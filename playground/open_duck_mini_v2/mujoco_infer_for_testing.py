import mujoco
import pickle
import numpy as np
import mujoco.viewer
import time
import argparse

from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase

USE_MOTOR_SPEED_LIMITS = True


class MjInferFixed(MJInferBase):
    def __init__(self, model_path, reference_data, onnx_model_path, standing):
        super().__init__(model_path)

        self.standing = standing

        # ===== FIXED COMMAND (EDIT THIS FOR EXPERIMENTS) =====
        self.commands = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # [vx, vy, yaw, neck, head_pitch, head_yaw, head_roll]

        # ===== PARAMS =====
        self.action_scale = 0.25
        self.dof_vel_scale = 0.05
        self.max_motor_velocity = 5.24

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        if not self.standing:
            self.PRM = PolyReferenceMotion(reference_data)

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])

        self.saved_obs = []

        print(f"Running FIXED command test: {self.commands}")

    # ===== OBSERVATION =====
    def get_obs(self, data, command):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)

        # 🚨 IMPORTANT: NO BIAS
        # accelerometer[0] += 1.3  ← REMOVED

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        contacts = self.get_feet_contacts(data)

        obs = np.concatenate(
            [
                gyro,
                accelerometer,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                self.imitation_phase,
            ]
        )

        return obs

    # ===== NO KEYBOARD =====
    def key_callback(self, keycode):
        pass  # Disabled

    # ===== MAIN LOOP =====
    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:

                counter = 0

                while True:
                    step_start = time.time()
                    mujoco.mj_step(self.model, self.data)
                    counter += 1

                    if counter % self.decimation == 0:

                        # ===== IMITATION PHASE =====
                        if not self.standing:
                            self.imitation_i += 1.0
                            self.imitation_i = (
                                self.imitation_i % self.PRM.nb_steps_in_period
                            )

                            self.imitation_phase = np.array(
                                [
                                    np.cos(
                                        self.imitation_i
                                        / self.PRM.nb_steps_in_period
                                        * 2 * np.pi
                                    ),
                                    np.sin(
                                        self.imitation_i
                                        / self.PRM.nb_steps_in_period
                                        * 2 * np.pi
                                    ),
                                ]
                            )

                        obs = self.get_obs(self.data, self.commands)
                        self.saved_obs.append(obs)

                        action = self.policy.infer(obs)

                        # Update action history
                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()

                        # Motor targets
                        self.motor_targets = (
                            self.default_actuator + action * self.action_scale
                        )

                        if USE_MOTOR_SPEED_LIMITS:
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets
                                - self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                                self.prev_motor_targets
                                + self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                            )
                            self.prev_motor_targets = self.motor_targets.copy()

                        self.data.ctrl = self.motor_targets.copy()

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))
            print("Saved observations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument("--standing", action="store_true", default=False)

    args = parser.parse_args()

    mjinfer = MjInferFixed(
        args.model_path,
        args.reference_data,
        args.onnx_model_path,
        args.standing,
    )
    mjinfer.run()