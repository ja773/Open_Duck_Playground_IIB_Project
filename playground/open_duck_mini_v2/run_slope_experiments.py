# save as:
# playground/open_duck_mini_v2/run_slope_experiments.py

import argparse
import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from playground.open_duck_mini_v2.mujoco_infer import MjInfer


EXPERIMENTS = [
    {
        "name": "flat",
        "angle": 0,
        "direction": "flat",
        "xml": "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    },
    {
        "name": "up_1deg",
        "angle": 1,
        "direction": "up",
        "xml": "playground/open_duck_mini_v2/xmls/scene_slope_up_1deg.xml",
    },
    {
        "name": "up_3deg",
        "angle": 3,
        "direction": "up",
        "xml": "playground/open_duck_mini_v2/xmls/scene_slope_up_3deg.xml",
    },
    {
        "name": "up_5deg",
        "angle": 5,
        "direction": "up",
        "xml": "playground/open_duck_mini_v2/xmls/scene_slope_up_5deg.xml",
    },
    {
        "name": "up_7deg",
        "angle": 7,
        "direction": "up",
        "xml": "playground/open_duck_mini_v2/xmls/scene_slope_up_7deg.xml",
    },
    {
        "name": "up_10deg",
        "angle": 10,
        "direction": "up",
        "xml": "playground/open_duck_mini_v2/xmls/scene_slope_up_10deg.xml",
    },
    {
        "name": "down_1deg",
        "angle": 1,
        "direction": "down",
        "xml": "playground/open_duck_mini_v2/xmls/scene_slope_down_1deg.xml",
    },
    {
        "name": "down_3deg",
        "angle": 3,
        "direction": "down",
        "xml": "playground/open_duck_mini_v2/xmls/scene_slope_down_3deg.xml",
    },
    {
        "name": "down_5deg",
        "angle": 5,
        "direction": "down",
        "xml": "playground/open_duck_mini_v2/xmls/scene_slope_down_5deg.xml",
    },
    {
        "name": "down_7deg",
        "angle": 7,
        "direction": "down",
        "xml": "playground/open_duck_mini_v2/xmls/scene_slope_down_7deg.xml",
    },
    {
        "name": "down_10deg",
        "angle": 10,
        "direction": "down",
        "xml": "playground/open_duck_mini_v2/xmls/scene_slope_down_10deg.xml",
    },
]




def terrain_angle_rad(exp):
    theta = math.radians(exp["angle"])

    if exp["direction"] == "up":
        return theta
    if exp["direction"] == "down":
        return -theta
    return 0.0


def terrain_height_at_x(exp, x):
    """
    Height of the ideal tilted terrain surface at horizontal coordinate x.

    This assumes the clean XML generator creates a single tilted surface
    passing through the origin-like frame used for the experiment.
    """
    alpha = terrain_angle_rad(exp)
    return x * math.tan(alpha)


def height_above_terrain(exp, data):
    x = float(data.qpos[0])
    z = float(data.qpos[2])
    return z - terrain_height_at_x(exp, x)


def quat_to_yaw(q):
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_to_rotmat_wxyz(q):
    w, x, y, z = q

    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,         1 - 2*x*x - 2*y*y],
    ])


def terrain_normal_world(exp):
    alpha = terrain_angle_rad(exp)

    # Terrain is rotated about y-axis.
    # Normal of flat plane [0,0,1] rotated by alpha around y.
    return np.array([
        math.sin(alpha),
        0.0,
        math.cos(alpha),
    ])


def body_up_world(data):
    R = quat_to_rotmat_wxyz(data.qpos[3:7])
    return R[:, 2]


def terrain_relative_tilt(exp, data):
    n = terrain_normal_world(exp)
    u = body_up_world(data)

    cosang = np.clip(np.dot(n, u) / (np.linalg.norm(n) * np.linalg.norm(u)), -1.0, 1.0)

    return math.acos(cosang)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_trial(
    exp,
    onnx_path,
    reference_data,
    max_time,
    vx,
    heading_hold=False,
):

    agent = MjInfer(
        model_path=exp["xml"],
        reference_data=reference_data,
        onnx_model_path=onnx_path,
        standing=False,
    )



    if not hasattr(agent, "prev_motor_targets"):
        agent.prev_motor_targets = agent.default_actuator.copy()

    data = agent.data
    model = agent.model

    initial_clearance = height_above_terrain(exp, data)
    min_clearance = max(0.04, initial_clearance - 0.08)

    fall_clearance_thresh = min_clearance
    fall_tilt_thresh = 1.0  # radians, about 57 degrees

    fall_reason = "none"
    fall_time_s = -1.0

    # command = [vx, vy, yaw_rate, neck_pitch, head_pitch, head_yaw, head_roll]
    agent.commands = [vx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    start_x = float(data.qpos[0])
    start_y = float(data.qpos[1])

    max_x = start_x
    entered_ramp = False
    completed_ramp = False
    left_corridor = False
    time_to_ramp = None
    time_on_ramp = 0.0
    corridor_half_width = 0.75

    target_yaw = quat_to_yaw(data.qpos[3:7])

    sim_time = 0.0
    counter = 0

    max_lateral_drift = 0.0
    max_yaw_error = 0.0

    trajectory = []

    min_clearance_seen = float("inf")
    max_tilt_seen = 0.0

    fell = False

    while sim_time < max_time:

        mujoco.mj_step(model, data)

        sim_time += model.opt.timestep
        counter += 1

        x = float(data.qpos[0])
        y = float(data.qpos[1])

        max_x = max(max_x, x)

        # if exp["ramp_start"] is not None:
        #     if x >= exp["ramp_start"] and not entered_ramp:
        #         entered_ramp = True
        #         time_to_ramp = sim_time

        #     if exp["ramp_start"] <= x <= exp["ramp_end"]:
        #         time_on_ramp += model.opt.timestep

        #     if x >= exp["ramp_end"]:
        #         completed_ramp = True

        if abs(y - start_y) > corridor_half_width:
            left_corridor = True

        yaw = quat_to_yaw(data.qpos[3:7])

        trajectory.append(
            {
                "t": sim_time,
                "x": x,
                "y": y,
                "yaw": yaw,
            }
        )

        yaw_error = wrap_angle(yaw - target_yaw)

        max_lateral_drift = max(
            max_lateral_drift,
            abs(y - start_y),
        )

        max_yaw_error = max(
            max_yaw_error,
            abs(yaw_error),
        )


        # simple fall detection
        if np.isnan(data.qpos).any() or np.isnan(data.qvel).any():
            fell = True
            fall_reason = "nan"
            break

        clearance = height_above_terrain(exp, data)
        tilt_rel = terrain_relative_tilt(exp, data)

        min_clearance_seen = min(min_clearance_seen, clearance)
        max_tilt_seen = max(max_tilt_seen, tilt_rel)

        if clearance < fall_clearance_thresh:
            fell = True
            fall_reason = "low_clearance"
            fall_time_s = sim_time
            break

        if tilt_rel > fall_tilt_thresh:
            fell = True
            fall_reason = "excess_tilt"
            fall_time_s = sim_time
            break

        if counter % agent.decimation == 0:

            if not agent.standing:
                agent.imitation_i += 1.0 * agent.phase_frequency_factor
                agent.imitation_i = (
                    agent.imitation_i % agent.PRM.nb_steps_in_period
                )

                agent.imitation_phase = np.array(
                    [
                        np.cos(
                            agent.imitation_i
                            / agent.PRM.nb_steps_in_period
                            * 2
                            * np.pi
                        ),
                        np.sin(
                            agent.imitation_i
                            / agent.PRM.nb_steps_in_period
                            * 2
                            * np.pi
                        ),
                    ]
                )

            if heading_hold:
                kp = 0.8
                yaw_cmd = np.clip(-kp * yaw_error, -0.3, 0.3)

                agent.commands[0] = vx
                agent.commands[1] = 0.0
                agent.commands[2] = float(yaw_cmd)

            obs = agent.get_obs(
                data,
                agent.commands,
            )

            action = agent.policy.infer(obs)

            agent.last_last_last_action = (
                agent.last_last_action.copy()
            )
            agent.last_last_action = (
                agent.last_action.copy()
            )
            agent.last_action = action.copy()

            agent.motor_targets = (
                agent.default_actuator
                + action * agent.action_scale
            )

            agent.motor_targets = np.clip(
                agent.motor_targets,
                agent.prev_motor_targets
                - agent.max_motor_velocity
                * (agent.sim_dt * agent.decimation),
                agent.prev_motor_targets
                + agent.max_motor_velocity
                * (agent.sim_dt * agent.decimation),
            )

            agent.prev_motor_targets = (
                agent.motor_targets.copy()
            )

            data.ctrl = agent.motor_targets.copy()

    #goal_distance = 0.5
    corridor_half_width = 0.75

    heading_failure = max_yaw_error > 1.2

    final_x = float(data.qpos[0])
    final_y = float(data.qpos[1])

    distance = final_x - start_x

    goal_distance = 0.5 if vx <= 0.1 else 1.0
    left_corridor = max_lateral_drift > 0.75
    heading_failure = max_yaw_error > 1.2

    success = int(
        (not fell)
        and distance >= goal_distance
        and not left_corridor
        and not heading_failure
    )

    row = {
        "experiment": exp["name"],
        "angle_deg": exp["angle"],
        "direction": exp["direction"],
        "success": success,
        "fell": int(fell),
        "left_corridor": int(left_corridor),
        "heading_failure": int(heading_failure),
        "time_s": sim_time,
        "distance_m": distance,
        "max_x_m": max_x - start_x,
        "max_lateral_drift_m": max_lateral_drift,
        "max_yaw_error_rad": max_yaw_error,
        "fall_reason": fall_reason,
        "fall_time_s": fall_time_s,
        "min_clearance_m": min_clearance_seen,
        "max_terrain_relative_tilt_rad": max_tilt_seen,
    }

    return row, trajectory


def summarise(rows):

    grouped = {}

    for r in rows:
        grouped.setdefault(r["experiment"], []).append(r)

    summary = []

    for exp_name, rs in grouped.items():

        n = len(rs)

        def mean(k):
            return sum(float(r[k]) for r in rs) / n
        
        def mean_positive(key):
            vals = [float(r[key]) for r in rs if float(r[key]) >= 0.0]
            return sum(vals) / len(vals) if vals else -1.0

        summary.append(
            {
                "experiment": exp_name,
                "angle_deg": rs[0]["angle_deg"],
                "direction": rs[0]["direction"],
                "trials": n,
                "success_rate": mean("success"),
                "fall_rate": mean("fell"),
                "left_corridor_rate": mean("left_corridor"),
                "mean_time_s": mean("time_s"),
                "mean_distance_m": mean("distance_m"),
                "mean_max_x_m": mean("max_x_m"),
                "mean_lateral_drift_m": mean("max_lateral_drift_m"),
                "mean_yaw_error_rad": mean("max_yaw_error_rad"),
                "heading_failure_rate": mean("heading_failure"),
                "mean_fall_time_s": mean_positive("fall_time_s"),
                "mean_min_clearance_m": mean("min_clearance_m"),
                "mean_max_tilt_rad": mean("max_terrain_relative_tilt_rad"),
            }
        )

    return sorted(
        summary,
        key=lambda x: (
            x["direction"],
            x["angle_deg"],
        ),
    )


def save_csv(path, rows):

    if not rows:
        return

    with open(path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def classify_failure(r):
    if r["success_rate"] >= 0.8:
        return "Success"
    if r["fall_rate"] >= 0.8:
        return "Fall"
    if r["left_corridor_rate"] >= 0.8:
        return "Drift / escape"
    if r["mean_yaw_error_rad"] > 1.2:
        return "Heading loss"
    if r["mean_distance_m"] < 0.5:
        return "Low progress"
    return "Mixed"


def save_latex_table(path, rows):
    with open(path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lrrrrrrrl}\n")
        f.write("\\hline\n")
        f.write(
            "Terrain & Succ. & Fall & Esc. & Time & Dist. & Drift & Yaw & Failure mode \\\\\n"
        )
        f.write("\\hline\n")

        for r in rows:
            failure_mode = classify_failure(r)

            f.write(
                f"{r['experiment']} & "
                f"{r['success_rate']:.2f} & "
                f"{r['fall_rate']:.2f} & "
                f"{r['left_corridor_rate']:.2f} & "
                f"{r['mean_time_s']:.2f} & "
                f"{r['mean_distance_m']:.2f} & "
                f"{r['mean_lateral_drift_m']:.2f} & "
                f"{r['mean_yaw_error_rad']:.2f} & "
                f"{failure_mode} \\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write(
            "\\caption{Baseline policy performance across slope conditions. "
            "Succ. denotes success rate, Fall denotes fall rate, and Esc. denotes "
            "the fraction of trials leaving the lateral corridor. Time denotes mean episode duration. Dist., Drift, "
            "and Yaw report mean forward distance, maximum lateral drift, and "
            "maximum yaw error respectively.}\n"
        )
        f.write("\\label{tab:baseline_slope_results}\n")
        f.write("\\end{table}\n")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--onnx_model_path",
        required=True,
    )

    parser.add_argument(
        "--reference_data",
        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    )

    parser.add_argument(
        "--trials_per_scene",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max_time",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--vx",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--heading_hold",
        action="store_true",
    )

    parser.add_argument(
        "--out_dir",
        default="results/baseline_eval",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    for exp in EXPERIMENTS:

        print(f"\n=== {exp['name']} ===")

        for trial in range(args.trials_per_scene):

            row, trajectory = run_trial(
                exp=exp,
                onnx_path=args.onnx_model_path,
                reference_data=args.reference_data,
                max_time=args.max_time,
                vx=args.vx,
                heading_hold=args.heading_hold,
            )

            row["trial"] = trial

            traj_path = out_dir / f"{exp['name']}_trial_{trial}_trajectory.json"
            with open(traj_path, "w") as f:
                json.dump(trajectory, f)

            all_rows.append(row)

            print(
                f"trial={trial} "
                f"success={row['success']} "
                f"dist={row['distance_m']:.2f} "
                f"drift={row['max_lateral_drift_m']:.2f} "
                f"yaw={row['max_yaw_error_rad']:.2f}"
            )

    summary = summarise(all_rows)

    save_csv(
        out_dir / "raw_trials.csv",
        all_rows,
    )

    save_csv(
        out_dir / "summary.csv",
        summary,
    )

    with open(
        out_dir / "summary.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    save_latex_table(
        out_dir / "latex_table.tex",
        summary,
    )

    print("\n==============================")
    print("Finished experiments.")
    print("==============================")

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()