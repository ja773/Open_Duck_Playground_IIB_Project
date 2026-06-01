# generate_slope_xmls_clean.py

from pathlib import Path
import math

OUT = Path("playground/open_duck_mini_v2/xmls")
OUT.mkdir(parents=True, exist_ok=True)

ANGLES = [1, 3, 5, 7, 10]

CTRL = """
    0.002
    0.053
    -0.63
    1.368
    -0.784
    0
    0
    0
    0
    -0.003
    -0.065
    0.635
    1.379
    -0.796
"""

def make_xml(angle_deg, direction):
    theta = math.radians(angle_deg)

    if direction == "up":
        euler_y = -theta
    elif direction == "down":
        euler_y = theta
    else:
        euler_y = 0.0

    half_len = 6.0
    half_width = 5.5
    half_thick = 0.05

    # Put robot near start of terrain, already on surface.
    robot_x = -2.5
    robot_y = 0.0

    # Height of tilted top surface at robot_x.
    # Approximate top height: z = -x * sin(euler_y) + half_thick*cos(theta)
    terrain_body_z = -half_thick * math.cos(theta)
    surface_z_at_robot = terrain_body_z - robot_x * math.sin(euler_y) + half_thick * math.cos(theta)

    robot_z = surface_z_at_robot + 0.15

    qpos = f"""
    {robot_x:.4f} {robot_y:.4f} {robot_z:.4f}
    1 0 0 0
    0.002
    0.053
    -0.63
    1.368
    -0.784
    0
    0
    0
    0
    -0.003
    -0.065
    0.635
    1.379
    -0.796
"""

    return f"""<mujoco model="scene">
    <include file="open_duck_mini_v2.xml"/>

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="160" elevation="-20"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge"
                 rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0"
                 width="300" height="300"/>
        <material name="groundplane" texture="groundplane"
                  texuniform="true" texrepeat="5 5" reflectance="0"/>
    </asset>

    <worldbody>
        <body name="floor" pos="0 0 {terrain_body_z:.4f}">
            <geom name="floor"
                  type="box"
                  size="{half_len:.4f} {half_width:.4f} {half_thick:.4f}"
                  euler="0 {euler_y:.6f} 0"
                  material="groundplane"
                  contype="1"
                  conaffinity="1"
                  priority="1"
                  friction="0.6"
                  condim="3"/>
        </body>
    </worldbody>

    <keyframe>
        <key name="home"
             qpos="{qpos}"
             ctrl="{CTRL}"/>
    </keyframe>
</mujoco>
"""

for angle in ANGLES:
    for direction in ["up", "down"]:
        path = OUT / f"scene_slope_{direction}_{angle}deg.xml"
        path.write_text(make_xml(angle, direction))
        print(f"Wrote {path}")