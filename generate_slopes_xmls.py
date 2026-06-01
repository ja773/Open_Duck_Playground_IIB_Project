from pathlib import Path
import math

OUT = Path("playground/open_duck_mini_v2/xmls")
OUT.mkdir(parents=True, exist_ok=True)

ANGLES = [1, 3, 5, 7, 10]

RAMP_START = 0.35
RAMP_END = 1.85
RAMP_HALF_LEN = (RAMP_END - RAMP_START) / 2
RAMP_CENTRE_X = (RAMP_START + RAMP_END) / 2

HOME_QPOS_FLAT = """
    0 0 0.15
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

def common_header():
    return """<mujoco model="scene">
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
"""

def keyframe(qpos):
    return f"""
    <keyframe>
        <key name="home"
             qpos="{qpos}"
             ctrl="{CTRL}"/>
    </keyframe>
</mujoco>
"""

def uphill_xml(angle_deg):
    theta = math.radians(angle_deg)
    ramp_half_len = RAMP_HALF_LEN
    ramp_half_thick = 0.03

    # Ramp centre x = 3.5, ramp spans x = 2.0 to 5.0.
    # Use original infinite plane as floor for uphill.
    ramp_z = ramp_half_len * math.sin(theta) - ramp_half_thick * math.cos(theta)
    upper_z = 2 * ramp_half_len * math.sin(theta)

    return common_header() + f"""
    <worldbody>
        <body name="floor">
            <geom name="floor"
                  size="0 0 0.01"
                  type="plane"
                  material="groundplane"
                  contype="1"
                  conaffinity="0"
                  priority="1"
                  friction="0.6"
                  condim="3"/>
        </body>

        <body name="ramp" pos="{RAMP_CENTRE_X:.4f} 0 {ramp_z:.4f}">
            <geom name="ramp_geom"
                  type="box"
                  size="{RAMP_HALF_LEN:.4f} 5.5 0.03"
                  pos="0 0 0.03"
                  euler="0 -{theta:.4f} 0"
                  material="groundplane"
                  contype="1"
                  conaffinity="1"
                  condim="3"
                  friction="0.6 0.1 0.1"/>
        </body>

        <body name="lower_floor" pos="{RAMP_END + 0.8:.4f} 0 {upper_z - 0.05:.4f}">
            <geom name="lower_floor_geom"
                  type="box"
                  size="1.0 5.8 0.05"
                  material="groundplane"
                  contype="1"
                  conaffinity="1"
                  condim="3"
                  friction="0.6 0.1 0.1"/>
        </body>
    </worldbody>
""" + keyframe(HOME_QPOS_FLAT)

def downhill_xml(angle_deg):
    theta = math.radians(angle_deg)
    ramp_half_len = RAMP_HALF_LEN
    ramp_half_thick = 0.03

    height_drop = 2 * ramp_half_len * math.sin(theta)
    ramp_z = ramp_half_len * math.sin(theta) - ramp_half_thick * math.cos(theta)

    start_platform_z = height_drop - 0.05
    robot_z = height_drop + 0.15

    home_qpos = HOME_QPOS_FLAT.replace("    0 0 0.15", f"    0 0 {robot_z:.4f}", 1)

    return common_header() + f"""
    <worldbody>
        <body name="floor" pos="0 0 {start_platform_z:.4f}">
            <geom name="floor"
                  type="box"
                  size="2.0 5.8 0.05"
                  material="groundplane"
                  contype="1"
                  conaffinity="1"
                  condim="3"
                  friction="0.6 0.1 0.1"/>
        </body>

        <body name="ramp" pos="{RAMP_CENTRE_X:.4f} 0 {ramp_z:.4f}">
            <geom name="ramp_geom"
                  type="box"
                  size="{RAMP_HALF_LEN:.4f} 5.5 0.03"
                  pos="0 0 0.03"
                  euler="0 {theta:.4f} 0"
                  material="groundplane"
                  contype="1"
                  conaffinity="1"
                  condim="3"
                  friction="0.6 0.1 0.1"/>
        </body>

        <body name="lower_floor" pos="{RAMP_END + 0.8:.4f} 0 -0.05">
            <geom name="lower_floor_geom"
                  type="box"
                  size="1.0 5.8 0.05"
                  material="groundplane"
                  contype="1"
                  conaffinity="1"
                  condim="3"
                  friction="0.6 0.1 0.1"/>
        </body>
    </worldbody>
""" + keyframe(home_qpos)

for angle in ANGLES:
    up_path = OUT / f"scene_slope_up_{angle}deg.xml"
    down_path = OUT / f"scene_slope_down_{angle}deg.xml"

    up_path.write_text(uphill_xml(angle))
    down_path.write_text(downhill_xml(angle))

    print(f"Wrote {up_path}")
    print(f"Wrote {down_path}")