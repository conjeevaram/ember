"""Generate the MuJoCo demo scenes (flat / obstacle course / fire) for both the
12-DOF physics body and the 29-DOF render-overlay body.

Scenes must sit next to the robot model so the model's relative ``meshdir``
resolves, so they are written into ``$UNITREE_RL_GYM/.../g1_description``. The
obstacle geometry is defined once (:func:`obstacle_geoms`) and shared between the
12-DOF physics scene and the 29-DOF overlay scene, so the overlay always lines
up with what the robot is physically standing on.

The course is intentionally smooth/rounded: the pretrained policy is *blind*, so
it only clears gentle, rounded obstacles -- rounded log -> up-ramp -> platform
-> down-ramp.
"""
from __future__ import annotations

import math

from .config import G1_MODEL_DIR

# Scene file names (written into G1_MODEL_DIR).
FLAT_12 = "g1_flat_demo.xml"
OBSTACLES_12 = "g1_obstacles_demo.xml"
FLAT_29 = "g1_flat29_demo.xml"
OBSTACLES_29 = "g1_obstacles29_demo.xml"
FIRE_12 = "g1_fire_demo.xml"
FIRE_29 = "g1_fire29_demo.xml"

# Fire prop: ground point (x, y, z) where the flame base sits, straight ahead of
# the spawn. Fixed and known so the controller can be validated against ground
# truth before the perception path is trusted. Query as scenes.FIRE_POSITION_WORLD.
FIRE_POSITION_WORLD = (4.0, 0.0, 0.0)

# 12-DOF physics body and the full 29-DOF (with hands) overlay body.
MODEL_12 = "g1_12dof.xml"
MODEL_29 = "g1_29dof_with_hand.xml"

_VISUAL = """  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
    <quality shadowsize="4096"/>
  </visual>"""

# Sky + checkered ground assets (only the 12-DOF model needs these; the 29-DOF
# model ships its own floor/sky).
_GROUND_ASSET = """  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.1 0.15 0.25"
             width="512" height="512"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.1"/>
  </asset>"""

_GROUND_BODY = """    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"
          condim="3" friction="1.0 0.005 0.0001"/>"""


def obstacle_geoms(bump_x=1.2, bump_radius=0.30, bump_expose=0.07,
                   ramp_x=2.2, ramp_run=1.2, rise=0.12,
                   plat_depth=1.0, width=1.4) -> str:
    """Fire-approach course a blind policy can traverse: rounded log -> up-ramp
    -> platform -> down-ramp. No sharp edges or drop-offs."""
    theta = math.atan2(rise, ramp_run)
    length = math.hypot(ramp_run, rise) + 0.1
    plat_x0 = ramp_x + ramp_run
    down_x = plat_x0 + plat_depth
    g = [
        f'<geom name="log" type="cylinder" size="{bump_radius} {width/2}" '
        f'pos="{bump_x} 0 {bump_expose - bump_radius}" euler="1.5708 0 0" '
        f'rgba="0.45 0.30 0.15 1" condim="3" friction="1.3 0.01 0.001"/>',
        f'<geom name="ramp_up" type="box" size="{length/2} {width/2} 0.03" '
        f'pos="{ramp_x + ramp_run/2} 0 {rise/2}" euler="0 {-theta} 0" '
        f'rgba="0.35 0.35 0.4 1" condim="3" friction="1.1 0.005 0.0001"/>',
        f'<geom name="platform" type="box" size="{plat_depth/2} {width/2} {rise/2}" '
        f'pos="{plat_x0 + plat_depth/2} 0 {rise/2}" '
        f'rgba="0.30 0.40 0.30 1" condim="3" friction="1.0 0.005 0.0001"/>',
        f'<geom name="ramp_down" type="box" size="{length/2} {width/2} 0.03" '
        f'pos="{down_x + ramp_run/2} 0 {rise/2}" euler="0 {theta} 0" '
        f'rgba="0.35 0.35 0.4 1" condim="3" friction="1.1 0.005 0.0001"/>',
    ]
    return "\n    ".join(g)


def fire_geom(pos=FIRE_POSITION_WORLD) -> str:
    """A bright flame prop: orange body + yellow tip. Non-colliding (contype=0)
    so the robot can walk up to it; it's a visual target for the detector."""
    x, y, z = pos
    return "\n    ".join([
        f'<geom name="fire_base" type="cylinder" size="0.18 0.30" '
        f'pos="{x} {y} {z + 0.30}" rgba="0.95 0.35 0.05 1" '
        f'contype="0" conaffinity="0"/>',
        f'<geom name="fire_tip" type="ellipsoid" size="0.15 0.15 0.28" '
        f'pos="{x} {y} {z + 0.72}" rgba="1.0 0.80 0.10 1" '
        f'contype="0" conaffinity="0"/>',
    ])


def _scene_12(model_label: str, statistic: str, body_extra: str = "") -> str:
    """A 12-DOF scene: the model has no floor/sky, so we add them here."""
    return f"""<mujoco model="{model_label}">
  <include file="{MODEL_12}"/>
  {statistic}
{_VISUAL}
{_GROUND_ASSET}
  <worldbody>
{_GROUND_BODY}
    {body_extra}
  </worldbody>
</mujoco>
"""


def _scene_29(model_label: str, statistic: str, body_extra: str = "") -> str:
    """A 29-DOF (render overlay) scene: the model ships its own floor/sky/light,
    so we only add obstacle geoms. Offscreen size is set in Python at load."""
    body = f"\n  <worldbody>\n    {body_extra}\n  </worldbody>" if body_extra else ""
    return f"""<mujoco model="{model_label}">
  <include file="{MODEL_29}"/>
  {statistic}
{_VISUAL}{body}
</mujoco>
"""


def render_scene(name: str) -> str:
    """Return the XML text for one of the named scenes."""
    if name == FLAT_12:
        return _scene_12("g1 flat scene", '<statistic center="0 0 0.8" extent="1.2"/>')
    if name == OBSTACLES_12:
        return _scene_12("g1 obstacle scene", '<statistic center="2 0 0.8" extent="3"/>',
                         obstacle_geoms())
    if name == FLAT_29:
        return _scene_29("g1 flat 29dof render", '<statistic center="0 0 0.8" extent="1.2"/>')
    if name == OBSTACLES_29:
        return _scene_29("g1 obstacles 29dof render", '<statistic center="2 0 0.8" extent="3"/>',
                         obstacle_geoms())
    if name == FIRE_12:
        return _scene_12("g1 fire scene", '<statistic center="2 0 0.8" extent="3"/>',
                         fire_geom())
    if name == FIRE_29:
        return _scene_29("g1 fire 29dof render", '<statistic center="2 0 0.8" extent="3"/>',
                         fire_geom())
    raise ValueError(f"unknown scene: {name}")


def write_scene(name: str) -> str:
    path = G1_MODEL_DIR / name
    path.write_text(render_scene(name))
    return str(path)


def ensure_scenes(force: bool = False) -> list[str]:
    """Generate all demo scenes into the model dir. Returns the paths written."""
    written = []
    for name in (FLAT_12, OBSTACLES_12, FLAT_29, OBSTACLES_29, FIRE_12, FIRE_29):
        path = G1_MODEL_DIR / name
        if force or not path.exists():
            written.append(write_scene(name))
    return written


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="(Re)generate G1 demo scenes.")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing scene files")
    args = p.parse_args()
    if not G1_MODEL_DIR.exists():
        raise SystemExit(f"model dir not found: {G1_MODEL_DIR}\n"
                         "Set $UNITREE_RL_GYM to your unitree_rl_gym checkout.")
    written = ensure_scenes(force=args.force)
    if written:
        print("wrote:\n  " + "\n  ".join(written))
    else:
        print("all scenes already present (use --force to overwrite)")


if __name__ == "__main__":
    main()
