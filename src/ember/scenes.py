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
from pathlib import Path

import numpy as np
from PIL import Image

from .config import G1_MODEL_DIR
from .spec import DebrisSpec, SceneSpec, TerrainSpec

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
# Multi-fire scenes use ``flame0..N`` bodies with lights ``firelight{i}a`` /
# ``firelight{i}b`` per fire (lower / upper; the single-fire names were
# ``firelight`` / ``firelight2``).
FIRE_POSITION_WORLD = (4.0, 0.0, 0.0)
FIRE_POSITIONS_DEFAULT = (FIRE_POSITION_WORLD,)

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


# Procedural wall height (m); tall enough that the blind walker cannot step over.
_WALL_HEIGHT = 0.8
# MuJoCo hfield base z must be > 0 (compiler rejects 0).
_HFIELD_BASE_Z = 0.01


def wall_geoms(walls, height: float = _WALL_HEIGHT) -> str:
    """Colliding box walls from ``(cx, cy, w, h, theta)`` tuples."""
    geoms = []
    for i, (cx, cy, w, h, theta) in enumerate(walls):
        geoms.append(
            f'<geom name="wall{i}" type="box" size="{w/2} {h/2} {height/2}" '
            f'pos="{cx} {cy} {height/2}" euler="0 0 {theta}" '
            f'rgba="0.35 0.35 0.4 1" condim="3" friction="1.1 0.005 0.0001"/>')
    return "\n    ".join(geoms)


def _terrain_png_path(spec_name: str) -> Path:
    return G1_MODEL_DIR / f"{spec_name}_hfield.png"


def write_terrain_png(terrain: TerrainSpec, path: Path) -> None:
    """Write a grayscale heightmap PNG for MuJoCo ``<hfield file=...>``."""
    rng = np.random.default_rng(terrain.seed)
    n = terrain.nrow
    u = np.linspace(0, 1, n)
    x, y = np.meshgrid(u, u)
    # Smooth undulation; peak normalized to 1 before scaling in XML via elevation.
    h = (np.sin(x * 6 + rng.uniform(0, 2 * math.pi))
         * np.cos(y * 5 + rng.uniform(0, 2 * math.pi)) * 0.35
         + np.sin((x + y) * 3) * 0.25 + 0.5)
    h = np.clip(h, 0.0, 1.0)
    gray = (h * 255).astype(np.uint8)
    Image.fromarray(gray).save(path)


def hfield_asset(terrain: TerrainSpec, spec_name: str) -> str:
    """``<asset>`` hfield entry; writes the PNG beside the scene XML."""
    png = _terrain_png_path(spec_name)
    write_terrain_png(terrain, png)
    # Absolute path: included g1_12dof.xml sets meshdir=meshes/, which breaks
    # relative hfield paths from the scene file.
    return (
        f'<hfield name="terrain_hf" file="{png}" '
        f'nrow="{terrain.nrow}" ncol="{terrain.ncol}" '
        f'size="{terrain.radius_x} {terrain.radius_y} {_HFIELD_BASE_Z} {terrain.elevation}"/>'
    )


def hfield_geom(terrain: TerrainSpec, bounds: tuple[float, float, float, float]) -> str:
    """Colliding heightfield geom centered on the scene bounds."""
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    return (
        f'<geom name="terrain" type="hfield" hfield="terrain_hf" '
        f'pos="{cx} {cy} 0" rgba="0.28 0.38 0.28 1" '
        f'condim="3" friction="1.0 0.005 0.0001"/>'
    )


def _statistic_from_bounds(bounds: tuple[float, float, float, float]) -> str:
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) / 2 + 0.5
    return f'<statistic center="{cx} {cy} 0.8" extent="{extent:.2f}"/>'


def _fire_positions_xy(fires: tuple[tuple[float, float], ...]) -> list[tuple[float, float, float]]:
    return [(x, y, 0.0) for x, y in fires]


def _terrain_hfield_line(spec: SceneSpec) -> str:
    if spec.terrain is None:
        return ""
    return hfield_asset(spec.terrain, spec.name)


def _body_extra_from_spec(spec: SceneSpec) -> str:
    parts = []
    if spec.walls:
        parts.append(wall_geoms(spec.walls))
    if spec.debris:
        parts.append(debris_geoms(spec.debris))
    if spec.terrain is not None:
        parts.append(hfield_geom(spec.terrain, spec.bounds))
    fire_xyz = _fire_positions_xy(spec.fires)
    if fire_xyz:
        parts.append(fire_geom(fire_xyz))
    return "\n    ".join(p for p in parts if p)


def _asset_extra_29(spec: SceneSpec, flame_xyz: list[tuple[float, float, float]]) -> str:
    parts = []
    if flame_xyz:
        parts.append(flame_assets().strip())
    line = _terrain_hfield_line(spec)
    if line:
        if parts:
            parts[0] = parts[0].replace("  </asset>", f"    {line}\n  </asset>")
        else:
            parts.append(f"<asset>\n    {line}\n  </asset>")
    if not parts:
        return ""
    return "\n  " + "\n  ".join(parts)


def _overlay_body_from_spec(spec: SceneSpec,
                            flame_xyz: list[tuple[float, float, float]]) -> str:
    """29-DOF overlay: walls + debris + terrain + emissive flames (aligned with physics)."""
    parts = []
    if spec.walls:
        parts.append(wall_geoms(spec.walls))
    if spec.debris:
        parts.append(debris_geoms(spec.debris))
    if spec.terrain is not None:
        parts.append(hfield_geom(spec.terrain, spec.bounds))
    if flame_xyz:
        parts.append(flame_bodies(flame_xyz))
    return "\n    ".join(p for p in parts if p)


def render_from_spec(spec: SceneSpec) -> tuple[str, str]:
    """Return (xml_12, xml_29) for a ``SceneSpec``."""
    stat = _statistic_from_bounds(spec.bounds)
    body = _body_extra_from_spec(spec)
    hfield_line = _terrain_hfield_line(spec)
    xml_12 = _scene_12(f"g1 {spec.name}", stat, body, hfield_line=hfield_line)
    flame_xyz = _fire_positions_xy(spec.fires)
    overlay = _overlay_body_from_spec(spec, flame_xyz)
    xml_29 = _scene_29(f"g1 {spec.name} 29dof", stat, overlay,
                       asset_extra=_asset_extra_29(spec, flame_xyz))
    return xml_12, xml_29


def spec_scene_names(name: str) -> tuple[str, str]:
    """12-DOF and 29-DOF filenames derived from ``spec.name``."""
    return f"g1_{name}_demo.xml", f"g1_{name}29_demo.xml"


def build(spec: SceneSpec) -> tuple[str, str]:
    """Write 12-DOF physics + 29-DOF overlay scenes from ``spec`` into ``G1_MODEL_DIR``.

    Robot start pose is carried only in the SceneSpec JSON (``spec.start``); G1Sim
    reset is out of Phase 1 scope and keyframes would fight the included model.
    """
    if not G1_MODEL_DIR.exists():
        raise FileNotFoundError(f"model dir not found: {G1_MODEL_DIR}")
    name_12, name_29 = spec_scene_names(spec.name)
    xml_12, xml_29 = render_from_spec(spec)
    path_12 = G1_MODEL_DIR / name_12
    path_29 = G1_MODEL_DIR / name_29
    path_12.write_text(xml_12)
    path_29.write_text(xml_29)
    return str(path_12), str(path_29)


_DEBRIS_FRICTION = 'condim="3" friction="1.1 0.005 0.0001"'
_LOG_FRICTION = 'condim="3" friction="1.3 0.01 0.001"'


def _world_xy(cx: float, cy: float, yaw: float, lx: float, ly: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return cx + c * lx - s * ly, cy + s * lx + c * ly


def _one_debris_geoms(index: int, d: DebrisSpec) -> str:
    """Emit colliding geoms for one debris item (blind-policy envelope)."""
    if d.kind == "log":
        r, half_len, expose = d.size_a, d.size_b, d.height
        wx, wy = d.cx, d.cy
        z = expose - r
        return (
            f'<geom name="debris{index}" type="cylinder" size="{r} {half_len}" '
            f'pos="{wx} {wy} {z}" euler="1.5708 0 {d.yaw}" '
            f'rgba="0.45 0.30 0.15 1" {_LOG_FRICTION}/>')
    if d.kind == "bump":
        rx, ry, peak = d.size_a, d.size_b, d.height
        rz = max(peak, min(rx, ry) * 0.5)
        z = peak - rz
        return (
            f'<geom name="debris{index}" type="ellipsoid" size="{rx} {ry} {rz}" '
            f'pos="{d.cx} {d.cy} {z}" euler="0 0 {d.yaw}" '
            f'rgba="0.40 0.35 0.28 1" {_DEBRIS_FRICTION}/>')
    if d.kind == "tier":
        width, depth, rise = d.size_a, d.size_b, d.height
        ramp_run = max(depth * 0.55, rise / 0.10)
        theta = math.atan2(rise, ramp_run)
        length = math.hypot(ramp_run, rise) + 0.08
        plat_depth = depth * 0.45
        plat_half = plat_depth / 2
        ramp_half = ramp_run / 2
        # Local +x is ramp approach; platform sits at positive x.
        ramp_lx = -(plat_half + ramp_half)
        plat_lx = 0.0
        ramp_wx, ramp_wy = _world_xy(d.cx, d.cy, d.yaw, ramp_lx, 0.0)
        plat_wx, plat_wy = _world_xy(d.cx, d.cy, d.yaw, plat_lx, 0.0)
        # Robot models declare <compiler angle="radian">; emit radians (theta, yaw)
        # directly, matching the log/bump branches and obstacle_geoms().
        return "\n    ".join([
            f'<geom name="tier{index}_ramp" type="box" size="{length/2} {width/2} 0.03" '
            f'pos="{ramp_wx} {ramp_wy} {rise/2}" euler="0 {-theta} {d.yaw}" '
            f'rgba="0.35 0.35 0.4 1" {_DEBRIS_FRICTION}/>',
            f'<geom name="tier{index}_plat" type="box" size="{plat_half} {width/2} {rise/2}" '
            f'pos="{plat_wx} {plat_wy} {rise/2}" euler="0 0 {d.yaw}" '
            f'rgba="0.30 0.40 0.30 1" condim="3" friction="1.0 0.005 0.0001"/>',
        ])
    raise ValueError(f"unknown debris kind: {d.kind!r}")


def debris_geoms(debris: tuple[DebrisSpec, ...]) -> str:
    """Colliding traversable debris: rounded logs, low bumps, ramped tiers."""
    parts = [_one_debris_geoms(i, d) for i, d in enumerate(debris)]
    return "\n    ".join(p for p in parts if p)


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


def _fire_positions(positions=None):
    """Normalize a single ``(x, y, z)`` or a list of them."""
    if positions is None:
        return list(FIRE_POSITIONS_DEFAULT)
    if isinstance(positions[0], (int, float)):
        return [tuple(positions)]
    return [tuple(p) for p in positions]


def fire_geom(positions=None) -> str:
    """Bright flame props for the bare 12-DOF fire scene: ``fire_base{i}`` +
    ``fire_tip{i}`` per position. Non-colliding (contype=0) so the robot can
    walk up to them; visual targets for the detector. The overlay scene gets
    the richer emissive flames (:func:`flame_bodies`)."""
    geoms = []
    for i, (x, y, z) in enumerate(_fire_positions(positions)):
        geoms.extend([
            f'<geom name="fire_base{i}" type="cylinder" size="0.18 0.30" '
            f'pos="{x} {y} {z + 0.30}" rgba="0.95 0.35 0.05 1" '
            f'contype="0" conaffinity="0"/>',
            f'<geom name="fire_tip{i}" type="ellipsoid" size="0.15 0.15 0.28" '
            f'pos="{x} {y} {z + 0.72}" rgba="1.0 0.80 0.10 1" '
            f'contype="0" conaffinity="0"/>',
        ])
    return "\n    ".join(geoms)


# Emissive flame: translucent red/orange shells (wide base -> narrow top) over
# an opaque white-hot core, plus two flickering fire lights. ember.effects
# animates these and adds the runtime smoke/embers/water.
_FLAME_MATERIALS = (
    ("m_core", "1.0 0.95 0.70 1.0", 1.0),   # white-hot core (opaque)
    ("m_mid", "1.0 0.60 0.15 0.55", 0.95),  # orange shell (translucent)
    ("m_out", "0.95 0.25 0.06 0.32", 0.85),  # red tongues (translucent)
)
# (material, size sx sy sz, pos px py pz) in the flame-body frame.
_FLAME_GEOMS = (
    ("m_out", "0.30 0.30 0.22", "0 0 0.22"),
    ("m_out", "0.24 0.24 0.24", "0.03 0.02 0.48"),
    ("m_out", "0.16 0.16 0.26", "-0.03 0.03 0.74"),
    ("m_out", "0.09 0.09 0.22", "0.02 -0.02 0.98"),
    ("m_mid", "0.20 0.20 0.20", "0 0 0.24"),
    ("m_mid", "0.14 0.14 0.22", "0.02 0.01 0.50"),
    ("m_mid", "0.08 0.08 0.18", "-0.02 0.02 0.74"),
    ("m_core", "0.12 0.12 0.16", "0 0 0.26"),
    ("m_core", "0.07 0.07 0.14", "0.01 0 0.46"),
)


def flame_assets() -> str:
    """``<asset>`` block with the emissive flame materials."""
    mats = "\n    ".join(
        f'<material name="{n}" rgba="{rgba}" emission="{e}"/>'
        for n, rgba, e in _FLAME_MATERIALS)
    return f"\n  <asset>\n    {mats}\n  </asset>"


def _flame_body_one(index: int, pos) -> str:
    """One emissive ``flame{index}`` body + ``firelight{index}a/b`` lights."""
    x, y, z = pos
    geoms = "\n      ".join(
        f'<geom type="ellipsoid" size="{size}" pos="{gpos}" material="{mat}" '
        f'contype="0" conaffinity="0"/>'
        for mat, size, gpos in _FLAME_GEOMS)
    return (
        f'<body name="flame{index}" pos="{x} {y} {z}">\n      {geoms}\n    </body>\n'
        f'    <light name="firelight{index}a" pos="{x} {y} {z + 0.45}" dir="0 0 -1" '
        f'diffuse="1.0 0.5 0.18" specular="0.5 0.25 0.06" attenuation="0.25 0 0.04"/>\n'
        f'    <light name="firelight{index}b" pos="{x - 0.3} {y} {z + 0.9}" dir="0 0 -1" '
        f'diffuse="1.0 0.5 0.18" specular="0.5 0.25 0.06" attenuation="0.4 0 0.06"/>')


def flame_bodies(positions=None) -> str:
    """Emissive ``flame0..N`` bodies + paired lights from a fire-position list.

    Per fire ``i``: body ``flame{i}``; lights ``firelight{i}a`` (lower) and
    ``firelight{i}b`` (upper). ``ember.effects.SceneFX`` discovers these by name.
    """
    return "\n    ".join(
        _flame_body_one(i, pos) for i, pos in enumerate(_fire_positions(positions)))


def flame_body(pos=FIRE_POSITION_WORLD) -> str:
    """Single-fire shorthand for :func:`flame_bodies`."""
    return flame_bodies([pos])


def _scene_12(model_label: str, statistic: str, body_extra: str = "",
              hfield_line: str = "") -> str:
    """A 12-DOF scene: the model has no floor/sky, so we add them here."""
    asset_block = _GROUND_ASSET
    if hfield_line:
        asset_block = asset_block.replace("  </asset>", f"    {hfield_line}\n  </asset>")
    return f"""<mujoco model="{model_label}">
  <include file="{MODEL_12}"/>
  {statistic}
{_VISUAL}
{asset_block}
  <worldbody>
{_GROUND_BODY}
    {body_extra}
  </worldbody>
</mujoco>
"""


def _scene_29(model_label: str, statistic: str, body_extra: str = "",
              asset_extra: str = "") -> str:
    """A 29-DOF (render overlay) scene: the model ships its own floor/sky/light,
    so we only add obstacle/flame geoms. Offscreen size is set in Python at
    load."""
    body = f"\n  <worldbody>\n    {body_extra}\n  </worldbody>" if body_extra else ""
    return f"""<mujoco model="{model_label}">
  <include file="{MODEL_29}"/>
  {statistic}
{_VISUAL}{asset_extra}{body}
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
                         flame_bodies(), flame_assets())
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
