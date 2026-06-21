"""Phase 1 scenegen / SceneSpec / MuJoCo load tests."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest

from ember import scenegen, scenes
from ember.config import G1_MODEL_DIR
from ember.spec import DebrisSpec, SceneSpec, TerrainSpec, from_dict, from_json, to_dict, to_json


pytestmark = pytest.mark.skipif(
    not G1_MODEL_DIR.exists(),
    reason=f"G1 model dir missing: {G1_MODEL_DIR}",
)


def test_determinism():
    a = scenegen.random_spec(42, n_fires=2, n_walls=3, n_debris=4, n_tiers=1)
    b = scenegen.random_spec(42, n_fires=2, n_walls=3, n_debris=4, n_tiers=1)
    assert a == b
    assert len(a.debris) == 4
    c = scenegen.random_spec(43, n_fires=2, n_walls=3, n_debris=4, n_tiers=1)
    assert a != c


def test_json_round_trip():
    spec = scenegen.random_spec(7, n_fires=3, n_walls=5, n_debris=3, n_tiers=1)
    assert from_dict(to_dict(spec)) == spec
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "spec.json"
        to_json(spec, path)
        assert from_json(path) == spec


def test_json_round_trip_with_debris():
    spec = scenegen.random_spec(17, n_fires=2, n_walls=3, n_debris=5, n_tiers=2)
    assert len(spec.debris) == 5
    assert from_dict(to_dict(spec)) == spec
    kinds = {d.kind for d in spec.debris}
    assert "tier" in kinds


def test_old_json_without_debris_key():
    # A spec dict predating the debris field (no "debris" key) must still load,
    # defaulting to empty debris. Built in-memory so it can't be invalidated by
    # build_scenes.py overwriting on-disk specs with debris.
    legacy_dict = {
        "name": "legacy",
        "bounds": [-1.0, 8.0, -4.0, 4.0],
        "start": [0.0, 0.0, 0.0],
        "walls": [],
        "fires": [[4.0, 0.0]],
        "terrain": None,
        "home": [0.5, 0.5],
        "seed": 0,
    }
    assert "debris" not in legacy_dict
    spec = from_dict(legacy_dict)
    assert spec.debris == ()
    assert from_dict(to_dict(spec)) == spec


def test_json_round_trip_with_terrain():
    spec = scenegen.random_spec(99, n_fires=2, n_walls=2, terrain=True)
    assert from_dict(to_dict(spec)) == spec
    assert spec.terrain is not None
    assert spec.terrain.elevation <= scenegen.TERRAIN_MAX_ELEVATION


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("n_fires", [1, 2, 3, 4])
@pytest.mark.parametrize("n_walls", [0, 2, 4, 6])
def test_validate_batch(seed, n_fires, n_walls):
    spec = scenegen.random_spec(seed + n_fires * 100 + n_walls * 10,
                                n_fires=n_fires, n_walls=n_walls, n_debris=3)
    scenegen.validate_spec(spec)
    for d in spec.debris:
        assert 0 < d.height <= scenegen.DEBRIS_MAX_HEIGHT


def _independent_reachability(spec: SceneSpec) -> bool:
    grid, xmin, ymin, res = scenegen.occupancy_grid(spec)
    si, sj = scenegen._world_to_cell(spec.start[0], spec.start[1], xmin, ymin, res)
    reachable = scenegen.flood_fill_reachable(grid, (si, sj))
    if not scenegen._goal_reachable(reachable, spec.home[0], spec.home[1], xmin, ymin, res):
        return False
    for fx, fy in spec.fires:
        if not scenegen._fire_reachable(reachable, fx, fy, xmin, ymin, res):
            return False
    return True


def test_reachability_independent_batch():
    for seed in range(50):
        for n_fires in (1, 2, 3):
            spec = scenegen.random_spec(seed, n_fires=n_fires, n_walls=4, n_debris=4)
            assert _independent_reachability(spec)


def test_debris_too_tall_rejected():
    base = scenegen.random_spec(3, n_fires=2, n_walls=2, n_debris=0)
    bad = DebrisSpec(kind="bump", cx=3.0, cy=0.0, yaw=0.0,
                     size_a=0.3, size_b=0.3, height=0.25)
    spec = SceneSpec(
        name=base.name, bounds=base.bounds, start=base.start, walls=base.walls,
        fires=base.fires, terrain=base.terrain, home=base.home, seed=base.seed,
        debris=(bad,),
    )
    with pytest.raises(ValueError, match="height .* outside"):
        scenegen.validate_spec(spec)


def test_debris_overlaps_wall_rejected():
    base = scenegen.random_spec(5, n_fires=2, n_walls=2, n_debris=0)
    cx, cy, w, h, theta = base.walls[0]
    bad = DebrisSpec(kind="log", cx=cx, cy=cy, yaw=0.0,
                     size_a=0.28, size_b=0.6, height=0.06)
    spec = SceneSpec(
        name=base.name, bounds=base.bounds, start=base.start, walls=base.walls,
        fires=base.fires, terrain=base.terrain, home=base.home, seed=base.seed,
        debris=(bad,),
    )
    with pytest.raises(ValueError, match="overlaps a wall"):
        scenegen.validate_spec(spec)


def _reachability_tuple(spec: SceneSpec) -> tuple:
    grid, xmin, ymin, res = scenegen.occupancy_grid(spec)
    si, sj = scenegen._world_to_cell(spec.start[0], spec.start[1], xmin, ymin, res)
    reachable = scenegen.flood_fill_reachable(grid, (si, sj))
    home_ok = scenegen._goal_reachable(reachable, spec.home[0], spec.home[1], xmin, ymin, res)
    fires_ok = tuple(
        scenegen._fire_reachable(reachable, fx, fy, xmin, ymin, res)
        for fx, fy in spec.fires
    )
    return grid.tobytes(), home_ok, fires_ok


def test_occupancy_invariance_with_debris():
    for seed in range(30):
        spec = scenegen.random_spec(seed, n_fires=2, n_walls=4, n_debris=4, n_tiers=1)
        assert len(spec.debris) >= 1
        stripped = SceneSpec(
            name=spec.name, bounds=spec.bounds, start=spec.start, walls=spec.walls,
            fires=spec.fires, terrain=spec.terrain, home=spec.home, seed=spec.seed,
            debris=(),
        )
        assert _reachability_tuple(spec) == _reachability_tuple(stripped)
        g1, _, _, _ = scenegen.occupancy_grid(spec)
        g2, _, _, _ = scenegen.occupancy_grid(stripped)
        assert np.array_equal(g1, g2)


def test_unreachable_fire_rejected():
    # Fire trapped inside a wall box; start outside.
    spec = SceneSpec(
        name="blocked",
        bounds=(-1.0, 8.0, -4.0, 4.0),
        start=(0.0, 0.0, 0.0),
        walls=((4.0, 0.0, 2.0, 2.0, 0.0),),
        fires=((4.0, 0.0),),
        terrain=None,
        home=(0.5, 0.5),
        seed=0,
    )
    with pytest.raises(ValueError, match="inside inflated wall|unreachable"):
        scenegen.validate_spec(spec)


def _count_named(model: mujoco.MjModel, prefix: str) -> int:
    return sum(1 for i in range(model.ngeom) if model.geom(i).name.startswith(prefix))


def _count_bodies(model: mujoco.MjModel, prefix: str) -> int:
    return sum(1 for i in range(model.nbody) if model.body(i).name.startswith(prefix))


def _debris_geom_count(spec: SceneSpec) -> int:
    n = 0
    for i, d in enumerate(spec.debris):
        if d.kind == "tier":
            n += 2
        else:
            n += 1
    return n


def _debris_geom_names(spec: SceneSpec) -> set[str]:
    names: set[str] = set()
    for i, d in enumerate(spec.debris):
        if d.kind == "tier":
            names.add(f"tier{i}_ramp")
            names.add(f"tier{i}_plat")
        else:
            names.add(f"debris{i}")
    return names


@pytest.mark.parametrize("seed", [0, 1, 5, 11, 23])
def test_mujoco_load(seed):
    spec = scenegen.random_spec(seed, n_fires=2, n_walls=4, n_debris=4, n_tiers=1)
    p12, p29 = scenes.build(spec)
    m12 = mujoco.MjModel.from_xml_path(p12)
    m29 = mujoco.MjModel.from_xml_path(p29)
    assert _count_named(m12, "wall") == len(spec.walls)
    assert _count_named(m12, "fire_base") == len(spec.fires)
    assert _count_bodies(m29, "flame") == len(spec.fires)
    assert _count_named(m29, "wall") == len(spec.walls)
    expected = _debris_geom_count(spec)
    assert _count_named(m12, "debris") + _count_named(m12, "tier") == expected
    assert _count_named(m29, "debris") + _count_named(m29, "tier") == expected
    names12 = {m12.geom(i).name for i in range(m12.ngeom)}
    names29 = {m29.geom(i).name for i in range(m29.ngeom)}
    debris_names = _debris_geom_names(spec)
    assert debris_names <= names12
    assert debris_names <= names29


def test_tier_ramp_orientation_is_radians():
    """The tier ramp must be tilted by theta=atan2(rise, ramp_run) (a few degrees),
    not by a huge wrapped angle. Catches the degrees-vs-radians bug: with the
    compiler in radian mode, emitting math.degrees(theta) over-rotates the ramp.

    We build a deterministic single-tier scene, load the compiled 12-DOF model,
    and read the ramp geom's world orientation. The ramp's local +z (surface
    normal) tilts from world vertical by exactly the ramp pitch, independent of
    yaw, so arccos(R[2,2]) == theta.
    """
    rise, depth, width = 0.12, 1.0, 1.0
    # Mirror scenes._one_debris_geoms tier geometry to get the intended pitch.
    ramp_run = max(depth * 0.55, rise / 0.10)
    theta = math.atan2(rise, ramp_run)
    assert theta < 0.2  # within the blind-policy envelope; tiny tilt.

    tier = DebrisSpec(kind="tier", cx=3.0, cy=0.0, yaw=0.7,
                      size_a=width, size_b=depth, height=rise)
    spec = SceneSpec(
        name="tier_orient",
        bounds=(-1.0, 8.0, -4.0, 4.0),
        start=(0.0, 0.0, 0.0),
        walls=(),
        fires=((4.0, 0.0),),
        terrain=None,
        home=(0.5, 0.5),
        seed=0,
        debris=(tier,),
    )
    p12, _ = scenes.build(spec)
    m12 = mujoco.MjModel.from_xml_path(p12)
    gid = mujoco.mj_name2id(m12, mujoco.mjtObj.mjOBJ_GEOM, "tier0_ramp")
    assert gid >= 0
    # geom_quat -> rotation matrix; local +z is the third column (surface normal).
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m12.geom_quat[gid])
    nz = R.reshape(3, 3)[2, 2]  # z-component of the ramp's local +z axis
    tilt = math.acos(max(-1.0, min(1.0, nz)))
    assert tilt == pytest.approx(theta, abs=0.02), (
        f"ramp tilt {tilt:.4f} rad != intended {theta:.4f} rad "
        "(degrees-vs-radians bug?)")


def test_mujoco_load_with_terrain():
    spec = scenegen.random_spec(777, n_fires=1, n_walls=2, terrain=True)
    p12, p29 = scenes.build(spec)
    m12 = mujoco.MjModel.from_xml_path(p12)
    m29 = mujoco.MjModel.from_xml_path(p29)
    assert m12.nhfield == 1
    assert m29.nhfield == 1
    assert spec.terrain is not None
    hf_id = mujoco.mj_name2id(m12, mujoco.mjtObj.mjOBJ_HFIELD, "terrain_hf")
    assert hf_id >= 0
    assert m12.hfield_size[hf_id][3] == pytest.approx(spec.terrain.elevation)


if __name__ == "__main__":
    test_determinism()
    test_json_round_trip()
    test_json_round_trip_with_terrain()
    for seed in range(50):
        for n_fires in (1, 2, 3, 4):
            for n_walls in (0, 2, 4, 6):
                spec = scenegen.random_spec(seed + n_fires * 100 + n_walls * 10,
                                            n_fires=n_fires, n_walls=n_walls)
                scenegen.validate_spec(spec)
    test_reachability_independent_batch()
    test_unreachable_fire_rejected()
    for seed in (0, 1, 5, 11, 23):
        test_mujoco_load(seed)
    test_mujoco_load_with_terrain()
    print("all tests passed")
