"""Phase 2–3 navigation: costmap, A*, approach controller, reachability."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from ember import scenegen
from ember.config import G1_MODEL_DIR, quat_yaw
from ember.nav import (
    ApproachController,
    Costmap,
    FACE_TOL,
    NAV_INFLATION,
    NAV_RES,
    SPRAY_STANDOFF,
    astar,
    best_standoff,
    line_of_sight,
    nearest_burning_fire,
    point_in_any_wall,
    safe_spray_point,
    standoff_point,
    WaypointFollower,
)
from ember.spec import SceneSpec, from_json

pytestmark = pytest.mark.skipif(
    not G1_MODEL_DIR.exists(),
    reason=f"G1 model dir missing: {G1_MODEL_DIR}",
)

_SPECS_DIR = Path(__file__).resolve().parent.parent / "scenes" / "specs"


def _hand_gap_costmap() -> Costmap:
    """9x9 grid with a vertical wall (column 4) and a gap at rows 3–5."""
    grid = np.zeros((9, 9), dtype=bool)
    for j in range(9):
        if j not in (3, 4, 5):
            grid[j, 4] = True
    return Costmap(grid, 0.0, 0.0, 1.0)


def test_costmap_walls_and_debris():
    spec = scenegen.random_spec(11, n_fires=2, n_walls=4, n_debris=4)
    cm = Costmap.from_spec(spec)
    xmin, xmax, ymin, ymax = spec.bounds
    for j in range(cm.shape[0]):
        for i in range(cm.shape[1]):
            wx = xmin + (i + 0.5) * cm.res
            wy = ymin + (j + 0.5) * cm.res
            wall_occ = point_in_any_wall(wx, wy, spec.walls, NAV_INFLATION)
            assert cm.is_free(i, j) == (not wall_occ)
    # Debris must not mark cells occupied.
    for d in spec.debris:
        ci, cj = cm.world_to_cell(d.cx, d.cy)
        if cm.in_bounds(ci, cj):
            assert cm.is_free(ci, cj)


def test_costmap_world_cell_roundtrip():
    spec = scenegen.random_spec(5, n_fires=1, n_walls=2)
    cm = Costmap.from_spec(spec)
    wx, wy = spec.start[0], spec.start[1]
    i, j = cm.world_to_cell(wx, wy)
    cx, cy = cm.cell_to_world(i, j)
    assert abs(cx - wx) <= cm.res / 2
    assert abs(cy - wy) <= cm.res / 2
    assert cm.is_free_world(cx, cy) == cm.is_free(i, j)


def test_astar_gap_no_corner_cut():
    cm = _hand_gap_costmap()
    path = astar(cm, (0.5, 0.5), (8.5, 0.5))
    assert len(path) >= 2
    assert path[0] == (0.5, 0.5)
    assert path[-1] == (8.5, 0.5)
    for wx, wy in path:
        i, j = cm.world_to_cell(wx, wy)
        assert cm.is_free(i, j)
    for k in range(len(path) - 1):
        assert line_of_sight(cm, path[k][0], path[k][1], path[k + 1][0], path[k + 1][1])
    assert any(3.0 <= p[1] <= 6.0 for p in path)


def test_astar_reachability_batch():
    """Every scenegen-validated fire/home must be pathable by astar."""
    failures = []
    for seed in range(30):
        spec = scenegen.random_spec(
            seed + 1000, n_fires=2 + seed % 3, n_walls=2 + seed % 4, n_debris=3)
        cm = Costmap.from_spec(spec)
        sx, sy = spec.start[0], spec.start[1]
        path_home = astar(cm, (sx, sy), spec.home)
        if not path_home:
            failures.append((seed, "home"))
        for fi, (fx, fy) in enumerate(spec.fires):
            gx, gy = standoff_point(fx, fy, sx, sy, SPRAY_STANDOFF)
            path = astar(cm, (sx, sy), (gx, gy))
            if not path:
                failures.append((seed, f"fire{fi}"))
    assert failures == [], f"astar failed: {failures}"


@pytest.mark.skipif(
    not G1_MODEL_DIR.exists(),
    reason="needs MuJoCo model",
)
def test_waypoint_follower_headless():
    """Plan A* and drive with WaypointFollower in a headless sim loop."""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from ember import scenes
    from ember.sim import G1Sim

    spec = scenegen.random_spec(2024, n_fires=1, n_walls=2, n_debris=0)
    scenes.build(spec)
    name_12, name_29 = scenes.spec_scene_names(spec.name)
    sim = G1Sim(
        scene_path=str(G1_MODEL_DIR / name_12),
        overlay_scene=str(G1_MODEL_DIR / name_29),
        spec=spec,
    )
    sim.heading_hold = False
    cm = Costmap.from_spec(spec)
    x0, y0 = spec.start[0], spec.start[1]
    tf = 0
    fx, fy = spec.fires[tf]
    gx, gy = standoff_point(fx, fy, x0, y0, SPRAY_STANDOFF)
    path = astar(cm, (x0, y0), (gx, gy))
    assert path, "expected non-empty path"
    follower = WaypointFollower(sim, path, goal_tol=0.45, cruise_vx=0.5)
    sim.navigator = follower
    max_steps = 4000
    done = False
    for _ in range(max_steps):
        dec = sim.control_decimation
        for _ in range(dec):
            from ember.sim import pd_control
            leg_tau = pd_control(
                sim.target_dof_pos, sim.data.qpos[sim.leg_qadr],
                sim.kps, np.zeros_like(sim.kds),
                sim.data.qvel[sim.leg_vadr], sim.kds,
            )
            sim.data.ctrl[sim.leg_act] = leg_tau
            import mujoco
            mujoco.mj_step(sim.model, sim.data)
        follower.update()
        sim._compute_action()
        if sim.data.qpos[2] < 0.35:
            break
        x, y = float(sim.data.qpos[0]), float(sim.data.qpos[1])
        if math.hypot(x - gx, y - gy) < 0.45:
            done = True
            break
    x, y = float(sim.data.qpos[0]), float(sim.data.qpos[1])
    dist = math.hypot(x - gx, y - gy)
    assert not sim.data.qpos[2] < 0.35, "robot fell"
    assert dist < 1.2 or done, f"final dist {dist:.2f} m, done={done}"
    progress = math.hypot(x - x0, y - y0)
    assert progress > 0.5, f"insufficient progress ({progress:.2f} m)"


def test_scenegen_reexports_match_nav():
    assert scenegen.NAV_RES == NAV_RES
    assert scenegen.NAV_INFLATION == NAV_INFLATION
    spec = scenegen.random_spec(3, n_fires=1, n_walls=2)
    g1, x1, y1, r1 = scenegen.occupancy_grid(spec)
    cm = Costmap.from_spec(spec)
    assert g1.shape == cm.occupied.shape
    assert np.array_equal(g1, cm.occupied)
    assert x1 == cm.xmin and y1 == cm.ymin and r1 == cm.res


def _all_test_specs() -> list[SceneSpec]:
    specs: list[SceneSpec] = []
    if _SPECS_DIR.is_dir():
        for path in sorted(_SPECS_DIR.glob("*.json")):
            specs.append(from_json(path))
    for seed in range(20):
        specs.append(scenegen.random_spec(seed + 5000, n_fires=2 + seed % 2,
                                          n_walls=2 + seed % 3, n_debris=2))
    return specs


def test_best_standoff_all_specs():
    failures = []
    for spec in _all_test_specs():
        cm = Costmap.from_spec(spec)
        sx, sy = spec.start[0], spec.start[1]
        for fi, (fx, fy) in enumerate(spec.fires):
            pt = best_standoff(cm, fx, fy, sx, sy)
            if pt is None:
                failures.append((spec.name, fi, "none"))
                continue
            if not cm.is_free_world(pt[0], pt[1]):
                failures.append((spec.name, fi, "occupied"))
            path = astar(cm, (sx, sy), pt)
            if not path:
                failures.append((spec.name, fi, "no_path"))
            dist = math.hypot(pt[0] - fx, pt[1] - fy)
            if abs(dist - SPRAY_STANDOFF) > cm.res * 2:
                failures.append((spec.name, fi, f"dist={dist:.2f}"))
    assert failures == [], f"best_standoff failed: {failures[:10]}"


def test_nearest_burning_fire_selection():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from ember import scenes
    from ember.sim import G1Sim

    spec = SceneSpec(
        name="fire_pick",
        bounds=(-1.0, 8.0, -4.0, 4.0),
        start=(0.0, 0.0, 0.0),
        walls=(),
        fires=((5.0, 0.0), (1.0, 0.0), (3.0, 3.0)),
        terrain=None,
        home=(0.5, 0.5),
        seed=0,
    )
    scenes.build(spec)
    name_12, name_29 = scenes.spec_scene_names(spec.name)
    sim = G1Sim(
        scene_path=str(G1_MODEL_DIR / name_12),
        overlay_scene=str(G1_MODEL_DIR / name_29),
        spec=spec,
    )
    assert sim.targeted_fire == 1
    sim.fire_health[1] = 0.0
    sim._advance_targeted_fire()
    assert sim.targeted_fire == 2


def test_stuck_detector():
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from ember import scenes
    from ember.sim import G1Sim, pd_control

    spec = SceneSpec(
        name="wall_block",
        bounds=(-1.0, 8.0, -4.0, 4.0),
        start=(0.0, 0.0, 0.0),
        walls=(
            (0.6, 0.0, 0.3, 6.0, 0.0),
            (0.3, 1.5, 0.3, 3.0, 0.0),
            (0.3, -1.5, 0.3, 3.0, 0.0),
        ),
        fires=((5.0, 0.0),),
        terrain=None,
        home=(0.5, 0.5),
        seed=0,
    )
    scenes.build(spec)
    name_12, name_29 = scenes.spec_scene_names(spec.name)
    sim = G1Sim(
        scene_path=str(G1_MODEL_DIR / name_12),
        overlay_scene=str(G1_MODEL_DIR / name_29),
        spec=spec,
    )
    sim.heading_hold = False
    sim.set_command(0.8, 0.0, 0.0)
    dec = sim.control_decimation
    steps = int(2.5 / (dec * sim.sim_dt)) + 1
    for _ in range(steps):
        for _ in range(dec):
            leg_tau = pd_control(
                sim.target_dof_pos, sim.data.qpos[sim.leg_qadr],
                sim.kps, np.zeros_like(sim.kds),
                sim.data.qvel[sim.leg_vadr], sim.kds,
            )
            sim.data.ctrl[sim.leg_act] = leg_tau
            mujoco.mj_step(sim.model, sim.data)
            sim.counter += 1
        sim._update_blocked()
        sim._compute_action()
    assert sim.get_state()["blocked"]
    sim.set_command(0.0, 0.0, 0.0)
    sim._update_blocked()
    assert not sim.get_state()["blocked"]


def test_nav_preview_path_goal():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from ember import scenes
    from ember.sim import G1Sim

    spec = scenegen.random_spec(7777, n_fires=2, n_walls=3, n_debris=0)
    scenes.build(spec)
    name_12, name_29 = scenes.spec_scene_names(spec.name)
    sim = G1Sim(
        scene_path=str(G1_MODEL_DIR / name_12),
        overlay_scene=str(G1_MODEL_DIR / name_29),
        spec=spec,
    )
    snap = sim.nav_snapshot()
    assert snap is not None
    path = snap["path"]
    assert len(path) >= 2
    tf = snap["targeted_fire"]
    fx, fy = spec.fires[tf]
    gx, gy = path[-1]
    dist = math.hypot(gx - fx, gy - fy)
    assert abs(dist - SPRAY_STANDOFF) <= NAV_RES * 2 + 0.05


def test_nearest_burning_fire_helper():
    fires = ((5.0, 0.0), (1.0, 0.0), (3.0, 3.0))
    health = [1.0, 0.0, 1.0]
    assert nearest_burning_fire(fires, health, 0.0, 0.0) == 2
    assert nearest_burning_fire(fires, [0.0, 0.0, 0.0], 0.0, 0.0) is None


def _sim_step(sim):
    from ember.sim import pd_control
    import mujoco

    dec = sim.control_decimation
    for _ in range(dec):
        leg_tau = pd_control(
            sim.target_dof_pos, sim.data.qpos[sim.leg_qadr],
            sim.kps, np.zeros_like(sim.kds),
            sim.data.qvel[sim.leg_vadr], sim.kds,
        )
        sim.data.ctrl[sim.leg_act] = leg_tau
        mujoco.mj_step(sim.model, sim.data)
    sim._compute_action()


def _proc_sim(seed: int = 2024):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from ember import scenes
    from ember.sim import G1Sim

    spec = scenegen.random_spec(seed, n_fires=1, n_walls=2, n_debris=0)
    scenes.build(spec)
    name_12, name_29 = scenes.spec_scene_names(spec.name)
    return G1Sim(
        scene_path=str(G1_MODEL_DIR / name_12),
        overlay_scene=str(G1_MODEL_DIR / name_29),
        spec=spec,
    ), spec


def test_safe_spray_point_all_specs():
    failures = []
    for spec in _all_test_specs():
        cm = Costmap.from_spec(spec)
        sx, sy = spec.start[0], spec.start[1]
        health = [1.0] * len(spec.fires)
        tf = nearest_burning_fire(spec.fires, health, sx, sy)
        if tf is None:
            continue
        fx, fy = spec.fires[tf]
        pt = safe_spray_point(cm, (fx, fy), spec.fires, (sx, sy), target_idx=tf)
        if pt is None:
            failures.append((spec.name, "none"))
            continue
        gx, gy = pt
        if not cm.is_free_world(gx, gy):
            failures.append((spec.name, "occupied"))
        path = astar(cm, (sx, sy), pt)
        if not path:
            failures.append((spec.name, "no_path"))
        for i, (ofx, ofy) in enumerate(spec.fires):
            if i == tf:
                continue
            if math.hypot(ofx - gx, ofy - gy) < 1.2 - 1e-6:
                failures.append((spec.name, f"fire_clearance{i}"))
        dist = math.hypot(gx - fx, gy - fy)
        if abs(dist - SPRAY_STANDOFF) > cm.res * 2 + 0.05:
            failures.append((spec.name, f"dist={dist:.2f}"))
    assert failures == [], f"safe_spray_point failed: {failures[:10]}"


@pytest.mark.skipif(
    not G1_MODEL_DIR.exists(),
    reason="needs MuJoCo model",
)
def test_approach_controller_headless():
    sim, spec = _proc_sim(3030)
    sim.heading_hold = False
    assert sim.set_approach(True)
    ctrl = sim.navigator
    assert isinstance(ctrl, ApproachController)
    tf = sim.targeted_fire
    fx, fy = spec.fires[tf]
    max_steps = 6000
    ready = False
    for _ in range(max_steps):
        _sim_step(sim)
        sim._update_blocked()
        if sim.navigator is not None:
            if sim.navigator.update():
                ready = True
                break
        if sim.data.qpos[2] < 0.35:
            break
    assert ready, "ApproachController did not reach READY"
    assert sim.data.qpos[2] > 0.4, "robot fell"
    x, y = float(sim.data.qpos[0]), float(sim.data.qpos[1])
    yaw = quat_yaw(sim.data.qpos[3:7])
    heading_target = math.atan2(fy - y, fx - x)
    heading_err = math.atan2(math.sin(heading_target - yaw),
                             math.cos(heading_target - yaw))
    assert abs(heading_err) < FACE_TOL + 0.08
    assert abs(float(sim.cmd[0])) < 0.15
    assert not point_in_any_wall(x, y, spec.walls)


@pytest.mark.skipif(
    not G1_MODEL_DIR.exists(),
    reason="needs MuJoCo model",
)
def test_proc_heading_hold_no_spin():
    sim, _spec = _proc_sim(4040)
    sim.heading_hold = True
    sim._manual_yaw = False
    sim.set_command(0.0, 0.0, 0.0)
    yaw0 = quat_yaw(sim.data.qpos[3:7])
    dec = sim.control_decimation
    steps = int(2.0 / (dec * sim.sim_dt)) + 1
    max_yaw_cmd = 0.0
    for _ in range(steps):
        _sim_step(sim)
        sim._update_blocked()
        if sim.heading_hold and not sim._manual_yaw:
            sim.cmd[2] = sim._steer()
        max_yaw_cmd = max(max_yaw_cmd, abs(float(sim.cmd[2])))
    yaw1 = quat_yaw(sim.data.qpos[3:7])
    d_yaw = abs(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))
    assert max_yaw_cmd < 0.05, f"yaw command should stay ~0, got {max_yaw_cmd}"
    assert d_yaw < 0.25, f"robot should not spin, d_yaw={d_yaw:.2f}"


def test_named_scene_steer_unchanged():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from ember import scenes
    from ember.sim import G1Sim, SCENES

    scenes.ensure_scenes()
    sim = G1Sim(scene_path=SCENES["flat"])
    sim.heading_target = 0.5
    sim.lateral_target = 0.0
    sim.data.qpos[1] = 0.3
    yaw_cmd = sim._steer()
    assert abs(yaw_cmd) > 0.05
