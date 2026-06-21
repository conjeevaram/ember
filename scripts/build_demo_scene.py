#!/usr/bin/env python3
"""Build ``demo_scene``: a small multi-room house for the firefighter demo.

The scene is a single-storey house with four rooms joined by doorways, fires
burning in several of those rooms, gentle uneven ground (a low heightfield),
half-buried "inset" logs, and a scatter of small low debris. Everything stays
inside the blind-policy envelope (short walls relative to the robot, low/rounded
debris, smooth terrain) so the pretrained walker can still move through it.

Outputs:
  - ``scenes/specs/demo_scene.json`` (auto-discovered by ``ember.viewer``)
  - ``g1_demo_scene_demo.xml`` / ``g1_demo_scene29_demo.xml`` in ``G1_MODEL_DIR``
    (only when the robot model dir is available)

Run:
    python scripts/build_demo_scene.py
"""
from __future__ import annotations

import math

import _bootstrap  # noqa: F401  (puts src/ on the path)
import numpy as np

from ember import scenegen, scenes
from ember.config import G1_MODEL_DIR
from ember.scenegen import _validate_one_debris, _debris_too_close
from ember.nav import point_in_any_wall
from ember.spec import DebrisSpec, SceneSpec, TerrainSpec, to_json

# --- house geometry -------------------------------------------------------- #
# House footprint sits in the right two-thirds of the bounds; the robot spawns
# in the open "yard" to the left and walks in through the front door.
BOUNDS = (-2.0, 7.0, -3.5, 3.5)
WALL_T = 0.15            # wall thickness (thin boxes; height comes from scenes.py)
HOUSE_X0, HOUSE_X1 = 1.0, 6.4
HOUSE_Y0, HOUSE_Y1 = -2.8, 2.8
DIV_X = 3.7             # interior vertical divider
DIV_Y = 0.0            # interior horizontal divider
DOOR = 1.35            # doorway clear width (> 2 * robot radius so it stays passable)

START = (-1.0, -1.4, 0.0)   # in the yard, facing +x toward the front door
HOME = (-1.2, 1.2)          # rally point back out in the yard

# Spawn zones kept clear of debris so the robot can be placed here (the stress
# harness spawns at each of these) without a debris-clearance violation. Each
# (x, y) gets ``SPAWN_KEEPOUT`` m of clearance from any debris footprint.
SPAWN_ANCHORS = (
    (-1.0, -1.4), (-1.5, -2.6), (-1.5, 2.6), (-0.2, 0.0),
    (2.3, -1.0), (2.3, 1.0), (5.05, -1.0), (5.05, 1.0),
    HOME,
)
SPAWN_KEEPOUT = 1.6


def _segments(axis: str, fixed: float, lo: float, hi: float,
              gaps: tuple[tuple[float, float], ...]) -> list[tuple[float, float, float, float, float]]:
    """Split a straight wall on ``axis`` into box segments, cutting out ``gaps``.

    ``axis='v'`` is a wall running along y at x=``fixed``; ``axis='h'`` runs along
    x at y=``fixed``. ``gaps`` are ``(center, width)`` intervals (doorways).
    """
    intervals = [(lo, hi)]
    for gc, gw in sorted(gaps):
        g0, g1 = gc - gw / 2, gc + gw / 2
        nxt: list[tuple[float, float]] = []
        for a, b in intervals:
            if g1 <= a or g0 >= b:
                nxt.append((a, b))
                continue
            if g0 > a:
                nxt.append((a, g0))
            if g1 < b:
                nxt.append((g1, b))
        intervals = nxt
    walls = []
    for a, b in intervals:
        if b - a < 1e-6:
            continue
        mid, length = (a + b) / 2, b - a
        if axis == "v":
            walls.append((fixed, mid, WALL_T, length, 0.0))
        else:
            walls.append((mid, fixed, length, WALL_T, 0.0))
    return walls


def _house_walls() -> tuple[tuple[float, float, float, float, float], ...]:
    walls: list[tuple[float, float, float, float, float]] = []
    # Outer shell. Front (left) wall has the entry door at y=-1.4.
    walls += _segments("v", HOUSE_X0, HOUSE_Y0, HOUSE_Y1, gaps=((-1.4, DOOR),))
    walls += _segments("v", HOUSE_X1, HOUSE_Y0, HOUSE_Y1, gaps=())
    walls += _segments("h", HOUSE_Y0, HOUSE_X0, HOUSE_X1, gaps=())
    walls += _segments("h", HOUSE_Y1, HOUSE_X0, HOUSE_X1, gaps=())
    # Interior dividers -> four rooms, each pair joined by a doorway.
    walls += _segments("v", DIV_X, HOUSE_Y0, HOUSE_Y1, gaps=((-1.4, DOOR), (1.4, DOOR)))
    walls += _segments("h", DIV_Y, HOUSE_X0, HOUSE_X1, gaps=((2.3, DOOR), (5.0, DOOR)))
    return tuple(walls)


# Fires burning in the four rooms (pushed toward the outer corners so the
# middle of each room stays open for the robot and the debris).
FIRES = (
    (2.3, 1.95),    # room A (top-left)
    (5.05, 1.95),   # room B (top-right)
    (2.3, -1.95),   # room C (bottom-left, by the entry)
    (5.05, -1.95),  # room D (bottom-right)
)


def _terrain(seed: int) -> TerrainSpec:
    xmin, xmax, ymin, ymax = BOUNDS
    return TerrainSpec(nrow=64, ncol=64,
                       radius_x=(xmax - xmin) / 2, radius_y=(ymax - ymin) / 2,
                       elevation=0.05, seed=seed)


def _scatter_debris(walls, n_logs: int, n_bumps: int, seed: int):
    """Rejection-sample half-buried logs + low bumps across the free floor."""
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = BOUNDS
    debris: list[DebrisSpec] = []
    targets = [("log", n_logs), ("bump", n_bumps)]
    for kind, count in targets:
        placed = 0
        for _ in range(4000):
            if placed >= count:
                break
            cx = float(rng.uniform(xmin + 0.6, xmax - 0.6))
            cy = float(rng.uniform(ymin + 0.6, ymax - 0.6))
            yaw = float(rng.uniform(0, math.pi))
            if point_in_any_wall(cx, cy, walls, 0.25):
                continue
            if any(math.hypot(cx - ax, cy - ay) < SPAWN_KEEPOUT
                   for ax, ay in SPAWN_ANCHORS):
                continue
            if kind == "log":
                d = DebrisSpec(kind="log", cx=cx, cy=cy, yaw=yaw,
                               size_a=float(rng.uniform(0.12, 0.16)),
                               size_b=float(rng.uniform(0.32, 0.5)),
                               height=float(rng.uniform(0.05, 0.07)))
            else:
                r = float(rng.uniform(0.2, 0.34))
                d = DebrisSpec(kind="bump", cx=cx, cy=cy, yaw=yaw,
                               size_a=r, size_b=float(rng.uniform(0.2, 0.34)),
                               height=float(rng.uniform(0.04, 0.08)))
            if any(_debris_too_close(d, e) for e in debris):
                continue
            try:
                _validate_one_debris(d, len(debris), BOUNDS, walls,
                                     (START[0], START[1]), FIRES)
            except ValueError:
                continue
            debris.append(d)
            placed += 1
    return tuple(debris)


def build_spec(seed: int = 7) -> SceneSpec:
    walls = _house_walls()
    debris = _scatter_debris(walls, n_logs=5, n_bumps=5, seed=seed)
    spec = SceneSpec(
        name="demo_scene",
        bounds=BOUNDS,
        start=START,
        walls=walls,
        fires=FIRES,
        terrain=_terrain(seed),
        home=HOME,
        seed=seed,
        debris=debris,
    )
    scenegen.validate_spec(spec)  # raises on any invariant violation
    return spec


def main() -> None:
    from pathlib import Path

    spec = build_spec()
    specs_dir = Path(__file__).resolve().parent.parent / "scenes" / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    json_path = specs_dir / f"{spec.name}.json"
    to_json(spec, json_path)
    print(f"validated demo_scene: {len(spec.walls)} wall segments, "
          f"{len(spec.fires)} fires, {len(spec.debris)} debris")
    print(f"JSON: {json_path}")
    if G1_MODEL_DIR.exists():
        p12, p29 = scenes.build(spec)
        print(f"XML:\n  {p12}\n  {p29}")
    else:
        print(f"(skipped XML: model dir not found at {G1_MODEL_DIR}; "
              f"the viewer will build it on load)")


if __name__ == "__main__":
    main()
