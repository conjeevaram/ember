"""Procedural SceneSpec generation and validation (Phase 1).

Deterministic given ``seed``; rejects specs that violate navigation or placement
invariants. Occupancy / reachability uses :mod:`ember.nav` (single source of truth).
"""
from __future__ import annotations

import math

import numpy as np

from .nav import (
    NAV_INFLATION,
    NAV_RES,
    SPRAY_STANDOFF,
    cell_in_grid as _cell_in_grid,
    flood_fill_reachable,
    occupancy_grid,
    point_in_any_wall,
    point_in_wall,
    world_to_cell as _world_to_cell,
)
from .spec import DebrisSpec, SceneSpec, TerrainSpec

ROBOT_RADIUS = NAV_INFLATION
BOUNDS_MARGIN = 0.45          # start/home/fires must stay inside bounds by this much
MIN_FIRE_SPACING = 1.0        # distinct fire targets (m)
WALL_HEIGHT = 0.8             # tall enough that the blind walker cannot step over
TERRAIN_MAX_ELEVATION = 0.08  # blind-policy envelope; steeper cells become obstacles
DEBRIS_MAX_HEIGHT = 0.12      # max exposed height / tier rise (m)
DEBRIS_START_CLEARANCE = 0.85
DEBRIS_FIRE_CLEARANCE = 0.55
DEBRIS_MIN_SPACING = 0.45
DEBRIS_KINDS = ("log", "bump", "tier")
MAX_GENERATION_ATTEMPTS = 2000

_DEFAULT_BOUNDS = (-1.0, 8.0, -4.0, 4.0)


def _inside_bounds(x: float, y: float, bounds: tuple[float, float, float, float],
                   margin: float) -> bool:
    xmin, xmax, ymin, ymax = bounds
    return (xmin + margin <= x <= xmax - margin and
            ymin + margin <= y <= ymax - margin)


def wall_corners(wall: tuple[float, float, float, float, float]) -> list[tuple[float, float]]:
    cx, cy, w, h, theta = wall
    c, s = math.cos(theta), math.sin(theta)
    corners = []
    for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        lx, ly = sx * w / 2, sy * h / 2
        wx = cx + c * lx - s * ly
        wy = cy + s * lx + c * ly
        corners.append((wx, wy))
    return corners


def _goal_reachable(reachable: np.ndarray, gx: float, gy: float,
                    xmin: float, ymin: float, res: float) -> bool:
    gi, gj = _world_to_cell(gx, gy, xmin, ymin, res)
    if not _cell_in_grid(gi, gj, reachable):
        return False
    return bool(reachable[gj, gi])


def _fire_reachable(reachable: np.ndarray, fx: float, fy: float,
                    xmin: float, ymin: float, res: float,
                    standoff: float = SPRAY_STANDOFF) -> bool:
    """True if some free cell within ``standoff`` of the fire connects to the start."""
    radius_cells = int(math.ceil(standoff / res)) + 1
    fi, fj = _world_to_cell(fx, fy, xmin, ymin, res)
    for dj in range(-radius_cells, radius_cells + 1):
        for di in range(-radius_cells, radius_cells + 1):
            ci, cj = fi + di, fj + dj
            if not _cell_in_grid(ci, cj, reachable):
                continue
            wx = xmin + (ci + 0.5) * res
            wy = ymin + (cj + 0.5) * res
            if math.hypot(wx - fx, wy - fy) > standoff:
                continue
            if reachable[cj, ci]:
                return True
    return False


def _tier_ramp_run(depth: float, rise: float) -> float:
    return max(depth * 0.55, rise / 0.10)


def debris_footprint_radius(d: DebrisSpec) -> float:
    """Conservative clearance radius for placement / overlap checks."""
    if d.kind == "log":
        return d.size_b + d.size_a
    if d.kind == "bump":
        return max(d.size_a, d.size_b)
    if d.kind == "tier":
        ramp_run = _tier_ramp_run(d.size_b, d.height)
        half_extent = d.size_b * 0.45 / 2 + ramp_run
        return math.hypot(d.size_a / 2, half_extent)
    raise ValueError(f"unknown debris kind: {d.kind!r}")


def debris_overlaps_wall(d: DebrisSpec,
                         walls: tuple[tuple[float, float, float, float, float], ...],
                         inflation: float = 0.05) -> bool:
    """True if debris footprint intersects any wall (sharp-edge guard)."""
    r = debris_footprint_radius(d)
    n = max(8, int(math.ceil(2 * math.pi * r / 0.15)))
    for k in range(n):
        ang = 2 * math.pi * k / n
        px = d.cx + r * math.cos(ang)
        py = d.cy + r * math.sin(ang)
        if point_in_any_wall(px, py, walls, inflation):
            return True
    return point_in_any_wall(d.cx, d.cy, walls, inflation + r * 0.25)


def _debris_too_close(a: DebrisSpec, b: DebrisSpec) -> bool:
    d = math.hypot(a.cx - b.cx, a.cy - b.cy)
    return d < debris_footprint_radius(a) + debris_footprint_radius(b) + DEBRIS_MIN_SPACING


def _validate_one_debris(d: DebrisSpec, index: int,
                         bounds: tuple[float, float, float, float],
                         walls: tuple[tuple[float, float, float, float, float], ...],
                         start_xy: tuple[float, float],
                         fires: tuple[tuple[float, float], ...]) -> None:
    if d.kind not in DEBRIS_KINDS:
        raise ValueError(f"debris {index}: unknown kind {d.kind!r}")
    if d.height <= 0 or d.height > DEBRIS_MAX_HEIGHT:
        raise ValueError(
            f"debris {index}: height {d.height:.3f} m outside (0, {DEBRIS_MAX_HEIGHT}]")
    if d.size_a <= 0 or d.size_b <= 0:
        raise ValueError(f"debris {index}: size_a/size_b must be positive")
    if d.kind == "log" and d.height > 0.08:
        raise ValueError(
            f"debris {index}: log expose {d.height:.3f} m > 0.08 m (blind-policy envelope)")
    r = debris_footprint_radius(d)
    if not _inside_bounds(d.cx, d.cy, bounds, r + 0.05):
        raise ValueError(
            f"debris {index} ({d.cx:.3f}, {d.cy:.3f}) outside bounds (footprint r={r:.2f} m)")
    if debris_overlaps_wall(d, walls):
        raise ValueError(f"debris {index} ({d.cx:.3f}, {d.cy:.3f}) overlaps a wall")
    sx, sy = start_xy
    if math.hypot(d.cx - sx, d.cy - sy) < DEBRIS_START_CLEARANCE + r:
        raise ValueError(
            f"debris {index} too close to start "
            f"({math.hypot(d.cx - sx, d.cy - sy):.3f} m < {DEBRIS_START_CLEARANCE + r:.3f} m)")
    for fi, (fx, fy) in enumerate(fires):
        if math.hypot(d.cx - fx, d.cy - fy) < DEBRIS_FIRE_CLEARANCE + r:
            raise ValueError(
                f"debris {index} too close to fire {fi} "
                f"({math.hypot(d.cx - fx, d.cy - fy):.3f} m)")


def _sample_debris(rng: np.random.Generator,
                   start: tuple[float, float, float],
                   fires: tuple[tuple[float, float], ...],
                   walls: tuple[tuple[float, float, float, float, float], ...],
                   bounds: tuple[float, float, float, float],
                   n_debris: int,
                   n_tiers: int) -> tuple[DebrisSpec, ...] | None:
    """Scatter debris along start→fire corridors; ``None`` if placement fails."""
    if n_debris <= 0:
        return ()
    sx, sy = start[0], start[1]
    debris: list[DebrisSpec] = []
    tier_quota = min(n_tiers, n_debris)
    for idx in range(n_debris):
        kind = "tier" if idx < tier_quota else str(rng.choice(["log", "bump"]))
        placed = False
        for _ in range(120):
            fi = int(rng.integers(0, len(fires)))
            fx, fy = fires[fi]
            t = float(rng.uniform(0.18, 0.82))
            bx = sx + t * (fx - sx)
            by = sy + t * (fy - sy)
            seg_len = math.hypot(fx - sx, fy - sy) or 1.0
            nx, ny = -(fy - sy) / seg_len, (fx - sx) / seg_len
            jitter = float(rng.uniform(-0.9, 0.9))
            cx = bx + nx * jitter
            cy = by + ny * jitter
            yaw = float(rng.uniform(0, math.pi))
            if kind == "log":
                size_a = float(rng.uniform(0.22, 0.32))
                size_b = float(rng.uniform(0.45, 0.85))
                height = float(rng.uniform(0.04, 0.07))
            elif kind == "bump":
                size_a = float(rng.uniform(0.25, 0.45))
                size_b = float(rng.uniform(0.25, 0.45))
                height = float(rng.uniform(0.04, 0.09))
            else:
                size_a = float(rng.uniform(0.9, 1.3))
                size_b = float(rng.uniform(0.9, 1.4))
                height = float(rng.uniform(0.08, DEBRIS_MAX_HEIGHT))
                yaw = math.atan2(fy - sy, fx - sx)
            d = DebrisSpec(kind=kind, cx=cx, cy=cy, yaw=yaw,
                           size_a=size_a, size_b=size_b, height=height)
            if any(_debris_too_close(d, existing) for existing in debris):
                continue
            try:
                _validate_one_debris(d, idx, bounds, walls, (sx, sy), fires)
            except ValueError:
                continue
            debris.append(d)
            placed = True
            break
        if not placed:
            return None
    return tuple(debris)


def validate_spec(spec: SceneSpec) -> None:
    """Raise ``ValueError`` with a precise message on any invariant violation."""
    bounds = spec.bounds
    xmin, xmax, ymin, ymax = bounds
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"invalid bounds {bounds}: xmin<xmax and ymin<ymax required")

    for label, x, y in (("start", spec.start[0], spec.start[1]),
                        ("home", spec.home[0], spec.home[1])):
        if not _inside_bounds(x, y, bounds, BOUNDS_MARGIN):
            raise ValueError(
                f"{label} ({x:.3f}, {y:.3f}) outside bounds {bounds} "
                f"(margin {BOUNDS_MARGIN} m)")

    for i, (fx, fy) in enumerate(spec.fires):
        if not _inside_bounds(fx, fy, bounds, BOUNDS_MARGIN):
            raise ValueError(
                f"fire {i} ({fx:.3f}, {fy:.3f}) outside bounds {bounds} "
                f"(margin {BOUNDS_MARGIN} m)")

    for label, x, y in (("start", spec.start[0], spec.start[1]),
                        ("home", spec.home[0], spec.home[1])):
        if point_in_any_wall(x, y, spec.walls, ROBOT_RADIUS):
            raise ValueError(f"{label} ({x:.3f}, {y:.3f}) inside inflated wall")

    for i, (fx, fy) in enumerate(spec.fires):
        if point_in_any_wall(fx, fy, spec.walls, ROBOT_RADIUS):
            raise ValueError(f"fire {i} ({fx:.3f}, {fy:.3f}) inside inflated wall")

    fires = spec.fires
    for a in range(len(fires)):
        for b in range(a + 1, len(fires)):
            d = math.hypot(fires[a][0] - fires[b][0], fires[a][1] - fires[b][1])
            if d < MIN_FIRE_SPACING:
                raise ValueError(
                    f"fires {a} and {b} too close ({d:.3f} m < {MIN_FIRE_SPACING} m)")

    for i, wall in enumerate(spec.walls):
        for cx, cy in wall_corners(wall):
            if not _inside_bounds(cx, cy, bounds, 0.05):
                raise ValueError(f"wall {i} corner ({cx:.3f}, {cy:.3f}) outside bounds")

    if spec.terrain is not None:
        t = spec.terrain
        if t.elevation <= 0 or t.elevation > TERRAIN_MAX_ELEVATION:
            raise ValueError(
                f"terrain elevation {t.elevation} m outside (0, {TERRAIN_MAX_ELEVATION}]")
        if t.nrow < 2 or t.ncol < 2:
            raise ValueError(f"terrain grid too small: nrow={t.nrow}, ncol={t.ncol}")
        if t.radius_x <= 0 or t.radius_y <= 0:
            raise ValueError(f"terrain radii must be positive: {t.radius_x}, {t.radius_y}")

    for i, d in enumerate(spec.debris):
        _validate_one_debris(d, i, bounds, spec.walls,
                             (spec.start[0], spec.start[1]), spec.fires)
    for a in range(len(spec.debris)):
        for b in range(a + 1, len(spec.debris)):
            if _debris_too_close(spec.debris[a], spec.debris[b]):
                raise ValueError(
                    f"debris {a} and {b} too close "
                    f"({math.hypot(spec.debris[a].cx - spec.debris[b].cx, spec.debris[a].cy - spec.debris[b].cy):.3f} m)")

    grid, gxmin, gymin, res = occupancy_grid(spec)
    si, sj = _world_to_cell(spec.start[0], spec.start[1], gxmin, gymin, res)
    if not _cell_in_grid(si, sj, grid):
        raise ValueError(f"start cell ({si}, {sj}) outside occupancy grid")
    if grid[sj, si]:
        raise ValueError("start lies inside occupied cell")

    reachable = flood_fill_reachable(grid, (si, sj))

    if not _goal_reachable(reachable, spec.home[0], spec.home[1], gxmin, gymin, res):
        raise ValueError(
            f"home ({spec.home[0]:.3f}, {spec.home[1]:.3f}) unreachable from start")

    for i, (fx, fy) in enumerate(spec.fires):
        if not _fire_reachable(reachable, fx, fy, gxmin, gymin, res):
            raise ValueError(
                f"fire {i} ({fx:.3f}, {fy:.3f}) unreachable "
                f"(no standoff cell within {SPRAY_STANDOFF} m connected to start)")


def _sample_wall(rng: np.random.Generator,
                 bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float, float]:
    xmin, xmax, ymin, ymax = bounds
    cx = rng.uniform(xmin + 1.0, xmax - 1.0)
    cy = rng.uniform(ymin + 0.8, ymax - 0.8)
    w = rng.uniform(0.6, 2.0)
    h = rng.uniform(0.4, 1.2)
    theta = rng.uniform(0, math.pi)
    return (cx, cy, w, h, theta)


def _sample_point(rng: np.random.Generator,
                  bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = bounds
    x = rng.uniform(xmin + BOUNDS_MARGIN, xmax - BOUNDS_MARGIN)
    y = rng.uniform(ymin + BOUNDS_MARGIN, ymax - BOUNDS_MARGIN)
    return (x, y)


def _make_terrain(rng: np.random.Generator, bounds: tuple[float, float, float, float],
                    seed: int) -> TerrainSpec:
    xmin, xmax, ymin, ymax = bounds
    radius_x = (xmax - xmin) / 2
    radius_y = (ymax - ymin) / 2
    elevation = float(rng.uniform(0.02, TERRAIN_MAX_ELEVATION))
    nrow = ncol = 64
    return TerrainSpec(nrow=nrow, ncol=ncol, radius_x=radius_x, radius_y=radius_y,
                       elevation=elevation, seed=seed)


def random_spec(seed: int, n_fires: int = 2, n_walls: int = 4, terrain: bool = False,
                n_debris: int = 4, n_tiers: int = 1,
                bounds: tuple[float, float, float, float] = _DEFAULT_BOUNDS,
                name: str | None = None) -> SceneSpec:
    """Deterministic procedural spec. Rejection-samples until ``validate_spec`` passes.

    Defaults: ``bounds=(-1, 8, -4, 4)``, start/home on the left third, walls 0.6–2.0 m
    wide, fires spread with >= 1 m spacing, terrain off unless requested, ~4 debris
    (1 tier + logs/bumps) scattered along start→fire corridors.
    """
    if n_fires < 1:
        raise ValueError("n_fires must be >= 1")
    if n_walls < 0:
        raise ValueError("n_walls must be >= 0")
    if n_debris < 0:
        raise ValueError("n_debris must be >= 0")
    if n_tiers < 0:
        raise ValueError("n_tiers must be >= 0")

    xmin, xmax, ymin, ymax = bounds
    scene_name = name or f"proc_{seed:05d}"

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng(seed * 100_000 + attempt)
        sx = rng.uniform(xmin + BOUNDS_MARGIN, xmin + (xmax - xmin) * 0.25)
        sy = rng.uniform(ymin + BOUNDS_MARGIN, ymax - BOUNDS_MARGIN)
        yaw = float(rng.uniform(-0.3, 0.3))
        start = (float(sx), float(sy), yaw)

        hx = float(rng.uniform(xmin + BOUNDS_MARGIN, xmin + (xmax - xmin) * 0.35))
        hy = float(rng.uniform(ymin + BOUNDS_MARGIN, ymax - BOUNDS_MARGIN))
        home = (hx, hy)

        walls = tuple(_sample_wall(rng, bounds) for _ in range(n_walls))

        fires: list[tuple[float, float]] = []
        fire_attempts = 0
        while len(fires) < n_fires and fire_attempts < 500:
            fire_attempts += 1
            fx, fy = _sample_point(rng, bounds)
            if point_in_any_wall(fx, fy, walls, ROBOT_RADIUS):
                continue
            if any(math.hypot(fx - ox, fy - oy) < MIN_FIRE_SPACING for ox, oy in fires):
                continue
            fires.append((fx, fy))

        if len(fires) < n_fires:
            continue

        debris = _sample_debris(rng, start, tuple(fires), walls, bounds, n_debris, n_tiers)
        if debris is None:
            continue

        terr = _make_terrain(rng, bounds, seed) if terrain else None

        spec = SceneSpec(
            name=scene_name,
            bounds=bounds,
            start=start,
            walls=walls,
            fires=tuple(fires),
            terrain=terr,
            home=home,
            seed=seed,
            debris=debris,
        )
        try:
            validate_spec(spec)
        except ValueError:
            continue
        return spec

    raise RuntimeError(
        f"could not generate valid SceneSpec after {MAX_GENERATION_ATTEMPTS} attempts "
        f"(seed={seed}, n_fires={n_fires}, n_walls={n_walls}, terrain={terrain}, "
        f"n_debris={n_debris}, n_tiers={n_tiers})")
