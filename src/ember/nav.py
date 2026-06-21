"""Navigation layer: occupancy costmap, A*, and waypoint following (Phase 2).

Single source of truth for grid resolution, wall inflation, and occupancy queries.
``scenegen`` imports from here so reachability validation matches ``astar``.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from .config import VX_RANGE, YAW_RANGE, clamp, quat_yaw
from .spec import SceneSpec, TerrainSpec

if TYPE_CHECKING:
    from .sim import G1Sim

NAV_RES = 0.10
NAV_INFLATION = 0.40  # robot radius (m)
SPRAY_STANDOFF = 2.3  # matches the water jet's forward reach (arc passes through
                      # sim.FIRE_AIM_HEIGHT here); centre of the 1.7-2.9 m hit window.

TERRAIN_MAX_SLOPE = 0.10  # max rise per horizontal meter (blind-policy envelope)
_HFIELD_BASE_Z = 0.01
_SNAP_RADIUS_CELLS = 12   # ~1.2 m at NAV_RES


def _rot_local(px: float, py: float, cx: float, cy: float, theta: float) -> tuple[float, float]:
    dx, dy = px - cx, py - cy
    c, s = math.cos(theta), math.sin(theta)
    return c * dx + s * dy, -s * dx + c * dy


def point_in_wall(px: float, py: float, wall: tuple[float, float, float, float, float],
                  inflation: float = 0.0) -> bool:
    """True if (px, py) lies inside a wall footprint expanded by ``inflation``."""
    cx, cy, w, h, theta = wall
    lx, ly = _rot_local(px, py, cx, cy, theta)
    return abs(lx) <= w / 2 + inflation and abs(ly) <= h / 2 + inflation


def point_in_any_wall(px: float, py: float,
                      walls: tuple[tuple[float, float, float, float, float], ...],
                      inflation: float = 0.0) -> bool:
    return any(point_in_wall(px, py, w, inflation) for w in walls)


def world_to_cell(x: float, y: float, xmin: float, ymin: float, res: float) -> tuple[int, int]:
    return int((x - xmin) / res), int((y - ymin) / res)


def cell_in_grid(i: int, j: int, grid: np.ndarray) -> bool:
    ny, nx = grid.shape
    return 0 <= i < nx and 0 <= j < ny


def occupancy_grid(spec: SceneSpec, res: float = NAV_RES,
                   inflation: float = NAV_INFLATION) -> tuple[np.ndarray, float, float, float]:
    """Return (occupied, xmin, ymin, res). ``occupied[j, i]`` is True when impassable."""
    cm = Costmap.from_spec(spec, res=res, inflation=inflation)
    return cm.occupied, cm.xmin, cm.ymin, cm.res


def flood_fill_reachable(grid: np.ndarray, start_ij: tuple[int, int]) -> np.ndarray:
    """8-connected BFS; returns bool array same shape as ``grid`` (True = reachable)."""
    ny, nx = grid.shape
    reachable = np.zeros((ny, nx), dtype=bool)
    si, sj = start_ij
    if not cell_in_grid(si, sj, grid) or grid[sj, si]:
        return reachable
    q: deque[tuple[int, int]] = deque([(si, sj)])
    reachable[sj, si] = True
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while q:
        ci, cj = q.popleft()
        for di, dj in neighbors:
            ni, nj = ci + di, cj + dj
            if not cell_in_grid(ni, nj, grid):
                continue
            if grid[nj, ni] or reachable[nj, ni]:
                continue
            reachable[nj, ni] = True
            q.append((ni, nj))
    return reachable


def _terrain_height_norm(terrain: TerrainSpec, u: float, v: float) -> float:
    """Normalized height [0, 1] at hfield UV (same formula as ``scenes.write_terrain_png``)."""
    rng = np.random.default_rng(terrain.seed)
    h = (math.sin(u * 6 + rng.uniform(0, 2 * math.pi))
         * math.cos(v * 5 + rng.uniform(0, 2 * math.pi)) * 0.35
         + math.sin((u + v) * 3) * 0.25 + 0.5)
    return float(np.clip(h, 0.0, 1.0))


def _terrain_height_world(terrain: TerrainSpec, wx: float, wy: float,
                          bounds: tuple[float, float, float, float]) -> float:
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    u = (wx - cx) / (2 * terrain.radius_x) + 0.5
    v = (wy - cy) / (2 * terrain.radius_y) + 0.5
    return _HFIELD_BASE_Z + _terrain_height_norm(terrain, u, v) * terrain.elevation


def _terrain_slope(terrain: TerrainSpec, wx: float, wy: float,
                   bounds: tuple[float, float, float, float], res: float) -> float:
    """Max absolute gradient (rise/run) at ``(wx, wy)``."""
    h0 = _terrain_height_world(terrain, wx, wy, bounds)
    slopes = []
    for dx, dy in ((res, 0), (-res, 0), (0, res), (0, -res)):
        h1 = _terrain_height_world(terrain, wx + dx, wy + dy, bounds)
        slopes.append(abs(h1 - h0) / res)
    return max(slopes)


class Costmap:
    """Occupancy grid for A*; walls inflated by robot radius. Debris are traversable."""

    def __init__(self, occupied: np.ndarray, xmin: float, ymin: float, res: float):
        self.occupied = occupied
        self.xmin = xmin
        self.ymin = ymin
        self.res = res
        self.shape = occupied.shape  # (ny, nx)

    @classmethod
    def from_spec(cls, spec: SceneSpec, res: float = NAV_RES,
                  inflation: float = NAV_INFLATION) -> Costmap:
        xmin, xmax, ymin, ymax = spec.bounds
        nx = max(1, int(math.ceil((xmax - xmin) / res)))
        ny = max(1, int(math.ceil((ymax - ymin) / res)))
        grid = np.zeros((ny, nx), dtype=bool)
        walls = spec.walls
        for j in range(ny):
            wy = ymin + (j + 0.5) * res
            for i in range(nx):
                wx = xmin + (i + 0.5) * res
                if point_in_any_wall(wx, wy, walls, inflation):
                    grid[j, i] = True
        if spec.terrain is not None:
            t = spec.terrain
            for j in range(ny):
                wy = ymin + (j + 0.5) * res
                for i in range(nx):
                    if grid[j, i]:
                        continue
                    wx = xmin + (i + 0.5) * res
                    if _terrain_slope(t, wx, wy, spec.bounds, res) > TERRAIN_MAX_SLOPE:
                        grid[j, i] = True
        return cls(grid, xmin, ymin, res)

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return world_to_cell(x, y, self.xmin, self.ymin, self.res)

    def cell_to_world(self, i: int, j: int) -> tuple[float, float]:
        return (self.xmin + (i + 0.5) * self.res,
                self.ymin + (j + 0.5) * self.res)

    def in_bounds(self, i: int, j: int) -> bool:
        return cell_in_grid(i, j, self.occupied)

    def is_free(self, i: int, j: int) -> bool:
        return self.in_bounds(i, j) and not self.occupied[j, i]

    def is_free_world(self, x: float, y: float) -> bool:
        i, j = self.world_to_cell(x, y)
        return self.is_free(i, j)


def standoff_point(fx: float, fy: float, from_x: float, from_y: float,
                   standoff: float = SPRAY_STANDOFF) -> tuple[float, float]:
    """World point ``standoff`` m from fire ``(fx, fy)`` toward ``(from_x, from_y)``."""
    dx = from_x - fx
    dy = from_y - fy
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (fx + standoff, fy)
    s = standoff / d
    return (fx + dx * s, fy + dy * s)


def nearest_burning_fire(fires: tuple[tuple[float, float], ...] | list,
                         health: list[float], from_x: float, from_y: float) -> int | None:
    """Index of the nearest fire with ``health > 0``, or ``None`` if none burn."""
    best_i: int | None = None
    best_d2 = float("inf")
    for i, (pos, h) in enumerate(zip(fires, health)):
        if h <= 0:
            continue
        fx, fy = pos[0], pos[1]
        d2 = (fx - from_x) ** 2 + (fy - from_y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def _has_wall_clearance(costmap: Costmap, wx: float, wy: float,
                        wall_margin: float) -> bool:
    """True if every cell center within ``wall_margin`` of ``(wx, wy)`` is free."""
    r_cells = int(math.ceil(wall_margin / costmap.res))
    ci, cj = costmap.world_to_cell(wx, wy)
    for dj in range(-r_cells, r_cells + 1):
        for di in range(-r_cells, r_cells + 1):
            cx, cy = costmap.cell_to_world(ci + di, cj + dj)
            if math.hypot(cx - wx, cy - wy) > wall_margin + costmap.res * 0.5:
                continue
            if not costmap.is_free(ci + di, cj + dj):
                return False
    return True


def _fire_clearance_ok(fires: tuple[tuple[float, float], ...] | list,
                       tx: float, ty: float, target_idx: int,
                       fire_clearance: float) -> bool:
    for i, pos in enumerate(fires):
        fpx, fpy = pos[0], pos[1]
        if i == target_idx:
            continue
        if math.hypot(fpx - tx, fpy - ty) < fire_clearance:
            return False
    return True


def _standoff_candidates(fx: float, fy: float, from_x: float, from_y: float,
                         standoff: float) -> list[tuple[float, float]]:
    candidates = [standoff_point(fx, fy, from_x, from_y, standoff)]
    for k in range(16):
        angle = 2 * math.pi * k / 16
        candidates.append((fx + standoff * math.cos(angle),
                           fy + standoff * math.sin(angle)))
    return candidates


def safe_spray_point(costmap: Costmap, target_fire: tuple[float, float],
                     fires: tuple[tuple[float, float], ...] | list,
                     from_xy: tuple[float, float],
                     standoff: float = SPRAY_STANDOFF,
                     wall_margin: float = 0.25,
                     fire_clearance: float = 1.2,
                     target_idx: int | None = None) -> tuple[float, float] | None:
    """Free, clearance-safe, A*-reachable spray point at ``standoff`` from target fire."""
    fx, fy = target_fire
    if target_idx is None:
        target_idx = next((i for i, p in enumerate(fires) if p[0] == fx and p[1] == fy), -1)
    start = from_xy
    best: tuple[float, float] | None = None
    best_len = float("inf")
    for gx, gy in _standoff_candidates(fx, fy, from_xy[0], from_xy[1], standoff):
        if not costmap.is_free_world(gx, gy):
            continue
        if not _has_wall_clearance(costmap, gx, gy, wall_margin):
            continue
        if not _fire_clearance_ok(fires, gx, gy, target_idx, fire_clearance):
            continue
        path = astar(costmap, start, (gx, gy))
        if not path:
            continue
        plen = sum(math.hypot(path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
                   for k in range(len(path) - 1))
        if plen < best_len or (plen == best_len and (best is None or (gx, gy) < best)):
            best_len = plen
            best = (gx, gy)
    return best


def best_standoff(costmap: Costmap, fx: float, fy: float, from_x: float, from_y: float,
                  standoff: float = SPRAY_STANDOFF) -> tuple[float, float] | None:
    """Free, A*-reachable standoff point; shortest path wins (deterministic tie-break)."""
    start = (from_x, from_y)
    best: tuple[float, float] | None = None
    best_len = float("inf")
    for gx, gy in _standoff_candidates(fx, fy, from_x, from_y, standoff):
        if not costmap.is_free_world(gx, gy):
            continue
        path = astar(costmap, start, (gx, gy))
        if not path:
            continue
        plen = sum(math.hypot(path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
                   for k in range(len(path) - 1))
        if plen < best_len or (plen == best_len and (best is None or (gx, gy) < best)):
            best_len = plen
            best = (gx, gy)
    return best


def _snap_to_free(costmap: Costmap, wx: float, wy: float,
                  max_cells: int = _SNAP_RADIUS_CELLS) -> tuple[float, float] | None:
    """Nearest free cell center within ``max_cells``; used when start/goal is occupied.

    Points outside the grid (e.g. the robot wandered off the map) are clamped to
    the nearest edge cell first so planning can still recover them.
    """
    ny, nx = costmap.occupied.shape
    ci, cj = costmap.world_to_cell(wx, wy)
    ci = min(max(ci, 0), nx - 1)
    cj = min(max(cj, 0), ny - 1)
    if costmap.is_free(ci, cj):
        return costmap.cell_to_world(ci, cj)
    best: tuple[float, float] | None = None
    best_d2 = float("inf")
    for r in range(1, max_cells + 1):
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                ni, nj = ci + di, cj + dj
                if not costmap.is_free(ni, nj):
                    continue
                cx, cy = costmap.cell_to_world(ni, nj)
                d2 = (cx - wx) ** 2 + (cy - wy) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = (cx, cy)
        if best is not None:
            return best
    return None


def _bresenham_cells(i0: int, j0: int, i1: int, j1: int) -> list[tuple[int, int]]:
    """Integer grid line (supercover-style via Bresenham)."""
    cells: list[tuple[int, int]] = []
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = di - dj
    i, j = i0, j0
    while True:
        cells.append((i, j))
        if i == i1 and j == j1:
            break
        e2 = 2 * err
        if e2 > -dj:
            err -= dj
            i += si
        if e2 < di:
            err += di
            j += sj
    return cells


def line_of_sight(costmap: Costmap, x0: float, y0: float, x1: float, y1: float) -> bool:
    """True if the straight segment is collision-free on the occupancy grid."""
    i0, j0 = costmap.world_to_cell(x0, y0)
    i1, j1 = costmap.world_to_cell(x1, y1)
    cells = _bresenham_cells(i0, j0, i1, j1)
    for i, j in cells:
        if not costmap.in_bounds(i, j) or not costmap.is_free(i, j):
            return False
    return True


def _simplify_path(costmap: Costmap, path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        far = i + 1
        for k in range(len(path) - 1, i, -1):
            if line_of_sight(costmap, path[i][0], path[i][1], path[k][0], path[k][1]):
                far = k
                break
        out.append(path[far])
        i = far
    return out


def astar(costmap: Costmap, start_xy: tuple[float, float],
          goal_xy: tuple[float, float]) -> list[tuple[float, float]]:
    """8-connected A* with octile heuristic and LoS path simplification.

    Occupied start/goal cells are snapped to the nearest free cell within ~1.2 m.
    Returns world waypoints from start to goal inclusive, or ``[]`` if no path.
    """
    sx, sy = start_xy
    gx, gy = goal_xy
    start_snap = _snap_to_free(costmap, sx, sy)
    goal_snap = _snap_to_free(costmap, gx, gy)
    if start_snap is None or goal_snap is None:
        return []
    sx, sy = start_snap
    gx, gy = goal_snap

    si, sj = costmap.world_to_cell(sx, sy)
    gi, gj = costmap.world_to_cell(gx, gy)
    if not costmap.is_free(si, sj) or not costmap.is_free(gi, gj):
        return []

    # Deterministic tie-breaking: (f, counter, i, j)
    counter = 0
    open_heap: list[tuple[float, int, int, int]] = []
    heapq.heappush(open_heap, (0.0, counter, si, sj))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {(si, sj): 0.0}
    closed: set[tuple[int, int]] = set()

    def h(i: int, j: int) -> float:
        dx = abs(gi - i)
        dy = abs(gj - j)
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        _, _, ci, cj = heapq.heappop(open_heap)
        if (ci, cj) in closed:
            continue
        closed.add((ci, cj))
        if ci == gi and cj == gj:
            break
        for di, dj in neighbors:
            ni, nj = ci + di, cj + dj
            if not costmap.is_free(ni, nj):
                continue
            if di != 0 and dj != 0:
                if not costmap.is_free(ci + di, cj) or not costmap.is_free(ci, cj + dj):
                    continue
            step = math.sqrt(2) if di and dj else 1.0
            ng = g_score[(ci, cj)] + step
            key = (ni, nj)
            if key in closed or ng >= g_score.get(key, float("inf")):
                continue
            g_score[key] = ng
            came_from[key] = (ci, cj)
            counter += 1
            heapq.heappush(open_heap, (ng + h(ni, nj), counter, ni, nj))

    if (gi, gj) not in came_from and (gi, gj) != (si, sj):
        return []

    cells: list[tuple[int, int]] = []
    cur = (gi, gj)
    while True:
        cells.append(cur)
        if cur == (si, sj):
            break
        cur = came_from[cur]
    cells.reverse()

    raw = [costmap.cell_to_world(i, j) for i, j in cells]
    return _simplify_path(costmap, raw)


class WaypointFollower:
    """Heading-P follower along an A* path; calls ``sim.set_command`` each tick."""

    def __init__(self, sim: G1Sim, path: list[tuple[float, float]],
                 goal_tol: float = 0.3, cruise_vx: float = 0.6):
        self.sim = sim
        self.path = path
        self.goal_tol = goal_tol
        self.cruise_vx = cruise_vx
        self._idx = 0

    def update(self) -> bool:
        """Advance waypoints and set velocity command. Returns True when done."""
        if not self.path:
            self.sim.set_command(0.0, 0.0, 0.0, _nav=True)
            return True

        state = self.sim.get_state()
        x, y = float(state["pos"][0]), float(state["pos"][1])
        yaw = quat_yaw(state["quat"])

        while self._idx < len(self.path) - 1:
            wx, wy = self.path[self._idx]
            if math.hypot(wx - x, wy - y) < self.goal_tol:
                self._idx += 1
            else:
                break

        wx, wy = self.path[self._idx]
        dx, dy = wx - x, wy - y
        dist = math.hypot(dx, dy)
        if self._idx == len(self.path) - 1 and dist < self.goal_tol:
            self.sim.set_command(0.0, 0.0, 0.0, _nav=True)
            return True

        heading_target = math.atan2(dy, dx)
        heading_err = math.atan2(math.sin(heading_target - yaw),
                                 math.cos(heading_target - yaw))
        yaw_cmd = clamp(1.8 * heading_err, *YAW_RANGE)
        abs_err = abs(heading_err)
        if abs_err > 0.5:
            vx = 0.0
        elif abs_err > 0.25:
            vx = self.cruise_vx * 0.25
        else:
            vx = clamp(self.cruise_vx * (1.0 - 0.8 * abs_err), 0.0, VX_RANGE[1])

        self.sim.set_command(vx, 0.0, yaw_cmd, _nav=True)
        return False


FACE_TOL = 0.12
FACE_HOLD_TICKS = 3


class ApproachController:
    """Navigate to spray point, face target fire, then hold READY."""

    def __init__(self, sim: G1Sim, path: list[tuple[float, float]],
                 fire_xy: tuple[float, float], target_idx: int,
                 goal_tol: float = 0.3, cruise_vx: float = 0.6,
                 face_tol: float = FACE_TOL, face_hold_ticks: int = FACE_HOLD_TICKS):
        self.sim = sim
        self.fire_xy = fire_xy
        self.target_idx = target_idx
        self.path = path
        self.face_tol = face_tol
        self.face_hold_ticks = face_hold_ticks
        self._phase = "navigate"
        self._face_ok = 0
        self.spray_goal = path[-1] if path else None
        self._follower = WaypointFollower(sim, path, goal_tol=goal_tol, cruise_vx=cruise_vx)

    @property
    def phase(self) -> str:
        return self._phase

    def _heading_err_to_fire(self) -> float:
        state = self.sim.get_state()
        x, y = float(state["pos"][0]), float(state["pos"][1])
        fx, fy = self.fire_xy
        yaw = quat_yaw(state["quat"])
        heading_target = math.atan2(fy - y, fx - x)
        return math.atan2(math.sin(heading_target - yaw),
                            math.cos(heading_target - yaw))

    def _face_command(self) -> None:
        err = self._heading_err_to_fire()
        yaw_cmd = clamp(1.8 * err, *YAW_RANGE)
        self.sim.set_command(0.0, 0.0, yaw_cmd, _nav=True)

    def update(self) -> bool:
        if self._phase == "navigate":
            if self._follower.update():
                self._phase = "face"
                self._face_ok = 0
            return False
        if self._phase == "face":
            err = self._heading_err_to_fire()
            if abs(err) < self.face_tol:
                self._face_ok += 1
            else:
                self._face_ok = 0
            self._face_command()
            if self._face_ok >= self.face_hold_ticks:
                self._phase = "ready"
                self.sim.set_command(0.0, 0.0, 0.0, _nav=True)
                return True
            return False
        self.sim.set_command(0.0, 0.0, 0.0, _nav=True)
        return True

    def replan(self) -> bool:
        self.sim._ensure_nav_costmap()
        cm = self.sim._nav_costmap
        if cm is None:
            return False
        state = self.sim.get_state()
        x, y = float(state["pos"][0]), float(state["pos"][1])
        fx, fy = self.fire_xy
        goal = safe_spray_point(cm, (fx, fy), self.sim.fire_positions, (x, y),
                                target_idx=self.target_idx)
        if goal is None:
            return False
        path = astar(cm, (x, y), goal)
        if not path:
            return False
        self.path = path
        self.spray_goal = goal
        self._follower = WaypointFollower(
            self.sim, path,
            goal_tol=self._follower.goal_tol,
            cruise_vx=self._follower.cruise_vx,
        )
        self._phase = "navigate"
        self._face_ok = 0
        return True
