"""Mission executor: FSM orchestration over sim navigation and spray (Phase 4)."""
from __future__ import annotations

import math
import threading
import time
from typing import Any

from .config import quat_yaw
from .mission import Mission, Task, TaskType
from .nav import Costmap, WaypointFollower, astar, nearest_burning_fire

DETECT_RADIUS = 4.0
DETECT_FOV_HALF = math.radians(50.0)
SPRAY_TIMEOUT = 25.0
NOT_HITTING_REAPPROACH = 5.0
APPROACH_TIMEOUT = 120.0
NAV_TIMEOUT = 180.0
HOME_TOL = 0.5
COVERAGE_STEP_CELLS = 5


class MissionExecutor:
    """Run a :class:`Mission` on a :class:`G1Sim` via polling (no physics stepping)."""

    def __init__(self, sim, mission: Mission, *, perception=None, tick_hz: float = 30.0):
        self.sim = sim
        self.mission = mission
        self.perception = perception
        self.tick_hz = tick_hz
        self._period = 1.0 / tick_hz

        self._tasks: list[Task] = list(mission.tasks)
        self._task_index = 0
        self._state = "idle"
        self._message = ""
        self._done = False
        self._running = False
        self._paused = False
        self._expect_nav = False
        self._reapproach_used = False
        self._search_seen: set[int] = set()
        self._active_path: list[tuple[float, float]] = []

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        if self.sim.spec is None:
            raise ValueError("MissionExecutor requires a procedural SceneSpec on sim")
        with self._lock:
            if self._running:
                return
            self._running = True
            self._done = False
            self._paused = False
            self._task_index = 0
            self._state = "plan"
            self._message = ""
            self._tasks = list(self.mission.tasks)
            self._stop_event.clear()
        # Take full control: the sim must stop its own auto-spray / auto-retarget
        # so it doesn't fight the mission FSM.
        self.sim.external_control = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mission-executor")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.sim.external_control = False
        with self._lock:
            self._running = False
            if not self._done:
                self._state = "idle"

    def preempt(self) -> None:
        # Hand control back to the operator while paused.
        self.sim.external_control = False
        with self._lock:
            self._paused = True
            self._state = "paused"
            self._expect_nav = False

    def resume(self) -> None:
        with self._lock:
            if not self._running or self._done:
                return
            self._paused = False
            self.sim.external_control = True
            if self._state == "paused":
                self._state = "plan"

    def status(self) -> dict[str, Any]:
        with self._lock:
            idx = self._task_index
            tasks = self._tasks
            cur = tasks[idx] if idx < len(tasks) else None
            return {
                "running": self._running,
                "state": self._state,
                "task_index": idx,
                "n_tasks": len(tasks),
                "current_task": (_task_dict(cur) if cur is not None else None),
                "tasks": [_task_dict(t) for t in tasks],
                "done": self._done,
                "message": self._message,
            }

    # -- main loop ---------------------------------------------------------- #

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if self._paused:
                self._sleep()
                continue

            with self._lock:
                if self._task_index >= len(self._tasks):
                    self._done = True
                    self._running = False
                    self._state = "done"
                    break
                task = self._tasks[self._task_index]

            try:
                ok = self._run_task(task)
            except _AbortTask as exc:
                self._set_message(str(exc))
                ok = True

            if self._stop_event.is_set():
                break
            if self._paused:
                continue
            if ok:
                with self._lock:
                    self._task_index += 1
                    self._state = "next" if self._task_index < len(self._tasks) else "done"

        with self._lock:
            if not self._done and self._stop_event.is_set():
                self._running = False
        # Mission finished (done / failed / stopped): halt the robot (otherwise
        # the last cruise velocity keeps it walking off in circles), shut the
        # hose, and hand autonomy back to the sim.
        self.sim.navigator = None
        self.sim.set_command(0.0, 0.0, 0.0)
        self.sim.set_spray(False)
        self.sim.external_control = False

    def _run_task(self, task: Task) -> bool:
        if task.type == TaskType.EXTINGUISH:
            self._set_state("plan")
            if task.target is None:
                return self._extinguish_all()
            return self._extinguish_one(task.target)
        if task.type == TaskType.VISIT:
            self._set_state("plan")
            return self._visit(task.target or 0)
        if task.type == TaskType.SEARCH:
            self._set_state("search")
            return self._search()
        if task.type == TaskType.RETURN:
            self._set_state("plan")
            return self._return_home()
        return True

    # -- task implementations ----------------------------------------------- #

    def _extinguish_one(self, target: int) -> bool:
        self.sim.set_targeted_fire(target)
        if not self._approach_and_ready():
            self._set_message(f"approach failed for fire {target}")
            return True
        return self._spray_until_out(target)

    def _extinguish_all(self) -> bool:
        while not self._stop_event.is_set() and not self._paused:
            st = self.sim.get_state()
            if st["fell"]:
                raise _AbortTask("robot fell during extinguish-all")
            idx = nearest_burning_fire(
                self.sim.fire_positions, self.sim.fire_health,
                float(st["pos"][0]), float(st["pos"][1]),
            )
            if idx is None:
                return True
            self._set_state("approach")
            self.sim.set_targeted_fire(idx)
            if not self._approach_and_ready():
                self._set_message(f"approach failed for fire {idx}")
                continue
            if not self._spray_until_out(idx):
                return False
        return not self._paused

    def _visit(self, target: int) -> bool:
        self.sim.set_targeted_fire(target)
        if not self._approach_and_ready():
            self._set_message(f"visit approach failed for fire {target}")
        # The target fire is still alive after a visit, so the sim's approach
        # state (_approach_enabled/_approach_ready) is never auto-cleared the way
        # it is after an extinguish. Release it explicitly: otherwise the stale
        # blocked-replan handler cancels the next navigator (e.g. return-home)
        # when the robot pivots in place, wedging it at the fire.
        self.sim.set_approach(False)
        self.sim.set_command(0.0, 0.0, 0.0)
        return True

    def _search(self) -> bool:
        self.sim.set_approach(False)
        self._search_seen = set()
        cm = Costmap.from_spec(self.sim.spec)
        waypoints = _coverage_waypoints(cm)
        if not waypoints:
            self._set_message("search: no coverage waypoints")
            return True

        path = _connect_waypoints(cm, waypoints)
        if not path:
            self._set_message("search: could not plan coverage path")
            return True

        self._active_path = path
        self.sim.navigator = WaypointFollower(self.sim, path)
        self._expect_nav = True
        deadline = time.monotonic() + NAV_TIMEOUT

        while not self._stop_event.is_set() and not self._paused:
            if time.monotonic() > deadline:
                self._set_message("search: coverage timeout")
                break

            st = self.sim.get_state()
            if st["fell"]:
                raise _AbortTask("robot fell during search")

            new = [i for i in self._detect_fires() if i not in self._search_seen]
            if new:
                for i in new:
                    self._search_seen.add(i)
                    with self._lock:
                        self._tasks.insert(self._task_index + 1, Task(TaskType.EXTINGUISH, i))
                self.sim.navigator = None
                self._expect_nav = False
                self._set_message(f"search detected fires {new}")
                return True

            if self.sim.navigator is None:
                self._expect_nav = False
                if self._path_goal_reached(st):
                    self._set_message("search: coverage complete, nothing found")
                    return True
                self._pause_external()
                return False

            self._sleep()

        self.sim.navigator = None
        self._expect_nav = False
        return not self._paused

    def _return_home(self) -> bool:
        self.sim.set_approach(False)
        cm = Costmap.from_spec(self.sim.spec)
        st = self.sim.get_state()
        x, y = float(st["pos"][0]), float(st["pos"][1])
        home = self.sim.spec.home
        path = astar(cm, (x, y), home)
        if not path:
            self._set_message("return: no path to home")
            return True

        self._set_state("return")
        self._active_path = path
        self.sim.navigator = WaypointFollower(self.sim, path)
        self._expect_nav = True
        deadline = time.monotonic() + NAV_TIMEOUT

        while not self._stop_event.is_set() and not self._paused:
            if time.monotonic() > deadline:
                self._set_message("return: navigation timeout")
                break

            st = self.sim.get_state()
            if st["fell"]:
                raise _AbortTask("robot fell during return")

            hx, hy = home
            if math.hypot(float(st["pos"][0]) - hx, float(st["pos"][1]) - hy) < HOME_TOL:
                self.sim.navigator = None
                self.sim.set_command(0.0, 0.0, 0.0)
                self._expect_nav = False
                return True

            if self.sim.navigator is None:
                self._expect_nav = False
                if math.hypot(float(st["pos"][0]) - hx, float(st["pos"][1]) - hy) < HOME_TOL:
                    return True
                self._pause_external()
                return False

            self._sleep()

        self.sim.navigator = None
        self._expect_nav = False
        return not self._paused

    # -- shared phases ------------------------------------------------------ #

    def _approach_and_ready(self) -> bool:
        self._set_state("approach")
        self._reapproach_used = False
        if not self.sim.set_approach(True):
            self._expect_nav = False
            return False
        self._expect_nav = True
        deadline = time.monotonic() + APPROACH_TIMEOUT

        while not self._stop_event.is_set() and not self._paused:
            if time.monotonic() > deadline:
                self._set_message("approach timeout")
                self.sim.set_approach(False)
                self._expect_nav = False
                return False

            st = self.sim.get_state()
            if st["fell"]:
                raise _AbortTask("robot fell during approach")

            if st.get("approach_phase") == "ready":
                self._expect_nav = False
                return True

            if self._external_approach_preempt(st):
                self._pause_external()
                return False

            self._sleep()

        return False

    def _spray_until_out(self, target: int) -> bool:
        self._set_state("spray")
        self.sim.set_spray(True)
        deadline = time.monotonic() + SPRAY_TIMEOUT
        not_hitting_since: float | None = None

        while not self._stop_event.is_set() and not self._paused:
            if time.monotonic() > deadline:
                self._set_message(f"spray timeout on fire {target}")
                break

            st = self.sim.get_state()
            if st["fell"]:
                self.sim.set_spray(False)
                raise _AbortTask("robot fell during spray")

            health = self.sim.fire_health[target] if target < len(self.sim.fire_health) else 0.0
            if health <= 0:
                break

            if st.get("hitting"):
                not_hitting_since = None
            else:
                if not_hitting_since is None:
                    not_hitting_since = time.monotonic()
                elif (time.monotonic() - not_hitting_since >= NOT_HITTING_REAPPROACH
                      and not self._reapproach_used):
                    self._reapproach_used = True
                    self.sim.set_spray(False)
                    if self._approach_and_ready():
                        self.sim.set_spray(True)
                        not_hitting_since = None
                    else:
                        self._set_message(f"re-approach failed for fire {target}")
                        return True

            self._sleep()

        self.sim.set_spray(False)
        self._set_state("verify")
        if target < len(self.sim.fire_health) and self.sim.fire_health[target] <= 0:
            return True
        self._set_message(f"verify failed for fire {target}")
        return True

    # -- perception / detection --------------------------------------------- #

    def _detect_fires(self) -> list[int]:
        if self.perception is not None:
            return list(self.perception.detect(self.sim))

        st = self.sim.get_state()
        x, y = float(st["pos"][0]), float(st["pos"][1])
        yaw = quat_yaw(st["quat"])
        found: list[int] = []
        for i, pos in enumerate(self.sim.fire_positions):
            if i >= len(self.sim.fire_health) or self.sim.fire_health[i] <= 0:
                continue
            fx, fy = float(pos[0]), float(pos[1])
            dx, dy = fx - x, fy - y
            if math.hypot(dx, dy) > DETECT_RADIUS:
                continue
            bearing = math.atan2(dy, dx)
            err = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))
            if abs(err) <= DETECT_FOV_HALF:
                found.append(i)
        return found

    # -- preemption --------------------------------------------------------- #

    def _external_approach_preempt(self, st: dict) -> bool:
        if not self._expect_nav:
            return False
        if st.get("approach_phase") == "ready":
            return False
        return self.sim.navigator is None and st.get("approach_phase") is None

    def _path_goal_reached(self, st: dict) -> bool:
        if not self._active_path:
            return True
        gx, gy = self._active_path[-1]
        x, y = float(st["pos"][0]), float(st["pos"][1])
        return math.hypot(x - gx, y - gy) < HOME_TOL * 2

    def _pause_external(self) -> None:
        with self._lock:
            self._paused = True
            self._state = "paused"
            self._expect_nav = False
            self._message = "paused: external override"

    # -- helpers ------------------------------------------------------------ #

    def _sleep(self) -> None:
        self._stop_event.wait(self._period)

    def _set_state(self, state: str) -> None:
        with self._lock:
            self._state = state

    def _set_message(self, msg: str) -> None:
        with self._lock:
            self._message = msg


class _AbortTask(Exception):
    pass


def _task_dict(task: Task) -> dict[str, Any]:
    return {"type": task.type.value, "target": task.target}


def _coverage_waypoints(cm: Costmap, step: int = COVERAGE_STEP_CELLS) -> list[tuple[float, float]]:
    ny, nx = cm.shape
    pts: list[tuple[float, float]] = []
    rows = range(0, ny, step)
    for row_idx, j in enumerate(rows):
        cols = range(0, nx, step) if row_idx % 2 == 0 else range(nx - 1, -1, -step)
        for i in cols:
            if cm.is_free(i, j):
                pts.append(cm.cell_to_world(i, j))
    return pts


def _connect_waypoints(cm: Costmap, waypoints: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not waypoints:
        return []
    path: list[tuple[float, float]] = []
    for k in range(len(waypoints) - 1):
        leg = astar(cm, waypoints[k], waypoints[k + 1])
        if not leg:
            continue
        if path:
            path.extend(leg[1:])
        else:
            path.extend(leg)
    if not path:
        path = list(waypoints)
    return path
