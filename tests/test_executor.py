"""Unit tests for :mod:`ember.executor` (Phase 4 mission FSM)."""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest

from ember.config import G1_MODEL_DIR
from ember.executor import MissionExecutor
from ember.mission import Mission, Task, TaskType
from ember.nav import WaypointFollower
from ember.spec import SceneSpec


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=float)


def _simple_spec(
    fires=((5.0, 0.0), (0.0, 5.0)),
    start=(0.0, 0.0, 0.0),
    home=(0.0, 0.0),
    bounds=(-2.0, 8.0, -2.0, 8.0),
) -> SceneSpec:
    return SceneSpec(
        name="test",
        bounds=bounds,
        start=start,
        walls=(),
        fires=fires,
        terrain=None,
        home=home,
        seed=1,
        debris=(),
    )


class _FakeApproachNav:
    """Minimal stand-in for ApproachController during fake ticks."""

    def __init__(self, ticks_to_ready: int = 3):
        self.phase = "navigate"
        self._t = 0
        self._ready_after = ticks_to_ready

    def update(self) -> bool:
        self._t += 1
        if self._t >= self._ready_after:
            self.phase = "ready"
            return True
        if self._t > 1:
            self.phase = "face"
        return False


class FakeSim:
    """Deterministic sim stub; advances on :meth:`get_state`."""

    def __init__(
        self,
        spec: SceneSpec,
        *,
        pos=(0.0, 0.0, 0.9),
        yaw: float = 0.0,
        fire_health: list[float] | None = None,
        approach_ticks: int = 3,
        spray_decay_per_tick: float = 0.5,
        move_speed: float = 2.0,
    ):
        self.spec = spec
        self.fire_positions = [(fx, fy, 0.45) for fx, fy in spec.fires]
        n = len(self.fire_positions)
        self.fire_health = list(fire_health if fire_health is not None else [1.0] * n)
        self.targeted_fire = 0
        self.navigator = None
        self._approach_enabled = False
        self._approach_ready = False
        self._approach_ticks = approach_ticks
        self._spraying = False
        self._hitting = False
        self._fell = False
        self._pos = np.array([pos[0], pos[1], pos[2] if len(pos) > 2 else 0.9], dtype=float)
        self._yaw = yaw
        self._spray_decay = spray_decay_per_tick
        self._move_speed = move_speed
        self._approach_fail_for: set[int] = set()
        self._in_tick = False

    def get_state(self) -> dict:
        if not self._in_tick:
            self._tick()
        phase = None
        if isinstance(self.navigator, _FakeApproachNav):
            phase = self.navigator.phase
        elif self._approach_ready:
            phase = "ready"
        return {
            "pos": self._pos.copy(),
            "quat": _yaw_quat(self._yaw).copy(),
            "cmd": np.zeros(3, dtype=float),
            "fell": self._fell,
            "spraying": self._spraying,
            "hitting": self._hitting,
            "fires": [{"pos": p, "health": h}
                      for p, h in zip(self.fire_positions, self.fire_health)],
            "targeted_fire": self.targeted_fire,
            "fire_health": self.fire_health[self.targeted_fire] if self.fire_health else 0.0,
            "blocked": False,
            "approach_phase": phase,
        }

    def set_targeted_fire(self, idx: int) -> int:
        if 0 <= idx < len(self.fire_positions) and self.fire_health[idx] > 0:
            self.targeted_fire = idx
        return self.targeted_fire

    def set_approach(self, on: bool) -> bool:
        self._approach_enabled = bool(on)
        self._approach_ready = False
        if not on:
            self.navigator = None
            return False
        if self.targeted_fire in self._approach_fail_for:
            self.navigator = None
            return False
        self.navigator = _FakeApproachNav(self._approach_ticks)
        return True

    def set_spray(self, on=None) -> bool:
        self._spraying = (not self._spraying) if on is None else bool(on)
        return self._spraying

    def set_command(self, vx=None, vy=None, yaw=None, _nav=False):
        if not _nav:
            self.navigator = None
            self._approach_enabled = False
            self._approach_ready = False
        dt = 1.0 / 30.0
        if yaw is not None:
            self._yaw += float(yaw) * dt
        if vx is not None or vy is not None:
            vx = float(vx or 0.0)
            vy = float(vy or 0.0)
            c, s = math.cos(self._yaw), math.sin(self._yaw)
            self._pos[0] += (c * vx - s * vy) * dt * self._move_speed
            self._pos[1] += (s * vx + c * vy) * dt * self._move_speed
        return np.zeros(3, dtype=float)

    def _tick(self) -> None:
        self._in_tick = True
        try:
            if self.navigator is not None:
                if isinstance(self.navigator, _FakeApproachNav):
                    if self.navigator.update():
                        self._approach_ready = True
                        self.navigator = None
                        self._approach_enabled = False
                elif isinstance(self.navigator, WaypointFollower):
                    if self.navigator.update():
                        self.navigator = None
        finally:
            self._in_tick = False

        ready = self._approach_ready
        self._hitting = False
        if self._spraying and ready and self.fire_health:
            tf = self.targeted_fire
            if 0 <= tf < len(self.fire_health) and self.fire_health[tf] > 0:
                self._hitting = True
                self.fire_health[tf] = max(0.0, self.fire_health[tf] - self._spray_decay)


def _wait_until(pred, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def _run_executor(sim, mission: Mission, timeout=5.0) -> MissionExecutor:
    ex = MissionExecutor(sim, mission, tick_hz=60.0)
    ex.start()
    _wait_until(lambda: ex.status()["done"] or ex.status()["state"] == "paused", timeout)
    return ex


def test_extinguish_single_fire():
    spec = _simple_spec(fires=((4.0, 0.0),))
    sim = FakeSim(spec)
    ex = _run_executor(sim, Mission((Task(TaskType.EXTINGUISH, 0),)))
    st = ex.status()
    assert st["done"]
    assert sim.fire_health[0] == 0.0
    ex.stop()


def test_extinguish_all_nearest_first():
    spec = _simple_spec(fires=((6.0, 0.0), (1.0, 0.0)))
    sim = FakeSim(spec, pos=(0.0, 0.0, 0.9))
    order: list[int] = []
    orig_set = sim.set_targeted_fire

    def track(idx):
        if 0 <= idx < len(sim.fire_health) and sim.fire_health[idx] > 0:
            order.append(idx)
        return orig_set(idx)

    sim.set_targeted_fire = track  # type: ignore[method-assign]
    ex = _run_executor(sim, Mission((Task(TaskType.EXTINGUISH, None),)))
    assert ex.status()["done"]
    assert sim.fire_health == [0.0, 0.0]
    assert order[0] == 1
    ex.stop()


def test_visit_no_spray():
    spec = _simple_spec(fires=((3.0, 0.0),))
    sim = FakeSim(spec)
    ex = _run_executor(sim, Mission((Task(TaskType.VISIT, 0),)))
    assert ex.status()["done"]
    assert sim.fire_health[0] == 1.0
    assert not sim._spraying
    ex.stop()


def test_search_detects_and_appends_extinguish():
    spec = _simple_spec(fires=((3.5, 0.0),), bounds=(-1.0, 6.0, -1.0, 6.0))
    sim = FakeSim(spec, pos=(0.0, 0.0, 0.9), yaw=0.0, move_speed=8.0)
    ex = MissionExecutor(
        sim,
        Mission((Task(TaskType.SEARCH,), Task(TaskType.EXTINGUISH, 0))),
        tick_hz=60.0,
    )
    ex.start()
    assert _wait_until(
        lambda: ex.status()["message"].startswith("search detected") or ex.status()["done"],
        5.0,
    )
    tasks = ex.status()["tasks"]
    assert sum(t["type"] == "extinguish" and t["target"] == 0 for t in tasks) >= 1
    ex.stop()


def test_return_reaches_home():
    spec = _simple_spec(fires=(), home=(4.0, 0.0))
    sim = FakeSim(spec, pos=(0.0, 0.0, 0.9), move_speed=10.0)
    ex = _run_executor(sim, Mission((Task(TaskType.RETURN,),)), timeout=8.0)
    assert ex.status()["done"]
    assert math.hypot(sim._pos[0] - 4.0, sim._pos[1]) < 0.6
    ex.stop()


def test_preempt_and_resume():
    spec = _simple_spec(fires=((5.0, 0.0),))
    sim = FakeSim(spec, approach_ticks=50)
    ex = MissionExecutor(sim, Mission((Task(TaskType.EXTINGUISH, 0),)), tick_hz=60.0)
    ex.start()
    assert _wait_until(lambda: ex.status()["state"] == "approach", 2.0)
    ex.preempt()
    assert ex.status()["state"] == "paused"
    ex.resume()
    assert _wait_until(lambda: ex.status()["done"], 5.0)
    assert sim.fire_health[0] == 0.0
    ex.stop()


def test_manual_navigator_clear_pauses():
    spec = _simple_spec(fires=(), home=(5.0, 0.0))
    sim = FakeSim(spec, pos=(0.0, 0.0, 0.9))
    ex = MissionExecutor(sim, Mission((Task(TaskType.RETURN,),)), tick_hz=60.0)
    ex.start()
    assert _wait_until(lambda: ex.status()["state"] == "return", 2.0)
    sim.navigator = None
    assert _wait_until(lambda: ex.status()["state"] == "paused", 2.0)
    ex.stop()


def test_status_shape_and_thread_safety():
    spec = _simple_spec(fires=((4.0, 0.0),))
    sim = FakeSim(spec, approach_ticks=20)
    ex = MissionExecutor(sim, Mission((Task(TaskType.EXTINGUISH, 0),)), tick_hz=60.0)
    ex.start()
    errors: list[str] = []

    def poll():
        try:
            for _ in range(200):
                st = ex.status()
                assert "running" in st and "state" in st and "tasks" in st
                assert "current_task" in st and "message" in st
                time.sleep(0.005)
        except Exception as exc:
            errors.append(str(exc))

    t = threading.Thread(target=poll)
    t.start()
    t.join(timeout=5.0)
    ex.stop()
    assert errors == []


def test_start_stop_clean_shutdown():
    spec = _simple_spec(fires=((4.0, 0.0),))
    sim = FakeSim(spec, approach_ticks=100)
    ex = MissionExecutor(sim, Mission((Task(TaskType.EXTINGUISH, 0),)), tick_hz=60.0)
    ex.start()
    time.sleep(0.1)
    ex.stop()
    assert not ex.status()["running"]
    assert ex._thread is None


@pytest.mark.skipif(not G1_MODEL_DIR.exists(), reason="G1 model dir missing")
def test_integration_extinguish_headless():
    import os
    import threading as th

    os.environ.setdefault("MUJOCO_GL", "egl")
    from ember import scenegen, scenes
    from ember.sim import G1Sim

    spec = scenegen.random_spec(9090, n_fires=1, n_walls=1, n_debris=0)
    scenes.build(spec)
    name_12, name_29 = scenes.spec_scene_names(spec.name)
    sim = G1Sim(
        scene_path=str(G1_MODEL_DIR / name_12),
        overlay_scene=str(G1_MODEL_DIR / name_29),
        spec=spec,
    )
    sim.heading_hold = False
    initial_health = sim.fire_health[0]

    th.Thread(target=sim.run, daemon=True).start()
    time.sleep(0.5)

    ex = MissionExecutor(sim, Mission((Task(TaskType.EXTINGUISH, None),)), tick_hz=20.0)
    ex.start()
    ok = _wait_until(
        lambda: ex.status()["done"] or sim.fire_health[0] < initial_health - 0.05,
        timeout=90.0,
    )
    ex.stop()
    sim._running = False
    assert ok
    assert sim.fire_health[0] < initial_health
