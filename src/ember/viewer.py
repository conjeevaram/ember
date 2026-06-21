"""Headless web viewer for the G1 firefighter sim.

Discovers the named demo scenes plus every ``scenes/specs/*.json`` procedural
spec, runs the active :class:`~ember.sim.G1Sim` on its own physics + render
threads, and serves a browser UI (keyboard driving, a top-down nav map, and
the autonomous "approach fire" controller) over MJPEG + JSON. Scenes hot-swap
under a lock without restarting the server.

Run:
    python -m ember.viewer --scene obstacles
    python -m ember.viewer --spec scenes/specs/scene_000.json
    ember-walk --scene obstacles            # if installed
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import scenes
from .config import G1_MODEL_DIR, RENDER_FPS
from .executor import MissionExecutor
from .mission import parse_mission
from .sim import G1Sim, OVERLAY_SCENES, SCENES
from .spec import from_json
from .streaming import FrameBuffer, create_app, mjpeg_response, serve

_SPECS_DIR = Path(__file__).resolve().parent.parent.parent / "scenes" / "specs"
_NAMED_LABELS = {"flat": "flat", "obstacles": "obstacles", "fire": "fire"}


@dataclass(frozen=True)
class SceneEntry:
    key: str
    label: str
    kind: Literal["named", "proc"]
    scene: str | None = None
    spec_path: Path | None = None


def discover_scene_catalog() -> list[SceneEntry]:
    """Named demos plus every ``scenes/specs/*.json`` (sorted)."""
    entries = [
        SceneEntry(key=k, label=_NAMED_LABELS[k], kind="named", scene=k)
        for k in ("flat", "obstacles", "fire")
    ]
    if _SPECS_DIR.is_dir():
        for path in sorted(_SPECS_DIR.glob("*.json")):
            spec = from_json(path)
            entries.append(SceneEntry(
                key=spec.name, label=spec.name, kind="proc", spec_path=path))
    return entries


# Module-level handle on the live sim, set by SimManager on each (re)build.
_SIM: G1Sim | None = None


class SimManager:
    """Owns the active :class:`G1Sim` and hot-swaps scenes under a lock."""

    def __init__(self, initial_key: str, *, overlay=True, render=True,
                 loop=False, initial_cmd=None):
        self._lock = threading.Lock()
        self._overlay = overlay
        self._render = render
        self._loop = loop
        self._initial_cmd = initial_cmd
        self.catalog = discover_scene_catalog()
        self._by_key = {e.key: e for e in self.catalog}
        if initial_key not in self._by_key:
            raise ValueError(f"unknown scene key: {initial_key}")
        self._current_key = initial_key
        self._physics_thread: threading.Thread | None = None
        self._retired: list[G1Sim] = []
        self._executor: MissionExecutor | None = None
        self._switch_unlocked(initial_key, first=True)

    def current_key(self) -> str:
        with self._lock:
            return self._current_key

    def catalog_list(self) -> list[dict]:
        return [{"key": e.key, "label": e.label} for e in self.catalog]

    def sim(self) -> G1Sim:
        with self._lock:
            if _SIM is None:
                raise RuntimeError("no active sim")
            return _SIM

    def frames(self) -> FrameBuffer:
        return self.sim().frames

    def cam_frames(self) -> FrameBuffer | None:
        sim = self.sim()
        return sim.cam_frames if sim.has_camera else None

    def has_camera(self) -> bool:
        return self.sim().has_camera

    def switch(self, key: str) -> str:
        with self._lock:
            if key not in self._by_key:
                raise ValueError(f"unknown scene key: {key}")
            if key == self._current_key:
                return key
            return self._switch_unlocked(key)

    def shutdown(self):
        with self._lock:
            self._stop_executor_unlocked()
            sim = _SIM
            if sim is not None:
                sim.stop()
            if self._physics_thread is not None:
                self._physics_thread.join(timeout=5.0)
                self._physics_thread = None

    # -- mission executor --------------------------------------------------- #

    def _stop_executor_unlocked(self) -> None:
        if self._executor is not None:
            self._executor.stop()
            self._executor = None

    def start_mission(self, prompt: str) -> dict:
        """Parse ``prompt`` against the active scene and run it autonomously.

        Missions need a procedural ``SceneSpec`` (the named demos have no
        costmap). Parsing happens outside the lock so a slow LLM call doesn't
        stall other requests."""
        sim = self.sim()
        if sim.spec is None:
            return {"error": "missions require a procedural scene"}
        mission = parse_mission(prompt, sim.spec)
        tasks = [{"type": t.type.value, "target": t.target} for t in mission]
        with self._lock:
            self._stop_executor_unlocked()
            ex = MissionExecutor(sim, mission)
            ex.start()
            self._executor = ex
        return {"ok": True, "prompt": prompt, "tasks": tasks}

    def mission_status(self) -> dict | None:
        ex = self._executor
        return ex.status() if ex is not None else None

    def stop_mission(self) -> None:
        with self._lock:
            self._stop_executor_unlocked()

    def preempt_executor(self) -> None:
        ex = self._executor
        if ex is not None:
            ex.preempt()

    def _build_sim(self, entry: SceneEntry) -> G1Sim:
        if entry.kind == "proc":
            spec = from_json(entry.spec_path)
            scenes.build(spec)
            name_12, name_29 = scenes.spec_scene_names(spec.name)
            scene_path = str(G1_MODEL_DIR / name_12)
            overlay = (str(G1_MODEL_DIR / name_29)
                       if self._overlay else None)
            return G1Sim(scene_path=scene_path, overlay_scene=overlay, spec=spec)
        scenes.ensure_scenes()
        scene_path = SCENES[entry.scene]
        overlay = (OVERLAY_SCENES.get(scene_path) if self._overlay else None)
        return G1Sim(scene_path=scene_path, overlay_scene=overlay)

    def _apply_policy(self, sim: G1Sim, entry: SceneEntry):
        if entry.kind == "proc":
            sim.auto_reset = False
            sim.heading_hold = False
            sim.set_command(0.0, 0.0, 0.0)
        elif entry.scene == "obstacles":
            sim.auto_reset = self._loop
            sim.heading_hold = self._loop
            sim.set_command(0.4, 0.0, 0.0)
        else:  # "flat" / "fire": walk forward, no auto-reset
            sim.auto_reset = False
            sim.heading_hold = False
            sim.set_command(0.5, 0.0, 0.0)

    def _switch_unlocked(self, key: str, first: bool = False) -> str:
        global _SIM
        self._stop_executor_unlocked()
        retiring: G1Sim | None = None
        if self._physics_thread is not None:
            retiring = _SIM
            if retiring is not None:
                retiring.stop()
            self._physics_thread.join(timeout=5.0)
            self._physics_thread = None
            if retiring is not None:
                self._retired.append(retiring)
            time.sleep(0.15)

        entry = self._by_key[key]
        sim = self._build_sim(entry)
        _SIM = sim
        if first and self._initial_cmd is not None:
            sim.set_command(*self._initial_cmd)
            sim.auto_reset = self._loop
            sim.heading_hold = self._loop
        else:
            self._apply_policy(sim, entry)

        self._physics_thread = threading.Thread(
            target=lambda: sim.run(render=self._render),
            daemon=True, name=f"g1-physics-{key}")
        self._physics_thread.start()
        self._current_key = key
        self._retired.clear()
        return key


_WEB_DIR = Path(__file__).resolve().parent / "web"
_PAGE = (_WEB_DIR / "console.html").read_text(encoding="utf-8")


def _make_app(mgr: SimManager):
    from flask import jsonify, request

    def register(app):
        @app.route("/cmd")
        def cmd():
            def f(name):
                v = request.args.get(name)
                return float(v) if v is not None else None
            mgr.preempt_executor()  # manual drive overrides an autonomous mission
            c = mgr.sim().set_command(vx=f("vx"), vy=f("vy"), yaw=f("yaw"))
            return jsonify(cmd=c.tolist())

        @app.route("/heading")
        def heading():
            on = request.args.get("on", "1") not in ("0", "false", "")
            mgr.sim().set_heading_hold(on)
            return jsonify(heading_hold=on)

        @app.route("/spray")
        def spray():
            on = request.args.get("on")
            return jsonify(spraying=mgr.sim().set_spray(
                None if on is None else on not in ("0", "false", "")))

        @app.route("/reignite")
        def reignite():
            return jsonify(fire_health=mgr.sim().reignite())

        @app.route("/wind")
        def wind_toggle():
            arg = request.args.get("on")
            sim = mgr.sim()
            on = (not sim.wind_enabled) if arg is None else arg not in ("0", "false", "")
            sim.enable_wind(on)
            return jsonify(wind_enabled=sim.wind_enabled)

        @app.route("/scenes")
        def scenes_list():
            return jsonify(scenes=mgr.catalog_list(), current=mgr.current_key(),
                           has_camera=mgr.has_camera())

        @app.route("/scene")
        def scene_switch():
            key = request.args.get("key") or request.args.get("name")
            if not key:
                return jsonify(error="missing key"), 400
            try:
                mgr.switch(key)
            except ValueError as e:
                return jsonify(error=str(e)), 400
            return jsonify(current=mgr.current_key(), has_camera=mgr.has_camera())

        @app.route("/nav")
        def nav_view():
            snap = mgr.sim().nav_snapshot()
            if snap is None:
                return jsonify(error="no procedural spec")
            return jsonify(snap)

        @app.route("/approach")
        def approach_toggle():
            on = request.args.get("on", "1") not in ("0", "false", "")
            active = mgr.sim().set_approach(on)
            phase = mgr.sim().get_state().get("approach_phase")
            return jsonify(approach=active, phase=phase)

        @app.route("/reset")
        def reset_pose():
            mgr.stop_mission()
            return jsonify(pos=mgr.sim().reset_to_start().tolist())

        @app.route("/mission")
        def mission_start():
            prompt = (request.args.get("prompt") or "").strip()
            if not prompt:
                return jsonify(error="missing prompt"), 400
            result = mgr.start_mission(prompt)
            return jsonify(result), (400 if "error" in result else 200)

        @app.route("/mission_status")
        def mission_status():
            return jsonify(mgr.mission_status() or {"running": False})

        @app.route("/mission_stop")
        def mission_stop():
            mgr.stop_mission()
            return jsonify(stopped=True)

        @app.route("/target")
        def target_fire():
            idx = request.args.get("idx", type=int)
            if idx is None:
                return jsonify(error="missing idx"), 400
            return jsonify(targeted_fire=mgr.sim().set_targeted_fire(idx))

        @app.route("/camera")
        def camera():
            if not mgr.has_camera():
                return "", 404
            return mjpeg_response(mgr.cam_frames, RENDER_FPS)

    def state_fn():
        s = mgr.sim().get_state()
        return dict(pos=s["pos"].tolist(), cmd=s["cmd"].tolist(),
                    lin_vel=s["lin_vel"].tolist(), wind=s.get("wind", [0.0, 0.0]),
                    wind_enabled=bool(mgr.sim().wind_enabled),
                    fell=bool(s["fell"]), sim_time=float(s["sim_time"]),
                    spraying=bool(s["spraying"]),
                    hitting=bool(s["hitting"]), fire_health=float(s["fire_health"]),
                    fires=[{"pos": list(f["pos"]), "health": float(f["health"])}
                           for f in s["fires"]],
                    targeted_fire=int(s["targeted_fire"]),
                    blocked=bool(s["blocked"]),
                    approach_phase=s["approach_phase"])

    return create_app(page_html=_PAGE, frame_source=mgr.frames,
                      state_fn=state_fn, register_routes=register)


def _load_dotenv() -> None:
    """Load ``KEY=VALUE`` lines from the repo-root ``.env`` into ``os.environ``
    (without overwriting anything already set).

    Lets *any* launcher -- ``python -m ember.viewer``, a teammate's restart, or
    a programmatic call -- pick up secrets like ``GEMINI_API_KEY`` so the LLM
    mission parser stays enabled without remembering to ``source .env``."""
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def start(block=True, serve_web=True, port=8088, render=True, initial_cmd=None,
          scene="flat", loop=False, overlay=True, spec=None):
    """Start the sim. With block=False, runs the loop on a background thread.

    Pass a ``SceneSpec`` via ``spec`` to run a procedural scene: the XMLs are
    (re)built from it and the robot spawns at ``spec.start``."""
    _load_dotenv()
    initial_key = spec.name if spec is not None else (
        scene if scene in SCENES else "flat")

    mgr = SimManager(initial_key, overlay=overlay, render=render,
                     loop=loop, initial_cmd=initial_cmd)

    if serve_web:
        serve(_make_app(mgr), port, label="G1 walker")

    if block:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            mgr.shutdown()
    return mgr.sim()


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="flat", help="flat | obstacles | <path.xml>")
    p.add_argument("--spec", default=None,
                   help="path to a SceneSpec JSON (procedural scene; spawns at start)")
    p.add_argument("--idle", action="store_true",
                   help="stand still until commanded (default: walk forward)")
    p.add_argument("--vx", type=float, default=None, help="initial forward speed")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--loop", dest="loop", action="store_true", default=None,
                   help="auto-reset & replay (default: on for obstacles)")
    p.add_argument("--no-loop", dest="loop", action="store_false")
    p.add_argument("--no-overlay", dest="overlay", action="store_false", default=True,
                   help="render the bare 12-DOF body (no arms)")
    args = p.parse_args()

    spec = from_json(args.spec) if args.spec else None

    # Procedural scenes have walls; stand still by default so the blind walker
    # doesn't march into one. Named demo scenes keep their walk-forward default.
    default_vx = 0.0 if spec is not None else (0.4 if args.scene == "obstacles" else 0.5)
    vx = 0.0 if args.idle else (args.vx if args.vx is not None else default_vx)
    loop = (args.scene == "obstacles") if args.loop is None else args.loop
    start(block=True, serve_web=True, port=args.port,
          initial_cmd=(vx, 0.0, 0.0), scene=args.scene, loop=loop,
          overlay=args.overlay, spec=spec)


if __name__ == "__main__":
    main()
