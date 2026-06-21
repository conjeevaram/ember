"""Offline full-resolution robot ego-camera capture for the demo mission.

Runs ``demo_scene`` headless (its own G1Sim physics + render threads, no web
server), enables gusty wind, executes the mission "put out 2 fires, then return
home" via the real MissionExecutor, and saves every robot-cam frame straight
from the ego renderer to ``recordings/robot_cam_frames/frame_*.jpg`` at full
resolution.

This is deliberately decoupled from the web viewer so it neither loads nor can
crash the live server, and the frames carry no MJPEG re-streaming overhead.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import threading
import time

# Must be set before mujoco / ember.config import.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ.get("EMBER_EGL_DEVICE", "0"))
os.environ.setdefault("EMBER_CAM_W", "1280")
os.environ.setdefault("EMBER_CAM_H", "960")
os.environ.setdefault("EMBER_QUALITY", "95")
os.environ.setdefault("EMBER_FPS", "30")

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ember import scenes                       # noqa: E402
from ember.config import G1_MODEL_DIR          # noqa: E402
from ember.executor import MissionExecutor     # noqa: E402
from ember.mission import parse_mission        # noqa: E402
from ember.sim import G1Sim                    # noqa: E402
from ember.spec import from_json               # noqa: E402

OUT = REPO / "recordings" / "robot_cam_frames"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=str(REPO / "scenes" / "specs" / "demo_scene.json"))
    ap.add_argument("--prompt", default="put out 2 fires, then return home")
    ap.add_argument("--fps", type=float, default=15.0, help="frame save rate")
    ap.add_argument("--max-s", type=float, default=220.0)
    ap.add_argument("--use-llm", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.jpg"):
        f.unlink()

    spec = from_json(args.spec)
    scenes.build(spec)
    name_12, name_29 = scenes.spec_scene_names(spec.name)

    sim = G1Sim(scene_path=str(G1_MODEL_DIR / name_12),
                overlay_scene=str(G1_MODEL_DIR / name_29), spec=spec)
    sim.heading_hold = False

    thread = threading.Thread(target=sim.run, daemon=True, name="cam-sim")
    thread.start()
    time.sleep(2.0)                            # let physics settle + first render

    if not sim.has_camera:
        print("ERROR: scene has no ego camera", file=sys.stderr)
        sys.exit(1)

    sim.enable_wind(True)                       # gusty wind

    mission = parse_mission(args.prompt, spec, use_llm=args.use_llm)
    print("parsed plan:", [(t.type.value, t.target) for t in mission])
    ex = MissionExecutor(sim, mission, tick_hz=20.0)
    ex.start()

    idx = 0
    min_dt = 1.0 / args.fps
    last = 0.0
    last_jpeg = None
    start = time.time()
    done = False
    while time.time() - start < args.max_s:
        now = time.time()
        if now - last >= min_dt:
            jpeg = sim.cam_frames.get()
            if jpeg is not None and jpeg is not last_jpeg:
                (OUT / f"frame_{idx:05d}.jpg").write_bytes(jpeg)
                idx += 1
                last_jpeg = jpeg
            last = now
        st = ex.status()
        if st and st.get("done"):
            done = True
            break
        time.sleep(0.01)

    # Capture a couple of extra seconds on the final (home) pose.
    tail_end = time.time() + 2.0
    while time.time() < tail_end:
        jpeg = sim.cam_frames.get()
        if jpeg is not None and jpeg is not last_jpeg:
            (OUT / f"frame_{idx:05d}.jpg").write_bytes(jpeg)
            idx += 1
            last_jpeg = jpeg
        time.sleep(min_dt)

    sim.stop()
    print(f"mission {'completed' if done else 'TIMED OUT'} after {time.time()-start:.1f}s")
    print(f"saved {idx} robot-cam frames -> {OUT}")


if __name__ == "__main__":
    main()
