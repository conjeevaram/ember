"""Record BOTH demo artifacts from a single, identical sim run.

One process, one ``G1Sim``: a browser records the whole web console to video
while an in-process thread saves the robot ego-camera frames straight off the
same sim's frame buffer. Because both outputs observe the exact same physics /
wind / spray realization, they are perfectly synchronized -- same run, same
stochastic behaviour.

Mission: "put out 2 fires, then return home" with gusty wind on ``demo_scene``.

Outputs:
    recordings/video_raw/*.webm        (-> convert to recordings/ember_demo.mp4)
    recordings/robot_cam_frames/*.jpg  (full-res ego frames, same run)
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

# Must be set before mujoco / torch / ember.config import.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ.get("EMBER_EGL_DEVICE", "0"))
# Cap BLAS/Torch threads: the policy MLP is tiny, so spreading it across all
# cores only adds sync overhead and starves the physics thread -> slow motion.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("EMBER_W", "1280")
os.environ.setdefault("EMBER_H", "720")
os.environ.setdefault("EMBER_CAM_W", "1280")
os.environ.setdefault("EMBER_CAM_H", "960")
os.environ.setdefault("EMBER_FPS", "30")
os.environ.setdefault("EMBER_QUALITY", "92")

import pathlib  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
torch.set_num_threads(1)

from ember import viewer            # noqa: E402
from ember.spec import from_json    # noqa: E402

OUT = REPO / "recordings"
CAM_DIR = OUT / "robot_cam_frames"
VIDEO_DIR = OUT / "video_raw"

_saving = threading.Event()
_stop = threading.Event()
_count = [0]


def cam_saver(sim, fps: float) -> None:
    """Save unique ego frames off the live sim's buffer while _saving is set."""
    CAM_DIR.mkdir(parents=True, exist_ok=True)
    min_dt = 1.0 / fps
    last = 0.0
    last_jpeg = None
    while not _stop.is_set():
        if _saving.is_set():
            jpeg = sim.cam_frames.get()
            now = time.time()
            if jpeg is not None and jpeg is not last_jpeg and now - last >= min_dt:
                (CAM_DIR / f"frame_{_count[0]:05d}.jpg").write_bytes(jpeg)
                _count[0] += 1
                last_jpeg = jpeg
                last = now
        time.sleep(0.005)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=str(REPO / "scenes" / "specs" / "demo_scene.json"))
    ap.add_argument("--prompt", default="put out 2 fires, then return home")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--cam-fps", type=float, default=15.0)
    ap.add_argument("--max-s", type=float, default=220.0)
    args = ap.parse_args()

    CAM_DIR.mkdir(parents=True, exist_ok=True)
    for f in CAM_DIR.glob("*.jpg"):
        f.unlink()
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    spec = from_json(args.spec)
    # Single sim + web server; returns the live G1Sim we also read frames from.
    sim = viewer.start(block=False, serve_web=True, port=args.port, spec=spec)
    time.sleep(3.0)  # server + first frames

    saver = threading.Thread(target=cam_saver, args=(sim, args.cam_fps), daemon=True)
    saver.start()

    base = f"http://127.0.0.1:{args.port}"
    from playwright.sync_api import sync_playwright

    size = {"width": 1920, "height": 1080}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True,
                                    args=["--force-device-scale-factor=1"])
        ctx = browser.new_context(viewport=size, record_video_dir=str(VIDEO_DIR),
                                  record_video_size=size, device_scale_factor=1)
        page = ctx.new_page()
        page.goto(base + "/", wait_until="domcontentloaded")
        page.wait_for_timeout(3000)

        # Start saving ego frames at the same moment the on-screen action begins,
        # so the frame sequence spans the same window as the video.
        _saving.set()
        page.click("#windBtn")                 # gusty wind (visible toggle)
        page.fill("#missionIn", args.prompt)
        page.wait_for_timeout(800)
        page.click("#missionBtn")              # run the mission

        start = time.time()
        done = False
        while time.time() - start < args.max_s:
            try:
                st = page.evaluate(
                    "async () => { const r = await fetch('/mission_status'); return await r.json(); }")
            except Exception:
                st = None
            if st and st.get("done"):
                done = True
                break
            page.wait_for_timeout(500)
        print(f"mission {'completed' if done else 'TIMED OUT'} after {time.time()-start:.1f}s")

        page.wait_for_timeout(3000)            # hold on the final frame
        _saving.clear()
        webm = page.video.path()
        ctx.close()
        browser.close()

    _stop.set()
    saver.join(timeout=5.0)
    print(f"raw video: {webm}")
    print(f"robot-cam frames (same run): {_count[0]} -> {CAM_DIR}")


if __name__ == "__main__":
    main()
