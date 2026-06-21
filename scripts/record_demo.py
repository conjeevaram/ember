"""Record an MP4 of the whole Ember web console for the demo mission.

Headless-Chrome screen recording of the **entire console page exactly as it is**
(third-person stream, robot-cam PiP, telemetry, nav map, mission panel, wind
gauge) while it runs "put out 2 fires, then return home" with gusty wind on
``demo_scene``.

Point ``--base`` at a viewer dedicated to recording (so it does not load or
disturb the live server). Robot-cam frames are captured separately and offline
by ``scripts/render_robot_cam.py``.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import httpx

REPO = pathlib.Path(__file__).resolve().parent.parent
VIDEO_DIR = REPO / "recordings" / "video_raw"


def _try(client: httpx.Client, url: str, tries: int = 5) -> None:
    for _ in range(tries):
        try:
            client.get(url)
            return
        except Exception:
            time.sleep(0.5)


def record(base: str, prompt: str, max_s: float) -> pathlib.Path:
    from playwright.sync_api import sync_playwright

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    size = {"width": 1920, "height": 1080}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True,
                                    args=["--force-device-scale-factor=1"])
        ctx = browser.new_context(viewport=size, record_video_dir=str(VIDEO_DIR),
                                  record_video_size=size, device_scale_factor=1)
        page = ctx.new_page()
        page.goto(base + "/", wait_until="domcontentloaded")
        page.wait_for_timeout(3000)            # let the MJPEG streams warm up

        page.click("#windBtn")                 # enable gusty wind (visible toggle)
        page.fill("#missionIn", prompt)
        page.wait_for_timeout(800)
        page.click("#missionBtn")              # run the mission

        start = time.time()
        done = False
        while time.time() - start < max_s:
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
        path = page.video.path()
        ctx.close()                            # flush the video to disk
        browser.close()
    return pathlib.Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8099")
    ap.add_argument("--prompt", default="put out 2 fires, then return home")
    ap.add_argument("--max-s", type=float, default=220.0)
    args = ap.parse_args()

    # Clean start: stop any mission, reset pose, wind off (so the toggle shows).
    with httpx.Client(timeout=10.0) as c:
        _try(c, args.base + "/mission_stop")
        _try(c, args.base + "/reset")
        try:
            if c.get(args.base + "/state").json().get("wind_enabled"):
                c.get(args.base + "/wind?on=0")
        except Exception:
            pass
    time.sleep(1.0)

    webm = record(args.base, args.prompt, args.max_s)
    print(f"raw video: {webm}")


if __name__ == "__main__":
    main()
