"""Headless MJPEG streaming and the shared Flask scaffolding for the demo.

The sim renders frames on its own thread (which owns the EGL/GL context) and
publishes the latest JPEG into a :class:`FrameBuffer`. The web app only ever
reads that buffer, so it never touches MuJoCo -- keeping the physics/render and
HTTP concerns cleanly separated.
"""
from __future__ import annotations

import io
import threading
import time
from typing import Callable, Union

from PIL import Image

from .config import JPEG_QUALITY, RENDER_FPS, advertise_host


def encode_jpeg(rgb, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class FrameBuffer:
    """Thread-safe holder for the most recent encoded JPEG frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None

    def set(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg

    def get(self) -> bytes | None:
        with self._lock:
            return self._jpeg


FrameSource = Union[FrameBuffer, Callable[[], FrameBuffer | None]]


def _resolve_buffer(source: FrameSource) -> FrameBuffer | None:
    if callable(source):
        return source()
    return source


def mjpeg_response(frame_source: FrameSource, fps: float = RENDER_FPS):
    """Flask MJPEG stream from a :class:`FrameBuffer` or a zero-arg callable
    that returns the current buffer (re-fetched each frame for hot-swaps)."""
    from flask import Response

    def gen():
        boundary = b"--frame"
        interval = 1.0 / fps
        while True:
            buf = _resolve_buffer(frame_source)
            if buf is None:
                time.sleep(0.03)
                continue
            jpeg = buf.get()
            if jpeg is None:
                time.sleep(0.03)
                continue
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() +
                   b"\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(interval)

    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def create_app(
    *,
    page_html: str,
    frame_buffer: FrameBuffer | None = None,
    frame_source: FrameSource | None = None,
    state_fn: Callable[[], dict],
    register_routes: Callable[["Flask"], None] | None = None,
    fps: float = RENDER_FPS,
):
    """Build the Flask app: ``/`` (page), ``/stream`` (MJPEG), ``/state``
    (JSON). Demo-specific control endpoints (and extra feeds) are added via
    ``register_routes``. Pass ``frame_source`` (or legacy ``frame_buffer``)
    for the MJPEG feed."""
    from flask import Flask, jsonify

    src: FrameSource = frame_source if frame_source is not None else frame_buffer

    app = Flask(__name__)

    @app.route("/")
    def index():
        return page_html

    @app.route("/stream")
    def stream():
        return mjpeg_response(src, fps)

    @app.route("/state")
    def state():
        return jsonify(state_fn())

    if register_routes is not None:
        register_routes(app)
    return app


def serve(app, port: int, label: str = "viewer") -> threading.Thread:
    """Run the Flask app on a daemon thread and print the URL."""
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port,
                               threaded=True, use_reloader=False),
        daemon=True,
    )
    t.start()
    print(f"\n  {label}:  http://{advertise_host()}:{port}/\n")
    return t
