"""Fire perception + approach state machine for the G1 firefighter (skeleton).

Pipeline:  RGB frame -> :class:`FlameDetector` (HSV blob) -> 2D centroid
        -> :class:`GroundPlaneProjector` (optional) -> 3D world point
        -> :class:`FireController` state machine -> velocity + arm commands.

This is the SKELETON: the detector + projector are functional, and the state
machine drives toward the flame and raises the arms, but the spray itself uses
the canned ``"aim"`` preset. The ballistic inverse-kinematics that aims the
nozzle (replacing ``"aim"`` with a computed :mod:`ember.arms` joint dict) is a
later layer -- the hook is marked ``TODO(ballistic-ik)`` in
:meth:`FireController.step`.

The controller is decoupled from the sim: it actuates through two callables
(``velocity_fn`` and ``arm_fn``) so you can unit-test the state machine with
fakes, or feed it webcam/video frames before wiring it to MuJoCo. Use
:meth:`FireController.from_locomotion` to bind it to the live walker.

OpenCV (``cv2``) is imported lazily inside :class:`FlameDetector` so this module
imports without it; install with ``pip install opencv-python-headless``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .config import clamp

Pixel = Tuple[float, float]
HsvRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


# --------------------------------------------------------------------------- #
# Flame detection
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """Result of one detector pass."""
    centroid: Optional[Pixel] = None        # (cx, cy) in pixels, or None
    area: float = 0.0                        # largest-blob area in px^2
    bbox: Optional[Tuple[int, int, int, int]] = None   # x, y, w, h
    mask: Optional[np.ndarray] = None        # uint8 binary mask (debug)

    @property
    def found(self) -> bool:
        return self.centroid is not None


# Default flame colour gates in OpenCV HSV (H in 0-179). Two bands: deep
# red-orange near the hue wrap, and orange-yellow. Tuned to the sim flame prop
# (orange 0.95,0.35,0.05 + yellow 1.0,0.80,0.10) and typical real flame.
DEFAULT_HSV_RANGES: tuple[HsvRange, ...] = (
    ((0, 120, 150), (15, 255, 255)),
    ((16, 90, 150), (38, 255, 255)),
)


class FlameDetector:
    """HSV-threshold + largest-blob flame detector.

    Args:
        hsv_ranges: list of (lo, hi) HSV gates OR'd together.
        min_area: ignore blobs smaller than this (px^2).
        input_format: ``"rgb"`` (MuJoCo renderer) or ``"bgr"`` (cv2 webcam/video).
        open_ksize / close_ksize: morphology kernel sizes (despeckle / fill).
    """

    def __init__(self, hsv_ranges: Sequence[HsvRange] = DEFAULT_HSV_RANGES,
                 min_area: float = 80.0, input_format: str = "rgb",
                 open_ksize: int = 3, close_ksize: int = 7):
        try:
            import cv2  # lazy: keep this module importable without OpenCV
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "FlameDetector needs OpenCV: pip install opencv-python-headless"
            ) from e
        self._cv2 = cv2
        self.hsv_ranges = [tuple(map(np.array, r)) for r in hsv_ranges]
        self.min_area = float(min_area)
        if input_format not in ("rgb", "bgr"):
            raise ValueError("input_format must be 'rgb' or 'bgr'")
        self.input_format = input_format
        self._open_k = np.ones((open_ksize, open_ksize), np.uint8)
        self._close_k = np.ones((close_ksize, close_ksize), np.uint8)

    def _mask(self, frame: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        code = cv2.COLOR_RGB2HSV if self.input_format == "rgb" else cv2.COLOR_BGR2HSV
        hsv = cv2.cvtColor(frame, code)
        mask = None
        for lo, hi in self.hsv_ranges:
            band = cv2.inRange(hsv, lo, hi)
            mask = band if mask is None else cv2.bitwise_or(mask, band)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_k)
        return mask

    def detect(self, frame: np.ndarray) -> Optional[Pixel]:
        """Return the flame centroid (cx, cy) in pixels, or None."""
        return self.detect_verbose(frame).centroid

    def detect_verbose(self, frame: np.ndarray) -> Detection:
        """Full detection result (centroid, area, bbox, mask) for debugging."""
        cv2 = self._cv2
        mask = self._mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Detection(mask=mask)
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < self.min_area:
            return Detection(area=area, mask=mask)
        m = cv2.moments(largest)
        if m["m00"] == 0:
            return Detection(area=area, mask=mask)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        return Detection(centroid=(cx, cy), area=area,
                         bbox=tuple(cv2.boundingRect(largest)), mask=mask)


# --------------------------------------------------------------------------- #
# Ground-plane back-projection
# --------------------------------------------------------------------------- #
class GroundPlaneProjector:
    """Back-project a pixel onto the world ground plane (z = ``ground_z``).

    Uses a pinhole model in the OpenCV camera convention (x right, y down,
    z forward). ``cam_rot`` is the 3x3 rotation whose columns are the camera
    axes expressed in world coordinates (i.e. world_point = cam_rot @ cam_point
    + cam_pos).
    """

    def __init__(self, intrinsics: Tuple[float, float, float, float],
                 ground_z: float = 0.0):
        self.fx, self.fy, self.cx, self.cy = intrinsics
        self.ground_z = float(ground_z)

    @classmethod
    def from_fovy(cls, fovy_deg: float, width: int, height: int,
                  ground_z: float = 0.0) -> "GroundPlaneProjector":
        """Build intrinsics from a vertical FOV (MuJoCo cameras specify fovy)."""
        fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
        return cls((fy, fy, width / 2.0, height / 2.0), ground_z)

    def project(self, pixel: Pixel, cam_pos, cam_rot) -> Optional[np.ndarray]:
        """Pixel -> 3D world point on the ground plane, or None if the ray does
        not cross the plane in front of the camera."""
        u, v = pixel
        cam_pos = np.asarray(cam_pos, dtype=float)
        cam_rot = np.asarray(cam_rot, dtype=float)
        ray_cam = np.array([(u - self.cx) / self.fx,
                            (v - self.cy) / self.fy, 1.0])
        ray_world = cam_rot @ ray_cam
        if abs(ray_world[2]) < 1e-9:
            return None
        t = (self.ground_z - cam_pos[2]) / ray_world[2]
        if t <= 0:  # plane is behind the camera
            return None
        return cam_pos + t * ray_world


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #
class FireState(Enum):
    SEARCH = "SEARCH"            # no flame in view -> rotate to find one
    TOO_FAR = "TOO_FAR"         # flame seen, out of spray range -> approach
    IN_RANGE = "IN_RANGE"       # close enough -> stop, square up, raise arms
    SPRAYING = "SPRAYING"       # aimed and spraying
    EXTINGUISHED = "EXTINGUISHED"  # done -> hold


@dataclass
class StepResult:
    state: FireState
    centroid: Optional[Pixel]
    area: float
    distance: Optional[float]
    cmd: dict
    arm: object  # preset name or joint dict


def _noop_velocity(vx=None, vy=None, yaw=None):
    return None


def _noop_arm(pose):
    return None


class FireController:
    """SEARCH -> TOO_FAR -> IN_RANGE -> SPRAYING -> EXTINGUISHED.

    Actuates via ``velocity_fn(vx, vy, yaw)`` and ``arm_fn(pose)``; both default
    to no-ops so the state machine can be exercised in tests. Range is taken
    from the known ``fire_position_world`` + ``robot_state['pos']`` when
    available (ground truth), else from a projected world point, else from a
    blob-area proxy (bigger blob == closer).
    """

    def __init__(self, velocity_fn: Callable = _noop_velocity,
                 arm_fn: Callable = _noop_arm,
                 detector: Optional[FlameDetector] = None,
                 projector: Optional[GroundPlaneProjector] = None,
                 fire_position_world: Optional[Sequence[float]] = None,
                 image_size: Tuple[int, int] = (640, 360),
                 in_range_m: float = 1.6, approach_vx: float = 0.5,
                 search_yaw: float = 0.5, center_tol: float = 0.12,
                 yaw_gain: float = 0.9, area_in_range: float = 9000.0,
                 spray_ticks: int = 150, lost_grace: int = 8):
        self.velocity_fn = velocity_fn
        self.arm_fn = arm_fn
        self.detector = detector  # may be None until OpenCV is available
        self.projector = projector
        self.fire_position_world = (np.asarray(fire_position_world, dtype=float)
                                    if fire_position_world is not None else None)
        self.image_w, self.image_h = image_size
        self.in_range_m = in_range_m
        self.approach_vx = approach_vx
        self.search_yaw = search_yaw
        self.center_tol = center_tol
        self.yaw_gain = yaw_gain
        self.area_in_range = area_in_range
        self.spray_ticks = spray_ticks
        self.lost_grace = lost_grace

        self.state = FireState.SEARCH
        self.last_detection: Optional[Detection] = None
        self._spray_count = 0
        self._lost_count = 0
        self._last_cmd = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
        self._last_arm: object = "carry"

    @classmethod
    def from_locomotion(cls, **kwargs) -> "FireController":
        """Bind to the live 12-DOF walker + overlay (see :mod:`ember.locomotion`).
        Defaults the fire position to the scene's known flame location."""
        from . import locomotion, scenes
        kwargs.setdefault("fire_position_world", scenes.FIRE_POSITION_WORLD)
        return cls(
            velocity_fn=locomotion.send_velocity_command,
            arm_fn=lambda pose: locomotion.get_sim().set_overlay_arm_pose(pose),
            **kwargs,
        )

    def reset(self):
        self.state = FireState.SEARCH
        self._spray_count = 0
        self._lost_count = 0
        self.last_detection = None

    # -- range estimation ---------------------------------------------------- #
    def _distance(self, robot_state: dict, det: Detection) -> Optional[float]:
        pos = robot_state.get("pos") if robot_state else None
        if self.fire_position_world is not None and pos is not None:
            d = np.asarray(pos, dtype=float)[:2] - self.fire_position_world[:2]
            return float(np.linalg.norm(d))
        # Perception fallback: project the centroid if we have a camera pose.
        if (self.projector is not None and det.centroid is not None
                and robot_state and "cam_pos" in robot_state
                and "cam_rot" in robot_state and pos is not None):
            world = self.projector.project(det.centroid, robot_state["cam_pos"],
                                           robot_state["cam_rot"])
            if world is not None:
                d = np.asarray(pos, dtype=float)[:2] - world[:2]
                return float(np.linalg.norm(d))
        return None  # caller falls back to the area proxy

    def _in_range(self, distance: Optional[float], det: Detection) -> bool:
        if distance is not None:
            return distance <= self.in_range_m
        return det.area >= self.area_in_range  # no metric range -> area proxy

    # -- actuation helpers --------------------------------------------------- #
    def _drive(self, vx=0.0, vy=0.0, yaw=0.0):
        self._last_cmd = {"vx": vx, "vy": vy, "yaw": yaw}
        self.velocity_fn(vx=vx, vy=vy, yaw=yaw)

    def _arm(self, pose):
        self._last_arm = pose
        self.arm_fn(pose)

    # -- one control tick ---------------------------------------------------- #
    def step(self, rgb_frame: np.ndarray, robot_state: Optional[dict] = None
             ) -> StepResult:
        """Advance the state machine one tick from a frame + robot state.

        ``robot_state`` is the dict from ``locomotion.get_state()`` (uses
        ``pos``; optionally ``cam_pos`` / ``cam_rot`` for the perception range
        path). Returns a :class:`StepResult` describing the decision."""
        if self.detector is None:
            raise RuntimeError("FireController.detector is not set "
                               "(install OpenCV and pass a FlameDetector)")
        robot_state = robot_state or {}
        det = self.detector.detect_verbose(rgb_frame)
        self.last_detection = det

        # Lost the flame: hold a short grace period, then search.
        if not det.found:
            self._lost_count += 1
            if (self.state not in (FireState.SPRAYING, FireState.EXTINGUISHED)
                    or self._lost_count > self.lost_grace):
                self.state = FireState.SEARCH
            if self.state == FireState.SEARCH:
                self._drive(yaw=self.search_yaw)
                self._arm("carry")
            return self._result(det, None)
        self._lost_count = 0

        cx, _ = det.centroid
        x_err = (cx - self.image_w / 2.0) / (self.image_w / 2.0)  # -1..+1
        yaw = clamp(-self.yaw_gain * x_err, -0.8, 0.8)
        distance = self._distance(robot_state, det)
        in_range = self._in_range(distance, det)

        if self.state in (FireState.SEARCH, FireState.TOO_FAR):
            self.state = FireState.IN_RANGE if in_range else FireState.TOO_FAR

        if self.state == FireState.TOO_FAR:
            # Slow down while still turning to center the flame.
            vx = self.approach_vx * (0.4 if abs(x_err) > 0.4 else 1.0)
            self._drive(vx=vx, yaw=yaw)
            self._arm("carry")

        elif self.state == FireState.IN_RANGE:
            self._drive(vx=0.0, yaw=yaw)   # square up to the flame
            self._arm("aim")
            if abs(x_err) < self.center_tol:
                self.state = FireState.SPRAYING
                self._spray_count = 0

        elif self.state == FireState.SPRAYING:
            self._drive(vx=0.0, yaw=0.0)
            # TODO(ballistic-ik): replace "aim" with a computed arm joint dict
            # that points the nozzle at fire_position_world given the standoff.
            self._arm("aim")
            self._spray_count += 1
            if self._spray_count >= self.spray_ticks:
                self.state = FireState.EXTINGUISHED

        elif self.state == FireState.EXTINGUISHED:
            self._drive(vx=0.0, yaw=0.0)
            self._arm("carry")

        return self._result(det, distance)

    def _result(self, det: Detection, distance: Optional[float]) -> StepResult:
        return StepResult(state=self.state, centroid=det.centroid,
                          area=det.area, distance=distance,
                          cmd=dict(self._last_cmd), arm=self._last_arm)
