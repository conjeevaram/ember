"""Centralized configuration: external asset paths, stream settings, command
ranges, and small math helpers. Everything here is overridable via environment
variables so the code carries no machine-specific absolute paths.

Environment variables
---------------------
External repos (cloned separately; see README):
    UNITREE_RL_GYM   path to unitree_rl_gym   (default: ~/unitree_rl_gym)

Video stream:
    EMBER_W, EMBER_H     frame size           (default: 640x360)
    EMBER_FPS            stream framerate      (default: 18)
    EMBER_QUALITY        JPEG quality 1-95     (default: 55)
    EMBER_HOST           advertised host shown in the startup URL (default:
                         auto-detected primary IP)

Robot ego camera (walker overlay POV):
    EMBER_CAM_W, EMBER_CAM_H   frame size      (default: 384x288)
    EMBER_CAM_FOVY             vertical FOV deg (default: 70)
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

import numpy as np


def _path_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


# --- external repositories (not vendored; cloned separately) --------------- #
HOME = Path.home()
UNITREE_RL_GYM = _path_env("UNITREE_RL_GYM", HOME / "unitree_rl_gym")

# Unitree 12-DOF RL walker assets.
G1_MODEL_DIR = UNITREE_RL_GYM / "resources" / "robots" / "g1_description"
POLICY_PATH = UNITREE_RL_GYM / "deploy" / "pre_train" / "g1" / "motion.pt"
DEPLOY_CONFIG_PATH = UNITREE_RL_GYM / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"

# --- video stream ---------------------------------------------------------- #
# Defaults tuned to stay smooth over a Tailscale DERP relay (~2 Mbps).
RENDER_W = int(os.environ.get("EMBER_W", 640))
RENDER_H = int(os.environ.get("EMBER_H", 360))
RENDER_FPS = float(os.environ.get("EMBER_FPS", 18))
JPEG_QUALITY = int(os.environ.get("EMBER_QUALITY", 55))

# Robot-mounted (ego / "helmet cam") view: a forward-looking camera on the
# overlay body's torso. Used for the live POV feed and (later) fire perception.
CAM_W = int(os.environ.get("EMBER_CAM_W", 384))
CAM_H = int(os.environ.get("EMBER_CAM_H", 288))
CAM_FOVY = float(os.environ.get("EMBER_CAM_FOVY", 70))

# --- velocity command clamps (m/s, m/s, rad/s) ----------------------------- #
VX_RANGE = (-0.6, 1.0)
VY_RANGE = (-0.4, 0.4)
YAW_RANGE = (-0.8, 0.8)


def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def gravity_orientation(quaternion) -> np.ndarray:
    """Projected gravity in the base frame from a base quaternion [w, x, y, z]."""
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def quat_yaw(quaternion) -> float:
    """World yaw (rad) from a base quaternion [w, x, y, z]."""
    w, x, y, z = quaternion
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def advertise_host() -> str:
    """Host to show in the startup URL. Uses $EMBER_HOST if set, else the
    primary outbound IP (works for LAN/Tailscale), else localhost."""
    host = os.environ.get("EMBER_HOST")
    if host:
        return host
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "localhost"
