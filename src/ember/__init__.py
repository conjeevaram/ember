"""Ember: Unitree G1 firefighting-humanoid demos in MuJoCo.

``ember.sim`` runs Unitree's pretrained 12-DOF RL walker (robust velocity
control + obstacle traversal) with a kinematic full-body arm overlay that
carries a fire-hose nozzle, rendered headlessly (EGL). ``ember.viewer`` hot-swaps
scenes and streams the sim to a browser.

Importing this package selects the headless GL backend before MuJoCo loads one.
"""
import os

# Headless GPU offscreen rendering. Must be set before mujoco imports a GL
# backend, so we do it at package import time.
os.environ.setdefault("MUJOCO_GL", "egl")

__version__ = "0.1.0"
