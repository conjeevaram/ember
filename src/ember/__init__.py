"""Ember: Unitree G1 firefighting-humanoid demos in MuJoCo.

Two locomotion stacks, both rendered headlessly (EGL) and streamed to a browser:

- ``ember.locomotion`` -- Unitree's pretrained 12-DOF RL walker (robust velocity
  control + obstacle traversal) with a kinematic full-body arm overlay.
- ``ember.gmt`` -- the GMT whole-body motion-tracking policy (physically
  articulated arms), steered with velocity commands.

Importing this package selects the headless GL backend before MuJoCo loads one.
"""
import os

# Headless GPU offscreen rendering. Must be set before mujoco imports a GL
# backend, so we do it at package import time.
os.environ.setdefault("MUJOCO_GL", "egl")

__version__ = "0.1.0"
