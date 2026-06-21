# ember

Unitree G1 firefighting-humanoid demo in MuJoCo, rendered headlessly (EGL) and
streamed to a browser.

- **`ember.locomotion`** — Unitree's pretrained **12-DOF** RL walker (robust
  `(vx, vy, yaw)` velocity control + obstacle traversal) with a kinematic
  full-body arm overlay that carries a fire-hose nozzle. Port **8088**.

## Install

```bash
pip install -e .            # core (walker + streaming)
pip install -e ".[fire]"    # + OpenCV flame detection
```

## External assets (not vendored)

Cloned separately; paths are env-overridable (defaults in parentheses):

| Var | Default | Holds |
| --- | --- | --- |
| `UNITREE_RL_GYM` | `~/unitree_rl_gym` | G1 model, 12-DOF policy, deploy config |

Stream/camera knobs: `EMBER_W`/`EMBER_H` (640x360), `EMBER_FPS` (18),
`EMBER_QUALITY` (55), `EMBER_HOST`, `EMBER_CAM_W`/`EMBER_CAM_H`/`EMBER_CAM_FOVY`.

## Run

```bash
python scripts/run_walker.py --scene fire        # 12-DOF walker -> :8088
python scripts/build_scenes.py --force           # (re)generate scene XMLs
```

Browser controls: **WASD** drive, **Q/E** strafe, **Space** stop, **H**
auto-steer, **1/2/3** for carry/aim/down arm poses; plus a robot POV inset.

## Programmatic control (fire-controller hook)

```python
from ember import locomotion
locomotion.start(block=False, scene="fire")        # flat ground + flame at 4 m
locomotion.send_velocity_command(vx=0.5, yaw=0.2)  # clamped (vx, vy, yaw)
locomotion.get_state()                             # pose, velocity, fell flag
sim = locomotion.get_sim()
sim.set_overlay_arm_pose("aim")                    # preset name, OR ...
sim.set_overlay_arm_pose({"left_elbow": 0.9})      # continuous joint dict
```

## Fire controller (skeleton)

`ember.fire_controller` turns a camera frame into approach + aim commands:

- `FlameDetector` — HSV-threshold + largest-blob centroid (needs `[fire]`).
- `GroundPlaneProjector` — back-projects a pixel onto the ground plane.
- `FireController` — `SEARCH → TOO_FAR → IN_RANGE → SPRAYING → EXTINGUISHED`,
  actuating via injected `velocity_fn` / `arm_fn` (testable with fakes).
  `FireController.from_locomotion()` binds it to the live walker.

The nozzle-aiming ballistic IK is still a stub (`TODO(ballistic-ik)`); spray
currently uses the `"aim"` preset.
