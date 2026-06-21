# ember

Unitree G1 firefighting-humanoid demo in MuJoCo, rendered headlessly (EGL) and
streamed to a browser.

- **`ember.sim`** — Unitree's pretrained **12-DOF** RL walker (robust
  `(vx, vy, yaw)` velocity control + obstacle traversal) with a kinematic
  full-body arm overlay that carries a fire-hose nozzle, plus the water-jet
  ballistics and A*/approach navigation hooks.
- **`ember.viewer`** — discovers the named + procedural scenes, hot-swaps them,
  and streams the sim to a browser (driving, nav map, autonomous approach).
  Port **8088**.

## Install

```bash
pip install -e .
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
auto-steer; plus a robot POV inset. The arms are locked in a hose-carry pose.

## Programmatic control

`viewer.start` returns the live `G1Sim`, which exposes the control API directly:

```python
from ember import viewer
sim = viewer.start(block=False, scene="fire")  # flat ground + flame at 4 m
sim.set_command(vx=0.5, yaw=0.2)               # clamped (vx, vy, yaw)
sim.get_state()                                # pose, velocity, fell flag, fires
```

For a procedural scene, pass a `SceneSpec` (`viewer.start(spec=...)`); the robot
spawns at `spec.start` and `sim.set_approach(True)` autonomously navigates to the
nearest burning fire and faces it.
