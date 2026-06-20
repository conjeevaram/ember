# G1 whole-body terrain policy — training kit (run on the 5090)

Goal: train a Unitree G1 **velocity-command** locomotion policy
(`G1JoystickRoughTerrain`) in MuJoCo Playground that handles **rough terrain
(stairs / debris)** and is trained on the **full articulated body**, so the arms
can be commanded without destabilizing it. Same `(vx, vy, yaw_rate)` interface
as the current `g1_walk.py`, so it drops into the existing web-viewer + WASD
demo — but with real terrain robustness and physical arms.

This is the **upgrade path**. The hackathon demo already works today on the
12-DOF walker + kinematic arm overlay (port 8088). Don't block the demo on this.

## Why this and not what we have
- **8088 (12-DOF Unitree policy):** great blind locomotion, but arms are welded
  decoration; we fake them with a kinematic render overlay.
- **8089 (GMT):** real arms, but a flat-ground motion tracker — poor on terrain.
- **This policy:** velocity command + rough terrain + full body in one RL run.
  Arms become commandable on top (Playground's joystick policy + a high-level
  arm command, the standard decomposition).

## Hardware / time
- **Use the 5090 (32 GB).** MJX speed = massive env parallelism = needs VRAM.
  The 5070 Ti laptop (12 GB) is fine for *running* policies, not training them.
- Expect ~**1.5–3 h** for a solid rough-terrain policy on a 5090, plus a
  one-time ~1 h setup (mostly the JAX/Blackwell gate below).

## Steps

### 0. Setup (one time)
```bash
bash setup_5090.sh           # venv + jax[cuda12] + playground, runs the gate
source .venv/bin/activate
```

### 1. GATE: JAX must see the GPU (make-or-break on Blackwell)
The 5090 is sm_120; an old jaxlib silently runs MJX on CPU (≈unusable).
```bash
python verify_jax_gpu.py     # must print your 5090 and "PASS"
```
If it fails: `pip install -U "jax[cuda12]"` (need a build with sm_120),
confirm `nvidia-smi` shows CUDA 12.8+/13.x. **Do not train until this passes.**

### 2. Train
```bash
python train_g1.py --env G1JoystickRoughTerrain --num_envs 8192
# OOM? drop --num_envs to 4096/2048.
# Flat-only (faster, no stairs): --env G1JoystickFlatTerrain
```
Equivalent official CLI (also fine): 
```bash
train-jax-ppo --env_name G1JoystickRoughTerrain --domain_randomization \
              --num_timesteps 200000000 --num_envs 8192
```
Checkpoints land in `checkpoints/G1JoystickRoughTerrain/`. Watch `reward` climb
in the console / `progress.json`.

### 3. Verify it learned (still on the 5090)
```bash
python export_policy.py --ckpt checkpoints/G1JoystickRoughTerrain \
                        --eval --steps 500 --video eval.mp4
```
A healthy joystick policy holds high reward and the video shows it walking and
tracking commands over terrain.

### 4. Bring it back to the tensor box
Copy the `checkpoints/G1JoystickRoughTerrain/` dir over (scp/rsync). Two deploy
routes (see `export_policy.py` docstring):
- **Route A (recommended):** install `jax` (CPU) + `brax` + `playground` on
  tensor and call `load_policy()` — exact, and CPU inference of this tiny MLP is
  trivial at 50 Hz. Blackwell GPU issues don't apply at inference.
- **Route B:** `--dump policy_export.npz` and run a numpy forward pass (no jax),
  if you want minimal deps.

### 5. Wire into the demo
The trained policy uses Playground's G1 joystick **observation/action layout**,
which differs from the current `unitree_rl_gym` policy in `g1_walk.py`. Final
integration = a small sim2sim step: replicate the env's observation in our plain
MuJoCo loop and swap the policy call, keeping the EGL web viewer and
`send_velocity_command(vx, vy, yaw)` interface. Do this once weights exist.

## Files
- `verify_jax_gpu.py` — Blackwell gate check (run first).
- `setup_5090.sh` — venv + deps + asset preload + gate.
- `train_g1.py` — programmatic Brax PPO training w/ checkpoints + progress log.
- `export_policy.py` — eval the policy and export weights (Route A / Route B).

## Risks / notes
- The JAX-on-Blackwell setup is the main thing that can eat an hour — the gate
  exists to catch it before you waste a training run.
- Playground's API can drift; if `train_g1.py` errors on a renamed arg, the
  `train-jax-ppo` CLI is the officially-maintained entry point.
- "Commandable arms" via this policy = the joystick locomotion policy + a
  high-level arm command layered on top (not arbitrary dexterous manipulation).
