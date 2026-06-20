#!/usr/bin/env bash
# One-time setup on the 5090 desktop. Creates a venv, installs JAX (CUDA12 /
# Blackwell), MuJoCo Playground, and deps, then runs the GPU gate check.
#
# Usage:   bash setup_5090.sh
# Assumes: NVIDIA driver with CUDA 12.8+ (check with `nvidia-smi`), Python 3.11+.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

echo "==> Python: $(python3 --version)"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip wheel

# JAX with CUDA 12 (must be recent enough for sm_120 / Blackwell).
echo "==> Installing jax[cuda12] ..."
pip install -U "jax[cuda12]"

# MuJoCo Playground + Brax PPO stack. playground pulls mujoco>=3.6, mujoco-mjx,
# brax, flax, orbax, etc. (see pyproject on PyPI).
echo "==> Installing mujoco_playground + training stack ..."
pip install -U playground mediapy tensorboardX

# Trigger one-time Menagerie asset download for the G1 envs.
echo "==> Pre-downloading G1 assets ..."
python -c "from mujoco_playground import locomotion; locomotion.load('G1JoystickRoughTerrain'); print('assets ok')" || \
  echo "(asset preload failed -- will download on first train run)"

echo "==> GATE CHECK: JAX-on-Blackwell"
python "$HERE/verify_jax_gpu.py"

echo
echo "Setup done. Activate with:  source $VENV/bin/activate"
echo "Then train with:            python $HERE/train_g1.py --env G1JoystickRoughTerrain"
