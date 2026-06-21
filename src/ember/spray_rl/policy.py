"""Spray-aim policy (SB3 checkpoint) with PID fallback."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ember.config import VX_RANGE, VY_RANGE, YAW_RANGE, clamp

DEFAULT_CHECKPOINT = Path("train_5070ti/spray_ppo.zip")


class PIDSprayController:
    """Proportional spray corrector on the body-frame landing error.

    The water lands roughly where the base points, so to cancel an error we move
    the base the opposite way: overshoot -> step back, landing-left -> strafe
    (and rotate) right. Nozzle pitch is left at its current elevation."""

    def __init__(self, *, kp_vx: float = 1.2, kp_vy: float = 1.4, kp_yaw: float = 0.4):
        self.kp_vx = kp_vx
        self.kp_vy = kp_vy
        self.kp_yaw = kp_yaw

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        err_fwd, err_lat = float(obs[2]), float(obs[3])
        vx = clamp(-self.kp_vx * err_fwd, *VX_RANGE)
        vy = clamp(-self.kp_vy * err_lat, *VY_RANGE)
        yaw = clamp(-self.kp_yaw * err_lat, *YAW_RANGE)
        return np.array([vx, vy, yaw], dtype=np.float32)


class SprayPolicy:
    """Wrapper around a trained SB3 PPO model."""

    def __init__(self, model):
        self._model = model

    @classmethod
    def load(cls, path: str | Path) -> SprayPolicy:
        from stable_baselines3 import PPO
        model = PPO.load(str(path))
        return cls(model)

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action, _ = self._model.predict(obs, deterministic=True)
        return np.asarray(action[0], dtype=np.float32)


def _self_test(policy: SprayPolicy, *, episodes: int = 4, seed: int = 0) -> bool:
    """Roll the policy out and check steady-state aim (second half of each
    episode, after it has had time to converge). Thresholds sit well above a
    broken/world-frame policy (~0.2 on-target) and below a working one (~0.8)."""
    from .env import SprayEnv
    env = SprayEnv(seed=seed)
    on, dists = 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        for t in range(env.max_steps):
            obs, _, term, trunc, info = env.step(policy.act(obs))
            if t >= env.max_steps // 2:
                on += int(info["on_target"])
                dists.append(info["landing_dist"])
            if term or trunc:
                break
    return bool(dists) and (on / len(dists)) > 0.5 and float(np.mean(dists)) < 0.30


def load_spray_controller(path: str | Path | None = None) -> PIDSprayController | SprayPolicy:
    """Load RL policy if checkpoint exists and passes a quick roll-out test."""
    ckpt = Path(path or DEFAULT_CHECKPOINT)
    if not ckpt.is_file():
        return PIDSprayController()
    try:
        policy = SprayPolicy.load(ckpt)
        if _self_test(policy):
            return policy
    except Exception:
        pass
    return PIDSprayController()
