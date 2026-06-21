"""Train PPO spray-aim policy on the kinematic surrogate."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from .env import SprayEnv


def _make_env(seed: int):
    def _init():
        env = SprayEnv(seed=seed)
        return env
    return _init


def evaluate_policy(model: PPO, *, n_episodes: int = 20, seed: int = 123) -> dict:
    env = SprayEnv(seed=seed)
    on_targets: list[float] = []
    dists: list[float] = []
    rewards: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_r = 0.0
        ep_on = 0
        ep_d: list[float] = []
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            ep_on += int(info["on_target"])
            ep_d.append(info["landing_dist"])
            if term or trunc:
                break
        on_targets.append(ep_on / max(len(ep_d), 1))
        dists.extend(ep_d)
        rewards.append(ep_r)
    return {
        "mean_reward": float(np.mean(rewards)),
        "on_target_rate": float(np.mean(on_targets)),
        "mean_landing_error": float(np.mean(dists)),
    }


def train(*, n_envs: int = 32, total_timesteps: int = 1_000_000,
          checkpoint: Path | None = None, seed: int = 0) -> dict:
    checkpoint = checkpoint or Path("train_5070ti/spray_ppo.zip")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", end="")
    if device == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print(" (CPU fallback)")

    env = SubprocVecEnv([_make_env(seed + i) for i in range(n_envs)])
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
        device=device,
        seed=seed,
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    wall = time.time() - t0
    model.save(str(checkpoint.with_suffix("")))  # SB3 adds .zip

    metrics = evaluate_policy(model, n_episodes=30, seed=999)
    metrics["wall_seconds"] = wall
    metrics["n_envs"] = n_envs
    metrics["total_timesteps"] = total_timesteps
    metrics["checkpoint"] = str(checkpoint)

    print(f"\nTraining wall time: {wall:.1f}s")
    print(f"mean_reward={metrics['mean_reward']:.3f} "
          f"on_target_rate={metrics['on_target_rate']:.3f} "
          f"mean_landing_error={metrics['mean_landing_error']:.3f}m")
    env.close()
    return metrics


def main():
    p = argparse.ArgumentParser(description="Train spray-aim PPO policy")
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--checkpoint", type=Path, default=Path("train_5070ti/spray_ppo.zip"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(n_envs=args.n_envs, total_timesteps=args.timesteps,
          checkpoint=args.checkpoint, seed=args.seed)


if __name__ == "__main__":
    main()
