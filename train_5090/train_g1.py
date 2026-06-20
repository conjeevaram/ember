"""
Train a Unitree G1 joystick (velocity-command) locomotion policy in MuJoCo
Playground (MJX) with Brax PPO, on the 5090.

Why this env: G1JoystickRoughTerrain learns to track a (vx, vy, yaw_rate)
command over rough/uneven terrain -- the SAME command interface as the existing
g1_walk.py send_velocity_command(), but trained for terrain (stairs/debris) and
on the full articulated body, so moving the arms no longer destabilizes it.

Run the GPU gate first:   python verify_jax_gpu.py
Then:                     python train_g1.py --env G1JoystickRoughTerrain

Checkpoints are written to ./checkpoints/<env>/<step>/ via Orbax. Point
export_policy.py at the latest one when training finishes.

Notes:
- This mirrors the documented Playground notebook workflow (registry.load ->
  brax_ppo_config -> ppo.train). If the Playground API has drifted by the time
  you run it, the equivalent official CLI is:
    train-jax-ppo --env_name G1JoystickRoughTerrain --domain_randomization \
                  --num_timesteps 200000000 --num_envs 8192
- On a 5090 (32 GB) expect roughly 1.5-3 h for a solid rough-terrain policy.
"""
import argparse
import functools
import json
import time
from datetime import datetime
from pathlib import Path

import jax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="G1JoystickRoughTerrain",
                   help="G1JoystickRoughTerrain (terrain) or G1JoystickFlatTerrain")
    p.add_argument("--num_timesteps", type=int, default=200_000_000)
    p.add_argument("--num_envs", type=int, default=8192,
                   help="parallel envs; 8192 fits a 32GB 5090, drop if OOM")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="checkpoints")
    args = p.parse_args()

    # Fail fast if JAX can't see the GPU (Blackwell gate).
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpus:
        raise SystemExit("No GPU visible to JAX -- run verify_jax_gpu.py first.")
    print("Training on:", gpus)

    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    env = registry.load(args.env)
    eval_env = registry.load(args.env)
    ppo_params = locomotion_params.brax_ppo_config(args.env)
    randomizer = registry.get_domain_randomizer(args.env)

    # Allow overriding the headline budget from the CLI.
    ppo_params.num_timesteps = args.num_timesteps
    ppo_params.num_envs = args.num_envs
    ppo_params.seed = args.seed

    ckpt_dir = Path(args.outdir).resolve() / args.env
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print("Checkpoints ->", ckpt_dir)

    t0 = time.time()
    progress_log = []

    def progress(step, metrics):
        r = float(metrics.get("eval/episode_reward", float("nan")))
        dt = time.time() - t0
        print(f"[{dt/60:6.1f} min] step {step:>12,}  reward {r:8.2f}")
        progress_log.append({"step": int(step), "reward": r, "wall_s": dt})
        (ckpt_dir / "progress.json").write_text(json.dumps(progress_log))

    # brax_ppo_config carries a network_factory partial; pass it through.
    train_fn = functools.partial(
        ppo.train,
        **{k: v for k, v in dict(ppo_params).items()
           if k not in ("network_factory",)},
        network_factory=ppo_params.network_factory,
        randomization_fn=randomizer,
        progress_fn=progress,
        save_checkpoint_path=str(ckpt_dir),
    )

    make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env)

    # Also drop a final params pickle next to the orbax checkpoints for easy
    # loading by export_policy.py.
    import pickle
    final = ckpt_dir / "final_params.pkl"
    with open(final, "wb") as f:
        pickle.dump(params, f)
    meta = {
        "env": args.env,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "num_timesteps": args.num_timesteps,
        "num_envs": args.num_envs,
        "obs_size": int(env.observation_size) if hasattr(env, "observation_size") else None,
        "action_size": int(env.action_size) if hasattr(env, "action_size") else None,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved final params ->", final)
    print("Done in %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
