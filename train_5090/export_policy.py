"""
Load a trained G1 Playground policy and (a) sanity-check it walks, and
(b) export portable weights for deployment back on the tensor box.

Two deployment routes are supported; pick based on what you want to install on
the deploy machine (tensor / 5070 Ti):

  ROUTE A (recommended, exact): keep jax + brax + playground installed and run
    the policy with jax on CPU. The network is tiny, so 50 Hz CPU inference is
    trivial and the Blackwell/sm_120 GPU issue is irrelevant at inference time.
    Use load_policy() below.

  ROUTE B (no jax at deploy): dump the MLP weights + observation normalizer to
    policy_export.npz and run a numpy forward pass. Lighter deps, but you must
    keep the numpy forward in sync with brax's network (activation + tanh mode).

Usage:
    python export_policy.py --ckpt checkpoints/G1JoystickRoughTerrain \
                            --eval --steps 500            # ROUTE A self-test
    python export_policy.py --ckpt checkpoints/G1JoystickRoughTerrain \
                            --dump policy_export.npz       # ROUTE B export
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def _load_params(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    pkl = ckpt_dir / "final_params.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    # Fall back to the latest orbax checkpoint subdir.
    from orbax import checkpoint as ocp
    steps = sorted((p for p in ckpt_dir.iterdir() if p.name.isdigit()),
                   key=lambda p: int(p.name))
    if not steps:
        raise FileNotFoundError(f"No final_params.pkl or orbax steps in {ckpt_dir}")
    latest = steps[-1]
    print("Restoring orbax checkpoint:", latest)
    return ocp.PyTreeCheckpointer().restore(str(latest))


def load_policy(ckpt_dir, env_name="G1JoystickRoughTerrain"):
    """ROUTE A: return (env, jit_policy_fn). policy_fn(obs_np)->action_np on CPU."""
    import jax
    jax.config.update("jax_platform_name", "cpu")  # CPU inference, no Blackwell issue
    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params
    from brax.training.agents.ppo import networks as ppo_networks

    env = registry.load(env_name)
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    nf = ppo_params.network_factory
    networks = nf(env.observation_size, env.action_size)
    make_policy = ppo_networks.make_inference_fn(networks)

    params = _load_params(ckpt_dir)
    policy = make_policy(params, deterministic=True)

    import jax.numpy as jnp
    def policy_fn(obs_np):
        act, _ = policy({"state": jnp.asarray(obs_np)} if isinstance(obs_np, dict)
                        else jnp.asarray(obs_np), jax.random.PRNGKey(0))
        return np.asarray(act)

    jit_fn = jax.jit(lambda o: policy(o, jax.random.PRNGKey(0))[0])
    return env, policy_fn, jit_fn


def evaluate(ckpt_dir, env_name, steps, video):
    """Run the policy in the Playground env headless; report reward, save video."""
    import jax
    import jax.numpy as jnp
    from mujoco_playground import registry
    env, policy_fn, jit_fn = load_policy(ckpt_dir, env_name)

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    frames, total_r = [], 0.0
    for i in range(steps):
        act = jit_fn(state.obs)
        state = step(state, act)
        total_r += float(state.reward)
        if video and i % 2 == 0:
            frames.append(env.render(state))
    print(f"eval: {steps} steps, total reward {total_r:.1f}, "
          f"avg {total_r/steps:.3f}/step")
    if video and frames:
        import mediapy
        mediapy.write_video(video, frames, fps=25)
        print("wrote", video)


def dump_npz(ckpt_dir, out):
    """ROUTE B: flatten params to a .npz (numpy). Keys preserve the pytree path
    so you can reconstruct the MLP forward pass without jax."""
    import jax
    params = _load_params(ckpt_dir)
    flat = {}
    leaves_with_path = jax.tree_util.tree_flatten_with_path(params)[0]
    for path, leaf in leaves_with_path:
        key = "/".join(str(getattr(p, "key", getattr(p, "idx", p))) for p in path)
        flat[key] = np.asarray(leaf)
    np.savez(out, **flat)
    print(f"wrote {out} with {len(flat)} arrays")
    print("keys:", list(flat.keys())[:12], "..." if len(flat) > 12 else "")
    print("NOTE: reconstruct obs normalization (mean/std) + MLP (Dense+activation,"
          " NormalTanh mode = tanh(mean)) in numpy to deploy without jax.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoints/<env> dir")
    p.add_argument("--env", default="G1JoystickRoughTerrain")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--video", default="")
    p.add_argument("--dump", default="")
    args = p.parse_args()

    if args.eval:
        evaluate(args.ckpt, args.env, args.steps, args.video or None)
    if args.dump:
        dump_npz(args.ckpt, args.dump)
    if not args.eval and not args.dump:
        print("Nothing to do; pass --eval and/or --dump. See module docstring.")


if __name__ == "__main__":
    main()
