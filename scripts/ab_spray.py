#!/usr/bin/env python3
"""A/B: reactive PID vs learned RL spray-correction, calm air vs gusty wind.

The water jet has a real time-of-flight, so where it lands is only *observed*
after a short delay, while the wind is sensed instantly at the nozzle. A purely
reactive controller (proportional on the delayed landing error) must therefore
chase stale information and trails the gusts; the RL policy also gets the instant
wind reading and learns to feed it forward, anticipating the deflection.

    python scripts/ab_spray.py
    python scripts/ab_spray.py --episodes 80 --plot

Runs are paired: every controller faces the same per-episode wind/noise
realizations, so the comparison is apples-to-apples.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (puts src/ on the path)
import numpy as np

from ember.spray_rl import kinematics as kin
from ember.spray_rl.env import SprayEnv
from ember.spray_rl.policy import DEFAULT_CHECKPOINT, PIDSprayController, SprayPolicy


def rollout(controller, *, wind: bool, episodes: int, seed: int,
            sigma: float, tau: float, trace: bool = False):
    """Paired rollout: returns (on_target_rate, mean_err, p90_err, trace).

    ``trace`` (steady-state landing error over one episode) is captured for the
    last episode when requested, for plotting."""
    env = SprayEnv(seed=seed, wind=wind)
    on, errs, last_trace = 0, [], []
    n_steady = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if wind:                       # pin the demo gust regime for the A/B
            env._wind.sigma = sigma
            env._wind.tau = tau
        ep_trace = []
        for t in range(env.max_steps):
            obs, _, term, trunc, info = env.step(controller.act(obs))
            ep_trace.append(info["landing_dist"])
            if t >= env.max_steps // 2:   # steady state (after convergence)
                on += int(info["on_target"])
                errs.append(info["landing_dist"])
                n_steady += 1
            if term or trunc:
                break
        if trace and ep == episodes - 1:
            last_trace = ep_trace
    return (on / max(n_steady, 1), float(np.mean(errs)),
            float(np.percentile(errs, 90)), last_trace)


def _fmt(rate, err, p90):
    return f"on-target {rate * 100:5.1f}%   mean {err:.3f} m   p90 {p90:.3f} m"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--sigma", type=float, default=kin.WIND_SIGMA_DEMO)
    ap.add_argument("--tau", type=float, default=kin.WIND_TAU)
    ap.add_argument("--plot", action="store_true",
                    help="save a landing-error-vs-time plot (needs matplotlib)")
    args = ap.parse_args()

    controllers: list[tuple[str, object]] = [("Reactive PID", PIDSprayController())]
    if DEFAULT_CHECKPOINT.is_file():
        try:
            controllers.append(("RL (wind-aware)", SprayPolicy.load(DEFAULT_CHECKPOINT)))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] RL checkpoint failed to load ({exc}); showing PID only.\n")
    else:
        print(f"[warn] no RL checkpoint at {DEFAULT_CHECKPOINT}; showing PID only.\n")

    kw = dict(episodes=args.episodes, seed=args.seed, sigma=args.sigma, tau=args.tau)
    print(f"Spray-correction A/B  |  {args.episodes} paired episodes, "
          f"gust sigma={args.sigma} m/s^2 tau={args.tau}s, "
          f"flight-lag={kin.FLIGHT_LAG_STEPS} steps\n")
    header = f"{'controller':<18}{'calm air':<48}{'gusty wind':<48}"
    print(header)
    print("-" * len(header))

    traces = {}
    for name, ctrl in controllers:
        calm = rollout(ctrl, wind=False, **kw)
        gust = rollout(ctrl, wind=True, trace=args.plot, **kw)
        traces[name] = gust[3]
        print(f"{name:<18}{_fmt(*calm[:3]):<48}{_fmt(*gust[:3]):<48}")

    if args.plot and any(traces.values()):
        _plot(traces, args)


def _plot(traces, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"\n[warn] plot skipped ({exc})")
        return
    out = "train_5070ti/ab_spray.png"
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, tr in traces.items():
        if tr:
            ax.plot(np.arange(len(tr)) * 0.02, tr, label=name, lw=1.6)
    ax.axhline(kin.HIT_RADIUS, ls="--", c="k", lw=1, label="hit radius")
    ax.set_xlabel("time (s)"); ax.set_ylabel("landing error (m)")
    ax.set_title(f"Gusty wind (sigma={args.sigma} m/s^2): PID trails, RL anticipates")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
