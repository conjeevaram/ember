"""Phase 6 spray-aim RL surrogate."""
from __future__ import annotations

import math

import numpy as np

from ember.nav import SPRAY_STANDOFF
from ember.spray_rl.env import SprayEnv
from ember.spray_rl import kinematics as kin
from ember.spray_rl.kinematics import (HIT_RADIUS, OBS_DIM, aim_point, build_obs,
                                       sample_disc_offset)
from ember.spray_rl.policy import PIDSprayController


def test_spray_env_spaces_and_step():
    env = SprayEnv(max_steps=20, seed=42)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert env.observation_space.contains(obs)
    action = env.action_space.sample()
    assert env.action_space.contains(action)
    obs2, reward, term, trunc, info = env.step(action)
    assert obs2.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert "landing_dist" in info
    assert "on_target" in info


def test_spray_env_reward_sanity():
    env = SprayEnv(max_steps=5, seed=0)
    env.reset()
    for _ in range(5):
        _, r, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        assert math.isfinite(r)
        assert r <= 5.0 + 1e-6           # bonus(5) - 2*dist - penalties
        if info["on_target"]:
            assert r > 0.0
        if term or trunc:
            break


def test_aim_point_hits_at_standoff():
    """The fixed jet's closest approach to the fire centre is within HIT_RADIUS
    when the robot faces the fire at the navigation standoff."""
    _, dist = aim_point(0.0, 0.0, 0.0, (SPRAY_STANDOFF, 0.0))
    assert dist < HIT_RADIUS, f"standoff not hittable: min3d={dist:.3f} m"


def test_disc_noise_radius():
    rng = np.random.default_rng(0)
    rs = [np.hypot(*sample_disc_offset(rng, 0.25)) for _ in range(500)]
    assert max(rs) <= 0.25 + 1e-9
    assert np.mean(rs) > 0.05


def test_build_obs_shape():
    obs = build_obs(fire_xy=(2.0, 0.0), robot_xy=(0.0, 0.0),
                    aim_xy=np.array([2.1, 0.05]), yaw=0.0,
                    vx=0.0, vy=0.0, yaw_rate=0.0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_build_obs_is_body_frame():
    """Spatial terms must rotate into the body frame so they match the actions.

    Robot facing +y (yaw=90 deg): a fire that is due east in world (+x) is to the
    robot's RIGHT, i.e. negative body-lateral; due north (+y) is straight ahead.
    A world-frame obs would mislabel these and the policy could not correct aim.
    """
    obs = build_obs(fire_xy=(1.0, 0.0), robot_xy=(0.0, 0.0),
                    aim_xy=np.array([1.0, 0.0]), yaw=math.pi / 2,
                    vx=0.0, vy=0.0, yaw_rate=0.0)
    fire_fwd, fire_lat = float(obs[0]), float(obs[1])
    assert abs(fire_fwd) < 1e-5            # east fire is not ahead
    assert fire_lat < -0.9                 # it is to the right (negative lateral)


def _pid_steady_on_target(*, wind: bool, episodes: int = 8, seed: int = 7) -> float:
    """Steady-state on-target fraction for the reactive PID over several episodes.

    When ``wind`` is on we pin the demo gust regime (strong, fast) rather than
    averaging over the calm-included training randomization, so the test asserts
    the actual A/B condition."""
    pid = PIDSprayController()
    env = SprayEnv(seed=seed, wind=wind)
    steady, on = [], 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if wind:
            env._wind.sigma = kin.WIND_SIGMA_DEMO
            env._wind.tau = kin.WIND_TAU
        for t in range(env.max_steps):
            obs, _, term, trunc, info = env.step(pid.act(obs))
            if t >= env.max_steps // 2:
                steady.append(info["landing_dist"])
                on += int(info["on_target"])
            if term or trunc:
                break
    return on / max(len(steady), 1)


def test_pid_controller_converges_on_env():
    """Body-frame PID must actually reduce landing error in calm air (guards the
    frame bug: a world-frame controller diverges to >1 m, on-target ~0.05)."""
    pid = PIDSprayController()
    env = SprayEnv(seed=7, wind=False)
    steady, on = [], 0
    for ep in range(6):
        obs, _ = env.reset(seed=7 + ep)
        for t in range(env.max_steps):
            obs, _, term, trunc, info = env.step(pid.act(obs))
            if t >= env.max_steps // 2:
                steady.append(info["landing_dist"])
                on += int(info["on_target"])
            if term or trunc:
                break
    assert np.mean(steady) < 0.30, f"steady error too high: {np.mean(steady):.3f} m"
    assert on / len(steady) > 0.5


def test_wind_degrades_reactive_pid():
    """Premise of the A/B: gusty wind the PID cannot anticipate collapses its
    on-target rate well below its calm-air performance (the RL policy, which sees
    a noisy wind reading, recovers this in deployment)."""
    calm = _pid_steady_on_target(wind=False)
    gusty = _pid_steady_on_target(wind=True)
    assert calm > 0.5
    assert gusty < calm - 0.15, f"wind should hurt reactive PID: calm={calm:.2f} gusty={gusty:.2f}"
