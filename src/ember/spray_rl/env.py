"""Pure-NumPy kinematic surrogate for spray-aim RL (no MuJoCo).

The nozzle is fixed; the policy moves the ego (strafe / rotate / walk fwd-back)
so the fixed jet's closest approach to the fire centre stays within HIT_RADIUS.
The reward tracks the true (deterministic) miss while the observation gets the
0.25 m disc-noisy aim feedback, so the policy must stay robust to the jitter.
"""
from __future__ import annotations

import math
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ember.config import VX_RANGE, VY_RANGE, YAW_RANGE, clamp
from ember.nav import SPRAY_STANDOFF

from . import kinematics as kin


class SprayEnv(gym.Env):
    """Spray correction at standoff via ego motion (vx, vy, yaw)."""

    metadata = {"render_modes": []}

    def __init__(self, *, max_steps: int = 250, seed: int | None = None,
                 wind: bool = True):
        super().__init__()
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._wind_enabled = wind
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(kin.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([VX_RANGE[0], VY_RANGE[0], YAW_RANGE[0]], dtype=np.float32),
            high=np.array([VX_RANGE[1], VY_RANGE[1], YAW_RANGE[1]], dtype=np.float32),
            dtype=np.float32,
        )
        self._x = self._y = self._yaw = 0.0
        self._vx = self._vy = self._yaw_rate = 0.0
        self._fire = (0.0, 0.0)
        self._step_count = 0
        self._noise_radius = kin.SPRAY_LANDING_NOISE_RADIUS
        self._lag_tau = 0.25
        self._fire_sigma = 0.08
        self._dt = 0.02
        self._wind = kin.GustyWind(dt=self._dt, rng=self._rng)
        self._lag_steps = kin.FLIGHT_LAG_STEPS
        self._aim_hist: deque[np.ndarray] = deque()
        self._aim_obs = np.zeros(2, dtype=float)   # delayed (observed) landing
        self._aim_now = np.zeros(2, dtype=float)    # current (true) landing

    def _domain_randomize(self) -> None:
        self._noise_radius = float(self._rng.uniform(0.15, 0.35))
        self._lag_tau = float(self._rng.uniform(0.15, 0.45))
        self._fire_sigma = float(self._rng.uniform(0.03, 0.15))
        self._lag_steps = int(self._rng.integers(kin.FLIGHT_LAG_STEPS - 4,
                                                 kin.FLIGHT_LAG_STEPS + 5))
        # Span calm -> strong so the same policy works wind-off and in gusts.
        sigma = float(self._rng.uniform(0.0, kin.WIND_SIGMA_DEMO * 1.2)) if self._wind_enabled else 0.0
        self._wind.sigma = sigma
        self._wind.tau = float(self._rng.uniform(0.6, 1.6))
        self._wind.reset(self._rng.normal(0.0, sigma, 2) if sigma > 0 else None)
        standoff = float(self._rng.uniform(SPRAY_STANDOFF * 0.9, SPRAY_STANDOFF * 1.1))
        self._fire, self._x, self._y, self._yaw = kin.random_fire_near_standoff(
            self._rng, standoff)
        self._vx = self._vy = self._yaw_rate = 0.0
        dist = self._refresh_aim()
        # Prime the flight-lag buffer so the first observations are valid.
        self._aim_hist = deque([self._aim_now.copy()] * self._lag_steps,
                               maxlen=self._lag_steps)
        self._aim_obs = self._aim_now.copy()
        return dist

    def _refresh_aim(self) -> float:
        """Update the current (true) noisy landing and return the true 3D miss."""
        aim_xy, dist = kin.aim_point(self._x, self._y, self._yaw, self._fire,
                                     wind=tuple(self._wind.value))
        nx, ny = kin.sample_disc_offset(self._rng, self._noise_radius)
        self._aim_now = np.array([aim_xy[0] + nx, aim_xy[1] + ny])
        return dist

    def _advance_lag(self) -> None:
        """Push the current landing and expose the one observed FLIGHT_LAG ago."""
        self._aim_obs = self._aim_hist[0].copy()
        self._aim_hist.append(self._aim_now.copy())

    def _obs(self) -> np.ndarray:
        return kin.build_obs(
            fire_xy=self._fire, robot_xy=(self._x, self._y), aim_xy=self._aim_obs,
            yaw=self._yaw, vx=self._vx, vy=self._vy, yaw_rate=self._yaw_rate,
            wind_xy=tuple(self._wind.observe()),
            wind_then_xy=tuple(self._wind.observe_delayed(self._lag_steps)),
            fire_pos_sigma=self._fire_sigma, rng=self._rng,
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._domain_randomize()
        self._step_count = 0
        return self._obs(), {}

    def step(self, action):
        vx_cmd = clamp(float(action[0]), *VX_RANGE)
        vy_cmd = clamp(float(action[1]), *VY_RANGE)
        yaw_cmd = clamp(float(action[2]), *YAW_RANGE)

        alpha = self._dt / (self._lag_tau + self._dt)
        self._vx += alpha * (vx_cmd - self._vx)
        self._vy += alpha * (vy_cmd - self._vy)
        self._yaw_rate += alpha * (yaw_cmd - self._yaw_rate)

        c, s = math.cos(self._yaw), math.sin(self._yaw)
        self._x += (c * self._vx - s * self._vy) * self._dt
        self._y += (s * self._vx + c * self._vy) * self._dt
        self._yaw += self._yaw_rate * self._dt

        self._wind.step()
        dist = self._refresh_aim()  # true 3D miss of the parcel launched now
        self._advance_lag()         # expose the delayed (observed) landing
        on_target = dist <= kin.HIT_RADIUS
        reward = -2.0 * dist + (5.0 if on_target else 0.0) - 0.01
        # Light roaming penalty only -- gust tracking needs active motion.
        reward -= 0.01 * (self._vx ** 2 + self._vy ** 2)

        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        info = {"landing_dist": dist, "on_target": on_target}
        return self._obs(), float(reward), terminated, False, info
