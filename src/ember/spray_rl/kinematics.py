"""Shared spray geometry + observation layout for the env and deployment.

The nozzle/arms are FIXED: the policy only drives the ego (vx, vy, yaw) so the
fixed jet lands on the fire. The muzzle pose + direction below are calibrated
(body frame, relative to the pelvis) from the live MuJoCo overlay while standing
at the rest nozzle pitch, so the surrogate's arc matches the real jet. The
deployed hit test is the closest approach of the arc to the fire centre at
``FIRE_AIM_HEIGHT``; the surrogate reuses the exact same ``effects`` math.
"""
from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from ember import effects
from ember.config import VX_RANGE, VY_RANGE, YAW_RANGE, clamp, quat_yaw

if TYPE_CHECKING:
    from ember.sim import G1Sim

# Calibrated against ember.sim's overlay (see scripts/calibrate notes): muzzle
# position relative to the pelvis and jet direction, both in the body frame.
MUZZLE_BODY_OFFSET = (0.6474, 0.1014, 0.0697)
DIR_BODY = (0.9994, 0.0002, -0.0331)
PELVIS_Z = 0.778

JET_SPEED = 7.0
FIRE_AIM_HEIGHT = 0.45
HIT_RADIUS = 0.25
SPRAY_LANDING_NOISE_RADIUS = 0.25

OBS_DIM = 12
# All spatial terms are in the robot BODY frame (x forward, y left) so they line
# up with the body-frame velocity actions. The two wind readings let a policy act
# as a Smith predictor: ``wind_now`` is sensed instantly, ``wind_then`` is the
# wind when the currently-observed (flight-lagged) splash was launched, so the
# pair reconstructs the true current aim error. A reactive PID uses neither.
OBS_SLICES = {
    "fire_body": slice(0, 2),         # fire bearing (forward, lateral)
    "aim_err_body": slice(2, 4),      # delayed, noisy aim point - fire (fwd, lat)
    "cmd": slice(4, 7),               # vx, vy, yaw_rate
    "wind_now": slice(7, 9),          # noisy wind accel now (forward, lateral)
    "wind_then": slice(9, 11),        # noisy wind accel a flight-lag ago
    "in_range": slice(11, 12),
}

# Gust strength (wind accel std, m/s^2) used for the wind-on demo / A/B. ~5
# deflects the jet by a sizable fraction of HIT_RADIUS at the standoff.
WIND_SIGMA_DEMO = 5.0
WIND_TAU = 0.9          # gust correlation time (s) -- faster than the loop can chase
WIND_OBS_SIGMA = 0.6    # noise on the wind reading (partial observability)

# Jet time-of-flight: the wind is sensed instantly at the nozzle, but where the
# water actually lands is only observed this many control steps later. That
# dead-time in the feedback path is what forces a reactive controller to detune
# and lag the gusts, while a feed-forward policy can act on the instant wind
# reading. dt=0.02 s, so 15 steps ~= 0.3 s.
FLIGHT_LAG_STEPS = 15


class GustyWind:
    """Ornstein-Uhlenbeck horizontal wind acceleration (m/s^2) on the jet.

    Mean-reverting to zero with correlation time ``tau`` and stationary std
    ``sigma``; gusts are slow enough that a feed-forward policy can lead them but
    a purely reactive controller lags."""

    _HIST = 40   # enough for any FLIGHT_LAG used

    def __init__(self, *, sigma: float = 0.0, tau: float = WIND_TAU, dt: float = 0.02,
                 rng: np.random.Generator | None = None):
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.dt = float(dt)
        self.rng = rng or np.random.default_rng()
        self.w = np.zeros(2, dtype=float)
        self._hist: deque[np.ndarray] = deque([self.w.copy()], maxlen=self._HIST)

    def step(self) -> np.ndarray:
        a = self.dt / self.tau
        kick = self.rng.normal(0.0, 1.0, 2) * self.sigma * math.sqrt(2.0 * self.dt / self.tau)
        self.w += -a * self.w + kick
        self._hist.append(self.w.copy())
        return self.w.copy()

    @property
    def value(self) -> np.ndarray:
        return self.w.copy()

    def value_delayed(self, lag: int) -> np.ndarray:
        """Wind ``lag`` steps ago (clamped to available history)."""
        if lag <= 0:
            return self.w.copy()
        idx = min(lag + 1, len(self._hist))
        return self._hist[-idx].copy()

    def reset(self, w0: np.ndarray | None = None) -> None:
        self.w = np.zeros(2) if w0 is None else np.asarray(w0, float).copy()
        self._hist = deque([self.w.copy()], maxlen=self._HIST)

    def observe(self, obs_sigma: float = WIND_OBS_SIGMA) -> np.ndarray:
        return self.w + self.rng.normal(0.0, obs_sigma, 2)

    def observe_delayed(self, lag: int, obs_sigma: float = WIND_OBS_SIGMA) -> np.ndarray:
        return self.value_delayed(lag) + self.rng.normal(0.0, obs_sigma, 2)


def sample_disc_offset(rng: np.random.Generator, radius: float) -> tuple[float, float]:
    """Uniform sample in a disc of ``radius`` (area-uniform)."""
    r = radius * math.sqrt(float(rng.random()))
    t = float(rng.uniform(0.0, 2.0 * math.pi))
    return r * math.cos(t), r * math.sin(t)


def muzzle_pose(x: float, y: float, yaw: float,
                *, pelvis_z: float = PELVIS_Z) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-nozzle muzzle position + unit direction for a base at (x, y, yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    ox, oy, oz = MUZZLE_BODY_OFFSET
    muzzle = np.array([x + c * ox - s * oy, y + s * ox + c * oy, pelvis_z + oz])
    dx, dy, dz = DIR_BODY
    direction = np.array([c * dx - s * dy, s * dx + c * dy, dz])
    direction /= np.linalg.norm(direction) + 1e-9
    return muzzle, direction


def aim_point(x: float, y: float, yaw: float, fire_xy: tuple[float, float],
              *, jet_speed: float = JET_SPEED, pelvis_z: float = PELVIS_Z,
              wind: tuple[float, float] = (0.0, 0.0)) -> tuple[np.ndarray, float]:
    """Closest approach of the fixed jet to the fire centre.

    Returns ``(xy, dist3d)``: the ground-plane xy of the arc point nearest the
    fire centre at ``FIRE_AIM_HEIGHT`` and that minimum 3D distance (the same
    quantity ``G1Sim._update_jet`` thresholds against ``HIT_RADIUS``). ``wind``
    deflects the arc, so the hittable ego pose shifts with the gust."""
    muzzle, direction = muzzle_pose(x, y, yaw, pelvis_z=pelvis_z)
    pts, _ = effects.trajectory(muzzle, direction, jet_speed, wind=wind)
    center = np.array([fire_xy[0], fire_xy[1], FIRE_AIM_HEIGHT])
    dists = np.linalg.norm(pts - center, axis=1)
    idx = int(np.argmin(dists))
    return pts[idx][:2], float(dists[idx])


def build_obs(*, fire_xy: tuple[float, float], robot_xy: tuple[float, float],
              aim_xy: np.ndarray, yaw: float, vx: float, vy: float, yaw_rate: float,
              wind_xy: tuple[float, float] = (0.0, 0.0),
              wind_then_xy: tuple[float, float] = (0.0, 0.0),
              fire_pos_sigma: float = 0.0,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Observation vector shared by :class:`SprayEnv` and deployment.

    Fire bearing, aim error and both wind readings are rotated into the body
    frame (by ``-yaw``) so they align with the body-frame velocity actions."""
    rng = rng or np.random.default_rng()
    dx = fire_xy[0] - robot_xy[0]
    dy = fire_xy[1] - robot_xy[1]
    if fire_pos_sigma > 0:
        dx += rng.normal(0.0, fire_pos_sigma)
        dy += rng.normal(0.0, fire_pos_sigma)
    err_x = float(aim_xy[0]) - fire_xy[0]
    err_y = float(aim_xy[1]) - fire_xy[1]
    in_range = 1.0 if math.hypot(err_x, err_y) <= HIT_RADIUS else 0.0
    c, s = math.cos(yaw), math.sin(yaw)

    def _body(ax, ay):
        return c * ax + s * ay, -s * ax + c * ay

    fire_fwd, fire_lat = _body(dx, dy)
    err_fwd, err_lat = _body(err_x, err_y)
    wn_fwd, wn_lat = _body(wind_xy[0], wind_xy[1])
    wt_fwd, wt_lat = _body(wind_then_xy[0], wind_then_xy[1])
    return np.array([fire_fwd, fire_lat, err_fwd, err_lat, vx, vy, yaw_rate,
                     wn_fwd, wn_lat, wt_fwd, wt_lat, in_range], dtype=np.float32)


def obs_from_sim(sim: G1Sim, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Build the policy observation from a live :class:`G1Sim`."""
    rng = rng or np.random.default_rng()
    tf = sim.targeted_fire
    fx, fy = sim.fire_positions[tf][0], sim.fire_positions[tf][1]
    x, y = float(sim.data.qpos[0]), float(sim.data.qpos[1])
    yaw = quat_yaw(sim.data.qpos[3:7])
    wind = sim.wind_value()
    wind_obs = sim.wind_reading(rng)
    wind_then_obs = sim.wind_reading_delayed(FLIGHT_LAG_STEPS, rng)
    aim_xy, _ = aim_point(x, y, yaw, (fx, fy), wind=tuple(wind))
    noise_r = getattr(sim, "_spray_noise_radius", SPRAY_LANDING_NOISE_RADIUS)
    nx, ny = sample_disc_offset(rng, noise_r)
    aim_now = np.array([aim_xy[0] + nx, aim_xy[1] + ny])
    # Wind is sensed now, but the landing is only observed after the jet's flight
    # (matches the surrogate's dead-time), so the policy corrects on a lagged aim.
    aim_obs = sim.push_aim(aim_now)
    with sim._cmd_lock:
        vx, vy, yaw_rate = float(sim.cmd[0]), float(sim.cmd[1]), float(sim.cmd[2])
    return build_obs(fire_xy=(fx, fy), robot_xy=(x, y), aim_xy=aim_obs, yaw=yaw,
                     vx=vx, vy=vy, yaw_rate=yaw_rate, wind_xy=tuple(wind_obs),
                     wind_then_xy=tuple(wind_then_obs), fire_pos_sigma=0.08, rng=rng)


def clamp_action(action: np.ndarray) -> tuple[float, float, float]:
    """Clamp policy output to deployed command ranges (vx, vy, yaw)."""
    return (clamp(float(action[0]), *VX_RANGE),
            clamp(float(action[1]), *VY_RANGE),
            clamp(float(action[2]), *YAW_RANGE))


def random_fire_near_standoff(rng: np.random.Generator, standoff: float
                              ) -> tuple[tuple[float, float], float, float, float]:
    """Domain-randomized fire + robot pose around ``standoff`` (robot faces fire)."""
    angle = float(rng.uniform(0, 2 * math.pi))
    fx = float(rng.uniform(-1.0, 1.0))
    fy = float(rng.uniform(-1.0, 1.0))
    s = float(rng.uniform(standoff * 0.7, standoff * 1.4))
    x = fx - s * math.cos(angle)
    y = fy - s * math.sin(angle)
    face = math.atan2(fy - y, fx - x)
    yaw = face + float(rng.uniform(-0.3, 0.3))
    return (fx, fy), x, y, yaw
