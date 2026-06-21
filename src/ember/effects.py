"""Procedural fire/water effects for the fire scene.

The static emissive flame + fire lights live in the FIRE overlay scene
(:mod:`ember.scenes`); here :func:`trajectory` / :func:`hit_index` give the real
ballistic water jet (and whether it reaches the fire), and :class:`SceneFX`
animates the flame (``animate``) and injects the transient water/steam/smoke/
ember geoms into a renderer's ``MjvScene`` (``decorate``). Nothing touches
physics -- it's pure decoration, and on non-fire scenes ``SceneFX`` is a no-op.

Flame bodies are named ``flame0..N`` with paired lights ``firelight{i}a`` /
``firelight{i}b`` per fire (see :func:`ember.scenes.flame_bodies`).
"""
from __future__ import annotations

import mujoco
import numpy as np

_GT = mujoco.mjtGeom
GRAVITY = 9.81

_FLAME_PREFIX = "flame"
_FIRE_RGB = np.array([1.0, 0.5, 0.18])


def _flame_light_names(index: int) -> tuple[str, str]:
    """Lower / upper fire-light names for ``flame{index}``."""
    return f"firelight{index}a", f"firelight{index}b"


def trajectory(muzzle, direction, speed, *, ground_z=0.0, dt=0.02, max_steps=200):
    """Ballistic path of a water parcel launched from ``muzzle`` along the unit
    ``direction`` at ``speed`` (m/s), under gravity, until it drops to
    ``ground_z``. Returns ``(points Nx3, landing 3)`` -- the real arc, so it
    overshoots / falls short / veers off when the nozzle is mis-aimed."""
    muzzle = np.asarray(muzzle, float)
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-9)
    vel = d * speed
    p = muzzle.copy()
    pts = [p.copy()]
    for _ in range(max_steps):
        vel = vel + np.array([0.0, 0.0, -GRAVITY]) * dt
        p = p + vel * dt
        pts.append(p.copy())
        if p[2] <= ground_z:
            pts[-1][2] = ground_z
            break
    return np.array(pts), pts[-1]


def hit_index(points, target, radius):
    """Index of the path point closest to ``target`` and whether the nearest
    approach is within ``radius``. ``(idx, hit, min_dist)``."""
    dists = np.linalg.norm(points - np.asarray(target, float), axis=1)
    idx = int(np.argmin(dists))
    return idx, bool(dists[idx] <= radius), float(dists[idx])


class SceneFX:
    """Animator + scene decorator for the fire scene, bound to a render model."""

    def __init__(self, model):
        BODY, LIGHT = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_LIGHT
        self._flames: list[dict] = []
        idx = 0
        while True:
            name = f"{_FLAME_PREFIX}{idx}"
            bid = mujoco.mj_name2id(model, BODY, name)
            if bid < 0:
                break
            g0 = model.body_geomadr[bid]
            lights = [i for i in
                      (mujoco.mj_name2id(model, LIGHT, n)
                       for n in _flame_light_names(idx))
                      if i >= 0]
            self._flames.append({
                "gslice": slice(g0, g0 + model.body_geomnum[bid]),
                "base_pos": model.geom_pos[g0:g0 + model.body_geomnum[bid]].copy(),
                "base_size": model.geom_size[g0:g0 + model.body_geomnum[bid]].copy(),
                "lights": lights,
            })
            idx += 1
        self.active = bool(self._flames)
        if not self.active:
            return
        self._rng = np.random.default_rng()

    def animate(self, model, health_list):
        """Flicker each fire's lights, jitter its flame, and scale by its
        ``health_list`` entry (1 = full, 0 = out). Mutates ``model`` arrays, so
        call once per frame just before ``mj_forward``."""
        if not self.active:
            return
        r = self._rng
        for i, flame in enumerate(self._flames):
            h = max(health_list[i] if i < len(health_list) else 0.0, 0.0)
            gslice = flame["gslice"]
            scale = (0.15 + 0.85 * h) if h > 0.02 else 0.0  # fully hide when out
            model.geom_size[gslice] = flame["base_size"] * scale
            model.geom_pos[gslice] = (
                flame["base_pos"] + r.normal(0, 0.02 * h, flame["base_pos"].shape))
            for lid in flame["lights"]:
                model.light_diffuse[lid] = _FIRE_RGB * h * (0.7 + 0.5 * r.random())

    def decorate(self, scene, *, fires=None, water=None, landing=None,
                 hit_idx=None, hitting=False, targeted_idx=0):
        """Inject transient FX geoms into ``scene``. ``fires`` is
        ``[(pos, health), ...]``; smoke/embers draw for every burning fire.
        ``water`` is the ballistic path (Nx3) aimed at ``targeted_idx``;
        ``hitting`` adds steam at that fire and truncates the stream at
        ``hit_idx``; otherwise the stream runs to ``landing`` and splashes there."""
        if not self.active:
            return
        fires = fires or []
        r = self._rng
        if water is not None and len(water) > 1:
            end = hit_idx + 1 if (hitting and hit_idx is not None) else len(water)
            self._stream(scene, water[:end])
            if hitting and fires and 0 <= targeted_idx < len(fires):
                self._steam(scene, np.asarray(fires[targeted_idx][0], float), r)
            elif landing is not None:
                self._splash(scene, np.asarray(landing, float), r)
        for pos, health in fires:
            if health > 0.02:
                fire = np.asarray(pos, float)
                self._smoke(scene, fire, r, health)
                self._embers(scene, fire, r, health)

    # -- transient geom builders -------------------------------------------- #
    def _stream(self, scene, pts):
        n = len(pts)
        for i in range(n - 1):
            u = i / n
            _capsule(scene, pts[i], pts[i + 1], 0.030 * (1 - 0.5 * u),
                     (0.70, 0.85, 1.0, 0.55 if u < 0.8 else 0.40))

    def _splash(self, scene, landing, r):
        for _ in range(10):
            j = r.normal(0, 0.12, 3)
            j[2] = abs(j[2]) * 1.5
            _sphere(scene, landing + j, 0.02 + 0.02 * r.random(),
                    (0.80, 0.90, 1.0, 0.45))

    def _steam(self, scene, fire, r):
        for _ in range(8):
            p = fire + [r.normal(0, 0.2), r.normal(0, 0.2), 0.5 + r.random() * 0.9]
            _sphere(scene, p, 0.13 + 0.10 * r.random(), (0.88, 0.88, 0.90, 0.20))

    def _smoke(self, scene, fire, r, h):
        for k in range(10):
            spread = 0.05 + 0.04 * k
            p = fire + [r.normal(0, spread), r.normal(0, spread), 1.2 + k * 0.32]
            _sphere(scene, p, (0.18 + 0.05 * k) * (0.4 + 0.6 * h),
                    (0.22, 0.22, 0.24, max(0.05, 0.30 - 0.025 * k) * h))

    def _embers(self, scene, fire, r, h):
        for _ in range(int(14 * h)):
            p = fire + [r.normal(0, 0.25), r.normal(0, 0.25), 0.6 + r.random() * 1.6]
            _sphere(scene, p, 0.012, (1.0, 0.6, 0.2, 0.95))


def _sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], _GT.mjGEOM_SPHERE,
                        np.array([radius, 0, 0]), np.asarray(pos, np.float64),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def _capsule(scene, a, b, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, _GT.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.zeros(9), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, _GT.mjGEOM_CAPSULE, radius,
                         np.asarray(a, np.float64), np.asarray(b, np.float64))
    scene.ngeom += 1
