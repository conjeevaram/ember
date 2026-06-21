"""SceneSpec — single source of truth for procedural scenes, costmaps, and XML."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TerrainSpec:
    """Gentle MuJoCo heightfield; amplitude must stay within the blind-policy envelope."""
    nrow: int
    ncol: int
    radius_x: float   # half-extent of the hfield geom in x (m)
    radius_y: float   # half-extent of the hfield geom in y (m)
    elevation: float  # max elevation above base (m); keep small (<= 0.08)
    seed: int


@dataclass(frozen=True)
class DebrisSpec:
    """Traversable colliding obstacle within the blind-policy envelope.

    ``kind`` selects geometry (see :func:`ember.scenes.debris_geoms`):

    - ``"log"``: half-buried cylinder — ``size_a``=radius, ``size_b``=half-length
      along axis, ``height``=exposed crest above ground (<= ~0.07 m typical).
    - ``"bump"``: low dome (ellipsoid) — ``size_a``/``size_b``=horizontal radii,
      ``height``=peak above ground.
    - ``"tier"``: gentle up-ramp + low platform — ``size_a``=width, ``size_b``=
      platform depth, ``height``=rise (<= ~0.12 m); ``yaw``=ramp approach direction.
    """
    kind: str
    cx: float
    cy: float
    yaw: float
    size_a: float
    size_b: float
    height: float


@dataclass(frozen=True)
class SceneSpec:
    name: str
    bounds: tuple[float, float, float, float]      # xmin, xmax, ymin, ymax
    start: tuple[float, float, float]              # x, y, yaw
    walls: tuple[tuple[float, float, float, float, float], ...]  # cx,cy,w,h,theta
    fires: tuple[tuple[float, float], ...]         # ground (x, y)
    terrain: TerrainSpec | None
    home: tuple[float, float]
    seed: int
    debris: tuple[DebrisSpec, ...] = ()


def _terrain_from_dict(d: dict[str, Any] | None) -> TerrainSpec | None:
    if d is None:
        return None
    return TerrainSpec(**d)


def _debris_from_dict(d: dict[str, Any]) -> DebrisSpec:
    return DebrisSpec(
        kind=str(d["kind"]),
        cx=float(d["cx"]),
        cy=float(d["cy"]),
        yaw=float(d["yaw"]),
        size_a=float(d["size_a"]),
        size_b=float(d["size_b"]),
        height=float(d["height"]),
    )


def to_dict(spec: SceneSpec) -> dict[str, Any]:
    d = asdict(spec)
    if spec.terrain is None:
        d["terrain"] = None
    return d


def from_dict(d: dict[str, Any]) -> SceneSpec:
    raw_debris = d.get("debris", ())
    return SceneSpec(
        name=d["name"],
        bounds=tuple(d["bounds"]),
        start=tuple(d["start"]),
        walls=tuple(tuple(w) for w in d["walls"]),
        fires=tuple(tuple(f) for f in d["fires"]),
        terrain=_terrain_from_dict(d.get("terrain")),
        home=tuple(d["home"]),
        seed=int(d["seed"]),
        debris=tuple(_debris_from_dict(x) for x in raw_debris),
    )


def to_json(spec: SceneSpec, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_dict(spec), indent=2) + "\n")


def from_json(path: str | Path) -> SceneSpec:
    return from_dict(json.loads(Path(path).read_text()))
