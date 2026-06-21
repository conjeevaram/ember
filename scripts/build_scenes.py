#!/usr/bin/env python3
"""Entry point: (re)generate demo or procedural scene XMLs.

    python scripts/build_scenes.py --force
    python scripts/build_scenes.py --random 8 --seed 0
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (puts src/ on the path)
import numpy as np

from ember import scenegen, scenes
from ember.config import G1_MODEL_DIR
from ember.spec import to_json


SPECS_DIR = Path(__file__).resolve().parent.parent / "scenes" / "specs"


def _build_random(n: int, seed: int, n_fires: int | None, n_walls: int, terrain: bool,
                  n_debris: int, n_tiers: int) -> None:
    if not G1_MODEL_DIR.exists():
        raise SystemExit(f"model dir not found: {G1_MODEL_DIR}\n"
                         "Set $UNITREE_RL_GYM to your unitree_rl_gym checkout.")
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    written_xml: list[str] = []
    written_json: list[str] = []
    fire_rng = np.random.default_rng(seed) if n_fires is None else None
    for i in range(n):
        s = seed + i
        scene_fires = n_fires if n_fires is not None else int(fire_rng.integers(1, 6))
        spec = scenegen.random_spec(s, n_fires=scene_fires, n_walls=n_walls, terrain=terrain,
                                    n_debris=n_debris, n_tiers=n_tiers)
        p12, p29 = scenes.build(spec)
        written_xml.extend([p12, p29])
        json_path = SPECS_DIR / f"{spec.name}.json"
        to_json(spec, json_path)
        written_json.append(str(json_path))
    print(f"generated {n} scene spec(s) (seeds {seed}..{seed + n - 1})")
    print("XML:\n  " + "\n  ".join(written_xml))
    print("JSON:\n  " + "\n  ".join(written_json))


def main() -> None:
    p = argparse.ArgumentParser(description="(Re)generate G1 demo or procedural scenes.")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing named demo scene files")
    p.add_argument("--random", type=int, metavar="N",
                   help="generate N procedural scenes from SceneSpec")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for --random (default: 0)")
    p.add_argument("--n-fires", type=int, default=None,
                   help="fixed fires per random scene (default: 1–5 per scene, seeded)")
    p.add_argument("--n-walls", type=int, default=4,
                   help="walls per random scene (default: 4)")
    p.add_argument("--terrain", action="store_true",
                   help="include gentle heightfield terrain")
    p.add_argument("--n-debris", type=int, default=4,
                   help="traversable debris per random scene (default: 4)")
    p.add_argument("--tiers", type=int, default=1, metavar="N",
                   help="low ramp+platform tiers among debris (default: 1)")
    args = p.parse_args()

    if args.random is not None:
        _build_random(args.random, args.seed, args.n_fires, args.n_walls, args.terrain,
                      args.n_debris, args.tiers)
        return

    if not G1_MODEL_DIR.exists():
        raise SystemExit(f"model dir not found: {G1_MODEL_DIR}\n"
                         "Set $UNITREE_RL_GYM to your unitree_rl_gym checkout.")
    written = scenes.ensure_scenes(force=args.force)
    if written:
        print("wrote:\n  " + "\n  ".join(written))
    else:
        print("all scenes already present (use --force to overwrite)")


if __name__ == "__main__":
    main()
