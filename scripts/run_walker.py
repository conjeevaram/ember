#!/usr/bin/env python3
"""Entry point: 12-DOF G1 walker + kinematic arm overlay (default port 8088).

    python scripts/run_walker.py --scene obstacles
"""
import _bootstrap  # noqa: F401  (puts src/ on the path)

from ember.locomotion import main

if __name__ == "__main__":
    main()
