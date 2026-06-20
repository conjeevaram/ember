#!/usr/bin/env python3
"""Entry point: GMT whole-body G1, velocity-steerable (default port 8089).

    python scripts/run_gmt.py --scene obstacles --motion basic_walk.pkl
"""
import _bootstrap  # noqa: F401  (puts src/ on the path)

from ember.gmt import main

if __name__ == "__main__":
    main()
