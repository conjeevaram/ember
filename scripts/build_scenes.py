#!/usr/bin/env python3
"""Entry point: (re)generate the demo scene XMLs into the G1 model dir.

    python scripts/build_scenes.py --force
"""
import _bootstrap  # noqa: F401  (puts src/ on the path)

from ember.scenes import main

if __name__ == "__main__":
    main()
