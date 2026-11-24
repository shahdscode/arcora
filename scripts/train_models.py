"""CLI script to run the Week 3 modeling pipeline (Issue #24).

Examples:
    python scripts/train_models.py
    python scripts/train_models.py --families baseline linear --targets home_win
    python scripts/train_models.py --evaluate-only --reuse-manifest
"""

from __future__ import annotations

import sys

from arcora.models.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
