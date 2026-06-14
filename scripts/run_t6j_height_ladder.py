#!/usr/bin/env python3
"""Deprecated wrapper — redirects to the renamed script.

.. deprecated::
    Use :func:`scripts.analyze_support_centering_height_ladder` instead.
    This script exists only for backward compatibility during the transition.
"""
import sys
import warnings
from pathlib import Path

warnings.warn(
    "scripts/run_t6j_height_ladder.py is deprecated. "
    "Use: python scripts/run_support_centering_height_ladder.py",
    DeprecationWarning,
    stacklevel=2,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_support_centering_height_ladder as new_module

sys.exit(new_module.main())