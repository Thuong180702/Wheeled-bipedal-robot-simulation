"""Pytest configuration for Phase 3D tests.

Ensures Phase 3C modules are importable before test collection.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Phase 3C modules must be imported first to register in sys.modules
import wheeled_biped.wbc.offline_rolling_constraints  # noqa: F401, E402
import wheeled_biped.wbc.phase3c_rolling_qp  # noqa: F401, E402
