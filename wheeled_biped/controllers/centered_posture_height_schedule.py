"""
Centered posture height schedule.

Provides smooth height-dependent hip_pitch_ref(height_m) and knee_ref(height_m)
functions fitted from the centered posture optimization results.

The functions are 4th-degree polynomials fitted over the calibrated [0.30, 0.48] m
height range, with clamping outside [0.28, 0.50] m.
"""

from pathlib import Path
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Coefficients from centered_posture_height_functions.json
# hip_pitch_ref(h) = c4*h^4 + c3*h^3 + c2*h^2 + c1*h + c0
# knee_ref(h)      = d4*h^4 + d3*h^3 + d2*h^2 + d1*h + d0
# ---------------------------------------------------------------------------
_HIP_PITCH_COEFFS = np.array([
    -445.3019944488364,  # h^4
    637.5059040349065,   # h^3
    -343.2588614305615,  # h^2
    81.9584579754549,    # h^1
    -6.128514902625229,  # h^0
])

_KNEE_COEFFS = np.array([
    -435.5456772116635,  # h^4
    627.1773006976054,   # h^3
    -341.0053479122672,  # h^2
    82.76750027306793,   # h^1
    -6.070302185600878,  # h^0
])

# Calibrated range
_MIN_HEIGHT_M = 0.28
_MAX_HEIGHT_M = 0.50

_CENTERED_POSTURE_FUNCTION_VERSION = "1.0"


def evaluate_centered_posture(height_m: float) -> Tuple[float, float, float, float]:
    """Compute centered hip_pitch_ref and knee_ref from height_m.

    Args:
        height_m: target CoM height in meters.

    Returns:
        (hip_pitch_ref_rad, knee_ref_rad, hip_roll_left, hip_roll_right).
        hip_roll values are always 0.0 (lateral bias is intrinsic, see
        docs/validation/current_height_posture_geometry_audit.md).
    """
    # Clamp input
    h = float(np.clip(height_m, _MIN_HEIGHT_M, _MAX_HEIGHT_M))

    # Evaluate polynomial using Horner's method for stability
    # p(h) = c4*h^4 + c3*h^3 + c2*h^2 + c1*h + c0
    hp = float(np.polyval(_HIP_PITCH_COEFFS, h))
    kn = float(np.polyval(_KNEE_COEFFS, h))

    return hp, kn, 0.0, 0.0


def centered_posture_function_version() -> str:
    """Return the version string for the loaded centered posture functions."""
    return _CENTERED_POSTURE_FUNCTION_VERSION


def centered_posture_supported_range_m() -> Tuple[float, float]:
    """Return the (min, max) height range the functions were calibrated for."""
    return (_MIN_HEIGHT_M, _MAX_HEIGHT_M)
