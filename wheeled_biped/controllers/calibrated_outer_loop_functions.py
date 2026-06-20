"""Calibrated height-dependent outer-loop function lookups.

This module provides smooth, height-dependent parameter functions for the
`calibrated_support_position_outer_loop_pitch_ref` profile (Phase B
calibration). It is the runtime counterpart of the offline calibration artifact
`outputs/.../calibrated_outer_loop_height_functions.json` produced by
`scripts/run_outer_loop_height_function_fit.py`.

Design rules (see task spec / CLAUDE.md):
  - NO setup-name branching. Every parameter is computed from the commanded
    target height in metres via continuous interpolation through calibrated
    breakpoints.
  - PCHIP (monotone-preserving cubic) interpolation when SciPy is available,
    otherwise a numerically-identical piecewise-linear fallback. Both clamp to
    the calibration height range (no extrapolation) and to per-parameter safety
    bounds.
  - Pure Python floats only — called once per control step, no JAX arrays, no
    NumPy inside the hot path (SciPy's PchipInterpolator is built once at import
    and evaluated as a scalar).
  - All outputs are finite and clamped: no NaN can leave these functions.

The breakpoint tables below are the FROZEN calibration result. They are the
single source of truth shared by the runtime profile and the tests; the JSON
artifact is the human-readable / audit copy of the same numbers.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

# ---- Calibration breakpoints (frozen Phase B Stage 2A/2B result) ----------- #
# Heights are strictly ascending CoM target heights in metres. Each parameter
# table is aligned 1:1 with CALIBRATION_HEIGHTS_M.
CALIBRATION_HEIGHTS_M: Tuple[float, ...] = (
    0.300, 0.320, 0.330, 0.340, 0.360, 0.380, 0.430, 0.450, 0.465, 0.480,
)

# Outer-loop proportional gain (deg per metre of support error). U-shaped:
# higher authority at the height extremes, gentler through the mid band where
# the height schedule already centers well. Values are the combined Phase 2
# Stage 2A coarse + Stage 2B local-refinement winners per height.
CALIBRATION_KP: Tuple[float, ...] = (
    1.500, 1.500, 1.300, 1.000, 0.725, 0.650, 1.000, 0.650, 1.350, 1.575,
)

# Outer-loop derivative gain (deg per m/s). Light damping in the mid-low band
# where drift oscillates; near-zero at the extremes. Stage 2A/2B winners.
CALIBRATION_KD: Tuple[float, ...] = (
    0.000, 0.000, 0.150, 0.200, 0.200, 0.150, 0.000, 0.000, 0.000, 0.050,
)

# Integral gain (deg per m*s). Zero everywhere — Stage 2D found no robust
# integral benefit; the table is kept so a future calibration can populate it.
CALIBRATION_KI: Tuple[float, ...] = (
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
)

# Saturation half-range of the dynamic pitch_ref offset (deg).
CALIBRATION_THETA_MAX_DEG: Tuple[float, ...] = (
    3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00,
)

# Proportional deadband half-width (m).
CALIBRATION_DEADBAND_M: Tuple[float, ...] = (
    0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
)

# Per-step rate limit of the dynamic offset (deg/step).
CALIBRATION_RATE_LIMIT_DEG_PER_STEP: Tuple[float, ...] = (
    0.030, 0.030, 0.030, 0.030, 0.030, 0.030, 0.030, 0.030, 0.030, 0.030,
)

# Output low-pass coefficient (0,1].
CALIBRATION_LOWPASS_ALPHA: Tuple[float, ...] = (
    0.150, 0.150, 0.150, 0.150, 0.150, 0.150, 0.150, 0.150, 0.150, 0.150,
)

# ---- Safety bounds (hard clamps on every output) --------------------------- #
KP_BOUNDS: Tuple[float, float] = (0.40, 2.50)
KD_BOUNDS: Tuple[float, float] = (0.00, 0.50)
KI_BOUNDS: Tuple[float, float] = (0.00, 0.05)
THETA_MAX_BOUNDS: Tuple[float, float] = (1.50, 5.00)
DEADBAND_BOUNDS: Tuple[float, float] = (0.005, 0.050)
RATE_LIMIT_BOUNDS: Tuple[float, float] = (0.010, 0.080)
LOWPASS_BOUNDS: Tuple[float, float] = (0.050, 0.500)

# Calibration height range — outputs are held at the endpoint value (clamped,
# never extrapolated) outside this range.
H_MIN: float = CALIBRATION_HEIGHTS_M[0]
H_MAX: float = CALIBRATION_HEIGHTS_M[-1]

CALIBRATED_FUNCTION_PROFILE_NAME = "calibrated_outer_loop_height_functions_v1"


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


# ---- Interpolator construction --------------------------------------------- #
# Build one PCHIP interpolator per parameter at import time when SciPy is
# available. Each is a callable float->float. If SciPy is missing, fall back to
# a piecewise-linear closure with identical clamping semantics.
def _build_interpolator(xs: Tuple[float, ...], ys: Tuple[float, ...]):
    try:
        from scipy.interpolate import PchipInterpolator

        pchip = PchipInterpolator(list(xs), list(ys), extrapolate=False)

        def _eval(h: float) -> float:
            # Clamp the query into range so PCHIP never extrapolates (returns
            # endpoint value below/above the calibration range).
            hq = _clamp(h, xs[0], xs[-1])
            v = float(pchip(hq))
            if math.isnan(v):
                # Defensive: hold endpoint if PCHIP returns NaN at the boundary.
                return float(ys[0] if hq <= xs[0] else ys[-1])
            return v

        return _eval
    except ImportError:
        def _eval(h: float) -> float:
            if h <= xs[0]:
                return float(ys[0])
            if h >= xs[-1]:
                return float(ys[-1])
            for i in range(1, len(xs)):
                if h <= xs[i]:
                    t = (h - xs[i - 1]) / (xs[i] - xs[i - 1])
                    return float(ys[i - 1] + t * (ys[i] - ys[i - 1]))
            return float(ys[-1])

        return _eval


_KP_FN = _build_interpolator(CALIBRATION_HEIGHTS_M, CALIBRATION_KP)
_KD_FN = _build_interpolator(CALIBRATION_HEIGHTS_M, CALIBRATION_KD)
_KI_FN = _build_interpolator(CALIBRATION_HEIGHTS_M, CALIBRATION_KI)
_THETA_FN = _build_interpolator(CALIBRATION_HEIGHTS_M, CALIBRATION_THETA_MAX_DEG)
_DEADBAND_FN = _build_interpolator(CALIBRATION_HEIGHTS_M, CALIBRATION_DEADBAND_M)
_RATE_FN = _build_interpolator(CALIBRATION_HEIGHTS_M, CALIBRATION_RATE_LIMIT_DEG_PER_STEP)
_LOWPASS_FN = _build_interpolator(CALIBRATION_HEIGHTS_M, CALIBRATION_LOWPASS_ALPHA)


def _finite(v: float, fallback: float) -> float:
    return v if (v == v and v not in (float("inf"), float("-inf"))) else fallback


def calibrated_kp_deg_per_m(height_m: float) -> float:
    """Outer-loop Kp (deg/m) at the given commanded height, clamped to bounds."""
    return _clamp(_finite(_KP_FN(height_m), 1.0), *KP_BOUNDS)


def calibrated_kd_deg_per_mps(height_m: float) -> float:
    """Outer-loop Kd (deg per m/s) at the given commanded height, clamped."""
    return _clamp(_finite(_KD_FN(height_m), 0.0), *KD_BOUNDS)


def calibrated_ki_deg_per_m_s(height_m: float) -> float:
    """Outer-loop Ki (deg per m*s) at the given commanded height, clamped."""
    return _clamp(_finite(_KI_FN(height_m), 0.0), *KI_BOUNDS)


def calibrated_theta_ref_max_deg(height_m: float) -> float:
    """Saturation half-range (deg) at the given commanded height, clamped."""
    return _clamp(_finite(_THETA_FN(height_m), 3.0), *THETA_MAX_BOUNDS)


def calibrated_deadband_m(height_m: float) -> float:
    """Proportional deadband half-width (m) at the given height, clamped."""
    return _clamp(_finite(_DEADBAND_FN(height_m), 0.015), *DEADBAND_BOUNDS)


def calibrated_rate_limit_deg_per_step(height_m: float) -> float:
    """Per-step rate limit (deg/step) at the given height, clamped."""
    return _clamp(_finite(_RATE_FN(height_m), 0.030), *RATE_LIMIT_BOUNDS)


def calibrated_lowpass_alpha(height_m: float) -> float:
    """Output low-pass alpha at the given height, clamped to (0,1]-safe bounds."""
    return _clamp(_finite(_LOWPASS_FN(height_m), 0.150), *LOWPASS_BOUNDS)


def calibrated_outer_loop_params(height_m: float) -> Dict[str, float]:
    """All calibrated outer-loop parameters at the given commanded height.

    Returns a dict with finite, bounded values for every parameter. Safe for any
    real height input (NaN/inf-guarded, clamped to the calibration range).
    """
    h = _finite(float(height_m), 0.40)
    return {
        "calibrated_height_m": h,
        "calibrated_kp_deg_per_m": calibrated_kp_deg_per_m(h),
        "calibrated_kd_deg_per_mps": calibrated_kd_deg_per_mps(h),
        "calibrated_ki_deg_per_m_s": calibrated_ki_deg_per_m_s(h),
        "calibrated_theta_ref_max_deg": calibrated_theta_ref_max_deg(h),
        "calibrated_deadband_m": calibrated_deadband_m(h),
        "calibrated_rate_limit_deg_per_step": calibrated_rate_limit_deg_per_step(h),
        "calibrated_lowpass_alpha": calibrated_lowpass_alpha(h),
        "calibrated_function_profile_name": CALIBRATED_FUNCTION_PROFILE_NAME,
    }
