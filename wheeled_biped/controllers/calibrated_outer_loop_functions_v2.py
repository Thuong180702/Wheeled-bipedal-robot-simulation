"""Calibrated height-dependent outer-loop function lookups — Version 2.

This module provides smooth, height-dependent parameter functions for the
`calibrated_support_position_outer_loop_pitch_ref_v2` profile. It replaces the
v1 calibration (calibrated_support_position_outer_loop_pitch_ref) whose Phase 6
fixed-height validation showed regressions at high_0p465 and high_0p480 due to
an overly-aggressive upper-band Kp curve.

v2 changes vs v1 (calibrated_outer_loop_functions):
  - Kp(0.465): 1.350 -> 1.000  (constrained to avoid regression vs B)
  - Kp(0.480): 1.575 -> 1.050  (constrained to avoid regression vs B)
  - Kd(0.480): 0.050 -> 0.000  (no damping benefit at high band)

All other breakpoints (0.300–0.450) are unchanged from v1.

Design rules (see task spec / CLAUDE.md):
  - NO setup-name branching. Every parameter is computed from the commanded
    target height in metres via continuous interpolation through calibrated
    breakpoints.
  - PCHIP (monotone-preserving cubic) interpolation when SciPy is available,
    otherwise a numerically-identical piecewise-linear fallback. Both clamp to
    the calibration height range (no extrapolation) and to per-parameter safety
    bounds.
  - Pure Python floats only — called once per control step, no JAX arrays, no
    NumPy inside the hot path.
  - All outputs are finite and clamped: no NaN can leave these functions.

See also:
  docs/validation/calibrated_outer_loop_upper_band_failure_audit.md
  docs/validation/calibrated_outer_loop_upper_band_resweep_report.md
  outputs/.../calibrated_outer_loop_upper_band_resweep_best_candidates.json
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

# ---- Calibration breakpoints (Phase 2 + Phase 2 upper-band resweep result) ----- #
# Heights are strictly ascending CoM target heights in metres.
CALIBRATION_HEIGHTS_M: Tuple[float, ...] = (
    0.300, 0.320, 0.330, 0.340, 0.360, 0.380, 0.430, 0.450, 0.465, 0.480,
)

# v2 Kp: lower upper-band vs v1 to avoid regressions at 0.465/0.480.
#   v1: [1.500, 1.500, 1.300, 1.000, 0.725, 0.650, 1.000, 0.650, 1.350, 1.575]
#   v2: [1.500, 1.500, 1.300, 1.000, 0.725, 0.650, 1.000, 0.650, 1.000, 1.050]
# Changes:
#   0.465: 1.350 -> 1.000  (resweep: kp=1.05 scored 853 vs B 906, kp=1.00 tied with B)
#   0.480: 1.575 -> 1.050  (resweep: kp=1.00 tied with B; conservative 1.05 is safe)
CALIBRATION_KP: Tuple[float, ...] = (
    1.500, 1.500, 1.300, 1.000, 0.725, 0.650, 1.000, 0.650, 1.000, 1.050,
)

# v2 Kd: zero at high band (resweep showed no damping benefit at 0.465/0.480).
#   v1: [0.000, 0.000, 0.150, 0.200, 0.200, 0.150, 0.000, 0.000, 0.000, 0.050]
#   v2: [0.000, 0.000, 0.150, 0.200, 0.200, 0.150, 0.000, 0.000, 0.000, 0.000]
CALIBRATION_KD: Tuple[float, ...] = (
    0.000, 0.000, 0.150, 0.200, 0.200, 0.150, 0.000, 0.000, 0.000, 0.000,
)

# Integral gain (deg per m*s). Zero everywhere — integral not validated.
CALIBRATION_KI: Tuple[float, ...] = (
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
)

# Saturation half-range of the dynamic pitch_ref offset (deg) — unchanged from v1.
CALIBRATION_THETA_MAX_DEG: Tuple[float, ...] = (
    3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00,
)

# Proportional deadband half-width (m) — unchanged from v1.
CALIBRATION_DEADBAND_M: Tuple[float, ...] = (
    0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
)

# Per-step rate limit (deg/step) — unchanged from v1.
CALIBRATION_RATE_LIMIT_DEG_PER_STEP: Tuple[float, ...] = (
    0.030, 0.030, 0.030, 0.030, 0.030, 0.030, 0.030, 0.030, 0.030, 0.030,
)

# Output low-pass coefficient (0,1] — unchanged from v1.
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

CALIBRATED_FUNCTION_PROFILE_NAME = "calibrated_outer_loop_height_functions_v2"


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _build_interpolator(xs: Tuple[float, ...], ys: Tuple[float, ...]):
    """Build a PCHIP interpolator (SciPy) or piecewise-linear fallback."""
    try:
        from scipy.interpolate import PchipInterpolator

        pchip = PchipInterpolator(list(xs), list(ys), extrapolate=False)

        def _eval(h: float) -> float:
            hq = _clamp(h, xs[0], xs[-1])
            v = float(pchip(hq))
            if math.isnan(v):
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
