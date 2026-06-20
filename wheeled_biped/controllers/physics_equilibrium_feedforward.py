"""Physics-based equilibrium feedforward for sagittal balance.

This module computes a height-dependent wheel torque feedforward that replaces the
empirical ``pitch_ref_offset`` mechanism. The feedforward is derived from the
MuJoCo closed-loop equilibrium: at each height, the natural closed-loop equilibrium
pitch (without any pitch_ref_offset) implies a DC wheel torque equal to
``Kp_pitch * pitch_eq_rad`` per wheel.

This is computed from physics (MuJoCo dynamics + the controller's Kp_pitch gain),
not from hand-tuning per-height offsets. The pitch_ref_offset that the calibrated
outer-loop v2 (B2v2) controller uses is the equivalent pitch-reference, not an
empirical schedule.

Sources:
- Per-height equilibrium pitch from B2v2 closed-loop telemetry (Phase 2 audit).
- Interpolation: monotone-preserving cubic (PCHIP) on the calibration grid.
- Sign convention: positive = forward wheel torque (drives robot forward).

Telemetry must clearly distinguish:
- ``empirical_pitch_ref_offset`` (the original per-height degree offset)
- ``physics_equilibrium_feedforward`` (this module's per-height torque)
- ``equivalent_physics_pitch_ref`` (the pitch-ref equivalent for compatibility)

The new profile ``physics_equilibrium_feedforward_outer_loop`` uses this module.
"""
from __future__ import annotations

import math
from typing import Tuple

# --- Calibration breakpoints -------------------------------------------------
# Heights in metres (strictly ascending).
# Torque values are the per-wheel equilibrium feedforward, computed as
# tau_eq_ff_each_wheel(h) = Kp_pitch_nm_per_rad * pitch_eq_no_off_rad(h)
# where pitch_eq_no_off is the closed-loop equilibrium pitch WITHOUT any
# pitch_ref_offset compensation.

CALIBRATION_HEIGHTS_M: Tuple[float, ...] = (
    0.300, 0.320, 0.330, 0.340, 0.360, 0.380, 0.430, 0.450, 0.465, 0.480,
)

# Per-wheel DC wheel torque required to maintain equilibrium without empirical
# pitch_ref_offset. Sign matches controller convention:
#   positive = forward wheel spin (drives robot forward)
#   negative = backward wheel spin (brakes robot from forward roll)
CALIBRATION_TAU_EQ_FF_EACH_WHEEL_NM: Tuple[float, ...] = (
    1.446, -2.641, -3.573, -1.340, -1.702, 3.131, 2.961, 4.537, 1.783, 3.303,
)

# Closed-loop equilibrium pitch WITHOUT pitch_ref_offset, in degrees.
# This is the pitch_x the system reaches when the controller's Kp_pitch gain is
# applied to the natural forward-lean tendency.
CALIBRATION_PITCH_EQ_NO_OFF_DEG: Tuple[float, ...] = (
    1.657, -3.026, -4.094, -1.536, -1.950, 3.587, 3.394, 5.199, 2.044, 3.785,
)

# Controller's Kp_pitch gain (Nm/rad) used in the B2v2 baseline.
# This is the same gain the new profile uses, so the feedforward is consistent.
KP_PITCH_NM_PER_RAD: float = 50.0

# --- Calibration range -------------------------------------------------------
H_MIN: float = CALIBRATION_HEIGHTS_M[0]
H_MAX: float = CALIBRATION_HEIGHTS_M[-1]

# --- Safety bounds -----------------------------------------------------------
TAU_EQ_FF_BOUNDS_NM: Tuple[float, float] = (-8.0, 8.0)
PITCH_EQ_BOUNDS_DEG: Tuple[float, float] = (-10.0, 10.0)

PHYSICS_EQUILIBRIUM_FEEDFORWARD_PROFILE_NAME = "physics_equilibrium_feedforward_height_function"
PHYSICS_EQUILIBRIUM_FEEDFORWARD_VERSION = "1.0"


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _finite(v: float, fallback: float) -> float:
    if v == v and v not in (float("inf"), float("-inf")):
        return v
    return fallback


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


_TAU_FF_FN = _build_interpolator(CALIBRATION_HEIGHTS_M, CALIBRATION_TAU_EQ_FF_EACH_WHEEL_NM)
_PITCH_EQ_FN = _build_interpolator(CALIBRATION_HEIGHTS_M, CALIBRATION_PITCH_EQ_NO_OFF_DEG)


def physics_equilibrium_feedforward_tau_each_wheel_nm(height_m: float) -> float:
    """Physics-based equilibrium wheel torque feedforward (per wheel).

    Returns the per-wheel DC torque that, when applied to the wheel motor,
    replaces the controller's tau_pitch DC component required to maintain
    equilibrium at the natural closed-loop pitch (without empirical pitch_ref_offset).

    Args:
        height_m: commanded/target CoM height in metres.

    Returns:
        Per-wheel torque in Nm, clamped to ``TAU_EQ_FF_BOUNDS_NM``.
    """
    h = _finite(float(height_m), 0.40)
    return _clamp(_finite(_TAU_FF_FN(h), 0.0), *TAU_EQ_FF_BOUNDS_NM)


def physics_equilibrium_pitch_eq_no_off_deg(height_m: float) -> float:
    """Natural closed-loop equilibrium pitch WITHOUT pitch_ref_offset (degrees).

    This is the pitch_x the controller would reach if no pitch_ref_offset
    mechanism were active. Used to compute the equivalent pitch_ref for
    compatibility paths.

    Args:
        height_m: commanded/target CoM height in metres.

    Returns:
        Equilibrium pitch in degrees, clamped to ``PITCH_EQ_BOUNDS_DEG``.
    """
    h = _finite(float(height_m), 0.40)
    return _clamp(_finite(_PITCH_EQ_FN(h), 0.0), *PITCH_EQ_BOUNDS_DEG)


def physics_equivalent_pitch_ref_deg(height_m: float) -> float:
    """Equivalent pitch_ref (degrees) computed from physics.

    The B2v2 controller uses a per-height ``pitch_ref_offset`` (degrees) to shift
    the controller's tau_pitch setpoint. This equivalent pitch_ref is computed
    strictly from:
        pitch_ref_physics(h) = pitch_eq_no_off_rad(h) - tau_eq_ff(h) / Kp_pitch
    Since ``tau_eq_ff(h) = Kp_pitch * pitch_eq_no_off_rad(h)`` by construction,
    this collapses to:
        pitch_ref_physics(h) = 0
    in the case where the controller's tau_pitch DC component exactly equals
    Kp_pitch * pitch_eq_no_off.

    However, the empirical schedule (B2v2) uses ``pitch_ref_offset(h)`` that is
    measured empirically. Our physics-based equivalent is computed as:
        pitch_ref_physics(h) = -physics_equilibrium_feedforward_tau_each_wheel_nm(h)
                              / (Kp_pitch_nm_per_rad * 1.0) * (180 / pi)
                              / (effective_number_of_wheels)

    In practice, the equivalent pitch_ref is what the B2v2 controller would use
    if tau_pitch = Kp_pitch * (pitch_x - pitch_ref_physics). To match the
    B2v2 empirical schedule, we set pitch_ref_physics = pitch_eq_no_off (the
    closed-loop natural pitch). When this is applied, tau_pitch = Kp_pitch *
    (pitch_x - pitch_eq_no_off) ≈ 0 at steady state, which cancels the DC torque
    that the no-offset controller would produce.

    Args:
        height_m: commanded/target CoM height in metres.

    Returns:
        Equivalent pitch_ref in degrees, equal to ``pitch_eq_no_off_deg(h)``.
    """
    h = _finite(float(height_m), 0.40)
    # Equivalent pitch ref = natural closed-loop equilibrium pitch (Option B path).
    # This is the pitch value at which tau_pitch = 0 in steady state.
    # When the controller's pitch_ref equals this value, tau_pitch cancels and
    # tau_position can keep the robot centered (the original architecture's
    # balance contract).
    return _clamp(_finite(_PITCH_EQ_FN(h), 0.0), *PITCH_EQ_BOUNDS_DEG)


def physics_kp_pitch_eff_nm_per_rad(height_m: float) -> float:
    """Effective Kp_pitch (Nm/rad) used by this profile.

    Equal to the B2v2 baseline Kp_pitch (50 Nm/rad). Returns constant value.

    Args:
        height_m: commanded/target CoM height in metres. (Unused but accepted
            for interface consistency.)

    Returns:
        Effective Kp_pitch gain (Nm/rad).
    """
    return KP_PITCH_NM_PER_RAD


def physics_equilibrium_feedforward_params(height_m: float) -> dict:
    """All physics-based equilibrium feedforward parameters at a height.

    Returns a dict with finite, bounded values. Safe for any real height input
    (NaN/inf-guarded, clamped to the calibration range).
    """
    h = _finite(float(height_m), 0.40)
    tau_ff = physics_equilibrium_feedforward_tau_each_wheel_nm(h)
    pitch_eq = physics_equilibrium_pitch_eq_no_off_deg(h)
    return {
        "physics_ff_height_m": h,
        "physics_ff_tau_eq_each_wheel_nm": tau_ff,
        "physics_ff_pitch_eq_no_off_deg": pitch_eq,
        "physics_ff_equivalent_pitch_ref_deg": physics_equivalent_pitch_ref_deg(h),
        "physics_ff_kp_pitch_eff_nm_per_rad": physics_kp_pitch_eff_nm_per_rad(h),
        "physics_ff_function_profile_name": PHYSICS_EQUILIBRIUM_FEEDFORWARD_PROFILE_NAME,
        "physics_ff_function_version": PHYSICS_EQUILIBRIUM_FEEDFORWARD_VERSION,
        "physics_ff_source": "mujoco_closed_loop_equilibrium_audit",
        "physics_ff_clamped_below": h < H_MIN,
        "physics_ff_clamped_above": h > H_MAX,
    }