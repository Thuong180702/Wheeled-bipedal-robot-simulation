"""Sagittal velocity-damped balance controller for balance-core architecture.

Analytic controller-by-construction approach: explicit state-feedback terms for
pitch balance, sagittal velocity damping, wheel velocity damping, and optional
weak position return. Built after the LQR/sysid path failed Gate 4 identification
(one-step R²=1.0 but 20-step rollout R²=-1.15e10, dominant eigenvalue λ=1.96).

This controller replaces SagittalWheelBalanceController when selected via
--sagittal-controller velocity-damped. Both controllers are mutually exclusive.

Output: nonzero torque only on wheel joints [4, 9].
"""

import math

import jax.numpy as jnp
from jax import Array
from dataclasses import dataclass, replace
from typing import Optional

from wheeled_biped.controllers.balance_core_types import (
    ACTION_DIM,
    WHEEL_INDICES,
    zeros_action,
)
from wheeled_biped.controllers.position_hold_capture_gate import (
    PositionHoldCaptureGate,
    CaptureGateDiagnostics,
)
from wheeled_biped.controllers.signal_filters import BiquadNotchFilter, FirstOrderLowPassFilter, smoothstep_gate


def smoothstep01(u: float) -> float:
    """Standard smoothstep interpolation: s(0)=0, s(1)=1, s'(0)=s'(1)=0."""
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def interpolate_pitch_ref_offset(
    height_m: float,
    heights_m: tuple[float, ...],
    offsets_deg: tuple[float, ...],
    clamp: bool = True,
) -> float:
    """Piecewise-linear lookup of the scheduled pitch_ref offset for a height.

    Used by the height_scheduled_pitch_equilibrium_trim structural fix: each
    height has a distinct equilibrium pitch, so the offset that centers support
    drift is height-dependent (see Phase 1 sweep). This is a pure offline lookup
    — no JAX arrays — called once per step on a Python float height.

    Args:
        height_m: query height (commanded/target CoM height in metres).
        heights_m: schedule breakpoints, strictly ascending.
        offsets_deg: offset at each breakpoint, same length as heights_m.
        clamp: when True (default) hold the endpoint offset outside the range;
            when False, linearly extrapolate from the two nearest breakpoints.

    Returns:
        Interpolated pitch_ref offset in degrees. Returns 0.0 if the schedule is
        empty or malformed (defensive: an empty schedule means "no offset").
    """
    n = len(heights_m)
    if n == 0 or n != len(offsets_deg):
        return 0.0
    if n == 1:
        return float(offsets_deg[0])

    # Below the lowest breakpoint.
    if height_m <= heights_m[0]:
        if clamp:
            return float(offsets_deg[0])
        h0, h1 = heights_m[0], heights_m[1]
        o0, o1 = offsets_deg[0], offsets_deg[1]
        t = (height_m - h0) / (h1 - h0)
        return float(o0 + t * (o1 - o0))

    # Above the highest breakpoint.
    if height_m >= heights_m[-1]:
        if clamp:
            return float(offsets_deg[-1])
        h0, h1 = heights_m[-2], heights_m[-1]
        o0, o1 = offsets_deg[-2], offsets_deg[-1]
        t = (height_m - h0) / (h1 - h0)
        return float(o0 + t * (o1 - o0))

    # Interior: find the bracketing segment.
    for i in range(1, n):
        if height_m <= heights_m[i]:
            h0, h1 = heights_m[i - 1], heights_m[i]
            o0, o1 = offsets_deg[i - 1], offsets_deg[i]
            t = (height_m - h0) / (h1 - h0)
            return float(o0 + t * (o1 - o0))
    return float(offsets_deg[-1])  # unreachable, defensive


def compute_outer_loop_pitch_ref(
    support_error_m: float,
    support_error_rate_m_s: float,
    integral_error_m_s: float,
    kp_deg_per_m: float,
    kd_deg_per_mps: float,
    ki_deg_per_m_s: float,
    deadband_m: float,
    theta_ref_max_deg: float,
) -> float:
    """Raw dynamic pitch_ref offset (deg) for the Phase B support-position outer loop.

    PD(+I) on the live support-position error, layered on top of the frozen
    height schedule. Pure Python float — no JAX — called once per control step.

    The restoring SIGN is carried entirely by the caller-supplied gain signs; this
    function applies no implicit sign convention. A positive ``support_error_m``
    (forward drift) times a positive ``kp_deg_per_m`` produces a positive offset.

    Order of operations:
      1. Deadband: the proportional term is zeroed while ``abs(support_error)`` is
         below ``deadband_m`` (no nudging while already centered). The derivative
         and integral terms are NOT deadbanded — damping/integral should keep
         acting through the band.
      2. Sum: ``Kp*error_after_deadband + Kd*error_rate + Ki*integral``.
      3. Saturation: clamp to ``[-theta_ref_max_deg, +theta_ref_max_deg]``.

    Args:
        support_error_m: live signed support-position error (m). Positive = forward.
        support_error_rate_m_s: low-passed derivative of the support error (m/s).
        integral_error_m_s: clamped integral accumulator (m*s); pass 0.0 when the
            integral path is disabled.
        kp_deg_per_m: proportional gain (deg per m). Sign selects restoring direction.
        kd_deg_per_mps: derivative gain (deg per m/s).
        ki_deg_per_m_s: integral gain (deg per m*s); 0.0 disables the integral path.
        deadband_m: proportional deadband half-width (m).
        theta_ref_max_deg: saturation half-range for the dynamic offset (deg).

    Returns:
        Saturated dynamic pitch_ref offset in degrees (before rate-limit/lowpass).
    """
    if abs(support_error_m) < deadband_m:
        error_p = 0.0
    else:
        error_p = support_error_m
    dynamic = (
        kp_deg_per_m * error_p
        + kd_deg_per_mps * support_error_rate_m_s
        + ki_deg_per_m_s * integral_error_m_s
    )
    if dynamic > theta_ref_max_deg:
        return float(theta_ref_max_deg)
    if dynamic < -theta_ref_max_deg:
        return float(-theta_ref_max_deg)
    return float(dynamic)


def apply_rate_limit(prev: float, target: float, max_delta: float) -> float:
    """Limit the per-step change from ``prev`` toward ``target`` to ``max_delta``.

    ``max_delta <= 0`` disables limiting (returns ``target`` unchanged). Pure float.
    """
    if max_delta <= 0.0:
        return float(target)
    delta = target - prev
    if delta > max_delta:
        return float(prev + max_delta)
    if delta < -max_delta:
        return float(prev - max_delta)
    return float(target)


def apply_lowpass(prev: float, target: float, alpha: float) -> float:
    """First-order low-pass: ``(1-alpha)*prev + alpha*target``.

    ``alpha <= 0`` holds ``prev``; ``alpha >= 1`` returns ``target`` (no filtering).
    Pure float.
    """
    if alpha <= 0.0:
        return float(prev)
    if alpha >= 1.0:
        return float(target)
    return float((1.0 - alpha) * prev + alpha * target)


def scheduled_k_position(
    z_ref: float,
    k_nominal: float,
    k_low_max: float,
    z_low: float,
    z_high: float,
) -> float:
    """Compute k_position as a smooth function of height.

    Uses smoothstep interpolation between k_nominal and k_low_max.

    Behavior:
        - z_ref = z_low  -> u = 1 -> smoothstep = 1 -> k_position = k_low_max
        - z_ref = z_high -> u = 0 -> smoothstep = 0 -> k_position = k_nominal
        - z_ref > z_high -> u clamped to 0 -> k_position = k_nominal
        - z_ref < z_low  -> u clamped to 1 -> k_position = k_low_max

    Args:
        z_ref: Commanded/reference height (m)
        k_nominal: k_position at nominal/high heights
        k_low_max: k_position at lowest heights
        z_low: Lower boundary where max authority applies
        z_high: Upper boundary where nominal authority applies

    Returns:
        Smoothly interpolated k_position value
    """
    u = (z_high - z_ref) / (z_high - z_low)
    s = smoothstep01(u)
    return k_nominal + (k_low_max - k_nominal) * s


def scheduled_k_wheel_velocity(
    z_ref: float,
    k_nominal: float,
    k_high_max: float,
    z_low: float,
    z_high: float,
) -> float:
    """Compute k_wheel_velocity as a smooth function of height for high-height damping.

    Uses smoothstep interpolation between k_nominal and k_high_max.
    This is the inverse of scheduled_k_position: k increases at HIGH heights.

    Behavior:
        - z_ref = z_low  -> u = 1 -> smoothstep = 1 -> k_wheel_velocity = k_nominal (no extra damping)
        - z_ref = z_high -> u = 0 -> smoothstep = 0 -> k_wheel_velocity = k_high_max (full extra damping)
        - z_ref > z_high -> u clamped to 0 -> k_wheel_velocity = k_high_max
        - z_ref < z_low  -> u clamped to 1 -> k_wheel_velocity = k_nominal

    Args:
        z_ref: Commanded/reference height (m)
        k_nominal: k_wheel_velocity at low/nominal heights
        k_high_max: k_wheel_velocity at highest heights
        z_low: Lower boundary where nominal damping applies
        z_high: Upper boundary where max damping applies

    Returns:
        Smoothly interpolated k_wheel_velocity value
    """
    u = (z_high - z_ref) / (z_high - z_low)
    s = smoothstep01(u)
    return k_high_max + (k_nominal - k_high_max) * s


@dataclass(frozen=True)
class SagittalAuthoritySchedule:
    profile_name: str = "baseline"
    applies_to_variants: tuple[str, ...] = ()
    position_tau_cap_scale: float = 1.0
    position_tau_cap_by_variant: tuple[tuple[str, float], ...] = ()
    pitch_tau_scale: float = 1.0
    pitch_tau_cap_nm: float | None = None
    velocity_damping_scale: float = 1.0
    support_velocity_scale: float = 1.0
    support_velocity_gain: float | None = None

    # Continuous k_position scheduling fields
    continuous_k_position: bool = False
    k_position_nominal: float = 40.0
    k_position_low_max: float = 80.0
    k_position_z_low: float = 0.300
    k_position_z_high: float = 0.393

    # Continuous max_position_tau scheduling (Phase 6 joint fix)
    continuous_max_position_tau: bool = False
    max_position_tau_nominal: float = 3.0
    max_position_tau_low_max: float = 6.0

    # Continuous k_velocity scheduling (Phase 6 joint fix)
    continuous_k_velocity: bool = False
    k_velocity_nominal: float = 15.0
    k_velocity_low_max: float = 25.0

    # Continuous k_wheel_velocity scheduling (Step E extreme height fix)
    continuous_k_wheel_velocity: bool = False
    k_wheel_velocity_nominal: float = 0.5
    k_wheel_velocity_high_max: float = 0.75
    k_wheel_velocity_z_low: float = 0.45
    k_wheel_velocity_z_high: float = 0.52

    # Continuous kd_pitch scheduling (Tall-height WIP damping fix).
    # Increases pitch-rate damping at tall heights to suppress the 2.5 Hz
    # wheeled inverted pendulum mode without affecting low-height behavior.
    # Maps: z_ref <= z_low -> kd_pitch_nominal (no extra damping)
    #       z_ref >= z_high -> kd_pitch_high_max (full extra damping)
    # Uses smoothstep interpolation between the two bounds.
    # Disabled by default so every legacy profile is unchanged.
    continuous_kd_pitch: bool = False
    kd_pitch_nominal: float = 10.0
    kd_pitch_high_max: float = 20.0
    kd_pitch_z_low: float = 0.40
    kd_pitch_z_high: float = 0.52

    # Position integral settings (Step E extreme height fix)
    enable_position_integral: bool = False
    ki_position_integral: float = 0.0  # 0.0 when disabled
    integral_max_abs: float = 0.0  # 0.0 when disabled

    # Pitch equilibrium trim (Phase 3 fix)
    # Positive offset makes controller target backward lean, reducing forward drift
    # Default 0.0, recommended values: 2.0-4.0 deg for high variants
    pitch_ref_offset_deg: float = 0.0

    # Height-scheduled pitch reference offset (structural fix, Phase 2).
    # When enabled, the pitch_ref offset is looked up from a per-height schedule
    # via piecewise-linear interpolation instead of using the static
    # pitch_ref_offset_deg. The static offset (pitch_equilibrium_trim) stays a
    # single forward lean tuned for high_0p480 and over-corrects low heights; the
    # schedule supplies the correct equilibrium-pitch offset at each height.
    # Disabled by default so baseline and all legacy profiles are unchanged.
    # heights_m must be ascending and the same length as offsets_deg. Below the
    # lowest / above the highest scheduled height the endpoint value is held when
    # pitch_ref_height_schedule_clamp is True.
    pitch_ref_height_schedule_enabled: bool = False
    pitch_ref_height_schedule_heights_m: tuple[float, ...] = ()
    pitch_ref_height_schedule_offsets_deg: tuple[float, ...] = ()
    pitch_ref_height_schedule_clamp: bool = True
    # Optional smoothing of the scheduled offset when the commanded height moves
    # (only relevant for height transitions; inert for fixed-height runs).
    # 0.0 disables each. rate_limit caps deg/step change; lowpass_alpha in (0,1].
    pitch_ref_offset_rate_limit_deg_per_step: float = 0.0
    pitch_ref_offset_lowpass_alpha: float = 0.0

    
    # Support-position outer loop (Phase B dynamic pitch-reference correction).
    # A bounded, gated, opt-in real-time correction to pitch_ref driven by the live
    # support-position error, layered ON TOP of the frozen height schedule above.
    # The schedule supplies the height-dependent DC operating point; this loop adds
    # slow centering feedback around it. Disabled by default so every legacy profile
    # (and height_scheduled_pitch_equilibrium_trim itself) is byte-for-byte unchanged.
    # The restoring SIGN is carried by the configured Kp sign, proven empirically in
    # the Phase 4 sign sweep — no sign is hard-coded as "correct". Integral is
    # disabled initially (Ki = 0). See
    # docs/validation/support_position_outer_loop_pitch_ref_design.md.
    outer_loop_enabled: bool = False
    outer_loop_kp_deg_per_m: float = 0.0
    outer_loop_kd_deg_per_mps: float = 0.0
    outer_loop_ki_deg_per_m_s: float = 0.0
    outer_loop_integral_enabled: bool = False
    outer_loop_integral_clamp_m_s: float = 0.05
    outer_loop_theta_ref_max_deg: float = 3.0
    outer_loop_theta_ref_rate_limit_deg_per_step: float = 0.03
    outer_loop_theta_ref_lowpass_alpha: float = 0.15
    outer_loop_support_error_deadband_m: float = 0.015
    outer_loop_support_velocity_lowpass_alpha: float = 0.20
    outer_loop_disable_if_abs_error_gt_m: float = 0.25
    outer_loop_disable_if_pitch_gt_deg: float = 12.0
    outer_loop_disable_if_roll_gt_deg: float = 5.0
    outer_loop_contact_required: bool = True
    outer_loop_height_schedule_required: bool = True

    # Support-position outer loop (Phase B dynamic centering).
    # A bounded, gated, opt-in additive nudge to pitch_ref driven by the live
    # support-position error, layered ON TOP of the frozen height schedule above.
    # Disabled by default so every legacy profile (and the Phase A base profile
    # height_scheduled_pitch_equilibrium_trim) is byte-for-byte unchanged: when
    # outer_loop_enabled is False the dynamic term is identically 0.0 and the
    # applied pitch_ref equals the scheduled offset.
    # The sign of the restoring direction lives entirely in the configured
    # outer_loop_kp_deg_per_m value; it is NOT hard-coded in the control law and
    # must be selected empirically (Phase 4 two-sign sweep). Integral disabled
    # initially (Ki = 0, integral_enabled = False) — PD only.
    # See docs/validation/support_position_outer_loop_pitch_ref_design.md.
    outer_loop_enabled: bool = False
    outer_loop_kp_deg_per_m: float = 0.0
    outer_loop_kd_deg_per_mps: float = 0.0
    outer_loop_ki_deg_per_m_s: float = 0.0
    outer_loop_integral_enabled: bool = False
    outer_loop_integral_clamp_m_s: float = 0.05
    outer_loop_theta_ref_max_deg: float = 3.0
    outer_loop_theta_ref_rate_limit_deg_per_step: float = 0.03
    outer_loop_theta_ref_lowpass_alpha: float = 0.15
    outer_loop_support_error_deadband_m: float = 0.015
    outer_loop_support_velocity_lowpass_alpha: float = 0.20
    outer_loop_disable_if_abs_error_gt_m: float = 0.25
    outer_loop_disable_if_pitch_gt_deg: float = 12.0
    outer_loop_disable_if_roll_gt_deg: float = 5.0
    outer_loop_contact_required: bool = True
    outer_loop_height_schedule_required: bool = True

    # Calibrated height-dependent outer loop (Phase B calibration, opt-in).
    # When True, the runtime computes Kp/Kd/Ki/theta_max/deadband/rate_limit/
    # lowpass from the continuous height functions in
    # wheeled_biped/controllers/calibrated_outer_loop_functions.py (fitted from
    # the Phase 2 per-height gain sweep) instead of the fixed scalar outer_loop_*
    # values above. NO setup-name branching — every parameter is interpolated
    # from the commanded target CoM height. Disabled by default so the base
    # support_position_outer_loop_pitch_ref profile and all legacy profiles keep
    # their fixed scalar gains byte-for-byte. See
    # docs/validation/calibrated_support_position_outer_loop_pitch_ref_final_report.md.
    calibrated_outer_loop_enabled: bool = False
    calibrated_outer_loop_function_version: str = "v1"

    # PFF low-band support shaping (opt-in candidate only).
    # Smoothly enables a bounded support-position pitch-ref correction around
    # 0.320 m without touching the physics equilibrium feedforward source.
    low_band_support_outer_loop_enabled: bool = False
    low_band_support_center_m: float = 0.320
    low_band_support_sigma_m: float = 0.006
    low_band_support_kp_peak_deg_per_m: float = 1.5
    low_band_support_theta_ref_max_peak_deg: float = 3.00
    low_band_support_pitch_ref_offset_peak_deg: float = 0.0
    # When True, the low-band support Kp blends with the base (calibrated) Kp
    # so that the effective Kp does not drop to zero at heights far from the
    # low-band center. At scale=1 (near center), Kp ≈ peak_kp; at scale=0 (far),
    # Kp ≈ base_kp. This is required for tall-height push recovery where the
    # support correction must remain active. Default False for backward compat.
    low_band_support_blend_with_base: bool = False

    # Physics-based equilibrium feedforward (Phase D, opt-in).
    # When True, the runtime reads tau_eq_ff(h) from
    # wheeled_biped/controllers/physics_equilibrium_feedforward.py and adds it
    # directly to the final wheel torque each step. This replaces the empirical
    # pitch_ref_height_schedule — see the physics_equilibrium_feedforward_outer_loop
    # profile below. The feedforward is derived from MuJoCo closed-loop
    # equilibrium dynamics, not hand-tuned per-height offsets. Disabled by
    # default so every legacy profile (and B2v2) is byte-for-byte unchanged.
    physics_equilibrium_feedforward_enabled: bool = False
    physics_eq_ff_clamp_to_height_range: bool = True
    physics_eq_ff_function_version: str = ""
    physics_eq_ff_max_abs_nm: float = 8.0  # safety clamp

    # Unified sagittal state-feedback no-offset controller.
    # Replaces the independent tau_pitch + tau_position + tau_velocity_damping
    # sum-of-torques architecture with a single coordinated sagittal command
    # from full state feedback. Requires pitch_ref_offset_deg = 0.0 and disables
    # all offset/trim/bias mechanisms. See
    # docs/validation/unified_sagittal_no_offset_design.md.
    # Disabled by default — opt-in via dedicated profile.
    enable_unified_sagittal_state_feedback: bool = False

    # Unified controller gains (scheduled by height if *_height_schedule is True)
    # support error proportional gain
    unified_kx: float = 40.0
    # support error rate gain
    unified_kv: float = 15.0
    # pitch proportional gain (+sign: forward lean -> forward torque; 0 = disabled in pure-support mode)
    unified_ktheta: float = 0.0
    # pitch rate gain
    unified_komega: float = 10.0
    # height error gain (typically 0 — height controlled by leg PD)
    unified_kh: float = 0.0
    # height rate gain (typically 0)
    unified_khdot: float = 0.0
    # Torque cap for unified controller (Nm)
    unified_torque_cap: float = 5.0
    # Rate limit (Nm/step)
    unified_rate_limit: float = 0.10

    # Height-scheduled unified gains
    unified_gain_height_schedule: bool = False
    unified_kx_nominal: float = 3.0
    unified_kx_low_max: float = 5.0
    unified_kv_nominal: float = 0.15
    unified_kv_low_max: float = 0.30
    unified_ktheta_nominal: float = 3.0
    unified_ktheta_low_max: float = 4.0
    unified_komega_nominal: float = 0.15
    unified_komega_low_max: float = 0.25
    unified_torque_cap_nominal: float = 6.0
    unified_torque_cap_low_max: float = 6.0

    # Unified controller mode classifier thresholds
    unified_drift_enter_m: float = 0.04
    unified_drift_exit_m: float = 0.02
    unified_push_pitch_enter_rad: float = 0.15
    unified_push_pitch_rate_enter_radps: float = 0.20
    unified_push_exit_rad: float = 0.05
    unified_height_transition_enter_m: float = 0.005
    unified_hip_yaw_risk_rad: float = 0.10
    unified_hip_yaw_danger_rad: float = 0.15
    unified_contact_degraded: int = 2  # fewer than this = degraded

    # Priority weights per mode
    unified_support_weight_steady: float = 1.0
    unified_pitch_weight_steady: float = 1.0
    unified_rate_weight_steady: float = 1.0
    unified_height_weight_steady: float = 0.5
    unified_support_weight_drift: float = 2.0
    unified_pitch_weight_drift: float = 0.7
    unified_rate_weight_drift: float = 1.5
    unified_support_weight_push: float = 0.5
    unified_pitch_weight_push: float = 2.0
    unified_rate_weight_push: float = 1.0
    unified_support_weight_transition: float = 0.7
    unified_pitch_weight_transition: float = 0.7
    unified_support_weight_degraded: float = 0.5
    unified_pitch_weight_degraded: float = 1.5
    unified_rate_weight_degraded: float = 0.5
    unified_support_weight_hip_yaw_risk: float = 0.5
    unified_pitch_weight_hip_yaw_risk: float = 0.5
    unified_rate_weight_hip_yaw_risk: float = 0.3

    integral_pitch_error_threshold_rad: float = 0.03
    integral_support_velocity_threshold_m_s: float = 0.03
    integral_wheel_velocity_threshold_rad_s: float = 1.0
    integral_min_com_z_m: float = 0.28
    integral_max_com_z_m: float = 0.50

    # Pitch-aware position scaling (Option C pitch-safe fix)
    enable_pitch_aware_position_scaling: bool = False
    pitch_soft_start: float = 0.06  # rad, scaling begins
    pitch_hard_limit: float = 0.10  # rad, full scaling applied
    min_pitch_scale: float = 0.7  # minimum scale factor at hard limit

    # Phase-aware recenter (F1_strategy - signed drift fix)
    # Decouples recentering from tau_position to avoid hip yaw coupling
    enable_phase_aware_recenter: bool = False
    k_recenter: float = 10.0  # Nm/m - gain for recenter term
    max_recenter_tau: float = 1.0  # Nm - max recenter torque (bounded, separate from balance)
    recenter_deadband_m: float = 0.01  # m - ignore small signed errors
    recenter_pitch_safe_threshold_rad: float = 0.05  # rad - pitch must be within this to recenter
    recenter_pitch_danger_threshold_rad: float = 0.10  # rad - pitch above this blocks recenter
    recenter_hip_yaw_safe_threshold_rad: float = 0.10  # rad - hip_yaw_abs_max must be below this
    recenter_smooth_alpha: float = 0.10  # smoothing factor for recenter term
    recenter_max_rate_per_step: float = 0.5  # Nm/step - rate limit for recenter term
    recenter_min_com_z_m: float = 0.28  # m - min height for recenter
    recenter_max_com_z_m: float = 0.50  # m - max height for recenter

    # Hysteresis recenter (F2_strategy - stateful recenter for stronger bias correction)
    # Holds recenter direction until error returns to exit target, preventing early reversal
    enable_hysteresis_recenter: bool = False
    hysteresis_outer_enter_m: float = 0.10  # m - outer threshold to enter recenter state
    hysteresis_exit_target_m: float = 0.00  # m - exit when error reaches this target
    hysteresis_opposite_overshoot_m: float = 0.01  # m - slight overshoot into opposite direction
    hysteresis_k_recenter: float = 10.0  # Nm/m - gain for hysteresis recenter term
    hysteresis_max_recenter_tau: float = 1.5  # Nm - max hysteresis recenter torque
    hysteresis_smooth_alpha: float = 0.10  # smoothing factor
    hysteresis_max_rate_per_step: float = 0.5  # Nm/step - rate limit
    hysteresis_deadband_m: float = 0.01  # m - ignore small errors in NEUTRAL state
    hysteresis_pitch_safe_threshold_rad: float = 0.05  # rad - pitch must be within this
    hysteresis_pitch_danger_threshold_rad: float = 0.10  # rad - pitch above this blocks recenter
    hysteresis_hip_yaw_safe_threshold_rad: float = 0.15  # rad - hip_yaw_abs_max must be below this
    hysteresis_min_com_z_m: float = 0.28  # m - min height for recenter
    hysteresis_max_com_z_m: float = 0.50  # m - max height for recenter

    # Bias cancellation (G1_strategy - persistent bias cancellation for one-sided drift)
    # Estimates persistent signed error bias and applies bounded opposite torque
    # Unlike F1/F2 which wait for natural drift, G1 estimates bias and cancels it proactively
    enable_bias_cancel: bool = False
    bias_cancel_k: float = 12.0  # Nm/m - gain for bias cancellation
    bias_cancel_max_tau: float = 1.5  # Nm - max bias cancellation torque
    bias_cancel_filter_alpha: float = 0.02  # filter coefficient for leaky integration
    bias_cancel_deadband_m: float = 0.02  # m - ignore small persistent errors
    bias_cancel_contact_gate: bool = True  # require valid contact
    bias_cancel_height_gate: bool = True  # require valid height
    bias_cancel_roll_gate: bool = True  # require valid roll
    bias_cancel_pitch_gate: bool = False  # NOT gated on pitch (pitch reversal doesn't produce negative drift)
    bias_cancel_min_com_z_m: float = 0.28  # m - min height for bias cancel
    bias_cancel_max_com_z_m: float = 0.50  # m - max height for bias cancel
    bias_cancel_roll_threshold_rad: float = 0.15  # rad - roll must be below this

    # Active Pitch Crossing (APC_strategy - explicit pitch-rate crossing controller)
    # Actively drives wheel torque to create controlled pitch-rate reversal
    # When robot has positive pitch AND positive signed drift, APC applies wheel torque
    # to reverse pitch_rate, allowing support to return toward 0.
    enable_active_pitch_crossing: bool = False
    apc_outer_enter_m: float = 0.10  # m - enter crossing when |signed_error| > this
    apc_inner_exit_m: float = 0.05  # m - exit crossing when |signed_error| <= this
    apc_opposite_overshoot_m: float = 0.01  # m - allow slight overshoot into opposite direction
    apc_pitch_enter_rad: float = 0.03  # rad - pitch must exceed this to enter crossing
    apc_pitch_safe_limit_rad: float = 0.08  # rad - reduce torque if pitch exceeds this
    apc_max_cross_tau: float = 1.5  # Nm - max crossing torque
    apc_smooth_alpha: float = 0.10  # smoothing factor for crossing torque
    apc_max_rate_per_step: float = 0.5  # Nm/step - rate limit
    apc_contact_gate: bool = True  # require valid contact
    apc_height_gate: bool = True  # require valid height
    apc_roll_gate: bool = True  # require valid roll
    apc_min_com_z_m: float = 0.28  # m - min height for crossing
    apc_max_com_z_m: float = 0.50  # m - max height for crossing
    apc_pitch_safe_threshold_rad: float = 0.05  # rad - pitch must be within this to enter
    apc_pitch_danger_threshold_rad: float = 0.10  # rad - pitch above this blocks crossing
    apc_roll_threshold_rad: float = 0.15  # rad - roll must be below this

    # APCR1d proportional soft band parameters
    apc_soft_enter_m: float = 0.05  # m - enter soft recenter when |error| > this
    apc_velocity_decay_enabled: bool = False  # reduce torque when moving toward zero
    apc_velocity_decay_factor: float = 0.5  # decay factor when moving toward zero
    apc_proportional_soft_band_mode: bool = False  # enable proportional torque shaping

    # APCR1e adaptive authority parameters
    # Automatically increases torque when error is not improving
    apc_adaptive_authority_enabled: bool = False  # enable adaptive authority
    apc_adaptive_base_tau: float = 0.55  # Nm - base starting torque
    apc_adaptive_max_tau: float = 1.20  # Nm - maximum adaptive torque
    apc_adaptive_boost_tau_max: float = 0.65  # Nm - maximum boost above base
    apc_adaptive_boost_start_error_m: float = 0.06  # m - error threshold for boost
    apc_adaptive_full_boost_error_m: float = 0.12  # m - error for full boost
    apc_adaptive_no_improvement_window_steps: int = 8  # steps without improvement
    apc_adaptive_startup_boost_steps: int = 50  # startup phase duration
    apc_adaptive_startup_boost_max_tau: float = 1.0  # Nm - max startup torque
    apc_adaptive_disable_vd_when_abs_e_gt: float = 0.10  # m - disable VD above this
    apc_adaptive_disable_vd_during_startup: bool = True  # disable VD in startup
    apc_adaptive_max_rate_per_step: float = 0.35  # Nm/step - rate limit

    # APCR1f adaptive fast response with phase brake parameters
    # Key differences from APCR1e:
    # - Earlier intervention (0.035m vs 0.05m)
    # - Faster rate limit (0.55 Nm/step vs 0.35 Nm/step)
    # - Higher max_tau (1.40 Nm vs 1.20 Nm)
    # - Phase brake when error returning toward zero
    # - Boost when error growing 3+ consecutive steps
    apc_fast_response_enabled: bool = False  # enable fast response mode
    apc_phase_brake_enabled: bool = False  # enable phase-aware braking
    apc_phase_brake_threshold_m: float = 0.08  # m - apply brake below this
    apc_phase_brake_damping_factor: float = 0.6  # reduce scale by this when braking
    apc_boost_rate_per_step: float = 0.25  # Nm/step - rate for adaptive boost
    apc_decay_rate_per_step: float = 0.45  # Nm/step - faster decay when returning
    apc_increasing_error_threshold_steps: int = 3  # boost when error grows 3+ steps
    apc_increasing_error_boost_factor: float = 0.3  # boost factor for growing error
    apc_fast_response_inner_deadband_m: float = 0.015  # m - earlier deadband
    apc_fast_response_soft_enter_m: float = 0.035  # m - earlier soft enter
    apc_fast_response_desired_band_m: float = 0.08  # m - wider comfortable band
    apc_fast_response_full_torque_m: float = 0.10  # m - full torque at this error
    apc_fast_response_emergency_m: float = 0.12  # m - emergency mode trigger
    apc_fast_response_base_tau: float = 0.45  # Nm - slightly lower base
    apc_fast_response_max_tau: float = 1.40  # Nm - higher ceiling
    apc_fast_response_boost_tau_max: float = 0.95  # Nm - larger boost capability
    apc_fast_response_startup_boost_max_tau: float = 1.20  # Nm - higher startup authority
    apc_fast_response_max_rate_per_step: float = 0.55  # Nm/step - faster response
    apc_fast_response_smooth_alpha: float = 0.18  # more responsive smoothing
    apc_fast_response_no_improvement_window: int = 5  # faster boost (5 vs 8 steps)

    # APCR1g Predictive Fast Response with Phase Brake parameters
    # Key differences from APCR1f:
    # - Predictive error: e_pred = e + lead_time_s * e_dot
    # - Earlier activation when predicted error exceeds threshold
    # - Predictive boost when predicted error indicates future overshoot
    # - Stronger phase brake with two thresholds (threshold + strong threshold)
    # - Faster response: higher max_tau (1.55 vs 1.40), faster rate (0.70 vs 0.55)
    apc_predictive_enabled: bool = False  # enable predictive error logic
    apc_lead_time_s: float = 0.10  # seconds to predict ahead
    apc_predicted_enter_m: float = 0.07  # activate when abs_pred > this AND moving_away
    apc_predicted_full_response_m: float = 0.10  # boost authority when abs_pred > this
    apc_predicted_emergency_m: float = 0.12  # emergency mode when abs_pred > this
    apc_predictive_inner_deadband_m: float = 0.012  # m - decay to zero below this
    apc_predictive_soft_enter_m: float = 0.030  # m - earlier soft enter
    apc_predictive_desired_band_m: float = 0.075  # m - tighter band
    apc_predictive_full_torque_m: float = 0.095  # m - full torque at this error
    apc_predictive_emergency_error_m: float = 0.115  # m - emergency mode trigger
    apc_predictive_base_tau: float = 0.45  # Nm - base starting torque
    apc_predictive_max_tau: float = 1.55  # Nm - higher ceiling
    apc_predictive_boost_tau_max: float = 1.10  # Nm - larger boost capability
    apc_predictive_startup_boost_max_tau: float = 1.25  # Nm - higher startup authority
    apc_predictive_max_rate_per_step: float = 0.70  # Nm/step - faster response
    apc_predictive_boost_rate_per_step: float = 0.35  # Nm/step - rate for adaptive boost
    apc_predictive_decay_rate_per_step: float = 0.55  # Nm/step - faster decay when returning
    apc_predictive_smooth_alpha: float = 0.22  # more responsive smoothing
    apc_predictive_no_improvement_window: int = 4  # boost after 4 steps without improvement
    apc_predictive_increasing_error_threshold_steps: int = 2  # boost when error grows 2+ steps
    apc_predictive_increasing_error_boost_factor: float = 0.35  # boost factor for growing error
    apc_predictive_phase_brake_enabled: bool = False  # enable phase-aware braking
    apc_predictive_phase_brake_threshold_m: float = 0.075  # m - apply brake below this
    apc_predictive_phase_brake_strong_threshold_m: float = 0.050  # m - strong brake closer to zero
    apc_predictive_phase_brake_factor: float = 0.55  # reduce scale by this when braking
    apc_predictive_phase_brake_strong_factor: float = 0.35  # stronger reduction near zero
    apc_predictive_disable_vd_when_abs_e_gt: float = 0.10  # m - disable VD above this
    apc_predictive_disable_vd_during_startup: bool = True  # disable VD in startup

    # APCR1h Drift Priority Override parameters
    # Key: uses APCR1f torque sign convention (NOT APCR1g)
    # Activates when drift > 0.08 AND moving away for faster recovery
    apc_drift_priority_enabled: bool = False  # enable drift priority override
    apc_drift_priority_enter_m: float = 0.08  # m - drift priority activates at this threshold
    apc_drift_priority_emergency_m: float = 0.12  # m - emergency clamp threshold
    apc_drift_priority_hard_m: float = 0.15  # m - hard safety threshold
    apc_drift_priority_base_tau: float = 0.45  # Nm - base torque
    apc_drift_priority_normal_max_tau: float = 1.40  # Nm - normal max (same as APCR1f)
    apc_drift_priority_drift_priority_max_tau: float = 1.65  # Nm - drift priority max
    apc_drift_priority_emergency_max_tau: float = 1.85  # Nm - emergency clamp max
    apc_drift_priority_startup_max_tau: float = 1.60  # Nm - startup boost max
    apc_drift_priority_normal_rate: float = 0.55  # Nm/step - normal rate (same as APCR1f)
    apc_drift_priority_drift_priority_rate: float = 0.85  # Nm/step - drift priority rate
    apc_drift_priority_emergency_rate: float = 1.00  # Nm/step - emergency rate
    apc_drift_priority_decay_rate: float = 0.55  # Nm/step - decay rate
    apc_drift_priority_phase_brake_disable_threshold_m: float = 0.10  # m - disable phase brake above this

    # APCR Recovery Gate Mode: separates hard safety from recovery activation
    # When enabled, APCR can activate during moderate pitch error instead of blocking
    active_pitch_crossing_recovery_gate_mode: bool = False
    apcr_pitch_hard_stop_rad: float = 0.30  # rad - absolute emergency stop, blocks APCR
    apcr_roll_hard_stop_rad: float = 0.15  # rad - lateral stability threshold
    apcr_min_com_z_m: float = 0.27  # m - minimum safe height
    apcr_max_com_z_m: float = 0.50  # m - maximum operating height

    # APCR1i Support Hysteresis Recenter parameters
    # Symmetric hysteresis state machine that holds recenter direction
    # until error reaches inner band or crosses to opposite side
    apc_hysteresis_enabled: bool = False  # enable hysteresis recenter
    apc_hysteresis_outer_enter_m: float = 0.08  # m - enter recenter when |e| > this
    apc_hysteresis_inner_exit_m: float = 0.03  # m - exit recenter when |e| <= this
    apc_hysteresis_opposite_release_m: float = 0.03  # m - allow overshoot into opposite
    apc_hysteresis_near_zero_m: float = 0.01  # m - error considered near zero
    apc_hysteresis_emergency_m: float = 0.12  # m - emergency clamp threshold
    apc_hysteresis_hard_m: float = 0.15  # m - hard safety threshold
    apc_hysteresis_base_tau: float = 0.45  # Nm - base starting torque
    apc_hysteresis_recenter_max_tau: float = 1.75  # Nm - max during recenter state
    apc_hysteresis_emergency_max_tau: float = 2.00  # Nm - max during emergency
    apc_hysteresis_hold_max_tau: float = 1.50  # Nm - max during hold-through-zero
    apc_hysteresis_normal_rate: float = 0.30  # Nm/step - normal rate
    apc_hysteresis_recenter_rate: float = 0.90  # Nm/step - recenter rate
    apc_hysteresis_emergency_rate: float = 1.00  # Nm/step - emergency rate
    apc_hysteresis_decay_rate: float = 0.50  # Nm/step - decay rate when returning
    apc_hysteresis_phase_brake_threshold_m: float = 0.05  # m - enable phase brake below this
    apc_hysteresis_phase_brake_disable_in_recenter: bool = True  # disable phase brake in recenter

    # APCR1l Pitch Suppress in Recenter (torque path fix)
    # Suppresses tau_pitch during RECENTER state to let APCR + tau_position correct drift
    # without interference from pitch-stabilizing torque that fights correction lean
    apc_hysteresis_pitch_suppress_in_recenter: bool = False  # enable pitch suppression in recenter

    # APCR1m Conditional Pitch Blend (conditional blending instead of hard suppression)
    # Blend tau_pitch based on error magnitude, with safety guards
    apc_pitch_blend_enabled: bool = False  # enable conditional pitch blend
    apc_pitch_blend_startup_guard_steps: int = 100  # No blending for first N steps
    apc_pitch_blend_safe_pitch_rad: float = 0.15  # Safe pitch threshold for blending
    apc_pitch_blend_safe_pitch_rate_rad_s: float = 0.5  # Safe pitch rate for blending
    apc_pitch_blend_min_com_z: float = 0.27  # Minimum height for blending
    apc_pitch_blend_max_roll_rad: float = 0.15  # Maximum roll for blending
    apc_pitch_blend_deep_error_m: float = 0.12  # |e| > this → scale_deep
    apc_pitch_blend_mid_error_m: float = 0.08  # |e| > this → scale_mid
    apc_pitch_blend_soft_error_m: float = 0.05  # |e| > this → scale_soft
    apc_pitch_blend_scale_deep: float = 0.0  # tau_pitch * 0.0 (effectively off)
    apc_pitch_blend_scale_mid: float = 0.25  # tau_pitch * 0.25
    apc_pitch_blend_scale_soft: float = 0.5  # tau_pitch * 0.5
    apc_pitch_blend_scale_near: float = 1.0  # tau_pitch * 1.0 (no blend)

    # APCR1n Recenter Priority Torque Boost
    # Based on APCR1h with targeted fixes:
    # 1. Wheel damping override during RECENTER when it fights drift
    # 2. Position cap boost during safe RECENTER
    recenter_priority_enabled: bool = False
    recenter_priority_startup_guard_steps: int = 100
    vd_wheel_damping_recenter_override_enabled: bool = False
    vd_wheel_damping_recenter_scale: float = 0.30  # Reduce to 30% of baseline
    vd_wheel_damping_recenter_min_abs_nm: float = 0.50  # Minimum damping preserved
    vd_wheel_damping_preserve_if_opposes_drift: bool = True
    position_cap_recenter_boost_enabled: bool = False
    position_cap_normal_nm: float = 3.0  # Current APCR1h cap
    position_cap_recenter_nm: float = 5.0  # Boosted cap during RECENTER
    position_cap_emergency_nm: float = 6.0  # Emergency cap
    position_cap_ramp_steps: int = 50  # Gradual ramp to boosted cap
    recenter_priority_safe_min_com_z: float = 0.27
    recenter_priority_safe_roll_rad: float = 0.15
    recenter_priority_safe_pitch_rad: float = 0.15

    # APCR1nD: Direct support drift trigger (decoupled from APC)
    # Activates based on direct drift conditions without requiring APC state
    recenter_priority_direct_enabled: bool = False
    recenter_priority_direct_enter_m: float = 0.08  # m - enter threshold
    recenter_priority_direct_emergency_m: float = 0.12  # m - emergency threshold
    recenter_priority_direct_hard_m: float = 0.15  # m - hard safety threshold
    recenter_priority_direct_exit_m: float = 0.02  # m - exit threshold (hysteresis)

    # APCR1nD Tuned Variants: Band-limited drift control tuning parameters
    # Five opt-in variants (T1-T5) addressing band control failure modes:
    # - T1: Early entry
    # - T2: Hold outside band
    # - T3: Early entry + hold
    # - T4: Stronger authority
    # - T5: Band-limited balanced (recommended)
    apcr1nd_tuned_enabled: bool = False
    apcr1nd_tuned_variant_name: str = ""
    apcr1nd_soft_enter_m: float = 0.05
    apcr1nd_direct_enter_m: float = 0.06
    apcr1nd_desired_band_m: float = 0.08
    apcr1nd_hard_band_m: float = 0.10
    apcr1nd_emergency_band_m: float = 0.12
    apcr1nd_release_inner_m: float = 0.03
    apcr1nd_hold_outside_band: bool = False
    apcr1nd_converging_release_steps: int = 15
    apcr1nd_position_cap_normal_nm: float = 3.5
    apcr1nd_position_cap_soft_nm: float = 4.0
    apcr1nd_position_cap_desired_nm: float = 5.0
    apcr1nd_position_cap_hard_nm: float = 6.0
    apcr1nd_position_cap_emergency_nm: float = 7.0
    apcr1nd_damping_scale_normal: float = 1.0
    apcr1nd_damping_scale_soft: float = 0.70
    apcr1nd_damping_scale_desired: float = 0.40
    apcr1nd_damping_scale_hard: float = 0.20
    apcr1nd_damping_scale_emergency: float = 0.10
    apcr1nd_preserve_damping_if_helps: bool = True

    # ── K2 JAX Dedicated Default V1: Pitch-damping enhancement ────────────
    # Candidate E v2 (2026-06-30). Adds continuous pitch-rate-dependent wheel
    # damping during oscillations. Smoothstep-gated, height-transition-aware,
    # zero steady-state effect. Enabled by default in DEFAULT_V1 profile.
    enable_pitch_damping_boost: bool = False
    pitch_damping_boost_kd: float = 3.0                # Nm/(rad/s)
    pitch_damping_rate_threshold_low: float = 0.035     # rad/s (~2 deg/s)
    pitch_damping_rate_threshold_high: float = 0.262    # rad/s (~15 deg/s)
    pitch_damping_height_gate_enabled: bool = True

    # ── Drift Controller (K2 JAX dedicated) ──────────────────────────────────
    # Coordinated wheel-torque drift correction with continuous state-dependent
    # gating. Corrects sagittal velocity drift, heading/yaw drift, and provides
    # weak position return — all through smoothstep-gated wheel torques.
    # Zero effect when disabled. Hardware-compatible estimator interface.
    enable_drift_controller: bool = False
    drift_k_vel: float = 6.0             # Nm/(m/s) velocity damping gain
    drift_k_pos: float = 1.5             # Nm/m position return gain (intentionally weak)
    drift_k_heading: float = 3.0         # Nm/rad heading hold proportional gain
    drift_k_heading_rate: float = 0.8    # Nm/(rad/s) heading rate damping
    drift_push_damp_mult: float = 1.5    # max additional velocity damping during push-like states
    drift_max_tau: float = 5.0           # Nm per-wheel max drift torque (smooth tanh bound)
    drift_hgate_low: float = 0.03        # CoM z-vel (m/s) below which height_gate ≈ 1.0
    drift_hgate_high: float = 0.15       # CoM z-vel (m/s) above which height_gate ≈ 0.0
    drift_pgate_low: float = 0.15        # drift distance (m) below which pos_gate ≈ 0.0
    drift_pgate_high: float = 0.80       # drift distance (m) above which pos_gate ≈ 1.0

    # ── Posture homing (F5/F12): restore hip_roll/hip_yaw to nominal q_ref when
    # settled so legs un-splay after a push. Stability-gated in the JAX step. ──
    enable_posture_homing: bool = False
    homing_kp_hip_roll: float = 0.0      # Nm/rad hip_roll restoring (V3 posture kp=0)
    homing_kp_hip_yaw: float = 0.0       # Nm/rad hip_yaw restoring boost
    homing_max_tau: float = 4.0          # Nm per-joint smooth tanh bound

    # ── Anchor position integral (V3_ANCHOR): PI position hold ──────────────
    # The P-only position loop parks the robot bias/k_position from home
    # (equilibrium-pitch torque bias exceeds the ABS trim cap). The integral
    # supplies the bias torque so the standing point converges to the latched
    # home. Adaptation freezes (continuous gates) when tilted, in bad contact,
    # or during commanded height transitions. ki = 0 disables (old behavior).
    anchor_position_ki: float = 0.0            # Nm/(m·s) integral gain
    anchor_integral_cap_nm: float = 0.0        # Nm anti-windup clamp
    anchor_integral_leak_per_step: float = 0.0  # per-step leak (forgetting)
    anchor_kvel_boost_scale: float = 0.0       # extra damping scale at quiet stance
    # (gated in JAX step by proximity/stability/height/quiet-EMA gates)
    anchor_leash_m: float = 0.0                # RESERVED param slot (leash mechanism
    # removed — it acted as a phase-lagged relay; superseded by the proximity gate)
    anchor_slew_m_s: float = 0.0               # RESERVED param slot (unused)
    anchor_kp_pitch_soft: float = 0.0          # pitch kp during recovery (0=off→keep 50);
    # scheduled stiff(50)→soft when displaced via the quiet-stance envelope

    # ── Heading hip-yaw stabilizer (low-authority soft heading impedance) ──
    # Acts on hip-yaw joints [1,6] with very low authority smooth bounded torque.
    # Corrects slow yaw drift without wheel differential. Yields to poor
    # stability, fast height motion, and hip-yaw divergence.
    enable_heading_hip_yaw: bool = False
    heading_hy_kp: float = 0.15          # Nm/rad — very low proportional gain
    heading_hy_kd: float = 0.05          # Nm/(rad/s) — mild damping
    heading_hy_max_tau: float = 0.8      # Nm per-joint smooth tanh bound

    # ── Anti-twist damping (reduce excessive hip-yaw divergence) ──────────
    # Applies opposing torques to hip-yaw joints [1,6] to damp left/right
    # asymmetry. Mild gains, smooth bounds. Does not lock legs.
    enable_anti_twist: bool = False
    anti_twist_kp: float = 0.3           # Nm/rad anti-twist proportional
    anti_twist_kd: float = 0.1           # Nm/(rad/s) anti-twist damping
    anti_twist_max_tau: float = 0.6      # Nm per-joint smooth tanh bound

    # ── Hip-yaw mean centering (weak return toward neutral) ────────────────
    # Gently brings both legs back toward zero-mean after disturbances.
    # Very weak authority. Yields under poor balance, divergence, and height motion.
    hy_mean_center_kp: float = 0.5        # Nm/rad weak centering proportional
    hy_mean_center_max_tau: float = 0.4   # Nm per-joint smooth tanh bound

    # ── Anti-twist divergence guard thresholds (V5 parameterization) ───────
    # Progressive kp boost when hip-yaw divergence enters the guard region.
    # guard_start: divergence at which boost begins (V3: 0.22, V4: 0.18)
    # guard_strong: divergence at which boost saturates (V3: 0.32, V4: 0.30)
    # guard_boost_max: maximum kp multiplier (V3: 3.5, V4: 5.0)
    anti_twist_guard_start_rad: float = 0.22
    anti_twist_guard_strong_rad: float = 0.32
    anti_twist_guard_boost_max: float = 3.5
    # V5 two-layer emergency guard: separate tanh cap for guard extra torque
    anti_twist_emergency_max_tau: float = 0.25  # Nm per-joint, separate from base

    # ── Heading twist yield gate thresholds (V5 parameterization) ──────────
    # Reduces heading authority as hip-yaw divergence grows.
    # yield_start: divergence at which heading yield begins (V3: 0.35 disabled, V4: 0.18)
    # yield_zero: divergence at which heading is fully suppressed (V3/V4: 0.35)
    # When yield_start >= yield_zero, the yield gate is disabled (always 1.0).
    heading_twist_yield_start_rad: float = 0.35
    heading_twist_yield_zero_rad: float = 0.35

    # ── Dynamic q_ref blend alpha (V5 parameterization) ─────────────────────
    # Fraction of dynamic (two-point-smooth) q_ref vs static equilibrium anchor.
    # Only applies to multi-segment dynamic cycles. V3: 0.40, V4: 0.60
    dynamic_q_ref_blend_alpha: float = 0.40

    # ── Split height gates for drift controller ───────────────────────────
    # Per-component height motion sensitivity. Wider gates = more active during
    # height transitions. Narrower gates = more suppressed.
    drift_hgate_vel_low: float = 0.05        # CoM z-vel (m/s) below which height_gate_vel ≈ 1.0
    drift_hgate_vel_high: float = 0.25       # CoM z-vel (m/s) above which height_gate_vel ≈ 0.0
    drift_hgate_heading_low: float = 0.02    # CoM z-vel (m/s) below which height_gate_heading ≈ 1.0
    drift_hgate_heading_high: float = 0.10   # CoM z-vel (m/s) above which height_gate_heading ≈ 0.0

    # ── Height trajectory speed (transition duration control) ─────────────
    height_transition_duration_s: float = 8.0  # Target duration for full height ramp

    # T6F Architecture Fix: Budget Cap Raise
    # Conditionally raises upstream max_position_tau cap during safe high-height emergency recenter
    # Addresses upstream 4.0 Nm clipping that prevents tuned cap authority from reaching wheels
    arch_fix_enabled: bool = False
    arch_fix_type: str = ""  # "budget_cap_raise", "emergency_bypass", etc.
    arch_fix_height_threshold_m: float = 0.45  # Only active at heights >= this
    arch_fix_hard_max_position_tau: float = 6.5  # Raised cap for hard band
    arch_fix_emergency_max_position_tau: float = 7.0  # Raised cap for emergency band

    # T6F Sign Fix: Enhanced Damping Override and Pitch Suppression
    # Conditionally disables fighting damping/pitch terms during arch_fix to preserve sign correctness
    # Problem: wheel velocity damping and pitch torque can fight position torque during high-authority recenter
    # Solution: disable damping when it fights tau_position, suppress pitch during large error
    sign_fix_enabled: bool = False
    sign_fix_disable_fighting_damping_during_arch_fix: bool = False
    sign_fix_suppress_pitch_during_arch_fix: bool = False
    sign_fix_pitch_error_threshold_m: float = 0.10  # m - suppress pitch when |error| > this AND arch_fix active
    sign_fix_suppress_pitch_rate: bool = False  # Future: also suppress pitch rate term

    # T6H Soft Blend Arch Fix: Soft modulation instead of hard suppression
    # Reduces pitch/damping authority by 50% (not 100%) during arch_fix
    # Preserves partial stabilization while reducing fighting terms
    t6h_enabled: bool = False
    t6h_soft_pitch_blend_factor: float = 0.50  # Pitch scale during blend (0.5 = 50% reduction)
    t6h_soft_damping_blend_factor: float = 0.50  # Damping scale during blend
    t6h_pitch_error_threshold_m: float = 0.10  # Apply blend when |error| > this
    t6h_pitch_safety_threshold_deg: float = 10.0  # Restore full pitch if |pitch| > this
    t6h_wheel_velocity_safety_threshold_rad_s: float = 7.0  # Restore full damping if |wheel_vel| > this

    # T6I Phase-Aware Release: Gradual cap decay when error converging
    # Detects convergence and releases high authority smoothly
    # Preserves full pitch/damping authority (no suppression)
    t6i_enabled: bool = False
    t6i_convergence_window_steps: int = 5  # Steps to track for convergence detection
    t6i_convergence_threshold_m: float = 0.12  # Error must be below this
    t6i_convergence_trend_threshold_m: float = 0.03  # Max error change to be converging
    t6i_cap_decay_rate_nm_per_step: float = 0.10  # Cap decay rate when converging
    t6i_cap_min_nm: float = 4.0  # Min cap (normal authority)
    t6i_max_cap_delta_per_step_nm: float = 0.30  # Rate limit for cap transitions

    # T6J Centering Bias Trim: slow bounded support-centering correction
    # Adds a small bias trim on top of T6I without suppressing pitch or damping
    t6j_bias_trim_enabled: bool = False
    t6j_bias_trim_window_steps: int = 200
    t6j_bias_trim_enter_threshold_m: float = 0.04
    t6j_bias_trim_exit_threshold_m: float = 0.015
    t6j_bias_trim_max_tau_nm: float = 0.35
    t6j_bias_trim_rate_nm_per_step: float = 0.01
    t6j_bias_trim_decay_rate_nm_per_step: float = 0.02
    t6j_bias_trim_only_when_upright: bool = True
    t6j_bias_trim_only_when_contact_stable: bool = True
    t6j_bias_trim_disable_if_pitch_gt_deg: float = 8.0
    t6j_bias_trim_disable_if_roll_gt_deg: float = 3.0
    t6j_bias_trim_disable_if_wheel_vel_gt_rad_s: float = 7.0
    t6j_bias_trim_disable_if_abs_error_gt_m: float = 0.22

    # Adaptive Centering Bias Trim: proportional, height-aware, guarded trim
    # Replaces the bang-bang T6J trim with smooth proportional authority
    # when adaptive_support_centering_trim profile is selected.
    adaptive_bias_trim_enabled: bool = False
    adaptive_bias_trim_replace_t6j: bool = True  # True = adaptive replaces T6J entirely
    adaptive_bias_window_steps: int = 300
    adaptive_bias_fast_window_steps: int = 100
    adaptive_bias_enter_threshold_m: float = 0.035
    adaptive_bias_exit_threshold_m: float = 0.012
    adaptive_bias_relief_hysteresis_m: float = 0.005
    adaptive_bias_k_tau_per_m: float = 5.0
    adaptive_bias_max_tau_low_nm: float = 0.35
    adaptive_bias_max_tau_high_nm: float = 0.50
    adaptive_bias_max_tau_extreme_nm: float = 0.55
    adaptive_bias_height_low_m: float = 0.38
    adaptive_bias_height_high_m: float = 0.48
    adaptive_bias_height_extreme_m: float = 0.52
    adaptive_bias_rate_nm_per_step: float = 0.006
    adaptive_bias_fast_rate_nm_per_step: float = 0.012
    adaptive_bias_decay_rate_nm_per_step: float = 0.018
    adaptive_bias_only_when_upright: bool = True
    adaptive_bias_only_when_contact_stable: bool = True
    adaptive_bias_disable_if_pitch_gt_deg: float = 12.0
    adaptive_bias_disable_if_roll_gt_deg: float = 5.0
    adaptive_bias_disable_if_abs_error_gt_m: float = 0.24
    adaptive_bias_disable_if_hip_yaw_gt_rad: float = 0.25
    adaptive_bias_zero_crossing_guard_enabled: bool = True
    adaptive_bias_zero_crossing_window_steps: int = 500
    adaptive_bias_zero_crossing_limit: int = 8
    adaptive_bias_zero_crossing_max_scale: float = 0.5
    adaptive_bias_sign_reversal_hold_steps: int = 100

    # Zero-Crossing Support Recenter: hysteresis recenter that forces drift to cross zero
    # Key difference from adaptive_bias_trim: holds correction until drift crosses to
    # opposite side, not just until near-zero. Enforces symmetric oscillation.
    enable_zero_crossing_recenter: bool = False
    zc_replace_adaptive: bool = False  # True = replace adaptive_bias_trim with ZC

    # Entry/exit thresholds
    zc_enter_m: float = 0.08           # Enter recenter when |e| > this
    zc_exit_m: float = 0.025           # Exit recenter when |e| <= this (with dwell)
    zc_cross_target_m: float = 0.02    # Target overshoot into opposite side
    zc_near_zero_band_m: float = 0.03  # Error considered "near zero"

    # Hold duration constraints
    zc_min_hold_steps: int = 50        # Minimum hold before considering release
    zc_max_hold_steps: int = 600       # Force exit after this many steps

    # Torque authority
    zc_base_tau_nm: float = 0.20       # Base correction torque
    zc_max_tau_nm: float = 0.65        # Maximum correction torque
    zc_rate_nm_per_step: float = 0.01  # Rate limit: increase toward target
    zc_decay_nm_per_step: float = 0.02 # Decay rate: return to zero

    # Error-proportional gain
    zc_error_gain_nm_per_m: float = 3.0  # Nm per meter of error

    # Optional velocity damping
    zc_velocity_gain: float = 0.0      # Damping term (0.0 = disabled)

    # Safety gates (absolute disable conditions)
    zc_disable_if_abs_error_gt_m: float = 0.25
    zc_disable_if_pitch_gt_deg: float = 12.0
    zc_disable_if_roll_gt_deg: float = 5.0
    zc_disable_if_hip_yaw_gt_rad: float = 0.25

    # Dwell time for exit (converging signal)
    zc_dwell_steps_for_exit: int = 30
    zc_dwell_target_within_m: float = 0.015

    # Early Zero-Crossing Support Recenter: exits at zero crossing, not opposite side
    # Key differences from ZC:
    # - Entry at 0.05 m (earlier) vs 0.08 m
    # - Exit at e <= 0 (not -0.02)
    # - No opposite-side target required
    # - Immediate decay after zero crossing
    enable_early_zero_crossing_recenter: bool = False
    ezc_replace_adaptive: bool = False  # True = replace adaptive_bias_trim with EZC
    ezc_replace_zc: bool = False  # True = replace old ZC with EZC

    # Entry/exit thresholds
    ezc_enter_m: float = 0.05           # Enter recenter when |e| > this
    ezc_exit_at_zero: bool = True       # Exit when e <= 0 (or e >= 0)
    ezc_zero_dwell_steps: int = 3       # Dwell at zero before decay starts
    ezc_reentry_m: float = 0.05         # Re-enter when |e| > this again

    # Hold duration constraints
    ezc_min_hold_steps: int = 0        # No minimum hold (exit at zero)
    ezc_max_hold_steps: int = 500      # Force exit after this many steps

    # Torque authority
    ezc_base_tau_nm: float = 0.18       # Base correction torque
    ezc_max_tau_nm: float = 0.55        # Maximum correction torque
    ezc_rate_nm_per_step: float = 0.012  # Rate limit: increase toward target
    ezc_decay_nm_per_step: float = 0.025  # Decay rate: return to zero

    # Error-proportional gain
    ezc_error_gain_nm_per_m: float = 3.0  # Nm per meter of error

    # Anti-rebound hold: keep decaying correction after zero crossing
    # Key fix for EZC_FAILURE_EXIT_TOO_EARLY_REBOUND
    # After crossing zero, keep a small decaying correction to prevent immediate rebound
    ezc_antirebound_enabled: bool = False  # Enable anti-rebound decay
    ezc_antirebound_decay_steps: int = 30  # Steps over which to decay after zero crossing
    ezc_antirebound_initial_ratio: float = 0.50  # Start at 50% of current tau

    # Safety gates (absolute disable conditions)
    ezc_disable_if_abs_error_gt_m: float = 0.25
    ezc_disable_if_pitch_gt_deg: float = 12.0
    ezc_disable_if_roll_gt_deg: float = 5.0
    ezc_disable_if_hip_yaw_gt_rad: float = 0.25

    # Pitch bias DC compensation (Phase 7 mechanism)
    # Removes slow residual tau_pitch DC component during stable upright posture.
    # Does NOT zero tau_pitch; does NOT suppress dynamic pitch correction.
    # See docs/validation/pitch_bias_compensated_zc_design.md for the full design.
    pitch_bias_comp_enabled: bool = False                         # Master enable
    pitch_bias_window_steps: int = 300                            # EMA window for tau_pitch estimate
    pitch_bias_max_comp_nm: float = 0.60                          # Hard cap on compensation (Nm)
    pitch_bias_comp_rate_nm_per_step: float = 0.005               # Rate limit growing comp
    pitch_bias_decay_rate_nm_per_step: float = 0.012              # Decay rate when gate fails
    pitch_bias_only_when_abs_pitch_lt_deg: float = 2.0            # Estimation gate: pitch upright
    pitch_bias_only_when_abs_error_lt_m: float = 0.12             # Estimation gate: drift small
    pitch_bias_disable_if_pitch_gt_deg: float = 12.0              # Hard safety disable on pitch
    pitch_bias_disable_if_roll_gt_deg: float = 5.0                # Hard safety disable on roll
    pitch_bias_disable_if_contact_unstable: bool = True           # Hard safety disable on contact
    pitch_bias_disable_if_height_lt_m: float = 0.25               # Hard safety disable on low height
    pitch_bias_gate_abs_error_soft_m: float = 0.12                # Soft gate (apply allowed)
    pitch_bias_gate_abs_error_hard_m: float = 0.20                # Hard gate (apply blocked)

    # ---- Notch / band-stop filter for 2.5 Hz WIP mode (K candidate family) ----
    # A causal IIR biquad notch filter centred on the observed 2.5 Hz WIP mode.
    # Only active at tall heights via the height gate.  Opt-in only — every
    # existing profile has enable_wip_notch_filter=False and is unchanged.
    enable_wip_notch_filter: bool = False

    # Target signal(s) to filter.  Allowed values:
    #   "pitch_rate"                     — pitch_rate in tau_pitch_rate term
    #   "wheel_velocity"                 — wheel_vel in tau_wheel_vel term
    #   "pitch_rate_and_wheel_velocity"  — both
    #   "support_velocity"               — support_vel in support velocity damping
    #   "all_damping_signals"            — pitch_rate + wheel_velocity + support_vel
    wip_notch_target_signal: str = "pitch_rate"

    # Filter centre frequency (Hz).  Telemetry shows ~2.4–2.5 Hz for pitch_rate.
    wip_notch_center_hz: float = 2.5

    # Quality factor (Q).  Higher Q = narrower notch.
    # Recommended: 4–8 for 100 Hz sample rate, 2.5 Hz centre.
    wip_notch_q: float = 6.0

    # Sample rate (Hz).  0 means auto-derive from controller dt.  100 Hz nominal.
    wip_notch_fs_hz: float = 0.0  # 0 = auto

    # Height gate for filter activation (smooth Hermite interpolation).
    # Below z_start → filter blend = 0 (fully raw).
    # Above z_full → filter blend = blend (fully filtered if blend=1).
    wip_notch_height_gate_start_m: float = 0.42
    wip_notch_height_gate_full_m: float = 0.48

    # Enable the height gate.  When False, the filter is always at full blend
    # (when enable_wip_notch_filter is True).
    wip_notch_gate_enabled: bool = True

    # Filter blend ratio.  0.0 = fully raw; 1.0 = fully filtered.
    # Intermediate values allow partial blending.
    wip_notch_filter_blend: float = 1.0

    # ---- Filter topology selection (K notch/filter sweep, audit-only) ----
    # Allowed values:
    #   "biquad_notch"          — current K1 biquad notch (default, unchanged)
    #   "first_order_lowpass"   — first-order IIR low-pass on pitch rate
    #   "notch_disabled"        — diagnostic: disable filter entirely
    # Audit-only.  K1 is "biquad_notch" and must remain unchanged.
    wip_notch_filter_type: str = "biquad_notch"

    # Cutoff frequency for first_order_lowpass filter type (Hz).
    # Ignored for biquad_notch filter type.
    wip_lowpass_cutoff_hz: float = 3.0

    # ---- L family: Coordinated sagittal state feedback (Phase 3) ---- #
    # When enabled, a coordinated state-feedback term is added to the wheel
    # torque AFTER the normal sagittal torque computation to synchronize
    # the pitch, support, and rate contributions.
    enable_coordinated_sagittal_feedback: bool = False
    # Selects the feedback gain function:
    #   "L1_low_freq"           — conservative low-frequency state feedback
    #   "L2_phase_lead"         — phase-lead compensation on pitch rate
    #   "L3_pitch_ref_stabilization" — pitch reference modulation
    #   "N1_mild_phase_lead"    — mild phase-lead for damping diagnostic
    coordinated_feedback_kind: str = "none"

    # ---- LR family: Replacement coordinated sagittal feedback (Phase 2) ---- #
    # Unlike the L family (which ADDS feedback on top of K1's existing terms),
    # LR REPLACES the sum-of-independent-torques with a single coordinated
    # feedback term. This avoids the torque double-counting that caused L1/L2/L3
    # failures (4-5 Nm RMS added to K1's existing 5-8 Nm).
    # Preserves equilibrium/feedforward path and notch filter.
    # Disabled by default — opt-in via LR profiles.
    enable_lr_replacement_feedback: bool = False
    # Selects the replacement gain function:
    #   "LR1_low_freq"              — LR coordinated low-frequency state feedback
    #   "LR2_phase_lead"            — LR with phase-lead compensation
    #   "LR3_pitch_ref_stabilized"  — LR with pitch reference stabilization
    lr_replacement_kind: str = "none"

    # ---- LP family: Priority sagittal allocator — pitch-first support-residual ---- #
    # Architectural alternative to LR/LRS coordinated feedback. Instead of a
    # single equal-priority sum, LP computes pitch stabilization first and
    # allocates support-centering torque only from residual safe authority.
    # Support correction is gated by pitch state safety, saturation headroom,
    # direction consistency, and slew limits. Preserves K1 EQ/FF baseline.
    # Disabled by default — opt-in via LP profiles.
    enable_lp_priority_allocator: bool = False
    # Selects the priority-allocation variant:
    #   "LP1_pitch_first_support_residual" — conservative pitch, soft support
    #   "LP2_pitch_strong_support_soft"    — stronger pitch-rate, softer support
    #   "LP3_support_recenter_when_safe"   — support only after pitch settles
    lp_allocator_kind: str = "none"

    # ---- M family: Body-yaw/wheel-yaw correct-actuator fix (Phase 4) ---- #
    # When enabled, adds body-yaw correction through differential wheel
    # velocity with support-aware gating.
    enable_body_yaw_wheel_stabilization: bool = False
    wheel_yaw_kp: float = 0.5
    wheel_yaw_kd: float = 0.1
    wheel_yaw_max_torque: float = 1.5
    wheel_yaw_height_gate_start_m: float = 0.34
    wheel_yaw_height_gate_full_m: float = 0.42
    wheel_yaw_activation_threshold_rad: float = 0.05
    wheel_yaw_support_gate_enabled: bool = True
    wheel_yaw_support_error_threshold_m: float = 0.15
    wheel_yaw_support_rate_threshold_mps: float = 0.05

    # N1 micro-sweep parameters (Phase 5)
    # Controls the height-scheduled mild phase-lead damping.
    # Only used when enable_coordinated_sagittal_feedback=True and
    # coordinated_feedback_kind="N1_mild_phase_lead".
    n1_rate_low: float = 0.3    # k_rate at low height (0.30 m)
    n1_rate_high: float = 0.5   # k_rate at high height (0.48 m)
    n1_lead_low: float = 0.02   # k_lead at low height (0.30 m)
    n1_lead_high: float = 0.04  # k_lead at high height (0.48 m)

    def is_active_for_variant(self, variant_name: str | None) -> bool:
        return variant_name is not None and variant_name in self.applies_to_variants

    def max_position_tau_for_variant(self, variant_name: str | None, baseline_max_position_tau: float) -> float:
        if not self.is_active_for_variant(variant_name):
            return baseline_max_position_tau
        for candidate_name, max_position_tau in self.position_tau_cap_by_variant:
            if candidate_name == variant_name:
                return float(max_position_tau)
        # If continuous_max_position_tau is enabled, use the low_max value
        # (the continuous scheduling will interpolate based on height)
        if self.continuous_max_position_tau:
            return float(self.max_position_tau_low_max)
        return baseline_max_position_tau * self.position_tau_cap_scale


BASELINE_AUTHORITY_SCHEDULE = SagittalAuthoritySchedule()

# Phase 6 Joint Low-Height Sagittal-Yaw Fix Profiles
# Evidence-based schedules addressing position_torque_cap_saturation,
# insufficient_velocity_damping, and support_velocity_underdamped failure modes

JOINT_FIX_J1_SUPPORT_CAP = SagittalAuthoritySchedule(
    profile_name="J1_support_cap",
    continuous_k_position=True,
    k_position_nominal=40.0,
    k_position_low_max=80.0,
    k_position_z_low=0.300,
    k_position_z_high=0.393,
    continuous_max_position_tau=True,
    max_position_tau_nominal=3.0,
    max_position_tau_low_max=6.0,
    # k_velocity unchanged (15.0 baseline)
)

JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING = SagittalAuthoritySchedule(
    profile_name="J2_support_cap_moderate_damping",
    continuous_k_position=True,
    k_position_nominal=40.0,
    k_position_low_max=80.0,
    k_position_z_low=0.300,
    k_position_z_high=0.393,
    continuous_max_position_tau=True,
    max_position_tau_nominal=3.0,
    max_position_tau_low_max=6.0,
    continuous_k_velocity=True,
    k_velocity_nominal=15.0,
    k_velocity_low_max=25.0,
)

JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING = SagittalAuthoritySchedule(
    profile_name="J3_support_cap_strong_damping",
    continuous_k_position=True,
    k_position_nominal=40.0,
    k_position_low_max=80.0,
    k_position_z_low=0.300,
    k_position_z_high=0.393,
    continuous_max_position_tau=True,
    max_position_tau_nominal=3.0,
    max_position_tau_low_max=6.0,
    continuous_k_velocity=True,
    k_velocity_nominal=15.0,
    k_velocity_low_max=30.0,
)

# Pitch-Safe Candidates (J2a-J2d family)
# Designed after audit showing position_authority_induces_pitch_overshoot
# Target: preserve support/hip-yaw improvements while keeping pitch < 0.10 rad

PITCH_SAFE_J2A_CONSERVATIVE = SagittalAuthoritySchedule(
    profile_name="J2a_conservative_position_cap",
    continuous_k_position=True,
    k_position_nominal=40.0,
    k_position_low_max=60.0,  # 50% increase vs 100% in J1-J3
    k_position_z_low=0.300,
    k_position_z_high=0.393,
    continuous_max_position_tau=True,
    max_position_tau_nominal=3.0,
    max_position_tau_low_max=4.5,  # 50% increase vs 100% in J1-J3
    continuous_k_velocity=True,
    k_velocity_nominal=15.0,
    k_velocity_low_max=22.0,  # 47% increase, moderate damping
)

PITCH_SAFE_J2B_BALANCED = SagittalAuthoritySchedule(
    profile_name="J2b_balanced_authority",
    continuous_k_position=True,
    k_position_nominal=40.0,
    k_position_low_max=65.0,  # 63% increase
    k_position_z_low=0.300,
    k_position_z_high=0.393,
    continuous_max_position_tau=True,
    max_position_tau_nominal=3.0,
    max_position_tau_low_max=5.0,  # 67% increase
    continuous_k_velocity=True,
    k_velocity_nominal=15.0,
    k_velocity_low_max=25.0,  # matches J2 proven effective
)

PITCH_SAFE_J2C_VELOCITY_PRIORITY = SagittalAuthoritySchedule(
    profile_name="J2c_velocity_damping_priority",
    continuous_k_position=True,
    k_position_nominal=40.0,
    k_position_low_max=60.0,  # conservative position
    k_position_z_low=0.300,
    k_position_z_high=0.393,
    continuous_max_position_tau=True,
    max_position_tau_nominal=3.0,
    max_position_tau_low_max=4.5,  # conservative tau cap
    continuous_k_velocity=True,
    k_velocity_nominal=15.0,
    k_velocity_low_max=28.0,  # aggressive damping to counter pitch
)

PITCH_SAFE_J2D_TAU_CAP_PRIORITY = SagittalAuthoritySchedule(
    profile_name="J2d_torque_cap_priority",
    continuous_k_position=True,
    k_position_nominal=40.0,
    k_position_low_max=70.0,  # 75% increase, higher stiffness
    k_position_z_low=0.300,
    k_position_z_high=0.393,
    continuous_max_position_tau=True,
    max_position_tau_nominal=3.0,
    max_position_tau_low_max=4.5,  # 50% increase, caps peak torque
    continuous_k_velocity=True,
    k_velocity_nominal=15.0,
    k_velocity_low_max=25.0,  # J2-level damping
)

# APCR1f Adaptive Fast Response with Phase Brake Profile
# Key differences from APCR1e:
# - Earlier intervention at 0.035m vs 0.05m
# - Faster rate limit 0.55 Nm/step vs 0.35 Nm/step
# - Higher max_tau 1.40 Nm vs 1.20 Nm
# - Phase brake when error returning toward zero
# - Boost when error growing 3+ consecutive steps
APCR1F_FAST_RESPONSE_PHASE_BRAKE = SagittalAuthoritySchedule(
    profile_name="APCR1f_adaptive_fast_response_phase_brake",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    # Proportional soft band mode enabled
    apc_proportional_soft_band_mode=True,
    apc_soft_enter_m=0.035,  # Earlier entry than APCR1e's 0.05
    apc_inner_exit_m=0.015,  # Earlier exit
    apc_outer_enter_m=0.10,  # Full torque threshold
    # Velocity decay
    apc_velocity_decay_enabled=True,
    apc_velocity_decay_factor=0.5,
    # APCR1f fast response parameters
    apc_fast_response_enabled=True,
    apc_phase_brake_enabled=True,
    apc_phase_brake_threshold_m=0.08,
    apc_phase_brake_damping_factor=0.6,
    apc_boost_rate_per_step=0.25,
    apc_decay_rate_per_step=0.45,
    apc_increasing_error_threshold_steps=3,
    apc_increasing_error_boost_factor=0.3,
    apc_fast_response_inner_deadband_m=0.015,
    apc_fast_response_soft_enter_m=0.035,
    apc_fast_response_desired_band_m=0.08,
    apc_fast_response_full_torque_m=0.10,
    apc_fast_response_emergency_m=0.12,
    apc_fast_response_base_tau=0.45,
    apc_fast_response_max_tau=1.40,
    apc_fast_response_boost_tau_max=0.95,
    apc_fast_response_startup_boost_max_tau=1.20,
    apc_fast_response_max_rate_per_step=0.55,
    apc_fast_response_smooth_alpha=0.18,
    apc_fast_response_no_improvement_window=5,
    # Recovery gate mode enabled
    active_pitch_crossing_recovery_gate_mode=True,
)

# APCR1g Predictive Fast Response with Phase Brake Profile
# Key differences from APCR1f:
# - Predictive error: e_pred = e + lead_time_s * e_dot
# - Earlier activation when predicted error exceeds threshold
# - Predictive boost when predicted error indicates future overshoot
# - Stronger phase brake with two thresholds
# - Higher max_tau (1.55 vs 1.40), faster rate (0.70 vs 0.55)
APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE = SagittalAuthoritySchedule(
    profile_name="APCR1g_predictive_fast_response_phase_brake",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    # Proportional soft band mode enabled
    apc_proportional_soft_band_mode=True,
    apc_soft_enter_m=0.030,  # Earlier entry than APCR1f's 0.035
    apc_inner_exit_m=0.012,  # Earlier exit
    apc_outer_enter_m=0.095,  # Full torque threshold
    # Velocity decay
    apc_velocity_decay_enabled=True,
    apc_velocity_decay_factor=0.5,
    # APCR1g predictive fast response parameters
    apc_predictive_enabled=True,
    apc_lead_time_s=0.10,  # Predict 100ms ahead
    apc_predicted_enter_m=0.07,  # Activate when abs_pred > this AND moving_away
    apc_predicted_full_response_m=0.10,  # Boost authority when abs_pred > this
    apc_predicted_emergency_m=0.12,  # Emergency mode when abs_pred > this
    apc_predictive_inner_deadband_m=0.012,
    apc_predictive_soft_enter_m=0.030,
    apc_predictive_desired_band_m=0.075,
    apc_predictive_full_torque_m=0.095,
    apc_predictive_emergency_error_m=0.115,
    apc_predictive_base_tau=0.45,
    apc_predictive_max_tau=1.55,  # Higher than APCR1f's 1.40
    apc_predictive_boost_tau_max=1.10,  # Higher than APCR1f's 0.95
    apc_predictive_startup_boost_max_tau=1.25,  # Higher than APCR1f's 1.20
    apc_predictive_max_rate_per_step=0.70,  # Faster than APCR1f's 0.55
    apc_predictive_boost_rate_per_step=0.35,
    apc_predictive_decay_rate_per_step=0.55,
    apc_predictive_smooth_alpha=0.22,  # More responsive than APCR1f's 0.18
    apc_predictive_no_improvement_window=4,  # Faster than APCR1f's 5
    apc_predictive_increasing_error_threshold_steps=2,  # Faster than APCR1f's 3
    apc_predictive_increasing_error_boost_factor=0.35,  # Higher than APCR1f's 0.30
    apc_predictive_phase_brake_enabled=True,
    apc_predictive_phase_brake_threshold_m=0.075,
    apc_predictive_phase_brake_strong_threshold_m=0.050,  # New: strong brake threshold
    apc_predictive_phase_brake_factor=0.55,  # Stronger than APCR1f's 0.60
    apc_predictive_phase_brake_strong_factor=0.35,  # New: strong brake factor
    apc_predictive_disable_vd_when_abs_e_gt=0.10,
    apc_predictive_disable_vd_during_startup=True,
    # Enable active pitch crossing for APCR1g
    enable_active_pitch_crossing=True,
    # Recovery gate mode enabled
    active_pitch_crossing_recovery_gate_mode=True,
)

# APCR1h Support Drift Priority with Fast Recenter Profile
# Key differences from APCR1f:
# - Based on APCR1f (correct torque sign), NOT APCR1g (wrong torque sign)
# - Drift priority override when abs_error > 0.08 AND moving away
# - Emergency clamp when abs_error > 0.12
# - Higher max_tau (1.65 vs 1.40), faster rate (0.85 vs 0.55)
# - Phase brake disabled when drift priority active
# - Allow higher wheel velocity for support recovery
# - Startup boost higher (1.60 vs 1.20)
APCR1H_SUPPORT_DRIFT_PRIORITY = SagittalAuthoritySchedule(
    profile_name="APCR1h_support_drift_priority_fast_recenter",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    # Proportional soft band mode enabled
    apc_proportional_soft_band_mode=True,
    apc_soft_enter_m=0.030,  # Earlier entry than APCR1f's 0.035
    apc_inner_exit_m=0.015,  # Earlier exit
    apc_outer_enter_m=0.095,  # Full torque threshold
    # Velocity decay
    apc_velocity_decay_enabled=True,
    apc_velocity_decay_factor=0.5,
    # APCR1f fast response parameters (CORRECT torque sign)
    apc_fast_response_enabled=True,
    apc_phase_brake_enabled=True,
    apc_phase_brake_threshold_m=0.08,
    apc_phase_brake_damping_factor=0.6,
    apc_boost_rate_per_step=0.25,
    apc_decay_rate_per_step=0.45,
    apc_increasing_error_threshold_steps=3,
    apc_increasing_error_boost_factor=0.3,
    apc_fast_response_inner_deadband_m=0.015,
    apc_fast_response_soft_enter_m=0.030,
    apc_fast_response_desired_band_m=0.08,
    apc_fast_response_full_torque_m=0.095,
    apc_fast_response_emergency_m=0.12,
    apc_fast_response_base_tau=0.45,
    apc_fast_response_max_tau=1.65,  # Higher than APCR1f's 1.40
    apc_fast_response_boost_tau_max=1.20,  # Higher than APCR1f's 0.95
    apc_fast_response_startup_boost_max_tau=1.60,  # Higher than APCR1f's 1.20
    apc_fast_response_max_rate_per_step=0.85,  # Faster than APCR1f's 0.55
    apc_fast_response_smooth_alpha=0.18,
    apc_fast_response_no_improvement_window=5,
    # Recovery gate mode enabled
    active_pitch_crossing_recovery_gate_mode=True,
    # APCR1h: Drift priority override parameters
    apc_drift_priority_enabled=True,
    apc_drift_priority_enter_m=0.08,  # Drift priority activates at this threshold
    apc_drift_priority_emergency_m=0.12,  # Emergency clamp threshold
    apc_drift_priority_hard_m=0.15,  # Hard safety threshold
    apc_drift_priority_base_tau=0.45,  # Same as APCR1f
    apc_drift_priority_normal_max_tau=1.40,  # Normal max (same as APCR1f)
    apc_drift_priority_drift_priority_max_tau=1.65,  # Drift priority max
    apc_drift_priority_emergency_max_tau=1.85,  # Emergency clamp max
    apc_drift_priority_startup_max_tau=1.60,  # Startup boost max
    apc_drift_priority_normal_rate=0.55,  # Normal rate (same as APCR1f)
    apc_drift_priority_drift_priority_rate=0.85,  # Drift priority rate
    apc_drift_priority_emergency_rate=1.00,  # Emergency rate
    apc_drift_priority_decay_rate=0.55,  # Decay rate
    apc_drift_priority_phase_brake_disable_threshold_m=0.10,  # Disable phase brake above this
)

# APCR1i Support Hysteresis Recenter Profile
# Symmetric hysteresis state machine that holds recenter direction
# until error reaches inner band or crosses to opposite side
# Key differences from APCR1h:
# - Full symmetric hysteresis state machine (NEUTRAL, RECENTER_FROM_POSITIVE, RECENTER_FROM_NEGATIVE, HOLD_THROUGH_ZERO)
# - Does NOT exit recenter when e_dot reverses while |e| > inner_exit_m
# - Holds direction through zero crossing until inside inner band
# - Phase brake disabled while outside inner band
APCR1I_SUPPORT_HYSTERESIS_RECENTER = SagittalAuthoritySchedule(
    profile_name="APCR1i_support_hysteresis_recenter",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    # Enable APCR for this profile
    enable_active_pitch_crossing=True,
    # WIDER pitch safe threshold to allow entry during moderate pitch error
    # APCR1i prioritizes drift recovery over pitch - pitch danger still blocks
    apc_pitch_safe_threshold_rad=0.15,  # 8.6 deg - wider than default (0.05 rad)
    apc_pitch_danger_threshold_rad=0.30,  # hard block at this threshold
    # Use APCR1i-specific thresholds for proportional soft band (not used but needed for telemetry)
    apc_outer_enter_m=0.08,  # Enter crossing when |e| > this (matches hysteresis outer_enter)
    apc_inner_exit_m=0.03,  # Exit crossing when |e| <= this (matches hysteresis inner_exit)
    # Hysteresis recenter parameters
    apc_hysteresis_enabled=True,
    apc_hysteresis_outer_enter_m=0.08,  # Enter recenter when |e| > this
    apc_hysteresis_inner_exit_m=0.03,  # Exit recenter when |e| <= this
    apc_hysteresis_opposite_release_m=0.03,  # Allow small overshoot into opposite
    apc_hysteresis_near_zero_m=0.01,  # Error considered near zero
    apc_hysteresis_emergency_m=0.12,  # Emergency clamp activates
    apc_hysteresis_hard_m=0.15,  # Hard safety activates
    apc_hysteresis_base_tau=0.45,  # Nm - base starting torque
    apc_hysteresis_recenter_max_tau=1.75,  # Nm - max during recenter
    apc_hysteresis_emergency_max_tau=2.00,  # Nm - max during emergency
    apc_hysteresis_hold_max_tau=1.50,  # Nm - max during hold-through-zero
    apc_hysteresis_normal_rate=0.30,  # Nm/step - normal rate
    apc_hysteresis_recenter_rate=0.90,  # Nm/step - recenter rate
    apc_hysteresis_emergency_rate=1.00,  # Nm/step - emergency rate
    apc_hysteresis_decay_rate=0.50,  # Nm/step - decay rate
    apc_hysteresis_phase_brake_threshold_m=0.05,  # Enable phase brake below this
    apc_hysteresis_phase_brake_disable_in_recenter=True,  # Disable in recenter state
    # Safety gates
    apc_contact_gate=True,
    apc_height_gate=True,
    apc_roll_gate=True,
    apc_min_com_z_m=0.27,
    apc_max_com_z_m=0.50,
    apc_roll_threshold_rad=0.15,
)

# APCR1j Support Hysteresis Higher Authority Profile
# Based on APCR1i but with higher torque authority to overcome the 1.5 Nm universal cap
# Root cause: APCR1i observed final APCR tau max = 1.5 Nm despite configured recenter_max_tau = 1.75 Nm
# Root cause: downstream apc_max_cross_tau = 1.5 universal clip overrides hysteresis authority
# Fix: explicitly set apc_max_cross_tau = 2.0 so hysteresis can reach 2.0 Nm
APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY = SagittalAuthoritySchedule(
    profile_name="APCR1j_support_hysteresis_higher_authority",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    # Enable APCR for this profile
    enable_active_pitch_crossing=True,
    # CRITICAL FIX: set apc_max_cross_tau = 2.0 to override the 1.5 Nm universal cap
    # This is the key difference from APCR1i
    apc_max_cross_tau=2.0,  # Nm - universal crossing torque cap (was 1.5 in APCR1i)
    # WIDER pitch safe threshold to allow entry during moderate pitch error
    # APCR1j prioritizes drift recovery over pitch - pitch danger still blocks
    apc_pitch_safe_threshold_rad=0.15,  # 8.6 deg - wider than default (0.05 rad)
    apc_pitch_danger_threshold_rad=0.30,  # hard block at this threshold
    # Use APCR1j-specific thresholds for proportional soft band (not used but needed for telemetry)
    apc_outer_enter_m=0.08,  # Enter crossing when |e| > this (matches hysteresis outer_enter)
    apc_inner_exit_m=0.03,  # Exit crossing when |e| <= this (matches hysteresis inner_exit)
    # Hysteresis recenter parameters - HIGHER AUTHORITY than APCR1i
    apc_hysteresis_enabled=True,
    apc_hysteresis_outer_enter_m=0.08,  # Enter recenter when |e| > this
    apc_hysteresis_inner_exit_m=0.03,  # Exit recenter when |e| <= this
    apc_hysteresis_opposite_release_m=0.03,  # Allow small overshoot into opposite
    apc_hysteresis_near_zero_m=0.01,  # Error considered near zero
    apc_hysteresis_emergency_m=0.12,  # Emergency clamp activates
    apc_hysteresis_hard_m=0.15,  # Hard safety activates
    apc_hysteresis_base_tau=0.45,  # Nm - base starting torque
    # HIGHER than APCR1i: 2.0 vs 1.75 Nm
    apc_hysteresis_recenter_max_tau=2.0,  # Nm - max during recenter (was 1.75 in APCR1i)
    # HIGHER than APCR1i: 2.2 vs 2.0 Nm
    apc_hysteresis_emergency_max_tau=2.2,  # Nm - max during emergency (was 2.00 in APCR1i)
    apc_hysteresis_hold_max_tau=1.75,  # Nm - max during hold-through-zero (was 1.50 in APCR1i)
    # FASTER than APCR1i: 1.1 vs 0.9 Nm/step
    apc_hysteresis_normal_rate=0.40,  # Nm/step - normal rate (was 0.30 in APCR1i)
    apc_hysteresis_recenter_rate=1.1,  # Nm/step - recenter rate (was 0.90 in APCR1i)
    # FASTER than APCR1i: 1.3 vs 1.0 Nm/step
    apc_hysteresis_emergency_rate=1.3,  # Nm/step - emergency rate (was 1.00 in APCR1i)
    apc_hysteresis_decay_rate=0.50,  # Nm/step - decay rate
    apc_hysteresis_phase_brake_threshold_m=0.05,  # Enable phase brake below this
    apc_hysteresis_phase_brake_disable_in_recenter=True,  # Disable in recenter state
    # Safety gates
    apc_contact_gate=True,
    apc_height_gate=True,
    apc_roll_gate=True,
    apc_min_com_z_m=0.27,
    apc_max_com_z_m=0.50,
    apc_roll_threshold_rad=0.15,
)

# APCR1k Support Hysteresis Early Entry Profile
# Based on APCR1j but with LOWER outer entry threshold to catch drift earlier
# Root cause: APCR1j analysis showed RECENTER starts at step 58 (e=0.0817m) allowing momentum buildup
# Fix: lower outer_enter_m from 0.08 to 0.05 to start RECENTER at step 46 (e=0.0521m)
# Keep same torque authority as APCR1j (2.0 Nm max)
APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY = SagittalAuthoritySchedule(
    profile_name="APCR1k_support_hysteresis_early_entry",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    # Enable APCR for this profile
    enable_active_pitch_crossing=True,
    # Keep same torque authority as APCR1j: 2.0 Nm
    apc_max_cross_tau=2.0,  # Nm - universal crossing torque cap (same as APCR1j)
    # WIDER pitch safe threshold to allow entry during moderate pitch error
    apc_pitch_safe_threshold_rad=0.15,  # 8.6 deg - same as APCR1j
    apc_pitch_danger_threshold_rad=0.30,  # hard block at this threshold - same as APCR1j
    # KEY CHANGE: lower outer enter threshold from 0.08 to 0.05
    # This catches drift earlier before momentum accumulates
    apc_outer_enter_m=0.05,  # Enter crossing when |e| > this (was 0.08 in APCR1j)
    apc_inner_exit_m=0.03,  # Exit crossing when |e| <= this (same as APCR1j)
    # Hysteresis recenter parameters - LOWER ENTRY THRESHOLD than APCR1j
    apc_hysteresis_enabled=True,
    apc_hysteresis_outer_enter_m=0.05,  # Enter recenter when |e| > this (was 0.08 in APCR1j)
    apc_hysteresis_inner_exit_m=0.03,  # Exit recenter when |e| <= this (same as APCR1j)
    apc_hysteresis_opposite_release_m=0.03,  # Allow small overshoot into opposite (same as APCR1j)
    apc_hysteresis_near_zero_m=0.01,  # Error considered near zero (same as APCR1j)
    apc_hysteresis_emergency_m=0.12,  # Emergency clamp activates (same as APCR1j)
    apc_hysteresis_hard_m=0.15,  # Hard safety activates (same as APCR1j)
    apc_hysteresis_base_tau=0.45,  # Nm - base starting torque (same as APCR1j)
    # Keep same torque limits as APCR1j: 2.0 Nm recenter, 2.2 Nm emergency
    apc_hysteresis_recenter_max_tau=2.0,  # Nm - max during recenter (same as APCR1j)
    apc_hysteresis_emergency_max_tau=2.2,  # Nm - max during emergency (same as APCR1j)
    apc_hysteresis_hold_max_tau=1.75,  # Nm - max during hold-through-zero (same as APCR1j)
    # Keep same rate limits as APCR1j: 1.1 Nm/step recenter, 1.3 Nm/step emergency
    apc_hysteresis_normal_rate=0.40,  # Nm/step - normal rate (same as APCR1j)
    apc_hysteresis_recenter_rate=1.1,  # Nm/step - recenter rate (same as APCR1j)
    apc_hysteresis_emergency_rate=1.3,  # Nm/step - emergency rate (same as APCR1j)
    apc_hysteresis_decay_rate=0.50,  # Nm/step - decay rate (same as APCR1j)
    apc_hysteresis_phase_brake_threshold_m=0.05,  # Enable phase brake below this (same as APCR1j)
    apc_hysteresis_phase_brake_disable_in_recenter=True,  # Disable in recenter state (same as APCR1j)
    # Safety gates - same as APCR1j
    apc_contact_gate=True,
    apc_height_gate=True,
    apc_roll_gate=True,
    apc_min_com_z_m=0.27,
    apc_max_com_z_m=0.50,
    apc_roll_threshold_rad=0.15,
)

# APCR1l Pitch Suppress in Recenter Profile
# Based on APCR1k but with pitch suppression during RECENTER state
# Root cause: tau_pitch produces positive torque when robot leans back (correcting drift)
# This fights the drift correction instead of helping
# Fix: suppress tau_pitch during RECENTER so APCR + tau_position can correct drift
APCR1L_PITCH_SUPPRESS_RECENTER = SagittalAuthoritySchedule(
    profile_name="APCR1l_pitch_suppress_recenter",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    # Enable APCR
    enable_active_pitch_crossing=True,
    # Same torque authority as APCR1k: 2.0 Nm
    apc_max_cross_tau=2.0,  # Nm - universal crossing torque cap
    # Same pitch safe thresholds as APCR1k
    apc_pitch_safe_threshold_rad=0.15,  # 8.6 deg
    apc_pitch_danger_threshold_rad=0.30,  # hard block
    # Same entry thresholds as APCR1k
    apc_outer_enter_m=0.05,  # Enter crossing when |e| > this
    apc_inner_exit_m=0.03,  # Exit crossing when |e| <= this
    # Same hysteresis parameters as APCR1k
    apc_hysteresis_enabled=True,
    apc_hysteresis_outer_enter_m=0.05,  # Enter recenter when |e| > this
    apc_hysteresis_inner_exit_m=0.03,  # Exit recenter when |e| <= this
    apc_hysteresis_opposite_release_m=0.03,  # Allow small overshoot
    apc_hysteresis_near_zero_m=0.01,  # Error considered near zero
    apc_hysteresis_emergency_m=0.12,  # Emergency clamp
    apc_hysteresis_hard_m=0.15,  # Hard safety
    apc_hysteresis_base_tau=0.45,  # Nm - base starting torque
    apc_hysteresis_recenter_max_tau=2.0,  # Nm - max during recenter
    apc_hysteresis_emergency_max_tau=2.2,  # Nm - max during emergency
    apc_hysteresis_hold_max_tau=1.75,  # Nm - max during hold
    apc_hysteresis_normal_rate=0.40,  # Nm/step
    apc_hysteresis_recenter_rate=1.1,  # Nm/step
    apc_hysteresis_emergency_rate=1.3,  # Nm/step
    apc_hysteresis_decay_rate=0.50,  # Nm/step
    apc_hysteresis_phase_brake_threshold_m=0.05,
    apc_hysteresis_phase_brake_disable_in_recenter=True,
    # KEY FIX: suppress tau_pitch during RECENTER state
    # This prevents pitch-stabilizing torque from fighting drift correction
    apc_hysteresis_pitch_suppress_in_recenter=True,
    # Safety gates
    apc_contact_gate=True,
    apc_height_gate=True,
    apc_roll_gate=True,
    apc_min_com_z_m=0.27,
    apc_max_com_z_m=0.50,
    apc_roll_threshold_rad=0.15,
)

# APCR1m Conditional Pitch Blend (conditional blending instead of hard suppression)
# Blend tau_pitch based on error magnitude, with startup guard and safety gates
APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER = SagittalAuthoritySchedule(
    profile_name="APCR1m_conditional_pitch_blend_recenter",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    # Enable APCR
    enable_active_pitch_crossing=True,
    # Same torque authority as APCR1k: 2.0 Nm
    apc_max_cross_tau=2.0,  # Nm - universal crossing torque cap
    # Same pitch safe thresholds as APCR1k
    apc_pitch_safe_threshold_rad=0.15,  # 8.6 deg
    apc_pitch_danger_threshold_rad=0.30,  # hard block
    # Same entry thresholds as APCR1k
    apc_outer_enter_m=0.05,  # Enter crossing when |e| > this
    apc_inner_exit_m=0.03,  # Exit crossing when |e| <= this
    # Same hysteresis parameters as APCR1k
    apc_hysteresis_enabled=True,
    apc_hysteresis_outer_enter_m=0.05,  # Enter recenter when |e| > this
    apc_hysteresis_inner_exit_m=0.03,  # Exit recenter when |e| <= this
    apc_hysteresis_opposite_release_m=0.03,  # Allow small overshoot
    apc_hysteresis_near_zero_m=0.01,  # Error considered near zero
    apc_hysteresis_emergency_m=0.12,  # Emergency clamp
    apc_hysteresis_hard_m=0.15,  # Hard safety
    apc_hysteresis_base_tau=0.45,  # Nm - base starting torque
    apc_hysteresis_recenter_max_tau=2.0,  # Nm - max during recenter
    apc_hysteresis_emergency_max_tau=2.2,  # Nm - max during emergency
    apc_hysteresis_hold_max_tau=1.75,  # Nm - max during hold
    apc_hysteresis_normal_rate=0.40,  # Nm/step
    apc_hysteresis_recenter_rate=1.1,  # Nm/step
    apc_hysteresis_emergency_rate=1.3,  # Nm/step
    apc_hysteresis_decay_rate=0.50,  # Nm/step
    apc_hysteresis_phase_brake_threshold_m=0.05,
    apc_hysteresis_phase_brake_disable_in_recenter=True,
    # KEY FIX: conditional pitch blend instead of hard suppression
    # Startup guard + safety gates + error-dependent scaling
    apc_pitch_blend_enabled=True,
    apc_pitch_blend_startup_guard_steps=100,  # No blending for first 100 steps
    apc_pitch_blend_safe_pitch_rad=0.15,  # Safe pitch threshold
    apc_pitch_blend_safe_pitch_rate_rad_s=0.5,  # Safe pitch rate
    apc_pitch_blend_min_com_z=0.27,  # Minimum height
    apc_pitch_blend_max_roll_rad=0.15,  # Maximum roll
    apc_pitch_blend_deep_error_m=0.12,  # |e| > 0.12 → scale 0.0
    apc_pitch_blend_mid_error_m=0.08,  # |e| > 0.08 → scale 0.25
    apc_pitch_blend_soft_error_m=0.05,  # |e| > 0.05 → scale 0.5
    apc_pitch_blend_scale_deep=0.0,  # tau_pitch * 0.0
    apc_pitch_blend_scale_mid=0.25,  # tau_pitch * 0.25
    apc_pitch_blend_scale_soft=0.5,  # tau_pitch * 0.5
    apc_pitch_blend_scale_near=1.0,  # tau_pitch * 1.0
    # Safety gates
    apc_contact_gate=True,
    apc_height_gate=True,
    apc_roll_gate=True,
    apc_min_com_z_m=0.27,
    apc_max_com_z_m=0.50,
    apc_roll_threshold_rad=0.15,
)

# APCR1n Recenter Priority Torque Boost
# Based on APCR1h with targeted fixes for support drift improvement:
# 1. Wheel damping override during RECENTER when it fights drift recovery
# 2. Position cap boost during safe RECENTER
# Root cause from APCR1m audit: wheel damping 3.5x too high (5.0 vs 1.4 Nm),
# position cap saturated 77.3%, final torque fights drift 62.8%
APCR1N_RECENTER_PRIORITY_TORQUE_BOOST = SagittalAuthoritySchedule(
    profile_name="APCR1n_recenter_priority_torque_boost",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    # APCR1h base configuration (copied from APCR1H_SUPPORT_DRIFT_PRIORITY)
    # Core scheduling parameters (REQUIRED for APCR1h baseline)
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    # APC configuration
    apc_proportional_soft_band_mode=True,
    apc_soft_enter_m=0.030,
    apc_inner_exit_m=0.015,
    apc_outer_enter_m=0.095,
    apc_velocity_decay_enabled=True,
    apc_velocity_decay_factor=0.5,
    apc_fast_response_enabled=True,
    apc_phase_brake_enabled=True,
    apc_phase_brake_threshold_m=0.08,
    apc_phase_brake_damping_factor=0.6,
    apc_boost_rate_per_step=0.25,
    apc_decay_rate_per_step=0.45,
    apc_increasing_error_threshold_steps=3,
    apc_increasing_error_boost_factor=0.3,
    apc_fast_response_inner_deadband_m=0.015,
    apc_fast_response_soft_enter_m=0.030,
    apc_fast_response_desired_band_m=0.08,
    apc_fast_response_full_torque_m=0.095,
    apc_fast_response_emergency_m=0.12,
    apc_fast_response_base_tau=0.45,
    apc_fast_response_max_tau=1.65,
    apc_fast_response_boost_tau_max=1.20,
    apc_fast_response_startup_boost_max_tau=1.60,
    apc_fast_response_max_rate_per_step=0.85,
    apc_fast_response_smooth_alpha=0.18,
    apc_fast_response_no_improvement_window=5,
    active_pitch_crossing_recovery_gate_mode=True,
    apc_drift_priority_enabled=True,
    apc_drift_priority_enter_m=0.08,
    apc_drift_priority_emergency_m=0.12,
    apc_drift_priority_hard_m=0.15,
    apc_drift_priority_base_tau=0.45,
    apc_drift_priority_normal_max_tau=1.40,
    apc_drift_priority_drift_priority_max_tau=1.65,
    apc_drift_priority_emergency_max_tau=1.85,
    apc_drift_priority_startup_max_tau=1.60,
    apc_drift_priority_normal_rate=0.55,
    apc_drift_priority_drift_priority_rate=0.85,
    apc_drift_priority_emergency_rate=1.00,
    apc_drift_priority_decay_rate=0.55,
    apc_drift_priority_phase_brake_disable_threshold_m=0.10,
    # APCR1n new fields: Recentering Priority
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_recenter_scale=0.30,
    vd_wheel_damping_recenter_min_abs_nm=0.50,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    position_cap_normal_nm=4.0,
    position_cap_recenter_nm=5.0,
    position_cap_emergency_nm=6.0,
    position_cap_ramp_steps=50,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
)

# APCR1nD Direct Support Recenter Features
# Key difference from APCR1n: uses DIRECT support drift trigger instead of APC dependency
# Activates based on direct drift conditions without requiring apc_enabled=True
APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES = SagittalAuthoritySchedule(
    profile_name="APCR1nD_direct_support_recenter_features",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    # APCR1h base configuration
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    # APCR1n recenter priority features
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_recenter_scale=0.30,
    vd_wheel_damping_recenter_min_abs_nm=0.50,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    position_cap_normal_nm=4.0,
    position_cap_recenter_nm=5.0,
    position_cap_emergency_nm=6.0,
    position_cap_ramp_steps=50,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    # APCR1nD: Direct support drift trigger (KEY DIFFERENCE)
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.08,
    recenter_priority_direct_emergency_m=0.12,
    recenter_priority_direct_hard_m=0.15,
    recenter_priority_direct_exit_m=0.02,
)

# APCR1nD Tuned Variants: Band-limited drift control
# Addressing APCR1nD band control failure (37.7% outside ±0.08 m)
# Root causes: moving-away gating too strict, authority too weak, late entry, early release

APCR1ND_T1_EARLY_ENTRY = SagittalAuthoritySchedule(
    profile_name="APCR1nD_T1_early_entry",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_recenter_scale=0.30,
    vd_wheel_damping_recenter_min_abs_nm=0.50,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    position_cap_normal_nm=4.0,
    position_cap_recenter_nm=5.0,
    position_cap_emergency_nm=6.0,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,  # Early entry
    recenter_priority_direct_exit_m=0.02,
    # Tuned variant config
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="T1",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.08,
    apcr1nd_release_inner_m=0.02,
    apcr1nd_hold_outside_band=False,
)

APCR1ND_T2_HOLD_OUTSIDE_BAND = SagittalAuthoritySchedule(
    profile_name="APCR1nD_T2_hold_outside_band",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_recenter_scale=0.30,
    vd_wheel_damping_recenter_min_abs_nm=0.50,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    position_cap_normal_nm=4.0,
    position_cap_recenter_nm=5.0,
    position_cap_emergency_nm=6.0,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.08,
    recenter_priority_direct_exit_m=0.05,
    # Tuned variant config
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="T2",
    apcr1nd_direct_enter_m=0.08,
    apcr1nd_desired_band_m=0.08,
    apcr1nd_release_inner_m=0.05,
    apcr1nd_hold_outside_band=True,
)

APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD = SagittalAuthoritySchedule(
    profile_name="APCR1nD_T3_early_entry_plus_hold",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_recenter_scale=0.30,
    vd_wheel_damping_recenter_min_abs_nm=0.50,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    position_cap_normal_nm=4.0,
    position_cap_recenter_nm=5.0,
    position_cap_emergency_nm=6.0,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    # Tuned variant config
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="T3",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.08,
    apcr1nd_release_inner_m=0.03,
    apcr1nd_hold_outside_band=True,
    apcr1nd_converging_release_steps=20,
)

APCR1ND_T4_STRONGER_AUTHORITY = SagittalAuthoritySchedule(
    profile_name="APCR1nD_T4_stronger_authority",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    # Tuned variant config
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="T4",
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_release_inner_m=0.03,
    apcr1nd_position_cap_normal_nm=4.0,
    apcr1nd_position_cap_desired_nm=6.0,
    apcr1nd_position_cap_emergency_nm=7.0,
    apcr1nd_damping_scale_desired=0.20,
    apcr1nd_damping_scale_hard=0.10,
)

BAND_LIMITED_SUPPORT_RECENTER = SagittalAuthoritySchedule(
    profile_name="band_limited_support_recenter",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    # Tuned variant config
    apcr1nd_tuned_enabled=True,
    # Band-Limited Support Recenter: Band-structure baseline for authority cap management
    apcr1nd_tuned_variant_name="band_limited",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.08,
    apcr1nd_hard_band_m=0.10,
    apcr1nd_emergency_band_m=0.12,
    apcr1nd_release_inner_m=0.03,
    apcr1nd_hold_outside_band=True,
    apcr1nd_converging_release_steps=15,
    apcr1nd_position_cap_normal_nm=4.0,
    apcr1nd_position_cap_soft_nm=4.5,
    apcr1nd_position_cap_desired_nm=5.5,
    apcr1nd_position_cap_hard_nm=6.5,
    apcr1nd_position_cap_emergency_nm=7.0,
    apcr1nd_damping_scale_normal=1.0,
    apcr1nd_damping_scale_soft=0.50,
    apcr1nd_damping_scale_desired=0.30,
    apcr1nd_damping_scale_hard=0.15,
    apcr1nd_damping_scale_emergency=0.10,
    apcr1nd_preserve_damping_if_helps=True,
)

# T6 High-Height Transient Suppression Variants
# Based on T5 high_0p480 failure audit: EMERGENCY_TOO_LATE + AUTHORITY_TOO_WEAK + DAMPING_TOO_STRONG

T6A_HIGH_EARLY_HARD_BAND = SagittalAuthoritySchedule(
    profile_name="T6A_high_early_hard_band",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    # T6A: Earlier entry into hard/emergency bands (vs T5)
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="T6A",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.07,  # tighter (was 0.08)
    apcr1nd_hard_band_m=0.085,     # tighter (was 0.10)
    apcr1nd_emergency_band_m=0.105, # tighter (was 0.12)
    apcr1nd_release_inner_m=0.03,
    apcr1nd_hold_outside_band=True,
    apcr1nd_converging_release_steps=15,
    apcr1nd_position_cap_normal_nm=4.0,
    apcr1nd_position_cap_soft_nm=4.5,
    apcr1nd_position_cap_desired_nm=5.5,
    apcr1nd_position_cap_hard_nm=6.5,
    apcr1nd_position_cap_emergency_nm=7.0,
    apcr1nd_damping_scale_normal=1.0,
    apcr1nd_damping_scale_soft=0.50,
    apcr1nd_damping_scale_desired=0.30,
    apcr1nd_damping_scale_hard=0.15,
    apcr1nd_damping_scale_emergency=0.10,
    apcr1nd_preserve_damping_if_helps=True,
)

T6B_HIGH_STRONGER_EMERGENCY = SagittalAuthoritySchedule(
    profile_name="T6B_high_stronger_emergency",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    # T6B: Stronger authority in high bands (vs T5)
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="T6B",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.08,
    apcr1nd_hard_band_m=0.10,
    apcr1nd_emergency_band_m=0.12,
    apcr1nd_release_inner_m=0.03,
    apcr1nd_hold_outside_band=True,
    apcr1nd_converging_release_steps=15,
    apcr1nd_position_cap_normal_nm=4.0,
    apcr1nd_position_cap_soft_nm=4.5,
    apcr1nd_position_cap_desired_nm=5.8,  # stronger (was 5.5)
    apcr1nd_position_cap_hard_nm=7.0,     # stronger (was 6.5)
    apcr1nd_position_cap_emergency_nm=8.0, # stronger (was 7.0)
    apcr1nd_damping_scale_normal=1.0,
    apcr1nd_damping_scale_soft=0.50,
    apcr1nd_damping_scale_desired=0.30,
    apcr1nd_damping_scale_hard=0.10,      # more aggressive (was 0.15)
    apcr1nd_damping_scale_emergency=0.05, # more aggressive (was 0.10)
    apcr1nd_preserve_damping_if_helps=True,
)

EMERGENCY_BUDGET_CAP_RAISE = SagittalAuthoritySchedule(
    profile_name="emergency_budget_cap_raise",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,  # Keep conservative nominal (inherited from T5)
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    # Emergency budget cap raise: based on band structure with architecture fix enabled
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="emergency_budget_cap_raise",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.08,
    apcr1nd_hard_band_m=0.10,
    apcr1nd_emergency_band_m=0.12,
    apcr1nd_release_inner_m=0.03,
    apcr1nd_hold_outside_band=True,
    apcr1nd_converging_release_steps=15,
    apcr1nd_position_cap_normal_nm=4.0,
    apcr1nd_position_cap_soft_nm=4.5,
    apcr1nd_position_cap_desired_nm=5.5,  # Same as T5
    apcr1nd_position_cap_hard_nm=6.5,     # Same as T5
    apcr1nd_position_cap_emergency_nm=7.0, # Same as T5
    apcr1nd_damping_scale_normal=1.0,
    apcr1nd_damping_scale_soft=0.50,
    apcr1nd_damping_scale_desired=0.30,
    apcr1nd_damping_scale_hard=0.15,
    apcr1nd_damping_scale_emergency=0.10,
    apcr1nd_preserve_damping_if_helps=True,
    # T6F Architecture Fix: Raise upstream cap during safe high-height emergency
    arch_fix_enabled=True,
    arch_fix_type="budget_cap_raise",
    arch_fix_height_threshold_m=0.45,
    arch_fix_hard_max_position_tau=6.5,       # Allow hard band to reach 6.5 Nm
    arch_fix_emergency_max_position_tau=7.0,  # Allow emergency band to reach 7.0 Nm (T5 tuned cap)
)

T6F_SIGN_CORRECTED = SagittalAuthoritySchedule(
    profile_name="T6F_sign_corrected",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    # T6F_sign_corrected: Based on T6F with sign fix enabled
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="T6F_sign_corrected",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.08,
    apcr1nd_hard_band_m=0.10,
    apcr1nd_emergency_band_m=0.12,
    apcr1nd_release_inner_m=0.03,
    apcr1nd_hold_outside_band=True,
    apcr1nd_converging_release_steps=15,
    apcr1nd_position_cap_normal_nm=4.0,
    apcr1nd_position_cap_soft_nm=4.5,
    apcr1nd_position_cap_desired_nm=5.5,
    apcr1nd_position_cap_hard_nm=6.5,
    apcr1nd_position_cap_emergency_nm=7.0,
    apcr1nd_damping_scale_normal=1.0,
    apcr1nd_damping_scale_soft=0.50,
    apcr1nd_damping_scale_desired=0.30,
    apcr1nd_damping_scale_hard=0.15,
    apcr1nd_damping_scale_emergency=0.10,
    apcr1nd_preserve_damping_if_helps=True,
    # T6F Architecture Fix: Preserve T6F budget cap raise mechanism
    arch_fix_enabled=True,
    arch_fix_type="budget_cap_raise",
    arch_fix_height_threshold_m=0.45,
    arch_fix_hard_max_position_tau=6.5,
    arch_fix_emergency_max_position_tau=7.0,
    # T6F Sign Fix: Enhanced damping override and pitch suppression
    sign_fix_enabled=True,
    sign_fix_disable_fighting_damping_during_arch_fix=True,
    sign_fix_suppress_pitch_during_arch_fix=True,
    sign_fix_pitch_error_threshold_m=0.10,
    sign_fix_suppress_pitch_rate=False,  # Keep pitch rate for now
)

T6H_SOFT_BLEND_ARCH_FIX = SagittalAuthoritySchedule(
    profile_name="T6H_soft_blend_arch_fix",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    # T6H: Based on T6F with soft blend instead of hard suppression
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="T6H",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.08,
    apcr1nd_hard_band_m=0.10,
    apcr1nd_emergency_band_m=0.12,
    apcr1nd_release_inner_m=0.03,
    apcr1nd_hold_outside_band=True,
    apcr1nd_converging_release_steps=15,
    apcr1nd_position_cap_normal_nm=4.0,
    apcr1nd_position_cap_soft_nm=4.5,
    apcr1nd_position_cap_desired_nm=5.5,
    apcr1nd_position_cap_hard_nm=6.5,
    apcr1nd_position_cap_emergency_nm=7.0,
    apcr1nd_damping_scale_normal=1.0,
    apcr1nd_damping_scale_soft=0.50,
    apcr1nd_damping_scale_desired=0.30,
    apcr1nd_damping_scale_hard=0.15,
    apcr1nd_damping_scale_emergency=0.10,
    apcr1nd_preserve_damping_if_helps=True,
    # T6H: Architecture fix enabled (budget cap raise)
    arch_fix_enabled=True,
    arch_fix_type="budget_cap_raise",
    arch_fix_height_threshold_m=0.45,
    arch_fix_hard_max_position_tau=6.5,
    arch_fix_emergency_max_position_tau=7.0,
    # T6H: Soft blend approach (reduces by 50%, not 100%)
    t6h_enabled=True,
    t6h_soft_pitch_blend_factor=0.50,
    t6h_soft_damping_blend_factor=0.50,
    t6h_pitch_error_threshold_m=0.10,
    t6h_pitch_safety_threshold_deg=10.0,
    t6h_wheel_velocity_safety_threshold_rad_s=7.0,
)

PHASE_AWARE_AUTHORITY_RELEASE = SagittalAuthoritySchedule(
    profile_name="phase_aware_authority_release",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    # Phase-aware authority release: based on emergency budget cap with phase-aware decay
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="phase_aware_authority_release",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.08,
    apcr1nd_hard_band_m=0.10,
    apcr1nd_emergency_band_m=0.12,
    apcr1nd_release_inner_m=0.03,
    apcr1nd_hold_outside_band=True,
    apcr1nd_converging_release_steps=15,
    apcr1nd_position_cap_normal_nm=4.0,
    apcr1nd_position_cap_soft_nm=4.5,
    apcr1nd_position_cap_desired_nm=5.5,
    apcr1nd_position_cap_hard_nm=6.5,
    apcr1nd_position_cap_emergency_nm=7.0,
    apcr1nd_damping_scale_normal=1.0,
    apcr1nd_damping_scale_soft=0.50,
    apcr1nd_damping_scale_desired=0.30,
    apcr1nd_damping_scale_hard=0.15,
    apcr1nd_damping_scale_emergency=0.10,
    apcr1nd_preserve_damping_if_helps=True,
    # T6I: Architecture fix enabled (budget cap raise)
    arch_fix_enabled=True,
    arch_fix_type="budget_cap_raise",
    arch_fix_height_threshold_m=0.45,
    arch_fix_hard_max_position_tau=6.5,
    arch_fix_emergency_max_position_tau=7.0,
    # T6I: Phase-aware release (gradual cap decay when converging)
    t6i_enabled=True,
    t6i_convergence_window_steps=5,
    t6i_convergence_threshold_m=0.12,
    t6i_convergence_trend_threshold_m=0.03,
    t6i_cap_decay_rate_nm_per_step=0.10,
    t6i_cap_min_nm=4.0,
    t6i_max_cap_delta_per_step_nm=0.30,
)

SUPPORT_CENTERING_BIAS_TRIM = SagittalAuthoritySchedule(
    profile_name="support_centering_bias_trim",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height", "high_0p430", "high_0p450", "high_0p465", "high_0p480"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="support_centering_bias_trim",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.08,
    apcr1nd_hard_band_m=0.10,
    apcr1nd_emergency_band_m=0.12,
    apcr1nd_release_inner_m=0.03,
    apcr1nd_hold_outside_band=True,
    apcr1nd_converging_release_steps=15,
    apcr1nd_position_cap_normal_nm=4.0,
    apcr1nd_position_cap_soft_nm=4.5,
    apcr1nd_position_cap_desired_nm=5.5,
    apcr1nd_position_cap_hard_nm=6.5,
    apcr1nd_position_cap_emergency_nm=7.0,
    apcr1nd_damping_scale_normal=1.0,
    apcr1nd_damping_scale_soft=0.50,
    apcr1nd_damping_scale_desired=0.30,
    apcr1nd_damping_scale_hard=0.15,
    apcr1nd_damping_scale_emergency=0.10,
    apcr1nd_preserve_damping_if_helps=True,
    arch_fix_enabled=True,
    arch_fix_type="budget_cap_raise",
    arch_fix_height_threshold_m=0.45,
    arch_fix_hard_max_position_tau=6.5,
    arch_fix_emergency_max_position_tau=7.0,
    t6i_enabled=True,
    t6i_convergence_window_steps=5,
    t6i_convergence_threshold_m=0.12,
    t6i_convergence_trend_threshold_m=0.03,
    t6i_cap_decay_rate_nm_per_step=0.10,
    t6i_cap_min_nm=4.0,
    t6i_max_cap_delta_per_step_nm=0.30,
    t6j_bias_trim_enabled=True,
    t6j_bias_trim_window_steps=200,
    t6j_bias_trim_enter_threshold_m=0.04,
    t6j_bias_trim_exit_threshold_m=0.015,
    t6j_bias_trim_max_tau_nm=0.35,
    t6j_bias_trim_rate_nm_per_step=0.01,
    t6j_bias_trim_decay_rate_nm_per_step=0.02,
    t6j_bias_trim_only_when_upright=True,
    t6j_bias_trim_only_when_contact_stable=True,
    t6j_bias_trim_disable_if_pitch_gt_deg=8.0,
    t6j_bias_trim_disable_if_roll_gt_deg=3.0,
    t6j_bias_trim_disable_if_wheel_vel_gt_rad_s=7.0,
    t6j_bias_trim_disable_if_abs_error_gt_m=0.22,
)

# Adaptive Centering Bias Trim: proportional, height-aware, guarded trim
# Inherits all SUPPORT_CENTERING_BIAS_TRIM settings and adds adaptive trim.
# When enabled, replaces the bang-bang T6J trim with proportional authority.
ADAPTIVE_SUPPORT_CENTERING_TRIM = SagittalAuthoritySchedule(
    profile_name="adaptive_support_centering_trim",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height", "high_0p430", "high_0p450", "high_0p465", "high_0p480"),
    # Inherit all SUPPORT_CENTERING_BIAS_TRIM settings
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="adaptive_support_centering_trim",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.08,
    apcr1nd_hard_band_m=0.10,
    apcr1nd_emergency_band_m=0.12,
    apcr1nd_release_inner_m=0.03,
    apcr1nd_hold_outside_band=True,
    apcr1nd_converging_release_steps=15,
    apcr1nd_position_cap_normal_nm=4.0,
    apcr1nd_position_cap_soft_nm=4.5,
    apcr1nd_position_cap_desired_nm=5.5,
    apcr1nd_position_cap_hard_nm=6.5,
    apcr1nd_position_cap_emergency_nm=7.0,
    apcr1nd_damping_scale_normal=1.0,
    apcr1nd_damping_scale_soft=0.50,
    apcr1nd_damping_scale_desired=0.30,
    apcr1nd_damping_scale_hard=0.15,
    apcr1nd_damping_scale_emergency=0.10,
    apcr1nd_preserve_damping_if_helps=True,
    arch_fix_enabled=True,
    arch_fix_type="budget_cap_raise",
    arch_fix_height_threshold_m=0.45,
    arch_fix_hard_max_position_tau=6.5,
    arch_fix_emergency_max_position_tau=7.0,
    t6i_enabled=True,
    t6i_convergence_window_steps=5,
    t6i_convergence_threshold_m=0.12,
    t6i_convergence_trend_threshold_m=0.03,
    t6i_cap_decay_rate_nm_per_step=0.10,
    t6i_cap_min_nm=4.0,
    t6i_max_cap_delta_per_step_nm=0.30,
    t6j_bias_trim_enabled=False,  # Replaced by adaptive trim
    # Adaptive centering bias trim (replaces bang-bang T6J)
    adaptive_bias_trim_enabled=True,
    adaptive_bias_trim_replace_t6j=True,
    adaptive_bias_window_steps=300,
    adaptive_bias_fast_window_steps=100,
    adaptive_bias_enter_threshold_m=0.035,
    adaptive_bias_exit_threshold_m=0.012,
    adaptive_bias_relief_hysteresis_m=0.005,
    adaptive_bias_k_tau_per_m=5.0,
    adaptive_bias_max_tau_low_nm=0.35,
    adaptive_bias_max_tau_high_nm=0.50,
    adaptive_bias_max_tau_extreme_nm=0.55,
    adaptive_bias_height_low_m=0.38,
    adaptive_bias_height_high_m=0.48,
    adaptive_bias_height_extreme_m=0.52,
    adaptive_bias_rate_nm_per_step=0.006,
    adaptive_bias_fast_rate_nm_per_step=0.012,
    adaptive_bias_decay_rate_nm_per_step=0.018,
    adaptive_bias_only_when_upright=True,
    adaptive_bias_only_when_contact_stable=True,
    adaptive_bias_disable_if_pitch_gt_deg=12.0,
    adaptive_bias_disable_if_roll_gt_deg=5.0,
    adaptive_bias_disable_if_abs_error_gt_m=0.24,
    adaptive_bias_disable_if_hip_yaw_gt_rad=0.25,
    adaptive_bias_zero_crossing_guard_enabled=True,
    adaptive_bias_zero_crossing_window_steps=500,
    adaptive_bias_zero_crossing_limit=8,
    adaptive_bias_zero_crossing_max_scale=0.5,
    adaptive_bias_sign_reversal_hold_steps=100,
)

T6C_HIGH_EARLY_PLUS_STRONGER = SagittalAuthoritySchedule(
    profile_name="T6C_high_early_plus_stronger",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    # T6C: Combined earlier entry + stronger authority
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="T6C",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.07,          # T6A
    apcr1nd_hard_band_m=0.085,            # T6A
    apcr1nd_emergency_band_m=0.105,       # T6A
    apcr1nd_release_inner_m=0.03,
    apcr1nd_hold_outside_band=True,
    apcr1nd_converging_release_steps=15,
    apcr1nd_position_cap_normal_nm=4.0,
    apcr1nd_position_cap_soft_nm=4.5,
    apcr1nd_position_cap_desired_nm=5.8,  # T6B
    apcr1nd_position_cap_hard_nm=7.0,     # T6B
    apcr1nd_position_cap_emergency_nm=8.0, # T6B
    apcr1nd_damping_scale_normal=1.0,
    apcr1nd_damping_scale_soft=0.50,
    apcr1nd_damping_scale_desired=0.25,   # slightly more aggressive
    apcr1nd_damping_scale_hard=0.10,      # T6B
    apcr1nd_damping_scale_emergency=0.05, # T6B
    apcr1nd_preserve_damping_if_helps=True,
)

# T6D and T6E require conditional logic - implemented as T6C for now
# (Transient-only and pitch-aware variants need additional state tracking)
T6D_HIGH_TRANSIENT_BOOST = T6C_HIGH_EARLY_PLUS_STRONGER
T6E_HIGH_PITCH_AWARE_BOOST = T6C_HIGH_EARLY_PLUS_STRONGER

# Zero-Crossing Support Recenter: hysteresis recenter that forces drift to cross zero
# Based on adaptive_support_centering_trim + ZC state machine
ZERO_CROSSING_SUPPORT_RECENTER = SagittalAuthoritySchedule(
    profile_name="zero_crossing_support_recenter",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height", "high_0p430", "high_0p450", "high_0p465", "high_0p480"),
    # Inherit all ADAPTIVE_SUPPORT_CENTERING_TRIM settings
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="zero_crossing_support_recenter",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.08,
    apcr1nd_hard_band_m=0.10,
    apcr1nd_emergency_band_m=0.12,
    apcr1nd_release_inner_m=0.03,
    apcr1nd_hold_outside_band=True,
    apcr1nd_converging_release_steps=15,
    apcr1nd_position_cap_normal_nm=4.0,
    apcr1nd_position_cap_soft_nm=4.5,
    apcr1nd_position_cap_desired_nm=5.5,
    apcr1nd_position_cap_hard_nm=6.5,
    apcr1nd_position_cap_emergency_nm=7.0,
    apcr1nd_damping_scale_normal=1.0,
    apcr1nd_damping_scale_soft=0.50,
    apcr1nd_damping_scale_desired=0.30,
    apcr1nd_damping_scale_hard=0.15,
    apcr1nd_damping_scale_emergency=0.10,
    apcr1nd_preserve_damping_if_helps=True,
    arch_fix_enabled=True,
    arch_fix_type="budget_cap_raise",
    arch_fix_height_threshold_m=0.45,
    arch_fix_hard_max_position_tau=6.5,
    arch_fix_emergency_max_position_tau=7.0,
    t6i_enabled=True,
    t6i_convergence_window_steps=5,
    t6i_convergence_threshold_m=0.12,
    t6i_convergence_trend_threshold_m=0.03,
    t6i_cap_decay_rate_nm_per_step=0.10,
    t6i_cap_min_nm=4.0,
    t6i_max_cap_delta_per_step_nm=0.30,
    t6j_bias_trim_enabled=False,
    adaptive_bias_trim_enabled=True,
    adaptive_bias_trim_replace_t6j=True,
    adaptive_bias_window_steps=300,
    adaptive_bias_fast_window_steps=100,
    adaptive_bias_enter_threshold_m=0.035,
    adaptive_bias_exit_threshold_m=0.012,
    adaptive_bias_relief_hysteresis_m=0.005,
    adaptive_bias_k_tau_per_m=5.0,
    adaptive_bias_max_tau_low_nm=0.35,
    adaptive_bias_max_tau_high_nm=0.50,
    adaptive_bias_max_tau_extreme_nm=0.55,
    adaptive_bias_height_low_m=0.38,
    adaptive_bias_height_high_m=0.48,
    adaptive_bias_height_extreme_m=0.52,
    adaptive_bias_rate_nm_per_step=0.006,
    adaptive_bias_fast_rate_nm_per_step=0.012,
    adaptive_bias_decay_rate_nm_per_step=0.018,
    adaptive_bias_only_when_upright=True,
    adaptive_bias_only_when_contact_stable=True,
    adaptive_bias_disable_if_pitch_gt_deg=12.0,
    adaptive_bias_disable_if_roll_gt_deg=5.0,
    adaptive_bias_disable_if_abs_error_gt_m=0.24,
    adaptive_bias_disable_if_hip_yaw_gt_rad=0.25,
    adaptive_bias_zero_crossing_guard_enabled=True,
    adaptive_bias_zero_crossing_window_steps=500,
    adaptive_bias_zero_crossing_limit=8,
    adaptive_bias_zero_crossing_max_scale=0.5,
    adaptive_bias_sign_reversal_hold_steps=100,
    # ZC recenter: NEW - hysteresis hold-through-zero recenter
    enable_zero_crossing_recenter=True,
    zc_replace_adaptive=False,  # ZC supplements adaptive trim
    zc_enter_m=0.08,
    zc_exit_m=0.025,
    zc_cross_target_m=0.02,
    zc_near_zero_band_m=0.03,
    zc_min_hold_steps=50,
    zc_max_hold_steps=600,
    zc_base_tau_nm=0.20,
    zc_max_tau_nm=0.65,
    zc_rate_nm_per_step=0.01,
    zc_decay_nm_per_step=0.02,
    zc_error_gain_nm_per_m=3.0,
    zc_velocity_gain=0.0,
    zc_disable_if_abs_error_gt_m=0.25,
    zc_disable_if_pitch_gt_deg=12.0,
    zc_disable_if_roll_gt_deg=5.0,
    zc_disable_if_hip_yaw_gt_rad=0.25,
    zc_dwell_steps_for_exit=30,
    zc_dwell_target_within_m=0.015,
)

# Early Zero-Crossing Support Recenter: exits at zero crossing, not opposite side
# Based on ZERO_CROSSING_SUPPORT_RECENTER with key changes:
# - Entry at 0.05 m (earlier) vs 0.08 m
# - Exit at e <= 0 (not -0.02)
# - No opposite-side target required
# - Immediate decay after zero crossing
EARLY_ZERO_CROSSING_RECENTER = SagittalAuthoritySchedule(
    profile_name="early_zero_crossing_recenter",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height", "high_0p430", "high_0p450", "high_0p465", "high_0p480"),
    # Inherit all ZERO_CROSSING_SUPPORT_RECENTER settings
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="early_zero_crossing_recenter",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.08,
    apcr1nd_hard_band_m=0.10,
    apcr1nd_emergency_band_m=0.12,
    apcr1nd_release_inner_m=0.03,
    apcr1nd_hold_outside_band=True,
    apcr1nd_converging_release_steps=15,
    apcr1nd_position_cap_normal_nm=4.0,
    apcr1nd_position_cap_soft_nm=4.5,
    apcr1nd_position_cap_desired_nm=5.5,
    apcr1nd_position_cap_hard_nm=6.5,
    apcr1nd_position_cap_emergency_nm=7.0,
    apcr1nd_damping_scale_normal=1.0,
    apcr1nd_damping_scale_soft=0.50,
    apcr1nd_damping_scale_desired=0.30,
    apcr1nd_damping_scale_hard=0.15,
    apcr1nd_damping_scale_emergency=0.10,
    apcr1nd_preserve_damping_if_helps=True,
    arch_fix_enabled=True,
    arch_fix_type="budget_cap_raise",
    arch_fix_height_threshold_m=0.45,
    arch_fix_hard_max_position_tau=6.5,
    arch_fix_emergency_max_position_tau=7.0,
    t6i_enabled=True,
    t6i_convergence_window_steps=5,
    t6i_convergence_threshold_m=0.12,
    t6i_convergence_trend_threshold_m=0.03,
    t6i_cap_decay_rate_nm_per_step=0.10,
    t6i_cap_min_nm=4.0,
    t6i_max_cap_delta_per_step_nm=0.30,
    t6j_bias_trim_enabled=False,
    adaptive_bias_trim_enabled=True,
    adaptive_bias_trim_replace_t6j=True,
    adaptive_bias_window_steps=300,
    adaptive_bias_fast_window_steps=100,
    adaptive_bias_enter_threshold_m=0.035,
    adaptive_bias_exit_threshold_m=0.012,
    adaptive_bias_relief_hysteresis_m=0.005,
    adaptive_bias_k_tau_per_m=5.0,
    adaptive_bias_max_tau_low_nm=0.35,
    adaptive_bias_max_tau_high_nm=0.50,
    adaptive_bias_max_tau_extreme_nm=0.55,
    adaptive_bias_height_low_m=0.38,
    adaptive_bias_height_high_m=0.48,
    adaptive_bias_height_extreme_m=0.52,
    adaptive_bias_rate_nm_per_step=0.006,
    adaptive_bias_fast_rate_nm_per_step=0.012,
    adaptive_bias_decay_rate_nm_per_step=0.018,
    adaptive_bias_only_when_upright=True,
    adaptive_bias_only_when_contact_stable=True,
    adaptive_bias_disable_if_pitch_gt_deg=12.0,
    adaptive_bias_disable_if_roll_gt_deg=5.0,
    adaptive_bias_disable_if_abs_error_gt_m=0.24,
    adaptive_bias_disable_if_hip_yaw_gt_rad=0.25,
    adaptive_bias_zero_crossing_guard_enabled=True,
    adaptive_bias_zero_crossing_window_steps=500,
    adaptive_bias_zero_crossing_limit=8,
    adaptive_bias_zero_crossing_max_scale=0.5,
    adaptive_bias_sign_reversal_hold_steps=100,
    # Early ZC recenter: EXITS AT ZERO, not opposite side
    enable_zero_crossing_recenter=False,  # Disable old ZC
    enable_early_zero_crossing_recenter=True,
    ezc_replace_adaptive=False,  # EZC supplements adaptive trim
    ezc_replace_zc=True,  # EZC replaces old ZC
    ezc_enter_m=0.05,  # Earlier entry than old ZC (0.08)
    ezc_exit_at_zero=True,  # Exit at zero, not -0.02
    ezc_zero_dwell_steps=3,  # Brief dwell at zero
    ezc_reentry_m=0.05,
    ezc_min_hold_steps=0,  # No minimum hold
    ezc_max_hold_steps=500,  # 500 steps max hold
    ezc_base_tau_nm=0.18,  # Slightly lower base torque
    ezc_max_tau_nm=0.55,  # Slightly lower max torque
    ezc_rate_nm_per_step=0.012,  # Faster rate
    ezc_decay_nm_per_step=0.025,  # Faster decay
    ezc_error_gain_nm_per_m=3.0,
    ezc_disable_if_abs_error_gt_m=0.25,
    ezc_disable_if_pitch_gt_deg=12.0,
    ezc_disable_if_roll_gt_deg=5.0,
    ezc_disable_if_hip_yaw_gt_rad=0.25,
)


# =====================================================================
# EARLY_ZERO_CROSSING_RECENTER_V2: Anti-rebound fix
# =====================================================================
# Fix for EZC_FAILURE_EXIT_TOO_EARLY_REBOUND
# Root cause: EZC exits at zero but positive bias (~+3.5 Nm) immediately
# returns drift to +0.10 to +0.20 m before EZC can re-enter.
#
# Changes from V1 (early_zero_crossing_recenter):
# - Stronger torque: base 0.25, max 0.70 (vs 0.18, 0.55)
# - Anti-rebound decay: keep decaying correction for 30 steps after crossing zero
# - Slower decay rate: 0.018 Nm/step (vs 0.025) to sustain correction longer
# - Longer zero dwell: 5 steps (vs 3) before entering anti-rebound decay
#
# Anti-rebound logic:
# When crossing zero (e <= 0), enter ANTIREBOUND_DECAY state instead of exiting.
# Keep current tau (ezc_antirebound_initial_ratio * current tau) and decay over
# ezc_antirebound_decay_steps (30). This prevents drift from immediately returning
# positive while tau_position recovers from the positive bias.
EARLY_ZERO_CROSSING_RECENTER_V2 = SagittalAuthoritySchedule(
    profile_name="early_zero_crossing_recenter_v2",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height", "high_0p430", "high_0p450", "high_0p465", "high_0p480"),
    # Inherit all EARLY_ZERO_CROSSING_RECENTER settings
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
    recenter_priority_direct_enabled=True,
    recenter_priority_direct_enter_m=0.06,
    recenter_priority_direct_exit_m=0.03,
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="early_zero_crossing_recenter_v2",
    apcr1nd_soft_enter_m=0.05,
    apcr1nd_direct_enter_m=0.06,
    apcr1nd_desired_band_m=0.08,
    apcr1nd_hard_band_m=0.10,
    apcr1nd_emergency_band_m=0.12,
    apcr1nd_release_inner_m=0.03,
    apcr1nd_hold_outside_band=True,
    apcr1nd_converging_release_steps=15,
    apcr1nd_position_cap_normal_nm=4.0,
    apcr1nd_position_cap_soft_nm=4.5,
    apcr1nd_position_cap_desired_nm=5.5,
    apcr1nd_position_cap_hard_nm=6.5,
    apcr1nd_position_cap_emergency_nm=7.0,
    apcr1nd_damping_scale_normal=1.0,
    apcr1nd_damping_scale_soft=0.50,
    apcr1nd_damping_scale_desired=0.30,
    apcr1nd_damping_scale_hard=0.15,
    apcr1nd_damping_scale_emergency=0.10,
    apcr1nd_preserve_damping_if_helps=True,
    arch_fix_enabled=True,
    arch_fix_type="budget_cap_raise",
    arch_fix_height_threshold_m=0.45,
    arch_fix_hard_max_position_tau=6.5,
    arch_fix_emergency_max_position_tau=7.0,
    t6i_enabled=True,
    t6i_convergence_window_steps=5,
    t6i_convergence_threshold_m=0.12,
    t6i_convergence_trend_threshold_m=0.03,
    t6i_cap_decay_rate_nm_per_step=0.10,
    t6i_cap_min_nm=4.0,
    t6i_max_cap_delta_per_step_nm=0.30,
    t6j_bias_trim_enabled=False,
    adaptive_bias_trim_enabled=True,
    adaptive_bias_trim_replace_t6j=True,
    adaptive_bias_window_steps=300,
    adaptive_bias_fast_window_steps=100,
    adaptive_bias_enter_threshold_m=0.035,
    adaptive_bias_exit_threshold_m=0.012,
    adaptive_bias_relief_hysteresis_m=0.005,
    adaptive_bias_k_tau_per_m=5.0,
    adaptive_bias_max_tau_low_nm=0.35,
    adaptive_bias_max_tau_high_nm=0.50,
    adaptive_bias_max_tau_extreme_nm=0.55,
    adaptive_bias_height_low_m=0.38,
    adaptive_bias_height_high_m=0.48,
    adaptive_bias_height_extreme_m=0.52,
    adaptive_bias_rate_nm_per_step=0.006,
    adaptive_bias_fast_rate_nm_per_step=0.012,
    adaptive_bias_decay_rate_nm_per_step=0.018,
    adaptive_bias_only_when_upright=True,
    adaptive_bias_only_when_contact_stable=True,
    adaptive_bias_disable_if_pitch_gt_deg=12.0,
    adaptive_bias_disable_if_roll_gt_deg=5.0,
    adaptive_bias_disable_if_abs_error_gt_m=0.24,
    adaptive_bias_disable_if_hip_yaw_gt_rad=0.25,
    adaptive_bias_zero_crossing_guard_enabled=True,
    adaptive_bias_zero_crossing_window_steps=500,
    adaptive_bias_zero_crossing_limit=8,
    adaptive_bias_zero_crossing_max_scale=0.5,
    adaptive_bias_sign_reversal_hold_steps=100,
    # Disable old ZC, enable EZC
    enable_zero_crossing_recenter=False,
    enable_early_zero_crossing_recenter=True,
    ezc_replace_adaptive=False,
    ezc_replace_zc=True,
    # EZC V2 parameters: stronger, with anti-rebound
    ezc_enter_m=0.05,
    ezc_exit_at_zero=True,
    ezc_zero_dwell_steps=5,  # Longer dwell before anti-rebound decay
    ezc_reentry_m=0.05,
    ezc_min_hold_steps=0,
    ezc_max_hold_steps=500,
    ezc_base_tau_nm=0.25,  # Stronger base (vs 0.18)
    ezc_max_tau_nm=0.70,   # Stronger max (vs 0.55)
    ezc_rate_nm_per_step=0.015,  # Faster ramp (vs 0.012)
    ezc_decay_nm_per_step=0.018,  # Slower decay (vs 0.025) - KEY CHANGE
    ezc_error_gain_nm_per_m=4.0,  # Stronger error gain (vs 3.0)
    # Anti-rebound configuration (NEW)
    ezc_antirebound_enabled=True,  # KEY FIX
    ezc_antirebound_decay_steps=30,  # Decay over 30 steps after zero crossing
    ezc_antirebound_initial_ratio=0.50,  # Start at 50% of current tau
    ezc_disable_if_abs_error_gt_m=0.25,
    ezc_disable_if_pitch_gt_deg=12.0,
    ezc_disable_if_roll_gt_deg=5.0,
    ezc_disable_if_hip_yaw_gt_rad=0.25,
)


# =====================================================================
# Phase 7: Pitch Bias Compensated Zero-Crossing Recenter
# Inherits all settings from EARLY_ZERO_CROSSING_RECENTER_V2 plus enables
# pitch bias DC compensation. See:
#   docs/validation/tau_pitch_positive_bias_audit.md
#   docs/validation/pitch_bias_compensated_zc_design.md
# =====================================================================
PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER = replace(
    EARLY_ZERO_CROSSING_RECENTER_V2,
    profile_name="pitch_bias_compensated_zero_crossing_recenter",
    pitch_bias_comp_enabled=True,
    pitch_bias_window_steps=300,
    pitch_bias_max_comp_nm=0.60,
    pitch_bias_comp_rate_nm_per_step=0.005,
    pitch_bias_decay_rate_nm_per_step=0.012,
    pitch_bias_only_when_abs_pitch_lt_deg=2.0,
    pitch_bias_only_when_abs_error_lt_m=0.12,
    pitch_bias_disable_if_pitch_gt_deg=12.0,
    pitch_bias_disable_if_roll_gt_deg=5.0,
    pitch_bias_disable_if_contact_unstable=True,
    pitch_bias_disable_if_height_lt_m=0.25,
    pitch_bias_gate_abs_error_soft_m=0.12,
    pitch_bias_gate_abs_error_hard_m=0.20,
)

# =====================================================================
# Pitch Equilibrium Trim (Phase 3 structural fix)
# =====================================================================
# ROOT CAUSE (see docs/validation/sagittal_root_cause_final_report.md):
# The robot settles into a forward-pitched equilibrium (~+3.3 deg) because
# the leg geometry at high heights places the CoM slightly forward of the
# wheel contact line. With pitch_ref=0, tau_pitch = kp_pitch * pitch_x is
# persistently positive (~+2.9 Nm), pushing wheels forward, while
# tau_position pulls backward and saturates. The two net to ~0 final wheel
# torque, freezing the robot in a forward-biased support stalemate and
# producing one-sided positive drift (80-92% positive).
#
# FIX: shift the pitch reference to the measured equilibrium pitch via a
# small positive offset. This makes tau_pitch oscillate symmetrically about
# zero instead of biasing forward, so support drift centers about zero.
# This is a coordination fix, NOT a suppression: full dynamic pitch gain is
# preserved; only the setpoint moves. Causal ablation (kp_pitch sweep +
# pitch_ref sweep) confirmed ROOT_CAUSE_PITCH_GAIN_TOO_HIGH relative to the
# equilibrium requirement, and +4 deg on high_0p480 yielded 5000-step
# pos%=46.7 / neg%=53.3 with no fall.
#
# Built on ADAPTIVE_SUPPORT_CENTERING_TRIM so all existing safety gates,
# recenter machinery, and authority scheduling remain intact. Opt-in only.
PITCH_EQUILIBRIUM_TRIM = replace(
    ADAPTIVE_SUPPORT_CENTERING_TRIM,
    profile_name="pitch_equilibrium_trim",
    pitch_ref_offset_deg=4.0,
)

# =====================================================================
# HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM (structural fix, Phase 2)
# =====================================================================
# The static pitch_equilibrium_trim above applies a single +4 deg forward
# lean tuned for high_0p480. But each height settles at a DIFFERENT
# equilibrium pitch, so a single offset over-corrects the low band (the
# 0.32-0.36 m heights settle at a NEGATIVE equilibrium pitch and need a
# negative offset, not +4). The Phase 1 blind 110-run height x offset sweep
# (docs/validation/height_scheduled_pitch_offset_sweep_report.md) selected the
# per-height offset that best centers signed support drift under the task
# metric (|pos%-50|, maxabs, P2P, out15%, posture/hip-yaw safety; final drift
# deliberately excluded). Verdict: HEIGHT_OFFSET_SWEEP_READY with a
# baseline-relative hip-yaw gate (the absolute 0.20 rad gate flagged behavior
# the accepted offset-0 adaptive baseline already exhibits).
#
# Selected per-height winners (raw, not smoothed — the user chose raw winners
# over a monotone fit because the low band's score-vs-offset curve is flat and
# a forced monotone ramp would discard the data-selected low-band offsets):
#   0.300 m -> +3   0.320 m -> -2   0.330 m -> -4   0.340 m ->  0
#   0.360 m -> -3   0.380 m -> +5   0.430 m -> +2   0.450 m -> +2
#   0.465 m -> +3   0.480 m -> +3
#
# Inherits ALL safety machinery from ADAPTIVE_SUPPORT_CENTERING_TRIM. The only
# differences from its parent are the profile name and the height schedule
# fields. pitch_ref_offset_deg stays 0.0 so that when the schedule is active it
# is the sole source of the offset (no double-application); the runtime uses
# the scheduled value when pitch_ref_height_schedule_enabled is True. Opt-in
# only — every other profile keeps the schedule disabled.
HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM = replace(
    ADAPTIVE_SUPPORT_CENTERING_TRIM,
    profile_name="height_scheduled_pitch_equilibrium_trim",
    pitch_ref_offset_deg=0.0,
    pitch_ref_height_schedule_enabled=True,
    pitch_ref_height_schedule_heights_m=(
        0.300, 0.320, 0.330, 0.340, 0.360, 0.380, 0.430, 0.450, 0.465, 0.480,
    ),
    pitch_ref_height_schedule_offsets_deg=(
        3.0, -2.0, -4.0, 0.0, -3.0, 5.0, 2.0, 2.0, 3.0, 3.0,
    ),
    pitch_ref_height_schedule_clamp=True,
)


# Phase B — Support-position outer-loop pitch reference.
# Opt-in dynamic centering layered on top of the frozen Phase A height schedule.
# Inherits the full HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM schedule and all
# safety machinery; only the outer_loop_* fields are turned on. The initial Kp
# below is a POSITIVE-sign placeholder consistent with the Phase A evidence
# (forward drift -> positive scheduled offset reduced it); the final sign and
# gain are selected empirically by the Phase 4 two-sign sweep and may be edited
# here. Kd starts at 0.0 (P-only) until the PD screening in Phase 4. Integral
# stays disabled. See docs/validation/support_position_outer_loop_pitch_ref_design.md.
SUPPORT_POSITION_OUTER_LOOP_PITCH_REF = replace(
    HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM,
    profile_name="support_position_outer_loop_pitch_ref",
    outer_loop_enabled=True,
    outer_loop_kp_deg_per_m=1.0,
    outer_loop_kd_deg_per_mps=0.0,
    outer_loop_ki_deg_per_m_s=0.0,
    outer_loop_integral_enabled=False,
)


# Phase B calibration — Calibrated support-position outer-loop pitch reference.
# Opt-in refinement of SUPPORT_POSITION_OUTER_LOOP_PITCH_REF (B) that replaces
# the single fixed (Kp=1.0, Kd=0.0, Ki=0.0) gains with smooth height-dependent
# functions fitted from the Phase 2 per-height gain sweep. The scalar
# outer_loop_* fields below are retained only as the OFF-RANGE fallback / default
# (they are overridden at runtime by the calibrated height functions when
# calibrated_outer_loop_enabled is True). Inherits the full frozen Phase A height
# schedule and every safety gate from B unchanged. The base B profile and all
# legacy profiles keep calibrated_outer_loop_enabled=False, so they are
# byte-for-byte unchanged. See
# wheeled_biped/controllers/calibrated_outer_loop_functions.py and
# docs/validation/calibrated_support_position_outer_loop_pitch_ref_final_report.md.
CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF = replace(
    SUPPORT_POSITION_OUTER_LOOP_PITCH_REF,
    profile_name="calibrated_support_position_outer_loop_pitch_ref",
    calibrated_outer_loop_enabled=True,
)

# Calibrated height-dependent outer loop (Phase B calibration v2, opt-in).
# v2 differs from the v1 calibrated profile only in the upper band (0.465, 0.480)
# where v1's Kp was too aggressive, causing regressions at 2000 steps. v2 lowers
# Kp(0.465) from 1.350 to 1.000 and Kp(0.480) from 1.575 to 1.050 based on
# targeted upper-band resweep results. All other breakpoints (0.300-0.450) are
# unchanged from v1. See
# wheeled_biped/controllers/calibrated_outer_loop_functions_v2.py and
# docs/validation/calibrated_outer_loop_upper_band_resweep_report.md.
CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF_V2 = replace(
    SUPPORT_POSITION_OUTER_LOOP_PITCH_REF,
    profile_name="calibrated_support_position_outer_loop_pitch_ref_v2",
    calibrated_outer_loop_enabled=True,
    calibrated_outer_loop_function_version="v2",
)

# =====================================================================
# Physics-Based Equilibrium Feedforward Outer Loop (Phase D, opt-in)
# =====================================================================
# Replaces the empirical pitch_ref_height_schedule with a physics-based
# equilibrium wheel torque feedforward. The feedforward is derived from the
# MuJoCo closed-loop equilibrium pitch at each height (without empirical offset)
# and applied directly as a wheel torque each step. This is NOT a hand-tuned
# offset — the equilibrium emerges from physics + controller Kp_pitch dynamics.
#
# Key design choices:
# - Inherits ALL safety infrastructure from CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF_V2.
# - Adds `physics_equilibrium_feedforward_enabled` flag and a height-dependent
#   `physics_eq_ff_*` schedule the runtime reads via the
#   `physics_equilibrium_feedforward` module.
# - When enabled, the controller ADDS the physics-derived DC wheel torque
#   feedforward to the final wheel torque each step, BEFORE the rate-limit and
#   low-pass stages. The controller's tau_pitch should remain approximately
#   zero in steady state (no empirical pitch_ref_offset is needed).
# - pitch_ref_height_schedule_enabled is set False; the empirical schedule is
#   not used. Telemetry emits `physics_equivalent_pitch_ref_deg = 0.0` and
#   `empirical_pitch_ref_offset_disabled = True` so callers can distinguish.
# - Disabled by default — opt-in only.
# See docs/validation/physics_equilibrium_feedforward_outer_loop_final_report.md.
PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP = replace(
    CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF_V2,
    profile_name="physics_equilibrium_feedforward_outer_loop",
    # Disable empirical pitch_ref_offset schedule
    pitch_ref_offset_deg=0.0,
    pitch_ref_height_schedule_enabled=False,
    pitch_ref_height_schedule_heights_m=(),
    pitch_ref_height_schedule_offsets_deg=(),
    # Enable physics-based equilibrium feedforward
    physics_equilibrium_feedforward_enabled=True,
    physics_eq_ff_clamp_to_height_range=True,
    # Telemetry provenance
    physics_eq_ff_function_version="1.0",
)


# Opt-in low-band support correction for the PFF Step C focused_low_0p320 case.
# The physics feedforward source and interpolation remain unchanged; this profile
# only re-enables a smooth, bounded support-position pitch-ref correction around
# 0.320 m where local telemetry shows a settling operating-point regression.
PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V1 = replace(
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP,
    profile_name="physics_equilibrium_feedforward_outer_loop_low_band_support_v1",
    outer_loop_height_schedule_required=False,
    calibrated_outer_loop_function_version="v2",
    low_band_support_outer_loop_enabled=True,
    low_band_support_center_m=0.320,
    low_band_support_sigma_m=0.006,
    low_band_support_kp_peak_deg_per_m=1.5,
    low_band_support_theta_ref_max_peak_deg=3.00,
    low_band_support_pitch_ref_offset_peak_deg=1.00,
)


# Opt-in v2 low-band support correction selected by
# physics_ff_low_band_support_v2_tuning. It preserves the same continuous
# low-band mechanism as v1 while narrowing the height support and reducing peak
# Kp to lower fixed-height low_0p320 P2P. Disabled by default.
PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2 = replace(
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP,
    profile_name="physics_equilibrium_feedforward_outer_loop_low_band_support_v2",
    outer_loop_height_schedule_required=False,
    calibrated_outer_loop_function_version="v2",
    low_band_support_outer_loop_enabled=True,
    low_band_support_center_m=0.320,
    low_band_support_sigma_m=0.004,
    low_band_support_kp_peak_deg_per_m=1.4,
    low_band_support_theta_ref_max_peak_deg=3.00,
    low_band_support_pitch_ref_offset_peak_deg=1.00,
)

# I_SUPPORT_REFERENCE_REACQUISITION_V1 — candidate I1 for support reference
# reacquisition and pitch-support limit-cycle suppression.
# Based on the low-band v2 sagittal schedule, with the critical fix that the
# low-band support Kp blends with the base (calibrated) Kp instead of replacing
# it. This ensures the support outer loop provides centering feedback at ALL
# heights, including the tall high_0p480 variant where the previous profile
# zeroed the effective Kp (height_scale ≈ 0 far from the 0.320 m low-band center).
#
# Key change vs v2: low_band_support_blend_with_base=True
# At scale=1 (near 0.320 m): Kp ≈ peak_kp = 1.4 deg/m (same as v2)
# At scale=0 (at 0.480 m): Kp ≈ base_kp = 1.050 deg/m (calibrated v2 outer loop)
# Smooth transition in between.
#
# This is an opt-in diagnostic candidate. D remains current-best.
# Disabled by default.
I_SUPPORT_REFERENCE_REACQUISITION_V1 = replace(
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    profile_name="i_support_reference_reacquisition_v1",
    low_band_support_blend_with_base=True,
)

# =====================================================================
# J_TALL_HEIGHT_SAGITTAL_WIP_DAMPING_V1 Family (Tall-height WIP damping)
# =====================================================================
# Opt-in candidate family for increasing sagittal damping at tall height
# to suppress the 2.505 Hz wheeled inverted pendulum pitch-support mode.
#
# Base: PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2
#       (same sagittal base as G1_sg080/D_MODE_HIP_YAW_DIV_V1)
#
# Mode-div parameters are applied via CLI flags (same as G1_sg080:
# kp=10, kd=0.5, mt=7.5, sl=0.30, sg=0.80).
#
# These profiles only add damping scheduling at tall heights. They do NOT
# change pitch Kp, support outer loop Kp, PFF source, or any low-height
# behavior. D remains current-best. J candidates are opt-in diagnostic only.
#
# All profiles: continuous height-scheduled damping increase above z_low,
# with smoothstep interpolation to z_high. This ensures zero change at
# low/nominal heights and progressive engagement from 0.40 m upward.
#
# J1 family — height-scheduled kd_pitch (pitch rate damping) increase at tall height.
#   Directly dampens the 2.5 Hz pitch oscillation component.
# J1a: mild increase (nominal 10.0 -> high_max 15.0)
# J1b: moderate increase (nominal 10.0 -> high_max 20.0)
# J1c: strong increase (nominal 10.0 -> high_max 30.0)
#
# J2 family — height-scheduled k_wheel_velocity increase at tall height.
#   Uses the existing continuous_k_wheel_velocity infrastructure.
#   Dampens the wheel velocity component of the WIP mode.
# J2a: mild increase (nominal 0.50 -> high_max 0.85)
# J2b: moderate increase (nominal 0.50 -> high_max 1.00)
# J2c: stronger increase (nominal 0.50 -> high_max 1.25)
#
# J3 family — Combined kd_pitch + k_wheel_velocity damping.
# J3a: mild combined (J1a + J2a)
# J3b: moderate combined (J1b + J2b)
# J3c: strong combined (J1c + J2c)
# =====================================================================

J1A_TALL_KD_PITCH_V1 = replace(
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    profile_name="j1a_tall_kd_pitch_v1",
    continuous_kd_pitch=True,
    kd_pitch_nominal=10.0,
    kd_pitch_high_max=15.0,
    kd_pitch_z_low=0.40,
    kd_pitch_z_high=0.52,
)

J1B_TALL_KD_PITCH_V1 = replace(
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    profile_name="j1b_tall_kd_pitch_v1",
    continuous_kd_pitch=True,
    kd_pitch_nominal=10.0,
    kd_pitch_high_max=20.0,
    kd_pitch_z_low=0.40,
    kd_pitch_z_high=0.52,
)

J1C_TALL_KD_PITCH_V1 = replace(
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    profile_name="j1c_tall_kd_pitch_v1",
    continuous_kd_pitch=True,
    kd_pitch_nominal=10.0,
    kd_pitch_high_max=30.0,
    kd_pitch_z_low=0.40,
    kd_pitch_z_high=0.52,
)

J2A_TALL_K_WHEEL_VEL_V1 = replace(
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    profile_name="j2a_tall_k_wheel_vel_v1",
    continuous_k_wheel_velocity=True,
    k_wheel_velocity_nominal=0.50,
    k_wheel_velocity_high_max=0.85,
    k_wheel_velocity_z_low=0.45,
    k_wheel_velocity_z_high=0.52,
)

J2B_TALL_K_WHEEL_VEL_V1 = replace(
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    profile_name="j2b_tall_k_wheel_vel_v1",
    continuous_k_wheel_velocity=True,
    k_wheel_velocity_nominal=0.50,
    k_wheel_velocity_high_max=1.00,
    k_wheel_velocity_z_low=0.45,
    k_wheel_velocity_z_high=0.52,
)

J2C_TALL_K_WHEEL_VEL_V1 = replace(
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    profile_name="j2c_tall_k_wheel_vel_v1",
    continuous_k_wheel_velocity=True,
    k_wheel_velocity_nominal=0.50,
    k_wheel_velocity_high_max=1.25,
    k_wheel_velocity_z_low=0.45,
    k_wheel_velocity_z_high=0.52,
)

J3A_TALL_COMBINED_V1 = replace(
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    profile_name="j3a_tall_combined_v1",
    continuous_kd_pitch=True,
    kd_pitch_nominal=10.0,
    kd_pitch_high_max=15.0,
    kd_pitch_z_low=0.40,
    kd_pitch_z_high=0.52,
    continuous_k_wheel_velocity=True,
    k_wheel_velocity_nominal=0.50,
    k_wheel_velocity_high_max=0.85,
    k_wheel_velocity_z_low=0.45,
    k_wheel_velocity_z_high=0.52,
)

J3B_TALL_COMBINED_V1 = replace(
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    profile_name="j3b_tall_combined_v1",
    continuous_kd_pitch=True,
    kd_pitch_nominal=10.0,
    kd_pitch_high_max=20.0,
    kd_pitch_z_low=0.40,
    kd_pitch_z_high=0.52,
    continuous_k_wheel_velocity=True,
    k_wheel_velocity_nominal=0.50,
    k_wheel_velocity_high_max=1.00,
    k_wheel_velocity_z_low=0.45,
    k_wheel_velocity_z_high=0.52,
)

# =====================================================================
# K_TARGETED_2P5HZ_WIP_NOTCH_V1 Family (Notch-filtered damping, 2.5 Hz WIP)
# =====================================================================
# Opt-in candidate family that applies a causal IIR biquad notch filter
# around the observed 2.5 Hz WIP mode to prevent phase-lagged damping
# signals from feeding the oscillation.
#
# Base: G1_sg080 (same sagittal as D_MODE_HIP_YAW_DIV_V1)
#       physics_equilibrium_feedforward_outer_loop_low_band_support_v2
#       + enable_wip_notch_filter=True + filter parameters.
#
# Mode-div parameters are applied via CLI flags (same as G1_sg080:
# kp=10, kd=0.5, mt=7.5, sl=0.30, sg=0.80).
#
# D remains current-best. K candidates are opt-in diagnostic only.
# =====================================================================

K1_PITCH_RATE_NOTCH = replace(
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    profile_name="k1_pitch_rate_notch_v1",
    enable_wip_notch_filter=True,
    wip_notch_target_signal="pitch_rate",
    wip_notch_center_hz=2.5,
    wip_notch_q=6.0,
    wip_notch_filter_blend=1.0,
    wip_notch_gate_enabled=True,
    wip_notch_height_gate_start_m=0.42,
    wip_notch_height_gate_full_m=0.48,
)
K1B_PITCH_RATE_NOTCH_2P3 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="k1b_pitch_rate_notch_2p3",
    wip_notch_center_hz=2.3,
)
K1C_PITCH_RATE_NOTCH_2P7 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="k1c_pitch_rate_notch_2p7",
    wip_notch_center_hz=2.7,
)
K1D_PITCH_RATE_NOTCH_Q4 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="k1d_pitch_rate_notch_q4",
    wip_notch_q=4.0,
)
K1E_PITCH_RATE_NOTCH_Q8 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="k1e_pitch_rate_notch_q8",
    wip_notch_q=8.0,
)
K1F_PITCH_RATE_NOTCH_BLEND075 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="k1f_pitch_rate_notch_blend075",
    wip_notch_filter_blend=0.75,
)
K1G_PITCH_RATE_NOTCH_BLEND050 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="k1g_pitch_rate_notch_blend050",
    wip_notch_filter_blend=0.50,
)
K2_NOTCH_LOW_Q_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="k2_notch_low_q_v1",
    wip_notch_q=2.0,
    # K2 JAX dedicated runner promotion — Phase 1 parameter parity fixes:
    # K2 inherits empty applies_to_variants from the SagittalAuthoritySchedule
    # default, which causes is_active_for_variant() to always return False and
    # _eff_velocity_damping_scale to stay at the baseline 1.0.  Set the intended
    # variant list and velocity_damping_scale so the canonical path matches the
    # documented K2 profile behaviour.
    applies_to_variants=(
        "low_0p300", "low_0p330", "low_0p360", "extreme_height",
        "high_0p430", "high_0p450", "high_0p465", "high_0p480",
    ),
    velocity_damping_scale=1.10,
    # APCR1ND hold_outside_band: keep the recentering-engaged state until the
    # support error falls below the inner release band (hysteresis).  The class
    # default is False (immediate release); K2 intends True for push-recovery
    # robustness.  Matches APCR1ND_T2 hold-outside-band variant.
    apcr1nd_hold_outside_band=True,
)

# ── K2 JAX Dedicated Default V1 ────────────────────────────────────────────────
# Promoted from Candidate E v2 (2026-06-30).
# Identical to K2_NOTCH_LOW_Q_V1 plus continuous pitch-damping enhancement.
# This is the OFFICIAL default controller for all future development.
K2_JAX_DEDICATED_DEFAULT_V1 = replace(
    K2_NOTCH_LOW_Q_V1,
    profile_name="k2_jax_dedicated_default_v1",
    # Phase 4 Candidate E v2: continuous pitch-damping enhancement.
    # Adds 3.0 Nm/(rad/s) additional pitch-rate damping at wheels during
    # oscillations (>2 deg/s), gated by height-velocity to avoid fighting
    # natural pitch during intentional height transitions.
    # Zero steady-state effect. Smoothstep-gated. No discrete thresholds.
    enable_pitch_damping_boost=True,
    pitch_damping_boost_kd=3.0,       # Nm/(rad/s)
    pitch_damping_rate_threshold_low=0.035,   # rad/s (2 deg/s)
    pitch_damping_rate_threshold_high=0.262,  # rad/s (15 deg/s)
    pitch_damping_height_gate_enabled=True,
)

# ── K2 JAX Dedicated Default V2 ────────────────────────────────────────────────
# Promoted from DRIFT_ITER2_VEL_ONLY_WIDE_GATE (2026-06-30).
# Identical to K2_JAX_DEDICATED_DEFAULT_V1 plus velocity-only drift damping.
#
# This is the OFFICIAL default controller for all future development.
# For rollback, use K2_JAX_DEDICATED_DEFAULT_V1.
#
# Configuration:
#   - Velocity damping only (k_vel=10.0, no heading, no position return)
#   - Wide height gate: smoothstep(0.03→0.15 m/s CoM z-velocity)
#   - Position gate remains configurable but k_pos=0 disables it
#   - No wheel-differential heading correction (known unsafe at low height)
#
# Known limitations:
#   - Does not fully solve heading/yaw drift
#   - Does not fully solve dynamic-height drift
#   - Next development should target: heading/yaw estimation, wheel asymmetry,
#     dynamic-height transition speed, dynamic-height drift, push drift decay.
K2_JAX_DEDICATED_DEFAULT_V2 = replace(
    K2_JAX_DEDICATED_DEFAULT_V1,
    profile_name="k2_jax_dedicated_default_v2",
    # Drift controller: velocity-only damping, no heading, no position return
    enable_drift_controller=True,
    drift_k_vel=10.0,              # Nm/(m/s) — validated best safe gain
    drift_k_pos=0.0,               # Nm/m — disabled (unsafe at low height)
    drift_k_heading=0.0,           # Nm/rad — disabled (unsafe at low height)
    drift_k_heading_rate=0.0,      # Nm/(rad/s) — disabled
    drift_max_tau=8.0,             # Nm per-wheel smooth tanh bound
    drift_push_damp_mult=1.5,      # Conservative push damping
    drift_hgate_low=0.03,          # CoM z-vel below 0.03 m/s → height_gate ≈ 1.0
    drift_hgate_high=0.15,         # CoM z-vel above 0.15 m/s → height_gate ≈ 0.0
    drift_pgate_low=0.15,          # drift distance below 0.15m → pos_gate ≈ 0.0
    drift_pgate_high=0.80,         # drift distance above 0.80m → pos_gate ≈ 1.0
)

# ═══════════════════════════════════════════════════════════════════════════════
# K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE
#
# Built on V2 (velocity-only drift damping). Adds:
#   - Low-authority hip-yaw heading stabilizer (no wheel differential)
#   - Anti-twist damping (reduce hip-yaw divergence)
#   - Split height gates (vel stays active during height transitions)
#   - Height transition speed control (5-10s target via S-curve)
# ═══════════════════════════════════════════════════════════════════════════════
K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE = replace(
    K2_JAX_DEDICATED_DEFAULT_V2,
    profile_name="k2_jax_dedicated_default_v2_heading_height_twist_candidate",
    # ── Heading hip-yaw stabilizer ──────────────────────────────────────────
    enable_heading_hip_yaw=True,
    heading_hy_kp=0.15,             # Nm/rad — very low, soft impedance
    heading_hy_kd=0.05,             # Nm/(rad/s) — mild damping
    heading_hy_max_tau=0.8,         # Nm per-joint smooth tanh bound
    # ── Anti-twist damping ──────────────────────────────────────────────────
    enable_anti_twist=True,
    anti_twist_kp=0.3,              # Nm/rad
    anti_twist_kd=0.1,              # Nm/(rad/s)
    anti_twist_max_tau=0.6,         # Nm per-joint
    # ── Split height gates ──────────────────────────────────────────────────
    # Velocity gate: WIDER — stays active during controlled height motion
    drift_hgate_vel_low=0.05,       # below 0.05 m/s → gate ≈ 1.0
    drift_hgate_vel_high=0.25,      # above 0.25 m/s → gate ≈ 0.0
    # Heading gate: NARROWER — reduces quickly during height transitions
    drift_hgate_heading_low=0.02,   # below 0.02 m/s → gate ≈ 1.0
    drift_hgate_heading_high=0.10,  # above 0.10 m/s → gate ≈ 0.0
    # Height transition speed
    height_transition_duration_s=8.0,
)

# ═══════════════════════════════════════════════════════════════════════════════
# K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V2
#
# Built on HHT Candidate with structural fixes from long telemetry diagnosis:
#   - TASK 1: Differential hip-yaw heading torque (left=+tau, right=-tau)
#            instead of symmetric (non-functional) torque.
#   - TASK 2: Reduced anti-twist authority to avoid fighting heading correction:
#            kp: 0.30→0.15, max_tau: 0.6→0.3 Nm
#   - TASK 3: Added weak hip-yaw mean centering to prevent outward leg drift
#   - TASK 4: (runtime-only) Multi-segment dynamic cycle q_ref fix
#   - TASK 5: Drift gates retuned after conflict resolution
#
# Hard principles:
#   - Velocity-only drift damping (V2). No wheel-differential heading.
#   - Continuous gates only. No discrete height buckets.
#   - No scenario-specific hacks. No silent profile fallback.
#   - Balance priority > drift/heading/yaw. No excessive leg twisting.
# ═══════════════════════════════════════════════════════════════════════════════
K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V2 = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE,
    profile_name="k2_jax_dedicated_default_v2_heading_height_twist_candidate_v2",
    # ── TASK 1: Stronger heading hip-yaw with differential torque ──────────
    # Sign convention: left=+tau, right=-tau → CW yaw moment (validated via
    # controlled yaw-error injection test).
    enable_heading_hip_yaw=True,
    heading_hy_kp=0.40,             # Nm/rad — increased from 0.15 for authority
    heading_hy_kd=0.10,             # Nm/(rad/s) — increased from 0.05
    heading_hy_max_tau=1.5,         # Nm per-joint — increased from 0.8
    # ── TASK 2: Reduced anti-twist authority ───────────────────────────────
    # Anti-twist should damp excessive divergence, not fight yaw correction.
    enable_anti_twist=True,
    anti_twist_kp=0.15,             # Nm/rad — reduced from 0.30
    anti_twist_kd=0.1,              # Nm/(rad/s) — unchanged (conservative)
    anti_twist_max_tau=0.3,         # Nm per-joint — reduced from 0.6
    # ── TASK 3: Weak hip-yaw mean centering ────────────────────────────────
    hy_mean_center_kp=0.5,          # Nm/rad — very weak centering
    hy_mean_center_max_tau=0.4,     # Nm per-joint smooth tanh bound
    # ── TASK 5: Slightly widened velocity gate (was over-gated 3-8x) ──────
    # After heading/anti-twist conflict fix, velocity damping can breathe.
    # Heading and position gates remain conservative. No wheel-diff heading.
    drift_hgate_vel_low=0.08,       # widened from 0.05
    drift_hgate_vel_high=0.35,      # widened from 0.25
    drift_hgate_heading_low=0.02,   # conservative (unchanged)
    drift_hgate_heading_high=0.10,  # conservative (unchanged)
    # Height transition speed
    height_transition_duration_s=8.0,
)

# ═══════════════════════════════════════════════════════════════════════════════
# K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3
#
# Built on V2 Candidate with three structural fixes from V2 telemetry:
#   - TASK 1: Widened heading stability gate (pitch full-gate 0.07 rad instead
#            of 0.035 rad) so heading torque can activate during normal balance.
#            Twist gate widened (full-gate 0.10 instead of 0.04 rad) so heading
#            can operate at typical divergence levels.
#   - TASK 2: Differential heading sign validated via telemetry correlation.
#   - TASK 3: (runtime-only) Multi-segment q_ref boundary blending.
#   - TASK 4: Soft divergence guard added to anti-twist damping.
#            Progressive boost (up to 3.5x) at divergence 0.22→0.32 rad.
#
# All V2 values preserved except heading gate thresholds and divergence guard.
# Drift gains unchanged (Task 5).
# ═══════════════════════════════════════════════════════════════════════════════
K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3 = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V2,
    profile_name="k2_jax_dedicated_default_v2_heading_height_twist_candidate_v3",
    # ── TASK 1: Same heading gains, gate thresholds relaxed at call site ─────
    # (heading_hy_kp/kd/max_tau unchanged from V2: 0.40, 0.10, 1.5)
    # (anti_twist_kp/kd/max_tau unchanged from V2: 0.15, 0.1, 0.3)
    # (hy_mean_center_kp/max_tau unchanged from V2: 0.5, 0.4)
    # (drift_hgate values unchanged from V2: vel 0.08/0.35, heading 0.02/0.10)
    # (height_transition_duration_s unchanged: 8.0)
    # The heading gate threshold changes are at the JAX call site in
    # k2_jax_controller.py, not parameterized in the profile.
    #
    # Pitch stability gate: full at 0.07 rad (was 0.035), zero at 0.21 rad
    # Roll stability gate: full at 0.035 rad (was 0.017), zero at 0.122 rad
    # Twist gate in heading: full at 0.10 rad (was 0.04), zero at 0.30 rad
    # Divergence guard boost: activates at 0.22 rad, 3.5x at 0.32 rad
)

# ═══════════════════════════════════════════════════════════════════════════════
# K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V4
#
# Built on V3 Candidate with two structural fixes from V3 telemetry:
#   - TASK 1: Dynamic cycle q_ref blend retuned from 40% dynamic/60% static
#            to 60% dynamic/40% static for better height tracking.
#            Boundary blending (60-step smoothstep) preserved.
#   - TASK 2: Divergence guard strengthened:
#            * Activation lowered from 0.22 rad to 0.18 rad
#            * Full boost increased from 3.5x to 5.0x at 0.30 rad (was 0.32)
#            * Heading twist yield gate added: yields heading at 0.18→0.35 rad
#   - TASK 3: Heading fix preserved — same gains, same differential sign.
#   - TASK 4: Realtime verified with no-telemetry runs.
#
# All V3 values preserved: heading gains (0.40, 0.10, 1.5), anti-twist base
# (0.15, 0.1, 0.3), mean centering (0.5, 0.4), drift gates unchanged.
# Divergence guard thresholds changed at JAX call site in k2_jax_controller.py.
# ═══════════════════════════════════════════════════════════════════════════════
K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V4 = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3,
    profile_name="k2_jax_dedicated_default_v2_heading_height_twist_candidate_v4",
    # ── TASK 1: Dynamic cycle blend retuned ─────────────────────────────────
    dynamic_q_ref_blend_alpha=0.60,  # 60% dynamic, 40% static (was 0.40 in V3)
    # ── TASK 2: Divergence guard strengthened ──────────────────────────────
    # Guard activation: 0.18 rad (was 0.22 rad in V3)
    # Guard boost: 5.0x max (was 3.5x in V3), full at 0.30 rad (was 0.32 in V3)
    anti_twist_guard_start_rad=0.18,
    anti_twist_guard_strong_rad=0.30,
    anti_twist_guard_boost_max=5.0,
    # Heading twist yield: active at 0.18→0.35 rad (was disabled in V3)
    heading_twist_yield_start_rad=0.18,
    heading_twist_yield_zero_rad=0.35,
    # ── TASK 3: Heading fix preserved ─────────────────────────────────────
    # Same gains, same differential sign, same gate thresholds as V3
    # (heading_hy_kp/kd/max_tau unchanged: 0.40, 0.10, 1.5)
    # (anti_twist_kp/kd/max_tau unchanged: 0.15, 0.1, 0.3)
    # (hy_mean_center_kp/max_tau unchanged: 0.5, 0.4)
    # (drift_hgate values unchanged from V2: vel 0.08/0.35, heading 0.02/0.10)
)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 ABLATION PROFILES — HHT V4 Regression Root-Cause Analysis
# ═══════════════════════════════════════════════════════════════════════════════

# ── Ablation A: HHT_ABLATE_V3_BASE ─────────────────────────────────────────
# Exact V3 behavior. All V3 guard thresholds, no heading twist yield, 40% blend.
# Purpose: baseline reference for ablation comparison.
HHT_ABLATE_V3_BASE = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3,
    profile_name="hht_ablate_v3_base",
    # V3 guard: 0.22→0.32 rad, 3.5x boost
    anti_twist_guard_start_rad=0.22,
    anti_twist_guard_strong_rad=0.32,
    anti_twist_guard_boost_max=3.5,
    # V3: no heading twist yield (disabled)
    heading_twist_yield_start_rad=0.35,
    heading_twist_yield_zero_rad=0.35,
    # V3: 40% dynamic blend
    dynamic_q_ref_blend_alpha=0.40,
)

# ── Ablation B: HHT_ABLATE_V3_PLUS_60_40_BLEND ────────────────────────────
# V3 behavior + only the 60/40 q_ref blend change.
# No divergence guard changes, no heading twist yield changes.
# Purpose: determine whether 60/40 blend is safe by itself.
HHT_ABLATE_V3_PLUS_60_40_BLEND = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3,
    profile_name="hht_ablate_v3_plus_60_40_blend",
    # V3 guard: 0.22→0.32 rad, 3.5x boost
    anti_twist_guard_start_rad=0.22,
    anti_twist_guard_strong_rad=0.32,
    anti_twist_guard_boost_max=3.5,
    # V3: no heading twist yield (disabled)
    heading_twist_yield_start_rad=0.35,
    heading_twist_yield_zero_rad=0.35,
    # V4: 60% dynamic blend (only V4 change applied)
    dynamic_q_ref_blend_alpha=0.60,
)

# ── Ablation C: HHT_ABLATE_V4_NO_GUARD_CHANGE ─────────────────────────────
# V4 but with divergence guard rolled back to V3 values.
# Keeps 60/40 blend and heading twist yield.
# Purpose: check whether strengthened divergence guard caused push fall.
HHT_ABLATE_V4_NO_GUARD_CHANGE = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V4,
    profile_name="hht_ablate_v4_no_guard_change",
    # Rollback guard to V3: 0.22→0.32 rad, 3.5x boost
    anti_twist_guard_start_rad=0.22,
    anti_twist_guard_strong_rad=0.32,
    anti_twist_guard_boost_max=3.5,
    # Keep V4 heading twist yield
    # Keep V4 60/40 blend
)

# ── Ablation D: HHT_ABLATE_V4_NO_HEADING_TWIST_YIELD ─────────────────────
# V4 but with heading twist yield gate disabled (set to V3 behavior).
# Keeps 60/40 blend and V4 guard.
# Purpose: check whether heading yield during high twist caused push regression.
HHT_ABLATE_V4_NO_HEADING_TWIST_YIELD = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V4,
    profile_name="hht_ablate_v4_no_heading_twist_yield",
    # Disable heading twist yield (V3 behavior: start >= zero)
    heading_twist_yield_start_rad=0.35,
    heading_twist_yield_zero_rad=0.35,
    # Keep V4 guard: 0.18→0.30, 5.0x
    # Keep V4 60/40 blend
)

# ── Ablation E: HHT_ABLATE_GUARD_CAP_TEST ────────────────────────────────
# V4 guard thresholds/boost, but increase anti_twist_max_tau from 0.3→0.45 Nm.
# Tests the theory that the 0.3 Nm tanh cap bottleneck prevents guard from working.
# Purpose: check if raising the torque cap allows the guard boost to take effect.
HHT_ABLATE_GUARD_CAP_TEST = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V4,
    profile_name="hht_ablate_guard_cap_test",
    # V4 guard: 0.18→0.30, 5.0x
    # V4 heading twist yield
    # V4 60/40 blend
    # Increased anti-twist torque cap
    anti_twist_max_tau=0.45,  # was 0.3 in V4/V3
)

# ═══════════════════════════════════════════════════════════════════════════════
# K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V5
#
# Built on V3 base + ablation-proven improvements:
#   - TASK 1: 60/40 dynamic q_ref blend (proven safe by ablation B:
#             +3.2 cm height, -55% displacement, -0.033 rad div, -0.2° pitch RMS).
#             Will test 65/35 and 70/30 blends for further tracking gains.
#   - TASK 2: Two-layer divergence guard (fixes V4 bottleneck):
#             * Layer 1: V3 base anti-twist (kp=0.15, max_tau=0.3, own tanh channel)
#             * Layer 2: Emergency guard at 0.28→0.34 rad, boost 3.5x, separate
#               tanh cap 0.25 Nm — never squeezed by Layer 1 cap.
#             * Do NOT multiply base kp into same tanh cap (V4 bottleneck).
#   - TASK 3: Push regression investigation — V4 fall was non-deterministic.
#             V5's delayed heading yield (0.30→0.38 rad, not 0.18→0.35) avoids
#             suppressing heading torque during normal push recovery.
#   - TASK 4: Heading sign/gate from V3 preserved. No gain increase.
#
# Hard principles: V3 base, 60/40 blend, two-layer emergency guard (0.28→0.34),
# heading yield delayed to 0.30→0.38 rad, no heading gain increase.
# ═══════════════════════════════════════════════════════════════════════════════
K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V5 = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3,
    profile_name="k2_jax_dedicated_default_v2_heading_height_twist_candidate_v5",
    # ── TASK 1: 60/40 dynamic q_ref blend (proven safe) ────────────────────
    dynamic_q_ref_blend_alpha=0.60,  # 60% dynamic, 40% static
    # ── TASK 2: Two-layer guard (V3 base + emergency extra) ───────────────
    # Layer 1: V3 guard behavior — boost kp at 0.22→0.32 rad, 3.5x, tanh 0.3 Nm
    # Layer 2: Emergency extra on separate tanh channel — never squeezed by base cap
    # The guard gate controls BOTH boost multiplier and emergency extra activation.
    anti_twist_guard_start_rad=0.22,    # V3 guard activation (was 0.28 — left gap)
    anti_twist_guard_strong_rad=0.32,   # V3 guard full (was 0.34)
    anti_twist_guard_boost_max=3.5,     # 3.5x extra kp at full gate
    anti_twist_emergency_max_tau=0.25,  # Nm — separate tanh cap for emergency extra
    # ── TASK 3: Delayed heading twist yield ───────────────────────────────
    # Heading yields only after divergence enters emergency region (0.30 rad),
    # fully suppressed at 0.38 rad. Maintains heading torque during normal recovery.
    heading_twist_yield_start_rad=0.30,  # rad — yield activation (was 0.18 in V4)
    heading_twist_yield_zero_rad=0.38,   # rad — fully suppressed (was 0.35 in V4)
    # ── TASK 4: Heading fix preserved ─────────────────────────────────────
    # Same gains, same differential sign: heading_hy_kp=0.40, kd=0.10, max_tau=1.5
    # anti_twist_kp=0.15, kd=0.1, max_tau=0.3, hy_mean_center_kp=0.5, max_tau=0.4
    # drift_hgate unchanged: vel 0.08/0.35, heading 0.02/0.10
)

# ═══════════════════════════════════════════════════════════════════════════════
# V3 AUDIT FIX CANDIDATE — evidence-backed fixes from V3 root-cause analysis
#
# Root Cause #1: Drift height gate uses position error (cm) but compares against
# thresholds 0.08-0.35 cm (0.8-3.5 mm). At typical height errors of 0.5-2 cm,
# the gate is always zero → drift velocity damping disabled for >99.8% of steps.
# Fix: Widen thresholds to 2-12 cm so gate stays open during normal tracking.
#
# Root Cause #2: Heading kp=0.40 generates <0.03 Nm peak torque at 5° yaw error.
# Fix: Increase heading kp modestly to 1.0 (2.5x) so heading generates meaningful
# correction without over-authority. Keep kd and max_tau unchanged.
#
# Root Cause #3: Dynamic q_ref blend 40/60 (V3) limits height tracking to 0.404 m
# with 0.48 m target. V4's 60/40 blend fixed this without safety regression.
# Fix: Adopt 60/40 blend (proven safe in HHT_ABLATE_V3_PLUS_60_40_BLEND tests).
#
# No other changes from V3. All gates remain continuous.
# ═══════════════════════════════════════════════════════════════════════════════
K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3,
    profile_name="k2_jax_dedicated_default_v2_heading_height_twist_candidate_v3_audit_fix",
    # ── Fix #1: Drift height gate thresholds widened for position-error units ──
    # com_z_vel_abs in drift controller = |height_error| * 100 (cm).
    # Old: hgate_vel 0.08→0.35 cm (0.8→3.5 mm) — always zero.
    # New: hgate_vel 2.0→12.0 cm (2→12 cm) — gate open during normal tracking.
    drift_hgate_vel_low=2.0,        # full gate below 2 cm height error
    drift_hgate_vel_high=12.0,      # zero gate above 12 cm height error
    # ── Fix #2: Stronger heading hip-yaw gain ──────────────────────────────────
    heading_hy_kp=1.0,              # Nm/rad — 2.5x V3 (0.40→1.0)
    # heading_hy_kd=0.10 unchanged — conservative
    # heading_hy_max_tau=1.5 unchanged — already sufficient
    # ── Fix #3: 60/40 dynamic q_ref blend for height tracking ─────────────────
    dynamic_q_ref_blend_alpha=0.60,  # 60% dynamic, 40% static (V4-proven value)
    # ── All other V3 parameters preserved ──────────────────────────────────────
)

# ═══════════════════════════════════════════════════════════════════════════════
# V3 AUDIT FIX V2 — heading gain midpoint to address lateral drift regression
# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT_FIX_V2 builds from AUDIT_FIX with one change: heading_hy_kp=0.70
# (midpoint between V3's 0.40 and AUDIT_FIX's 1.0).
#
# Rationale: AUDIT_FIX (kp=1.0) showed lateral drift regression (+311%)
# and push yaw worsening (+15%). Stronger differential hip-yaw heading
# torque may be injecting lateral/yaw-side forces. kp=0.70 tests whether
# the regression scales proportionally with gain.
#
# Preserved from AUDIT_FIX:
#   drift_hgate_vel_low  = 2.0
#   drift_hgate_vel_high = 12.0
#   dynamic_q_ref_blend_alpha = 0.60
#
# No other changes. All gates remain continuous, no scenario hacks.
K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX_V2 = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3,
    profile_name="k2_jax_dedicated_default_v2_heading_height_twist_candidate_v3_audit_fix_v2",
    # ── Fix #1: Drift height gate thresholds widened for position-error units ──
    drift_hgate_vel_low=2.0,        # full gate below 2 cm height error
    drift_hgate_vel_high=12.0,      # zero gate above 12 cm height error
    # ── Fix #2: Heading hip-yaw gain at midpoint (0.70 vs V3=0.40, AUDIT_FIX=1.0)
    heading_hy_kp=0.70,             # Nm/rad — 1.75x V3, 0.70x AUDIT_FIX
    # heading_hy_kd=0.10 unchanged
    # heading_hy_max_tau=1.5 unchanged
    # ── Fix #3: 60/40 dynamic q_ref blend for height tracking ─────────────────
    dynamic_q_ref_blend_alpha=0.60,  # 60% dynamic, 40% static
    # ── All other V3 parameters preserved ──────────────────────────────────────
)

# ═══════════════════════════════════════════════════════════════════════════════
# V3 AUDIT FIX V2 FINAL — promote candidate from 5-point micro-ablation
# ═══════════════════════════════════════════════════════════════════════════════
# Micro-ablation across kp ∈ {0.40, 0.55, 0.70, 0.85, 1.00} revealed that kp=0.55
# achieves near-zero yaw error (-0.50° at fixed 0.400 m) with lateral drift
# nearly identical to V3 (-0.030 m vs -0.028 m). This is NON-MONOTONIC — kp=0.70
# is WORSE than both kp=0.55 and kp=0.85 for fixed-height yaw.
#
# Three evidence-backed changes from V3:
#   1. drift_hgate_vel_low/high:  0.08/0.35 → 2.0/12.0  (fix disabled drift gate)
#   2. heading_hy_kp:             0.40       → 0.55       (optimal from 5-pt sweep)
#   3. dynamic_q_ref_blend_alpha: 0.40       → 0.60       (better height tracking)
#
# All other V3 parameters preserved.
K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX_V2_FINAL = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3,
    profile_name="k2_jax_dedicated_default_v2_heading_height_twist_candidate_v3_audit_fix_v2_final",
    # ── Fix #1: Drift height gate thresholds widened for position-error units ──
    drift_hgate_vel_low=2.0,        # full gate below 2 cm height error
    drift_hgate_vel_high=12.0,      # zero gate above 12 cm height error
    # ── Fix #2: Heading hip-yaw gain — optimal from 5-point micro-ablation ─────
    heading_hy_kp=0.55,             # Nm/rad — 1.375x V3, near-zero yaw at fixed ht
    # heading_hy_kd=0.10 unchanged
    # heading_hy_max_tau=1.5 unchanged
    # ── Fix #3: 60/40 dynamic q_ref blend for height tracking ─────────────────
    dynamic_q_ref_blend_alpha=0.60,  # 60% dynamic, 40% static
    # ── All other V3 parameters preserved ──────────────────────────────────────
)

# ═══════════════════════════════════════════════════════════════════════════════
# K2_JAX_DEDICATED_DEFAULT_V3 — OFFICIAL DEFAULT (promoted 2026-07-01)
# ═══════════════════════════════════════════════════════════════════════════════
# Promoted from V3_AUDIT_FIX_V2_FINAL after comprehensive validation:
#   - 5-point heading-gain micro-ablation identified kp=0.55 as optimal
#   - 37/37 validation scenarios survive, 0 falls, 0 SAFETY_FAIL
#   - Fixed-height yaw: -0.50° vs V2/V3's 5.27° (NEAR-ZERO)
#   - Lateral drift at mid height: -0.030m (nearly identical to V3's -0.028m)
#   - Dynamic height: 0.436m max vs V3's 0.404m (+8%)
#   - Dynamic displacement: 1.37m vs V3's 3.09m (-56%)
#   - Drift height gate: 100% operational vs V3's 0.2%
#   - Performance: 121-127 Hz without telemetry
#
# Three evidence-backed changes from V3:
#   1. drift_hgate_vel_low/high:  0.08/0.35 → 2.0/12.0  (fix disabled drift gate)
#   2. heading_hy_kp:             0.40       → 0.55       (optimal from 5-pt sweep)
#   3. dynamic_q_ref_blend_alpha: 0.40       → 0.60       (better height tracking)
#
# Rollback: use --profile K2_JAX_DEDICATED_DEFAULT_V2
K2_JAX_DEDICATED_DEFAULT_V3 = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX_V2_FINAL,
    profile_name="k2_jax_dedicated_default_v3",
)

# ═══════════════════════════════════════════════════════════════════════════════
# K2_JAX_DEDICATED_DEFAULT_V3_HOMING — rollback (default 2026-07-19 → 2026-07-21,
# superseded by K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR)
# V3 + post-push HOMING (audit F5/F12). Realtime runner defaults to this profile;
# rollback to K2_JAX_DEDICATED_DEFAULT_V3 (no homing) if needed. Validated: quick
# 48-scenario suite 0 falls and ≥ WBC-assist on all metrics; push recovery returns
# legs+heading to nominal; 20 s stand-up/sit-down height sweep tracks CoM to ~3 mm.
# ═══════════════════════════════════════════════════════════════════════════════
# V3 recovers from pushes without falling but does NOT return to the initial pose:
#   - legs stay splayed (hip_roll abducted: V3 posture kp_hip_roll=0 → no restoring)
#   - hip_yaw scissored (friction-pinned)
#   - body keeps a residual yaw offset and the wheels keep rolling (no yaw/position
#     return loop — drift_k_pos/k_heading were 0)
# This profile adds, all STABILITY-GATED (inactive during a disturbance):
#   - posture homing: hip_roll + hip_yaw restoring PD toward nominal q_ref (F12)
#   - wheel-differential heading return (F6-b sign fix) + sagittal position return (F5)
#   - widened drift heading height-gate so the return activates during normal balance
# Mechanism verified: hip_roll splay dev 0.154→0.017 rad with homing; wheel-diff
# yaw gain has strong authority (2 Nm → >100°). Gains kept conservative.
K2_JAX_DEDICATED_DEFAULT_V3_HOMING = replace(
    K2_JAX_DEDICATED_DEFAULT_V3,
    profile_name="k2_jax_dedicated_default_v3_homing",
    # ── F12: posture homing (un-splay legs when settled) ──
    # Gains from a 2D sweep on r_thigh 90N: hip_roll dev 0.2°/joint, hip_yaw
    # 2.3°/joint (baseline 4.3°/~9°). kp_hip_roll kept high enough to hold hip_roll
    # against the coupling from the strong hip_yaw restoring.
    enable_posture_homing=True,
    homing_kp_hip_roll=15.0,   # Nm/rad — closes hip_roll abduction (was 0 in V3)
    homing_kp_hip_yaw=25.0,    # Nm/rad — relieves hip_yaw scissor (friction-limited)
    homing_max_tau=10.0,       # Nm per-joint bound
    # ── F5: yaw + sagittal position return via wheels (drift controller) ──
    drift_k_heading=2.0,       # Nm/rad — wheel-differential yaw return (F6-b sign)
    drift_k_heading_rate=0.6,  # Nm/(rad/s) — yaw-rate damping
    drift_k_pos=2.0,           # Nm/m — weak sagittal position return
    drift_hgate_heading_low=2.0,   # widen heading height-gate to cm scale (F10)
    drift_hgate_heading_high=12.0,
)

# ═══════════════════════════════════════════════════════════════════════════════
# K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR — OFFICIAL DEFAULT (promoted 2026-07-21)
# V3_HOMING + anchored standing. Promotion suite (--quick, 48 scenarios):
# 0 falls, 48× ASSIST_EQUIVALENT. Rollback: K2_JAX_DEDICATED_DEFAULT_V3_HOMING.
# ═══════════════════════════════════════════════════════════════════════════════
# Root cause (instrumented idle, 2026-07-21): with P-only position control the
# robot parks ~6 cm from home where tau_position (+1.37 Nm mean) cancels the
# untrimmed equilibrium-pitch bias (tau_pitch −1.23 Nm mean; ABS trim caps at
# 0.35 Nm), and oscillates ±4–6 cm (limit cycle, under-damped).
# This profile makes the latched home (position/heading/posture) a true anchor:
#   - anchor position integral: supplies the bias torque → steady-state error → 0
#   - raised velocity_damping_scale: shrinks the idle limit-cycle amplitude
# Balance keeps absolute priority: integral adaptation freezes (continuous
# gates) when tilted/pushed or during height transitions; heading return,
# posture homing, and far-range drift return are inherited from V3_HOMING.
K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR = replace(
    K2_JAX_DEDICATED_DEFAULT_V3_HOMING,
    profile_name="k2_jax_dedicated_default_v3_anchor",
    # ARCHITECTURE: anchor = HOMING + prox(|err|)·[integral + damping boost].
    # Every anchor mechanism is confined to the anchor neighborhood (master
    # proximity gate, full ≤5 cm / off ≥15 cm) — outside it the controller IS
    # V3_HOMING, whose displaced-return behavior is proven. Modifying the
    # displaced regime (error leash/tanh shaping) destabilized it every time.
    # Position integral stiffness. History: 8.0 → 4.0 (weaker spring = fewer
    # idle oscillation cycles). A 2026-08-01 campaign tried to raise it and
    # was REJECTED on measured evidence — do not raise it again without
    # repeating that campaign.
    #
    # The leaky integral has finite DC gain k_I,dc = ki·dt·(1−λ)/λ, so the
    # standing bias torque leaves a steady-state sagittal offset ∝ 1/ki. The
    # offset shrinks exactly as predicted (N=5 each, clean):
    #     ki       4      20      40      80     160
    #     k_I,dc  7.96   39.8    79.6   159.2   318.4   Nm/m
    #     offset -27.0  -17.5   -12.5    -8.4    -5.5   mm
    # but the robustness sweep's two noise+delay cells (med noise × 10/30 ms,
    # N=20 per cell, identical seeds across arms) price it:
    #     falls   7/40  13/40   10/40   14/40   17/40
    # Cochran–Armitage trend on log2(ki): z=2.34, p=0.019; ki=160 alone is
    # Fisher p=0.027 vs ki=4. The cost is graded, not a threshold, so no
    # intermediate value is safe — ki=40 looks clean (p=0.59) only because
    # N=40 cannot resolve 17.5% from 25%.
    #
    # Mechanism: H(s) = ki·dt/(λ + s·dt) — raising ki lifts DC gain AND
    # mid-band gain (with 90° lag) in the band the 10–30 ms actuator delay
    # already eats. Lowering λ lifts DC gain alone, but stretches forgetting
    # to 20 s and degraded window jitter 12× (0.298 → 3.463 mm). Separating
    # DC gain from mid-band gain needs a lead/filtered-PI restructure and a
    # full 48-scenario re-qualification, not a constant change.
    #
    # All zero-delay cells stay clean at every ki, and idle/push/flight/
    # terrain/rate-limit metrics were flat — the regression is specific to
    # noise AND delay together. Note idle_rms at high ki looks *better* in
    # the failing cells purely from survivor bias (fewer trials contribute).
    anchor_position_ki=4.0,
    anchor_integral_cap_nm=2.0,         # Nm — covers the ~1.3 Nm standing bias
    anchor_integral_leak_per_step=5e-3,  # ~2 s forgetting; prevents windup
    anchor_kvel_boost_scale=5.0,         # GATED quiet-stance boost (was 1.9)
    drift_k_vel=15.0,                    # was 10.0; drift sagittal damping
    # UNGATED base damping kept low for clean acceleration.
    # 15 × 1.5 = 22.5 Nm/(m/s). Settle ~5.4s, driving clean (no ripple).
    velocity_damping_scale=1.5,
    # Apply to ALL height variants (not just low-height ones). V3_ANCHOR is
    # the anchor profile — the velocity damping scale should apply everywhere.
    applies_to_variants=("nominal", "low_tiny", "high_tiny", "low_small",
                         "high_small", "low_0p300", "low_0p330", "low_0p360",
                         "extreme_height"),
    # Pitch-stiffness schedule: soft (35) while recovering → wider push capture
    # (360° min 40→60 N, median 75→90 N); returns to stiff (50) once settled so
    # idle stand-still + ringdown decay are preserved. Gated by the slow
    # quiet-stance envelope (no cycle-frequency modulation).
    anchor_kp_pitch_soft=35.0,
)

# ═══════════════════════════════════════════════════════════════════════════════
# ACC_DELAY_HARDENED — evaluation variant, NOT promoted (2026-08-02)
# ═══════════════════════════════════════════════════════════════════════════════
# V3_ANCHOR with the ungated sagittal velocity damping doubled (15 × 1.5 = 22.5
# → 15 × 3.0 = 45 Nm/(m/s)).  Exists to price the actuator-delay cliff; the
# shipped default is unchanged and remains K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR.
#
# Motivation: a substep-resolution delay sweep (scripts/delay_cliff_resolution.py,
# 2 ms granularity) put the push-margin knee between 6 and 8 ms — BELOW the
# ~9.5 ms end-to-end budget estimated for a non-RTOS build. A one-knob screen at
# 8 ms over four profile constants (scripts/delay_retune_sweep.py) found the
# shipped gain set sitting in a sharp local dip: F_max 51.0 N at the shipped
# value against 78–93 N at almost every neighbouring value of every knob, in
# BOTH directions. That shape says phase alignment, not an exhausted margin.
#
# Measured, clean sensing, F_max (N), N=1 (this harness is deterministic clean):
#   delay ms      0     2     4     6     8    10    12    14    16    20
#   vds=1.5    95.5  94.4  93.2  93.2  51.0  56.9  41.6  40.5  28.8  10.0
#   vds=3.0    95.5  94.4  94.4  94.4  93.2  60.4  79.1  68.6  41.6  10.0
# → knee moves 6–8 ms → 8–10 ms; never worse at any delay; identical at 0–2 ms.
# Both collapse to the 10 N search floor at 20 ms, so this buys headroom below
# the cliff, not a different failure mode.
#
# Price measured so far: idle lateral RMS 0.790 → 0.801 mm (+1.4%, N=3, clean),
# i.e. none resolvable. Under the noise+delay conjunction that is ACC's binding
# constraint it is BETTER, not worse (8 ms, N=5: low 76.8 → 92.0, medium
# 84.3 → 90.2 N).
#
# Why this is NOT promoted: velocity_damping_scale is the constant the V3_ANCHOR
# comment above ties to "clean acceleration ... settle ~5.4 s, driving clean (no
# ripple)". Doubling it is exactly the kind of change the 48-scenario promotion
# suite exists to adjudicate, and driving/terrain/flight behaviour has NOT been
# re-measured here. Promotion requires that suite.
ACC_DELAY_HARDENED = replace(
    K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR,
    profile_name="acc_delay_hardened",
    velocity_damping_scale=3.0,
)

# ═══════════════════════════════════════════════════════════════════════════════
# V3 AUDIT FIX V2 MICRO-ABLATIONS — heading gain sweep for tradeoff mapping
# ═══════════════════════════════════════════════════════════════════════════════
# Two micro-ablation profiles to fit the heading-gain tradeoff curve:
#  - kp=0.55 between V3(0.40) and V2(0.70)
#  - kp=0.85 between V2(0.70) and FIX(1.0)
# Used only if kp=0.70 push yaw is worse than both extremes.
# All other AUDIT_FIX corrections preserved.
V3_AUDIT_FIX_KP_055 = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX,
    profile_name="v3_audit_fix_kp_055",
    heading_hy_kp=0.55,              # Nm/rad — 1.375x V3, ~midpoint V3↔V2
)
V3_AUDIT_FIX_KP_085 = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX,
    profile_name="v3_audit_fix_kp_085",
    heading_hy_kp=0.85,              # Nm/rad — ~midpoint V2↔FIX
)

# ═══════════════════════════════════════════════════════════════════════════════
# V3 AUDIT ABLATIONS — single-factor changes for root-cause validation
# ═══════════════════════════════════════════════════════════════════════════════

# A: Disable heading hip-yaw (isolate heading contribution to yaw/twist)
V3_AUDIT_HEADING_OFF = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3,
    profile_name="v3_audit_heading_off",
    enable_heading_hip_yaw=False,
)

# D: Heading gate always open under normal pitch/roll (test if heading torque
#    is effective when not gated by the twist/error sub-gates)
V3_AUDIT_HEADING_GATE_OPEN = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3,
    profile_name="v3_audit_heading_gate_open",
    # Use V3 base with inherently more open gates — the gate thresholds
    # are at the JAX call site; this profile variant exists for runner
    # identification. Actual changes in k2_jax_controller.py call site.
)

# F: Dynamic q_ref 100% static (test whether static anchor causes height tracking failure)
V3_AUDIT_QREF_STATIC = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3,
    profile_name="v3_audit_qref_static",
    dynamic_q_ref_blend_alpha=0.0,  # 100% static equilibrium anchor
)

# G: Dynamic q_ref 100% dynamic (test whether dynamic-only q_ref is stable)
V3_AUDIT_QREF_DYNAMIC = replace(
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3,
    profile_name="v3_audit_qref_dynamic",
    dynamic_q_ref_blend_alpha=1.0,  # 100% dynamic q_ref
)

# ── K2 JAX Dedicated Default V1 + Drift Controller ────────────────────────────
# Candidate: adds coordinated wheel-torque drift correction with continuous
# stability/height/contact/hip-yaw gating. All gates are smoothstep — no hard
# thresholds, no scenario flags, no lateral pseudo-force.
# Ablation-safe: set enable_drift_controller=False to revert to DEFAULT_V1
# through the same code path.
K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED = replace(
    K2_JAX_DEDICATED_DEFAULT_V1,
    profile_name="k2_jax_dedicated_default_v1_drift_fixed",
    # Drift controller: coordinated wheel-torque correction
    enable_drift_controller=True,
    drift_k_vel=6.0,              # Nm/(m/s) — conservative first pass
    drift_k_pos=1.5,              # Nm/m — intentionally weak
    drift_k_heading=3.0,          # Nm/rad
    drift_k_heading_rate=0.8,     # Nm/(rad/s)
    drift_push_damp_mult=1.5,     # max 2.5x damping during push-like states
    drift_max_tau=5.0,            # Nm per-wheel smooth tanh bound
    drift_hgate_low=0.03,         # CoM z-vel below 0.03 m/s → height_gate ≈ 1.0
    drift_hgate_high=0.15,        # CoM z-vel above 0.15 m/s → height_gate ≈ 0.0
    drift_pgate_low=0.15,         # drift distance below 0.15m → pos_gate ≈ 0.0
    drift_pgate_high=0.80,        # drift distance above 0.80m → pos_gate ≈ 1.0
)

# ─── Drift Iteration 2 Variants ──────────────────────────────────────────────

# Variant A: Velocity damping only, wide gate, no position/heading
DRIFT_ITER2_VEL_ONLY_WIDE_GATE = replace(
    K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED,
    profile_name="drift_iter2_vel_only_wide_gate",
    drift_k_vel=10.0,
    drift_k_pos=0.0,
    drift_k_heading=0.0,
    drift_k_heading_rate=0.0,
    drift_max_tau=8.0,
)

# Variant B: Velocity damping + heading hold, wide gate
DRIFT_ITER2_VEL_HEADING_WIDE_GATE = replace(
    K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED,
    profile_name="drift_iter2_vel_heading_wide_gate",
    drift_k_vel=10.0,
    drift_k_pos=0.0,
    drift_k_heading=5.0,
    drift_k_heading_rate=1.5,
    drift_max_tau=8.0,
)

# Variant C: Velocity + heading + late position return
DRIFT_ITER2_VEL_HEADING_LATE_POSITION = replace(
    K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED,
    profile_name="drift_iter2_vel_heading_late_position",
    drift_k_vel=10.0,
    drift_k_pos=1.5,
    drift_k_heading=5.0,
    drift_k_heading_rate=1.5,
    drift_max_tau=8.0,
    drift_pgate_low=0.15,
    drift_pgate_high=0.80,
)

# Variant D: Push damping emphasis (higher push_damp_mult)
DRIFT_ITER2_PUSH_DAMPING = replace(
    K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED,
    profile_name="drift_iter2_push_damping",
    drift_k_vel=10.0,
    drift_k_pos=0.0,
    drift_k_heading=5.0,
    drift_k_heading_rate=1.5,
    drift_max_tau=8.0,
    drift_push_damp_mult=3.0,
)

# Variant E: Dynamic height yield (late position, wide height gate)
DRIFT_ITER2_DYNAMIC_YIELD = replace(
    K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED,
    profile_name="drift_iter2_dynamic_yield",
    drift_k_vel=10.0,
    drift_k_pos=0.0,
    drift_k_heading=5.0,
    drift_k_heading_rate=1.5,
    drift_max_tau=8.0,
    drift_hgate_low=0.03,
    drift_hgate_high=0.15,
    drift_pgate_low=0.50,
    drift_pgate_high=1.50,
)


K2_WHEEL_VEL_NOTCH = replace(
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    profile_name="k2_wheel_vel_notch_v1",
    enable_wip_notch_filter=True,
    wip_notch_target_signal="wheel_velocity",
    wip_notch_center_hz=2.5,
    wip_notch_q=6.0,
    wip_notch_filter_blend=1.0,
    wip_notch_gate_enabled=True,
    wip_notch_height_gate_start_m=0.42,
    wip_notch_height_gate_full_m=0.48,
)
K3_PITCH_RATE_WHEEL_VEL_NOTCH = replace(
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    profile_name="k3_pitch_rate_wheel_vel_notch_v1",
    enable_wip_notch_filter=True,
    wip_notch_target_signal="pitch_rate_and_wheel_velocity",
    wip_notch_center_hz=2.5,
    wip_notch_q=6.0,
    wip_notch_filter_blend=1.0,
    wip_notch_gate_enabled=True,
    wip_notch_height_gate_start_m=0.42,
    wip_notch_height_gate_full_m=0.48,
)
K3B_PITCH_RATE_WHEEL_VEL_NOTCH_BLEND075 = replace(
    K3_PITCH_RATE_WHEEL_VEL_NOTCH,
    profile_name="k3b_pitch_rate_wheel_vel_notch_blend075",
    wip_notch_filter_blend=0.75,
)

# =====================================================================
# K_SWEEP factory — audit-only filter parameter sweep profiles
# =====================================================================
# These profiles are generated programmatically from K1_PITCH_RATE_NOTCH.
# All are audit-only — none may become current-best.
# =====================================================================


def _make_sweep_profile(base, name, **overrides):
    """Create an audit-only sweep profile from a base profile.

    Marks the profile as audit-only by appending '--audit-sweep' to the name.
    All sweep profiles inherit from K1_PITCH_RATE_NOTCH.
    """
    return replace(base, profile_name=name, **overrides)


# ── Group A: Centre frequency sweep (Q=6, blend=1.0, biquad_notch) ──
K_SWEEP_FC_1P50 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_fc_1p50",
    wip_notch_center_hz=1.5, wip_notch_q=6.0, wip_notch_filter_blend=1.0,
    wip_notch_filter_type="biquad_notch",
)
K_SWEEP_FC_1P75 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_fc_1p75",
    wip_notch_center_hz=1.75, wip_notch_q=6.0, wip_notch_filter_blend=1.0,
    wip_notch_filter_type="biquad_notch",
)
K_SWEEP_FC_2P00 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_fc_2p00",
    wip_notch_center_hz=2.0, wip_notch_q=6.0, wip_notch_filter_blend=1.0,
    wip_notch_filter_type="biquad_notch",
)
K_SWEEP_FC_2P25 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_fc_2p25",
    wip_notch_center_hz=2.25, wip_notch_q=6.0, wip_notch_filter_blend=1.0,
    wip_notch_filter_type="biquad_notch",
)
K_SWEEP_FC_2P75 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_fc_2p75",
    wip_notch_center_hz=2.75, wip_notch_q=6.0, wip_notch_filter_blend=1.0,
    wip_notch_filter_type="biquad_notch",
)
K_SWEEP_FC_3P00 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_fc_3p00",
    wip_notch_center_hz=3.0, wip_notch_q=6.0, wip_notch_filter_blend=1.0,
    wip_notch_filter_type="biquad_notch",
)
K_SWEEP_FC_3P25 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_fc_3p25",
    wip_notch_center_hz=3.25, wip_notch_q=6.0, wip_notch_filter_blend=1.0,
    wip_notch_filter_type="biquad_notch",
)
K_SWEEP_FC_3P50 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_fc_3p50",
    wip_notch_center_hz=3.5, wip_notch_q=6.0, wip_notch_filter_blend=1.0,
    wip_notch_filter_type="biquad_notch",
)

# ── Group B: Q sweep (fc=2.5, blend=1.0, biquad_notch) ──
K_SWEEP_Q_2P0 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_q_2p0",
    wip_notch_center_hz=2.5, wip_notch_q=2.0, wip_notch_filter_blend=1.0,
    wip_notch_filter_type="biquad_notch",
)
K_SWEEP_Q_3P0 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_q_3p0",
    wip_notch_center_hz=2.5, wip_notch_q=3.0, wip_notch_filter_blend=1.0,
    wip_notch_filter_type="biquad_notch",
)
K_SWEEP_Q_8P0 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_q_8p0",
    wip_notch_center_hz=2.5, wip_notch_q=8.0, wip_notch_filter_blend=1.0,
    wip_notch_filter_type="biquad_notch",
)
K_SWEEP_Q_10P0 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_q_10p0",
    wip_notch_center_hz=2.5, wip_notch_q=10.0, wip_notch_filter_blend=1.0,
    wip_notch_filter_type="biquad_notch",
)

# ── Group C: Blend sweep (fc=2.5, Q=6, biquad_notch) ──
K_SWEEP_BLEND_0P00 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_blend_0p00",
    wip_notch_center_hz=2.5, wip_notch_q=6.0, wip_notch_filter_blend=0.0,
    wip_notch_filter_type="biquad_notch",
)
K_SWEEP_BLEND_0P25 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_blend_0p25",
    wip_notch_center_hz=2.5, wip_notch_q=6.0, wip_notch_filter_blend=0.25,
    wip_notch_filter_type="biquad_notch",
)

# ── Group D: Topology variants ──
# Notch-disabled diagnostic
K_SWEEP_NOTCH_DISABLED = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_notch_disabled",
    wip_notch_filter_type="notch_disabled",
)

# First-order lowpass variants (cutoff sweep: 3.0, 4.0, 5.0, 6.0 Hz)
K_SWEEP_LP_3P0 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_lp_3p0",
    wip_notch_filter_type="first_order_lowpass",
    wip_lowpass_cutoff_hz=3.0,
)
K_SWEEP_LP_4P0 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_lp_4p0",
    wip_notch_filter_type="first_order_lowpass",
    wip_lowpass_cutoff_hz=4.0,
)
K_SWEEP_LP_5P0 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_lp_5p0",
    wip_notch_filter_type="first_order_lowpass",
    wip_lowpass_cutoff_hz=5.0,
)
K_SWEEP_LP_6P0 = _make_sweep_profile(
    K1_PITCH_RATE_NOTCH, "k_sweep_lp_6p0",
    wip_notch_filter_type="first_order_lowpass",
    wip_lowpass_cutoff_hz=6.0,
)

# ── All sweep profile names for registry ──
ALL_K_SWEEP_PROFILES = {
    # Group A: centre frequency
    "k_sweep_fc_1p50": K_SWEEP_FC_1P50,
    "k_sweep_fc_1p75": K_SWEEP_FC_1P75,
    "k_sweep_fc_2p00": K_SWEEP_FC_2P00,
    "k_sweep_fc_2p25": K_SWEEP_FC_2P25,
    "k_sweep_fc_2p75": K_SWEEP_FC_2P75,
    "k_sweep_fc_3p00": K_SWEEP_FC_3P00,
    "k_sweep_fc_3p25": K_SWEEP_FC_3P25,
    "k_sweep_fc_3p50": K_SWEEP_FC_3P50,
    # Group B: Q
    "k_sweep_q_2p0": K_SWEEP_Q_2P0,
    "k_sweep_q_3p0": K_SWEEP_Q_3P0,
    "k_sweep_q_8p0": K_SWEEP_Q_8P0,
    "k_sweep_q_10p0": K_SWEEP_Q_10P0,
    # Group C: blend
    "k_sweep_blend_0p00": K_SWEEP_BLEND_0P00,
    "k_sweep_blend_0p25": K_SWEEP_BLEND_0P25,
    # Group D: topology
    "k_sweep_notch_disabled": K_SWEEP_NOTCH_DISABLED,
    "k_sweep_lp_3p0": K_SWEEP_LP_3P0,
    "k_sweep_lp_4p0": K_SWEEP_LP_4P0,
    "k_sweep_lp_5p0": K_SWEEP_LP_5P0,
    "k_sweep_lp_6p0": K_SWEEP_LP_6P0,
}


# =====================================================================
# L Family — K1 + Coordinated Sagittal State Feedback (Phase 3)
# =====================================================================
# Base: K1 (K1_PITCH_RATE_NOTCH)
#
# Purpose: Replace the independent sagittal torque summation with a
# coordinated state-feedback command that accounts for coupled sagittal
# states. The goal is to suppress the 2.5 Hz WIP mode by synchronizing
# pitch, support, and rate contributions instead of letting them fight.
#
# State vector: [pitch_error, pitch_rate_effective, support_position_error,
#                support_velocity, wheel_velocity_average]
#
# Controller form:
#   tau_coordinated = K1_base_torque + coordinated_feedback(x)
#
# Where coordinated_feedback uses manually specified gains in an LQR-like
# structure, NOT arbitrary independent term summation.
#
# Telemetry fields in sagittal_diag:
#   L_enabled, L_candidate_kind, L_state_vector,
#   L_gains, L_feedback_torque, L_base_torque, L_final_torque
#
# Key rule: Do NOT modify K1 parameters. Add coordinated_feedback on top.


def _coordinated_feedback_gains_L1(height_m: float) -> dict:
    """L1 gains: conservative state feedback focused on suppressing 2.5 Hz.

    Gains are height-scheduled between low and high heights.
    State vector order: [pitch, pitch_rate, support_err, support_vel, wheel_vel_mean]

    At tall heights (0.48 m), pitch-rate and support-velocity gains are
    tuned to provide phase-coherent damping at ~2.5 Hz.
    At low heights (0.33 m), same gains but lower overall torque authority.
    """
    h_norm = max(0.0, min(1.0, (height_m - 0.30) / (0.48 - 0.30)))
    # Gains interpolate between low-height and high-height values
    k_pitch = 8.0 + (5.0 - 8.0) * h_norm           # pitch error (Nm/rad)
    k_pitch_rate = 0.8 + (1.5 - 0.8) * h_norm        # pitch rate (Nm/(rad/s))
    k_support = -15.0 + (-20.0 - (-15.0)) * h_norm    # support error (Nm/m)
    k_support_vel = -0.5 + (-1.0 - (-0.5)) * h_norm   # support vel (Nm/(m/s))
    k_wheel_vel = 0.0                                 # wheel vel not used in L1
    return {
        "k_pitch": float(k_pitch),
        "k_pitch_rate": float(k_pitch_rate),
        "k_support": float(k_support),
        "k_support_vel": float(k_support_vel),
        "k_wheel_vel": float(k_wheel_vel),
        "kind": "coordinated_low_freq_state_feedback",
    }


def _coordinated_feedback_gains_L2(height_m: float) -> dict:
    """L2 gains: coordinated feedback with phase-lead compensation.

    Adds a small phase-lead on the pitch_rate path to reduce the ~90° phase
    lag that causes damping to feed the 2.5 Hz mode. The lead compensation
    is implemented as an additional term on pitch rate error rate (pitch
    acceleration proxy) to create a phase-advanced damping component.

    Lead: tau_lead = k_lead * d(pitch_rate)/dt (acceleration proxy)
    """
    h_norm = max(0.0, min(1.0, (height_m - 0.30) / (0.48 - 0.30)))
    k_pitch = 8.0 + (5.0 - 8.0) * h_norm
    k_pitch_rate = 0.8 + (1.2 - 0.8) * h_norm        # slightly lower rate gain than L1
    k_support = -15.0 + (-20.0 - (-15.0)) * h_norm
    k_support_vel = -0.5 + (-1.0 - (-0.5)) * h_norm
    k_lead = 0.05 + (0.08 - 0.05) * h_norm            # phase-lead on pitch acceleration proxy
    return {
        "k_pitch": float(k_pitch),
        "k_pitch_rate": float(k_pitch_rate),
        "k_support": float(k_support),
        "k_support_vel": float(k_support_vel),
        "k_lead": float(k_lead),
        "kind": "coordinated_phase_lead_compensation",
    }


def _coordinated_feedback_gains_L3(height_m: float) -> dict:
    """L3 gains: coordinated feedback + pitch reference stabilization.

    Small pitch reference correction based on support state. The correction
    is a physical, state-tied modification: when support error is large, the
    pitch reference is shifted slightly to reduce the pitch-vs-support
    conflict without suppressing pitch torque or support torque.

    pitch_ref_mod = small_k * support_error (deg)

    The correction is amplitude-limited to prevent anti-phase injection.
    """
    h_norm = max(0.0, min(1.0, (height_m - 0.30) / (0.48 - 0.30)))
    k_pitch = 8.0 + (5.0 - 8.0) * h_norm
    k_pitch_rate = 0.8 + (1.5 - 0.8) * h_norm
    k_support = -15.0 + (-20.0 - (-15.0)) * h_norm
    k_support_vel = -0.5 + (-1.0 - (-0.5)) * h_norm
    pitch_ref_gain = 1.5 + (2.5 - 1.5) * h_norm       # deg/m of support error
    pitch_ref_max = 1.0 + (1.5 - 1.0) * h_norm         # max correction (deg)
    return {
        "k_pitch": float(k_pitch),
        "k_pitch_rate": float(k_pitch_rate),
        "k_support": float(k_support),
        "k_support_vel": float(k_support_vel),
        "pitch_ref_gain": float(pitch_ref_gain),
        "pitch_ref_max": float(pitch_ref_max),
        "kind": "coordinated_pitch_ref_stabilization",
    }


# ---- LR family: Replacement coordinated sagittal feedback gain functions ---- #
# These use similar gains to the L family but the feedback REPLACES the
# sum-of-independent-torques rather than adding on top. The gains are
# dimensioned to be the TOTAL feedback, not an additive supplement.
# Height-scheduled with conservative bounds to avoid the L family's
# torque double-counting failure.

def _lr_replacement_gains_LR1(height_m: float) -> dict:
    """LR1 gains: replacement coordinated low-frequency feedback.

    Replaces the sum-of-independent-torques with a single coordinated
    command. Gains are the full feedback authority, not additive.
    """
    h_norm = max(0.0, min(1.0, (height_m - 0.30) / (0.48 - 0.30)))
    # Moderate replacement gains — total authority, not additive supplement
    k_pitch = 6.0 + (3.5 - 6.0) * h_norm           # pitch error (Nm/rad) — moderate
    k_pitch_rate = 0.6 + (1.2 - 0.6) * h_norm        # pitch rate (Nm/(rad/s))
    k_support = -8.0 + (-12.0 - (-8.0)) * h_norm      # support error (Nm/m)
    k_support_vel = -0.3 + (-0.6 - (-0.3)) * h_norm   # support vel (Nm/(m/s))
    return {
        "k_pitch": float(k_pitch),
        "k_pitch_rate": float(k_pitch_rate),
        "k_support": float(k_support),
        "k_support_vel": float(k_support_vel),
        "kind": "lr_replacement_low_freq_state_feedback",
    }


def _lr_replacement_gains_LR2(height_m: float) -> dict:
    """LR2 gains: replacement coordinated feedback with phase-lead.

    Adds phase-lead compensation on the pitch_rate path. Gains are
    the full replacement authority.
    """
    h_norm = max(0.0, min(1.0, (height_m - 0.30) / (0.48 - 0.30)))
    k_pitch = 6.0 + (3.5 - 6.0) * h_norm
    k_pitch_rate = 0.5 + (1.0 - 0.5) * h_norm        # slightly lower rate gain than LR1
    k_support = -8.0 + (-12.0 - (-8.0)) * h_norm
    k_support_vel = -0.3 + (-0.6 - (-0.3)) * h_norm
    k_lead = 0.04 + (0.06 - 0.04) * h_norm            # phase-lead on pitch acceleration
    return {
        "k_pitch": float(k_pitch),
        "k_pitch_rate": float(k_pitch_rate),
        "k_support": float(k_support),
        "k_support_vel": float(k_support_vel),
        "k_lead": float(k_lead),
        "kind": "lr_replacement_phase_lead_compensation",
    }


def _lr_replacement_gains_LR3(height_m: float) -> dict:
    """LR3 gains: replacement coordinated feedback with pitch ref stabilization.

    Includes small pitch reference correction based on support state.
    Gains are the full replacement authority.
    """
    h_norm = max(0.0, min(1.0, (height_m - 0.30) / (0.48 - 0.30)))
    k_pitch = 6.0 + (3.5 - 6.0) * h_norm
    k_pitch_rate = 0.6 + (1.2 - 0.6) * h_norm
    k_support = -8.0 + (-12.0 - (-8.0)) * h_norm
    k_support_vel = -0.3 + (-0.6 - (-0.3)) * h_norm
    pitch_ref_gain = 1.0 + (2.0 - 1.0) * h_norm        # deg/m of support error
    pitch_ref_max_deg = 0.8 + (1.2 - 0.8) * h_norm      # max correction (deg)
    return {
        "k_pitch": float(k_pitch),
        "k_pitch_rate": float(k_pitch_rate),
        "k_support": float(k_support),
        "k_support_vel": float(k_support_vel),
        "pitch_ref_gain": float(pitch_ref_gain),
        "pitch_ref_max_deg": float(pitch_ref_max_deg),
        "kind": "lr_replacement_pitch_ref_stabilization",
    }


# ---- LRS Family: Sign-audited constrained gain sweep ---- #
# All signs confirmed correct by Phase 1 audit (2026-06-24).
# Failure mode is gain magnitude, not sign.
# Hard bounds: k_pitch <= 15, k_pitch_rate <= 3, |k_support| <= 2.5x LR1, |k_support_vel| <= 2.5x LR1.

def _lrs_replacement_gains_S1(height_m: float) -> dict:
    """LRS1: Support-dominant — increase support position/velocity authority.

    Target: fix support drift while keeping pitch gains moderate.
    k_support ≈ 1.8x LR1, k_support_vel ≈ 1.8x LR1.
    """
    h_norm = max(0.0, min(1.0, (height_m - 0.30) / (0.48 - 0.30)))
    k_pitch = 6.0 + (3.5 - 6.0) * h_norm           # same as LR1
    k_pitch_rate = 0.6 + (1.2 - 0.6) * h_norm        # same as LR1
    # 1.8x LR1 support gains
    k_support = -14.4 + (-21.6 - (-14.4)) * h_norm    # 1.8x: -12→-21.6 at h=0.48
    k_support_vel = -0.54 + (-1.08 - (-0.54)) * h_norm  # 1.8x: -0.6→-1.08 at h=0.48
    return {
        "k_pitch": float(k_pitch),
        "k_pitch_rate": float(k_pitch_rate),
        "k_support": float(k_support),
        "k_support_vel": float(k_support_vel),
        "kind": "lrs1_support_dominant",
    }


def _lrs_replacement_gains_S2(height_m: float) -> dict:
    """LRS2: Pitch-rate damping — increase damping around 0.5 Hz.

    k_pitch_rate ≈ 2.5x LR1, other gains at LR1 level.
    """
    h_norm = max(0.0, min(1.0, (height_m - 0.30) / (0.48 - 0.30)))
    k_pitch = 6.0 + (3.5 - 6.0) * h_norm           # same as LR1
    # 2.5x LR1 pitch rate gain (capped at hard bound 3.0)
    k_pitch_rate_base = 1.5 + (3.0 - 1.5) * h_norm   # 2.5x LR1 baseline
    k_pitch_rate = min(k_pitch_rate_base, 3.0)        # hard bound
    k_support = -8.0 + (-12.0 - (-8.0)) * h_norm      # same as LR1
    k_support_vel = -0.3 + (-0.6 - (-0.3)) * h_norm   # same as LR1
    return {
        "k_pitch": float(k_pitch),
        "k_pitch_rate": float(k_pitch_rate),
        "k_support": float(k_support),
        "k_support_vel": float(k_support_vel),
        "kind": "lrs2_pitch_rate_damping",
    }


def _lrs_replacement_gains_S3(height_m: float) -> dict:
    """LRS3: Balanced medium — moderate increase across all gains.

    k_pitch ≈ 1.5x LR1, k_pitch_rate ≈ 2x LR1, support ≈ 1.5x LR1.
    """
    h_norm = max(0.0, min(1.0, (height_m - 0.30) / (0.48 - 0.30)))
    k_pitch = 9.0 + (5.25 - 9.0) * h_norm             # 1.5x: 3.5→5.25 at h=0.48
    k_pitch_rate = 1.2 + (2.4 - 1.2) * h_norm          # 2x: 1.2→2.4 at h=0.48
    k_support = -12.0 + (-18.0 - (-12.0)) * h_norm     # 1.5x: -12→-18 at h=0.48
    k_support_vel = -0.45 + (-0.9 - (-0.45)) * h_norm  # 1.5x: -0.6→-0.9 at h=0.48
    return {
        "k_pitch": float(k_pitch),
        "k_pitch_rate": float(k_pitch_rate),
        "k_support": float(k_support),
        "k_support_vel": float(k_support_vel),
        "kind": "lrs3_balanced_medium",
    }


# =============================================================================
# LP PRIORITY SAGITTAL ALLOCATOR GAIN FUNCTIONS
# =============================================================================
# Architectural alternative to LR/LRS coordinated feedback. Instead of a single
# equal-priority sum tau = k_pitch*pitch + k_pitch_rate*pitch_rate +
# k_support*support + k_support_vel*support_vel, LP uses:
#
#   tau_common = tau_eq_ff_pass_through
#              + tau_pitch_priority
#              + tau_support_residual_allocated
#
# where pitch priority gets first access to dynamic authority and support
# centering only uses remaining residual, gated by pitch safety, saturation
# headroom, direction consistency, and slew limits.
#
# Hard safety bounds (same as LRS for comparability):
#   |k_pitch_lp| <= 15 Nm/rad
#   |k_pitch_rate_lp| <= 3 Nm/(rad/s)
#   |k_support_lp| <= 30 (2.5× LR1 baseline)
#   |k_support_vel_lp| <= 1.5 (2.5× LR1 baseline)


def _lp_priority_gains_LP1(height_m: float) -> dict:
    """LP1: Conservative pitch priority, soft support residual.

    Pitch gets moderate authority. Support centering is soft and only allowed
    when pitch state is controlled. Designed to test whether pitch-first
    allocation can complete 3000 steps where LR/LRS failed.
    """
    h_norm = max(0.0, min(1.0, (height_m - 0.30) / (0.48 - 0.30)))
    k_pitch_lp = 8.0 + (5.0 - 8.0) * h_norm               # moderate pitch stiffness
    k_pitch_rate_lp = 2.0 + (1.2 - 2.0) * h_norm            # moderate pitch damping
    k_support_lp = -10.0 + (-16.0 - (-10.0)) * h_norm       # moderate support centering
    k_support_vel_lp = -0.4 + (-0.8 - (-0.4)) * h_norm      # moderate support velocity damping
    return {
        "k_pitch_lp": float(k_pitch_lp),
        "k_pitch_rate_lp": float(k_pitch_rate_lp),
        "k_support_lp": float(k_support_lp),
        "k_support_vel_lp": float(k_support_vel_lp),
        # Safety gates
        "pitch_safe_low_deg": 5.0,        # full support below this pitch
        "pitch_safe_high_deg": 12.0,       # zero support above this pitch
        "rate_safe_low_deg_s": 30.0,       # full support below this pitch rate
        "rate_safe_high_deg_s": 80.0,      # zero support above this pitch rate
        # Allocation limits
        "pitch_priority_limit_nm": 5.0,     # max pitch priority torque
        "support_residual_fraction": 0.6,   # fraction of residual authority for support
        "support_slew_limit_nm_per_step": 0.3,  # max support torque change per step
        "support_deadband_m": 0.02,         # ignore small support errors
        # Direction gate
        "direction_gate_enabled": True,
        "kind": "lp1_pitch_first_support_residual",
    }


def _lp_priority_gains_LP2(height_m: float) -> dict:
    """LP2: Stronger pitch-rate stabilization, softer support.

    Higher pitch-rate damping with tighter support gates. Goal: test whether
    stronger pitch damping + attenuated support can complete 3000 steps.
    """
    h_norm = max(0.0, min(1.0, (height_m - 0.30) / (0.48 - 0.30)))
    k_pitch_lp = 6.0 + (4.0 - 6.0) * h_norm                # slightly lower pitch stiffness
    k_pitch_rate_lp = 2.8 + (1.8 - 2.8) * h_norm            # stronger pitch damping
    k_pitch_rate_lp = min(k_pitch_rate_lp, 3.0)              # hard bound
    k_support_lp = -8.0 + (-12.0 - (-8.0)) * h_norm         # softer support
    k_support_vel_lp = -0.3 + (-0.6 - (-0.3)) * h_norm      # softer support velocity
    return {
        "k_pitch_lp": float(k_pitch_lp),
        "k_pitch_rate_lp": float(k_pitch_rate_lp),
        "k_support_lp": float(k_support_lp),
        "k_support_vel_lp": float(k_support_vel_lp),
        # Tighter safety gates
        "pitch_safe_low_deg": 4.0,
        "pitch_safe_high_deg": 10.0,
        "rate_safe_low_deg_s": 25.0,
        "rate_safe_high_deg_s": 70.0,
        # Allocation limits
        "pitch_priority_limit_nm": 4.5,
        "support_residual_fraction": 0.4,   # less residual for support
        "support_slew_limit_nm_per_step": 0.2,
        "support_deadband_m": 0.03,
        "direction_gate_enabled": True,
        "kind": "lp2_pitch_strong_support_soft",
    }


def _lp_priority_gains_LP3(height_m: float) -> dict:
    """LP3: Support recentering delayed/gated — only when pitch is safe.

    Support correction is held at zero until post-push pitch settles below a
    strict threshold. After settling, support recentering is enabled with
    moderate gains. Goal: test temporal separation of pitch stabilization
    and support recovery.
    """
    h_norm = max(0.0, min(1.0, (height_m - 0.30) / (0.48 - 0.30)))
    k_pitch_lp = 10.0 + (6.0 - 10.0) * h_norm               # strong pitch priority
    k_pitch_rate_lp = 2.2 + (1.4 - 2.2) * h_norm             # moderate pitch damping
    k_support_lp = -12.0 + (-18.0 - (-12.0)) * h_norm        # strong support (when active)
    k_support_vel_lp = -0.5 + (-1.0 - (-0.5)) * h_norm       # moderate support vel damping
    return {
        "k_pitch_lp": float(k_pitch_lp),
        "k_pitch_rate_lp": float(k_pitch_rate_lp),
        "k_support_lp": float(k_support_lp),
        "k_support_vel_lp": float(k_support_vel_lp),
        # Very tight safety gates — support only when pitch is well-controlled
        "pitch_safe_low_deg": 3.0,
        "pitch_safe_high_deg": 7.0,
        "rate_safe_low_deg_s": 20.0,
        "rate_safe_high_deg_s": 50.0,
        # Support settling: require pitch_abs below settle_threshold for N steps
        "pitch_settle_threshold_deg": 4.0,
        "pitch_settle_steps_required": 50,
        # Allocation limits
        "pitch_priority_limit_nm": 6.0,
        "support_residual_fraction": 0.5,
        "support_slew_limit_nm_per_step": 0.15,  # slower support ramp
        "support_deadband_m": 0.015,
        "direction_gate_enabled": True,
        "kind": "lp3_support_recenter_when_safe",
    }


# ---- LR Family Profile Constants ---- #
# These profiles REPLACE the sum-of-independent-torques with coordinated
# feedback, preserving equilibrium/feedforward and notch filter.
# Built on K1_PITCH_RATE_NOTCH.

LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="lr1_k1_replacement_coordinated_low_freq_v1",
    enable_lr_replacement_feedback=True,
    lr_replacement_kind="LR1_low_freq",
)

LR2_K1_REPLACEMENT_PHASE_LEAD_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="lr2_k1_replacement_phase_lead_v1",
    enable_lr_replacement_feedback=True,
    lr_replacement_kind="LR2_phase_lead",
)

LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="lr3_k1_replacement_pitch_ref_stabilized_v1",
    enable_lr_replacement_feedback=True,
    lr_replacement_kind="LR3_pitch_ref_stabilized",
)


# ---- LRS Family: Sign-audited constrained gain sweep profiles ---- #
# All signs confirmed correct. Failure mode is gain magnitude, not sign.
# See: scripts/audit_lr_support_drift_sign_phase.py (Phase 1 audit, 2026-06-24).
# Built on K1_PITCH_RATE_NOTCH, opt-in only.

LRS1_SUPPORT_DOMINANT_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="lrs1_support_dominant_v1",
    enable_lr_replacement_feedback=True,
    lr_replacement_kind="LRS1_support_dominant",
)

LRS2_PITCH_RATE_DAMPING_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="lrs2_pitch_rate_damping_v1",
    enable_lr_replacement_feedback=True,
    lr_replacement_kind="LRS2_pitch_rate_damping",
)

LRS3_BALANCED_MEDIUM_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="lrs3_balanced_medium_v1",
    enable_lr_replacement_feedback=True,
    lr_replacement_kind="LRS3_balanced_medium",
)


# ---- LP Family: Priority Sagittal Allocator Profiles ---- #
# Pitch-first support-residual architecture. Resolves the LR/LRS support-pitch
# coupling by allocating pitch stabilization torque first and support-centering
# torque only from residual safe authority, gated by pitch safety, saturation
# headroom, direction consistency, and slew limits.
# Built on K1_PITCH_RATE_NOTCH, opt-in only. Preserves K1 EQ/FF baseline.
# See: docs/validation/lp_priority_sagittal_allocator_report.md

LP1_K1_PRIORITY_PITCH_FIRST_SUPPORT_RESIDUAL_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="lp1_k1_priority_pitch_first_support_residual_v1",
    enable_lp_priority_allocator=True,
    lp_allocator_kind="LP1_pitch_first_support_residual",
)

LP2_K1_PRIORITY_PITCH_STRONG_SUPPORT_SOFT_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="lp2_k1_priority_pitch_strong_support_soft_v1",
    enable_lp_priority_allocator=True,
    lp_allocator_kind="LP2_pitch_strong_support_soft",
)

LP3_K1_PRIORITY_SUPPORT_RECENTER_WHEN_SAFE_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="lp3_k1_priority_support_recenter_when_safe_v1",
    enable_lp_priority_allocator=True,
    lp_allocator_kind="LP3_support_recenter_when_safe",
)


# L1 — Lowest-risk coordinated feedback
# Adds small coordinated LQR-style state feedback on top of K1's notch.
# The feedback torque is added to the wheel torque AFTER the normal sagittal
# torque computation, so K1's existing terms are unchanged.
L1_K1_COORDINATED_LOW_FREQ_FEEDBACK = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="l1_k1_coordinated_low_freq_feedback_v1",
    # Metadata: the coordinated feedback function is selected at runtime
    # via a new field on SagittalAuthoritySchedule.
    enable_coordinated_sagittal_feedback=True,
    coordinated_feedback_kind="L1_low_freq",
)

# L2 — Coordinated with phase-lead compensation
L2_K1_COORDINATED_PHASE_LEAD = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="l2_k1_coordinated_phase_lead_v1",
    enable_coordinated_sagittal_feedback=True,
    coordinated_feedback_kind="L2_phase_lead",
)

# L3 — Coordinated with pitch reference stabilization
L3_K1_COORDINATED_PITCH_REF_STABILIZATION = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="l3_k1_coordinated_pitch_ref_stabilization_v1",
    enable_coordinated_sagittal_feedback=True,
    coordinated_feedback_kind="L3_pitch_ref_stabilization",
)

# =====================================================================
# M Family — K1 + Body-Yaw/Wheel-Yaw Correct-Actuator Fix (Phase 4)
# =====================================================================
# Base: K1 (K1_PITCH_RATE_NOTCH)
#
# Purpose: Reduce D4/D5 hip_yaw > 0.35 rad by addressing body yaw drift
# through the correct actuator path (differential wheel velocity), not
# through hip-yaw torque increase.
#
# Key difference from old E candidate: M uses sagittal-coordinated wheel
# yaw that accounts for pitch/support state to avoid yaw-spin instability.
# The wheel yaw torque is modulated by a support-confidence gate and does
# NOT fight the mode-div divergence controller.
#
# Telemetry: M_enabled, M_candidate_kind, M_wheel_yaw_torque,
#            M_body_yaw_error, M_support_gate, M_yaw_correlation


# M1 — Low-band body-yaw damping through differential wheel velocity.
# Uses low gain with smooth yaw-rate-based activation.
M1_K1_BODY_YAW_DIFF_WHEEL_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="m1_k1_body_yaw_diff_wheel_v1",
    enable_body_yaw_wheel_stabilization=True,
    wheel_yaw_kp=0.5,
    wheel_yaw_kd=0.1,
    wheel_yaw_max_torque=1.5,
    wheel_yaw_height_gate_start_m=0.34,
    wheel_yaw_height_gate_full_m=0.42,
    wheel_yaw_activation_threshold_rad=0.05,
    wheel_yaw_support_gate_enabled=True,
)

# M2 — Support-aware body-yaw damping.
# Modulates wheel-yaw correction based on support/contact confidence.
# Avoids injecting yaw torque during poor support states.
M2_K1_BODY_YAW_SUPPORT_AWARE_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="m2_k1_body_yaw_support_aware_v1",
    enable_body_yaw_wheel_stabilization=True,
    wheel_yaw_kp=0.8,
    wheel_yaw_kd=0.15,
    wheel_yaw_max_torque=2.0,
    wheel_yaw_height_gate_start_m=0.34,
    wheel_yaw_height_gate_full_m=0.42,
    wheel_yaw_activation_threshold_rad=0.05,
    wheel_yaw_support_gate_enabled=True,
    wheel_yaw_support_error_threshold_m=0.15,
    wheel_yaw_support_rate_threshold_mps=0.05,
)

# =====================================================================
# N Family — K1 + Mild Phase-Compensated Damping Diagnostic (Phase 5)
# =====================================================================
# Base: K1 (K1_PITCH_RATE_NOTCH)
#
# Purpose: Check whether K1 notch plus very mild coordinated damping
# can recover the transient J3a benefit without J3a's growing oscillation.
#
# Restriction: No J3a as-is. No K3 combined notch. No wheel_velocity
# notch full blend. Abort if RMS worsens vs K1.

# N1 — Very mild phase-lead-compensated pitch rate damping increment.
# Uses the same phase-lead concept from L2 but at a much lower level,
# applied only to the pitch_rate path (not the full sagittal command).
N1_K1_MILD_PHASE_LEAD_DAMPING = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="n1_k1_mild_phase_lead_damping_v1",
    enable_coordinated_sagittal_feedback=True,
    coordinated_feedback_kind="N1_mild_phase_lead",
)

# N1 micro-sweep variants (K1 + mild parameter changes)
# All stay within bounds: k_rate <= 0.6, k_lead <= 0.06

# N1b: slightly higher rate and lead
N1B_K1_MILD_PHASE_LEAD_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="n1b_k1_mild_phase_lead_v1",
    enable_coordinated_sagittal_feedback=True,
    coordinated_feedback_kind="N1_mild_phase_lead",
    n1_rate_low=0.4,
    n1_rate_high=0.6,
    n1_lead_low=0.03,
    n1_lead_high=0.06,
)

# N1c: same rate as N1b but slightly lower lead
N1C_K1_MILD_PHASE_LEAD_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="n1c_k1_mild_phase_lead_v1",
    enable_coordinated_sagittal_feedback=True,
    coordinated_feedback_kind="N1_mild_phase_lead",
    n1_rate_low=0.4,
    n1_rate_high=0.6,
    n1_lead_low=0.025,
    n1_lead_high=0.05,
)

# N1d: same lead as N1b but slightly lower rate
N1D_K1_MILD_PHASE_LEAD_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="n1d_k1_mild_phase_lead_v1",
    enable_coordinated_sagittal_feedback=True,
    coordinated_feedback_kind="N1_mild_phase_lead",
    n1_rate_low=0.35,
    n1_rate_high=0.55,
    n1_lead_low=0.03,
    n1_lead_high=0.06,
)

# =====================================================================
# Unified Sagittal State-Feedback No-Offset Controller
# =====================================================================
# Opt-in profile that replaces the independent tau_pitch + tau_position +
# tau_velocity_damping sum-of-torques architecture with a single coordinated
# sagittal command from full state feedback. The mode classifier detects 8
# operating modes and applies priority-weighted arbitration so the six state
# terms share the same torque budget toward the SAME goal.
#
# Key design choices:
# - pitch_ref_offset_deg = 0.0 (no pitch offset at all)
# - pitch_ref_height_schedule_enabled = False
# - All offset/trim/bias mechanisms disabled
# - One unified command replaces tau_pitch, tau_position, outer_loop
# - Mode classifier + priority arbitration
# - Height-scheduled gains (continuous gain scheduling)
# - Safety gates for contact/roll/hip-yaw/torque-cap/rate-limit
#
# Built on HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM's safety infrastructure
# for contact/roll/height gates but overrides ALL sagittal control
# computation. Disabled by default — opt-in only.
# See docs/validation/unified_sagittal_no_offset_design.md.
UNIFIED_SAGITTAL_STATE_FEEDBACK_NO_OFFSET = replace(
    HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM,
    profile_name="unified_sagittal_state_feedback_no_offset",
    # Disable ALL offset/trim/bias mechanisms
    pitch_ref_offset_deg=0.0,
    pitch_ref_height_schedule_enabled=False,
    pitch_ref_height_schedule_heights_m=(),
    pitch_ref_height_schedule_offsets_deg=(),
    outer_loop_enabled=False,
    calibrated_outer_loop_enabled=False,
    pitch_bias_comp_enabled=False,
    t6j_bias_trim_enabled=False,
    adaptive_bias_trim_enabled=False,
    enable_phase_aware_recenter=False,
    enable_hysteresis_recenter=False,
    enable_bias_cancel=False,
    enable_active_pitch_crossing=False,
    # Enable unified state-feedback mode
    enable_unified_sagittal_state_feedback=True,
    # Tuned gains for no-offset operation.
    # Pitch-primary architecture: tau = Ktheta*pitch + Komega*pitch_rate - Ki*∫err dt
    # No separate tau_position term — avoids the structural pitch-vs-support conflict.
    # The integral slowly winds up to cancel steady-state drift, replacing the
    # pitch_ref_offset without introducing a fixed bias.
    unified_kx=0.0,
    unified_kv=0.0,
    unified_ktheta=30.0,
    unified_komega=10.0,
    unified_kh=0.0,
    unified_khdot=0.0,
    unified_torque_cap=6.0,
    unified_rate_limit=1.0,
    # Gain scheduling disabled for initial discovery
    unified_gain_height_schedule=False,
    unified_torque_cap_nominal=5.0,
    unified_torque_cap_low_max=6.0,
)

# Backward-compatible aliases — development identifiers → semantic constants.
# These allow existing imports and scripts to keep working. The primary names
# (BAND_LIMITED_SUPPORT_RECENTER, EMERGENCY_BUDGET_CAP_RAISE, etc.) should be
# used in new code.
APCR1ND_T5_BAND_LIMITED_BALANCED = BAND_LIMITED_SUPPORT_RECENTER  # legacy
T6F_BUDGET_CAP_RAISE = EMERGENCY_BUDGET_CAP_RAISE                  # legacy
T6I_PHASE_AWARE_RELEASE = PHASE_AWARE_AUTHORITY_RELEASE            # legacy
T6J_CENTERING_BIAS_TRIM = SUPPORT_CENTERING_BIAS_TRIM             # legacy
ADAPTIVE_CENTERING_BIAS_TRIM = ADAPTIVE_SUPPORT_CENTERING_TRIM  # legacy alias

# Profile registry for CLI selection
JOINT_FIX_PROFILES = {
    "baseline": BASELINE_AUTHORITY_SCHEDULE,
    "J1": JOINT_FIX_J1_SUPPORT_CAP,
    "J2": JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING,
    "J3": JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING,
    "J2a": PITCH_SAFE_J2A_CONSERVATIVE,
    "J2b": PITCH_SAFE_J2B_BALANCED,
    "J2c": PITCH_SAFE_J2C_VELOCITY_PRIORITY,
    "J2d": PITCH_SAFE_J2D_TAU_CAP_PRIORITY,
    "APCR1f_adaptive_fast_response_phase_brake": APCR1F_FAST_RESPONSE_PHASE_BRAKE,
    "APCR1g_predictive_fast_response_phase_brake": APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE,
    "APCR1h_support_drift_priority_fast_recenter": APCR1H_SUPPORT_DRIFT_PRIORITY,
    "APCR1i_support_hysteresis_recenter": APCR1I_SUPPORT_HYSTERESIS_RECENTER,
    "APCR1j_support_hysteresis_higher_authority": APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
    "APCR1k_support_hysteresis_early_entry": APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
    "APCR1l_pitch_suppress_recenter": APCR1L_PITCH_SUPPRESS_RECENTER,
    "APCR1m_conditional_pitch_blend_recenter": APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
    "APCR1n_recenter_priority_torque_boost": APCR1N_RECENTER_PRIORITY_TORQUE_BOOST,
    "APCR1nD_direct_support_recenter_features": APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
    # APCR1nD Tuned Variants
    "APCR1nD_T1_early_entry": APCR1ND_T1_EARLY_ENTRY,
    "APCR1nD_T2_hold_outside_band": APCR1ND_T2_HOLD_OUTSIDE_BAND,
    "APCR1nD_T3_early_entry_plus_hold": APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD,
    "APCR1nD_T4_stronger_authority": APCR1ND_T4_STRONGER_AUTHORITY,
    "APCR1nD_T5_band_limited_balanced": BAND_LIMITED_SUPPORT_RECENTER,  # legacy alias
    # Semantic: Band-Limited Support Recenter
    "band_limited_support_recenter": BAND_LIMITED_SUPPORT_RECENTER,
    # T6 High-Height Transient Suppression Variants
    "T6A_high_early_hard_band": T6A_HIGH_EARLY_HARD_BAND,
    "T6B_high_stronger_emergency": T6B_HIGH_STRONGER_EMERGENCY,
    "T6C_high_early_plus_stronger": T6C_HIGH_EARLY_PLUS_STRONGER,
    "T6D_high_transient_boost": T6D_HIGH_TRANSIENT_BOOST,
    "T6E_high_pitch_aware_boost": T6E_HIGH_PITCH_AWARE_BOOST,
    "T6F_budget_cap_raise": EMERGENCY_BUDGET_CAP_RAISE,         # legacy alias
    "emergency_budget_cap_raise": EMERGENCY_BUDGET_CAP_RAISE,   # semantic
    "T6F_sign_corrected": T6F_SIGN_CORRECTED,
    "T6H_soft_blend_arch_fix": T6H_SOFT_BLEND_ARCH_FIX,
    "T6I_phase_aware_release": PHASE_AWARE_AUTHORITY_RELEASE,   # legacy alias
    "phase_aware_authority_release": PHASE_AWARE_AUTHORITY_RELEASE,  # semantic
    "T6J_centering_bias_trim": SUPPORT_CENTERING_BIAS_TRIM,      # legacy alias
    "support_centering_bias_trim": SUPPORT_CENTERING_BIAS_TRIM, # semantic
    "adaptive_support_centering_trim": ADAPTIVE_SUPPORT_CENTERING_TRIM,  # opt-in proportional adaptive trim
    "zero_crossing_support_recenter": ZERO_CROSSING_SUPPORT_RECENTER,  # ZC hysteresis recenter
    "early_zero_crossing_recenter": EARLY_ZERO_CROSSING_RECENTER,  # Early ZC: exits at zero, not opposite side
    "early_zero_crossing_recenter_v2": EARLY_ZERO_CROSSING_RECENTER_V2,  # V2: anti-rebound fix for EZC_FAILURE_EXIT_TOO_EARLY_REBOUND
    "pitch_bias_compensated_zero_crossing_recenter": PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER,  # Phase 7: EZC V2 + pitch DC bias compensation
    "pitch_equilibrium_trim": PITCH_EQUILIBRIUM_TRIM,  # Phase 3 structural fix: shift pitch reference to equilibrium to center support drift
    "height_scheduled_pitch_equilibrium_trim": HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM,  # Phase 2 structural fix: per-height pitch_ref offset schedule
    "support_position_outer_loop_pitch_ref": SUPPORT_POSITION_OUTER_LOOP_PITCH_REF,  # Phase B dynamic outer loop on top of height schedule
    "calibrated_support_position_outer_loop_pitch_ref": CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF,  # v1 — failed Phase 6 upper-band regressions
    "calibrated_support_position_outer_loop_pitch_ref_v2": CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF_V2,  # v2 — opt-in, no regressions  # Phase B calibration: height-dependent outer-loop gains
    # Physics-based equilibrium feedforward outer loop (Phase D, opt-in)
    "physics_equilibrium_feedforward_outer_loop": PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP,
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v1": PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V1,
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2": PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    # I_SUPPORT_REFERENCE_REACQUISITION_V1 — candidate I1 (opt-in, diagnostic only)
    "i_support_reference_reacquisition_v1": I_SUPPORT_REFERENCE_REACQUISITION_V1,
    # Unified sagittal state-feedback no-offset controller
    "unified_sagittal_state_feedback_no_offset": UNIFIED_SAGITTAL_STATE_FEEDBACK_NO_OFFSET,
}


class SagittalVelocityDampedBalanceController:
    """Wheel-based sagittal balance with explicit velocity and position damping.

    Control law:
        tau = k_pitch * pitch_x
            + k_pitch_rate * pitch_rate_x
            + k_velocity * (-sagittal_velocity)
            + k_wheel_velocity * (-wheel_velocity_mean)
            + k_position * (-sagittal_position_error)
            + k_support_velocity * (-support_position_velocity)

    Signs verified by unit tests:
        - positive pitch → restoring torque (opposes tilt)
        - positive pitch_rate → damping torque (opposes angular velocity)
        - positive sagittal_velocity → torque reducing forward velocity
        - positive wheel_velocity_mean → opposing torque
        - positive sagittal_position_error → weak return tendency
        - positive support_position_velocity → damping torque (opposes support drift)
    """

    def __init__(
        self,
        kp_pitch: float = 50.0,
        kd_pitch: float = 10.0,
        kp_cp: float = 0.0,
        kd_com_vy: float = 5.0,
        k_velocity: float = 0.0,
        k_wheel_velocity: float = 0.5,
        k_position: float = 0.0,
        k_support_velocity: float = 0.0,
        max_position_tau: float = 3.0,
        wheel_torque_sign: float = 1.0,
        max_tau_wheel: float = 5.0,
        enable_capture_gate: bool = False,
        capture_gate_config: Optional[dict] = None,
        dt: float = 0.01,
        enable_torque_budget_aware_position: bool = False,
        position_tau_budget_cap: float = 7.0,
        enable_position_integral: bool = False,
        ki_position_integral: float = 0.0,
        integral_max_abs: float = 1.0,
        integral_pitch_error_threshold_rad: float = 0.03,
        integral_roll_error_threshold_rad: float = 0.05,
        integral_pitch_rate_threshold_rad_s: float = 0.05,
        integral_support_velocity_threshold_m_s: float = 0.03,
        integral_wheel_velocity_threshold_rad_s: float = 1.0,
        integral_min_com_z_m: float = 0.38,
        integral_max_com_z_m: float = 0.43,
        authority_schedule: SagittalAuthoritySchedule | None = None,
    ):
        if wheel_torque_sign not in [1.0, -1.0]:
            raise ValueError(f"wheel_torque_sign must be +1.0 or -1.0, got {wheel_torque_sign}")

        self.kp_pitch = kp_pitch
        self.kd_pitch = kd_pitch
        self.kp_cp = kp_cp
        self.kd_com_vy = kd_com_vy
        self.k_velocity = k_velocity
        self.k_wheel_velocity = k_wheel_velocity
        self.k_position = k_position
        self.k_support_velocity = k_support_velocity
        self.max_position_tau = max_position_tau
        self.wheel_torque_sign = wheel_torque_sign
        self.max_tau_wheel = max_tau_wheel
        self.enable_capture_gate = enable_capture_gate
        self.dt = dt
        self.enable_torque_budget_aware_position = enable_torque_budget_aware_position
        self.position_tau_budget_cap = position_tau_budget_cap
        self.enable_position_integral = enable_position_integral
        self.ki_position_integral = ki_position_integral
        self.integral_max_abs = integral_max_abs
        self.integral_pitch_error_threshold_rad = integral_pitch_error_threshold_rad
        self.integral_roll_error_threshold_rad = integral_roll_error_threshold_rad
        self.integral_pitch_rate_threshold_rad_s = integral_pitch_rate_threshold_rad_s
        self.integral_support_velocity_threshold_m_s = integral_support_velocity_threshold_m_s
        self.integral_wheel_velocity_threshold_rad_s = integral_wheel_velocity_threshold_rad_s
        self.integral_min_com_z_m = integral_min_com_z_m
        self.integral_max_com_z_m = integral_max_com_z_m
        self.authority_schedule = authority_schedule or BASELINE_AUTHORITY_SCHEDULE
        self.position_integral_error = 0.0

        # State for support position velocity computation
        self.prev_support_position_error_m = 0.0

        # State for continuous k_position scheduling: first-order filtered com_z
        self._filtered_com_z = 0.4  # Initialize to default com_z

        # State for phase-aware recenter (F1_strategy)
        self._prev_recenter_tau = 0.0

        # State for hysteresis recenter (F2_strategy)
        self._hysteresis_state = "NEUTRAL"  # NEUTRAL, RECENTER_FROM_POSITIVE, RECENTER_FROM_NEGATIVE
        self._hysteresis_prev_tau = 0.0  # Previous smoothed torque for smoothing
        self._hysteresis_state_entry_count = 0
        self._hysteresis_state_exit_count = 0
        self._hysteresis_safety_override_count = 0

        # State for bias cancellation (G1_strategy)
        self._bias_cancel_estimate = 0.0  # Low-pass filtered signed error estimate
        self._bias_cancel_prev_tau = 0.0  # Previous smoothed bias torque

        # State for Active Pitch Crossing (APC_strategy)
        # States: NEUTRAL, CROSS_FROM_POSITIVE, CROSS_FROM_NEGATIVE, HOLD_RECENTER_TO_ZERO
        self._apc_state = "NEUTRAL"
        self._apc_prev_tau = 0.0  # Previous smoothed crossing torque
        self._apc_state_entry_count = 0
        self._apc_state_exit_count = 0
        self._apc_safety_override_count = 0
        self._apc_persistent_tau_sign_steps = 0  # Count steps with persistent tau sign
        self._apc_prev_tau_sign = 0  # Track previous tau sign for persistence detection

        # State for APCR1e adaptive authority tracking
        self._apc_adaptive_no_improvement_count = 0  # Steps since error decreased
        self._apc_adaptive_prev_abs_error = 0.0  # Previous absolute error for improvement detection

        # State for APCR1f fast response with phase brake
        self._apc_fast_response_increasing_error_count = 0  # Steps with growing error
        self._apc_fast_response_prev_error = 0.0  # Previous signed error
        self._apc_fast_response_prev_tau = 0.0  # Previous tau before rate limit
        self._apc_fast_response_phase_brake_active = False  # Phase brake engaged
        self._apc_fast_response_adaptive_tau_limit = 0.0  # Current adaptive max tau

        # State for APCR1g Predictive Fast Response with Phase Brake
        self._apc_predictive_prev_error = 0.0  # Previous signed error for prediction
        self._apc_predictive_prev_error_rate = 0.0  # Previous error rate for prediction
        self._apc_predictive_predicted_error = 0.0  # Current predicted error
        self._apc_predictive_no_improvement_count = 0  # Steps without improvement
        self._apc_predictive_increasing_error_count = 0  # Steps with growing error
        self._apc_predictive_adaptive_tau_limit = 0.0  # Current adaptive max tau
        self._apc_predictive_phase_brake_active = False  # Phase brake engaged
        self._apc_predictive_phase_brake_strong_active = False  # Strong phase brake engaged
        self._apc_predictive_predictive_trigger_active = False  # Predictive trigger engaged
        self._apc_predictive_predictive_boost_active = False  # Predictive boost engaged
        self._apc_predictive_prev_abs_error = 0.0  # Previous absolute error for improvement detection

        # State for APCR1h Drift Priority Override
        self._apc_drift_priority_active = False  # Drift priority mode active
        self._apc_drift_priority_emergency_active = False  # Emergency clamp active

        # State for T6I Phase-Aware Release
        # Track recent errors for convergence detection
        self._t6i_error_history = []  # List of recent errors (max length = convergence_window_steps)
        self._t6i_current_cap = 4.0  # Current position cap (starts at nominal)
        self._t6i_converging = False  # Convergence detected flag
        self._t6j_bias_error_history = []  # Rolling signed error history for centering trim
        self._t6j_bias_trim_tau = 0.0  # Current applied trim torque
        self._t6j_bias_trim_target_tau = 0.0  # Current target trim torque before rate limiting
        self._t6j_bias_positive_duration_steps = 0
        self._t6j_bias_negative_duration_steps = 0
        # State for Adaptive Centering Bias Trim (proportional, height-aware, guarded)
        self._adaptive_bias_trim_tau = 0.0           # current applied trim torque
        self._adaptive_bias_trim_target_tau = 0.0    # target before rate limiting
        self._adaptive_bias_slow_error_history = []  # slow window signed errors
        self._adaptive_bias_fast_error_history = []  # fast window signed errors
        self._adaptive_bias_zero_crossing_history = []  # signed errors for crossing detection
        self._adaptive_bias_crossing_count = 0        # crossings in current window
        self._adaptive_bias_guard_trigger_count = 0   # consecutive guard triggers
        self._adaptive_bias_prev_trim_sign = 0        # +1/-1/0
        self._adaptive_bias_prev_error_sign = 0       # +1/-1/0
        self._adaptive_bias_hold_steps = 0            # steps in sign-reversal hold
        self._adaptive_bias_positive_area = 0.0       # accumulated positive drift area
        self._adaptive_bias_negative_area = 0.0       # accumulated negative drift area
        self._apc_drift_priority_tau_limit = 0.0  # Current drift priority tau limit
        self._apc_drift_priority_rate_limit = 0.0  # Current drift priority rate limit
        self._apc_drift_priority_prev_tau = 0.0  # Previous tau for rate limiting
        self._apc_drift_priority_steps_since_hard_drift = 0  # Steps since hard drift (>0.15)
        self._apc_drift_priority_error_rate_reversal_achieved = False  # e_dot sign reversed
        self._apc_drift_priority_prev_error = 0.0  # Previous error for e_dot sign detection

        # State for APCR1i Support Hysteresis Recenter
        self._apc_hysteresis_state = "NEUTRAL"  # NEUTRAL, RECENTER_FROM_POSITIVE, RECENTER_FROM_NEGATIVE, HOLD_THROUGH_ZERO
        self._apc_hysteresis_prev_tau = 0.0  # Previous tau for rate limiting
        self._apc_hysteresis_state_entry_count = 0  # State entry count for telemetry
        self._apc_hysteresis_state_exit_count = 0  # State exit count for telemetry
        self._apc_hysteresis_entry_e = 0.0  # Error at state entry
        self._apc_hysteresis_exit_e = 0.0  # Error at state exit
        self._apc_hysteresis_emergency_active = False  # Emergency clamp active
        self._apc_hysteresis_prev_e = 0.0  # Previous error for e_dot detection

        # State for APCR1nD Direct Support Drift Trigger
        self._apcr1nd_step_counter = 0  # Step counter for startup guard
        self._apcr1nd_prev_error = 0.0  # Previous signed error for e_dot detection
        self._apcr1nd_direct_recenter_priority_active = False  # Direct recenter active
        self._apcr1nd_hysteresis_state = "NEUTRAL"  # Hysteresis state for direct trigger
        self._apcr1nd_prev_tau = 0.0  # Previous tau for smoothing

        # State for APCR1nD Tuned Variants
        self._apcr1nd_tuned_converging_steps = 0  # Consecutive converging steps for release
        self._apcr1nd_tuned_recenter_held = False  # Recenter held outside band
        self._apcr1nd_wd_override_active = False  # Phase 0: wheel damping override applied this step

        # State for Zero-Crossing Support Recenter (ZC)
        self._zc_state = "CENTER_IDLE"  # CENTER_IDLE, RECENTER_FROM_POSITIVE, RECENTER_FROM_NEGATIVE, HOLD_THROUGH_ZERO, SAFETY_DECAY
        self._zc_state_id = 0           # 0=IDLE, 1=POS, 2=NEG, 3=HOLD, 4=DECAY
        self._zc_direction = 0          # -1, 0, +1 correction direction
        self._zc_hold_steps = 0         # steps in current hold
        self._zc_dwell_steps = 0        # dwell steps in near-zero band
        self._zc_tau = 0.0              # current applied ZC correction torque
        self._zc_target_tau = 0.0       # target before rate limiting
        self._zc_enter_event = 0         # cumulative enter events
        self._zc_exit_event = 0          # cumulative exit events
        self._zc_crossed_zero = False    # True if current episode crossed zero
        self._zc_cross_target_reached = False  # True if crossed to opposite side
        self._zc_episode_id = 0          # episode counter
        self._zc_episode_start_error = 0.0  # error at episode start
        self._zc_episode_min_error = 0.0  # min error in positive episode
        self._zc_episode_max_error = 0.0  # max error in negative episode
        self._zc_safety_gate_pass = True  # safety gate status
        self._zc_block_reason = "none"   # block reason for telemetry
        self._zc_exit_reason = "none"     # exit reason for telemetry

        # State for Early Zero-Crossing Support Recenter (EZC)
        # Key differences: exits at zero, not opposite side; earlier entry at 0.05 m
        # V2 adds: ANTIREBOUND_DECAY state for anti-rebound hold
        self._ezc_state = "CENTER_IDLE"  # CENTER_IDLE, RECENTER_FROM_POSITIVE, RECENTER_FROM_NEGATIVE, ZERO_CROSSED_DECAY, ANTIREBOUND_DECAY, SAFETY_DECAY
        self._ezc_state_id = 0           # 0=IDLE, 1=POS, 2=NEG, 3=DECAY, 4=SAFETY, 5=ANTIREBOUND
        self._ezc_direction = 0           # -1, 0, +1 correction direction
        self._ezc_hold_steps = 0          # steps in current hold
        self._ezc_tau = 0.0               # current applied EZC correction torque
        self._ezc_target_tau = 0.0        # target before rate limiting
        self._ezc_enter_event = 0         # cumulative enter events
        self._ezc_zero_cross_exit_event = 0  # cumulative exits at zero
        self._ezc_safety_exit_event = 0   # cumulative exits due to safety
        self._ezc_crossed_zero = False     # True if current episode crossed zero
        self._ezc_zero_dwell_steps = 0    # dwell steps after zero crossing
        self._ezc_episode_id = 0          # episode counter
        self._ezc_episode_start_error = 0.0  # error at episode start
        self._ezc_episode_min_error = 0.0 # min error in positive episode
        self._ezc_episode_max_error = 0.0 # max error in negative episode
        self._ezc_safety_gate_pass = True # safety gate status
        self._ezc_block_reason = "none"   # block reason for telemetry
        self._ezc_exit_reason = "none"    # exit reason for telemetry
        # Anti-rebound state (V2)
        self._ezc_antirebound_steps = 0   # steps in anti-rebound decay
        self._ezc_antirebound_tau_start = 0.0  # tau at start of anti-rebound

        # Pitch bias DC compensation state (Phase 7)
        self._pitch_bias_estimate_nm = 0.0   # EMA of tau_pitch in stable windows
        self._pitch_bias_samples = 0          # number of EMA updates
        self._pitch_bias_comp_tau_nm = 0.0   # current bounded compensation

        # Notch filter state (K candidate family — 2.5 Hz WIP mode)
        self._wip_notch_pitch_rate: BiquadNotchFilter | None = None
        self._wip_notch_wheel_left: BiquadNotchFilter | None = None
        self._wip_notch_wheel_right: BiquadNotchFilter | None = None
        self._wip_notch_support_vel: BiquadNotchFilter | None = None
        self._wip_notch_fs_hz: float = 0.0

        # Unified sagittal state-feedback controller state
        self._prev_unified_tau_cmd = 0.0      # previous step's tau_cmd for rate limiting
        self._no_offset_int_error = 0.0        # integral accumulator for no-offset controller

        # L family: coordinated state-feedback state (Phase 3)
        self._prev_pitch_rate_for_L = 0.0      # previous pitch_rate for phase-lead computation
        self._prev_pitch_rate_for_N = 0.0      # previous pitch_rate for N1 mild damping
        self._prev_pitch_rate_for_LR = 0.0     # previous pitch_rate for LR2 phase-lead computation
        self._lp_prev_support_allocated = 0.0  # LP slew-limit state
        self._lp_pitch_settle_counter = 0       # LP3 pitch-settle counter

        # Initialize capture gate if enabled
        if self.enable_capture_gate:
            gate_config = capture_gate_config or {}
            self.capture_gate = PositionHoldCaptureGate(**gate_config)
        else:
            self.capture_gate = None

    def _compute_tuned_band_state(self, abs_error: float) -> tuple[str, int]:
        """Compute band state for tuned variants telemetry.

        Returns:
            (band_name, band_id) where band_id: 0=normal, 1=soft, 2=desired, 3=hard, 4=emergency
        """
        if not self.authority_schedule.apcr1nd_tuned_enabled:
            return "none", 0

        emergency_band_m = self.authority_schedule.apcr1nd_emergency_band_m
        hard_band_m = self.authority_schedule.apcr1nd_hard_band_m
        desired_band_m = self.authority_schedule.apcr1nd_desired_band_m
        soft_enter_m = self.authority_schedule.apcr1nd_soft_enter_m

        if abs_error >= emergency_band_m:
            return "emergency", 4
        elif abs_error >= hard_band_m:
            return "hard", 3
        elif abs_error >= desired_band_m:
            return "desired", 2
        elif abs_error >= soft_enter_m:
            return "soft", 1
        else:
            return "normal", 0

    def compute(
        self,
        pitch_x_rad: float,
        pitch_rate_x_rad_s: float,
        sagittal_velocity_m_s: float,
        wheel_vel_left_rad_s: float,
        wheel_vel_right_rad_s: float,
        sagittal_position_error_m: float = 0.0,
        com_y_m: float = 0.0,
        com_vy_m_s: float = 0.0,
        support_center_y_m: float = 0.0,
        com_z_m: float = 0.4,
        roll_y_rad: float = 0.0,
        contact_valid: bool = True,
        height_variant_name: str | None = None,
        commanded_height_ref_m: float | None = None,
    ) -> tuple[Array, dict]:
        """Compute sagittal velocity-damped balance torque and diagnostics.

        Args:
            pitch_x_rad: Body pitch angle in robot frame (rad).
            pitch_rate_x_rad_s: Body pitch rate in robot frame (rad/s).
            sagittal_velocity_m_s: CoM velocity projected onto initial-heading
                sagittal axis (m/s). Positive = forward.
            wheel_vel_left_rad_s: Left wheel velocity (rad/s).
            wheel_vel_right_rad_s: Right wheel velocity (rad/s).
            sagittal_position_error_m: Sagittal displacement from equilibrium
                along initial-heading axis (m). Positive = forward of reference.
            com_y_m: CoM position in sagittal direction (m). Required for capture gate.
            com_vy_m_s: CoM velocity in sagittal direction (m/s). Required for capture gate.
            support_center_y_m: Support center position in sagittal direction (m). Required for capture gate.
            com_z_m: CoM height (m). Required for capture gate.

        Returns:
            tau: Torque vector (10,) with nonzero values only at wheel indices [4, 9].
            diagnostics: Dictionary with per-term decomposition and saturation info.
        """
        wheel_vel_mean = 0.5 * (wheel_vel_left_rad_s + wheel_vel_right_rad_s)
        schedule_active = self.authority_schedule.is_active_for_variant(height_variant_name)

        # ---- Continuous height-scheduled parameters ----
        # Determine scheduling height source
        if commanded_height_ref_m is not None:
            schedule_height_ref = commanded_height_ref_m
            schedule_height_source = "target_reference"
        else:
            # Fallback: use first-order filtered current com_z
            alpha_filter = 0.9  # Slow filter to avoid gain oscillation
            self._filtered_com_z = alpha_filter * self._filtered_com_z + (1.0 - alpha_filter) * float(com_z_m)
            schedule_height_ref = self._filtered_com_z
            schedule_height_source = "filtered_current_fallback"

        # k_position scheduling
        effective_k_position = self.k_position
        if self.authority_schedule.continuous_k_position:
            effective_k_position = scheduled_k_position(
                z_ref=schedule_height_ref,
                k_nominal=self.authority_schedule.k_position_nominal,
                k_low_max=self.authority_schedule.k_position_low_max,
                z_low=self.authority_schedule.k_position_z_low,
                z_high=self.authority_schedule.k_position_z_high,
            )
            # Compute smoothstep variables for telemetry and active flag
            u_raw = (self.authority_schedule.k_position_z_high - schedule_height_ref) / (
                self.authority_schedule.k_position_z_high - self.authority_schedule.k_position_z_low
            )
            u_clamped = max(0.0, min(1.0, u_raw))
            smoothstep_value = u_clamped * u_clamped * (3.0 - 2.0 * u_clamped)

            u_for_telemetry = u_clamped  # normalized position [0,1]
            schedule_smoothstep = smoothstep_value
        else:
            u_for_telemetry = 0.0
            smoothstep_value = 0.0
            schedule_smoothstep = 0.0

        # max_position_tau scheduling (Phase 6 joint fix)
        if self.authority_schedule.continuous_max_position_tau:
            effective_max_position_tau = scheduled_k_position(
                z_ref=schedule_height_ref,
                k_nominal=self.authority_schedule.max_position_tau_nominal,
                k_low_max=self.authority_schedule.max_position_tau_low_max,
                z_low=self.authority_schedule.k_position_z_low,
                z_high=self.authority_schedule.k_position_z_high,
            )
        else:
            effective_max_position_tau = self.authority_schedule.max_position_tau_for_variant(
                height_variant_name,
                self.max_position_tau,
            )

        # T6F Architecture Fix: Conditionally raise upstream cap during safe high-height emergency
        # This allows emergency recenter authority > 4.0 Nm to reach wheels
        arch_fix_active = False
        arch_fix_reason = "none"
        arch_fix_height_gate_pass = False
        arch_fix_band_gate_pass = False
        arch_fix_safety_gate_pass = False
        arch_fix_recenter_gate_pass = False
        arch_fix_requested_cap = 0.0
        effective_max_position_tau_before_arch_fix = float(effective_max_position_tau)

        # T6F Sign Fix: Initialize telemetry variables
        sign_fix_active = False
        sign_fix_damping_disabled = False
        sign_fix_damping_helped = False
        sign_fix_damping_fought = False
        sign_fix_damping_original_nm = 0.0
        sign_fix_damping_after_nm = 0.0
        sign_fix_pitch_suppressed = False
        sign_fix_pitch_original_nm = 0.0
        sign_fix_pitch_after_nm = 0.0

        # T6H telemetry variables (initialized before use)
        t6h_soft_pitch_blend_active = False
        t6h_pitch_blend_factor = 1.0
        t6h_pitch_safety_active = False
        t6h_soft_damping_blend_active = False
        t6h_damping_blend_factor = 1.0
        t6h_wheel_velocity_safety_active = False

        # T6I telemetry variables (initialized before use)
        t6i_error_converging = False
        t6i_error_trend = 0.0
        t6i_target_cap = 4.0
        t6i_current_cap = 4.0
        t6i_cap_delta_this_step = 0.0
        t6i_cap_change_rate_limited = False
        t6i_release_reason = "none"

        # T6J telemetry variables (initialized before use)
        t6j_bias_trim_enabled = bool(self.authority_schedule.t6j_bias_trim_enabled)
        t6j_bias_trim_active = False
        t6j_bias_mean_error_m = 0.0
        t6j_bias_window_steps = int(self.authority_schedule.t6j_bias_trim_window_steps)
        t6j_bias_trim_target_tau_nm = 0.0
        t6j_bias_trim_tau_nm = float(self._t6j_bias_trim_tau)
        t6j_bias_trim_rate_limited = False
        t6j_bias_safety_gate_pass = False
        t6j_bias_block_reason = "disabled"
        t6j_bias_applied_to_final_tau = 0.0
        t6j_bias_expected_direction_correct = False

        # Adaptive centering bias trim telemetry variables (initialized before use)
        adaptive_bias_trim_enabled = bool(self.authority_schedule.adaptive_bias_trim_enabled)
        adaptive_bias_trim_active = False
        adaptive_bias_mean_error_m = 0.0
        adaptive_bias_fast_mean_error_m = 0.0
        adaptive_bias_effective_error_m = 0.0
        adaptive_bias_target_tau_nm = 0.0
        adaptive_bias_tau_nm = float(self._adaptive_bias_trim_tau)
        adaptive_bias_max_tau_current_nm = 0.0
        adaptive_bias_height_scale = 0.0
        adaptive_bias_rate_used_nm_per_step = 0.0
        adaptive_bias_zero_crossing_count = 0
        adaptive_bias_zero_crossing_guard_active = False
        adaptive_bias_near_zero_relief_active = False
        adaptive_bias_sign_reversal_blocked = False
        adaptive_bias_safety_gate_pass = False
        adaptive_bias_block_reason = "disabled"
        adaptive_bias_expected_direction_correct = False
        adaptive_bias_positive_area = 0.0
        adaptive_bias_negative_area = 0.0
        adaptive_bias_symmetry_ratio = 0.0
        adaptive_bias_hip_yaw_gate_pass = True
        adaptive_bias_hip_yaw_abs_max = 0.0

        if self.authority_schedule.arch_fix_enabled:
            # Gate 1: Height threshold (only at high heights >= 0.45m)
            arch_fix_height_gate_pass = schedule_height_ref >= self.authority_schedule.arch_fix_height_threshold_m

            # Gate 2: Band state (hard or emergency)
            # We need to check APCR1nD band state - this will be computed later
            # For now, use a forward reference that we'll populate in the APCR1nD section
            # Temporary: always False here, will be updated after APCR1nD computes band
            arch_fix_band_gate_pass = False  # Will be set after APCR1nD band computation

            # Gate 3: Safety gates (contact/height/roll/pitch)
            arch_fix_safety_gate_pass = (
                contact_valid
                and com_z_m >= self.authority_schedule.recenter_priority_safe_min_com_z
                and abs(float(roll_y_rad)) <= self.authority_schedule.recenter_priority_safe_roll_rad
                and abs(float(pitch_x_rad)) <= self.authority_schedule.recenter_priority_safe_pitch_rad
            )

            # Gate 4: Recenter priority active
            # Will be set after APCR1nD section computes this
            arch_fix_recenter_gate_pass = False  # Will be set after APCR1nD

            # Architecture fix will be applied after APCR1nD band state is known
            # Placeholder values for now
            arch_fix_reason = "awaiting_apcr1nd_state"


        # k_velocity scheduling (Phase 6 joint fix)
        if self.authority_schedule.continuous_k_velocity:
            effective_k_velocity = scheduled_k_position(
                z_ref=schedule_height_ref,
                k_nominal=self.authority_schedule.k_velocity_nominal,
                k_low_max=self.authority_schedule.k_velocity_low_max,
                z_low=self.authority_schedule.k_position_z_low,
                z_high=self.authority_schedule.k_position_z_high,
            )
        else:
            effective_k_velocity = self.k_velocity

        # k_wheel_velocity scheduling (Step E extreme height fix)
        # Note: Uses scheduled_k_wheel_velocity which maps z_low -> k_nominal, z_high -> k_high_max
        # (inverse of k_position scheduling which targets low heights)
        if self.authority_schedule.continuous_k_wheel_velocity:
            effective_k_wheel_velocity = scheduled_k_wheel_velocity(
                z_ref=schedule_height_ref,
                k_nominal=self.authority_schedule.k_wheel_velocity_nominal,
                k_high_max=self.authority_schedule.k_wheel_velocity_high_max,
                z_low=self.authority_schedule.k_wheel_velocity_z_low,
                z_high=self.authority_schedule.k_wheel_velocity_z_high,
            )
            high_height_wheel_damping_active = effective_k_wheel_velocity > self.authority_schedule.k_wheel_velocity_nominal
        else:
            effective_k_wheel_velocity = self.k_wheel_velocity
            high_height_wheel_damping_active = False

        # kd_pitch scheduling (Tall-height WIP damping fix, J candidate).
        # Uses the same smoothstep function as k_wheel_velocity (increases at tall heights).
        if self.authority_schedule.continuous_kd_pitch:
            effective_kd_pitch = scheduled_k_wheel_velocity(
                z_ref=schedule_height_ref,
                k_nominal=self.authority_schedule.kd_pitch_nominal,
                k_high_max=self.authority_schedule.kd_pitch_high_max,
                z_low=self.authority_schedule.kd_pitch_z_low,
                z_high=self.authority_schedule.kd_pitch_z_high,
            )
            high_height_kd_pitch_active = effective_kd_pitch > self.authority_schedule.kd_pitch_nominal
        else:
            effective_kd_pitch = self.kd_pitch
            high_height_kd_pitch_active = False

        SMALL_EPSILON = 1e-6
        low_height_sagittal_schedule_active = (
            (self.authority_schedule.continuous_k_position or
             self.authority_schedule.continuous_max_position_tau or
             self.authority_schedule.continuous_k_velocity or
             self.authority_schedule.continuous_k_wheel_velocity)
            and smoothstep_value > SMALL_EPSILON
        )

        # Legacy variant-based scales (kept for backward compatibility)
        effective_pitch_scale = self.authority_schedule.pitch_tau_scale if schedule_active else 1.0
        effective_pitch_tau_cap = self.authority_schedule.pitch_tau_cap_nm if schedule_active else None
        effective_velocity_damping_scale = self.authority_schedule.velocity_damping_scale if schedule_active else 1.0
        effective_support_velocity_scale = self.authority_schedule.support_velocity_scale if schedule_active else 1.0
        effective_support_velocity_gain = (
            self.authority_schedule.support_velocity_gain
            if schedule_active and self.authority_schedule.support_velocity_gain is not None
            else self.k_support_velocity
        )

        # Compute support position velocity (numerical derivative)
        # This is the rate of change of support-center position error in initial-heading frame
        support_position_velocity_m_s = (sagittal_position_error_m - self.prev_support_position_error_m) / self.dt
        self.prev_support_position_error_m = sagittal_position_error_m

        # ---- Notch filter for 2.5 Hz WIP mode (K candidate family) ----
        # Applies causal IIR biquad notch filter to selected damping input signals
        # to prevent phase-lagged damping from feeding the 2.5 Hz oscillation mode.
        # Only active when enable_wip_notch_filter is True on the authority schedule.
        notch_enabled = self.authority_schedule.enable_wip_notch_filter
        # Audit-only: notch_disabled filter type forces filter off for diagnostics
        if self.authority_schedule.wip_notch_filter_type == "notch_disabled":
            notch_enabled = False
        notch_target = self.authority_schedule.wip_notch_target_signal
        notch_center_hz = self.authority_schedule.wip_notch_center_hz
        notch_q = self.authority_schedule.wip_notch_q
        notch_blend = self.authority_schedule.wip_notch_filter_blend

        # Derive fs from dt if not explicitly set
        if self._wip_notch_fs_hz <= 0:
            self._wip_notch_fs_hz = float(1.0 / self.dt) if self.dt > 0 else 100.0
        fs_hz = self.authority_schedule.wip_notch_fs_hz if self.authority_schedule.wip_notch_fs_hz > 0 else self._wip_notch_fs_hz

        # Compute height gate
        if notch_enabled and self.authority_schedule.wip_notch_gate_enabled:
            notch_height_gate = smoothstep_gate(
                schedule_height_ref,
                self.authority_schedule.wip_notch_height_gate_start_m,
                self.authority_schedule.wip_notch_height_gate_full_m,
            )
        else:
            notch_height_gate = 1.0 if notch_enabled else 0.0

        # Telemetry: raw signals (always captured)
        pitch_rate_raw = float(pitch_rate_x_rad_s)
        wheel_left_raw = float(wheel_vel_left_rad_s)
        wheel_right_raw = float(wheel_vel_right_rad_s)
        support_vel_raw = float(support_position_velocity_m_s)

        # Notched signals (may equal raw if filter disabled)
        pitch_rate_notched = pitch_rate_raw
        wheel_left_notched = wheel_left_raw
        wheel_right_notched = wheel_right_raw
        support_vel_notched = support_vel_raw

        # Lazy-init filters
        notch_filter_valid = False
        if notch_enabled:
            try:
                filter_type = self.authority_schedule.wip_notch_filter_type
                if self._wip_notch_pitch_rate is None:
                    if filter_type == "first_order_lowpass":
                        lp_cutoff = self.authority_schedule.wip_lowpass_cutoff_hz
                        self._wip_notch_pitch_rate = FirstOrderLowPassFilter(fs_hz=fs_hz, cutoff_hz=lp_cutoff)
                        self._wip_notch_wheel_left = FirstOrderLowPassFilter(fs_hz=fs_hz, cutoff_hz=lp_cutoff)
                        self._wip_notch_wheel_right = FirstOrderLowPassFilter(fs_hz=fs_hz, cutoff_hz=lp_cutoff)
                        self._wip_notch_support_vel = FirstOrderLowPassFilter(fs_hz=fs_hz, cutoff_hz=lp_cutoff)
                    else:
                        # Default: biquad_notch (K1 behaviour preserved)
                        self._wip_notch_pitch_rate = BiquadNotchFilter(fs_hz=fs_hz, fc_hz=notch_center_hz, Q=notch_q)
                        self._wip_notch_wheel_left = BiquadNotchFilter(fs_hz=fs_hz, fc_hz=notch_center_hz, Q=notch_q)
                        self._wip_notch_wheel_right = BiquadNotchFilter(fs_hz=fs_hz, fc_hz=notch_center_hz, Q=notch_q)
                        self._wip_notch_support_vel = BiquadNotchFilter(fs_hz=fs_hz, fc_hz=notch_center_hz, Q=notch_q)

                # Update filters
                should_filter_pr = notch_target in ("pitch_rate", "pitch_rate_and_wheel_velocity", "all_damping_signals")
                should_filter_wv = notch_target in ("wheel_velocity", "pitch_rate_and_wheel_velocity", "all_damping_signals")
                should_filter_sv = notch_target in ("support_velocity", "all_damping_signals")

                # Apply filter per signal
                if should_filter_pr:
                    pitch_rate_notched = self._wip_notch_pitch_rate.update(pitch_rate_raw)
                if should_filter_wv:
                    wheel_left_notched = self._wip_notch_wheel_left.update(wheel_left_raw)
                    wheel_right_notched = self._wip_notch_wheel_right.update(wheel_right_raw)
                if should_filter_sv:
                    support_vel_notched = self._wip_notch_support_vel.update(support_vel_raw)

                notch_filter_valid = True
            except Exception:
                # If filter fails, fall back to raw signals
                notch_filter_valid = False

        # Blend: gate * blend controls how much filtered vs raw signal is used
        gate = notch_height_gate * notch_blend if notch_enabled else 0.0
        pitch_rate_effective = (1.0 - gate) * pitch_rate_raw + gate * pitch_rate_notched
        wheel_left_effective = (1.0 - gate) * wheel_left_raw + gate * wheel_left_notched
        wheel_right_effective = (1.0 - gate) * wheel_right_raw + gate * wheel_right_notched
        support_vel_effective = (1.0 - gate) * support_vel_raw + gate * support_vel_notched

        # For telemetry: compute signal delta
        notch_signal_delta_pr = float(pitch_rate_effective - pitch_rate_raw)
        notch_signal_delta_wl = float(wheel_left_effective - wheel_left_raw)
        notch_signal_delta_wr = float(wheel_right_effective - wheel_right_raw)

        # Use effective (notched or raw) signals for damping computations
        # Replace raw signals for the remainder of compute()
        # Note: wheel_vel_mean is already computed from raw — recompute if filter active
        if notch_enabled and gate > 1e-9:
            wheel_vel_mean = 0.5 * (wheel_left_effective + wheel_right_effective)

        # Override pitch_rate_x_rad_s and wheel velocity references for damping terms
        pitch_rate_for_damping = pitch_rate_effective
        wheel_left_for_damping = wheel_left_effective
        wheel_right_for_damping = wheel_right_effective
        support_vel_for_damping = support_vel_effective

        # =====================================================================
        # UNIFIED SAGITTAL STATE-FEEDBACK NO-OFFSET CONTROLLER
        # =====================================================================
        # When enabled, replaces all of tau_pitch, tau_position, tau_velocity,
        # tau_support_velocity, tau_cp, tau_com_vy, recenter, hysteresis, bias,
        # APC, and outer-loop with a single coordinated state-feedback command.
        #
        # The unified controller uses priority-weighted mode arbitration to
        # prevent the torque conflict documented in
        # docs/validation/unified_no_offset_state_conflict_audit.md.
        # =====================================================================
        no_offset_active = self.authority_schedule.enable_unified_sagittal_state_feedback
        no_offset_mode = "disabled"
        no_offset_gate_pass = True
        no_offset_block_reason = "none"
        no_offset_kx = 0.0
        no_offset_kv = 0.0
        no_offset_ktheta = 0.0
        no_offset_komega = 0.0
        no_offset_kh = 0.0
        no_offset_khdot = 0.0
        no_offset_tau_support_state = 0.0
        no_offset_tau_pitch_state = 0.0
        no_offset_tau_rate_state = 0.0
        no_offset_tau_height_state = 0.0
        no_offset_priority_support = 1.0
        no_offset_priority_pitch = 1.0
        no_offset_priority_rate = 1.0
        no_offset_tau_total_raw = 0.0
        no_offset_tau_total_limited = 0.0
        no_offset_torque_cap = 0.0
        no_offset_rate_limit = 0.0
        no_offset_saturation_active = False
        no_offset_arbitration_reason = "disabled"
        no_offset_pitch_ref_offset_deg = 0.0
        unified_tau_cmd = None  # If computed, used at assembly to replace tau_common

        if no_offset_active:
            sched = self.authority_schedule
            no_offset_pitch_ref_offset_deg = 0.0
            abs_support_error = abs(float(sagittal_position_error_m))
            abs_pitch = abs(float(pitch_x_rad))
            abs_roll = abs(float(roll_y_rad))
            abs_height_err = abs(float(com_z_m) - float(commanded_height_ref_m if commanded_height_ref_m is not None else com_z_m))
            hip_yaw_abs_max = 0.0  # Not available in compute method; set from diagnostics externally

            # --- Safety gates ---
            if not contact_valid:
                no_offset_gate_pass = False
                no_offset_block_reason = "contact_invalid"
            elif com_z_m < 0.28 or com_z_m > 0.50:
                no_offset_gate_pass = False
                no_offset_block_reason = "height_unsafe"
            elif abs_roll > 0.15:
                no_offset_gate_pass = False
                no_offset_block_reason = "roll_unsafe"

            # --- Mode classifier ---
            # Mode classification affects priority weights but does not
            # change the fundamental control law. When Ktheta=0 (pure
            # support-centering), the weights are always 1.0 (no arbitration
            # needed since there's only one objective).
            no_offset_priority_support = 1.0
            no_offset_priority_pitch = 1.0
            no_offset_priority_rate = 1.0

            pitch_rate_large = abs(float(pitch_rate_x_rad_s)) > sched.unified_push_pitch_rate_enter_radps
            pitch_large = abs_pitch > sched.unified_push_pitch_enter_rad
            drift_detected = abs_support_error > sched.unified_drift_enter_m
            height_changing = abs_height_err > sched.unified_height_transition_enter_m
            hip_yaw_risky = hip_yaw_abs_max > sched.unified_hip_yaw_risk_rad
            hip_yaw_danger = hip_yaw_abs_max > sched.unified_hip_yaw_danger_rad
            contact_ok = contact_valid

            if not no_offset_gate_pass:
                no_offset_mode = "BLOCKED"
            elif not contact_ok:
                no_offset_mode = "CONTACT_DEGRADED"
            elif hip_yaw_danger:
                no_offset_mode = "HIP_YAW_RISK"
            elif pitch_large or pitch_rate_large:
                no_offset_mode = "PUSH_RECOVERY"
            elif drift_detected:
                no_offset_mode = "DRIFT_RECOVERY"
            elif height_changing:
                no_offset_mode = "HEIGHT_TRANSITION"
            else:
                no_offset_mode = "STEADY"

            # --- Height-scheduled gains ---
            h_norm = max(0.0, min(1.0, (float(com_z_m) - 0.30) / (0.48 - 0.30)))
            if sched.unified_gain_height_schedule:
                no_offset_kx = sched.unified_kx_nominal + (sched.unified_kx_low_max - sched.unified_kx_nominal) * (1.0 - h_norm)
                no_offset_kv = sched.unified_kv_nominal + (sched.unified_kv_low_max - sched.unified_kv_nominal) * (1.0 - h_norm)
                no_offset_ktheta = sched.unified_ktheta_nominal + (sched.unified_ktheta_low_max - sched.unified_ktheta_nominal) * (1.0 - h_norm)
                no_offset_komega = sched.unified_komega_nominal + (sched.unified_komega_low_max - sched.unified_komega_nominal) * (1.0 - h_norm)
                no_offset_torque_cap = sched.unified_torque_cap_nominal + (sched.unified_torque_cap_low_max - sched.unified_torque_cap_nominal) * (1.0 - h_norm)
            else:
                no_offset_kx = sched.unified_kx
                no_offset_kv = sched.unified_kv
                no_offset_ktheta = sched.unified_ktheta
                no_offset_komega = sched.unified_komega
                no_offset_torque_cap = sched.unified_torque_cap
            no_offset_rate_limit = sched.unified_rate_limit
            no_offset_kh = sched.unified_kh
            no_offset_khdot = sched.unified_khdot

            # --- Priority weights (simplified — pitch-primary architecture) ---
            # With the pitch-primary architecture (no separate tau_position),
            # weights are always 1.0. The sign-aware coordination is handled
            # by the integral term, not by weighting.
            no_offset_priority_support = 1.0
            no_offset_priority_pitch = 1.0
            no_offset_priority_rate = 1.0
            no_offset_arbitration_reason = no_offset_mode.lower()

            # --- Unified state-feedback command ---
            # tau_pitch_fb = +Ktheta * pitch_x (forward correction for forward lean)
            # tau_support = -Kx * err - Kv * vel (support-centering PD)
            # tau_integral = -Ki * ∫err dt (anti-windup bounded integral)
            #
            # The integral term is critical for no-offset operation: it winds up
            # to cancel the DC torque from tau_pitch_fb, eliminating steady-state
            # drift without requiring a pitch_ref_offset. Integral is bounded to
            # prevent windup during large transients.
            # EQUILIBRIUM PITCH HIGH-PASS: use Ktheta*(pitch - pitch_eqm) so the DC
            # component of equilibrium lean (~3 deg at h=0.48) does not produce
            # unwanted forward torque. pitch_eqm is a slow EMA (~10s time constant).
            if not hasattr(self, '_pitch_eqm_estimate'):
                self._pitch_eqm_estimate = float(pitch_x_rad)
            alpha_eqm = 0.010  # 100-step EMA @100Hz ≈ 10s time constant
            self._pitch_eqm_estimate = (1.0 - alpha_eqm) * self._pitch_eqm_estimate + alpha_eqm * float(pitch_x_rad)
            pitch_deviation = float(pitch_x_rad) - self._pitch_eqm_estimate

            tau_pitch_fb = +no_offset_ktheta * pitch_deviation
            tau_support = 0.0

            # Integral on support error (eliminates steady-state drift without offset)
            self._no_offset_int_error += sagittal_position_error_m * self.dt
            k_bound = max(no_offset_ktheta * 0.5, 1.0)
            max_int = 3.0 / k_bound
            self._no_offset_int_error = max(-max_int, min(max_int, self._no_offset_int_error))
            ki_eff = no_offset_ktheta * 0.020
            tau_integral = -ki_eff * self._no_offset_int_error

            tau_rate = +no_offset_komega * pitch_rate_x_rad_s
            tau_height = 0.0

            # Unified command: high-pass pitch + drift integral + pitch rate damping
            unified_tau_cmd_raw = tau_pitch_fb + tau_rate + tau_integral

            no_offset_tau_support_state = 0.0  # no separate support term
            no_offset_tau_pitch_state = float(tau_pitch_fb)
            no_offset_tau_rate_state = float(tau_rate)
            no_offset_tau_height_state = float(tau_integral)  # reuse height_state for integral telemetry
            no_offset_tau_total_raw = float(unified_tau_cmd_raw)

            # --- Torque cap ---
            if abs(unified_tau_cmd_raw) > no_offset_torque_cap:
                unified_tau_cmd_limited = float(jnp.clip(unified_tau_cmd_raw, -no_offset_torque_cap, no_offset_torque_cap))
                no_offset_saturation_active = True
            else:
                unified_tau_cmd_limited = float(unified_tau_cmd_raw)

            # Rate limit
            if abs(unified_tau_cmd_limited - self._prev_unified_tau_cmd) > no_offset_rate_limit:
                delta = unified_tau_cmd_limited - self._prev_unified_tau_cmd
                delta = max(-no_offset_rate_limit, min(no_offset_rate_limit, delta))
                unified_tau_cmd_limited = self._prev_unified_tau_cmd + delta

            self._prev_unified_tau_cmd = unified_tau_cmd_limited
            no_offset_tau_total_limited = float(unified_tau_cmd_limited)

            # Store for final assembly
            unified_tau_cmd = unified_tau_cmd_limited

        # Per-wheel damping (separate for each wheel)
        tau_wheel_vel_left = -effective_k_wheel_velocity * wheel_left_for_damping
        tau_wheel_vel_right = -effective_k_wheel_velocity * wheel_right_for_damping

        # APCR1l: Check if pitch suppression should be applied during RECENTER state
        # During RECENTER, tau_pitch fights drift correction (robot leans back intentionally,
        # but tau_pitch produces positive torque that accelerates forward motion).
        # Suppressing tau_pitch lets APCR + tau_position correct drift without interference.
        apc_recenter_active = self._apc_hysteresis_state in ("RECENTER_FROM_POSITIVE", "RECENTER_FROM_NEGATIVE")
        pitch_suppress_active = (
            self.authority_schedule.apc_hysteresis_pitch_suppress_in_recenter
            and apc_recenter_active
        )

        # Always compute tau_pitch_raw for telemetry (even if suppressed)
        tau_pitch_raw_orig = self.kp_pitch * pitch_x_rad

        # Common balance terms
        if pitch_suppress_active:
            # Suppress tau_pitch during RECENTER to let APCR + tau_position correct drift
            # Compute all terms for telemetry even though they won't be used
            tau_pitch_raw = tau_pitch_raw_orig
            tau_pitch_scheduled = tau_pitch_raw_orig * effective_pitch_scale
            tau_pitch = 0.0
            tau_pitch_clipped = 0.0
        else:
            tau_pitch_raw = self.kp_pitch * pitch_x_rad
            tau_pitch_scheduled = tau_pitch_raw * effective_pitch_scale
            if effective_pitch_tau_cap is None:
                tau_pitch = tau_pitch_scheduled
                tau_pitch_clipped = tau_pitch_scheduled
            else:
                tau_pitch = float(jnp.clip(tau_pitch_scheduled, -effective_pitch_tau_cap, effective_pitch_tau_cap))
                tau_pitch_clipped = tau_pitch

        # =====================================================================
        # Pitch Bias DC Compensation (Phase 7 mechanism)
        # Estimates and removes only the slow tau_pitch DC component during
        # stable upright posture. Does NOT zero tau_pitch, does NOT suppress
        # dynamic pitch correction. See docs/validation/tau_pitch_positive_bias_audit.md
        # and docs/validation/pitch_bias_compensated_zc_design.md.
        # =====================================================================
        tau_pitch_before_bias_comp = float(tau_pitch)
        pitch_bias_gate_pass = False
        pitch_bias_block_reason = "disabled"
        pitch_bias_comp_tau = 0.0
        pitch_bias_estimation_active = False

        if self.authority_schedule.pitch_bias_comp_enabled:
            sched = self.authority_schedule
            abs_pitch_deg = abs(float(pitch_x_rad)) * 180.0 / math.pi
            abs_roll_deg = abs(float(roll_y_rad)) * 180.0 / math.pi
            abs_error_pbc = abs(float(sagittal_position_error_m))

            # Hard safety gates - never apply if any of these fail
            safety_pass = True
            if sched.pitch_bias_disable_if_contact_unstable and not contact_valid:
                safety_pass = False
                pitch_bias_block_reason = "contact_invalid"
            elif float(com_z_m) < sched.pitch_bias_disable_if_height_lt_m:
                safety_pass = False
                pitch_bias_block_reason = "height_unsafe"
            elif abs_pitch_deg > sched.pitch_bias_disable_if_pitch_gt_deg:
                safety_pass = False
                pitch_bias_block_reason = "pitch_unsafe"
            elif abs_roll_deg > sched.pitch_bias_disable_if_roll_gt_deg:
                safety_pass = False
                pitch_bias_block_reason = "roll_unsafe"

            if safety_pass:
                # Estimation window: only when robot is upright AND drift is small
                pitch_bias_estimation_active = (
                    abs_pitch_deg < sched.pitch_bias_only_when_abs_pitch_lt_deg
                    and abs_error_pbc < sched.pitch_bias_only_when_abs_error_lt_m
                )

                # Apply gate: hard band disables, soft band allows, estimation window forces apply
                if abs_error_pbc >= sched.pitch_bias_gate_abs_error_hard_m:
                    pitch_bias_gate_pass = False
                    pitch_bias_block_reason = "error_hard_gate"
                elif pitch_bias_estimation_active:
                    pitch_bias_gate_pass = True
                    pitch_bias_block_reason = "in_estimation_window"
                elif abs_error_pbc < sched.pitch_bias_gate_abs_error_soft_m:
                    pitch_bias_gate_pass = True
                    pitch_bias_block_reason = "soft_gate_pass"
                else:
                    pitch_bias_gate_pass = False
                    pitch_bias_block_reason = "outside_apply_window"

                # Update EMA estimate only during estimation window
                if pitch_bias_estimation_active:
                    window = max(1, int(sched.pitch_bias_window_steps))
                    alpha = 1.0 / window
                    self._pitch_bias_estimate_nm = (
                        (1.0 - alpha) * self._pitch_bias_estimate_nm
                        + alpha * tau_pitch_before_bias_comp
                    )
                    self._pitch_bias_samples += 1

                # Rate-limit current compensation toward target
                # Only compensate the positive DC residual (negative bias should not be amplified)
                target_comp = max(0.0, self._pitch_bias_estimate_nm)
                target_comp = min(target_comp, sched.pitch_bias_max_comp_nm)

                if pitch_bias_gate_pass:
                    rate = sched.pitch_bias_comp_rate_nm_per_step
                    if self._pitch_bias_comp_tau_nm < target_comp:
                        self._pitch_bias_comp_tau_nm = min(
                            target_comp,
                            self._pitch_bias_comp_tau_nm + rate,
                        )
                    elif self._pitch_bias_comp_tau_nm > target_comp:
                        decay = sched.pitch_bias_decay_rate_nm_per_step
                        self._pitch_bias_comp_tau_nm = max(
                            target_comp,
                            self._pitch_bias_comp_tau_nm - decay,
                        )
                    pitch_bias_comp_tau = self._pitch_bias_comp_tau_nm
                else:
                    # Gate fails: decay toward zero (do not apply)
                    decay = sched.pitch_bias_decay_rate_nm_per_step
                    self._pitch_bias_comp_tau_nm = max(
                        0.0,
                        self._pitch_bias_comp_tau_nm - decay,
                    )
                    pitch_bias_comp_tau = 0.0
            else:
                # Safety gate failed - decay any active compensation
                decay = self.authority_schedule.pitch_bias_decay_rate_nm_per_step
                self._pitch_bias_comp_tau_nm = max(
                    0.0,
                    self._pitch_bias_comp_tau_nm - decay,
                )
                pitch_bias_comp_tau = 0.0

            # Apply compensation - subtract from tau_pitch (positive comp -> reduces positive tau_pitch)
            tau_pitch = tau_pitch - pitch_bias_comp_tau
            tau_pitch_clipped = tau_pitch

        tau_pitch_after_bias_comp = float(tau_pitch)

        # APCR1m: Conditional pitch blend (instead of hard suppression)
        # Blend tau_pitch based on error magnitude, with startup guard and safety gates
        apc_pitch_blend_active = False
        apc_pitch_blend_scale = 1.0
        apc_pitch_blend_block_reason = "none"
        apc_pitch_blend_startup_guard_active = False
        apc_pitch_blend_recenter_active = apc_recenter_active
        apc_pitch_blend_pitch_safe = True
        apc_pitch_blend_height_safe = True
        apc_pitch_blend_contact_safe = True
        tau_pitch_before_blend = tau_pitch

        if self.authority_schedule.apc_pitch_blend_enabled:
            # Track steps for startup guard (need instance variable)
            if not hasattr(self, '_apc_pitch_blend_step_counter'):
                self._apc_pitch_blend_step_counter = 0
            current_step = self._apc_pitch_blend_step_counter
            self._apc_pitch_blend_step_counter += 1

            # Check startup guard
            startup_guard_steps = self.authority_schedule.apc_pitch_blend_startup_guard_steps
            if current_step < startup_guard_steps:
                apc_pitch_blend_startup_guard_active = True
                apc_pitch_blend_block_reason = "startup"
            elif not apc_recenter_active:
                # Never blend outside RECENTER state
                apc_pitch_blend_block_reason = "not_recenter"
            else:
                # Check safety conditions
                abs_pitch = abs(float(pitch_x_rad))
                abs_pitch_rate = abs(float(pitch_rate_x_rad_s))
                abs_roll = abs(float(roll_y_rad)) if roll_y_rad is not None else 0.0

                pitch_safe = abs_pitch < self.authority_schedule.apc_pitch_blend_safe_pitch_rad
                pitch_rate_safe = abs_pitch_rate < self.authority_schedule.apc_pitch_blend_safe_pitch_rate_rad_s
                height_safe = float(com_z_m) > self.authority_schedule.apc_pitch_blend_min_com_z
                roll_safe = abs_roll < self.authority_schedule.apc_pitch_blend_max_roll_rad
                contact_safe = contact_valid

                apc_pitch_blend_pitch_safe = pitch_safe and pitch_rate_safe
                apc_pitch_blend_height_safe = height_safe
                apc_pitch_blend_contact_safe = contact_safe

                if not (pitch_safe and height_safe and roll_safe and contact_safe):
                    apc_pitch_blend_block_reason = "safety"
                else:
                    # All conditions met - compute blend scale based on error magnitude
                    abs_error = abs(sagittal_position_error_m)
                    deep_thresh = self.authority_schedule.apc_pitch_blend_deep_error_m
                    mid_thresh = self.authority_schedule.apc_pitch_blend_mid_error_m
                    soft_thresh = self.authority_schedule.apc_pitch_blend_soft_error_m

                    if abs_error > deep_thresh:
                        apc_pitch_blend_scale = self.authority_schedule.apc_pitch_blend_scale_deep
                    elif abs_error > mid_thresh:
                        apc_pitch_blend_scale = self.authority_schedule.apc_pitch_blend_scale_mid
                    elif abs_error > soft_thresh:
                        apc_pitch_blend_scale = self.authority_schedule.apc_pitch_blend_scale_soft
                    else:
                        apc_pitch_blend_scale = self.authority_schedule.apc_pitch_blend_scale_near

                    # Only mark as active if scale < 1.0
                    if apc_pitch_blend_scale < 1.0:
                        apc_pitch_blend_active = True
                        # Apply blend
                        tau_pitch = tau_pitch * apc_pitch_blend_scale
                        tau_pitch_clipped = tau_pitch

        # =====================================================================
        # T6F Sign Fix: Enhanced Pitch Suppression
        # MOVED AFTER ARCH_FIX_ACTIVE IS SET (Phase 2 fix for Phase 6 bug)
        # Previously at line 2027, now after line 2253
        # =====================================================================
        # Pitch suppression moved to after arch_fix_active computation

        tau_pitch_rate = effective_kd_pitch * pitch_rate_for_damping
        tau_sagittal_velocity = -effective_k_velocity * effective_velocity_damping_scale * sagittal_velocity_m_s

        # Support position velocity damping term
        # Directly opposes support-center drift velocity to prevent transient position excursions
        tau_support_velocity = -effective_support_velocity_gain * effective_support_velocity_scale * support_vel_for_damping

        # Capture-point-like term matching baseline controller's cp/com_vy contributions
        # Uses sagittal_position_error as proxy for cp_error and sagittal_velocity as proxy for com_vy
        # when running in initial-heading frame mode. Disabled (0.0) by default to avoid
        # fighting the separate k_position/k_velocity terms.
        tau_cp = -self.kp_cp * sagittal_position_error_m
        tau_com_vy = -self.kd_com_vy * sagittal_velocity_m_s

        # Position hold term with optional smart capture gating
        tau_position_p = -effective_k_position * sagittal_position_error_m
        tau_position_raw = tau_position_p

        integral_active = False
        integral_gate_reason = "disabled"
        integral_saturation_flag = False
        if self.enable_position_integral:
            abs_pitch_error = abs(float(pitch_x_rad))
            abs_pitch_rate = abs(float(pitch_rate_x_rad_s))
            abs_support_velocity = abs(float(support_position_velocity_m_s))
            abs_wheel_velocity_mean = abs(float(wheel_vel_mean))
            com_z_safe = self.integral_min_com_z_m <= float(com_z_m) <= self.integral_max_com_z_m

            if abs_pitch_error > self.integral_pitch_error_threshold_rad:
                integral_gate_reason = "pitch_error_large"
            elif abs_pitch_rate > self.integral_pitch_rate_threshold_rad_s:
                integral_gate_reason = "pitch_rate_large"
            elif abs_support_velocity > self.integral_support_velocity_threshold_m_s:
                integral_gate_reason = "support_velocity_large"
            elif abs_wheel_velocity_mean > self.integral_wheel_velocity_threshold_rad_s:
                integral_gate_reason = "wheel_velocity_large"
            elif not com_z_safe:
                integral_gate_reason = "height_unsafe"
            elif abs(float(roll_y_rad)) > self.integral_roll_error_threshold_rad:
                integral_gate_reason = "roll_error_large"
            elif not contact_valid:
                integral_gate_reason = "contact_invalid"
            else:
                integral_active = True
                integral_gate_reason = "safe_steady_state"

            if integral_active:
                self.position_integral_error += float(sagittal_position_error_m) * self.dt
            else:
                self.position_integral_error = 0.0

        tau_position_integral_unclipped = -self.ki_position_integral * self.position_integral_error
        tau_position_integral = float(max(
            -self.integral_max_abs,
            min(self.integral_max_abs, float(tau_position_integral_unclipped)),
        ))
        integral_saturation_flag = abs(tau_position_integral - float(tau_position_integral_unclipped)) > 1e-9
        if not integral_active:
            tau_position_integral = 0.0
            integral_saturation_flag = False

        tau_position_raw = tau_position_p + tau_position_integral

        # Apply capture gate if enabled
        capture_gate_diagnostics = None
        if self.enable_capture_gate and self.capture_gate is not None:
            tau_position_before_clip, capture_gate_diagnostics = self.capture_gate.apply_gate(
                tau_position_raw=tau_position_raw,
                pitch_x_rad=pitch_x_rad,
                pitch_rate_x_rad_s=pitch_rate_x_rad_s,
                com_y_m=com_y_m,
                com_vy_m_s=com_vy_m_s,
                support_center_y_m=support_center_y_m,
                com_z_m=com_z_m,
            )
        else:
            tau_position_before_clip = tau_position_raw

        # Pitch-aware position scaling (Option C pitch-safe fix)
        pitch_aware_position_scale = 1.0
        pitch_aware_active = False
        tau_position_before_pitch_scale = tau_position_before_clip

        if self.authority_schedule.enable_pitch_aware_position_scaling:
            pitch_abs = abs(float(pitch_x_rad))
            pitch_soft_start = self.authority_schedule.pitch_soft_start
            pitch_hard_limit = self.authority_schedule.pitch_hard_limit
            min_pitch_scale = self.authority_schedule.min_pitch_scale

            if pitch_abs > pitch_soft_start:
                # Compute smoothstep scaling
                u_pitch = (pitch_abs - pitch_soft_start) / (pitch_hard_limit - pitch_soft_start)
                u_pitch_clamped = max(0.0, min(1.0, u_pitch))
                smoothstep_pitch = u_pitch_clamped * u_pitch_clamped * (3.0 - 2.0 * u_pitch_clamped)

                # Scale from 1.0 to min_pitch_scale
                pitch_aware_position_scale = 1.0 - smoothstep_pitch * (1.0 - min_pitch_scale)
                pitch_aware_active = True

            # Apply scaling to position torque
            tau_position_before_clip = tau_position_before_clip * pitch_aware_position_scale

        # Torque-budget-aware position authority allocation
        if self.enable_torque_budget_aware_position:
            # Compute balance torque before position (all terms except position)
            tau_balance_before_position = (
                tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
                tau_support_velocity + tau_cp + tau_com_vy +
                0.5 * (tau_wheel_vel_left + tau_wheel_vel_right)
            )

            tau_position_lower_bound = -self.max_tau_wheel - float(tau_balance_before_position)
            tau_position_upper_bound = self.max_tau_wheel - float(tau_balance_before_position)

            tau_balance_before_position_log = float(tau_balance_before_position)

            tau_position = float(
                jnp.clip(
                    tau_position_before_clip,
                    tau_position_lower_bound,
                    tau_position_upper_bound,
                )
            )
            tau_position_total_bound_clipped = abs(tau_position - float(tau_position_before_clip)) > 1e-9
            tau_pitch_reserve_applied = 0.0

            if abs(tau_position_before_clip) < 1e-9:
                tau_position_saturation_reason = "none"
                position_authority_reason = "within_bounds"
            elif tau_position <= tau_position_lower_bound + 1e-9 and tau_position_before_clip < tau_position_lower_bound:
                tau_position_saturation_reason = "lower_bound"
                position_authority_reason = "lower_bound"
            elif tau_position >= tau_position_upper_bound - 1e-9 and tau_position_before_clip > tau_position_upper_bound:
                tau_position_saturation_reason = "upper_bound"
                position_authority_reason = "upper_bound"
            else:
                tau_position_saturation_reason = "none"
                position_authority_reason = "within_bounds"

            tau_position_saturated = tau_position_total_bound_clipped

            tau_position_budget_available = float(
                tau_position_upper_bound if tau_position_before_clip >= 0.0 else -tau_position_lower_bound
            )
            tau_position_budget_allowed = float(
                max(0.0, tau_position_upper_bound)
                if tau_position_before_clip >= 0.0
                else max(0.0, -tau_position_lower_bound)
            )
            tau_position_budget_cap = float(self.position_tau_budget_cap)
            pitch_reserve_tau_log = 0.0
            position_authority_mode = "total_torque_bound"

        else:
            # T6F Architecture Fix: Apply before upstream clip
            # Check if we should raise the cap based on height, drift severity, and safety
            if self.authority_schedule.arch_fix_enabled:
                # Recompute gates now that we have all necessary state
                arch_fix_height_gate_pass = schedule_height_ref >= self.authority_schedule.arch_fix_height_threshold_m

                # Determine band state from abs_error (same logic as APCR1nD)
                abs_error = abs(float(sagittal_position_error_m))
                emergency_band_m = self.authority_schedule.apcr1nd_emergency_band_m
                hard_band_m = self.authority_schedule.apcr1nd_hard_band_m

                # Band gate: hard or emergency
                in_hard_band = abs_error >= hard_band_m and abs_error < emergency_band_m
                in_emergency_band = abs_error >= emergency_band_m
                arch_fix_band_gate_pass = in_hard_band or in_emergency_band

                # Safety gate: same as APCR1n
                arch_fix_safety_gate_pass = (
                    contact_valid
                    and com_z_m >= self.authority_schedule.recenter_priority_safe_min_com_z
                    and abs(float(roll_y_rad)) <= self.authority_schedule.recenter_priority_safe_roll_rad
                    and abs(float(pitch_x_rad)) <= self.authority_schedule.recenter_priority_safe_pitch_rad
                )

                # Recenter gate: will be determined by APCR1nD later, but we can check if enabled
                arch_fix_recenter_gate_pass = self.authority_schedule.recenter_priority_direct_enabled

                # All gates must pass
                all_gates_pass = (
                    arch_fix_height_gate_pass
                    and arch_fix_band_gate_pass
                    and arch_fix_safety_gate_pass
                    and arch_fix_recenter_gate_pass
                )

                if all_gates_pass:
                    # Determine which cap to apply based on band
                    if in_emergency_band:
                        arch_fix_requested_cap = self.authority_schedule.arch_fix_emergency_max_position_tau
                        arch_fix_reason = "emergency_band"
                    elif in_hard_band:
                        arch_fix_requested_cap = self.authority_schedule.arch_fix_hard_max_position_tau
                        arch_fix_reason = "hard_band"
                    else:
                        arch_fix_requested_cap = effective_max_position_tau
                        arch_fix_reason = "band_below_hard"

                    # Raise the cap (take max to ensure we don't lower it)
                    effective_max_position_tau = max(
                        float(effective_max_position_tau),
                        float(arch_fix_requested_cap)
                    )
                    arch_fix_active = True

                    # =====================================================================
                    # T6H Soft Blend Arch Fix: Soft Pitch Blending
                    # Reduces pitch authority by 50% (not 100%) during arch_fix
                    # Preserves partial stabilization while reducing fighting terms
                    # =====================================================================
                    t6h_soft_pitch_blend_active = False
                    t6h_pitch_blend_factor = 1.0
                    t6h_pitch_safety_active = False

                    if self.authority_schedule.t6h_enabled and arch_fix_active:
                        abs_sagittal_error = abs(float(sagittal_position_error_m))
                        pitch_error_threshold = self.authority_schedule.t6h_pitch_error_threshold_m
                        pitch_safety_threshold_rad = self.authority_schedule.t6h_pitch_safety_threshold_deg * (3.14159 / 180.0)

                        # Check pitch safety override
                        if abs(float(pitch_x_rad)) > pitch_safety_threshold_rad:
                            # Pitch too large - restore full pitch control
                            t6h_pitch_blend_factor = 1.0
                            t6h_pitch_safety_active = True
                        elif abs_sagittal_error > pitch_error_threshold:
                            # Error large - apply soft blend
                            t6h_pitch_blend_factor = self.authority_schedule.t6h_soft_pitch_blend_factor
                            t6h_soft_pitch_blend_active = True
                        else:
                            # Error small - preserve full pitch
                            t6h_pitch_blend_factor = 1.0

                        # Apply blend to pitch torque
                        tau_pitch = tau_pitch * t6h_pitch_blend_factor
                        tau_pitch_clipped = tau_pitch_clipped * t6h_pitch_blend_factor

                    # =====================================================================
                    # T6I Phase-Aware Release: Convergence Detection and Cap Decay
                    # Detects error convergence and gradually releases high authority
                    # Preserves full pitch/damping authority (no suppression)
                    # =====================================================================
                    t6i_error_converging = False
                    t6i_error_trend = 0.0
                    t6i_target_cap = float(effective_max_position_tau)
                    t6i_current_cap = self._t6i_current_cap
                    t6i_cap_delta_this_step = 0.0
                    t6i_cap_change_rate_limited = False
                    t6i_release_reason = "none"

                    if self.authority_schedule.t6i_enabled and arch_fix_active:
                        # Update error history
                        current_error = float(sagittal_position_error_m)
                        self._t6i_error_history.append(current_error)
                        max_history = self.authority_schedule.t6i_convergence_window_steps
                        if len(self._t6i_error_history) > max_history:
                            self._t6i_error_history.pop(0)

                        # Detect convergence
                        if len(self._t6i_error_history) >= self.authority_schedule.t6i_convergence_window_steps:
                            recent_errors = self._t6i_error_history[-5:]
                            error_trend = recent_errors[-1] - recent_errors[0]
                            abs_error = abs(current_error)

                            # Converging if: error < threshold AND trend shows decreasing magnitude
                            converging = (
                                abs_error < self.authority_schedule.t6i_convergence_threshold_m and
                                abs(error_trend) < self.authority_schedule.t6i_convergence_trend_threshold_m and
                                # Same sign (not crossing zero rapidly)
                                (current_error * recent_errors[0] > 0 or abs(current_error) < 0.01)
                            )

                            t6i_error_converging = converging
                            t6i_error_trend = error_trend
                            self._t6i_converging = converging

                        # Compute target cap
                        if self._t6i_converging:
                            # Decay toward minimum cap
                            decay_rate = self.authority_schedule.t6i_cap_decay_rate_nm_per_step
                            min_cap = self.authority_schedule.t6i_cap_min_nm
                            t6i_target_cap = max(min_cap, self._t6i_current_cap - decay_rate)
                            t6i_release_reason = "converging"
                        else:
                            # Use arch_fix raised cap
                            if arch_fix_band_gate_pass:
                                t6i_target_cap = float(arch_fix_requested_cap)
                                t6i_release_reason = "arch_fix_active"
                            else:
                                t6i_target_cap = float(effective_max_position_tau)
                                t6i_release_reason = "normal"

                        # Rate-limit cap transition
                        max_delta = self.authority_schedule.t6i_max_cap_delta_per_step_nm
                        cap_delta = t6i_target_cap - self._t6i_current_cap

                        if abs(cap_delta) > max_delta:
                            cap_delta = max_delta if cap_delta > 0 else -max_delta
                            t6i_cap_change_rate_limited = True

                        self._t6i_current_cap = self._t6i_current_cap + cap_delta
                        t6i_current_cap = self._t6i_current_cap
                        t6i_cap_delta_this_step = cap_delta

                        # Apply T6I cap for both converging and non-converging arch-fix-active states
                        effective_max_position_tau = self._t6i_current_cap

                    # =====================================================================
                    # T6F Sign Fix: Enhanced Pitch Suppression
                    # Suppress pitch completely when arch_fix active AND abs(error) > threshold
                    # MOVED HERE (Phase 2 fix): Previously at line 2027 BEFORE arch_fix_active was set
                    # =====================================================================
                    if (self.authority_schedule.sign_fix_enabled and
                        self.authority_schedule.sign_fix_suppress_pitch_during_arch_fix):

                        abs_sagittal_error = abs(float(sagittal_position_error_m))
                        pitch_error_threshold = self.authority_schedule.sign_fix_pitch_error_threshold_m

                        if abs_sagittal_error > pitch_error_threshold:
                            # Suppress pitch during large error + arch_fix
                            sign_fix_pitch_original_nm = float(tau_pitch)
                            tau_pitch = 0.0
                            tau_pitch_clipped = 0.0
                            sign_fix_pitch_after_nm = 0.0
                            sign_fix_pitch_suppressed = True
                        else:
                            # Preserve pitch for small error
                            sign_fix_pitch_original_nm = float(tau_pitch)
                            sign_fix_pitch_after_nm = float(tau_pitch)

                else:
                    # Determine which gate failed
                    if not arch_fix_height_gate_pass:
                        arch_fix_reason = "height_below_threshold"
                    elif not arch_fix_band_gate_pass:
                        arch_fix_reason = "band_below_hard"
                    elif not arch_fix_safety_gate_pass:
                        arch_fix_reason = "safety_gate_fail"
                    elif not arch_fix_recenter_gate_pass:
                        arch_fix_reason = "recenter_disabled"
                    else:
                        arch_fix_reason = "unknown_gate_fail"

                    arch_fix_requested_cap = 0.0
                    arch_fix_active = False

            # Legacy fixed-cap clipping (now uses potentially raised effective_max_position_tau)
            tau_position = float(jnp.clip(tau_position_before_clip, -effective_max_position_tau, effective_max_position_tau))
            if t6j_bias_trim_enabled:
                signed_trim_error = float(sagittal_position_error_m)
                self._t6j_bias_error_history.append(signed_trim_error)
                max_bias_history = max(1, int(self.authority_schedule.t6j_bias_trim_window_steps))
                if len(self._t6j_bias_error_history) > max_bias_history:
                    self._t6j_bias_error_history.pop(0)

                t6j_bias_mean_error_m = float(sum(self._t6j_bias_error_history) / len(self._t6j_bias_error_history))
                if t6j_bias_mean_error_m > 0.0:
                    self._t6j_bias_positive_duration_steps += 1
                    self._t6j_bias_negative_duration_steps = 0
                elif t6j_bias_mean_error_m < 0.0:
                    self._t6j_bias_negative_duration_steps += 1
                    self._t6j_bias_positive_duration_steps = 0
                else:
                    self._t6j_bias_positive_duration_steps = 0
                    self._t6j_bias_negative_duration_steps = 0

                abs_pitch_deg = abs(float(pitch_x_rad)) * 180.0 / 3.141592653589793
                abs_roll_deg = abs(float(roll_y_rad)) * 180.0 / 3.141592653589793
                abs_wheel_vel = abs(float(wheel_vel_mean))
                abs_error = abs(signed_trim_error)
                upright_ok = (
                    (not self.authority_schedule.t6j_bias_trim_only_when_upright) or
                    (
                        abs_pitch_deg <= self.authority_schedule.t6j_bias_trim_disable_if_pitch_gt_deg and
                        abs_roll_deg <= self.authority_schedule.t6j_bias_trim_disable_if_roll_gt_deg
                    )
                )
                contact_ok = (
                    (not self.authority_schedule.t6j_bias_trim_only_when_contact_stable) or
                    bool(contact_valid)
                )
                wheel_ok = abs_wheel_vel <= self.authority_schedule.t6j_bias_trim_disable_if_wheel_vel_gt_rad_s
                error_ok = abs_error <= self.authority_schedule.t6j_bias_trim_disable_if_abs_error_gt_m
                t6j_bias_safety_gate_pass = bool(upright_ok and contact_ok and wheel_ok and error_ok)

                if not upright_ok:
                    t6j_bias_block_reason = "upright_gate_fail"
                elif not contact_ok:
                    t6j_bias_block_reason = "contact_unstable"
                elif not wheel_ok:
                    t6j_bias_block_reason = "wheel_velocity_high"
                elif not error_ok:
                    t6j_bias_block_reason = "abs_error_too_large"
                else:
                    enter_threshold = self.authority_schedule.t6j_bias_trim_enter_threshold_m
                    exit_threshold = self.authority_schedule.t6j_bias_trim_exit_threshold_m
                    max_tau = self.authority_schedule.t6j_bias_trim_max_tau_nm
                    if t6j_bias_mean_error_m >= enter_threshold:
                        t6j_bias_trim_target_tau_nm = -max_tau
                        t6j_bias_trim_active = True
                        t6j_bias_block_reason = "positive_bias_correcting"
                    elif t6j_bias_mean_error_m <= -enter_threshold:
                        t6j_bias_trim_target_tau_nm = max_tau
                        t6j_bias_trim_active = True
                        t6j_bias_block_reason = "negative_bias_correcting"
                    elif abs(t6j_bias_mean_error_m) <= exit_threshold:
                        t6j_bias_trim_target_tau_nm = 0.0
                        t6j_bias_block_reason = "inside_exit_threshold"
                    else:
                        t6j_bias_trim_target_tau_nm = 0.0
                        t6j_bias_block_reason = "hold_between_thresholds"

                if not t6j_bias_safety_gate_pass:
                    t6j_bias_trim_target_tau_nm = 0.0

                current_trim = float(self._t6j_bias_trim_tau)
                target_trim = float(max(-self.authority_schedule.t6j_bias_trim_max_tau_nm, min(self.authority_schedule.t6j_bias_trim_max_tau_nm, t6j_bias_trim_target_tau_nm)))
                max_step = (
                    self.authority_schedule.t6j_bias_trim_decay_rate_nm_per_step
                    if abs(target_trim) < abs(current_trim)
                    else self.authority_schedule.t6j_bias_trim_rate_nm_per_step
                )
                trim_delta = target_trim - current_trim
                if abs(trim_delta) > max_step:
                    trim_delta = max_step if trim_delta > 0.0 else -max_step
                    t6j_bias_trim_rate_limited = True
                updated_trim = current_trim + trim_delta
                updated_trim = max(-self.authority_schedule.t6j_bias_trim_max_tau_nm, min(self.authority_schedule.t6j_bias_trim_max_tau_nm, updated_trim))
                self._t6j_bias_trim_target_tau = target_trim
                self._t6j_bias_trim_tau = updated_trim
                t6j_bias_trim_target_tau_nm = target_trim
                t6j_bias_trim_tau_nm = updated_trim
                t6j_bias_applied_to_final_tau = updated_trim
                if t6j_bias_mean_error_m > 0.0:
                    t6j_bias_expected_direction_correct = updated_trim <= 0.0
                elif t6j_bias_mean_error_m < 0.0:
                    t6j_bias_expected_direction_correct = updated_trim >= 0.0
                else:
                    t6j_bias_expected_direction_correct = abs(updated_trim) < 1e-9

                tau_position_with_trim = float(tau_position + updated_trim)
                tau_position = float(jnp.clip(tau_position_with_trim, -effective_max_position_tau, effective_max_position_tau))
                t6j_bias_applied_to_final_tau = float(updated_trim)

            # =====================================================================
            # Adaptive Centering Bias Trim (ASCT): proportional, height-aware trim
            # Replaces bang-bang T6J with smooth proportional authority
            # =====================================================================
            if adaptive_bias_trim_enabled:
                signed_error = float(sagittal_position_error_m)
                sch = self.authority_schedule

                # --- Update slow and fast error histories ---
                self._adaptive_bias_slow_error_history.append(signed_error)
                if len(self._adaptive_bias_slow_error_history) > int(sch.adaptive_bias_window_steps):
                    self._adaptive_bias_slow_error_history.pop(0)
                self._adaptive_bias_fast_error_history.append(signed_error)
                if len(self._adaptive_bias_fast_error_history) > int(sch.adaptive_bias_fast_window_steps):
                    self._adaptive_bias_fast_error_history.pop(0)

                # Compute means
                slow_n = max(1, len(self._adaptive_bias_slow_error_history))
                fast_n = max(1, len(self._adaptive_bias_fast_error_history))
                mean_err = sum(self._adaptive_bias_slow_error_history) / slow_n
                fast_mean_err = sum(self._adaptive_bias_fast_error_history) / fast_n
                adaptive_bias_mean_error_m = mean_err
                adaptive_bias_fast_mean_error_m = fast_mean_err

                # --- Height-scheduled max trim ---
                com_z = float(com_z_m)
                z_low = float(sch.adaptive_bias_height_low_m)
                z_high = float(sch.adaptive_bias_height_high_m)
                z_extreme = float(sch.adaptive_bias_height_extreme_m)
                max_low = float(sch.adaptive_bias_max_tau_low_nm)
                max_high = float(sch.adaptive_bias_max_tau_high_nm)
                max_extreme = float(sch.adaptive_bias_max_tau_extreme_nm)
                if com_z <= z_low:
                    max_tau_current = max_low
                    adaptive_bias_height_scale = 0.0
                elif com_z >= z_extreme:
                    max_tau_current = max_extreme
                    adaptive_bias_height_scale = 1.0
                else:
                    # Linear interpolation: low->high for [z_low, z_high], high->extreme for [z_high, z_extreme]
                    if com_z <= z_high:
                        t = (com_z - z_low) / max(z_high - z_low, 1e-9)
                    else:
                        t = 1.0 + (com_z - z_high) / max(z_extreme - z_high, 1e-9)
                    t = max(0.0, min(2.0, t))
                    if t <= 1.0:
                        max_tau_current = max_low + (max_high - max_low) * t
                    else:
                        max_tau_current = max_high + (max_extreme - max_high) * (t - 1.0)
                    adaptive_bias_height_scale = min(1.0, t)
                adaptive_bias_max_tau_current_nm = max_tau_current

                # --- Zero-crossing detection and guard ---
                self._adaptive_bias_zero_crossing_history.append(signed_error)
                max_zc_len = max(1, int(sch.adaptive_bias_zero_crossing_window_steps))
                if len(self._adaptive_bias_zero_crossing_history) > max_zc_len:
                    self._adaptive_bias_zero_crossing_history.pop(0)
                zc_window = self._adaptive_bias_zero_crossing_history
                zc_count = 0
                for i in range(1, len(zc_window)):
                    if (zc_window[i-1] < 0) != (zc_window[i] < 0):
                        zc_count += 1
                adaptive_bias_zero_crossing_count = zc_count
                zc_guard_active = False
                if sch.adaptive_bias_zero_crossing_guard_enabled and zc_count > int(sch.adaptive_bias_zero_crossing_limit):
                    zc_guard_active = True
                    self._adaptive_bias_guard_trigger_count += 1
                    if self._adaptive_bias_guard_trigger_count >= 3:
                        # Reset counter after 3 consecutive triggers
                        self._adaptive_bias_guard_trigger_count = 0
                else:
                    self._adaptive_bias_guard_trigger_count = 0
                adaptive_bias_zero_crossing_guard_active = zc_guard_active

                # Apply guard scale
                guard_scale = 1.0
                if zc_guard_active:
                    guard_scale = float(sch.adaptive_bias_zero_crossing_max_scale)
                max_tau_guarded = max_tau_current * guard_scale

                # --- Positive/negative area in slow window ---
                pos_a = sum(v for v in self._adaptive_bias_slow_error_history if v > 0)
                neg_a = abs(sum(v for v in self._adaptive_bias_slow_error_history if v < 0))
                total_a = pos_a + neg_a
                adaptive_bias_positive_area = pos_a
                adaptive_bias_negative_area = neg_a
                adaptive_bias_symmetry_ratio = (abs(pos_a - neg_a) / total_a) if total_a > 1e-9 else 0.0

                # --- Safety gates ---
                abs_pitch_deg = abs(float(pitch_x_rad)) * 180.0 / math.pi
                abs_roll_deg = abs(float(roll_y_rad)) * 180.0 / math.pi
                contact_ok = (not sch.adaptive_bias_only_when_contact_stable) or bool(contact_valid)
                upright_ok = (
                    (not sch.adaptive_bias_only_when_upright)
                    or (abs_pitch_deg <= sch.adaptive_bias_disable_if_pitch_gt_deg
                        and abs_roll_deg <= sch.adaptive_bias_disable_if_roll_gt_deg)
                )
                # hip_yaw_abs_max not directly available in compute() scope;
                # use the same pattern as the rest of the controller: default to 0.0
                # and let telemetry reveal it post-hoc. This avoids a hard dependency
                # on an absent local variable.
                try:
                    hy_val = float(hip_yaw_abs_max_tracking)
                except (NameError, TypeError, ValueError):
                    hy_val = 0.0
                hy_ok = hy_val <= float(sch.adaptive_bias_disable_if_hip_yaw_gt_rad)
                adaptive_bias_hip_yaw_gate_pass = hy_ok
                adaptive_bias_hip_yaw_abs_max = hy_val
                abs_error_ok = abs(signed_error) <= float(sch.adaptive_bias_disable_if_abs_error_gt_m)
                pitch_ok = abs_pitch_deg <= float(sch.adaptive_bias_disable_if_pitch_gt_deg)
                roll_ok = abs_roll_deg <= float(sch.adaptive_bias_disable_if_roll_gt_deg)

                safety_pass = bool(contact_ok and upright_ok and hy_ok and abs_error_ok)
                adaptive_bias_safety_gate_pass = safety_pass

                if not contact_ok:
                    adaptive_bias_block_reason = "contact_unstable"
                elif not upright_ok:
                    adaptive_bias_block_reason = "upright_gate_fail"
                elif not hy_ok:
                    adaptive_bias_block_reason = "hip_yaw_unsafe"
                elif not abs_error_ok:
                    adaptive_bias_block_reason = "abs_error_too_large"
                elif zc_guard_active:
                    adaptive_bias_block_reason = "zero_crossing_guard"
                else:
                    adaptive_bias_block_reason = "ok"

                # --- Proportional target computation ---
                enter_th = float(sch.adaptive_bias_enter_threshold_m)
                exit_th = float(sch.adaptive_bias_exit_threshold_m)
                relief_th = float(sch.adaptive_bias_relief_hysteresis_m)
                k_tau = float(sch.adaptive_bias_k_tau_per_m)

                sign_err = 1.0 if mean_err > 1e-9 else (-1.0 if mean_err < -1e-9 else 0.0)
                err_sign_changed = (sign_err != 0) and (sign_err != self._adaptive_bias_prev_error_sign)

                # Sign-reversal guard: hold before reversing
                if err_sign_changed:
                    self._adaptive_bias_hold_steps = int(sch.adaptive_bias_sign_reversal_hold_steps)
                    self._adaptive_bias_prev_error_sign = sign_err
                elif self._adaptive_bias_hold_steps > 0:
                    self._adaptive_bias_hold_steps -= 1
                    self._adaptive_bias_prev_error_sign = sign_err

                sign_reversal_blocked = (self._adaptive_bias_hold_steps > 0) and err_sign_changed
                adaptive_bias_sign_reversal_blocked = sign_reversal_blocked

                # Near-zero relief
                near_zero = abs(mean_err) <= exit_th
                in_hysteresis = abs(mean_err) <= exit_th + relief_th
                adaptive_bias_near_zero_relief_active = near_zero

                if near_zero:
                    # Inside exit threshold: target zero
                    raw_target = 0.0
                elif sign_reversal_blocked:
                    # Sign reversal in progress: decay toward zero
                    raw_target = 0.0
                elif in_hysteresis:
                    # In hysteresis band: hold current
                    raw_target = float(self._adaptive_bias_trim_tau)
                else:
                    # Normal proportional target
                    eff_err = mean_err - sign_err * exit_th
                    raw_target = -k_tau * eff_err

                adaptive_bias_effective_error_m = mean_err - sign_err * exit_th if not near_zero and not sign_reversal_blocked else 0.0

                # Apply height/guard ceiling
                clipped_target = max(-max_tau_guarded, min(max_tau_guarded, raw_target))
                adaptive_bias_target_tau_nm = float(clipped_target)

                # Rate limiting
                current_trim_a = float(self._adaptive_bias_trim_tau)
                is_decay = abs(clipped_target) < abs(current_trim_a)
                rate = float(sch.adaptive_bias_decay_rate_nm_per_step) if is_decay else float(sch.adaptive_bias_rate_nm_per_step)
                adaptive_bias_rate_used_nm_per_step = rate

                trim_delta_a = clipped_target - current_trim_a
                if abs(trim_delta_a) > rate:
                    trim_delta_a = rate if trim_delta_a > 0.0 else -rate
                updated_trim_a = current_trim_a + trim_delta_a
                updated_trim_a = max(-max_tau_guarded, min(max_tau_guarded, updated_trim_a))

                self._adaptive_bias_trim_target_tau = clipped_target
                self._adaptive_bias_trim_tau = updated_trim_a
                adaptive_bias_tau_nm = updated_trim_a

                # Direction correctness
                if mean_err > 1e-9:
                    adaptive_bias_expected_direction_correct = updated_trim_a <= 0.0
                elif mean_err < -1e-9:
                    adaptive_bias_expected_direction_correct = updated_trim_a >= 0.0
                else:
                    adaptive_bias_expected_direction_correct = abs(updated_trim_a) < 1e-9

                # Phase 0: ABS trim state/timing trace — capture all intermediates
                # for JAX parity comparison. Accessed by simulate_hierarchical_controller.py.
                self._py_abs_trim_trace = {
                    'signed_error': float(signed_error),
                    'mean_err': float(mean_err),
                    'fast_mean_err': float(fast_mean_err),
                    'sign_err': float(sign_err),
                    'max_tau_current': float(max_tau_current),
                    'max_tau_g': float(max_tau_guarded),
                    'guard_scale': float(guard_scale),
                    'raw_target': float(raw_target),
                    'clipped_target': float(clipped_target),
                    'is_decay': bool(is_decay),
                    'rate': float(rate),
                    'trim_delta': float(trim_delta_a),
                    'new_trim': float(updated_trim_a),
                    'safety_pass': bool(safety_pass),
                    'trim_to_apply': float(updated_trim_a),  # same as new_trim, gated later
                    'tau_position_before_trim': float(tau_position),
                    'tau_position_raw': float(tau_position_raw),
                    'effective_max_position_tau': float(effective_max_position_tau),
                    'hold_steps': int(self._adaptive_bias_hold_steps),
                    'err_sign_changed': bool(err_sign_changed),
                    'sign_rev_blocked': bool(sign_reversal_blocked),
                    'near_zero': bool(near_zero),
                    'in_hysteresis': bool(in_hysteresis),
                    'zc_guard_active': bool(zc_guard_active),
                    # Phase 0: safety gate component diagnostics for parity tracing
                    'safety_contact_ok': bool(contact_ok),
                    'safety_upright_ok': bool(upright_ok),
                    'safety_hy_ok': bool(hy_ok),
                    'safety_abs_error_ok': bool(abs_error_ok),
                    'safety_pitch_deg': float(abs_pitch_deg),
                    'safety_roll_deg': float(abs_roll_deg),
                }

                # Apply to tau_position AFTER T6J block (or independently if T6J disabled)
                adaptive_trim_to_apply = float(updated_trim_a)
                if safety_pass:
                    # Only apply if safety gates pass
                    tau_position = float(jnp.clip(
                        tau_position + adaptive_trim_to_apply,
                        -effective_max_position_tau,
                        effective_max_position_tau
                    ))
                    # "Active" = meaningful corrective trim is being applied, not merely
                    # that the gates passed. Trim is inactive at near-zero error, during
                    # sign-reversal hold, or when magnitude is negligible.
                    adaptive_bias_trim_active = bool(
                        (not near_zero)
                        and (not sign_reversal_blocked)
                        and abs(updated_trim_a) > 1e-9
                    )
                    if adaptive_bias_trim_active:
                        adaptive_bias_block_reason = "ok"
            tau_position_total_bound_clipped = False
            tau_position_saturated = abs(tau_position_before_clip) >= effective_max_position_tau * 0.99
            tau_position_saturation_reason = "fixed_cap" if tau_position_saturated else "none"
            position_authority_reason = tau_position_saturation_reason

            # Telemetry placeholders
            tau_balance_before_position_log = 0.0
            tau_position_budget_available = 0.0
            tau_position_budget_allowed = 0.0
            tau_position_budget_cap = float(effective_max_position_tau)
            pitch_reserve_tau_log = 0.0
            tau_pitch_reserve_applied = 0.0
            tau_position_lower_bound = -float(effective_max_position_tau)
            tau_position_upper_bound = float(effective_max_position_tau)
            position_authority_mode = "scheduled_fixed_cap" if schedule_active else "fixed_cap"

        # =====================================================================
        # Zero-Crossing Support Recenter (ZC): hysteresis hold-through-zero
        # Supplements adaptive_bias_trim when enable_zero_crossing_recenter is True
        # Forces drift to cross to opposite side before releasing correction
        # =====================================================================
        zc_state = "CENTER_IDLE"
        zc_state_id = 0
        zc_active = False
        zc_direction = 0
        zc_tau_nm = 0.0
        zc_target_tau_nm = 0.0
        zc_crossed_zero = False
        zc_cross_target_reached = False
        zc_safety_gate_pass = True
        zc_block_reason = "none"
        zc_expected_direction_correct = True

        if self.authority_schedule.enable_zero_crossing_recenter:
            sch = self.authority_schedule

            # Get primary drift signal (active_pitch_crossing_signed_error_m)
            signed_error = float(sagittal_position_error_m)
            abs_error = abs(signed_error)
            pitch_rad = float(pitch_x_rad)
            roll_rad = float(roll_y_rad)

            # Get hip_yaw if available (column 534)
            hip_yaw_abs = 0.0
            if hasattr(self, '_hip_yaw_for_zc'):
                hip_yaw_abs = abs(self._hip_yaw_for_zc)

            # ZC Safety gate
            if abs_error > sch.zc_disable_if_abs_error_gt_m:
                zc_safety_gate_pass = False
                zc_block_reason = "error_too_large"
            elif abs(pitch_rad) > sch.zc_disable_if_pitch_gt_deg * (3.14159 / 180.0):
                zc_safety_gate_pass = False
                zc_block_reason = "pitch_unsafe"
            elif abs(roll_rad) > sch.zc_disable_if_roll_gt_deg * (3.14159 / 180.0):
                zc_safety_gate_pass = False
                zc_block_reason = "roll_unsafe"
            elif hip_yaw_abs > sch.zc_disable_if_hip_yaw_gt_rad:
                zc_safety_gate_pass = False
                zc_block_reason = "hip_yaw_unsafe"

            # State machine transitions
            prev_state = self._zc_state

            if self._zc_state == "CENTER_IDLE":
                # Check entry conditions
                if zc_safety_gate_pass:
                    if signed_error > sch.zc_enter_m:
                        self._zc_state = "RECENTER_FROM_POSITIVE"
                        self._zc_direction = -1  # negative correction
                        self._zc_hold_steps = 0
                        self._zc_dwell_steps = 0
                        self._zc_enter_event += 1
                        self._zc_episode_id += 1
                        self._zc_episode_start_error = signed_error
                        self._zc_episode_min_error = signed_error
                        self._zc_crossed_zero = False
                        self._zc_cross_target_reached = False
                        self._zc_tau = 0.0
                        self._zc_target_tau = 0.0
                    elif signed_error < -sch.zc_enter_m:
                        self._zc_state = "RECENTER_FROM_NEGATIVE"
                        self._zc_direction = +1  # positive correction
                        self._zc_hold_steps = 0
                        self._zc_dwell_steps = 0
                        self._zc_enter_event += 1
                        self._zc_episode_id += 1
                        self._zc_episode_start_error = signed_error
                        self._zc_episode_max_error = signed_error
                        self._zc_crossed_zero = False
                        self._zc_cross_target_reached = False
                        self._zc_tau = 0.0
                        self._zc_target_tau = 0.0

            elif self._zc_state == "RECENTER_FROM_POSITIVE":
                # Apply negative correction
                self._zc_hold_steps += 1
                self._zc_episode_min_error = min(self._zc_episode_min_error, signed_error)

                # Check if crossed zero
                if signed_error < 0:
                    self._zc_crossed_zero = True
                if signed_error < -sch.zc_cross_target_m:
                    self._zc_cross_target_reached = True

                # Target torque: base + error proportional
                target_tau = sch.zc_base_tau_nm + sch.zc_error_gain_nm_per_m * abs_error
                target_tau = min(target_tau, sch.zc_max_tau_nm)
                target_tau = max(target_tau, sch.zc_base_tau_nm)
                self._zc_target_tau = target_tau

                # Exit conditions
                exit_to_decay = False
                if signed_error <= -sch.zc_cross_target_m:
                    # Crossed to opposite side - success
                    self._zc_state = "HOLD_THROUGH_ZERO"
                    self._zc_exit_event += 1
                    self._zc_exit_reason = "cross_target"
                    self._zc_hold_steps = 0
                elif self._zc_hold_steps >= sch.zc_max_hold_steps:
                    # Max hold reached
                    self._zc_state = "SAFETY_DECAY"
                    self._zc_exit_event += 1
                    self._zc_exit_reason = "max_hold"
                    self._zc_hold_steps = 0
                elif zc_safety_gate_pass and abs_error <= sch.zc_near_zero_band_m and self._zc_hold_steps >= sch.zc_min_hold_steps:
                    # In near-zero band after min hold - check dwell
                    self._zc_dwell_steps += 1
                    if self._zc_dwell_steps >= sch.zc_dwell_steps_for_exit:
                        self._zc_state = "SAFETY_DECAY"
                        self._zc_exit_event += 1
                        self._zc_exit_reason = "dwell_exit"
                        self._zc_hold_steps = 0
                        self._zc_dwell_steps = 0
                elif not zc_safety_gate_pass:
                    # Safety gate failed
                    self._zc_state = "SAFETY_DECAY"
                    self._zc_exit_event += 1
                    self._zc_exit_reason = "safety_gate"
                    self._zc_hold_steps = 0

            elif self._zc_state == "RECENTER_FROM_NEGATIVE":
                # Apply positive correction
                self._zc_hold_steps += 1
                self._zc_episode_max_error = max(self._zc_episode_max_error, signed_error)

                # Check if crossed zero
                if signed_error > 0:
                    self._zc_crossed_zero = True
                if signed_error > sch.zc_cross_target_m:
                    self._zc_cross_target_reached = True

                # Target torque
                target_tau = sch.zc_base_tau_nm + sch.zc_error_gain_nm_per_m * abs_error
                target_tau = min(target_tau, sch.zc_max_tau_nm)
                target_tau = max(target_tau, sch.zc_base_tau_nm)
                self._zc_target_tau = target_tau

                # Exit conditions
                if signed_error >= sch.zc_cross_target_m:
                    self._zc_state = "HOLD_THROUGH_ZERO"
                    self._zc_exit_event += 1
                    self._zc_exit_reason = "cross_target"
                    self._zc_hold_steps = 0
                elif self._zc_hold_steps >= sch.zc_max_hold_steps:
                    self._zc_state = "SAFETY_DECAY"
                    self._zc_exit_event += 1
                    self._zc_exit_reason = "max_hold"
                    self._zc_hold_steps = 0
                elif zc_safety_gate_pass and abs_error <= sch.zc_near_zero_band_m and self._zc_hold_steps >= sch.zc_min_hold_steps:
                    self._zc_dwell_steps += 1
                    if self._zc_dwell_steps >= sch.zc_dwell_steps_for_exit:
                        self._zc_state = "SAFETY_DECAY"
                        self._zc_exit_event += 1
                        self._zc_exit_reason = "dwell_exit"
                        self._zc_hold_steps = 0
                        self._zc_dwell_steps = 0
                elif not zc_safety_gate_pass:
                    self._zc_state = "SAFETY_DECAY"
                    self._zc_exit_event += 1
                    self._zc_exit_reason = "safety_gate"
                    self._zc_hold_steps = 0

            elif self._zc_state == "HOLD_THROUGH_ZERO":
                # Continue reduced correction to ensure full crossing
                self._zc_hold_steps += 1
                hold_tau = min(self._zc_target_tau, sch.zc_base_tau_nm)
                self._zc_target_tau = hold_tau

                # Exit conditions
                if abs_error <= sch.zc_exit_m and self._zc_hold_steps >= sch.zc_min_hold_steps:
                    self._zc_state = "SAFETY_DECAY"
                    self._zc_exit_event += 1
                    self._zc_exit_reason = "hold_through_zero"
                    self._zc_hold_steps = 0
                elif self._zc_hold_steps >= sch.zc_max_hold_steps:
                    self._zc_state = "SAFETY_DECAY"
                    self._zc_exit_event += 1
                    self._zc_exit_reason = "max_hold"
                    self._zc_hold_steps = 0

            elif self._zc_state == "SAFETY_DECAY":
                # Decay correction toward zero
                if abs(self._zc_tau) > 1e-9:
                    decay = sch.zc_decay_nm_per_step
                    if self._zc_tau > 0:
                        self._zc_tau = max(0, self._zc_tau - decay)
                    else:
                        self._zc_tau = min(0, self._zc_tau + decay)
                else:
                    self._zc_tau = 0.0
                    self._zc_direction = 0
                    self._zc_state = "CENTER_IDLE"

            # Rate limit toward target (except in SAFETY_DECAY)
            if self._zc_state != "SAFETY_DECAY" and abs(self._zc_target_tau) > 1e-9:
                is_increase = abs(self._zc_target_tau) > abs(self._zc_tau)
                rate = sch.zc_rate_nm_per_step if is_increase else sch.zc_decay_nm_per_step
                delta = self._zc_target_tau - self._zc_tau
                if abs(delta) > rate:
                    delta = rate if delta > 0 else -rate
                self._zc_tau = self._zc_tau + delta

            # Apply ZC correction to tau_position
            if self._zc_state not in ("CENTER_IDLE", "SAFETY_DECAY"):
                zc_active = True
                # Direction sign is in _zc_direction (-1 or +1), tau is positive magnitude
                zc_tau_signed = self._zc_direction * abs(self._zc_tau)
                tau_position = float(jnp.clip(
                    tau_position + zc_tau_signed,
                    -effective_max_position_tau,
                    effective_max_position_tau
                ))

            # Telemetry output
            zc_state = self._zc_state
            zc_state_id = {"CENTER_IDLE": 0, "RECENTER_FROM_POSITIVE": 1, "RECENTER_FROM_NEGATIVE": 2,
                          "HOLD_THROUGH_ZERO": 3, "SAFETY_DECAY": 4}.get(zc_state, 0)
            zc_direction = self._zc_direction
            zc_tau_nm = float(self._zc_tau) * self._zc_direction  # signed tau
            zc_target_tau_nm = self._zc_target_tau * self._zc_direction  # signed
            zc_crossed_zero = self._zc_crossed_zero
            zc_cross_target_reached = self._zc_cross_target_reached
            zc_safety_gate_pass = zc_safety_gate_pass
            zc_block_reason = zc_block_reason if not zc_safety_gate_pass else "none"

            # Direction correctness
            if signed_error > 1e-9:
                zc_expected_direction_correct = (zc_tau_nm <= 0)
            elif signed_error < -1e-9:
                zc_expected_direction_correct = (zc_tau_nm >= 0)
            else:
                zc_expected_direction_correct = True

        # =====================================================================
        # Early Zero-Crossing Support Recenter (EZC)
        # Key differences from old ZC:
        # - Entry at 0.05 m (earlier) vs 0.08 m
        # - Exit at e <= 0 (not -0.02)
        # - No opposite-side target required
        # - Immediate decay after zero crossing
        # =====================================================================
        ezc_state = "CENTER_IDLE"
        ezc_state_id = 0
        ezc_active = False
        ezc_direction = 0
        ezc_tau_nm = 0.0
        ezc_target_tau_nm = 0.0
        ezc_crossed_zero = False
        ezc_safety_gate_pass = True
        ezc_block_reason = "none"
        ezc_expected_direction_correct = True

        if self.authority_schedule.enable_early_zero_crossing_recenter:
            sch = self.authority_schedule

            # Get primary drift signal (active_pitch_crossing_signed_error_m)
            signed_error = float(sagittal_position_error_m)
            abs_error = abs(signed_error)
            pitch_rad = float(pitch_x_rad)
            roll_rad = float(roll_y_rad)

            # Get hip_yaw if available
            hip_yaw_abs = 0.0
            if hasattr(self, '_hip_yaw_for_ezc'):
                hip_yaw_abs = abs(self._hip_yaw_for_ezc)

            # EZC Safety gate
            if abs_error > sch.ezc_disable_if_abs_error_gt_m:
                ezc_safety_gate_pass = False
                ezc_block_reason = "error_too_large"
            elif abs(pitch_rad) > sch.ezc_disable_if_pitch_gt_deg * (3.14159 / 180.0):
                ezc_safety_gate_pass = False
                ezc_block_reason = "pitch_unsafe"
            elif abs(roll_rad) > sch.ezc_disable_if_roll_gt_deg * (3.14159 / 180.0):
                ezc_safety_gate_pass = False
                ezc_block_reason = "roll_unsafe"
            elif hip_yaw_abs > sch.ezc_disable_if_hip_yaw_gt_rad:
                ezc_safety_gate_pass = False
                ezc_block_reason = "hip_yaw_unsafe"

            # State machine transitions
            if self._ezc_state == "CENTER_IDLE":
                # Check entry conditions
                if ezc_safety_gate_pass:
                    if signed_error > sch.ezc_enter_m:
                        self._ezc_state = "RECENTER_FROM_POSITIVE"
                        self._ezc_direction = -1  # negative correction
                        self._ezc_hold_steps = 0
                        self._ezc_zero_dwell_steps = 0
                        self._ezc_enter_event += 1
                        self._ezc_episode_id += 1
                        self._ezc_episode_start_error = signed_error
                        self._ezc_episode_min_error = signed_error
                        self._ezc_crossed_zero = False
                        self._ezc_tau = 0.0
                        self._ezc_target_tau = 0.0
                        self._ezc_exit_reason = "none"
                    elif signed_error < -sch.ezc_enter_m:
                        self._ezc_state = "RECENTER_FROM_NEGATIVE"
                        self._ezc_direction = +1  # positive correction
                        self._ezc_hold_steps = 0
                        self._ezc_zero_dwell_steps = 0
                        self._ezc_enter_event += 1
                        self._ezc_episode_id += 1
                        self._ezc_episode_start_error = signed_error
                        self._ezc_episode_max_error = signed_error
                        self._ezc_crossed_zero = False
                        self._ezc_tau = 0.0
                        self._ezc_target_tau = 0.0
                        self._ezc_exit_reason = "none"

            elif self._ezc_state == "RECENTER_FROM_POSITIVE":
                # Apply negative correction
                self._ezc_hold_steps += 1
                self._ezc_episode_min_error = min(self._ezc_episode_min_error, signed_error)

                # Check if crossed zero
                if signed_error < 0:
                    self._ezc_crossed_zero = True

                # Target torque: base + error proportional
                target_tau = sch.ezc_base_tau_nm + sch.ezc_error_gain_nm_per_m * abs_error
                target_tau = min(target_tau, sch.ezc_max_tau_nm)
                target_tau = max(target_tau, sch.ezc_base_tau_nm)
                self._ezc_target_tau = target_tau

                # Exit conditions - EXIT AT ZERO, not opposite side
                if signed_error <= 0:
                    # CROSSED ZERO - check if anti-rebound is enabled (V2)
                    if sch.ezc_antirebound_enabled:
                        # Enter ANTIREBOUND_DECAY instead of immediate decay
                        self._ezc_state = "ANTIREBOUND_DECAY"
                        self._ezc_antirebound_steps = 0
                        # Start with ezc_antirebound_initial_ratio of current tau
                        self._ezc_antirebound_tau_start = abs(self._ezc_tau) * sch.ezc_antirebound_initial_ratio
                        self._ezc_target_tau = self._ezc_antirebound_tau_start
                        self._ezc_zero_cross_exit_event += 1
                        self._ezc_exit_reason = "anti_rebound"
                        self._ezc_hold_steps = 0
                    else:
                        # Original behavior: exit to ZERO_CROSSED_DECAY
                        self._ezc_state = "ZERO_CROSSED_DECAY"
                        self._ezc_zero_cross_exit_event += 1
                        self._ezc_exit_reason = "zero_cross"
                        self._ezc_hold_steps = 0
                        self._ezc_target_tau = 0.0  # Start decaying toward zero
                elif self._ezc_hold_steps >= sch.ezc_max_hold_steps:
                    # Max hold reached
                    self._ezc_state = "SAFETY_DECAY"
                    self._ezc_safety_exit_event += 1
                    self._ezc_exit_reason = "max_hold"
                    self._ezc_hold_steps = 0
                elif not ezc_safety_gate_pass:
                    # Safety gate failed
                    self._ezc_state = "SAFETY_DECAY"
                    self._ezc_safety_exit_event += 1
                    self._ezc_exit_reason = "safety_gate"
                    self._ezc_hold_steps = 0

            elif self._ezc_state == "RECENTER_FROM_NEGATIVE":
                # Apply positive correction
                self._ezc_hold_steps += 1
                self._ezc_episode_max_error = max(self._ezc_episode_max_error, signed_error)

                # Check if crossed zero
                if signed_error > 0:
                    self._ezc_crossed_zero = True

                # Target torque
                target_tau = sch.ezc_base_tau_nm + sch.ezc_error_gain_nm_per_m * abs_error
                target_tau = min(target_tau, sch.ezc_max_tau_nm)
                target_tau = max(target_tau, sch.ezc_base_tau_nm)
                self._ezc_target_tau = target_tau

                # Exit conditions - EXIT AT ZERO, not opposite side
                if signed_error >= 0:
                    # CROSSED ZERO - check if anti-rebound is enabled (V2)
                    if sch.ezc_antirebound_enabled:
                        # Enter ANTIREBOUND_DECAY instead of immediate decay
                        self._ezc_state = "ANTIREBOUND_DECAY"
                        self._ezc_antirebound_steps = 0
                        # Start with ezc_antirebound_initial_ratio of current tau
                        self._ezc_antirebound_tau_start = abs(self._ezc_tau) * sch.ezc_antirebound_initial_ratio
                        self._ezc_target_tau = self._ezc_antirebound_tau_start
                        self._ezc_zero_cross_exit_event += 1
                        self._ezc_exit_reason = "anti_rebound"
                        self._ezc_hold_steps = 0
                    else:
                        # Original behavior: exit to ZERO_CROSSED_DECAY
                        self._ezc_state = "ZERO_CROSSED_DECAY"
                        self._ezc_zero_cross_exit_event += 1
                        self._ezc_exit_reason = "zero_cross"
                        self._ezc_hold_steps = 0
                        self._ezc_target_tau = 0.0  # Start decaying toward zero
                elif self._ezc_hold_steps >= sch.ezc_max_hold_steps:
                    self._ezc_state = "SAFETY_DECAY"
                    self._ezc_safety_exit_event += 1
                    self._ezc_exit_reason = "max_hold"
                    self._ezc_hold_steps = 0
                elif not ezc_safety_gate_pass:
                    self._ezc_state = "SAFETY_DECAY"
                    self._ezc_safety_exit_event += 1
                    self._ezc_exit_reason = "safety_gate"
                    self._ezc_hold_steps = 0

            elif self._ezc_state == "ZERO_CROSSED_DECAY":
                # Decay correction toward zero after zero crossing
                self._ezc_hold_steps += 1
                self._ezc_zero_dwell_steps += 1
                self._ezc_target_tau = 0.0  # Target is zero

                # Decay tau toward zero
                if abs(self._ezc_tau) > 1e-9:
                    decay = sch.ezc_decay_nm_per_step
                    if self._ezc_tau > 0:
                        self._ezc_tau = max(0, self._ezc_tau - decay)
                    else:
                        self._ezc_tau = min(0, self._ezc_tau + decay)
                else:
                    self._ezc_tau = 0.0
                    self._ezc_direction = 0
                    self._ezc_state = "CENTER_IDLE"

            elif self._ezc_state == "ANTIREBOUND_DECAY":
                # V2 Anti-rebound hold: decay slowly after zero crossing
                # This prevents immediate rebound while tau_position recovers
                self._ezc_hold_steps += 1
                self._ezc_antirebound_steps += 1

                # Check if anti-rebound is complete
                decay_steps = sch.ezc_antirebound_decay_steps
                if self._ezc_antirebound_steps >= decay_steps:
                    # Decay complete - exit to idle
                    self._ezc_tau = 0.0
                    self._ezc_direction = 0
                    self._ezc_state = "CENTER_IDLE"
                    self._ezc_exit_reason = "antirebound_complete"
                else:
                    # Linear decay from ezc_antirebound_initial_ratio * tau to 0
                    progress = self._ezc_antirebound_steps / decay_steps
                    target_tau = self._ezc_antirebound_tau_start * (1.0 - progress)
                    self._ezc_target_tau = target_tau

                    # Smoothly decay tau (use smaller decay rate for smoother transition)
                    if abs(self._ezc_tau) > 1e-9:
                        decay = sch.ezc_decay_nm_per_step * 0.5  # Slower decay during anti-rebound
                        if self._ezc_tau > 0:
                            self._ezc_tau = max(0, self._ezc_tau - decay)
                        else:
                            self._ezc_tau = min(0, self._ezc_tau + decay)
                    else:
                        self._ezc_tau = 0.0
                        self._ezc_direction = 0
                        self._ezc_state = "CENTER_IDLE"

                # Check for re-entry: if error grows back to enter threshold while in anti-rebound
                # Allow re-entry to handle large rebound quickly
                if signed_error > sch.ezc_enter_m:
                    # Re-enter recenter state
                    if self._ezc_direction < 0:  # Was correcting positive drift
                        self._ezc_state = "RECENTER_FROM_POSITIVE"
                        self._ezc_hold_steps = 0
                        self._ezc_enter_event += 1
                        self._ezc_episode_id += 1
                        self._ezc_episode_start_error = signed_error
                        self._ezc_episode_min_error = signed_error
                        self._ezc_tau = 0.0
                        self._ezc_target_tau = 0.0
                        self._ezc_exit_reason = "none"
                    # Note: No need to check RECENTER_FROM_NEGATIVE since we only correct positive drift

            elif self._ezc_state == "SAFETY_DECAY":
                # Decay correction toward zero if safety gate failed
                if abs(self._ezc_tau) > 1e-9:
                    decay = sch.ezc_decay_nm_per_step
                    if self._ezc_tau > 0:
                        self._ezc_tau = max(0, self._ezc_tau - decay)
                    else:
                        self._ezc_tau = min(0, self._ezc_tau + decay)
                else:
                    self._ezc_tau = 0.0
                    self._ezc_direction = 0
                    self._ezc_state = "CENTER_IDLE"

            # Rate limit toward target (except in decay states)
            if self._ezc_state not in ("ZERO_CROSSED_DECAY", "SAFETY_DECAY", "CENTER_IDLE") and abs(self._ezc_target_tau) > 1e-9:
                is_increase = abs(self._ezc_target_tau) > abs(self._ezc_tau)
                rate = sch.ezc_rate_nm_per_step if is_increase else sch.ezc_decay_nm_per_step
                delta = self._ezc_target_tau - self._ezc_tau
                if abs(delta) > rate:
                    delta = rate if delta > 0 else -rate
                self._ezc_tau = self._ezc_tau + delta

            # Apply EZC correction to tau_position
            if self._ezc_state not in ("CENTER_IDLE", "SAFETY_DECAY"):
                ezc_active = True
                # Direction sign is in _ezc_direction (-1 or +1), tau is positive magnitude
                ezc_tau_signed = self._ezc_direction * abs(self._ezc_tau)
                tau_position = float(jnp.clip(
                    tau_position + ezc_tau_signed,
                    -effective_max_position_tau,
                    effective_max_position_tau
                ))

            # Telemetry output
            ezc_state = self._ezc_state
            ezc_state_id_map = {
                "CENTER_IDLE": 0,
                "RECENTER_FROM_POSITIVE": 1,
                "RECENTER_FROM_NEGATIVE": 2,
                "ZERO_CROSSED_DECAY": 3,
                "SAFETY_DECAY": 4,
                "ANTIREBOUND_DECAY": 5  # V2 anti-rebound state
            }
            ezc_state_id = ezc_state_id_map.get(ezc_state, 0)
            ezc_direction = self._ezc_direction
            ezc_tau_nm = float(self._ezc_tau) * self._ezc_direction  # signed tau
            ezc_target_tau_nm = self._ezc_target_tau * self._ezc_direction  # signed
            ezc_crossed_zero = self._ezc_crossed_zero
            ezc_safety_gate_pass = ezc_safety_gate_pass
            ezc_block_reason = ezc_block_reason if not ezc_safety_gate_pass else "none"

            # Direction correctness
            if signed_error > 1e-9:
                ezc_expected_direction_correct = (ezc_tau_nm <= 0)
            elif signed_error < -1e-9:
                ezc_expected_direction_correct = (ezc_tau_nm >= 0)
            else:
                ezc_expected_direction_correct = True

        # =====================================================================
        # APCR1nD: Direct Support Drift Trigger (BEFORE APCR1n block)
        # Activates based on direct drift conditions WITHOUT requiring APC state
        # This is the KEY DIFFERENCE from APCR1n which depends on _apc_drift_priority_active
        # =====================================================================
        # APCR1nD telemetry variables
        apcr1nd_direct_recenter_priority_active = False
        apcr1nd_direct_recenter_eligible = False
        apcr1nd_direct_recenter_block_reason = "none"
        apcr1nd_moving_away = False
        apcr1nd_abs_error = 0.0
        apcr1nd_error_rate = 0.0

        if self.authority_schedule.recenter_priority_direct_enabled:
            # Track steps for startup guard
            current_step = self._apcr1nd_step_counter
            self._apcr1nd_step_counter += 1

            # Startup guard
            startup_guard_steps = self.authority_schedule.recenter_priority_startup_guard_steps
            if current_step < startup_guard_steps:
                apcr1nd_direct_recenter_block_reason = "startup_guard"
            else:
                # Compute drift conditions
                signed_error = float(sagittal_position_error_m)
                abs_error = abs(signed_error)
                e_dot = signed_error - self._apcr1nd_prev_error
                self._apcr1nd_prev_error = signed_error
                moving_away = signed_error * e_dot > 0.0
                converging = not moving_away and abs(e_dot) > 1e-6

                # Telemetry
                apcr1nd_moving_away = moving_away
                apcr1nd_abs_error = abs_error
                apcr1nd_error_rate = e_dot

                # Safety gates
                abs_pitch = abs(float(pitch_x_rad))
                abs_roll = abs(float(roll_y_rad)) if roll_y_rad is not None else 0.0
                com_z_safe = float(com_z_m) >= self.authority_schedule.recenter_priority_safe_min_com_z
                roll_safe = abs_roll <= self.authority_schedule.recenter_priority_safe_roll_rad
                pitch_safe = abs_pitch <= self.authority_schedule.recenter_priority_safe_pitch_rad

                # APCR1nD Tuned Variants Logic
                if self.authority_schedule.apcr1nd_tuned_enabled:
                    # Use tuned thresholds
                    soft_enter_m = self.authority_schedule.apcr1nd_soft_enter_m
                    direct_enter_m = self.authority_schedule.apcr1nd_direct_enter_m
                    desired_band_m = self.authority_schedule.apcr1nd_desired_band_m
                    release_inner_m = self.authority_schedule.apcr1nd_release_inner_m
                    hold_outside_band = self.authority_schedule.apcr1nd_hold_outside_band
                    converging_release_steps = self.authority_schedule.apcr1nd_converging_release_steps

                    # Update converging steps counter
                    if converging:
                        self._apcr1nd_tuned_converging_steps += 1
                    else:
                        self._apcr1nd_tuned_converging_steps = 0

                    # Safety gate check
                    safety_pass = contact_valid and com_z_safe and roll_safe and pitch_safe

                    if not safety_pass:
                        if not contact_valid:
                            apcr1nd_direct_recenter_block_reason = "contact_invalid"
                        elif not com_z_safe:
                            apcr1nd_direct_recenter_block_reason = "height_unsafe"
                        elif not roll_safe:
                            apcr1nd_direct_recenter_block_reason = "roll_unsafe"
                        else:
                            apcr1nd_direct_recenter_block_reason = "pitch_unsafe"
                        apcr1nd_direct_recenter_eligible = False
                        apcr1nd_direct_recenter_priority_active = False
                        self._apcr1nd_tuned_recenter_held = False
                    else:
                        # Activation logic
                        # Activate if:
                        # 1. abs_error >= direct_enter_m AND moving_away (initial entry)
                        # 2. abs_error >= desired_band_m regardless of moving_away (emergency entry)
                        # 3. Already active AND abs_error > release_inner_m (hold)

                        prev_active = self._apcr1nd_tuned_recenter_held

                        # Entry conditions
                        soft_entry = abs_error >= soft_enter_m and abs_error < direct_enter_m and moving_away
                        direct_entry = abs_error >= direct_enter_m and moving_away
                        emergency_entry = abs_error >= desired_band_m

                        # Hold condition
                        hold_condition = prev_active and abs_error > release_inner_m
                        hold_outside_band_condition = hold_outside_band and abs_error > desired_band_m

                        # Release condition
                        release_by_inner_band = abs_error <= release_inner_m
                        release_by_converging = (
                            converging and
                            self._apcr1nd_tuned_converging_steps >= converging_release_steps and
                            abs_error <= desired_band_m * 0.75  # Allow release if well below desired band
                        )

                        # Decision
                        if release_by_inner_band or release_by_converging:
                            apcr1nd_direct_recenter_priority_active = False
                            apcr1nd_direct_recenter_eligible = True
                            apcr1nd_direct_recenter_block_reason = "released_inner_band"
                            self._apcr1nd_tuned_recenter_held = False
                        elif emergency_entry or hold_outside_band_condition:
                            apcr1nd_direct_recenter_priority_active = True
                            apcr1nd_direct_recenter_eligible = True
                            apcr1nd_direct_recenter_block_reason = "none"
                            self._apcr1nd_tuned_recenter_held = True
                        elif direct_entry or soft_entry:
                            apcr1nd_direct_recenter_priority_active = True
                            apcr1nd_direct_recenter_eligible = True
                            apcr1nd_direct_recenter_block_reason = "none"
                            self._apcr1nd_tuned_recenter_held = True
                        elif hold_condition:
                            apcr1nd_direct_recenter_priority_active = True
                            apcr1nd_direct_recenter_eligible = True
                            apcr1nd_direct_recenter_block_reason = "none"
                            self._apcr1nd_tuned_recenter_held = True
                        else:
                            apcr1nd_direct_recenter_priority_active = False
                            apcr1nd_direct_recenter_eligible = True
                            apcr1nd_direct_recenter_block_reason = "below_threshold"
                            self._apcr1nd_tuned_recenter_held = False

                else:
                    # Original APCR1nD baseline logic
                    enter_thresh = self.authority_schedule.recenter_priority_direct_enter_m
                    exit_thresh = self.authority_schedule.recenter_priority_direct_exit_m

                    if not contact_valid:
                        apcr1nd_direct_recenter_block_reason = "contact_invalid"
                    elif not com_z_safe:
                        apcr1nd_direct_recenter_block_reason = "height_unsafe"
                    elif not roll_safe:
                        apcr1nd_direct_recenter_block_reason = "roll_unsafe"
                    elif not pitch_safe:
                        apcr1nd_direct_recenter_block_reason = "pitch_unsafe"
                    elif abs_error < exit_thresh:
                        # Inside exit band - eligible but not active
                        apcr1nd_direct_recenter_eligible = True
                        apcr1nd_direct_recenter_block_reason = "within_exit_band"
                    elif abs_error > enter_thresh and moving_away:
                        # Large error AND moving away - ACTIVE
                        apcr1nd_direct_recenter_eligible = True
                        apcr1nd_direct_recenter_priority_active = True
                    elif abs_error > enter_thresh:
                        # Large error but moving toward zero - eligible, not priority
                        apcr1nd_direct_recenter_eligible = True
                        apcr1nd_direct_recenter_block_reason = "eligible_but_converging"
                    else:
                        apcr1nd_direct_recenter_block_reason = "below_enter_threshold"

        # =====================================================================
        # APCR1n: Recenter Priority Torque Boost
        # Based on APCR1h with targeted fixes:
        # 1. Wheel damping override during RECENTER when it fights drift recovery
        # 2. Position cap boost during safe RECENTER
        # =====================================================================
        # APCR1n telemetry variables
        apcr1n_recenter_priority_active = False
        apcr1n_startup_guard_active = False
        apcr1n_wheel_damping_override_active = False
        self._apcr1nd_wd_override_active = False  # Phase 0: reset for this step
        apcr1n_wheel_damping_scale = 1.0
        apcr1n_wheel_damping_before = 0.0
        apcr1n_wheel_damping_after = 0.0
        apcr1n_wheel_damping_fights_drift = False
        apcr1n_position_cap_boost_active = False
        apcr1n_position_cap_current = float(effective_max_position_tau)
        apcr1n_tau_position_raw = float(tau_position_before_clip)
        apcr1n_tau_position_after_cap = float(tau_position)
        apcr1n_position_saturated = bool(tau_position_saturated)
        apcr1n_safety_gate_pass = False
        apcr1n_final_torque_direction_correct = True
        apcr1n_final_torque_fights_drift = False
        apcr1n_physical_drift_column_used = "active_pitch_crossing_signed_error_m"

        if self.authority_schedule.recenter_priority_enabled:
            # Track steps for startup guard
            if not hasattr(self, '_apcr1n_step_counter'):
                self._apcr1n_step_counter = 0
            current_step = self._apcr1n_step_counter
            self._apcr1n_step_counter += 1

            # Check startup guard
            startup_guard_steps = self.authority_schedule.recenter_priority_startup_guard_steps
            if current_step < startup_guard_steps:
                apcr1n_startup_guard_active = True
            else:
                # KEY DIFFERENCE: Use direct trigger when available
                # APCR1nD profile uses direct drift trigger instead of _apc_drift_priority_active
                if self.authority_schedule.recenter_priority_direct_enabled:
                    # APCR1nD: Use the direct support drift trigger
                    apcr1n_recenter_priority_active = apcr1nd_direct_recenter_priority_active
                else:
                    # Original APCR1n: depends on _apc_drift_priority_active
                    apcr1n_recenter_priority_active = self._apc_drift_priority_active

                if apcr1n_recenter_priority_active:
                    # Check safety gates for position cap boost
                    abs_pitch = abs(float(pitch_x_rad))
                    abs_roll = abs(float(roll_y_rad)) if roll_y_rad is not None else 0.0
                    com_z_safe = float(com_z_m) >= self.authority_schedule.recenter_priority_safe_min_com_z
                    roll_safe = abs_roll <= self.authority_schedule.recenter_priority_safe_roll_rad
                    pitch_safe_gate = abs_pitch <= self.authority_schedule.recenter_priority_safe_pitch_rad

                    apcr1n_safety_gate_pass = (
                        contact_valid and com_z_safe and roll_safe and pitch_safe_gate
                    )

                    # Determine physical drift sign (positive drift = forward)
                    signed_error = float(sagittal_position_error_m)  # positive = forward drift
                    drift_sign = 1.0 if signed_error > 0 else -1.0

                    # Compute wheel damping component (negative of wheel velocity)
                    # Positive wheel vel = forward rotation = backward drift correction
                    # tau_wheel_vel = -k_wheel * wheel_vel
                    # If drift is positive (forward) and wheel_vel is positive (forward),
                    # tau_wheel_vel is negative (backward) = fighting forward drift = GOOD
                    # If drift is positive (forward) and wheel_vel is negative (backward),
                    # tau_wheel_vel is positive (forward) = fighting backward drift = BAD
                    wheel_vel_mean = 0.5 * (wheel_vel_left_rad_s + wheel_vel_right_rad_s)
                    wheel_vel_sign = 1.0 if wheel_vel_mean > 0 else -1.0

                    # Wheel damping fights drift if:
                    # - drift_sign and wheel_vel_sign have SAME sign (both positive or both negative)
                    # This means wheel is accelerating drift rather than correcting it
                    apcr1n_wheel_damping_fights_drift = (
                        abs(drift_sign - wheel_vel_sign) < 0.5  # Same sign
                    )

                    # Wheel damping override
                    if self.authority_schedule.vd_wheel_damping_recenter_override_enabled:
                        # Determine damping scale
                        if self.authority_schedule.apcr1nd_tuned_enabled:
                            # Tuned variant: band-aware damping scaling
                            abs_error = abs(signed_error)
                            hard_band_m = self.authority_schedule.apcr1nd_hard_band_m
                            emergency_band_m = self.authority_schedule.apcr1nd_emergency_band_m
                            desired_band_m = self.authority_schedule.apcr1nd_desired_band_m
                            soft_enter_m = self.authority_schedule.apcr1nd_soft_enter_m

                            # Determine band state and damping scale
                            if abs_error >= emergency_band_m:
                                wheel_scale = self.authority_schedule.apcr1nd_damping_scale_emergency
                            elif abs_error >= hard_band_m:
                                wheel_scale = self.authority_schedule.apcr1nd_damping_scale_hard
                            elif abs_error >= desired_band_m:
                                wheel_scale = self.authority_schedule.apcr1nd_damping_scale_desired
                            elif abs_error >= soft_enter_m:
                                wheel_scale = self.authority_schedule.apcr1nd_damping_scale_soft
                            else:
                                wheel_scale = self.authority_schedule.apcr1nd_damping_scale_normal

                            # Preserve damping if it helps recovery
                            if self.authority_schedule.apcr1nd_preserve_damping_if_helps:
                                # Check if damping opposes drift
                                damping_opposes_drift = not apcr1n_wheel_damping_fights_drift
                                if damping_opposes_drift:
                                    wheel_scale = 1.0  # Keep full damping
                        else:
                            # Original APCR1n: use configured scale when damping fights drift
                            if apcr1n_wheel_damping_fights_drift:
                                wheel_scale = self.authority_schedule.vd_wheel_damping_recenter_scale
                            else:
                                wheel_scale = 1.0

                        # Apply override only if damping fights drift OR tuned variant is active
                        apply_override = (
                            (not self.authority_schedule.apcr1nd_tuned_enabled and apcr1n_wheel_damping_fights_drift) or
                            (self.authority_schedule.apcr1nd_tuned_enabled and wheel_scale < 1.0)
                        )

                        if apply_override:
                            # Compute original wheel damping torques
                            tau_wheel_vel_left_orig = tau_wheel_vel_left
                            tau_wheel_vel_right_orig = tau_wheel_vel_right

                            # Apply override
                            tau_wheel_vel_left = tau_wheel_vel_left * wheel_scale
                            tau_wheel_vel_right = tau_wheel_vel_right * wheel_scale

                            # Ensure minimum damping is preserved if configured
                            min_damping = self.authority_schedule.vd_wheel_damping_recenter_min_abs_nm
                            if abs(tau_wheel_vel_left) < min_damping:
                                tau_wheel_vel_left = min_damping * (1.0 if tau_wheel_vel_left >= 0 else -1.0)
                            if abs(tau_wheel_vel_right) < min_damping:
                                tau_wheel_vel_right = min_damping * (1.0 if tau_wheel_vel_right >= 0 else -1.0)

                            # Telemetry
                            apcr1n_wheel_damping_override_active = True
                            self._apcr1nd_wd_override_active = True  # Phase 0: JAX parity state
                            apcr1n_wheel_damping_scale = wheel_scale
                            apcr1n_wheel_damping_before = float(tau_wheel_vel_left_orig)
                            apcr1n_wheel_damping_after = float(tau_wheel_vel_left)

                    # =====================================================================
                    # T6F Sign Fix: Enhanced Damping Override
                    # Disable damping completely (not scaled) when it fights position torque during arch_fix
                    # =====================================================================
                    if (self.authority_schedule.sign_fix_enabled and
                        self.authority_schedule.sign_fix_disable_fighting_damping_during_arch_fix and
                        arch_fix_active):

                        sign_fix_active = True

                        # Check if damping fights position torque
                        tau_damping_mean = 0.5 * (tau_wheel_vel_left + tau_wheel_vel_right)
                        sign_position = 1.0 if tau_position > 0 else -1.0 if tau_position < 0 else 0.0
                        sign_damping = 1.0 if tau_damping_mean > 0 else -1.0 if tau_damping_mean < 0 else 0.0

                        damping_fights_position = (sign_position * sign_damping < 0)

                        if damping_fights_position:
                            # Damping fights position - disable completely
                            sign_fix_damping_original_nm = float(tau_damping_mean)
                            tau_wheel_vel_left = 0.0
                            tau_wheel_vel_right = 0.0
                            sign_fix_damping_after_nm = 0.0
                            sign_fix_damping_disabled = True
                            sign_fix_damping_fought = True
                        else:
                            # Damping helps or is neutral - preserve it
                            sign_fix_damping_original_nm = float(tau_damping_mean)
                            sign_fix_damping_after_nm = float(tau_damping_mean)
                            sign_fix_damping_helped = True

                    # =====================================================================
                    # T6H Soft Blend Arch Fix: Soft Damping Blending
                    # Reduces damping authority by 50% (not 100%) when it fights position
                    # Preserves partial energy dissipation while reducing fighting
                    # =====================================================================
                    t6h_soft_damping_blend_active = False
                    t6h_damping_blend_factor = 1.0
                    t6h_wheel_velocity_safety_active = False

                    if self.authority_schedule.t6h_enabled and arch_fix_active:
                        # Check wheel velocity safety override
                        mean_wheel_vel = 0.5 * (wheel_vel_left_rad_s + wheel_vel_right_rad_s)
                        wheel_velocity_safety_threshold = self.authority_schedule.t6h_wheel_velocity_safety_threshold_rad_s

                        if abs(float(mean_wheel_vel)) > wheel_velocity_safety_threshold:
                            # Wheel velocity too high - restore full damping
                            t6h_damping_blend_factor = 1.0
                            t6h_wheel_velocity_safety_active = True
                        else:
                            # Check if damping opposes position correction
                            tau_damping_mean = 0.5 * (tau_wheel_vel_left + tau_wheel_vel_right)
                            sign_position = 1.0 if tau_position > 0 else -1.0 if tau_position < 0 else 0.0
                            sign_damping = 1.0 if tau_damping_mean > 0 else -1.0 if tau_damping_mean < 0 else 0.0
                            damping_opposes_position = (sign_position * sign_damping < 0)

                            if damping_opposes_position:
                                # Damping opposes position - apply soft blend
                                t6h_damping_blend_factor = self.authority_schedule.t6h_soft_damping_blend_factor
                                t6h_soft_damping_blend_active = True
                            else:
                                # Damping helps or neutral - preserve full damping
                                t6h_damping_blend_factor = 1.0

                        # Apply blend to damping torques
                        tau_wheel_vel_left = tau_wheel_vel_left * t6h_damping_blend_factor
                        tau_wheel_vel_right = tau_wheel_vel_right * t6h_damping_blend_factor

                    # Position cap boost
                    if (self.authority_schedule.position_cap_recenter_boost_enabled and
                        apcr1n_safety_gate_pass):
                        # Determine boosted cap
                        if self.authority_schedule.apcr1nd_tuned_enabled:
                            # Tuned variant: band-aware position cap scaling
                            abs_error = abs(signed_error)
                            hard_band_m = self.authority_schedule.apcr1nd_hard_band_m
                            emergency_band_m = self.authority_schedule.apcr1nd_emergency_band_m
                            desired_band_m = self.authority_schedule.apcr1nd_desired_band_m
                            soft_enter_m = self.authority_schedule.apcr1nd_soft_enter_m

                            # Determine band state and position cap
                            if abs_error >= emergency_band_m:
                                boosted_cap = self.authority_schedule.apcr1nd_position_cap_emergency_nm
                            elif abs_error >= hard_band_m:
                                boosted_cap = self.authority_schedule.apcr1nd_position_cap_hard_nm
                            elif abs_error >= desired_band_m:
                                boosted_cap = self.authority_schedule.apcr1nd_position_cap_desired_nm
                            elif abs_error >= soft_enter_m:
                                boosted_cap = self.authority_schedule.apcr1nd_position_cap_soft_nm
                            else:
                                boosted_cap = self.authority_schedule.apcr1nd_position_cap_normal_nm
                        else:
                            # Original APCR1n: use configured boosted cap
                            boosted_cap = self.authority_schedule.position_cap_recenter_nm

                        tau_position_before_boost = tau_position

                        # Apply boosted cap to tau_position
                        tau_position = float(jnp.clip(tau_position, -boosted_cap, boosted_cap))

                        # Update effective_max_position_tau for telemetry
                        apcr1n_position_cap_boost_active = True
                        apcr1n_position_cap_current = boosted_cap
                        apcr1n_tau_position_after_cap = float(tau_position)
                        # Check if position was saturated by the new cap
                        apcr1n_position_saturated = abs(tau_position_before_boost) > boosted_cap

                        # Update saturation flag
                        tau_position_saturated = apcr1n_position_saturated

        # =====================================================================
        # Phase-aware recenter (F1_strategy - signed drift fix)
        # Decouples recentering from tau_position to avoid hip yaw coupling
        # =====================================================================
        # Phase detection: determine if it's safe to apply recenter torque
        pitch_abs = abs(float(pitch_x_rad))
        pitch_safe = pitch_abs < self.authority_schedule.recenter_pitch_safe_threshold_rad
        # Also safe if pitch is recovering (pitch and pitch_rate have opposite signs)
        pitch_recovering = float(pitch_x_rad) * float(pitch_rate_x_rad_s) < 0
        pitch_safe = pitch_safe or pitch_recovering
        # Danger threshold: block recenter if pitch is too large
        pitch_danger = pitch_abs > self.authority_schedule.recenter_pitch_danger_threshold_rad

        # Get hip_yaw_abs_max from diagnostics (set to 0 if not available yet)
        hip_yaw_abs_max = 0.0  # Will be set from telemetry if available

        # Height safety check
        com_z_safe = (
            self.authority_schedule.recenter_min_com_z_m <= float(com_z_m) <= self.authority_schedule.recenter_max_com_z_m
        )

        # Signed support error (hip_yaw_comp_support_error_m)
        # positive = forward drift
        signed_error = float(sagittal_position_error_m)  # This is the yaw-aware compensated error

        # Deadband check
        within_deadband = abs(signed_error) <= self.authority_schedule.recenter_deadband_m

        # Overall gate: recenter is active only if all conditions are met
        phase_recenter_enabled = self.authority_schedule.enable_phase_aware_recenter
        recenter_gate_safe = pitch_safe and not pitch_danger and com_z_safe and contact_valid
        recenter_deadband_active = within_deadband

        # Compute gate reason for telemetry
        if not phase_recenter_enabled:
            gate_reason = "disabled"
        elif pitch_danger:
            gate_reason = "pitch_danger"
        elif not pitch_safe:
            gate_reason = "pitch_unsafe"
        elif not com_z_safe:
            gate_reason = "height_unsafe"
        elif not contact_valid:
            gate_reason = "contact_invalid"
        elif within_deadband:
            gate_reason = "deadband"
        else:
            gate_reason = "active"

        # Compute raw recenter torque
        if phase_recenter_enabled and recenter_gate_safe and not recenter_deadband_active:
            # recenter_tau = -k_recenter * signed_error
            # If signed_error > 0 (forward drift), recenter_tau < 0 (push backward)
            raw_recenter_tau = -self.authority_schedule.k_recenter * signed_error
            # Clip to max recenter torque
            raw_recenter_tau = float(jnp.clip(
                raw_recenter_tau,
                -self.authority_schedule.max_recenter_tau,
                self.authority_schedule.max_recenter_tau
            ))
        else:
            raw_recenter_tau = 0.0

        # Smooth the recenter torque to avoid discontinuous jumps
        alpha = self.authority_schedule.recenter_smooth_alpha
        smoothed_recenter_tau = alpha * raw_recenter_tau + (1.0 - alpha) * self._prev_recenter_tau

        # Rate limit: prevent too fast changes
        max_rate = self.authority_schedule.recenter_max_rate_per_step
        if smoothed_recenter_tau > self._prev_recenter_tau + max_rate:
            final_recenter_tau = self._prev_recenter_tau + max_rate
        elif smoothed_recenter_tau < self._prev_recenter_tau - max_rate:
            final_recenter_tau = self._prev_recenter_tau - max_rate
        else:
            final_recenter_tau = smoothed_recenter_tau

        # Store for next step
        self._prev_recenter_tau = final_recenter_tau

        # Apply recenter torque to wheel torque
        # recenter_tau is added to tau_common, which then goes to both wheels
        recenter_tau_clipped = float(jnp.clip(
            final_recenter_tau,
            -self.authority_schedule.max_recenter_tau,
            self.authority_schedule.max_recenter_tau
        ))

        # =====================================================================
        # Hysteresis recenter (F2_strategy - stateful recenter for stronger bias correction)
        # Holds recenter direction until error returns to exit target, preventing early reversal
        # =====================================================================
        hysteresis_recenter_enabled = self.authority_schedule.enable_hysteresis_recenter

        # Hysteresis safety checks
        hyst_pitch_abs = abs(float(pitch_x_rad))
        hyst_pitch_safe = hyst_pitch_abs < self.authority_schedule.hysteresis_pitch_safe_threshold_rad
        hyst_pitch_recovering = float(pitch_x_rad) * float(pitch_rate_x_rad_s) < 0
        hyst_pitch_safe = hyst_pitch_safe or hyst_pitch_recovering
        hyst_pitch_danger = hyst_pitch_abs > self.authority_schedule.hysteresis_pitch_danger_threshold_rad
        hyst_com_z_safe = (
            self.authority_schedule.hysteresis_min_com_z_m <= float(com_z_m) <= self.authority_schedule.hysteresis_max_com_z_m
        )
        hyst_hip_yaw_safe = hip_yaw_abs_max < self.authority_schedule.hysteresis_hip_yaw_safe_threshold_rad
        hyst_gate_safe = hyst_pitch_safe and not hyst_pitch_danger and hyst_com_z_safe and contact_valid and hyst_hip_yaw_safe

        # Hysteresis state machine
        # States: NEUTRAL, RECENTER_FROM_POSITIVE, RECENTER_FROM_NEGATIVE
        hysteresis_state = self._hysteresis_state
        hysteresis_safety_override = False

        # Outer enter thresholds
        outer_enter = self.authority_schedule.hysteresis_outer_enter_m
        exit_target = self.authority_schedule.hysteresis_exit_target_m
        opposite_overshoot = self.authority_schedule.hysteresis_opposite_overshoot_m
        deadband = self.authority_schedule.hysteresis_deadband_m

        # Exit targets with slight overshoot
        exit_target_positive = exit_target - opposite_overshoot  # For RECENTER_FROM_POSITIVE: exit when error <= this (negative)
        exit_target_negative = exit_target + opposite_overshoot  # For RECENTER_FROM_NEGATIVE: exit when error >= this (positive)

        # Safety override: exit any recenter state to NEUTRAL
        if hysteresis_state != "NEUTRAL" and not hyst_gate_safe:
            hysteresis_state = "NEUTRAL"
            hysteresis_safety_override = True
            self._hysteresis_safety_override_count += 1

        # State transitions
        if hysteresis_state == "NEUTRAL":
            # Check if we should enter a recenter state
            if hysteresis_recenter_enabled and hyst_gate_safe and not within_deadband:
                if signed_error > outer_enter:
                    hysteresis_state = "RECENTER_FROM_POSITIVE"
                    self._hysteresis_state_entry_count += 1
                elif signed_error < -outer_enter:
                    hysteresis_state = "RECENTER_FROM_NEGATIVE"
                    self._hysteresis_state_entry_count += 1

        elif hysteresis_state == "RECENTER_FROM_POSITIVE":
            # Keep applying negative torque until error returns to exit target
            if signed_error <= exit_target_positive:
                hysteresis_state = "NEUTRAL"
                self._hysteresis_state_exit_count += 1

        elif hysteresis_state == "RECENTER_FROM_NEGATIVE":
            # Keep applying positive torque until error returns to exit target
            if signed_error >= exit_target_negative:
                hysteresis_state = "NEUTRAL"
                self._hysteresis_state_exit_count += 1

        # Store the new state
        self._hysteresis_state = hysteresis_state

        # Compute hysteresis recenter torque based on state
        hyst_raw_tau = 0.0
        if hysteresis_state == "RECENTER_FROM_POSITIVE":
            # Apply negative torque to push back from positive drift
            # Target error is negative (opposite direction of drift)
            target_error = exit_target_positive
            hyst_raw_tau = -self.authority_schedule.hysteresis_k_recenter * (signed_error - target_error)
        elif hysteresis_state == "RECENTER_FROM_NEGATIVE":
            # Apply positive torque to push back from negative drift
            # Target error is positive (opposite direction of drift)
            target_error = exit_target_negative
            hyst_raw_tau = -self.authority_schedule.hysteresis_k_recenter * (signed_error - target_error)

        # Clip hysteresis recenter torque
        hyst_raw_tau = float(jnp.clip(
            hyst_raw_tau,
            -self.authority_schedule.hysteresis_max_recenter_tau,
            self.authority_schedule.hysteresis_max_recenter_tau
        ))

        # Smooth hysteresis recenter torque
        hyst_alpha = self.authority_schedule.hysteresis_smooth_alpha
        hyst_smoothed_tau = hyst_alpha * hyst_raw_tau + (1.0 - hyst_alpha) * self._hysteresis_prev_tau

        # Rate limit hysteresis recenter
        hyst_max_rate = self.authority_schedule.hysteresis_max_rate_per_step
        if hyst_smoothed_tau > self._hysteresis_prev_tau + hyst_max_rate:
            hyst_final_tau = self._hysteresis_prev_tau + hyst_max_rate
        elif hyst_smoothed_tau < self._hysteresis_prev_tau - hyst_max_rate:
            hyst_final_tau = self._hysteresis_prev_tau - hyst_max_rate
        else:
            hyst_final_tau = hyst_smoothed_tau

        # Store for next step
        self._hysteresis_prev_tau = hyst_final_tau

        # Clip final hysteresis torque
        hyst_tau_clipped = float(jnp.clip(
            hyst_final_tau,
            -self.authority_schedule.hysteresis_max_recenter_tau,
            self.authority_schedule.hysteresis_max_recenter_tau
        ))

        # Hysteresis active flag
        hysteresis_active = hysteresis_state != "NEUTRAL"

        # Hysteresis gate reason for telemetry
        if not hysteresis_recenter_enabled:
            hyst_gate_reason = "disabled"
        elif hysteresis_safety_override:
            hyst_gate_reason = "safety_override"
        elif hyst_pitch_danger:
            hyst_gate_reason = "pitch_danger"
        elif not hyst_pitch_safe:
            hyst_gate_reason = "pitch_unsafe"
        elif not hyst_com_z_safe:
            hyst_gate_reason = "height_unsafe"
        elif not contact_valid:
            hyst_gate_reason = "contact_invalid"
        elif not hyst_hip_yaw_safe:
            hyst_gate_reason = "hip_yaw_unsafe"
        elif within_deadband:
            hyst_gate_reason = "deadband"
        elif hysteresis_state == "NEUTRAL":
            hyst_gate_reason = "waiting_for_threshold"
        else:
            hyst_gate_reason = "active"

        # =====================================================================
        # Bias cancellation (G1_strategy - persistent bias cancellation)
        # Estimates persistent signed error bias and applies bounded opposite torque
        # Unlike F1/F2 which wait for natural drift, G1 estimates bias and cancels proactively
        # =====================================================================
        bias_cancel_enabled = self.authority_schedule.enable_bias_cancel

        # Bias cancellation safety checks
        bias_com_z_safe = (
            self.authority_schedule.bias_cancel_min_com_z_m <= float(com_z_m) <= self.authority_schedule.bias_cancel_max_com_z_m
        )
        bias_roll_safe = abs(float(roll_y_rad)) < self.authority_schedule.bias_cancel_roll_threshold_rad
        bias_contact_safe = contact_valid if self.authority_schedule.bias_cancel_contact_gate else True
        bias_height_safe = bias_com_z_safe if self.authority_schedule.bias_cancel_height_gate else True
        bias_roll_gate_safe = bias_roll_safe if self.authority_schedule.bias_cancel_roll_gate else True
        # NOT gated on pitch (key difference from F1/F2 - pitch reversal doesn't produce negative drift)
        bias_gate_safe = bias_contact_safe and bias_height_safe and bias_roll_gate_safe

        # Deadband check for bias estimation
        within_bias_deadband = abs(signed_error) <= self.authority_schedule.bias_cancel_deadband_m

        # Compute gate reason for telemetry
        if not bias_cancel_enabled:
            bias_gate_reason = "disabled"
        elif not bias_contact_safe:
            bias_gate_reason = "contact_invalid"
        elif not bias_height_safe:
            bias_gate_reason = "height_unsafe"
        elif not bias_roll_gate_safe:
            bias_gate_reason = "roll_unsafe"
        elif within_bias_deadband:
            bias_gate_reason = "deadband"
        else:
            bias_gate_reason = "active"

        # Update bias estimate using low-pass filter (leaky integration)
        if bias_cancel_enabled and bias_gate_safe:
            # Filter coefficient: higher alpha = faster response, more noise
            # Using low alpha (0.02) for slow, smooth bias estimation
            alpha = self.authority_schedule.bias_cancel_filter_alpha
            self._bias_cancel_estimate = (
                alpha * signed_error + (1.0 - alpha) * self._bias_cancel_estimate
            )
        else:
            # Decay estimate when not active (leaky integration)
            decay = 0.95  # Decay toward zero when not applying
            self._bias_cancel_estimate = decay * self._bias_cancel_estimate

        # Compute bias cancellation torque
        # bias_tau = -k * bias_estimate
        # If bias_estimate > 0 (positive persistent bias), bias_tau < 0 (push backward)
        if bias_cancel_enabled and bias_gate_safe and not within_bias_deadband:
            bias_raw_tau = -self.authority_schedule.bias_cancel_k * self._bias_cancel_estimate
        else:
            bias_raw_tau = 0.0

        # Clip bias cancellation torque
        bias_raw_tau = float(jnp.clip(
            bias_raw_tau,
            -self.authority_schedule.bias_cancel_max_tau,
            self.authority_schedule.bias_cancel_max_tau
        ))

        # Smooth bias cancellation torque
        bias_alpha = 0.05  # Slightly higher alpha for smoothing
        bias_smoothed_tau = bias_alpha * bias_raw_tau + (1.0 - bias_alpha) * self._bias_cancel_prev_tau

        # Rate limit bias cancellation
        bias_max_rate = 0.3  # Nm/step - slower than F1/F2
        if bias_smoothed_tau > self._bias_cancel_prev_tau + bias_max_rate:
            bias_final_tau = self._bias_cancel_prev_tau + bias_max_rate
        elif bias_smoothed_tau < self._bias_cancel_prev_tau - bias_max_rate:
            bias_final_tau = self._bias_cancel_prev_tau - bias_max_rate
        else:
            bias_final_tau = bias_smoothed_tau

        # Store for next step
        self._bias_cancel_prev_tau = bias_final_tau

        # Clip final bias cancellation torque
        bias_tau_clipped = float(jnp.clip(
            bias_final_tau,
            -self.authority_schedule.bias_cancel_max_tau,
            self.authority_schedule.bias_cancel_max_tau
        ))

        # Bias cancellation active flag
        bias_cancel_active = bias_cancel_enabled and bias_gate_safe and not within_bias_deadband

        # =====================================================================
        # Active Pitch Crossing (APC_strategy - explicit pitch-rate crossing controller)
        # Actively drives wheel torque to create controlled pitch-rate reversal
        # When robot has positive pitch AND positive signed drift, APC applies
        # wheel torque to reverse pitch_rate, allowing support to return toward 0.
        # =====================================================================
        apc_enabled = self.authority_schedule.enable_active_pitch_crossing

        # APC safety checks
        apc_com_z_safe = (
            self.authority_schedule.apc_min_com_z_m <= float(com_z_m) <= self.authority_schedule.apc_max_com_z_m
        )
        apc_roll_safe = abs(float(roll_y_rad)) < self.authority_schedule.apc_roll_threshold_rad
        apc_contact_safe = contact_valid if self.authority_schedule.apc_contact_gate else True
        apc_height_safe = apc_com_z_safe if self.authority_schedule.apc_height_gate else True

        # Pitch safety check
        apc_pitch_abs = abs(float(pitch_x_rad))
        apc_pitch_safe = apc_pitch_abs < self.authority_schedule.apc_pitch_safe_threshold_rad
        apc_pitch_recovering = float(pitch_x_rad) * float(pitch_rate_x_rad_s) < 0
        apc_pitch_safe = apc_pitch_safe or apc_pitch_recovering
        apc_pitch_danger = apc_pitch_abs > self.authority_schedule.apc_pitch_danger_threshold_rad

        # Gate safety
        apc_gate_safe = apc_contact_safe and apc_height_safe and apc_roll_safe and apc_pitch_safe and not apc_pitch_danger

        # =====================================================================
        # APCR Recovery Gate Mode (alternative gate for active recovery)
        # Separates hard safety from recovery activation
        # APCR can activate during moderate pitch error instead of blocking
        # =====================================================================
        apc_recovery_gate_mode = self.authority_schedule.active_pitch_crossing_recovery_gate_mode
        apc_hard_safety_gate = True  # Default: hard safety allows activation
        apc_recovery_gate = True  # Default: recovery gate allows activation

        if apc_recovery_gate_mode:
            # Hard safety gate: blocks only if truly unsafe (pitch exceeds hard stop, etc.)
            apcr_pitch_hard_stop = self.authority_schedule.apcr_pitch_hard_stop_rad
            apcr_roll_hard_stop = self.authority_schedule.apcr_roll_hard_stop_rad
            apcr_min_com_z = self.authority_schedule.apcr_min_com_z_m
            apcr_max_com_z = self.authority_schedule.apcr_max_com_z_m

            # Hard safety blocks if pitch exceeds hard emergency threshold
            apc_hard_safety_gate = (
                apc_pitch_abs <= apcr_pitch_hard_stop
                and abs(float(roll_y_rad)) <= apcr_roll_hard_stop
                and apcr_min_com_z <= float(com_z_m) <= apcr_max_com_z
                and apc_contact_safe
            )

            # Recovery gate allows activation when pitch and drift are in same direction
            # (This is the entry condition, not a block)
            # Exit conditions are handled separately in state transitions
            apc_recovery_gate = apc_hard_safety_gate  # Recovery gate requires hard safety

        # Override gate safety with recovery mode if enabled
        if apc_recovery_gate_mode:
            apc_gate_safe = apc_hard_safety_gate

        # =====================================================================
        # APCR1d Proportional Soft Band Mode
        # Uses symmetric proportional torque shaping instead of bang-bang state machine
        # Key differences:
        # - Intervenes earlier (soft_enter instead of outer_enter)
        # - Uses proportional torque based on abs(error)
        # - Velocity-aware decay to prevent overshoot
        # - Inherently symmetric for positive and negative error
        # Only activated when apc_proportional_soft_band_mode is True
        # =====================================================================
        apc_soft_enter = self.authority_schedule.apc_soft_enter_m
        apc_proportional_mode = (
            self.authority_schedule.apc_proportional_soft_band_mode and apc_soft_enter > 0.0
        )  # APCR1d uses proportional mode with soft_enter > 0

        # Initialize common variables
        apc_inner_exit = self.authority_schedule.apc_inner_exit_m
        apc_outer_enter = self.authority_schedule.apc_outer_enter_m
        apc_safety_override = False
        proportional_scale = 0.0
        velocity_decay_active = False
        velocity_decay_disabled_reason = "none"
        # APCR1f: Use sagittal_velocity for error rate to detect direction of error motion
        apc_error_rate = sagittal_velocity_m_s  # m/s
        # Initialize APCR1e adaptive authority variables
        adaptive_enabled = False
        adaptive_max_tau = self.authority_schedule.apc_max_cross_tau
        boost_tau = 0.0
        boost_reason = "none"
        startup_boost_active = False

        # APCR1f Adaptive Fast Response - initialize early so available for both branches
        fast_response_enabled = self.authority_schedule.apc_fast_response_enabled
        phase_brake_enabled = self.authority_schedule.apc_phase_brake_enabled
        predictive_enabled = self.authority_schedule.apc_predictive_enabled
        drift_priority_enabled = self.authority_schedule.apc_drift_priority_enabled
        hysteresis_enabled = self.authority_schedule.apc_hysteresis_enabled

        # APCR1d proportional soft band logic
        # Exclude predictive and hysteresis profiles so they take dedicated elif branches
        if apc_proportional_mode and apc_enabled and apc_gate_safe and not predictive_enabled and not hysteresis_enabled:
            # Symmetric proportional control based on abs(error)
            abs_error = abs(signed_error)

            # Compute moving_away_from_zero and moving_toward_zero for APCR1f
            # These need to be computed inside the proportional block because they depend on signed_error
            moving_away_from_zero = signed_error * apc_error_rate > 0.0
            moving_toward_zero = signed_error * apc_error_rate < 0.0

            # Determine state
            apc_proportional_state = "NEUTRAL"
            if abs_error > apc_soft_enter:
                apc_proportional_state = "SOFT_RECENTER"

            # Compute proportional scale
            apc_inner_deadband = apc_inner_exit  # Reuse inner_exit as deadband
            apc_full_torque_error = apc_outer_enter  # Reuse outer_enter as full-torque threshold

            if abs_error <= apc_inner_deadband:
                proportional_scale = 0.0
            elif abs_error >= apc_full_torque_error:
                proportional_scale = 1.0
            else:
                # Smooth interpolation between deadband and full-torque threshold
                u = (abs_error - apc_inner_deadband) / (apc_full_torque_error - apc_inner_deadband)
                u_clamped = max(0.0, min(1.0, u))
                proportional_scale = u_clamped * u_clamped * (3.0 - 2.0 * u_clamped)

            # Direction: push toward zero
            apc_direction = -1.0 if signed_error > 0.0 else (1.0 if signed_error < 0.0 else 0.0)

            # Velocity decay: reduce torque if error is moving toward zero
            if self.authority_schedule.apc_velocity_decay_enabled:
                # e * e_dot < 0 means error is moving toward zero
                if signed_error * apc_error_rate < 0.0:
                    # Check if velocity decay should be disabled
                    vd_disabled = False
                    if self.authority_schedule.apc_adaptive_authority_enabled:
                        if self.authority_schedule.apc_adaptive_disable_vd_during_startup and self._apc_adaptive_no_improvement_count < self.authority_schedule.apc_adaptive_startup_boost_steps:
                            vd_disabled = True
                            velocity_decay_disabled_reason = "startup"
                        elif abs_error > self.authority_schedule.apc_adaptive_disable_vd_when_abs_e_gt:
                            vd_disabled = True
                            velocity_decay_disabled_reason = "high_error"

                    if not vd_disabled:
                        proportional_scale = proportional_scale * self.authority_schedule.apc_velocity_decay_factor
                        velocity_decay_active = True

            # =====================================================================
            # APCR1f Adaptive Fast Response with Phase Brake
            # Only activated when apc_fast_response_enabled is True
            # =====================================================================
            # Enable adaptive authority for profiles with adaptive_authority_enabled
            # This applies to APCR1e (via apc_adaptive_authority_enabled), APCR1f (via fast_response_enabled),
            # APCR1g (via predictive_enabled), and APCR1h (via drift_priority_enabled)
            if self.authority_schedule.apc_adaptive_authority_enabled or fast_response_enabled or predictive_enabled or drift_priority_enabled:
                adaptive_enabled = True
                # Track no improvement
                if abs_error >= self._apc_adaptive_prev_abs_error:
                    self._apc_adaptive_no_improvement_count += 1
                else:
                    self._apc_adaptive_no_improvement_count = 0
                self._apc_adaptive_prev_abs_error = abs_error

                # Track increasing error (3+ consecutive steps)
                if abs_error > self._apc_fast_response_prev_error:
                    self._apc_fast_response_increasing_error_count += 1
                else:
                    self._apc_fast_response_increasing_error_count = 0
                self._apc_fast_response_prev_error = abs_error

                # APCR1f parameters
                base_tau = self.authority_schedule.apc_fast_response_base_tau
                max_boost = self.authority_schedule.apc_fast_response_boost_tau_max
                desired_band = self.authority_schedule.apc_fast_response_desired_band_m
                phase_brake_threshold = self.authority_schedule.apc_phase_brake_threshold_m
                phase_brake_damping = self.authority_schedule.apc_phase_brake_damping_factor

                # Calculate boost based on APCR1f conditions
                boost_tau = 0.0
                boost_reason = "none"

                # Condition 1: Beyond desired band
                if abs_error > desired_band:
                    boost_tau = min(max_boost, boost_tau + max_boost * 0.5)
                    if boost_reason == "none":
                        boost_reason = "beyond_band"

                # Condition 2: Moving away from zero (APCR1f: disable velocity decay, increase boost faster)
                if moving_away_from_zero and abs_error >= self.authority_schedule.apc_adaptive_boost_start_error_m:
                    boost_tau = min(max_boost, boost_tau + max_boost * 0.5)  # 0.5 vs 0.4 in APCR1e
                    if boost_reason == "none":
                        boost_reason = "moving_away"

                # Condition 3: No improvement for N steps (APCR1f: 5 steps vs 8 in APCR1e)
                if self._apc_adaptive_no_improvement_count >= self.authority_schedule.apc_fast_response_no_improvement_window:
                    boost_tau = min(max_boost, boost_tau + max_boost * 0.3)
                    if boost_reason == "none":
                        boost_reason = "no_improvement"

                # Condition 4: Startup boost
                if self._apc_adaptive_no_improvement_count < self.authority_schedule.apc_adaptive_startup_boost_steps:
                    if abs_error > 0.04 or float(pitch_x_rad) > 0.02:
                        boost_tau = min(max_boost, boost_tau + 0.3)
                        if boost_reason == "none":
                            boost_reason = "startup"
                        startup_boost_active = True

                # Condition 5: APCR1f - Increasing error for 3+ steps
                if self._apc_fast_response_increasing_error_count >= self.authority_schedule.apc_increasing_error_threshold_steps:
                    boost_tau = min(max_boost, boost_tau + max_boost * self.authority_schedule.apc_increasing_error_boost_factor)
                    if boost_reason == "none":
                        boost_reason = "increasing_error"

                # Final adaptive max tau (APCR1f: higher ceiling)
                self._apc_fast_response_adaptive_tau_limit = base_tau + boost_tau

                # APCR1f Phase Brake: When error is already returning toward zero
                # Applies when: moving_toward_zero AND abs_error > phase_brake_threshold
                # Phase brake reduces scale to prevent overshoot as error returns to zero
                self._apc_fast_response_phase_brake_active = False
                if phase_brake_enabled and moving_toward_zero:
                    if abs_error > 0.10:
                        # Error still very large, don't decay too early
                        pass  # maintain current scale
                    elif abs_error > phase_brake_threshold:
                        # Apply phase brake - reduce scale by damping factor
                        proportional_scale = proportional_scale * phase_brake_damping
                        self._apc_fast_response_phase_brake_active = True
                    else:
                        # Error near zero, decay quickly
                        proportional_scale = proportional_scale * phase_brake_damping * phase_brake_damping
                        self._apc_fast_response_phase_brake_active = True

                # APCR1h: Drift Priority Override
                # Applies when abs_error > drift_priority_enter AND moving away
                # Uses higher tau limit and rate to force drift back toward zero
                self._apc_drift_priority_active = False
                self._apc_drift_priority_emergency_active = False
                drift_priority_reason = "none"
                selected_tau_limit = adaptive_max_tau
                selected_rate_limit = self.authority_schedule.apc_fast_response_max_rate_per_step

                if drift_priority_enabled:
                    drift_priority_enter = self.authority_schedule.apc_drift_priority_enter_m
                    emergency_threshold = self.authority_schedule.apc_drift_priority_emergency_m
                    hard_threshold = self.authority_schedule.apc_drift_priority_hard_m

                    # Compute current error rate (e_dot)
                    e_dot = signed_error - self._apc_drift_priority_prev_error
                    self._apc_drift_priority_prev_error = signed_error
                    moving_away_drift = signed_error * e_dot > 0.0  # drift moving away from zero

                    # Track steps since hard drift (> 0.15)
                    if abs_error > hard_threshold:
                        self._apc_drift_priority_steps_since_hard_drift += 1
                    else:
                        self._apc_drift_priority_steps_since_hard_drift = 0

                    # Check if error rate reversal was achieved (e_dot sign changed)
                    if e_dot * self._apc_drift_priority_prev_error < 0 and abs(self._apc_drift_priority_prev_error) > 0.01:
                        self._apc_drift_priority_error_rate_reversal_achieved = True
                    self._apc_drift_priority_prev_error = e_dot

                    # Determine drift priority level
                    if abs_error > emergency_threshold and moving_away_drift:
                        # Emergency clamp mode
                        self._apc_drift_priority_emergency_active = True
                        self._apc_drift_priority_active = True
                        selected_tau_limit = self.authority_schedule.apc_drift_priority_emergency_max_tau
                        selected_rate_limit = self.authority_schedule.apc_drift_priority_emergency_rate
                        drift_priority_reason = "emergency"
                    elif abs_error > drift_priority_enter and moving_away_drift:
                        # Drift priority mode
                        self._apc_drift_priority_active = True
                        selected_tau_limit = self.authority_schedule.apc_drift_priority_drift_priority_max_tau
                        selected_rate_limit = self.authority_schedule.apc_drift_priority_drift_priority_rate
                        drift_priority_reason = "drift_priority"

                    # Apply drift priority tau limit (overrides adaptive max)
                    if self._apc_drift_priority_active:
                        self._apc_drift_priority_tau_limit = selected_tau_limit
                        self._apc_drift_priority_rate_limit = selected_rate_limit

                        # Apply rate limiting to tau
                        tau_delta = selected_tau_limit - self._apc_drift_priority_prev_tau
                        rate_limited_delta = max(-selected_rate_limit, min(selected_rate_limit, tau_delta))
                        adaptive_max_tau = self._apc_drift_priority_prev_tau + rate_limited_delta
                        self._apc_drift_priority_prev_tau = adaptive_max_tau

                    # Phase brake is disabled when drift priority is active (to allow full correction)
                    if self._apc_drift_priority_active:
                        self._apc_fast_response_phase_brake_active = False
                        # Don't reduce scale - let full torque through
                    else:
                        # Use APCR1f max tau only when drift priority is NOT active
                        adaptive_max_tau = self._apc_fast_response_adaptive_tau_limit

            # Raw torque with proportional shaping (using adaptive_max_tau)
            apc_raw_tau = apc_direction * adaptive_max_tau * proportional_scale

            # APCR1f: Track tau before rate limit for telemetry
            apc_tau_before_rate_limit = apc_raw_tau

            # Pitch-aware scale (same as original)
            pitch_aware_scale = 1.0
            if apc_pitch_abs > self.authority_schedule.apc_pitch_safe_limit_rad:
                pitch_aware_scale = self.authority_schedule.apc_pitch_safe_limit_rad / apc_pitch_abs
                pitch_aware_scale = max(0.1, pitch_aware_scale)
            apc_raw_tau = apc_raw_tau * pitch_aware_scale

            # State machine tracking for telemetry (simplified for proportional mode)
            if apc_proportional_state == "SOFT_RECENTER" and self._apc_state == "NEUTRAL":
                self._apc_state_entry_count += 1
            elif apc_proportional_state == "NEUTRAL" and self._apc_state != "NEUTRAL":
                self._apc_state_exit_count += 1

            apc_state = apc_proportional_state
            apc_active = apc_proportional_state == "SOFT_RECENTER"

        # =====================================================================
        # APCR1i Support Hysteresis Recenter
        # Symmetric hysteresis state machine that holds recenter direction
        # until error reaches inner band or crosses to opposite side
        # Key differences from APCR1h:
        # - Full symmetric hysteresis state machine
        # - Does NOT exit when e_dot reverses while |e| > inner_exit_m
        # - Holds direction through zero crossing until inside inner band
        # =====================================================================
        elif self.authority_schedule.apc_hysteresis_enabled and apc_enabled and apc_gate_safe:
            abs_error = abs(signed_error)

            # Get hysteresis parameters
            outer_enter = self.authority_schedule.apc_hysteresis_outer_enter_m
            inner_exit = self.authority_schedule.apc_hysteresis_inner_exit_m
            opposite_release = self.authority_schedule.apc_hysteresis_opposite_release_m
            near_zero = self.authority_schedule.apc_hysteresis_near_zero_m
            emergency_threshold = self.authority_schedule.apc_hysteresis_emergency_m

            # Compute error rate
            e_dot = signed_error - self._apc_hysteresis_prev_e
            self._apc_hysteresis_prev_e = signed_error
            moving_away_from_zero = signed_error * e_dot > 0.0
            moving_toward_zero = signed_error * e_dot < 0.0

            # State machine transitions
            prev_state = self._apc_hysteresis_state

            if self._apc_hysteresis_state == "NEUTRAL":
                # Entry conditions from NEUTRAL
                if signed_error > outer_enter:
                    self._apc_hysteresis_state = "RECENTER_FROM_POSITIVE"
                    self._apc_hysteresis_entry_e = signed_error
                    self._apc_hysteresis_state_entry_count += 1
                elif signed_error < -outer_enter:
                    self._apc_hysteresis_state = "RECENTER_FROM_NEGATIVE"
                    self._apc_hysteresis_entry_e = signed_error
                    self._apc_hysteresis_state_entry_count += 1

            elif self._apc_hysteresis_state == "RECENTER_FROM_POSITIVE":
                # Exit conditions from RECENTER_FROM_POSITIVE
                # Priority 1: Check for overshoot into opposite direction FIRST
                if signed_error < -opposite_release:
                    # Exit to RECENTER_FROM_NEGATIVE when overshoot into opposite direction
                    self._apc_hysteresis_exit_e = signed_error
                    self._apc_hysteresis_state = "RECENTER_FROM_NEGATIVE"
                    self._apc_hysteresis_state_entry_count += 1
                # Priority 2: Normal exit when error returns to inner band
                elif signed_error <= inner_exit and e_dot < 0:
                    self._apc_hysteresis_exit_e = signed_error
                    self._apc_hysteresis_state = "NEUTRAL"
                    self._apc_hysteresis_state_exit_count += 1
                # Priority 3: Exit when near zero
                elif signed_error <= near_zero and e_dot < 0:
                    self._apc_hysteresis_exit_e = signed_error
                    self._apc_hysteresis_state = "NEUTRAL"
                    self._apc_hysteresis_state_exit_count += 1

            elif self._apc_hysteresis_state == "RECENTER_FROM_NEGATIVE":
                # Exit conditions from RECENTER_FROM_NEGATIVE
                # Priority 1: Check for overshoot into opposite direction FIRST
                if signed_error > opposite_release:
                    # Exit to RECENTER_FROM_POSITIVE when overshoot into opposite direction
                    self._apc_hysteresis_exit_e = signed_error
                    self._apc_hysteresis_state = "RECENTER_FROM_POSITIVE"
                    self._apc_hysteresis_state_entry_count += 1
                # Priority 2: Normal exit when error returns to inner band
                elif signed_error >= -inner_exit and e_dot > 0:
                    self._apc_hysteresis_exit_e = signed_error
                    self._apc_hysteresis_state = "NEUTRAL"
                    self._apc_hysteresis_state_exit_count += 1
                # Priority 3: Exit when near zero
                elif signed_error >= -near_zero and e_dot > 0:
                    self._apc_hysteresis_exit_e = signed_error
                    self._apc_hysteresis_state = "NEUTRAL"
                    self._apc_hysteresis_state_exit_count += 1

            elif self._apc_hysteresis_state == "HOLD_THROUGH_ZERO":
                # Exit conditions from HOLD_THROUGH_ZERO
                if abs_error < inner_exit and abs(e_dot) < 0.01:
                    # Exit when error near zero and velocity low
                    self._apc_hysteresis_exit_e = signed_error
                    self._apc_hysteresis_state = "NEUTRAL"
                    self._apc_hysteresis_state_exit_count += 1
                elif signed_error > opposite_release:
                    # Re-enter RECENTER_FROM_POSITIVE
                    self._apc_hysteresis_state = "RECENTER_FROM_POSITIVE"
                    self._apc_hysteresis_entry_e = signed_error
                    self._apc_hysteresis_state_entry_count += 1
                elif signed_error < -opposite_release:
                    # Re-enter RECENTER_FROM_NEGATIVE
                    self._apc_hysteresis_state = "RECENTER_FROM_NEGATIVE"
                    self._apc_hysteresis_entry_e = signed_error
                    self._apc_hysteresis_state_entry_count += 1

            # Determine torque direction and magnitude based on state
            if self._apc_hysteresis_state == "NEUTRAL":
                apc_tau = 0.0
                apc_direction = "none"
                apc_active = False
                # Phase brake enabled in NEUTRAL
                self._apc_fast_response_phase_brake_active = True

            elif self._apc_hysteresis_state in ("RECENTER_FROM_POSITIVE", "RECENTER_FROM_NEGATIVE"):
                # Apply recenter torque
                apc_active = True
                # Determine direction: push toward zero
                apc_direction = "negative" if self._apc_hysteresis_state == "RECENTER_FROM_POSITIVE" else "positive"
                tau_sign = -1.0 if self._apc_hysteresis_state == "RECENTER_FROM_POSITIVE" else 1.0

                # Determine max tau based on emergency condition
                self._apc_hysteresis_emergency_active = abs_error > emergency_threshold
                if self._apc_hysteresis_emergency_active:
                    max_tau = self.authority_schedule.apc_hysteresis_emergency_max_tau
                    rate_limit = self.authority_schedule.apc_hysteresis_emergency_rate
                else:
                    max_tau = self.authority_schedule.apc_hysteresis_recenter_max_tau
                    rate_limit = self.authority_schedule.apc_hysteresis_recenter_rate

                # Apply rate limiting
                tau_delta = max_tau - abs(self._apc_hysteresis_prev_tau)
                rate_limited_tau = abs(self._apc_hysteresis_prev_tau) + min(abs(tau_delta), rate_limit)
                rate_limited_tau = min(rate_limited_tau, max_tau)

                apc_tau = tau_sign * rate_limited_tau
                self._apc_hysteresis_prev_tau = apc_tau

                # Phase brake disabled in recenter state
                self._apc_fast_response_phase_brake_active = False

            elif self._apc_hysteresis_state == "HOLD_THROUGH_ZERO":
                # Hold through zero - continue applying torque in same direction
                apc_active = True
                # Keep same direction as previous
                prev_direction = "positive" if self._apc_hysteresis_prev_tau > 0 else "negative"
                apc_direction = prev_direction
                tau_sign = 1.0 if prev_direction == "positive" else -1.0

                max_tau = self.authority_schedule.apc_hysteresis_hold_max_tau
                rate_limit = self.authority_schedule.apc_hysteresis_emergency_rate

                # Apply rate limiting
                tau_delta = max_tau - abs(self._apc_hysteresis_prev_tau)
                rate_limited_tau = abs(self._apc_hysteresis_prev_tau) + min(abs(tau_delta), rate_limit)
                rate_limited_tau = min(rate_limited_tau, max_tau)

                apc_tau = tau_sign * rate_limited_tau
                self._apc_hysteresis_prev_tau = apc_tau

                # Phase brake disabled in hold state
                self._apc_fast_response_phase_brake_active = False

            # State machine completed - set state for telemetry
            apc_state = self._apc_hysteresis_state

            # Set common variables for APCR1i
            apc_raw_tau = apc_tau
            apc_active = self._apc_hysteresis_state != "NEUTRAL"
            apc_tau_before_rate_limit = apc_tau

        # =====================================================================
        # APCR1g Predictive Fast Response with Phase Brake
        # Uses predictive error: e_pred = e + lead_time * e_dot
        # Activates earlier when predicted error exceeds thresholds
        # =====================================================================
        elif self.authority_schedule.apc_predictive_enabled and apc_enabled and apc_gate_safe:
            # Symmetric proportional control based on abs(error) and predicted error
            abs_error = abs(signed_error)

            # Compute error rate from sagittal velocity (rate of change of drift)
            apc_error_rate = sagittal_velocity_m_s  # m/s

            # Compute predicted error: e_pred = e + lead_time * e_dot
            lead_time = self.authority_schedule.apc_lead_time_s
            predicted_error = signed_error + lead_time * apc_error_rate
            abs_predicted_error = abs(predicted_error)

            # Track predicted error for telemetry
            self._apc_predictive_predicted_error = predicted_error

            # Compute moving_away_from_zero and moving_toward_zero
            moving_away_from_zero = signed_error * apc_error_rate > 0.0
            moving_toward_zero = signed_error * apc_error_rate < 0.0

            # Predictive activation: activate when predicted error exceeds threshold AND moving away
            predictive_enter_threshold = self.authority_schedule.apc_predicted_enter_m
            predictive_enter_active = (
                abs_predicted_error > predictive_enter_threshold and moving_away_from_zero
            )
            self._apc_predictive_predictive_trigger_active = predictive_enter_active

            # Determine which error to use for proportional scale
            # Use predicted error for activation check if predictive trigger is active
            error_for_activation = abs_predicted_error if predictive_enter_active else abs_error

            # Determine state
            apc_predictive_inner_deadband = self.authority_schedule.apc_predictive_inner_deadband_m
            apc_predictive_soft_enter = self.authority_schedule.apc_predictive_soft_enter_m
            apc_predictive_full_torque = self.authority_schedule.apc_predictive_full_torque_m
            apc_predictive_desired_band = self.authority_schedule.apc_predictive_desired_band_m
            apc_predictive_emergency = self.authority_schedule.apc_predictive_emergency_error_m

            apc_proportional_state = "NEUTRAL"
            if error_for_activation > apc_predictive_soft_enter:
                apc_proportional_state = "SOFT_RECENTER"

            # Compute proportional scale
            if abs_error <= apc_predictive_inner_deadband:
                proportional_scale = 0.0
            elif abs_error >= apc_predictive_full_torque:
                proportional_scale = 1.0
            else:
                # Smooth interpolation between deadband and full-torque threshold
                u = (abs_error - apc_predictive_inner_deadband) / (apc_predictive_full_torque - apc_predictive_inner_deadband)
                u_clamped = max(0.0, min(1.0, u))
                proportional_scale = u_clamped * u_clamped * (3.0 - 2.0 * u_clamped)

            # Direction: push toward zero
            # Sign convention: e > 0 → tau > 0 (positive torque pushes backward)
            # Use predicted error direction if it's larger than current error
            if abs_predicted_error > abs_error:
                apc_direction = 1.0 if predicted_error > 0.0 else (-1.0 if predicted_error < 0.0 else 0.0)
            else:
                apc_direction = 1.0 if signed_error > 0.0 else (-1.0 if signed_error < 0.0 else 0.0)

            # Track no improvement
            if abs_error >= self._apc_predictive_prev_abs_error:
                self._apc_predictive_no_improvement_count += 1
            else:
                self._apc_predictive_no_improvement_count = 0
            self._apc_predictive_prev_abs_error = abs_error

            # Track increasing error
            if abs_error > self._apc_predictive_prev_error:
                self._apc_predictive_increasing_error_count += 1
            else:
                self._apc_predictive_increasing_error_count = 0
            self._apc_predictive_prev_error = abs_error

            # Calculate adaptive authority
            base_tau = self.authority_schedule.apc_predictive_base_tau
            max_tau = self.authority_schedule.apc_predictive_max_tau
            max_boost = self.authority_schedule.apc_predictive_boost_tau_max
            startup_boost_max = self.authority_schedule.apc_predictive_startup_boost_max_tau
            startup_boost_steps = 50  # Same as APCR1f

            # Calculate boost based on conditions
            boost_tau = 0.0
            boost_reason = "none"
            self._apc_predictive_predictive_boost_active = False

            # Condition 1: Predictive boost - when predicted error indicates future overshoot
            predicted_full_threshold = self.authority_schedule.apc_predicted_full_response_m
            if abs_predicted_error > predicted_full_threshold and moving_away_from_zero:
                boost_tau = min(max_boost, boost_tau + max_boost * 0.5)
                self._apc_predictive_predictive_boost_active = True
                if boost_reason == "none":
                    boost_reason = "predictive_boost"

            # Condition 2: Beyond desired band
            if abs_error > apc_predictive_desired_band:
                boost_tau = min(max_boost, boost_tau + max_boost * 0.3)
                if boost_reason == "none":
                    boost_reason = "beyond_band"

            # Condition 3: Moving away from zero
            if moving_away_from_zero and abs_error >= self.authority_schedule.apc_adaptive_boost_start_error_m:
                boost_tau = min(max_boost, boost_tau + max_boost * 0.5)
                if boost_reason == "none":
                    boost_reason = "moving_away"

            # Condition 4: No improvement for N steps
            no_improve_window = self.authority_schedule.apc_predictive_no_improvement_window
            if self._apc_predictive_no_improvement_count >= no_improve_window:
                boost_tau = min(max_boost, boost_tau + max_boost * 0.3)
                if boost_reason == "none":
                    boost_reason = "no_improvement"

            # Condition 5: Startup boost
            if self._apc_predictive_no_improvement_count < startup_boost_steps:
                if abs_error > 0.04 or float(pitch_x_rad) > 0.02:
                    boost_tau = min(startup_boost_max - base_tau, boost_tau + 0.3)
                    if boost_reason == "none":
                        boost_reason = "startup"
                    startup_boost_active = True

            # Condition 6: Increasing error for 2+ steps
            increase_threshold = self.authority_schedule.apc_predictive_increasing_error_threshold_steps
            if self._apc_predictive_increasing_error_count >= increase_threshold:
                boost_tau = min(max_boost, boost_tau + max_boost * self.authority_schedule.apc_predictive_increasing_error_boost_factor)
                if boost_reason == "none":
                    boost_reason = "increasing_error"

            # Final adaptive max tau
            self._apc_predictive_adaptive_tau_limit = min(base_tau + boost_tau, max_tau)

            # Phase brake: When error is already returning toward zero
            phase_brake_enabled = self.authority_schedule.apc_predictive_phase_brake_enabled
            phase_brake_threshold = self.authority_schedule.apc_predictive_phase_brake_threshold_m
            phase_brake_strong_threshold = self.authority_schedule.apc_predictive_phase_brake_strong_threshold_m
            phase_brake_factor = self.authority_schedule.apc_predictive_phase_brake_factor
            phase_brake_strong_factor = self.authority_schedule.apc_predictive_phase_brake_strong_factor

            self._apc_predictive_phase_brake_active = False
            self._apc_predictive_phase_brake_strong_active = False

            if phase_brake_enabled and moving_toward_zero:
                if abs_error > 0.10:
                    # Error still very large, don't decay too early
                    pass  # maintain current scale
                elif abs_error > phase_brake_strong_threshold:
                    # Apply phase brake - reduce scale by damping factor
                    proportional_scale = proportional_scale * phase_brake_factor
                    self._apc_predictive_phase_brake_active = True
                else:
                    # Error near zero, apply strong brake and decay quickly
                    proportional_scale = proportional_scale * phase_brake_strong_factor
                    self._apc_predictive_phase_brake_active = True
                    self._apc_predictive_phase_brake_strong_active = True

            # Use adaptive max tau
            adaptive_max_tau = self._apc_predictive_adaptive_tau_limit

            # Velocity decay disabled at high error
            vd_disabled_predictive = False
            if abs_error > self.authority_schedule.apc_predictive_disable_vd_when_abs_e_gt:
                vd_disabled_predictive = True

            # Apply velocity decay if not disabled and moving toward zero
            if not vd_disabled_predictive and moving_toward_zero:
                vd_factor = self.authority_schedule.apc_velocity_decay_factor
                proportional_scale = proportional_scale * vd_factor
                velocity_decay_active = True

            # Raw torque with proportional shaping
            apc_raw_tau = apc_direction * adaptive_max_tau * proportional_scale

            # Track tau before rate limit for telemetry
            apc_tau_before_rate_limit = apc_raw_tau

            # Pitch-aware scale
            pitch_aware_scale = 1.0
            if apc_pitch_abs > self.authority_schedule.apc_pitch_safe_limit_rad:
                pitch_aware_scale = self.authority_schedule.apc_pitch_safe_limit_rad / apc_pitch_abs
                pitch_aware_scale = max(0.1, pitch_aware_scale)
            apc_raw_tau = apc_raw_tau * pitch_aware_scale

            # State machine tracking for telemetry
            if apc_proportional_state == "SOFT_RECENTER" and self._apc_state == "NEUTRAL":
                self._apc_state_entry_count += 1
            elif apc_proportional_state == "NEUTRAL" and self._apc_state != "NEUTRAL":
                self._apc_state_exit_count += 1

            apc_state = apc_proportional_state
            apc_active = apc_proportional_state == "SOFT_RECENTER"

        else:
            # Original bang-bang state machine logic
            # Outer enter thresholds
            apc_opposite_overshoot = self.authority_schedule.apc_opposite_overshoot_m
            apc_pitch_enter = self.authority_schedule.apc_pitch_enter_rad

            # Exit targets with slight overshoot
            apc_exit_target_positive = apc_inner_exit - apc_opposite_overshoot  # For CROSS_FROM_POSITIVE
            apc_exit_target_negative = -apc_inner_exit + apc_opposite_overshoot  # For CROSS_FROM_NEGATIVE

            # Safety override: exit any crossing state to NEUTRAL
            apc_safety_override = False
            apc_state = self._apc_state
            if apc_state != "NEUTRAL" and not apc_gate_safe:
                apc_state = "NEUTRAL"
                apc_safety_override = True
                self._apc_safety_override_count += 1

            # Track persistent tau sign for alternative entry detection
            current_tau_sign = 1 if float(tau_pitch) > 0.5 else (-1 if float(tau_pitch) < -0.5 else 0)
            if current_tau_sign != 0 and current_tau_sign == self._apc_prev_tau_sign:
                self._apc_persistent_tau_sign_steps += 1
            else:
                self._apc_persistent_tau_sign_steps = 0
            self._apc_prev_tau_sign = current_tau_sign
            tau_pitch_persistent_positive = self._apc_persistent_tau_sign_steps >= 5 and current_tau_sign > 0
            tau_pitch_persistent_negative = self._apc_persistent_tau_sign_steps >= 5 and current_tau_sign < 0

            # State transitions
            if apc_state == "NEUTRAL":
                # Check if we should enter a crossing state
                if apc_enabled and apc_gate_safe:
                    # Enter CROSS_FROM_POSITIVE: signed_error > outer AND (pitch > threshold OR tau_pitch persistent positive)
                    if signed_error > apc_outer_enter and (float(pitch_x_rad) > apc_pitch_enter or tau_pitch_persistent_positive):
                        apc_state = "CROSS_FROM_POSITIVE"
                        self._apc_state_entry_count += 1
                    # Enter CROSS_FROM_NEGATIVE: signed_error < -outer AND (pitch < -threshold OR tau_pitch persistent negative)
                    elif signed_error < -apc_outer_enter and (float(pitch_x_rad) < -apc_pitch_enter or tau_pitch_persistent_negative):
                        apc_state = "CROSS_FROM_NEGATIVE"
                        self._apc_state_entry_count += 1

            elif apc_state == "CROSS_FROM_POSITIVE":
                # Keep applying negative torque until exit condition
                # Exit when: signed_error <= inner_exit_m OR signed_error < 0 (crossed slightly)
                if signed_error <= apc_exit_target_positive:
                    apc_state = "NEUTRAL"
                    self._apc_state_exit_count += 1

            elif apc_state == "CROSS_FROM_NEGATIVE":
                # Keep applying positive torque until exit condition
                # Exit when: signed_error >= -inner_exit_m OR signed_error > 0 (crossed slightly)
                if signed_error >= apc_exit_target_negative:
                    apc_state = "NEUTRAL"
                    self._apc_state_exit_count += 1

            # Store the new state
            self._apc_state = apc_state

            # Compute APC torque based on state
            apc_raw_tau = 0.0
            if apc_state == "CROSS_FROM_POSITIVE":
                # Apply NEGATIVE torque to make pitch_rate negative (push back from forward lean)
                # Reduce torque if pitch is already large (avoid overcorrection)
                pitch_aware_scale = 1.0
                if apc_pitch_abs > self.authority_schedule.apc_pitch_safe_limit_rad:
                    pitch_aware_scale = self.authority_schedule.apc_pitch_safe_limit_rad / apc_pitch_abs
                    pitch_aware_scale = max(0.1, pitch_aware_scale)  # At least 10% torque
                apc_raw_tau = -self.authority_schedule.apc_max_cross_tau * pitch_aware_scale
            elif apc_state == "CROSS_FROM_NEGATIVE":
                # Apply POSITIVE torque to make pitch_rate positive (push back from backward lean)
                pitch_aware_scale = 1.0
                if apc_pitch_abs > self.authority_schedule.apc_pitch_safe_limit_rad:
                    pitch_aware_scale = self.authority_schedule.apc_pitch_safe_limit_rad / apc_pitch_abs
                    pitch_aware_scale = max(0.1, pitch_aware_scale)
                apc_raw_tau = self.authority_schedule.apc_max_cross_tau * pitch_aware_scale

            # APC active flag
            apc_active = apc_state != "NEUTRAL"

            # For bang-bang mode, track tau before rate limit
            apc_tau_before_rate_limit = apc_raw_tau

        # Clip APC torque
        apc_raw_tau = float(jnp.clip(
            apc_raw_tau,
            -self.authority_schedule.apc_max_cross_tau,
            self.authority_schedule.apc_max_cross_tau
        ))

        # Smooth APC torque (APCR1f/APCR1g: more responsive smoothing)
        if predictive_enabled:
            apc_alpha = self.authority_schedule.apc_predictive_smooth_alpha
        elif fast_response_enabled:
            apc_alpha = self.authority_schedule.apc_fast_response_smooth_alpha
        else:
            apc_alpha = self.authority_schedule.apc_smooth_alpha
        apc_smoothed_tau = apc_alpha * apc_raw_tau + (1.0 - apc_alpha) * self._apc_prev_tau

        # Rate limit APC torque (APCR1f/APCR1g: faster rate limit)
        if predictive_enabled:
            apc_max_rate = self.authority_schedule.apc_predictive_max_rate_per_step
        elif fast_response_enabled:
            apc_max_rate = self.authority_schedule.apc_fast_response_max_rate_per_step
        else:
            apc_max_rate = self.authority_schedule.apc_max_rate_per_step
        if apc_smoothed_tau > self._apc_prev_tau + apc_max_rate:
            apc_final_tau = self._apc_prev_tau + apc_max_rate
        elif apc_smoothed_tau < self._apc_prev_tau - apc_max_rate:
            apc_final_tau = self._apc_prev_tau - apc_max_rate
        else:
            apc_final_tau = apc_smoothed_tau

        # Store for next step
        self._apc_prev_tau = apc_final_tau

        # Clip final APC torque
        apc_tau_clipped = float(jnp.clip(
            apc_final_tau,
            -self.authority_schedule.apc_max_cross_tau,
            self.authority_schedule.apc_max_cross_tau
        ))

        # APC active flag
        apc_active = apc_state != "NEUTRAL"

        # APC gate reason for telemetry
        if not apc_enabled:
            apc_gate_reason = "disabled"
        elif apc_safety_override:
            apc_gate_reason = "safety_override"
        elif apc_pitch_danger:
            apc_gate_reason = "pitch_danger"
        elif not apc_pitch_safe:
            apc_gate_reason = "pitch_unsafe"
        elif not apc_contact_safe:
            apc_gate_reason = "contact_invalid"
        elif not apc_height_safe:
            apc_gate_reason = "height_unsafe"
        elif not apc_roll_safe:
            apc_gate_reason = "roll_unsafe"
        elif apc_state == "NEUTRAL":
            apc_gate_reason = "waiting_for_threshold"
        else:
            apc_gate_reason = "active"

        # ---- LP family: Priority Sagittal Allocator (pitch-first support-residual) ---- #
        # Architectural alternative to LR/LRS. Computes pitch priority first, then
        # allocates support-centering torque only from residual safe authority.
        # Support is gated by pitch safety, saturation headroom, direction
        # consistency, and slew limits. Preserves K1 EQ/FF baseline.
        LP_enabled = self.authority_schedule.enable_lp_priority_allocator
        LP_kind = self.authority_schedule.lp_allocator_kind
        LP_gains = {}
        if LP_enabled and LP_kind.startswith("LP"):
            height_fb = float(com_z_m)
            if LP_kind == "LP1_pitch_first_support_residual":
                LP_gains = _lp_priority_gains_LP1(height_fb)
            elif LP_kind == "LP2_pitch_strong_support_soft":
                LP_gains = _lp_priority_gains_LP2(height_fb)
            elif LP_kind == "LP3_support_recenter_when_safe":
                LP_gains = _lp_priority_gains_LP3(height_fb)

        # LP telemetry defaults (populated when LP is active, zero otherwise)
        LP_eq_ff_pass_through = 0.0
        LP_pitch_error_rad = 0.0
        LP_pitch_rate_rad_s = 0.0
        LP_pitch_priority_raw = 0.0
        LP_pitch_priority = 0.0
        LP_pitch_priority_limit = 0.0
        LP_pitch_abs_gate = 0.0
        LP_pitch_rate_gate = 0.0
        LP_saturation_gate = 0.0
        LP_direction_gate = 0.0
        LP_support_gate = 0.0
        LP_support_error_m = 0.0
        LP_support_velocity_m_s = 0.0
        LP_support_raw = 0.0
        LP_support_allocated_raw = 0.0
        LP_support_allocated = 0.0
        LP_support_slew_limited = 0.0
        LP_support_limit_nm = 0.0
        LP_residual_authority_nm = 0.0
        LP_support_residual_fraction = 0.0
        LP_tau_total_preclip = 0.0
        LP_support_suppressed_reason = "lp_disabled"
        LP_near_saturation = False
        LP_support_direction_assists = False

        # ---- LR family: Replacement coordinated sagittal state feedback ---- #
        # Unlike the L family (which ADDS on top), LR REPLACES the independent
        # dynamic damping terms (tau_pitch_rate, tau_sagittal_velocity,
        # tau_support_velocity) with a single coordinated state-feedback command.
        #
        # CRITICAL (EQ/FF pass-through fix): LR PRESERVES K1's equilibrium/
        # feedforward baseline (tau_pitch + tau_position + tau_cp + tau_com_vy).
        # These terms carry the pitch equilibrium authority through the pitch
        # reference offset (pitch_eq + outer_loop + PFF) and the position
        # centering bias. Without this pass-through, the LR path has zero
        # equilibrium/feedforward and the total torque is ~10x too weak.
        #
        # Architecture:
        #   tau_common = tau_eq_ff_pass_through + LR_dynamic_feedback
        # where:
        #   tau_eq_ff_pass_through = tau_pitch + tau_position + tau_cp + tau_com_vy
        #     (equilibrium/feedforward — carries static authority to maintain height)
        #   LR_dynamic_feedback = k_pitch*pitch + k_pitch_rate*pitch_rate
        #     + k_support*support_err + k_support_vel*support_vel
        #     (replaces tau_pitch_rate + tau_sagittal_velocity + tau_support_velocity)
        LR_enabled = self.authority_schedule.enable_lr_replacement_feedback
        LR_kind = self.authority_schedule.lr_replacement_kind
        LR_feedback_torque = 0.0
        LR_k1_existing_estimate = 0.0
        LR_eq_ff_pass_through = 0.0
        LR_removed_dynamic_terms_estimate = 0.0
        LR_gains = {}
        LR_replacement_mode = "none"
        # LRS component-wise torque decomposition (for sign/phase audit)
        LRS_tau_pitch_component = 0.0
        LRS_tau_pitch_rate_component = 0.0
        LRS_tau_support_component = 0.0
        LRS_tau_support_vel_component = 0.0
        LR_state_vector = [
            float(pitch_x_rad),
            float(pitch_rate_for_damping),
            float(sagittal_position_error_m),
            float(support_position_velocity_m_s),
            float(wheel_vel_mean),
        ]
        if LR_enabled and (LR_kind.startswith("LR") or LR_kind.startswith("LRS")):
            height_fb = float(com_z_m)
            if LR_kind == "LR1_low_freq":
                LR_gains = _lr_replacement_gains_LR1(height_fb)
                g = LR_gains
                LR_feedback_torque = (
                    g["k_pitch"] * LR_state_vector[0]
                    + g["k_pitch_rate"] * LR_state_vector[1]
                    + g["k_support"] * LR_state_vector[2]
                    + g["k_support_vel"] * LR_state_vector[3]
                )
            elif LR_kind == "LR2_phase_lead":
                LR_gains = _lr_replacement_gains_LR2(height_fb)
                g = LR_gains
                pitch_accel = (LR_state_vector[1] - getattr(self, '_prev_pitch_rate_for_LR', 0.0)) / max(self.dt, 1e-6)
                self._prev_pitch_rate_for_LR = LR_state_vector[1]
                LR_feedback_torque = (
                    g["k_pitch"] * LR_state_vector[0]
                    + g["k_pitch_rate"] * LR_state_vector[1]
                    + g["k_support"] * LR_state_vector[2]
                    + g["k_support_vel"] * LR_state_vector[3]
                    + g.get("k_lead", 0.0) * pitch_accel
                )
            elif LR_kind == "LR3_pitch_ref_stabilized":
                LR_gains = _lr_replacement_gains_LR3(height_fb)
                g = LR_gains
                pitch_ref_mod = g["pitch_ref_gain"] * LR_state_vector[2]
                pitch_ref_mod = max(-g["pitch_ref_max_deg"], min(g["pitch_ref_max_deg"], pitch_ref_mod))
                pitch_ref_mod_rad = math.radians(pitch_ref_mod)
                LR_feedback_torque = (
                    g["k_pitch"] * LR_state_vector[0]
                    + g["k_pitch_rate"] * LR_state_vector[1]
                    + g["k_support"] * LR_state_vector[2]
                    + g["k_support_vel"] * LR_state_vector[3]
                    + self.kp_pitch * pitch_ref_mod_rad
                )
            elif LR_kind == "LRS1_support_dominant":
                LR_gains = _lrs_replacement_gains_S1(height_fb)
                g = LR_gains
                LR_feedback_torque = (
                    g["k_pitch"] * LR_state_vector[0]
                    + g["k_pitch_rate"] * LR_state_vector[1]
                    + g["k_support"] * LR_state_vector[2]
                    + g["k_support_vel"] * LR_state_vector[3]
                )
            elif LR_kind == "LRS2_pitch_rate_damping":
                LR_gains = _lrs_replacement_gains_S2(height_fb)
                g = LR_gains
                LR_feedback_torque = (
                    g["k_pitch"] * LR_state_vector[0]
                    + g["k_pitch_rate"] * LR_state_vector[1]
                    + g["k_support"] * LR_state_vector[2]
                    + g["k_support_vel"] * LR_state_vector[3]
                )
            elif LR_kind == "LRS3_balanced_medium":
                LR_gains = _lrs_replacement_gains_S3(height_fb)
                g = LR_gains
                LR_feedback_torque = (
                    g["k_pitch"] * LR_state_vector[0]
                    + g["k_pitch_rate"] * LR_state_vector[1]
                    + g["k_support"] * LR_state_vector[2]
                    + g["k_support_vel"] * LR_state_vector[3]
                )
            # Component-wise torque decomposition for LRS variants
            if LR_kind.startswith("LRS") and isinstance(LR_gains, dict) and LR_gains:
                g = LR_gains
                LRS_tau_pitch_component = float(g["k_pitch"] * LR_state_vector[0])
                LRS_tau_pitch_rate_component = float(g["k_pitch_rate"] * LR_state_vector[1])
                LRS_tau_support_component = float(g["k_support"] * LR_state_vector[2])
                LRS_tau_support_vel_component = float(g["k_support_vel"] * LR_state_vector[3])
            else:
                LRS_tau_pitch_component = 0.0
                LRS_tau_pitch_rate_component = 0.0
                LRS_tau_support_component = 0.0
                LRS_tau_support_vel_component = 0.0
            # Capture what K1's full sum-of-torques would have been
            LR_k1_existing_estimate = float(
                tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
                tau_support_velocity + tau_position + tau_cp + tau_com_vy
            )
            # Equilibrium/feedforward pass-through: preserve K1's pitch equilibrium,
            # position centering, and capture-point corrections.
            # These carry the static authority to maintain the target height.
            LR_eq_ff_pass_through = float(tau_pitch + tau_position + tau_cp + tau_com_vy)
            # Dynamic terms that LR replaces (captured before zeroing for telemetry)
            LR_removed_dynamic_terms_estimate = float(
                tau_pitch_rate + tau_sagittal_velocity + tau_support_velocity
            )
            LR_replacement_mode = "eq_ff_pass_through"

        # Common scalar command (before per-wheel damping)
        # No internal clipping - let the composer handle torque limits like baseline does
        if unified_tau_cmd is not None:
            # Unified controller replaces all torque components with a single coordinated command
            tau_common_unclipped = unified_tau_cmd
            # Set individual terms to 0 for telemetry consistency (they're not used)
            tau_pitch = 0.0
            tau_pitch_raw = tau_pitch_raw_orig if 'tau_pitch_raw_orig' in dir() else 0.0
            tau_pitch_scheduled = 0.0
            tau_pitch_clipped = 0.0
            tau_pitch_rate = 0.0
            tau_sagittal_velocity = 0.0
            tau_support_velocity = 0.0
            tau_position = 0.0
            tau_position_raw = 0.0
            tau_position_p = 0.0
            tau_position_integral = 0.0
            tau_cp = 0.0
            tau_com_vy = 0.0
            recenter_tau_clipped = 0.0
            hyst_tau_clipped = 0.0
            bias_tau_clipped = 0.0
            apc_tau_clipped = 0.0
        elif LR_enabled and (LR_kind.startswith("LR") or LR_kind.startswith("LRS")):
            # LR replacement with EQ/FF pass-through:
            # tau_common = tau_eq_ff_pass_through + LR_dynamic_feedback
            # where tau_eq_ff_pass_through preserves K1's equilibrium/feedforward
            # (tau_pitch + tau_position + tau_cp + tau_com_vy) and
            # LR_dynamic_feedback replaces the independent dynamic terms
            # (tau_pitch_rate + tau_sagittal_velocity + tau_support_velocity).
            tau_common_unclipped = LR_eq_ff_pass_through + LR_feedback_torque
            # Zero only the dynamic terms that LR replaces (for clean telemetry).
            # tau_pitch, tau_position, tau_cp, tau_com_vy are KEPT
            # — they carry the equilibrium/feedforward pass-through.
            tau_pitch_rate = 0.0
            tau_sagittal_velocity = 0.0
            tau_support_velocity = 0.0
        elif LP_enabled and LP_kind.startswith("LP"):
            # ---- LP family: Priority Sagittal Allocator ---- #
            # Pitch-first support-residual architecture.
            #
            # tau_common = tau_eq_ff_pass_through
            #            + tau_pitch_priority
            #            + tau_support_residual_allocated
            #
            # where:
            #   tau_eq_ff_pass_through = tau_pitch + tau_position + tau_cp + tau_com_vy
            #     (K1 equilibrium/feedforward — preserved unchanged)
            #   tau_pitch_priority = clamped pitch/pitch_rate damping
            #     (gets first access to dynamic torque authority)
            #   tau_support_residual_allocated = gated, slew-limited support centering
            #     (only uses remaining residual authority after pitch priority)
            #
            # Support is attenuated when pitch|pitch_rate is high, when near
            # torque saturation, or when support correction would worsen pitch.

            # --- Step 1: EQ/FF pass-through (preserved K1 baseline) ---
            LP_eq_ff_pass_through = float(tau_pitch + tau_position + tau_cp + tau_com_vy)

            # --- Step 2: Pitch priority torque ---
            LP_pitch_error_rad = float(pitch_x_rad)
            LP_pitch_rate_rad_s = float(pitch_rate_for_damping)
            LP_pitch_priority_raw = float(
                LP_gains["k_pitch_lp"] * LP_pitch_error_rad
                + LP_gains["k_pitch_rate_lp"] * LP_pitch_rate_rad_s
            )
            LP_pitch_priority_limit = float(LP_gains["pitch_priority_limit_nm"])
            LP_pitch_priority = max(-LP_pitch_priority_limit,
                                    min(LP_pitch_priority_limit, LP_pitch_priority_raw))

            # --- Step 3: Safety gates ---
            LP_pitch_abs_deg = float(abs(math.degrees(LP_pitch_error_rad)))
            LP_pitch_rate_abs_deg_s = float(abs(math.degrees(LP_pitch_rate_rad_s)))

            # pitch_abs_gate: 1.0 at or below safe_low, 0.0 at or above safe_high
            _psl = float(LP_gains["pitch_safe_low_deg"])
            _psh = float(LP_gains["pitch_safe_high_deg"])
            if LP_pitch_abs_deg <= _psl:
                LP_pitch_abs_gate = 1.0
            elif LP_pitch_abs_deg >= _psh:
                LP_pitch_abs_gate = 0.0
            else:
                LP_pitch_abs_gate = 1.0 - (LP_pitch_abs_deg - _psl) / (_psh - _psl)

            # pitch_rate_gate: 1.0 at or below safe_low, 0.0 at or above safe_high
            _rsl = float(LP_gains["rate_safe_low_deg_s"])
            _rsh = float(LP_gains["rate_safe_high_deg_s"])
            if LP_pitch_rate_abs_deg_s <= _rsl:
                LP_pitch_rate_gate = 1.0
            elif LP_pitch_rate_abs_deg_s >= _rsh:
                LP_pitch_rate_gate = 0.0
            else:
                LP_pitch_rate_gate = 1.0 - (LP_pitch_rate_abs_deg_s - _rsl) / (_rsh - _rsl)

            # saturation_gate: decreases as total torque approaches max_tau_wheel
            LP_pre_support_torque = float(abs(LP_eq_ff_pass_through + LP_pitch_priority))
            LP_saturation_gate = max(0.0, 1.0 - LP_pre_support_torque / max(self.max_tau_wheel * 0.85, 1e-6))
            LP_saturation_gate = min(1.0, max(0.0, LP_saturation_gate))

            # direction_gate: attenuate if support torque direction would worsen pitch
            LP_support_error_m = float(sagittal_position_error_m)
            LP_support_velocity_m_s = float(support_position_velocity_m_s)
            LP_support_raw = float(
                LP_gains["k_support_lp"] * LP_support_error_m
                + LP_gains["k_support_vel_lp"] * LP_support_velocity_m_s
            )
            LP_direction_gate = 1.0
            LP_support_direction_assists = False
            if LP_gains.get("direction_gate_enabled", False):
                # Support torque should move robot opposite to support error direction.
                # If support_error > 0 (robot drifted forward), support torque should
                # push backward (negative). If pitch > 0 (leaning forward), the
                # support torque may worsen pitch if it adds forward torque.
                # Simple heuristic: if support_raw and pitch_error have opposite signs
                # AND pitch_error is significant, the support torque is helping hold
                # pitch (robot is leaning one way, support pushes the other way to
                # restore). If they have the SAME sign, support may worsen pitch.
                if LP_pitch_abs_deg > 3.0:
                    _supp_sign = 1.0 if LP_support_raw > 0 else (-1.0 if LP_support_raw < 0 else 0.0)
                    _pitch_sign = 1.0 if LP_pitch_error_rad > 0 else (-1.0 if LP_pitch_error_rad < 0 else 0.0)
                    if _supp_sign != 0 and _pitch_sign != 0:
                        if _supp_sign == _pitch_sign:
                            # Same sign — support may worsen pitch, attenuate
                            LP_direction_gate = 0.3
                        else:
                            # Opposite signs — support helps pitch
                            LP_direction_gate = 1.0
                            LP_support_direction_assists = True

            # --- Step 4: Composite support gate ---
            LP_support_gate = float(
                LP_pitch_abs_gate * LP_pitch_rate_gate * LP_saturation_gate * LP_direction_gate
            )

            # --- Step 5: Support deadband ---
            LP_support_deadband_m = float(LP_gains.get("support_deadband_m", 0.02))
            if abs(LP_support_error_m) < LP_support_deadband_m:
                LP_support_gate = 0.0

            # --- Step 6: Residual authority and support limit ---
            LP_residual_authority_nm = float(max(0.0, self.max_tau_wheel * 0.85 - LP_pre_support_torque))
            LP_support_residual_fraction = float(LP_gains["support_residual_fraction"])
            LP_support_limit_nm = float(LP_residual_authority_nm * LP_support_residual_fraction)

            LP_support_allocated_raw = float(LP_support_raw * LP_support_gate)
            LP_support_allocated = max(-LP_support_limit_nm,
                                       min(LP_support_limit_nm, LP_support_allocated_raw))

            # --- Step 7: Slew limit on support allocated torque ---
            LP_support_slew_limit = float(LP_gains.get("support_slew_limit_nm_per_step", 0.3))
            LP_prev_support_allocated = float(getattr(self, '_lp_prev_support_allocated', 0.0))
            LP_support_slew_limited = float(
                LP_prev_support_allocated
                + max(-LP_support_slew_limit,
                      min(LP_support_slew_limit, LP_support_allocated - LP_prev_support_allocated))
            )
            self._lp_prev_support_allocated = LP_support_slew_limited

            # --- Step 8: LP3 settling counter (delayed support activation) ---
            LP_pitch_settled = True  # default for LP1/LP2
            LP_settle_counter = getattr(self, '_lp_pitch_settle_counter', 0)
            LP_settle_required = int(LP_gains.get("pitch_settle_steps_required", 0))
            LP_settle_threshold = float(LP_gains.get("pitch_settle_threshold_deg", 999.0))
            if LP_kind == "LP3_support_recenter_when_safe" and LP_settle_required > 0:
                if LP_pitch_abs_deg < LP_settle_threshold:
                    LP_settle_counter = LP_settle_counter + 1
                else:
                    LP_settle_counter = 0
                self._lp_pitch_settle_counter = LP_settle_counter
                LP_pitch_settled = LP_settle_counter >= LP_settle_required
                if not LP_pitch_settled:
                    LP_support_slew_limited = 0.0
                    LP_support_gate = 0.0

            # --- Step 9: Support suppression reason ---
            LP_support_suppressed_reason = "none"
            if LP_support_gate < 0.01:
                reasons = []
                if LP_pitch_abs_gate < 0.01:
                    reasons.append("pitch_abs")
                if LP_pitch_rate_gate < 0.01:
                    reasons.append("pitch_rate")
                if LP_saturation_gate < 0.01:
                    reasons.append("saturation")
                if LP_direction_gate < 0.3:
                    reasons.append("direction")
                if abs(LP_support_error_m) < LP_support_deadband_m:
                    reasons.append("deadband")
                if not LP_pitch_settled:
                    reasons.append("pitch_not_settled")
                LP_support_suppressed_reason = "+".join(reasons) if reasons else "gate_zero"

            # --- Step 10: Compose final torque ---
            LP_tau_total_preclip = float(
                LP_eq_ff_pass_through + LP_pitch_priority + LP_support_slew_limited
            )

            # Near-saturation flag
            LP_near_saturation = bool(
                abs(LP_tau_total_preclip) >= self.max_tau_wheel * 0.95
            )

            tau_common_unclipped = LP_tau_total_preclip
            # Zero the dynamic terms that LP replaces (for clean telemetry).
            # tau_pitch, tau_position, tau_cp, tau_com_vy are KEPT
            # — they carry the equilibrium/feedforward pass-through.
            tau_pitch_rate = 0.0
            tau_sagittal_velocity = 0.0
            tau_support_velocity = 0.0
        else:
            tau_common_unclipped = (
                tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
                tau_support_velocity + tau_position + tau_cp + tau_com_vy
            )
        # Add phase-aware recenter torque (decoupled from tau_position)
        tau_common_unclipped = tau_common_unclipped + recenter_tau_clipped
        # Add hysteresis recenter torque (F2_strategy)
        tau_common_unclipped = tau_common_unclipped + hyst_tau_clipped
        # Add bias cancellation torque (G1_strategy)
        tau_common_unclipped = tau_common_unclipped + bias_tau_clipped
        # Add Active Pitch Crossing torque (APC_strategy)
        tau_common_unclipped = tau_common_unclipped + apc_tau_clipped

        # ---- L family: Coordinated sagittal state feedback (Phase 3) ---- #
        # Adds a coordinated state-feedback term on top of K1's existing terms.
        # This synchronizes pitch, support, rate contributions to avoid the
        # torque-phase-conflict that feeds the 2.5 Hz WIP mode.
        L_enabled = self.authority_schedule.enable_coordinated_sagittal_feedback
        L_candidate_kind = self.authority_schedule.coordinated_feedback_kind
        L_state_vector = [
            float(pitch_x_rad),
            float(pitch_rate_for_damping),
            float(sagittal_position_error_m),
            float(support_position_velocity_m_s),
            float(wheel_vel_mean),
        ]
        L_gains = {}
        L_feedback_torque = 0.0
        if L_enabled and L_candidate_kind.startswith("L"):
            height_fb = float(com_z_m)
            if L_candidate_kind == "L1_low_freq":
                L_gains = _coordinated_feedback_gains_L1(height_fb)
                g = L_gains
                L_feedback_torque = (
                    g["k_pitch"] * L_state_vector[0]
                    + g["k_pitch_rate"] * L_state_vector[1]
                    + g["k_support"] * L_state_vector[2]
                    + g["k_support_vel"] * L_state_vector[3]
                )
            elif L_candidate_kind == "L2_phase_lead":
                L_gains = _coordinated_feedback_gains_L2(height_fb)
                g = L_gains
                # Pitch acceleration proxy: use pitch_rate derivative (pitch rate diff / dt)
                pitch_accel = (L_state_vector[1] - self._prev_pitch_rate_for_L) / max(self.dt, 1e-6)
                L_feedback_torque = (
                    g["k_pitch"] * L_state_vector[0]
                    + g["k_pitch_rate"] * L_state_vector[1]
                    + g["k_support"] * L_state_vector[2]
                    + g["k_support_vel"] * L_state_vector[3]
                    + g.get("k_lead", 0.0) * pitch_accel
                )
                self._prev_pitch_rate_for_L = L_state_vector[1]
            elif L_candidate_kind == "L3_pitch_ref_stabilization":
                L_gains = _coordinated_feedback_gains_L3(height_fb)
                g = L_gains
                # Pitch reference modulation based on support error
                pitch_ref_mod = g["pitch_ref_gain"] * L_state_vector[2]
                pitch_ref_mod = max(-g["pitch_ref_max"], min(g["pitch_ref_max"], pitch_ref_mod))
                # Convert ref mod (deg) to additional torque: use Kp_pitch * ref_mod(rad)
                pitch_ref_mod_rad = math.radians(pitch_ref_mod)
                L_feedback_torque = (
                    g["k_pitch"] * L_state_vector[0]
                    + g["k_pitch_rate"] * L_state_vector[1]
                    + g["k_support"] * L_state_vector[2]
                    + g["k_support_vel"] * L_state_vector[3]
                    + self.kp_pitch * pitch_ref_mod_rad
                )
            # Add coordinated feedback torque to common command
            tau_common_unclipped = tau_common_unclipped + L_feedback_torque

        # ---- N1 mild phase-lead damping diagnostic (Phase 5) ---- #
        if L_enabled and L_candidate_kind == "N1_mild_phase_lead":
            height_fb = float(com_z_m)
            h_norm = max(0.0, min(1.0, (height_fb - 0.30) / (0.48 - 0.30)))
            # Very mild phase-lead-compensated pitch rate damping
            # Parameters from profile for micro-sweep support
            sched = self.authority_schedule
            k_rate = sched.n1_rate_low + (sched.n1_rate_high - sched.n1_rate_low) * h_norm
            k_lead = sched.n1_lead_low + (sched.n1_lead_high - sched.n1_lead_low) * h_norm
            pitch_accel = (float(pitch_rate_for_damping) - getattr(self, '_prev_pitch_rate_for_N', 0.0)) / max(self.dt, 1e-6)
            self._prev_pitch_rate_for_N = float(pitch_rate_for_damping)
            N_feedback_torque = k_rate * float(pitch_rate_for_damping) + k_lead * pitch_accel
            tau_common_unclipped = tau_common_unclipped + N_feedback_torque
            L_feedback_torque = N_feedback_torque
            L_candidate_kind = "N1_mild_phase_lead"
            L_gains = {"k_rate": float(k_rate), "k_lead": float(k_lead)}

        # ---- M family: Body-yaw / wheel-yaw correct-actuator fix (Phase 4) ---- #
        # Adds body-yaw correction through differential wheel velocity with
        # support-aware gating. Does NOT fight mode-div divergence controller.
        M_enabled = self.authority_schedule.enable_body_yaw_wheel_stabilization
        M_wheel_yaw_torque = 0.0
        M_body_yaw_error = 0.0
        M_support_gate = 1.0
        M_yaw_correlation = 0.0
        if M_enabled:
            sched = self.authority_schedule
            # Yaw error: use yaw from orientation (negative pitch_yaw or available yaw signal)
            # In the sagittal controller, yaw error is approximated from body orientation
            yaw_error = float(0.0)  # Will be populated from external yaw signal
            # For now, this is a stub — the actual yaw error is computed in the
            # simulation harness and passed via commanded_height_ref_m or a new field.

        # ---- End of L/M/N candidate additions ---- #

        tau_total_before_final_clip = float(tau_common_unclipped + (tau_wheel_vel_left + tau_wheel_vel_right) / 2.0)
        tau_common = self.wheel_torque_sign * tau_common_unclipped

        # Per-wheel torque with common command + individual wheel damping
        tau_left = tau_common + tau_wheel_vel_left
        tau_right = tau_common + tau_wheel_vel_right
        tau_total_after_final_clip = float(0.5 * (tau_left + tau_right))

        # Compute final wheel torque margin
        final_wheel_torque_max = max(abs(float(tau_left)), abs(float(tau_right)))
        final_wheel_torque_margin = self.max_tau_wheel - final_wheel_torque_max

        # Build output vector
        tau = zeros_action()
        tau = tau.at[4].set(tau_left)
        tau = tau.at[9].set(tau_right)

        saturated = bool(
            abs(float(tau_common)) >= self.max_tau_wheel * 0.99
            or abs(float(tau_left)) >= self.max_tau_wheel * 0.99
            or abs(float(tau_right)) >= self.max_tau_wheel * 0.99
        )

        diagnostics = {
            "tau_pitch": float(tau_pitch),
            "tau_pitch_raw": float(tau_pitch_raw),
            "tau_pitch_scheduled": float(tau_pitch_scheduled),
            "tau_pitch_clipped": float(tau_pitch_clipped),
            "tau_pitch_rate": float(tau_pitch_rate),
            "tau_pitch_rate_raw_signal": float(effective_kd_pitch * pitch_rate_raw),
            "tau_pitch_rate_filtered_signal": float(effective_kd_pitch * pitch_rate_notched),
            "tau_wheel_velocity_left_raw_signal": float(-effective_k_wheel_velocity * wheel_left_raw),
            "tau_wheel_velocity_left_filtered_signal": float(-effective_k_wheel_velocity * wheel_left_notched),
            "tau_wheel_velocity_right_raw_signal": float(-effective_k_wheel_velocity * wheel_right_raw),
            "tau_wheel_velocity_right_filtered_signal": float(-effective_k_wheel_velocity * wheel_right_notched),
            "tau_cp": float(tau_cp),
            "tau_com_vy": float(tau_com_vy),
            "tau_sagittal_velocity": float(tau_sagittal_velocity),
            "tau_support_velocity": float(tau_support_velocity),
            "tau_wheel_velocity_left": float(tau_wheel_vel_left),
            "tau_wheel_velocity_right": float(tau_wheel_vel_right),
            "tau_position_raw": float(tau_position_raw),
            "tau_position_p": float(tau_position_p),
            "tau_position_i": float(tau_position_integral),
            "tau_position_integral": float(tau_position_integral),
            "tau_position_total": float(tau_position_raw),
            "position_integral_error": float(self.position_integral_error),
            "integral_active": bool(integral_active),
            "integral_gate_reason": integral_gate_reason,
            "integral_saturation_flag": bool(integral_saturation_flag),
            "pitch_error_x_rad": float(pitch_x_rad),
            "wheel_velocity_mean_rad_s": float(wheel_vel_mean),
            "com_z_m": float(com_z_m),
            "tau_position_before_clip": float(tau_position_before_clip),
            "tau_position": float(tau_position),
            "tau_position_clipped": float(tau_position),
            "tau_position_lower_bound": float(tau_position_lower_bound),
            "tau_position_upper_bound": float(tau_position_upper_bound),
            "tau_position_total_bound_clipped": bool(tau_position_total_bound_clipped),
            "position_authority_mode": position_authority_mode,
            "position_authority_reason": position_authority_reason,
            "max_position_tau": float(self.max_position_tau),
            "sagittal_schedule_profile": self.authority_schedule.profile_name,
            "high_height_schedule_active": bool(schedule_active),
            "effective_max_position_tau": float(effective_max_position_tau),
            "effective_pitch_scale": float(effective_pitch_scale),
            "effective_pitch_tau_cap": "none" if effective_pitch_tau_cap is None else float(effective_pitch_tau_cap),
            "effective_velocity_damping_scale": float(effective_velocity_damping_scale),
            "effective_support_velocity_scale": float(effective_support_velocity_scale),
            "effective_support_velocity_gain": float(effective_support_velocity_gain),
            "tau_pitch_to_position_ratio": float(abs(tau_pitch) / max(abs(tau_position), 1e-9)),
            "height_variant_name": height_variant_name or "none",
            "tau_position_saturation_flag": bool(tau_position_saturated),
            "tau_position_saturation_reason": tau_position_saturation_reason,
            "tau_balance_before_position": tau_balance_before_position_log,
            "tau_position_budget_available": tau_position_budget_available,
            "tau_position_budget_allowed": tau_position_budget_allowed,
            "tau_position_budget_cap": tau_position_budget_cap,
            "pitch_reserve_tau": pitch_reserve_tau_log,
            "tau_pitch_reserve_applied": float(tau_pitch_reserve_applied),
            "enable_torque_budget_aware_position": self.enable_torque_budget_aware_position,
            "tau_common_unclipped": float(tau_common_unclipped),
            "tau_common_clipped": float(tau_common),
            "tau_left": float(tau_left),
            "tau_right": float(tau_right),
            "tau_total_before_final_clip": tau_total_before_final_clip,
            "tau_total_after_final_clip": tau_total_after_final_clip,
            "tau_total_unclipped": tau_total_before_final_clip,
            "tau_total_clipped": tau_total_after_final_clip,
            "final_wheel_torque_margin": final_wheel_torque_margin,
            "saturated": saturated,
            "wheel_vel_mean_rad_s": float(wheel_vel_mean),
            "sagittal_position_error_m": float(sagittal_position_error_m),
            "sagittal_velocity_m_s": float(sagittal_velocity_m_s),
            "support_position_velocity_m_s": float(support_position_velocity_m_s),
            "pitch_x_rad": float(pitch_x_rad),
            "pitch_rate_x_rad_s": float(pitch_rate_x_rad_s),
            "wheel_torque_sign": self.wheel_torque_sign,
            "k_support_velocity": self.k_support_velocity,
            # ---- Continuous k_position scheduling telemetry ----
            "schedule_height_source": schedule_height_source,
            "schedule_height_reference_m": float(schedule_height_ref),
            "filtered_current_com_z_m": float(self._filtered_com_z),
            "effective_k_position": float(effective_k_position),
            "k_position_schedule_u": float(u_for_telemetry),
            "k_position_schedule_smoothstep": float(schedule_smoothstep),
            "low_height_sagittal_schedule_active": bool(low_height_sagittal_schedule_active),
            "k_position_nominal": float(self.authority_schedule.k_position_nominal),
            "k_position_low_max": float(self.authority_schedule.k_position_low_max),
            "k_position_z_low": float(self.authority_schedule.k_position_z_low),
            "k_position_z_high": float(self.authority_schedule.k_position_z_high),
            "effective_k_velocity": float(effective_k_velocity),
            # ---- High-height k_wheel_velocity scheduling telemetry (Step E extreme height fix) ----
            "effective_k_wheel_velocity": float(effective_k_wheel_velocity),
            "high_height_wheel_damping_active": bool(high_height_wheel_damping_active),
            "k_wheel_velocity_nominal": float(self.authority_schedule.k_wheel_velocity_nominal),
            "k_wheel_velocity_high_max": float(self.authority_schedule.k_wheel_velocity_high_max),
            "k_wheel_velocity_z_low": float(self.authority_schedule.k_wheel_velocity_z_low),
            "k_wheel_velocity_z_high": float(self.authority_schedule.k_wheel_velocity_z_high),
            # ---- Tall-height kd_pitch scheduling telemetry (J candidate) ----
            "effective_kd_pitch": float(effective_kd_pitch),
            "high_height_kd_pitch_active": bool(high_height_kd_pitch_active),
            "kd_pitch_nominal": float(self.authority_schedule.kd_pitch_nominal),
            "kd_pitch_high_max": float(self.authority_schedule.kd_pitch_high_max),
            "kd_pitch_z_low": float(self.authority_schedule.kd_pitch_z_low),
            "kd_pitch_z_high": float(self.authority_schedule.kd_pitch_z_high),
            # ---- Notch filter telemetry (K candidate — 2.5 Hz WIP notch) ----
            "wip_notch_enabled": bool(notch_enabled),
            "wip_notch_target_signal": str(notch_target),
            "wip_notch_center_hz": float(notch_center_hz),
            "wip_notch_q": float(notch_q),
            "wip_notch_filter_type": str(self.authority_schedule.wip_notch_filter_type),
            "wip_lowpass_cutoff_hz": float(self.authority_schedule.wip_lowpass_cutoff_hz),
            "wip_notch_fs_hz": float(fs_hz),
            "wip_notch_height_gate": float(notch_height_gate),
            "wip_notch_filter_blend": float(notch_blend),
            "wip_notch_filter_valid": bool(notch_filter_valid),
            "pitch_rate_raw": float(pitch_rate_raw),
            "pitch_rate_notched": float(pitch_rate_notched),
            "pitch_rate_effective": float(pitch_rate_effective),
            "wheel_velocity_left_raw": float(wheel_left_raw),
            "wheel_velocity_left_notched": float(wheel_left_notched),
            "wheel_velocity_left_effective": float(wheel_left_for_damping),
            "wheel_velocity_right_raw": float(wheel_right_raw),
            "wheel_velocity_right_notched": float(wheel_right_notched),
            "wheel_velocity_right_effective": float(wheel_right_for_damping),
            "support_velocity_raw": float(support_vel_raw),
            "support_velocity_notched": float(support_vel_notched),
            "support_velocity_effective": float(support_vel_for_damping),
            "notch_signal_delta_pr": float(notch_signal_delta_pr),
            "notch_signal_delta_wl": float(notch_signal_delta_wl),
            "notch_signal_delta_wr": float(notch_signal_delta_wr),
            # ---- L family: Coordinated sagittal state feedback telemetry ----
            "L_enabled": bool(L_enabled),
            "L_candidate_kind": str(L_candidate_kind),
            "L_state_pitch_rad": float(L_state_vector[0]) if isinstance(L_state_vector, list) and len(L_state_vector) > 0 else 0.0,
            "L_state_pitch_rate_rad_s": float(L_state_vector[1]) if isinstance(L_state_vector, list) and len(L_state_vector) > 1 else 0.0,
            "L_state_support_error_m": float(L_state_vector[2]) if isinstance(L_state_vector, list) and len(L_state_vector) > 2 else 0.0,
            "L_state_wheel_vel_rad_s": float(L_state_vector[4]) if isinstance(L_state_vector, list) and len(L_state_vector) > 4 else 0.0,
            "L_feedback_torque_nm": float(L_feedback_torque),
            "L_gains_kind": str(L_gains.get("kind", "none")) if isinstance(L_gains, dict) else "none",
            # ---- LR family: Replacement coordinated feedback telemetry ---- #
            "LR_enabled": bool(LR_enabled),
            "LR_candidate_kind": str(LR_kind) if LR_enabled else "none",
            "LR_state_pitch_rad": float(LR_state_vector[0]) if isinstance(LR_state_vector, list) and len(LR_state_vector) > 0 else 0.0,
            "LR_state_pitch_rate_rad_s": float(LR_state_vector[1]) if isinstance(LR_state_vector, list) and len(LR_state_vector) > 1 else 0.0,
            "LR_state_support_error_m": float(LR_state_vector[2]) if isinstance(LR_state_vector, list) and len(LR_state_vector) > 2 else 0.0,
            "LR_state_support_velocity_m_s": float(LR_state_vector[3]) if isinstance(LR_state_vector, list) and len(LR_state_vector) > 3 else 0.0,
            "LR_state_wheel_vel_rad_s": float(LR_state_vector[4]) if isinstance(LR_state_vector, list) and len(LR_state_vector) > 4 else 0.0,
            "LR_feedback_torque_nm": float(LR_feedback_torque),
            "LR_dynamic_feedback_torque_nm": float(LR_feedback_torque),
            "LR_eq_ff_pass_through_nm": float(LR_eq_ff_pass_through),
            "LR_total_command_preclip_nm": float(LR_eq_ff_pass_through + LR_feedback_torque)
                if LR_enabled and (LR_kind.startswith("LR") or LR_kind.startswith("LRS")) else 0.0,
            "LR_total_command_postclip_nm": float(tau_common),
            "LR_k1_existing_estimate_nm": float(LR_k1_existing_estimate),
            "LR_removed_dynamic_terms_estimate_nm": float(LR_removed_dynamic_terms_estimate),
            "LR_eq_ff_estimate_nm": float(LR_eq_ff_pass_through),
            "LR_replacement_mode": str(LR_replacement_mode),
            "LR_gains_kind": str(LR_gains.get("kind", "none")) if isinstance(LR_gains, dict) else "none",
            # ---- LRS family: Component-wise torque decomposition ---- #
            "LRS_tau_pitch_component_nm": float(LRS_tau_pitch_component),
            "LRS_tau_pitch_rate_component_nm": float(LRS_tau_pitch_rate_component),
            "LRS_tau_support_component_nm": float(LRS_tau_support_component),
            "LRS_tau_support_vel_component_nm": float(LRS_tau_support_vel_component),
            # ---- LP family: Priority Sagittal Allocator telemetry ---- #
            "LP_enabled": bool(LP_enabled),
            "LP_candidate_kind": str(LP_kind) if LP_enabled else "none",
            "LP_allocator_mode": str(LP_kind) if LP_enabled else "none",
            "LP_tau_eq_ff_nm": float(LP_eq_ff_pass_through) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_tau_pitch_priority_raw_nm": float(LP_pitch_priority_raw) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_tau_pitch_priority_nm": float(LP_pitch_priority) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_tau_support_raw_nm": float(LP_support_raw) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_tau_support_allocated_nm": float(LP_support_allocated) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_tau_support_slew_limited_nm": float(LP_support_slew_limited) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_tau_total_preclip_nm": float(LP_tau_total_preclip) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_tau_total_postclip_nm": float(tau_common) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_pitch_abs_gate": float(LP_pitch_abs_gate) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_pitch_rate_gate": float(LP_pitch_rate_gate) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_saturation_gate": float(LP_saturation_gate) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_direction_gate": float(LP_direction_gate) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_support_gate": float(LP_support_gate) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_residual_authority_nm": float(LP_residual_authority_nm) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_support_limit_nm": float(LP_support_limit_nm) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_support_residual_fraction": float(LP_support_residual_fraction) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_pitch_error_rad": float(LP_pitch_error_rad) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_pitch_rate_effective_rad_s": float(LP_pitch_rate_rad_s) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_support_error_m": float(LP_support_error_m) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_support_velocity_m_s": float(LP_support_velocity_m_s) if LP_enabled and LP_kind.startswith("LP") else 0.0,
            "LP_support_suppressed_reason": str(LP_support_suppressed_reason) if LP_enabled and LP_kind.startswith("LP") else "lp_disabled",
            "LP_near_saturation": bool(LP_near_saturation) if LP_enabled and LP_kind.startswith("LP") else False,
            "LP_support_direction_assists_pitch_error": bool(LP_support_direction_assists) if LP_enabled and LP_kind.startswith("LP") else False,
            "LP_gains_kind": str(LP_gains.get("kind", "none")) if isinstance(LP_gains, dict) else "none",
            # ---- M family: Body-yaw/wheel-yaw telemetry ----
            "M_enabled": bool(M_enabled),
            "M_wheel_yaw_torque_nm": float(M_wheel_yaw_torque),
            "M_body_yaw_error_rad": float(M_body_yaw_error),
            "M_support_gate": float(M_support_gate),
            # ---- Pitch-aware position scaling telemetry ----
            "pitch_aware_position_scaling_enabled": bool(self.authority_schedule.enable_pitch_aware_position_scaling),
            "pitch_aware_position_scale": float(pitch_aware_position_scale),
            "pitch_aware_active": bool(pitch_aware_active),
            "pitch_soft_start": float(self.authority_schedule.pitch_soft_start),
            "pitch_hard_limit": float(self.authority_schedule.pitch_hard_limit),
            "min_pitch_scale": float(self.authority_schedule.min_pitch_scale),
            "tau_position_before_pitch_scale": float(tau_position_before_pitch_scale),
            "tau_position_after_pitch_scale": float(tau_position_before_clip),
            # ---- T6F Architecture Fix telemetry ----
            "arch_fix_enabled": bool(self.authority_schedule.arch_fix_enabled),
            "arch_fix_type": str(self.authority_schedule.arch_fix_type),
            "arch_fix_active": bool(arch_fix_active),
            "arch_fix_reason": str(arch_fix_reason),
            "arch_fix_height_gate_pass": bool(arch_fix_height_gate_pass),
            "arch_fix_band_gate_pass": bool(arch_fix_band_gate_pass),
            "arch_fix_safety_gate_pass": bool(arch_fix_safety_gate_pass),
            "arch_fix_recenter_gate_pass": bool(arch_fix_recenter_gate_pass),
            "effective_max_position_tau_before_arch_fix": float(effective_max_position_tau_before_arch_fix),
            "effective_max_position_tau_after_arch_fix": float(effective_max_position_tau),
            "arch_fix_requested_cap": float(arch_fix_requested_cap),
            "arch_fix_upstream_clip_active": bool(tau_position_saturated),
            "arch_fix_tau_position_before_clip": float(tau_position_before_clip),
            "arch_fix_tau_position_after_upstream_clip": float(tau_position),
            "arch_fix_torque_transmitted_above_4nm": bool(abs(tau_position) > 4.0),
            "arch_fix_height_threshold_m": float(self.authority_schedule.arch_fix_height_threshold_m),
            "arch_fix_hard_max_position_tau": float(self.authority_schedule.arch_fix_hard_max_position_tau),
            "arch_fix_emergency_max_position_tau": float(self.authority_schedule.arch_fix_emergency_max_position_tau),
            # ---- T6F Sign Fix telemetry ----
            "sign_fix_enabled": bool(self.authority_schedule.sign_fix_enabled),
            "sign_fix_active": bool(sign_fix_active),
            "sign_fix_damping_disabled": bool(sign_fix_damping_disabled),
            "sign_fix_damping_helped": bool(sign_fix_damping_helped),
            "sign_fix_damping_fought": bool(sign_fix_damping_fought),
            "sign_fix_damping_original_nm": float(sign_fix_damping_original_nm),
            "sign_fix_damping_after_nm": float(sign_fix_damping_after_nm),
            "sign_fix_pitch_suppressed": bool(sign_fix_pitch_suppressed),
            "sign_fix_pitch_original_nm": float(sign_fix_pitch_original_nm),
            "sign_fix_pitch_after_nm": float(sign_fix_pitch_after_nm),
            "sign_fix_tau_position_sign": int(1 if tau_position > 0 else (-1 if tau_position < 0 else 0)),
            "sign_fix_damping_sign": int(1 if sign_fix_damping_original_nm > 0 else (-1 if sign_fix_damping_original_nm < 0 else 0)),
            "sign_fix_pitch_sign": int(1 if sign_fix_pitch_original_nm > 0 else (-1 if sign_fix_pitch_original_nm < 0 else 0)),
            "sign_fix_final_tau_sign": int(1 if tau_left > 0 else (-1 if tau_left < 0 else 0)),
            "sign_fix_drift_sign": int(1 if sagittal_position_error_m > 0 else (-1 if sagittal_position_error_m < 0 else 0)),
            "sign_fix_final_sign_correct": bool(
                (tau_left > 0 and sagittal_position_error_m < 0) or
                (tau_left < 0 and sagittal_position_error_m > 0) or
                (abs(tau_left) < 0.1 and abs(sagittal_position_error_m) < 0.01)
            ),
            "sign_fix_reason": str(
                "sign_fix_active" if sign_fix_active else
                ("arch_fix_inactive" if not arch_fix_active else "sign_fix_disabled")
            ),
            # ---- T6H Soft Blend Arch Fix telemetry ----
            "t6h_soft_pitch_blend_active": bool(t6h_soft_pitch_blend_active),
            "t6h_pitch_blend_factor": float(t6h_pitch_blend_factor),
            "t6h_pitch_safety_active": bool(t6h_pitch_safety_active),
            "t6h_soft_damping_blend_active": bool(t6h_soft_damping_blend_active),
            "t6h_damping_blend_factor": float(t6h_damping_blend_factor),
            "t6h_wheel_velocity_safety_active": bool(t6h_wheel_velocity_safety_active),
            # ---- T6I Phase-Aware Release telemetry ----
            "t6i_error_converging": bool(t6i_error_converging),
            "t6i_error_trend": float(t6i_error_trend),
            "t6i_target_cap": float(t6i_target_cap),
            "t6i_current_cap": float(t6i_current_cap),
            "t6i_cap_delta_this_step": float(t6i_cap_delta_this_step),
            "t6i_cap_change_rate_limited": bool(t6i_cap_change_rate_limited),
            "t6i_release_reason": str(t6i_release_reason),
            # ---- T6J Centering Bias Trim telemetry ----
            "t6j_bias_trim_enabled": bool(t6j_bias_trim_enabled),
            "t6j_bias_trim_active": bool(t6j_bias_trim_active),
            "t6j_bias_mean_error_m": float(t6j_bias_mean_error_m),
            "t6j_bias_window_steps": int(t6j_bias_window_steps),
            "t6j_bias_trim_tau_nm": float(t6j_bias_trim_tau_nm),
            "t6j_bias_trim_target_tau_nm": float(t6j_bias_trim_target_tau_nm),
            "t6j_bias_trim_rate_limited": bool(t6j_bias_trim_rate_limited),
            "t6j_bias_positive_duration_steps": int(self._t6j_bias_positive_duration_steps),
            "t6j_bias_negative_duration_steps": int(self._t6j_bias_negative_duration_steps),
            "t6j_bias_safety_gate_pass": bool(t6j_bias_safety_gate_pass),
            "t6j_bias_block_reason": str(t6j_bias_block_reason),
            "t6j_bias_applied_to_final_tau": float(t6j_bias_applied_to_final_tau),
            "t6j_bias_expected_direction_correct": bool(t6j_bias_expected_direction_correct),
            # ---- Adaptive Centering Bias Trim telemetry ----
            "adaptive_bias_trim_enabled": bool(adaptive_bias_trim_enabled),
            "adaptive_bias_trim_active": bool(adaptive_bias_trim_active),
            "adaptive_bias_mean_error_m": float(adaptive_bias_mean_error_m),
            "adaptive_bias_fast_mean_error_m": float(adaptive_bias_fast_mean_error_m),
            "adaptive_bias_effective_error_m": float(adaptive_bias_effective_error_m),
            "adaptive_bias_target_tau_nm": float(adaptive_bias_target_tau_nm),
            "adaptive_bias_tau_nm": float(adaptive_bias_tau_nm),
            "adaptive_bias_max_tau_current_nm": float(adaptive_bias_max_tau_current_nm),
            "adaptive_bias_height_scale": float(adaptive_bias_height_scale),
            "adaptive_bias_rate_used_nm_per_step": float(adaptive_bias_rate_used_nm_per_step),
            "adaptive_bias_zero_crossing_count": int(adaptive_bias_zero_crossing_count),
            "adaptive_bias_zero_crossing_guard_active": bool(adaptive_bias_zero_crossing_guard_active),
            "adaptive_bias_near_zero_relief_active": bool(adaptive_bias_near_zero_relief_active),
            "adaptive_bias_sign_reversal_blocked": bool(adaptive_bias_sign_reversal_blocked),
            "adaptive_bias_safety_gate_pass": bool(adaptive_bias_safety_gate_pass),
            "adaptive_bias_block_reason": str(adaptive_bias_block_reason),
            "adaptive_bias_expected_direction_correct": bool(adaptive_bias_expected_direction_correct),
            "adaptive_bias_positive_area": float(adaptive_bias_positive_area),
            "adaptive_bias_negative_area": float(adaptive_bias_negative_area),
            "adaptive_bias_symmetry_ratio": float(adaptive_bias_symmetry_ratio),
            "adaptive_bias_hip_yaw_gate_pass": bool(adaptive_bias_hip_yaw_gate_pass),
            "adaptive_bias_hip_yaw_abs_max": float(adaptive_bias_hip_yaw_abs_max),
            # ---- Zero-Crossing Support Recenter (ZC) telemetry ----
            "zc_state": str(zc_state),
            "zc_state_id": int(zc_state_id),
            "zc_active": bool(zc_active),
            "zc_direction": int(zc_direction),
            "zc_enter_event": int(self._zc_enter_event),
            "zc_exit_event": int(self._zc_exit_event),
            "zc_hold_steps": int(self._zc_hold_steps),
            "zc_tau_nm": float(zc_tau_nm),
            "zc_target_tau_nm": float(zc_target_tau_nm),
            "zc_crossed_zero": bool(zc_crossed_zero),
            "zc_cross_target_reached": bool(zc_cross_target_reached),
            "zc_safety_gate_pass": bool(zc_safety_gate_pass),
            "zc_block_reason": str(zc_block_reason),
            "zc_episode_id": int(self._zc_episode_id),
            "zc_episode_start_error": float(self._zc_episode_start_error),
            "zc_episode_min_error": float(self._zc_episode_min_error),
            "zc_episode_max_error": float(self._zc_episode_max_error),
            "zc_expected_direction_correct": bool(zc_expected_direction_correct),
            # ---- Early Zero-Crossing Recenter telemetry (EZC) ----
            "ezc_state_id": int(ezc_state_id),
            "ezc_active": bool(ezc_active),
            "ezc_direction": int(ezc_direction),
            "ezc_enter_event": int(self._ezc_enter_event),
            "ezc_zero_cross_exit_event": int(self._ezc_zero_cross_exit_event),
            "ezc_safety_exit_event": int(self._ezc_safety_exit_event),
            "ezc_hold_steps": int(self._ezc_hold_steps),
            "ezc_tau_nm": float(ezc_tau_nm),
            "ezc_target_tau_nm": float(ezc_target_tau_nm),
            "ezc_crossed_zero": bool(ezc_crossed_zero),
            "ezc_zero_dwell_steps": int(self._ezc_zero_dwell_steps),
            "ezc_safety_gate_pass": bool(ezc_safety_gate_pass),
            "ezc_block_reason": str(ezc_block_reason),
            "ezc_episode_id": int(self._ezc_episode_id),
            "ezc_episode_start_error": float(self._ezc_episode_start_error),
            "ezc_episode_min_error": float(self._ezc_episode_min_error),
            "ezc_episode_max_error": float(self._ezc_episode_max_error),
            "ezc_expected_direction_correct": bool(ezc_expected_direction_correct),
            "ezc_exit_reason": str(self._ezc_exit_reason),
            # V2 Anti-rebound telemetry
            "ezc_antirebound_steps": int(self._ezc_antirebound_steps),
            "ezc_antirebound_tau_start": float(self._ezc_antirebound_tau_start),
            # ---- Pitch Bias DC Compensation telemetry (Phase 7) ----
            "pitch_bias_comp_enabled": bool(self.authority_schedule.pitch_bias_comp_enabled),
            "pitch_bias_comp_active": bool(pitch_bias_gate_pass),
            "pitch_bias_estimation_active": bool(pitch_bias_estimation_active),
            "pitch_bias_estimate_nm": float(self._pitch_bias_estimate_nm),
            "pitch_bias_comp_tau_nm": float(pitch_bias_comp_tau),
            "pitch_bias_samples": int(self._pitch_bias_samples),
            "pitch_bias_block_reason": str(pitch_bias_block_reason),
            "tau_pitch_before_bias_comp": float(tau_pitch_before_bias_comp),
            "tau_pitch_after_bias_comp": float(tau_pitch_after_bias_comp),
            # ---- Phase-aware recenter telemetry (F1_strategy) ----
            "phase_recenter_enabled": bool(phase_recenter_enabled),
            "phase_recenter_active": bool(recenter_gate_safe and not recenter_deadband_active),
            "phase_recenter_gate_safe": bool(recenter_gate_safe),
            "phase_recenter_signed_error_m": float(signed_error),
            "phase_recenter_raw_tau": float(raw_recenter_tau),
            "phase_recenter_tau": float(final_recenter_tau),
            "phase_recenter_tau_clipped": float(recenter_tau_clipped),
            "phase_recenter_smooth_alpha": float(alpha),
            "phase_recenter_gate_reason": str(gate_reason),
            "phase_recenter_pitch_safe": bool(pitch_safe),
            "phase_recenter_pitch_danger": bool(pitch_danger),
            "phase_recenter_hip_yaw_safe": bool(hip_yaw_abs_max < self.authority_schedule.recenter_hip_yaw_safe_threshold_rad),
            "phase_recenter_contact_safe": bool(contact_valid),
            "phase_recenter_height_safe": bool(com_z_safe),
            "phase_recenter_deadband_active": bool(recenter_deadband_active),
            # ---- Hysteresis recenter telemetry (F2_strategy) ----
            "hysteresis_recenter_enabled": bool(hysteresis_recenter_enabled),
            "hysteresis_recenter_state": str(hysteresis_state),
            "hysteresis_recenter_state_id": 0 if hysteresis_state == "NEUTRAL" else (1 if hysteresis_state == "RECENTER_FROM_POSITIVE" else 2),
            "hysteresis_recenter_outer_enter_m": float(outer_enter),
            "hysteresis_recenter_exit_target_m": float(exit_target),
            "hysteresis_recenter_signed_error_m": float(signed_error),
            "hysteresis_recenter_target_error_m": float(target_error) if hysteresis_state != "NEUTRAL" else 0.0,
            "hysteresis_recenter_raw_tau": float(hyst_raw_tau),
            "hysteresis_recenter_tau": float(hyst_final_tau),
            "hysteresis_recenter_tau_clipped": float(hyst_tau_clipped),
            "hysteresis_recenter_active": bool(hysteresis_active),
            "hysteresis_recenter_state_entry_count": int(self._hysteresis_state_entry_count),
            "hysteresis_recenter_state_exit_count": int(self._hysteresis_state_exit_count),
            "hysteresis_recenter_safety_override": bool(hysteresis_safety_override),
            "hysteresis_recenter_gate_reason": str(hyst_gate_reason),
            "hysteresis_recenter_pitch_safe": bool(hyst_pitch_safe),
            "hysteresis_recenter_pitch_danger": bool(hyst_pitch_danger),
            "hysteresis_recenter_hip_yaw_safe": bool(hyst_hip_yaw_safe),
            "hysteresis_recenter_contact_safe": bool(contact_valid),
            "hysteresis_recenter_height_safe": bool(hyst_com_z_safe),
            "hysteresis_recenter_max_tau": float(self.authority_schedule.hysteresis_max_recenter_tau),
            "hysteresis_recenter_k_recenter": float(self.authority_schedule.hysteresis_k_recenter),
            # ---- Bias cancellation telemetry (G1_strategy) ----
            "bias_cancel_enabled": bool(bias_cancel_enabled),
            "bias_cancel_active": bool(bias_cancel_active),
            "bias_cancel_signed_error_m": float(signed_error),
            "bias_cancel_estimate_m": float(self._bias_cancel_estimate),
            "bias_cancel_raw_tau": float(bias_raw_tau),
            "bias_cancel_tau": float(bias_final_tau),
            "bias_cancel_tau_clipped": float(bias_tau_clipped),
            "bias_cancel_gate_reason": str(bias_gate_reason),
            "bias_cancel_contact_safe": bool(bias_contact_safe),
            "bias_cancel_height_safe": bool(bias_height_safe),
            "bias_cancel_roll_safe": bool(bias_roll_safe),
            "bias_cancel_k": float(self.authority_schedule.bias_cancel_k),
            "bias_cancel_max_tau": float(self.authority_schedule.bias_cancel_max_tau),
            "bias_cancel_filter_alpha": float(self.authority_schedule.bias_cancel_filter_alpha),
            "bias_cancel_deadband_m": float(self.authority_schedule.bias_cancel_deadband_m),
            # ---- Active Pitch Crossing telemetry (APC_strategy) ----
            "active_pitch_crossing_enabled": bool(apc_enabled),
            "active_pitch_crossing_state": str(apc_state),
            # State ID mapping for both bang-bang and proportional modes
            "active_pitch_crossing_state_id": (
                0 if apc_state == "NEUTRAL" else
                (1 if apc_state in ("CROSS_FROM_POSITIVE", "SOFT_RECENTER") else 2)
            ),
            "active_pitch_crossing_active": bool(apc_active),
            "active_pitch_crossing_signed_error_m": float(signed_error),
            "active_pitch_crossing_pitch_x": float(pitch_x_rad),
            "active_pitch_crossing_pitch_rate": float(pitch_rate_x_rad_s),
            "active_pitch_crossing_raw_tau": float(apc_raw_tau),
            "active_pitch_crossing_tau": float(apc_final_tau),
            "active_pitch_crossing_tau_clipped": float(apc_tau_clipped),
            # Target direction: proportional mode uses SOFT_RECENTER for both directions
            # APCR1i uses RECENTER_FROM_POSITIVE/NEGATIVE/HOLD_THROUGH_ZERO states
            "active_pitch_crossing_target_direction": (
                "negative" if apc_state in ("CROSS_FROM_POSITIVE", "RECENTER_FROM_POSITIVE") else
                ("positive" if apc_state in ("CROSS_FROM_NEGATIVE", "RECENTER_FROM_NEGATIVE") else "none")
            ),
            "active_pitch_crossing_inner_exit_m": float(apc_inner_exit),
            "active_pitch_crossing_outer_enter_m": float(apc_outer_enter),
            "active_pitch_crossing_state_entry_count": int(self._apc_state_entry_count),
            "active_pitch_crossing_state_exit_count": int(self._apc_state_exit_count),
            "active_pitch_crossing_safety_override": bool(apc_safety_override),
            "active_pitch_crossing_gate_reason": str(apc_gate_reason),
            "active_pitch_crossing_contact_safe": bool(apc_contact_safe),
            "active_pitch_crossing_height_safe": bool(apc_height_safe),
            "active_pitch_crossing_roll_safe": bool(apc_roll_safe),
            "active_pitch_crossing_pitch_safe": bool(apc_pitch_safe),
            "active_pitch_crossing_pitch_danger": bool(apc_pitch_danger),
            # APCR recovery gate mode telemetry
            "active_pitch_crossing_recovery_gate_mode": bool(apc_recovery_gate_mode),
            "active_pitch_crossing_hard_safety_gate": bool(apc_hard_safety_gate),
            "active_pitch_crossing_recovery_gate": bool(apc_recovery_gate),
            "active_pitch_crossing_max_tau": float(self.authority_schedule.apc_max_cross_tau),
            "active_pitch_crossing_smooth_alpha": float(apc_alpha),
            # ---- APCR1d proportional soft band telemetry ----
            "active_pitch_crossing_torque_mode": "proportional_soft_band" if apc_proportional_mode else "bang_bang",
            "active_pitch_crossing_soft_enter_m": float(apc_soft_enter),
            "active_pitch_crossing_inner_deadband_m": float(apc_inner_exit),  # Reuse as deadband
            "active_pitch_crossing_full_torque_error_m": float(apc_outer_enter),  # Reuse as threshold
            "active_pitch_crossing_desired_band_m": float(apc_soft_enter),  # Target band
            "active_pitch_crossing_abs_error_m": float(abs(signed_error)),
            "active_pitch_crossing_error_rate_mps": float(apc_error_rate),
            "active_pitch_crossing_error_moving_toward_zero": bool(signed_error * apc_error_rate < 0.0),
            "active_pitch_crossing_proportional_scale": float(proportional_scale),
            "active_pitch_crossing_velocity_decay_enabled": bool(self.authority_schedule.apc_velocity_decay_enabled),
            "active_pitch_crossing_velocity_decay_factor": float(self.authority_schedule.apc_velocity_decay_factor),
            "active_pitch_crossing_velocity_decay_active": bool(velocity_decay_active),
            # ---- APCR1e adaptive authority telemetry ----
            "active_pitch_crossing_adaptive_enabled": bool(adaptive_enabled),
            "active_pitch_crossing_base_tau": float(self.authority_schedule.apc_adaptive_base_tau) if adaptive_enabled else 0.0,
            "active_pitch_crossing_adaptive_max_tau": float(adaptive_max_tau) if adaptive_enabled else float(self.authority_schedule.apc_max_cross_tau),
            "active_pitch_crossing_boost_tau": float(boost_tau) if adaptive_enabled else 0.0,
            "active_pitch_crossing_boost_reason": str(boost_reason) if adaptive_enabled else "none",
            "active_pitch_crossing_moving_away_from_zero": bool(moving_away_from_zero) if adaptive_enabled else False,
            "active_pitch_crossing_moving_toward_zero": bool(moving_toward_zero) if adaptive_enabled else False,
            "active_pitch_crossing_no_improvement_count": int(self._apc_adaptive_no_improvement_count) if adaptive_enabled else 0,
            "active_pitch_crossing_startup_boost_active": bool(startup_boost_active) if adaptive_enabled else False,
            "active_pitch_crossing_velocity_decay_disabled_reason": str(velocity_decay_disabled_reason) if adaptive_enabled else "none",
            # ---- APCR1f fast response with phase brake telemetry ----
            "active_pitch_crossing_fast_response_enabled": bool(fast_response_enabled) if 'fast_response_enabled' in dir() else False,
            "active_pitch_crossing_phase_brake_enabled": bool(phase_brake_enabled) if 'phase_brake_enabled' in dir() else False,
            "active_pitch_crossing_boost_rate": float(self.authority_schedule.apc_boost_rate_per_step) if fast_response_enabled else 0.0,
            "active_pitch_crossing_decay_rate": float(self.authority_schedule.apc_decay_rate_per_step) if fast_response_enabled else 0.0,
            "active_pitch_crossing_phase_brake_active": bool(self._apc_fast_response_phase_brake_active) if fast_response_enabled else False,
            "active_pitch_crossing_increasing_error_count": int(self._apc_fast_response_increasing_error_count) if fast_response_enabled else 0,
            "active_pitch_crossing_adaptive_tau_limit": float(self._apc_fast_response_adaptive_tau_limit) if fast_response_enabled else 0.0,
            "active_pitch_crossing_tau_before_rate_limit": float(apc_tau_before_rate_limit) if fast_response_enabled and apc_tau_before_rate_limit is not None else float(apc_raw_tau),
            "active_pitch_crossing_tau_after_rate_limit": float(apc_final_tau) if fast_response_enabled else float(apc_final_tau),
            # ---- APCR1g predictive fast response with phase brake telemetry ----
            "active_pitch_crossing_predictive_enabled": bool(self.authority_schedule.apc_predictive_enabled),
            "active_pitch_crossing_lead_time_s": float(self.authority_schedule.apc_lead_time_s),
            "active_pitch_crossing_predicted_error_m": float(self._apc_predictive_predicted_error),
            "active_pitch_crossing_abs_predicted_error_m": float(abs(self._apc_predictive_predicted_error)),
            "active_pitch_crossing_predicted_enter_m": float(self.authority_schedule.apc_predicted_enter_m),
            "active_pitch_crossing_predicted_full_response_m": float(self.authority_schedule.apc_predicted_full_response_m),
            "active_pitch_crossing_predictive_trigger_active": bool(self._apc_predictive_predictive_trigger_active),
            "active_pitch_crossing_predictive_boost_active": bool(self._apc_predictive_predictive_boost_active),
            "active_pitch_crossing_phase_brake_strong_active": bool(self._apc_predictive_phase_brake_strong_active),
            "active_pitch_crossing_phase_brake_factor_current": float(
                self.authority_schedule.apc_predictive_phase_brake_strong_factor if self._apc_predictive_phase_brake_strong_active
                else self.authority_schedule.apc_predictive_phase_brake_factor if self._apc_predictive_phase_brake_active
                else 1.0
            ),
            # ---- APCR1h drift priority telemetry ----
            "active_pitch_crossing_drift_priority_enabled": bool(self.authority_schedule.apc_drift_priority_enabled),
            "active_pitch_crossing_drift_priority_active": bool(self._apc_drift_priority_active),
            "active_pitch_crossing_emergency_drift_clamp_active": bool(self._apc_drift_priority_emergency_active),
            "active_pitch_crossing_drift_priority_reason": str(drift_priority_reason) if 'drift_priority_reason' in dir() else "none",
            "active_pitch_crossing_drift_priority_tau_limit": float(self._apc_drift_priority_tau_limit) if hasattr(self, '_apc_drift_priority_tau_limit') else 0.0,
            "active_pitch_crossing_selected_tau_limit": float(selected_tau_limit) if 'selected_tau_limit' in dir() else 0.0,
            "active_pitch_crossing_selected_rate_limit": float(selected_rate_limit) if 'selected_rate_limit' in dir() else 0.0,
            "active_pitch_crossing_support_priority_over_pitch": bool(self._apc_drift_priority_active),
            # Phase brake disabled reason: check both drift priority and hysteresis
            "active_pitch_crossing_phase_brake_disabled_reason": (
                "drift_priority" if self._apc_drift_priority_active else
                ("hysteresis_recenter" if self._apc_hysteresis_state != "NEUTRAL" else "none")
            ),
            "active_pitch_crossing_drift_clamp_success": bool(self._apc_drift_priority_error_rate_reversal_achieved) if hasattr(self, '_apc_drift_priority_error_rate_reversal_achieved') else False,
            "active_pitch_crossing_steps_since_hard_drift": int(self._apc_drift_priority_steps_since_hard_drift) if hasattr(self, '_apc_drift_priority_steps_since_hard_drift') else 0,
            "active_pitch_crossing_error_rate_reversal_achieved": bool(self._apc_drift_priority_error_rate_reversal_achieved) if hasattr(self, '_apc_drift_priority_error_rate_reversal_achieved') else False,
            "active_pitch_crossing_physical_drift_column_used": "sagittal_position_error_m",
            "active_pitch_crossing_wheel_velocity_monitor_only": True,  # APCR1h does not restrict wheel velocity
            # ---- APCR1i hysteresis recenter telemetry ----
            "active_pitch_crossing_hysteresis_enabled": bool(self.authority_schedule.apc_hysteresis_enabled),
            "active_pitch_crossing_hysteresis_state": str(self._apc_hysteresis_state),
            "active_pitch_crossing_hysteresis_state_id": int({"NEUTRAL": 0, "RECENTER_FROM_POSITIVE": 1, "RECENTER_FROM_NEGATIVE": 2, "HOLD_THROUGH_ZERO": 3}.get(self._apc_hysteresis_state, 0)),
            "active_pitch_crossing_hysteresis_entry_e": float(self._apc_hysteresis_entry_e),
            "active_pitch_crossing_hysteresis_exit_e": float(self._apc_hysteresis_exit_e),
            "active_pitch_crossing_hysteresis_entry_count": int(self._apc_hysteresis_state_entry_count),
            "active_pitch_crossing_hysteresis_exit_count": int(self._apc_hysteresis_state_exit_count),
            "active_pitch_crossing_hysteresis_inner_exit_m": float(self.authority_schedule.apc_hysteresis_inner_exit_m),
            "active_pitch_crossing_hysteresis_opposite_release_m": float(self.authority_schedule.apc_hysteresis_opposite_release_m),
            "active_pitch_crossing_hysteresis_emergency_active": bool(self._apc_hysteresis_emergency_active),
            "active_pitch_crossing_physical_drift_column_used": "sagittal_position_error_m",
            # APCR1l pitch suppression telemetry
            "apcr1l_pitch_suppress_active": bool(pitch_suppress_active),
            "apcr1l_recenter_state": str(self._apc_hysteresis_state),
            # Log what tau_pitch WOULD have been without suppression
            "apcr1l_tau_pitch_before_suppress": float(tau_pitch_raw_orig) if pitch_suppress_active else float(tau_pitch_raw),
            # APCR1m conditional pitch blend telemetry
            "apcr1m_pitch_blend_active": bool(apc_pitch_blend_active),
            "apcr1m_pitch_blend_scale": float(apc_pitch_blend_scale),
            "apcr1m_pitch_blend_block_reason": str(apc_pitch_blend_block_reason),
            "apcr1m_tau_pitch_before_blend": float(tau_pitch_before_blend),
            "apcr1m_tau_pitch_after_blend": float(tau_pitch),
            "apcr1m_startup_guard_active": bool(apc_pitch_blend_startup_guard_active),
            "apcr1m_recenter_active": bool(apc_pitch_blend_recenter_active),
            "apcr1m_pitch_safe": bool(apc_pitch_blend_pitch_safe),
            "apcr1m_height_safe": bool(apc_pitch_blend_height_safe),
            "apcr1m_contact_safe": bool(apc_pitch_blend_contact_safe),
            # APCR1nD direct support drift trigger telemetry
            "apcr1nd_direct_recenter_priority_active": bool(apcr1nd_direct_recenter_priority_active),
            "apcr1nd_direct_recenter_eligible": bool(apcr1nd_direct_recenter_eligible),
            "apcr1nd_direct_recenter_block_reason": str(apcr1nd_direct_recenter_block_reason),
            "apcr1nd_moving_away": bool(apcr1nd_moving_away),
            "apcr1nd_abs_error": float(apcr1nd_abs_error),
            "apcr1nd_error_rate": float(apcr1nd_error_rate),
            # APCR1nD tuned variants telemetry
            "tuned_variant_name": str(self.authority_schedule.apcr1nd_tuned_variant_name) if self.authority_schedule.apcr1nd_tuned_enabled else "",
            "tuned_recenter_active": bool(apcr1nd_direct_recenter_priority_active) if self.authority_schedule.apcr1nd_tuned_enabled else False,
            "tuned_band_state": self._compute_tuned_band_state(apcr1nd_abs_error)[0] if self.authority_schedule.apcr1nd_tuned_enabled else "none",
            "tuned_band_state_id": self._compute_tuned_band_state(apcr1nd_abs_error)[1] if self.authority_schedule.apcr1nd_tuned_enabled else 0,
            "tuned_abs_error": float(apcr1nd_abs_error) if self.authority_schedule.apcr1nd_tuned_enabled else 0.0,
            "tuned_error_rate": float(apcr1nd_error_rate) if self.authority_schedule.apcr1nd_tuned_enabled else 0.0,
            "tuned_moving_away": bool(apcr1nd_moving_away) if self.authority_schedule.apcr1nd_tuned_enabled else False,
            "tuned_converging": bool(apcr1nd_error_rate * float(sagittal_position_error_m) < 0) if self.authority_schedule.apcr1nd_tuned_enabled else False,
            "tuned_release_allowed": bool(apcr1nd_abs_error <= self.authority_schedule.apcr1nd_release_inner_m) if self.authority_schedule.apcr1nd_tuned_enabled else False,
            "tuned_active_reason": str(apcr1nd_direct_recenter_block_reason) if self.authority_schedule.apcr1nd_tuned_enabled else "none",
            "tuned_block_reason": str(apcr1nd_direct_recenter_block_reason) if self.authority_schedule.apcr1nd_tuned_enabled else "none",
            "tuned_position_cap_current": float(apcr1n_position_cap_current) if self.authority_schedule.apcr1nd_tuned_enabled else 0.0,
            "tuned_wheel_damping_scale": float(apcr1n_wheel_damping_scale) if self.authority_schedule.apcr1nd_tuned_enabled else 1.0,
            "tuned_wheel_damping_override_active": bool(apcr1n_wheel_damping_override_active) if self.authority_schedule.apcr1nd_tuned_enabled else False,
            "tuned_outside_band_active": bool(apcr1nd_direct_recenter_priority_active and apcr1nd_abs_error > self.authority_schedule.apcr1nd_desired_band_m) if self.authority_schedule.apcr1nd_tuned_enabled else False,
            "tuned_outside_band_inactive": bool(not apcr1nd_direct_recenter_priority_active and apcr1nd_abs_error > self.authority_schedule.apcr1nd_desired_band_m) if self.authority_schedule.apcr1nd_tuned_enabled else False,
            "tuned_recenter_held": bool(self._apcr1nd_tuned_recenter_held) if self.authority_schedule.apcr1nd_tuned_enabled else False,
            "tuned_release_counter": int(self._apcr1nd_tuned_converging_steps) if self.authority_schedule.apcr1nd_tuned_enabled else 0,
            "tuned_final_torque_direction_correct": bool(apcr1n_final_torque_direction_correct) if self.authority_schedule.apcr1nd_tuned_enabled else True,
            # APCR1n recenter priority telemetry
            "apcr1n_recenter_priority_active": bool(apcr1n_recenter_priority_active),
            "apcr1n_startup_guard_active": bool(apcr1n_startup_guard_active),
            "apcr1n_wheel_damping_override_active": bool(apcr1n_wheel_damping_override_active),
            "apcr1n_wheel_damping_scale": float(apcr1n_wheel_damping_scale),
            "apcr1n_wheel_damping_before": float(apcr1n_wheel_damping_before),
            "apcr1n_wheel_damping_after": float(apcr1n_wheel_damping_after),
            "apcr1n_wheel_damping_fights_drift": bool(apcr1n_wheel_damping_fights_drift),
            "apcr1n_position_cap_boost_active": bool(apcr1n_position_cap_boost_active),
            "apcr1n_position_cap_current": float(apcr1n_position_cap_current),
            "apcr1n_tau_position_raw": float(apcr1n_tau_position_raw),
            "apcr1n_tau_position_after_cap": float(apcr1n_tau_position_after_cap),
            "apcr1n_position_saturated": bool(apcr1n_position_saturated),
            "apcr1n_safety_gate_pass": bool(apcr1n_safety_gate_pass),
            "apcr1n_final_torque_direction_correct": bool(apcr1n_final_torque_direction_correct),
            "apcr1n_final_torque_fights_drift": bool(apcr1n_final_torque_fights_drift),
            "apcr1n_physical_drift_column_used": str(apcr1n_physical_drift_column_used),
            "final_wheel_tau_with_apc": float(tau_common_unclipped + (tau_wheel_vel_left + tau_wheel_vel_right) / 2.0),
            "final_wheel_tau_without_apc": float(tau_common_unclipped - apc_tau_clipped + (tau_wheel_vel_left + tau_wheel_vel_right) / 2.0),
            # ---- Unified sagittal state-feedback no-offset controller telemetry ----
            "no_offset_controller_active": bool(no_offset_active),
            "no_offset_mode": str(no_offset_mode),
            "no_offset_gate_pass": bool(no_offset_gate_pass),
            "no_offset_block_reason": str(no_offset_block_reason),
            "no_offset_kx": float(no_offset_kx),
            "no_offset_kv": float(no_offset_kv),
            "no_offset_ktheta": float(no_offset_ktheta),
            "no_offset_komega": float(no_offset_komega),
            "no_offset_kh": float(no_offset_kh),
            "no_offset_khdot": float(no_offset_khdot),
            "no_offset_tau_support_state": float(no_offset_tau_support_state),
            "no_offset_tau_pitch_state": float(no_offset_tau_pitch_state),
            "no_offset_tau_rate_state": float(no_offset_tau_rate_state),
            "no_offset_tau_height_state": float(no_offset_tau_height_state),
            "no_offset_priority_support": float(no_offset_priority_support),
            "no_offset_priority_pitch": float(no_offset_priority_pitch),
            "no_offset_priority_rate": float(no_offset_priority_rate),
            "no_offset_tau_total_raw": float(no_offset_tau_total_raw),
            "no_offset_tau_total_limited": float(no_offset_tau_total_limited),
            "no_offset_torque_cap": float(no_offset_torque_cap),
            "no_offset_rate_limit": float(no_offset_rate_limit),
            "no_offset_saturation_active": bool(no_offset_saturation_active),
            "no_offset_arbitration_reason": str(no_offset_arbitration_reason),
            "no_offset_pitch_ref_offset_deg": float(no_offset_pitch_ref_offset_deg),
            # === K1 Augmented Telemetry — Phase 1 (read-only, behavior-neutral) ===
            # A. Pitch-rate notch / filter path
            "k1_raw_pitch_rate_x": float(pitch_rate_raw),
            "k1_filtered_pitch_rate_x": float(pitch_rate_effective),
            "k1_notch_output": float(pitch_rate_notched),
            "k1_notch_input": float(pitch_rate_raw),
            "k1_notch_state_1": float(self._wip_notch_pitch_rate.get_state()[0]) if self._wip_notch_pitch_rate is not None else 0.0,
            "k1_notch_state_2": float(self._wip_notch_pitch_rate.get_state()[1]) if self._wip_notch_pitch_rate is not None else 0.0,
            "k1_notch_state_y1": float(self._wip_notch_pitch_rate.get_state()[2]) if self._wip_notch_pitch_rate is not None else 0.0,
            "k1_notch_state_y2": float(self._wip_notch_pitch_rate.get_state()[3]) if self._wip_notch_pitch_rate is not None else 0.0,
            "k1_notch_enabled": bool(notch_enabled),
            "k1_notch_blend": float(notch_blend),
            "k1_notch_center_hz": float(notch_center_hz),
            "k1_notch_q": float(notch_q),
            "k1_notch_height_gate_alpha": float(notch_height_gate),
            "k1_notch_filter_type": str(self.authority_schedule.wip_notch_filter_type),
            "k1_lowpass_cutoff_hz": float(self.authority_schedule.wip_lowpass_cutoff_hz),
            # B. Torque decomposition before clipping
            "k1_tau_pitch_raw": float(tau_pitch),
            "k1_tau_pitch_rate_raw": float(tau_pitch_rate),
            "k1_tau_position_raw": float(tau_position_before_clip),
            "k1_tau_com_velocity_raw": float(tau_sagittal_velocity),
            "k1_tau_wheel_velocity_raw": float(tau_wheel_vel_left + tau_wheel_vel_right),
            "k1_tau_support_velocity_raw": float(tau_support_velocity),
            "k1_tau_eq_ff_raw": 0.0,
            "k1_tau_common_preclip": float(tau_common_unclipped),
            "k1_tau_left_preclip": float(tau_common_unclipped + tau_wheel_vel_left),
            "k1_tau_right_preclip": float(tau_common_unclipped + tau_wheel_vel_right),
            # C. Torque clipping / saturation
            "k1_tau_position_cap_active": bool(tau_position_saturated),
            "k1_tau_position_cap_margin_nm": float(effective_max_position_tau - abs(float(tau_position_before_clip))),
            "k1_tau_total_clip_active": bool(saturated),
            "k1_tau_total_clip_margin_nm": float(final_wheel_torque_margin),
            "k1_tau_left_postclip": float(tau_left),
            "k1_tau_right_postclip": float(tau_right),
            "k1_tau_clip_delta_left": float((tau_common_unclipped + tau_wheel_vel_left) - tau_left),
            "k1_tau_clip_delta_right": float((tau_common_unclipped + tau_wheel_vel_right) - tau_right),
            "k1_tau_clip_delta_common": float(tau_common_unclipped - tau_common),
            "k1_saturation_fraction_window_50": -1.0,
            "k1_saturation_fraction_window_200": -1.0,
            # D. Support / coupling diagnostics
            "k1_support_error_m": float(sagittal_position_error_m),
            "k1_support_velocity_m_s": float(support_position_velocity_m_s),
            "k1_com_y_velocity_m_s": float(sagittal_velocity_m_s),
            "k1_pitch_support_phase_lag_s_est": 0.0,
            "k1_pitch_support_corr_window_200": 0.0,
            # E. Controller mode flags
            "k1_feedback_mode": str("balance-core"),
            "k1_profile_name": str(self.authority_schedule.profile_name),
            "k1_current_best_id": "K2_NOTCH_LOW_Q_V1",  # K2 promoted 2026-06-25 (Step C/E/D passed; K1 legacy)
            "k1_audit_ablation_mode": "none",
            "k1_telemetry_augmented_version": 1,
        }

        # Add capture gate diagnostics if enabled
        if capture_gate_diagnostics is not None:
            diagnostics.update({
                "capture_gate_enabled": True,
                "capture_gate_required_direction": capture_gate_diagnostics.required_capture_direction,
                "capture_gate_tau_position_direction": capture_gate_diagnostics.tau_position_direction,
                "capture_gate_position_opposes_capture": capture_gate_diagnostics.position_opposes_capture,
                "capture_gate_factor": capture_gate_diagnostics.gate_factor,
                "capture_gate_active": capture_gate_diagnostics.gate_active,
                "capture_gate_reason": capture_gate_diagnostics.gate_reason,
                "capture_gate_pitch_reversal": capture_gate_diagnostics.pitch_reversal_detected,
                "capture_gate_capture_recovery": capture_gate_diagnostics.capture_recovery_detected,
                "capture_gate_tau_position_gated": capture_gate_diagnostics.tau_position_gated,
                "capture_gate_cp_relative_to_support_m": capture_gate_diagnostics.capture_point_relative_to_support_m,
                "capture_gate_com_support_error_m": capture_gate_diagnostics.com_support_error_y_m,
            })
        else:
            diagnostics["capture_gate_enabled"] = False

        return tau, diagnostics
