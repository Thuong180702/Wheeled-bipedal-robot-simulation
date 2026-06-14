"""Sagittal velocity-damped balance controller for balance-core architecture.

Analytic controller-by-construction approach: explicit state-feedback terms for
pitch balance, sagittal velocity damping, wheel velocity damping, and optional
weak position return. Built after the LQR/sysid path failed Gate 4 identification
(one-step R²=1.0 but 20-step rollout R²=-1.15e10, dominant eigenvalue λ=1.96).

This controller replaces SagittalWheelBalanceController when selected via
--sagittal-controller velocity-damped. Both controllers are mutually exclusive.

Output: nonzero torque only on wheel joints [4, 9].
"""

import jax.numpy as jnp
from jax import Array
from dataclasses import dataclass
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


def smoothstep01(u: float) -> float:
    """Standard smoothstep interpolation: s(0)=0, s(1)=1, s'(0)=s'(1)=0."""
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


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

    # Position integral settings (Step E extreme height fix)
    enable_position_integral: bool = False
    ki_position_integral: float = 0.0  # 0.0 when disabled
    integral_max_abs: float = 0.0  # 0.0 when disabled
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
}

# Backward-compatible aliases — development identifiers → semantic constants.
# These allow existing imports and scripts to keep working. The primary names
# (BAND_LIMITED_SUPPORT_RECENTER, EMERGENCY_BUDGET_CAP_RAISE, etc.) should be
# used in new code.
APCR1ND_T5_BAND_LIMITED_BALANCED = BAND_LIMITED_SUPPORT_RECENTER  # legacy
T6F_BUDGET_CAP_RAISE = EMERGENCY_BUDGET_CAP_RAISE                  # legacy
T6I_PHASE_AWARE_RELEASE = PHASE_AWARE_AUTHORITY_RELEASE            # legacy
T6J_CENTERING_BIAS_TRIM = SUPPORT_CENTERING_BIAS_TRIM             # legacy


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

        # Per-wheel damping (separate for each wheel)
        tau_wheel_vel_left = -effective_k_wheel_velocity * wheel_vel_left_rad_s
        tau_wheel_vel_right = -effective_k_wheel_velocity * wheel_vel_right_rad_s

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

        tau_pitch_rate = self.kd_pitch * pitch_rate_x_rad_s
        tau_sagittal_velocity = -effective_k_velocity * effective_velocity_damping_scale * sagittal_velocity_m_s

        # Support position velocity damping term
        # Directly opposes support-center drift velocity to prevent transient position excursions
        tau_support_velocity = -effective_support_velocity_gain * effective_support_velocity_scale * support_position_velocity_m_s

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

        # Common scalar command (before per-wheel damping)
        # No internal clipping - let the composer handle torque limits like baseline does
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
