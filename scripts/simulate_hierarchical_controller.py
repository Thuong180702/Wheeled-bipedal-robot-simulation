"""Simulate hierarchical controller with full telemetry logging.

Runs the three-level hierarchical controller in MuJoCo simulation and records:
- Joint positions, velocities, torques
- CoM position and velocity
- Capture point
- Controller torques (WBC, Momentum, Posture)
- Fall detection and termination conditions

Saves telemetry to CSV for post-analysis.

Usage:
    python scripts/simulate_hierarchical_controller.py              # Headless simulation
    python scripts/simulate_hierarchical_controller.py --visual     # Visual simulation with viewer
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.momentum_coordinator import (
    MomentumCoordinator,
    MomentumCoordinatorConfig,
)
from wheeled_biped.controllers.orientation_utils import (
    compute_orientation_from_gravity,
    compute_orientation_from_quaternion,
)
from wheeled_biped.controllers.posture_regularizer import (
    PostureRegularizer,
    PostureRegularizerConfig,
)
from wheeled_biped.controllers.leg_position_controller import LegPositionController
from wheeled_biped.controllers.static_posture_holding_controller import StaticPostureHoldingController
from wheeled_biped.controllers.static_feedforward_controller import (
    StaticFeedforwardController,
    load_empirical_feedforward_from_telemetry,
)
from wheeled_biped.controllers.stage2b_roll_direct_controller import Stage2BRollDirectController
from wheeled_biped.controllers.stage2b_sagittal_wheel_controller import Stage2BSagittalWheelController
from wheeled_biped.controllers.stage2c_sagittal_state_feedback_controller import Stage2CSagittalStateFeedbackController
from wheeled_biped.controllers.stage2d_sagittal_lqr_controller import Stage2DSagittalLQRController
from wheeled_biped.controllers.contact_jacobian import ContactJacobian
from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer
from wheeled_biped.controllers.contact_supervisor import ContactSupervisor
from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController
from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    SagittalAuthoritySchedule,
    SagittalVelocityDampedBalanceController,
    JOINT_FIX_J1_SUPPORT_CAP,
    JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING,
    JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING,
    PITCH_SAFE_J2A_CONSERVATIVE,
    PITCH_SAFE_J2B_BALANCED,
    PITCH_SAFE_J2C_VELOCITY_PRIORITY,
    PITCH_SAFE_J2D_TAU_CAP_PRIORITY,
    APCR1ND_T1_EARLY_ENTRY,
    APCR1ND_T2_HOLD_OUTSIDE_BAND,
    APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD,
    APCR1ND_T4_STRONGER_AUTHORITY,
    APCR1ND_T5_BAND_LIMITED_BALANCED,
    T6A_HIGH_EARLY_HARD_BAND,
    T6B_HIGH_STRONGER_EMERGENCY,
    T6C_HIGH_EARLY_PLUS_STRONGER,
    T6D_HIGH_TRANSIENT_BOOST,
    T6E_HIGH_PITCH_AWARE_BOOST,
    T6F_BUDGET_CAP_RAISE,
    T6F_SIGN_CORRECTED,
    T6H_SOFT_BLEND_ARCH_FIX,
    T6I_PHASE_AWARE_RELEASE,
    T6J_CENTERING_BIAS_TRIM,
    # Semantic aliases (point to same objects as the legacy names above)
    EMERGENCY_BUDGET_CAP_RAISE,
    PHASE_AWARE_AUTHORITY_RELEASE,
    SUPPORT_CENTERING_BIAS_TRIM,
    ADAPTIVE_SUPPORT_CENTERING_TRIM,
    BAND_LIMITED_SUPPORT_RECENTER,
    ZERO_CROSSING_SUPPORT_RECENTER,
    EARLY_ZERO_CROSSING_RECENTER,
    EARLY_ZERO_CROSSING_RECENTER_V2,
    PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER,
    PITCH_EQUILIBRIUM_TRIM,
    HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM,
    SUPPORT_POSITION_OUTER_LOOP_PITCH_REF,
    CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF,
    CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF_V2,
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP,
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V1,
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    I_SUPPORT_REFERENCE_REACQUISITION_V1,
    J1A_TALL_KD_PITCH_V1,
    J1B_TALL_KD_PITCH_V1,
    J1C_TALL_KD_PITCH_V1,
    J2A_TALL_K_WHEEL_VEL_V1,
    J2B_TALL_K_WHEEL_VEL_V1,
    J2C_TALL_K_WHEEL_VEL_V1,
    J3A_TALL_COMBINED_V1,
    J3B_TALL_COMBINED_V1,
    K1_PITCH_RATE_NOTCH,
    K1B_PITCH_RATE_NOTCH_2P3,
    K1C_PITCH_RATE_NOTCH_2P7,
    K1D_PITCH_RATE_NOTCH_Q4,
    K1E_PITCH_RATE_NOTCH_Q8,
    K1F_PITCH_RATE_NOTCH_BLEND075,
    K1G_PITCH_RATE_NOTCH_BLEND050,
    K2_NOTCH_LOW_Q_V1,
    K2_WHEEL_VEL_NOTCH,
    K3_PITCH_RATE_WHEEL_VEL_NOTCH,
    K3B_PITCH_RATE_WHEEL_VEL_NOTCH_BLEND075,
    ALL_K_SWEEP_PROFILES,
    L1_K1_COORDINATED_LOW_FREQ_FEEDBACK,
    L2_K1_COORDINATED_PHASE_LEAD,
    L3_K1_COORDINATED_PITCH_REF_STABILIZATION,
    LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
    LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
    LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1,
    LRS1_SUPPORT_DOMINANT_V1,
    LRS2_PITCH_RATE_DAMPING_V1,
    LRS3_BALANCED_MEDIUM_V1,
    LP1_K1_PRIORITY_PITCH_FIRST_SUPPORT_RESIDUAL_V1,
    LP2_K1_PRIORITY_PITCH_STRONG_SUPPORT_SOFT_V1,
    LP3_K1_PRIORITY_SUPPORT_RECENTER_WHEN_SAFE_V1,
    M1_K1_BODY_YAW_DIFF_WHEEL_V1,
    M2_K1_BODY_YAW_SUPPORT_AWARE_V1,
    N1_K1_MILD_PHASE_LEAD_DAMPING,
    N1B_K1_MILD_PHASE_LEAD_V1,
    N1C_K1_MILD_PHASE_LEAD_V1,
    N1D_K1_MILD_PHASE_LEAD_V1,
    UNIFIED_SAGITTAL_STATE_FEEDBACK_NO_OFFSET,
    interpolate_pitch_ref_offset,
    compute_outer_loop_pitch_ref,
    apply_rate_limit,
    apply_lowpass,
)
from wheeled_biped.controllers.pitch_rate_consistency_estimator import PitchRateConsistencyEstimator
from wheeled_biped.controllers.sagittal_balance_state import (
    project_sagittal_displacement,
    project_sagittal_velocity,
    compute_support_center_xy,
)
from wheeled_biped.controllers.shape_posture_controller import (
    BALANCE_CORE_HIP_YAW_AUTHORITY,
    ShapePostureController,
)
from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController
from wheeled_biped.controllers.yaw_controller import YawController
from wheeled_biped.controllers.differential_wheel_yaw_stabilizer import (
    DifferentialWheelYawStabilizer,
)
from wheeled_biped.controllers.balance_core_types import make_balance_core_telemetry_columns
from wheeled_biped.validation.telemetry_adapter import (
    add_validation_telemetry_fields,
    normalize_balance_core_owner_names,
)


STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD = np.array([
    0.0, 0.0, 4.1, -15.5, 0.0,
    0.0, 0.0, 3.2, -15.8, 0.0,
], dtype=np.float64)

HIGH_HEIGHT_VARIANTS = ("high_tiny", "high_small")
EXTREME_HEIGHT_VARIANTS = ("min_operational_height", "max_operational_height")
BOUNDARY_HEIGHT_VARIANTS = ("low_0p300", "high_0p480")
D2_HEIGHT_VARIANTS = HIGH_HEIGHT_VARIANTS + EXTREME_HEIGHT_VARIANTS + BOUNDARY_HEIGHT_VARIANTS
SAGITTAL_AUTHORITY_PROFILES = {
    "baseline": SagittalAuthoritySchedule(),
    "candidate_A_position_cap": SagittalAuthoritySchedule(
        profile_name="candidate_A_position_cap",
        applies_to_variants=HIGH_HEIGHT_VARIANTS,
        position_tau_cap_scale=4.0 / 3.0,
    ),
    "candidate_B_balanced": SagittalAuthoritySchedule(
        profile_name="candidate_B_balanced",
        applies_to_variants=HIGH_HEIGHT_VARIANTS,
        position_tau_cap_scale=4.0 / 3.0,
        pitch_tau_scale=0.9,
    ),
    "candidate_C_stronger_position": SagittalAuthoritySchedule(
        profile_name="candidate_C_stronger_position",
        applies_to_variants=HIGH_HEIGHT_VARIANTS,
        position_tau_cap_scale=4.5 / 3.0,
        pitch_tau_scale=0.85,
    ),
    "candidate_A2_height_staged": SagittalAuthoritySchedule(
        profile_name="candidate_A2_height_staged",
        applies_to_variants=HIGH_HEIGHT_VARIANTS,
        position_tau_cap_by_variant=(
            ("high_tiny", 4.0),
            ("high_small", 4.5),
        ),
        pitch_tau_scale=1.0,
    ),
    "candidate_D1_support_velocity_light": SagittalAuthoritySchedule(
        profile_name="candidate_D1_support_velocity_light",
        applies_to_variants=HIGH_HEIGHT_VARIANTS,
        position_tau_cap_by_variant=(
            ("high_tiny", 4.0),
            ("high_small", 4.0),
        ),
        pitch_tau_scale=1.0,
        support_velocity_gain=0.2,
    ),
    "candidate_D2_wheel_velocity_damping_light": SagittalAuthoritySchedule(
        profile_name="candidate_D2_wheel_velocity_damping_light",
        applies_to_variants=D2_HEIGHT_VARIANTS,
        position_tau_cap_by_variant=(
            ("high_tiny", 4.0),
            ("high_small", 4.0),
            ("min_operational_height", 4.0),
            ("max_operational_height", 4.0),
            ("low_0p300", 4.0),
            ("high_0p480", 4.0),
        ),
        pitch_tau_scale=1.0,
        velocity_damping_scale=1.10,
    ),
    "candidate_E1_k60_continuous": SagittalAuthoritySchedule(
        profile_name="candidate_E1_k60_continuous",
        applies_to_variants=(),  # Not variant-specific - formula-based
        continuous_k_position=True,
        k_position_nominal=40.0,
        k_position_low_max=60.0,
        k_position_z_low=0.300,
        k_position_z_high=0.393,
    ),
    "candidate_E2_k80_continuous": SagittalAuthoritySchedule(
        profile_name="candidate_E2_k80_continuous",
        applies_to_variants=(),  # Not variant-specific - formula-based
        continuous_k_position=True,
        k_position_nominal=40.0,
        k_position_low_max=80.0,
        k_position_z_low=0.300,
        k_position_z_high=0.393,
    ),
    "candidate_E3_k100_continuous": SagittalAuthoritySchedule(
        profile_name="candidate_E3_k100_continuous",
        applies_to_variants=(),  # Not variant-specific - formula-based
        continuous_k_position=True,
        k_position_nominal=40.0,
        k_position_low_max=100.0,
        k_position_z_low=0.300,
        k_position_z_high=0.393,
    ),
    # Phase 6 Joint Low-Height Sagittal-Yaw Fix Profiles
    "J1": JOINT_FIX_J1_SUPPORT_CAP,
    "J2": JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING,
    "J3": JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING,
    # Pitch-Safe Candidates (J2a-J2d family)
    "J2a": PITCH_SAFE_J2A_CONSERVATIVE,
    "J2b": PITCH_SAFE_J2B_BALANCED,
    "J2c": PITCH_SAFE_J2C_VELOCITY_PRIORITY,
    "J2d": PITCH_SAFE_J2D_TAU_CAP_PRIORITY,
    # Step E Extreme Height Support Fix Candidates
    # Priority: 1. support_position_error, 2. wheel_velocity transient, 3. hip_yaw (after support/wheel)
    # These profiles are opt-in only - do NOT modify candidate_D2_wheel_velocity_damping_light
    "E1_support_integral": SagittalAuthoritySchedule(
        profile_name="E1_support_integral",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480 only
        # Enable position integral for steady-state error correction
        # FIX: Raise pitch_error threshold from 0.03 to 0.12 rad to allow integral
        # to activate during normal low_0p300 pitch oscillations (max 0.11 rad).
        # At 0.03 rad, the integral was blocked for 349/500 steps (69.8%).
        # At 0.12 rad, the integral can activate during normal operation while
        # still blocking during extreme pitch events that indicate fall risk.
        enable_position_integral=True,
        ki_position_integral=2.0,
        integral_max_abs=1.0,
        integral_pitch_error_threshold_rad=0.12,  # Was 0.03 - raised to allow low_0p300 normal pitch
        integral_support_velocity_threshold_m_s=0.03,
        integral_wheel_velocity_threshold_rad_s=1.0,
        integral_min_com_z_m=0.28,
        integral_max_com_z_m=0.50,
        # Keep position cap at 4.0 Nm (same as D2)
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        # Keep velocity damping at D2 level (1.10 scale via variant)
        velocity_damping_scale=1.10,
    ),
    "E2_support_integral_higher_cap": SagittalAuthoritySchedule(
        profile_name="E2_support_integral_higher_cap",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,
        # Enable position integral (same as E1)
        enable_position_integral=True,
        ki_position_integral=2.0,
        integral_max_abs=1.0,
        integral_pitch_error_threshold_rad=0.03,
        integral_support_velocity_threshold_m_s=0.03,
        integral_wheel_velocity_threshold_rad_s=1.0,
        integral_min_com_z_m=0.28,
        integral_max_com_z_m=0.50,
        # Increase position cap to 5.0 Nm (25% increase)
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=5.0,
        # Keep velocity damping at D2 level
        velocity_damping_scale=1.10,
    ),
    # E2b: Same as E2 but with integral gate aligned to E1 (0.12 rad vs 0.03 rad).
    # Hypothesis: E2's 0.03 rad threshold was too restrictive, causing tau_position
    # to accumulate aggressively which drove hip_yaw divergence. By widening to 0.12 rad
    # (E1's value), the integral accumulates more naturally without windup-driven
    # torque spikes that couple to hip_yaw through kinematic coupling.
    "E2b_support_integral_higher_cap_aligned_gate": SagittalAuthoritySchedule(
        profile_name="E2b_support_integral_higher_cap_aligned_gate",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,
        # Enable position integral (same as E1/E2)
        enable_position_integral=True,
        ki_position_integral=2.0,
        integral_max_abs=1.0,
        integral_pitch_error_threshold_rad=0.12,  # KEY CHANGE: align to E1 value
        integral_support_velocity_threshold_m_s=0.03,
        integral_wheel_velocity_threshold_rad_s=1.0,
        integral_min_com_z_m=0.28,
        integral_max_com_z_m=0.50,
        # Keep position cap at E2 level (5.0 Nm)
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=5.0,
        # Keep velocity damping at D2 level
        velocity_damping_scale=1.10,
    ),
    "E3_support_integral_cap_wheel_damping": SagittalAuthoritySchedule(
        profile_name="E3_support_integral_cap_wheel_damping",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,
        # Enable position integral (same as E1/E2)
        enable_position_integral=True,
        ki_position_integral=2.0,
        integral_max_abs=1.0,
        integral_pitch_error_threshold_rad=0.03,
        integral_support_velocity_threshold_m_s=0.03,
        integral_wheel_velocity_threshold_rad_s=1.0,
        integral_min_com_z_m=0.28,
        integral_max_com_z_m=0.50,
        # Increase position cap (same as E2)
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=5.0,
        # Add high-height wheel damping for high_0p480 transient
        continuous_k_wheel_velocity=True,
        k_wheel_velocity_nominal=0.5,
        k_wheel_velocity_high_max=0.75,  # 50% increase at high heights
        k_wheel_velocity_z_low=0.45,
        k_wheel_velocity_z_high=0.52,
    ),
    # F1_phase_aware_recenter_wider_yaw_gate
    # F1 with wider hip_yaw gate (0.15 rad vs 0.10 rad)
    # Fixes circular dependency: F1 hip_yaw gate 0.10 blocked recentering when D2 already reaches ~0.1018 rad
    # This allows recentering to activate more often and reduce signed support bias
    "F1_phase_aware_recenter_wider_yaw_gate": SagittalAuthoritySchedule(
        profile_name="F1_phase_aware_recenter_wider_yaw_gate",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Phase-aware recenter - decoupled from tau_position
        enable_phase_aware_recenter=True,
        k_recenter=10.0,  # Nm/m
        max_recenter_tau=1.0,  # Nm - bounded, separate from balance
        recenter_deadband_m=0.01,  # m
        recenter_pitch_safe_threshold_rad=0.05,  # rad
        recenter_pitch_danger_threshold_rad=0.10,  # rad
        recenter_hip_yaw_safe_threshold_rad=0.15,  # rad - WIDER (D2 reaches ~0.1018, was blocking recenter)
        recenter_smooth_alpha=0.10,
        recenter_max_rate_per_step=0.5,  # Nm/step
        recenter_min_com_z_m=0.28,  # m
        recenter_max_com_z_m=0.50,  # m
    ),
    # F1_phase_aware_recenter_wider_yaw_gate_low_tau (F1c)
    # Conservative variant with lower max_recenter_tau (0.5 Nm vs 1.0 Nm)
    # Fallback if F1b reduces drift but causes hip-yaw or wheel velocity instability
    "F1_phase_aware_recenter_wider_yaw_gate_low_tau": SagittalAuthoritySchedule(
        profile_name="F1_phase_aware_recenter_wider_yaw_gate_low_tau",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Phase-aware recenter - decoupled from tau_position
        enable_phase_aware_recenter=True,
        k_recenter=10.0,  # Nm/m
        max_recenter_tau=0.5,  # Nm - CONSERVATIVE (half of F1b)
        recenter_deadband_m=0.01,  # m
        recenter_pitch_safe_threshold_rad=0.05,  # rad
        recenter_pitch_danger_threshold_rad=0.10,  # rad
        recenter_hip_yaw_safe_threshold_rad=0.15,  # rad - same as F1b
        recenter_smooth_alpha=0.10,
        recenter_max_rate_per_step=0.25,  # Nm/step - slower than F1b
        recenter_min_com_z_m=0.28,  # m
        recenter_max_com_z_m=0.50,  # m
    ),
    # F2a_hysteresis_recenter_moderate
    # Hysteresis recenter with moderate torque (1.5 Nm max)
    # Holds recenter direction until signed_error returns to exit target
    # This fixes the one-sided ratcheting that F1b does not fully solve
    "F2a_hysteresis_recenter_moderate": SagittalAuthoritySchedule(
        profile_name="F2a_hysteresis_recenter_moderate",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Hysteresis recenter (F2_strategy) - stateful recenter
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,  # m - enter when error exceeds this
        hysteresis_exit_target_m=0.00,  # m - exit when error reaches this
        hysteresis_opposite_overshoot_m=0.01,  # m - slight overshoot into opposite direction
        hysteresis_k_recenter=10.0,  # Nm/m - gain
        hysteresis_max_recenter_tau=1.5,  # Nm - moderate torque
        hysteresis_smooth_alpha=0.10,
        hysteresis_max_rate_per_step=0.5,  # Nm/step
        hysteresis_deadband_m=0.01,  # m - ignore small errors in NEUTRAL
        hysteresis_pitch_safe_threshold_rad=0.05,  # rad
        hysteresis_pitch_danger_threshold_rad=0.10,  # rad
        hysteresis_hip_yaw_safe_threshold_rad=0.15,  # rad - wider gate
        hysteresis_min_com_z_m=0.28,  # m
        hysteresis_max_com_z_m=0.50,  # m
    ),
    # F2b_hysteresis_recenter_strong
    # Stronger hysteresis recenter with higher torque (2.0 Nm max)
    # Only run if F2a improves but not enough
    "F2b_hysteresis_recenter_strong": SagittalAuthoritySchedule(
        profile_name="F2b_hysteresis_recenter_strong",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Hysteresis recenter (F2_strategy) - stronger variant
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,  # m - enter when error exceeds this
        hysteresis_exit_target_m=0.00,  # m - exit when error reaches this
        hysteresis_opposite_overshoot_m=0.02,  # m - larger overshoot into opposite direction
        hysteresis_k_recenter=12.0,  # Nm/m - higher gain
        hysteresis_max_recenter_tau=2.0,  # Nm - stronger torque
        hysteresis_smooth_alpha=0.10,
        hysteresis_max_rate_per_step=0.5,  # Nm/step
        hysteresis_deadband_m=0.01,  # m - ignore small errors in NEUTRAL
        hysteresis_pitch_safe_threshold_rad=0.05,  # rad
        hysteresis_pitch_danger_threshold_rad=0.10,  # rad
        hysteresis_hip_yaw_safe_threshold_rad=0.15,  # rad - wider gate
        hysteresis_min_com_z_m=0.28,  # m
        hysteresis_max_com_z_m=0.50,  # m
    ),
    # G1a_bias_cancel_moderate
    # Persistent bias cancellation for one-sided positive drift
    # Estimates persistent signed error bias and applies bounded opposite torque
    # Unlike F2 which waits for natural negative drift, G1 estimates bias and cancels proactively
    # Key difference from F2: does NOT gate on pitch (pitch reversal doesn't produce negative drift)
    "G1a_bias_cancel_moderate": SagittalAuthoritySchedule(
        profile_name="G1a_bias_cancel_moderate",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Bias cancellation (G1_strategy) - persistent bias cancellation
        enable_bias_cancel=True,
        bias_cancel_k=12.0,  # Nm/m - moderate gain
        bias_cancel_max_tau=1.5,  # Nm - bounded torque
        bias_cancel_filter_alpha=0.02,  # slow filter for smooth bias estimation
        bias_cancel_deadband_m=0.02,  # m - ignore small persistent errors
        bias_cancel_contact_gate=True,  # require valid contact
        bias_cancel_height_gate=True,  # require valid height
        bias_cancel_roll_gate=True,  # require valid roll
        bias_cancel_pitch_gate=False,  # NOT gated on pitch (key difference from F2)
        bias_cancel_min_com_z_m=0.28,  # m
        bias_cancel_max_com_z_m=0.50,  # m
        bias_cancel_roll_threshold_rad=0.15,  # rad
    ),
    # G1b_bias_cancel_strong
    # Stronger bias cancellation with higher torque (2.0 Nm max)
    # Only run if G1a improves but not enough
    "G1b_bias_cancel_strong": SagittalAuthoritySchedule(
        profile_name="G1b_bias_cancel_strong",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Bias cancellation (G1_strategy) - stronger variant
        enable_bias_cancel=True,
        bias_cancel_k=15.0,  # Nm/m - higher gain
        bias_cancel_max_tau=2.0,  # Nm - stronger torque
        bias_cancel_filter_alpha=0.03,  # faster filter
        bias_cancel_deadband_m=0.02,  # m - ignore small persistent errors
        bias_cancel_contact_gate=True,
        bias_cancel_height_gate=True,
        bias_cancel_roll_gate=True,
        bias_cancel_pitch_gate=False,  # NOT gated on pitch
        bias_cancel_min_com_z_m=0.28,  # m
        bias_cancel_max_com_z_m=0.50,  # m
        bias_cancel_roll_threshold_rad=0.15,  # rad
    ),
    # APC1_active_pitch_crossing_moderate
    # Active Pitch Crossing controller with moderate torque (1.5 Nm max)
    # Explicitly drives wheel torque to create controlled pitch-rate reversal
    # When robot has positive pitch AND positive signed drift, APC applies wheel torque
    # to reverse pitch_rate, allowing support to return toward 0.
    "APC1_active_pitch_crossing_moderate": SagittalAuthoritySchedule(
        profile_name="APC1_active_pitch_crossing_moderate",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Active Pitch Crossing (APC_strategy) - explicit pitch-rate crossing
        enable_active_pitch_crossing=True,
        apc_outer_enter_m=0.10,  # m - enter crossing when |signed_error| > this
        apc_inner_exit_m=0.05,  # m - exit crossing when |signed_error| <= this
        apc_opposite_overshoot_m=0.01,  # m - allow slight overshoot
        apc_pitch_enter_rad=0.03,  # rad - pitch must exceed this to enter
        apc_pitch_safe_limit_rad=0.08,  # rad - reduce torque if pitch exceeds this
        apc_max_cross_tau=1.5,  # Nm - max crossing torque
        apc_smooth_alpha=0.10,
        apc_max_rate_per_step=0.5,  # Nm/step
        apc_contact_gate=True,
        apc_height_gate=True,
        apc_roll_gate=True,
        apc_min_com_z_m=0.28,  # m
        apc_max_com_z_m=0.50,  # m
        apc_pitch_safe_threshold_rad=0.05,  # rad
        apc_pitch_danger_threshold_rad=0.10,  # rad
        apc_roll_threshold_rad=0.15,  # rad
    ),
    # APC2_active_pitch_crossing_stronger
    # Stronger Active Pitch Crossing with higher torque (2.0 Nm max)
    # Only run if APC1 improves but not enough
    "APC2_active_pitch_crossing_stronger": SagittalAuthoritySchedule(
        profile_name="APC2_active_pitch_crossing_stronger",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Active Pitch Crossing (APC_strategy) - stronger variant
        enable_active_pitch_crossing=True,
        apc_outer_enter_m=0.10,  # m
        apc_inner_exit_m=0.05,  # m
        apc_opposite_overshoot_m=0.01,  # m
        apc_pitch_enter_rad=0.03,  # rad
        apc_pitch_safe_limit_rad=0.10,  # rad - higher limit for stronger variant
        apc_max_cross_tau=2.0,  # Nm - stronger torque
        apc_smooth_alpha=0.10,
        apc_max_rate_per_step=0.5,  # Nm/step
        apc_contact_gate=True,
        apc_height_gate=True,
        apc_roll_gate=True,
        apc_min_com_z_m=0.28,  # m
        apc_max_com_z_m=0.50,  # m
        apc_pitch_safe_threshold_rad=0.05,  # rad
        apc_pitch_danger_threshold_rad=0.10,  # rad
        apc_roll_threshold_rad=0.15,  # rad
    ),

    # APCR1_active_pitch_crossing_recovery_moderate
    # NEW: Active Pitch Recovery with recovery gate mode
    # Uses separate hard safety gate from recovery activation gate
    # Hard safety only blocks at pitch > 0.30 rad (vs old 0.10 rad)
    # APCR can activate during moderate pitch error when drift is present
    "APCR1_active_pitch_crossing_recovery_moderate": SagittalAuthoritySchedule(
        profile_name="APCR1_active_pitch_crossing_recovery_moderate",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Active Pitch Recovery (APCR_strategy) - with recovery gate mode
        enable_active_pitch_crossing=True,
        active_pitch_crossing_recovery_gate_mode=True,  # NEW: separate hard safety from recovery
        apc_outer_enter_m=0.10,  # m - enter when |signed_error| > this
        apc_inner_exit_m=0.05,  # m - exit when |signed_error| <= this
        apc_opposite_overshoot_m=0.01,  # m - allow slight overshoot
        apc_pitch_enter_rad=0.03,  # rad - pitch must exceed this to enter
        apc_pitch_safe_limit_rad=0.08,  # rad - reduce torque if pitch exceeds this
        apc_max_cross_tau=1.0,  # Nm - moderate recovery torque
        apc_smooth_alpha=0.10,
        apc_max_rate_per_step=0.4,  # Nm/step - slightly slower rate
        apc_contact_gate=True,
        apc_height_gate=True,
        apc_roll_gate=True,
        apc_min_com_z_m=0.28,  # m
        apc_max_com_z_m=0.50,  # m
        apc_pitch_safe_threshold_rad=0.05,  # rad - used for OLD gate mode only
        apc_pitch_danger_threshold_rad=0.10,  # rad - used for OLD gate mode only
        apc_roll_threshold_rad=0.15,  # rad
        # APCR recovery gate parameters
        apcr_pitch_hard_stop_rad=0.30,  # rad - hard stop, blocks APCR (17.2 deg)
        apcr_roll_hard_stop_rad=0.15,  # rad - lateral stability
        apcr_min_com_z_m=0.27,  # m - minimum safe height
        apcr_max_com_z_m=0.50,  # m - maximum operating height
    ),

    # APCR1b_active_pitch_crossing_early_release
    # Same as APCR1 but with earlier release to reduce oscillation amplitude
    # inner_exit_m: 0.05 -> 0.07 (exit earlier)
    # opposite_overshoot_m: 0.01 -> 0.00 (no overshoot allowance)
    # Rationale: APCR1 releases too late, causing excessive band violations
    "APCR1b_active_pitch_crossing_early_release": SagittalAuthoritySchedule(
        profile_name="APCR1b_active_pitch_crossing_early_release",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Active Pitch Recovery (APCR_strategy) - with recovery gate mode
        enable_active_pitch_crossing=True,
        active_pitch_crossing_recovery_gate_mode=True,  # same as APCR1
        apc_outer_enter_m=0.10,  # m - same as APCR1
        apc_inner_exit_m=0.07,  # m - CHANGED from 0.05: exit earlier
        apc_opposite_overshoot_m=0.00,  # m - CHANGED from 0.01: no overshoot
        apc_pitch_enter_rad=0.03,  # rad - same as APCR1
        apc_pitch_safe_limit_rad=0.08,  # rad - same as APCR1
        apc_max_cross_tau=1.0,  # Nm - same as APCR1
        apc_smooth_alpha=0.10,
        apc_max_rate_per_step=0.4,  # Nm/step - same as APCR1
        apc_contact_gate=True,
        apc_height_gate=True,
        apc_roll_gate=True,
        apc_min_com_z_m=0.28,  # m - same as APCR1
        apc_max_com_z_m=0.50,  # m - same as APCR1
        apc_pitch_safe_threshold_rad=0.05,  # rad - same as APCR1
        apc_pitch_danger_threshold_rad=0.10,  # rad - same as APCR1
        apc_roll_threshold_rad=0.15,  # rad - same as APCR1
        # APCR recovery gate parameters - same as APCR1
        apcr_pitch_hard_stop_rad=0.30,  # rad - hard stop, blocks APCR
        apcr_roll_hard_stop_rad=0.15,  # rad - lateral stability
        apcr_min_com_z_m=0.27,  # m - minimum safe height
        apcr_max_com_z_m=0.50,  # m - maximum operating height
    ),

    # APCR1c_active_pitch_crossing_early_activation
    # Same as APCR1b but with earlier entry to reduce band violations
    # outer_enter_m: 0.10 -> 0.08 (enter when |signed_error| > 0.08 instead of 0.10)
    # Rationale: APCR1b still allows drift to approach the +0.15 band before recovery starts
    "APCR1c_active_pitch_crossing_early_activation": SagittalAuthoritySchedule(
        profile_name="APCR1c_active_pitch_crossing_early_activation",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Active Pitch Recovery (APCR_strategy) - with recovery gate mode
        enable_active_pitch_crossing=True,
        active_pitch_crossing_recovery_gate_mode=True,  # same as APCR1/APCR1b
        apc_outer_enter_m=0.08,  # m - CHANGED from 0.10: enter earlier
        apc_inner_exit_m=0.07,  # m - same as APCR1b
        apc_opposite_overshoot_m=0.00,  # m - same as APCR1b
        apc_pitch_enter_rad=0.03,  # rad - same as APCR1b
        apc_pitch_safe_limit_rad=0.08,  # rad - same as APCR1b
        apc_max_cross_tau=1.0,  # Nm - same as APCR1b
        apc_smooth_alpha=0.10,
        apc_max_rate_per_step=0.4,  # Nm/step - same as APCR1b
        apc_contact_gate=True,
        apc_height_gate=True,
        apc_roll_gate=True,
        apc_min_com_z_m=0.28,  # m - same as APCR1b
        apc_max_com_z_m=0.50,  # m - same as APCR1b
        apc_pitch_safe_threshold_rad=0.05,  # rad - same as APCR1b
        apc_pitch_danger_threshold_rad=0.10,  # rad - same as APCR1b
        apc_roll_threshold_rad=0.15,  # rad - same as APCR1b
        # APCR recovery gate parameters - same as APCR1b
        apcr_pitch_hard_stop_rad=0.30,  # rad - hard stop, blocks APCR
        apcr_roll_hard_stop_rad=0.15,  # rad - lateral stability
        apcr_min_com_z_m=0.27,  # m - minimum safe height
        apcr_max_com_z_m=0.50,  # m - maximum operating height
    ),

    # APCR1d_symmetric_soft_band_control
    # Symmetric proportional torque shaping instead of bang-bang control
    # Design goals:
    # 1. Intervene earlier (soft_enter = 0.05 m instead of 0.08 m)
    # 2. Use softer proportional torque instead of constant torque
    # 3. Reduce positive max drift vs APCR1c
    # 4. Avoid excessive negative overshoot
    # 5. Keep amplitude smaller than APCR1c (P2P < 0.20 m)
    # 6. Preserve stability
    #
    # Key differences from APCR1c:
    # - Torque mode: proportional_soft_band instead of state-machine bang-bang
    # - Entry threshold: 0.05 m (APCR1c: 0.08 m)
    # - Exit deadband: 0.02 m (APCR1c: 0.07 m)
    # - Full torque at: 0.08 m (APCR1c: constant 1.0 Nm)
    # - Max torque: 0.75 Nm (APCR1c: 1.0 Nm)
    # - Velocity decay: enabled with 0.5 factor
    # - Symmetry: inherently symmetric via abs(error)
    "APCR1d_symmetric_soft_band_control": SagittalAuthoritySchedule(
        profile_name="APCR1d_symmetric_soft_band_control",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Active Pitch Recovery (APCR_strategy) - with proportional soft band mode
        enable_active_pitch_crossing=True,
        active_pitch_crossing_recovery_gate_mode=True,  # same as APCR1/APCR1b/APCR1c
        # Proportional soft band mode parameters
        apc_proportional_soft_band_mode=True,  # KEY: enable proportional mode
        apc_soft_enter_m=0.05,  # m - enter soft recenter when |error| > this
        apc_inner_exit_m=0.02,  # m - exit when |error| <= this
        apc_opposite_overshoot_m=0.00,  # m - no overshoot allowance
        apc_pitch_enter_rad=0.03,  # rad - pitch threshold to enter
        apc_pitch_safe_limit_rad=0.08,  # rad - reduce torque if pitch exceeds this
        apc_max_cross_tau=0.75,  # Nm - max torque (lower than APCR1c's 1.0)
        apc_smooth_alpha=0.10,
        apc_max_rate_per_step=0.30,  # Nm/step - rate limit
        apc_contact_gate=True,
        apc_height_gate=True,
        apc_roll_gate=True,
        apc_min_com_z_m=0.28,  # m
        apc_max_com_z_m=0.50,  # m
        apc_pitch_safe_threshold_rad=0.05,  # rad
        apc_pitch_danger_threshold_rad=0.10,  # rad
        apc_roll_threshold_rad=0.15,  # rad
        # Velocity decay parameters
        apc_velocity_decay_enabled=True,  # KEY: enable velocity decay
        apc_velocity_decay_factor=0.5,  # reduce torque by 50% when moving toward zero
        # APCR recovery gate parameters
        apcr_pitch_hard_stop_rad=0.30,  # rad - hard stop, blocks APCR
        apcr_roll_hard_stop_rad=0.15,  # rad - lateral stability
        apcr_min_com_z_m=0.27,  # m - minimum safe height
        apcr_max_com_z_m=0.50,  # m - maximum operating height
    ),

    # APCR1e_adaptive_symmetric_soft_band
    # Adaptive authority profile that automatically increases torque when error is not improving
    # Based on APCR1d but with adaptive max_tau that can increase from base_tau to max_tau
    "APCR1e_adaptive_symmetric_soft_band": SagittalAuthoritySchedule(
        profile_name="APCR1e_adaptive_symmetric_soft_band",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Active Pitch Recovery (APCR_strategy) - with adaptive authority mode
        enable_active_pitch_crossing=True,
        active_pitch_crossing_recovery_gate_mode=True,  # same as APCR1 family
        # Proportional soft band mode parameters
        apc_proportional_soft_band_mode=True,  # KEY: enable proportional mode
        apc_soft_enter_m=0.045,  # m - enter soft recenter when |error| > this
        apc_inner_exit_m=0.02,  # m - exit when |error| <= this
        apc_opposite_overshoot_m=0.00,  # m - no overshoot allowance
        apc_pitch_enter_rad=0.03,  # rad - pitch threshold to enter
        apc_pitch_safe_limit_rad=0.08,  # rad - reduce torque if pitch exceeds this
        apc_smooth_alpha=0.10,
        apc_max_rate_per_step=0.30,  # Nm/step - rate limit
        apc_contact_gate=True,
        apc_height_gate=True,
        apc_roll_gate=True,
        apc_min_com_z_m=0.28,  # m
        apc_max_com_z_m=0.50,  # m
        apc_pitch_safe_threshold_rad=0.05,  # rad
        apc_pitch_danger_threshold_rad=0.10,  # rad
        apc_roll_threshold_rad=0.15,  # rad
        # Velocity decay parameters
        apc_velocity_decay_enabled=True,  # KEY: enable velocity decay
        apc_velocity_decay_factor=0.5,  # reduce torque by 50% when moving toward zero
        # APCR1e adaptive authority parameters
        apc_adaptive_authority_enabled=True,  # KEY: enable adaptive authority
        apc_adaptive_base_tau=0.55,  # Nm - base starting torque
        apc_adaptive_max_tau=1.20,  # Nm - maximum adaptive torque
        apc_adaptive_boost_tau_max=0.65,  # Nm - maximum boost above base
        apc_adaptive_boost_start_error_m=0.06,  # m - error threshold for boost
        apc_adaptive_full_boost_error_m=0.12,  # m - error for full boost
        apc_adaptive_no_improvement_window_steps=8,  # steps without improvement
        apc_adaptive_startup_boost_steps=50,  # startup phase duration
        apc_adaptive_startup_boost_max_tau=1.0,  # Nm - max startup torque
        apc_adaptive_disable_vd_when_abs_e_gt=0.10,  # m - disable VD above this
        apc_adaptive_disable_vd_during_startup=True,  # disable VD in startup
        # APCR recovery gate parameters
        apcr_pitch_hard_stop_rad=0.30,  # rad - hard stop, blocks APCR
        apcr_roll_hard_stop_rad=0.15,  # rad - lateral stability
        apcr_min_com_z_m=0.27,  # m - minimum safe height
        apcr_max_com_z_m=0.50,  # m - maximum operating height
    ),

    # APCR1f_adaptive_fast_response_phase_brake
    # Adaptive fast response with phase-aware braking
    # Key differences from APCR1e:
    # - Earlier intervention at 0.035m vs 0.05m
    # - Faster rate limit 0.55 Nm/step vs 0.35 Nm/step
    # - Higher max_tau 1.40 Nm vs 1.20 Nm
    # - Phase brake when error returning toward zero
    # - Boost when error growing 3+ consecutive steps
    "APCR1f_adaptive_fast_response_phase_brake": SagittalAuthoritySchedule(
        profile_name="APCR1f_adaptive_fast_response_phase_brake",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Active Pitch Recovery (APCR_strategy) - with fast response + phase brake
        enable_active_pitch_crossing=True,
        active_pitch_crossing_recovery_gate_mode=True,  # same as APCR1 family
        # Proportional soft band mode parameters
        apc_proportional_soft_band_mode=True,  # KEY: enable proportional mode
        apc_soft_enter_m=0.035,  # m - CHANGED: earlier entry than APCR1e's 0.045
        apc_inner_exit_m=0.015,  # m - earlier exit
        apc_opposite_overshoot_m=0.00,  # m - no overshoot allowance
        apc_pitch_enter_rad=0.03,  # rad - pitch threshold to enter
        apc_pitch_safe_limit_rad=0.08,  # rad - reduce torque if pitch exceeds this
        apc_smooth_alpha=0.18,  # CHANGED: more responsive smoothing
        apc_max_rate_per_step=0.55,  # CHANGED: faster response
        apc_contact_gate=True,
        apc_height_gate=True,
        apc_roll_gate=True,
        apc_min_com_z_m=0.28,  # m
        apc_max_com_z_m=0.50,  # m
        apc_pitch_safe_threshold_rad=0.05,  # rad
        apc_pitch_danger_threshold_rad=0.10,  # rad
        apc_roll_threshold_rad=0.15,  # rad
        # Velocity decay parameters
        apc_velocity_decay_enabled=True,  # KEY: enable velocity decay
        apc_velocity_decay_factor=0.5,  # reduce torque by 50% when moving toward zero
        # APCR1e adaptive authority parameters (base settings)
        apc_adaptive_authority_enabled=True,  # KEY: enable adaptive authority
        apc_adaptive_base_tau=0.45,  # Nm - CHANGED: slightly lower base
        apc_adaptive_max_tau=1.40,  # Nm - CHANGED: higher ceiling
        apc_adaptive_boost_tau_max=0.95,  # Nm - CHANGED: larger boost capability
        apc_adaptive_boost_start_error_m=0.06,  # m - error threshold for boost
        apc_adaptive_full_boost_error_m=0.10,  # m - CHANGED: full boost at smaller error
        apc_adaptive_no_improvement_window_steps=5,  # CHANGED: faster boost (5 vs 8)
        apc_adaptive_startup_boost_steps=50,  # startup phase duration
        apc_adaptive_startup_boost_max_tau=1.20,  # Nm - CHANGED: higher startup authority
        apc_adaptive_disable_vd_when_abs_e_gt=0.10,  # m - disable VD above this
        apc_adaptive_disable_vd_during_startup=True,  # disable VD in startup
        apc_adaptive_max_rate_per_step=0.55,  # CHANGED: faster rate
        # APCR1f fast response with phase brake parameters
        apc_fast_response_enabled=True,  # KEY: enable fast response
        apc_phase_brake_enabled=True,  # KEY: enable phase brake
        apc_phase_brake_threshold_m=0.08,  # m - apply brake below this
        apc_phase_brake_damping_factor=0.6,  # reduce scale by this when braking
        apc_boost_rate_per_step=0.25,  # Nm/step - rate for adaptive boost
        apc_decay_rate_per_step=0.45,  # Nm/step - faster decay when returning
        apc_increasing_error_threshold_steps=3,  # boost when error grows 3+ steps
        apc_increasing_error_boost_factor=0.3,  # boost factor for growing error
        apc_fast_response_inner_deadband_m=0.015,  # m - earlier deadband
        apc_fast_response_soft_enter_m=0.035,  # m - earlier soft enter
        apc_fast_response_desired_band_m=0.08,  # m - wider comfortable band
        apc_fast_response_full_torque_m=0.10,  # m - full torque at this error
        apc_fast_response_emergency_m=0.12,  # m - emergency mode trigger
        apc_fast_response_base_tau=0.45,  # Nm - slightly lower base
        apc_fast_response_max_tau=1.40,  # Nm - higher ceiling
        apc_fast_response_boost_tau_max=0.95,  # Nm - larger boost capability
        apc_fast_response_startup_boost_max_tau=1.20,  # Nm - higher startup authority
        apc_fast_response_max_rate_per_step=0.55,  # Nm/step - faster response
        apc_fast_response_smooth_alpha=0.18,  # more responsive smoothing
        apc_fast_response_no_improvement_window=5,  # faster boost (5 vs 8 steps)
        # APCR recovery gate parameters
        apcr_pitch_hard_stop_rad=0.30,  # rad - hard stop, blocks APCR
        apcr_roll_hard_stop_rad=0.15,  # rad - lateral stability
        apcr_min_com_z_m=0.27,  # m - minimum safe height
        apcr_max_com_z_m=0.50,  # m - maximum operating height
    ),

    # APCR1g_predictive_fast_response_phase_brake
    # Predictive fast response with phase-aware braking
    # Key differences from APCR1f:
    # - Predictive error: e_pred = e + lead_time_s * e_dot
    # - Earlier activation when predicted error exceeds threshold
    # - Predictive boost when predicted error indicates future overshoot
    # - Stronger phase brake with two thresholds
    # - Higher max_tau (1.55 vs 1.40), faster rate (0.70 vs 0.55)
    "APCR1g_predictive_fast_response_phase_brake": SagittalAuthoritySchedule(
        profile_name="APCR1g_predictive_fast_response_phase_brake",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Active Pitch Recovery (APCR_strategy) - with predictive fast response
        enable_active_pitch_crossing=True,
        active_pitch_crossing_recovery_gate_mode=True,
        # Proportional soft band mode parameters
        apc_proportional_soft_band_mode=True,  # KEY: enable proportional mode
        apc_soft_enter_m=0.030,  # m - CHANGED: earlier entry than APCR1f's 0.035
        apc_inner_exit_m=0.012,  # m - earlier exit
        apc_opposite_overshoot_m=0.00,  # m - no overshoot allowance
        apc_pitch_enter_rad=0.03,  # rad - pitch threshold to enter
        apc_pitch_safe_limit_rad=0.08,  # rad - reduce torque if pitch exceeds this
        apc_smooth_alpha=0.22,  # CHANGED: more responsive smoothing
        apc_max_rate_per_step=0.70,  # CHANGED: faster response
        apc_contact_gate=True,
        apc_height_gate=True,
        apc_roll_gate=True,
        apc_min_com_z_m=0.28,  # m
        apc_max_com_z_m=0.50,  # m
        apc_pitch_safe_threshold_rad=0.05,  # rad
        apc_pitch_danger_threshold_rad=0.10,  # rad
        apc_roll_threshold_rad=0.15,  # rad
        # Velocity decay parameters
        apc_velocity_decay_enabled=True,  # KEY: enable velocity decay
        apc_velocity_decay_factor=0.5,  # reduce torque by 50% when moving toward zero
        # APCR1e adaptive authority parameters (base settings)
        apc_adaptive_authority_enabled=True,  # KEY: enable adaptive authority
        apc_adaptive_base_tau=0.45,  # Nm - slightly lower base
        apc_adaptive_max_tau=1.55,  # Nm - CHANGED: higher ceiling
        apc_adaptive_boost_tau_max=1.10,  # Nm - CHANGED: larger boost capability
        apc_adaptive_boost_start_error_m=0.06,  # m - error threshold for boost
        apc_adaptive_full_boost_error_m=0.095,  # m - full boost at smaller error
        apc_adaptive_no_improvement_window_steps=4,  # CHANGED: faster boost (4 vs 5)
        apc_adaptive_startup_boost_steps=50,  # startup phase duration
        apc_adaptive_startup_boost_max_tau=1.25,  # Nm - CHANGED: higher startup authority
        apc_adaptive_disable_vd_when_abs_e_gt=0.10,  # m - disable VD above this
        apc_adaptive_disable_vd_during_startup=True,  # disable VD in startup
        apc_adaptive_max_rate_per_step=0.70,  # CHANGED: faster rate
        # APCR1f fast response with phase brake parameters
        apc_fast_response_enabled=True,  # KEY: enable fast response
        apc_phase_brake_enabled=True,  # KEY: enable phase brake
        apc_phase_brake_threshold_m=0.075,  # m - CHANGED: tighter threshold
        apc_phase_brake_damping_factor=0.55,  # CHANGED: stronger damping
        apc_boost_rate_per_step=0.35,  # Nm/step - CHANGED: faster boost rate
        apc_decay_rate_per_step=0.55,  # Nm/step - faster decay when returning
        apc_increasing_error_threshold_steps=2,  # CHANGED: faster boost (2 vs 3)
        apc_increasing_error_boost_factor=0.35,  # CHANGED: higher factor
        apc_fast_response_inner_deadband_m=0.012,  # m - earlier deadband
        apc_fast_response_soft_enter_m=0.030,  # m - earlier soft enter
        apc_fast_response_desired_band_m=0.075,  # m - tighter band
        apc_fast_response_full_torque_m=0.095,  # m - full torque at smaller error
        apc_fast_response_emergency_m=0.115,  # m - emergency mode trigger
        apc_fast_response_base_tau=0.45,  # Nm - slightly lower base
        apc_fast_response_max_tau=1.55,  # Nm - higher ceiling
        apc_fast_response_boost_tau_max=1.10,  # Nm - larger boost capability
        apc_fast_response_startup_boost_max_tau=1.25,  # Nm - higher startup authority
        apc_fast_response_max_rate_per_step=0.70,  # Nm/step - faster response
        apc_fast_response_smooth_alpha=0.22,  # more responsive smoothing
        apc_fast_response_no_improvement_window=4,  # faster boost (4 vs 5 steps)
        # APCR1g predictive fast response parameters
        apc_predictive_enabled=True,  # KEY: enable predictive error logic
        apc_lead_time_s=0.10,  # seconds to predict ahead
        apc_predicted_enter_m=0.07,  # activate when abs_pred > this AND moving_away
        apc_predicted_full_response_m=0.10,  # boost authority when abs_pred > this
        apc_predicted_emergency_m=0.12,  # emergency mode when abs_pred > this
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
        # APCR recovery gate parameters
        apcr_pitch_hard_stop_rad=0.30,  # rad - hard stop, blocks APCR
        apcr_roll_hard_stop_rad=0.15,  # rad - lateral stability
        apcr_min_com_z_m=0.27,  # m - minimum safe height
        apcr_max_com_z_m=0.50,  # m - maximum operating height
    ),

    # APCR1h_support_drift_priority_fast_recenter
    # Based on APCR1f (correct torque sign), NOT APCR1g (wrong torque sign)
    # Key changes:
    # - Correct torque sign for support recovery (negative tau when drift > 0)
    # - Drift priority override: higher torque + disabled phase brake when drift runaway
    # - Emergency clamp: when drift > 0.12m
    # - Monitor wheel velocity but do NOT penalize for drift reduction
    "APCR1h_support_drift_priority_fast_recenter": SagittalAuthoritySchedule(
        profile_name="APCR1h_support_drift_priority_fast_recenter",
        applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
        # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=4.0,
        velocity_damping_scale=1.10,
        # Active Pitch Recovery (APCR_strategy) - with drift priority
        enable_active_pitch_crossing=True,
        active_pitch_crossing_recovery_gate_mode=True,
        # Proportional soft band mode parameters
        apc_proportional_soft_band_mode=True,
        apc_soft_enter_m=0.030,  # m - soft enter threshold
        apc_inner_exit_m=0.015,  # m - inner exit threshold
        apc_opposite_overshoot_m=0.00,  # m - no overshoot allowance
        apc_pitch_enter_rad=0.03,  # rad - pitch threshold to enter
        apc_pitch_safe_limit_rad=0.08,  # rad - reduce torque if pitch exceeds this
        apc_smooth_alpha=0.18,  # responsive smoothing
        apc_max_rate_per_step=0.55,  # normal APCR response rate
        apc_contact_gate=True,
        apc_height_gate=True,
        apc_roll_gate=True,
        apc_min_com_z_m=0.28,  # m
        apc_max_com_z_m=0.50,  # m
        apc_pitch_safe_threshold_rad=0.05,  # rad
        apc_pitch_danger_threshold_rad=0.10,  # rad
        apc_roll_threshold_rad=0.15,  # rad
        # Velocity decay parameters
        apc_velocity_decay_enabled=True,
        apc_velocity_decay_factor=0.5,  # reduce torque by 50% when moving toward zero
        # APCR1f adaptive authority parameters (base settings - CORRECT torque sign)
        apc_adaptive_authority_enabled=True,
        apc_adaptive_base_tau=0.45,  # Nm
        apc_adaptive_max_tau=1.25,  # Nm - normal APCR ceiling
        apc_adaptive_boost_tau_max=0.95,  # Nm
        apc_adaptive_boost_start_error_m=0.06,  # m
        apc_adaptive_full_boost_error_m=0.095,  # m
        apc_adaptive_no_improvement_window_steps=5,
        apc_adaptive_startup_boost_steps=500,  # startup phase duration
        apc_adaptive_startup_boost_max_tau=1.60,  # Nm - higher startup authority
        apc_adaptive_disable_vd_when_abs_e_gt=0.10,  # m
        apc_adaptive_disable_vd_during_startup=True,
        apc_adaptive_max_rate_per_step=0.55,  # normal rate
        # APCR1f fast response with phase brake parameters
        apc_fast_response_enabled=True,
        apc_phase_brake_enabled=True,
        apc_phase_brake_threshold_m=0.06,  # m - phase brake when within this
        apc_phase_brake_damping_factor=0.60,  # phase brake damping
        apc_boost_rate_per_step=0.35,  # Nm/step
        apc_decay_rate_per_step=0.55,  # Nm/step
        apc_increasing_error_threshold_steps=3,
        apc_increasing_error_boost_factor=0.30,
        apc_fast_response_inner_deadband_m=0.015,
        apc_fast_response_soft_enter_m=0.030,
        apc_fast_response_desired_band_m=0.08,
        apc_fast_response_full_torque_m=0.10,
        apc_fast_response_emergency_m=0.12,
        apc_fast_response_base_tau=0.45,  # Nm
        apc_fast_response_max_tau=1.25,  # Nm - normal APCR ceiling
        apc_fast_response_boost_tau_max=0.95,  # Nm
        apc_fast_response_startup_boost_max_tau=1.60,  # Nm - higher startup
        apc_fast_response_max_rate_per_step=0.55,  # Nm/step
        apc_fast_response_smooth_alpha=0.18,
        apc_fast_response_no_improvement_window=5,
        # APCR1h DRIFT PRIORITY parameters (NEW)
        # These override normal APCR when drift exceeds threshold
        apc_drift_priority_enabled=True,  # KEY: enable drift priority
        apc_drift_priority_enter_m=0.08,  # m - drift priority activates
        apc_drift_priority_normal_max_tau=1.25,  # Nm - normal max (higher than APCR1f)
        apc_drift_priority_drift_priority_max_tau=1.65,  # Nm - drift priority max
        apc_drift_priority_emergency_max_tau=1.85,  # Nm - emergency clamp max
        apc_drift_priority_startup_max_tau=1.60,  # Nm - startup boost max
        apc_drift_priority_normal_rate=0.55,  # Nm/step - normal rate
        apc_drift_priority_drift_priority_rate=0.85,  # Nm/step - drift priority rate
        apc_drift_priority_emergency_rate=1.00,  # Nm/step - emergency rate
        apc_drift_priority_decay_rate=0.55,  # Nm/step - decay rate
        apc_drift_priority_phase_brake_disable_threshold_m=0.10,  # disable phase brake above this
        apc_drift_priority_base_tau=0.45,  # Nm - base torque
        apc_drift_priority_emergency_m=0.12,  # m - emergency clamp threshold
        apc_drift_priority_hard_m=0.15,  # m - hard safety threshold
        # APCR recovery gate parameters
        apcr_pitch_hard_stop_rad=0.30,  # rad - hard stop, blocks APCR
        apcr_roll_hard_stop_rad=0.15,  # rad - lateral stability
        apcr_min_com_z_m=0.27,  # m - minimum safe height
        apcr_max_com_z_m=0.50,  # m - maximum operating height
    ),

    # APCR1i_support_hysteresis_recenter
    # Symmetric hysteresis state machine for support drift recenter
    # Key principle: hold recenter until error crosses near zero, then symmetric release
    "APCR1i_support_hysteresis_recenter": SagittalAuthoritySchedule(
        profile_name="APCR1i_support_hysteresis_recenter",
        applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
        # Enable APCR for this profile
        enable_active_pitch_crossing=True,
        # WIDER pitch safe threshold to allow entry during moderate pitch error
        # APCR1i prioritizes drift recovery over pitch - pitch danger still blocks
        apc_pitch_safe_threshold_rad=0.15,  # 8.6 deg - wider than APCR1h (0.05 rad)
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
    ),

    # APCR1j_support_hysteresis_higher_authority
    # Based on APCR1i but with higher torque authority to overcome the 1.5 Nm universal cap
    # Root cause: APCR1i observed final APCR tau max = 1.5 Nm despite configured recenter_max_tau = 1.75 Nm
    # Fix: explicitly set apc_max_cross_tau = 2.0 so hysteresis can reach 2.0 Nm
    "APCR1j_support_hysteresis_higher_authority": SagittalAuthoritySchedule(
        profile_name="APCR1j_support_hysteresis_higher_authority",
        applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
        # Enable APCR for this profile
        enable_active_pitch_crossing=True,
        # CRITICAL FIX: set apc_max_cross_tau = 2.0 to override the 1.5 Nm universal cap
        apc_max_cross_tau=2.0,  # Nm - universal crossing torque cap (was 1.5 in APCR1i)
        # WIDER pitch safe threshold to allow entry during moderate pitch error
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
    ),

    # APCR1k_support_hysteresis_early_entry
    # Based on APCR1j but with LOWER outer entry threshold to catch drift earlier
    # Root cause: APCR1j analysis showed RECENTER starts at step 58 (e=0.0817m) allowing momentum buildup
    # Fix: lower outer_enter_m from 0.08 to 0.05 to start RECENTER at step 46 (e=0.0521m)
    # Keep same torque authority as APCR1j (2.0 Nm max)
    "APCR1k_support_hysteresis_early_entry": SagittalAuthoritySchedule(
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
    ),
    # APCR1m_conditional_pitch_blend_recenter
    # Conditional pitch blending instead of hard suppression
    # Blend tau_pitch based on error magnitude, with startup guard and safety gates
    "APCR1m_conditional_pitch_blend_recenter": SagittalAuthoritySchedule(
        profile_name="APCR1m_conditional_pitch_blend_recenter",
        applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
        enable_active_pitch_crossing=True,
        apc_max_cross_tau=2.0,  # Nm - same as APCR1k
        apc_pitch_safe_threshold_rad=0.15,  # 8.6 deg - same as APCR1k
        apc_pitch_danger_threshold_rad=0.30,  # hard block - same as APCR1k
        apc_outer_enter_m=0.05,  # Enter crossing - same as APCR1k
        apc_inner_exit_m=0.03,  # Exit crossing - same as APCR1k
        apc_hysteresis_enabled=True,
        apc_hysteresis_outer_enter_m=0.05,  # Enter recenter - same as APCR1k
        apc_hysteresis_inner_exit_m=0.03,  # Exit recenter - same as APCR1k
        apc_hysteresis_opposite_release_m=0.03,
        apc_hysteresis_near_zero_m=0.01,
        apc_hysteresis_emergency_m=0.12,
        apc_hysteresis_hard_m=0.15,
        apc_hysteresis_base_tau=0.45,
        apc_hysteresis_recenter_max_tau=2.0,
        apc_hysteresis_emergency_max_tau=2.2,
        apc_hysteresis_hold_max_tau=1.75,
        apc_hysteresis_normal_rate=0.40,
        apc_hysteresis_recenter_rate=1.1,
        apc_hysteresis_emergency_rate=1.3,
        apc_hysteresis_decay_rate=0.50,
        apc_hysteresis_phase_brake_threshold_m=0.05,
        apc_hysteresis_phase_brake_disable_in_recenter=True,
        # KEY: conditional pitch blend parameters
        apc_pitch_blend_enabled=True,
        apc_pitch_blend_startup_guard_steps=100,
        apc_pitch_blend_safe_pitch_rad=0.15,
        apc_pitch_blend_safe_pitch_rate_rad_s=0.5,
        apc_pitch_blend_min_com_z=0.27,
        apc_pitch_blend_max_roll_rad=0.15,
        apc_pitch_blend_deep_error_m=0.12,
        apc_pitch_blend_mid_error_m=0.08,
        apc_pitch_blend_soft_error_m=0.05,
        apc_pitch_blend_scale_deep=0.0,
        apc_pitch_blend_scale_mid=0.25,
        apc_pitch_blend_scale_soft=0.5,
        apc_pitch_blend_scale_near=1.0,
        # Safety gates
        apc_contact_gate=True,
        apc_height_gate=True,
        apc_roll_gate=True,
        apc_min_com_z_m=0.27,
        apc_max_com_z_m=0.50,
        apc_roll_threshold_rad=0.15,
    ),
    # APCR1n_recenter_priority_torque_boost
    # Based on APCR1h with wheel damping override and position cap boost during RECENTER
    "APCR1n_recenter_priority_torque_boost": SagittalAuthoritySchedule(
        profile_name="APCR1n_recenter_priority_torque_boost",
        applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
        # APCR1h base configuration
        continuous_max_position_tau=True,  # Added: must match APCR1h
        max_position_tau_nominal=4.0,  # Added: must match APCR1h (was 3.0 in initial design)
        velocity_damping_scale=1.10,  # Added: must match APCR1h
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
        position_cap_normal_nm=4.0,  # FIXED: Was 3.0, should match APCR1h's 4.0
        position_cap_recenter_nm=5.0,
        position_cap_emergency_nm=6.0,
        position_cap_ramp_steps=50,
        recenter_priority_safe_min_com_z=0.27,
        recenter_priority_safe_roll_rad=0.15,
        recenter_priority_safe_pitch_rad=0.15,
    ),
    # APCR1nD_direct_support_recenter_features
    # Based on APCR1n but with DIRECT support drift trigger instead of APC dependency
    # KEY DIFFERENCE: Does NOT require enable_active_pitch_crossing=True
    # This fixes the issue where APCR1n features never activated because APC was disabled
    "APCR1nD_direct_support_recenter_features": SagittalAuthoritySchedule(
        profile_name="APCR1nD_direct_support_recenter_features",
        applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
        # APCR1h base configuration (same as APCR1n)
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        velocity_damping_scale=1.10,
        # APCR1n recenter priority features (same as APCR1n)
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
        # APCR1nD: Direct support drift trigger (KEY NEW FEATURE)
        recenter_priority_direct_enabled=True,
        recenter_priority_direct_enter_m=0.08,
        recenter_priority_direct_emergency_m=0.12,
        recenter_priority_direct_hard_m=0.15,
        recenter_priority_direct_exit_m=0.02,
    ),
    # APCR1nD Tuned Variants (Phase 3-4)
    "APCR1nD_T1_early_entry": APCR1ND_T1_EARLY_ENTRY,
    "APCR1nD_T2_hold_outside_band": APCR1ND_T2_HOLD_OUTSIDE_BAND,
    "APCR1nD_T3_early_entry_plus_hold": APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD,
    "APCR1nD_T4_stronger_authority": APCR1ND_T4_STRONGER_AUTHORITY,
    "APCR1nD_T5_band_limited_balanced": BAND_LIMITED_SUPPORT_RECENTER,  # legacy alias
    "band_limited_support_recenter": BAND_LIMITED_SUPPORT_RECENTER,     # semantic
    # T6 High-Height Transient Suppression Variants (Phase 5)
    "T6A_high_early_hard_band": T6A_HIGH_EARLY_HARD_BAND,
    "T6B_high_stronger_emergency": T6B_HIGH_STRONGER_EMERGENCY,
    "T6C_high_early_plus_stronger": T6C_HIGH_EARLY_PLUS_STRONGER,
    "T6D_high_transient_boost": T6D_HIGH_TRANSIENT_BOOST,
    "T6E_high_pitch_aware_boost": T6E_HIGH_PITCH_AWARE_BOOST,
    "T6F_budget_cap_raise": EMERGENCY_BUDGET_CAP_RAISE,            # legacy alias
    "emergency_budget_cap_raise": EMERGENCY_BUDGET_CAP_RAISE,      # semantic
    "T6F_sign_corrected": T6F_SIGN_CORRECTED,
    "T6H_soft_blend_arch_fix": T6H_SOFT_BLEND_ARCH_FIX,
    "T6I_phase_aware_release": PHASE_AWARE_AUTHORITY_RELEASE,      # legacy alias
    "phase_aware_authority_release": PHASE_AWARE_AUTHORITY_RELEASE,  # semantic
    "T6J_centering_bias_trim": SUPPORT_CENTERING_BIAS_TRIM,          # legacy alias
    "support_centering_bias_trim": SUPPORT_CENTERING_BIAS_TRIM,    # semantic
    "adaptive_support_centering_trim": ADAPTIVE_SUPPORT_CENTERING_TRIM,  # opt-in adaptive trim
    "zero_crossing_support_recenter": ZERO_CROSSING_SUPPORT_RECENTER,  # ZC hysteresis recenter
    "early_zero_crossing_recenter": EARLY_ZERO_CROSSING_RECENTER,  # Early ZC: exits at zero, not opposite side
    "early_zero_crossing_recenter_v2": EARLY_ZERO_CROSSING_RECENTER_V2,  # V2: anti-rebound fix
    "pitch_bias_compensated_zero_crossing_recenter": PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER,  # Phase 7: EZC V2 + pitch DC bias compensation
    "pitch_equilibrium_trim": PITCH_EQUILIBRIUM_TRIM,  # Phase 3 structural fix: shift pitch reference to equilibrium to center support drift
    "height_scheduled_pitch_equilibrium_trim": HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM,  # Phase 2 structural fix: per-height scheduled offset
    "support_position_outer_loop_pitch_ref": SUPPORT_POSITION_OUTER_LOOP_PITCH_REF,  # Phase B dynamic centering outer loop
    "calibrated_support_position_outer_loop_pitch_ref": CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF,  # Phase B calibration: height-dependent outer-loop gains
    "calibrated_support_position_outer_loop_pitch_ref_v2": CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF_V2,  # Phase B calibration v2: smoothed upper-band Kp
    "physics_equilibrium_feedforward_outer_loop": PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP,  # Phase D: physics-based equilibrium wheel torque feedforward
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v1": PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V1,  # PFF low-band support correction candidate
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2": PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,  # PFF low-band support correction v2 candidate
    # I_SUPPORT_REFERENCE_REACQUISITION_V1 — opt-in diagnostic candidate (support reference blend fix)
    "i_support_reference_reacquisition_v1": I_SUPPORT_REFERENCE_REACQUISITION_V1,  # I1 candidate
    # J_TALL_HEIGHT_SAGITTAL_WIP_DAMPING_V1 family — opt-in diagnostic candidates.
    # Increases pitch-rate damping (kd_pitch) and/or wheel velocity damping
    # (k_wheel_velocity) at tall heights via continuous height scheduling.
    # All use the same low-band v2 sagittal base as D_MODE_HIP_YAW_DIV_V1.
    "j1a_tall_kd_pitch_v1": J1A_TALL_KD_PITCH_V1,
    "j1b_tall_kd_pitch_v1": J1B_TALL_KD_PITCH_V1,
    "j1c_tall_kd_pitch_v1": J1C_TALL_KD_PITCH_V1,
    "j2a_tall_k_wheel_vel_v1": J2A_TALL_K_WHEEL_VEL_V1,
    "j2b_tall_k_wheel_vel_v1": J2B_TALL_K_WHEEL_VEL_V1,
    "j2c_tall_k_wheel_vel_v1": J2C_TALL_K_WHEEL_VEL_V1,
    "j3a_tall_combined_v1": J3A_TALL_COMBINED_V1,
    "j3b_tall_combined_v1": J3B_TALL_COMBINED_V1,
    # K_TARGETED_2P5HZ_WIP_NOTCH_V1 family — opt-in diagnostic candidates.
    # Applies a causal IIR biquad notch filter around ~2.5 Hz on selected
    # damping input signals to prevent phase-lagged damping from feeding
    # the WIP oscillation mode. Uses the same low-band v2 sagittal base.
    "k1_pitch_rate_notch_v1": K1_PITCH_RATE_NOTCH,
    "k1b_pitch_rate_notch_2p3": K1B_PITCH_RATE_NOTCH_2P3,
    "k1c_pitch_rate_notch_2p7": K1C_PITCH_RATE_NOTCH_2P7,
    "k1d_pitch_rate_notch_q4": K1D_PITCH_RATE_NOTCH_Q4,
    "k1e_pitch_rate_notch_q8": K1E_PITCH_RATE_NOTCH_Q8,
    "k1f_pitch_rate_notch_blend075": K1F_PITCH_RATE_NOTCH_BLEND075,
    "k1g_pitch_rate_notch_blend050": K1G_PITCH_RATE_NOTCH_BLEND050,
    "k2_notch_low_q_v1": K2_NOTCH_LOW_Q_V1,
    "k2_wheel_vel_notch_v1": K2_WHEEL_VEL_NOTCH,
    "k3_pitch_rate_wheel_vel_notch_v1": K3_PITCH_RATE_WHEEL_VEL_NOTCH,
    "k3b_pitch_rate_wheel_vel_notch_blend075": K3B_PITCH_RATE_WHEEL_VEL_NOTCH_BLEND075,
    # K_SWEEP audit-only filter parameter sweep profiles
    **ALL_K_SWEEP_PROFILES,
    # L_K1_COORDINATED_SAGITTAL_STATE_FEEDBACK_V1 family — Phase 3
    # K1 + coordinated sagittal state feedback for sustained posture recovery
    "l1_k1_coordinated_low_freq_feedback_v1": L1_K1_COORDINATED_LOW_FREQ_FEEDBACK,
    "l2_k1_coordinated_phase_lead_v1": L2_K1_COORDINATED_PHASE_LEAD,
    "l3_k1_coordinated_pitch_ref_stabilization_v1": L3_K1_COORDINATED_PITCH_REF_STABILIZATION,
    # LR_K1_REPLACEMENT_COORDINATED_FEEDBACK_V1 family — Replacement architecture
    # K1 + replacement coordinated sagittal state feedback (not additive like L)
    "lr1_k1_replacement_coordinated_low_freq_v1": LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
    "lr2_k1_replacement_phase_lead_v1": LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
    "lr3_k1_replacement_pitch_ref_stabilized_v1": LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1,
    # LRS family — Sign-audited constrained gain sweep (2026-06-24)
    # All signs confirmed correct. Failure mode: gain magnitude.
    "lrs1_support_dominant_v1": LRS1_SUPPORT_DOMINANT_V1,
    "lrs2_pitch_rate_damping_v1": LRS2_PITCH_RATE_DAMPING_V1,
    "lrs3_balanced_medium_v1": LRS3_BALANCED_MEDIUM_V1,
    # LP family — Priority Sagittal Allocator (2026-06-24)
    # Pitch-first support-residual architecture. Resolves LR/LRS support-pitch
    # coupling via priority-based torque allocation.
    "lp1_k1_priority_pitch_first_support_residual_v1": LP1_K1_PRIORITY_PITCH_FIRST_SUPPORT_RESIDUAL_V1,
    "lp2_k1_priority_pitch_strong_support_soft_v1": LP2_K1_PRIORITY_PITCH_STRONG_SUPPORT_SOFT_V1,
    "lp3_k1_priority_support_recenter_when_safe_v1": LP3_K1_PRIORITY_SUPPORT_RECENTER_WHEN_SAFE_V1,
    # M_K1_BODY_YAW_CORRECT_ACTUATOR_V1 family — Phase 4
    # K1 + body-yaw/wheel-yaw correct-actuator fix for D4/D5 hip-yaw
    "m1_k1_body_yaw_diff_wheel_v1": M1_K1_BODY_YAW_DIFF_WHEEL_V1,
    "m2_k1_body_yaw_support_aware_v1": M2_K1_BODY_YAW_SUPPORT_AWARE_V1,
    # N_K1_MILD_DAMPING_DIAGNOSTIC_V1 — Phase 5 (optional diagnostic)
    "n1_k1_mild_phase_lead_damping_v1": N1_K1_MILD_PHASE_LEAD_DAMPING,
    # N_K1_MILD_DAMPING_DIAGNOSTIC_V1 — N1 micro-sweep variants
    "n1b_k1_mild_phase_lead_v1": N1B_K1_MILD_PHASE_LEAD_V1,
    "n1c_k1_mild_phase_lead_v1": N1C_K1_MILD_PHASE_LEAD_V1,
    "n1d_k1_mild_phase_lead_v1": N1D_K1_MILD_PHASE_LEAD_V1,
    # D_MODE_HIP_YAW_DIV_V1 — current-best architecture-correct candidate.
    # Resolves to the low-band v2 sagittal schedule; the divergence-mode
    # controller is enabled separately at runtime via --enable-mode-hip-yaw-divergence.
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1": PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,  # current-best candidate (sagittal + mode-div flags)
    # E_MODE_HIP_YAW_DIV_PLUS_WHEEL_YAW_V1 — opt-in architecture-fix candidate.
    # Combines D's mode-based hip-yaw divergence controller with the differential
    # wheel-yaw stabilizer for body-yaw correction through the correct actuator path.
    # Resolves to the same low-band v2 sagittal schedule; wheel-yaw and mode-div
    # are enabled separately at runtime via CLI flags. Opt-in only — NOT default.
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_wheel_yaw_v1": PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,  # opt-in candidate (sagittal + mode-div + wheel-yaw flags)
    "unified_sagittal_state_feedback_no_offset": UNIFIED_SAGITTAL_STATE_FEEDBACK_NO_OFFSET,  # Unified sagittal state-feedback no-offset controller
}


def resolve_sagittal_authority_schedule(profile_name: str) -> SagittalAuthoritySchedule:
    return SAGITTAL_AUTHORITY_PROFILES[profile_name]


# =============================================================================
# Boundary Yaw-Position Coupling Fix Profiles (Phase 2)
# =============================================================================
# Profiles for fixing yaw-position coupling at boundary heights (0.300m, 0.480m).
# Only activates for boundary variants (low_0p300, high_0p480).
# All profiles preserve nominal/standard variant behavior.

class BoundaryYawPositionFixState:
    """Runtime state for boundary yaw-position fix profiles (per-simulation, not per-step)."""

    def __init__(
        self,
        profile: str,
        boundary_kp: float,
        boundary_kd: float,
        integral_gain: float,
        integral_max: float,
    ):
        self.profile = profile
        self.boundary_kp = boundary_kp
        self.boundary_kd = boundary_kd
        self.integral_gain = integral_gain
        self.integral_max = integral_max
        # Integral state per joint (left, right hip_yaw)
        self.integral_error_left = 0.0
        self.integral_error_right = 0.0
        # Bias torque applied per joint (for integral profile)
        self.bias_tau_left = 0.0
        self.bias_tau_right = 0.0

    def reset(self):
        """Reset integral state at start of simulation."""
        self.integral_error_left = 0.0
        self.integral_error_right = 0.0
        self.bias_tau_left = 0.0
        self.bias_tau_right = 0.0

    def is_boundary_variant(self, variant_name: str | None) -> bool:
        return variant_name in BOUNDARY_HEIGHT_VARIANTS

    def is_active(self, variant_name: str | None) -> bool:
        """Profile is active only for boundary variants AND non-baseline profiles."""
        return self.is_boundary_variant(variant_name) and self.profile != "baseline"

    def uses_yaw_aware_compensation(self) -> bool:
        return self.profile in (
            "yaw_aware_position_only",
            "yaw_aware_plus_boundary_hip_yaw",
            "yaw_aware_plus_integral_light",
        )

    def uses_boundary_hip_yaw(self) -> bool:
        return self.profile in (
            "boundary_hip_yaw_profile",
            "yaw_aware_plus_boundary_hip_yaw",
        )

    def uses_integral(self) -> bool:
        return self.profile in (
            "boundary_hip_yaw_integral_light",
            "yaw_aware_plus_integral_light",
        )

    def get_effective_hip_yaw_kp(self, default_kp: float, variant_name: str | None) -> float:
        """Return effective kp: higher for boundary variants if boundary profile active."""
        if self.is_active(variant_name) and self.uses_boundary_hip_yaw():
            return self.boundary_kp
        return default_kp

    def get_effective_hip_yaw_kd(self, default_kd: float, variant_name: str | None) -> float:
        """Return effective kd: higher for boundary variants if boundary profile active."""
        if self.is_active(variant_name) and self.uses_boundary_hip_yaw():
            return self.boundary_kd
        return default_kd

    def update_integral(
        self,
        l_hip_yaw_error: float,
        r_hip_yaw_error: float,
        dt: float,
    ) -> tuple[float, float, bool, bool]:
        """Update integral state and return bias torques. Returns (bias_left, bias_right, integral_active, clamp_active)."""
        if not self.uses_integral():
            return 0.0, 0.0, False, False

        # Integrate error for boundary variants
        self.integral_error_left += l_hip_yaw_error * dt
        self.integral_error_right += r_hip_yaw_error * dt

        # Anti-windup clamp
        clamp_active = False
        if abs(self.integral_error_left) > self.integral_max:
            self.integral_error_left = float(np.sign(self.integral_error_left) * self.integral_max)
            clamp_active = True
        if abs(self.integral_error_right) > self.integral_max:
            self.integral_error_right = float(np.sign(self.integral_error_right) * self.integral_max)
            clamp_active = True

        # Compute bias torques
        self.bias_tau_left = -self.integral_gain * self.integral_error_left
        self.bias_tau_right = -self.integral_gain * self.integral_error_right

        # Clamp bias torques
        self.bias_tau_left = float(np.clip(self.bias_tau_left, -self.integral_max, self.integral_max))
        self.bias_tau_right = float(np.clip(self.bias_tau_right, -self.integral_max, self.integral_max))

        return self.bias_tau_left, self.bias_tau_right, True, clamp_active

    def apply_yaw_aware_position_compensation(
        self,
        raw_sagittal_error: float,
        raw_lateral_error: float,
        yaw_error: float,
        yaw_compensation_gain: float = 1.0,
        max_compensation: float = 0.05,
    ) -> tuple[float, float]:
        """Compensate support position error for yaw-induced apparent drift.

        When the robot rotates by yaw angle theta, the support center appears to
        shift by approximately d*sin(theta) in the lateral direction and
        d*(1-cos(theta)) in the sagittal direction (where d is the axle offset).

        This method subtracts the yaw-induced apparent drift from the measured
        position error so that the position controller only responds to true drift.

        Args:
            raw_sagittal_error: Measured sagittal position error (m)
            raw_lateral_error: Measured lateral position error (m)
            yaw_error: Current yaw error from equilibrium (rad)
            yaw_compensation_gain: Scale factor for compensation (0.0 to 1.0)
            max_compensation: Maximum compensation magnitude (m)

        Returns:
            (compensated_sagittal_error, compensated_lateral_error)
        """
        if not self.uses_yaw_aware_compensation():
            return raw_sagittal_error, raw_lateral_error

        # Estimate yaw-induced apparent displacement
        # Using a conservative approximation: the apparent lateral shift
        # from yaw rotation is approximately yaw_error * (effective lever arm)
        # For a wheeled biped, the relevant lever arm is roughly the axle offset
        # Assume ~0.1m axle offset for this estimation
        axle_offset_m = 0.10
        yaw_apparent_lateral = axle_offset_m * np.sin(yaw_error)
        yaw_apparent_sagittal = axle_offset_m * (1.0 - np.cos(yaw_error))

        # Compensate by subtracting the yaw-induced component
        compensation_lateral = yaw_compensation_gain * yaw_apparent_lateral
        compensation_sagittal = yaw_compensation_gain * yaw_apparent_sagittal

        # Clamp compensation magnitude
        compensation_lateral = float(np.clip(compensation_lateral, -max_compensation, max_compensation))
        compensation_sagittal = float(np.clip(compensation_sagittal, -max_compensation, max_compensation))

        compensated_sagittal = raw_sagittal_error - compensation_sagittal
        compensated_lateral = raw_lateral_error - compensation_lateral

        return compensated_sagittal, compensated_lateral


def resolve_boundary_yaw_position_fix_state(args) -> BoundaryYawPositionFixState:
    """Create a BoundaryYawPositionFixState from parsed command-line arguments."""
    return BoundaryYawPositionFixState(
        profile=args.boundary_yaw_position_profile,
        boundary_kp=args.boundary_hip_yaw_kp,
        boundary_kd=args.boundary_hip_yaw_kd,
        integral_gain=args.boundary_hip_yaw_integral_gain,
        integral_max=args.boundary_hip_yaw_integral_max,
    )


def get_stage2b_default_empirical_feedforward() -> np.ndarray:
    return STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD.copy()


def resolve_stage2b_empirical_feedforward(telemetry_path: str | None) -> np.ndarray:
    if telemetry_path is None:
        return get_stage2b_default_empirical_feedforward()
    return load_empirical_feedforward_from_telemetry(telemetry_path)


def check_termination(qpos, com_height, robot_pitch_x, robot_roll_y,
                      height_floor_m: float = 0.35):
    """Check if robot should terminate (fall detection).

    Uses robot-frame orientation (pitch_x, roll_y) for termination, not Euler angles.

    Args:
        height_floor_m: Minimum allowed CoM height. Defaults to 0.35 m.
            When a height-variant setup is active, this is set to
            achieved_com_z - 0.05 m so low-height variants are not
            spuriously terminated.
    """
    # Height check
    if com_height < height_floor_m:
        return True, "height_too_low"

    # Orientation check using robot-frame orientation (45 degrees threshold)
    if abs(robot_pitch_x) > 0.785 or abs(robot_roll_y) > 0.785:
        return True, f"orientation_fail_pitch_x_{robot_pitch_x:.2f}_roll_y_{robot_roll_y:.2f}"

    return False, None


def measure_wheel_floor_contact(model, data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id):
    min_dist = None
    total_fz = 0.0
    contact_count = 0
    wheel_geom_ids = {l_wheel_geom_id, r_wheel_geom_id}

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        involves_wheel = g1 in wheel_geom_ids or g2 in wheel_geom_ids
        if not (involves_floor and involves_wheel):
            continue

        contact_count += 1
        d = float(c.dist)
        min_dist = d if min_dist is None else min(min_dist, d)

        force_contact = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force_contact)
        frame = np.array(c.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        total_fz += float(force_world[2])

    return {
        "min_dist": min_dist,
        "total_fz": total_fz,
        "contact_count": contact_count,
    }


def classify_floor_contacts(model, data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id):
    left_wheel_floor_contact = False
    right_wheel_floor_contact = False
    non_wheel_floor_contacts = 0
    total_wheel_floor_fz = 0.0
    contact_dist_min = None
    contact_dist_max = None

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        if not involves_floor:
            continue

        dist = float(c.dist)
        if contact_dist_min is None:
            contact_dist_min = dist
            contact_dist_max = dist
        else:
            contact_dist_min = min(contact_dist_min, dist)
            contact_dist_max = max(contact_dist_max, dist)

        involves_l_wheel = g1 == l_wheel_geom_id or g2 == l_wheel_geom_id
        involves_r_wheel = g1 == r_wheel_geom_id or g2 == r_wheel_geom_id

        if involves_l_wheel or involves_r_wheel:
            left_wheel_floor_contact = left_wheel_floor_contact or involves_l_wheel
            right_wheel_floor_contact = right_wheel_floor_contact or involves_r_wheel
            force_contact = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force_contact)
            frame = np.array(c.frame).reshape(3, 3)
            force_world = frame.T @ force_contact[:3]
            total_wheel_floor_fz += float(force_world[2])
        else:
            non_wheel_floor_contacts += 1

    return {
        "left_wheel_floor_contact": left_wheel_floor_contact,
        "right_wheel_floor_contact": right_wheel_floor_contact,
        "non_wheel_floor_contacts": non_wheel_floor_contacts,
        "total_wheel_floor_fz": total_wheel_floor_fz,
        "contact_dist_min": contact_dist_min if contact_dist_min is not None else 0.0,
        "contact_dist_max": contact_dist_max if contact_dist_max is not None else 0.0,
    }


def apply_initial_root_z_perturbation(
    model,
    data,
    perturbation_m: float,
    nominal_equilibrium_com_z_m: float,
    initial_com_z_m_after_perturbation: float | None = None,
):
    data.qpos[2] += perturbation_m
    data.qvel[:] = [0.0] * len(data.qvel)
    data.qacc[:] = [0.0] * len(data.qacc)
    mujoco.mj_forward(model, data)
    if initial_com_z_m_after_perturbation is None:
        initial_com_z_m_after_perturbation = nominal_equilibrium_com_z_m + perturbation_m
    return {
        "initial_root_z_perturbation_m": float(perturbation_m),
        "nominal_equilibrium_com_z_m": float(nominal_equilibrium_com_z_m),
        "initial_com_z_m_after_perturbation": float(initial_com_z_m_after_perturbation),
        "perturbation_applied_after_equilibrium_capture": True,
    }


def calibrate_root_z_for_wheel_floor_contact(model, data, target_dist=-5e-4, max_iters=5):
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        stats = measure_wheel_floor_contact(
            model, data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id
        )
        min_dist = stats["min_dist"]
        if min_dist is None:
            break

        delta_z = target_dist - min_dist
        if abs(delta_z) < 1e-7:
            break

        data.qpos[2] += delta_z
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0

    mujoco.mj_forward(model, data)
    return {
        "floor_geom_id": floor_geom_id,
        "l_wheel_geom_id": l_wheel_geom_id,
        "r_wheel_geom_id": r_wheel_geom_id,
    }


def build_stage2b_drift_audit_field_names():
    return [
        "com_z", "com_vz", "pitch_x", "pitch_rate_x", "roll_y", "roll_rate_y", "yaw_z",
        "com_x", "com_y", "cp_x", "cp_y",
        "com_error_x", "com_error_y", "com_error_z",
        "cp_error_x", "cp_error_y",
        "pitch_error", "roll_error", "height_error",
        "left_wheel_floor_contact", "right_wheel_floor_contact", "total_wheel_floor_fz",
        "left_fz_actual", "right_fz_actual", "fz_asymmetry_actual",
        "non_wheel_floor_contacts", "contact_dist_min", "contact_dist_max",
        "correction_wrench_Fx", "correction_wrench_Fy", "correction_wrench_Fz",
        "correction_wrench_Mx", "correction_wrench_My", "correction_wrench_Mz",
        "correction_Fy_com", "correction_Fy_cp", "correction_Fy_pitch", "correction_My_roll",
        "distributor_f_left", "distributor_f_right", "distributor_fz_sum",
        "tau_hip_roll", "tau_contact", "tau_wbc_correction", "tau_wbc_after_authority_clip",
        "tau_static_feedforward", "tau_static_posture", "tau_total_raw", "tau_final",
        "saturation_flags", "rate_limit_flags",
    ]


# K1 augmented telemetry fields — must be forwarded from controller diagnostics to CSV.
# Each field is read-only in the controller; this list mirrors the diagnostics dict keys.
K1_AUGMENTED_TELEMETRY_FIELDS = [
    # A. Pitch-rate notch / filter path
    "k1_raw_pitch_rate_x",
    "k1_filtered_pitch_rate_x",
    "k1_notch_output",
    "k1_notch_input",
    "k1_notch_state_1",
    "k1_notch_state_2",
    "k1_notch_state_y1",
    "k1_notch_state_y2",
    "k1_notch_enabled",
    "k1_notch_blend",
    "k1_notch_center_hz",
    "k1_notch_q",
    "k1_notch_height_gate_alpha",
    "k1_notch_filter_type",
    "k1_lowpass_cutoff_hz",
    # B. Torque decomposition before clipping
    "k1_tau_pitch_raw",
    "k1_tau_pitch_rate_raw",
    "k1_tau_position_raw",
    "k1_tau_com_velocity_raw",
    "k1_tau_wheel_velocity_raw",
    "k1_tau_support_velocity_raw",
    "k1_tau_eq_ff_raw",
    "k1_tau_common_preclip",
    "k1_tau_left_preclip",
    "k1_tau_right_preclip",
    # C. Torque clipping / saturation
    "k1_tau_position_cap_active",
    "k1_tau_position_cap_margin_nm",
    "k1_tau_total_clip_active",
    "k1_tau_total_clip_margin_nm",
    "k1_tau_left_postclip",
    "k1_tau_right_postclip",
    "k1_tau_clip_delta_left",
    "k1_tau_clip_delta_right",
    "k1_tau_clip_delta_common",
    "k1_saturation_fraction_window_50",
    "k1_saturation_fraction_window_200",
    # D. Support / coupling diagnostics
    "k1_support_error_m",
    "k1_support_velocity_m_s",
    "k1_com_y_velocity_m_s",
    "k1_pitch_support_phase_lag_s_est",
    "k1_pitch_support_corr_window_200",
    # E. Controller mode flags
    "k1_feedback_mode",
    "k1_profile_name",
    "k1_current_best_id",
    "k1_audit_ablation_mode",
    "k1_telemetry_augmented_version",
]


def build_step1_telemetry_template():
    return {
        "tau_wbc_per_joint": [],
        "tau_wbc_scaled_per_joint": [],
        "tau_hip_roll_centering_per_joint": [],
        "tau_posture_per_joint": [],
        "tau_leg_position_per_joint": [],
        "tau_wheel_balance_per_joint": [],
        "tau_static_feedforward_per_joint": [],  # Stage 2B
        "tau_total_per_joint": [],
        "tau_total_raw_per_joint": [],
        "tau_total_clipped_per_joint": [],
        "tau_smooth_per_joint": [],
        "support_ratio_support_joints": [],
        "support_ratio_mean": [],
        "torque_rate_limit_enabled": [],
        "per_actuator_wbc_authority_enabled": [],
        "wbc_joint_scaling_enabled": [],
        "initialize_tau_prev_from_wbc_enabled": [],
        "hip_roll_abs_max": [],
        "hip_yaw_abs_max": [],
        "push_active": [],
        "push_force_x": [],
        "push_force_y": [],
        "push_schedule_entries": [],
        "hip_pitch_error_max": [],
        "knee_error_max": [],
        "wheel_balance_torque": [],
        "control_mode": [],
        "feedforward_enabled": [],  # Stage 2B
        "feedforward_norm": [],  # Stage 2B
        "tau_total_unclipped": [],
        "tau_total_clipped": [],
        "tau_total_before_final_clip": [],
        "tau_total_after_final_clip": [],
        "tau_position_lower_bound": [],
        "tau_position_upper_bound": [],
        "tau_position_total_bound_clipped": [],
        "position_authority_mode": [],
        "position_authority_reason": [],
        "wheel_torque_saturation_left": [],
        "wheel_torque_saturation_right": [],
        "wheel_torque_rate_saturation_left": [],
        "wheel_torque_rate_saturation_right": [],
        "initial_root_z_perturbation_m": [],
        "nominal_equilibrium_com_z_m": [],
        "initial_com_z_m_after_perturbation": [],
        "perturbation_applied_after_equilibrium_capture": [],
    }


def compute_step1_joint_diagnostics(joint_pos, joint_pos_error):
    hip_roll_indices = jnp.array([0, 5])
    hip_yaw_indices = jnp.array([1, 6])
    hip_pitch_indices = jnp.array([2, 7])
    knee_indices = jnp.array([3, 8])

    return {
        "control_mode": "upright",
        "hip_roll_abs_max": float(jnp.max(jnp.abs(joint_pos[hip_roll_indices]))),
        "hip_yaw_abs_max": float(jnp.max(jnp.abs(joint_pos[hip_yaw_indices]))),
        "hip_pitch_error_max": float(jnp.max(jnp.abs(joint_pos_error[hip_pitch_indices]))),
        "knee_error_max": float(jnp.max(jnp.abs(joint_pos_error[knee_indices]))),
        "wheel_balance_torque": 0.0,
    }


def build_step3_wbc_joint_scale():
    return jnp.array([1.0, 0.3, 0.75, 0.75, 1.0, 1.0, 0.3, 0.75, 0.75, 1.0])


def compute_step6_control_mode(
    roll_rad,
    pitch_rad,
    upright_roll_threshold=0.20,
    upright_pitch_threshold=0.15,
    recovery_roll_threshold=0.30,
    recovery_pitch_threshold=0.25,
):
    if abs(roll_rad) > recovery_roll_threshold or abs(pitch_rad) > recovery_pitch_threshold:
        return "recovery"
    if abs(roll_rad) < upright_roll_threshold and abs(pitch_rad) < upright_pitch_threshold:
        return "upright"
    return "transition"


def build_step6_wbc_joint_scale(control_mode):
    return build_step3_wbc_joint_scale()


def compute_step6_hip_roll_authority_scale(control_mode):
    if control_mode == "transition":
        return 0.5
    return 1.0


def compute_step4_hip_roll_centering(
    joint_pos,
    joint_vel,
    deadband=0.25,
    kp=20.0,
    kd=1.0,
    max_torque=12.0,
):
    tau = jnp.zeros(10)
    hip_roll_indices = jnp.array([0, 5])
    hip_roll_pos = joint_pos[hip_roll_indices]
    hip_roll_vel = joint_vel[hip_roll_indices]
    excess = jnp.maximum(jnp.abs(hip_roll_pos) - deadband, 0.0)
    tau_raw = -kp * excess * jnp.sign(hip_roll_pos) - kd * hip_roll_vel
    tau_limited = jnp.clip(tau_raw, -max_torque, max_torque)
    return tau.at[hip_roll_indices].set(tau_limited)


def compute_step5_wheel_balance(
    pitch_rad,
    pitch_rate_rad_s,
    capture_point_error_y,
    kp_pitch=10.0,
    kd_pitch=2.0,
    k_cp=4.0,
    max_torque=4.0,
):
    # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
    # Wheel balance uses sagittal capture-point error on Y-axis.
    tau_wheel = kp_pitch * pitch_rad + kd_pitch * pitch_rate_rad_s + k_cp * capture_point_error_y
    tau_wheel = jnp.clip(tau_wheel, -max_torque, max_torque)
    tau = jnp.zeros(10)
    return tau.at[jnp.array([4, 9])].set(tau_wheel)


def log_wrapper_telemetry(step, telemetry):
    """Log StaticBalanceController wrapper telemetry for diagnostics."""
    print(f"[WRAPPER][step={step}] Support joint bias removed: {telemetry['support_joint_bias_removed']}")
    print(f"[WRAPPER][step={step}] Posture error: {telemetry['posture_error_norm']:.6f} rad")
    print(f"[WRAPPER][step={step}] CoM height error: {telemetry['com_height_error']:.6f} m")
    print(f"[WRAPPER][step={step}] Pitch error: {telemetry['pitch_x_error']:.6f} rad")
    print(f"[WRAPPER][step={step}] Roll error: {telemetry['roll_y_error']:.6f} rad")


def compute_step2_torque_components(
    leg_position_controller,
    joint_pos,
    joint_vel,
    target_joint_pos,
    tau_wbc,
    tau_posture,
    tau_wheel_secondary,
    tau_inverse_dynamics,
    wbc_joint_scale=None,
    tau_hip_roll_centering=None,
    tau_wheel_balance=None,
):
    tau_leg_position = leg_position_controller.compute_leg_torques(
        joint_pos,
        joint_vel,
        target_joint_pos,
    )
    if wbc_joint_scale is None:
        wbc_joint_scale = jnp.ones(10)
    if tau_hip_roll_centering is None:
        tau_hip_roll_centering = jnp.zeros(10)
    if tau_wheel_balance is None:
        tau_wheel_balance = jnp.zeros(10)
    tau_wbc_scaled = tau_wbc * wbc_joint_scale
    tau_total_raw = (
        tau_wbc_scaled
        + tau_hip_roll_centering
        + tau_leg_position
        + tau_posture
        + tau_wheel_secondary
        + tau_wheel_balance
        + tau_inverse_dynamics
    )
    return {
        "tau_wbc_scaled": tau_wbc_scaled,
        "tau_hip_roll_centering": tau_hip_roll_centering,
        "tau_leg_position": tau_leg_position,
        "tau_wheel_balance": tau_wheel_balance,
        "tau_total_raw": tau_total_raw,
    }


def is_balance_core_mode(args) -> bool:
    return args.controller_mode in {"balance-core", "standing-balance"}


def resolve_active_sagittal_controller_set(sagittal_controller_choice: str) -> set[str]:
    """Return the set of active sagittal controller names for mutual exclusion verification.

    Only one sagittal controller may be active at a time.
    """
    return {sagittal_controller_choice}


def validate_balance_core_mode_args(args):
    """Validate that balance-core mode does not use incompatible legacy flags.

    Args:
        args: Parsed command-line arguments

    Raises:
        ValueError: If balance-core mode is used with incompatible legacy flags
    """
    if args.controller_mode != "balance-core":
        return

    incompatible_flags = []

    if args.enable_static_dynamics_wrapper:
        incompatible_flags.append("--enable-static-dynamics-wrapper")
    if args.enable_secondary_wheel_balance:
        incompatible_flags.append("--enable-secondary-wheel-balance")
    if args.enable_stage2_static_posture_hold:
        incompatible_flags.append("--enable-stage2-static-posture-hold")
    if args.enable_stage2b_gravity_feedforward:
        incompatible_flags.append("--enable-stage2b-gravity-feedforward")
    if args.enable_stage2b_roll_direct:
        incompatible_flags.append("--enable-stage2b-roll-direct")
    if args.enable_stage2b_sagittal_wheel:
        incompatible_flags.append("--enable-stage2b-sagittal-wheel")
    if args.enable_stage2c_sagittal_state_feedback:
        incompatible_flags.append("--enable-stage2c-sagittal-state-feedback")
    if args.enable_stage2d_sagittal_lqr:
        incompatible_flags.append("--enable-stage2d-sagittal-lqr")
    if args.initialize_tau_prev_from_wbc:
        incompatible_flags.append("--initialize-tau-prev-from-wbc")
    if args.use_per_actuator_wbc_authority:
        incompatible_flags.append("--use-per-actuator-wbc-authority")

    if incompatible_flags:
        raise ValueError(
            f"balance-core mode is incompatible with the following legacy flags: "
            f"{', '.join(incompatible_flags)}"
        )


def resolve_support_feedforward_vector():
    """Return empirical support feedforward vector for balance-core mode.

    Returns:
        np.ndarray: 10-element support feedforward vector with empirical hip-pitch and knee torques
    """
    return get_stage2b_default_empirical_feedforward()


def append_balance_core_telemetry(
    telemetry: dict,
    result,
    centroidal_state,
    contact_output,
    cp_error_y_m: float,
    wheel_vel_left_rad_s: float,
    wheel_vel_right_rad_s: float,
    wheel_acc_left_rad_s2: float,
    wheel_acc_right_rad_s2: float,
    hip_roll_pos: tuple[float, float] | None = None,
    hip_roll_ref: tuple[float, float] | None = None,
):
    """Append balance-core state and torque telemetry for one control tick.

    Args:
        telemetry: Telemetry dict with balance-core columns initialized
        result: BalanceCoreTorqueResult with torque composition output
        centroidal_state: Centroidal state with body orientation and CoM
        contact_output: ContactSupervisorOutput with contact classification
        cp_error_y_m: Capture point error in y direction [m]
        wheel_vel_left_rad_s: Left wheel velocity [rad/s]
        wheel_vel_right_rad_s: Right wheel velocity [rad/s]
        wheel_acc_left_rad_s2: Left wheel acceleration [rad/s^2]
        wheel_acc_right_rad_s2: Right wheel acceleration [rad/s^2]
    """
    wheel_vel_mean = 0.5 * (wheel_vel_left_rad_s + wheel_vel_right_rad_s)
    wheel_acc_mean = 0.5 * (wheel_acc_left_rad_s2 + wheel_acc_right_rad_s2)

    hip_roll_left = None if hip_roll_pos is None else float(hip_roll_pos[0])
    hip_roll_right = None if hip_roll_pos is None else float(hip_roll_pos[1])
    hip_roll_ref_left = None if hip_roll_ref is None else float(hip_roll_ref[0])
    hip_roll_ref_right = None if hip_roll_ref is None else float(hip_roll_ref[1])
    hip_roll_common_component = None
    hip_roll_symmetric_component = None
    hip_roll_abs_max = None
    hip_roll_error_left = None
    hip_roll_error_right = None
    if hip_roll_pos is not None:
        hip_roll_common_component = 0.5 * (hip_roll_left + hip_roll_right)
        hip_roll_symmetric_component = 0.5 * (hip_roll_left - hip_roll_right)
        hip_roll_abs_max = max(abs(hip_roll_left), abs(hip_roll_right))
    if hip_roll_pos is not None and hip_roll_ref is not None:
        hip_roll_error_left = hip_roll_ref_left - hip_roll_left
        hip_roll_error_right = hip_roll_ref_right - hip_roll_right

    # Append state fields
    state_values = {
        "pitch_x_rad": float(centroidal_state.body_pitch_x),
        "roll_y_rad": float(centroidal_state.body_roll_y),
        "yaw_z_rad": float(centroidal_state.body_yaw_z),
        "pitch_rate_x_rad_s": float(centroidal_state.body_pitch_rate_x),
        "roll_rate_y_rad_s": float(centroidal_state.body_roll_rate_y),
        "yaw_rate_z_rad_s": float(centroidal_state.body_yaw_rate_z),
        "com_x_m": float(centroidal_state.com_pos[0]),
        "com_y_m": float(centroidal_state.com_pos[1]),
        "com_z_m": float(centroidal_state.com_pos[2]),
        "com_vx_m_s": float(centroidal_state.com_vel[0]),
        "com_vy_m_s": float(centroidal_state.com_vel[1]),
        "com_vz_m_s": float(centroidal_state.com_vel[2]),
        "cp_x_m": float(centroidal_state.capture_point[0]),
        "cp_y_m": float(centroidal_state.capture_point[1]),
        "cp_error_y_m": float(cp_error_y_m),
        "wheel_vel_left_rad_s": float(wheel_vel_left_rad_s),
        "wheel_vel_right_rad_s": float(wheel_vel_right_rad_s),
        "wheel_vel_mean_rad_s": float(wheel_vel_mean),
        "wheel_acc_left_rad_s2": float(wheel_acc_left_rad_s2),
        "wheel_acc_right_rad_s2": float(wheel_acc_right_rad_s2),
        "wheel_acc_mean_rad_s2": float(wheel_acc_mean),
        "left_wheel_contact": bool(contact_output.left_wheel_contact),
        "right_wheel_contact": bool(contact_output.right_wheel_contact),
        "contact_supervisor_state": contact_output.state.value,
        "contact_previous_state": contact_output.previous_state.value if contact_output.previous_state is not None else "none",
        "contact_duration_s": float(contact_output.contact_duration_s),
        "contact_transition_event": contact_output.transition_event,
        "contact_force_valid": bool(contact_output.contact_force_valid),
        "contact_recovery_hook_fields": str(contact_output.recovery_hook_fields),
        "hip_roll_left_rad": hip_roll_left,
        "hip_roll_right_rad": hip_roll_right,
        "hip_roll_common_component_rad": hip_roll_common_component,
        "hip_roll_symmetric_component_rad": hip_roll_symmetric_component,
        "hip_roll_abs_max_rad": hip_roll_abs_max,
        "hip_roll_ref_left_rad": hip_roll_ref_left,
        "hip_roll_ref_right_rad": hip_roll_ref_right,
        "hip_roll_error_left_rad": hip_roll_error_left,
        "hip_roll_error_right_rad": hip_roll_error_right,
    }
    for name, value in state_values.items():
        telemetry[name].append(value)

    # Append torque fields from result.telemetry
    # Per-joint arrays are tuples and need comma-separated string conversion for CSV
    # Use setdefault to handle dynamic diagnostics fields (e.g. tuned telemetry)
    for name, value in result.telemetry.items():
        if isinstance(value, tuple):
            telemetry.setdefault(name, []).append(",".join(str(v) for v in value))
        else:
            telemetry.setdefault(name, []).append(value)


def zero_legacy_torque_sources_for_balance_core():
    return {
        "tau_wbc_correction": jnp.zeros(10),
        "tau_wbc_scaled": jnp.zeros(10),
        "tau_posture": jnp.zeros(10),
        "tau_leg_position": jnp.zeros(10),
        "tau_hip_roll_centering": jnp.zeros(10),
        "tau_wheel_balance": jnp.zeros(10),
        "tau_inverse_dynamics": jnp.zeros(10),
    }


def build_balance_core_controllers(
    control_dt: float,
    support_feedforward_vector: np.ndarray,
    torque_limit: np.ndarray,
    max_torque_rate: np.ndarray,
    sagittal_controller_choice: str = "baseline",
    vd_k_position: float = 40.0,
    vd_k_velocity: float = 15.0,
    vd_k_support_velocity: float = 0.0,
    vd_max_position_tau: float = 3.0,
    vd_k_pitch: float = 50.0,
    vd_pitch_ref_offset_deg: float = 0.0,
    vd_enable_capture_gate: bool = False,
    vd_capture_gate_pitch_threshold: float = 0.05,
    vd_capture_gate_conflict_factor: float = 0.0,
    vd_capture_gate_smooth_steps: int = 10,
    vd_capture_gate_use_cp: bool = True,
    vd_enable_torque_budget_aware_position: bool = False,
    vd_position_tau_budget_cap: float = 7.0,
    vd_enable_position_integral: bool = False,
    vd_ki_position_integral: float = 0.0,
    vd_integral_max_abs: float = 1.0,
    vd_integral_pitch_error_threshold_rad: float = 0.03,
    vd_integral_roll_error_threshold_rad: float = 0.05,
    vd_integral_pitch_rate_threshold_rad_s: float = 0.05,
    vd_integral_support_velocity_threshold_m_s: float = 0.03,
    vd_integral_wheel_velocity_threshold_rad_s: float = 1.0,
    vd_integral_min_com_z_m: float = 0.38,
    vd_integral_max_com_z_m: float = 0.43,
    sagittal_authority_schedule: SagittalAuthoritySchedule | None = None,
    shape_kp_hip_yaw: float | None = None,
    shape_kd_hip_yaw: float | None = None,
    enable_hip_yaw_support_feedforward: bool = False,
    hip_yaw_support_k: float = 0.0,
    hip_yaw_support_tau_max: float = 1.0,
    hip_yaw_support_sign: float = 1.0,
    enable_hip_yaw_divergence_damping: bool = False,
    hip_yaw_divergence_k: float = 0.0,
    hip_yaw_divergence_kd: float = 0.0,
    hip_yaw_divergence_tau_max: float = 0.5,
    hip_yaw_divergence_z_low: float = 0.300,
    hip_yaw_divergence_z_high: float = 0.393,
    # Differential wheel yaw stabilizer (BODY_YAW_WRONG_ACTUATOR fix)
    enable_wheel_yaw_stabilizer: bool = False,
    wheel_yaw_kp: float = 5.0,
    wheel_yaw_kd: float = 1.5,
    wheel_yaw_max_torque: float = 5.0,
    wheel_yaw_lowpass_alpha: float = 1.0,
    wheel_yaw_height_gate_low: float = 0.250,
    wheel_yaw_height_gate_high: float = 0.350,
    yaw_controller_kp: float = 8.0,
    yaw_controller_kd: float = 2.0,
    yaw_controller_max_torque: float = 5.0,
):
    """Build all balance-core controller components.

    Args:
        control_dt: Control timestep in seconds
        support_feedforward_vector: 10-element empirical support torque vector
        torque_limit: Per-joint torque limits [Nm], shape (10,)
        max_torque_rate: Per-joint max torque rate [Nm/s], shape (10,)
        sagittal_controller_choice: "baseline" or "velocity-damped"

    Returns:
        dict: Dictionary with keys:
            - contact_supervisor: ContactSupervisor instance
            - shape_posture: ShapePostureController instance
            - support_feedforward: SupportFeedforwardController instance
            - sagittal_wheel_balance: SagittalWheelBalanceController or SagittalVelocityDampedBalanceController
            - lateral_roll_balance: LateralRollBalanceController instance
            - composer: BalanceCoreTorqueComposer instance
            - sagittal_controller_name: str identifier for telemetry
    """
    # Instantiate contact supervisor
    contact_supervisor = ContactSupervisor(control_dt=control_dt)

    # Instantiate shape-posture controller
    # Use overrides if provided, otherwise use balance-core defaults
    effective_kp_hip_yaw = (
        shape_kp_hip_yaw
        if shape_kp_hip_yaw is not None
        else BALANCE_CORE_HIP_YAW_AUTHORITY.kp_hip_yaw
    )
    effective_kd_hip_yaw = (
        shape_kd_hip_yaw
        if shape_kd_hip_yaw is not None
        else BALANCE_CORE_HIP_YAW_AUTHORITY.kd_hip_yaw
    )

    shape_posture = ShapePostureController(
        kp_hip_yaw=effective_kp_hip_yaw,
        kd_hip_yaw=effective_kd_hip_yaw,
        kp_hip_pitch=30.0,
        kd_hip_pitch=4.0,
        kp_knee=40.0,
        kd_knee=5.0,
        enable_hip_yaw_support_feedforward=enable_hip_yaw_support_feedforward,
        k_support_hip_yaw=hip_yaw_support_k,
        tau_max_support_comp=hip_yaw_support_tau_max,
        support_comp_sign=hip_yaw_support_sign,
        enable_hip_yaw_divergence_damping=enable_hip_yaw_divergence_damping,
        k_divergence=hip_yaw_divergence_k,
        k_divergence_rate=hip_yaw_divergence_kd,
        tau_max_divergence=hip_yaw_divergence_tau_max,
        divergence_gate_z_low=hip_yaw_divergence_z_low,
        divergence_gate_z_high=hip_yaw_divergence_z_high,
    )

    # Instantiate support feedforward controller
    support_feedforward = SupportFeedforwardController(
        support_vector=jnp.array(support_feedforward_vector),
        joint_group="hip_pitch_knee",
        scale=0.5,
    )

    # Instantiate sagittal controller (mutually exclusive selection)
    if sagittal_controller_choice == "velocity-damped":
        # Build capture gate config if enabled
        capture_gate_config = None
        if vd_enable_capture_gate:
            capture_gate_config = {
                "pitch_threshold_rad": vd_capture_gate_pitch_threshold,
                "gate_factor_conflict": vd_capture_gate_conflict_factor,
                "gate_factor_normal": 1.0,
                "smooth_ramp_steps": vd_capture_gate_smooth_steps,
                "enable_capture_point": vd_capture_gate_use_cp,
                "gravity_m_s2": 9.81,
            }

        sagittal_wheel_balance = SagittalVelocityDampedBalanceController(
            kp_pitch=vd_k_pitch,
            kd_pitch=10.0,
            kp_cp=0.0,  # Step E coupling fix: disable tau_cp to prevent destructive cancellation with tau_pitch
            kd_com_vy=5.0,
            k_velocity=vd_k_velocity,
            k_wheel_velocity=0.5,
            k_position=vd_k_position,
            k_support_velocity=vd_k_support_velocity,
            max_position_tau=vd_max_position_tau,
            wheel_torque_sign=1.0,
            max_tau_wheel=5.0,
            enable_capture_gate=(capture_gate_config is not None),
            capture_gate_config=capture_gate_config,
            dt=control_dt,
            enable_torque_budget_aware_position=vd_enable_torque_budget_aware_position,
            position_tau_budget_cap=vd_position_tau_budget_cap,
            enable_position_integral=vd_enable_position_integral,
            ki_position_integral=vd_ki_position_integral,
            integral_max_abs=vd_integral_max_abs,
            integral_pitch_error_threshold_rad=vd_integral_pitch_error_threshold_rad,
            integral_roll_error_threshold_rad=vd_integral_roll_error_threshold_rad,
            integral_pitch_rate_threshold_rad_s=vd_integral_pitch_rate_threshold_rad_s,
            integral_support_velocity_threshold_m_s=vd_integral_support_velocity_threshold_m_s,
            integral_wheel_velocity_threshold_rad_s=vd_integral_wheel_velocity_threshold_rad_s,
            integral_min_com_z_m=vd_integral_min_com_z_m,
            integral_max_com_z_m=vd_integral_max_com_z_m,
            authority_schedule=sagittal_authority_schedule,
        )
        sagittal_controller_name = "velocity-damped"
    else:
        sagittal_wheel_balance = SagittalWheelBalanceController(
            kp_pitch=50.0,
            kd_pitch=10.0,
            kp_cp=30.0,
            kd_com_vy=5.0,
            kd_wheel_vel=0.5,
            wheel_torque_sign=1.0,
        )
        sagittal_controller_name = "baseline"

    # Instantiate lateral roll balance controller
    lateral_roll_balance = LateralRollBalanceController(
        kp_roll=40.0,
        kd_roll=8.0,
        max_roll_moment=50.0,
        hip_roll_torque_sign=1.0,
    )

    # Instantiate yaw controller (antisymmetric hip-yaw, always created)
    # When wheel yaw stabilizer is also enabled, the YawController still runs at
    # full capacity and wheel yaw adds EXTRA correction — this avoids the problem
    # where reducing YawController authority causes body yaw to diverge and fall.
    yaw_controller = YawController(
        kp_yaw=yaw_controller_kp,
        kd_yaw=yaw_controller_kd,
        max_yaw_torque=yaw_controller_max_torque,
    )

    # Instantiate differential wheel yaw stabilizer (opt-in BODY_YAW_WRONG_ACTUATOR fix)
    # Activation via --enable-wheel-yaw-stabilizer CLI flag OR profile-based M activation
    # (enable_body_yaw_wheel_stabilization=True in the sagittal authority schedule).
    wheel_yaw_stabilizer = None
    m_profile_activation = False
    if sagittal_authority_schedule is not None and sagittal_authority_schedule.enable_body_yaw_wheel_stabilization:
        # M family profile activation: use profile parameters
        m_profile_activation = True
        wheel_yaw_stabilizer = DifferentialWheelYawStabilizer(
            kp_yaw=sagittal_authority_schedule.wheel_yaw_kp,
            kd_yaw=sagittal_authority_schedule.wheel_yaw_kd,
            max_yaw_torque=sagittal_authority_schedule.wheel_yaw_max_torque,
            lowpass_alpha=wheel_yaw_lowpass_alpha,
            height_gate_low=sagittal_authority_schedule.wheel_yaw_height_gate_start_m,
            height_gate_high=sagittal_authority_schedule.wheel_yaw_height_gate_full_m,
        )
    elif enable_wheel_yaw_stabilizer:
        wheel_yaw_stabilizer = DifferentialWheelYawStabilizer(
            kp_yaw=wheel_yaw_kp,
            kd_yaw=wheel_yaw_kd,
            max_yaw_torque=wheel_yaw_max_torque,
            lowpass_alpha=wheel_yaw_lowpass_alpha,
            height_gate_low=wheel_yaw_height_gate_low,
            height_gate_high=wheel_yaw_height_gate_high,
        )

    # Instantiate torque composer
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.array(torque_limit),
        max_torque_rate=jnp.array(max_torque_rate),
        control_dt=control_dt,
    )

    return {
        "contact_supervisor": contact_supervisor,
        "shape_posture": shape_posture,
        "support_feedforward": support_feedforward,
        "sagittal_wheel_balance": sagittal_wheel_balance,
        "lateral_roll_balance": lateral_roll_balance,
        "yaw_controller": yaw_controller,
        "composer": composer,
        "sagittal_controller_name": sagittal_controller_name,
        "wheel_yaw_stabilizer": wheel_yaw_stabilizer,
    }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Simulate hierarchical controller with telemetry"
    )
    parser.add_argument(
        "--visual", action="store_true", help="Run with MuJoCo viewer (visual mode)"
    )
    # ---- Visual realtime pacing flags ---- #
    parser.add_argument(
        "--visual-realtime-factor",
        type=float,
        default=1.0,
        help="Target realtime factor for visual pacing (default 1.0 = 1:1 wall clock). "
             "Values > 1.0 run faster than realtime; < 1.0 run slower. "
             "Set to 0 to disable pacing entirely (run as fast as possible).",
    )
    parser.add_argument(
        "--visual-sync-hz",
        type=float,
        default=30.0,
        help="Target viewer sync rate in Hz (default 30). "
             "Syncs the MuJoCo viewer at this approximate rate, decoupled from control rate. "
             "Lower values reduce render overhead but make viewer less responsive.",
    )
    parser.add_argument(
        "--visual-disable-realtime-pacing",
        action="store_true",
        help="Disable all realtime pacing sleep in visual mode (run as fast as possible).",
    )
    parser.add_argument(
        "--visual-profile-timing",
        action="store_true",
        help="Print detailed per-step timing diagnostics in visual mode.",
    )
    parser.add_argument(
        "--profile-controller",
        action="store_true",
        help="Profile per-component controller timing and emit JSON breakdown at end of run. "
             "Accumulates wall-clock time for: centroidal_estimator, capture_estimator, "
             "shape_posture, sagittal_wheel_balance, lateral_roll, yaw, composer, "
             "telemetry, torque_apply, and total per-step. "
             "Output path: outputs/profile/stage1_controller_profile_breakdown.json",
    )
    parser.add_argument(
        "--wbc-quiet",
        action="store_true",
        help="Suppress per-step WBC diagnostic prints (PID, wheel torque, force feedback, pipeline). "
             "Automatically enabled in --visual mode. Use this to benchmark headless without print overhead.",
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of 100 Hz control steps to simulate"
    )
    parser.add_argument(
        "--controller-mode",
        type=str,
        default="legacy",
        choices=["legacy", "balance-core", "standing-balance"],
        help="Controller mode: legacy (all features), balance-core (clean WBC), standing-balance (future)",
    )
    parser.add_argument(
        "--controller-backend",
        type=str,
        default="python",
        choices=["python", "jax"],
        help="Controller backend: python (default, reference), jax (JIT-accelerated, opt-in). "
             "JAX backend requires balance-core mode. Python backend is always available.",
    )
    parser.add_argument(
        "--enable-secondary-wheel-balance",
        action="store_true",
        help="Enable secondary wheel-balance torque path (default: disabled for WBC-only wheel torque)",
    )
    parser.add_argument("--disable-torque-rate-limit", action="store_true")
    parser.add_argument("--initialize-tau-prev-from-wbc", action="store_true")
    parser.add_argument("--disable-wbc-joint-scale", action="store_true")
    parser.add_argument("--use-per-actuator-wbc-authority", action="store_true")
    parser.add_argument("--height-damping", type=float, default=0.0)
    parser.add_argument(
        '--enable-static-dynamics-wrapper',
        action='store_true',
        default=False,
        help='Enable StaticBalanceController wrapper to cancel WBC equilibrium bias'
    )
    parser.add_argument(
        '--enable-stage2-static-posture-hold',
        action='store_true',
        default=False,
        help='Enable Stage 2: StaticPostureHoldingController + correction-only WBC'
    )
    parser.add_argument('--static-kp-hip-pitch', type=float, default=30.0, help='StaticPostureHoldingController kp_hip_pitch')
    parser.add_argument('--static-kd-hip-pitch', type=float, default=4.0, help='StaticPostureHoldingController kd_hip_pitch')
    parser.add_argument('--static-kp-knee', type=float, default=40.0, help='StaticPostureHoldingController kp_knee')
    parser.add_argument('--static-kd-knee', type=float, default=5.0, help='StaticPostureHoldingController kd_knee')
    parser.add_argument('--static-max-torque-hip-pitch', type=float, default=30.0, help='StaticPostureHoldingController max_torque_hip_pitch')
    parser.add_argument('--static-max-torque-knee', type=float, default=30.0, help='StaticPostureHoldingController max_torque_knee')
    # Stage 2B: Gravity feedforward compensation
    parser.add_argument(
        '--enable-stage2b-gravity-feedforward',
        action='store_true',
        default=False,
        help='Enable Stage 2B: StaticFeedforwardController for gravity compensation (validated: +empirical, scale=0.5, knee, instant)'
    )
    parser.add_argument('--stage2b-feedforward-scale', type=float, default=0.5, help='Stage 2B feedforward scale factor (default: 0.5, validated)')
    parser.add_argument('--stage2b-feedforward-joint-group', type=str, default='knee', choices=['knee', 'hip_pitch', 'hip_pitch_knee'], help='Stage 2B feedforward joint group (default: knee, validated)')
    parser.add_argument('--stage2b-feedforward-ramp', type=str, default='instant', choices=['instant', 'short', 'medium'], help='Stage 2B feedforward ramp mode (default: instant, validated)')
    parser.add_argument('--stage2b-feedforward-sign', type=str, default='positive', choices=['positive', 'negative'], help='Stage 2B feedforward sign (default: positive, validated)')
    parser.add_argument('--stage2b-feedforward-telemetry-path', type=str, default=None, help='Optional telemetry CSV path to override fixed Stage 2B empirical feedforward default')
    parser.add_argument('--stage2b-ablation-mode', type=str, default='E', choices=['A', 'B', 'C', 'D', 'E'], help='Stage 2B ablation mode: A=ff+posture, B=+wbc, C=+hip_roll_centering, D=+wheel_balance, E=full stack')
    parser.add_argument('--disable-wbc-correction', action='store_true', default=False, help='Disable WBC correction torque in Stage 2B ablation')
    parser.add_argument('--disable-hip-roll-centering', action='store_true', default=False, help='Disable hip-roll centering torque in Stage 2B ablation')
    parser.add_argument('--disable-wheel-balance', action='store_true', default=False, help='Disable wheel-balance torque in Stage 2B ablation')
    # Stage 2B: Direct roll controller
    parser.add_argument('--enable-stage2b-roll-direct', action='store_true', default=False, help='Enable Stage 2B direct roll controller (hip_roll PD only, no WBC contact path)')
    parser.add_argument('--stage2b-roll-kp', type=float, default=100.0, help='Stage 2B direct roll kp gain (Nm/rad)')
    parser.add_argument('--stage2b-roll-kd', type=float, default=20.0, help='Stage 2B direct roll kd gain (Nm/(rad/s))')
    parser.add_argument('--stage2b-roll-tau-max', type=float, default=15.0, help='Stage 2B direct roll max hip_roll torque per side (Nm)')
    # Stage 2B: Sagittal wheel controller
    parser.add_argument('--enable-stage2b-sagittal-wheel', action='store_true', default=False, help='Enable Stage 2B sagittal wheel controller (direct wheel PD for pitch)')
    parser.add_argument('--stage2b-sagittal-k-pitch', type=float, default=10.0, help='Stage 2B sagittal k_pitch gain (Nm/rad)')
    parser.add_argument('--stage2b-sagittal-k-pitch-rate', type=float, default=2.0, help='Stage 2B sagittal k_pitch_rate gain (Nm/(rad/s))')
    parser.add_argument('--stage2b-sagittal-k-cp', type=float, default=4.0, help='Stage 2B sagittal k_cp gain (Nm/m)')
    parser.add_argument('--stage2b-sagittal-k-com-y', type=float, default=0.0, help='Stage 2B sagittal k_com_y gain (Nm/m)')
    parser.add_argument('--stage2b-sagittal-k-com-vy', type=float, default=2.0, help='Stage 2B sagittal k_com_vy gain (Nm/(m/s))')
    parser.add_argument('--stage2b-sagittal-max-tau', type=float, default=3.0, help='Stage 2B sagittal max wheel torque (Nm)')
    # Stage 2C: Sagittal state-feedback controller with wheel velocity damping
    parser.add_argument('--enable-stage2c-sagittal-state-feedback', action='store_true', default=False, help='Enable Stage 2C sagittal state-feedback controller (full state feedback with wheel velocity damping)')
    parser.add_argument('--stage2c-k-pitch', type=float, default=20.0, help='Stage 2C k_pitch gain (Nm/rad)')
    parser.add_argument('--stage2c-k-pitch-rate', type=float, default=6.0, help='Stage 2C k_pitch_rate gain (Nm/(rad/s))')
    parser.add_argument('--stage2c-k-com-y', type=float, default=0.0, help='Stage 2C k_com_y gain (Nm/m)')
    parser.add_argument('--stage2c-k-com-vy', type=float, default=0.0, help='Stage 2C k_com_vy gain (Nm/(m/s))')
    parser.add_argument('--stage2c-k-cp-y', type=float, default=8.0, help='Stage 2C k_cp_y gain (Nm/m)')
    parser.add_argument('--stage2c-k-wheel-vel', type=float, default=0.3, help='Stage 2C k_wheel_vel damping gain (Nm/(rad/s))')
    parser.add_argument('--stage2c-max-tau', type=float, default=8.0, help='Stage 2C max wheel torque (Nm)')
    # Stage 2D: Sagittal LQR controller (model-based, identified dynamics)
    parser.add_argument('--enable-stage2d-sagittal-lqr', action='store_true', default=False, help='Enable Stage 2D sagittal LQR controller (model-based with identified dynamics)')
    parser.add_argument('--stage2d-lqr-config', type=str, default='A', choices=['A', 'B', 'C', 'D'], help='Stage 2D LQR configuration (A=baseline, B=increased, C=high, D=aggressive)')
    parser.add_argument('--stage2d-model-path', type=str, default='outputs/stage2d_sysid/identified_model.npz', help='Path to identified model from Phase 1')
    parser.add_argument(
        '--initial-root-z-perturbation',
        type=float,
        default=0.0,
        help='Apply an initial root-z perturbation after equilibrium capture and before rollout',
    )
    # Step D: push disturbance parameters (mjx-style periodic random push via xfrc_applied).
    # These are NO-OP for the C++ mj_step path — push is applied via subprocess sims only.
    parser.add_argument(
        '--push-enabled',
        action='store_true',
        default=False,
        help='Enable periodic random push disturbance (Step D validation only).',
    )
    parser.add_argument(
        '--push-magnitude-n',
        type=float,
        default=15.0,
        help='Push magnitude in Newtons applied to torso (random direction, periodic). Default: 15.0 N.',
    )
    parser.add_argument(
        '--push-interval-steps',
        type=int,
        default=200,
        help='Interval in steps between push events. Default: 200.',
    )
    parser.add_argument(
        '--push-duration-steps',
        type=int,
        default=5,
        help='Duration in steps each push lasts. Default: 5.',
    )
    parser.add_argument(
        '--push-count',
        type=int,
        default=None,
        help='Override number of push events. Default: None (use steps // push_interval).',
    )
    parser.add_argument(
        '--push-start-step',
        type=int,
        default=None,
        help='Override start step for the first push. Default: None (use 50 + rand).',
    )
    parser.add_argument(
        '--sagittal-push-only',
        action='store_true',
        default=False,
        help='Force all pushes in the +y (forward sagittal) direction instead of random angles.',
    )
    parser.add_argument(
        '--push-sequence-file',
        type=str,
        default=None,
        help='Path to JSON file containing a push sequence: list of [step, force_x_N, force_y_N, duration_steps]. '
             'When provided, overrides --push-magnitude-n/--push-interval-steps for deterministic excitation.',
    )
    parser.add_argument(
        '--telemetry-decimation',
        type=int,
        default=1,
        help='Write first row, every Nth row, and final/termination row to main telemetry CSV',
    )
    parser.add_argument(
        '--failure-window-steps',
        type=int,
        default=0,
        help='Preserve the last N full-rate rows in a failure-window CSV if the run terminates early',
    )
    parser.add_argument(
        '--write-run-summary-sidecar',
        action='store_true',
        help='Write whole-run summary sidecar JSON with authoritative simulated-step counts and full-rate maxima',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for telemetry/summary output. Default: outputs/hierarchical_controller_sim. '
             'Set a unique path per run to avoid collisions when running sims in parallel.',
    )
    parser.add_argument(
        "--height-variant-setup",
        type=str,
        default=None,
        help='Path to height variant setup JSON (from B2-B4 validation) for true standing-height variant initialization',
    )
    parser.add_argument(
        "--sagittal-controller",
        type=str,
        default="baseline",
        choices=["baseline", "velocity-damped"],
        help="Select sagittal wheel controller: baseline (SagittalWheelBalanceController) or velocity-damped (SagittalVelocityDampedBalanceController)",
    )
    parser.add_argument(
        "--vd-k-pitch",
        type=float,
        default=50.0,
        help="velocity-damped controller kp_pitch gain (Nm/rad). Default: 50.0. CAUTION: Reducing this below 25 may destabilize pitch control.",
    )
    parser.add_argument(
        "--vd-pitch-ref-offset-deg",
        type=float,
        default=0.0,
        help="Pitch reference offset in degrees. Positive = forward lean reference, Negative = backward lean reference. Default: 0.0 (zero pitch reference). CAUTION: Large offsets may destabilize.",
    )
    # Phase B support-position outer-loop gain overrides (screening only).
    # When set (not None) they override the resolved profile's outer_loop_* gains.
    # Used by the Phase 4 sign/gain sweep so candidates can be screened without
    # editing the source constant. Default None = use the profile value.
    parser.add_argument(
        "--vd-outer-loop-kp-deg-per-m",
        type=float,
        default=None,
        help="Override outer_loop_kp_deg_per_m for the support-position outer loop. Default: None (use profile). Sign selects restoring direction.",
    )
    parser.add_argument(
        "--vd-outer-loop-kd-deg-per-mps",
        type=float,
        default=None,
        help="Override outer_loop_kd_deg_per_mps for the support-position outer loop. Default: None (use profile).",
    )
    parser.add_argument(
        "--vd-outer-loop-ki-deg-per-m-s",
        type=float,
        default=None,
        help="Override outer_loop_ki_deg_per_m_s for the support-position outer loop. Default: None (use profile).",
    )
    parser.add_argument(
        "--vd-outer-loop-integral-enabled",
        action="store_true",
        help="Force-enable the support-position outer-loop integral path (calibration sweep).",
    )
    parser.add_argument(
        "--vd-outer-loop-theta-ref-max-deg",
        type=float,
        default=None,
        help="Override outer_loop_theta_ref_max_deg (saturation half-range, deg). Default: None (use profile).",
    )
    parser.add_argument(
        "--vd-outer-loop-deadband-m",
        type=float,
        default=None,
        help="Override outer_loop_support_error_deadband_m (m). Default: None (use profile).",
    )
    parser.add_argument(
        "--vd-outer-loop-rate-limit-deg-per-step",
        type=float,
        default=None,
        help="Override outer_loop_theta_ref_rate_limit_deg_per_step. Default: None (use profile).",
    )
    parser.add_argument(
        "--vd-outer-loop-lowpass-alpha",
        type=float,
        default=None,
        help="Override outer_loop_theta_ref_lowpass_alpha. Default: None (use profile).",
    )
    parser.add_argument(
        "--vd-low-band-support-sigma-m",
        type=float,
        default=None,
        help="Override low-band support Gaussian sigma for validation sweeps only. Default: None (use profile).",
    )
    parser.add_argument(
        "--vd-low-band-support-kp-peak-deg-per-m",
        type=float,
        default=None,
        help="Override low-band support peak Kp for validation sweeps only. Default: None (use profile).",
    )
    parser.add_argument(
        "--vd-low-band-support-pitch-ref-offset-peak-deg",
        type=float,
        default=None,
        help="Override low-band support peak pitch-ref trim for validation sweeps only. Default: None (use profile).",
    )
    parser.add_argument(
        "--vd-k-position",
        type=float,
        default=40.0,
        help="velocity-damped controller k_position gain (Nm/m). Default: 40.0 (Step E position-return migration: restores effective return coefficient after kp_cp=0.0)",
    )
    parser.add_argument(
        "--vd-k-velocity",
        type=float,
        default=15.0,
        help="velocity-damped controller k_velocity gain (Nm/(m/s)). Default: 15.0 (F4c)",
    )
    parser.add_argument(
        "--vd-k-support-velocity",
        type=float,
        default=0.0,
        help="velocity-damped controller k_support_velocity gain (Nm/(m/s)). Damps support-center velocity to prevent position drift. Default: 0.0 (disabled)",
    )
    parser.add_argument(
        "--vd-max-position-tau",
        type=float,
        default=3.0,
        help="velocity-damped controller max position-return torque (Nm). Clips tau_position before summing. Default: 3.0",
    )
    parser.add_argument(
        "--vd-pitch-rate-filter-alpha",
        type=float,
        default=0.3,
        help="Pitch rate consistency estimator low-pass filter alpha [0,1]. Higher = more filtering. Default: 0.3",
    )
    parser.add_argument(
        "--vd-pitch-rate-min-sign-check",
        type=float,
        default=0.01,
        help="Minimum absolute pitch rate (rad/s) to check sign consistency. Default: 0.01",
    )
    parser.add_argument(
        "--vd-enable-pitch-rate-correction",
        action="store_true",
        default=False,
        help="Enable pitch rate consistency correction in velocity-damped controller. DISABLED by default (fix was ineffective and caused height variant regressions).",
    )
    parser.add_argument(
        "--vd-position-ramp-steps",
        type=int,
        default=0,
        help="Ramp position-hold authority from 0 to full over this many steps. 0 = instant (default). Diagnostic for transient disambiguation.",
    )
    parser.add_argument(
        "--vd-balance-safety-scheduling",
        action="store_true",
        default=False,
        help="Enable balance-safety scheduling: reduce tau_position when pitch/height is unsafe. Diagnostic for transient disambiguation.",
    )
    parser.add_argument(
        "--vd-safety-pitch-threshold-deg",
        type=float,
        default=3.0,
        help="Pitch threshold (deg) for balance-safety scheduling. Default: 3.0",
    )
    parser.add_argument(
        "--vd-safety-com-z-threshold-m",
        type=float,
        default=0.38,
        help="COM Z threshold (m) for balance-safety scheduling. Default: 0.38",
    )
    # Transient capture diagnostic variants (Task 2)
    parser.add_argument(
        "--vd-transient-capture-mode",
        type=str,
        default="none",
        choices=["none", "T1", "T2", "T3", "T4"],
        help="Transient capture diagnostic mode: none (baseline), T1 (position freeze), T2 (position scaling), T3 (pitch-rate boost), T4 (combined). Default: none",
    )
    parser.add_argument(
        "--vd-transient-pitch-threshold-deg",
        type=float,
        default=3.0,
        help="Pitch threshold (deg) for transient detection. Default: 3.0",
    )
    parser.add_argument(
        "--vd-transient-pitch-rate-threshold",
        type=float,
        default=0.3,
        help="Pitch rate threshold (rad/s) for transient detection. Default: 0.3",
    )
    parser.add_argument(
        "--vd-transient-pitch-rate-boost-factor",
        type=float,
        default=2.0,
        help="Pitch rate damping boost factor during transient (T3/T4). Default: 2.0",
    )
    parser.add_argument(
        "--vd-transient-position-scale-min",
        type=float,
        default=0.0,
        help="Minimum position authority scale during transient (T2/T4). Default: 0.0",
    )
    # Smart position hold capture gate (Fix D)
    parser.add_argument(
        "--vd-enable-capture-gate",
        action="store_true",
        default=False,
        help="Enable smart position hold capture gate (Fix D). Gates tau_position only when it opposes required pitch capture direction.",
    )
    parser.add_argument(
        "--vd-capture-gate-pitch-threshold",
        type=float,
        default=0.05,
        help="Pitch threshold (rad) for capture gate activation. Default: 0.05 (~2.9 deg)",
    )
    parser.add_argument(
        "--vd-capture-gate-conflict-factor",
        type=float,
        default=0.0,
        help="Gate factor during conflict (0.0 = fully gate, 1.0 = no gating). Default: 0.0",
    )
    parser.add_argument(
        "--vd-capture-gate-smooth-steps",
        type=int,
        default=10,
        help="Number of steps for smooth gate factor transitions. Default: 10",
    )
    parser.add_argument(
        "--vd-capture-gate-use-cp",
        action="store_true",
        default=True,
        help="Use capture point for direction detection (default: True). If False, uses pitch sign only.",
    )
    # Torque-budget-aware position authority (Step E fix)
    parser.add_argument(
        "--vd-enable-torque-budget-aware-position",
        action="store_true",
        default=False,
        help="Enable torque-budget-aware position authority allocation. Replaces fixed max_position_tau with dynamic budget based on available wheel torque.",
    )
    parser.add_argument(
        "--vd-position-tau-budget-cap",
        type=float,
        default=7.0,
        help="Maximum position authority cap (Nm) for torque-budget-aware mode. Default: 7.0",
    )
    parser.add_argument(
        "--vd-enable-position-integral",
        action="store_true",
        default=False,
        help="Enable steady-state-only centering integral for final support-position bias correction.",
    )
    parser.add_argument(
        "--vd-ki-position-integral",
        type=float,
        default=0.0,
        help="Steady-state centering integral gain (Nm/(m*s)). Default: 0.0",
    )
    parser.add_argument(
        "--vd-integral-max-abs",
        type=float,
        default=1.0,
        help="Maximum absolute integral torque contribution (Nm). Default: 1.0",
    )
    parser.add_argument(
        "--vd-integral-pitch-error-threshold-rad",
        type=float,
        default=0.03,
        help="Maximum pitch error for integral activation (rad). Default: 0.03",
    )
    parser.add_argument(
        "--vd-integral-roll-error-threshold-rad",
        type=float,
        default=0.05,
        help="Maximum roll error for integral activation (rad). Default: 0.05",
    )
    parser.add_argument(
        "--vd-integral-pitch-rate-threshold-rad-s",
        type=float,
        default=0.05,
        help="Maximum pitch rate for integral activation (rad/s). Default: 0.05",
    )
    parser.add_argument(
        "--vd-integral-support-velocity-threshold-m-s",
        type=float,
        default=0.03,
        help="Maximum support-position velocity for integral activation (m/s). Default: 0.03",
    )
    parser.add_argument(
        "--vd-integral-wheel-velocity-threshold-rad-s",
        type=float,
        default=1.0,
        help="Maximum mean wheel velocity for integral activation (rad/s). Default: 1.0",
    )
    parser.add_argument(
        "--vd-integral-min-com-z-m",
        type=float,
        default=0.38,
        help="Minimum safe COM height for integral activation (m). Default: 0.38",
    )
    parser.add_argument(
        "--vd-integral-max-com-z-m",
        type=float,
        default=0.43,
        help="Maximum safe COM height for integral activation (m). Default: 0.43",
    )
    parser.add_argument(
        "--vd-sagittal-authority-profile",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "candidate_A_position_cap",
            "candidate_A2_height_staged",
            "candidate_B_balanced",
            "candidate_C_stronger_position",
            "candidate_D1_support_velocity_light",
            "candidate_D2_wheel_velocity_damping_light",
            "candidate_E1_k60_continuous",
            "candidate_E2_k80_continuous",
            "candidate_E3_k100_continuous",
            "E1_support_integral",
            "E2_support_integral_higher_cap",
            "E2b_support_integral_higher_cap_aligned_gate",
            "E3_support_integral_cap_wheel_damping",
            "J1",
            "J2",
            "J3",
            "J2a",
            "J2b",
            "J2c",
            "J2d",
            "F1_phase_aware_recenter_velocity_shaping",
            "F1_phase_aware_recenter_wider_yaw_gate",
            "F1_phase_aware_recenter_wider_yaw_gate_low_tau",
            "F2a_hysteresis_recenter_moderate",
            "F2b_hysteresis_recenter_strong",
            "G1a_bias_cancel_moderate",
            "G1b_bias_cancel_strong",
            "APC1_active_pitch_crossing_moderate",
            "APC2_active_pitch_crossing_stronger",
            "APCR1_active_pitch_crossing_recovery_moderate",
            "APCR1b_active_pitch_crossing_early_release",
            "APCR1c_active_pitch_crossing_early_activation",
            "APCR1d_symmetric_soft_band_control",
            "APCR1e_adaptive_symmetric_soft_band",
            "APCR1f_adaptive_fast_response_phase_brake",
            "APCR1g_predictive_fast_response_phase_brake",
            "APCR1h_support_drift_priority_fast_recenter",
            "APCR1i_support_hysteresis_recenter",
            "APCR1j_support_hysteresis_higher_authority",
            "APCR1k_support_hysteresis_early_entry",
            "APCR1l_pitch_suppress_recenter",
            "APCR1m_conditional_pitch_blend_recenter",
            "APCR1n_recenter_priority_torque_boost",
            "APCR1nD_direct_support_recenter_features",
            "APCR1nD_T1_early_entry",
            "APCR1nD_T2_hold_outside_band",
            "APCR1nD_T3_early_entry_plus_hold",
            "APCR1nD_T4_stronger_authority",
            "APCR1nD_T5_band_limited_balanced",
            "T6A_high_early_hard_band",
            "T6B_high_stronger_emergency",
            "T6C_high_early_plus_stronger",
            "T6D_high_transient_boost",
            "T6E_high_pitch_aware_boost",
            "T6F_budget_cap_raise",
            "emergency_budget_cap_raise",
            "T6F_sign_corrected",
            "T6H_soft_blend_arch_fix",
            "T6I_phase_aware_release",
            "phase_aware_authority_release",
            "T6J_centering_bias_trim",
            "support_centering_bias_trim",
            "adaptive_support_centering_trim",
            "zero_crossing_support_recenter",
            "early_zero_crossing_recenter",
            "early_zero_crossing_recenter_v2",
            "pitch_bias_compensated_zero_crossing_recenter",
            "pitch_equilibrium_trim",
            "height_scheduled_pitch_equilibrium_trim",
            "support_position_outer_loop_pitch_ref",
            "calibrated_support_position_outer_loop_pitch_ref",
            "calibrated_support_position_outer_loop_pitch_ref_v2",
            "physics_equilibrium_feedforward_outer_loop",
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v1",
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v2",
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1",
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_wheel_yaw_v1",
            "i_support_reference_reacquisition_v1",
            "j1a_tall_kd_pitch_v1",
            "j1b_tall_kd_pitch_v1",
            "j1c_tall_kd_pitch_v1",
            "j2a_tall_k_wheel_vel_v1",
            "j2b_tall_k_wheel_vel_v1",
            "j2c_tall_k_wheel_vel_v1",
            "j3a_tall_combined_v1",
            "j3b_tall_combined_v1",
            "k1_pitch_rate_notch_v1",
            "k1b_pitch_rate_notch_2p3",
            "k1c_pitch_rate_notch_2p7",
            "k1d_pitch_rate_notch_q4",
            "k1e_pitch_rate_notch_q8",
            "k1f_pitch_rate_notch_blend075",
            "k1g_pitch_rate_notch_blend050",
            "k2_notch_low_q_v1",
            "k2_wheel_vel_notch_v1",
            "k3_pitch_rate_wheel_vel_notch_v1",
            "k3b_pitch_rate_wheel_vel_notch_blend075",
            # L_K1_COORDINATED_SAGITTAL_STATE_FEEDBACK_V1 family
            "l1_k1_coordinated_low_freq_feedback_v1",
            "l2_k1_coordinated_phase_lead_v1",
            "l3_k1_coordinated_pitch_ref_stabilization_v1",
            # LR_K1_REPLACEMENT_COORDINATED_FEEDBACK_V1 family
            "lr1_k1_replacement_coordinated_low_freq_v1",
            "lr2_k1_replacement_phase_lead_v1",
            "lr3_k1_replacement_pitch_ref_stabilized_v1",
            # LRS family — Sign-audited constrained gain sweep
            "lrs1_support_dominant_v1",
            "lrs2_pitch_rate_damping_v1",
            "lrs3_balanced_medium_v1",
            # LP family — Priority Sagittal Allocator
            "lp1_k1_priority_pitch_first_support_residual_v1",
            "lp2_k1_priority_pitch_strong_support_soft_v1",
            "lp3_k1_priority_support_recenter_when_safe_v1",
            # M_K1_BODY_YAW_CORRECT_ACTUATOR_V1 family
            "m1_k1_body_yaw_diff_wheel_v1",
            "m2_k1_body_yaw_support_aware_v1",
            # N_K1_MILD_DAMPING_DIAGNOSTIC_V1
            "n1_k1_mild_phase_lead_damping_v1",
            # N_K1_MILD_DAMPING_MICRO_SWEEP_VARIANTS
            "n1b_k1_mild_phase_lead_v1",
            "n1c_k1_mild_phase_lead_v1",
            "n1d_k1_mild_phase_lead_v1",
            "unified_sagittal_state_feedback_no_offset",
            "band_limited_support_recenter",
            # K_SWEEP audit-only filter parameter sweep profiles
            "k_sweep_fc_1p50", "k_sweep_fc_1p75", "k_sweep_fc_2p00",
            "k_sweep_fc_2p25", "k_sweep_fc_2p75", "k_sweep_fc_3p00",
            "k_sweep_fc_3p25", "k_sweep_fc_3p50",
            "k_sweep_q_2p0", "k_sweep_q_3p0", "k_sweep_q_8p0", "k_sweep_q_10p0",
            "k_sweep_blend_0p00", "k_sweep_blend_0p25",
            "k_sweep_notch_disabled",
            "k_sweep_lp_3p0", "k_sweep_lp_4p0", "k_sweep_lp_5p0", "k_sweep_lp_6p0",
        ],
        help="Height-variant-aware sagittal authority schedule. Default: baseline",
    )
    # Shape posture controller overrides (for isolation experiments)
    parser.add_argument(
        "--shape-kp-hip-yaw",
        type=float,
        default=None,
        help="Override shape posture controller kp_hip_yaw (default: 15.0 for balance-core). For isolation experiments only.",
    )
    parser.add_argument(
        "--shape-kd-hip-yaw",
        type=float,
        default=None,
        help="Override shape posture controller kd_hip_yaw (default: 3.0 for balance-core). For isolation experiments only.",
    )
    # HY-FF: Hip-yaw support-error feedforward compensation (candidate fix)
    parser.add_argument(
        "--enable-hip-yaw-support-feedforward",
        action="store_true",
        help="Enable hip-yaw support-error feedforward compensation (HY-FF candidate fix).",
    )
    parser.add_argument(
        "--hip-yaw-support-k",
        type=float,
        default=0.0,
        help="Support-error feedforward gain for hip-yaw (HY-FF). Default: 0.0 (disabled).",
    )
    parser.add_argument(
        "--hip-yaw-support-tau-max",
        type=float,
        default=1.0,
        help="Maximum compensation torque for hip-yaw support feedforward (Nm). Default: 1.0.",
    )
    parser.add_argument(
        "--hip-yaw-support-sign",
        type=float,
        default=1.0,
        help="Sign of support-error feedforward compensation (+1.0 or -1.0). Default: +1.0.",
    )
    # HY2-DIV: Hip-yaw divergence damping (Phase 3 candidate)
    parser.add_argument(
        "--enable-hip-yaw-divergence-damping",
        action="store_true",
        help="Enable hip-yaw divergence damping (HY2-DIV candidate fix).",
    )
    parser.add_argument(
        "--hip-yaw-divergence-k",
        type=float,
        default=0.0,
        help="Divergence proportional gain for hip-yaw (HY2-DIV). Default: 0.0 (disabled).",
    )
    parser.add_argument(
        "--hip-yaw-divergence-kd",
        type=float,
        default=0.0,
        help="Divergence derivative gain for hip-yaw (HY2-DIV). Default: 0.0 (disabled).",
    )
    parser.add_argument(
        "--hip-yaw-divergence-tau-max",
        type=float,
        default=0.5,
        help="Maximum divergence damping torque for hip-yaw (Nm). Default: 0.5.",
    )
    parser.add_argument(
        "--hip-yaw-divergence-z-low",
        type=float,
        default=0.300,
        help="Lower height threshold for HY2-DIV gate (m). Default: 0.300.",
    )
    parser.add_argument(
        "--hip-yaw-divergence-z-high",
        type=float,
        default=0.393,
        help="Upper height threshold for HY2-DIV gate (m). Default: 0.393.",
    )
    parser.add_argument(
        "--boundary-yaw-position-profile",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "yaw_aware_position_only",
            "boundary_hip_yaw_profile",
            "yaw_aware_plus_boundary_hip_yaw",
            "boundary_hip_yaw_integral_light",
            "yaw_aware_plus_integral_light",
        ],
        help="Boundary-height yaw-position coupling fix profile. Default: baseline (no changes)",
    )
    parser.add_argument(
        "--boundary-hip-yaw-kp",
        type=float,
        default=22.0,
        help="Boundary-only hip-yaw kp when boundary_hip_yaw_profile is active (default: 22.0)",
    )
    parser.add_argument(
        "--boundary-hip-yaw-kd",
        type=float,
        default=4.5,
        help="Boundary-only hip-yaw kd when boundary_hip_yaw_profile is active (default: 4.5)",
    )
    parser.add_argument(
        "--boundary-hip-yaw-integral-gain",
        type=float,
        default=2.0,
        help="Weak hip-yaw integral gain for boundary variants (default: 2.0)",
    )
    parser.add_argument(
        "--boundary-hip-yaw-integral-max",
        type=float,
        default=1.0,
        help="Hip-yaw integral anti-windup clamp in Nm (default: 1.0)",
    )
    parser.add_argument(
        "--yaw-controller-kp",
        type=float,
        default=8.0,
        help="YawController proportional gain on yaw error [Nm/rad]. Default: 8.0",
    )
    parser.add_argument(
        "--yaw-controller-kd",
        type=float,
        default=2.0,
        help="YawController derivative gain on yaw rate [Nm/(rad/s)]. Default: 2.0",
    )
    parser.add_argument(
        "--yaw-controller-max-torque",
        type=float,
        default=5.0,
        help="YawController max antisymmetric hip-yaw torque [Nm]. Default: 5.0",
    )

    # ---- Differential Wheel Yaw Stabilizer (BODY_YAW_WRONG_ACTUATOR fix) ---- #
    parser.add_argument(
        "--enable-wheel-yaw-stabilizer",
        action="store_true",
        default=False,
        help="Enable differential wheel yaw stabilizer. Body-yaw correction is split: "
             "wheels handle bulk via differential torque, hip-yaw handles fine correction "
             "at reduced gain (30%%). Opt-in only.",
    )
    parser.add_argument(
        "--wheel-yaw-kp",
        type=float,
        default=5.0,
        help="Wheel yaw stabilizer proportional gain on yaw error [Nm/rad]. Default: 5.0",
    )
    parser.add_argument(
        "--wheel-yaw-kd",
        type=float,
        default=1.5,
        help="Wheel yaw stabilizer derivative gain on yaw rate [Nm/(rad/s)]. Default: 1.5",
    )
    parser.add_argument(
        "--wheel-yaw-max-torque",
        type=float,
        default=5.0,
        help="Wheel yaw stabilizer max antisymmetric torque per wheel [Nm]. Default: 5.0",
    )
    parser.add_argument(
        "--wheel-yaw-lowpass-alpha",
        type=float,
        default=1.0,
        help="Wheel yaw stabilizer output lowpass alpha [0,1]. 1.0 = no filter (default). "
             "The composer handles rate limiting, so no filter is safe.",
    )
    parser.add_argument(
        "--wheel-yaw-height-gate-low",
        type=float,
        default=0.250,
        help="Height below which wheel yaw is fully gated off [m]. Default: 0.250",
    )
    parser.add_argument(
        "--wheel-yaw-height-gate-high",
        type=float,
        default=0.350,
        help="Height above which full wheel yaw authority applies [m]. Default: 0.350",
    )

    # ---- Mode-Based Hip-Yaw Divergence Controller (architecture fix candidate) ---- #
    # Opt-in only. When --enable-mode-hip-yaw-divergence is set the antisymmetric
    # hip-yaw component owned by the divergence mode is computed by
    # ``ModeBasedHipYawDivergenceController`` and added into the hip-yaw torque
    # AFTER the YawController's body-yaw correction. This is independent from the
    # older ``enable_hip_yaw_divergence_damping`` flag which lives inside
    # ShapePostureController.
    parser.add_argument(
        "--enable-mode-hip-yaw-divergence",
        action="store_true",
        default=False,
        help="Enable mode-based hip-yaw divergence controller (architecture fix candidate). "
             "Default: disabled. Opt-in only.",
    )
    parser.add_argument(
        "--mode-hip-yaw-div-kp",
        type=float,
        default=1.0,
        help="Mode-based hip-yaw divergence proportional gain [Nm/rad]. Default: 1.0",
    )
    parser.add_argument(
        "--mode-hip-yaw-div-kd",
        type=float,
        default=0.10,
        help="Mode-based hip-yaw divergence derivative gain [Nm/(rad/s)]. Default: 0.10",
    )
    parser.add_argument(
        "--mode-hip-yaw-div-max-torque",
        type=float,
        default=1.0,
        help="Mode-based hip-yaw divergence max torque magnitude [Nm]. Default: 1.0",
    )
    parser.add_argument(
        "--mode-hip-yaw-div-soft-limit-rad",
        type=float,
        default=0.30,
        help="Mode-based hip-yaw divergence soft limit height [m]. Default: 0.30",
    )
    parser.add_argument(
        "--mode-hip-yaw-div-soft-gain",
        type=float,
        default=0.10,
        help="Mode-based hip-yaw divergence soft limit gain above soft-limit [m]. Default: 0.10",
    )
    parser.add_argument(
        "--mode-hip-yaw-div-ref-source",
        type=str,
        default="target",
        choices=["target", "zero_only_for_debug"],
        help="Mode-based hip-yaw divergence reference source. Default: 'target'.",
    )
    # ---- Support-aware mode-div gating (opt-in) ---- #
    parser.add_argument(
        "--mode-hip-yaw-div-support-enabled",
        action="store_true",
        default=False,
        help="Enable support-aware mode-div authority gating",
    )
    parser.add_argument(
        "--mode-hip-yaw-div-support-threshold-m",
        type=float,
        default=0.25,
        help="Support error threshold (m) below which support gate = 1.0",
    )
    parser.add_argument(
        "--mode-hip-yaw-div-support-width-m",
        type=float,
        default=0.10,
        help="Support error width (m) over which gate transitions from 1.0 to min",
    )
    parser.add_argument(
        "--mode-hip-yaw-div-support-min-gate",
        type=float,
        default=0.70,
        help="Minimum support-error gate value (at large support error)",
    )
    parser.add_argument(
        "--mode-hip-yaw-div-support-rate-threshold-mps",
        type=float,
        default=0.05,
        help="Support error rate threshold (m/s) below which rate gate = 1.0",
    )
    parser.add_argument(
        "--mode-hip-yaw-div-support-rate-width-mps",
        type=float,
        default=0.03,
        help="Support error rate width (m/s) over which rate gate transitions",
    )
    parser.add_argument(
        "--mode-hip-yaw-div-support-rate-min-gate",
        type=float,
        default=0.70,
        help="Minimum support-rate gate value",
    )

    # ---- Dynamic height trajectory (for true dynamic-height validation) ---- #
    parser.add_argument(
        "--dynamic-height-trajectory",
        type=str,
        default=None,
        help="Path to JSON file defining height waypoints for dynamic-height simulation. "
             "Format: {\"height_profile_name\": \"...\", \"waypoints\": [{\"step\": N, \"height_m\": H}, ...]}. "
             "Height is linearly interpolated between waypoints during the simulation. "
             "When active, the robot's posture is updated dynamically from height_cmd "
             "via the posture regularizer, and commanded_height_ref_m tracks the trajectory.",
    )

    args = parser.parse_args()

    # Validate balance-core mode arguments
    validate_balance_core_mode_args(args)

    print("=" * 80)
    print("Hierarchical Controller Simulation with Telemetry")
    print(f"Mode: {'VISUAL' if args.visual else 'HEADLESS'}")
    print("=" * 80)

    # Create output directory
    _out_dir_arg = getattr(args, "output_dir", None)
    output_dir = Path(_out_dir_arg) if _out_dir_arg else Path("outputs/hierarchical_controller_sim")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load robot model
    model_path = "assets/robot/wheeled_biped_real.xml"
    print(f"\nLoading model: {model_path}")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    contact_jacobian = ContactJacobian(mj_model)

    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
    l_wheel_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    # Load height-variant setup if provided (B5-B10 validation)
    height_variant_setup = None
    if args.height_variant_setup:
        with open(args.height_variant_setup, "r") as f:
            height_variant_setup = json.load(f)
        print(f"[HEIGHT VARIANT] Loaded setup: {height_variant_setup['variant_name']}")
        print(f"[HEIGHT VARIANT] Target CoM Z: {height_variant_setup['target_com_z_m']:.6f} m")

    # ---- Dynamic height trajectory (opt-in) ---- #
    dynamic_height_traj = None  # Will hold {"profile_name": str, "waypoints": [(step, height_m), ...], "interp_fn": callable}
    if args.dynamic_height_trajectory:
        import json as _json
        with open(args.dynamic_height_trajectory, "r") as f:
            traj_data = _json.load(f)
        waypoints = [(wp["step"], wp["height_m"]) for wp in traj_data["waypoints"]]
        waypoints.sort(key=lambda x: x[0])
        # Pre-compute segments for fast lookup
        traj_segments = []
        for i in range(len(waypoints) - 1):
            s0, h0 = waypoints[i]
            s1, h1 = waypoints[i + 1]
            if s1 > s0:
                traj_segments.append((s0, s1, h0, h1))
        def _interp_height(step: int) -> float:
            if step <= traj_segments[0][0]:
                return traj_segments[0][2]
            for s0, s1, h0, h1 in traj_segments:
                if s0 <= step < s1:
                    frac = (step - s0) / (s1 - s0)
                    return h0 + frac * (h1 - h0)
            return traj_segments[-1][3]
        dynamic_height_traj = {
            "profile_name": traj_data.get("height_profile_name", "unknown"),
            "waypoints": waypoints,
            "interp_fn": _interp_height,
        }
        print(f"[DYNAMIC HEIGHT] Loaded trajectory: {dynamic_height_traj['profile_name']} "
              f"({len(waypoints)} waypoints, {max(w[0] for w in waypoints)} steps)")

    # Initialize robot on ground using keyframe 0 or height-variant setup
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        print("[OK] Robot initialized using keyframe 0")

        # Apply height-variant posture if provided
        if height_variant_setup:
            print("[HEIGHT VARIANT] Applying variant posture...")
            # Apply hip_pitch and knee references (symmetric left/right)
            mj_data.qpos[9] = height_variant_setup["hip_pitch_ref"]   # l_hip_pitch
            mj_data.qpos[10] = height_variant_setup["knee_ref"]        # l_knee
            mj_data.qpos[14] = height_variant_setup["hip_pitch_ref"]  # r_hip_pitch
            mj_data.qpos[15] = height_variant_setup["knee_ref"]        # r_knee

            # Apply hip_roll references if different from keyframe
            mj_data.qpos[7] = height_variant_setup["hip_roll_left"]   # l_hip_roll
            mj_data.qpos[12] = height_variant_setup["hip_roll_right"] # r_hip_roll

            # Apply hip_yaw references if available
            if "hip_yaw_left" in height_variant_setup:
                mj_data.qpos[8] = height_variant_setup["hip_yaw_left"]   # l_hip_yaw
                mj_data.qpos[13] = height_variant_setup["hip_yaw_right"] # r_hip_yaw

            # Zero velocities and accelerations
            mj_data.qvel[:] = 0.0
            mj_data.qacc[:] = 0.0

            print(f"[HEIGHT VARIANT] Hip pitch: {height_variant_setup['hip_pitch_ref']:.4f} rad")
            print(f"[HEIGHT VARIANT] Knee: {height_variant_setup['knee_ref']:.4f} rad")

    print("\n=== INITIALIZATION DIAGNOSTICS ===")
    root_z_before_calib = float(mj_data.qpos[2])
    mujoco.mj_forward(mj_model, mj_data)
    before_contact = measure_wheel_floor_contact(
        mj_model, mj_data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id
    )
    before_min_dist = before_contact["min_dist"]
    print(f"[INIT CALIB] root_z before calibration: {root_z_before_calib:+.6f}")
    print(
        "[INIT CALIB] min wheel-floor contact.dist before calibration: "
        f"{before_min_dist:+.6f}" if before_min_dist is not None else "[INIT CALIB] min wheel-floor contact.dist before calibration: <none>"
    )

    # Use pre-calibrated root_z from height-variant setup if available
    if height_variant_setup and "calibrated_root_z_m" in height_variant_setup:
        print(f"[HEIGHT VARIANT] Using pre-calibrated root_z: {height_variant_setup['calibrated_root_z_m']:.6f} m")
        mj_data.qpos[2] = height_variant_setup["calibrated_root_z_m"]
        mujoco.mj_forward(mj_model, mj_data)
    else:
        # Standard root_z calibration for nominal/keyframe initialization
        calibrate_root_z_for_wheel_floor_contact(
            mj_model,
            mj_data,
            target_dist=-5e-4,
            max_iters=5,
        )

    root_z_after_calib = float(mj_data.qpos[2])
    after_contact = measure_wheel_floor_contact(
        mj_model, mj_data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id
    )
    after_min_dist = after_contact["min_dist"]
    print(f"[INIT CALIB] root_z after calibration: {root_z_after_calib:+.6f}")
    print(
        "[INIT CALIB] min wheel-floor contact.dist after calibration: "
        f"{after_min_dist:+.6f}" if after_min_dist is not None else "[INIT CALIB] min wheel-floor contact.dist after calibration: <none>"
    )

    # Initialize controllers
    print("\nInitializing hierarchical controller...")
    robot_mass = float(np.sum(mj_model.body_mass))
    gravity = float(abs(mj_model.opt.gravity[2]))
    centroidal_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=robot_mass, torso_inertia=jnp.array([0.1, 0.1, 0.05])
        ),
        mj_model=mj_model,
    )
    capture_estimator = CapturePointEstimator(
        CapturePointEstimatorConfig(gravity=gravity, min_height=0.35)
    )
    wbc_controller = IntegratedWBC(
        mj_model,
        k_roll=60.0,
        k_roll_rate=12.0,
        k_roll_integral=0.0,
        k_pitch=300.0,  # TUNED: Optimal balance - strong enough without oscillation (tested: 800 too high, 150 too low)
        k_pitch_rate=15.0,  # TUNED: Proportional damping with 20:1 ratio
        k_com_lateral=15.0,
        k_com_lateral_damping=3.0,
        k_com_sagittal=50.0,  # INCREASED: Faster CoM positioning to help wheels move under CoM (was 30.0)
        k_com_sagittal_damping=6.0,  # MODERATE INCREASE: 3x higher for proper damping
        k_cp_lateral=50.0,  # REVERTED: Back to best config (Test 2: 46 steps with k_cp_lateral=25.0)
        k_cp_sagittal=100.0,  # REVERTED: Back to best config (Test 2: 46 steps with k_cp_sagittal=50.0)
        k_height=50.0,  # OPTIMAL: Best balance between contact maintenance and overshoot prevention (tested: 80 and 150 too high)
        k_height_damping=args.height_damping,
        robot_mass=robot_mass,
        gravity=gravity,
        max_roll_moment=25.0,
        wbc_authority_budget=0.95,  # INCREASED: Use more motor capability (0.95 × 60 = 57 Nm limit)
        max_actuator_torque=60.0,  # Increased from 30 to 60 Nm
        force_feedback_gain=0.2,  # FIXED: Reduced from 0.8 to 0.2 to eliminate phase lag oscillations (was causing 3.3x scale swings)
        force_feedback_warmup_steps=5,  # FIXED: Added 5-step warmup to avoid reacting to mj_forward artifacts at t=0
        tau_hip_roll_max=15.0,
        max_force_asymmetry=60.0,  # INCREASED: Allow larger asymmetry to prevent wheel liftoff (was 40.0)
        min_wheel_force=20.0,  # INCREASED: Higher minimum to prevent wheel liftoff (was 10.0)
        roll_integral_limit=0.52,  # Anti-windup limit: ~30 degrees
        dt=mj_model.opt.timestep,
        use_per_actuator_authority=args.use_per_actuator_wbc_authority,
        verbose=not (args.visual or getattr(args, "wbc_quiet", False)),
    )
    momentum_coordinator = MomentumCoordinator(
        MomentumCoordinatorConfig(
            k_momentum_lateral=0.8,
            k_momentum_sagittal=1.2,
            k_angular_roll=1.5,
            k_feedforward=5.0,
            k_feedforward_hip=2.0,
            momentum_authority_budget=0.15,  # 15% of 60 Nm = 9 Nm
        )
    )
    posture_regularizer = PostureRegularizer(
        PostureRegularizerConfig(
            k_posture=10.0,
            k_hip_roll=3.0,
            k_hip_yaw=1.5,
            k_hip_pitch=30.0,
            k_knee=30.0,
            k_wheel=0.0,
            hip_roll_deadband=0.15,  # ±8.6° - LARGE deadband, hip roll must be free for balance
            hip_yaw_deadband=0.02,  # ±1.1° - tighter, yaw drift is bad
            hip_pitch_deadband=0.035,  # ±2.0° - reduced for earlier activation
            knee_deadband=0.05,  # ±2.9° - reduced for earlier activation
            wbc_error_threshold=0.3,
            momentum_activity_threshold=0.1,
            momentum_active_scale=0.5,
            posture_authority_budget=0.40,
            max_actuator_torque=60.0,
        )
    )

    # Secondary posture controller only; WBC is the primary torque path.
    leg_position_controller = LegPositionController(
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=20.0,
        kd_hip_pitch=3.0,
        kp_knee=35.0,
        kd_knee=4.0,
        max_torque=25.0,
    )

    # Stage 2: Static posture holding controller for correction-only WBC
    static_posture_controller = None
    if args.enable_stage2_static_posture_hold:
        static_posture_controller = StaticPostureHoldingController(
            kp_hip_roll=5.0,
            kd_hip_roll=1.0,
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            kp_hip_pitch=args.static_kp_hip_pitch,
            kd_hip_pitch=args.static_kd_hip_pitch,
            kp_knee=args.static_kp_knee,
            kd_knee=args.static_kd_knee,
            max_torque_hip_roll=15.0,
            max_torque_hip_yaw=15.0,
            max_torque_hip_pitch=args.static_max_torque_hip_pitch,
            max_torque_knee=args.static_max_torque_knee,
        )
        print(f"[STAGE 2] StaticPostureHoldingController initialized with gains:")
        print(f"  kp_hip_pitch={args.static_kp_hip_pitch}, kd_hip_pitch={args.static_kd_hip_pitch}")
        print(f"  kp_knee={args.static_kp_knee}, kd_knee={args.static_kd_knee}")

    # Stage 2B: Static feedforward controller for gravity compensation
    static_feedforward_controller = None
    wbc_controller.set_correction_only_mode(False)
    if args.enable_stage2b_gravity_feedforward:
        if not args.enable_stage2_static_posture_hold:
            raise ValueError("Stage 2B feedforward requires Stage 2 static posture hold (--enable-stage2-static-posture-hold)")

        empirical_ff = resolve_stage2b_empirical_feedforward(args.stage2b_feedforward_telemetry_path)

        static_feedforward_controller = StaticFeedforwardController(
            empirical_feedforward=empirical_ff,
            scale=args.stage2b_feedforward_scale,
            joint_group=args.stage2b_feedforward_joint_group,
            ramp_mode=args.stage2b_feedforward_ramp,
            sign=args.stage2b_feedforward_sign,
        )
        print(f"[STAGE 2B] StaticFeedforwardController initialized:")
        if args.stage2b_feedforward_telemetry_path is None:
            print("  Empirical feedforward source: fixed validated default")
        else:
            print(f"  Empirical feedforward source: telemetry override ({Path(args.stage2b_feedforward_telemetry_path).name})")
        print(f"  Sign: {args.stage2b_feedforward_sign}")
        print(f"  Scale: {args.stage2b_feedforward_scale}")
        print(f"  Joint group: {args.stage2b_feedforward_joint_group}")
        print(f"  Ramp mode: {args.stage2b_feedforward_ramp}")
        print(f"  Ablation mode: {args.stage2b_ablation_mode}")
        print(f"  Effective feedforward (knee): {empirical_ff[3] * args.stage2b_feedforward_scale:.2f}, {empirical_ff[8] * args.stage2b_feedforward_scale:.2f} Nm")
        wbc_controller.set_correction_only_mode(True)
        print("  WBC distributor input mode: correction-only")

    # Stage 2B: Direct roll controller (alternative to WBC contact path)
    stage2b_roll_direct_controller = None
    if args.enable_stage2b_roll_direct:
        if not args.enable_stage2_static_posture_hold:
            raise ValueError("Stage 2B direct roll requires Stage 2 static posture hold (--enable-stage2-static-posture-hold)")

        stage2b_roll_direct_controller = Stage2BRollDirectController(
            k_roll=args.stage2b_roll_kp,
            k_roll_rate=args.stage2b_roll_kd,
            k_roll_integral=0.0,
            tau_hip_roll_max=args.stage2b_roll_tau_max,
            max_roll_moment=args.stage2b_roll_tau_max * 2.0,
        )
        print(f"[STAGE 2B] Stage2BRollDirectController initialized:")
        print(f"  k_roll: {args.stage2b_roll_kp} Nm/rad")
        print(f"  k_roll_rate: {args.stage2b_roll_kd} Nm/(rad/s)")
        print(f"  tau_hip_roll_max: {args.stage2b_roll_tau_max} Nm")
        print(f"  Direct roll mode: WBC contact path disabled for roll")

    # Stage 2B: Sagittal wheel controller (alternative to WBC wheel path)
    stage2b_sagittal_wheel_controller = None
    if args.enable_stage2b_sagittal_wheel:
        if not args.enable_stage2_static_posture_hold:
            raise ValueError("Stage 2B sagittal wheel requires Stage 2 static posture hold (--enable-stage2-static-posture-hold)")

        stage2b_sagittal_wheel_controller = Stage2BSagittalWheelController(
            k_pitch=args.stage2b_sagittal_k_pitch,
            k_pitch_rate=args.stage2b_sagittal_k_pitch_rate,
            k_cp=args.stage2b_sagittal_k_cp,
            k_com_y=args.stage2b_sagittal_k_com_y,
            k_com_vy=args.stage2b_sagittal_k_com_vy,
            max_tau_wheel=args.stage2b_sagittal_max_tau,
        )
        print(f"[STAGE 2B] Stage2BSagittalWheelController initialized:")
        print(f"  k_pitch: {args.stage2b_sagittal_k_pitch} Nm/rad")
        print(f"  k_pitch_rate: {args.stage2b_sagittal_k_pitch_rate} Nm/(rad/s)")
        print(f"  k_cp: {args.stage2b_sagittal_k_cp} Nm/m")
        print(f"  k_com_y: {args.stage2b_sagittal_k_com_y} Nm/m")
        print(f"  k_com_vy: {args.stage2b_sagittal_k_com_vy} Nm/(m/s)")
        print(f"  max_tau_wheel: {args.stage2b_sagittal_max_tau} Nm")
        print(f"  Direct wheel mode: WBC wheel path disabled for pitch")

    # Stage 2C: Sagittal state-feedback controller (alternative to Stage 2B)
    stage2c_sagittal_state_feedback_controller = None
    if args.enable_stage2c_sagittal_state_feedback:
        if not args.enable_stage2_static_posture_hold:
            raise ValueError("Stage 2C sagittal state-feedback requires Stage 2 static posture hold (--enable-stage2-static-posture-hold)")
        if args.enable_stage2b_sagittal_wheel:
            raise ValueError("Stage 2C and Stage 2B sagittal controllers are mutually exclusive")

        stage2c_sagittal_state_feedback_controller = Stage2CSagittalStateFeedbackController(
            k_pitch=args.stage2c_k_pitch,
            k_pitch_rate=args.stage2c_k_pitch_rate,
            k_com_y=args.stage2c_k_com_y,
            k_com_vy=args.stage2c_k_com_vy,
            k_cp_y=args.stage2c_k_cp_y,
            k_wheel_vel=args.stage2c_k_wheel_vel,
            max_tau_wheel=args.stage2c_max_tau,
        )
        print(f"[STAGE 2C] Stage2CSagittalStateFeedbackController initialized:")
        print(f"  k_pitch: {args.stage2c_k_pitch} Nm/rad")
        print(f"  k_pitch_rate: {args.stage2c_k_pitch_rate} Nm/(rad/s)")
        print(f"  k_com_y: {args.stage2c_k_com_y} Nm/m")
        print(f"  k_com_vy: {args.stage2c_k_com_vy} Nm/(m/s)")
        print(f"  k_cp_y: {args.stage2c_k_cp_y} Nm/m")
        print(f"  k_wheel_vel: {args.stage2c_k_wheel_vel} Nm/(rad/s)")
        print(f"  max_tau_wheel: {args.stage2c_max_tau} Nm")
        print(f"  State-feedback mode: Full state feedback with wheel velocity damping")

    # Stage 2D: Sagittal LQR controller (model-based, identified dynamics)
    stage2d_sagittal_lqr_controller = None
    if args.enable_stage2d_sagittal_lqr:
        if not args.enable_stage2_static_posture_hold:
            raise ValueError("Stage 2D sagittal LQR requires Stage 2 static posture hold (--enable-stage2-static-posture-hold)")
        if args.enable_stage2b_sagittal_wheel:
            raise ValueError("Stage 2D and Stage 2B sagittal controllers are mutually exclusive")
        if args.enable_stage2c_sagittal_state_feedback:
            raise ValueError("Stage 2D and Stage 2C sagittal controllers are mutually exclusive")

        # Load identified model and create LQR controller
        model_path = Path(args.stage2d_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Stage 2D model file not found: {model_path}\n"
                f"Run Phase 1 system identification first:\n"
                f"  python scripts/identify_stage2d_sagittal_dynamics.py"
            )

        stage2d_sagittal_lqr_controller = Stage2DSagittalLQRController.from_identified_model(
            model_path=str(model_path),
            config=args.stage2d_lqr_config,
        )
        print(f"[STAGE 2D] Stage2DSagittalLQRController initialized:")
        print(f"  Model: {model_path.name}")
        print(f"  Config: {args.stage2d_lqr_config}")
        stage2d_sagittal_lqr_controller.print_analysis()

    print("[OK] Controllers initialized (wheeled biped architecture)")

    # Initialize StaticBalanceController wrapper if enabled
    static_balance_wrapper = None
    if args.enable_static_dynamics_wrapper:
        from wheeled_biped.controllers.static_balance_controller import StaticBalanceController

        print("\n[WRAPPER] Initializing StaticBalanceController wrapper...")
        calibration_config = {
            'target_contact_dist': -5e-4,
        }
        static_balance_wrapper = StaticBalanceController(
            mj_model,
            mj_data,
            wbc_controller,
            calibration_config=calibration_config,
        )
        print("[OK] StaticBalanceController wrapper initialized")

    # JIT-compile controller functions for real-time performance
    print("\nJIT-compiling controller functions...")

    # Create dummy inputs for compilation
    dummy_obs = jnp.zeros(42)
    dummy_state = centroidal_estimator.estimate(dummy_obs, mj_data, None)[0]
    dummy_state = capture_estimator.update(dummy_state)
    dummy_joint_pos = jnp.zeros(10)

    # WBC controller cannot be JIT compiled (uses MuJoCo data)
    # It will be called directly without JIT

    # Compile Momentum coordinator
    @jax.jit
    def compute_momentum_jit(obs, state):
        return momentum_coordinator.compute_momentum_coordinator_torque(obs, state)

    # Compile Posture regularizer
    @jax.jit
    def compute_posture_jit(joint_pos, wbc_error_mag, momentum_mag, height_cmd):
        return posture_regularizer.compute_posture_regularizer_torque(
            joint_pos, wbc_error_mag, momentum_mag, height_cmd
        )

    # Warmup compilation (WBC not JIT-compiled)
    _ = compute_momentum_jit(dummy_obs, dummy_state)
    _ = compute_posture_jit(dummy_joint_pos, 0.5, 0.1, 0.55)

    print("[OK] JIT compilation complete - controllers ready for real-time operation")

    # Stage 2: Set equilibrium reference for WBC and StaticPostureHoldingController
    print("\n[STAGE 2] Setting equilibrium reference...")
    # Capture equilibrium state after calibration
    mujoco.mj_forward(mj_model, mj_data)
    equilibrium_joint_pos = jnp.array(mj_data.qpos[7:17])

    # Set equilibrium reference for correction-only WBC (always needed)
    centroidal_state_eq, com_pos_eq = centroidal_estimator.estimate(jnp.zeros(42), mj_data, None)
    centroidal_state_eq = capture_estimator.update(centroidal_state_eq)

    base_body_id = 1
    R_eq = np.array(mj_data.xmat[base_body_id]).reshape(3, 3)
    gravity_world = np.array([0.0, 0.0, -gravity])
    gravity_body_eq = R_eq.T @ gravity_world
    pitch_x_eq, roll_y_eq = compute_orientation_from_gravity(jnp.array(gravity_body_eq))

    # Initial-heading sagittal axis for velocity-damped controller
    yaw_eq = float(centroidal_state_eq.body_yaw_z)
    sagittal_axis_xy_initial = (float(np.sin(yaw_eq)), float(np.cos(yaw_eq)))

    # Equilibrium support center (wheel midpoint) for support-position error
    # A wheeled biped is allowed to move its COM relative to the support center during pitch balance.
    # Position hold must track the wheel support center, NOT the COM.
    mujoco.mj_forward(mj_model, mj_data)
    l_wheel_xpos_eq = tuple(float(mj_data.xpos[l_wheel_body_id][i]) for i in range(3))
    r_wheel_xpos_eq = tuple(float(mj_data.xpos[r_wheel_body_id][i]) for i in range(3))
    support_center_eq_xy = compute_support_center_xy(l_wheel_xpos_eq, r_wheel_xpos_eq)
    print(f"[STAGE 2] Support center equilibrium: ({support_center_eq_xy[0]:.6f}, {support_center_eq_xy[1]:.6f}) m")
    print(f"  COM equilibrium: ({float(centroidal_state_eq.com_pos[0]):.6f}, {float(centroidal_state_eq.com_pos[1]):.6f}) m")

    wbc_controller.wrench_computer.set_equilibrium_reference(
        com_pos=centroidal_state_eq.com_pos,
        com_z=float(centroidal_state_eq.com_pos[2]),
        pitch_x=float(pitch_x_eq),
        roll_y=float(roll_y_eq),
        capture_point=centroidal_state_eq.capture_point,
        joint_pos=equilibrium_joint_pos,
    )
    print(f"[STAGE 2] WBC equilibrium reference set:")
    print(f"  CoM: [{float(centroidal_state_eq.com_pos[0]):.6f}, {float(centroidal_state_eq.com_pos[1]):.6f}, {float(centroidal_state_eq.com_pos[2]):.6f}] m")
    print(f"  Pitch: {float(pitch_x_eq)*57.3:.2f} deg, Roll: {float(roll_y_eq)*57.3:.2f} deg")
    print(f"  Joint pos: {[f'{float(x):.3f}' for x in equilibrium_joint_pos]}")

    # Set equilibrium reference for StaticPostureHoldingController if enabled
    if static_posture_controller is not None:
        static_posture_controller.set_equilibrium_reference(equilibrium_joint_pos)
        print(f"[STAGE 2] StaticPostureHoldingController equilibrium reference set")

    # E0 position-containment experiments were removed from runtime.
    # Failure analyses remain under outputs/balance_core_position_containment/.

    # Set equilibrium reference for Stage2B direct roll controller if enabled
    if stage2b_roll_direct_controller is not None:
        stage2b_roll_direct_controller.set_equilibrium_reference(float(roll_y_eq))
        print(f"[STAGE 2B] Stage2BRollDirectController equilibrium reference set: {float(roll_y_eq)*57.3:.2f} deg")

    # Set equilibrium reference for Stage2B sagittal wheel controller if enabled
    if stage2b_sagittal_wheel_controller is not None:
        stage2b_sagittal_wheel_controller.set_equilibrium_reference(
            pitch_x=float(pitch_x_eq),
            cp_y=float(centroidal_state_eq.capture_point[1]),
            com_y=float(centroidal_state_eq.com_pos[1]),
        )
        print(f"[STAGE 2B] Stage2BSagittalWheelController equilibrium reference set:")
        print(f"  Pitch: {float(pitch_x_eq)*57.3:.2f} deg")
        print(f"  CP Y: {float(centroidal_state_eq.capture_point[1]):.6f} m")
        print(f"  CoM Y: {float(centroidal_state_eq.com_pos[1]):.6f} m")

    if stage2c_sagittal_state_feedback_controller is not None:
        stage2c_sagittal_state_feedback_controller.set_equilibrium_reference(
            pitch_x=float(pitch_x_eq),
            com_y=float(centroidal_state_eq.com_pos[1]),
            cp_y=float(centroidal_state_eq.capture_point[1]),
        )
        print(f"[STAGE 2C] Stage2CSagittalStateFeedbackController equilibrium reference set:")
        print(f"  Pitch: {float(pitch_x_eq)*57.3:.2f} deg")
        print(f"  CoM Y: {float(centroidal_state_eq.com_pos[1]):.6f} m")
        print(f"  CP Y: {float(centroidal_state_eq.capture_point[1]):.6f} m")

    perturbation_metadata = {
        "initial_root_z_perturbation_m": 0.0,
        "nominal_equilibrium_com_z_m": float(centroidal_state_eq.com_pos[2]),
        "initial_com_z_m_after_perturbation": float(centroidal_state_eq.com_pos[2]),
        "perturbation_applied_after_equilibrium_capture": False,
    }
    if args.initial_root_z_perturbation != 0.0:
        apply_initial_root_z_perturbation(
            model=mj_model,
            data=mj_data,
            perturbation_m=args.initial_root_z_perturbation,
            nominal_equilibrium_com_z_m=float(centroidal_state_eq.com_pos[2]),
        )
        centroidal_state_after_perturbation, _ = centroidal_estimator.estimate(
            jnp.zeros(42), mj_data, None
        )
        perturbation_metadata = {
            "initial_root_z_perturbation_m": float(args.initial_root_z_perturbation),
            "nominal_equilibrium_com_z_m": float(centroidal_state_eq.com_pos[2]),
            "initial_com_z_m_after_perturbation": float(
                centroidal_state_after_perturbation.com_pos[2]
            ),
            "perturbation_applied_after_equilibrium_capture": True,
        }
        print("[STUDY] Applied initial root-z perturbation after equilibrium capture")
        print(f"  perturbation: {perturbation_metadata['initial_root_z_perturbation_m']:+.6f} m")
        print(f"  nominal equilibrium com_z: {perturbation_metadata['nominal_equilibrium_com_z_m']:.6f} m")
        print(f"  initial com_z after perturbation: {perturbation_metadata['initial_com_z_m_after_perturbation']:.6f} m")

    # Telemetry storage (all keys must be initialized here for decimation to work)
    telemetry = {
        "source_step_index": [],
        "time": [],
        "mass_kg": [],
        "weight_N": [],
        "com_x": [],
        "com_y": [],
        "com_z": [],
        "com_vx": [],
        "com_vy": [],
        "com_vz": [],
        "cp_x": [],
        "cp_y": [],
        "tau_wbc_max": [],
        "tau_wheel_actual_max": [],  # Actual wheel torques from applied tau_smooth at indices [4, 9]
        "tau_posture_max": [],
        "tau_total_max": [],
        # Euler angles (world-frame, for reference only)
        "euler_roll_x": [],
        "euler_pitch_y": [],
        "euler_yaw_z": [],
        # Robot-frame orientation (used for control and termination)
        "robot_pitch_x": [],
        "robot_roll_y": [],
        "robot_yaw_z": [],
        "roll_rate_rad_s": [],
        "pitch_rate_rad_s": [],
        "yaw_rate_rad_s": [],
        "height_cmd": [],  # Track adaptive height command
        "left_contact_active": [],
        "right_contact_active": [],
        "n_contacts": [],
        "contact_force_valid": [],
        "left_contact_force_world_x": [],
        "left_contact_force_world_y": [],
        "left_contact_force_world_z": [],
        "right_contact_force_world_x": [],
        "right_contact_force_world_y": [],
        "right_contact_force_world_z": [],
        "total_contact_force_z": [],
        "joint_pos": [],
        "joint_vel": [],
        "terminated": [],
        "termination_reason": [],
        # QP-specific metrics
        "qp_solve_time_ms": [],
        "qp_converged": [],
        "qp_error": [],
        "wrench_error_norm": [],
        "f_left_z": [],
        "f_right_z": [],
        "force_distribution_feasible": [],
        "force_distribution_reason": [],
        "distributed_left_fx": [],
        "distributed_left_fy": [],
        "distributed_left_fz": [],
        "distributed_right_fx": [],
        "distributed_right_fy": [],
        "distributed_right_fz": [],
        "tau_saturation_rate": [],
        # Desired wrench components
        "desired_wrench_Fx": [],
        "desired_wrench_Fy": [],
        "desired_wrench_Fz": [],
        "desired_wrench_Mx": [],
        "desired_wrench_My": [],
        "desired_wrench_Mz": [],
        # Motor tracking diagnostics
        "target_joint_pos": [],  # Target positions from posture regularizer
        "joint_pos_error": [],  # Position error per joint (target - actual)
        "joint_pos_error_norm": [],  # L2 norm of position error
        "joint_vel_norm": [],  # L2 norm of joint velocities
        "tau_wbc_norm": [],  # L2 norm of WBC torques
        "tau_posture_norm": [],  # L2 norm of posture torques
        "tau_inverse_dynamics_norm": [],  # L2 norm of inverse dynamics torques
        "tau_total_norm": [],  # L2 norm of total torques
        "tau_rate_unlimited": [],  # Torque rate before rate limiting (Nm/s)
        "tau_rate_limited": [],  # Torque rate after rate limiting (Nm/s)
        # Stage 2B ablation diagnostics
        "active_wheels": [],
        "left_wheel_floor_contact": [],
        "right_wheel_floor_contact": [],
        "non_wheel_floor_contacts": [],
        "total_wheel_floor_fz": [],
        "correction_wrench_norm": [],
        "correction_wrench_Fx": [],
        "correction_wrench_Fy": [],
        "correction_wrench_Fz": [],
        "correction_wrench_Mx": [],
        "correction_wrench_My": [],
        "correction_wrench_Mz": [],
        "ablation_mode": [],
        # B500 drift audit fields
        "pitch_x": [],
        "pitch_rate_x": [],
        "roll_y": [],
        "roll_rate_y": [],
        "yaw_z": [],
        "hip_roll_left_rad": [],
        "hip_roll_right_rad": [],
        "hip_roll_common_component_rad": [],
        "hip_roll_symmetric_component_rad": [],
        "hip_roll_abs_max_rad": [],
        "hip_roll_ref_left_rad": [],
        "hip_roll_ref_right_rad": [],
        "hip_roll_error_left_rad": [],
        "hip_roll_error_right_rad": [],
        "yaw_drift_from_initial_rad": [],
        "com_error_x": [],
        "com_error_y": [],
        "com_error_z": [],
        "cp_error_x": [],
        "cp_error_y": [],
        "pitch_error": [],
        "roll_error": [],
        "height_error": [],
        "left_fz_actual": [],
        "right_fz_actual": [],
        "fz_asymmetry_actual": [],
        "contact_dist_min": [],
        "contact_dist_max": [],
        "correction_Fy_com": [],
        "correction_Fy_cp": [],
        "correction_Fy_pitch": [],
        "correction_My_roll": [],
        "distributor_f_left": [],
        "distributor_f_right": [],
        "tau_hip_roll": [],
        "tau_contact": [],
        "tau_wbc_correction": [],
        "tau_wbc_after_authority_clip": [],
        "tau_static_feedforward": [],
        "tau_static_posture": [],
        "saturation_flags": [],
        "rate_limit_flags": [],
        # Wheel torque pipeline telemetry
        "tau_stage2b_sagittal_wheel_l": [],
        "tau_stage2b_sagittal_wheel_r": [],
        "tau_total_raw_l_wheel": [],
        "tau_total_raw_r_wheel": [],
        "tau_total_clipped_l_wheel": [],
        "tau_total_clipped_r_wheel": [],
        "tau_smooth_l_wheel": [],
        "tau_smooth_r_wheel": [],
        "ctrl_l_wheel": [],
        "ctrl_r_wheel": [],
        "qvel_l_wheel": [],
        "qvel_r_wheel": [],
        "sagittal_term_pitch": [],
        "sagittal_term_pitch_rate": [],
        "sagittal_term_cp": [],
        "sagittal_term_com_vy": [],
        "sagittal_term_wheel_vel_left": [],
        "sagittal_term_wheel_vel_right": [],
        "sagittal_balance_torque_raw": [],
        "sagittal_balance_torque_clipped": [],
        "sagittal_balance_torque_final": [],
        "sagittal_pitch_error": [],
        "sagittal_cp_error_y": [],
        "sagittal_tau_wheel_cmd": [],
        "sagittal_saturated": [],
        # Stage 2C: Sagittal state-feedback telemetry
        "stage2c_pitch_error": [],
        "stage2c_pitch_rate_x": [],
        "stage2c_com_y_error": [],
        "stage2c_com_vy": [],
        "stage2c_cp_y_error": [],
        "stage2c_wheel_vel_left": [],
        "stage2c_wheel_vel_right": [],
        "stage2c_wheel_vel_mean": [],
        "stage2c_term_pitch": [],
        "stage2c_term_pitch_rate": [],
        "stage2c_term_com_y": [],
        "stage2c_term_com_vy": [],
        "stage2c_term_cp_y": [],
        "stage2c_term_wheel_vel": [],
        "stage2c_tau_wheel_raw": [],
        "stage2c_tau_wheel_clipped": [],
        "stage2c_saturated": [],
        # Stage 2D: Sagittal LQR telemetry
        "stage2d_pitch_x": [],
        "stage2d_pitch_rate_x": [],
        "stage2d_cp_error_y": [],
        "stage2d_com_vy": [],
        "stage2d_wheel_vel_mean": [],
        "stage2d_u_raw": [],
        "stage2d_u_clipped": [],
        "stage2d_saturated": [],
        "stage2d_contrib_pitch_x": [],
        "stage2d_contrib_pitch_rate_x": [],
        "stage2d_contrib_cp_error_y": [],
        "stage2d_contrib_com_vy": [],
        "stage2d_contrib_wheel_vel_mean": [],
        "stage2d_config": [],
        # Control-time vs post-step orientation/rate telemetry
        "control_pitch_x": [],
        "control_pitch_rate_x": [],
        "control_roll_y": [],
        "control_roll_rate_y": [],
        "log_pitch_x": [],
        "log_pitch_rate_x": [],
        "log_roll_y": [],
        "log_roll_rate_y": [],
        "fd_pitch_rate_x": [],
        "fd_roll_rate_y": [],
        "sagittal_controller_input_pitch_x": [],
        "sagittal_controller_input_pitch_rate_x": [],
        "sagittal_controller_input_cp_y": [],
        "sagittal_controller_input_com_y": [],
        "sagittal_controller_input_com_vy": [],
        "sagittal_position_error_m": [],
        "sagittal_velocity_m_s": [],
        "support_position_velocity_m_s": [],
        "tau_position": [],
        "tau_position_raw": [],
        "position_integral_error": [],
        "tau_position_integral": [],
        "integral_active": [],
        "integral_gate_reason": [],
        "integral_saturation_flag": [],
        "tau_position_p": [],
        "tau_position_i": [],
        "tau_position_total": [],
        "tau_position_clipped": [],
        "tau_support_velocity": [],
        "tau_pitch": [],
        "tau_pitch_raw": [],
        "tau_pitch_scheduled": [],
        "tau_pitch_clipped": [],
        "tau_pitch_to_position_ratio": [],
        "sagittal_schedule_profile": [],
        "high_height_schedule_active": [],
        "effective_max_position_tau": [],
        "effective_pitch_scale": [],
        "effective_pitch_tau_cap": [],
        "effective_velocity_damping_scale": [],
        "effective_support_velocity_scale": [],
        "low_height_sagittal_schedule_active": [],
        "effective_k_position": [],
        "effective_k_velocity": [],
        "sagittal_schedule_height_reference_m": [],
        "sagittal_schedule_height_source": [],
        "sagittal_schedule_u": [],
        "sagittal_schedule_smoothstep": [],
        "tau_pitch_rate": [],
        "tau_pitch_rate_raw_signal": [],
        "tau_pitch_rate_filtered_signal": [],
        "pitch_rate_raw_rad_s": [],
        "pitch_rate_notched_rad_s": [],
        "pitch_rate_effective_rad_s": [],
        "wip_notch_height_gate": [],
        "wip_notch_filter_valid": [],
        "dynamic_height_active": [],
        "dynamic_height_target_m": [],
        "notch_height_gate_from_traj": [],
        "tau_sagittal_velocity": [],
        "tau_wheel_velocity_left": [],
        "tau_wheel_velocity_right": [],
        "max_position_tau": [],
        "tau_position_saturation_flag": [],
        "tau_position_saturation_reason": [],
        "tau_balance_before_position": [],
        "tau_position_budget_available": [],
        "tau_position_budget_allowed": [],
        "tau_position_budget_cap": [],
        "pitch_reserve_tau": [],
        "tau_pitch_reserve_applied": [],
        "enable_torque_budget_aware_position": [],
        "tau_position_lower_bound": [],
        "tau_position_upper_bound": [],
        "tau_position_total_bound_clipped": [],
        "position_authority_mode": [],
        "position_authority_reason": [],
        "tau_total_unclipped": [],
        "tau_total_clipped": [],
        "tau_total_before_final_clip": [],
        "tau_total_after_final_clip": [],
        "final_wheel_torque_margin": [],
        "k_support_velocity": [],
        "support_position_error_m": [],
        "com_position_error_sagittal_m": [],
        "pitch_x_ref_rad": [],
        "pitch_x_error_rad": [],
        # Phase B support-position outer-loop telemetry
        "outer_loop_active": [],
        "outer_loop_support_error_m": [],
        "outer_loop_support_error_rate_mps": [],
        "outer_loop_pitch_ref_dynamic_deg": [],
        "outer_loop_pitch_ref_total_deg": [],
        "outer_loop_pitch_ref_limited_deg": [],
        "outer_loop_pitch_ref_rate_limited_deg": [],
        "outer_loop_integral_m_s": [],
        "outer_loop_gate_pass": [],
        "outer_loop_block_reason": [],
        "outer_loop_sign_selected": [],
        "support_outer_loop_height_scale": [],
        "support_outer_loop_kp_effective": [],
        "support_outer_loop_kd_effective": [],
        "support_outer_loop_pitch_ref_offset_deg": [],
        "support_outer_loop_pitch_ref_contrib": [],
        "support_outer_loop_cap_active": [],
        "pitch_ref_offset_scheduled_deg": [],
        "pitch_ref_total_after_outer_loop_deg": [],
        "pitch_x_error_after_outer_loop_rad": [],
        "wheel_torque_saturation_left": [],
        "wheel_torque_saturation_right": [],
        "wheel_torque_rate_saturation_left": [],
        "wheel_torque_rate_saturation_right": [],
        # Capture gate telemetry (velocity-damped controller only)
        "capture_gate_enabled": [],
        "capture_gate_required_direction": [],
        "capture_gate_tau_position_direction": [],
        "capture_gate_position_opposes_capture": [],
        "capture_gate_factor": [],
        "capture_gate_active": [],
        "capture_gate_reason": [],
        "capture_gate_pitch_reversal": [],
        "capture_gate_capture_recovery": [],
        "capture_gate_tau_position_gated": [],
        "capture_gate_cp_relative_to_support_m": [],
        "capture_gate_com_support_error_m": [],
        # Pitch rate consistency estimator telemetry
        "pitch_rate_measured_x_rad_s": [],
        "pitch_rate_fd_x_rad_s": [],
        "pitch_rate_corrected_x_rad_s": [],
        "pitch_rate_consistency_error_rad_s": [],
        "pitch_rate_sign_mismatch": [],
        "pitch_rate_source_used": [],
        # Transient capture diagnostic telemetry
        "transient_detected": [],
        "transient_by_pitch": [],
        "transient_by_pitch_rate": [],
        "transient_by_height": [],
        "pitch_rate_boost_factor": [],
        "pitch_rate_for_control_boosted": [],
        "transient_capture_mode": [],
        # Pitch-aware position scaling telemetry
        "pitch_aware_position_scaling_enabled": [],
        "pitch_aware_position_scale": [],
        "pitch_aware_active": [],
        "pitch_soft_start": [],
        "pitch_hard_limit": [],
        "min_pitch_scale": [],
        "tau_position_before_pitch_scale": [],
        "tau_position_after_pitch_scale": [],
        # Phase-aware recenter telemetry (F1_strategy)
        "phase_recenter_enabled": [],
        "phase_recenter_active": [],
        "phase_recenter_gate_safe": [],
        "phase_recenter_signed_error_m": [],
        "phase_recenter_raw_tau": [],
        "phase_recenter_tau": [],
        "phase_recenter_tau_clipped": [],
        "phase_recenter_smooth_alpha": [],
        "phase_recenter_gate_reason": [],
        "phase_recenter_pitch_safe": [],
        "phase_recenter_pitch_danger": [],
        "phase_recenter_contact_safe": [],
        "phase_recenter_height_safe": [],
        "phase_recenter_deadband_active": [],
        # Hysteresis recenter fields (F2_strategy)
        "hysteresis_recenter_enabled": [],
        "hysteresis_recenter_state": [],
        "hysteresis_recenter_state_id": [],
        "hysteresis_recenter_outer_enter_m": [],
        "hysteresis_recenter_exit_target_m": [],
        "hysteresis_recenter_signed_error_m": [],
        "hysteresis_recenter_target_error_m": [],
        "hysteresis_recenter_raw_tau": [],
        "hysteresis_recenter_tau": [],
        "hysteresis_recenter_tau_clipped": [],
        "hysteresis_recenter_active": [],
        "hysteresis_recenter_state_entry_count": [],
        "hysteresis_recenter_state_exit_count": [],
        "hysteresis_recenter_safety_override": [],
        "hysteresis_recenter_gate_reason": [],
        "sagittal_axis_x_initial": [],
        "sagittal_axis_y_initial": [],
        "raw_com_vx": [],
        "raw_com_vy": [],
        "projected_sagittal_velocity_m_s": [],
        "actual_sagittal_velocity_passed_to_controller_m_s": [],
        "tau_wheel_total_raw_left": [],
        "tau_wheel_total_raw_right": [],
        "tau_wheel_total_clipped_left": [],
        "tau_wheel_total_clipped_right": [],
        "wheel_torque_margin_left": [],
        "wheel_torque_margin_right": [],
        "wheel_torque_rate_limit_active_left": [],
        "wheel_torque_rate_limit_active_right": [],
        "l_hip_yaw_pos": [],
        "r_hip_yaw_pos": [],
        "l_hip_yaw_ref": [],
        "r_hip_yaw_ref": [],
        "l_hip_yaw_error": [],
        "r_hip_yaw_error": [],
        "l_hip_yaw_vel": [],
        "r_hip_yaw_vel": [],
        "hip_yaw_error_rms": [],
        "l_hip_yaw_tau_shape_raw": [],
        "r_hip_yaw_tau_shape_raw": [],
        "l_hip_yaw_tau_shape_final": [],
        "r_hip_yaw_tau_shape_final": [],
        # HY-FF: Hip-yaw support-error feedforward compensation telemetry
        "hip_yaw_comp_active": [],
        "hip_yaw_comp_height_gate": [],
        "hip_yaw_comp_support_error_m": [],
        "hip_yaw_comp_tau_left": [],
        "hip_yaw_comp_tau_right": [],
        "hip_yaw_comp_tau_left_clipped": [],
        "hip_yaw_comp_tau_right_clipped": [],
        "hip_yaw_comp_sign": [],
        "hip_yaw_comp_k_support": [],
        "hip_yaw_comp_tau_max": [],
        # HY2-DIV: Hip-yaw divergence damping telemetry
        # Note: hip_yaw_div_active is deprecated; use hip_yaw_div_enabled and hip_yaw_div_gate_active
        "hip_yaw_div_enabled": [],
        "hip_yaw_div_gate_active": [],
        "hip_yaw_div_active": [],  # Deprecated alias for backward compatibility
        "hip_yaw_div_height_gate": [],
        "hip_yaw_div_effective_k": [],
        "hip_yaw_div_effective_kd": [],
        "hip_yaw_div_effective_tau_max": [],
        "hip_yaw_div_left": [],
        "hip_yaw_div_right": [],
        "hip_yaw_div_left_clipped": [],
        "hip_yaw_div_right_clipped": [],
        "hip_yaw_div_k_divergence": [],
        "hip_yaw_div_k_divergence_rate": [],
        "hip_yaw_div_tau_max": [],
        "hip_yaw_div_z_low": [],
        "hip_yaw_div_z_high": [],
        # Mode-Based Hip-Yaw Divergence Controller (architecture fix candidate) — opt-in
        "mode_hip_yaw_div_enabled": [],
        "mode_hip_yaw_div_kp": [],
        "mode_hip_yaw_div_kd": [],
        "mode_hip_yaw_div_max_torque": [],
        "mode_hip_yaw_div_soft_limit_rad": [],
        "mode_hip_yaw_div_soft_gain": [],
        "mode_hip_yaw_div_ref_source": [],
        "mode_hip_yaw_div_height_gate": [],
        "mode_hip_yaw_div_tau_left": [],
        "mode_hip_yaw_div_tau_right": [],
        "mode_hip_yaw_div_tau_left_raw": [],
        "mode_hip_yaw_div_tau_right_raw": [],
        "mode_hip_yaw_div_tau_left_raw": [],
        "mode_hip_yaw_div_tau_right_raw": [],
        "mode_hip_yaw_div_tau_left_sat": [],
        "mode_hip_yaw_div_tau_right_sat": [],
        "mode_hip_yaw_div_torque_margin_left": [],
        "mode_hip_yaw_div_torque_margin_right": [],
        "mode_hip_yaw_div_error": [],
        "mode_hip_yaw_div_rate": [],
        "mode_hip_yaw_div_ref": [],
        # Support-aware mode-div gating telemetry (opt-in)
        "mode_hip_yaw_div_support_gate_enabled": [],
        "mode_hip_yaw_div_support_error_m": [],
        "mode_hip_yaw_div_support_error_rate_mps": [],
        "mode_hip_yaw_div_support_error_gate": [],
        "mode_hip_yaw_div_support_rate_gate": [],
        "mode_hip_yaw_div_effective_support_gate": [],
        "mode_hip_yaw_div_combined_gate": [],
        "hip_yaw_mode_ownership_violation": [],
        # HY-FF debug telemetry
        "hy_ff_height_passed_to_shape": [],
        "hy_ff_support_error_passed_to_shape": [],
        "hy_ff_support_error_from_sagittal": [],
        "hy_ff_prev_support_error": [],
        "hy_ff_setup_target_com_z_m": [],
        "hy_ff_setup_achieved_com_z_m": [],
        "hy_ff_root_z_m": [],
        "hy_ff_current_com_z_m": [],
        "hip_yaw_torque_sign_correct_left": [],
        "hip_yaw_torque_sign_correct_right": [],
        "hip_yaw_torque_saturation_flag_left": [],
        "hip_yaw_torque_saturation_flag_right": [],
        "hip_yaw_torque_margin_left": [],
        "hip_yaw_torque_margin_right": [],
        "variant_name": [],
        "height_variant_target_com_z_m": [],
        "height_variant_achieved_com_z_m": [],
        "height_variant_root_z_m": [],
        "height_variant_hip_pitch_ref": [],
        "height_variant_knee_ref": [],
        "shape_posture_reference_source": [],
        "equilibrium_capture_after_variant_applied": [],
        "target_com_z_m": [],
        "current_com_z_m": [],
        "height_error_m": [],
        "root_z_m": [],
        "support_center_ref_x": [],
        "support_center_ref_y": [],
        "support_center_x": [],
        "support_center_y": [],
        "support_position_reference_source": [],
        "support_reference_captured_after_variant": [],
        # Yaw-position coupling diagnostic telemetry
        "root_yaw_z_rad": [],
        "yaw_z_rad": [],
        "yaw_error_from_equilibrium_rad": [],
        "hip_yaw_asymmetry": [],
        "hip_yaw_divergence": [],
        "yaw_induced_position_error_x_m": [],
        "yaw_induced_position_error_y_m": [],
        "yaw_induced_position_error_norm_m": [],
        "yaw_aware_position_compensation_active": [],
        "yaw_aware_sagittal_error_compensated_m": [],
        "yaw_aware_lateral_error_compensated_m": [],
        "effective_kp_hip_yaw": [],
        "effective_kd_hip_yaw": [],
        "hip_yaw_integral_active": [],
        "hip_yaw_integral_clamp": [],
        "hip_yaw_integral_error_left": [],
        "hip_yaw_integral_error_right": [],
        "hip_yaw_bias_tau_left": [],
        "hip_yaw_bias_tau_right": [],
        "hip_yaw_bias_active": [],
        "tau_position_yaw_compensated_raw": [],
        "tau_position_yaw_compensated_clipped": [],
        "boundary_yaw_position_profile": [],
        "boundary_profile_active": [],
        "hip_yaw_abs_max_tracking": [],
        "hip_yaw_abs_max_threshold": [],
        # APCR (Active Pitch Crossing Recovery) telemetry fields
        # These are generated by SagittalVelocityDampedBalanceController.compute() but
        # were not being captured in telemetry. Added 2026-06-08 to fix APCR validation.
        "active_pitch_crossing_enabled": [],
        "active_pitch_crossing_recovery_gate_mode": [],
        "active_pitch_crossing_state": [],
        "active_pitch_crossing_state_id": [],
        "active_pitch_crossing_active": [],
        "active_pitch_crossing_signed_error_m": [],
        "active_pitch_crossing_pitch_x": [],
        "active_pitch_crossing_pitch_rate": [],
        "active_pitch_crossing_raw_tau": [],
        "active_pitch_crossing_tau": [],
        "active_pitch_crossing_tau_clipped": [],
        "active_pitch_crossing_target_direction": [],
        "active_pitch_crossing_outer_enter_m": [],
        "active_pitch_crossing_inner_exit_m": [],
        "active_pitch_crossing_pitch_hard_stop_rad": [],
        "active_pitch_crossing_hard_safety_gate": [],
        "active_pitch_crossing_recovery_gate": [],
        "active_pitch_crossing_gate_reason": [],
        "active_pitch_crossing_state_entry_count": [],
        "active_pitch_crossing_state_exit_count": [],
        "active_pitch_crossing_safety_override": [],
        "active_pitch_crossing_contact_safe": [],
        "active_pitch_crossing_height_safe": [],
        "active_pitch_crossing_roll_safe": [],
        "active_pitch_crossing_pitch_safe": [],
        "active_pitch_crossing_pitch_danger": [],
        "active_pitch_crossing_max_tau": [],
        "active_pitch_crossing_smooth_alpha": [],
        # APCR1i hysteresis recenter telemetry
        "active_pitch_crossing_hysteresis_enabled": [],
        "active_pitch_crossing_hysteresis_state": [],
        "active_pitch_crossing_hysteresis_state_id": [],
        "active_pitch_crossing_hysteresis_entry_e": [],
        "active_pitch_crossing_hysteresis_exit_e": [],
        "active_pitch_crossing_hysteresis_entry_count": [],
        "active_pitch_crossing_hysteresis_exit_count": [],
        "active_pitch_crossing_hysteresis_inner_exit_m": [],
        "active_pitch_crossing_hysteresis_opposite_release_m": [],
        "active_pitch_crossing_hysteresis_emergency_active": [],
        "final_wheel_tau_with_apc": [],
        "final_wheel_tau_without_apc": [],
        # APCR1l pitch suppression telemetry
        "apcr1l_pitch_suppress_active": [],
        "apcr1l_recenter_state": [],
        "apcr1l_tau_pitch_before_suppress": [],
        # APCR1m conditional pitch blend telemetry
        "apcr1m_pitch_blend_active": [],
        "apcr1m_pitch_blend_scale": [],
        "apcr1m_pitch_blend_block_reason": [],
        "apcr1m_tau_pitch_before_blend": [],
        "apcr1m_tau_pitch_after_blend": [],
        "apcr1m_startup_guard_active": [],
        "apcr1m_recenter_active": [],
        "apcr1m_pitch_safe": [],
        "apcr1m_height_safe": [],
        "apcr1m_contact_safe": [],
        "apcr1m_roll_safe": [],
        "apcr1m_pitch_rate_safe": [],
        # APCR1n recenter priority telemetry columns
        "apcr1n_recenter_priority_active": [],
        "apcr1n_startup_guard_active": [],
        "apcr1n_wheel_damping_override_active": [],
        "apcr1n_wheel_damping_scale": [],
        "apcr1n_wheel_damping_before": [],
        "apcr1n_wheel_damping_after": [],
        "apcr1n_wheel_damping_fights_drift": [],
        "apcr1n_position_cap_boost_active": [],
        "apcr1n_position_cap_current": [],
        "apcr1n_tau_position_raw": [],
        "apcr1n_tau_position_after_cap": [],
        "apcr1n_position_saturated": [],
        "apcr1n_safety_gate_pass": [],
        "apcr1n_final_torque_direction_correct": [],
        "apcr1n_final_torque_fights_drift": [],
        "apcr1n_physical_drift_column_used": [],
        # APCR1nD direct support recenter telemetry
        "apcr1nd_direct_recenter_priority_active": [],
        "apcr1nd_direct_recenter_eligible": [],
        "apcr1nd_direct_recenter_block_reason": [],
        "apcr1nd_moving_away": [],
        "apcr1nd_abs_error": [],
        "apcr1nd_error_rate": [],
        # Wheel yaw stabilizer telemetry
        "wheel_yaw_enabled": [],
        "wheel_yaw_error": [],
        "wheel_yaw_rate": [],
        "wheel_yaw_tau_left": [],
        "wheel_yaw_tau_right": [],
        "wheel_yaw_saturated": [],
        "wheel_yaw_profile_activated": [],
        "wheel_yaw_kp": [],
        "wheel_yaw_kd": [],
        "wheel_yaw_max_torque": [],
        "wheel_yaw_height_gate": [],
        "wheel_yaw_tau_diff": [],
        "wheel_yaw_use_numerical_rate": [],
        # Body yaw and hip-yaw ownership telemetry
        "body_yaw_owner": [],
        "hip_yaw_divergence_owner": [],
        # Yaw controller hip-yaw contribution telemetry
        "yaw_controller_tau_hip_yaw_left": [],
        "yaw_controller_tau_hip_yaw_right": [],
        # Hip-yaw mode decomposition telemetry
        "hip_yaw_common_error_rad": [],
        "hip_yaw_common_error_sum_abs_rad": [],
        "hip_yaw_divergence_error_rad": [],
        "hip_yaw_asymmetry_abs_rad": [],
        "hip_yaw_div_common_ratio": [],
    }
    telemetry.update(build_step1_telemetry_template())

    # Initialize balance-core telemetry columns if in balance-core mode
    if is_balance_core_mode(args):
        for key, values in make_balance_core_telemetry_columns().items():
            telemetry.setdefault(key, values)

    # Add profile identity telemetry fields (Phase 1 fix for T6F sign correctness investigation)
    telemetry.setdefault("controller_mode", [])
    telemetry.setdefault("sagittal_controller", [])
    telemetry.setdefault("vd_sagittal_authority_profile", [])
    telemetry.setdefault("height_variant_setup_name", [])

    # Pre-register K1 augmented telemetry fields in the telemetry dict
    # so the CSV writer includes them (values flow from sagittal_diag each step).
    for _k1_field in K1_AUGMENTED_TELEMETRY_FIELDS:
        telemetry.setdefault(_k1_field, [])

    # Simulation parameters
    max_steps = args.steps
    control_dt = 0.01  # 100 Hz
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)
    prev_control_com_pos = None
    tau_prev = jnp.array(mj_data.ctrl)  # Initialize previous torque from current control
    prev_support_error = 0.0  # Previous-step support position error for HY-FF (m)

    # Actuator limits (used by both balance-core and legacy modes for telemetry)
    torque_limit = np.array(mj_model.actuator_ctrlrange[:, 1])
    torque_limit_jax = jnp.array(torque_limit)  # JAX copy for wheel yaw clipping
    max_torque_rate = np.full(10, 400.0)  # 400 Nm/s per joint

    # Balance-core controller instantiation
    balance_core_controllers = None
    if is_balance_core_mode(args):
        support_feedforward_vector = resolve_support_feedforward_vector()
        sagittal_choice = getattr(args, "sagittal_controller", "baseline")
        sagittal_authority_schedule = resolve_sagittal_authority_schedule(args.vd_sagittal_authority_profile)

        # Extract integral parameters from profile for E1/E2/E3 extreme height profiles
        # These profiles have enable_position_integral=True in their schedule
        profile = sagittal_authority_schedule
        if profile.enable_position_integral:
            # Use profile's integral settings instead of CLI defaults
            vd_enable_position_integral = True
            vd_ki_position_integral = profile.ki_position_integral
            vd_integral_max_abs = profile.integral_max_abs
            vd_integral_pitch_error_threshold_rad = profile.integral_pitch_error_threshold_rad
            vd_integral_support_velocity_threshold_m_s = profile.integral_support_velocity_threshold_m_s
            vd_integral_wheel_velocity_threshold_rad_s = profile.integral_wheel_velocity_threshold_rad_s
            vd_integral_min_com_z_m = profile.integral_min_com_z_m
            vd_integral_max_com_z_m = profile.integral_max_com_z_m
            print(f"[EXTREME HEIGHT PROFILE] {profile.profile_name}: integral enabled (ki={vd_ki_position_integral})")
        else:
            # Use CLI values for non-extreme profiles
            vd_enable_position_integral = args.vd_enable_position_integral
            vd_ki_position_integral = args.vd_ki_position_integral
            vd_integral_max_abs = args.vd_integral_max_abs
            vd_integral_pitch_error_threshold_rad = args.vd_integral_pitch_error_threshold_rad
            vd_integral_support_velocity_threshold_m_s = args.vd_integral_support_velocity_threshold_m_s
            vd_integral_wheel_velocity_threshold_rad_s = args.vd_integral_wheel_velocity_threshold_rad_s
            vd_integral_min_com_z_m = args.vd_integral_min_com_z_m
            vd_integral_max_com_z_m = args.vd_integral_max_com_z_m

        # Pitch reference offset (Phase 3 structural fix + causal ablation experiments).
        # Profile-driven offset takes precedence when the schedule defines a nonzero
        # pitch_ref_offset_deg; otherwise fall back to the CLI value (default 0.0).
        # This keeps the fix opt-in: baseline and all legacy profiles use 0.0.
        vd_pitch_ref_offset_deg = args.vd_pitch_ref_offset_deg
        profile_pitch_ref_offset = float(getattr(profile, "pitch_ref_offset_deg", 0.0))
        if abs(profile_pitch_ref_offset) > 1e-9:
            vd_pitch_ref_offset_deg = profile_pitch_ref_offset
            print(f"[PITCH EQUILIBRIUM TRIM] {profile.profile_name}: pitch_ref_offset={vd_pitch_ref_offset_deg:+.2f} deg")

        # Height-scheduled pitch reference offset (Phase 2 structural fix).
        # When the profile enables the schedule, the per-height offset (looked up by
        # piecewise-linear interpolation on the commanded/target CoM height) replaces
        # the static offset. The schedule height is constant for fixed-height
        # validation runs. Opt-in only: every legacy profile keeps the schedule
        # disabled, so this branch is inert for them and the static path above stands.
        pitch_ref_schedule_enabled = bool(
            getattr(profile, "pitch_ref_height_schedule_enabled", False)
        )
        pitch_ref_schedule_height_m = 0.0
        pitch_ref_offset_scheduled_deg = 0.0
        if pitch_ref_schedule_enabled:
            if height_variant_setup is not None:
                pitch_ref_schedule_height_m = float(
                    height_variant_setup.get("target_com_z_m", 0.40)
                )
            else:
                pitch_ref_schedule_height_m = 0.40
            pitch_ref_offset_scheduled_deg = interpolate_pitch_ref_offset(
                pitch_ref_schedule_height_m,
                tuple(profile.pitch_ref_height_schedule_heights_m),
                tuple(profile.pitch_ref_height_schedule_offsets_deg),
                clamp=bool(profile.pitch_ref_height_schedule_clamp),
            )
            vd_pitch_ref_offset_deg = pitch_ref_offset_scheduled_deg
            print(
                f"[HEIGHT SCHEDULED PITCH TRIM] {profile.profile_name}: "
                f"height={pitch_ref_schedule_height_m:.3f} m -> "
                f"pitch_ref_offset={vd_pitch_ref_offset_deg:+.2f} deg"
            )

        # Physics-based equilibrium feedforward (Phase D, opt-in).
        # When the profile enables it, the empirical pitch_ref_offset mechanism
        # (either static or height-scheduled) is REPLACED by a physics-derived
        # pitch_ref value that emerges from the closed-loop equilibrium dynamics.
        # The implementation uses the EQUIVALENT PITCH_REF path (Option B in the
        # task spec): pitch_ref_physics(h) = pitch_eq_no_off_deg(h). When the
        # controller's pitch_ref equals this value, tau_pitch = Kp_pitch *
        # (pitch_x - pitch_ref) ≈ 0 at steady state, which is exactly what the
        # empirical pitch_ref_offset achieves. The difference: the value is
        # derived from MuJoCo closed-loop dynamics (not hand-tuned per-height
        # sweep). No setup-name branching, no hand tuning.
        #
        # IMPORTANT: A direct additive wheel torque feedforward (Option A) was
        # tried first and FAILED — the controller's tau_position does not
        # cancel a constant torque injection, so the robot accelerates
        # forward. The equivalent pitch_ref path preserves the controller's
        # state-dependent torque balance contract.
        physics_ff_enabled = bool(
            getattr(profile, "physics_equilibrium_feedforward_enabled", False)
        )
        physics_ff_height_m = 0.0
        physics_ff_tau_eq_each_wheel_nm = 0.0
        physics_ff_pitch_eq_no_off_deg = 0.0
        physics_ff_function_version = ""
        physics_ff_clamped = False
        if physics_ff_enabled:
            from wheeled_biped.controllers.physics_equilibrium_feedforward import (
                physics_equilibrium_feedforward_params,
            )
            if height_variant_setup is not None:
                physics_ff_height_m = float(
                    height_variant_setup.get("target_com_z_m", 0.40)
                )
            else:
                physics_ff_height_m = 0.40
            _pff = physics_equilibrium_feedforward_params(physics_ff_height_m)
            physics_ff_tau_eq_each_wheel_nm = _pff["physics_ff_tau_eq_each_wheel_nm"]
            physics_ff_pitch_eq_no_off_deg = _pff["physics_ff_pitch_eq_no_off_deg"]
            physics_ff_function_version = _pff["physics_ff_function_version"]
            physics_ff_clamped = bool(_pff["physics_ff_clamped_below"] or _pff["physics_ff_clamped_above"])
            # Equivalent pitch_ref path: set pitch_ref_total = pitch_eq_no_off_deg
            # This makes tau_pitch = 0 at steady state, matching the empirical
            # schedule but with values derived from MuJoCo closed-loop dynamics.
            vd_pitch_ref_offset_deg = physics_ff_pitch_eq_no_off_deg
            pitch_ref_offset_scheduled_deg = physics_ff_pitch_eq_no_off_deg
            print(
                f"[PHYSICS EQ FEEDFORWARD] {profile.profile_name}: "
                f"height={physics_ff_height_m:.3f} m -> "
                f"equivalent pitch_ref={physics_ff_pitch_eq_no_off_deg:+.3f} deg, "
                f"tau_eq_ff_each_wheel={physics_ff_tau_eq_each_wheel_nm:+.3f} Nm "
                f"({_pff['physics_ff_function_profile_name']} v{physics_ff_function_version})"
            )

        # Support-position outer loop (Phase B dynamic centering).
        # Read the outer_loop_* config from the resolved profile. Disabled by
        # default for every legacy profile, so the dynamic term stays 0.0 and the
        # applied pitch_ref equals the scheduled offset above (Phase A unchanged).
        # The loop only activates when the base profile also enables the height
        # schedule (outer_loop_height_schedule_required), binding it to Phase A.
        outer_loop_enabled = bool(getattr(profile, "outer_loop_enabled", False))
        outer_loop_schedule_bound = bool(
            getattr(profile, "outer_loop_height_schedule_required", True)
        )
        outer_loop_active_profile = (
            outer_loop_enabled
            and (pitch_ref_schedule_enabled or not outer_loop_schedule_bound)
        )
        outer_loop_kp = float(getattr(profile, "outer_loop_kp_deg_per_m", 0.0))
        outer_loop_kd = float(getattr(profile, "outer_loop_kd_deg_per_mps", 0.0))
        outer_loop_ki = float(getattr(profile, "outer_loop_ki_deg_per_m_s", 0.0))
        # Phase 4 screening overrides: CLI gains take precedence when provided.
        if getattr(args, "vd_outer_loop_kp_deg_per_m", None) is not None:
            outer_loop_kp = float(args.vd_outer_loop_kp_deg_per_m)
        if getattr(args, "vd_outer_loop_kd_deg_per_mps", None) is not None:
            outer_loop_kd = float(args.vd_outer_loop_kd_deg_per_mps)
        if getattr(args, "vd_outer_loop_ki_deg_per_m_s", None) is not None:
            outer_loop_ki = float(args.vd_outer_loop_ki_deg_per_m_s)
        outer_loop_integral_enabled = bool(
            getattr(profile, "outer_loop_integral_enabled", False)
        )
        if getattr(args, "vd_outer_loop_integral_enabled", False):
            outer_loop_integral_enabled = True
        outer_loop_integral_clamp = float(
            getattr(profile, "outer_loop_integral_clamp_m_s", 0.05)
        )
        outer_loop_theta_ref_max = float(
            getattr(profile, "outer_loop_theta_ref_max_deg", 3.0)
        )
        if getattr(args, "vd_outer_loop_theta_ref_max_deg", None) is not None:
            outer_loop_theta_ref_max = float(args.vd_outer_loop_theta_ref_max_deg)
        outer_loop_theta_rate_limit = float(
            getattr(profile, "outer_loop_theta_ref_rate_limit_deg_per_step", 0.03)
        )
        if getattr(args, "vd_outer_loop_rate_limit_deg_per_step", None) is not None:
            outer_loop_theta_rate_limit = float(args.vd_outer_loop_rate_limit_deg_per_step)
        outer_loop_theta_lowpass = float(
            getattr(profile, "outer_loop_theta_ref_lowpass_alpha", 0.15)
        )
        if getattr(args, "vd_outer_loop_lowpass_alpha", None) is not None:
            outer_loop_theta_lowpass = float(args.vd_outer_loop_lowpass_alpha)
        outer_loop_error_deadband = float(
            getattr(profile, "outer_loop_support_error_deadband_m", 0.015)
        )
        if getattr(args, "vd_outer_loop_deadband_m", None) is not None:
            outer_loop_error_deadband = float(args.vd_outer_loop_deadband_m)
        outer_loop_vel_lowpass = float(
            getattr(profile, "outer_loop_support_velocity_lowpass_alpha", 0.20)
        )
        outer_loop_disable_abs_error = float(
            getattr(profile, "outer_loop_disable_if_abs_error_gt_m", 0.25)
        )
        outer_loop_disable_pitch_deg = float(
            getattr(profile, "outer_loop_disable_if_pitch_gt_deg", 12.0)
        )
        outer_loop_disable_roll_deg = float(
            getattr(profile, "outer_loop_disable_if_roll_gt_deg", 5.0)
        )
        outer_loop_contact_required = bool(
            getattr(profile, "outer_loop_contact_required", True)
        )

        # Calibrated height-dependent outer loop (Phase B calibration, opt-in).
        # When the profile enables it, the scalar Kp/Kd/Ki/theta/deadband/rate/
        # lowpass above are REPLACED by smooth continuous functions of the
        # commanded target CoM height (no setup-name branching). CLI overrides
        # still take precedence (they are applied above and below this block).
        calibrated_outer_loop_enabled = bool(
            getattr(profile, "calibrated_outer_loop_enabled", False)
        )
        calibrated_function_profile_name = ""
        calibrated_height_m_telemetry = 0.0
        support_outer_loop_height_scale = 0.0
        support_outer_loop_kp_effective = outer_loop_kp
        support_outer_loop_kd_effective = outer_loop_kd
        support_outer_loop_pitch_ref_offset_deg = 0.0
        support_outer_loop_cap_active = False
        support_outer_loop_pitch_ref_contrib = 0.0
        if calibrated_outer_loop_enabled and outer_loop_active_profile:
            # Select the correct calibration version based on profile name.
            profile_name = profile.profile_name
            cal_version = str(getattr(profile, "calibrated_outer_loop_function_version", "v1"))
            if profile_name == "calibrated_support_position_outer_loop_pitch_ref_v2" or cal_version == "v2":
                from wheeled_biped.controllers.calibrated_outer_loop_functions_v2 import (
                    calibrated_outer_loop_params as _cal_params_v2,
                )
                _cal_params = _cal_params_v2
            else:
                from wheeled_biped.controllers.calibrated_outer_loop_functions import (
                    calibrated_outer_loop_params as _cal_params_v1,
                )
                _cal_params = _cal_params_v1
            cal_height_m = (
                float(height_variant_setup.get("target_com_z_m", 0.40))
                if height_variant_setup is not None
                else 0.40
            )
            cal = _cal_params(cal_height_m)
            calibrated_function_profile_name = cal["calibrated_function_profile_name"]
            calibrated_height_m_telemetry = cal_height_m
            # CLI overrides win over the calibrated functions (sweep/diagnostic use).
            if getattr(args, "vd_outer_loop_kp_deg_per_m", None) is None:
                outer_loop_kp = cal["calibrated_kp_deg_per_m"]
            if getattr(args, "vd_outer_loop_kd_deg_per_mps", None) is None:
                outer_loop_kd = cal["calibrated_kd_deg_per_mps"]
            if getattr(args, "vd_outer_loop_ki_deg_per_m_s", None) is None:
                outer_loop_ki = cal["calibrated_ki_deg_per_m_s"]
            if getattr(args, "vd_outer_loop_theta_ref_max_deg", None) is None:
                outer_loop_theta_ref_max = cal["calibrated_theta_ref_max_deg"]
            if getattr(args, "vd_outer_loop_deadband_m", None) is None:
                outer_loop_error_deadband = cal["calibrated_deadband_m"]
            if getattr(args, "vd_outer_loop_rate_limit_deg_per_step", None) is None:
                outer_loop_theta_rate_limit = cal["calibrated_rate_limit_deg_per_step"]
            if getattr(args, "vd_outer_loop_lowpass_alpha", None) is None:
                outer_loop_theta_lowpass = cal["calibrated_lowpass_alpha"]
            # Enable integral only if the calibrated Ki is nonzero (anti-windup
            # gating happens in the control loop as for the manual integral).
            if outer_loop_ki != 0.0:
                outer_loop_integral_enabled = True
            print(
                f"[CALIBRATED OUTER LOOP] {profile.profile_name}: "
                f"height={cal_height_m:.3f} m -> Kp={outer_loop_kp:.3f} "
                f"Kd={outer_loop_kd:.3f} Ki={outer_loop_ki:.4f} "
                f"theta_max={outer_loop_theta_ref_max:.2f} db={outer_loop_error_deadband:.4f} "
                f"({calibrated_function_profile_name})"
            )

        if bool(getattr(profile, "low_band_support_outer_loop_enabled", False)) and outer_loop_active_profile:
            from wheeled_biped.controllers.support_outer_loop_low_band import (
                low_band_support_outer_loop_params,
            )

            shape_height_m = (
                float(height_variant_setup.get("target_com_z_m", 0.40))
                if height_variant_setup is not None
                else 0.40
            )
            low_band_sigma_m = float(getattr(profile, "low_band_support_sigma_m", 0.006))
            low_band_kp_peak_deg_per_m = float(getattr(profile, "low_band_support_kp_peak_deg_per_m", 7.0))
            low_band_pitch_ref_offset_peak_deg = float(
                getattr(profile, "low_band_support_pitch_ref_offset_peak_deg", 0.0)
            )
            if getattr(args, "vd_low_band_support_sigma_m", None) is not None:
                low_band_sigma_m = float(args.vd_low_band_support_sigma_m)
            if getattr(args, "vd_low_band_support_kp_peak_deg_per_m", None) is not None:
                low_band_kp_peak_deg_per_m = float(args.vd_low_band_support_kp_peak_deg_per_m)
            if getattr(args, "vd_low_band_support_pitch_ref_offset_peak_deg", None) is not None:
                low_band_pitch_ref_offset_peak_deg = float(args.vd_low_band_support_pitch_ref_offset_peak_deg)
            shaped = low_band_support_outer_loop_params(
                shape_height_m,
                base_kp_deg_per_m=outer_loop_kp,
                base_kd_deg_per_mps=outer_loop_kd,
                base_theta_ref_max_deg=outer_loop_theta_ref_max,
                center_m=float(getattr(profile, "low_band_support_center_m", 0.320)),
                sigma_m=low_band_sigma_m,
                peak_kp_deg_per_m=low_band_kp_peak_deg_per_m,
                peak_theta_ref_max_deg=float(getattr(profile, "low_band_support_theta_ref_max_peak_deg", 0.90)),
                peak_pitch_ref_offset_deg=low_band_pitch_ref_offset_peak_deg,
                blend_with_base=bool(getattr(profile, "low_band_support_blend_with_base", False)),
            )
            support_outer_loop_height_scale = float(shaped["support_outer_loop_height_scale"])
            outer_loop_kp = float(shaped["support_outer_loop_kp_effective"])
            outer_loop_kd = float(shaped["support_outer_loop_kd_effective"])
            outer_loop_theta_ref_max = float(shaped["support_outer_loop_theta_ref_max_effective_deg"])
            support_outer_loop_pitch_ref_offset_deg = float(shaped["support_outer_loop_pitch_ref_offset_deg"])
            support_outer_loop_kp_effective = outer_loop_kp
            support_outer_loop_kd_effective = outer_loop_kd
            print(
                f"[LOW-BAND SUPPORT SHAPING] {profile.profile_name}: "
                f"height={shape_height_m:.3f} m scale={support_outer_loop_height_scale:.3f} "
                f"Kp={outer_loop_kp:.3f} Kd={outer_loop_kd:.3f} "
                f"theta_max={outer_loop_theta_ref_max:.2f} "
                f"pitch_ref_offset={support_outer_loop_pitch_ref_offset_deg:+.2f} "
                f"blend={bool(getattr(profile, 'low_band_support_blend_with_base', False))}"
            )

        outer_loop_sign_selected = "none"
        if outer_loop_active_profile:
            outer_loop_sign_selected = "positive" if outer_loop_kp >= 0.0 else "negative"
        # Per-run outer-loop state (reset each simulation).
        outer_loop_prev_support_error_m = None  # set on first control step
        outer_loop_support_error_rate_smoothed = 0.0
        outer_loop_pitch_ref_smoothed_deg = 0.0
        outer_loop_integral_accum_m_s = 0.0
        if outer_loop_active_profile:
            print(
                f"[OUTER LOOP] {profile.profile_name}: enabled "
                f"Kp={outer_loop_kp:+.3f} deg/m Kd={outer_loop_kd:+.3f} deg/(m/s) "
                f"Ki={outer_loop_ki:+.3f} (integral={'on' if outer_loop_integral_enabled else 'off'}) "
                f"sign={outer_loop_sign_selected} theta_max={outer_loop_theta_ref_max:.1f} deg"
            )

        balance_core_controllers = build_balance_core_controllers(
            control_dt=control_dt,
            support_feedforward_vector=support_feedforward_vector,
            torque_limit=torque_limit,
            max_torque_rate=max_torque_rate,
            sagittal_controller_choice=sagittal_choice,
            vd_k_position=args.vd_k_position,
            vd_k_velocity=args.vd_k_velocity,
            vd_k_support_velocity=args.vd_k_support_velocity,
            vd_max_position_tau=args.vd_max_position_tau,
            vd_k_pitch=args.vd_k_pitch,
            vd_pitch_ref_offset_deg=vd_pitch_ref_offset_deg,
            vd_enable_capture_gate=args.vd_enable_capture_gate,
            vd_capture_gate_pitch_threshold=args.vd_capture_gate_pitch_threshold,
            vd_capture_gate_conflict_factor=args.vd_capture_gate_conflict_factor,
            vd_capture_gate_smooth_steps=args.vd_capture_gate_smooth_steps,
            vd_capture_gate_use_cp=args.vd_capture_gate_use_cp,
            vd_enable_torque_budget_aware_position=args.vd_enable_torque_budget_aware_position,
            vd_position_tau_budget_cap=args.vd_position_tau_budget_cap,
            vd_enable_position_integral=vd_enable_position_integral,
            vd_ki_position_integral=vd_ki_position_integral,
            vd_integral_max_abs=vd_integral_max_abs,
            vd_integral_pitch_error_threshold_rad=vd_integral_pitch_error_threshold_rad,
            vd_integral_roll_error_threshold_rad=args.vd_integral_roll_error_threshold_rad,
            vd_integral_pitch_rate_threshold_rad_s=args.vd_integral_pitch_rate_threshold_rad_s,
            vd_integral_support_velocity_threshold_m_s=vd_integral_support_velocity_threshold_m_s,
            vd_integral_wheel_velocity_threshold_rad_s=vd_integral_wheel_velocity_threshold_rad_s,
            vd_integral_min_com_z_m=vd_integral_min_com_z_m,
            vd_integral_max_com_z_m=vd_integral_max_com_z_m,
            sagittal_authority_schedule=sagittal_authority_schedule,
            shape_kp_hip_yaw=args.shape_kp_hip_yaw,
            shape_kd_hip_yaw=args.shape_kd_hip_yaw,
            enable_hip_yaw_support_feedforward=args.enable_hip_yaw_support_feedforward,
            hip_yaw_support_k=args.hip_yaw_support_k,
            hip_yaw_support_tau_max=args.hip_yaw_support_tau_max,
            hip_yaw_support_sign=args.hip_yaw_support_sign,
            enable_hip_yaw_divergence_damping=args.enable_hip_yaw_divergence_damping,
            hip_yaw_divergence_k=args.hip_yaw_divergence_k,
            hip_yaw_divergence_kd=args.hip_yaw_divergence_kd,
            hip_yaw_divergence_tau_max=args.hip_yaw_divergence_tau_max,
            hip_yaw_divergence_z_low=args.hip_yaw_divergence_z_low,
            hip_yaw_divergence_z_high=args.hip_yaw_divergence_z_high,
            enable_wheel_yaw_stabilizer=args.enable_wheel_yaw_stabilizer,
            wheel_yaw_kp=args.wheel_yaw_kp,
            wheel_yaw_kd=args.wheel_yaw_kd,
            wheel_yaw_max_torque=args.wheel_yaw_max_torque,
            wheel_yaw_lowpass_alpha=args.wheel_yaw_lowpass_alpha,
            wheel_yaw_height_gate_low=args.wheel_yaw_height_gate_low,
            wheel_yaw_height_gate_high=args.wheel_yaw_height_gate_high,
            yaw_controller_kp=args.yaw_controller_kp,
            yaw_controller_kd=args.yaw_controller_kd,
            yaw_controller_max_torque=args.yaw_controller_max_torque,
        )
        sagittal_name = balance_core_controllers["sagittal_controller_name"]
        print(f"[BALANCE-CORE] Functional four-source controller stack enabled")
        print(f"[BALANCE-CORE] Sagittal controller: {sagittal_name}")

    # Boundary yaw-position coupling fix state
    boundary_fix = resolve_boundary_yaw_position_fix_state(args)
    boundary_fix.reset()
    if boundary_fix.profile != "baseline":
        print(f"[BOUNDARY FIX] Profile: {boundary_fix.profile}")
        print(f"[BOUNDARY FIX] Boundary kp_hip_yaw={boundary_fix.boundary_kp}, kd_hip_yaw={boundary_fix.boundary_kd}")
        print(f"[BOUNDARY FIX] Integral gain={boundary_fix.integral_gain}, max={boundary_fix.integral_max}")

    # For finite-difference rate computation
    prev_log_pitch_x = None
    prev_log_roll_y = None

    # Pitch rate consistency estimator for velocity-damped controller
    pitch_rate_estimator = PitchRateConsistencyEstimator(
        dt=control_dt,
        min_rate_for_sign_check=args.vd_pitch_rate_min_sign_check,
        filter_alpha=args.vd_pitch_rate_filter_alpha,
    )
    if args.sagittal_controller == "velocity-damped":
        correction_status = "ENABLED" if args.vd_enable_pitch_rate_correction else "DISABLED (default)"
        print(f"[PITCH RATE ESTIMATOR] Initialized: filter_alpha={args.vd_pitch_rate_filter_alpha}, min_sign_check={args.vd_pitch_rate_min_sign_check} rad/s")
        print(f"[PITCH RATE ESTIMATOR] Correction in active control: {correction_status}")

    # Wheel velocity memory for balance-core mode
    prev_wheel_vel_left = 0.0
    prev_wheel_vel_right = 0.0

    # --- Long-run logging state ---
    telemetry_decimation = max(1, int(getattr(args, "telemetry_decimation", 1)))
    failure_window_steps = max(0, int(getattr(args, "failure_window_steps", 0)))
    write_run_summary_sidecar = getattr(args, "write_run_summary_sidecar", False)

    failure_window_buffer: deque = deque(maxlen=failure_window_steps) if failure_window_steps > 0 else deque()
    last_full_rate_row = None
    last_full_rate_step = -1

    def make_rms_accumulator() -> dict:
        return {"count": 0, "sum_sq": 0.0}

    def update_rms_accumulator(accumulator: dict, value: float) -> None:
        accumulator["count"] += 1
        accumulator["sum_sq"] += float(value) * float(value)

    def finalize_rms_accumulator(accumulator: dict) -> float:
        if accumulator["count"] <= 0:
            return 0.0
        return float(np.sqrt(accumulator["sum_sq"] / accumulator["count"]))

    full_rate_summary = {
        "actual_steps": 0,
        "survived_steps": 0,
        "pitch_x_min": None,
        "pitch_x_max": None,
        "pitch_x_rms": make_rms_accumulator(),
        "roll_y_min": None,
        "roll_y_max": None,
        "roll_y_rms": make_rms_accumulator(),
        "com_z_min": None,
        "com_z_max": None,
        "com_z_initial": None,
        "com_z_final": None,
        "wheel_vel_mean_min": None,
        "wheel_vel_mean_max": None,
        "wheel_vel_mean_rms": make_rms_accumulator(),
        "wheel_vel_mean_initial": None,
        "wheel_vel_mean_final": None,
        "ownership_violation_count_max": 0,
        "hidden_torque_norm_max": 0.0,
        "tau_wbc_norm_max": 0.0,
        "torque_saturation_rate_max": 0.0,
        "torque_saturation_fraction_mean": 0.0,
        "torque_rate_saturation_rate_max": 0.0,
        "torque_rate_saturation_fraction_mean": 0.0,
        "contact_state_counts": {},
        "metric_integrity": {
            "source": "full_rate_online",
            "limitations": [],
        },
    }

    def update_min_max(summary: dict, min_key: str, max_key: str, value: float) -> None:
        current_min = summary[min_key]
        current_max = summary[max_key]
        summary[min_key] = value if current_min is None else min(current_min, value)
        summary[max_key] = value if current_max is None else max(current_max, value)

    def update_full_rate_summary(
        *,
        pitch_x_value: float,
        roll_y_value: float,
        com_z_value: float,
        wheel_vel_mean_value: float,
        ownership_violation_count_value: int,
        hidden_torque_norm_value: float,
        tau_wbc_norm_value: float,
        torque_saturation_rate_value: float,
        torque_rate_saturation_rate_value: float,
        contact_state_value: str,
    ) -> None:
        full_rate_summary["actual_steps"] += 1
        full_rate_summary["survived_steps"] = full_rate_summary["actual_steps"]

        update_min_max(full_rate_summary, "pitch_x_min", "pitch_x_max", pitch_x_value)
        update_rms_accumulator(full_rate_summary["pitch_x_rms"], pitch_x_value)

        update_min_max(full_rate_summary, "roll_y_min", "roll_y_max", roll_y_value)
        update_rms_accumulator(full_rate_summary["roll_y_rms"], roll_y_value)

        update_min_max(full_rate_summary, "com_z_min", "com_z_max", com_z_value)
        if full_rate_summary["com_z_initial"] is None:
            full_rate_summary["com_z_initial"] = com_z_value
        full_rate_summary["com_z_final"] = com_z_value

        update_min_max(full_rate_summary, "wheel_vel_mean_min", "wheel_vel_mean_max", wheel_vel_mean_value)
        update_rms_accumulator(full_rate_summary["wheel_vel_mean_rms"], wheel_vel_mean_value)
        if full_rate_summary["wheel_vel_mean_initial"] is None:
            full_rate_summary["wheel_vel_mean_initial"] = wheel_vel_mean_value
        full_rate_summary["wheel_vel_mean_final"] = wheel_vel_mean_value

        full_rate_summary["ownership_violation_count_max"] = max(
            int(full_rate_summary["ownership_violation_count_max"]),
            int(ownership_violation_count_value),
        )
        full_rate_summary["hidden_torque_norm_max"] = max(
            float(full_rate_summary["hidden_torque_norm_max"]),
            float(hidden_torque_norm_value),
        )
        full_rate_summary["tau_wbc_norm_max"] = max(
            float(full_rate_summary["tau_wbc_norm_max"]),
            float(tau_wbc_norm_value),
        )
        full_rate_summary["torque_saturation_rate_max"] = max(
            float(full_rate_summary["torque_saturation_rate_max"]),
            float(torque_saturation_rate_value),
        )
        full_rate_summary["torque_rate_saturation_rate_max"] = max(
            float(full_rate_summary["torque_rate_saturation_rate_max"]),
            float(torque_rate_saturation_rate_value),
        )
        full_rate_summary["torque_saturation_fraction_mean"] += float(torque_saturation_rate_value)
        full_rate_summary["torque_rate_saturation_fraction_mean"] += float(torque_rate_saturation_rate_value)
        full_rate_summary["contact_state_counts"][contact_state_value] = (
            int(full_rate_summary["contact_state_counts"].get(contact_state_value, 0)) + 1
        )

    def finalize_full_rate_summary() -> dict:
        total_steps = max(int(full_rate_summary["actual_steps"]), 1)
        com_z_initial = full_rate_summary["com_z_initial"]
        com_z_final = full_rate_summary["com_z_final"]
        wheel_vel_mean_initial = full_rate_summary["wheel_vel_mean_initial"]
        wheel_vel_mean_final = full_rate_summary["wheel_vel_mean_final"]
        contact_state_counts = dict(sorted(full_rate_summary["contact_state_counts"].items()))
        most_common_contact_state = None
        if contact_state_counts:
            most_common_contact_state = max(contact_state_counts.items(), key=lambda item: item[1])[0]

        return {
            "actual_steps": int(full_rate_summary["actual_steps"]),
            "survived_steps": int(full_rate_summary["survived_steps"]),
            "pitch_x": {
                "min": 0.0 if full_rate_summary["pitch_x_min"] is None else float(full_rate_summary["pitch_x_min"]),
                "max": 0.0 if full_rate_summary["pitch_x_max"] is None else float(full_rate_summary["pitch_x_max"]),
                "rms": finalize_rms_accumulator(full_rate_summary["pitch_x_rms"]),
            },
            "roll_y": {
                "min": 0.0 if full_rate_summary["roll_y_min"] is None else float(full_rate_summary["roll_y_min"]),
                "max": 0.0 if full_rate_summary["roll_y_max"] is None else float(full_rate_summary["roll_y_max"]),
                "rms": finalize_rms_accumulator(full_rate_summary["roll_y_rms"]),
            },
            "com_z": {
                "min": 0.0 if full_rate_summary["com_z_min"] is None else float(full_rate_summary["com_z_min"]),
                "max": 0.0 if full_rate_summary["com_z_max"] is None else float(full_rate_summary["com_z_max"]),
                "drift": 0.0 if com_z_initial is None or com_z_final is None else float(com_z_final - com_z_initial),
            },
            "wheel_vel_mean": {
                "min": 0.0 if full_rate_summary["wheel_vel_mean_min"] is None else float(full_rate_summary["wheel_vel_mean_min"]),
                "max": 0.0 if full_rate_summary["wheel_vel_mean_max"] is None else float(full_rate_summary["wheel_vel_mean_max"]),
                "rms": finalize_rms_accumulator(full_rate_summary["wheel_vel_mean_rms"]),
            },
            "wheel_velocity_trend": 0.0 if wheel_vel_mean_initial is None or wheel_vel_mean_final is None else float(wheel_vel_mean_final - wheel_vel_mean_initial),
            "ownership_violation_count_max": int(full_rate_summary["ownership_violation_count_max"]),
            "hidden_torque_norm_max": float(full_rate_summary["hidden_torque_norm_max"]),
            "tau_wbc_norm_max": float(full_rate_summary["tau_wbc_norm_max"]),
            "torque_saturation": {
                "fraction_max": float(full_rate_summary["torque_saturation_rate_max"]),
                "fraction_mean": float(full_rate_summary["torque_saturation_fraction_mean"] / total_steps),
            },
            "torque_rate_saturation": {
                "fraction_max": float(full_rate_summary["torque_rate_saturation_rate_max"]),
                "fraction_mean": float(full_rate_summary["torque_rate_saturation_fraction_mean"] / total_steps),
            },
            "contact_state_summary": {
                "counts": contact_state_counts,
                "most_common_state": most_common_contact_state,
            },
            "metric_integrity": dict(full_rate_summary["metric_integrity"]),
        }

    def should_keep_main_telemetry_row(source_step_index: int, is_terminating: bool) -> bool:
        if telemetry_decimation <= 1:
            return True
        if source_step_index == 0:
            return True
        if is_terminating:
            return True
        return (source_step_index % telemetry_decimation) == 0

    def snapshot_last_telemetry_row() -> dict:
        result = {}
        for key, values in telemetry.items():
            if values:
                result[key] = values[-1]
            else:
                result[key] = None
        return result

    def drop_last_telemetry_row() -> None:
        for values in telemetry.values():
            if values:
                values.pop()

    def append_telemetry_row(row: dict) -> None:
        for key in telemetry.keys():
            telemetry[key].append(row[key])

    print(
        f"\nRunning simulation for {max_steps} steps ({max_steps * control_dt:.1f} seconds)"
    )
    print("=" * 80)

    start_time = time.time()

    # Stage 5: JAX backend initialization (once before loop)
    _backend = getattr(args, "controller_backend", "python")
    _jax_enabled = (_backend == "jax")
    _jax_step_fn = None
    _jax_params = None
    _jax_state = None
    _jax_compile_time_s = 0.0
    if _jax_enabled:
        if not is_balance_core_mode(args):
            print("[JAX BACKEND] ERROR: --controller-backend jax requires --controller-mode balance-core")
            sys.exit(1)
        import jax as _jax
        _jax.config.update("jax_enable_x64", True)
        from wheeled_biped.controllers.k2_jax_controller import (
            pack_state_k2, pack_params_stage2, k2_jax_controller_step,
            K2_JAX_INPUT_SIZE, pack_input_k2,
        )
        import jax.numpy as _jnp

        _t_compile_start = time.perf_counter()
        _jax_params = pack_params_stage2(
            fs_hz=100.0, fc_hz=2.5, Q=2.0,
            torque_limit=_jnp.ones(10) * float(torque_limit[0]) if hasattr(torque_limit, '__len__') else 10.0,
            max_torque_rate=_jnp.ones(10) * 400.0,
            control_dt=float(control_dt),
        )
        _jax_state = pack_state_k2()
        _jax_step_fn = _jax.jit(k2_jax_controller_step)
        # Warmup compile
        _dummy_in = _jnp.zeros(K2_JAX_INPUT_SIZE, dtype=_jnp.float64)
        _ = _jax_step_fn(_jax_state, _dummy_in, _jax_params)
        _ = _jax_step_fn(_jax_state, _dummy_in, _jax_params)
        _jax_compile_time_s = time.perf_counter() - _t_compile_start
        print(f"[JAX BACKEND] Enabled — JIT compile time: {_jax_compile_time_s:.2f}s")
        print(f"[JAX BACKEND] State size: {_jax_state.shape[0]}, Params size: {_jax_params.shape[0]}, Input size: {K2_JAX_INPUT_SIZE}")

    # Stage 1: Per-component controller profiling accumulators (--profile-controller)
    _profile_enabled = getattr(args, "profile_controller", False)
    _profile_timing = {
        "centroidal_control_ms": 0.0,
        "capture_control_ms": 0.0,
        "balance_core_block_ms": 0.0,
        "centroidal_log_ms": 0.0,
        "capture_log_ms": 0.0,
        "telemetry_ms": 0.0,
        "total_step_ms": 0.0,
        "step_count": 0,
    }

    terminated = False
    termination_reason = None
    step = 0
    height_cmd = 0.40  # Match equilibrium CoM height from compute_equilibrium_keyframe.py
    initial_yaw_z = float(centroidal_state_eq.body_yaw_z)

    # Dynamic height trajectory tracking state
    dynamic_height_active = dynamic_height_traj is not None
    dynamic_height_target_m = float(height_cmd)
    dynamic_height_actual_m = float(centroidal_state_eq.com_pos[2]) if hasattr(centroidal_state_eq, 'com_pos') else 0.40
    dynamic_height_notch_gate = 0.0

    # Dynamic termination height floor for low-height variants
    if height_variant_setup is not None:
        achieved_com_z = float(height_variant_setup.get("achieved_com_z_m", 0.40))
        termination_height_floor_m = achieved_com_z - 0.05
        print(f"[HEIGHT VARIANT] Termination height floor: {termination_height_floor_m:.3f} m "
              f"(achieved_com_z - 0.05)")
    else:
        termination_height_floor_m = 0.35

    # Step D: push disturbance state (deterministic per-run pseudo-random schedule).
    # The C++ mj_step path does not accept xfrc_applied via this entrypoint; instead
    # we emulate the push via the existing --initial-root-z-perturbation machinery
    # applied at scheduled steps by injecting velocity directly into mj_data.qvel
    # for push_duration_steps before each mj_step.  This is the simplest faithful
    # proxy: an impulse on the torso CoM in the sagittal direction.
    # Direction sign convention: +1 = forward push (sagittal +y), -1 = backward.
    push_enabled = bool(getattr(args, "push_enabled", False))
    push_mag_n = float(getattr(args, "push_magnitude_n", 15.0))
    push_interval = int(getattr(args, "push_interval_steps", 200))
    push_duration = int(getattr(args, "push_duration_steps", 5))
    push_count_override = getattr(args, "push_count", None)
    push_start_step_override = getattr(args, "push_start_step", None)
    sagittal_push_only = bool(getattr(args, "sagittal_push_only", False))
    push_sequence_file = getattr(args, "push_sequence_file", None)
    push_rng = random.Random(20260617)
    push_schedule = []  # list of (start_step, end_step, fx_N, fy_N)
    push_active_count = 0
    push_applied_count = 0
    # Per-step push state for telemetry
    push_active_now = False
    push_fx_now = 0.0
    push_fy_now = 0.0

    # Load push sequence from JSON file (deterministic excitation for identification)
    if push_sequence_file and not push_enabled:
        import json as _json_push
        with open(push_sequence_file, "r") as _f:
            _seq_data = _json_push.load(_f)
        # Accept either a flat list of entries or a dict with a "sequence" key
        if isinstance(_seq_data, list):
            _seq_entries = _seq_data
        elif isinstance(_seq_data, dict):
            _seq_entries = _seq_data.get("sequence", [])
        else:
            _seq_entries = []
        for _entry in _seq_entries:
            _step = int(_entry[0])
            _fx = float(_entry[1])
            _fy = float(_entry[2])
            _dur = int(_entry[3])
            if 0 <= _step < max_steps:
                push_schedule.append((_step, _step + _dur, _fx, _fy))
        if push_schedule:
            push_enabled = True
            print(
                f"[PUSH] sequence-file: n_pushes={len(push_schedule)} "
                f"from {push_sequence_file}"
            )
        else:
            print(f"[PUSH] sequence-file {push_sequence_file} produced empty schedule; push disabled")

    if push_enabled and not push_schedule:
        n_pushes = (max(1, max_steps // push_interval)
                    if push_count_override is None else push_count_override)
        for i in range(n_pushes):
            if push_start_step_override is not None:
                start = push_start_step_override + i * push_interval
            else:
                start = 50 + i * push_interval + push_rng.randint(-15, 15)
            if start < 0 or start >= max_steps:
                continue
            if sagittal_push_only:
                # Forward sagittal push (+y direction)
                angle = math.pi / 2
            else:
                angle = push_rng.uniform(0, 2 * math.pi)
            # Convert Newton-impulse to a velocity change on torso CoM.
            # Approx mass ~ robot_mass; dq = J * F * dt where dt is one control step.
            push_force_x = push_mag_n * math.cos(angle)
            push_force_y = push_mag_n * math.sin(angle)
            push_dur = push_duration
            push_schedule.append((start, start + push_dur, push_force_x, push_force_y))
        print(
            f"[PUSH] enabled: n_pushes={len(push_schedule)} magnitude={push_mag_n:.1f} N "
            f"interval={push_interval} duration={push_duration}"
        )

    def _apply_pending_push():
        """If current step falls inside any scheduled push window, add a velocity impulse."""
        nonlocal push_active_count, push_active_now, push_fx_now, push_fy_now
        if not push_enabled:
            push_active_now = False
            push_fx_now = 0.0
            push_fy_now = 0.0
            return
        # Determine if any push is active at this step.
        active = [(fx, fy) for s, e, fx, fy in push_schedule if s <= step < e]
        if not active:
            mj_data.xfrc_applied[:] = 0
            push_active_now = False
            push_fx_now = 0.0
            push_fy_now = 0.0
            return
        # Sum contributions (rare to overlap; sum conservatively).
        fx_total = sum(fx for fx, _ in active)
        fy_total = sum(fy for _, fy in active)
        # xfrc_applied shape: (nbody, 6) — [fx, fy, fz, tx, ty, tz]; body 1 is torso.
        mj_data.xfrc_applied[1, 0] = fx_total
        mj_data.xfrc_applied[1, 1] = fy_total
        push_active_count += 1
        push_active_now = True
        push_fx_now = fx_total
        push_fy_now = fy_total

    def simulation_step():
        nonlocal prev_control_com_pos, terminated, termination_reason, step, height_cmd, tau_prev, prev_log_pitch_x, prev_log_roll_y, prev_wheel_vel_left, prev_wheel_vel_right, torque_limit, max_torque_rate, last_full_rate_row, last_full_rate_step, full_rate_summary, prev_support_error, outer_loop_prev_support_error_m, outer_loop_support_error_rate_smoothed, outer_loop_pitch_ref_smoothed_deg, outer_loop_integral_accum_m_s
        nonlocal dynamic_height_target_m, dynamic_height_actual_m, dynamic_height_notch_gate
        nonlocal _profile_enabled, _profile_timing
        nonlocal _jax_enabled, _jax_step_fn, _jax_params, _jax_state

        if terminated or step >= max_steps:
            return False

        # ---- Dynamic height update (if trajectory active) ---- #
        if dynamic_height_active:
            dynamic_height_target_m = dynamic_height_traj["interp_fn"](step)
            height_cmd = dynamic_height_target_m
            # Update setup dict in-place so downstream reads of height_variant_setup["target_com_z_m"]
            # automatically get the current dynamic target.
            if height_variant_setup is not None:
                height_variant_setup["target_com_z_m"] = dynamic_height_target_m
            # Compute notch gate for telemetry (replicates smoothstep_gate for 0.42-0.48 m)
            if dynamic_height_target_m <= 0.42:
                dynamic_height_notch_gate = 0.0
            elif dynamic_height_target_m >= 0.48:
                dynamic_height_notch_gate = 1.0
            else:
                u = (dynamic_height_target_m - 0.42) / (0.48 - 0.42)
                dynamic_height_notch_gate = u * u * (3.0 - 2.0 * u)

        # Convert MuJoCo data to JAX arrays for controller
        qpos_jax = jnp.array(mj_data.qpos)
        qvel_jax = jnp.array(mj_data.qvel)

        # Compute gravity in body frame from base quaternion
        # qpos[3:7] is base quaternion [w, x, y, z]
        quat = np.array(mj_data.qpos[3:7])
        # Rotate world gravity [0, 0, -9.81] into body frame
        # Using quaternion rotation: v' = q * v * q^-1
        # For efficiency, use rotation matrix from MuJoCo
        base_body_id = 1  # torso is body 1
        R = np.array(mj_data.xmat[base_body_id]).reshape(
            3, 3
        )  # Rotation matrix (world to body)
        gravity_world = np.array([0.0, 0.0, -9.81])
        gravity_body = R.T @ gravity_world  # R.T transforms world to body frame

        # Stage 1: per-step profiling
        _t_step_start = time.perf_counter() if _profile_enabled else 0.0

        # Phase 1: Control-time state estimation.
        # Use previous CONTROL sample CoM for velocity finite-difference.
        prev_control_before_estimate = prev_control_com_pos
        _t0 = time.perf_counter() if _profile_enabled else 0.0
        centroidal_state_control, control_com_pos = centroidal_estimator.estimate(
            jnp.zeros(42), mj_data, prev_control_com_pos
        )
        prev_control_com_pos = control_com_pos
        _t1 = time.perf_counter() if _profile_enabled else 0.0
        centroidal_state_control = capture_estimator.update(centroidal_state_control)
        _t2 = time.perf_counter() if _profile_enabled else 0.0
        if _profile_enabled:
            _profile_timing["centroidal_control_ms"] += (_t1 - _t0) * 1000.0
            _profile_timing["capture_control_ms"] += (_t2 - _t1) * 1000.0

        # Update dynamic height actual from centroidal state
        dynamic_height_actual_m = float(centroidal_state_control.com_pos[2])

        # Construct observation with ACTUAL gravity from IMU
        obs = jnp.zeros(42)
        obs = obs.at[0:3].set(jnp.array(gravity_body))  # Real gravity in body frame
        obs = obs.at[6:16].set(qpos_jax[7:17])
        obs = obs.at[16:26].set(qvel_jax[6:16])
        obs = obs.at[36].set(height_cmd)  # Height command (adaptive, matches keyframe CoM)
        obs = obs.at[37].set(centroidal_state_control.com_pos[2])

        joint_pos = qpos_jax[7:17]
        joint_vel = qvel_jax[6:16]

        # Phase 2-4: Compute controller torques
        # WBC uses unified QP force distribution (not JIT-compiled)
        # Command the ACTUAL CoM height from keyframe configuration
        # With base_z=0.55, hip_pitch=0.95, knee=1.70 → CoM is at ~0.42m with wheels on ground

        # ADAPTIVE HEIGHT ADJUSTMENT: Maintain stability margin by raising height when unstable
        # Extract orientation from gravity vector
        gravity_body = obs[0:3]
        pitch_x_rad, roll_y_rad = compute_orientation_from_gravity(gravity_body)
        control_mode = compute_step6_control_mode(roll_y_rad, pitch_x_rad)

        # Check contact state
        left_contact = centroidal_state_control.left_wheel_contact
        right_contact = centroidal_state_control.right_wheel_contact
        active_wheels = int(left_contact) + int(right_contact)

        # Keep height_cmd constant at 0.40m (no adaptive adjustment)
        # Adaptive height adjustment was causing instability by reducing natural frequency
        # when the robot was already unstable

        tau_wbc, qp_diagnostics = wbc_controller.compute_wbc_torque_with_diagnostics(
            mj_data,
            obs,
            centroidal_state_control,
            height_cmd,
            hip_roll_authority_scale=compute_step6_hip_roll_authority_scale(control_mode),
        )

        # Apply StaticBalanceController wrapper if enabled
        if static_balance_wrapper is not None:
            # Build current_state dict with required keys
            current_state = {
                'com_z': float(centroidal_state_control.com_pos[2]),
                'pitch_x': float(pitch_x_rad),
                'roll_y': float(roll_y_rad),
                'joint_pos': np.array(joint_pos),
                'com_vel': np.array(centroidal_state_control.com_vel),
                'angular_vel': np.array([
                    centroidal_state_control.roll_rate,
                    centroidal_state_control.pitch_rate,
                    centroidal_state_control.yaw_rate,
                ]),
            }

            # Apply wrapper to remove equilibrium bias
            tau_wbc_wrapped, wrapper_telemetry = static_balance_wrapper.wrap(
                np.array(tau_wbc),
                current_state,
            )

            # Log wrapper telemetry for first 20 steps
            if step < 20:
                log_wrapper_telemetry(step, wrapper_telemetry)

            # Use wrapped torque for rest of pipeline
            tau_wbc = jnp.array(tau_wbc_wrapped)

        # Diagnostic: log WBC output on first step
        if step == 0 and not args.visual:
            print(f"\n[WBC DIAGNOSTIC - Step 0]")

            # Show computed orientation from gravity vector using unified computation
            gravity_body = obs[0:3]
            pitch_x_computed, roll_y_computed = compute_orientation_from_gravity(gravity_body)
            print(f"Computed orientation from gravity:")
            print(f"  Roll(Y): {roll_y_computed*57.3:.2f} deg")
            print(f"  Pitch(X): {pitch_x_computed*57.3:.2f} deg")
            print(f"  Gravity vector: [{obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}]")

            print(
                f"\nDesired wrench: Fx={qp_diagnostics['desired_wrench_Fx']:.2f}, "
                f"Fy={qp_diagnostics['desired_wrench_Fy']:.2f}, "
                f"Fz={qp_diagnostics['desired_wrench_Fz']:.2f}, "
                f"Mx={qp_diagnostics['desired_wrench_Mx']:.2f}, "
                f"My={qp_diagnostics['desired_wrench_My']:.2f}, "
                f"Mz={qp_diagnostics['desired_wrench_Mz']:.2f}"
            )
            print(f"QP solution:")
            print(
                f"  f_left:  [{qp_diagnostics['f_left'][0]:.2f}, {qp_diagnostics['f_left'][1]:.2f}, {qp_diagnostics['f_left'][2]:.2f}] N"
            )
            print(
                f"  f_right: [{qp_diagnostics['f_right'][0]:.2f}, {qp_diagnostics['f_right'][1]:.2f}, {qp_diagnostics['f_right'][2]:.2f}] N"
            )
            print(
                f"  tau_hip_roll: [{qp_diagnostics['tau_hip_roll'][0]:.2f}, {qp_diagnostics['tau_hip_roll'][1]:.2f}] Nm"
            )
            print(f"WBC torques: {tau_wbc}")
            print(f"Max WBC torque: {float(jnp.max(jnp.abs(tau_wbc))):.2f} Nm")
            print(f"QP solve time: {qp_diagnostics['solve_time_ms']:.2f} ms")
            print(f"Wrench error: {qp_diagnostics['wrench_error_norm']:.6f} N/Nm")

            # Check actual contact forces in simulation
            print(f"\nActual contact forces from MuJoCo:")
            total_contact_force_z = 0.0
            for i in range(mj_data.ncon):
                contact = mj_data.contact[i]
                geom1_name = mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                )
                geom2_name = mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                )
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(mj_model, mj_data, i, contact_force)
                contact_frame = np.array(contact.frame).reshape(3, 3)
                force_world = contact_frame.T @ contact_force[:3]
                total_contact_force_z += force_world[2]
                print(
                    f"  Contact {i}: {geom1_name} - {geom2_name}, world_fz: {force_world[2]:.2f} N"
                )
            weight_n = robot_mass * gravity
            print(
                f"  Raw mj_forward contact force z: {total_contact_force_z:.2f} N (weight: {weight_n:.2f} N)"
            )
            print(f"  Contact force valid for feedback: {qp_diagnostics['contact_force_valid']}")
            print(f"  Desired Fz: {qp_diagnostics['desired_wrench_Fz']:.2f} N")

            # Force feedback diagnostics
            print(f"\nForce feedback control:")
            print(f"  Actual Fz: {qp_diagnostics['actual_fz_total']:.2f} N")
            print(f"  Desired Fz: {qp_diagnostics['desired_fz_total']:.2f} N")
            print(f"  Force scale: {qp_diagnostics['force_scale']:.3f}x")
            print(f"  Feedback gain: {wbc_controller.force_feedback_gain}")

            # Check Jacobian mapping
            from wheeled_biped.controllers.contact_jacobian import ContactJacobian

            contact_jac = ContactJacobian(mj_model)
            J_left, J_right = contact_jac.compute_wheel_jacobians(mj_data)
            print(f"\nJacobian diagnostics:")
            print(f"  J_left vertical (z) row: {J_left[2, :]}")
            print(f"  J_right vertical (z) row: {J_right[2, :]}")
            print(f"  Expected torque from 73.71 N vertical force:")
            tau_left_expected = J_left.T @ np.array([0.0, 0.0, 73.71])
            tau_right_expected = J_right.T @ np.array([0.0, 0.0, 73.71])
            print(f"    Left leg: {tau_left_expected}")
            print(f"    Right leg: {tau_right_expected}")
            print(f"Note: Using unified QP force distribution with hip roll torques\n")

        # WBC is the primary torque path. Posture is secondary and budgeted.
        wbc_error_magnitude = float(jnp.linalg.norm(qp_diagnostics.get('wrench_error_norm', 0.0)))
        momentum_magnitude = 0.0  # Not using momentum coordinator in this test

        # Stage 2: Use StaticPostureHoldingController if enabled, otherwise use PostureRegularizer
        if static_posture_controller is not None:
            # Stage 2: Static posture holding for correction-only WBC
            tau_static_posture, posture_diag = static_posture_controller.compute_posture_holding_torque(
                joint_pos, joint_vel
            )
            tau_posture = jnp.zeros(10)  # Disable PostureRegularizer
            tau_leg_position = jnp.zeros(10)  # Disable LegPositionController

            # Stage 2B: Compute feedforward torque if enabled
            if static_feedforward_controller is not None:
                tau_static_feedforward = jnp.array(static_feedforward_controller.compute_feedforward())
            else:
                tau_static_feedforward = jnp.zeros(10)

            if step < 10:
                print(f"[STAGE 2][step={step}] tau_static_posture={np.array(tau_static_posture)}")
                print(f"[STAGE 2][step={step}] posture_error_norm={posture_diag['posture_error_norm']:.6f}")
                if static_feedforward_controller is not None:
                    print(f"[STAGE 2B][step={step}] tau_static_feedforward={np.array(tau_static_feedforward)}")
        else:
            # Legacy path: PostureRegularizer
            tau_posture = compute_posture_jit(joint_pos, wbc_error_magnitude, momentum_magnitude, height_cmd)
            tau_static_posture = jnp.zeros(10)
            tau_static_feedforward = jnp.zeros(10)

        tau_wheel_secondary = jnp.zeros(10)
        RAW_INVERSE_DYNAMICS_DIAGNOSTIC_ENABLED = False
        if RAW_INVERSE_DYNAMICS_DIAGNOSTIC_ENABLED:
            mujoco.mj_inverse(mj_model, mj_data)
            tau_inverse_dynamics = jnp.array(mj_data.qfrc_inverse[6:16])
        else:
            tau_inverse_dynamics = jnp.zeros(10)

        # FIX: Use setup equilibrium for target_joint_pos when height-variant setup is provided.
        # This prevents initial joint errors that cause tau_pitch bias and forward lean.
        # Before: target_joint_pos from posture_regularizer.height_targets (h=0.40 -> hip_pitch=0.9261)
        # After: target_joint_pos from setup.equilibrium_joint_pos (hip_pitch=1.3761 for low_0p300)
        # When dynamic height trajectory is active, always use posture regularizer for active height tracking.
        if dynamic_height_active:
            target_joint_pos = posture_regularizer.compute_target_posture_from_height(height_cmd)
        elif height_variant_setup is not None and "equilibrium_joint_pos" in height_variant_setup:
            target_joint_pos = jnp.array(height_variant_setup["equilibrium_joint_pos"])
        else:
            target_joint_pos = posture_regularizer.compute_target_posture_from_height(height_cmd)
        joint_pos_error = target_joint_pos - joint_pos
        tau_hip_roll_centering_raw = compute_step4_hip_roll_centering(joint_pos, joint_vel)
        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        capture_point_error_y = float(centroidal_state_control.capture_point[1] - centroidal_state_control.com_pos[1])
        if args.enable_secondary_wheel_balance:
            tau_wheel_balance_raw = compute_step5_wheel_balance(
                pitch_x_rad,
                centroidal_state_control.pitch_rate,
                capture_point_error_y,
            )
        else:
            tau_wheel_balance_raw = jnp.zeros(10)
        if args.disable_wbc_joint_scale:
            wbc_joint_scale = jnp.ones(10)
        else:
            wbc_joint_scale = build_step6_wbc_joint_scale(control_mode)

        # Compute scaled WBC torque (used in both paths)
        tau_wbc_scaled = tau_wbc * wbc_joint_scale

        # Stage 2B ablation gating for component isolation
        if static_posture_controller is not None and static_feedforward_controller is not None:
            mode = args.stage2b_ablation_mode
            include_wbc = mode in ["B", "C", "D", "E"] and (not args.disable_wbc_correction)
            include_hip_roll = mode in ["C", "E"] and (not args.disable_hip_roll_centering)
            include_wheel_balance = mode in ["D", "E"] and (not args.disable_wheel_balance)
        else:
            mode = "LEGACY"
            include_wbc = True
            include_hip_roll = True
            include_wheel_balance = True

        # CRITICAL FIX: Disable legacy torque sources in balance-core mode
        # Balance-core uses clean four-source architecture:
        #   - ShapePostureController (not legacy static posture)
        #   - SupportFeedforwardController (not legacy feedforward)
        #   - LateralRollBalanceController (not legacy hip roll centering)
        #   - SagittalVelocityDampedBalanceController (not legacy wheel balance)
        if args.controller_mode == "balance-core":
            include_wbc = False
            include_hip_roll = False
            include_wheel_balance = False

        # Stage 2B: Compute direct roll controller torque if enabled
        tau_stage2b_roll_direct = jnp.zeros(10)
        roll_direct_diagnostics = {}
        if stage2b_roll_direct_controller is not None:
            tau_stage2b_roll_direct, roll_direct_diagnostics = stage2b_roll_direct_controller.compute_roll_torques(
                roll_y=float(centroidal_state_control.body_roll_y),
                roll_rate_y=float(centroidal_state_control.body_roll_rate_y),
            )

        # Stage 2B: Compute sagittal wheel controller torque if enabled
        tau_stage2b_sagittal_wheel = jnp.zeros(10)
        sagittal_wheel_diagnostics = {}
        sagittal_controller_input_pitch_x = 0.0
        sagittal_controller_input_pitch_rate_x = 0.0
        sagittal_controller_input_cp_y = 0.0
        sagittal_controller_input_com_y = 0.0
        sagittal_controller_input_com_vy = 0.0
        if stage2b_sagittal_wheel_controller is not None:
            sagittal_controller_input_pitch_x = float(centroidal_state_control.body_pitch_x)
            sagittal_controller_input_pitch_rate_x = float(centroidal_state_control.body_pitch_rate_x)
            sagittal_controller_input_cp_y = float(centroidal_state_control.capture_point[1])
            sagittal_controller_input_com_y = float(centroidal_state_control.com_pos[1])
            sagittal_controller_input_com_vy = float(centroidal_state_control.com_vel[1])
            tau_stage2b_sagittal_wheel, sagittal_wheel_diagnostics = stage2b_sagittal_wheel_controller.compute_wheel_torques(
                pitch_x=sagittal_controller_input_pitch_x,
                pitch_rate_x=sagittal_controller_input_pitch_rate_x,
                cp_y=sagittal_controller_input_cp_y,
                com_y=sagittal_controller_input_com_y,
                com_vy=sagittal_controller_input_com_vy,
            )

        # Stage 2C: Compute sagittal state-feedback controller torque if enabled
        tau_stage2c_sagittal_state_feedback = jnp.zeros(10)
        stage2c_diagnostics = {}
        if stage2c_sagittal_state_feedback_controller is not None:
            sagittal_controller_input_pitch_x = float(centroidal_state_control.body_pitch_x)
            sagittal_controller_input_pitch_rate_x = float(centroidal_state_control.body_pitch_rate_x)
            sagittal_controller_input_cp_y = float(centroidal_state_control.capture_point[1])
            sagittal_controller_input_com_y = float(centroidal_state_control.com_pos[1])
            sagittal_controller_input_com_vy = float(centroidal_state_control.com_vel[1])
            # Extract wheel velocities from qvel: joint indices 4 (l_wheel) and 9 (r_wheel)
            # qvel indices are offset by -1 from joint indices: qvel[10] = l_wheel, qvel[15] = r_wheel
            wheel_vel_left = float(joint_vel[4])  # l_wheel velocity
            wheel_vel_right = float(joint_vel[9])  # r_wheel velocity
            tau_stage2c_sagittal_state_feedback, stage2c_diagnostics = stage2c_sagittal_state_feedback_controller.compute_wheel_torques(
                pitch_x=sagittal_controller_input_pitch_x,
                pitch_rate_x=sagittal_controller_input_pitch_rate_x,
                com_y=sagittal_controller_input_com_y,
                com_vy=sagittal_controller_input_com_vy,
                cp_y=sagittal_controller_input_cp_y,
                wheel_vel_left=wheel_vel_left,
                wheel_vel_right=wheel_vel_right,
            )

        # Stage 2D: Compute sagittal LQR controller torque if enabled
        tau_stage2d_sagittal_lqr = jnp.zeros(10)
        stage2d_diagnostics = {}
        if stage2d_sagittal_lqr_controller is not None:
            sagittal_controller_input_pitch_x = float(centroidal_state_control.body_pitch_x)
            sagittal_controller_input_pitch_rate_x = float(centroidal_state_control.body_pitch_rate_x)
            sagittal_controller_input_cp_y = float(centroidal_state_control.capture_point[1])
            sagittal_controller_input_com_vy = float(centroidal_state_control.com_vel[1])
            wheel_vel_left = float(joint_vel[4])  # l_wheel velocity
            wheel_vel_right = float(joint_vel[9])  # r_wheel velocity
            tau_stage2d_sagittal_lqr, stage2d_diagnostics = stage2d_sagittal_lqr_controller.compute_wheel_torques(
                pitch_x=sagittal_controller_input_pitch_x,
                pitch_rate_x=sagittal_controller_input_pitch_rate_x,
                cp_y=sagittal_controller_input_cp_y,
                com_vy=sagittal_controller_input_com_vy,
                wheel_vel_left=wheel_vel_left,
                wheel_vel_right=wheel_vel_right,
            )

        # Stage 2B joint ownership mask: WBC only controls hip_roll and wheels
        # Static feedforward/posture own hip_pitch/knee to prevent conflict
        # If direct roll controller is enabled, WBC does not control hip_roll
        # If sagittal wheel controller (Stage 2B or Stage 2C) is enabled, WBC does not control wheels
        if static_posture_controller is not None and static_feedforward_controller is not None and include_wbc:
            tau_wbc_stage2b = jnp.zeros(10)
            # Only include hip_roll from WBC if direct roll controller is NOT enabled
            if stage2b_roll_direct_controller is None:
                tau_wbc_stage2b = tau_wbc_stage2b.at[0].set(tau_wbc_scaled[0])  # l_hip_roll
                tau_wbc_stage2b = tau_wbc_stage2b.at[5].set(tau_wbc_scaled[5])  # r_hip_roll
            # Only include wheels from WBC if sagittal controllers (Stage 2B, 2C, or 2D) are NOT enabled
            if (stage2b_sagittal_wheel_controller is None and
                stage2c_sagittal_state_feedback_controller is None and
                stage2d_sagittal_lqr_controller is None):
                tau_wbc_stage2b = tau_wbc_stage2b.at[4].set(tau_wbc_scaled[4])  # l_wheel
                tau_wbc_stage2b = tau_wbc_stage2b.at[9].set(tau_wbc_scaled[9])  # r_wheel
            tau_wbc_correction = tau_wbc_stage2b
        else:
            tau_wbc_correction = tau_wbc_scaled if include_wbc else jnp.zeros(10)

        tau_hip_roll_centering = tau_hip_roll_centering_raw if include_hip_roll else jnp.zeros(10)
        tau_wheel_balance = tau_wheel_balance_raw if include_wheel_balance else jnp.zeros(10)

        # Default sagittal diagnostics — overwritten inside balance-core branch
        sagittal_diag = {}

        # Default pitch rate estimate — overwritten inside velocity-damped controller branch
        from wheeled_biped.controllers.pitch_rate_consistency_estimator import PitchRateEstimate
        pitch_rate_estimate = PitchRateEstimate(
            pitch_rate_corrected=0.0,
            pitch_rate_measured=0.0,
            pitch_rate_fd=0.0,
            consistency_error=0.0,
            sign_mismatch=False,
            source_used="N/A",
            filter_alpha=0.0,
        )

        # Balance-core runtime branch: route torque through composer
        _t_bc_start = time.perf_counter() if _profile_enabled else 0.0
        if is_balance_core_mode(args):
            contact_output = balance_core_controllers["contact_supervisor"].update(
                left_wheel_contact=bool(centroidal_state_control.left_wheel_contact),
                right_wheel_contact=bool(centroidal_state_control.right_wheel_contact),
                contact_force_valid=bool(centroidal_state_control.contact_force_valid),
                left_normal_force_n=float(centroidal_state_control.left_wheel_force),
                right_normal_force_n=float(centroidal_state_control.right_wheel_force),
            )

            # HY-FF debug: Capture values being passed to shape_posture.compute()
            hy_ff_height_input = float(height_variant_setup.get("target_com_z_m", height_cmd)) if height_variant_setup else float(height_cmd)
            hy_ff_support_error_input = prev_support_error
            hy_ff_setup_target = float(height_variant_setup.get("target_com_z_m", 0.0)) if height_variant_setup else 0.0
            hy_ff_setup_achieved = float(height_variant_setup.get("achieved_com_z_m", 0.0)) if height_variant_setup else 0.0
            hy_ff_root_z = float(mj_data.qpos[2]) if len(mj_data.qpos) > 2 else 0.0
            hy_ff_current_com_z = float(centroidal_state_control.com_pos[2])

            tau_shape_posture, shape_diag = balance_core_controllers["shape_posture"].compute(
                q_ref=equilibrium_joint_pos,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                posture_weight=1.0,
                contact_degraded_scale=1.0,
                support_position_error=hy_ff_support_error_input,  # Use previous-step support error (sagittal computes after shape)
                target_com_height=hy_ff_height_input,
            )

            # Phase 2/3: Boundary yaw-position fix — modify tau_shape_posture
            # for boundary variants if a non-baseline profile is active.
            variant_name_for_fix = height_variant_setup.get("variant_name") if height_variant_setup else None
            if boundary_fix.is_active(variant_name_for_fix):
                # Get hip-yaw errors
                l_yaw_err = float(equilibrium_joint_pos[1]) - float(joint_pos[1])
                r_yaw_err = float(equilibrium_joint_pos[6]) - float(joint_pos[6])

                # Candidate 2/3: Boundary-only hip-yaw profile (override kp/kd)
                if boundary_fix.uses_boundary_hip_yaw():
                    effective_kp = boundary_fix.boundary_kp
                    effective_kd = boundary_fix.boundary_kd
                    # Recompute hip-yaw torques with boundary gains
                    l_yaw_vel = float(joint_vel[1])
                    r_yaw_vel = float(joint_vel[6])
                    l_yaw_tau_raw = effective_kp * l_yaw_err - effective_kd * l_yaw_vel
                    r_yaw_tau_raw = effective_kp * r_yaw_err - effective_kd * r_yaw_vel
                    tau_shape_posture = tau_shape_posture.at[1].set(l_yaw_tau_raw)
                    tau_shape_posture = tau_shape_posture.at[6].set(r_yaw_tau_raw)

                # Candidate 4/5: Weak hip-yaw integral/bias
                if boundary_fix.uses_integral():
                    bias_l, bias_r, integral_active, clamp_active = boundary_fix.update_integral(
                        l_hip_yaw_error=l_yaw_err,
                        r_hip_yaw_error=r_yaw_err,
                        dt=control_dt,
                    )
                    # Add bias to tau_shape_posture on hip-yaw joints
                    tau_shape_posture = tau_shape_posture.at[1].set(float(tau_shape_posture[1]) + bias_l)
                    tau_shape_posture = tau_shape_posture.at[6].set(float(tau_shape_posture[6]) + bias_r)
            tau_support_feedforward, support_diag = balance_core_controllers["support_feedforward"].compute()

            wheel_vel_left = float(joint_vel[4])
            wheel_vel_right = float(joint_vel[9])
            wheel_acc_left = (wheel_vel_left - prev_wheel_vel_left) / control_dt
            wheel_acc_right = (wheel_vel_right - prev_wheel_vel_right) / control_dt
            prev_wheel_vel_left = wheel_vel_left
            prev_wheel_vel_right = wheel_vel_right

            # E0c: Position containment via capture-point reference shaping
            # FAILED EXPERIMENT - DO NOT USE
            # Failed catastrophically: 63.72 m drift (worse than 35.22 m baseline)
            # Root cause: CP bias was ineffective, did not prevent forward drift
            # Kept for research documentation only. Must remain disabled.

            cp_error_y_m = float(centroidal_state_control.capture_point[1] - centroidal_state_control.com_pos[1])

            # Sagittal controller dispatch (mutually exclusive)
            sagittal_ctrl_name = balance_core_controllers.get("sagittal_controller_name", "baseline")
            if sagittal_ctrl_name == "velocity-damped":
                # SagittalVelocityDampedBalanceController
                # Support-position error: track wheel midpoint, NOT COM.
                # A wheeled biped is allowed to move its COM relative to the support center
                # during pitch balance. Using COM position as the standing-position error
                # causes tau_position to fight tau_pitch during the transient (COM moves
                # forward when pitching forward, even if wheels haven't moved).
                l_wheel_xpos_ctrl = tuple(float(mj_data.xpos[l_wheel_body_id][i]) for i in range(3))
                r_wheel_xpos_ctrl = tuple(float(mj_data.xpos[r_wheel_body_id][i]) for i in range(3))
                support_center_ctrl_xy = compute_support_center_xy(l_wheel_xpos_ctrl, r_wheel_xpos_ctrl)

                # C1/C3: Yaw-aware position compensation for boundary variants
                # When hip-yaw has drifted, the raw sagittal/lateral position errors contain
                # apparent components from yaw rotation. Subtract them to isolate true drift.
                current_yaw_z = float(centroidal_state_control.body_yaw_z) if hasattr(centroidal_state_control, 'body_yaw_z') else 0.0
                yaw_error_from_eq = current_yaw_z - initial_yaw_z
                mean_hip_yaw_error = 0.5 * (float(equilibrium_joint_pos[1]) - float(joint_pos[1]) + float(equilibrium_joint_pos[6]) - float(joint_pos[6]))
                raw_lateral_error = support_center_ctrl_xy[0] - support_center_eq_xy[0]
                raw_sagittal_error = project_sagittal_displacement(
                    origin_xy=support_center_eq_xy,
                    sagittal_axis_xy=sagittal_axis_xy_initial,
                    current_xy=support_center_ctrl_xy,
                )
                compensated_sagittal_error, compensated_lateral_error = boundary_fix.apply_yaw_aware_position_compensation(
                    raw_sagittal_error=raw_sagittal_error,
                    raw_lateral_error=raw_lateral_error,
                    yaw_error=mean_hip_yaw_error,
                    yaw_compensation_gain=1.0,
                    max_compensation=0.05,
                )
                # Use compensated error for sagittal position tracking
                sag_pos_error = compensated_sagittal_error
                # COM position error (diagnostic only — not used for position hold)
                com_pos_error_sagittal = project_sagittal_displacement(
                    origin_xy=(float(com_pos_eq[0]), float(com_pos_eq[1])),
                    sagittal_axis_xy=sagittal_axis_xy_initial,
                    current_xy=(float(centroidal_state_control.com_pos[0]), float(centroidal_state_control.com_pos[1])),
                )
                sag_vel = project_sagittal_velocity(
                    sagittal_axis_xy=sagittal_axis_xy_initial,
                    velocity_xy=(float(centroidal_state_control.com_vel[0]), float(centroidal_state_control.com_vel[1])),
                )
                # ---- Phase B: support-position outer loop ----------------------
                # Bounded, gated, opt-in dynamic nudge to pitch_ref driven by the
                # live (unscaled) support-position error, layered on top of the
                # frozen height-scheduled offset. Inert for every legacy profile
                # (outer_loop_active_profile is False): the dynamic term stays 0.0
                # and pitch_ref_total_deg == vd_pitch_ref_offset_deg.
                # See docs/validation/support_position_outer_loop_pitch_ref_design.md.
                outer_loop_pitch_ref_dynamic_deg = 0.0
                outer_loop_pitch_ref_total_deg = (
                    float(vd_pitch_ref_offset_deg)
                    + float(support_outer_loop_pitch_ref_offset_deg)
                )
                outer_loop_support_error_rate_mps = 0.0
                outer_loop_gate_pass = False
                outer_loop_block_reason = "disabled"
                support_outer_loop_cap_active_step = False
                support_outer_loop_pitch_ref_contrib_step = 0.0
                if outer_loop_active_profile:
                    ol_support_error = float(sag_pos_error)
                    # Low-passed numerical derivative of the support error.
                    if outer_loop_prev_support_error_m is None:
                        ol_rate_raw = 0.0
                    else:
                        ol_rate_raw = (
                            ol_support_error - outer_loop_prev_support_error_m
                        ) / control_dt
                    outer_loop_support_error_rate_smoothed = apply_lowpass(
                        outer_loop_support_error_rate_smoothed,
                        ol_rate_raw,
                        outer_loop_vel_lowpass,
                    )
                    outer_loop_prev_support_error_m = ol_support_error
                    outer_loop_support_error_rate_mps = (
                        outer_loop_support_error_rate_smoothed
                    )
                    # Safety gates (additive; never relax inner-loop gates).
                    ol_pitch_deg = abs(float(centroidal_state_control.body_pitch_x)) * 180.0 / math.pi
                    ol_roll_deg = abs(float(centroidal_state_control.body_roll_y)) * 180.0 / math.pi
                    ol_contact_ok = bool(
                        contact_output.left_wheel_contact
                        and contact_output.right_wheel_contact
                        and contact_output.contact_force_valid
                    )
                    if outer_loop_contact_required and not ol_contact_ok:
                        outer_loop_block_reason = "contact_invalid"
                    elif abs(ol_support_error) > outer_loop_disable_abs_error:
                        outer_loop_block_reason = "error_too_large"
                    elif ol_pitch_deg > outer_loop_disable_pitch_deg:
                        outer_loop_block_reason = "pitch_unsafe"
                    elif ol_roll_deg > outer_loop_disable_roll_deg:
                        outer_loop_block_reason = "roll_unsafe"
                    else:
                        outer_loop_gate_pass = True
                        outer_loop_block_reason = "active"

                    if outer_loop_gate_pass:
                        # Integral path (disabled initially: Ki=0, integral off).
                        if outer_loop_integral_enabled:
                            outer_loop_integral_accum_m_s += ol_support_error * control_dt
                            if outer_loop_integral_accum_m_s > outer_loop_integral_clamp:
                                outer_loop_integral_accum_m_s = outer_loop_integral_clamp
                            elif outer_loop_integral_accum_m_s < -outer_loop_integral_clamp:
                                outer_loop_integral_accum_m_s = -outer_loop_integral_clamp
                        target_dynamic_deg = compute_outer_loop_pitch_ref(
                            support_error_m=ol_support_error,
                            support_error_rate_m_s=outer_loop_support_error_rate_smoothed,
                            integral_error_m_s=outer_loop_integral_accum_m_s,
                            kp_deg_per_m=outer_loop_kp,
                            kd_deg_per_mps=outer_loop_kd,
                            ki_deg_per_m_s=outer_loop_ki,
                            deadband_m=outer_loop_error_deadband,
                            theta_ref_max_deg=outer_loop_theta_ref_max,
                        )
                        support_outer_loop_cap_active_step = (
                            abs(target_dynamic_deg) >= outer_loop_theta_ref_max - 1e-9
                        )
                    else:
                        # Gated off: decay the dynamic term toward 0 (no step).
                        target_dynamic_deg = 0.0
                    # Rate-limit then low-pass toward the target (or toward 0).
                    rate_limited = apply_rate_limit(
                        outer_loop_pitch_ref_smoothed_deg,
                        target_dynamic_deg,
                        outer_loop_theta_rate_limit,
                    )
                    outer_loop_pitch_ref_smoothed_deg = apply_lowpass(
                        outer_loop_pitch_ref_smoothed_deg,
                        rate_limited,
                        outer_loop_theta_lowpass,
                    )
                    outer_loop_pitch_ref_dynamic_deg = (
                        outer_loop_pitch_ref_smoothed_deg
                    )
                    support_outer_loop_pitch_ref_contrib_step = outer_loop_pitch_ref_dynamic_deg
                    outer_loop_pitch_ref_total_deg = (
                        float(vd_pitch_ref_offset_deg)
                        + float(support_outer_loop_pitch_ref_offset_deg)
                        + outer_loop_pitch_ref_dynamic_deg
                    )

                # Pitch error relative to equilibrium reference.
                # Using raw pitch_x means the controller fights any residual equilibrium pitch offset.
                pitch_x_ref = float(pitch_x_eq) + math.radians(outer_loop_pitch_ref_total_deg)
                pitch_x_error = float(centroidal_state_control.body_pitch_x) - pitch_x_ref

                # Pitch rate consistency estimator: DISABLED by default in active control.
                # The fix did not reduce the transient peak (0.595 m unchanged) and caused
                # height variant regressions (both high_5cm and low_5cm fell due to filter lag).
                # Estimator is still called for diagnostic telemetry only.
                pitch_rate_estimate = pitch_rate_estimator.estimate(
                    pitch_x=float(centroidal_state_control.body_pitch_x),
                    pitch_rate_measured=float(centroidal_state_control.body_pitch_rate_x),
                )

                # Use raw measured pitch rate by default, corrected rate only if explicitly enabled.
                if args.vd_enable_pitch_rate_correction:
                    pitch_rate_for_control = pitch_rate_estimate.pitch_rate_corrected
                else:
                    pitch_rate_for_control = float(centroidal_state_control.body_pitch_rate_x)

                # Diagnostic: position authority scaling for transient disambiguation
                position_authority_scale = 1.0
                pitch_rate_boost_factor = 1.0
                transient_detected = False

                # Transient detection for T1-T4 modes
                pitch_deg = abs(float(centroidal_state_control.body_pitch_x)) * 57.3
                pitch_rate_abs = abs(float(centroidal_state_control.body_pitch_rate_x))
                com_z_m = float(centroidal_state_control.com_pos[2])
                com_vz = float(centroidal_state_control.com_vel[2])

                # Detect transient condition
                transient_by_pitch = pitch_deg > args.vd_transient_pitch_threshold_deg
                transient_by_pitch_rate = pitch_rate_abs > args.vd_transient_pitch_rate_threshold
                transient_by_height = com_z_m < 0.38 and com_vz < -0.01
                transient_detected = transient_by_pitch or transient_by_pitch_rate or transient_by_height

                # Apply transient capture mode
                transient_mode = args.vd_transient_capture_mode
                if transient_mode == "T1":
                    # T1: Position hold freeze during transient
                    if transient_detected:
                        position_authority_scale = 0.0
                elif transient_mode == "T2":
                    # T2: Position authority scaling (continuous)
                    if transient_detected:
                        # Scale down based on pitch magnitude
                        pitch_excess = max(0.0, pitch_deg - args.vd_transient_pitch_threshold_deg)
                        position_authority_scale = max(
                            args.vd_transient_position_scale_min,
                            1.0 - pitch_excess / 5.0
                        )
                elif transient_mode == "T3":
                    # T3: Pitch-rate transient boost (no position change)
                    if transient_detected:
                        pitch_rate_boost_factor = args.vd_transient_pitch_rate_boost_factor
                elif transient_mode == "T4":
                    # T4: Combined scaling + pitch-rate boost
                    if transient_detected:
                        pitch_excess = max(0.0, pitch_deg - args.vd_transient_pitch_threshold_deg)
                        position_authority_scale = max(
                            args.vd_transient_position_scale_min,
                            1.0 - pitch_excess / 5.0
                        )
                        pitch_rate_boost_factor = args.vd_transient_pitch_rate_boost_factor

                # Legacy Config C: Ramp-in position authority over first N steps
                if args.vd_position_ramp_steps > 0:
                    ramp_progress = min(1.0, step / args.vd_position_ramp_steps)
                    position_authority_scale *= ramp_progress

                # Legacy Config D: Balance-safety scheduling - reduce position authority when unsafe
                if args.vd_balance_safety_scheduling:
                    if pitch_deg > args.vd_safety_pitch_threshold_deg:
                        pitch_excess = pitch_deg - args.vd_safety_pitch_threshold_deg
                        pitch_scale = max(0.0, 1.0 - pitch_excess / 5.0)
                        position_authority_scale *= pitch_scale
                    if com_z_m < args.vd_safety_com_z_threshold_m:
                        height_deficit = args.vd_safety_com_z_threshold_m - com_z_m
                        height_scale = max(0.0, 1.0 - height_deficit / 0.03)
                        position_authority_scale *= height_scale

                # Apply position authority scaling to position error
                sag_pos_error_scaled = sag_pos_error * position_authority_scale

                # Apply pitch rate boost for T3/T4
                pitch_rate_for_control_boosted = pitch_rate_for_control * pitch_rate_boost_factor

                # Compute support center position for capture gate
                support_center_y_m = float(support_center_ctrl_xy[1])
                com_y_m = float(centroidal_state_control.com_pos[1])
                com_vy_m_s = float(centroidal_state_control.com_vel[1])

                tau_sagittal_wheel_balance, sagittal_diag = balance_core_controllers["sagittal_wheel_balance"].compute(
                    pitch_x_rad=pitch_x_error,
                    pitch_rate_x_rad_s=pitch_rate_for_control_boosted,
                    sagittal_velocity_m_s=float(centroidal_state_control.com_vel[1]),
                    wheel_vel_left_rad_s=wheel_vel_left,
                    wheel_vel_right_rad_s=wheel_vel_right,
                    sagittal_position_error_m=sag_pos_error_scaled,
                    com_y_m=com_y_m,
                    com_vy_m_s=com_vy_m_s,
                    support_center_y_m=support_center_y_m,
                    com_z_m=com_z_m,
                    roll_y_rad=float(centroidal_state_control.body_roll_y),
                    contact_valid=bool(contact_output.left_wheel_contact and contact_output.right_wheel_contact and contact_output.contact_force_valid),
                    height_variant_name=height_variant_setup.get("variant_name") if height_variant_setup else None,
                    commanded_height_ref_m=height_variant_setup.get("target_com_z_m") if height_variant_setup else None,
                )
                sagittal_diag["support_position_error_m"] = float(sag_pos_error)
                sagittal_diag["support_position_error_scaled_m"] = float(sag_pos_error_scaled)
                sagittal_diag["position_authority_scale"] = float(position_authority_scale)
                sagittal_diag["com_position_error_sagittal_m"] = float(com_pos_error_sagittal)
                sagittal_diag["pitch_x_ref_rad"] = pitch_x_ref
                sagittal_diag["pitch_x_error_rad"] = pitch_x_error
                # Phase B support-position outer-loop diagnostics
                sagittal_diag["outer_loop_active"] = bool(outer_loop_active_profile and outer_loop_gate_pass)
                sagittal_diag["outer_loop_support_error_m"] = float(sag_pos_error)
                sagittal_diag["outer_loop_support_error_rate_mps"] = float(outer_loop_support_error_rate_mps)
                sagittal_diag["outer_loop_pitch_ref_dynamic_deg"] = float(outer_loop_pitch_ref_dynamic_deg)
                sagittal_diag["outer_loop_pitch_ref_total_deg"] = float(outer_loop_pitch_ref_total_deg)
                sagittal_diag["outer_loop_pitch_ref_limited_deg"] = float(outer_loop_pitch_ref_dynamic_deg)
                sagittal_diag["outer_loop_pitch_ref_rate_limited_deg"] = float(outer_loop_pitch_ref_dynamic_deg)
                sagittal_diag["outer_loop_integral_m_s"] = float(outer_loop_integral_accum_m_s)
                sagittal_diag["outer_loop_gate_pass"] = bool(outer_loop_gate_pass)
                sagittal_diag["outer_loop_block_reason"] = str(outer_loop_block_reason)
                sagittal_diag["outer_loop_sign_selected"] = str(outer_loop_sign_selected)
                sagittal_diag["support_outer_loop_height_scale"] = float(support_outer_loop_height_scale)
                sagittal_diag["support_outer_loop_kp_effective"] = float(support_outer_loop_kp_effective)
                sagittal_diag["support_outer_loop_kd_effective"] = float(support_outer_loop_kd_effective)
                sagittal_diag["support_outer_loop_pitch_ref_offset_deg"] = float(support_outer_loop_pitch_ref_offset_deg)
                sagittal_diag["support_outer_loop_pitch_ref_contrib"] = float(support_outer_loop_pitch_ref_contrib_step)
                sagittal_diag["support_outer_loop_cap_active"] = bool(support_outer_loop_cap_active_step)
                sagittal_diag["pitch_ref_offset_scheduled_deg"] = float(pitch_ref_offset_scheduled_deg)
                sagittal_diag["pitch_ref_total_after_outer_loop_deg"] = float(outer_loop_pitch_ref_total_deg)
                sagittal_diag["pitch_x_error_after_outer_loop_rad"] = float(pitch_x_error)
                # Calibrated outer-loop telemetry (Phase B calibration)
                sagittal_diag["calibrated_outer_loop_active"] = calibrated_outer_loop_enabled
                sagittal_diag["calibrated_function_profile_name"] = calibrated_function_profile_name
                sagittal_diag["calibrated_height_m"] = float(calibrated_height_m_telemetry)
                sagittal_diag["calibrated_kp_deg_per_m"] = outer_loop_kp if calibrated_outer_loop_enabled else 0.0
                sagittal_diag["calibrated_kd_deg_per_mps"] = outer_loop_kd if calibrated_outer_loop_enabled else 0.0
                sagittal_diag["calibrated_ki_deg_per_m_s"] = outer_loop_ki if calibrated_outer_loop_enabled else 0.0
                sagittal_diag["calibrated_theta_ref_max_deg"] = outer_loop_theta_ref_max if calibrated_outer_loop_enabled else 0.0
                sagittal_diag["calibrated_deadband_m"] = outer_loop_error_deadband if calibrated_outer_loop_enabled else 0.0
                sagittal_diag["calibrated_rate_limit_deg_per_step"] = outer_loop_theta_rate_limit if calibrated_outer_loop_enabled else 0.0
                sagittal_diag["calibrated_lowpass_alpha"] = outer_loop_theta_lowpass if calibrated_outer_loop_enabled else 0.0
                sagittal_diag["calibrated_integral_active"] = bool(outer_loop_integral_enabled and calibrated_outer_loop_enabled)
                sagittal_diag["calibrated_integral_value"] = outer_loop_integral_accum_m_s if calibrated_outer_loop_enabled else 0.0
                # Physics-based equilibrium feedforward telemetry (Phase D)
                sagittal_diag["physics_ff_enabled"] = bool(physics_ff_enabled)
                sagittal_diag["physics_ff_height_m"] = float(physics_ff_height_m)
                sagittal_diag["physics_ff_tau_eq_each_wheel_nm"] = float(physics_ff_tau_eq_each_wheel_nm)
                sagittal_diag["physics_ff_pitch_eq_no_off_deg"] = float(physics_ff_pitch_eq_no_off_deg)
                sagittal_diag["physics_ff_function_version"] = str(physics_ff_function_version)
                sagittal_diag["physics_ff_clamped"] = bool(physics_ff_clamped)
                sagittal_diag["empirical_pitch_ref_offset_disabled"] = bool(physics_ff_enabled)
                sagittal_diag["physics_equivalent_pitch_ref_deg"] = 0.0 if physics_ff_enabled else float(vd_pitch_ref_offset_deg)
                # Transient capture diagnostics
                sagittal_diag["transient_detected"] = transient_detected
                sagittal_diag["transient_by_pitch"] = transient_by_pitch
                sagittal_diag["transient_by_pitch_rate"] = transient_by_pitch_rate
                sagittal_diag["transient_by_height"] = transient_by_height
                sagittal_diag["pitch_rate_boost_factor"] = float(pitch_rate_boost_factor)
                sagittal_diag["pitch_rate_for_control_boosted"] = float(pitch_rate_for_control_boosted)
                sagittal_diag["transient_capture_mode"] = transient_mode
                sagittal_diag["sagittal_axis_x_initial"] = float(sagittal_axis_xy_initial[0])
                sagittal_diag["sagittal_axis_y_initial"] = float(sagittal_axis_xy_initial[1])
                sagittal_diag["raw_com_vx"] = float(centroidal_state_control.com_vel[0])
                sagittal_diag["raw_com_vy"] = float(centroidal_state_control.com_vel[1])
                sagittal_diag["projected_sagittal_velocity_m_s"] = float(sag_vel)
                sagittal_diag["actual_sagittal_velocity_passed_to_controller_m_s"] = float(centroidal_state_control.com_vel[1])
                sagittal_diag["support_center_x"] = float(support_center_ctrl_xy[0])
                sagittal_diag["support_center_y"] = float(support_center_ctrl_xy[1])
            else:
                # Baseline SagittalWheelBalanceController
                tau_sagittal_wheel_balance, sagittal_diag = balance_core_controllers["sagittal_wheel_balance"].compute(
                    pitch_x_rad=float(centroidal_state_control.body_pitch_x),
                    pitch_rate_x_rad_s=float(centroidal_state_control.body_pitch_rate_x),
                    cp_error_y_m=cp_error_y_m,
                    com_vy_m_s=float(centroidal_state_control.com_vel[1]),
                    wheel_vel_left_rad_s=wheel_vel_left,
                    wheel_vel_right_rad_s=wheel_vel_right,
                    outer_position_bias=0.0,
                    position_y_m=float(centroidal_state_control.com_pos[1]),
                    roll_y_rad=float(centroidal_state_control.body_roll_y),
                )
            tau_lateral_roll_balance, lateral_diag = balance_core_controllers["lateral_roll_balance"].compute(
                roll_y_rad=float(centroidal_state_control.body_roll_y),
                roll_rate_y_rad_s=float(centroidal_state_control.body_roll_rate_y),
                hip_roll_pos=(float(joint_pos[0]), float(joint_pos[5])),
                hip_roll_vel=(float(joint_vel[0]), float(joint_vel[5])),
                hip_roll_ref=(float(equilibrium_joint_pos[0]), float(equilibrium_joint_pos[5])),
            )

            # Compute yaw stabilization torque
            # Compute yaw directly from quaternion since centroidal_state yaw is NaN during control phase
            quat = np.array(mj_data.qpos[3:7])
            _, _, current_yaw = compute_orientation_from_quaternion(quat)
            yaw_rate = float(mj_data.qvel[5])  # Body-frame yaw rate (z-axis angular velocity)
            yaw_error = 0.0 - current_yaw  # Reference yaw is zero

            # Check if differential wheel yaw stabilizer is active
            wheel_yaw_stabilizer = balance_core_controllers.get("wheel_yaw_stabilizer")
            wheel_yaw_enabled = wheel_yaw_stabilizer is not None

            if wheel_yaw_enabled:
                # === WHEEL YAW stabilizer: reduced hip-yaw + post-composer wheel yaw ===
                # The YawController already has a reduced max_yaw_torque=2.0 Nm
                # (set at build time) to keep hip_yaw < 0.35 rad.
                # Wheel yaw torque is applied AFTER the composer as a direct additive
                # on the final tau_smooth, so it does NOT compete with the sagittal
                # balance torque budget or the composer's rate limiting.
                tau_yaw, yaw_diag = balance_core_controllers["yaw_controller"].compute(
                    yaw_error=yaw_error,
                    yaw_rate=yaw_rate,
                )
                # YawController goes to hip-yaw joints as normal (reduced max)
                tau_shape_posture_with_yaw = tau_shape_posture.at[1].add(tau_yaw[1])
                tau_shape_posture_with_yaw = tau_shape_posture_with_yaw.at[6].add(tau_yaw[6])
                # Compute wheel yaw (additional yaw correction via wheels)
                com_z_for_yaw = float(centroidal_state_control.com_pos[2])
                tau_wheel_yaw, wheel_yaw_diag = wheel_yaw_stabilizer.compute(
                    yaw_error=yaw_error,
                    yaw_rate=yaw_rate,
                    current_height_m=com_z_for_yaw,
                )
                yaw_diag["wheel_yaw_enabled"] = True
                # Check if M family profile activated the stabilizer (vs CLI)
                m_profile_name = balance_core_controllers["sagittal_wheel_balance"].authority_schedule.profile_name
                yaw_diag["wheel_yaw_profile_activated"] = bool(m_profile_name.startswith("m"))
                yaw_diag.update(wheel_yaw_diag)
            else:
                # === Legacy behavior: yaw via hip-yaw joints ===
                tau_yaw, yaw_diag = balance_core_controllers["yaw_controller"].compute(
                    yaw_error=yaw_error,
                    yaw_rate=yaw_rate,
                )
                # Compose yaw torque with shape posture at hip-yaw joints [1, 6]
                tau_shape_posture_with_yaw = tau_shape_posture.at[1].add(tau_yaw[1])
                tau_shape_posture_with_yaw = tau_shape_posture_with_yaw.at[6].add(tau_yaw[6])
                yaw_diag["wheel_yaw_enabled"] = False

            # ------------------------------------------------------------------
            # Mode-Based Hip-Yaw Divergence Controller (architecture fix candidate)
            # Opt-in. Computes an antisymmetric hip-yaw torque on the divergence
            # mode (left - right). The reference comes from the posture target
            # via the balance-core's tau_shape_posture reference.
            #
            # Sign convention: positive div_error (left ahead of right) ->
            # left torque negative, right torque positive.
            # ------------------------------------------------------------------
            mode_hip_yaw_div_enabled = bool(getattr(args, "enable_mode_hip_yaw_divergence", False))
            mode_div_tau_left = 0.0
            mode_div_tau_right = 0.0
            mode_div_tau_left_raw = 0.0
            mode_div_tau_right_raw = 0.0
            mode_div_error = 0.0
            mode_div_rate = 0.0
            mode_div_ref = 0.0
            mode_div_height_gate = 0.0
            mode_div_tau_left_sat = False
            mode_div_tau_right_sat = False
            mode_ownership_violation = 0
            mode_div_support_error_gate = 1.0
            mode_div_support_rate_gate = 1.0
            mode_div_effective_support_gate = 1.0
            mode_div_combined_gate = 1.0
            mode_div_support_error_val = 0.0
            mode_div_support_error_rate_val = 0.0
            if mode_hip_yaw_div_enabled:
                from wheeled_biped.controllers.hip_yaw_mode_math import (
                    decompose,
                    torque_recompose,
                    sign_for_divergence_correction,
                )
                from wheeled_biped.controllers.mode_based_hip_yaw_divergence_controller import (
                    HipYawState,
                    ModeBasedHipYawDivergenceController,
                )
                mode_div_cfg = {
                    "enabled": True,
                    "kp_div": float(args.mode_hip_yaw_div_kp),
                    "kd_div": float(args.mode_hip_yaw_div_kd),
                    "max_torque": float(args.mode_hip_yaw_div_max_torque),
                    "soft_limit_rad": float(args.mode_hip_yaw_div_soft_limit_rad),
                    "soft_limit_gain": float(args.mode_hip_yaw_div_soft_gain),
                    "ref_source": str(args.mode_hip_yaw_div_ref_source),
                    # Support-aware gating (opt-in)
                    "support_gate_enabled": bool(getattr(args, "mode_hip_yaw_div_support_enabled", False)),
                    "support_threshold_m": float(args.mode_hip_yaw_div_support_threshold_m),
                    "support_width_m": float(args.mode_hip_yaw_div_support_width_m),
                    "support_min_gate": float(args.mode_hip_yaw_div_support_min_gate),
                    "support_rate_threshold_mps": float(args.mode_hip_yaw_div_support_rate_threshold_mps),
                    "support_rate_width_mps": float(args.mode_hip_yaw_div_support_rate_width_mps),
                    "support_rate_min_gate": float(args.mode_hip_yaw_div_support_rate_min_gate),
                }
                mode_div_ctrl = ModeBasedHipYawDivergenceController(mode_div_cfg)
                l_pos = float(joint_pos[1])
                r_pos = float(joint_pos[6])
                l_vel = float(joint_vel[1])
                r_vel = float(joint_vel[6])
                l_ref = float(equilibrium_joint_pos[1])
                r_ref = float(equilibrium_joint_pos[6])
                # Reference for the divergence mode from posture target
                ref_common, ref_div = decompose(l_ref, r_ref)
                _act_common, actual_div = decompose(l_pos, r_pos)
                div_rate = l_vel - r_vel
                # Support state for support-aware gating
                support_error = float(sagittal_diag.get("support_position_error_m", 0.0))
                support_error_rate = float(sagittal_diag.get("outer_loop_support_error_rate_mps", 0.0))
                mode_div_support_error_val = support_error
                mode_div_support_error_rate_val = support_error_rate
                state = HipYawState(
                    div_error=actual_div - ref_div,
                    div_rate=div_rate,
                    height=float(centroidal_state_control.com_pos[2]),
                    support_error=support_error,
                    support_error_rate=support_error_rate,
                )
                mode_div_out = mode_div_ctrl.compute(state)
                mode_div_tau_left = float(mode_div_out["tau_left"])
                mode_div_tau_right = float(mode_div_out["tau_right"])
                mode_div_tau_left_raw = float(mode_div_out.get("tau_left_raw", mode_div_tau_left))
                mode_div_tau_right_raw = float(mode_div_out.get("tau_right_raw", mode_div_tau_right))
                mode_div_error = float(state.div_error)
                mode_div_rate = float(div_rate)
                mode_div_ref = float(ref_div)
                mode_div_height_gate = float(mode_div_ctrl._height_gate(state.height))
                # Support-aware gate telemetry
                mode_div_support_error_gate = float(mode_div_out.get("support_error_gate", 1.0))
                mode_div_support_rate_gate = float(mode_div_out.get("support_rate_gate", 1.0))
                mode_div_effective_support_gate = float(mode_div_out.get("effective_support_gate", 1.0))
                mode_div_combined_gate = float(mode_div_out.get("combined_gate", mode_div_height_gate))
                # Reconstruct from (raw_common + 0.5 * div, raw_common - 0.5 * div)
                # but apply only the antisymmetric component on hip-yaw indices 1, 6.
                # Saturation flag
                mode_div_tau_left_sat = abs(mode_div_tau_left) >= (
                    float(args.mode_hip_yaw_div_max_torque) - 1e-6
                )
                mode_div_tau_right_sat = abs(mode_div_tau_right) >= (
                    float(args.mode_hip_yaw_div_max_torque) - 1e-6
                )
                # Sanity check sign convention
                sign_for_divergence_correction(state.div_error, state.div_rate)
                # Add the antisymmetric torque to the hip-yaw slots in
                # tau_shape_posture_with_yaw BEFORE composer rate limiting.
                tau_shape_posture_with_yaw = tau_shape_posture_with_yaw.at[1].add(mode_div_tau_left)
                tau_shape_posture_with_yaw = tau_shape_posture_with_yaw.at[6].add(mode_div_tau_right)
                # Ownership: divergence mode is owned by mode_based_divergence.
                # Track any double-write to the same mode (placeholder count).
                mode_ownership_violation = 0

            # Update previous-step support error for next iteration's HY-FF
            prev_support_error = sagittal_diag.get("support_position_error_m", 0.0)

            balance_core_result = balance_core_controllers["composer"].compose(
                tau_shape_posture=tau_shape_posture_with_yaw,
                tau_support_feedforward=tau_support_feedforward,
                tau_sagittal_wheel_balance=tau_sagittal_wheel_balance,
                tau_lateral_roll_balance=tau_lateral_roll_balance,
                tau_prev=tau_prev,
            )

            tau_total_raw = balance_core_result.tau_total_raw
            tau_total_clipped = balance_core_result.tau_total_clipped
            tau_smooth = balance_core_result.tau_final
            tau_prev = tau_smooth

            # === JAX fast-path override (Stage 5) ===
            # When backend=jax, replace Python-computed tau_smooth with JAX output.
            # Python path still runs for telemetry but torque is from JAX.
            if _jax_enabled:
                _t_jax_start = time.perf_counter()
                # Pass pitch_x_error (ALREADY adjusted by sim loop).
                # JAX step must NOT apply pitch_ref_offset internally when receiving
                # pre-adjusted pitch — the sim loop applies the offset via vd_pitch_ref_offset_deg.
                _jax_input = pack_input_k2(
                    pitch_x_rad=float(pitch_x_error) if 'pitch_x_error' in dir() else 0.0,
                    pitch_rate_x_rad_s=float(pitch_rate_for_control_boosted) if 'pitch_rate_for_control_boosted' in dir() else float(centroidal_state_control.body_pitch_rate_x),
                    roll_y_rad=float(centroidal_state_control.body_roll_y),
                    roll_rate_y_rad_s=float(centroidal_state_control.body_roll_rate_y),
                    yaw_error_rad=float(centroidal_state_control.body_yaw_z - initial_yaw_z),
                    yaw_rate_rad_s=float(centroidal_state_control.body_yaw_rate_z),
                    com_z_m=float(centroidal_state_control.com_pos[2]),
                    com_vy_m_s=float(centroidal_state_control.com_vel[1]),
                    sagittal_velocity_m_s=float(centroidal_state_control.com_vel[1]),
                    sagittal_position_error_m=float(prev_support_error),
                    wheel_vel_left_rad_s=float(joint_vel[4]),
                    wheel_vel_right_rad_s=float(joint_vel[9]),
                    support_velocity_m_s=0.0,
                    commanded_height_ref_m=float(height_cmd),
                    hip_yaw_div_error=float(joint_pos[1] - joint_pos[6]),
                    hip_yaw_div_rate=float(joint_vel[1] - joint_vel[6]),
                    joint_pos=jnp.array(joint_pos),
                    joint_vel=jnp.array(joint_vel),
                    q_ref=jnp.array(equilibrium_joint_pos),
                    support_position_error_m=float(prev_support_error),
                )
                _jax_tau, _jax_state, _jax_diag = _jax_step_fn(_jax_state, _jax_input, _jax_params)
                tau_smooth = _jax_tau
                tau_total_clipped = _jax_tau  # sync for rate-limiting code below
                tau_prev = tau_smooth  # sync tau_prev for rate limiting on next step
                if _profile_enabled:
                    _dt_jax = (time.perf_counter() - _t_jax_start) * 1000.0
                    _profile_timing["jax_step_ms"] = _profile_timing.get("jax_step_ms", 0.0) + _dt_jax

            # Apply wheel yaw torque POST-composer to tau_smooth directly.
            # This does NOT compete with sagittal balance torque budget since
            # the composer has already determined the final wheel torque and
            # tau_prev is set BEFORE this addition (so rate limiting is unaffected
            # on the next step — wheel yaw is a fast additive correction).
            if wheel_yaw_enabled:
                tau_smooth = tau_smooth.at[4].add(tau_wheel_yaw[4])
                tau_smooth = tau_smooth.at[9].add(tau_wheel_yaw[9])
                # Re-clip to actuator limits for safety
                tau_smooth = jnp.clip(tau_smooth, -torque_limit_jax, torque_limit_jax)

            # Append balance-core telemetry (before physics FF, so we see composer output)
            append_balance_core_telemetry(
                telemetry,
                balance_core_result,
                centroidal_state_control,
                contact_output,
                cp_error_y_m=cp_error_y_m,
                wheel_vel_left_rad_s=wheel_vel_left,
                wheel_vel_right_rad_s=wheel_vel_right,
                wheel_acc_left_rad_s2=wheel_acc_left,
                wheel_acc_right_rad_s2=wheel_acc_right,
                hip_roll_pos=(float(joint_pos[0]), float(joint_pos[5])),
                hip_roll_ref=(float(equilibrium_joint_pos[0]), float(equilibrium_joint_pos[5])),
            )

            # Physics-based equilibrium feedforward (Phase D, opt-in).
            # When enabled via Option B (equivalent pitch_ref path), the value of
            # vd_pitch_ref_offset_deg is set to pitch_eq_no_off_deg(h) above,
            # which makes the controller's tau_pitch = 0 at steady state. No
            # additive torque injection is needed here (Option A direct-torque
            # was tried and FAILED — see comment block above the physics_ff
            # branch). We still emit telemetry so callers can distinguish the
            # two telemetry sources (physics-derived vs empirical schedule).
            if physics_ff_enabled:
                sagittal_diag["physics_ff_applied_each_wheel_nm"] = 0.0  # Option A: not used
                sagittal_diag["physics_ff_final_wheel_tau_with_ff"] = float(0.5 * (np.array(tau_total_clipped)[4] + np.array(tau_total_clipped)[9]))
                sagittal_diag["physics_ff_final_wheel_tau_without_ff"] = float(0.5 * (np.array(tau_total_clipped)[4] + np.array(tau_total_clipped)[9]))
                sagittal_diag["physics_ff_active_this_step"] = True
            else:
                sagittal_diag["physics_ff_applied_each_wheel_nm"] = 0.0
                sagittal_diag["physics_ff_final_wheel_tau_with_ff"] = 0.0
                sagittal_diag["physics_ff_final_wheel_tau_without_ff"] = 0.0
                sagittal_diag["physics_ff_active_this_step"] = False

            # Zero legacy torques for telemetry clarity
            legacy_zeros = zero_legacy_torque_sources_for_balance_core()
            tau_wbc_correction = legacy_zeros["tau_wbc_correction"]
            tau_wbc_scaled = legacy_zeros["tau_wbc_scaled"]
            tau_posture = legacy_zeros["tau_posture"]
            # Reassign balance-core torques to legacy variable names for telemetry compatibility
            tau_static_posture = tau_shape_posture
            tau_static_feedforward = tau_support_feedforward
            tau_leg_position = legacy_zeros["tau_leg_position"]
            tau_hip_roll_centering = legacy_zeros["tau_hip_roll_centering"]
            tau_wheel_balance = legacy_zeros["tau_wheel_balance"]
            tau_inverse_dynamics = legacy_zeros["tau_inverse_dynamics"]
        # Stage 2: Modify torque combination for static posture holding
        elif static_posture_controller is not None:
            # A/B/C/D/E ablations over Stage 2B/2C/2D stack
            tau_total_raw = (
                tau_static_feedforward
                + tau_static_posture
                + tau_wbc_correction
                + tau_stage2b_roll_direct
                + tau_stage2b_sagittal_wheel
                + tau_stage2c_sagittal_state_feedback
                + tau_stage2d_sagittal_lqr
                + tau_hip_roll_centering
                + tau_wheel_balance
                + tau_inverse_dynamics
            )
            tau_leg_position = jnp.zeros(10)  # Not used in Stage 2
        else:
            # Legacy path: original torque combination
            tau_leg_position = leg_position_controller.compute_leg_torques(
                joint_pos,
                joint_vel,
                target_joint_pos,
            )
            tau_total_raw = (
                tau_wbc_scaled
                + tau_hip_roll_centering
                + tau_leg_position
                + tau_posture
                + tau_wheel_secondary
                + tau_wheel_balance
                + tau_inverse_dynamics
            )

        if _profile_enabled:
            _profile_timing["balance_core_block_ms"] += (time.perf_counter() - _t_bc_start) * 1000.0

        # Balance-core already handled clipping in composer; only apply legacy processing for other modes
        if not is_balance_core_mode(args):
            torque_limit = jnp.array(mj_model.actuator_ctrlrange[:, 1])
            tau_total_clipped = jnp.clip(tau_total_raw, -torque_limit, torque_limit)
            tau_saturation_rate = float(jnp.mean(jnp.abs(tau_total_raw) > torque_limit))

            if step == 0 and args.initialize_tau_prev_from_wbc:
                tau_prev = tau_total_clipped
        else:
            # Balance-core mode: saturation rate already computed in composer
            tau_saturation_rate = float(jnp.mean(jnp.abs(tau_total_raw) > jnp.array(mj_model.actuator_ctrlrange[:, 1])))

        # Compute motor tracking telemetry
        step1_diagnostics = compute_step1_joint_diagnostics(joint_pos, joint_pos_error)
        step1_diagnostics["control_mode"] = control_mode
        joint_pos_error_norm = float(jnp.linalg.norm(joint_pos_error))
        joint_vel_norm = float(jnp.linalg.norm(mj_data.qvel[6:16]))
        tau_wbc_norm = float(jnp.linalg.norm(tau_wbc))
        tau_posture_norm = float(jnp.linalg.norm(tau_posture))
        tau_inverse_dynamics_norm = float(jnp.linalg.norm(tau_inverse_dynamics))
        tau_total_norm = float(jnp.linalg.norm(tau_total_clipped))

        # Balance-core already handled rate limiting in composer; only apply legacy processing for other modes
        if not is_balance_core_mode(args):
            # Compute torque rate (Nm/s) and optionally apply limiting.
            tau_rate_unlimited = float(jnp.linalg.norm(tau_total_clipped - tau_prev) / control_dt)
            max_torque_rate = 400.0
            tau_rate_vec = (tau_total_clipped - tau_prev) / control_dt
            tau_rate_vec_clipped = jnp.clip(tau_rate_vec, -max_torque_rate, max_torque_rate)

            if args.disable_torque_rate_limit:
                tau_smooth = tau_total_clipped
                tau_rate_limited = tau_rate_unlimited
            else:
                tau_smooth = tau_prev + tau_rate_vec_clipped * control_dt
                tau_rate_limited = float(jnp.linalg.norm(tau_rate_vec_clipped))

            tau_prev = tau_smooth
        else:
            # Balance-core mode: rate limiting already applied in composer
            tau_rate_vec = (tau_total_clipped - tau_prev) / control_dt
            tau_rate_unlimited = float(jnp.linalg.norm(tau_rate_vec))
            tau_rate_limited = tau_rate_unlimited  # Composer already applied rate limiting

        sat_flags_vec = (np.abs(np.array(tau_total_raw)) > np.array(torque_limit)).astype(int)
        rate_flags_vec = (np.abs(np.array(tau_rate_vec)) > np.array(max_torque_rate)).astype(int)

        # Early-step support torque parity diagnostics
        j_left_dbg, j_right_dbg = contact_jacobian.compute_wheel_jacobians(mj_data)
        f_up_left = jnp.array([0.0, 0.0, robot_mass * gravity / 2.0])
        f_up_right = jnp.array([0.0, 0.0, robot_mass * gravity / 2.0])
        tau_ideal = j_left_dbg.T @ f_up_left + j_right_dbg.T @ f_up_right
        support_indices = [2, 3, 7, 8]
        support_ratios = [
            float(jnp.abs(tau_smooth[idx]) / jnp.maximum(jnp.abs(tau_ideal[idx]), 1e-6))
            for idx in support_indices
        ]
        support_ratio_mean = float(np.mean(support_ratios))

        if step < 10 and not args.visual:
            print(f"[EARLY SUPPORT][step={step}] tau_wbc={np.array(tau_wbc)}")
            print(f"[EARLY SUPPORT][step={step}] tau_wbc_scaled={np.array(tau_wbc_scaled)}")
            print(f"[EARLY SUPPORT][step={step}] tau_total_raw={np.array(tau_total_raw)}")
            print(f"[EARLY SUPPORT][step={step}] tau_total_clipped={np.array(tau_total_clipped)}")
            print(f"[EARLY SUPPORT][step={step}] tau_smooth={np.array(tau_smooth)}")
            print(
                f"[EARLY SUPPORT][step={step}] support_ratio_[2,3,7,8]={support_ratios}, mean={support_ratio_mean:.4f}, "
                f"rate_limit_enabled={not args.disable_torque_rate_limit}, "
                f"per_actuator_wbc_authority={args.use_per_actuator_wbc_authority}, "
                f"wbc_joint_scaling_enabled={not args.disable_wbc_joint_scale}"
            )

            # Stage2B joint ownership diagnostics
            if static_posture_controller is not None and static_feedforward_controller is not None:
                support_joints = [2, 3, 7, 8]  # hip_pitch/knee
                print(f"[STAGE2B OWNERSHIP][step={step}] tau_wbc_raw[2,3,7,8]={[float(tau_wbc_scaled[i]) for i in support_joints]}")
                print(f"[STAGE2B OWNERSHIP][step={step}] tau_wbc_correction[2,3,7,8]={[float(tau_wbc_correction[i]) for i in support_joints]}")
                print(f"[STAGE2B OWNERSHIP][step={step}] tau_static_feedforward[2,3,7,8]={[float(tau_static_feedforward[i]) for i in support_joints]}")
                print(f"[STAGE2B OWNERSHIP][step={step}] tau_static_posture[2,3,7,8]={[float(tau_static_posture[i]) for i in support_joints]}")
                print(f"[STAGE2B OWNERSHIP][step={step}] tau_total_raw[2,3,7,8]={[float(tau_total_raw[i]) for i in support_joints]}")

                # Knee state diagnostics
                knee_indices = [3, 8]  # l_knee, r_knee
                knee_pos = [float(joint_pos[i]) for i in knee_indices]
                knee_vel = [float(joint_vel[i]) for i in knee_indices]
                knee_error = [float(joint_pos_error[i]) for i in knee_indices]
                print(f"[STAGE2B OWNERSHIP][step={step}] knee_pos[3,8]={knee_pos}, knee_vel={knee_vel}, knee_error={knee_error}")

                # CoM and orientation state
                com_z = float(centroidal_state_control.com_pos[2])
                com_vz = float(centroidal_state_control.com_vel[2])

                # Direct roll controller diagnostics
                if stage2b_roll_direct_controller is not None:
                    print(f"[STAGE2B ROLL DIRECT][step={step}] roll_error={roll_direct_diagnostics.get('roll_error', 0.0):.6f} rad ({roll_direct_diagnostics.get('roll_error', 0.0)*57.3:.2f} deg)")
                    print(f"[STAGE2B ROLL DIRECT][step={step}] m_roll_cmd={roll_direct_diagnostics.get('m_roll_cmd', 0.0):+.2f} Nm")
                    print(f"[STAGE2B ROLL DIRECT][step={step}] tau_hip_roll_left={roll_direct_diagnostics.get('tau_hip_roll_left', 0.0):+.2f} Nm")
                    print(f"[STAGE2B ROLL DIRECT][step={step}] tau_hip_roll_right={roll_direct_diagnostics.get('tau_hip_roll_right', 0.0):+.2f} Nm")
                    print(f"[STAGE2B ROLL DIRECT][step={step}] saturated={roll_direct_diagnostics.get('moment_saturated', False)}")
                    print(f"[STAGE2B ROLL DIRECT][step={step}] tau_stage2b_roll_direct[0,5]={[float(tau_stage2b_roll_direct[0]), float(tau_stage2b_roll_direct[5])]}")

                # Sagittal wheel controller diagnostics
                if stage2b_sagittal_wheel_controller is not None:
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] pitch_error={sagittal_wheel_diagnostics.get('pitch_error', 0.0):.6f} rad ({sagittal_wheel_diagnostics.get('pitch_error', 0.0)*57.3:.2f} deg)")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] pitch_rate_x={sagittal_wheel_diagnostics.get('pitch_rate_x', 0.0):+.6f} rad/s")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] cp_error_y={sagittal_wheel_diagnostics.get('cp_error_y', 0.0):+.6f} m")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_wheel_cmd={sagittal_wheel_diagnostics.get('tau_wheel_cmd', 0.0):+.2f} Nm")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] saturated={sagittal_wheel_diagnostics.get('saturated', False)}")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_stage2b_sagittal_wheel[4,9]={[float(tau_stage2b_sagittal_wheel[4]), float(tau_stage2b_sagittal_wheel[9])]}")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_total_raw[4,9]={[float(tau_total_raw[4]), float(tau_total_raw[9])]}")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_static_feedforward[4,9]={[float(tau_static_feedforward[4]), float(tau_static_feedforward[9])]}")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_static_posture[4,9]={[float(tau_static_posture[4]), float(tau_static_posture[9])]}")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_wbc_correction[4,9]={[float(tau_wbc_correction[4]), float(tau_wbc_correction[9])]}")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_hip_roll_centering[4,9]={[float(tau_hip_roll_centering[4]), float(tau_hip_roll_centering[9])]}")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_wheel_balance[4,9]={[float(tau_wheel_balance[4]), float(tau_wheel_balance[9])]}")
                pitch_deg = float(pitch_x_rad) * 57.3
                roll_deg = float(roll_y_rad) * 57.3
                print(f"[STAGE2B OWNERSHIP][step={step}] com_z={com_z:.4f}m, com_vz={com_vz:.4f}m/s, pitch={pitch_deg:.2f}deg, roll={roll_deg:.2f}deg")

        # Apply final torques
        mj_data.ctrl[:] = np.array(tau_smooth)

        # Step D: apply scheduled push disturbance (xfrc_applied on torso body 1).
        _apply_pending_push()

        # POINT 5: After first mj_step (only on step 0)
        if step == 0:
            # Step simulation once to get constraint forces
            mujoco.mj_step(mj_model, mj_data)
            post_step_contact = measure_wheel_floor_contact(
                mj_model,
                mj_data,
                floor_geom_id,
                l_wheel_geom_id,
                r_wheel_geom_id,
            )
            first_total_fz = post_step_contact["total_fz"]
            weight_n = robot_mass * gravity
            ratio = first_total_fz / max(weight_n, 1e-6)
            print(f"[INIT CALIB] first post-step total wheel-floor Fz: {first_total_fz:+.6f} N")
            print(f"[INIT CALIB] first post-step total_fz/weight: {ratio:+.6f}")
            print("=== END INITIALIZATION DIAGNOSTICS ===\n")

            # Continue with remaining substeps
            for _ in range(n_substeps - 1):
                mujoco.mj_step(mj_model, mj_data)
        else:
            # Normal simulation: all substeps
            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)

        # Re-estimate centroidal/contact state after physics stepping for logging.
        # Do NOT overwrite prev_control_com_pos from logging sample.
        _t_log0 = time.perf_counter() if _profile_enabled else 0.0
        centroidal_state_log, logged_com_pos = centroidal_estimator.estimate(
            jnp.zeros(42), mj_data, control_com_pos
        )
        _t_log1 = time.perf_counter() if _profile_enabled else 0.0
        centroidal_state_log = capture_estimator.update(centroidal_state_log)
        _t_log2 = time.perf_counter() if _profile_enabled else 0.0
        if _profile_enabled:
            _profile_timing["centroidal_log_ms"] += (_t_log1 - _t_log0) * 1000.0
            _profile_timing["capture_log_ms"] += (_t_log2 - _t_log1) * 1000.0

        if step < 20 and not args.visual:
            prev_ctrl_txt = (
                "None"
                if prev_control_before_estimate is None
                else np.array2string(np.array(prev_control_before_estimate), precision=6)
            )
            print(
                f"[LIFECYCLE][step={step}] prev_control_com_pos={prev_ctrl_txt}, "
                f"control_com_pos={np.array(control_com_pos)}, "
                f"control_com_vel={np.array(centroidal_state_control.com_vel)}, "
                f"cp_x={float(centroidal_state_control.capture_point[0]):+.6f}, "
                f"cp_y={float(centroidal_state_control.capture_point[1]):+.6f}, "
                f"com_vx={float(centroidal_state_control.com_vel[0]):+.6f}, "
                f"com_vy={float(centroidal_state_control.com_vel[1]):+.6f}, "
                f"com_vz={float(centroidal_state_control.com_vel[2]):+.6f}"
            )
            print(
                f"[LIFECYCLE][step={step}] log_com_pos={np.array(logged_com_pos)}, "
                f"log_com_vel={np.array(centroidal_state_log.com_vel)}, "
                f"log_cp_x={float(centroidal_state_log.capture_point[0]):+.6f}, "
                f"log_cp_y={float(centroidal_state_log.capture_point[1]):+.6f}, "
                f"log_com_vx={float(centroidal_state_log.com_vel[0]):+.6f}, "
                f"log_com_vy={float(centroidal_state_log.com_vel[1]):+.6f}, "
                f"log_com_vz={float(centroidal_state_log.com_vel[2]):+.6f}"
            )

        # Contact classification from MuJoCo geoms
        contact_class = classify_floor_contacts(
            mj_model,
            mj_data,
            floor_geom_id,
            l_wheel_geom_id,
            r_wheel_geom_id,
        )

        # Compute both Euler angles and robot-frame orientation
        quat = np.array(mj_data.qpos[3:7])  # [w, x, y, z]
        euler_roll_x, euler_pitch_y, euler_yaw_z = compute_orientation_from_quaternion(quat)

        # Robot-frame orientation from gravity vector (used for control and termination)
        robot_pitch_x = float(centroidal_state_log.body_pitch_x)
        robot_roll_y = float(centroidal_state_log.body_roll_y)
        robot_yaw_z = float(centroidal_state_log.body_yaw_z)

        # Check termination using robot-frame orientation
        com_height = float(centroidal_state_log.com_pos[2])
        terminated, termination_reason = check_termination(
            mj_data.qpos, com_height, robot_pitch_x, robot_roll_y,
            height_floor_m=termination_height_floor_m,
        )

        # Wrench diagnostics with explicit separation:
        # - full_wrench: baseline + correction
        # - correction_wrench: equilibrium-relative correction only
        full_wrench = np.array([
            qp_diagnostics["desired_wrench_Fx"],
            qp_diagnostics["desired_wrench_Fy"],
            qp_diagnostics["desired_wrench_Fz"],
            qp_diagnostics["desired_wrench_Mx"],
            qp_diagnostics["desired_wrench_My"],
            qp_diagnostics["desired_wrench_Mz"],
        ])
        full_wrench_norm = float(np.linalg.norm(full_wrench))
        correction_wrench_norm = float(qp_diagnostics.get("correction_wrench_norm", full_wrench_norm))

        _t_telem_start = time.perf_counter() if _profile_enabled else 0.0
        telemetry["source_step_index"].append(step)
        telemetry["time"].append(step * control_dt)

        # Profile identity telemetry (Phase 1 fix for T6F sign correctness investigation)
        if is_balance_core_mode(args):
            telemetry["controller_mode"].append("balance-core")
            telemetry["sagittal_controller"].append(getattr(args, "sagittal_controller", "baseline"))
            telemetry["vd_sagittal_authority_profile"].append(getattr(args, "vd_sagittal_authority_profile", "baseline"))
            height_setup_name = getattr(args, "height_variant_setup", None)
            if height_setup_name and isinstance(height_setup_name, str) and not height_setup_name.endswith(".json"):
                telemetry["height_variant_setup_name"].append(height_setup_name)
            elif height_setup_name:
                # Extract name from path
                setup_name = Path(height_setup_name).stem if height_setup_name else ""
                telemetry["height_variant_setup_name"].append(setup_name)
            else:
                telemetry["height_variant_setup_name"].append("")
        else:
            telemetry["controller_mode"].append("legacy")
            telemetry["sagittal_controller"].append("")
            telemetry["vd_sagittal_authority_profile"].append("")
            telemetry["height_variant_setup_name"].append("")

        telemetry["mass_kg"].append(robot_mass)
        telemetry["weight_N"].append(robot_mass * gravity)
        telemetry["com_x"].append(float(centroidal_state_log.com_pos[0]))
        telemetry["com_y"].append(float(centroidal_state_log.com_pos[1]))
        telemetry["com_z"].append(com_height)
        telemetry["com_vx"].append(float(centroidal_state_log.com_vel[0]))
        telemetry["com_vy"].append(float(centroidal_state_log.com_vel[1]))
        telemetry["com_vz"].append(float(centroidal_state_log.com_vel[2]))
        telemetry["cp_x"].append(float(centroidal_state_log.capture_point[0]))
        telemetry["cp_y"].append(float(centroidal_state_log.capture_point[1]))
        telemetry["tau_wbc_max"].append(float(jnp.max(jnp.abs(tau_wbc))))
        # Track actual wheel torques at indices [4, 9] from applied torque (tau_smooth)
        wheel_indices = jnp.array([4, 9])
        tau_wheel_actual = jnp.max(jnp.abs(tau_smooth[wheel_indices]))
        telemetry["tau_wheel_actual_max"].append(float(tau_wheel_actual))
        # Stage 2: Log max of tau_static_posture if enabled, otherwise tau_posture
        if static_posture_controller is not None:
            telemetry["tau_posture_max"].append(float(jnp.max(jnp.abs(tau_static_posture))))
        else:
            telemetry["tau_posture_max"].append(float(jnp.max(jnp.abs(tau_posture))))
        telemetry["tau_total_max"].append(float(jnp.max(jnp.abs(tau_smooth))))
        # Euler angles (world-frame, for reference only)
        telemetry["euler_roll_x"].append(euler_roll_x)
        telemetry["euler_pitch_y"].append(euler_pitch_y)
        telemetry["euler_yaw_z"].append(euler_yaw_z)
        # Robot-frame orientation (used for control and termination)
        telemetry["robot_pitch_x"].append(robot_pitch_x)
        telemetry["robot_roll_y"].append(robot_roll_y)
        telemetry["robot_yaw_z"].append(robot_yaw_z)
        telemetry["roll_rate_rad_s"].append(float(centroidal_state_log.roll_rate))
        telemetry["pitch_rate_rad_s"].append(float(centroidal_state_log.pitch_rate))
        telemetry["yaw_rate_rad_s"].append(float(centroidal_state_log.yaw_rate))
        telemetry["height_cmd"].append(height_cmd)  # Log adaptive height command
        telemetry["left_contact_active"].append(bool(centroidal_state_log.left_wheel_contact))
        telemetry["right_contact_active"].append(bool(centroidal_state_log.right_wheel_contact))
        telemetry["active_wheels"].append(int(active_wheels))
        telemetry["left_wheel_floor_contact"].append(bool(contact_class["left_wheel_floor_contact"]))
        telemetry["right_wheel_floor_contact"].append(bool(contact_class["right_wheel_floor_contact"]))
        telemetry["non_wheel_floor_contacts"].append(int(contact_class["non_wheel_floor_contacts"]))
        telemetry["total_wheel_floor_fz"].append(float(contact_class["total_wheel_floor_fz"]))
        telemetry["n_contacts"].append(int(mj_data.ncon))
        telemetry["contact_force_valid"].append(bool(centroidal_state_log.contact_force_valid))
        telemetry["left_contact_force_world_x"].append(float(centroidal_state_log.left_contact_force_world[0]))
        telemetry["left_contact_force_world_y"].append(float(centroidal_state_log.left_contact_force_world[1]))
        telemetry["left_contact_force_world_z"].append(float(centroidal_state_log.left_contact_force_world[2]))
        telemetry["right_contact_force_world_x"].append(float(centroidal_state_log.right_contact_force_world[0]))
        telemetry["right_contact_force_world_y"].append(float(centroidal_state_log.right_contact_force_world[1]))
        telemetry["right_contact_force_world_z"].append(float(centroidal_state_log.right_contact_force_world[2]))
        telemetry["total_contact_force_z"].append(float(centroidal_state_log.total_contact_force_z))
        telemetry["joint_pos"].append(",".join(f"{x:.4f}" for x in np.array(joint_pos)))
        telemetry["joint_vel"].append(
            ",".join(f"{x:.4f}" for x in np.array(mj_data.qvel[6:16]))
        )
        telemetry["terminated"].append(terminated)
        telemetry["termination_reason"].append(termination_reason or "")
        # QP metrics
        telemetry["qp_solve_time_ms"].append(qp_diagnostics["solve_time_ms"])
        telemetry["qp_converged"].append(
            1
        )  # Will be updated with actual convergence status
        telemetry["qp_error"].append(0.0)  # Will be updated with actual error
        telemetry["wrench_error_norm"].append(qp_diagnostics["wrench_error_norm"])
        telemetry["f_left_z"].append(qp_diagnostics["f_left_z"])
        telemetry["f_right_z"].append(qp_diagnostics["f_right_z"])
        telemetry["force_distribution_feasible"].append(qp_diagnostics["force_distribution_feasible"])
        telemetry["force_distribution_reason"].append(qp_diagnostics["force_distribution_reason"])
        telemetry["distributed_left_fx"].append(qp_diagnostics["distributed_left_fx"])
        telemetry["distributed_left_fy"].append(qp_diagnostics["distributed_left_fy"])
        telemetry["distributed_left_fz"].append(qp_diagnostics["distributed_left_fz"])
        telemetry["distributed_right_fx"].append(qp_diagnostics["distributed_right_fx"])
        telemetry["distributed_right_fy"].append(qp_diagnostics["distributed_right_fy"])
        telemetry["distributed_right_fz"].append(qp_diagnostics["distributed_right_fz"])
        telemetry["tau_saturation_rate"].append(tau_saturation_rate)
        # Desired wrench components
        telemetry["desired_wrench_Fx"].append(qp_diagnostics["desired_wrench_Fx"])
        telemetry["desired_wrench_Fy"].append(qp_diagnostics["desired_wrench_Fy"])
        telemetry["desired_wrench_Fz"].append(qp_diagnostics["desired_wrench_Fz"])
        telemetry["desired_wrench_Mx"].append(qp_diagnostics["desired_wrench_Mx"])
        telemetry["desired_wrench_My"].append(qp_diagnostics["desired_wrench_My"])
        telemetry["desired_wrench_Mz"].append(qp_diagnostics["desired_wrench_Mz"])
        telemetry["correction_wrench_norm"].append(correction_wrench_norm)
        telemetry["correction_wrench_Fx"].append(float(qp_diagnostics.get("correction_wrench_Fx", qp_diagnostics["desired_wrench_Fx"])))
        telemetry["correction_wrench_Fy"].append(float(qp_diagnostics.get("correction_wrench_Fy", qp_diagnostics["desired_wrench_Fy"])))
        telemetry["correction_wrench_Fz"].append(float(qp_diagnostics.get("correction_wrench_Fz", qp_diagnostics["desired_wrench_Fz"])))
        telemetry["correction_wrench_Mx"].append(float(qp_diagnostics.get("correction_wrench_Mx", 0.0)))
        telemetry["correction_wrench_My"].append(float(qp_diagnostics.get("correction_wrench_My", qp_diagnostics["desired_wrench_My"])))
        telemetry["correction_wrench_Mz"].append(float(qp_diagnostics.get("correction_wrench_Mz", 0.0)))
        telemetry["ablation_mode"].append(mode)
        # B500 drift audit fields (use control state, not log state)
        telemetry["pitch_x"].append(float(pitch_x_rad))
        telemetry["pitch_rate_x"].append(float(centroidal_state_control.pitch_rate_x))
        telemetry["roll_y"].append(float(roll_y_rad))
        telemetry["roll_rate_y"].append(float(centroidal_state_control.roll_rate_y))
        telemetry["yaw_z"].append(float(centroidal_state_control.yaw_z))
        telemetry["hip_roll_left_rad"].append(float(joint_pos[0]))
        telemetry["hip_roll_right_rad"].append(float(joint_pos[5]))
        hip_roll_common_component = 0.5 * float(joint_pos[0] + joint_pos[5])
        hip_roll_symmetric_component = 0.5 * float(joint_pos[0] - joint_pos[5])
        telemetry["hip_roll_common_component_rad"].append(hip_roll_common_component)
        telemetry["hip_roll_symmetric_component_rad"].append(hip_roll_symmetric_component)
        telemetry["hip_roll_abs_max_rad"].append(max(abs(float(joint_pos[0])), abs(float(joint_pos[5]))))
        telemetry["hip_roll_ref_left_rad"].append(float(equilibrium_joint_pos[0]))
        telemetry["hip_roll_ref_right_rad"].append(float(equilibrium_joint_pos[5]))
        telemetry["hip_roll_error_left_rad"].append(float(equilibrium_joint_pos[0] - joint_pos[0]))
        telemetry["hip_roll_error_right_rad"].append(float(equilibrium_joint_pos[5] - joint_pos[5]))
        telemetry["yaw_drift_from_initial_rad"].append(float(centroidal_state_control.yaw_z) - initial_yaw_z)

        # Control-time vs post-step orientation/rate telemetry
        telemetry["control_pitch_x"].append(float(centroidal_state_control.body_pitch_x))
        telemetry["control_pitch_rate_x"].append(float(centroidal_state_control.body_pitch_rate_x))
        telemetry["control_roll_y"].append(float(centroidal_state_control.body_roll_y))
        telemetry["control_roll_rate_y"].append(float(centroidal_state_control.body_roll_rate_y))
        telemetry["log_pitch_x"].append(float(centroidal_state_log.body_pitch_x))
        telemetry["log_pitch_rate_x"].append(float(centroidal_state_log.body_pitch_rate_x))
        telemetry["log_roll_y"].append(float(centroidal_state_log.body_roll_y))
        telemetry["log_roll_rate_y"].append(float(centroidal_state_log.body_roll_rate_y))

        # Compute finite-difference rates from logged orientation
        if prev_log_pitch_x is not None:
            fd_pitch_rate_x = (float(centroidal_state_log.body_pitch_x) - prev_log_pitch_x) / control_dt
            fd_roll_rate_y = (float(centroidal_state_log.body_roll_y) - prev_log_roll_y) / control_dt
        else:
            fd_pitch_rate_x = 0.0
            fd_roll_rate_y = 0.0
        telemetry["fd_pitch_rate_x"].append(fd_pitch_rate_x)
        telemetry["fd_roll_rate_y"].append(fd_roll_rate_y)

        # Update previous logged orientation for next step
        prev_log_pitch_x = float(centroidal_state_log.body_pitch_x)
        prev_log_roll_y = float(centroidal_state_log.body_roll_y)

        # Sagittal controller input telemetry
        telemetry["sagittal_controller_input_pitch_x"].append(sagittal_controller_input_pitch_x)
        telemetry["sagittal_controller_input_pitch_rate_x"].append(sagittal_controller_input_pitch_rate_x)
        telemetry["sagittal_controller_input_cp_y"].append(sagittal_controller_input_cp_y)
        telemetry["sagittal_controller_input_com_y"].append(sagittal_controller_input_com_y)
        telemetry["sagittal_controller_input_com_vy"].append(sagittal_controller_input_com_vy)
        telemetry["sagittal_position_error_m"].append(sagittal_diag.get("sagittal_position_error_m", 0.0))
        telemetry["sagittal_velocity_m_s"].append(sagittal_diag.get("sagittal_velocity_m_s", 0.0))
        telemetry["support_position_velocity_m_s"].append(sagittal_diag.get("support_position_velocity_m_s", 0.0))
        telemetry["tau_position"].append(sagittal_diag.get("tau_position", 0.0))
        telemetry["tau_position_raw"].append(sagittal_diag.get("tau_position_raw", 0.0))
        telemetry["position_integral_error"].append(sagittal_diag.get("position_integral_error", 0.0))
        telemetry["tau_position_integral"].append(sagittal_diag.get("tau_position_integral", 0.0))
        telemetry["integral_active"].append(sagittal_diag.get("integral_active", False))
        telemetry["integral_gate_reason"].append(sagittal_diag.get("integral_gate_reason", "disabled"))
        telemetry["integral_saturation_flag"].append(sagittal_diag.get("integral_saturation_flag", False))
        telemetry["tau_position_p"].append(sagittal_diag.get("tau_position_p", 0.0))
        telemetry["tau_position_i"].append(sagittal_diag.get("tau_position_i", 0.0))
        telemetry["tau_position_total"].append(sagittal_diag.get("tau_position_total", 0.0))
        telemetry["tau_position_clipped"].append(sagittal_diag.get("tau_position_clipped", 0.0))
        telemetry["tau_support_velocity"].append(sagittal_diag.get("tau_support_velocity", 0.0))
        telemetry["tau_pitch"].append(sagittal_diag.get("tau_pitch", 0.0))
        telemetry["tau_pitch_raw"].append(sagittal_diag.get("tau_pitch_raw", sagittal_diag.get("tau_pitch", 0.0)))
        telemetry["tau_pitch_scheduled"].append(sagittal_diag.get("tau_pitch_scheduled", sagittal_diag.get("tau_pitch", 0.0)))
        telemetry["tau_pitch_clipped"].append(sagittal_diag.get("tau_pitch_clipped", sagittal_diag.get("tau_pitch", 0.0)))
        telemetry["tau_pitch_to_position_ratio"].append(sagittal_diag.get("tau_pitch_to_position_ratio", 0.0))
        telemetry["sagittal_schedule_profile"].append(sagittal_diag.get("sagittal_schedule_profile", "baseline"))
        telemetry["high_height_schedule_active"].append(sagittal_diag.get("high_height_schedule_active", False))
        telemetry["effective_max_position_tau"].append(sagittal_diag.get("effective_max_position_tau", sagittal_diag.get("max_position_tau", 0.0)))
        telemetry["effective_pitch_scale"].append(sagittal_diag.get("effective_pitch_scale", 1.0))
        telemetry["effective_pitch_tau_cap"].append(sagittal_diag.get("effective_pitch_tau_cap", "none"))
        telemetry["effective_velocity_damping_scale"].append(sagittal_diag.get("effective_velocity_damping_scale", 1.0))
        telemetry["effective_support_velocity_scale"].append(sagittal_diag.get("effective_support_velocity_scale", 1.0))
        # Phase 6 continuous schedule telemetry
        telemetry["low_height_sagittal_schedule_active"].append(sagittal_diag.get("low_height_sagittal_schedule_active", False))
        telemetry["effective_k_position"].append(sagittal_diag.get("effective_k_position", sagittal_diag.get("k_position", 40.0)))
        telemetry["effective_k_velocity"].append(sagittal_diag.get("effective_k_velocity", 15.0))
        telemetry["sagittal_schedule_height_reference_m"].append(sagittal_diag.get("schedule_height_reference_m", 0.4))
        telemetry["sagittal_schedule_height_source"].append(sagittal_diag.get("schedule_height_source", "unknown"))
        telemetry["sagittal_schedule_u"].append(sagittal_diag.get("k_position_schedule_u", 0.0))
        telemetry["sagittal_schedule_smoothstep"].append(sagittal_diag.get("k_position_schedule_smoothstep", 0.0))
        telemetry["tau_pitch_rate"].append(sagittal_diag.get("tau_pitch_rate", 0.0))
        telemetry["tau_pitch_rate_raw_signal"].append(sagittal_diag.get("tau_pitch_rate_raw_signal", 0.0))
        telemetry["tau_pitch_rate_filtered_signal"].append(sagittal_diag.get("tau_pitch_rate_filtered_signal", 0.0))
        telemetry["pitch_rate_raw_rad_s"].append(sagittal_diag.get("pitch_rate_raw", 0.0))
        telemetry["pitch_rate_notched_rad_s"].append(sagittal_diag.get("pitch_rate_notched", 0.0))
        telemetry["pitch_rate_effective_rad_s"].append(sagittal_diag.get("pitch_rate_effective", 0.0))
        telemetry["wip_notch_height_gate"].append(sagittal_diag.get("wip_notch_height_gate", 0.0))
        telemetry["wip_notch_filter_valid"].append(sagittal_diag.get("wip_notch_filter_valid", False))
        telemetry["dynamic_height_active"].append(dynamic_height_active)
        telemetry["dynamic_height_target_m"].append(dynamic_height_target_m if dynamic_height_active else 0.0)
        telemetry["notch_height_gate_from_traj"].append(dynamic_height_notch_gate if dynamic_height_active else 0.0)
        telemetry["tau_sagittal_velocity"].append(sagittal_diag.get("tau_sagittal_velocity", 0.0))
        telemetry["tau_wheel_velocity_left"].append(sagittal_diag.get("tau_wheel_velocity_left", 0.0))
        telemetry["tau_wheel_velocity_right"].append(sagittal_diag.get("tau_wheel_velocity_right", 0.0))
        telemetry["max_position_tau"].append(sagittal_diag.get("max_position_tau", 0.0))
        telemetry["tau_position_saturation_flag"].append(sagittal_diag.get("tau_position_saturation_flag", False))
        telemetry["tau_position_saturation_reason"].append(sagittal_diag.get("tau_position_saturation_reason", "none"))
        telemetry["tau_balance_before_position"].append(sagittal_diag.get("tau_balance_before_position", 0.0))
        telemetry["tau_position_budget_available"].append(sagittal_diag.get("tau_position_budget_available", 0.0))
        telemetry["tau_position_budget_allowed"].append(sagittal_diag.get("tau_position_budget_allowed", 0.0))
        telemetry["tau_position_budget_cap"].append(sagittal_diag.get("tau_position_budget_cap", 0.0))
        telemetry["pitch_reserve_tau"].append(sagittal_diag.get("pitch_reserve_tau", 0.0))
        telemetry["tau_pitch_reserve_applied"].append(sagittal_diag.get("tau_pitch_reserve_applied", 0.0))
        telemetry["enable_torque_budget_aware_position"].append(sagittal_diag.get("enable_torque_budget_aware_position", False))
        telemetry["tau_position_lower_bound"].append(sagittal_diag.get("tau_position_lower_bound", 0.0))
        telemetry["tau_position_upper_bound"].append(sagittal_diag.get("tau_position_upper_bound", 0.0))
        telemetry["tau_position_total_bound_clipped"].append(sagittal_diag.get("tau_position_total_bound_clipped", False))
        telemetry["position_authority_mode"].append(sagittal_diag.get("position_authority_mode", "none"))
        telemetry["position_authority_reason"].append(sagittal_diag.get("position_authority_reason", "none"))
        telemetry["tau_total_unclipped"].append(sagittal_diag.get("tau_total_unclipped", 0.0))
        telemetry["tau_total_clipped"].append(sagittal_diag.get("tau_total_clipped", 0.0))
        telemetry["tau_total_before_final_clip"].append(sagittal_diag.get("tau_total_before_final_clip", 0.0))
        telemetry["tau_total_after_final_clip"].append(sagittal_diag.get("tau_total_after_final_clip", 0.0))
        telemetry["final_wheel_torque_margin"].append(sagittal_diag.get("final_wheel_torque_margin", 0.0))
        telemetry["k_support_velocity"].append(sagittal_diag.get("k_support_velocity", 0.0))
        telemetry["support_position_error_m"].append(sagittal_diag.get("support_position_error_m", 0.0))
        telemetry["com_position_error_sagittal_m"].append(sagittal_diag.get("com_position_error_sagittal_m", 0.0))
        telemetry["pitch_x_ref_rad"].append(sagittal_diag.get("pitch_x_ref_rad", 0.0))
        telemetry["pitch_x_error_rad"].append(sagittal_diag.get("pitch_x_error_rad", 0.0))

        # Phase B support-position outer-loop telemetry (zeros / "disabled" for legacy profiles)
        telemetry["outer_loop_active"].append(sagittal_diag.get("outer_loop_active", False))
        telemetry["outer_loop_support_error_m"].append(sagittal_diag.get("outer_loop_support_error_m", 0.0))
        telemetry["outer_loop_support_error_rate_mps"].append(sagittal_diag.get("outer_loop_support_error_rate_mps", 0.0))
        telemetry["outer_loop_pitch_ref_dynamic_deg"].append(sagittal_diag.get("outer_loop_pitch_ref_dynamic_deg", 0.0))
        telemetry["outer_loop_pitch_ref_total_deg"].append(sagittal_diag.get("outer_loop_pitch_ref_total_deg", 0.0))
        telemetry["outer_loop_pitch_ref_limited_deg"].append(sagittal_diag.get("outer_loop_pitch_ref_limited_deg", 0.0))
        telemetry["outer_loop_pitch_ref_rate_limited_deg"].append(sagittal_diag.get("outer_loop_pitch_ref_rate_limited_deg", 0.0))
        telemetry["outer_loop_integral_m_s"].append(sagittal_diag.get("outer_loop_integral_m_s", 0.0))
        telemetry["outer_loop_gate_pass"].append(sagittal_diag.get("outer_loop_gate_pass", False))
        telemetry["outer_loop_block_reason"].append(sagittal_diag.get("outer_loop_block_reason", "disabled"))
        telemetry["outer_loop_sign_selected"].append(sagittal_diag.get("outer_loop_sign_selected", "none"))
        telemetry["support_outer_loop_height_scale"].append(sagittal_diag.get("support_outer_loop_height_scale", 0.0))
        telemetry["support_outer_loop_kp_effective"].append(sagittal_diag.get("support_outer_loop_kp_effective", 0.0))
        telemetry["support_outer_loop_kd_effective"].append(sagittal_diag.get("support_outer_loop_kd_effective", 0.0))
        telemetry["support_outer_loop_pitch_ref_offset_deg"].append(sagittal_diag.get("support_outer_loop_pitch_ref_offset_deg", 0.0))
        telemetry["support_outer_loop_pitch_ref_contrib"].append(sagittal_diag.get("support_outer_loop_pitch_ref_contrib", 0.0))
        telemetry["support_outer_loop_cap_active"].append(sagittal_diag.get("support_outer_loop_cap_active", False))
        telemetry["pitch_ref_offset_scheduled_deg"].append(sagittal_diag.get("pitch_ref_offset_scheduled_deg", 0.0))
        telemetry["pitch_ref_total_after_outer_loop_deg"].append(sagittal_diag.get("pitch_ref_total_after_outer_loop_deg", 0.0))
        telemetry["pitch_x_error_after_outer_loop_rad"].append(sagittal_diag.get("pitch_x_error_after_outer_loop_rad", 0.0))
        # Calibrated outer-loop telemetry (Phase B calibration)
        telemetry.setdefault("calibrated_outer_loop_active", []).append(sagittal_diag.get("calibrated_outer_loop_active", False))
        telemetry.setdefault("calibrated_function_profile_name", []).append(sagittal_diag.get("calibrated_function_profile_name", ""))
        telemetry.setdefault("calibrated_height_m", []).append(sagittal_diag.get("calibrated_height_m", 0.0))
        telemetry.setdefault("calibrated_kp_deg_per_m", []).append(sagittal_diag.get("calibrated_kp_deg_per_m", 0.0))
        telemetry.setdefault("calibrated_kd_deg_per_mps", []).append(sagittal_diag.get("calibrated_kd_deg_per_mps", 0.0))
        telemetry.setdefault("calibrated_ki_deg_per_m_s", []).append(sagittal_diag.get("calibrated_ki_deg_per_m_s", 0.0))
        telemetry.setdefault("calibrated_theta_ref_max_deg", []).append(sagittal_diag.get("calibrated_theta_ref_max_deg", 0.0))
        telemetry.setdefault("calibrated_deadband_m", []).append(sagittal_diag.get("calibrated_deadband_m", 0.0))
        telemetry.setdefault("calibrated_rate_limit_deg_per_step", []).append(sagittal_diag.get("calibrated_rate_limit_deg_per_step", 0.0))
        telemetry.setdefault("calibrated_lowpass_alpha", []).append(sagittal_diag.get("calibrated_lowpass_alpha", 0.0))
        telemetry.setdefault("calibrated_integral_active", []).append(sagittal_diag.get("calibrated_integral_active", False))
        telemetry.setdefault("calibrated_integral_value", []).append(sagittal_diag.get("calibrated_integral_value", 0.0))
        # Physics-based equilibrium feedforward telemetry (Phase D)
        telemetry.setdefault("physics_ff_enabled", []).append(sagittal_diag.get("physics_ff_enabled", False))
        telemetry.setdefault("physics_ff_height_m", []).append(sagittal_diag.get("physics_ff_height_m", 0.0))
        telemetry.setdefault("physics_ff_tau_eq_each_wheel_nm", []).append(sagittal_diag.get("physics_ff_tau_eq_each_wheel_nm", 0.0))
        telemetry.setdefault("physics_ff_pitch_eq_no_off_deg", []).append(sagittal_diag.get("physics_ff_pitch_eq_no_off_deg", 0.0))
        telemetry.setdefault("physics_ff_function_version", []).append(sagittal_diag.get("physics_ff_function_version", ""))
        telemetry.setdefault("physics_ff_clamped", []).append(sagittal_diag.get("physics_ff_clamped", False))
        telemetry.setdefault("physics_ff_applied_each_wheel_nm", []).append(sagittal_diag.get("physics_ff_applied_each_wheel_nm", 0.0))
        telemetry.setdefault("physics_ff_final_wheel_tau_with_ff", []).append(sagittal_diag.get("physics_ff_final_wheel_tau_with_ff", 0.0))
        telemetry.setdefault("physics_ff_final_wheel_tau_without_ff", []).append(sagittal_diag.get("physics_ff_final_wheel_tau_without_ff", 0.0))
        telemetry.setdefault("physics_ff_active_this_step", []).append(sagittal_diag.get("physics_ff_active_this_step", False))
        telemetry.setdefault("empirical_pitch_ref_offset_disabled", []).append(sagittal_diag.get("empirical_pitch_ref_offset_disabled", False))
        telemetry.setdefault("physics_equivalent_pitch_ref_deg", []).append(sagittal_diag.get("physics_equivalent_pitch_ref_deg", 0.0))

        # Auto-forward all K1 augmented telemetry fields from controller diagnostics to CSV
        for _k1_field in K1_AUGMENTED_TELEMETRY_FIELDS:
            telemetry.setdefault(_k1_field, []).append(sagittal_diag.get(_k1_field, 0.0))

        # Capture gate telemetry (velocity-damped controller only)
        telemetry["capture_gate_enabled"].append(sagittal_diag.get("capture_gate_enabled", False))
        telemetry["capture_gate_required_direction"].append(sagittal_diag.get("capture_gate_required_direction", 0.0))
        telemetry["capture_gate_tau_position_direction"].append(sagittal_diag.get("capture_gate_tau_position_direction", 0.0))
        telemetry["capture_gate_position_opposes_capture"].append(sagittal_diag.get("capture_gate_position_opposes_capture", False))
        telemetry["capture_gate_factor"].append(sagittal_diag.get("capture_gate_factor", 1.0))
        telemetry["capture_gate_active"].append(sagittal_diag.get("capture_gate_active", False))
        telemetry["capture_gate_reason"].append(sagittal_diag.get("capture_gate_reason", "N/A"))
        telemetry["capture_gate_pitch_reversal"].append(sagittal_diag.get("capture_gate_pitch_reversal", False))
        telemetry["capture_gate_capture_recovery"].append(sagittal_diag.get("capture_gate_capture_recovery", False))
        telemetry["capture_gate_tau_position_gated"].append(sagittal_diag.get("capture_gate_tau_position_gated", 0.0))
        telemetry["capture_gate_cp_relative_to_support_m"].append(sagittal_diag.get("capture_gate_cp_relative_to_support_m", 0.0))
        telemetry["capture_gate_com_support_error_m"].append(sagittal_diag.get("capture_gate_com_support_error_m", 0.0))

        # Pitch-aware position scaling telemetry (if enabled)
        telemetry["pitch_aware_position_scaling_enabled"].append(sagittal_diag.get("pitch_aware_position_scaling_enabled", False))
        telemetry["pitch_aware_position_scale"].append(sagittal_diag.get("pitch_aware_position_scale", 1.0))
        telemetry["pitch_aware_active"].append(sagittal_diag.get("pitch_aware_active", False))
        telemetry["pitch_soft_start"].append(sagittal_diag.get("pitch_soft_start", 0.06))
        telemetry["pitch_hard_limit"].append(sagittal_diag.get("pitch_hard_limit", 0.10))
        telemetry["min_pitch_scale"].append(sagittal_diag.get("min_pitch_scale", 0.7))
        telemetry["tau_position_before_pitch_scale"].append(sagittal_diag.get("tau_position_before_pitch_scale", 0.0))
        telemetry["tau_position_after_pitch_scale"].append(sagittal_diag.get("tau_position_after_pitch_scale", 0.0))

        # Phase-aware recenter telemetry (F1_strategy - if enabled)
        telemetry["phase_recenter_enabled"].append(sagittal_diag.get("phase_recenter_enabled", False))
        telemetry["phase_recenter_active"].append(sagittal_diag.get("phase_recenter_active", False))
        telemetry["phase_recenter_gate_safe"].append(sagittal_diag.get("phase_recenter_gate_safe", False))
        telemetry["phase_recenter_signed_error_m"].append(sagittal_diag.get("phase_recenter_signed_error_m", 0.0))
        telemetry["phase_recenter_raw_tau"].append(sagittal_diag.get("phase_recenter_raw_tau", 0.0))
        telemetry["phase_recenter_tau"].append(sagittal_diag.get("phase_recenter_tau", 0.0))
        telemetry["phase_recenter_tau_clipped"].append(sagittal_diag.get("phase_recenter_tau_clipped", 0.0))
        telemetry["phase_recenter_smooth_alpha"].append(sagittal_diag.get("phase_recenter_smooth_alpha", 0.0))
        telemetry["phase_recenter_gate_reason"].append(str(sagittal_diag.get("phase_recenter_gate_reason", "unknown")))
        telemetry["phase_recenter_pitch_safe"].append(sagittal_diag.get("phase_recenter_pitch_safe", False))
        telemetry["phase_recenter_pitch_danger"].append(sagittal_diag.get("phase_recenter_pitch_danger", False))
        telemetry["phase_recenter_contact_safe"].append(sagittal_diag.get("phase_recenter_contact_safe", True))
        telemetry["phase_recenter_height_safe"].append(sagittal_diag.get("phase_recenter_height_safe", True))
        telemetry["phase_recenter_deadband_active"].append(sagittal_diag.get("phase_recenter_deadband_active", False))

        # Hysteresis recenter telemetry (F2_strategy - if enabled)
        telemetry["hysteresis_recenter_enabled"].append(sagittal_diag.get("hysteresis_recenter_enabled", False))
        telemetry["hysteresis_recenter_state"].append(str(sagittal_diag.get("hysteresis_recenter_state", "NEUTRAL")))
        telemetry["hysteresis_recenter_state_id"].append(sagittal_diag.get("hysteresis_recenter_state_id", 0))
        telemetry["hysteresis_recenter_outer_enter_m"].append(sagittal_diag.get("hysteresis_recenter_outer_enter_m", 0.10))
        telemetry["hysteresis_recenter_exit_target_m"].append(sagittal_diag.get("hysteresis_recenter_exit_target_m", 0.00))
        telemetry["hysteresis_recenter_signed_error_m"].append(sagittal_diag.get("hysteresis_recenter_signed_error_m", 0.0))
        telemetry["hysteresis_recenter_target_error_m"].append(sagittal_diag.get("hysteresis_recenter_target_error_m", 0.0))
        telemetry["hysteresis_recenter_raw_tau"].append(sagittal_diag.get("hysteresis_recenter_raw_tau", 0.0))
        telemetry["hysteresis_recenter_tau"].append(sagittal_diag.get("hysteresis_recenter_tau", 0.0))
        telemetry["hysteresis_recenter_tau_clipped"].append(sagittal_diag.get("hysteresis_recenter_tau_clipped", 0.0))
        telemetry["hysteresis_recenter_active"].append(sagittal_diag.get("hysteresis_recenter_active", False))
        telemetry["hysteresis_recenter_state_entry_count"].append(sagittal_diag.get("hysteresis_recenter_state_entry_count", 0))
        telemetry["hysteresis_recenter_state_exit_count"].append(sagittal_diag.get("hysteresis_recenter_state_exit_count", 0))
        telemetry["hysteresis_recenter_safety_override"].append(sagittal_diag.get("hysteresis_recenter_safety_override", False))
        telemetry["hysteresis_recenter_gate_reason"].append(str(sagittal_diag.get("hysteresis_recenter_gate_reason", "unknown")))

        # APCR (Active Pitch Crossing Recovery) telemetry
        # Captured from sagittal_diag generated by SagittalVelocityDampedBalanceController.compute()
        # Added 2026-06-08 to fix APCR telemetry validation gap
        telemetry["active_pitch_crossing_enabled"].append(sagittal_diag.get("active_pitch_crossing_enabled", False))
        telemetry["active_pitch_crossing_recovery_gate_mode"].append(sagittal_diag.get("active_pitch_crossing_recovery_gate_mode", False))
        telemetry["active_pitch_crossing_state"].append(str(sagittal_diag.get("active_pitch_crossing_state", "DISABLED")))
        telemetry["active_pitch_crossing_state_id"].append(int(sagittal_diag.get("active_pitch_crossing_state_id", 0)))
        telemetry["active_pitch_crossing_active"].append(sagittal_diag.get("active_pitch_crossing_active", False))
        telemetry["active_pitch_crossing_signed_error_m"].append(float(sagittal_diag.get("active_pitch_crossing_signed_error_m", 0.0)))
        telemetry["active_pitch_crossing_pitch_x"].append(float(sagittal_diag.get("active_pitch_crossing_pitch_x", 0.0)))
        telemetry["active_pitch_crossing_pitch_rate"].append(float(sagittal_diag.get("active_pitch_crossing_pitch_rate", 0.0)))
        telemetry["active_pitch_crossing_raw_tau"].append(float(sagittal_diag.get("active_pitch_crossing_raw_tau", 0.0)))
        telemetry["active_pitch_crossing_tau"].append(float(sagittal_diag.get("active_pitch_crossing_tau", 0.0)))
        telemetry["active_pitch_crossing_tau_clipped"].append(float(sagittal_diag.get("active_pitch_crossing_tau_clipped", 0.0)))
        telemetry["active_pitch_crossing_target_direction"].append(str(sagittal_diag.get("active_pitch_crossing_target_direction", "none")))
        telemetry["active_pitch_crossing_outer_enter_m"].append(float(sagittal_diag.get("active_pitch_crossing_outer_enter_m", 0.0)))
        telemetry["active_pitch_crossing_inner_exit_m"].append(float(sagittal_diag.get("active_pitch_crossing_inner_exit_m", 0.0)))
        telemetry["active_pitch_crossing_pitch_hard_stop_rad"].append(float(sagittal_diag.get("active_pitch_crossing_pitch_hard_stop_rad", 0.0)))
        telemetry["active_pitch_crossing_hard_safety_gate"].append(sagittal_diag.get("active_pitch_crossing_hard_safety_gate", False))
        telemetry["active_pitch_crossing_recovery_gate"].append(sagittal_diag.get("active_pitch_crossing_recovery_gate", False))
        telemetry["active_pitch_crossing_gate_reason"].append(str(sagittal_diag.get("active_pitch_crossing_gate_reason", "unknown")))
        telemetry["active_pitch_crossing_state_entry_count"].append(int(sagittal_diag.get("active_pitch_crossing_state_entry_count", 0)))
        telemetry["active_pitch_crossing_state_exit_count"].append(int(sagittal_diag.get("active_pitch_crossing_state_exit_count", 0)))
        telemetry["active_pitch_crossing_safety_override"].append(sagittal_diag.get("active_pitch_crossing_safety_override", False))
        telemetry["active_pitch_crossing_contact_safe"].append(sagittal_diag.get("active_pitch_crossing_contact_safe", True))
        telemetry["active_pitch_crossing_height_safe"].append(sagittal_diag.get("active_pitch_crossing_height_safe", True))
        telemetry["active_pitch_crossing_roll_safe"].append(sagittal_diag.get("active_pitch_crossing_roll_safe", True))
        telemetry["active_pitch_crossing_pitch_safe"].append(sagittal_diag.get("active_pitch_crossing_pitch_safe", True))
        telemetry["active_pitch_crossing_pitch_danger"].append(sagittal_diag.get("active_pitch_crossing_pitch_danger", False))
        telemetry["active_pitch_crossing_max_tau"].append(float(sagittal_diag.get("active_pitch_crossing_max_tau", 0.0)))
        telemetry["active_pitch_crossing_smooth_alpha"].append(float(sagittal_diag.get("active_pitch_crossing_smooth_alpha", 0.0)))
        # APCR1i hysteresis recenter telemetry
        telemetry["active_pitch_crossing_hysteresis_enabled"].append(sagittal_diag.get("active_pitch_crossing_hysteresis_enabled", False))
        telemetry["active_pitch_crossing_hysteresis_state"].append(str(sagittal_diag.get("active_pitch_crossing_hysteresis_state", "NEUTRAL")))
        telemetry["active_pitch_crossing_hysteresis_state_id"].append(int(sagittal_diag.get("active_pitch_crossing_hysteresis_state_id", 0)))
        telemetry["active_pitch_crossing_hysteresis_entry_e"].append(float(sagittal_diag.get("active_pitch_crossing_hysteresis_entry_e", 0.0)))
        telemetry["active_pitch_crossing_hysteresis_exit_e"].append(float(sagittal_diag.get("active_pitch_crossing_hysteresis_exit_e", 0.0)))
        telemetry["active_pitch_crossing_hysteresis_entry_count"].append(int(sagittal_diag.get("active_pitch_crossing_hysteresis_entry_count", 0)))
        telemetry["active_pitch_crossing_hysteresis_exit_count"].append(int(sagittal_diag.get("active_pitch_crossing_hysteresis_exit_count", 0)))
        telemetry["active_pitch_crossing_hysteresis_inner_exit_m"].append(float(sagittal_diag.get("active_pitch_crossing_hysteresis_inner_exit_m", 0.0)))
        telemetry["active_pitch_crossing_hysteresis_opposite_release_m"].append(float(sagittal_diag.get("active_pitch_crossing_hysteresis_opposite_release_m", 0.0)))
        telemetry["active_pitch_crossing_hysteresis_emergency_active"].append(sagittal_diag.get("active_pitch_crossing_hysteresis_emergency_active", False))
        telemetry["final_wheel_tau_with_apc"].append(float(sagittal_diag.get("final_wheel_tau_with_apc", 0.0)))
        telemetry["final_wheel_tau_without_apc"].append(float(sagittal_diag.get("final_wheel_tau_without_apc", 0.0)))

        # APCR1l pitch suppression telemetry
        telemetry["apcr1l_pitch_suppress_active"].append(sagittal_diag.get("apcr1l_pitch_suppress_active", False))
        telemetry["apcr1l_recenter_state"].append(str(sagittal_diag.get("apcr1l_recenter_state", "NEUTRAL")))
        telemetry["apcr1l_tau_pitch_before_suppress"].append(float(sagittal_diag.get("apcr1l_tau_pitch_before_suppress", 0.0)))

        # APCR1m conditional pitch blend telemetry
        telemetry["apcr1m_pitch_blend_active"].append(sagittal_diag.get("apcr1m_pitch_blend_active", False))
        telemetry["apcr1m_pitch_blend_scale"].append(float(sagittal_diag.get("apcr1m_pitch_blend_scale", 1.0)))
        telemetry["apcr1m_pitch_blend_block_reason"].append(str(sagittal_diag.get("apcr1m_pitch_blend_block_reason", "none")))
        telemetry["apcr1m_tau_pitch_before_blend"].append(float(sagittal_diag.get("apcr1m_tau_pitch_before_blend", 0.0)))
        telemetry["apcr1m_tau_pitch_after_blend"].append(float(sagittal_diag.get("apcr1m_tau_pitch_after_blend", 0.0)))
        telemetry["apcr1m_startup_guard_active"].append(sagittal_diag.get("apcr1m_startup_guard_active", False))
        telemetry["apcr1m_recenter_active"].append(sagittal_diag.get("apcr1m_recenter_active", False))
        telemetry["apcr1m_pitch_safe"].append(sagittal_diag.get("apcr1m_pitch_safe", True))
        telemetry["apcr1m_height_safe"].append(sagittal_diag.get("apcr1m_height_safe", True))
        telemetry["apcr1m_contact_safe"].append(sagittal_diag.get("apcr1m_contact_safe", True))
        telemetry["apcr1m_roll_safe"].append(sagittal_diag.get("apcr1m_roll_safe", True))
        telemetry["apcr1m_pitch_rate_safe"].append(sagittal_diag.get("apcr1m_pitch_rate_safe", True))

        # APCR1n recenter priority telemetry
        telemetry["apcr1n_recenter_priority_active"].append(sagittal_diag.get("apcr1n_recenter_priority_active", False))
        telemetry["apcr1n_startup_guard_active"].append(sagittal_diag.get("apcr1n_startup_guard_active", True))
        telemetry["apcr1n_wheel_damping_override_active"].append(sagittal_diag.get("apcr1n_wheel_damping_override_active", False))
        telemetry["apcr1n_wheel_damping_scale"].append(float(sagittal_diag.get("apcr1n_wheel_damping_scale", 1.0)))
        telemetry["apcr1n_wheel_damping_before"].append(float(sagittal_diag.get("apcr1n_wheel_damping_before", 0.0)))
        telemetry["apcr1n_wheel_damping_after"].append(float(sagittal_diag.get("apcr1n_wheel_damping_after", 0.0)))
        telemetry["apcr1n_wheel_damping_fights_drift"].append(sagittal_diag.get("apcr1n_wheel_damping_fights_drift", False))
        telemetry["apcr1n_position_cap_boost_active"].append(sagittal_diag.get("apcr1n_position_cap_boost_active", False))
        telemetry["apcr1n_position_cap_current"].append(float(sagittal_diag.get("apcr1n_position_cap_current", 3.0)))
        telemetry["apcr1n_tau_position_raw"].append(float(sagittal_diag.get("apcr1n_tau_position_raw", 0.0)))
        telemetry["apcr1n_tau_position_after_cap"].append(float(sagittal_diag.get("apcr1n_tau_position_after_cap", 0.0)))
        telemetry["apcr1n_position_saturated"].append(sagittal_diag.get("apcr1n_position_saturated", False))
        telemetry["apcr1n_safety_gate_pass"].append(sagittal_diag.get("apcr1n_safety_gate_pass", True))
        telemetry["apcr1n_final_torque_direction_correct"].append(sagittal_diag.get("apcr1n_final_torque_direction_correct", True))
        telemetry["apcr1n_final_torque_fights_drift"].append(sagittal_diag.get("apcr1n_final_torque_fights_drift", False))
        telemetry["apcr1n_physical_drift_column_used"].append(str(sagittal_diag.get("apcr1n_physical_drift_column_used", "unknown")))

        # APCR1nD direct support recenter telemetry
        telemetry["apcr1nd_direct_recenter_priority_active"].append(bool(sagittal_diag.get("apcr1nd_direct_recenter_priority_active", False)))
        telemetry["apcr1nd_direct_recenter_eligible"].append(bool(sagittal_diag.get("apcr1nd_direct_recenter_eligible", False)))
        telemetry["apcr1nd_direct_recenter_block_reason"].append(str(sagittal_diag.get("apcr1nd_direct_recenter_block_reason", "")))
        telemetry["apcr1nd_moving_away"].append(bool(sagittal_diag.get("apcr1nd_moving_away", False)))
        telemetry["apcr1nd_abs_error"].append(float(sagittal_diag.get("apcr1nd_abs_error", 0.0)))
        telemetry["apcr1nd_error_rate"].append(float(sagittal_diag.get("apcr1nd_error_rate", 0.0)))

        # Append all remaining sagittal diagnostics fields (including tuned telemetry)
        # Use setdefault to dynamically create columns for new fields
        if is_balance_core_mode(args):
            for key, value in sagittal_diag.items():
                # Skip fields already explicitly handled above
                if key not in telemetry or len(telemetry[key]) < step:
                    if isinstance(value, (int, float, bool, str)):
                        telemetry.setdefault(key, []).append(value)
                    else:
                        # Convert other types to string for CSV compatibility
                        telemetry.setdefault(key, []).append(str(value))

        # Pitch rate consistency estimator telemetry (velocity-damped controller only)
        telemetry["sagittal_axis_y_initial"].append(float(sagittal_diag.get("sagittal_axis_y_initial", sagittal_axis_xy_initial[1])))
        telemetry["raw_com_vx"].append(float(sagittal_diag.get("raw_com_vx", centroidal_state_control.com_vel[0])))
        telemetry["raw_com_vy"].append(float(sagittal_diag.get("raw_com_vy", centroidal_state_control.com_vel[1])))
        telemetry["projected_sagittal_velocity_m_s"].append(float(sagittal_diag.get("projected_sagittal_velocity_m_s", 0.0)))
        telemetry["actual_sagittal_velocity_passed_to_controller_m_s"].append(float(sagittal_diag.get("actual_sagittal_velocity_passed_to_controller_m_s", sagittal_diag.get("sagittal_velocity_m_s", 0.0))))
        telemetry["tau_wheel_total_raw_left"].append(float(sagittal_diag.get("tau_left", 0.0)))
        telemetry["tau_wheel_total_raw_right"].append(float(sagittal_diag.get("tau_right", 0.0)))
        telemetry["tau_wheel_total_clipped_left"].append(float(tau_total_clipped[4]))
        telemetry["tau_wheel_total_clipped_right"].append(float(tau_total_clipped[9]))
        telemetry["wheel_torque_margin_left"].append(float(torque_limit[4] - abs(float(tau_total_raw[4]))))
        telemetry["wheel_torque_margin_right"].append(float(torque_limit[9] - abs(float(tau_total_raw[9]))))
        telemetry["wheel_torque_rate_limit_active_left"].append(bool(rate_flags_vec[4]))
        telemetry["wheel_torque_rate_limit_active_right"].append(bool(rate_flags_vec[9]))

        l_hip_yaw_pos = float(joint_pos[1])
        r_hip_yaw_pos = float(joint_pos[6])
        l_hip_yaw_ref = float(equilibrium_joint_pos[1])
        r_hip_yaw_ref = float(equilibrium_joint_pos[6])
        l_hip_yaw_error = l_hip_yaw_ref - l_hip_yaw_pos
        r_hip_yaw_error = r_hip_yaw_ref - r_hip_yaw_pos
        l_hip_yaw_vel = float(joint_vel[1])
        r_hip_yaw_vel = float(joint_vel[6])
        l_hip_yaw_tau_raw = l_hip_yaw_error * balance_core_controllers["shape_posture"].kp_hip_yaw - l_hip_yaw_vel * balance_core_controllers["shape_posture"].kd_hip_yaw if is_balance_core_mode(args) else 0.0
        r_hip_yaw_tau_raw = r_hip_yaw_error * balance_core_controllers["shape_posture"].kp_hip_yaw - r_hip_yaw_vel * balance_core_controllers["shape_posture"].kd_hip_yaw if is_balance_core_mode(args) else 0.0
        telemetry["l_hip_yaw_pos"].append(l_hip_yaw_pos)
        telemetry["r_hip_yaw_pos"].append(r_hip_yaw_pos)
        telemetry["l_hip_yaw_ref"].append(l_hip_yaw_ref)
        telemetry["r_hip_yaw_ref"].append(r_hip_yaw_ref)
        telemetry["l_hip_yaw_error"].append(l_hip_yaw_error)
        telemetry["r_hip_yaw_error"].append(r_hip_yaw_error)
        telemetry["l_hip_yaw_vel"].append(l_hip_yaw_vel)
        telemetry["r_hip_yaw_vel"].append(r_hip_yaw_vel)
        telemetry["hip_yaw_error_rms"].append(float(np.sqrt(0.5 * (l_hip_yaw_error**2 + r_hip_yaw_error**2))))
        telemetry["l_hip_yaw_tau_shape_raw"].append(l_hip_yaw_tau_raw)
        telemetry["r_hip_yaw_tau_shape_raw"].append(r_hip_yaw_tau_raw)
        telemetry["l_hip_yaw_tau_shape_final"].append(float(tau_shape_posture[1]) if is_balance_core_mode(args) else 0.0)
        telemetry["r_hip_yaw_tau_shape_final"].append(float(tau_shape_posture[6]) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_torque_sign_correct_left"].append(abs(l_hip_yaw_error) < 1e-9 or l_hip_yaw_error * (float(tau_shape_posture[1]) if is_balance_core_mode(args) else 0.0) >= 0.0)
        telemetry["hip_yaw_torque_sign_correct_right"].append(abs(r_hip_yaw_error) < 1e-9 or r_hip_yaw_error * (float(tau_shape_posture[6]) if is_balance_core_mode(args) else 0.0) >= 0.0)
        telemetry["hip_yaw_torque_saturation_flag_left"].append(bool(sat_flags_vec[1]))
        telemetry["hip_yaw_torque_saturation_flag_right"].append(bool(sat_flags_vec[6]))
        telemetry["hip_yaw_torque_margin_left"].append(float(torque_limit[1] - abs(float(tau_total_raw[1]))))
        telemetry["hip_yaw_torque_margin_right"].append(float(torque_limit[6] - abs(float(tau_total_raw[6]))))

        # HY-FF: Hip-yaw support-error feedforward compensation telemetry
        telemetry["hip_yaw_comp_active"].append(shape_diag.get("hip_yaw_comp_active", False) if is_balance_core_mode(args) else False)
        telemetry["hip_yaw_comp_height_gate"].append(shape_diag.get("hip_yaw_comp_height_gate", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_comp_support_error_m"].append(shape_diag.get("hip_yaw_comp_support_error_m", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_comp_tau_left"].append(shape_diag.get("hip_yaw_comp_tau_left", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_comp_tau_right"].append(shape_diag.get("hip_yaw_comp_tau_right", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_comp_tau_left_clipped"].append(shape_diag.get("hip_yaw_comp_tau_left_clipped", False) if is_balance_core_mode(args) else False)
        telemetry["hip_yaw_comp_tau_right_clipped"].append(shape_diag.get("hip_yaw_comp_tau_right_clipped", False) if is_balance_core_mode(args) else False)
        telemetry["hip_yaw_comp_sign"].append(shape_diag.get("hip_yaw_comp_sign", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_comp_k_support"].append(shape_diag.get("hip_yaw_comp_k_support", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_comp_tau_max"].append(shape_diag.get("hip_yaw_comp_tau_max", 0.0) if is_balance_core_mode(args) else 0.0)

        # HY2-DIV: Hip-yaw divergence damping telemetry
        telemetry["hip_yaw_div_enabled"].append(shape_diag.get("hip_yaw_div_enabled", False) if is_balance_core_mode(args) else False)
        telemetry["hip_yaw_div_gate_active"].append(shape_diag.get("hip_yaw_div_gate_active", False) if is_balance_core_mode(args) else False)
        telemetry["hip_yaw_div_active"].append(shape_diag.get("hip_yaw_div_active", False) if is_balance_core_mode(args) else False)
        telemetry["hip_yaw_div_height_gate"].append(shape_diag.get("hip_yaw_div_height_gate", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_div_effective_k"].append(shape_diag.get("hip_yaw_div_effective_k", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_div_effective_kd"].append(shape_diag.get("hip_yaw_div_effective_kd", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_div_effective_tau_max"].append(shape_diag.get("hip_yaw_div_effective_tau_max", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_div_left"].append(shape_diag.get("hip_yaw_div_left", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_div_right"].append(shape_diag.get("hip_yaw_div_right", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_div_left_clipped"].append(shape_diag.get("hip_yaw_div_left_clipped", False) if is_balance_core_mode(args) else False)
        telemetry["hip_yaw_div_right_clipped"].append(shape_diag.get("hip_yaw_div_right_clipped", False) if is_balance_core_mode(args) else False)
        telemetry["hip_yaw_div_k_divergence"].append(shape_diag.get("hip_yaw_div_k_divergence", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_div_k_divergence_rate"].append(shape_diag.get("hip_yaw_div_k_divergence_rate", 0.0) if is_balance_core_mode(args) else 0.0)
        # Mode-Based Hip-Yaw Divergence Controller telemetry (opt-in)
        telemetry["mode_hip_yaw_div_enabled"].append(bool(mode_hip_yaw_div_enabled))
        telemetry["mode_hip_yaw_div_kp"].append(float(getattr(args, "mode_hip_yaw_div_kp", 0.0)))
        telemetry["mode_hip_yaw_div_kd"].append(float(getattr(args, "mode_hip_yaw_div_kd", 0.0)))
        telemetry["mode_hip_yaw_div_max_torque"].append(float(getattr(args, "mode_hip_yaw_div_max_torque", 0.0)))
        telemetry["mode_hip_yaw_div_soft_limit_rad"].append(float(getattr(args, "mode_hip_yaw_div_soft_limit_rad", 0.0)))
        telemetry["mode_hip_yaw_div_soft_gain"].append(float(getattr(args, "mode_hip_yaw_div_soft_gain", 0.0)))
        telemetry["mode_hip_yaw_div_ref_source"].append(str(getattr(args, "mode_hip_yaw_div_ref_source", "target")))
        telemetry["mode_hip_yaw_div_height_gate"].append(float(mode_div_height_gate))
        telemetry["mode_hip_yaw_div_tau_left"].append(float(mode_div_tau_left))
        telemetry["mode_hip_yaw_div_tau_right"].append(float(mode_div_tau_right))
        telemetry["mode_hip_yaw_div_tau_left_raw"].append(float(mode_div_tau_left_raw))
        telemetry["mode_hip_yaw_div_tau_right_raw"].append(float(mode_div_tau_right_raw))
        telemetry["mode_hip_yaw_div_tau_left_sat"].append(bool(mode_div_tau_left_sat))
        telemetry["mode_hip_yaw_div_tau_right_sat"].append(bool(mode_div_tau_right_sat))
        telemetry["mode_hip_yaw_div_torque_margin_left"].append(
            float(args.mode_hip_yaw_div_max_torque) - abs(float(mode_div_tau_left_raw))
            if mode_hip_yaw_div_enabled else 0.0
        )
        telemetry["mode_hip_yaw_div_torque_margin_right"].append(
            float(args.mode_hip_yaw_div_max_torque) - abs(float(mode_div_tau_right_raw))
            if mode_hip_yaw_div_enabled else 0.0
        )
        telemetry["mode_hip_yaw_div_error"].append(float(mode_div_error))
        telemetry["mode_hip_yaw_div_rate"].append(float(mode_div_rate))
        telemetry["mode_hip_yaw_div_ref"].append(float(mode_div_ref))
        # Support-aware mode-div gating telemetry
        telemetry["mode_hip_yaw_div_support_gate_enabled"].append(
            bool(getattr(args, "mode_hip_yaw_div_support_enabled", False))
        )
        telemetry["mode_hip_yaw_div_support_error_m"].append(
            float(mode_div_support_error_val) if mode_hip_yaw_div_enabled else 0.0
        )
        telemetry["mode_hip_yaw_div_support_error_rate_mps"].append(
            float(mode_div_support_error_rate_val) if mode_hip_yaw_div_enabled else 0.0
        )
        telemetry["mode_hip_yaw_div_support_error_gate"].append(float(mode_div_support_error_gate))
        telemetry["mode_hip_yaw_div_support_rate_gate"].append(float(mode_div_support_rate_gate))
        telemetry["mode_hip_yaw_div_effective_support_gate"].append(float(mode_div_effective_support_gate))
        telemetry["mode_hip_yaw_div_combined_gate"].append(float(mode_div_combined_gate))
        telemetry["hip_yaw_mode_ownership_violation"].append(int(mode_ownership_violation))
        telemetry["hip_yaw_div_tau_max"].append(shape_diag.get("hip_yaw_div_tau_max", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_div_z_low"].append(shape_diag.get("hip_yaw_div_z_low", 0.0) if is_balance_core_mode(args) else 0.0)
        telemetry["hip_yaw_div_z_high"].append(shape_diag.get("hip_yaw_div_z_high", 0.0) if is_balance_core_mode(args) else 0.0)

        # HY-FF debug telemetry
        if is_balance_core_mode(args):
            telemetry["hy_ff_height_passed_to_shape"].append(hy_ff_height_input)
            telemetry["hy_ff_support_error_passed_to_shape"].append(hy_ff_support_error_input)
            telemetry["hy_ff_support_error_from_sagittal"].append(sagittal_diag.get("support_position_error_m", 0.0))
            telemetry["hy_ff_prev_support_error"].append(prev_support_error)
            telemetry["hy_ff_setup_target_com_z_m"].append(hy_ff_setup_target)
            telemetry["hy_ff_setup_achieved_com_z_m"].append(hy_ff_setup_achieved)
            telemetry["hy_ff_root_z_m"].append(hy_ff_root_z)
            telemetry["hy_ff_current_com_z_m"].append(hy_ff_current_com_z)
        else:
            telemetry["hy_ff_height_passed_to_shape"].append(0.0)
            telemetry["hy_ff_support_error_passed_to_shape"].append(0.0)
            telemetry["hy_ff_support_error_from_sagittal"].append(0.0)
            telemetry["hy_ff_prev_support_error"].append(0.0)
            telemetry["hy_ff_setup_target_com_z_m"].append(0.0)
            telemetry["hy_ff_setup_achieved_com_z_m"].append(0.0)
            telemetry["hy_ff_root_z_m"].append(0.0)
            telemetry["hy_ff_current_com_z_m"].append(0.0)

        # Yaw-position coupling diagnostic telemetry
        root_yaw_z = float(mj_data.qpos[6]) if len(mj_data.qpos) > 6 else 0.0  # qpos[6] = torso yaw (approximate)
        yaw_z = float(centroidal_state_control.body_yaw_z) if hasattr(centroidal_state_control, 'body_yaw_z') else 0.0
        yaw_error_eq = yaw_z - initial_yaw_z
        l_hip_yaw_err_val = l_hip_yaw_error
        r_hip_yaw_err_val = r_hip_yaw_error
        hip_yaw_asym = abs(l_hip_yaw_err_val + r_hip_yaw_err_val)
        hip_yaw_div = abs(l_hip_yaw_err_val - r_hip_yaw_err_val)
        telemetry["root_yaw_z_rad"].append(root_yaw_z)
        telemetry["yaw_z_rad"].append(yaw_z)
        telemetry["yaw_error_from_equilibrium_rad"].append(yaw_error_eq)
        telemetry["hip_yaw_asymmetry"].append(hip_yaw_asym)
        telemetry["hip_yaw_divergence"].append(hip_yaw_div)

        # Yaw-induced position error estimation
        # When hip yaw drifts by angle theta, the support center appears to shift
        # by approximately d * sin(theta) where d is half the axle-to-axle distance
        # in the sagittal direction. Use wheel separation as approximate d.
        l_wheel_pos = np.array([float(mj_data.xpos[l_wheel_body_id][i]) for i in range(3)])
        r_wheel_pos = np.array([float(mj_data.xpos[r_wheel_body_id][i]) for i in range(3)])
        wheel_sep_y = abs(l_wheel_pos[1] - r_wheel_pos[1])
        mean_yaw_error = 0.5 * (l_hip_yaw_err_val + r_hip_yaw_err_val)
        yaw_induced_x = wheel_sep_y * np.sin(mean_yaw_error)
        yaw_induced_y = wheel_sep_y * (1.0 - np.cos(mean_yaw_error))
        yaw_induced_norm = np.sqrt(yaw_induced_x**2 + yaw_induced_y**2)
        telemetry["yaw_induced_position_error_x_m"].append(float(yaw_induced_x))
        telemetry["yaw_induced_position_error_y_m"].append(float(yaw_induced_y))
        telemetry["yaw_induced_position_error_norm_m"].append(float(yaw_induced_norm))

        # Yaw-aware compensation telemetry — populated by profile system
        variant_name_for_telem = height_variant_setup.get("variant_name") if height_variant_setup else None
        fix_active = boundary_fix.is_active(variant_name_for_telem)
        yaw_aware_active = boundary_fix.uses_yaw_aware_compensation() and fix_active
        boundary_profile_name = boundary_fix.profile
        effective_kp_val = boundary_fix.get_effective_hip_yaw_kp(
            balance_core_controllers["shape_posture"].kp_hip_yaw, variant_name_for_telem
        ) if is_balance_core_mode(args) else 0.0
        effective_kd_val = boundary_fix.get_effective_hip_yaw_kd(
            balance_core_controllers["shape_posture"].kd_hip_yaw, variant_name_for_telem
        ) if is_balance_core_mode(args) else 0.0
        telemetry["yaw_aware_position_compensation_active"].append(yaw_aware_active)
        telemetry["yaw_aware_sagittal_error_compensated_m"].append(float(sagittal_diag.get("support_position_error_scaled_m", sagittal_diag.get("sagittal_position_error_m", 0.0))))
        telemetry["yaw_aware_lateral_error_compensated_m"].append(compensated_lateral_error if yaw_aware_active else 0.0)
        telemetry["effective_kp_hip_yaw"].append(float(effective_kp_val))
        telemetry["effective_kd_hip_yaw"].append(float(effective_kd_val))
        telemetry["hip_yaw_integral_active"].append(boundary_fix.uses_integral() and fix_active)
        telemetry["hip_yaw_integral_clamp"].append(1.0 if (boundary_fix.uses_integral() and boundary_fix.integral_error_left >= boundary_fix.integral_max) else 0.0)
        telemetry["hip_yaw_integral_error_left"].append(float(boundary_fix.integral_error_left))
        telemetry["hip_yaw_integral_error_right"].append(float(boundary_fix.integral_error_right))
        telemetry["hip_yaw_bias_tau_left"].append(float(boundary_fix.bias_tau_left))
        telemetry["hip_yaw_bias_tau_right"].append(float(boundary_fix.bias_tau_right))
        telemetry["hip_yaw_bias_active"].append(boundary_fix.uses_integral() and fix_active)
        telemetry["tau_position_yaw_compensated_raw"].append(float(sagittal_diag.get("tau_position_raw", 0.0)))
        telemetry["tau_position_yaw_compensated_clipped"].append(float(sagittal_diag.get("tau_position", 0.0)))
        telemetry["boundary_yaw_position_profile"].append(boundary_profile_name)
        telemetry["boundary_profile_active"].append(fix_active)
        telemetry["hip_yaw_abs_max_tracking"].append(float(max(abs(l_hip_yaw_pos), abs(r_hip_yaw_pos))))
        telemetry["hip_yaw_abs_max_threshold"].append(0.07)

        # Wheel yaw stabilizer telemetry (populated from yaw_diag)
        is_wheel_yaw = yaw_diag.get("wheel_yaw_enabled", False)
        telemetry["wheel_yaw_enabled"].append(is_wheel_yaw)
        if is_wheel_yaw:
            telemetry["wheel_yaw_error"].append(yaw_diag.get("wheel_yaw_error", 0.0))
            telemetry["wheel_yaw_rate"].append(yaw_diag.get("wheel_yaw_rate", 0.0))
            telemetry["wheel_yaw_tau_left"].append(yaw_diag.get("wheel_yaw_tau_left", 0.0))
            telemetry["wheel_yaw_tau_right"].append(yaw_diag.get("wheel_yaw_tau_right", 0.0))
            telemetry["wheel_yaw_saturated"].append(yaw_diag.get("wheel_yaw_saturated", False))
            telemetry["wheel_yaw_profile_activated"].append(yaw_diag.get("wheel_yaw_profile_activated", False))
            telemetry["wheel_yaw_kp"].append(yaw_diag.get("wheel_yaw_kp", 0.0))
            telemetry["wheel_yaw_kd"].append(yaw_diag.get("wheel_yaw_kd", 0.0))
            telemetry["wheel_yaw_max_torque"].append(yaw_diag.get("wheel_yaw_max_torque", 0.0))
            telemetry["wheel_yaw_height_gate"].append(yaw_diag.get("wheel_yaw_height_gate", 0.0))
            telemetry["wheel_yaw_use_numerical_rate"].append(yaw_diag.get("wheel_yaw_use_numerical_rate", False))
            tl = yaw_diag.get("wheel_yaw_tau_left", 0.0)
            tr = yaw_diag.get("wheel_yaw_tau_right", 0.0)
            telemetry["wheel_yaw_tau_diff"].append(float(tl - tr))
        else:
            telemetry["wheel_yaw_error"].append(0.0)
            telemetry["wheel_yaw_rate"].append(0.0)
            telemetry["wheel_yaw_tau_left"].append(0.0)
            telemetry["wheel_yaw_tau_right"].append(0.0)
            telemetry["wheel_yaw_saturated"].append(False)
            telemetry["wheel_yaw_profile_activated"].append(False)
            telemetry["wheel_yaw_kp"].append(0.0)
            telemetry["wheel_yaw_kd"].append(0.0)
            telemetry["wheel_yaw_max_torque"].append(0.0)
            telemetry["wheel_yaw_height_gate"].append(0.0)
            telemetry["wheel_yaw_use_numerical_rate"].append(False)
            telemetry["wheel_yaw_tau_diff"].append(0.0)

        # Body yaw and hip-yaw ownership telemetry
        # body_yaw_owner: "wheel_yaw_stabilizer" when enabled, else "yaw_controller"
        telemetry["body_yaw_owner"].append(
            "wheel_yaw_stabilizer" if is_wheel_yaw else "yaw_controller"
        )
        # hip_yaw_divergence_owner: "mode_based_divergence" when enabled, else "shape_posture"
        telemetry["hip_yaw_divergence_owner"].append(
            "mode_based_divergence" if mode_hip_yaw_div_enabled else "shape_posture"
        )
        # Yaw controller hip-yaw torque contribution (present regardless of wheel_yaw)
        telemetry["yaw_controller_tau_hip_yaw_left"].append(
            yaw_diag.get("tau_yaw_left", 0.0)
        )
        telemetry["yaw_controller_tau_hip_yaw_right"].append(
            yaw_diag.get("tau_yaw_right", 0.0)
        )

        # Hip-yaw mode decomposition telemetry
        l_hip_yaw_err_rad = l_hip_yaw_error
        r_hip_yaw_err_rad = r_hip_yaw_error
        hip_yaw_common = 0.5 * (l_hip_yaw_err_rad + r_hip_yaw_err_rad)
        hip_yaw_divergence = l_hip_yaw_err_rad - r_hip_yaw_err_rad
        hip_yaw_common_sum_abs = abs(l_hip_yaw_err_rad + r_hip_yaw_err_rad)
        hip_yaw_asymmetry = abs(l_hip_yaw_err_rad - r_hip_yaw_err_rad)
        div_common_ratio = (
            hip_yaw_asymmetry / (abs(hip_yaw_common) + 1e-12)
            if abs(hip_yaw_common) > 1e-12
            else float('inf')
        )
        telemetry["hip_yaw_common_error_rad"].append(float(hip_yaw_common))
        telemetry["hip_yaw_common_error_sum_abs_rad"].append(float(hip_yaw_common_sum_abs))
        telemetry["hip_yaw_divergence_error_rad"].append(float(hip_yaw_divergence))
        telemetry["hip_yaw_asymmetry_abs_rad"].append(float(hip_yaw_asymmetry))
        telemetry["hip_yaw_div_common_ratio"].append(float(div_common_ratio))
        telemetry["variant_name"].append(height_variant_setup.get("variant_name", "nominal_keyframe") if height_variant_setup else "nominal_keyframe")
        telemetry["height_variant_target_com_z_m"].append(float(height_variant_setup.get("target_com_z_m", height_cmd)) if height_variant_setup else float(height_cmd))
        telemetry["height_variant_achieved_com_z_m"].append(float(height_variant_setup.get("achieved_com_z_m", height_cmd)) if height_variant_setup else float(height_cmd))
        telemetry["height_variant_root_z_m"].append(float(height_variant_setup.get("calibrated_root_z_m", mj_data.qpos[2])) if height_variant_setup else float(mj_data.qpos[2]))
        telemetry["height_variant_hip_pitch_ref"].append(float(height_variant_setup.get("hip_pitch_ref", equilibrium_joint_pos[2])) if height_variant_setup else float(equilibrium_joint_pos[2]))
        telemetry["height_variant_knee_ref"].append(float(height_variant_setup.get("knee_ref", equilibrium_joint_pos[3])) if height_variant_setup else float(equilibrium_joint_pos[3]))
        telemetry["shape_posture_reference_source"].append("height_variant_equilibrium_joint_pos" if height_variant_setup else "nominal_equilibrium_joint_pos")
        telemetry["equilibrium_capture_after_variant_applied"].append(bool(height_variant_setup is not None))

        telemetry["com_error_x"].append(float(qp_diagnostics.get("com_error_x", 0.0)))
        telemetry["com_error_y"].append(float(qp_diagnostics.get("com_error_y", 0.0)))
        telemetry["com_error_z"].append(float(qp_diagnostics.get("com_error_z", 0.0)))
        telemetry["cp_error_x"].append(float(qp_diagnostics.get("cp_error_x", 0.0)))
        telemetry["cp_error_y"].append(float(qp_diagnostics.get("cp_error_y", 0.0)))
        telemetry["pitch_error"].append(float(qp_diagnostics.get("pitch_error", 0.0)))
        telemetry["roll_error"].append(float(qp_diagnostics.get("roll_error", 0.0)))
        telemetry["height_error"].append(float(qp_diagnostics.get("height_error", 0.0)))
        telemetry["target_com_z_m"].append(float(height_variant_setup.get("achieved_com_z_m", height_cmd)) if height_variant_setup else float(height_cmd))
        telemetry["current_com_z_m"].append(float(centroidal_state_control.com_pos[2]))
        telemetry["height_error_m"].append(float(centroidal_state_control.com_pos[2]) - (float(height_variant_setup.get("achieved_com_z_m", height_cmd)) if height_variant_setup else float(height_cmd)))
        telemetry["root_z_m"].append(float(mj_data.qpos[2]))
        telemetry["support_center_ref_x"].append(float(support_center_eq_xy[0]))
        telemetry["support_center_ref_y"].append(float(support_center_eq_xy[1]))
        telemetry["support_center_x"].append(float(sagittal_diag.get("support_center_x", support_center_eq_xy[0])))
        telemetry["support_center_y"].append(float(sagittal_diag.get("support_center_y", support_center_eq_xy[1])))
        telemetry["support_position_reference_source"].append("height_variant_equilibrium_support_center" if height_variant_setup else "nominal_equilibrium_support_center")
        telemetry["support_reference_captured_after_variant"].append(bool(height_variant_setup is not None))
        telemetry["left_fz_actual"].append(float(centroidal_state_log.left_contact_force_world[2]))
        telemetry["right_fz_actual"].append(float(centroidal_state_log.right_contact_force_world[2]))
        telemetry["fz_asymmetry_actual"].append(float(centroidal_state_log.left_contact_force_world[2] - centroidal_state_log.right_contact_force_world[2]))
        telemetry["contact_dist_min"].append(float(contact_class["contact_dist_min"]))
        telemetry["contact_dist_max"].append(float(contact_class["contact_dist_max"]))
        telemetry["correction_Fy_com"].append(float(qp_diagnostics.get("correction_Fy_com", 0.0)))
        telemetry["correction_Fy_cp"].append(float(qp_diagnostics.get("correction_Fy_cp", 0.0)))
        telemetry["correction_Fy_pitch"].append(float(qp_diagnostics.get("correction_Fy_pitch", 0.0)))
        telemetry["correction_My_roll"].append(float(qp_diagnostics.get("correction_My_roll", 0.0)))
        telemetry["distributor_f_left"].append(",".join(f"{x:.4f}" for x in np.array(qp_diagnostics.get("f_left", jnp.zeros(3)))))
        telemetry["distributor_f_right"].append(",".join(f"{x:.4f}" for x in np.array(qp_diagnostics.get("f_right", jnp.zeros(3)))))
        telemetry["tau_hip_roll"].append(",".join(f"{x:.4f}" for x in np.array(qp_diagnostics.get("tau_hip_roll", jnp.zeros(2)))))
        tau_contact_val = jnp.zeros(10)  # Placeholder if not available
        telemetry["tau_contact"].append(",".join(f"{x:.4f}" for x in np.array(tau_contact_val)))

        # Legacy compatibility fields: in balance-core mode, reflect balance-core torques or zeros
        if is_balance_core_mode(args):
            telemetry["tau_wbc_correction"].append(",".join(f"{x:.4f}" for x in np.zeros(10)))
            telemetry["tau_wbc_after_authority_clip"].append(",".join(f"{x:.4f}" for x in np.zeros(10)))
            telemetry["tau_static_feedforward"].append(",".join(f"{x:.4f}" for x in np.array(tau_support_feedforward)))
            telemetry["tau_static_posture"].append(",".join(f"{x:.4f}" for x in np.array(tau_shape_posture)))
        else:
            telemetry["tau_wbc_correction"].append(",".join(f"{x:.4f}" for x in np.array(tau_wbc_correction)))
            telemetry["tau_wbc_after_authority_clip"].append(",".join(f"{x:.4f}" for x in np.array(tau_wbc)))
            telemetry["tau_static_feedforward"].append(",".join(f"{x:.4f}" for x in np.array(tau_static_feedforward)))
            telemetry["tau_static_posture"].append(",".join(f"{x:.4f}" for x in np.array(tau_static_posture)))
        torque_rate_saturation_rate = float(np.mean(rate_flags_vec))
        hidden_torque_norm_value = float(
            np.linalg.norm(np.array(tau_wheel_balance))
            + np.linalg.norm(np.array(tau_hip_roll_centering))
            + np.linalg.norm(np.array(tau_static_posture if static_posture_controller is not None else tau_posture))
            + np.linalg.norm(np.array(tau_leg_position))
        )
        contact_state_value = "legacy"
        if is_balance_core_mode(args):
            contact_state_value = str(contact_output.state.value)
        elif static_posture_controller is not None:
            contact_state_value = "stage2_static_posture"

        update_full_rate_summary(
            pitch_x_value=float(pitch_x_rad),
            roll_y_value=float(roll_y_rad),
            com_z_value=com_height,
            wheel_vel_mean_value=float(0.5 * (float(joint_vel[4]) + float(joint_vel[9]))),
            ownership_violation_count_value=int(balance_core_result.ownership_violation_count) if is_balance_core_mode(args) else 0,
            hidden_torque_norm_value=hidden_torque_norm_value,
            tau_wbc_norm_value=tau_wbc_norm,
            torque_saturation_rate_value=tau_saturation_rate,
            torque_rate_saturation_rate_value=torque_rate_saturation_rate,
            contact_state_value=contact_state_value,
        )

        telemetry["saturation_flags"].append(",".join(f"{x}" for x in sat_flags_vec))
        telemetry["rate_limit_flags"].append(",".join(f"{x}" for x in rate_flags_vec))
        telemetry["wheel_torque_saturation_left"].append(bool(sat_flags_vec[4]))
        telemetry["wheel_torque_saturation_right"].append(bool(sat_flags_vec[9]))
        telemetry["wheel_torque_rate_saturation_left"].append(bool(rate_flags_vec[4]))
        telemetry["wheel_torque_rate_saturation_right"].append(bool(rate_flags_vec[9]))

        # Motor tracking diagnostics
        telemetry["target_joint_pos"].append(",".join(f"{x:.4f}" for x in np.array(target_joint_pos)))
        telemetry["joint_pos_error"].append(",".join(f"{x:.4f}" for x in np.array(joint_pos_error)))
        telemetry["joint_pos_error_norm"].append(joint_pos_error_norm)
        telemetry["joint_vel_norm"].append(joint_vel_norm)
        telemetry["tau_wbc_norm"].append(tau_wbc_norm)
        telemetry["tau_posture_norm"].append(tau_posture_norm)
        telemetry["tau_inverse_dynamics_norm"].append(tau_inverse_dynamics_norm)
        telemetry["tau_total_norm"].append(tau_total_norm)
        telemetry["tau_rate_unlimited"].append(tau_rate_unlimited)
        telemetry["tau_rate_limited"].append(tau_rate_limited)
        telemetry["tau_wbc_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_wbc)))
        telemetry["tau_wbc_scaled_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_wbc_scaled)))
        telemetry["tau_hip_roll_centering_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_hip_roll_centering)))
        # Stage 2: Log tau_static_posture if enabled, otherwise tau_posture
        if static_posture_controller is not None:
            telemetry["tau_posture_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_static_posture)))
        else:
            telemetry["tau_posture_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_posture)))
        # Stage 2B: Log tau_static_feedforward
        telemetry["tau_static_feedforward_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_static_feedforward)))
        telemetry["feedforward_enabled"].append(static_feedforward_controller is not None)
        telemetry["feedforward_norm"].append(float(jnp.linalg.norm(tau_static_feedforward)))
        telemetry["tau_leg_position_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_leg_position)))
        telemetry["tau_wheel_balance_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_wheel_balance)))
        telemetry["tau_total_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_smooth)))
        telemetry["tau_total_raw_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_total_raw)))
        telemetry["tau_total_clipped_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_total_clipped)))
        telemetry["tau_smooth_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_smooth)))
        telemetry["support_ratio_support_joints"].append(",".join(f"{x:.4f}" for x in support_ratios))
        telemetry["support_ratio_mean"].append(support_ratio_mean)
        telemetry["torque_rate_limit_enabled"].append(not args.disable_torque_rate_limit)
        telemetry["per_actuator_wbc_authority_enabled"].append(args.use_per_actuator_wbc_authority)
        telemetry["wbc_joint_scaling_enabled"].append(not args.disable_wbc_joint_scale)
        telemetry["initialize_tau_prev_from_wbc_enabled"].append(args.initialize_tau_prev_from_wbc)
        telemetry["hip_roll_abs_max"].append(step1_diagnostics["hip_roll_abs_max"])
        telemetry["hip_yaw_abs_max"].append(step1_diagnostics["hip_yaw_abs_max"])
        # Push disturbance telemetry
        telemetry["push_active"].append(push_active_now)
        telemetry["push_force_x"].append(push_fx_now)
        telemetry["push_force_y"].append(push_fy_now)
        telemetry["push_schedule_entries"].append(len(push_schedule))
        telemetry["hip_pitch_error_max"].append(step1_diagnostics["hip_pitch_error_max"])
        telemetry["knee_error_max"].append(step1_diagnostics["knee_error_max"])
        telemetry["wheel_balance_torque"].append(step1_diagnostics["wheel_balance_torque"])
        telemetry["control_mode"].append(step1_diagnostics["control_mode"])
        # Wheel torque pipeline telemetry
        telemetry["tau_stage2b_sagittal_wheel_l"].append(float(tau_stage2b_sagittal_wheel[4]))
        telemetry["tau_stage2b_sagittal_wheel_r"].append(float(tau_stage2b_sagittal_wheel[9]))
        telemetry["tau_total_raw_l_wheel"].append(float(tau_total_raw[4]))
        telemetry["tau_total_raw_r_wheel"].append(float(tau_total_raw[9]))
        telemetry["tau_total_clipped_l_wheel"].append(float(tau_total_clipped[4]))
        telemetry["tau_total_clipped_r_wheel"].append(float(tau_total_clipped[9]))
        telemetry["tau_smooth_l_wheel"].append(float(tau_smooth[4]))
        telemetry["tau_smooth_r_wheel"].append(float(tau_smooth[9]))
        telemetry["ctrl_l_wheel"].append(float(mj_data.ctrl[4]))
        telemetry["ctrl_r_wheel"].append(float(mj_data.ctrl[9]))
        telemetry["qvel_l_wheel"].append(float(mj_data.qvel[10]))  # l_wheel joint velocity
        telemetry["qvel_r_wheel"].append(float(mj_data.qvel[15]))  # r_wheel joint velocity
        telemetry["sagittal_term_pitch"].append(sagittal_diag.get("term_pitch", sagittal_diag.get("tau_pitch", 0.0)))
        telemetry["sagittal_term_pitch_rate"].append(sagittal_diag.get("term_pitch_rate", sagittal_diag.get("tau_pitch_rate", 0.0)))
        telemetry["sagittal_term_cp"].append(sagittal_diag.get("term_cp", 0.0))
        telemetry["sagittal_term_com_vy"].append(sagittal_diag.get("term_com_vy", sagittal_diag.get("tau_sagittal_velocity", 0.0)))
        telemetry["sagittal_term_wheel_vel_left"].append(sagittal_diag.get("term_wheel_vel_left", sagittal_diag.get("tau_wheel_velocity_left", 0.0)))
        telemetry["sagittal_term_wheel_vel_right"].append(sagittal_diag.get("term_wheel_vel_right", sagittal_diag.get("tau_wheel_velocity_right", 0.0)))
        telemetry["sagittal_balance_torque_raw"].append(sagittal_diag.get("balance_torque_raw", sagittal_diag.get("tau_common_unclipped", 0.0)))
        telemetry["sagittal_balance_torque_clipped"].append(sagittal_diag.get("balance_torque_raw", sagittal_diag.get("tau_common_clipped", 0.0)))
        telemetry["sagittal_balance_torque_final"].append(0.5 * (sagittal_diag.get("tau_left", 0.0) + sagittal_diag.get("tau_right", 0.0)))
        telemetry["sagittal_pitch_error"].append(sagittal_wheel_diagnostics.get("pitch_error", 0.0))
        telemetry["sagittal_cp_error_y"].append(sagittal_wheel_diagnostics.get("cp_error_y", 0.0))
        telemetry["sagittal_tau_wheel_cmd"].append(sagittal_wheel_diagnostics.get("tau_wheel_cmd", 0.0))
        telemetry["sagittal_saturated"].append(sagittal_wheel_diagnostics.get("saturated", False))
        # Stage 2C telemetry
        telemetry["stage2c_pitch_error"].append(stage2c_diagnostics.get("pitch_error", 0.0))
        telemetry["stage2c_pitch_rate_x"].append(stage2c_diagnostics.get("pitch_rate_x", 0.0))
        telemetry["stage2c_com_y_error"].append(stage2c_diagnostics.get("com_y_error", 0.0))
        telemetry["stage2c_com_vy"].append(stage2c_diagnostics.get("com_vy", 0.0))
        telemetry["stage2c_cp_y_error"].append(stage2c_diagnostics.get("cp_y_error", 0.0))
        telemetry["stage2c_wheel_vel_left"].append(stage2c_diagnostics.get("wheel_vel_left", 0.0))
        telemetry["stage2c_wheel_vel_right"].append(stage2c_diagnostics.get("wheel_vel_right", 0.0))
        telemetry["stage2c_wheel_vel_mean"].append(stage2c_diagnostics.get("wheel_vel_mean", 0.0))
        telemetry["stage2c_term_pitch"].append(stage2c_diagnostics.get("term_pitch", 0.0))
        telemetry["stage2c_term_pitch_rate"].append(stage2c_diagnostics.get("term_pitch_rate", 0.0))
        telemetry["stage2c_term_com_y"].append(stage2c_diagnostics.get("term_com_y", 0.0))
        telemetry["stage2c_term_com_vy"].append(stage2c_diagnostics.get("term_com_vy", 0.0))
        telemetry["stage2c_term_cp_y"].append(stage2c_diagnostics.get("term_cp_y", 0.0))
        telemetry["stage2c_term_wheel_vel"].append(stage2c_diagnostics.get("term_wheel_vel", 0.0))
        telemetry["stage2c_tau_wheel_raw"].append(stage2c_diagnostics.get("tau_wheel_raw", 0.0))
        telemetry["stage2c_tau_wheel_clipped"].append(stage2c_diagnostics.get("tau_wheel_clipped", 0.0))
        telemetry["stage2c_saturated"].append(stage2c_diagnostics.get("saturated", False))
        # Stage 2D telemetry
        telemetry["stage2d_pitch_x"].append(stage2d_diagnostics.get("pitch_x", 0.0))
        telemetry["stage2d_pitch_rate_x"].append(stage2d_diagnostics.get("pitch_rate_x", 0.0))
        telemetry["stage2d_cp_error_y"].append(stage2d_diagnostics.get("cp_error_y", 0.0))
        telemetry["stage2d_com_vy"].append(stage2d_diagnostics.get("com_vy", 0.0))
        telemetry["stage2d_wheel_vel_mean"].append(stage2d_diagnostics.get("wheel_vel_mean", 0.0))
        telemetry["stage2d_u_raw"].append(stage2d_diagnostics.get("u_raw", 0.0))
        telemetry["stage2d_u_clipped"].append(stage2d_diagnostics.get("u_clipped", 0.0))
        telemetry["stage2d_saturated"].append(stage2d_diagnostics.get("saturated", False))
        telemetry["stage2d_contrib_pitch_x"].append(stage2d_diagnostics.get("contrib_pitch_x", 0.0))
        telemetry["stage2d_contrib_pitch_rate_x"].append(stage2d_diagnostics.get("contrib_pitch_rate_x", 0.0))
        telemetry["stage2d_contrib_cp_error_y"].append(stage2d_diagnostics.get("contrib_cp_error_y", 0.0))
        telemetry["stage2d_contrib_com_vy"].append(stage2d_diagnostics.get("contrib_com_vy", 0.0))
        telemetry["stage2d_contrib_wheel_vel_mean"].append(stage2d_diagnostics.get("contrib_wheel_vel_mean", 0.0))
        telemetry["stage2d_config"].append(stage2d_diagnostics.get("config", ""))
        telemetry["initial_root_z_perturbation_m"].append(perturbation_metadata["initial_root_z_perturbation_m"])
        telemetry["nominal_equilibrium_com_z_m"].append(perturbation_metadata["nominal_equilibrium_com_z_m"])
        telemetry["initial_com_z_m_after_perturbation"].append(perturbation_metadata["initial_com_z_m_after_perturbation"])
        telemetry["perturbation_applied_after_equilibrium_capture"].append(
            perturbation_metadata["perturbation_applied_after_equilibrium_capture"]
        )

        # Save the current telemetry row for failure window and decimation
        # Guard against empty telemetry (should not happen, but be defensive)
        if telemetry["source_step_index"]:
            current_row = snapshot_last_telemetry_row()
            last_full_rate_row = dict(current_row)
            last_full_rate_step = step
            if failure_window_steps > 0:
                failure_window_buffer.append(dict(current_row))

            if not should_keep_main_telemetry_row(step, terminated):
                drop_last_telemetry_row()
        else:
            # Fallback: telemetry not initialized yet, skip decimation logic
            last_full_rate_row = None
            last_full_rate_step = step

        if step < 20 and static_feedforward_controller is not None and not args.visual:
            idx = [2, 3, 7, 8]
            sat_flags = np.abs(np.array(tau_total_raw)) > np.array(torque_limit)
            rate_flags = np.abs(np.array(tau_rate_vec)) > max_torque_rate
            wc = wbc_controller.wrench_computer
            eq_com = np.array(wc.equilibrium_com_pos) if wc.equilibrium_com_pos is not None else np.zeros(3)
            eq_cp = np.array(wc.equilibrium_capture_point) if wc.equilibrium_capture_point is not None else np.zeros(2)
            cur_com = np.array(centroidal_state_log.com_pos)
            cur_cp = np.array(centroidal_state_log.capture_point)
            com_err = cur_com - eq_com
            cp_err = cur_cp - eq_cp

            print(
                f"[B0-AUDIT][step={step}][mode={mode}] "
                f"tau_static_feedforward[2,3,7,8]={np.array(tau_static_feedforward)[idx]} "
                f"tau_static_posture[2,3,7,8]={np.array(tau_static_posture)[idx]} "
                f"tau_wbc_correction[2,3,7,8]={np.array(tau_wbc_correction)[idx]}"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"tau_total_raw[2,3,7,8]={np.array(tau_total_raw)[idx]} "
                f"tau_final[2,3,7,8]={np.array(tau_smooth)[idx]} "
                f"sat_flags[2,3,7,8]={sat_flags[idx].astype(int)} "
                f"rate_limit_flags[2,3,7,8]={rate_flags[idx].astype(int)}"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"correction_wrench_norm={float(qp_diagnostics.get('correction_wrench_norm', correction_wrench_norm)):+.6f} "
                f"correction_wrench_Fx={float(qp_diagnostics.get('correction_wrench_Fx', 0.0)):+.6f} "
                f"correction_wrench_Fy={float(qp_diagnostics.get('correction_wrench_Fy', 0.0)):+.6f} "
                f"correction_wrench_Fz={float(qp_diagnostics.get('correction_wrench_Fz', 0.0)):+.6f} "
                f"correction_wrench_My={float(qp_diagnostics.get('correction_wrench_My', 0.0)):+.6f}"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"correction_Fy_com={float(qp_diagnostics.get('correction_Fy_com', 0.0)):+.6f} "
                f"correction_Fy_cp={float(qp_diagnostics.get('correction_Fy_cp', 0.0)):+.6f} "
                f"correction_Fy_pitch={float(qp_diagnostics.get('correction_Fy_pitch', 0.0)):+.6f}"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"baseline_fz={float(qp_diagnostics.get('baseline_fz', 0.0)):+.6f} "
                f"distributor_input_wrench=[{float(qp_diagnostics.get('distributor_input_wrench_Fx', 0.0)):+.6f},"
                f" {float(qp_diagnostics.get('distributor_input_wrench_Fy', 0.0)):+.6f},"
                f" {float(qp_diagnostics.get('distributor_input_wrench_Fz', 0.0)):+.6f},"
                f" {float(qp_diagnostics.get('distributor_input_wrench_Mx', 0.0)):+.6f},"
                f" {float(qp_diagnostics.get('distributor_input_wrench_My', 0.0)):+.6f},"
                f" {float(qp_diagnostics.get('distributor_input_wrench_Mz', 0.0)):+.6f}] "
                f"distributor_fz_sum={float(qp_diagnostics.get('distributor_fz_sum', 0.0)):+.6f}"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"force_feedback_scale={float(qp_diagnostics.get('force_scale', 1.0)):+.6f} "
                f"force_feedback_enabled={bool(qp_diagnostics.get('force_feedback_enabled', False))} "
                f"force_feedback_mode={qp_diagnostics.get('force_feedback_mode', 'unknown')}"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"equilibrium_com_pos={eq_com.tolist()} current_com_pos={cur_com.tolist()} com_error={com_err.tolist()} "
                f"equilibrium_capture_point={eq_cp.tolist()} current_capture_point={cur_cp.tolist()} cp_error={cp_err.tolist()}"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"pitch_x={float(pitch_x_rad):+.6f} pitch_error={float(qp_diagnostics.get('pitch_error', 0.0)):+.6f} "
                f"roll_y={float(roll_y_rad):+.6f} roll_error={float(qp_diagnostics.get('roll_error', 0.0)):+.6f} "
                f"height_error={float(qp_diagnostics.get('height_error', 0.0)):+.6f} "
                f"gravity_body=[{float(obs[0]):+.6f}, {float(obs[1]):+.6f}, {float(obs[2]):+.6f}]"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"active_wheels={active_wheels} "
                f"left_wheel_floor_contact={contact_class['left_wheel_floor_contact']} "
                f"right_wheel_floor_contact={contact_class['right_wheel_floor_contact']} "
                f"total_wheel_floor_fz={contact_class['total_wheel_floor_fz']:+.6f}"
            )

        # Progress updates with orientation feedback
        # In visual mode, reduce print frequency to avoid I/O stutter
        _progress_interval = 100 if args.visual else 10
        if (step + 1) % _progress_interval == 0 or step < 5:
            elapsed = time.time() - start_time
            # Show what controller is sensing using unified orientation computation
            gravity_body = obs[0:3]
            pitch_x_sensed, roll_y_sensed = compute_orientation_from_gravity(gravity_body)
            pitch_sensed = float(pitch_x_sensed) * 57.3
            roll_sensed = float(roll_y_sensed) * 57.3
            print(
                f"Step {step + 1}: h={com_height:.3f}m, "
                f"euler_pitch={euler_pitch_y*57.3:.1f}deg (sensed={pitch_sensed:.1f}deg), "
                f"euler_roll={euler_roll_x*57.3:.1f}deg (sensed={roll_sensed:.1f}deg), "
                f"robot_pitch_x={robot_pitch_x*57.3:.1f}deg, robot_roll_y={robot_roll_y*57.3:.1f}deg, "
                f"gravity=[{obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}]"
            )

        if terminated:
            print(f"\n[TERMINATED] at step {step + 1}: {termination_reason}")
            return False

        if _profile_enabled:
            _profile_timing["telemetry_ms"] += (time.perf_counter() - _t_telem_start) * 1000.0
            _profile_timing["total_step_ms"] += (time.perf_counter() - _t_step_start) * 1000.0
            _profile_timing["step_count"] += 1

        step += 1
        return True

    # ---- Visual realtime pacing configuration ---- #
    if args.visual:
        # Read pacing flags
        visual_realtime_factor = float(getattr(args, "visual_realtime_factor", 1.0) or 1.0)
        visual_sync_hz = float(getattr(args, "visual_sync_hz", 30.0) or 30.0)
        visual_disable_pacing = bool(getattr(args, "visual_disable_realtime_pacing", False))
        visual_profile_timing = bool(getattr(args, "visual_profile_timing", False))

        # Clamp to sensible ranges
        if visual_realtime_factor <= 0.0:
            visual_disable_pacing = True
            visual_realtime_factor = 1.0  # placeholder for reporting
        visual_sync_hz = max(5.0, min(visual_sync_hz, 120.0))

        # Compute pacing parameters from actual control_dt (0.01 s = 100 Hz)
        control_hz = 1.0 / control_dt
        sim_duration_s = max_steps * control_dt
        pacing_dt = control_dt / max(visual_realtime_factor, 1e-6)
        sync_interval_s = 1.0 / visual_sync_hz

        # Viewer sync scheduling (decoupled from step count)
        last_viewer_sync_time = 0.0  # sim-time of last sync

        # Timing profiling accumulators
        profile_step_times_s = [] if visual_profile_timing else None
        profile_sync_times_s = [] if visual_profile_timing else None
        profile_sleep_times_s = [] if visual_profile_timing else None
        profile_n_syncs = 0
        profile_n_overslept = 0
        cumul_sleep_debt_s = 0.0

        if visual_disable_pacing:
            print("\nLaunching MuJoCo viewer...")
            print("Close the viewer window to end simulation and save telemetry.")
            print(f"Control: {control_hz:.0f} Hz | Viewer sync: {visual_sync_hz:.0f} Hz")
            print("Realtime pacing: DISABLED (running as fast as possible)")
        else:
            print("\nLaunching MuJoCo viewer...")
            print("Close the viewer window to end simulation and save telemetry.")
            print(
                f"Control: {control_hz:.0f} Hz | Viewer sync: {visual_sync_hz:.0f} Hz | "
                f"Realtime factor: {visual_realtime_factor:.2f}"
            )
        print(f"Expected sim duration: {sim_duration_s:.1f} s ({max_steps} steps)")
        if visual_profile_timing:
            print("[PROFILING] Timing diagnostics enabled. Expect slightly higher overhead.")

        sim_start_time = time.time()
        step_start_time = sim_start_time

        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            while viewer.is_running():
                t_before_step = time.time()
                if visual_profile_timing:
                    step_start_time = t_before_step

                if not simulation_step():
                    break

                t_after_step = time.time()
                step_wall_elapsed = t_after_step - t_before_step

                if visual_profile_timing:
                    profile_step_times_s.append(step_wall_elapsed)

                # Viewer sync: decoupled from step count — sync when enough wall time has passed
                wall_since_sync = t_after_step - (sim_start_time + last_viewer_sync_time)
                if wall_since_sync >= sync_interval_s:
                    t_before_sync = time.time()
                    viewer.sync()
                    t_after_sync = time.time()
                    if visual_profile_timing:
                        profile_sync_times_s.append(t_after_sync - t_before_sync)
                    profile_n_syncs += 1
                    last_viewer_sync_time = t_after_sync - sim_start_time

                # Realtime pacing (unless disabled)
                if not visual_disable_pacing:
                    # Target wall time for this step: sim_start + step * pacing_dt
                    target_time = sim_start_time + step * pacing_dt
                    current_time = time.time()
                    sleep_time = target_time - current_time

                    if sleep_time > 0:
                        # Apply sleep-debt floor: don't oversleep more than 1 control_dt
                        cumul_sleep_debt_s = min(cumul_sleep_debt_s, 0.0)
                        effective_sleep = max(sleep_time + cumul_sleep_debt_s, 0.0)
                        if effective_sleep > 0.0:
                            time.sleep(effective_sleep)
                            actual_sleep = time.time() - current_time
                            cumul_sleep_debt_s += sleep_time - actual_sleep
                            if visual_profile_timing:
                                profile_sleep_times_s.append(actual_sleep)
                        else:
                            cumul_sleep_debt_s += sleep_time
                            if visual_profile_timing:
                                profile_sleep_times_s.append(0.0)
                    else:
                        # Running behind: accumulate negative debt (but capped)
                        cumul_sleep_debt_s = max(cumul_sleep_debt_s + sleep_time, -control_dt)
                        if visual_profile_timing:
                            profile_sleep_times_s.append(0.0)
                        profile_n_overslept += 0  # track oversleep count via debt

        # Collect final timing for summary
        visual_elapsed_wall = time.time() - sim_start_time
        visual_sim_s = step * control_dt
        visual_achieved_rf = visual_sim_s / max(visual_elapsed_wall, 1e-6)
        visual_target_rf = visual_realtime_factor if not visual_disable_pacing else float("inf")
    else:
        # ---- Headless mode ---- #
        visual_elapsed_wall = None
        visual_sim_s = None
        visual_achieved_rf = None
        visual_target_rf = None
        visual_sync_hz = None
        visual_realtime_factor = None
        visual_disable_pacing = None
        visual_profile_timing = False
        profile_n_syncs = 0
        profile_step_times_s = None
        profile_sync_times_s = None
        profile_sleep_times_s = None

        while simulation_step():
            pass

    # Stage 1: Emit per-component controller profile report
    if _profile_enabled and _profile_timing["step_count"] > 0:
        import os as _os
        _profile_dir = Path("outputs/profile")
        _profile_dir.mkdir(parents=True, exist_ok=True)
        _profile_path = _profile_dir / "stage1_controller_profile_breakdown.json"
        _n = _profile_timing["step_count"]
        _profile_report = {
            "profile": "stage1_k2_python_controller",
            "backend": "python",
            "step_count": _n,
            "control_dt_s": control_dt,
            "timing_mean_ms": {
                "centroidal_control": round(_profile_timing["centroidal_control_ms"] / _n, 4),
                "capture_control": round(_profile_timing["capture_control_ms"] / _n, 4),
                "balance_core_block": round(_profile_timing["balance_core_block_ms"] / _n, 4),
                "centroidal_log": round(_profile_timing["centroidal_log_ms"] / _n, 4),
                "capture_log": round(_profile_timing["capture_log_ms"] / _n, 4),
                "telemetry": round(_profile_timing["telemetry_ms"] / _n, 4),
                "total_per_step": round(_profile_timing["total_step_ms"] / _n, 4),
            },
            "timing_total_s": {k: round(v / 1000.0, 4) for k, v in _profile_timing.items() if k != "step_count"},
            "duplicate_call_analysis": {
                "centroidal_estimate_called_twice_per_step": True,
                "capture_estimator_update_called_twice_per_step": True,
                "centroidal_control_vs_log_ratio": round(
                    _profile_timing["centroidal_control_ms"] / max(_profile_timing["centroidal_log_ms"], 1e-9), 2
                ),
                "capture_control_vs_log_ratio": round(
                    _profile_timing["capture_control_ms"] / max(_profile_timing["capture_log_ms"], 1e-9), 2
                ),
                "duplicate_removed": False,
                "duplicate_removal_blocked_by": "capture_estimator uses min_height=0.35m vs centroidal estimator min_height=0.1m; behavior would change at low heights (0.33m K2 minimum)",
                "estimated_savings_if_removed_ms": round(
                    (_profile_timing["centroidal_log_ms"] + _profile_timing["capture_log_ms"]) / _n, 4
                ),
            },
        }
        with open(_profile_path, "w") as _f:
            json.dump(_profile_report, _f, indent=2)
        print(f"\n[PROFILE] Controller profile report saved to: {_profile_path}")
        print(f"[PROFILE] Mean per-step breakdown (ms):")
        for _k, _v in _profile_report["timing_mean_ms"].items():
            print(f"  {_k}: {_v:.4f}")
        print(f"[PROFILE] Duplicate centroidal estimate (log): {_profile_report['timing_mean_ms']['centroidal_log']:.4f} ms/step")
        print(f"[PROFILE] Duplicate capture update (log):    {_profile_report['timing_mean_ms']['capture_log']:.4f} ms/step")
        print(f"[PROFILE] Total duplicate overhead:           {_profile_report['duplicate_call_analysis']['estimated_savings_if_removed_ms']:.4f} ms/step")

    elapsed_time = time.time() - start_time
    simulated_steps = int(full_rate_summary["actual_steps"])
    finalized_summary_metrics = finalize_full_rate_summary()

    if last_full_rate_row is not None and (
        len(telemetry["source_step_index"]) == 0
        or telemetry["source_step_index"][-1] != last_full_rate_step
    ):
        append_telemetry_row(last_full_rate_row)

    # Save telemetry to CSV
    csv_path = output_dir / f"telemetry_{int(time.time())}.csv"

    # Add validation-compatible telemetry fields
    add_validation_telemetry_fields(
        telemetry,
        control_dt,
        csv_path,
        survival_steps_override=simulated_steps,
    )

    # Normalize balance-core owner names if in balance-core mode
    if is_balance_core_mode(args):
        normalize_balance_core_owner_names(telemetry)

    # Check telemetry state before writing CSV
    # Calculate n_rows from non-empty columns only (empty columns have 0 entries)
    non_empty_cols = [v for v in telemetry.values() if len(v) > 0]
    n_rows = min(len(v) for v in non_empty_cols) if non_empty_cols else 0
    populated_cols = {k: len(v) for k, v in telemetry.items() if len(v) > 0}
    empty_cols = [k for k, v in telemetry.items() if len(v) == 0]

    print(f"[TELEMETRY] Columns: total={len(telemetry)}, populated={len(populated_cols)}, empty={len(empty_cols)}")
    print(f"[TELEMETRY] Data rows (n_rows): {n_rows}")
    if empty_cols:
        print(f"[TELEMETRY] First 10 empty columns: {empty_cols[:10]}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(telemetry.keys())
        # Write rows safely - only include columns that have data for each row
        for i in range(n_rows):
            row = []
            for k in telemetry.keys():
                if len(telemetry[k]) > i:
                    row.append(telemetry[k][i])
                else:
                    row.append(None)  # Missing data for this row
            writer.writerow(row)

    if terminated and failure_window_steps > 0 and len(failure_window_buffer) > 0:
        failure_window_path = output_dir / f"failure_window_{simulated_steps}.csv"
        failure_window_telemetry = {key: [] for key in failure_window_buffer[0].keys()}
        for row in failure_window_buffer:
            for key in failure_window_telemetry.keys():
                failure_window_telemetry[key].append(row[key])
        add_validation_telemetry_fields(
            failure_window_telemetry,
            control_dt,
            failure_window_path,
            survival_steps_override=simulated_steps,
        )
        if is_balance_core_mode(args):
            normalize_balance_core_owner_names(failure_window_telemetry)
        with open(failure_window_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(failure_window_telemetry.keys())
            for i in range(len(failure_window_telemetry["time"])):
                writer.writerow([failure_window_telemetry[k][i] for k in failure_window_telemetry.keys()])

    if write_run_summary_sidecar:
        sidecar_path = output_dir / f"telemetry_{simulated_steps}.summary.json"
        sidecar_payload = {
            "requested_steps": max_steps,
            "actual_steps": simulated_steps,
            "survived_steps": simulated_steps,
            "terminated": bool(terminated),
            "termination_reason": termination_reason or "completed",
            "final_sim_time_s": float(simulated_steps * control_dt),
            "wall_clock_time_s": float(elapsed_time),
            "telemetry_decimation": telemetry_decimation,
            "failure_window_steps": failure_window_steps,
            "written_telemetry_rows": n_rows,  # Use n_rows to match actual CSV rows
            "telemetry_columns_total": len(telemetry),
            "telemetry_columns_populated": len(populated_cols),
            "telemetry_columns_empty": len(empty_cols),
            **finalized_summary_metrics,
        }
        # Visual pacing metadata
        if args.visual:
            sim_s = simulated_steps * control_dt
            wall_s = visual_elapsed_wall or elapsed_time
            sidecar_payload["visual_pacing"] = {
                "mode": "visual",
                "target_realtime_factor": visual_realtime_factor if not visual_disable_pacing else None,
                "achieved_realtime_factor": float(sim_s / max(wall_s, 1e-6)),
                "pacing_disabled": visual_disable_pacing,
                "control_hz": float(1.0 / control_dt),
                "control_dt_s": float(control_dt),
                "physics_dt_s": float(physics_dt),
                "n_substeps_per_control": int(n_substeps),
                "target_viewer_sync_hz": float(visual_sync_hz) if visual_sync_hz else None,
                "viewer_sync_count": int(profile_n_syncs),
                "wall_clock_s": float(wall_s),
                "sim_time_s": float(sim_s),
                "mean_step_time_ms": float(wall_s / max(simulated_steps, 1) * 1000),
            }
        else:
            sidecar_payload["visual_pacing"] = {"mode": "headless"}
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_payload, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("Simulation Summary")
    print("=" * 80)
    print(f"Mode: {'VISUAL' if args.visual else 'HEADLESS'}")
    print(f"Total simulated steps: {simulated_steps}")
    print(f"Written telemetry rows: {len(telemetry['time'])}")
    print(f"Simulation time: {simulated_steps * control_dt:.1f} seconds")
    print(f"Wall clock time: {elapsed_time:.1f} seconds")

    # Realtime factor reporting
    if args.visual:
        sim_s = simulated_steps * control_dt
        wall_s = visual_elapsed_wall or elapsed_time
        achieved_rf = sim_s / max(wall_s, 1e-6)
        target_rf = visual_realtime_factor if not visual_disable_pacing else float("inf")

        print(f"\n--- Visual Realtime Pacing ---")
        print(f"Target realtime factor: {'∞ (no pacing)' if visual_disable_pacing else f'{target_rf:.2f}x'}")
        print(f"Achieved realtime factor: {achieved_rf:.3f}x")
        if achieved_rf < 0.95 * target_rf and not visual_disable_pacing:
            bottleneck_ratio = target_rf / max(achieved_rf, 1e-6)
            print(f"  ⚠ Realtime target NOT met — simulation is {bottleneck_ratio:.1f}x slower than target")
            print(f"  Likely causes: controller compute > pacing interval, viewer render overhead, CPU bound")
        elif achieved_rf >= 0.95 * target_rf and not visual_disable_pacing:
            print(f"  ✓ Realtime target met within ±5%")
        print(f"Control rate: {1.0/control_dt:.0f} Hz (control_dt={control_dt:.3f}s)")
        print(f"Viewer sync rate: {visual_sync_hz:.0f} Hz (target)")
        print(f"Viewer sync count: {profile_n_syncs}")
        print(f"Mean step time: {wall_s/max(simulated_steps,1)*1000:.1f} ms (target: {control_dt*1000:.1f} ms @ {1.0/control_dt:.0f} Hz)")

        # Profiling details
        if visual_profile_timing and profile_step_times_s:
            step_times = np.array(profile_step_times_s)
            print(f"\n--- Step Timing Profile (n={len(step_times)}) ---")
            print(f"  Mean: {np.mean(step_times)*1000:.2f} ms")
            print(f"  Median: {np.median(step_times)*1000:.2f} ms")
            print(f"  P50: {np.percentile(step_times,50)*1000:.2f} ms")
            print(f"  P95: {np.percentile(step_times,95)*1000:.2f} ms")
            print(f"  P99: {np.percentile(step_times,99)*1000:.2f} ms")
            print(f"  Max: {np.max(step_times)*1000:.2f} ms")
            print(f"  Std: {np.std(step_times)*1000:.2f} ms")
            if profile_sync_times_s:
                sync_times = np.array(profile_sync_times_s)
                print(f"\n  Viewer sync (n={len(sync_times)}):")
                print(f"    Mean: {np.mean(sync_times)*1000:.2f} ms")
                print(f"    P95: {np.percentile(sync_times,95)*1000:.2f} ms")
                print(f"    Max: {np.max(sync_times)*1000:.2f} ms")
            if profile_sleep_times_s:
                sleep_times = np.array(profile_sleep_times_s)
                nonsleep = np.sum(sleep_times > 0)
                print(f"\n  Sleep (n={len(sleep_times)}, slept={nonsleep} steps):")
                if nonsleep > 0:
                    pos_sleep = sleep_times[sleep_times > 0]
                    print(f"    Mean when sleeping: {np.mean(pos_sleep)*1000:.2f} ms")
                    print(f"    Max sleep: {np.max(pos_sleep)*1000:.2f} ms")
                print(f"    Steps with zero/negative sleep: {len(sleep_times) - nonsleep}")
                print(f"    Sleep ratio: {nonsleep}/{len(sleep_times)} ({100*nonsleep/max(len(sleep_times),1):.1f}%)")

    print(f"Terminated: {terminated}")
    if terminated:
        print(f"Termination reason: {termination_reason}")
    else:
        print("Status: [OK] Completed full simulation without falling")

    print(
        f"\nCoM height range: {finalized_summary_metrics['com_z']['min']:.3f} - {finalized_summary_metrics['com_z']['max']:.3f} m"
    )
    print(
        f"Robot pitch_x range: {finalized_summary_metrics['pitch_x']['min']*57.3:.1f} - {finalized_summary_metrics['pitch_x']['max']*57.3:.1f} deg"
    )
    print(
        f"Robot roll_y range: {finalized_summary_metrics['roll_y']['min']*57.3:.1f} - {finalized_summary_metrics['roll_y']['max']*57.3:.1f} deg"
    )

    print(f"\nMax torques (wheeled biped architecture):")
    max_hip_roll = max(telemetry["tau_wbc_max"])
    max_wheels = max(telemetry["tau_wheel_actual_max"])
    max_legs = max(telemetry["tau_posture_max"])
    max_total = max(telemetry["tau_total_max"])
    print(f"  Hip roll: {max_hip_roll:.2f} Nm")
    print(f"  Wheels: {max_wheels:.2f} Nm")
    print(f"  Legs: {max_legs:.2f} Nm")
    print(f"  Total: {max_total:.2f} Nm")

    print(f"\nTelemetry saved to: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
