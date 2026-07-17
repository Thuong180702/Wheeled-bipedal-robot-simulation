"""Shape-posture controller for balance-core architecture.

Includes optional support-error feedforward compensation for hip-yaw
disturbance rejection (HY-FF candidate fix).
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.balance_core_types import (
    ACTION_DIM,
    SUPPORT_SHAPE_INDICES,
    zeros_action,
)


@dataclass(frozen=True)
class HipYawAuthorityProfile:
    name: str
    kp_hip_yaw: float
    kd_hip_yaw: float


BASELINE_HIP_YAW_AUTHORITY = HipYawAuthorityProfile(
    name="baseline_current",
    kp_hip_yaw=5.0,
    kd_hip_yaw=1.0,
)

BALANCE_CORE_HIP_YAW_AUTHORITY = HipYawAuthorityProfile(
    name="balance_core_candidate_b",
    kp_hip_yaw=15.0,
    kd_hip_yaw=3.0,
)


# ============================================================================
# HIP-YAW DIVERGENCE DAMPING PROFILE
# ============================================================================
# Phase 3 candidate: HY2-DIV
# Addresses divergence-dominant hip-yaw error by applying antisymmetric
# torque proportional to left/right hip-yaw error difference.
# Enabled only at low heights via smooth height gate.
# ============================================================================

@dataclass(frozen=True)
class HipYawDivergenceProfile:
    name: str
    k_divergence: float  # Proportional gain on divergence (|l - r|)
    k_divergence_rate: float  # Derivative gain on divergence rate
    tau_max_divergence: float  # Maximum antisymmetric torque magnitude
    z_low: float = 0.300
    z_high: float = 0.393

    def gate(self, z_ref: float) -> float:
        """Compute smooth height gate."""
        u = jnp.clip((self.z_high - z_ref) / (self.z_high - self.z_low), 0.0, 1.0)
        return 3.0 * u**2 - 2.0 * u**3


HY2_DIV_BASELINE = HipYawDivergenceProfile(
    name="hy2_div_baseline",
    k_divergence=5.0,
    k_divergence_rate=1.0,
    tau_max_divergence=0.5,
)

HY2_DIV_AGGRESSIVE = HipYawDivergenceProfile(
    name="hy2_div_aggressive",
    k_divergence=10.0,
    k_divergence_rate=2.0,
    tau_max_divergence=1.0,
)

# ============================================================================
# PHASE 3 AUTHORITY CANDIDATE PROFILES
# ============================================================================
# Candidate Group A: Low-height authority only
# Candidate Group B: Height-gate coverage
# Candidate Group C: Strong damping (only if A/B partial)

# Group A: Low-height authority only
HY2_DIV_A0 = HipYawDivergenceProfile(
    name="hy2_div_A0",
    k_divergence=5.0,
    k_divergence_rate=1.0,
    tau_max_divergence=0.5,
    z_low=0.300,
    z_high=0.393,
)

HY2_DIV_A1 = HipYawDivergenceProfile(
    name="hy2_div_A1",
    k_divergence=5.0,
    k_divergence_rate=1.0,
    tau_max_divergence=1.0,
    z_low=0.300,
    z_high=0.393,
)

HY2_DIV_A2 = HipYawDivergenceProfile(
    name="hy2_div_A2",
    k_divergence=5.0,
    k_divergence_rate=1.0,
    tau_max_divergence=2.0,
    z_low=0.300,
    z_high=0.393,
)

HY2_DIV_A3 = HipYawDivergenceProfile(
    name="hy2_div_A3",
    k_divergence=7.5,
    k_divergence_rate=1.5,
    tau_max_divergence=1.0,
    z_low=0.300,
    z_high=0.393,
)

# Group B: Height-gate coverage
HY2_DIV_B1 = HipYawDivergenceProfile(
    name="hy2_div_B1",
    k_divergence=5.0,
    k_divergence_rate=1.0,
    tau_max_divergence=1.0,
    z_low=0.300,
    z_high=0.500,
)

HY2_DIV_B2 = HipYawDivergenceProfile(
    name="hy2_div_B2",
    k_divergence=5.0,
    k_divergence_rate=1.0,
    tau_max_divergence=2.0,
    z_low=0.300,
    z_high=0.500,
)

HY2_DIV_B3 = HipYawDivergenceProfile(
    name="hy2_div_B3",
    k_divergence=7.5,
    k_divergence_rate=1.5,
    tau_max_divergence=1.0,
    z_low=0.300,
    z_high=0.500,
)

# Group C: Strong damping (only if A/B partial)
HY2_DIV_C1 = HipYawDivergenceProfile(
    name="hy2_div_C1",
    k_divergence=10.0,
    k_divergence_rate=2.0,
    tau_max_divergence=1.5,
    z_low=0.300,
    z_high=0.500,
)

# All candidates for iteration
ALL_HY2_DIV_CANDIDATES = [
    HY2_DIV_A0,
    HY2_DIV_A1,
    HY2_DIV_A2,
    HY2_DIV_A3,
    HY2_DIV_B1,
    HY2_DIV_B2,
    HY2_DIV_B3,
    HY2_DIV_C1,
]


def compute_hip_yaw_support_feedforward_height_gate(z_ref: float) -> float:
    """Smooth height-based activation gate for hip-yaw support feedforward.

    Returns 1.0 at z <= 0.300 m (full compensation at extreme flexion)
    Returns 0.0 at z >= 0.393 m (no compensation at nominal/tall heights)
    Smoothstep transition between (continuous, C1).

    Args:
        z_ref: Target CoM height in meters

    Returns:
        Activation gate value in [0, 1]
    """
    z_low = 0.300
    z_high = 0.393

    u = jnp.clip((z_high - z_ref) / (z_high - z_low), 0.0, 1.0)
    s = 3.0 * u**2 - 2.0 * u**3  # smoothstep: C1 continuous

    return s


class ShapePostureController:
    """Joint-space PD posture controller on support-shape joints only.

    Optionally includes support-error feedforward compensation for hip-yaw
    disturbance rejection (HY-FF candidate fix).
    """

    def __init__(
        self,
        kp_hip_yaw: float = 5.0,
        kd_hip_yaw: float = 1.0,
        kp_hip_pitch: float = 30.0,
        kd_hip_pitch: float = 4.0,
        kp_knee: float = 40.0,
        kd_knee: float = 5.0,
        enable_hip_yaw_support_feedforward: bool = False,
        k_support_hip_yaw: float = 0.0,
        tau_max_support_comp: float = 1.0,
        support_comp_sign: float = 1.0,
        # HY2-DIV: divergence damping
        enable_hip_yaw_divergence_damping: bool = False,
        k_divergence: float = 0.0,
        k_divergence_rate: float = 0.0,
        tau_max_divergence: float = 0.5,
        divergence_gate_z_low: float = 0.300,
        divergence_gate_z_high: float = 0.393,
    ):
        self.kp_hip_yaw = kp_hip_yaw
        self.kd_hip_yaw = kd_hip_yaw
        self.kp_hip_pitch = kp_hip_pitch
        self.kd_hip_pitch = kd_hip_pitch
        self.kp_knee = kp_knee
        self.kd_knee = kd_knee

        # HY-FF: support-error feedforward for hip-yaw disturbance rejection
        self.enable_hip_yaw_support_feedforward = enable_hip_yaw_support_feedforward
        self.k_support_hip_yaw = k_support_hip_yaw
        self.tau_max_support_comp = tau_max_support_comp
        self.support_comp_sign = support_comp_sign

        # HY2-DIV: divergence damping
        self.enable_hip_yaw_divergence_damping = enable_hip_yaw_divergence_damping
        self.k_divergence = k_divergence
        self.k_divergence_rate = k_divergence_rate
        self.tau_max_divergence = tau_max_divergence
        self.divergence_gate_z_low = divergence_gate_z_low
        self.divergence_gate_z_high = divergence_gate_z_high

        # Configure the mode‑based controller using existing fields when enabled
        from .mode_based_hip_yaw_divergence_controller import ModeBasedHipYawDivergenceController
        cfg = {
            "enabled": enable_hip_yaw_divergence_damping,
            "kp_div": k_divergence,
            "kd_div": k_divergence_rate,
            "max_torque": tau_max_divergence,
            "soft_limit_rad": divergence_gate_z_low,
            "soft_limit_gain": divergence_gate_z_high - divergence_gate_z_low,
            "ref_source": "target",
        }
        self._mode_based_controller = ModeBasedHipYawDivergenceController(cfg)

    def compute(
        self,
        q_ref: Array,
        joint_pos: Array,
        joint_vel: Array,
        posture_weight: float = 1.0,
        contact_degraded_scale: float = 1.0,
        support_position_error: float = 0.0,
        target_com_height: float = 0.45,
    ) -> tuple[Array, dict]:
        """Compute posture torque and diagnostics.

        Produces nonzero torque only on support-shape joints:
        hip-yaw [1,6], hip-pitch [2,7], knee [3,8].

        Args:
            q_ref: Reference joint positions [10]
            joint_pos: Current joint positions [10]
            joint_vel: Current joint velocities [10]
            posture_weight: Global posture authority scale
            contact_degraded_scale: Contact-based authority scale
            support_position_error: Forward support position error (m), for HY-FF
            target_com_height: Target CoM height (m), for HY-FF height gate

        Returns:
            tau: Joint torque command [10]
            diagnostics: Dict with posture metrics and HY-FF telemetry
        """
        if q_ref.shape != (ACTION_DIM,):
            raise ValueError(f"q_ref must be shape ({ACTION_DIM},), got {q_ref.shape}")
        if joint_pos.shape != (ACTION_DIM,):
            raise ValueError(
                f"joint_pos must be shape ({ACTION_DIM},), got {joint_pos.shape}"
            )
        if joint_vel.shape != (ACTION_DIM,):
            raise ValueError(
                f"joint_vel must be shape ({ACTION_DIM},), got {joint_vel.shape}"
            )

        authority_scale = posture_weight * contact_degraded_scale
        posture_error = q_ref - joint_pos

        tau = zeros_action()

        # Compute HY-FF support-error feedforward compensation
        height_gate = 0.0
        tau_comp_left_raw = 0.0
        tau_comp_right_raw = 0.0
        tau_comp_left_final = 0.0
        tau_comp_right_final = 0.0
        tau_comp_left_clipped = False
        tau_comp_right_clipped = False

        if self.enable_hip_yaw_support_feedforward and self.k_support_hip_yaw != 0.0:
            # Compute smooth height gate (1.0 at low heights, 0.0 at nominal)
            height_gate = compute_hip_yaw_support_feedforward_height_gate(target_com_height)

            # Compute raw compensation torque
            # Left/right hip-yaw have opposite sign to correct yaw divergence
            tau_comp_left_raw = self.support_comp_sign * self.k_support_hip_yaw * support_position_error * height_gate
            tau_comp_right_raw = -self.support_comp_sign * self.k_support_hip_yaw * support_position_error * height_gate

            # Clamp compensation
            tau_comp_left_final = jnp.clip(tau_comp_left_raw, -self.tau_max_support_comp, self.tau_max_support_comp)
            tau_comp_right_final = jnp.clip(tau_comp_right_raw, -self.tau_max_support_comp, self.tau_max_support_comp)

            # Detect clipping
            tau_comp_left_clipped = jnp.abs(tau_comp_left_raw) > self.tau_max_support_comp
            tau_comp_right_clipped = jnp.abs(tau_comp_right_raw) > self.tau_max_support_comp

        # Compute HY2-DIV divergence damping (antisymmetric compensation) using mode‑based controller
        div_gate = 0.0
        tau_div_left_raw = 0.0
        tau_div_right_raw = 0.0
        tau_div_left_clipped = False
        tau_div_right_clipped = False

        if self.enable_hip_yaw_divergence_damping:
            # Compute smooth height gate (preserve original behavior for telemetry)
            u = jnp.clip((self.divergence_gate_z_high - target_com_height) / (self.divergence_gate_z_high - self.divergence_gate_z_low), 0.0, 1.0)
            div_gate = 3.0 * u**2 - 2.0 * u**3

            # Build state for controller
            class _HipYawState:
                def __init__(self, div_error: float, div_rate: float, height: float):
                    self.div_error = div_error
                    self.div_rate = div_rate
                    self.height = height

            l_error = posture_error[1]
            r_error = posture_error[6]
            divergence = l_error - r_error
            divergence_rate = joint_vel[1] - joint_vel[6]
            state = _HipYawState(div_error=divergence, div_rate=divergence_rate, height=target_com_height)

            result = self._mode_based_controller.compute(state)
            # Apply height gate to the controller's output (controller already applies its own gate, but we retain original gating semantics)
            tau_div_left_raw = result["tau_left"] * div_gate
            tau_div_right_raw = result["tau_right"] * div_gate

            # Apply torque clamp
            tau_div_left_raw = jnp.clip(tau_div_left_raw, -self.tau_max_divergence, self.tau_max_divergence)
            tau_div_right_raw = jnp.clip(tau_div_right_raw, -self.tau_max_divergence, self.tau_max_divergence)

            # Detect clipping based on configured max torque
            tau_div_left_clipped = jnp.abs(tau_div_left_raw) >= self.tau_max_divergence - 1e-6
            tau_div_right_clipped = jnp.abs(tau_div_right_raw) >= self.tau_max_divergence - 1e-6

        # Hip-yaw: PD + optional HY-FF compensation + optional HY2-DIV divergence damping
        # Standard PD control: torque = kp * error - kd * velocity
        # Positive error (pos < ref) -> positive torque (increases position)
        # Negative error (pos > ref) -> negative torque (decreases position)
        for idx in [1, 6]:
            tau_pd = self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx]
            tau_comp = tau_comp_left_final if idx == 1 else tau_comp_right_final
            tau_div = tau_div_left_raw if idx == 1 else tau_div_right_raw
            tau_total = authority_scale * tau_pd + tau_comp + tau_div
            tau = tau.at[idx].set(tau_total)

        for idx in [2, 7]:
            tau_raw = self.kp_hip_pitch * posture_error[idx] - self.kd_hip_pitch * joint_vel[idx]
            tau = tau.at[idx].set(authority_scale * tau_raw)

        for idx in [3, 8]:
            tau_raw = self.kp_knee * posture_error[idx] - self.kd_knee * joint_vel[idx]
            tau = tau.at[idx].set(authority_scale * tau_raw)

        diagnostics = {
            "posture_error_norm": float(jnp.linalg.norm(posture_error[SUPPORT_SHAPE_INDICES])),
            "torque_norm": float(jnp.linalg.norm(tau[SUPPORT_SHAPE_INDICES])),
            # HY-FF telemetry
            "hip_yaw_comp_active": self.enable_hip_yaw_support_feedforward,
            "hip_yaw_comp_height_gate": float(height_gate),
            "hip_yaw_comp_support_error_m": float(support_position_error),
            "hip_yaw_comp_tau_left": float(tau_comp_left_raw),
            "hip_yaw_comp_tau_right": float(tau_comp_right_raw),
            "hip_yaw_comp_tau_left_clipped": bool(tau_comp_left_clipped),
            "hip_yaw_comp_tau_right_clipped": bool(tau_comp_right_clipped),
            "hip_yaw_comp_sign": float(self.support_comp_sign),
            "hip_yaw_comp_k_support": float(self.k_support_hip_yaw),
            "hip_yaw_comp_tau_max": float(self.tau_max_support_comp),
            # HY2-DIV telemetry
            # hy2_div_enabled: controller is enabled (config flag)
            "hip_yaw_div_enabled": self.enable_hip_yaw_divergence_damping,
            # hy2_div_gate_active: gate > epsilon (operationally active)
            "hip_yaw_div_gate_active": bool(div_gate > 1e-6),
            # hy2_div_height_gate: continuous smoothstep gate value
            "hip_yaw_div_height_gate": float(div_gate),
            # hy2_div_effective_k/kd: gate-scaled gains (actual control authority)
            "hip_yaw_div_effective_k": float(self.k_divergence * div_gate),
            "hip_yaw_div_effective_kd": float(self.k_divergence_rate * div_gate),
            # hy2_div_effective_tau_max: torque limit (not gate-scaled; gate scales the command)
            "hip_yaw_div_effective_tau_max": float(self.tau_max_divergence),
            # hy2_div_torque: raw torque commands
            "hip_yaw_div_left": float(tau_div_left_raw),
            "hip_yaw_div_right": float(tau_div_right_raw),
            # hy2_div_clipped: saturation flags
            "hip_yaw_div_left_clipped": bool(tau_div_left_clipped),
            "hip_yaw_div_right_clipped": bool(tau_div_right_clipped),
            # hy2_div_config: static config values
            "hip_yaw_div_k_divergence": float(self.k_divergence),
            "hip_yaw_div_k_divergence_rate": float(self.k_divergence_rate),
            "hip_yaw_div_tau_max": float(self.tau_max_divergence),
            "hip_yaw_div_z_low": float(self.divergence_gate_z_low),
            "hip_yaw_div_z_high": float(self.divergence_gate_z_high),
        }

        return tau, diagnostics
