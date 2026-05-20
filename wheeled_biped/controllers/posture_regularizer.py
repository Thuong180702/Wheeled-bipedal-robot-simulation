# wheeled_biped/controllers/posture_regularizer.py
"""Posture regularization for Level 3 stabilization.

Provides weak posture restoration with momentum coordinator gating and per-joint
deadbands with 20% authority budget. Always available as backup when WBC saturates.
"""

import chex
import jax.numpy as jnp
from jax import Array


@chex.dataclass(frozen=True)
class PostureRegularizerConfig:
    """Configuration for posture regularizer."""

    # Proportional gain (used for any joint group without an override)
    k_posture: float = 10.0
    k_hip_roll: float | None = None
    k_hip_yaw: float | None = None
    k_hip_pitch: float | None = None
    k_knee: float | None = None
    k_wheel: float = 0.0

    # Per-joint deadbands (radians) - reduced for earlier activation
    hip_roll_deadband: float = 0.15  # ±8.6° - LARGE deadband, hip roll must be free for balance
    hip_yaw_deadband: float = 0.02  # ±1.1° - tighter, yaw drift is bad
    hip_pitch_deadband: float = 0.035  # ±2.0° - reduced from ±4.6° for earlier activation
    knee_deadband: float = 0.05  # ±2.9° - reduced from ±5.7° for earlier activation
    wheel_deadband: float = 0.0  # wheels don't have posture target

    # Gating thresholds
    wbc_error_threshold: float = 0.3  # 30% of WBC capacity
    momentum_active_scale: float = 0.5  # 50% authority when momentum active
    momentum_activity_threshold: float = (
        0.1  # Threshold to detect momentum coordinator activity
    )

    # Authority budget
    posture_authority_budget: float = 0.2  # 20% of actuator range
    max_actuator_torque: float = 30.0


class PostureRegularizer:
    """Posture regularizer for Level 3 stabilization."""

    def __init__(self, config: PostureRegularizerConfig):
        """Initialize posture regularizer.

        Args:
            config: PostureRegularizerConfig with gains and thresholds
        """
        self.config = config

        # Robot geometry (from URDF)
        self.hip_to_knee_length = 0.26  # hip_pitch joint to knee joint (thigh length)
        self.knee_to_wheel_length = 0.28  # knee joint to wheel center (shin length)
        self.hip_height_offset = 0.05  # hip joint height above base

        self._joint_gains = jnp.array([
            self.config.k_hip_roll if self.config.k_hip_roll is not None else self.config.k_posture,
            self.config.k_hip_yaw if self.config.k_hip_yaw is not None else self.config.k_posture,
            self.config.k_hip_pitch if self.config.k_hip_pitch is not None else self.config.k_posture,
            self.config.k_knee if self.config.k_knee is not None else self.config.k_posture,
            self.config.k_wheel,
            self.config.k_hip_roll if self.config.k_hip_roll is not None else self.config.k_posture,
            self.config.k_hip_yaw if self.config.k_hip_yaw is not None else self.config.k_posture,
            self.config.k_hip_pitch if self.config.k_hip_pitch is not None else self.config.k_posture,
            self.config.k_knee if self.config.k_knee is not None else self.config.k_posture,
            self.config.k_wheel,
        ])

        # Height-dependent target lookup table
        # Format: (height_m, l_hip_pitch, l_knee, r_hip_pitch, r_knee)
        # Equilibrium configuration at h=0.534m (from keyframe)
        self.height_targets = jnp.array([
            [0.40, 0.926052, 1.748364, 0.926052, 1.748364],  # Current standing keyframe
            [0.45, 0.78, 1.85, 0.78, 1.85],  # Mid-low
            [0.50, 0.72, 1.75, 0.72, 1.75],  # Mid
            [0.534, 0.668271, 1.698462, 0.668302, 1.698341],  # Equilibrium (from keyframe)
            [0.55, 0.65, 1.65, 0.65, 1.65],  # Mid-high
            [0.60, 0.58, 1.50, 0.58, 1.50],  # High
            [0.65, 0.50, 1.35, 0.50, 1.35],  # Higher
            [0.70, 0.42, 1.20, 0.42, 1.20],  # Highest - more upright
        ])

    def compute_target_posture_from_height(self, height_cmd: float) -> Array:
        """Compute target posture from height command using interpolation.

        Interpolates between height-dependent equilibrium configurations to provide
        smooth target transitions. Uses equilibrium keyframe at h=0.534m as anchor point.

        Args:
            height_cmd: Desired CoM height in meters (0.40 to 0.70)

        Returns:
            Target joint positions (10,) interpolated from height lookup table
        """
        # Clamp height to valid range
        height_clamped = jnp.clip(height_cmd, 0.40, 0.70)

        # Extract height column and target columns
        heights = self.height_targets[:, 0]
        l_hip_pitch_targets = self.height_targets[:, 1]
        l_knee_targets = self.height_targets[:, 2]
        r_hip_pitch_targets = self.height_targets[:, 3]
        r_knee_targets = self.height_targets[:, 4]

        # Linear interpolation for each joint
        l_hip_pitch = jnp.interp(height_clamped, heights, l_hip_pitch_targets)
        l_knee = jnp.interp(height_clamped, heights, l_knee_targets)
        r_hip_pitch = jnp.interp(height_clamped, heights, r_hip_pitch_targets)
        r_knee = jnp.interp(height_clamped, heights, r_knee_targets)

        # Construct full target posture
        # Hip roll has NO target - must be free for WBC balance control
        # Hip yaw has small target to prevent drift
        # Hip pitch/knee interpolated from height for posture maintenance
        target_pos = jnp.array([
            0.0,          # l_hip_roll - NO TARGET (free for balance)
            -0.000740,    # l_hip_yaw - equilibrium value
            l_hip_pitch,  # l_hip_pitch - interpolated
            l_knee,       # l_knee - interpolated
            0.0,          # l_wheel - no target
            0.0,          # r_hip_roll - NO TARGET (free for balance)
            0.000859,     # r_hip_yaw - equilibrium value
            r_hip_pitch,  # r_hip_pitch - interpolated
            r_knee,       # r_knee - interpolated
            0.0,          # r_wheel - no target
        ])

        return target_pos

    def compute_posture_restoration_torque(self, joint_pos: Array, height_cmd: float | None = None) -> Array:
        """Compute posture restoration torque with per-joint deadbands.

        Args:
            joint_pos: Joint position array (10,) - current joint angles
            height_cmd: Desired height command for adaptive IK

        Returns:
            Posture restoration torque array (10,) opposing posture errors
        """
        if height_cmd is None:
            target_pos = jnp.zeros(10)
        else:
            target_pos = self.compute_target_posture_from_height(height_cmd)

        # Compute posture errors
        posture_error = joint_pos - target_pos

        # Per-joint deadbands
        deadbands = jnp.array(
            [
                self.config.hip_roll_deadband,  # 0: left hip roll
                self.config.hip_yaw_deadband,  # 1: left hip yaw
                self.config.hip_pitch_deadband,  # 2: left hip pitch
                self.config.knee_deadband,  # 3: left knee
                self.config.wheel_deadband,  # 4: left wheel
                self.config.hip_roll_deadband,  # 5: right hip roll
                self.config.hip_yaw_deadband,  # 6: right hip yaw
                self.config.hip_pitch_deadband,  # 7: right hip pitch
                self.config.knee_deadband,  # 8: right knee
                self.config.wheel_deadband,  # 9: right wheel
            ]
        )

        # JAX-compatible deadband using jnp.where
        # Only apply torque if error exceeds deadband
        active = jnp.where(jnp.abs(posture_error) > deadbands, 1.0, 0.0)

        # Proportional control with deadband gating
        tau = -self._joint_gains * posture_error * active

        return tau

    def apply_wbc_error_gate(
        self, joint_pos: Array, wbc_error_magnitude: float, height_cmd: float | None = None
    ) -> Array:
        """Apply WBC error gate to posture restoration.

        If WBC error exceeds threshold, completely disable posture regularization.

        Args:
            joint_pos: Joint position array (10,)
            wbc_error_magnitude: WBC error magnitude (normalized 0-1)
            height_cmd: Desired height command for adaptive IK

        Returns:
            Gated posture torque array (10,)
        """
        # Compute base posture restoration torque
        tau_posture = self.compute_posture_restoration_torque(joint_pos, height_cmd)

        # JAX-compatible gating using jnp.where
        # Disable completely if WBC error exceeds threshold
        gate = jnp.where(
            wbc_error_magnitude > self.config.wbc_error_threshold,
            0.0,
            1.0,
        )

        # Apply gate
        tau_gated = tau_posture * gate

        return tau_gated

    def apply_momentum_gate(self, joint_pos: Array, momentum_magnitude: float, height_cmd: float | None = None) -> Array:
        """Apply momentum coordinator gate to posture restoration.

        Reduces posture authority by 50% when momentum coordinator is active.

        Args:
            joint_pos: Joint position array (10,)
            momentum_magnitude: Momentum coordinator activity magnitude (0-1)
            height_cmd: Desired height command for adaptive IK

        Returns:
            Gated posture torque array (10,)
        """
        # Compute base posture restoration torque
        tau_posture = self.compute_posture_restoration_torque(joint_pos, height_cmd)

        # JAX-compatible gating using jnp.where
        # Reduce to 50% if momentum coordinator is active (magnitude > threshold)
        gate = jnp.where(
            momentum_magnitude > self.config.momentum_activity_threshold,
            self.config.momentum_active_scale,
            1.0,
        )

        # Apply gate
        tau_gated = tau_posture * gate

        return tau_gated

    def clip_to_authority_budget(self, tau: Array) -> Array:
        """Clip torque to posture regularizer authority budget.

        Args:
            tau: Desired torque array (10,)

        Returns:
            Clipped torque array (10,) within 20% authority budget
        """
        # Compute budget limit
        budget_limit = self.config.posture_authority_budget * self.config.max_actuator_torque

        # Find maximum absolute torque
        max_tau = jnp.max(jnp.abs(tau))

        # JAX-compatible conditional scaling
        scale_factor = jnp.where(max_tau <= budget_limit, 1.0, budget_limit / max_tau)
        tau_clipped = tau * scale_factor

        return tau_clipped

    def compute_posture_regularizer_torque(
        self,
        joint_pos: Array,
        wbc_error_magnitude: float,
        momentum_magnitude: float,
        height_cmd: float | None = None,
    ) -> Array:
        """Compute integrated posture regularizer torque with momentum gating.

        Combines posture restoration with momentum coordinator gating, then applies
        20% authority budget. Posture provides continuous backup, especially when
        WBC saturates.

        Args:
            joint_pos: Joint position array (10,)
            wbc_error_magnitude: WBC error magnitude (normalized 0-1) - unused, kept for API compatibility
            momentum_magnitude: Momentum coordinator activity magnitude (0-1)
            height_cmd: Desired height command for adaptive IK

        Returns:
            Posture regularizer torque array (10,) with gating and budget clipping
        """
        # Compute base posture restoration torque
        tau_posture = self.compute_posture_restoration_torque(joint_pos, height_cmd)

        # Apply momentum coordinator gate (reduce to 50% if active)
        momentum_gate = jnp.where(
            momentum_magnitude > self.config.momentum_activity_threshold,
            self.config.momentum_active_scale,
            1.0,
        )
        tau_posture = tau_posture * momentum_gate

        # Clip to 20% authority budget
        tau_posture = self.clip_to_authority_budget(tau_posture)

        return tau_posture
