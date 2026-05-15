# wheeled_biped/controllers/posture_regularizer.py
"""Posture regularization for Level 3 stabilization.

Provides weak posture restoration with two-level gating (WBC error gate,
momentum coordinator gate) and per-joint deadbands with 20% authority budget.
"""

import chex
import jax.numpy as jnp
from jax import Array


@chex.dataclass(frozen=True)
class PostureRegularizerConfig:
    """Configuration for posture regularizer."""
    # Proportional gain
    k_posture: float = 2.0  # Weak compared to WBC gains

    # Per-joint deadbands (radians)
    hip_roll_deadband: float = 0.05  # ±2.9° - allow lateral sway
    hip_yaw_deadband: float = 0.03  # ±1.7° - tighter, yaw drift is bad
    hip_pitch_deadband: float = 0.08  # ±4.6° - allow squat variation
    knee_deadband: float = 0.10  # ±5.7° - allow knee bend variation
    wheel_deadband: float = 0.0  # wheels don't have posture target

    # Gating thresholds
    wbc_error_threshold: float = 0.3  # 30% of WBC capacity
    momentum_active_scale: float = 0.5  # 50% authority when momentum active
    momentum_activity_threshold: float = 0.1  # Threshold to detect momentum coordinator activity

    # Authority budget
    posture_authority_budget: float = 0.2  # 20% of actuator range


class PostureRegularizer:
    """Posture regularizer for Level 3 stabilization."""

    def __init__(self, config: PostureRegularizerConfig):
        """Initialize posture regularizer.

        Args:
            config: PostureRegularizerConfig with gains and thresholds
        """
        self.config = config

    def compute_posture_restoration_torque(self, joint_pos: Array) -> Array:
        """Compute posture restoration torque with per-joint deadbands.

        Args:
            joint_pos: Joint position array (10,) - current joint angles

        Returns:
            Posture restoration torque array (10,) opposing posture errors
        """
        # Target posture is zero for all joints
        target_pos = jnp.zeros(10)

        # Compute posture errors
        posture_error = joint_pos - target_pos

        # Per-joint deadbands
        deadbands = jnp.array([
            self.config.hip_roll_deadband,   # 0: left hip roll
            self.config.hip_yaw_deadband,    # 1: left hip yaw
            self.config.hip_pitch_deadband,  # 2: left hip pitch
            self.config.knee_deadband,       # 3: left knee
            self.config.wheel_deadband,      # 4: left wheel
            self.config.hip_roll_deadband,   # 5: right hip roll
            self.config.hip_yaw_deadband,    # 6: right hip yaw
            self.config.hip_pitch_deadband,  # 7: right hip pitch
            self.config.knee_deadband,       # 8: right knee
            self.config.wheel_deadband,      # 9: right wheel
        ])

        # JAX-compatible deadband using jnp.where
        # Only apply torque if error exceeds deadband
        active = jnp.where(jnp.abs(posture_error) > deadbands, 1.0, 0.0)

        # Proportional control with deadband gating
        tau = -self.config.k_posture * posture_error * active

        return tau

    def apply_wbc_error_gate(self, joint_pos: Array, wbc_error_magnitude: float) -> Array:
        """Apply WBC error gate to posture restoration.

        If WBC error exceeds threshold, completely disable posture regularization.

        Args:
            joint_pos: Joint position array (10,)
            wbc_error_magnitude: WBC error magnitude (normalized 0-1)

        Returns:
            Gated posture torque array (10,)
        """
        # Compute base posture restoration torque
        tau_posture = self.compute_posture_restoration_torque(joint_pos)

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

    def apply_momentum_gate(self, joint_pos: Array, momentum_magnitude: float) -> Array:
        """Apply momentum coordinator gate to posture restoration.

        Reduces posture authority by 50% when momentum coordinator is active.

        Args:
            joint_pos: Joint position array (10,)
            momentum_magnitude: Momentum coordinator activity magnitude (0-1)

        Returns:
            Gated posture torque array (10,)
        """
        # Compute base posture restoration torque
        tau_posture = self.compute_posture_restoration_torque(joint_pos)

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
