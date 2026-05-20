"""Position controller for leg joints to maintain balanced configuration.

Wheeled bipeds require the leg joints (hip_pitch, knee) to stay at fixed angles
that maintain the CoM over the wheel contact points. This controller provides
strong PD position control on these joints while allowing wheels and hip_roll
to actively balance.
"""

import jax.numpy as jnp
from jax import Array


class LegPositionController:
    """PD position controller for leg joints."""

    def __init__(
        self,
        target_hip_pitch: float = 0.95,
        target_knee: float = 1.70,
        kp_hip_pitch: float = 20.0,  # Reduced from 100 for compliance
        kd_hip_pitch: float = 2.0,   # Reduced from 10
        kp_knee: float = 30.0,        # Reduced from 150
        kd_knee: float = 3.0,         # Reduced from 15
        max_torque: float = 30.0,     # Limit to 50% of actuator capacity
    ):
        """Initialize leg position controller.

        Args:
            target_hip_pitch: Target hip_pitch angle (rad) for balanced configuration
            target_knee: Target knee angle (rad) for balanced configuration
            kp_hip_pitch: Proportional gain for hip_pitch
            kd_hip_pitch: Derivative gain for hip_pitch
            kp_knee: Proportional gain for knee
            kd_knee: Derivative gain for knee
            max_torque: Maximum torque per joint (Nm)
        """
        self.target_hip_pitch = target_hip_pitch
        self.target_knee = target_knee
        self.kp_hip_pitch = kp_hip_pitch
        self.kd_hip_pitch = kd_hip_pitch
        self.kp_knee = kp_knee
        self.kd_knee = kd_knee
        self.max_torque = max_torque

        # Joint indices in 10-DOF array
        self.LEG_INDICES = [2, 3, 7, 8]  # l_hip_pitch, l_knee, r_hip_pitch, r_knee

    def compute_leg_torques(self, joint_pos: Array, joint_vel: Array) -> Array:
        """Compute PD control torques to hold legs at balanced configuration.

        Args:
            joint_pos: Current joint positions (10,)
            joint_vel: Current joint velocities (10,)

        Returns:
            Joint torques (10,) with position control on leg joints, zeros elsewhere
        """
        tau = jnp.zeros(10)

        # Left hip_pitch (index 2)
        pos_error = self.target_hip_pitch - joint_pos[2]
        vel_error = 0.0 - joint_vel[2]
        tau_raw = self.kp_hip_pitch * pos_error + self.kd_hip_pitch * vel_error
        tau = tau.at[2].set(jnp.clip(tau_raw, -self.max_torque, self.max_torque))

        # Left knee (index 3)
        pos_error = self.target_knee - joint_pos[3]
        vel_error = 0.0 - joint_vel[3]
        tau_raw = self.kp_knee * pos_error + self.kd_knee * vel_error
        tau = tau.at[3].set(jnp.clip(tau_raw, -self.max_torque, self.max_torque))

        # Right hip_pitch (index 7)
        pos_error = self.target_hip_pitch - joint_pos[7]
        vel_error = 0.0 - joint_vel[7]
        tau_raw = self.kp_hip_pitch * pos_error + self.kd_hip_pitch * vel_error
        tau = tau.at[7].set(jnp.clip(tau_raw, -self.max_torque, self.max_torque))

        # Right knee (index 8)
        pos_error = self.target_knee - joint_pos[8]
        vel_error = 0.0 - joint_vel[8]
        tau_raw = self.kp_knee * pos_error + self.kd_knee * vel_error
        tau = tau.at[8].set(jnp.clip(tau_raw, -self.max_torque, self.max_torque))

        return tau

    def mask_wbc_torques(self, tau_wbc: Array) -> Array:
        """Zero out WBC torques on leg joints to prevent interference.

        Args:
            tau_wbc: WBC torques (10,)

        Returns:
            Masked WBC torques with leg joints zeroed
        """
        # Zero out leg joint torques individually (JAX requires this approach)
        tau_masked = tau_wbc.at[2].set(0.0)  # l_hip_pitch
        tau_masked = tau_masked.at[3].set(0.0)  # l_knee
        tau_masked = tau_masked.at[7].set(0.0)  # r_hip_pitch
        tau_masked = tau_masked.at[8].set(0.0)  # r_knee
        return tau_masked
