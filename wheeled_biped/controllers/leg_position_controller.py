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
        kp_hip_yaw: float = 5.0,
        kd_hip_yaw: float = 1.0,
        kp_hip_pitch: float = 20.0,
        kd_hip_pitch: float = 2.0,
        kp_knee: float = 30.0,
        kd_knee: float = 3.0,
        max_torque: float = 30.0,
    ):
        """Initialize leg position controller.

        Args:
            kp_hip_yaw: Proportional gain for hip_yaw
            kd_hip_yaw: Derivative gain for hip_yaw
            kp_hip_pitch: Proportional gain for hip_pitch
            kd_hip_pitch: Derivative gain for hip_pitch
            kp_knee: Proportional gain for knee
            kd_knee: Derivative gain for knee
            max_torque: Maximum torque per joint (Nm)
        """
        self.kp_hip_yaw = kp_hip_yaw
        self.kd_hip_yaw = kd_hip_yaw
        self.kp_hip_pitch = kp_hip_pitch
        self.kd_hip_pitch = kd_hip_pitch
        self.kp_knee = kp_knee
        self.kd_knee = kd_knee
        self.max_torque = max_torque

        self.LEG_POSTURE_INDICES = [1, 2, 3, 6, 7, 8]

    def compute_leg_torques(self, joint_pos: Array, joint_vel: Array, target_joint_pos: Array) -> Array:
        """Compute PD control torques from the per-step posture target.

        Args:
            joint_pos: Current joint positions (10,)
            joint_vel: Current joint velocities (10,)
            target_joint_pos: Target joint positions (10,)

        Returns:
            Joint torques (10,) with hip-roll and wheels left at zero
        """
        tau = jnp.zeros(10)
        joint_gains = {
            1: (self.kp_hip_yaw, self.kd_hip_yaw),
            2: (self.kp_hip_pitch, self.kd_hip_pitch),
            3: (self.kp_knee, self.kd_knee),
            6: (self.kp_hip_yaw, self.kd_hip_yaw),
            7: (self.kp_hip_pitch, self.kd_hip_pitch),
            8: (self.kp_knee, self.kd_knee),
        }

        for joint_idx, (kp, kd) in joint_gains.items():
            pos_error = target_joint_pos[joint_idx] - joint_pos[joint_idx]
            vel_error = -joint_vel[joint_idx]
            tau_raw = kp * pos_error + kd * vel_error
            tau = tau.at[joint_idx].set(jnp.clip(tau_raw, -self.max_torque, self.max_torque))

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
