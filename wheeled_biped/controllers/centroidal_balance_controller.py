"""Centroidal balance controller with integrated CoM and capture point tracking."""

import chex
import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


@chex.dataclass
class CentroidalBalanceConfig:
    """Configuration for centroidal balance controller."""
    # Roll stabilization (from Step 5.25)
    k_roll: float = 20.0
    k_roll_rate: float = 4.0

    # CoM regulation
    k_com_lateral: float = 15.0
    k_com_lateral_damping: float = 3.0
    k_com_sagittal: float = 10.0
    k_com_sagittal_damping: float = 2.0

    # Capture point tracking
    k_cp_lateral: float = 25.0
    k_cp_sagittal: float = 20.0
    k_cp_wheel_diff: float = 8.0

    # Height tracking
    k_height: float = 5.0

    # Task weights
    w_roll: float = 1.0
    w_com: float = 0.8
    w_cp: float = 1.2  # Highest priority - divergence is critical
    w_height: float = 0.6

    # Deadbands
    com_deadband_lateral: float = 0.02  # meters
    com_deadband_sagittal: float = 0.03  # meters
    cp_deadband: float = 0.05  # meters

    # Authority budget
    wbc_authority_budget: float = 0.6  # 60% of actuator range


class CentroidalBalanceController:
    """Centroidal WBC with CoM regulation and capture point tracking."""

    def __init__(self, config: CentroidalBalanceConfig):
        self.config = config

    def compute_roll_stabilization_torque(self, obs: Array) -> Array:
        """Compute roll stabilization torque for hip roll joints.

        Args:
            obs: Observation array with gravity_body at [0:3], base_ang_vel at [6:9]

        Returns:
            Torque array (10,) with roll correction on hip roll joints
        """
        # Compute roll from gravity vector in body frame
        # roll = atan2(gravity_y, gravity_z)
        roll = jnp.arctan2(obs[1], obs[2])

        # Extract roll rate from base angular velocity
        roll_rate = obs[6]

        # PD control: tau = -k_p * error - k_d * error_rate
        tau_hip_roll = -self.config.k_roll * roll - self.config.k_roll_rate * roll_rate

        # Apply to both hip roll joints (symmetric)
        tau = jnp.zeros(10)
        tau = tau.at[0].set(tau_hip_roll)  # left hip roll
        tau = tau.at[5].set(tau_hip_roll)  # right hip roll

        return tau

    def compute_com_regulation_torque(self, state: CentroidalState) -> Array:
        """Compute CoM regulation torque with deadband control.

        Args:
            state: CentroidalState with com_pos and com_vel

        Returns:
            Torque array (10,) with CoM correction on hip roll and wheels
        """
        # Extract CoM position and velocity
        com_x = state.com_pos[0]  # sagittal (forward)
        com_y = state.com_pos[1]  # lateral (sideways)
        com_vx = state.com_vel[0]
        com_vy = state.com_vel[1]

        # Apply deadband to lateral error
        com_y_error = jnp.where(
            jnp.abs(com_y) < self.config.com_deadband_lateral,
            0.0,
            com_y
        )

        # Apply deadband to sagittal error
        com_x_error = jnp.where(
            jnp.abs(com_x) < self.config.com_deadband_sagittal,
            0.0,
            com_x
        )

        # Lateral CoM error → hip roll torques (symmetric)
        tau_lateral = -self.config.k_com_lateral * com_y_error - self.config.k_com_lateral_damping * com_vy

        # Sagittal CoM error → wheel torques (common mode)
        tau_sagittal = -self.config.k_com_sagittal * com_x_error - self.config.k_com_sagittal_damping * com_vx

        # Build torque vector
        tau = jnp.zeros(10)
        tau = tau.at[0].set(tau_lateral)  # left hip roll
        tau = tau.at[5].set(tau_lateral)  # right hip roll
        tau = tau.at[4].set(tau_sagittal)  # left wheel
        tau = tau.at[9].set(tau_sagittal)  # right wheel

        return tau

    def compute_capture_point_tracking_torque(self, state: CentroidalState) -> Array:
        """Compute capture point tracking torque with deadband control.

        Args:
            state: CentroidalState with capture_point and divergence

        Returns:
            Torque array (10,) with CP correction on hip roll and wheels
        """
        # Extract capture point error (divergence from support center)
        cp_x = state.capture_point[0]  # sagittal
        cp_y = state.capture_point[1]  # lateral

        # Compute magnitude for deadband check
        cp_error_mag = jnp.sqrt(cp_x**2 + cp_y**2)

        # Apply deadband using JAX-compatible conditional
        # If error magnitude is below deadband, zero out both components
        cp_x_active = jnp.where(
            cp_error_mag < self.config.cp_deadband,
            0.0,
            cp_x
        )
        cp_y_active = jnp.where(
            cp_error_mag < self.config.cp_deadband,
            0.0,
            cp_y
        )

        # Lateral divergence → hip roll torques (asymmetric for differential correction)
        tau_lateral = -self.config.k_cp_lateral * cp_y_active

        # Sagittal divergence → wheel torques (common mode)
        tau_sagittal = -self.config.k_cp_sagittal * cp_x_active

        # Additional wheel differential for lateral correction
        tau_wheel_diff = -self.config.k_cp_wheel_diff * cp_y_active

        # Build torque vector
        tau = jnp.zeros(10)
        tau = tau.at[0].set(tau_lateral)  # left hip roll
        tau = tau.at[5].set(tau_lateral)  # right hip roll
        tau = tau.at[4].set(tau_sagittal + tau_wheel_diff)  # left wheel
        tau = tau.at[9].set(tau_sagittal - tau_wheel_diff)  # right wheel (opposite for differential)

        return tau

    def compute_height_tracking_torque(self, obs: Array, state: CentroidalState) -> Array:
        """Compute height tracking torque using simplified IK.

        Args:
            obs: Observation array with height_cmd at index 39
            state: CentroidalState with current CoM height

        Returns:
            Torque array (10,) with height correction on hip pitch and knee
        """
        # Extract height command and current height
        height_cmd = obs[39]
        height_current = state.com_pos[2]

        # Height error
        height_error = height_cmd - height_current

        # Simplified IK: distribute height correction to hip pitch and knee
        # Positive error (need to extend) → negative hip pitch, positive knee
        # This is a simplified mapping; real IK would be more sophisticated
        tau_hip_pitch = -self.config.k_height * height_error
        tau_knee = self.config.k_height * height_error * 1.5  # knee has more leverage

        # Build torque vector
        tau = jnp.zeros(10)
        tau = tau.at[2].set(tau_hip_pitch)  # left hip pitch
        tau = tau.at[3].set(tau_knee)  # left knee
        tau = tau.at[7].set(tau_hip_pitch)  # right hip pitch
        tau = tau.at[8].set(tau_knee)  # right knee

        return tau

    def clip_to_authority_budget(self, tau: Array, budget: float) -> Array:
        """Clip torque vector to authority budget while preserving proportions.

        Args:
            tau: Desired torque array (10,)
            budget: Authority budget as fraction of max actuator range (0.0-1.0)

        Returns:
            Clipped torque array (10,) scaled to fit within budget
        """
        # Assume max actuator torque is 30 Nm (typical for this robot)
        max_actuator_torque = 30.0

        # Compute budget limit
        budget_limit = budget * max_actuator_torque

        # Find maximum absolute torque
        max_tau = jnp.max(jnp.abs(tau))

        # Use JAX-compatible conditional: scale down if exceeds budget, otherwise return as-is
        scale_factor = jnp.where(max_tau <= budget_limit, 1.0, budget_limit / max_tau)
        tau_clipped = tau * scale_factor

        return tau_clipped

    def compute_centroidal_wbc_torque(self, obs: Array, state: CentroidalState) -> Array:
        """Compute integrated centroidal WBC torque with all objectives.

        Args:
            obs: Observation array
            state: CentroidalState with all centroidal quantities

        Returns:
            WBC torque array (10,) clipped to 60% authority budget
        """
        # Compute individual objective torques
        tau_roll = self.compute_roll_stabilization_torque(obs)
        tau_com = self.compute_com_regulation_torque(state)
        tau_cp = self.compute_capture_point_tracking_torque(state)
        tau_height = self.compute_height_tracking_torque(obs, state)

        # Weighted fusion
        tau_wbc_desired = (
            self.config.w_roll * tau_roll +
            self.config.w_com * tau_com +
            self.config.w_cp * tau_cp +
            self.config.w_height * tau_height
        )

        # Clip to 60% authority budget
        tau_wbc = self.clip_to_authority_budget(tau_wbc_desired, self.config.wbc_authority_budget)

        return tau_wbc
