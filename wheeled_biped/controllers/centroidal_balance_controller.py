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

    # Deadbands
    com_deadband_lateral: float = 0.02  # meters
    com_deadband_sagittal: float = 0.03  # meters

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
