"""Centroidal balance controller with integrated CoM and capture point tracking."""

import chex
import jax.numpy as jnp
from jax import Array


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
