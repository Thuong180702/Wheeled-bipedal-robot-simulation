"""Momentum coordinator for Level 2 stabilization.

Provides reactive momentum damping, proactive feedforward compensation,
and contact-aware recovery with 20% authority budget.
"""

import chex
import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


@chex.dataclass(frozen=True)
class MomentumCoordinatorConfig:
    """Configuration for momentum coordinator."""
    # Momentum damping
    k_momentum_lateral: float = 0.8
    k_momentum_sagittal: float = 1.2
    k_angular_roll: float = 1.5

    # Feedforward compensation
    k_feedforward: float = 5.0
    k_feedforward_hip: float = 2.0
    height_transition_threshold: float = 0.05  # m/s

    # Contact-aware recovery
    k_contact_recovery: float = 10.0
    k_contact_wheel_diff: float = 4.0
    unloading_threshold: float = 0.3  # 30% force asymmetry

    # Deadbands
    momentum_deadband_linear: float = 0.5  # kg*m/s
    momentum_deadband_angular: float = 0.2  # kg*m^2/s

    # Authority budget
    momentum_authority_budget: float = 0.2  # 20% of actuator range


class MomentumCoordinator:
    """Momentum coordinator for Level 2 stabilization."""

    def __init__(self, config: MomentumCoordinatorConfig):
        """Initialize momentum coordinator.

        Args:
            config: MomentumCoordinatorConfig with gains and thresholds
        """
        self.config = config

    def compute_momentum_damping_torque(self, state: CentroidalState) -> Array:
        """Compute momentum damping torque to prevent oscillation buildup.

        Args:
            state: CentroidalState with linear and angular momentum

        Returns:
            Damping torque array (10,) opposing unwanted momentum
        """
        tau = jnp.zeros(10)

        # Linear momentum damping (lateral and sagittal)
        linear_momentum_mag = jnp.sqrt(
            state.linear_momentum[0]**2 + state.linear_momentum[1]**2
        )

        # JAX-compatible deadband using jnp.where
        linear_active = jnp.where(
            linear_momentum_mag > self.config.momentum_deadband_linear,
            1.0,
            0.0,
        )

        # Lateral momentum → hip roll damping
        lateral_momentum = state.linear_momentum[1]
        tau_lateral = -self.config.k_momentum_lateral * lateral_momentum * linear_active
        tau = tau.at[0].set(tau_lateral)  # left hip roll
        tau = tau.at[5].set(tau_lateral)  # right hip roll

        # Sagittal momentum → wheel damping
        sagittal_momentum = state.linear_momentum[0]
        tau_sagittal = -self.config.k_momentum_sagittal * sagittal_momentum * linear_active
        tau = tau.at[4].set(tau_sagittal)  # left wheel
        tau = tau.at[9].set(tau_sagittal)  # right wheel

        # Angular momentum damping (roll axis most critical)
        angular_momentum_mag = jnp.abs(state.angular_momentum[0])

        angular_active = jnp.where(
            angular_momentum_mag > self.config.momentum_deadband_angular,
            1.0,
            0.0,
        )

        # Roll momentum → differential hip roll
        roll_momentum = state.angular_momentum[0]
        tau_angular_left = -self.config.k_angular_roll * roll_momentum * angular_active
        tau_angular_right = self.config.k_angular_roll * roll_momentum * angular_active

        tau = tau.at[0].add(tau_angular_left)  # left hip roll (add to existing)
        tau = tau.at[5].add(tau_angular_right)  # right hip roll (opposite sign)

        return tau
