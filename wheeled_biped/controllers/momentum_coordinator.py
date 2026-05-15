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
