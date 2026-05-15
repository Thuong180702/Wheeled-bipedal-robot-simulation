"""Tests for MomentumCoordinator."""

import jax.numpy as jnp
import pytest
from wheeled_biped.controllers.momentum_coordinator import (
    MomentumCoordinator,
    MomentumCoordinatorConfig,
)
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


def test_momentum_coordinator_creation():
    """Test MomentumCoordinator can be created with config."""
    config = MomentumCoordinatorConfig(
        k_momentum_lateral=0.8,
        k_momentum_sagittal=1.2,
        k_angular_roll=1.5,
        momentum_authority_budget=0.2,
    )
    coordinator = MomentumCoordinator(config)

    assert coordinator.config.k_momentum_lateral == 0.8
    assert coordinator.config.momentum_authority_budget == 0.2
