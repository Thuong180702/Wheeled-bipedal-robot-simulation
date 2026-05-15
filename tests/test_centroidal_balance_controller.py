# tests/test_centroidal_balance_controller.py
import jax.numpy as jnp
import pytest
from wheeled_biped.controllers.centroidal_balance_controller import (
    CentroidalBalanceController,
    CentroidalBalanceConfig,
)


def test_centroidal_balance_controller_creation():
    """Test CentroidalBalanceController can be created with config."""
    config = CentroidalBalanceConfig(
        # Roll stabilization
        k_roll=20.0,
        k_roll_rate=4.0,

        # CoM regulation
        k_com_lateral=15.0,
        k_com_lateral_damping=3.0,
        k_com_sagittal=10.0,
        k_com_sagittal_damping=2.0,

        # Deadbands
        com_deadband_lateral=0.02,
        com_deadband_sagittal=0.03,

        # Authority budget
        wbc_authority_budget=0.6,
    )
    controller = CentroidalBalanceController(config)

    assert controller.config.k_roll == 20.0
    assert controller.config.wbc_authority_budget == 0.6
