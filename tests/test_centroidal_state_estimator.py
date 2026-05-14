"""Tests for centroidal state estimation."""

import numpy as np
import pytest
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


def test_centroidal_state_creation():
    """Test CentroidalState dataclass can be created with all required fields."""
    state = CentroidalState(
        com_pos=np.array([0.0, 0.0, 0.6]),
        com_vel=np.array([0.0, 0.0, 0.0]),
        capture_point=np.array([0.0, 0.0]),
        divergence=np.array([0.0, 0.0]),
        linear_momentum=np.array([0.0, 0.0, 0.0]),
        angular_momentum=np.array([0.0, 0.0, 0.0]),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=100.0,
        right_wheel_force=100.0,
    )

    assert state.com_pos.shape == (3,)
    assert state.com_vel.shape == (3,)
    assert state.capture_point.shape == (2,)
    assert state.divergence.shape == (2,)
    assert state.linear_momentum.shape == (3,)
    assert state.angular_momentum.shape == (3,)
    assert isinstance(state.left_wheel_contact, bool)
    assert isinstance(state.right_wheel_contact, bool)
    assert isinstance(state.left_wheel_force, float)
    assert isinstance(state.right_wheel_force, float)
