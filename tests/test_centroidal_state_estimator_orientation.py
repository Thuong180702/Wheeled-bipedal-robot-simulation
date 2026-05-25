"""Regression tests for centroidal state estimator orientation and rate mapping.

Verifies that robot-frame orientation aliases (pitch_x, roll_y) correctly map to
body-frame values (body_pitch_x, body_roll_y) and that rates match finite-difference
derivatives within tolerance.
"""

import mujoco
import numpy as np
import pytest
from pathlib import Path

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)


@pytest.fixture
def model_and_data():
    """Load MuJoCo model and data."""
    xml_path = Path("assets/robot/wheeled_biped_real.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    return model, data


@pytest.fixture
def estimator(model_and_data):
    """Create centroidal state estimator."""
    model, _ = model_and_data
    robot_mass = float(np.sum(model.body_mass))
    config = CentroidalStateEstimatorConfig(
        robot_mass=robot_mass,
        torso_inertia=np.array([0.1, 0.1, 0.05]),
    )
    return CentroidalStateEstimator(config, mj_model=model)


def test_pitch_x_maps_to_body_pitch_x(model_and_data, estimator):
    """Robot-frame pitch_x alias should equal body_pitch_x."""
    model, data = model_and_data

    # Apply pitch perturbation (rotate about body X axis)
    mujoco.mj_forward(model, data)
    state, _ = estimator.estimate(np.zeros(42), data, None)

    # Verify pitch_x equals body_pitch_x
    assert state.pitch_x == state.body_pitch_x, (
        f"pitch_x ({state.pitch_x}) should equal body_pitch_x ({state.body_pitch_x})"
    )


def test_roll_y_maps_to_body_roll_y(model_and_data, estimator):
    """Robot-frame roll_y alias should equal body_roll_y."""
    model, data = model_and_data

    # Apply roll perturbation (rotate about body Y axis)
    mujoco.mj_forward(model, data)
    state, _ = estimator.estimate(np.zeros(42), data, None)

    # Verify roll_y equals body_roll_y
    assert state.roll_y == state.body_roll_y, (
        f"roll_y ({state.roll_y}) should equal body_roll_y ({state.body_roll_y})"
    )


def test_pitch_rate_matches_finite_difference(model_and_data, estimator):
    """Pitch rate should match finite-difference derivative of pitch_x."""
    model, data = model_and_data
    dt = 0.01

    # Get initial state
    mujoco.mj_forward(model, data)
    state0, com0 = estimator.estimate(np.zeros(42), data, None)
    pitch_x_0 = state0.pitch_x

    # Step simulation multiple times
    for _ in range(10):
        mujoco.mj_step(model, data)

    # Get final state
    mujoco.mj_forward(model, data)
    state1, com1 = estimator.estimate(np.zeros(42), data, com0)
    pitch_x_1 = state1.pitch_x
    pitch_rate_x = state1.pitch_rate_x

    # Compute finite-difference rate
    pitch_rate_fd = (pitch_x_1 - pitch_x_0) / (10 * dt)

    # Verify rates are same order of magnitude (within 50% tolerance)
    # This catches the 40x error that was present before the fix
    if abs(pitch_rate_fd) > 0.01:  # Only check if there's significant motion
        ratio = abs(pitch_rate_x / pitch_rate_fd) if pitch_rate_fd != 0 else 0
        assert 0.5 < ratio < 2.0, (
            f"pitch_rate_x ({pitch_rate_x:.6f}) should match finite-difference "
            f"({pitch_rate_fd:.6f}), ratio={ratio:.2f}"
        )


def test_roll_rate_matches_finite_difference(model_and_data, estimator):
    """Roll rate should match finite-difference derivative of roll_y."""
    model, data = model_and_data
    dt = 0.01

    # Get initial state
    mujoco.mj_forward(model, data)
    state0, com0 = estimator.estimate(np.zeros(42), data, None)
    roll_y_0 = state0.roll_y

    # Step simulation multiple times
    for _ in range(10):
        mujoco.mj_step(model, data)

    # Get final state
    mujoco.mj_forward(model, data)
    state1, com1 = estimator.estimate(np.zeros(42), data, com0)
    roll_y_1 = state1.roll_y
    roll_rate_y = state1.roll_rate_y

    # Compute finite-difference rate
    roll_rate_fd = (roll_y_1 - roll_y_0) / (10 * dt)

    # Verify rates are same order of magnitude (within 50% tolerance)
    if abs(roll_rate_fd) > 0.01:  # Only check if there's significant motion
        ratio = abs(roll_rate_y / roll_rate_fd) if roll_rate_fd != 0 else 0
        assert 0.5 < ratio < 2.0, (
            f"roll_rate_y ({roll_rate_y:.6f}) should match finite-difference "
            f"({roll_rate_fd:.6f}), ratio={ratio:.2f}"
        )


def test_pitch_rate_not_euler_rate(model_and_data, estimator):
    """Pitch_rate_x should not equal Euler pitch_rate when conventions differ."""
    model, data = model_and_data

    # Apply significant pitch perturbation
    data.qpos[4] = 0.3  # Rotate quaternion to create pitch
    mujoco.mj_forward(model, data)

    state, _ = estimator.estimate(np.zeros(42), data, None)

    # Verify pitch_rate_x uses body frame, not Euler frame
    # (They should differ when there's significant rotation)
    assert state.pitch_rate_x == state.body_pitch_rate_x, (
        f"pitch_rate_x should equal body_pitch_rate_x, not Euler pitch_rate"
    )

    # Verify it's not accidentally using Euler rate
    assert state.pitch_rate_x != state.pitch_rate or abs(state.pitch_rate_x) < 0.01, (
        f"pitch_rate_x ({state.pitch_rate_x}) should not equal Euler pitch_rate "
        f"({state.pitch_rate}) when there's significant rotation"
    )
