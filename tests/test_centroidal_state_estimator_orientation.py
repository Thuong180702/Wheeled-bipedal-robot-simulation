"""Regression tests for centroidal state estimator orientation and rate mapping.

Verifies that robot-frame orientation aliases (pitch_x, roll_y, yaw_z) correctly map to
body-frame values and that rate aliases preserve body-frame angular velocity component
signs and axis identity.
"""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    return np.array([
        math.cos(half),
        axis[0] * math.sin(half),
        axis[1] * math.sin(half),
        axis[2] * math.sin(half),
    ])


def left_multiply_quat(delta_quat, quat):
    w1, x1, y1, z1 = delta_quat
    w2, x2, y2, z2 = quat
    quat_new = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    return quat_new / np.linalg.norm(quat_new)


def set_base_rotation(data, axis, angle):
    data.qpos[3:7] = left_multiply_quat(quat_from_axis_angle(axis, angle), np.array(data.qpos[3:7]))


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

    mujoco.mj_forward(model, data)
    state, _ = estimator.estimate(np.zeros(42), data, None)

    assert state.pitch_x == state.body_pitch_x, (
        f"pitch_x ({state.pitch_x}) should equal body_pitch_x ({state.body_pitch_x})"
    )


def test_roll_y_maps_to_body_roll_y(model_and_data, estimator):
    """Robot-frame roll_y alias should equal body_roll_y."""
    model, data = model_and_data

    mujoco.mj_forward(model, data)
    state, _ = estimator.estimate(np.zeros(42), data, None)

    assert state.roll_y == state.body_roll_y, (
        f"roll_y ({state.roll_y}) should equal body_roll_y ({state.body_roll_y})"
    )


def test_yaw_z_maps_to_body_yaw_z(model_and_data, estimator):
    """Robot-frame yaw_z alias should equal body_yaw_z."""
    model, data = model_and_data

    mujoco.mj_forward(model, data)
    state, _ = estimator.estimate(np.zeros(42), data, None)

    assert state.yaw_z == state.body_yaw_z, (
        f"yaw_z ({state.yaw_z}) should equal body_yaw_z ({state.body_yaw_z})"
    )


def test_x_axis_rotation_primarily_affects_pitch_x(model_and_data, estimator):
    """X-axis body rotation should map to pitch_x with negligible roll/yaw leakage."""
    model, data = model_and_data

    set_base_rotation(data, [1.0, 0.0, 0.0], 0.12)
    mujoco.mj_forward(model, data)
    state, _ = estimator.estimate(np.zeros(42), data, None)

    assert abs(state.pitch_x) > 0.1
    assert abs(state.roll_y) / abs(state.pitch_x) < 1e-3
    assert abs(state.yaw_z) / abs(state.pitch_x) < 1e-3


def test_y_axis_rotation_primarily_affects_roll_y(model_and_data, estimator):
    """Y-axis body rotation should map to roll_y with negligible pitch/yaw leakage."""
    model, data = model_and_data

    set_base_rotation(data, [0.0, 1.0, 0.0], -0.12)
    mujoco.mj_forward(model, data)
    state, _ = estimator.estimate(np.zeros(42), data, None)

    assert abs(state.roll_y) > 0.1
    assert abs(state.pitch_x) / abs(state.roll_y) < 1e-3
    assert abs(state.yaw_z) / abs(state.roll_y) < 1e-3


def test_z_axis_rotation_primarily_affects_yaw_z(model_and_data, estimator):
    """Z-axis body rotation should map to yaw_z with negligible pitch/roll leakage."""
    model, data = model_and_data

    set_base_rotation(data, [0.0, 0.0, 1.0], 0.12)
    mujoco.mj_forward(model, data)
    state, _ = estimator.estimate(np.zeros(42), data, None)

    assert abs(state.yaw_z) > 0.1
    assert abs(state.pitch_x) / abs(state.yaw_z) < 1e-3
    assert abs(state.roll_y) / abs(state.yaw_z) < 1e-3


def test_pitch_rate_alias_maps_to_body_x_angular_velocity(model_and_data, estimator):
    """Pitch rate alias should equal body X angular velocity."""
    model, data = model_and_data

    data.qvel[3:6] = np.array([0.7, -0.2, 0.1])
    mujoco.mj_forward(model, data)
    state, _ = estimator.estimate(np.zeros(42), data, None)

    assert state.pitch_rate_x == pytest.approx(0.7)
    assert state.roll_rate_y == pytest.approx(-0.2)
    assert state.yaw_rate_z == pytest.approx(0.1)


def test_roll_rate_alias_maps_to_body_y_angular_velocity(model_and_data, estimator):
    """Roll rate alias should preserve Y-axis body angular velocity sign."""
    model, data = model_and_data

    data.qvel[3:6] = np.array([0.0, -0.45, 0.0])
    mujoco.mj_forward(model, data)
    state, _ = estimator.estimate(np.zeros(42), data, None)

    assert state.pitch_rate_x == pytest.approx(0.0)
    assert state.roll_rate_y == pytest.approx(-0.45)
    assert state.yaw_rate_z == pytest.approx(0.0)


def test_yaw_rate_alias_maps_to_body_z_angular_velocity(model_and_data, estimator):
    """Yaw rate alias should preserve Z-axis body angular velocity sign."""
    model, data = model_and_data

    data.qvel[3:6] = np.array([0.0, 0.0, 0.33])
    mujoco.mj_forward(model, data)
    state, _ = estimator.estimate(np.zeros(42), data, None)

    assert state.pitch_rate_x == pytest.approx(0.0)
    assert state.roll_rate_y == pytest.approx(0.0)
    assert state.yaw_rate_z == pytest.approx(0.33)


def test_pitch_rate_not_euler_rate(model_and_data, estimator):
    """Pitch_rate_x should not equal Euler pitch_rate when conventions differ."""
    model, data = model_and_data

    data.qpos[4] = 0.3
    mujoco.mj_forward(model, data)

    state, _ = estimator.estimate(np.zeros(42), data, None)

    assert state.pitch_rate_x == state.body_pitch_rate_x, (
        f"pitch_rate_x should equal body_pitch_rate_x, not Euler pitch_rate"
    )

    assert state.pitch_rate_x != state.pitch_rate or abs(state.pitch_rate_x) < 0.01, (
        f"pitch_rate_x ({state.pitch_rate_x}) should not equal Euler pitch_rate "
        f"({state.pitch_rate}) when there's significant rotation"
    )
