"""Tests for centroidal state estimation."""

import jax.numpy as jnp
import mujoco
import pytest
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalState,
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)


def test_centroidal_state_creation():
    """Test CentroidalState dataclass can be created with all required fields."""
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.6]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.array([0.0, 0.0]),
        divergence=jnp.array([0.0, 0.0]),
        linear_momentum=jnp.array([0.0, 0.0, 0.0]),
        angular_momentum=jnp.array([0.0, 0.0, 0.0]),
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


def test_com_extraction_from_mjx_data():
    """Test CoM position and velocity extraction from MJX data."""
    config = CentroidalStateEstimatorConfig(
        robot_mass=15.0,  # kg
        torso_inertia=jnp.array([0.5, 0.5, 0.3]),  # kg⋅m²
    )
    estimator = CentroidalStateEstimator(config)

    # Mock MJX data structure
    class MockData:
        def __init__(self):
            # subtree_com[1] is torso CoM in world frame
            self.subtree_com = jnp.array([
                [0.0, 0.0, 0.0],  # world origin
                [0.1, 0.05, 0.6],  # torso CoM
            ])
            # Velocity computed from position derivative (simplified)
            self.qvel = jnp.zeros(16)  # 10 joints + 6 base DOF

    # Mock observation (not used for CoM extraction, but needed for interface)
    obs = jnp.zeros(42)

    data = MockData()
    state, com_pos = estimator.estimate(obs, data)

    # Verify CoM extraction
    assert jnp.allclose(state.com_pos, jnp.array([0.1, 0.05, 0.6]))
    assert jnp.allclose(com_pos, jnp.array([0.1, 0.05, 0.6]))
    assert state.com_vel.shape == (3,)


def test_first_call_zero_velocity():
    """Test that first call returns zero velocity when prev_com_pos is None."""
    config = CentroidalStateEstimatorConfig(
        robot_mass=15.0,
        torso_inertia=jnp.array([0.5, 0.5, 0.3]),
    )
    estimator = CentroidalStateEstimator(config)

    class MockData:
        def __init__(self):
            self.subtree_com = jnp.array([
                [0.0, 0.0, 0.0],
                [0.1, 0.05, 0.6],
            ])
            self.qvel = jnp.zeros(16)

    obs = jnp.zeros(42)
    data = MockData()

    # First call with prev_com_pos=None should return zero velocity
    state, com_pos = estimator.estimate(obs, data, prev_com_pos=None)

    assert jnp.allclose(state.com_vel, jnp.zeros(3))
    assert jnp.allclose(com_pos, jnp.array([0.1, 0.05, 0.6]))


def test_velocity_computation_via_finite_difference():
    """Test that velocity is correctly computed via finite difference."""
    config = CentroidalStateEstimatorConfig(
        robot_mass=15.0,
        torso_inertia=jnp.array([0.5, 0.5, 0.3]),
    )
    estimator = CentroidalStateEstimator(config)

    class MockData:
        def __init__(self, com_pos):
            self.subtree_com = jnp.array([
                [0.0, 0.0, 0.0],
                com_pos,
            ])
            self.qvel = jnp.zeros(16)

    obs = jnp.zeros(42)

    # First call at t=0
    data1 = MockData(jnp.array([0.0, 0.0, 0.6]))
    state1, com_pos1 = estimator.estimate(obs, data1, prev_com_pos=None)

    # Verify first call returns zero velocity
    assert jnp.allclose(state1.com_vel, jnp.zeros(3))

    # Second call at t=0.01s with CoM moved by [0.02, 0.01, -0.01] m
    data2 = MockData(jnp.array([0.02, 0.01, 0.59]))
    state2, com_pos2 = estimator.estimate(obs, data2, prev_com_pos=com_pos1)

    # Expected velocity: delta_pos / dt = [0.02, 0.01, -0.01] / 0.01 = [2.0, 1.0, -1.0] m/s
    expected_vel = jnp.array([2.0, 1.0, -1.0])
    assert jnp.allclose(state2.com_vel, expected_vel, atol=1e-6)

    # Verify CoM position is correct
    assert jnp.allclose(state2.com_pos, jnp.array([0.02, 0.01, 0.59]))
    assert jnp.allclose(com_pos2, jnp.array([0.02, 0.01, 0.59]))


def test_contact_force_extraction():
    """Test contact force extraction from MJX contact data."""
    config = CentroidalStateEstimatorConfig(
        robot_mass=15.0,
        torso_inertia=jnp.array([0.5, 0.5, 0.3]),
    )
    estimator = CentroidalStateEstimator(config)

    # Mock MJX data with contact information
    class MockContact:
        def __init__(self):
            # Simulate 2 active contacts (left and right wheels)
            self.force = jnp.array([
                [0.0, 0.0, 75.0, 0.0, 0.0, 0.0],   # Left wheel: 75N normal force
                [0.0, 0.0, 80.0, 0.0, 0.0, 0.0],   # Right wheel: 80N normal force
            ])
            # geom1 and geom2 identify which geoms are in contact
            self.geom1 = jnp.array([5, 6])  # Wheel geom IDs
            self.geom2 = jnp.array([0, 0])  # Ground geom ID

    class MockData:
        def __init__(self):
            self.subtree_com = jnp.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.6],
            ])
            self.qvel = jnp.zeros(16)
            self.contact = MockContact()

    obs = jnp.zeros(42)
    data = MockData()

    state, _ = estimator.estimate(obs, data, prev_com_pos=None)

    # Verify contact extraction
    assert state.left_wheel_contact == True
    assert state.right_wheel_contact == True
    assert abs(state.left_wheel_force - 75.0) < 1e-6
    assert abs(state.right_wheel_force - 80.0) < 1e-6


MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


def make_model_data():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def test_wheel_geom_ids_are_resolved_by_name():
    model, _ = make_model_data()
    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=8.1,
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )
    assert estimator.left_wheel_geom_id == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision"
    )
    assert estimator.right_wheel_geom_id == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision"
    )


def test_reset_keyframe_detects_contact_but_marks_forward_force_invalid():
    model, data = make_model_data()
    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=sum(model.body_mass),
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )
    state, _ = estimator.estimate(jnp.zeros(42), data, None)
    assert state.left_wheel_contact
    assert state.right_wheel_contact
    assert not state.contact_force_valid
    assert state.left_wheel_force == 0.0
    assert state.right_wheel_force == 0.0
    assert state.total_contact_force_z == 0.0


def test_contact_force_is_valid_after_mj_step():
    model, data = make_model_data()
    mujoco.mj_step(model, data)
    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=sum(model.body_mass),
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )
    state, _ = estimator.estimate(jnp.zeros(42), data, None)
    assert state.left_wheel_contact
    assert state.right_wheel_contact
    assert state.contact_force_valid
    assert state.left_wheel_force > 0.0
    assert state.right_wheel_force > 0.0
    assert state.total_contact_force_z > 0.5 * sum(model.body_mass) * abs(model.opt.gravity[2])
