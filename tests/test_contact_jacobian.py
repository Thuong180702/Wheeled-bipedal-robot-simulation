import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from wheeled_biped.controllers.contact_jacobian import ContactJacobian


@pytest.fixture
def mj_model():
    """Load robot model."""
    model_path = "assets/robot/wheeled_biped_real.xml"
    return mujoco.MjModel.from_xml_path(model_path)


@pytest.fixture
def mj_data(mj_model):
    """Create MuJoCo data."""
    return mujoco.MjData(mj_model)


def test_compute_hip_roll_moment_contribution(mj_model, mj_data):
    """Test that hip roll torques map to roll moment (Mx)."""
    jacobian = ContactJacobian(mj_model)

    tau_hip_roll = np.array([-12.5, 12.5])

    mx = jacobian.compute_hip_roll_moment_contribution(tau_hip_roll)

    assert isinstance(mx, (float, np.floating))
    assert mx == pytest.approx(25.0, abs=1e-6)


def test_map_forces_with_hip_roll_torques(mj_model, mj_data):
    """Test mapping contact forces + hip roll torques to joint torques."""
    jacobian = ContactJacobian(mj_model)

    # Zero contact forces
    f_left = np.zeros(3)
    f_right = np.zeros(3)

    # Non-zero hip roll torques
    tau_hip_roll = np.array([1.0, 2.0])

    # Map to joint torques
    tau = jacobian.map_contact_forces_to_torques(
        mj_data, f_left, f_right, tau_hip_roll=tau_hip_roll
    )

    # Should return 10D joint torques
    assert tau.shape == (10,)

    # Hip roll joints should have the commanded torques
    # l_hip_roll is joint 0, r_hip_roll is joint 5
    assert tau[0] == pytest.approx(1.0, abs=1e-6)
    assert tau[5] == pytest.approx(2.0, abs=1e-6)


def test_build_wrench_matrix_dimensions(mj_model, mj_data):
    """Test that wrench matrix has correct dimensions."""
    jacobian = ContactJacobian(mj_model)

    # Compute wheel positions (dummy values for now)
    wheel_pos_left = np.array([0.135, 0.0, 0.0])
    wheel_pos_right = np.array([-0.135, 0.0, 0.0])

    # Build wrench matrix
    A_wrench = jacobian.build_wrench_matrix(
        mj_data, wheel_pos_left, wheel_pos_right
    )

    # Should be (6, 8): 6 wrench components, 8 decision variables
    assert A_wrench.shape == (6, 8)


def test_wrench_matrix_force_mapping(mj_model, mj_data):
    """Test that wrench matrix correctly maps forces."""
    jacobian = ContactJacobian(mj_model)

    # Wheel positions at CoM height (z=0 relative to CoM)
    wheel_pos_left = np.array([0.135, 0.0, 0.0])
    wheel_pos_right = np.array([-0.135, 0.0, 0.0])

    A_wrench = jacobian.build_wrench_matrix(
        mj_data, wheel_pos_left, wheel_pos_right
    )

    # Test case: vertical forces only
    decision_vars = np.array([
        0.0, 0.0, 50.0,  # f_left: [0, 0, 50N]
        0.0, 0.0, 50.0,  # f_right: [0, 0, 50N]
        0.0, 0.0         # tau_hip_roll: [0, 0]
    ])

    wrench = A_wrench @ decision_vars

    # Expected: Fz = 100N, all other components = 0
    assert wrench[0] == pytest.approx(0.0, abs=1e-6)  # Fx
    assert wrench[1] == pytest.approx(0.0, abs=1e-6)  # Fy
    assert wrench[2] == pytest.approx(100.0, abs=1e-6)  # Fz
    assert wrench[3] == pytest.approx(0.0, abs=1e-6)  # Mx
    assert wrench[4] == pytest.approx(0.0, abs=1e-6)  # My
    assert wrench[5] == pytest.approx(0.0, abs=1e-6)  # Mz


def test_wrench_matrix_hip_roll_contribution(mj_model, mj_data):
    """Test that hip roll torques contribute to roll moment."""
    jacobian = ContactJacobian(mj_model)

    wheel_pos_left = np.array([0.135, 0.0, 0.0])
    wheel_pos_right = np.array([-0.135, 0.0, 0.0])

    A_wrench = jacobian.build_wrench_matrix(
        mj_data, wheel_pos_left, wheel_pos_right
    )

    decision_vars = np.array([
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        -12.5, 12.5,
    ])

    wrench = A_wrench @ decision_vars

    assert wrench[0] == pytest.approx(0.0, abs=1e-6)  # Fx
    assert wrench[1] == pytest.approx(0.0, abs=1e-6)  # Fy
    assert wrench[2] == pytest.approx(0.0, abs=1e-6)  # Fz
    assert wrench[3] == pytest.approx(25.0, abs=1e-6)  # Mx
    assert wrench[4] == pytest.approx(0.0, abs=1e-6)  # My
    assert wrench[5] == pytest.approx(0.0, abs=1e-6)  # Mz


def test_wrench_matrix_vertical_force_asymmetry_creates_roll_moment(mj_model, mj_data):
    jacobian = ContactJacobian(mj_model)

    wheel_pos_left = np.array([0.135, 0.0, 0.0])
    wheel_pos_right = np.array([-0.135, 0.0, 0.0])

    A_wrench = jacobian.build_wrench_matrix(
        mj_data, wheel_pos_left, wheel_pos_right
    )

    decision_vars = np.array([
        0.0, 0.0, 55.0,
        0.0, 0.0, 25.0,
        0.0, 0.0,
    ])

    wrench = A_wrench @ decision_vars

    assert wrench[2] == pytest.approx(80.0, abs=1e-6)
    assert wrench[3] == pytest.approx(4.05, abs=1e-6)
    assert wrench[4] == pytest.approx(0.0, abs=1e-6)


def test_wrench_matrix_pitch_moment(mj_model, mj_data):
    """Test that forward forces at wheel height create pitch moment."""
    jacobian = ContactJacobian(mj_model)

    # Wheels below CoM (negative z)
    wheel_pos_left = np.array([0.135, 0.0, -0.3])
    wheel_pos_right = np.array([-0.135, 0.0, -0.3])

    A_wrench = jacobian.build_wrench_matrix(
        mj_data, wheel_pos_left, wheel_pos_right
    )

    # Test case: forward forces
    decision_vars = np.array([
        10.0, 0.0, 0.0,  # f_left: [10N forward, 0, 0]
        10.0, 0.0, 0.0,  # f_right: [10N forward, 0, 0]
        0.0, 0.0         # tau_hip_roll: zero
    ])

    wrench = A_wrench @ decision_vars

    # Expected: Fx = 20N, My = r_z * Fx = -0.3 * 20 = -6 Nm
    assert wrench[0] == pytest.approx(20.0, abs=1e-6)  # Fx
    assert wrench[4] == pytest.approx(-6.0, abs=1e-6)  # My


def make_reset_model_data():
    model = mujoco.MjModel.from_xml_path("assets/robot/wheeled_biped_real.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def test_wheel_jacobians_have_expected_shape():
    model, data = make_reset_model_data()
    contact_jacobian = ContactJacobian(model)
    j_left, j_right = contact_jacobian.compute_wheel_jacobians(data)
    assert j_left.shape == (3, 10)
    assert j_right.shape == (3, 10)


def test_symmetric_upward_force_maps_to_nonzero_leg_torque():
    model, data = make_reset_model_data()
    contact_jacobian = ContactJacobian(model)
    tau = contact_jacobian.map_contact_forces_to_torques(
        data,
        jnp.array([0.0, 0.0, 40.0]),
        jnp.array([0.0, 0.0, 40.0]),
        jnp.array([0.0, 0.0]),
    )
    assert tau.shape == (10,)
    assert abs(float(tau[3])) > 1.0
    assert abs(float(tau[8])) > 1.0
    assert np.isfinite(np.array(tau)).all()


def test_force_mapping_diagnostics_include_jacobian_and_torque_terms():
    model, data = make_reset_model_data()
    contact_jacobian = ContactJacobian(model)
    diagnostics = contact_jacobian.compute_force_mapping_diagnostics(
        data,
        jnp.array([0.0, 0.0, 40.0]),
        jnp.array([0.0, 0.0, 40.0]),
    )
    assert diagnostics["left_jacobian_z_row"].shape == (10,)
    assert diagnostics["right_jacobian_z_row"].shape == (10,)
    assert diagnostics["tau_left_from_force"].shape == (10,)
    assert diagnostics["tau_right_from_force"].shape == (10,)
    assert diagnostics["tau_total_from_force"].shape == (10,)
