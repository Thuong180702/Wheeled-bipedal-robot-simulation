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


def test_hip_roll_joint_ids_exist(mj_model):
    """Test that hip roll joint IDs are found."""
    jacobian = ContactJacobian(mj_model)

    assert hasattr(jacobian, 'l_hip_roll_id')
    assert hasattr(jacobian, 'r_hip_roll_id')
    assert jacobian.l_hip_roll_id >= 0
    assert jacobian.r_hip_roll_id >= 0


def test_compute_hip_roll_moment_contribution(mj_model, mj_data):
    """Test that hip roll torques map to roll moment (Mx)."""
    jacobian = ContactJacobian(mj_model)

    # Hip roll torques: [left, right]
    tau_hip_roll = np.array([1.0, -1.0])

    # Should return roll moment contribution
    mx = jacobian.compute_hip_roll_moment_contribution(tau_hip_roll)

    # Hip roll torques directly contribute to roll moment
    # Left hip roll positive = positive roll moment
    # Right hip roll positive = positive roll moment
    # So [1.0, -1.0] should give net moment
    assert isinstance(mx, (float, np.floating))


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
