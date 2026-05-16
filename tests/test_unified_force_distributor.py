import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from wheeled_biped.controllers.unified_force_distributor import UnifiedForceDistributor


@pytest.fixture
def mj_model():
    """Load robot model."""
    model_path = "assets/robot/wheeled_biped_real.xml"
    return mujoco.MjModel.from_xml_path(model_path)


def test_force_distributor_initialization(mj_model):
    """Test that UnifiedForceDistributor can be instantiated."""
    distributor = UnifiedForceDistributor(
        mj_model=mj_model,
        w_force=0.01,
        w_torque=0.1,
        w_smoothness=0.5,
        tau_hip_roll_max=10.0,
    )

    assert distributor.w_force == 0.01
    assert distributor.w_torque == 0.1
    assert distributor.w_smoothness == 0.5
    assert distributor.tau_hip_roll_max == 10.0


def test_distribute_wrench_signature(mj_model):
    """Test that distribute_wrench has correct signature."""
    distributor = UnifiedForceDistributor(mj_model=mj_model)

    # Create MuJoCo data
    mj_data = mujoco.MjData(mj_model)

    # Dummy inputs
    desired_wrench = np.array([0.0, 0.0, 147.0, 0.0, 0.0, 0.0])  # Just gravity compensation
    wheel_pos_left = np.array([0.135, 0.0, -0.3])
    wheel_pos_right = np.array([-0.135, 0.0, -0.3])

    # Should raise NotImplementedError (not yet implemented)
    with pytest.raises(NotImplementedError, match="QP solving not yet implemented"):
        distributor.distribute_wrench(
            mj_data, desired_wrench, wheel_pos_left, wheel_pos_right
        )


def test_prev_solution_initialization(mj_model):
    """Test that previous solution is initialized to zeros."""
    distributor = UnifiedForceDistributor(mj_model=mj_model)

    # Previous solution should be 8D zeros
    assert distributor.prev_solution.shape == (8,)
    assert jnp.allclose(distributor.prev_solution, jnp.zeros(8))
