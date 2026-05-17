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


def test_build_cost_matrix_p(mj_model):
    """Test that cost matrix P has correct structure."""
    distributor = UnifiedForceDistributor(
        mj_model=mj_model,
        w_force=0.01,
        w_torque=0.1,
    )

    P = distributor._build_cost_matrix_p()

    # Should be (8, 8) diagonal
    assert P.shape == (8, 8)

    # Check diagonal values
    # First 6 elements: w_force for wheel forces
    assert P[0, 0] == pytest.approx(0.01)
    assert P[1, 1] == pytest.approx(0.01)
    assert P[2, 2] == pytest.approx(0.01)
    assert P[3, 3] == pytest.approx(0.01)
    assert P[4, 4] == pytest.approx(0.01)
    assert P[5, 5] == pytest.approx(0.01)

    # Last 2 elements: w_torque for hip roll torques
    assert P[6, 6] == pytest.approx(0.1)
    assert P[7, 7] == pytest.approx(0.1)

    # Off-diagonal should be zero
    assert jnp.allclose(P - jnp.diag(jnp.diag(P)), 0.0)


def test_build_linear_cost_q(mj_model):
    """Test that linear cost q implements smoothness penalty."""
    distributor = UnifiedForceDistributor(
        mj_model=mj_model,
        w_force=0.01,
        w_torque=0.1,
        w_smoothness=0.5,
    )

    # Set previous solution
    x_prev = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    distributor.prev_solution = x_prev

    q = distributor._build_linear_cost_q()

    # Should be (8,) vector
    assert q.shape == (8,)

    # q = -2 * w_smoothness * P @ x_prev
    P = distributor._build_cost_matrix_p()
    expected_q = -2.0 * 0.5 * (P @ x_prev)

    assert jnp.allclose(q, expected_q)


def test_linear_cost_q_zero_smoothness(mj_model):
    """Test that q is zero when smoothness weight is zero."""
    distributor = UnifiedForceDistributor(
        mj_model=mj_model,
        w_smoothness=0.0,
    )

    # Set non-zero previous solution
    distributor.prev_solution = jnp.ones(8)

    q = distributor._build_linear_cost_q()

    # Should be all zeros when w_smoothness = 0
    assert jnp.allclose(q, jnp.zeros(8))


def test_build_equality_constraints(mj_model):
    """Test that equality constraint matrices are built correctly."""
    distributor = UnifiedForceDistributor(mj_model=mj_model)

    # Create MuJoCo data
    mj_data = mujoco.MjData(mj_model)

    # Desired wrench
    desired_wrench = jnp.array([10.0, 5.0, 147.0, 2.0, 3.0, 1.0])

    # Wheel positions
    wheel_pos_left = jnp.array([0.135, 0.0, -0.3])
    wheel_pos_right = jnp.array([-0.135, 0.0, -0.3])

    A_eq, b_eq = distributor._build_equality_constraints(
        mj_data, desired_wrench, wheel_pos_left, wheel_pos_right
    )

    # A_eq should be wrench matrix (6, 8)
    assert A_eq.shape == (6, 8)

    # b_eq should be desired wrench (6,)
    assert b_eq.shape == (6,)
    assert jnp.allclose(b_eq, desired_wrench)


def test_build_inequality_bounds(mj_model):
    """Test that inequality constraint bounds are built correctly."""
    distributor = UnifiedForceDistributor(
        mj_model=mj_model,
        tau_hip_roll_max=10.0,
    )

    lower, upper = distributor._build_inequality_bounds()

    # Should be (8,) vectors
    assert lower.shape == (8,)
    assert upper.shape == (8,)

    # Contact forces: fz >= 0 (compressive), fx/fy unbounded
    # First 6 elements (wheel forces): lower = -inf for x/y, 0 for z
    assert lower[0] == pytest.approx(-jnp.inf)  # f_left_x
    assert lower[1] == pytest.approx(-jnp.inf)  # f_left_y
    assert lower[2] == pytest.approx(0.0)       # f_left_z (compressive)
    assert lower[3] == pytest.approx(-jnp.inf)  # f_right_x
    assert lower[4] == pytest.approx(-jnp.inf)  # f_right_y
    assert lower[5] == pytest.approx(0.0)       # f_right_z (compressive)

    # Hip roll torques: -tau_max <= tau <= tau_max
    assert lower[6] == pytest.approx(-10.0)
    assert lower[7] == pytest.approx(-10.0)

    # Upper bounds: inf for forces, tau_max for torques
    assert upper[0] == pytest.approx(jnp.inf)
    assert upper[1] == pytest.approx(jnp.inf)
    assert upper[2] == pytest.approx(jnp.inf)
    assert upper[3] == pytest.approx(jnp.inf)
    assert upper[4] == pytest.approx(jnp.inf)
    assert upper[5] == pytest.approx(jnp.inf)
    assert upper[6] == pytest.approx(10.0)
    assert upper[7] == pytest.approx(10.0)
