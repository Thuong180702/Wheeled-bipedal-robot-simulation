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


@pytest.fixture
def mj_data(mj_model):
    """Create MuJoCo data."""
    return mujoco.MjData(mj_model)


@pytest.fixture
def wheel_positions():
    """Standard wheel positions relative to CoM."""
    wheel_pos_left = jnp.array([0.135, 0.0, -0.3])
    wheel_pos_right = jnp.array([-0.135, 0.0, -0.3])
    return wheel_pos_left, wheel_pos_right


def test_force_distributor_initialization(mj_model):
    """Test that UnifiedForceDistributor can be instantiated."""
    distributor = UnifiedForceDistributor(
        mj_model=mj_model,
        w_force=0.01,
        w_torque=0.1,
        w_smoothness=0.5,
        w_wrench=100.0,
        tau_hip_roll_max=10.0,
    )

    assert distributor.w_force == 0.01
    assert distributor.w_torque == 0.1
    assert distributor.w_smoothness == 0.5
    assert distributor.w_wrench == 100.0
    assert distributor.tau_hip_roll_max == 10.0


def test_prev_solution_initialization(mj_model):
    """Test that previous solution is initialized to zeros."""
    distributor = UnifiedForceDistributor(mj_model=mj_model)

    # Previous solution should be 8D zeros
    assert distributor.prev_solution.shape == (8,)
    assert jnp.allclose(distributor.prev_solution, jnp.zeros(8))


def test_build_cost_matrix_p(mj_model, mj_data, wheel_positions):
    """Test that cost matrix P has correct structure with soft constraints."""
    distributor = UnifiedForceDistributor(
        mj_model=mj_model,
        w_force=0.01,
        w_torque=0.1,
        w_wrench=100.0,
    )

    wheel_pos_left, wheel_pos_right = wheel_positions
    P = distributor._build_cost_matrix_p(mj_data, wheel_pos_left, wheel_pos_right)

    # Should be (8, 8) symmetric positive definite
    assert P.shape == (8, 8)

    # Should be symmetric
    assert jnp.allclose(P, P.T)

    # Diagonal should be positive (effort + soft wrench terms)
    assert jnp.all(jnp.diag(P) > 0)


def test_build_linear_cost_q(mj_model, mj_data, wheel_positions):
    """Test that linear cost q includes smoothness and soft wrench tracking."""
    distributor = UnifiedForceDistributor(
        mj_model=mj_model,
        w_force=0.01,
        w_torque=0.1,
        w_smoothness=0.5,
        w_wrench=100.0,
    )

    # Set previous solution
    x_prev = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    distributor.prev_solution = x_prev

    desired_wrench = jnp.array([0.0, 0.0, 147.0, 0.0, 0.0, 0.0])
    wheel_pos_left, wheel_pos_right = wheel_positions

    q = distributor._build_linear_cost_q(
        mj_data, desired_wrench, wheel_pos_left, wheel_pos_right
    )

    # Should be (8,) vector
    assert q.shape == (8,)

    # q should be non-zero (includes smoothness and wrench tracking terms)
    assert not jnp.allclose(q, jnp.zeros(8))


def test_linear_cost_q_zero_smoothness(mj_model, mj_data, wheel_positions):
    """Test that q only includes wrench tracking when smoothness weight is zero."""
    distributor = UnifiedForceDistributor(
        mj_model=mj_model,
        w_smoothness=0.0,
        w_wrench=100.0,
    )

    # Set non-zero previous solution
    distributor.prev_solution = jnp.ones(8)

    desired_wrench = jnp.array([0.0, 0.0, 147.0, 0.0, 0.0, 0.0])
    wheel_pos_left, wheel_pos_right = wheel_positions

    q = distributor._build_linear_cost_q(
        mj_data, desired_wrench, wheel_pos_left, wheel_pos_right
    )

    # Should be non-zero (wrench tracking term remains)
    assert not jnp.allclose(q, jnp.zeros(8))


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


def test_distribute_wrench_basic(mj_model, mj_data, wheel_positions):
    """Test basic wrench distribution with gravity compensation."""
    distributor = UnifiedForceDistributor(
        mj_model=mj_model,
        w_wrench=1000.0,  # High weight for accurate tracking
    )

    # Desired wrench: just gravity compensation (147N = 15kg * 9.81)
    desired_wrench = jnp.array([0.0, 0.0, 147.0, 0.0, 0.0, 0.0])

    wheel_pos_left, wheel_pos_right = wheel_positions

    # Distribute wrench
    f_left, f_right, tau_hip_roll = distributor.distribute_wrench(
        mj_data, desired_wrench, wheel_pos_left, wheel_pos_right
    )

    # Check shapes
    assert f_left.shape == (3,)
    assert f_right.shape == (3,)
    assert tau_hip_roll.shape == (2,)

    # Check vertical forces sum to ~147N (soft constraint, allow tolerance)
    total_fz = f_left[2] + f_right[2]
    assert total_fz == pytest.approx(147.0, abs=5.0)

    # Check forces are compressive (fz >= 0)
    assert f_left[2] >= -2e-6  # Allow small numerical error from QP solver
    assert f_right[2] >= -2e-6


def test_distribute_wrench_roll_moment(mj_model, mj_data, wheel_positions):
    """Test that roll moment is distributed to hip roll torques."""
    distributor = UnifiedForceDistributor(
        mj_model=mj_model,
        w_wrench=1000.0,
    )

    # Desired wrench: gravity + roll moment
    desired_wrench = jnp.array([0.0, 0.0, 147.0, 5.0, 0.0, 0.0])  # 5Nm roll

    wheel_pos_left, wheel_pos_right = wheel_positions

    f_left, f_right, tau_hip_roll = distributor.distribute_wrench(
        mj_data, desired_wrench, wheel_pos_left, wheel_pos_right
    )

    # Hip roll torques should be non-zero to generate roll moment
    assert jnp.abs(tau_hip_roll).sum() > 0.1

    # Total hip roll torque should contribute to roll moment
    # (soft constraint, so allow larger tolerance)
    assert tau_hip_roll[0] + tau_hip_roll[1] == pytest.approx(5.0, abs=2.0)


def test_warm_starting(mj_model, mj_data, wheel_positions):
    """Test that previous solution is updated after each solve."""
    distributor = UnifiedForceDistributor(mj_model=mj_model)

    desired_wrench = jnp.array([0.0, 0.0, 147.0, 0.0, 0.0, 0.0])
    wheel_pos_left, wheel_pos_right = wheel_positions

    # First solve
    f_left_1, f_right_1, tau_hip_roll_1 = distributor.distribute_wrench(
        mj_data, desired_wrench, wheel_pos_left, wheel_pos_right
    )

    # Check that prev_solution was updated
    expected_prev = jnp.concatenate([f_left_1, f_right_1, tau_hip_roll_1])
    assert jnp.allclose(distributor.prev_solution, expected_prev)

    # Second solve with same inputs
    f_left_2, f_right_2, tau_hip_roll_2 = distributor.distribute_wrench(
        mj_data, desired_wrench, wheel_pos_left, wheel_pos_right
    )

    # Solutions should be very similar (QP is deterministic with same inputs)
    # Allow slightly larger tolerance due to solver convergence variations
    assert jnp.allclose(f_left_1, f_left_2, atol=1e-2)
    assert jnp.allclose(f_right_1, f_right_2, atol=1e-2)


def test_soft_constraint_feasibility(mj_model, mj_data, wheel_positions):
    """Test that soft constraints make QP always feasible."""
    distributor = UnifiedForceDistributor(mj_model=mj_model)

    # Infeasible hard constraint scenario: large wrench with limited torque
    desired_wrench = jnp.array([50.0, 50.0, 200.0, 20.0, 20.0, 10.0])

    wheel_pos_left, wheel_pos_right = wheel_positions

    # Should not raise exception (soft constraints guarantee feasibility)
    f_left, f_right, tau_hip_roll = distributor.distribute_wrench(
        mj_data, desired_wrench, wheel_pos_left, wheel_pos_right
    )

    # Check shapes
    assert f_left.shape == (3,)
    assert f_right.shape == (3,)
    assert tau_hip_roll.shape == (2,)

    # Check constraints are satisfied (allow small numerical error from QP solver)
    assert f_left[2] >= -2e-6  # Compressive
    assert f_right[2] >= -2e-6  # Compressive
    assert jnp.abs(tau_hip_roll[0]) <= distributor.tau_hip_roll_max + 1e-6
    assert jnp.abs(tau_hip_roll[1]) <= distributor.tau_hip_roll_max + 1e-6
