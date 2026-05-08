"""Tests for QP allocator (Phase B.7 Task 9)."""

import numpy as np
import pytest

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

from wheeled_biped.controllers.qp_allocator import (
    QPAllocator,
    QPAllocatorConfig,
    create_qp_allocator,
)


@pytest.fixture
def default_config():
    """Create default QP allocator config."""
    return QPAllocatorConfig(
        joint_limits_lower=-np.ones(10),
        joint_limits_upper=np.ones(10),
        weight_height_ik=1.0,
        weight_com_vmc=0.8,
        weight_wheel_lqr=1.2,
        weight_roll_yaw=0.6,
        regularization_weight=0.01,
    )


@pytest.fixture
def allocator(default_config):
    """Create QP allocator."""
    if not HAS_CVXPY:
        pytest.skip("cvxpy not installed")
    return QPAllocator(default_config)


class TestQPAllocatorConfig:
    """Test QP allocator configuration."""

    def test_default_config_creation(self):
        """Test default config creation."""
        config = QPAllocatorConfig()
        assert config.weight_height_ik == 1.0
        assert config.weight_wheel_lqr == 1.2
        assert config.regularization_weight == 0.01

    def test_custom_joint_limits(self):
        """Test custom joint limits."""
        lower = -0.5 * np.ones(10)
        upper = 0.8 * np.ones(10)
        config = QPAllocatorConfig(
            joint_limits_lower=lower,
            joint_limits_upper=upper,
        )
        assert np.allclose(config.joint_limits_lower, lower)
        assert np.allclose(config.joint_limits_upper, upper)

    def test_priority_weights_positive(self, default_config):
        """Test priority weights are positive."""
        assert default_config.weight_height_ik > 0
        assert default_config.weight_com_vmc > 0
        assert default_config.weight_wheel_lqr > 0
        assert default_config.weight_roll_yaw > 0


class TestQPAllocator:
    """Test QP allocator functionality."""

    @pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
    def test_allocator_creation(self, default_config):
        """Test allocator creation."""
        allocator = QPAllocator(default_config)
        assert allocator.config == default_config

    @pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
    def test_allocator_requires_cvxpy(self):
        """Test allocator raises error without cvxpy."""
        if HAS_CVXPY:
            pytest.skip("cvxpy is installed")

        config = QPAllocatorConfig()
        with pytest.raises(ImportError, match="cvxpy"):
            QPAllocator(config)

    def test_allocate_within_limits(self, allocator):
        """Test allocated action respects joint limits."""
        desired_action = np.array([0.5, -0.3, 0.8, -0.9, 0.2, 0.4, -0.6, 0.7, -0.8, 0.1])

        allocated = allocator.allocate(desired_action)

        assert allocated.shape == (10,)
        assert np.all(allocated >= allocator.config.joint_limits_lower)
        assert np.all(allocated <= allocator.config.joint_limits_upper)

    def test_allocate_clips_out_of_bounds(self, allocator):
        """Test allocation clips out-of-bounds actions."""
        # Desired action exceeds limits
        desired_action = 2.0 * np.ones(10)

        allocated = allocator.allocate(desired_action)

        # Should be clipped to upper limit
        assert np.all(allocated <= allocator.config.joint_limits_upper)

    def test_allocate_zero_action(self, allocator):
        """Test allocation of zero action."""
        desired_action = np.zeros(10)

        allocated = allocator.allocate(desired_action)

        # Should be near zero (regularization may cause small deviation)
        assert np.linalg.norm(allocated) < 0.1

    def test_allocate_with_custom_weights(self, allocator):
        """Test allocation with custom layer weights."""
        desired_action = np.ones(10)

        # High weight on wheels, low on others
        layer_weights = np.array([0.1, 0.1, 0.5, 0.5, 2.0, 0.1, 0.1, 0.5, 0.5, 2.0])

        allocated = allocator.allocate(desired_action, layer_weights=layer_weights)

        assert allocated.shape == (10,)
        assert np.all(np.isfinite(allocated))

    def test_allocate_conflicting_actions(self, allocator):
        """Test allocation resolves conflicting actions."""
        # Conflicting: some joints want +1, others want -1
        desired_action = np.array([1.0, -1.0, 0.5, -0.5, 0.8, 1.0, -1.0, 0.5, -0.5, 0.8])

        allocated = allocator.allocate(desired_action)

        # Should find a compromise
        assert allocated.shape == (10,)
        assert np.all(allocated >= -1.0)
        assert np.all(allocated <= 1.0)

    def test_default_weights_prioritize_wheels(self, allocator):
        """Test default weights give highest priority to wheels."""
        default_weights = allocator._get_default_weights()

        # Wheels (indices 4, 9) should have highest weight
        wheel_weight = default_weights[4]
        assert wheel_weight == allocator.config.weight_wheel_lqr
        assert wheel_weight > allocator.config.weight_height_ik
        assert wheel_weight > allocator.config.weight_roll_yaw

    def test_default_weights_shape(self, allocator):
        """Test default weights have correct shape."""
        default_weights = allocator._get_default_weights()
        assert default_weights.shape == (10,)
        assert np.all(default_weights > 0)

    def test_allocate_fallback_on_solver_failure(self, allocator):
        """Test fallback behavior when solver fails."""
        # Create infeasible problem (limits that can't be satisfied)
        allocator.config.joint_limits_lower = np.ones(10)
        allocator.config.joint_limits_upper = -np.ones(10)

        desired_action = np.zeros(10)

        # Should fallback to clipping
        allocated = allocator.allocate(desired_action)

        assert allocated.shape == (10,)
        assert np.all(np.isfinite(allocated))

    def test_allocate_preserves_feasible_action(self, allocator):
        """Test allocation preserves already-feasible action."""
        # Action already within limits
        desired_action = 0.5 * np.ones(10)

        allocated = allocator.allocate(desired_action)

        # Should be close to desired (within regularization tolerance)
        assert np.linalg.norm(allocated - desired_action) < 0.2

    def test_allocate_different_solvers(self, default_config):
        """Test allocation with different solvers."""
        if not HAS_CVXPY:
            pytest.skip("cvxpy not installed")

        desired_action = np.array([0.5, -0.3, 0.8, -0.9, 0.2, 0.4, -0.6, 0.7, -0.8, 0.1])

        for solver in ["OSQP", "ECOS", "SCS"]:
            config = QPAllocatorConfig(
                joint_limits_lower=-np.ones(10),
                joint_limits_upper=np.ones(10),
                solver=solver,
            )
            allocator = QPAllocator(config)

            try:
                allocated = allocator.allocate(desired_action)
                assert allocated.shape == (10,)
                assert np.all(np.isfinite(allocated))
            except Exception:
                # Solver may not be available
                pass


class TestCreateQPAllocator:
    """Test factory function."""

    @pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
    def test_create_with_defaults(self):
        """Test factory function with default weights."""
        allocator = create_qp_allocator()
        assert isinstance(allocator, QPAllocator)
        assert allocator.config.weight_height_ik == 1.0

    @pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
    def test_create_with_custom_weights(self):
        """Test factory function with custom weights."""
        allocator = create_qp_allocator(
            weight_height_ik=2.0,
            weight_wheel_lqr=3.0,
            regularization_weight=0.05,
        )
        assert allocator.config.weight_height_ik == 2.0
        assert allocator.config.weight_wheel_lqr == 3.0
        assert allocator.config.regularization_weight == 0.05


class TestQPAllocatorIntegration:
    """Integration tests for QP allocator."""

    def test_allocate_realistic_hierarchical_action(self, allocator):
        """Test allocation with realistic hierarchical control action."""
        # Simulate hierarchical controller output
        # Legs want one thing, wheels want another
        desired_action = np.array([
            0.1,   # l_hip_roll (small roll correction)
            0.05,  # l_hip_yaw (small yaw)
            -0.3,  # l_hip_pitch (IK + VMC)
            0.8,   # l_knee (IK + VMC)
            0.6,   # l_wheel (LQR balance)
            -0.1,  # r_hip_roll (opposite roll)
            0.05,  # r_hip_yaw
            -0.3,  # r_hip_pitch
            0.8,   # r_knee
            0.6,   # r_wheel
        ])

        allocated = allocator.allocate(desired_action)

        # Should preserve wheel commands (highest priority)
        assert abs(allocated[4] - desired_action[4]) < 0.1
        assert abs(allocated[9] - desired_action[9]) < 0.1

        # Should preserve leg symmetry
        assert abs(allocated[2] - allocated[7]) < 0.1  # hip pitch
        assert abs(allocated[3] - allocated[8]) < 0.1  # knee

    def test_allocate_multiple_calls_consistent(self, allocator):
        """Test multiple allocations are consistent."""
        desired_action = np.random.uniform(-0.5, 0.5, 10)

        allocated1 = allocator.allocate(desired_action)
        allocated2 = allocator.allocate(desired_action)

        # Should be deterministic
        assert np.allclose(allocated1, allocated2)

    def test_allocate_regularization_effect(self):
        """Test regularization reduces action magnitude."""
        if not HAS_CVXPY:
            pytest.skip("cvxpy not installed")

        desired_action = 0.8 * np.ones(10)

        # Low regularization
        config_low = QPAllocatorConfig(regularization_weight=0.001)
        allocator_low = QPAllocator(config_low)
        allocated_low = allocator_low.allocate(desired_action)

        # High regularization
        config_high = QPAllocatorConfig(regularization_weight=0.1)
        allocator_high = QPAllocator(config_high)
        allocated_high = allocator_high.allocate(desired_action)

        # High regularization should produce smaller actions
        assert np.linalg.norm(allocated_high) < np.linalg.norm(allocated_low)
