"""
Phase B.9 Step 5.19: Authority Reallocation Tests

Tests PID output clamping to reserve actuator headroom for WBC corrections.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from wheeled_biped.sim.low_level_control import hybrid_pid_plus_torque_control


def test_pid_authority_fraction_default():
    """Test backward compatibility: default pid_authority_fraction=1.0."""
    pid_ctrl = jnp.array([10.0, -10.0, 15.0, -15.0, 5.0, 10.0, -10.0, 15.0, -15.0, 5.0])
    residual = jnp.zeros(10)
    ctrl_min = jnp.full(10, -30.0)
    ctrl_max = jnp.full(10, 30.0)

    final, res = hybrid_pid_plus_torque_control(
        pid_ctrl, residual, ctrl_min, ctrl_max,
        max_ctrl_fraction=1.0,
        allow_mask=None,
        pid_authority_fraction=1.0,  # default
    )

    # With fraction=1.0, PID should not be clamped
    np.testing.assert_array_almost_equal(final, pid_ctrl)


def test_pid_authority_fraction_clamping():
    """Test PID output is clamped to reserved fraction."""
    # PID wants to output 20 Nm
    pid_ctrl = jnp.array([20.0, -20.0, 20.0, -20.0, 10.0, 20.0, -20.0, 20.0, -20.0, 10.0])
    residual = jnp.zeros(10)
    ctrl_min = jnp.full(10, -30.0)
    ctrl_max = jnp.full(10, 30.0)

    # Limit PID to 70% of range
    final, res = hybrid_pid_plus_torque_control(
        pid_ctrl, residual, ctrl_min, ctrl_max,
        max_ctrl_fraction=1.0,
        allow_mask=None,
        pid_authority_fraction=0.7,
    )

    # PID should be clamped to ±21 Nm (70% of ±30)
    expected_pid_clamped = jnp.array([20.0, -20.0, 20.0, -20.0, 10.0, 20.0, -20.0, 20.0, -20.0, 10.0])
    expected_pid_clamped = jnp.clip(expected_pid_clamped, ctrl_min * 0.7, ctrl_max * 0.7)

    np.testing.assert_array_almost_equal(final, expected_pid_clamped)


def test_pid_authority_fraction_reserves_headroom():
    """Test that clamping PID reserves headroom for WBC residuals."""
    # PID wants to saturate at 30 Nm
    pid_ctrl = jnp.array([30.0, -30.0, 30.0, -30.0, 15.0, 30.0, -30.0, 30.0, -30.0, 15.0])

    # WBC wants to add 5 Nm correction
    residual_normalized = jnp.array([0.5, -0.5, 0.5, -0.5, 0.25, 0.5, -0.5, 0.5, -0.5, 0.25])

    ctrl_min = jnp.full(10, -30.0)
    ctrl_max = jnp.full(10, 30.0)

    # Without clamping (fraction=1.0): PID saturates, WBC gets clipped
    final_no_clamp, _ = hybrid_pid_plus_torque_control(
        pid_ctrl, residual_normalized, ctrl_min, ctrl_max,
        max_ctrl_fraction=0.5,  # WBC uses 50% of range = ±15 Nm
        allow_mask=None,
        pid_authority_fraction=1.0,
    )

    # With clamping (fraction=0.7): PID limited to ±21, WBC can add up to ±15
    final_with_clamp, _ = hybrid_pid_plus_torque_control(
        pid_ctrl, residual_normalized, ctrl_min, ctrl_max,
        max_ctrl_fraction=0.5,
        allow_mask=None,
        pid_authority_fraction=0.7,
    )

    # Without clamping: PID=30, residual=7.5, final=30 (clipped, residual lost)
    # With clamping: PID=21, residual=7.5, final=28.5 (residual delivered)

    # Check that with clamping, more WBC authority is delivered
    assert np.abs(final_with_clamp[0]) < np.abs(final_no_clamp[0])


def test_pid_authority_fraction_bounds():
    """Test pid_authority_fraction is bounded to [0, 1]."""
    pid_ctrl = jnp.array([20.0] * 10)
    residual = jnp.zeros(10)
    ctrl_min = jnp.full(10, -30.0)
    ctrl_max = jnp.full(10, 30.0)

    # Test fraction > 1.0 is clamped to 1.0
    final_over, _ = hybrid_pid_plus_torque_control(
        pid_ctrl, residual, ctrl_min, ctrl_max,
        max_ctrl_fraction=1.0,
        allow_mask=None,
        pid_authority_fraction=1.5,
    )
    np.testing.assert_array_almost_equal(final_over, pid_ctrl)

    # Test fraction < 0.0 is clamped to 0.0
    final_under, _ = hybrid_pid_plus_torque_control(
        pid_ctrl, residual, ctrl_min, ctrl_max,
        max_ctrl_fraction=1.0,
        allow_mask=None,
        pid_authority_fraction=-0.5,
    )
    np.testing.assert_array_almost_equal(final_under, jnp.zeros(10))


def test_action_dimension_unchanged():
    """Test action dimension remains 10."""
    pid_ctrl = jnp.zeros(10)
    residual = jnp.zeros(10)
    ctrl_min = jnp.full(10, -30.0)
    ctrl_max = jnp.full(10, 30.0)

    final, res = hybrid_pid_plus_torque_control(
        pid_ctrl, residual, ctrl_min, ctrl_max,
        max_ctrl_fraction=1.0,
        allow_mask=None,
        pid_authority_fraction=0.7,
    )

    assert final.shape == (10,)
    assert res.shape == (10,)


def test_ctrlrange_respected():
    """Test final control respects ctrlrange limits."""
    # PID wants 25 Nm, WBC wants 10 Nm
    pid_ctrl = jnp.array([25.0] * 10)
    residual_normalized = jnp.array([0.5] * 10)

    ctrl_min = jnp.full(10, -30.0)
    ctrl_max = jnp.full(10, 30.0)

    final, _ = hybrid_pid_plus_torque_control(
        pid_ctrl, residual_normalized, ctrl_min, ctrl_max,
        max_ctrl_fraction=0.5,
        allow_mask=None,
        pid_authority_fraction=0.7,
    )

    # Final must respect ctrlrange
    assert np.all(final >= ctrl_min)
    assert np.all(final <= ctrl_max)


def test_backward_compatibility():
    """Test that omitting pid_authority_fraction maintains old behavior."""
    pid_ctrl = jnp.array([20.0] * 10)
    residual = jnp.zeros(10)
    ctrl_min = jnp.full(10, -30.0)
    ctrl_max = jnp.full(10, 30.0)

    # Old call signature (no pid_authority_fraction)
    final_old, _ = hybrid_pid_plus_torque_control(
        pid_ctrl, residual, ctrl_min, ctrl_max,
        max_ctrl_fraction=1.0,
        allow_mask=None,
    )

    # New call signature with default
    final_new, _ = hybrid_pid_plus_torque_control(
        pid_ctrl, residual, ctrl_min, ctrl_max,
        max_ctrl_fraction=1.0,
        allow_mask=None,
        pid_authority_fraction=1.0,
    )

    np.testing.assert_array_almost_equal(final_old, final_new)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
