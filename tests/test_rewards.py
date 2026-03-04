"""
Tests cho reward functions.

Kiểm tra:
  - Phạm vi giá trị đúng
  - Gradient tính được (differentiable)
  - Edge cases
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.rewards.reward_functions import (
    compute_total_reward,
    penalty_action_rate,
    penalty_joint_torque,
    penalty_joint_velocity,
    reward_alive,
    reward_foot_clearance,
    reward_height,
    reward_tracking_velocity,
    reward_upright,
)


class TestRewardUpright:
    """Kiểm tra reward_upright."""

    def test_perfect_upright(self):
        """Quaternion [1,0,0,0] (đứng thẳng) → reward ~1.0."""
        quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        r = reward_upright(quat)
        assert float(r) > 0.95

    def test_fallen(self):
        """Quaternion nằm ngang → reward gần 0."""
        # Xoay 90° quanh y (nằm xuống)
        quat = jnp.array([0.707, 0.0, 0.707, 0.0])
        r = reward_upright(quat)
        assert float(r) < 0.2

    def test_range(self):
        """Reward phải trong [0, 1]."""
        for _ in range(20):
            quat = jax.random.normal(jax.random.PRNGKey(np.random.randint(1000)), (4,))
            quat = quat / jnp.linalg.norm(quat)
            r = reward_upright(quat)
            assert 0.0 <= float(r) <= 1.0


class TestRewardHeight:
    """Kiểm tra reward_height."""

    def test_at_target(self):
        """Ở đúng chiều cao mục tiêu → reward ~1.0."""
        r = reward_height(jnp.float32(0.65), target_height=0.65)
        assert float(r) > 0.95

    def test_far_from_target(self):
        """Xa mục tiêu → reward gần 0."""
        r = reward_height(jnp.float32(0.1), target_height=0.65)
        assert float(r) < 0.1


class TestRewardTrackingVelocity:
    """Kiểm tra velocity tracking reward."""

    def test_perfect_tracking(self):
        """Vận tốc khớp chính xác cmd → reward cao."""
        r = reward_tracking_velocity(
            base_vel_x=jnp.float32(1.0),
            base_vel_y=jnp.float32(0.0),
            base_ang_vel_z=jnp.float32(0.0),
            cmd_vel_x=jnp.float32(1.0),
            cmd_vel_y=jnp.float32(0.0),
            cmd_ang_vel_z=jnp.float32(0.0),
        )
        assert float(r) > 0.9

    def test_large_error(self):
        """Sai lệch lớn → reward thấp."""
        r = reward_tracking_velocity(
            base_vel_x=jnp.float32(-2.0),
            base_vel_y=jnp.float32(0.0),
            base_ang_vel_z=jnp.float32(-2.0),
            cmd_vel_x=jnp.float32(2.0),
            cmd_vel_y=jnp.float32(0.0),
            cmd_ang_vel_z=jnp.float32(2.0),
        )
        assert float(r) < 0.1


class TestPenalties:
    """Kiểm tra các penalty functions."""

    def test_zero_torque_zero_penalty(self):
        """Mô-men = 0 → penalty = 0."""
        p = penalty_joint_torque(jnp.zeros(10))
        assert float(p) == 0.0

    def test_positive_penalty(self):
        """Mô-men > 0 → penalty > 0."""
        p = penalty_joint_torque(jnp.ones(10) * 5.0)
        assert float(p) > 0.0

    def test_action_rate_zero(self):
        """Action không đổi → penalty = 0."""
        a = jnp.ones(10)
        p = penalty_action_rate(a, a)
        assert float(p) < 1e-6

    def test_action_rate_nonzero(self):
        """Action thay đổi → penalty > 0."""
        p = penalty_action_rate(jnp.ones(10), jnp.zeros(10))
        assert float(p) > 0.0


class TestComputeTotalReward:
    """Kiểm tra hàm tổng hợp reward."""

    def test_weighted_sum(self):
        """Tổng reward = weighted sum."""
        components = {
            "a": jnp.float32(1.0),
            "b": jnp.float32(2.0),
        }
        weights = {"a": 0.5, "b": 0.3}
        total = compute_total_reward(components, weights)
        expected = 0.5 * 1.0 + 0.3 * 2.0
        assert abs(float(total) - expected) < 1e-5

    def test_missing_weight_ignored(self):
        """Component không có weight → bị bỏ qua."""
        components = {"a": jnp.float32(100.0)}
        weights = {}
        total = compute_total_reward(components, weights)
        assert float(total) == 0.0


class TestDifferentiability:
    """Kiểm tra gradient tính được (cho RL)."""

    def test_upright_has_gradient(self):
        """reward_upright phải có gradient."""
        grad_fn = jax.grad(lambda q: reward_upright(q))
        quat = jnp.array([1.0, 0.01, 0.01, 0.0])
        quat = quat / jnp.linalg.norm(quat)
        grad = grad_fn(quat)
        assert not np.any(np.isnan(grad))

    def test_height_has_gradient(self):
        """reward_height phải có gradient."""
        grad_fn = jax.grad(lambda h: reward_height(h, 0.65))
        grad = grad_fn(jnp.float32(0.6))
        assert not np.isnan(float(grad))
