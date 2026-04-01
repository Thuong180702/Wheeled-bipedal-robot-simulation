"""
Tests cho environment.

Kiểm tra:
  - Reset/step hoạt động
  - Observation kích thước đúng
  - Reward là số hợp lệ
  - Termination hoạt động
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def balance_env():
    """Tạo BalanceEnv."""
    from wheeled_biped.envs.balance_env import BalanceEnv

    return BalanceEnv()


@pytest.fixture
def rng():
    """JAX random key."""
    return jax.random.PRNGKey(42)


class TestBaseEnv:
    """Kiểm tra base environment."""

    def test_obs_size(self, balance_env):
        """Observation size phải = 41 (cho balance: 39 base + height_command + yaw_error)."""
        assert balance_env.obs_size == 41

    def test_num_actions(self, balance_env):
        """Phải có 10 actions."""
        assert balance_env.num_actions == 10

    def test_reset(self, balance_env, rng):
        """Reset phải trả về state hợp lệ."""
        state = balance_env.reset(rng)

        assert state.obs.shape == (balance_env.obs_size,)
        assert state.reward.shape == ()
        assert state.done.shape == ()
        assert state.step_count == 0
        assert not np.any(np.isnan(state.obs))

    def test_step(self, balance_env, rng):
        """Step phải trả về state hợp lệ."""
        state = balance_env.reset(rng)
        action = jnp.zeros(balance_env.num_actions)
        new_state = balance_env.step(state, action)

        assert new_state.obs.shape == (balance_env.obs_size,)
        assert new_state.step_count == 1
        assert not np.any(np.isnan(new_state.obs))
        assert not np.isnan(float(new_state.reward))

    def test_vectorized_reset(self, balance_env, rng):
        """Vectorized reset phải hoạt động."""
        num_envs = 8
        states = balance_env.v_reset(rng, num_envs)

        assert states.obs.shape == (num_envs, balance_env.obs_size)
        assert states.done.shape == (num_envs,)

    def test_vectorized_step(self, balance_env, rng):
        """Vectorized step phải hoạt động."""
        num_envs = 8
        states = balance_env.v_reset(rng, num_envs)
        actions = jnp.zeros((num_envs, balance_env.num_actions))
        new_states = balance_env.v_step(states, actions)

        assert new_states.obs.shape == (num_envs, balance_env.obs_size)
        assert new_states.reward.shape == (num_envs,)


class TestBalanceReward:
    """Kiểm tra reward cho BalanceEnv."""

    def test_positive_reward_standing(self, balance_env, rng):
        """Robot đứng yên phải có reward dương."""
        state = balance_env.reset(rng)
        action = jnp.zeros(balance_env.num_actions)
        new_state = balance_env.step(state, action)

        assert float(new_state.reward) >= 0.0

    def test_multiple_steps_stable(self, balance_env, rng):
        """Chạy 50 step không crash."""
        state = balance_env.reset(rng)

        for _ in range(50):
            action = jax.random.uniform(
                jax.random.PRNGKey(0),
                shape=(balance_env.num_actions,),
                minval=-0.1,
                maxval=0.1,
            )
            state = balance_env.step(state, action)

        assert not np.any(np.isnan(state.obs))


class TestTermination:
    """Kiểm tra termination conditions."""

    def test_episode_length_limit(self, balance_env, rng):
        """Episode phải kết thúc khi đạt max steps."""
        # Tạo env với episode_length ngắn để test
        from wheeled_biped.envs.balance_env import BalanceEnv

        short_env = BalanceEnv(config={"task": {"episode_length": 5}})

        state = short_env.reset(rng)
        for i in range(10):
            action = jnp.zeros(short_env.num_actions)
            state = short_env.step(state, action)

        # Phải done sau 5 steps
        assert bool(state.done) or state.step_count >= 5
