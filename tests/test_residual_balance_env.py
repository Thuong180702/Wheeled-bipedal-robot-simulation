"""Tests for ResidualBalanceEnv."""

import jax
import jax.numpy as jnp
import pytest

from wheeled_biped.envs.residual_balance_env import ResidualBalanceEnv


@pytest.fixture
def residual_env():
    """Create a ResidualBalanceEnv for testing."""
    config = {
        "task": {
            "episode_length": 100,
            "initial_min_height": 0.68,
        },
        "residual": {
            "prior_config": "configs/controllers/gain_scheduled_lqr.yaml",
            "residual_scale": [0.10, 0.05, 0.20, 0.20, 0.40, 0.10, 0.05, 0.20, 0.20, 0.40],
        },
        "low_level_pid": {
            "enabled": True,
            "action_smoothing_alpha": 0.5,
            "disable_pid_action_bias": True,
        },
        "rewards": {
            "body_level": 1.5,
            "height": 2.0,
            "alive": 0.3,
            "residual_magnitude": -0.02,
            "residual_rate": -0.03,
            "residual_saturation": -0.05,
        },
        "termination": {
            "max_tilt_rad": 0.8,
            "min_height": 0.3,
        },
    }
    return ResidualBalanceEnv(config=config)


def test_residual_env_obs_size(residual_env):
    """Test that ResidualBalanceEnv has 52-dim observation."""
    assert residual_env.obs_size == 52, f"Expected obs_size=52, got {residual_env.obs_size}"


def test_residual_env_action_size(residual_env):
    """Test that ResidualBalanceEnv has 10-dim action."""
    assert residual_env.num_actions == 10, f"Expected num_actions=10, got {residual_env.num_actions}"


def test_residual_env_reset(residual_env):
    """Test that reset returns 52-dim observation with base_action_abs appended."""
    rng = jax.random.PRNGKey(42)
    state = residual_env.reset(rng)

    # Check observation shape
    assert state.obs.shape == (52,), f"Expected obs shape (52,), got {state.obs.shape}"

    # Check that base_action_abs is in info
    assert "base_action_abs" in state.info, "base_action_abs not in info"
    assert state.info["base_action_abs"].shape == (10,), \
        f"Expected base_action_abs shape (10,), got {state.info['base_action_abs'].shape}"

    # Check that last 10 dims of obs match base_action_abs
    base_action_from_obs = state.obs[42:]
    base_action_from_info = state.info["base_action_abs"]
    assert jnp.allclose(base_action_from_obs, base_action_from_info), \
        "Last 10 dims of obs do not match base_action_abs in info"


def test_residual_env_step(residual_env):
    """Test that step composes residual action correctly."""
    rng = jax.random.PRNGKey(42)
    state = residual_env.reset(rng)

    # Policy outputs residual_action
    residual_action = jnp.array([0.1, -0.1, 0.2, -0.2, 0.3, 0.1, -0.1, 0.2, -0.2, 0.3])

    # Step
    new_state = residual_env.step(state, residual_action)

    # Check observation shape
    assert new_state.obs.shape == (52,), f"Expected obs shape (52,), got {new_state.obs.shape}"

    # Check that all action components are logged
    assert "base_action_abs" in new_state.info, "base_action_abs not in info"
    assert "residual_action" in new_state.info, "residual_action not in info"
    assert "residual_scaled" in new_state.info, "residual_scaled not in info"
    assert "final_action_abs" in new_state.info, "final_action_abs not in info"
    assert "residual_norm" in new_state.info, "residual_norm not in info"
    assert "residual_saturation_rate" in new_state.info, "residual_saturation_rate not in info"

    # Check shapes
    assert new_state.info["base_action_abs"].shape == (10,)
    assert new_state.info["residual_action"].shape == (10,)
    assert new_state.info["residual_scaled"].shape == (10,)
    assert new_state.info["final_action_abs"].shape == (10,)

    # Check that residual_action matches input
    assert jnp.allclose(new_state.info["residual_action"], residual_action), \
        "residual_action in info does not match input"


def test_residual_env_no_nan_rollout(residual_env):
    """Test that a short rollout produces no NaNs."""
    rng = jax.random.PRNGKey(42)
    state = residual_env.reset(rng)

    for _ in range(10):
        # Random residual action
        rng, action_rng = jax.random.split(rng)
        residual_action = jax.random.uniform(action_rng, (10,), minval=-1.0, maxval=1.0)

        # Step
        state = residual_env.step(state, residual_action)

        # Check for NaNs
        assert not jnp.any(jnp.isnan(state.obs)), "NaN in observation"
        assert not jnp.isnan(state.reward), "NaN in reward"
        assert "base_action_abs" in state.info
        assert not jnp.any(jnp.isnan(state.info["base_action_abs"])), "NaN in base_action_abs"
        assert not jnp.any(jnp.isnan(state.info["final_action_abs"])), "NaN in final_action_abs"


def test_residual_env_base_action_in_obs(residual_env):
    """Test that base_action_abs is consistently appended to observation."""
    rng = jax.random.PRNGKey(42)
    state = residual_env.reset(rng)

    for _ in range(5):
        # Random residual action
        rng, action_rng = jax.random.split(rng)
        residual_action = jax.random.uniform(action_rng, (10,), minval=-1.0, maxval=1.0)

        # Step
        state = residual_env.step(state, residual_action)

        # Check that last 10 dims of obs match base_action_abs in info
        base_action_from_obs = state.obs[42:]
        base_action_from_info = state.info["base_action_abs"]
        assert jnp.allclose(base_action_from_obs, base_action_from_info, atol=1e-6), \
            f"Mismatch: obs[42:]={base_action_from_obs}, info={base_action_from_info}"


def test_residual_env_zero_residual_returns_base(residual_env):
    """Test that zero residual action produces zero residual_scaled."""
    rng = jax.random.PRNGKey(42)
    state = residual_env.reset(rng)

    # Zero residual action
    residual_action = jnp.zeros(10)

    # Step
    new_state = residual_env.step(state, residual_action)

    # Check that residual_scaled is zero (the key invariant)
    residual_scaled = new_state.info["residual_scaled"]
    assert jnp.allclose(residual_scaled, jnp.zeros(10), atol=1e-6), \
        f"Zero residual_action should give zero residual_scaled, got {residual_scaled}"

    # Check that residual_norm is near zero
    residual_norm = new_state.info["residual_norm"]
    assert residual_norm < 1e-5, f"Expected residual_norm ≈ 0, got {residual_norm}"


def test_residual_env_clipping(residual_env):
    """Test that final_action_abs is clipped to [-1, 1]."""
    rng = jax.random.PRNGKey(42)
    state = residual_env.reset(rng)

    # Large residual action to force clipping
    residual_action = jnp.ones(10) * 10.0  # Very large

    # Step
    new_state = residual_env.step(state, residual_action)

    # Check that final_action_abs is clipped
    final_action = new_state.info["final_action_abs"]
    assert jnp.all(final_action >= -1.0), f"final_action_abs below -1: {final_action}"
    assert jnp.all(final_action <= 1.0), f"final_action_abs above 1: {final_action}"


def test_residual_env_residual_scale(residual_env):
    """Test that residual_scale is applied correctly."""
    rng = jax.random.PRNGKey(42)
    state = residual_env.reset(rng)

    # Unit residual action
    residual_action = jnp.ones(10)

    # Step
    new_state = residual_env.step(state, residual_action)

    # Check that residual_scaled matches expected scale
    residual_scaled = new_state.info["residual_scaled"]
    expected_scale = jnp.array([0.10, 0.05, 0.20, 0.20, 0.40, 0.10, 0.05, 0.20, 0.20, 0.40])

    assert jnp.allclose(residual_scaled, expected_scale, atol=1e-6), \
        f"residual_scaled={residual_scaled}, expected={expected_scale}"


def test_residual_env_requires_pid_bias_disabled():
    """Test that ResidualBalanceEnv raises ValueError if disable_pid_action_bias is not true."""
    config_without_bias_disabled = {
        "task": {
            "episode_length": 100,
            "initial_min_height": 0.68,
        },
        "residual": {
            "prior_config": "configs/controllers/gain_scheduled_lqr.yaml",
            "residual_scale": [0.10, 0.05, 0.20, 0.20, 0.40, 0.10, 0.05, 0.20, 0.20, 0.40],
        },
        "low_level_pid": {
            "enabled": True,
            "action_smoothing_alpha": 0.5,
            "disable_pid_action_bias": False,  # Explicitly false
        },
        "rewards": {
            "body_level": 1.5,
            "height": 2.0,
            "alive": 0.3,
        },
        "termination": {
            "max_tilt_rad": 0.8,
            "min_height": 0.3,
        },
    }

    with pytest.raises(ValueError, match="ResidualBalanceEnv requires disable_pid_action_bias=true"):
        ResidualBalanceEnv(config=config_without_bias_disabled)


def test_residual_env_reward_uses_current_actions():
    """Test that reward computation uses current residual actions, not stale prev_state.info.

    This verifies that residual-specific reward terms (residual_magnitude, residual_rate,
    residual_saturation) are computed from the current step's residual actions, not from
    stale info carried over from the previous state.
    """
    rng = jax.random.PRNGKey(42)

    config = {
        "task": {
            "episode_length": 100,
            "initial_min_height": 0.68,
        },
        "residual": {
            "prior_config": "configs/controllers/gain_scheduled_lqr.yaml",
            "residual_scale": [0.10, 0.05, 0.20, 0.20, 0.40, 0.10, 0.05, 0.20, 0.20, 0.40],
        },
        "low_level_pid": {
            "enabled": True,
            "action_smoothing_alpha": 0.5,
            "disable_pid_action_bias": True,
        },
        "rewards": {
            "body_level": 1.5,
            "height": 2.0,
            "alive": 0.3,
            "residual_magnitude": -0.02,
            "residual_rate": -0.03,
            "residual_saturation": -0.05,
        },
        "termination": {
            "max_tilt_rad": 0.8,
            "min_height": 0.3,
        },
    }

    env = ResidualBalanceEnv(config=config)
    state = env.reset(rng)

    # Step 1: zero residual action
    residual_action_1 = jnp.zeros(10)
    state_1 = env.step(state, residual_action_1)

    # Step 2: large residual action
    residual_action_2 = jnp.ones(10) * 0.8
    state_2 = env.step(state_1, residual_action_2)

    # Verify that state_2.info contains the large residual action, not zeros
    assert jnp.allclose(state_2.info["residual_action"], residual_action_2, atol=1e-6), \
        "state_2.info should contain current residual_action, not stale zeros"

    # Verify that residual_norm reflects the large action (scaled)
    residual_norm_2 = state_2.info["residual_norm"]
    residual_scale = jnp.array([0.10, 0.05, 0.20, 0.20, 0.40, 0.10, 0.05, 0.20, 0.20, 0.40])
    expected_scaled_2 = residual_scale * residual_action_2
    expected_norm_2 = jnp.linalg.norm(expected_scaled_2)
    assert jnp.allclose(residual_norm_2, expected_norm_2, atol=1e-5), \
        f"residual_norm should reflect current scaled action: expected {expected_norm_2}, got {residual_norm_2}"

    # Step 3: back to zero residual action
    residual_action_3 = jnp.zeros(10)
    state_3 = env.step(state_2, residual_action_3)

    # Verify that state_3.info contains zeros, not the previous large action
    assert jnp.allclose(state_3.info["residual_action"], residual_action_3, atol=1e-6), \
        "state_3.info should contain current zero residual_action, not stale large action"

    # Verify that residual_norm is near zero
    residual_norm_3 = state_3.info["residual_norm"]
    assert residual_norm_3 < 1e-5, \
        f"residual_norm should be near zero for zero action: got {residual_norm_3}"

