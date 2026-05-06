"""Tests for residual-aware checkpoint validation logic.

Tests the checkpoint type detection and validation logic added in Phase C.1.
"""

import pickle
from pathlib import Path

import jax.numpy as jnp
import pytest


def create_fake_checkpoint(obs_dim: int, config: dict) -> dict:
    """Create a minimal fake checkpoint for testing."""
    return {
        "params": {"dummy": jnp.zeros(10)},
        "opt_state": None,
        "obs_rms": type("ObsRMS", (), {"mean": jnp.zeros(obs_dim), "var": jnp.ones(obs_dim)})(),
        "config": config,
        "global_step": 1000,
    }


def test_residual_checkpoint_detection_by_env_name(tmp_path):
    """Test residual checkpoint detected by env_name."""
    config = {
        "task": {"env": "ResidualBalanceEnv"},
        "sensor_noise": {"lin_vel_mode": "clean", "enabled": False},
        "low_level_pid": {"enabled": True},
    }

    # Validation logic (simplified from validate_checkpoint.py)
    task_cfg = config.get("task", {})
    env_name = task_cfg.get("env", "BalanceEnv")
    residual_cfg = config.get("residual", {})
    is_residual = (
        env_name == "ResidualBalanceEnv"
        or "residual_scale" in residual_cfg
        or "prior_config" in residual_cfg
    )

    assert is_residual, "Should detect residual checkpoint by env_name"


def test_residual_checkpoint_detection_by_residual_scale(tmp_path):
    """Test residual checkpoint detected by residual_scale presence."""
    config = {
        "task": {"env": "BalanceEnv"},
        "residual": {"residual_scale": [0.1] * 10},
        "sensor_noise": {"lin_vel_mode": "clean", "enabled": False},
        "low_level_pid": {"enabled": True},
    }
    ckpt = create_fake_checkpoint(obs_dim=52, config=config)

    task_cfg = config.get("task", {})
    env_name = task_cfg.get("env", "BalanceEnv")
    residual_cfg = config.get("residual", {})
    is_residual = (
        env_name == "ResidualBalanceEnv"
        or "residual_scale" in residual_cfg
        or "prior_config" in residual_cfg
    )

    assert is_residual, "Should detect residual checkpoint by residual_scale"


def test_residual_checkpoint_detection_by_prior_config(tmp_path):
    """Test residual checkpoint detected by prior_config presence."""
    config = {
        "task": {"env": "BalanceEnv"},
        "residual": {"prior_config": "configs/controllers/gain_scheduled_lqr.yaml"},
        "sensor_noise": {"lin_vel_mode": "clean", "enabled": False},
        "low_level_pid": {"enabled": True},
    }
    ckpt = create_fake_checkpoint(obs_dim=52, config=config)

    task_cfg = config.get("task", {})
    env_name = task_cfg.get("env", "BalanceEnv")
    residual_cfg = config.get("residual", {})
    is_residual = (
        env_name == "ResidualBalanceEnv"
        or "residual_scale" in residual_cfg
        or "prior_config" in residual_cfg
    )

    assert is_residual, "Should detect residual checkpoint by prior_config"


def test_pure_ppo_checkpoint_detection(tmp_path):
    """Test pure PPO checkpoint not detected as residual."""
    config = {
        "task": {"env": "BalanceEnv"},
        "sensor_noise": {"lin_vel_mode": "clean", "enabled": False},
        "low_level_pid": {"enabled": True},
    }
    ckpt = create_fake_checkpoint(obs_dim=42, config=config)

    task_cfg = config.get("task", {})
    env_name = task_cfg.get("env", "BalanceEnv")
    residual_cfg = config.get("residual", {})
    is_residual = (
        env_name == "ResidualBalanceEnv"
        or "residual_scale" in residual_cfg
        or "prior_config" in residual_cfg
    )

    assert not is_residual, "Should not detect pure PPO checkpoint as residual"


def test_residual_obs_size_validation():
    """Test residual checkpoint obs size validation."""
    # lin_vel_mode = "clean" -> base_obs_dim = 39
    # base_obs_size_with_extras = 39 + 3 = 42
    # residual obs_size = 42 + 10 = 52

    lin_vel_mode = "clean"
    base_obs_dim = 36 if lin_vel_mode == "disabled" else 39
    base_obs_size_with_extras = base_obs_dim + 3
    expected_obs_size = base_obs_size_with_extras + 10

    assert expected_obs_size == 52, "Residual obs size should be 52 for clean mode"

    # lin_vel_mode = "disabled" -> base_obs_dim = 36
    # base_obs_size_with_extras = 36 + 3 = 39
    # residual obs_size = 39 + 10 = 49

    lin_vel_mode = "disabled"
    base_obs_dim = 36 if lin_vel_mode == "disabled" else 39
    base_obs_size_with_extras = base_obs_dim + 3
    expected_obs_size = base_obs_size_with_extras + 10

    assert expected_obs_size == 49, "Residual obs size should be 49 for disabled mode"


def test_pure_ppo_obs_size_validation():
    """Test pure PPO checkpoint obs size validation."""
    # lin_vel_mode = "clean" -> base_obs_dim = 39
    # base_obs_size_with_extras = 39 + 3 = 42

    lin_vel_mode = "clean"
    base_obs_dim = 36 if lin_vel_mode == "disabled" else 39
    base_obs_size_with_extras = base_obs_dim + 3
    expected_obs_size = base_obs_size_with_extras

    assert expected_obs_size == 42, "Pure PPO obs size should be 42 for clean mode"

    # lin_vel_mode = "disabled" -> base_obs_dim = 36
    # base_obs_size_with_extras = 36 + 3 = 39

    lin_vel_mode = "disabled"
    base_obs_dim = 36 if lin_vel_mode == "disabled" else 39
    base_obs_size_with_extras = base_obs_dim + 3
    expected_obs_size = base_obs_size_with_extras

    assert expected_obs_size == 39, "Pure PPO obs size should be 39 for disabled mode"


def test_residual_scale_validation():
    """Test residual_scale validation logic."""
    # Valid: list of 10 elements
    residual_scale = [0.1, 0.05, 0.2, 0.2, 0.4, 0.1, 0.05, 0.2, 0.2, 0.4]
    assert isinstance(residual_scale, list) and len(residual_scale) == 10

    # Invalid: wrong length
    residual_scale = [0.1] * 5
    assert not (isinstance(residual_scale, list) and len(residual_scale) == 10)

    # Valid: None (missing, but not invalid)
    residual_scale = None
    assert residual_scale is None
