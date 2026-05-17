from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from wheeled_biped.controllers.action_codec import (
    ACTION_DIM,
    L_HIP_PITCH,
    L_HIP_ROLL,
    L_HIP_YAW,
    L_KNEE,
    L_WHEEL,
    R_HIP_PITCH,
    R_HIP_ROLL,
    R_HIP_YAW,
    R_KNEE,
    R_WHEEL,
)
from wheeled_biped.envs.balance_env import BalanceEnv

from scripts.phase_b9_step5_18b_hybrid_pid_torque_rollout_validation import (
    CANDIDATES,
    activation_config,
    compute_torque_residual_action,
    run_activation_trace,
    run_episode,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _base_config() -> dict:
    return {
        "low_level_pid": {"enabled": True, "disable_pid_action_bias": True},
        "domain_randomization": {"enabled": False},
        "sensor_noise": {"enabled": False},
        "task": {"initial_min_height": 0.60, "episode_length": 20},
    }


def test_hybrid_candidate_activates_torque_control():
    cfg = activation_config(_base_config(), CANDIDATES[0])
    env = BalanceEnv(cfg)
    state = env.step(env.reset(jax.random.PRNGKey(5180)), jnp.zeros(ACTION_DIM))

    assert env._low_level_mode == "hybrid_pid_plus_torque"
    assert bool(state.info["torque_control_enabled"])
    assert int(state.info["low_level_mode_code"]) == 2


def test_qfrc_applied_is_not_used_by_step5_18b_rollout():
    rows, summary = run_activation_trace(candidate=CANDIDATES[0], steps=3)

    assert rows
    assert summary["uses_deployable_actuator_ctrl_only"] is True
    assert summary["qfrc_applied_abs_max"] == 0.0


def test_final_ctrl_is_raw_pid_plus_bounded_torque_residual():
    cfg = activation_config(_base_config(), CANDIDATES[0])
    env = BalanceEnv(cfg)
    state = env.reset(jax.random.PRNGKey(5181))
    residual = jnp.array([0.4, 0.0, -0.2, 0.1, 0.0, -0.4, 0.0, -0.2, -0.1, 0.0], dtype=jnp.float32)
    state = state._replace(info={**state.info, "torque_residual_action": residual})

    next_state = env.step(state, jnp.zeros(ACTION_DIM))
    raw = np.array(next_state.info["raw_pid_ctrl"])
    torque = np.array(next_state.info["torque_residual_ctrl"])
    final = np.array(next_state.info["final_actuator_ctrl"])
    ctrl_min = np.array(env._ctrl_min)
    ctrl_max = np.array(env._ctrl_max)

    np.testing.assert_allclose(final, np.clip(raw + torque, ctrl_min, ctrl_max), atol=1e-6)


def test_actuator_ctrl_respects_ctrlrange_in_rollout():
    result = run_episode(candidate=CANDIDATES[1], seed=5182, max_steps=5)

    assert result["max_ctrl_margin_violation"] <= 1e-6
    assert result["actuator_saturation_rate"] >= 0.0


def test_torque_residual_can_be_nonzero_in_rollout():
    result = run_episode(candidate=CANDIDATES[1], seed=5183, max_steps=5)

    assert result["torque_residual_nonzero_steps"] > 0
    assert result["mean_torque_residual_abs"] > 0.0


def test_default_pid_position_velocity_path_unchanged():
    baseline_env = BalanceEnv(_base_config())
    cfg = activation_config(_base_config(), CANDIDATES[0])
    cfg["low_level_control"]["torque_control"]["enabled"] = False
    disabled_env = BalanceEnv(cfg)
    action = jnp.linspace(-0.1, 0.1, ACTION_DIM)
    rng = jax.random.PRNGKey(5184)

    baseline_state = baseline_env.step(baseline_env.reset(rng), action)
    disabled_state = disabled_env.step(disabled_env.reset(rng), action)

    np.testing.assert_allclose(
        np.array(baseline_state.info["final_actuator_ctrl"]),
        np.array(disabled_state.info["final_actuator_ctrl"]),
        atol=1e-6,
    )


def test_balance_residual_yaml_unchanged_by_step5_18b_helpers():
    path = PROJECT_ROOT / "configs" / "training" / "balance_residual.yaml"
    before = path.read_text(encoding="utf-8")

    compute_torque_residual_action(np.zeros(42, dtype=np.float32), CANDIDATES[0])

    assert path.read_text(encoding="utf-8") == before


def test_action_dimension_and_order_unchanged():
    assert ACTION_DIM == 10
    assert [
        L_HIP_ROLL,
        L_HIP_YAW,
        L_HIP_PITCH,
        L_KNEE,
        L_WHEEL,
        R_HIP_ROLL,
        R_HIP_YAW,
        R_HIP_PITCH,
        R_KNEE,
        R_WHEEL,
    ] == list(range(10))


def test_config_mutation_does_not_leak_between_envs():
    cfg = activation_config(_base_config(), CANDIDATES[0])
    env_a = BalanceEnv(cfg)
    env_a._low_level_mode = "pid_position_velocity"
    env_b = BalanceEnv(activation_config(_base_config(), CANDIDATES[0]))

    assert env_b._low_level_mode == "hybrid_pid_plus_torque"
