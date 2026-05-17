from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
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
from wheeled_biped.utils.config import get_model_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _base_config() -> dict:
    return {
        "low_level_pid": {"enabled": True, "disable_pid_action_bias": True},
        "domain_randomization": {"enabled": False},
        "sensor_noise": {"enabled": False},
        "task": {"initial_min_height": 0.60, "episode_length": 20},
    }


def test_default_low_level_mode_remains_pid_position_velocity():
    env = BalanceEnv(_base_config())

    assert env._low_level_mode == "pid_position_velocity"


def test_pid_baseline_path_unchanged_when_torque_mode_disabled():
    cfg = _base_config()
    cfg["low_level_control"] = {
        "mode": "motor_torque",
        "torque_control": {"enabled": False, "max_ctrl_fraction": 0.5},
    }
    baseline_env = BalanceEnv(_base_config())
    disabled_env = BalanceEnv(cfg)
    action = jnp.linspace(-0.2, 0.2, ACTION_DIM)
    rng = jax.random.PRNGKey(18)

    baseline_state = baseline_env.step(baseline_env.reset(rng), action)
    disabled_state = disabled_env.step(disabled_env.reset(rng), action)

    np.testing.assert_allclose(
        np.array(baseline_state.info["final_actuator_ctrl"]),
        np.array(disabled_state.info["final_actuator_ctrl"]),
        atol=1e-6,
    )


def test_motor_torque_helper_maps_normalized_action_to_direct_ctrl():
    from wheeled_biped.sim.low_level_control import normalized_motor_torque_control

    ctrl = normalized_motor_torque_control(
        jnp.array([-1.0, 0.0, 1.0]),
        jnp.array([-10.0, -20.0, -30.0]),
        jnp.array([10.0, 20.0, 30.0]),
        max_ctrl_fraction=0.5,
    )

    np.testing.assert_allclose(np.array(ctrl), np.array([-5.0, 0.0, 15.0]), atol=1e-7)


def test_hybrid_helper_adds_bounded_torque_residual_to_pid_ctrl():
    from wheeled_biped.sim.low_level_control import hybrid_pid_plus_torque_control

    final, residual = hybrid_pid_plus_torque_control(
        jnp.array([8.0, -8.0, 0.0]),
        jnp.array([1.0, -1.0, 1.0]),
        jnp.array([-10.0, -10.0, -10.0]),
        jnp.array([10.0, 10.0, 10.0]),
        max_ctrl_fraction=0.25,
    )

    np.testing.assert_allclose(np.array(residual), np.array([2.5, -2.5, 2.5]), atol=1e-7)
    np.testing.assert_allclose(np.array(final), np.array([10.0, -10.0, 2.5]), atol=1e-7)


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


def test_actuator_index_matches_action_index_for_all_motors():
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    expected = [
        "l_hip_roll_motor",
        "l_hip_yaw_motor",
        "l_hip_pitch_motor",
        "l_knee_motor",
        "l_wheel_motor",
        "r_hip_roll_motor",
        "r_hip_yaw_motor",
        "r_hip_pitch_motor",
        "r_knee_motor",
        "r_wheel_motor",
    ]

    assert model.nu == ACTION_DIM
    assert [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)] == expected


def test_motor_torque_mode_writes_actuator_ctrl_directly():
    cfg = _base_config()
    cfg["low_level_control"] = {
        "mode": "motor_torque",
        "torque_control": {"enabled": True, "max_ctrl_fraction": 0.5, "allow_hip_yaw_torque": False},
    }
    env = BalanceEnv(cfg)
    action = jnp.array([1.0, 1.0, -1.0, 0.5, -0.5, -1.0, 1.0, 0.25, -0.25, 0.0])
    state = env.step(env.reset(jax.random.PRNGKey(1818)), action)
    ctrl = np.array(state.info["final_actuator_ctrl"])

    expected = np.array([7.5, 0.0, -15.0, 7.5, -3.75, -7.5, 0.0, 3.75, -3.75, 0.0])
    np.testing.assert_allclose(ctrl, expected, atol=1e-6)


def test_hybrid_pid_plus_torque_telemetry_exists_when_enabled():
    cfg = _base_config()
    cfg["low_level_control"] = {
        "mode": "hybrid_pid_plus_torque",
        "torque_control": {"enabled": True, "max_ctrl_fraction": 0.25, "allow_hip_yaw_torque": False},
    }
    env = BalanceEnv(cfg)
    state = env.reset(jax.random.PRNGKey(1819))
    residual = jnp.ones(ACTION_DIM, dtype=jnp.float32)
    state = state._replace(info={**state.info, "torque_residual_action": residual})
    next_state = env.step(state, jnp.zeros(ACTION_DIM))

    for key in [
        "raw_pid_ctrl",
        "torque_residual_ctrl",
        "final_actuator_ctrl",
        "actuator_saturation_flags",
        "low_level_mode_code",
        "torque_control_enabled",
        "torque_safety_disabled",
    ]:
        assert key in next_state.info
    assert bool(next_state.info["torque_control_enabled"])
    assert int(next_state.info["low_level_mode_code"]) == 2
    assert np.array(next_state.info["torque_residual_ctrl"]).shape == (ACTION_DIM,)
    assert np.array(next_state.info["torque_residual_ctrl"])[L_HIP_YAW] == 0.0
    assert np.array(next_state.info["torque_residual_ctrl"])[R_HIP_YAW] == 0.0


def test_balance_residual_yaml_unchanged_by_motor_torque_helpers():
    from wheeled_biped.sim.low_level_control import normalized_motor_torque_control

    path = PROJECT_ROOT / "configs" / "training" / "balance_residual.yaml"
    before = path.read_text(encoding="utf-8")

    normalized_motor_torque_control(
        jnp.zeros(ACTION_DIM),
        jnp.full((ACTION_DIM,), -1.0),
        jnp.ones(ACTION_DIM),
    )

    assert path.read_text(encoding="utf-8") == before


def test_config_mutation_does_not_leak_between_envs():
    cfg = _base_config()
    cfg["low_level_control"] = {"mode": "motor_torque", "torque_control": {"enabled": True}}
    env_a = BalanceEnv(cfg)
    env_a._low_level_mode = "hybrid_pid_plus_torque"
    env_b = BalanceEnv(_base_config())

    assert env_b._low_level_mode == "pid_position_velocity"
