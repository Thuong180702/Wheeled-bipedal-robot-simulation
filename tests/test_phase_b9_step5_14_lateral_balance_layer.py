from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from scripts.phase_b9_step5_13_reset_equilibrium_fix import rpy_to_quat
from scripts.phase_b9_step5_14_lateral_balance_layer import refresh_balance_obs_after_data_edit
from scripts.phase_b9_step5_lqr_gain_strengthening import (
    apply_balanced_root_init,
    load_balanced_init_table,
)
from wheeled_biped.controllers.action_codec import (
    ACTION_DIM,
    L_HIP_ROLL,
    L_HIP_YAW,
    R_HIP_ROLL,
    R_HIP_YAW,
)
from wheeled_biped.controllers.dual_rate_balance_controller import (
    DualRateBalanceController,
    DualRateConfig,
)
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "controllers" / "dual_rate_balance_controller_b9.yaml"


def _controller() -> DualRateBalanceController:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    return DualRateBalanceController(DualRateConfig.from_yaml(CONFIG_PATH), model)


def _obs_with_roll(roll: float = 0.0, roll_rate: float = 0.0) -> np.ndarray:
    obs = np.zeros(42, dtype=np.float32)
    obs[2] = -1.0
    obs[7] = roll_rate
    obs[39] = (0.60 - 0.40) / (0.65 - 0.40)
    obs[40] = obs[39]
    obs[0] = 0.0
    obs[1] = np.sin(roll)
    obs[2] = -np.cos(roll)
    return obs


def test_lateral_balance_disabled_by_default_config():
    cfg = DualRateConfig.from_yaml(CONFIG_PATH)

    assert cfg.lateral_balance_enabled is False


def test_controller_output_shape_remains_10():
    action = _controller().compute_action(_obs_with_roll())

    assert action.shape == (ACTION_DIM,)


def test_action_ordering_unchanged_for_roll_yaw_indices():
    assert L_HIP_ROLL == 0
    assert L_HIP_YAW == 1
    assert R_HIP_ROLL == 5
    assert R_HIP_YAW == 6


def test_lateral_hip_roll_correction_is_bounded_when_enabled():
    controller = _controller()
    controller.config.lateral_balance_enabled = True
    controller.config.lateral_k_roll = 100.0
    controller.config.lateral_max_correction = 0.15

    action = controller.compute_action(_obs_with_roll(roll=0.2))

    np.testing.assert_allclose(abs(float(action[L_HIP_ROLL])), 0.15, atol=1e-6)
    np.testing.assert_allclose(abs(float(action[R_HIP_ROLL])), 0.15, atol=1e-6)
    np.testing.assert_allclose(float(action[L_HIP_ROLL]), -float(action[R_HIP_ROLL]), atol=1e-6)


def test_lateral_correction_sign_is_configurable():
    controller = _controller()
    controller.config.lateral_balance_enabled = True
    controller.config.lateral_k_roll = 1.0
    controller.config.lateral_max_correction = 0.5
    controller.config.lateral_sign = 1.0
    positive_action = controller.compute_action(_obs_with_roll(roll=0.1))

    controller = _controller()
    controller.config.lateral_balance_enabled = True
    controller.config.lateral_k_roll = 1.0
    controller.config.lateral_max_correction = 0.5
    controller.config.lateral_sign = -1.0
    negative_action = controller.compute_action(_obs_with_roll(roll=0.1))

    assert abs(float(positive_action[L_HIP_ROLL])) > 1e-6
    assert abs(float(negative_action[L_HIP_ROLL])) > 1e-6
    assert np.sign(positive_action[L_HIP_ROLL]) == -np.sign(negative_action[L_HIP_ROLL])
    assert np.sign(positive_action[R_HIP_ROLL]) == -np.sign(negative_action[R_HIP_ROLL])


def test_lateral_config_mutation_does_not_leak_between_loads():
    cfg_a = DualRateConfig.from_yaml(CONFIG_PATH)
    cfg_a.lateral_balance_enabled = True
    cfg_b = DualRateConfig.from_yaml(CONFIG_PATH)

    assert cfg_b.lateral_balance_enabled is False


def test_balance_residual_yaml_unchanged_by_lateral_controller_call():
    path = PROJECT_ROOT / "configs" / "training" / "balance_residual.yaml"
    before = path.read_text(encoding="utf-8")

    controller = _controller()
    controller.compute_action(_obs_with_roll(roll=0.1))

    assert path.read_text(encoding="utf-8") == before




def test_full_root_reset_path_remains_active():
    env = BalanceEnv({
        "low_level_pid": {"enabled": True, "disable_pid_action_bias": True},
        "domain_randomization": {"enabled": False},
        "sensor_noise": {"enabled": False},
    })
    state = env.reset(jax.random.PRNGKey(123))
    table = load_balanced_init_table()
    init = table[0.60]

    data = apply_balanced_root_init(state.mjx_data, 0.60, table)

    assert float(data.qpos[0]) == np.float32(init["root_x"])
    assert float(data.qpos[2]) == np.float32(init["root_z"])


def test_sign_response_uses_refreshed_observation_after_roll_perturbation():
    env = BalanceEnv({
        "low_level_pid": {"enabled": True, "disable_pid_action_bias": True},
        "domain_randomization": {"enabled": False},
        "sensor_noise": {"enabled": False},
    })
    state = env.reset(jax.random.PRNGKey(456))
    table = load_balanced_init_table()
    data = apply_balanced_root_init(state.mjx_data, 0.60, table)
    qpos = data.qpos.at[3:7].set(jnp.array(rpy_to_quat(np.deg2rad(2.0), 0.0, 0.0), dtype=data.qpos.dtype))
    state = state._replace(mjx_data=data.replace(qpos=qpos, qvel=jnp.zeros_like(data.qvel)))

    refreshed = refresh_balance_obs_after_data_edit(env, state)
    roll = np.arcsin(np.clip(np.array(refreshed)[1], -1.0, 1.0))

    np.testing.assert_allclose(abs(np.rad2deg(roll)), 2.0, atol=0.05)
