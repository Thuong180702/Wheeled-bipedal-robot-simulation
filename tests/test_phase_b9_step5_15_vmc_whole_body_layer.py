from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from scripts.phase_b9_step5_lqr_gain_strengthening import (
    apply_balanced_root_init,
    load_balanced_init_table,
)
from wheeled_biped.controllers.action_codec import (
    ACTION_DIM,
    L_HIP_ROLL,
    L_HIP_YAW,
    L_KNEE,
    R_HIP_ROLL,
    R_HIP_YAW,
    R_KNEE,
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
    obs[0] = 0.0
    obs[1] = np.sin(roll)
    obs[2] = -np.cos(roll)
    obs[7] = roll_rate
    obs[39] = (0.60 - 0.40) / (0.65 - 0.40)
    obs[40] = obs[39]
    return obs


def test_vmc_disabled_by_default_config():
    cfg = DualRateConfig.from_yaml(CONFIG_PATH)

    assert cfg.vmc_enabled is False


def test_controller_output_shape_remains_10_with_vmc_config_present():
    action = _controller().compute_action(_obs_with_roll())

    assert action.shape == (ACTION_DIM,)


def test_action_ordering_unchanged_for_vmc_roll_yaw_and_knee_indices():
    assert L_HIP_ROLL == 0
    assert L_HIP_YAW == 1
    assert L_KNEE == 3
    assert R_HIP_ROLL == 5
    assert R_HIP_YAW == 6
    assert R_KNEE == 8


def test_vmc_correction_is_bounded_when_enabled():
    controller = _controller()
    controller.config.vmc_enabled = True
    controller.config.vmc_mapping = "hip_roll_leg_length"
    controller.config.vmc_k_roll = 100.0
    controller.config.vmc_a_roll = 1.0
    controller.config.vmc_max_delta_support = 0.30
    controller.config.vmc_max_hip_roll_correction = 0.12
    controller.config.vmc_max_leg_length_correction = 0.08

    action = controller.compute_action(_obs_with_roll(roll=0.2))

    np.testing.assert_allclose(abs(float(action[L_HIP_ROLL])), 0.12, atol=1e-6)
    np.testing.assert_allclose(abs(float(action[R_HIP_ROLL])), 0.12, atol=1e-6)
    np.testing.assert_allclose(abs(float(action[L_KNEE] - action[R_KNEE])), 0.16, atol=1e-6)


def test_vmc_config_mutation_does_not_leak_between_loads():
    cfg_a = DualRateConfig.from_yaml(CONFIG_PATH)
    cfg_a.vmc_enabled = True
    cfg_b = DualRateConfig.from_yaml(CONFIG_PATH)

    assert cfg_b.vmc_enabled is False


def test_balance_residual_yaml_unchanged_by_vmc_controller_call():
    path = PROJECT_ROOT / "configs" / "training" / "balance_residual.yaml"
    before = path.read_text(encoding="utf-8")

    controller = _controller()
    controller.config.vmc_enabled = True
    controller.config.vmc_mapping = "combined_weak"
    controller.config.vmc_k_roll = 1.0
    controller.config.vmc_a_roll = 1.0
    controller.compute_action(_obs_with_roll(roll=0.1))

    assert path.read_text(encoding="utf-8") == before


def test_full_root_reset_path_remains_active_for_vmc_step():
    env = BalanceEnv({
        "low_level_pid": {"enabled": True, "disable_pid_action_bias": True},
        "domain_randomization": {"enabled": False},
        "sensor_noise": {"enabled": False},
    })
    state = env.reset(jax.random.PRNGKey(515))
    table = load_balanced_init_table()
    init = table[0.60]

    data = apply_balanced_root_init(state.mjx_data, 0.60, table)

    assert float(data.qpos[0]) == np.float32(init["root_x"])
    assert float(data.qpos[2]) == np.float32(init["root_z"])


def test_failed_step5_14_lateral_layer_is_not_enabled_by_default():
    cfg = DualRateConfig.from_yaml(CONFIG_PATH)

    assert cfg.lateral_balance_enabled is False


def test_vmc_telemetry_exists_when_enabled():
    controller = _controller()
    controller.config.vmc_enabled = True
    controller.config.vmc_mapping = "force_balance_only"
    controller.config.vmc_k_force_diff = 1.0
    controller.config.vmc_a_force = 1.0
    controller.config.vmc_external_force_diff_error = 0.25
    controller.config.vmc_max_delta_support = 0.10

    controller.compute_action(_obs_with_roll())
    telemetry = controller.get_telemetry()["vmc_whole_body"]

    assert telemetry["enabled"] is True
    assert telemetry["mapping"] == "force_balance_only"
    assert abs(float(telemetry["delta_support"])) <= 0.10
    assert "desired_force_balance" in telemetry
