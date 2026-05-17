from __future__ import annotations

from pathlib import Path

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
from wheeled_biped.controllers.dual_rate_balance_controller import (
    DualRateBalanceController,
    DualRateConfig,
)
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


def test_wbc_vmc_disabled_by_default_config():
    cfg = DualRateConfig.from_yaml(CONFIG_PATH)

    assert cfg.wbc_vmc_enabled is False


def test_controller_output_shape_remains_10_with_wbc_config_present():
    action = _controller().compute_action(_obs_with_roll())

    assert action.shape == (ACTION_DIM,)


def test_wbc_telemetry_exists_when_enabled():
    controller = _controller()
    controller.config.wbc_vmc_enabled = True
    controller.config.wbc_vmc_k_roll = 1.0
    controller.config.wbc_vmc_max_delta_fz = 10.0
    controller.config.wbc_vmc_max_hip_roll_offset = 0.05

    controller.compute_action(_obs_with_roll(roll=0.1))
    telemetry = controller.get_telemetry()["wbc_vmc"]

    expected_keys = {
        "enabled",
        "tau_roll_des",
        "Fy_des",
        "Fz_des",
        "delta_Fz_des",
        "Fz_left_des",
        "Fz_right_des",
        "force_error",
        "hip_roll_offset_left",
        "hip_roll_offset_right",
        "hip_pitch_offset_left",
        "hip_pitch_offset_right",
        "knee_offset_left",
        "knee_offset_right",
        "wheel_diff_cmd",
        "clamped",
        "wheel_unload_flag",
        "mapping_mode",
    }
    assert telemetry["enabled"] is True
    assert expected_keys.issubset(telemetry.keys())


def test_wbc_correction_is_bounded_and_uses_only_allowed_indices():
    baseline = _controller().compute_action(_obs_with_roll(roll=0.2))
    controller = _controller()
    controller.config.wbc_vmc_enabled = True
    controller.config.wbc_vmc_k_roll = 1000.0
    controller.config.wbc_vmc_max_delta_fz = 5.0
    controller.config.wbc_vmc_max_hip_roll_offset = 0.11
    controller.config.wbc_vmc_max_hip_pitch_offset = 0.07
    controller.config.wbc_vmc_max_knee_offset = 0.09
    controller.config.wbc_vmc_max_wheel_diff_cmd = 0.03
    controller.config.wbc_vmc_use_wheel_diff = True

    action = controller.compute_action(_obs_with_roll(roll=0.2))
    changed = {i for i, delta in enumerate(np.abs(action - baseline)) if delta > 1e-6}

    assert changed.issubset({L_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_WHEEL, R_HIP_ROLL, R_HIP_PITCH, R_KNEE, R_WHEEL})
    assert L_HIP_YAW not in changed
    assert R_HIP_YAW not in changed
    assert abs(float(action[L_HIP_ROLL] - baseline[L_HIP_ROLL])) <= 0.11 + 1e-6
    assert abs(float(action[R_HIP_ROLL] - baseline[R_HIP_ROLL])) <= 0.11 + 1e-6
    assert abs(float(action[L_HIP_PITCH] - baseline[L_HIP_PITCH])) <= 0.07 + 1e-6
    assert abs(float(action[R_HIP_PITCH] - baseline[R_HIP_PITCH])) <= 0.07 + 1e-6
    assert abs(float(action[L_KNEE] - baseline[L_KNEE])) <= 0.09 + 1e-6
    assert abs(float(action[R_KNEE] - baseline[R_KNEE])) <= 0.09 + 1e-6


def test_wbc_config_mutation_does_not_leak_between_loads():
    cfg_a = DualRateConfig.from_yaml(CONFIG_PATH)
    cfg_a.wbc_vmc_enabled = True
    cfg_b = DualRateConfig.from_yaml(CONFIG_PATH)

    assert cfg_b.wbc_vmc_enabled is False


def test_balance_residual_yaml_unchanged_by_wbc_controller_call():
    path = PROJECT_ROOT / "configs" / "training" / "balance_residual.yaml"
    before = path.read_text(encoding="utf-8")

    controller = _controller()
    controller.config.wbc_vmc_enabled = True
    controller.config.wbc_vmc_k_roll = 1.0
    controller.config.wbc_vmc_max_delta_fz = 5.0
    controller.compute_action(_obs_with_roll(roll=0.1))

    assert path.read_text(encoding="utf-8") == before


def test_failed_step5_14_and_step5_15_layers_not_enabled_by_default():
    cfg = DualRateConfig.from_yaml(CONFIG_PATH)

    assert cfg.lateral_balance_enabled is False
    assert cfg.vmc_enabled is False


def test_jacobian_mapping_signs_are_left_right_consistent():
    from scripts.phase_b9_step5_16_jacobian_wbc_vmc import joint_delta_from_force_fraction

    left = joint_delta_from_force_fraction(0.5, side="left")
    right = joint_delta_from_force_fraction(0.5, side="right")

    assert left["hip_roll"] == -right["hip_roll"]
    assert left["hip_pitch"] == right["hip_pitch"]
    assert left["knee"] == -right["knee"]
