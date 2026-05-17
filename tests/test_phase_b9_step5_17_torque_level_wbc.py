from __future__ import annotations

from pathlib import Path

import jax
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


def test_torque_wbc_disabled_by_default_config():
    cfg = DualRateConfig.from_yaml(CONFIG_PATH)

    assert cfg.torque_wbc_enabled is False


def test_diagnostic_only_flag_true_by_default():
    cfg = DualRateConfig.from_yaml(CONFIG_PATH)

    assert cfg.torque_wbc_diagnostic_only is True


def test_baseline_pid_action_unchanged_when_torque_wbc_disabled():
    obs = _obs_with_roll(roll=0.1)
    a = _controller().compute_action(obs)
    b = _controller().compute_action(obs)

    np.testing.assert_allclose(a, b, atol=1e-7)


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


def test_qfrc_applied_writes_only_allowed_joint_indices():
    from wheeled_biped.sim.torque_wbc import apply_qfrc_applied_torque

    env = BalanceEnv({
        "low_level_pid": {"enabled": True, "disable_pid_action_bias": True},
        "domain_randomization": {"enabled": False},
        "sensor_noise": {"enabled": False},
    })
    state = env.reset(jax.random.PRNGKey(17))
    command = np.arange(10, dtype=np.float32) + 1.0

    data, qfrc = apply_qfrc_applied_torque(state.mjx_data, command)
    qfrc_np = np.array(qfrc)

    np.testing.assert_allclose(qfrc_np[:6], 0.0, atol=1e-7)
    assert qfrc_np[6 + L_HIP_YAW] == 0.0
    assert qfrc_np[6 + R_HIP_YAW] == 0.0
    for action_idx in [L_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_WHEEL, R_HIP_ROLL, R_HIP_PITCH, R_KNEE, R_WHEEL]:
        assert qfrc_np[6 + action_idx] == command[action_idx]
    np.testing.assert_allclose(np.array(data.qfrc_applied), qfrc_np, atol=1e-7)


def test_torque_commands_are_bounded_and_telemetry_exists():
    from wheeled_biped.sim.torque_wbc import (
        TorqueWbcGains,
        TorqueWbcLimits,
        compute_diagnostic_torque_wbc,
    )

    command, telemetry = compute_diagnostic_torque_wbc(
        _obs_with_roll(roll=0.3, roll_rate=1.0),
        TorqueWbcGains(k_roll=100.0, k_roll_rate=100.0, k_com_y=100.0),
        TorqueWbcLimits(max_joint_torque=2.5, max_wheel_torque=1.0),
        mode="torque_roll_plus_lateral",
        diagnostic_only=True,
    )

    assert np.max(np.abs(command[[L_HIP_ROLL, L_HIP_PITCH, L_KNEE, R_HIP_ROLL, R_HIP_PITCH, R_KNEE]])) <= 2.5
    assert np.max(np.abs(command[[L_WHEEL, R_WHEEL]])) <= 1.0
    assert telemetry["enabled"] is True
    assert telemetry["diagnostic_only"] is True
    assert telemetry["mode"] == "torque_roll_plus_lateral"
    for key in [
        "tau_roll_des",
        "Fy_des",
        "delta_Fz_des",
        "joint_torque_commands",
        "qfrc_applied_indices",
        "torque_clamped",
        "contact_force_response",
        "roll_response",
    ]:
        assert key in telemetry


def test_config_mutation_does_not_leak_between_loads():
    cfg_a = DualRateConfig.from_yaml(CONFIG_PATH)
    cfg_a.torque_wbc_enabled = True
    cfg_b = DualRateConfig.from_yaml(CONFIG_PATH)

    assert cfg_b.torque_wbc_enabled is False


def test_balance_residual_yaml_unchanged_by_torque_helpers():
    from wheeled_biped.sim.torque_wbc import (
        TorqueWbcGains,
        TorqueWbcLimits,
        compute_diagnostic_torque_wbc,
    )

    path = PROJECT_ROOT / "configs" / "training" / "balance_residual.yaml"
    before = path.read_text(encoding="utf-8")

    compute_diagnostic_torque_wbc(
        _obs_with_roll(roll=0.1),
        TorqueWbcGains(k_roll=1.0),
        TorqueWbcLimits(max_joint_torque=1.0),
    )

    assert path.read_text(encoding="utf-8") == before
