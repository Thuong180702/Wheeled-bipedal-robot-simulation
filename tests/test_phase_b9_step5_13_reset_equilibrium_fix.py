from __future__ import annotations

import copy
from pathlib import Path

import jax
import mujoco
import numpy as np
import yaml

from scripts.phase_b9_step5_lqr_gain_strengthening import (
    apply_balanced_root_init,
    load_balanced_init_table,
)
from wheeled_biped.envs.balance_env import BalanceEnv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _expected_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([roll, pitch, yaw]), b"xyz")
    return quat


def _reset_state():
    env = BalanceEnv({
        "low_level_pid": {"enabled": True, "disable_pid_action_bias": True},
        "domain_randomization": {"enabled": False},
        "sensor_noise": {"enabled": False},
    })
    return env.reset(jax.random.PRNGKey(123))


def test_step5_full_balanced_root_init_applies_root_position():
    table = load_balanced_init_table()
    height = 0.60
    init = table[height]
    state = _reset_state()
    before_y = float(state.mjx_data.qpos[1])

    data = apply_balanced_root_init(state.mjx_data, height, table)

    assert float(data.qpos[0]) == np.float32(init["root_x"])
    assert float(data.qpos[1]) == before_y
    assert float(data.qpos[2]) == np.float32(init["root_z"])


def test_step5_full_balanced_root_init_applies_root_orientation_as_quaternion():
    table = load_balanced_init_table()
    height = 0.60
    init = table[height]
    state = _reset_state()

    data = apply_balanced_root_init(state.mjx_data, height, table)

    expected = _expected_quat(init["root_roll"], init["root_pitch"], 0.0)
    np.testing.assert_allclose(np.array(data.qpos[3:7]), expected, atol=1e-6)


def test_step5_full_balanced_root_init_applies_symmetric_joint_targets():
    table = load_balanced_init_table()
    height = 0.60
    init = table[height]
    state = _reset_state()

    data = apply_balanced_root_init(state.mjx_data, height, table)

    expected = np.array([
        0.0, 0.0, init["hip_pitch"], init["knee"], 0.0,
        0.0, 0.0, init["hip_pitch"], init["knee"], 0.0,
    ], dtype=np.float32)
    np.testing.assert_allclose(np.array(data.qpos[7:17]), expected, atol=1e-6)


def test_step5_full_balanced_root_init_zeroes_all_velocities():
    table = load_balanced_init_table()
    state = _reset_state()
    moving = state.mjx_data.replace(qvel=state.mjx_data.qvel + 1.0)

    data = apply_balanced_root_init(moving, 0.60, table)

    np.testing.assert_allclose(np.array(data.qvel), np.zeros_like(np.array(data.qvel)), atol=1e-7)


def test_step5_full_balanced_root_init_does_not_mutate_table():
    table = load_balanced_init_table()
    original = copy.deepcopy(table)
    state = _reset_state()

    apply_balanced_root_init(state.mjx_data, 0.60, table)

    assert table == original


def test_balance_residual_config_is_unchanged_by_step5_init():
    path = PROJECT_ROOT / "configs" / "training" / "balance_residual.yaml"
    before = path.read_text(encoding="utf-8")
    table = load_balanced_init_table()
    state = _reset_state()

    apply_balanced_root_init(state.mjx_data, 0.60, table)

    after = path.read_text(encoding="utf-8")
    assert after == before


def test_current_balanced_root_table_does_not_have_severe_penetration_after_repair():
    table_path = PROJECT_ROOT / "configs" / "controllers" / "b9_balanced_root_init_table.yaml"
    with open(table_path, "r", encoding="utf-8") as f:
        table = yaml.safe_load(f)["balanced_root_initialization"]["heights"]

    for height_key, init in table.items():
        assert "root_z" in init, height_key
        assert "root_roll" in init, height_key
        assert "root_pitch" in init, height_key
