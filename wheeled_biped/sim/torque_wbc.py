from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from wheeled_biped.controllers.action_codec import (
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

ALLOWED_TORQUE_ACTION_INDICES = (
    L_HIP_ROLL,
    L_HIP_PITCH,
    L_KNEE,
    L_WHEEL,
    R_HIP_ROLL,
    R_HIP_PITCH,
    R_KNEE,
    R_WHEEL,
)


@dataclass(frozen=True)
class TorqueWbcGains:
    k_roll: float = 0.0
    k_roll_rate: float = 0.0
    k_com_y: float = 0.0
    k_com_y_rate: float = 0.0
    k_height: float = 0.0
    k_height_rate: float = 0.0


@dataclass(frozen=True)
class TorqueWbcLimits:
    max_joint_torque: float = 0.0
    max_wheel_torque: float = 0.0
    max_body_wrench: float = 0.0
    max_torque_rate: float = 0.0


def _obs_value(obs: np.ndarray, index: int, default: float = 0.0) -> float:
    return float(obs[index]) if obs.size > index else default


def compute_diagnostic_torque_wbc(
    obs: np.ndarray,
    gains: TorqueWbcGains,
    limits: TorqueWbcLimits,
    *,
    mode: str = "torque_roll_plus_lateral",
    diagnostic_only: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    obs = np.asarray(obs, dtype=np.float32)
    roll = float(np.arcsin(np.clip(_obs_value(obs, 1), -1.0, 1.0)))
    roll_rate = _obs_value(obs, 7)
    com_y = _obs_value(obs, 4)
    com_y_rate = _obs_value(obs, 4)
    height_error = _obs_value(obs, 40) - _obs_value(obs, 39)
    height_rate = _obs_value(obs, 5)

    tau_roll_des = -(gains.k_roll * roll + gains.k_roll_rate * roll_rate)
    fy_des = -(gains.k_com_y * com_y + gains.k_com_y_rate * com_y_rate)
    fz_term = -(gains.k_height * height_error + gains.k_height_rate * height_rate)
    delta_fz_des = tau_roll_des / 0.23 + 0.1 * fy_des + fz_term

    command = np.zeros(10, dtype=np.float32)
    joint_limit = abs(float(limits.max_joint_torque))
    wheel_limit = abs(float(limits.max_wheel_torque))
    roll_cmd = float(np.clip(tau_roll_des, -joint_limit, joint_limit))
    lateral_cmd = float(np.clip(fy_des * 0.05, -joint_limit, joint_limit))
    leg_length_cmd = float(np.clip(delta_fz_des * 0.02, -joint_limit, joint_limit))
    wheel_cmd = float(np.clip(fy_des * 0.02, -wheel_limit, wheel_limit))

    if mode in {"torque_roll_only", "hybrid_pid_plus_torque_roll", "torque_roll_plus_lateral", "conservative_torque_wbc"}:
        command[L_HIP_ROLL] += roll_cmd
        command[R_HIP_ROLL] -= roll_cmd
    if mode in {"torque_lateral_com_only", "torque_roll_plus_lateral", "conservative_torque_wbc"}:
        command[L_HIP_PITCH] += lateral_cmd + leg_length_cmd
        command[R_HIP_PITCH] += lateral_cmd - leg_length_cmd
        command[L_KNEE] -= leg_length_cmd
        command[R_KNEE] += leg_length_cmd
        command[L_WHEEL] += wheel_cmd
        command[R_WHEEL] -= wheel_cmd
    if mode == "conservative_torque_wbc":
        command *= 0.5

    pre_clip = command.copy()
    leg_indices = [L_HIP_ROLL, L_HIP_YAW, L_HIP_PITCH, L_KNEE, R_HIP_ROLL, R_HIP_YAW, R_HIP_PITCH, R_KNEE]
    wheel_indices = [L_WHEEL, R_WHEEL]
    command[leg_indices] = np.clip(command[leg_indices], -joint_limit, joint_limit)
    command[wheel_indices] = np.clip(command[wheel_indices], -wheel_limit, wheel_limit)
    command[L_HIP_YAW] = 0.0
    command[R_HIP_YAW] = 0.0

    telemetry = {
        "enabled": True,
        "diagnostic_only": bool(diagnostic_only),
        "mode": mode,
        "tau_roll_des": float(tau_roll_des),
        "Fy_des": float(fy_des),
        "delta_Fz_des": float(delta_fz_des),
        "joint_torque_commands": command.tolist(),
        "qfrc_applied_indices": [6 + int(i) for i in ALLOWED_TORQUE_ACTION_INDICES],
        "torque_clamped": bool(np.any(np.abs(pre_clip - command) > 1e-7)),
        "contact_force_response": "diagnostic_not_measured_in_helper",
        "roll_response": "diagnostic_not_measured_in_helper",
    }
    return command, telemetry


def apply_qfrc_applied_torque(
    mjx_data: mjx.Data,
    joint_torque_commands: np.ndarray | jnp.ndarray,
    allowed_action_indices: Iterable[int] | None = None,
) -> tuple[mjx.Data, jnp.ndarray]:
    allowed = tuple(ALLOWED_TORQUE_ACTION_INDICES if allowed_action_indices is None else allowed_action_indices)
    command = jnp.asarray(joint_torque_commands, dtype=mjx_data.qfrc_applied.dtype)
    qfrc = jnp.zeros_like(mjx_data.qfrc_applied)
    for action_idx in allowed:
        if action_idx in (L_HIP_YAW, R_HIP_YAW):
            continue
        qfrc = qfrc.at[6 + int(action_idx)].set(command[int(action_idx)])
    return mjx_data.replace(qfrc_applied=qfrc), qfrc
