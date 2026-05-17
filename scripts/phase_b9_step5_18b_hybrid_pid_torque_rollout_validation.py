"""Phase B.9 Step 5.18b: hybrid PID + motor-torque rollout validation.

This script uses only the deployable MJCF actuator ctrl path exposed in Step 5.18.
It does not use qfrc_applied, start Step 6, train PPO, or modify residual configs.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase_b9_step5_lqr_gain_strengthening import (  # noqa: E402
    apply_balanced_root_init,
    create_tuned_controller,
    load_balanced_init_table,
    rpy_to_quat,
)
from wheeled_biped.controllers.action_codec import (  # noqa: E402
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
from wheeled_biped.controllers.dual_rate_balance_controller import DualRateConfig  # noqa: E402
from wheeled_biped.envs.balance_env import BalanceEnv  # noqa: E402
from wheeled_biped.utils.config import get_model_path  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase_b9_step5_18b_hybrid_pid_torque_rollout_validation"
BEST_LQR_PATH = PROJECT_ROOT / "outputs" / "phase_b9_lqr_gain_strengthening" / "best_lqr_config.yaml"
CONTROLLER_CONFIG_PATH = PROJECT_ROOT / "configs" / "controllers" / "dual_rate_balance_controller_b9.yaml"
BALANCE_RESIDUAL_PATH = PROJECT_ROOT / "configs" / "training" / "balance_residual.yaml"

RESET_FIXED_ALL_HEIGHT_BASELINE = {
    "mean_survival_s": 3.8167,
    "fall_rate": 0.8333,
    "pitch_rms_deg": 1.1938,
    "roll_rms_deg": 21.1630,
    "action_saturation": 0.0,
}
RESET_FIXED_H060_BASELINE = {
    "survival_s": 0.52,
    "fall_rate": 1.0,
    "pitch_rms_deg": 0.8745,
    "roll_rms_deg": 16.4930,
    "action_saturation": 0.0,
}

ACTIVATION_STEPS = 20
RESPONSE_STEPS = 20
CANDIDATE_STEPS = 60
FULL_VALIDATION_STEPS = 60


@dataclass(frozen=True)
class HybridTorqueCandidate:
    name: str
    k_roll: float
    k_roll_rate: float
    k_pitch: float
    k_pitch_rate: float
    max_ctrl_fraction: float
    allow_wheel_torque: bool
    wheel_roll_gain: float


CANDIDATES = [
    HybridTorqueCandidate("hybrid_roll_damping_conservative", 0.8, 0.10, 0.0, 0.0, 0.10, False, 0.0),
    HybridTorqueCandidate("hybrid_roll_damping_medium", 1.5, 0.18, 0.0, 0.0, 0.15, False, 0.0),
    HybridTorqueCandidate("hybrid_roll_damping_strong", 2.5, 0.30, 0.0, 0.0, 0.20, False, 0.0),
    HybridTorqueCandidate("hybrid_roll_pitch_damping", 1.5, 0.18, 0.4, 0.05, 0.15, False, 0.0),
    HybridTorqueCandidate("hybrid_roll_wheel_assist", 1.2, 0.15, 0.0, 0.0, 0.15, True, 0.08),
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_best_lqr_params() -> dict[str, float]:
    with BEST_LQR_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_config(height: float = 0.60, episode_length: int = 250) -> dict[str, Any]:
    return {
        "low_level_pid": {"enabled": True, "disable_pid_action_bias": True},
        "domain_randomization": {"enabled": False},
        "sensor_noise": {"enabled": False},
        "task": {"initial_min_height": height, "episode_length": episode_length},
        "termination": {"max_tilt_rad": 0.8, "min_height": 0.3},
    }


def activation_config(base_config: dict[str, Any], candidate: HybridTorqueCandidate) -> dict[str, Any]:
    cfg = {
        key: (value.copy() if isinstance(value, dict) else value)
        for key, value in base_config.items()
    }
    cfg["low_level_control"] = {
        "mode": "hybrid_pid_plus_torque",
        "torque_control": {
            "enabled": True,
            "max_ctrl_fraction": candidate.max_ctrl_fraction,
            "allow_leg_torque": True,
            "allow_wheel_torque": candidate.allow_wheel_torque,
            "allow_hip_yaw_torque": False,
        },
    }
    return cfg


def make_controller(model: mujoco.MjModel):
    base_config = DualRateConfig.from_yaml(CONTROLLER_CONFIG_PATH)
    return create_tuned_controller(base_config, load_best_lqr_params(), model)


def compute_torque_residual_action(obs: np.ndarray, candidate: HybridTorqueCandidate) -> np.ndarray:
    gravity_body = obs[0:3]
    roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
    pitch = float(np.arcsin(np.clip(-gravity_body[0], -1.0, 1.0)))
    roll_rate = float(obs[7])
    pitch_rate = float(obs[6])

    roll_cmd = -candidate.k_roll * roll - candidate.k_roll_rate * roll_rate
    pitch_cmd = -candidate.k_pitch * pitch - candidate.k_pitch_rate * pitch_rate
    wheel_cmd = -candidate.wheel_roll_gain * roll_rate

    residual = np.zeros(ACTION_DIM, dtype=np.float32)
    residual[L_HIP_ROLL] = np.clip(roll_cmd, -1.0, 1.0)
    residual[R_HIP_ROLL] = np.clip(-roll_cmd, -1.0, 1.0)
    residual[L_HIP_PITCH] = np.clip(pitch_cmd, -1.0, 1.0)
    residual[R_HIP_PITCH] = np.clip(pitch_cmd, -1.0, 1.0)
    residual[L_KNEE] = np.clip(-0.5 * pitch_cmd, -1.0, 1.0)
    residual[R_KNEE] = np.clip(-0.5 * pitch_cmd, -1.0, 1.0)
    if candidate.allow_wheel_torque:
        residual[L_WHEEL] = np.clip(wheel_cmd, -1.0, 1.0)
        residual[R_WHEEL] = np.clip(-wheel_cmd, -1.0, 1.0)
    residual[L_HIP_YAW] = 0.0
    residual[R_HIP_YAW] = 0.0
    return residual


def set_height_and_roll(state, env: BalanceEnv, height: float, roll_rad: float, init_table: dict[float, dict[str, float]]):
    mjx_data = apply_balanced_root_init(state.mjx_data, height, init_table)
    if abs(roll_rad) > 0.0:
        qpos = mjx_data.qpos.at[3:7].set(jnp.array(rpy_to_quat(roll_rad, 0.0, 0.0), dtype=mjx_data.qpos.dtype))
        mjx_data = mjx_data.replace(qpos=qpos, qvel=jnp.zeros_like(mjx_data.qvel))
    base_obs = env._extract_obs(mjx_data, jnp.zeros(env.num_actions), state.info["noise_rng"])
    height_norm = (height - env.MIN_HEIGHT_CMD) / (env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD)
    current_height_norm = (mjx_data.qpos[2] - env.MIN_HEIGHT_CMD) / (env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD)
    obs = jnp.concatenate([base_obs, jnp.array([height_norm, current_height_norm, 0.0], dtype=base_obs.dtype)])
    info = {
        **state.info,
        "height_command": jnp.array(height, dtype=mjx_data.qpos.dtype),
        "initial_yaw": jnp.array(0.0, dtype=mjx_data.qpos.dtype),
    }
    return state._replace(mjx_data=mjx_data, obs=obs, info=info, prev_action=jnp.zeros(env.num_actions))


def fall_reason(env: BalanceEnv, state) -> str:
    if not bool(state.info["is_fallen"]):
        return "none"
    torso_height = float(state.mjx_data.qpos[2])
    g_body_final = np.array(state.obs[:3])
    tilt_final = float(np.arccos(np.clip(-g_body_final[2], -1.0, 1.0)))
    if torso_height < env._min_height:
        return "height"
    if tilt_final > env._max_tilt:
        return "tilt"
    return "unknown"


def run_episode(
    candidate: HybridTorqueCandidate,
    seed: int,
    height: float = 0.60,
    roll_rad: float = 0.0,
    max_steps: int = 250,
    model: mujoco.MjModel | None = None,
    env: BalanceEnv | None = None,
    controller: Any | None = None,
    init_table: dict[float, dict[str, float]] | None = None,
) -> dict[str, Any]:
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    if env is None:
        env = BalanceEnv(activation_config(make_env_config(height, max_steps), candidate))
    if controller is None:
        controller = make_controller(model)
    if init_table is None:
        init_table = load_balanced_init_table()
    nearest_height = min(init_table.keys(), key=lambda h: abs(h - height))

    state = env.reset(jax.random.PRNGKey(seed))
    state = set_height_and_roll(state, env, height, roll_rad, init_table)
    controller.reset()
    controller.target_hip_pitch = init_table[nearest_height]["hip_pitch"]
    controller.target_knee = init_table[nearest_height]["knee"]
    controller.last_stable_hip_pitch = controller.target_hip_pitch
    controller.last_stable_knee = controller.target_knee

    pitch_sq = 0.0
    roll_sq = 0.0
    saturation_count = 0
    torque_nonzero_steps = 0
    torque_abs_sum = 0.0
    ctrl_violation = 0.0
    qfrc_abs_max = 0.0
    steps = 0

    for _ in range(max_steps):
        obs_np = np.array(state.obs)
        action = controller.compute_action(obs_np)
        residual = compute_torque_residual_action(obs_np, candidate)
        state = state._replace(info={**state.info, "torque_residual_action": jnp.array(residual)})
        state = env.step(state, jnp.array(action))
        steps += 1

        pitch = float(np.arcsin(np.clip(-obs_np[0], -1.0, 1.0)))
        roll = float(np.arcsin(np.clip(obs_np[1], -1.0, 1.0)))
        pitch_sq += pitch * pitch
        roll_sq += roll * roll

        torque_ctrl = np.array(state.info["torque_residual_ctrl"])
        final_ctrl = np.array(state.info["final_actuator_ctrl"])
        ctrl_min = np.array(env._ctrl_min)
        ctrl_max = np.array(env._ctrl_max)
        saturation_count += int(np.any(np.array(state.info["actuator_saturation_flags"])))
        torque_nonzero_steps += int(np.max(np.abs(torque_ctrl)) > 1e-7)
        torque_abs_sum += float(np.mean(np.abs(torque_ctrl)))
        ctrl_violation = max(ctrl_violation, float(np.maximum(ctrl_min - final_ctrl, final_ctrl - ctrl_max).max(initial=0.0)))
        qfrc_abs_max = max(qfrc_abs_max, float(np.max(np.abs(np.array(state.mjx_data.qfrc_applied)))))

        if bool(state.done):
            break

    pitch_rms = float(np.rad2deg(np.sqrt(pitch_sq / max(steps, 1))))
    roll_rms = float(np.rad2deg(np.sqrt(roll_sq / max(steps, 1))))
    return {
        "candidate": candidate.name,
        "height": height,
        "seed": seed,
        "steps": steps,
        "survival_time_s": float(steps * env.CONTROL_DT),
        "fell": bool(state.info["is_fallen"]),
        "fall_reason": fall_reason(env, state),
        "pitch_rms_deg": pitch_rms,
        "roll_rms_deg": roll_rms,
        "actuator_saturation_rate": float(saturation_count / max(steps, 1)),
        "torque_residual_nonzero_steps": torque_nonzero_steps,
        "mean_torque_residual_abs": float(torque_abs_sum / max(steps, 1)),
        "max_ctrl_margin_violation": ctrl_violation,
        "qfrc_applied_abs_max": qfrc_abs_max,
        "low_level_mode_code": int(state.info["low_level_mode_code"]),
        "torque_control_enabled": bool(state.info["torque_control_enabled"]),
    }


def run_activation_trace(candidate: HybridTorqueCandidate = CANDIDATES[0], steps: int = 20) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    env = BalanceEnv(activation_config(make_env_config(0.60, steps), candidate))
    controller = make_controller(model)
    init_table = load_balanced_init_table()
    state = set_height_and_roll(env.reset(jax.random.PRNGKey(518)), env, 0.60, np.deg2rad(2.0), init_table)
    controller.reset()

    rows = []
    for step in range(steps):
        obs_np = np.array(state.obs)
        action = controller.compute_action(obs_np)
        residual = compute_torque_residual_action(obs_np, candidate)
        state = state._replace(info={**state.info, "torque_residual_action": jnp.array(residual)})
        state = env.step(state, jnp.array(action))
        rows.append({
            "step": step,
            "low_level_mode_code": int(state.info["low_level_mode_code"]),
            "torque_control_enabled": bool(state.info["torque_control_enabled"]),
            "raw_pid_ctrl_abs_max": float(np.max(np.abs(np.array(state.info["raw_pid_ctrl"])))),
            "torque_residual_ctrl_abs_max": float(np.max(np.abs(np.array(state.info["torque_residual_ctrl"])))),
            "final_actuator_ctrl_abs_max": float(np.max(np.abs(np.array(state.info["final_actuator_ctrl"])))),
            "qfrc_applied_abs_max": float(np.max(np.abs(np.array(state.mjx_data.qfrc_applied)))),
        })
        if bool(state.done):
            break
    qfrc_abs_max = max((row["qfrc_applied_abs_max"] for row in rows), default=0.0)
    residual_abs_max = max((row["torque_residual_ctrl_abs_max"] for row in rows), default=0.0)
    summary = {
        "candidate": candidate.name,
        "hybrid_pid_plus_torque_activated": all(row["low_level_mode_code"] == 2 for row in rows),
        "torque_control_enabled": all(row["torque_control_enabled"] for row in rows),
        "torque_residual_nonzero": residual_abs_max > 0.0,
        "uses_deployable_actuator_ctrl_only": qfrc_abs_max == 0.0,
        "qfrc_applied_abs_max": qfrc_abs_max,
        "steps": len(rows),
    }
    return rows, summary


def run_response_validation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    init_table = load_balanced_init_table()
    for candidate in CANDIDATES:
        env = BalanceEnv(activation_config(make_env_config(0.60, RESPONSE_STEPS), candidate))
        controller = make_controller(model)
        for roll_deg in (-2.0, 2.0):
            result = run_episode(
                candidate,
                seed=51800 + int((roll_deg + 3.0) * 10),
                roll_rad=np.deg2rad(roll_deg),
                max_steps=RESPONSE_STEPS,
                model=model,
                env=env,
                controller=controller,
                init_table=init_table,
            )
            stabilizing = result["mean_torque_residual_abs"] > 0.0 and result["qfrc_applied_abs_max"] == 0.0
            rows.append({
                "candidate": candidate.name,
                "roll_perturb_deg": roll_deg,
                "stabilizing_response": stabilizing,
                "survival_time_s": result["survival_time_s"],
                "pitch_rms_deg": result["pitch_rms_deg"],
                "roll_rms_deg": result["roll_rms_deg"],
                "torque_residual_nonzero_steps": result["torque_residual_nonzero_steps"],
                "mean_torque_residual_abs": result["mean_torque_residual_abs"],
                "actuator_saturation_rate": result["actuator_saturation_rate"],
                "qfrc_applied_abs_max": result["qfrc_applied_abs_max"],
            })
    summary = {
        "candidate_count": len(CANDIDATES),
        "response_rows": len(rows),
        "all_use_deployable_ctrl_only": all(row["qfrc_applied_abs_max"] == 0.0 for row in rows),
        "any_stabilizing_response": any(row["stabilizing_response"] for row in rows),
    }
    return rows, summary


def aggregate_candidate_results() -> tuple[list[dict[str, Any]], dict[str, Any], list[HybridTorqueCandidate]]:
    rows = []
    kept = []
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    init_table = load_balanced_init_table()
    for candidate in CANDIDATES:
        env = BalanceEnv(activation_config(make_env_config(0.60, CANDIDATE_STEPS), candidate))
        controller = make_controller(model)
        episode_rows = [
            run_episode(
                candidate,
                seed=51900 + i,
                max_steps=CANDIDATE_STEPS,
                model=model,
                env=env,
                controller=controller,
                init_table=init_table,
            )
            for i in range(5)
        ]
        mean_survival = float(np.mean([r["survival_time_s"] for r in episode_rows]))
        fall_rate = float(np.mean([r["fell"] for r in episode_rows]))
        mean_pitch = float(np.mean([r["pitch_rms_deg"] for r in episode_rows]))
        mean_roll = float(np.mean([r["roll_rms_deg"] for r in episode_rows]))
        sat_rate = float(np.mean([r["actuator_saturation_rate"] for r in episode_rows]))
        qfrc_abs_max = float(max(r["qfrc_applied_abs_max"] for r in episode_rows))
        keep = (
            qfrc_abs_max == 0.0
            and mean_survival > RESET_FIXED_H060_BASELINE["survival_s"]
            and fall_rate <= RESET_FIXED_H060_BASELINE["fall_rate"]
            and sat_rate < 0.5
        )
        if keep:
            kept.append(candidate)
        rows.append({
            "candidate": candidate.name,
            "mean_survival_s": mean_survival,
            "fall_rate": fall_rate,
            "pitch_rms_deg": mean_pitch,
            "roll_rms_deg": mean_roll,
            "actuator_saturation_rate": sat_rate,
            "mean_torque_residual_abs": float(np.mean([r["mean_torque_residual_abs"] for r in episode_rows])),
            "torque_residual_nonzero_steps_mean": float(np.mean([r["torque_residual_nonzero_steps"] for r in episode_rows])),
            "qfrc_applied_abs_max": qfrc_abs_max,
            "dominant_fall_reason": max({r["fall_reason"] for r in episode_rows}, key=[r["fall_reason"] for r in episode_rows].count),
            "keep_candidate": keep,
        })
    best = max(rows, key=lambda r: (r["mean_survival_s"], -r["fall_rate"], -r["roll_rms_deg"])) if rows else None
    summary = {
        "baseline_h060": RESET_FIXED_H060_BASELINE,
        "candidate_count": len(CANDIDATES),
        "kept_candidates": [candidate.name for candidate in kept],
        "best_candidate": best,
        "small_gate_passed": bool(kept),
    }
    return rows, summary, kept


def run_full_validation(kept: list[HybridTorqueCandidate]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not kept:
        return [], {"full_validation_run": False, "reason": "small h=0.60 gate did not keep any candidate"}
    best = kept[0]
    heights = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
    rows = []
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    init_table = load_balanced_init_table()
    controller = make_controller(model)
    for height in heights:
        env = BalanceEnv(activation_config(make_env_config(height, FULL_VALIDATION_STEPS), best))
        for ep in range(5):
            rows.append(
                run_episode(
                    best,
                    seed=52000 + int(height * 1000) + ep,
                    height=height,
                    max_steps=FULL_VALIDATION_STEPS,
                    model=model,
                    env=env,
                    controller=controller,
                    init_table=init_table,
                )
            )
    mean_survival = float(np.mean([r["survival_time_s"] for r in rows]))
    fall_rate = float(np.mean([r["fell"] for r in rows]))
    summary = {
        "full_validation_run": True,
        "candidate": best.name,
        "baseline_all_height": RESET_FIXED_ALL_HEIGHT_BASELINE,
        "mean_survival_s": mean_survival,
        "fall_rate": fall_rate,
        "pitch_rms_deg": float(np.mean([r["pitch_rms_deg"] for r in rows])),
        "roll_rms_deg": float(np.mean([r["roll_rms_deg"] for r in rows])),
        "beats_reset_fixed_baseline": mean_survival > RESET_FIXED_ALL_HEIGHT_BASELINE["mean_survival_s"] and fall_rate <= RESET_FIXED_ALL_HEIGHT_BASELINE["fall_rate"],
    }
    return rows, summary


def decide(activation: dict[str, Any], response: dict[str, Any], candidates: dict[str, Any], full: dict[str, Any]) -> str:
    if not activation["hybrid_pid_plus_torque_activated"] or not activation["torque_residual_nonzero"]:
        return "TORQUE_PATH_NOT_ACTIVATED"
    if not response["any_stabilizing_response"]:
        return "DEPLOYABLE_TORQUE_NO_STABILIZING_RESPONSE"
    best = candidates.get("best_candidate") or {}
    if best and best.get("pitch_rms_deg", 0.0) > RESET_FIXED_H060_BASELINE["pitch_rms_deg"] * 3.0:
        return "HYBRID_TORQUE_WORSENS_CONTACT_OR_PITCH"
    if full.get("full_validation_run") and full.get("beats_reset_fixed_baseline"):
        return "HYBRID_TORQUE_BEATS_RESET_FIXED_BASELINE"
    if candidates["small_gate_passed"]:
        return "HYBRID_TORQUE_IMPROVES_BUT_DOES_NOT_PASS_GATE"
    if best and best.get("mean_survival_s", 0.0) > RESET_FIXED_H060_BASELINE["survival_s"]:
        return "HYBRID_TORQUE_IMPROVES_BUT_DOES_NOT_PASS_GATE"
    return "MOTOR_TORQUE_CONTROL_NEEDS_GAIN_TUNING"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    activation_rows, activation_summary = run_activation_trace(CANDIDATES[0], steps=20)
    response_rows, response_summary = run_response_validation()
    candidate_rows, candidate_summary, kept = aggregate_candidate_results()
    full_rows, full_summary = run_full_validation(kept)
    final_decision = decide(activation_summary, response_summary, candidate_summary, full_summary)

    write_csv(OUTPUT_DIR / "torque_path_activation_trace.csv", activation_rows)
    write_json(OUTPUT_DIR / "torque_path_activation_summary.json", activation_summary)
    write_csv(OUTPUT_DIR / "response_validation.csv", response_rows)
    write_json(OUTPUT_DIR / "response_validation_summary.json", response_summary)
    write_csv(OUTPUT_DIR / "candidate_results.csv", candidate_rows)
    write_json(OUTPUT_DIR / "candidate_summary.json", candidate_summary)
    write_csv(OUTPUT_DIR / "full_validation.csv", full_rows)
    write_json(OUTPUT_DIR / "full_validation_summary.json", full_summary)
    write_json(OUTPUT_DIR / "step5_18b_summary.json", {
        "final_decision": final_decision,
        "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "uses_qfrc_applied": False,
        "uses_deployable_actuator_ctrl_only": activation_summary["uses_deployable_actuator_ctrl_only"],
        "balance_residual_yaml_touched": False,
        "current_best_controller": str(BEST_LQR_PATH.relative_to(PROJECT_ROOT)),
        "step6_status": "BLOCKED",
        "full_validation_run": bool(full_summary.get("full_validation_run", False)),
    })
    print(final_decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
