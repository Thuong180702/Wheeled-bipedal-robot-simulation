"""Phase B.9 Step 5.15: minimal VMC / whole-body force distribution evaluation."""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase_b9_posture_geometry_inspection import contact_forces_by_wheel, wheel_bottom_heights  # noqa: E402
from scripts.phase_b9_step5_13_reset_equilibrium_fix import expected_weight, quat_to_rpy, rpy_to_quat  # noqa: E402
from scripts.phase_b9_step5_14_lateral_balance_layer import refresh_balance_obs_after_data_edit  # noqa: E402
from scripts.phase_b9_step5_lqr_gain_strengthening import (  # noqa: E402
    apply_balanced_root_init,
    create_tuned_controller,
    load_balanced_init_table,
)
from wheeled_biped.controllers.action_codec import L_HIP_ROLL, L_KNEE, R_HIP_ROLL, R_KNEE  # noqa: E402
from wheeled_biped.controllers.dual_rate_balance_controller import DualRateBalanceController, DualRateConfig  # noqa: E402
from wheeled_biped.envs.balance_env import BalanceEnv  # noqa: E402
from wheeled_biped.utils.config import get_model_path  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase_b9_step5_15_vmc_whole_body_layer"
BEST_LQR_PATH = PROJECT_ROOT / "outputs" / "phase_b9_lqr_gain_strengthening" / "best_lqr_config.yaml"
CONTROLLER_CONFIG_PATH = PROJECT_ROOT / "configs" / "controllers" / "dual_rate_balance_controller_b9.yaml"
STEP513_REVALIDATION_CSV = PROJECT_ROOT / "outputs" / "phase_b9_step5_13_reset_equilibrium_fix" / "step5_revalidation_after_reset_fix.csv"
STEP513_REVALIDATION_SUMMARY = PROJECT_ROOT / "outputs" / "phase_b9_step5_13_reset_equilibrium_fix" / "step5_revalidation_after_reset_fix_summary.json"
BALANCE_RESIDUAL_PATH = PROJECT_ROOT / "configs" / "training" / "balance_residual.yaml"

BASELINE_HEIGHTS = [0.60, 0.40]
VALID_HEIGHTS = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
EPISODES_PER_HEIGHT = 5
MAX_STEPS = 1000


@dataclass(frozen=True)
class VmcCandidate:
    name: str
    mapping: str
    k_roll: float = 0.0
    k_roll_rate: float = 0.0
    k_com_y: float = 0.0
    k_com_y_rate: float = 0.0
    k_force_diff: float = 0.0
    a_roll: float = 0.0
    a_com: float = 0.0
    a_force: float = 0.0
    max_delta_support: float = 0.0
    max_hip_roll_correction: float = 0.0
    max_leg_length_correction: float = 0.0
    sign: float = 1.0


RESPONSE_MAPPINGS = [
    VmcCandidate("no_vmc", "disabled"),
    VmcCandidate("differential_hip_roll_plus_height_knee", "hip_roll_leg_length", k_roll=0.7, k_roll_rate=0.06, a_roll=1.0, max_delta_support=0.12, max_hip_roll_correction=0.08, max_leg_length_correction=0.04),
    VmcCandidate("differential_leg_length_knee_only", "leg_length_only", k_roll=0.7, k_roll_rate=0.06, a_roll=1.0, max_delta_support=0.12, max_leg_length_correction=0.06),
    VmcCandidate("hip_roll_plus_differential_leg_length", "hip_roll_leg_length", k_roll=0.7, k_roll_rate=0.06, a_roll=1.0, max_delta_support=0.12, max_hip_roll_correction=0.06, max_leg_length_correction=0.06),
    VmcCandidate("contact_force_balancing_mode", "force_balance_only", k_force_diff=0.10, a_force=1.0, max_delta_support=0.10, max_leg_length_correction=0.05),
    VmcCandidate("weak_combined_vmc", "combined_weak", k_roll=0.45, k_roll_rate=0.04, k_com_y=0.04, k_com_y_rate=0.02, k_force_diff=0.05, a_roll=1.0, a_com=0.5, a_force=0.5, max_delta_support=0.10, max_hip_roll_correction=0.04, max_leg_length_correction=0.04),
]

CANDIDATES = [
    VmcCandidate("vmc_force_balance_only", "force_balance_only", k_force_diff=0.10, a_force=1.0, max_delta_support=0.10, max_leg_length_correction=0.05),
    VmcCandidate("vmc_roll_torque_only", "hip_roll_leg_length", k_roll=0.7, k_roll_rate=0.06, a_roll=1.0, max_delta_support=0.12, max_hip_roll_correction=0.06, max_leg_length_correction=0.06),
    VmcCandidate("vmc_com_y_only", "leg_length_only", k_com_y=0.08, k_com_y_rate=0.04, a_com=1.0, max_delta_support=0.08, max_leg_length_correction=0.04),
    VmcCandidate("vmc_roll_plus_force_balance", "hip_roll_leg_length", k_roll=0.6, k_roll_rate=0.05, k_force_diff=0.06, a_roll=1.0, a_force=0.5, max_delta_support=0.12, max_hip_roll_correction=0.05, max_leg_length_correction=0.05),
    VmcCandidate("vmc_roll_plus_com_y", "hip_roll_leg_length", k_roll=0.6, k_roll_rate=0.05, k_com_y=0.05, k_com_y_rate=0.025, a_roll=1.0, a_com=0.5, max_delta_support=0.10, max_hip_roll_correction=0.04, max_leg_length_correction=0.05),
    VmcCandidate("vmc_combined_weak", "combined_weak", k_roll=0.45, k_roll_rate=0.04, k_com_y=0.04, k_com_y_rate=0.02, k_force_diff=0.05, a_roll=1.0, a_com=0.5, a_force=0.5, max_delta_support=0.10, max_hip_roll_correction=0.04, max_leg_length_correction=0.04),
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_best_lqr_params() -> dict[str, float]:
    return yaml.safe_load(BEST_LQR_PATH.read_text(encoding="utf-8"))


def make_env() -> BalanceEnv:
    return BalanceEnv({
        "episode_length": MAX_STEPS,
        "low_level_pid": {"enabled": True, "disable_pid_action_bias": True},
        "domain_randomization": {"enabled": False},
        "sensor_noise": {"enabled": False},
    })


def make_controller(model: mujoco.MjModel, candidate: VmcCandidate | None = None) -> DualRateBalanceController:
    base_config = DualRateConfig.from_yaml(CONTROLLER_CONFIG_PATH)
    controller = create_tuned_controller(base_config, load_best_lqr_params(), model)
    controller.slow_loop_interval = 999999
    controller.config.lateral_balance_enabled = False
    if candidate and candidate.name != "no_vmc" and candidate.max_delta_support > 0.0:
        controller.config.vmc_enabled = True
        controller.config.vmc_mapping = candidate.mapping
        controller.config.vmc_k_roll = candidate.k_roll
        controller.config.vmc_k_roll_rate = candidate.k_roll_rate
        controller.config.vmc_k_com_y = candidate.k_com_y
        controller.config.vmc_k_com_y_rate = candidate.k_com_y_rate
        controller.config.vmc_k_force_diff = candidate.k_force_diff
        controller.config.vmc_a_roll = candidate.a_roll
        controller.config.vmc_a_com = candidate.a_com
        controller.config.vmc_a_force = candidate.a_force
        controller.config.vmc_max_delta_support = candidate.max_delta_support
        controller.config.vmc_max_hip_roll_correction = candidate.max_hip_roll_correction
        controller.config.vmc_max_leg_length_correction = candidate.max_leg_length_correction
        controller.config.vmc_sign = candidate.sign
    return controller


def freeze_posture(controller: DualRateBalanceController, height: float, table: dict[float, dict[str, float]]) -> None:
    nearest = min(table.keys(), key=lambda h: abs(h - height))
    init = table[nearest]
    controller.target_hip_pitch = float(init["hip_pitch"])
    controller.target_knee = float(init["knee"])
    controller.last_stable_hip_pitch = controller.target_hip_pitch
    controller.last_stable_knee = controller.target_knee


def cpu_contact_snapshot(model: mujoco.MjModel, mjx_data, height: float, phase: str) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:] = np.array(mjx_data.qpos)
    data.qvel[:] = np.array(mjx_data.qvel)
    mujoco.mj_forward(model, data)
    left_clear, right_clear = wheel_bottom_heights(model, data)
    left_force, right_force = contact_forces_by_wheel(model, data)
    left_force = 0.0 if math.isnan(left_force) else float(left_force)
    right_force = 0.0 if math.isnan(right_force) else float(right_force)
    total_force = left_force + right_force
    force_diff = left_force - right_force
    roll, pitch, _ = quat_to_rpy(data.qpos[3:7].copy())
    return {
        "height": height,
        "phase": phase,
        "left_clearance_m": float(left_clear),
        "right_clearance_m": float(right_clear),
        "min_clearance_m": float(min(left_clear, right_clear)),
        "left_force_n": left_force,
        "right_force_n": right_force,
        "total_force_n": total_force,
        "force_diff_n": force_diff,
        "normalized_force_diff": force_diff / expected_weight(model),
        "force_imbalance_ratio": abs(force_diff) / total_force if total_force > 1e-9 else math.inf,
        "force_to_weight_ratio": total_force / expected_weight(model),
        "roll_deg": math.degrees(roll),
        "pitch_deg": math.degrees(pitch),
        "roll_rate_rad_s": float(data.qvel[3]),
        "pitch_rate_rad_s": float(data.qvel[4]),
        "severe_penetration": bool(min(left_clear, right_clear) < -0.01),
        "multi_kn_contact": bool(total_force > 2000.0),
        "wheel_unloaded": bool(min(left_force, right_force) < 1e-6),
    }


def obs_roll_pitch(obs: np.ndarray) -> tuple[float, float]:
    gravity_body = obs[0:3]
    pitch = float(np.arcsin(np.clip(-gravity_body[0], -1.0, 1.0)))
    roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
    return roll, pitch


def load_reset_fixed_baselines() -> tuple[dict[str, float], dict[str, float]]:
    all_height = json.loads(STEP513_REVALIDATION_SUMMARY.read_text(encoding="utf-8"))
    with STEP513_REVALIDATION_CSV.open("r", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if abs(float(row["height"]) - 0.60) < 1e-9]
    h060 = {
        "mean_survival_s": float(np.mean([float(row["survival_time_s"]) for row in rows])),
        "fall_rate": float(np.mean([row["fell"] == "True" for row in rows])),
        "mean_pitch_rms_deg": float(np.mean([float(row["pitch_rms_deg"]) for row in rows])),
        "mean_roll_rms_deg": float(np.mean([float(row["roll_rms_deg"]) for row in rows])),
        "mean_action_saturation_rate": float(np.mean([float(row["action_saturation_rate"]) for row in rows])),
    }
    all_baseline = {
        "mean_survival_s": float(all_height["mean_survival_s"]),
        "fall_rate": float(all_height["fall_rate"]),
        "mean_pitch_rms_deg": float(all_height["mean_pitch_rms_deg"]),
        "mean_roll_rms_deg": float(all_height["mean_roll_rms_deg"]),
        "mean_action_saturation_rate": float(all_height["mean_action_saturation_rate"]),
    }
    return h060, all_baseline


def run_episode(model: mujoco.MjModel, env: BalanceEnv, table: dict[float, dict[str, float]], height: float, episode: int, candidate: VmcCandidate | None) -> dict[str, Any]:
    controller = make_controller(model, candidate)
    rng = jax.random.PRNGKey(51_500 + episode + int(height * 1000))
    state = env.reset(rng)
    state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, table))
    controller.reset()
    freeze_posture(controller, height, table)

    pitch_sq = 0.0
    roll_sq = 0.0
    roll_abs_max = 0.0
    action_sat = 0
    correction_abs_sum = 0.0
    force_imbalance_max = 0.0
    clearance_abs_max = 0.0
    wheel_unload_count = 0
    first_failure_variable = "none"
    steps = 0

    for _ in range(MAX_STEPS):
        obs_np = np.array(state.obs)
        roll, pitch = obs_roll_pitch(obs_np)
        snap = cpu_contact_snapshot(model, state.mjx_data, height, "during_rollout")
        force_error = snap["normalized_force_diff"] if math.isfinite(float(snap["normalized_force_diff"])) else 0.0
        controller.config.vmc_external_force_diff_error = float(np.clip(force_error, -1.0, 1.0))
        action = controller.compute_action(obs_np)
        telemetry = controller.get_telemetry()["vmc_whole_body"]

        pitch_sq += pitch ** 2
        roll_sq += roll ** 2
        roll_abs_max = max(roll_abs_max, abs(roll))
        action_sat += int(np.max(np.abs(action)) >= 0.99)
        correction_abs_sum += abs(float(telemetry["delta_support"]))
        force_imbalance_max = max(force_imbalance_max, float(snap["force_imbalance_ratio"]) if math.isfinite(float(snap["force_imbalance_ratio"])) else 0.0)
        clearance_abs_max = max(clearance_abs_max, abs(float(snap["left_clearance_m"])), abs(float(snap["right_clearance_m"])))
        wheel_unload_count += int(bool(snap["wheel_unloaded"]))

        state = env.step(state, jnp.array(action))
        steps += 1

        if bool(state.done):
            torso_height = float(state.mjx_data.qpos[2])
            tilt = float(np.arccos(np.clip(-np.array(state.obs)[2], -1.0, 1.0)))
            if torso_height < env._min_height:
                first_failure_variable = "height"
            elif tilt > env._max_tilt:
                first_failure_variable = "tilt"
            else:
                first_failure_variable = "unknown"
            break

    fell = bool(state.info["is_fallen"])
    return {
        "candidate": candidate.name if candidate else "baseline",
        "height": height,
        "episode": episode,
        "steps": steps,
        "survival_time_s": steps * env.CONTROL_DT,
        "fell": fell,
        "fall_reason": first_failure_variable if fell else "none",
        "pitch_rms_deg": math.degrees(math.sqrt(pitch_sq / steps)) if steps else 0.0,
        "roll_rms_deg": math.degrees(math.sqrt(roll_sq / steps)) if steps else 0.0,
        "roll_abs_max_deg": math.degrees(roll_abs_max),
        "action_saturation_rate": action_sat / steps if steps else 0.0,
        "first_failure_variable": first_failure_variable,
        "max_contact_force_imbalance_ratio": force_imbalance_max,
        "max_abs_clearance_m": clearance_abs_max,
        "wheel_unloading_rate": wheel_unload_count / steps if steps else 0.0,
        "mean_vmc_delta_support_abs": correction_abs_sum / steps if steps else 0.0,
    }


def summarize(rows: list[dict[str, Any]], group_key: str = "candidate") -> list[dict[str, Any]]:
    out = []
    for key in sorted({row[group_key] for row in rows}):
        group = [row for row in rows if row[group_key] == key]
        fall_reasons = [row["fall_reason"] for row in group]
        dominant = max(set(fall_reasons), key=fall_reasons.count)
        out.append({
            group_key: key,
            "episodes": len(group),
            "mean_survival_s": float(np.mean([row["survival_time_s"] for row in group])),
            "fall_rate": float(np.mean([row["fell"] for row in group])),
            "mean_pitch_rms_deg": float(np.mean([row["pitch_rms_deg"] for row in group])),
            "mean_roll_rms_deg": float(np.mean([row["roll_rms_deg"] for row in group])),
            "mean_roll_abs_max_deg": float(np.mean([row["roll_abs_max_deg"] for row in group])),
            "mean_action_saturation_rate": float(np.mean([row["action_saturation_rate"] for row in group])),
            "max_contact_force_imbalance_ratio": float(np.max([row["max_contact_force_imbalance_ratio"] for row in group])),
            "max_abs_clearance_m": float(np.max([row["max_abs_clearance_m"] for row in group])),
            "mean_wheel_unloading_rate": float(np.mean([row["wheel_unloading_rate"] for row in group])),
            "mean_vmc_delta_support_abs": float(np.mean([row["mean_vmc_delta_support_abs"] for row in group])),
            "dominant_fall_reason": dominant,
        })
    return out


def run_baseline_trace(model: mujoco.MjModel, env: BalanceEnv, table: dict[float, dict[str, float]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for height in BASELINE_HEIGHTS:
        for episode in range(EPISODES_PER_HEIGHT):
            rows.append(run_episode(model, env, table, height, episode, None))
    summary_rows = summarize(rows)
    return rows, {"by_candidate": summary_rows, "episodes": len(rows)}


def run_contact_force_response(model: mujoco.MjModel, env: BalanceEnv, table: dict[float, dict[str, float]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for perturb_deg in [2.0, -2.0]:
        for candidate in RESPONSE_MAPPINGS:
            controller = make_controller(model, candidate)
            state = env.reset(jax.random.PRNGKey(53_000 + int(perturb_deg * 10)))
            state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, 0.60, table))
            qpos = state.mjx_data.qpos.at[3:7].set(jnp.array(rpy_to_quat(math.radians(perturb_deg), 0.0, 0.0), dtype=state.mjx_data.qpos.dtype))
            state = state._replace(mjx_data=state.mjx_data.replace(qpos=qpos, qvel=jnp.zeros_like(state.mjx_data.qvel)))
            state = state._replace(obs=refresh_balance_obs_after_data_edit(env, state))
            controller.reset()
            freeze_posture(controller, 0.60, table)
            obs0 = np.array(state.obs)
            roll0, pitch0 = obs_roll_pitch(obs0)
            pre = cpu_contact_snapshot(model, state.mjx_data, 0.60, "pre_response")
            controller.config.vmc_external_force_diff_error = float(np.clip(pre["normalized_force_diff"], -1.0, 1.0))
            action = controller.compute_action(obs0)
            telemetry = controller.get_telemetry()["vmc_whole_body"]
            state1 = env.step(state, jnp.array(action))
            post = cpu_contact_snapshot(model, state1.mjx_data, 0.60, "post_response")
            obs1 = np.array(state1.obs)
            roll1, pitch1 = obs_roll_pitch(obs1)
            force_diff_moves_to_zero = abs(post["force_diff_n"]) < abs(pre["force_diff_n"])
            roll_magnitude_reduced = abs(roll1) < abs(roll0)
            rows.append({
                "mapping": candidate.name,
                "perturb_deg": perturb_deg,
                "roll0_deg": math.degrees(roll0),
                "roll1_deg": math.degrees(roll1),
                "roll_delta_deg": math.degrees(roll1 - roll0),
                "roll_magnitude_reduced": roll_magnitude_reduced,
                "pitch_delta_deg": math.degrees(pitch1 - pitch0),
                "pre_left_force_n": pre["left_force_n"],
                "pre_right_force_n": pre["right_force_n"],
                "post_left_force_n": post["left_force_n"],
                "post_right_force_n": post["right_force_n"],
                "pre_force_diff_n": pre["force_diff_n"],
                "post_force_diff_n": post["force_diff_n"],
                "force_diff_moves_to_zero": force_diff_moves_to_zero,
                "delta_support": telemetry["delta_support"],
                "hip_roll_correction": telemetry["hip_roll_correction"],
                "leg_length_correction": telemetry["leg_length_correction"],
                "left_hip_roll_action": float(action[L_HIP_ROLL]),
                "right_hip_roll_action": float(action[R_HIP_ROLL]),
                "left_knee_action": float(action[L_KNEE]),
                "right_knee_action": float(action[R_KNEE]),
                "min_clearance_m": post["min_clearance_m"],
                "wheel_unloaded": post["wheel_unloaded"],
                "action_saturated": bool(np.max(np.abs(action)) >= 0.99),
            })
    enabled = [row for row in rows if row["mapping"] != "no_vmc"]
    force_authority_rows = [row for row in enabled if abs(float(row["post_force_diff_n"] - row["pre_force_diff_n"])) > 1e-6]
    stabilizing_rows = [row for row in enabled if row["force_diff_moves_to_zero"] or row["roll_magnitude_reduced"]]
    pitch_bad = any(abs(float(row["pitch_delta_deg"])) > 5.0 for row in enabled)
    clearance_bad = any(float(row["min_clearance_m"]) < -0.01 for row in enabled)
    summary = {
        "rows": len(rows),
        "force_authority_detected": len(force_authority_rows) > 0,
        "stabilizing_response_detected": len(stabilizing_rows) > 0,
        "pitch_disturbance_bad": pitch_bad,
        "clearance_bad": clearance_bad,
        "vmc_mapping_has_force_authority": len(force_authority_rows) > 0 and not pitch_bad and not clearance_bad,
    }
    return rows, summary


def run_small_eval(model: mujoco.MjModel, env: BalanceEnv, table: dict[float, dict[str, float]], h060_baseline: dict[str, float]) -> tuple[list[dict[str, Any]], dict[str, Any], list[VmcCandidate]]:
    rows = []
    for candidate in CANDIDATES:
        for episode in range(EPISODES_PER_HEIGHT):
            rows.append(run_episode(model, env, table, 0.60, episode, candidate))
    summaries = summarize(rows)
    kept = []
    for candidate in CANDIDATES:
        s = next(row for row in summaries if row["candidate"] == candidate.name)
        passes = (
            s["mean_survival_s"] > h060_baseline["mean_survival_s"]
            and (s["mean_roll_rms_deg"] < h060_baseline["mean_roll_rms_deg"] or s["mean_roll_abs_max_deg"] < 20.0)
            and s["mean_pitch_rms_deg"] <= h060_baseline["mean_pitch_rms_deg"] + 2.0
            and s["mean_wheel_unloading_rate"] < 0.95
            and s["max_abs_clearance_m"] < 0.05
            and s["mean_action_saturation_rate"] < 0.05
        )
        s["kept"] = passes
        if passes:
            kept.append(candidate)
    return rows, {"h060_baseline": h060_baseline, "candidates": summaries, "kept_candidates": [c.name for c in kept]}, kept


def run_full_validation(model: mujoco.MjModel, env: BalanceEnv, table: dict[float, dict[str, float]], candidates: list[VmcCandidate], all_baseline: dict[str, float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for candidate in candidates[:2]:
        for height in VALID_HEIGHTS:
            for episode in range(EPISODES_PER_HEIGHT):
                rows.append(run_episode(model, env, table, height, episode, candidate))
    summaries = summarize(rows) if rows else []
    best = None
    for s in summaries:
        s["beats_reset_fixed_baseline"] = (
            s["mean_survival_s"] > all_baseline["mean_survival_s"]
            and s["fall_rate"] < all_baseline["fall_rate"]
            and s["mean_roll_rms_deg"] < all_baseline["mean_roll_rms_deg"]
            and s["mean_pitch_rms_deg"] <= all_baseline["mean_pitch_rms_deg"] + 2.0
            and s["mean_action_saturation_rate"] < 0.05
        )
        if best is None or s["mean_survival_s"] > best["mean_survival_s"]:
            best = s
    return rows, {"all_height_baseline": all_baseline, "candidates": summaries, "best": best}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    balance_residual_before = BALANCE_RESIDUAL_PATH.read_text(encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    env = make_env()
    table = load_balanced_init_table()
    h060_baseline, all_baseline = load_reset_fixed_baselines()

    baseline_rows, baseline_summary = run_baseline_trace(model, env, table)
    baseline_summary["h060_reset_fixed_baseline"] = h060_baseline
    baseline_summary["all_height_reset_fixed_baseline"] = all_baseline
    write_csv(OUTPUT_DIR / "baseline_trace.csv", baseline_rows)
    write_json(OUTPUT_DIR / "baseline_trace_summary.json", baseline_summary)

    response_rows, response_summary = run_contact_force_response(model, env, table)
    write_csv(OUTPUT_DIR / "contact_force_response.csv", response_rows)
    write_json(OUTPUT_DIR / "contact_force_response_summary.json", response_summary)
    if not response_summary["vmc_mapping_has_force_authority"]:
        write_json(OUTPUT_DIR / "candidate_summary.json", {
            "final_decision": "A. VMC_MAPPING_NO_FORCE_AUTHORITY",
            "h060_baseline": h060_baseline,
            "balance_residual_unchanged": BALANCE_RESIDUAL_PATH.read_text(encoding="utf-8") == balance_residual_before,
            "step6_status": "BLOCKED",
        })
        print("VMC_MAPPING_NO_FORCE_AUTHORITY")
        return

    candidate_rows, candidate_summary, kept = run_small_eval(model, env, table, h060_baseline)
    write_csv(OUTPUT_DIR / "candidate_results.csv", candidate_rows)
    write_json(OUTPUT_DIR / "candidate_summary.json", candidate_summary)

    full_rows: list[dict[str, Any]] = []
    full_fieldnames = [
        "candidate", "height", "episode", "steps", "survival_time_s", "fell", "fall_reason",
        "pitch_rms_deg", "roll_rms_deg", "roll_abs_max_deg", "action_saturation_rate",
        "first_failure_variable", "max_contact_force_imbalance_ratio", "max_abs_clearance_m",
        "wheel_unloading_rate", "mean_vmc_delta_support_abs",
    ]
    full_summary: dict[str, Any] = {"all_height_baseline": all_baseline, "candidates": [], "best": None}
    final_decision = "E. DEEPER_WHOLE_BODY_CONTROLLER_REQUIRED"
    if kept:
        full_rows, full_summary = run_full_validation(model, env, table, kept, all_baseline)
        if any(row.get("beats_reset_fixed_baseline") for row in full_summary["candidates"]):
            final_decision = "D. VMC_BEATS_RESET_FIXED_BASELINE"
            best = full_summary["best"]
            best_candidate = next(c for c in kept if c.name == best["candidate"])
            write_json(OUTPUT_DIR / "best_vmc_summary.json", best)
            best_config = {
                "source_controller": "outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml",
                "vmc_whole_body": best_candidate.__dict__,
            }
            (OUTPUT_DIR / "best_vmc_config.yaml").write_text(yaml.dump(best_config, sort_keys=False), encoding="utf-8")
        else:
            final_decision = "C. VMC_IMPROVES_BUT_DOES_NOT_PASS_GATE"
    write_csv(OUTPUT_DIR / "full_validation.csv", full_rows, full_fieldnames)
    full_summary["final_decision"] = final_decision
    full_summary["balance_residual_unchanged"] = BALANCE_RESIDUAL_PATH.read_text(encoding="utf-8") == balance_residual_before
    full_summary["step6_status"] = "BLOCKED"
    write_json(OUTPUT_DIR / "full_validation_summary.json", full_summary)

    print(json.dumps({"kept": [c.name for c in kept], "final_decision": final_decision}, indent=2))


if __name__ == "__main__":
    main()
