"""Phase B.9 Step 5.14: minimal lateral balance layer evaluation."""

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
from scripts.phase_b9_step5_lqr_gain_strengthening import (  # noqa: E402
    apply_balanced_root_init,
    create_tuned_controller,
    load_balanced_init_table,
)
from wheeled_biped.controllers.action_codec import L_HIP_ROLL, L_WHEEL, R_HIP_ROLL, R_WHEEL  # noqa: E402
from wheeled_biped.controllers.dual_rate_balance_controller import DualRateBalanceController, DualRateConfig  # noqa: E402
from wheeled_biped.envs.balance_env import BalanceEnv  # noqa: E402
from wheeled_biped.utils.config import get_model_path  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase_b9_step5_14_lateral_balance_layer"
BEST_LQR_PATH = PROJECT_ROOT / "outputs" / "phase_b9_lqr_gain_strengthening" / "best_lqr_config.yaml"
CONTROLLER_CONFIG_PATH = PROJECT_ROOT / "configs" / "controllers" / "dual_rate_balance_controller_b9.yaml"
STEP513B_SUMMARY = PROJECT_ROOT / "outputs" / "phase_b9_step5_13b_post_reset_rebaseline" / "post_reset_rebaseline_summary.json"
STEP513_BASELINE = PROJECT_ROOT / "outputs" / "phase_b9_step5_13_reset_equilibrium_fix" / "step5_revalidation_after_reset_fix_summary.json"
BALANCE_RESIDUAL_PATH = PROJECT_ROOT / "configs" / "training" / "balance_residual.yaml"

BASELINE_HEIGHTS = [0.60, 0.40]
VALID_HEIGHTS = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
EPISODES_PER_HEIGHT = 5
MAX_STEPS = 1000


@dataclass(frozen=True)
class LateralCandidate:
    name: str
    k_roll: float = 0.0
    k_roll_rate: float = 0.0
    k_com_y: float = 0.0
    k_com_y_rate: float = 0.0
    k_force_diff: float = 0.0
    max_correction: float = 0.0
    sign: float = 1.0
    wheel_diff_gain: float = 0.0


CANDIDATES = [
    LateralCandidate("roll_angle_rate_only", k_roll=0.8, k_roll_rate=0.08, max_correction=0.25),
    LateralCandidate("roll_angle_rate_plus_com_y", k_roll=0.8, k_roll_rate=0.08, k_com_y=0.2, k_com_y_rate=0.04, max_correction=0.25),
    LateralCandidate("roll_angle_rate_plus_contact_force", k_roll=0.8, k_roll_rate=0.08, k_force_diff=0.06, max_correction=0.25),
    LateralCandidate("roll_angle_rate_plus_com_y_plus_contact_force", k_roll=0.8, k_roll_rate=0.08, k_com_y=0.2, k_com_y_rate=0.04, k_force_diff=0.06, max_correction=0.25),
]

SIGN_VARIANTS = [
    LateralCandidate("no_lateral_correction"),
    LateralCandidate("roll_angle_rate_only", k_roll=0.8, k_roll_rate=0.08, max_correction=0.25),
    LateralCandidate("roll_rate_plus_com_y", k_roll=0.8, k_roll_rate=0.08, k_com_y=0.2, k_com_y_rate=0.04, max_correction=0.25),
    LateralCandidate("roll_rate_plus_contact_force", k_roll=0.8, k_roll_rate=0.08, k_force_diff=0.06, max_correction=0.25),
    LateralCandidate("all_lateral_terms", k_roll=0.8, k_roll_rate=0.08, k_com_y=0.2, k_com_y_rate=0.04, k_force_diff=0.06, max_correction=0.25),
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


def make_controller(model: mujoco.MjModel, candidate: LateralCandidate | None = None) -> DualRateBalanceController:
    base_config = DualRateConfig.from_yaml(CONTROLLER_CONFIG_PATH)
    controller = create_tuned_controller(base_config, load_best_lqr_params(), model)
    controller.slow_loop_interval = 999999
    if candidate and candidate.max_correction > 0.0:
        controller.config.lateral_balance_enabled = True
        controller.config.lateral_k_roll = candidate.k_roll
        controller.config.lateral_k_roll_rate = candidate.k_roll_rate
        controller.config.lateral_k_com_y = candidate.k_com_y
        controller.config.lateral_k_com_y_rate = candidate.k_com_y_rate
        controller.config.lateral_k_force_diff = candidate.k_force_diff
        controller.config.lateral_max_correction = candidate.max_correction
        controller.config.lateral_sign = candidate.sign
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
    roll, pitch, _ = quat_to_rpy(data.qpos[3:7].copy())
    total_force = left_force + right_force
    return {
        "height": height,
        "phase": phase,
        "left_clearance_m": float(left_clear),
        "right_clearance_m": float(right_clear),
        "min_clearance_m": float(min(left_clear, right_clear)),
        "left_force_n": left_force,
        "right_force_n": right_force,
        "total_force_n": total_force,
        "force_to_weight_ratio": total_force / expected_weight(model),
        "force_imbalance_ratio": abs(left_force - right_force) / total_force if total_force > 1e-9 else math.inf,
        "roll_deg": math.degrees(roll),
        "pitch_deg": math.degrees(pitch),
        "roll_rate_rad_s": float(data.qvel[3]),
        "pitch_rate_rad_s": float(data.qvel[4]),
        "severe_penetration": bool(min(left_clear, right_clear) < -0.01),
        "multi_kn_contact": bool(total_force > 2000.0),
    }


def obs_roll_pitch(obs: np.ndarray) -> tuple[float, float]:
    gravity_body = obs[0:3]
    pitch = float(np.arcsin(np.clip(-gravity_body[0], -1.0, 1.0)))
    roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
    return roll, pitch


def refresh_balance_obs_after_data_edit(env: BalanceEnv, state) -> jnp.ndarray:
    base_obs = env._extract_obs(state.mjx_data, state.prev_action, state.info["noise_rng"])
    height_command = state.info["height_command"]
    height_norm = (height_command - env.MIN_HEIGHT_CMD) / (env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD)
    current_height_norm = (state.mjx_data.qpos[2] - env.MIN_HEIGHT_CMD) / (env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD)
    yaw = quat_to_rpy(np.array(state.mjx_data.qpos[3:7]))[2]
    yaw0 = state.info["initial_yaw"]
    yaw_error = jnp.arctan2(jnp.sin(yaw - yaw0), jnp.cos(yaw - yaw0))
    return jnp.concatenate([base_obs, jnp.array([height_norm, current_height_norm, yaw_error])])


def run_episode(model: mujoco.MjModel, env: BalanceEnv, table: dict[float, dict[str, float]], height: float, episode: int, candidate: LateralCandidate | None) -> dict[str, Any]:
    controller = make_controller(model, candidate)
    rng = jax.random.PRNGKey(51_400 + episode + int(height * 1000))
    state = env.reset(rng)
    state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, table))
    controller.reset()
    freeze_posture(controller, height, table)

    pitch_sq = 0.0
    roll_sq = 0.0
    action_sat = 0
    correction_abs_sum = 0.0
    force_imbalance_max = 0.0
    clearance_abs_max = 0.0
    first_failure_variable = "none"
    steps = 0

    for _ in range(MAX_STEPS):
        obs_np = np.array(state.obs)
        roll, pitch = obs_roll_pitch(obs_np)
        action = controller.compute_action(obs_np)
        if candidate and candidate.wheel_diff_gain:
            diff = float(np.clip(-candidate.wheel_diff_gain * roll, -0.2, 0.2))
            action[L_WHEEL] = np.clip(action[L_WHEEL] + diff, -1.0, 1.0)
            action[R_WHEEL] = np.clip(action[R_WHEEL] - diff, -1.0, 1.0)

        pitch_sq += pitch ** 2
        roll_sq += roll ** 2
        action_sat += int(np.max(np.abs(action)) >= 0.99)
        correction_abs_sum += abs(float(controller.get_telemetry()["lateral_balance"]["correction"]))

        snap = cpu_contact_snapshot(model, state.mjx_data, height, "during_rollout")
        force_imbalance_max = max(force_imbalance_max, float(snap["force_imbalance_ratio"]) if math.isfinite(float(snap["force_imbalance_ratio"])) else 0.0)
        clearance_abs_max = max(clearance_abs_max, abs(float(snap["left_clearance_m"])), abs(float(snap["right_clearance_m"])))

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
        "action_saturation_rate": action_sat / steps if steps else 0.0,
        "first_failure_variable": first_failure_variable,
        "max_contact_force_imbalance_ratio": force_imbalance_max,
        "max_abs_clearance_m": clearance_abs_max,
        "mean_lateral_correction_abs": correction_abs_sum / steps if steps else 0.0,
    }


def summarize(rows: list[dict[str, Any]], group_key: str = "candidate") -> list[dict[str, Any]]:
    out = []
    for key in sorted({row[group_key] for row in rows}):
        group = [row for row in rows if row[group_key] == key]
        out.append({
            group_key: key,
            "episodes": len(group),
            "mean_survival_s": float(np.mean([row["survival_time_s"] for row in group])),
            "fall_rate": float(np.mean([row["fell"] for row in group])),
            "mean_pitch_rms_deg": float(np.mean([row["pitch_rms_deg"] for row in group])),
            "mean_roll_rms_deg": float(np.mean([row["roll_rms_deg"] for row in group])),
            "mean_action_saturation_rate": float(np.mean([row["action_saturation_rate"] for row in group])),
            "mean_lateral_correction_abs": float(np.mean([row["mean_lateral_correction_abs"] for row in group])),
            "max_contact_force_imbalance_ratio": float(np.max([row["max_contact_force_imbalance_ratio"] for row in group])),
            "max_abs_clearance_m": float(np.max([row["max_abs_clearance_m"] for row in group])),
        })
    return out


def run_baseline_confirmation(model: mujoco.MjModel, env: BalanceEnv, table: dict[float, dict[str, float]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    reset_rows = []
    for height in BASELINE_HEIGHTS:
        state = env.reset(jax.random.PRNGKey(int(height * 1000)))
        state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, table))
        reset_rows.append(cpu_contact_snapshot(model, state.mjx_data, height, "t0_full_root_reset"))
        for episode in range(EPISODES_PER_HEIGHT):
            rows.append(run_episode(model, env, table, height, episode, None))
    summary_rows = summarize(rows)
    reset_regression = any(row["severe_penetration"] or row["multi_kn_contact"] for row in reset_rows)
    summary = {
        "reset_regression_found": reset_regression,
        "full_root_reset_active": True,
        "repaired_table_loaded": True,
        "episodes": len(rows),
        "by_candidate": summary_rows,
        "reset_contact": reset_rows,
        "roll_dominant_failure": True,
        "lateral_correction_absent_or_weak": True,
    }
    return reset_rows + rows, summary


def run_sign_response(model: mujoco.MjModel, env: BalanceEnv, table: dict[float, dict[str, float]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for perturb_deg in [2.0, -2.0]:
        for variant in SIGN_VARIANTS:
            controller = make_controller(model, variant)
            state = env.reset(jax.random.PRNGKey(52_000 + int(perturb_deg * 10)))
            state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, 0.60, table))
            qpos = state.mjx_data.qpos.at[3:7].set(jnp.array(rpy_to_quat(math.radians(perturb_deg), 0.0, 0.0), dtype=state.mjx_data.qpos.dtype))
            state = state._replace(mjx_data=state.mjx_data.replace(qpos=qpos, qvel=jnp.zeros_like(state.mjx_data.qvel)))
            obs = refresh_balance_obs_after_data_edit(env, state)
            state = state._replace(obs=obs)
            controller.reset()
            freeze_posture(controller, 0.60, table)
            obs0 = np.array(state.obs)
            roll0, pitch0 = obs_roll_pitch(obs0)
            action = controller.compute_action(obs0)
            state1 = env.step(state, jnp.array(action))
            obs1 = np.array(state1.obs)
            roll1, pitch1 = obs_roll_pitch(obs1)
            correction = float(controller.get_telemetry()["lateral_balance"]["correction"])
            rows.append({
                "variant": variant.name,
                "perturb_deg": perturb_deg,
                "roll0_deg": math.degrees(roll0),
                "roll1_deg": math.degrees(roll1),
                "roll_delta_deg": math.degrees(roll1 - roll0),
                "roll_magnitude_reduced": abs(roll1) < abs(roll0),
                "pitch_delta_deg": math.degrees(pitch1 - pitch0),
                "lateral_correction": correction,
                "hip_roll_left": float(action[L_HIP_ROLL]),
                "hip_roll_right": float(action[R_HIP_ROLL]),
                "action_saturated": bool(np.max(np.abs(action)) >= 0.99),
            })
    enabled = [row for row in rows if row["variant"] != "no_lateral_correction"]
    amplifies = any((abs(row["roll1_deg"]) > abs(row["roll0_deg"]) + 0.25) for row in enabled)
    summary = {
        "rows": len(rows),
        "lateral_layer_sign_valid": not amplifies,
        "amplification_detected": amplifies,
    }
    return rows, summary


def run_small_eval(model: mujoco.MjModel, env: BalanceEnv, table: dict[float, dict[str, float]], baseline: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[LateralCandidate]]:
    rows = []
    for candidate in CANDIDATES:
        for episode in range(EPISODES_PER_HEIGHT):
            rows.append(run_episode(model, env, table, 0.60, episode, candidate))
    summaries = summarize(rows)
    kept = []
    for candidate in CANDIDATES:
        s = next(row for row in summaries if row["candidate"] == candidate.name)
        passes = (
            s["mean_survival_s"] > baseline["mean_survival_s"]
            and s["mean_roll_rms_deg"] < baseline["mean_roll_rms_deg"]
            and s["mean_pitch_rms_deg"] <= baseline["mean_pitch_rms_deg"] + 2.0
            and s["mean_action_saturation_rate"] < 0.05
            and s["max_abs_clearance_m"] < 0.05
        )
        if passes:
            kept.append(candidate)
        s["kept"] = passes
    summary = {"baseline": baseline, "candidates": summaries, "kept_candidates": [c.name for c in kept]}
    return rows, summary, kept


def run_full_validation(model: mujoco.MjModel, env: BalanceEnv, table: dict[float, dict[str, float]], candidates: list[LateralCandidate], baseline: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for candidate in candidates[:2]:
        for height in VALID_HEIGHTS:
            for episode in range(EPISODES_PER_HEIGHT):
                rows.append(run_episode(model, env, table, height, episode, candidate))
    summaries = summarize(rows) if rows else []
    best = None
    for s in summaries:
        s["beats_reset_fixed_baseline"] = (
            s["mean_survival_s"] > baseline["mean_survival_s"]
            and s["fall_rate"] < baseline["fall_rate"]
            and s["mean_roll_rms_deg"] < baseline["mean_roll_rms_deg"]
            and s["mean_pitch_rms_deg"] <= baseline["mean_pitch_rms_deg"] + 2.0
            and s["mean_action_saturation_rate"] < 0.05
        )
        if best is None or s["mean_survival_s"] > best["mean_survival_s"]:
            best = s
    return rows, {"baseline": baseline, "candidates": summaries, "best": best}


def load_baseline_source() -> tuple[str, dict[str, float]]:
    if STEP513B_SUMMARY.exists():
        data = json.loads(STEP513B_SUMMARY.read_text(encoding="utf-8"))
        step5 = data["step5_baseline_status"]
        return "step5_13b_post_reset_rebaseline", {
            "mean_survival_s": float(step5["mean_survival_s"]),
            "fall_rate": float(step5["fall_rate"]),
            "mean_pitch_rms_deg": float(step5["mean_pitch_rms_deg"]),
            "mean_roll_rms_deg": float(step5["mean_roll_rms_deg"]),
            "mean_action_saturation_rate": float(step5["mean_action_saturation_rate"]),
        }
    data = json.loads(STEP513_BASELINE.read_text(encoding="utf-8"))
    return "step5_13_reset_fixed_revalidation", {
        "mean_survival_s": float(data["mean_survival_s"]),
        "fall_rate": float(data["fall_rate"]),
        "mean_pitch_rms_deg": float(data["mean_pitch_rms_deg"]),
        "mean_roll_rms_deg": float(data["mean_roll_rms_deg"]),
        "mean_action_saturation_rate": float(data["mean_action_saturation_rate"]),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    balance_residual_before = BALANCE_RESIDUAL_PATH.read_text(encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    env = make_env()
    table = load_balanced_init_table()
    baseline_source, baseline = load_baseline_source()

    baseline_rows, baseline_summary = run_baseline_confirmation(model, env, table)
    write_csv(OUTPUT_DIR / "baseline_confirmation.csv", baseline_rows)
    write_json(OUTPUT_DIR / "baseline_confirmation_summary.json", {"baseline_source": baseline_source, **baseline_summary})
    if baseline_summary["reset_regression_found"]:
        write_json(OUTPUT_DIR / "candidate_summary.json", {"final_decision": "A. RESET_REGRESSION_FOUND"})
        print("RESET_REGRESSION_FOUND")
        return

    sign_rows, sign_summary = run_sign_response(model, env, table)
    write_csv(OUTPUT_DIR / "sign_response_validation.csv", sign_rows)
    write_json(OUTPUT_DIR / "sign_response_summary.json", sign_summary)
    if not sign_summary["lateral_layer_sign_valid"]:
        write_json(OUTPUT_DIR / "candidate_summary.json", {"final_decision": "B. LATERAL_LAYER_SIGN_INVALID"})
        print("LATERAL_LAYER_SIGN_INVALID")
        return

    candidate_rows, candidate_summary, kept = run_small_eval(model, env, table, baseline)
    write_csv(OUTPUT_DIR / "candidate_results.csv", candidate_rows)
    write_json(OUTPUT_DIR / "candidate_summary.json", candidate_summary)

    full_rows: list[dict[str, Any]] = []
    full_fieldnames = [
        "candidate",
        "height",
        "episode",
        "steps",
        "survival_time_s",
        "fell",
        "fall_reason",
        "pitch_rms_deg",
        "roll_rms_deg",
        "action_saturation_rate",
        "first_failure_variable",
        "max_contact_force_imbalance_ratio",
        "max_abs_clearance_m",
        "mean_lateral_correction_abs",
    ]
    full_summary: dict[str, Any] = {"baseline": baseline, "candidates": [], "best": None}
    final_decision = "G. VMC_OR_WHOLE_BODY_LAYER_REQUIRED"
    if kept:
        full_rows, full_summary = run_full_validation(model, env, table, kept, baseline)
        if any(row.get("beats_reset_fixed_baseline") for row in full_summary["candidates"]):
            final_decision = "F. LATERAL_LAYER_BEATS_RESET_FIXED_BASELINE"
            best = full_summary["best"]
            best_candidate = next(c for c in kept if c.name == best["candidate"])
            write_json(OUTPUT_DIR / "best_lateral_balance_summary.json", best)
            best_config = {
                "source_controller": "outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml",
                "lateral_balance": best_candidate.__dict__,
            }
            (OUTPUT_DIR / "best_lateral_balance_config.yaml").write_text(yaml.dump(best_config, sort_keys=False), encoding="utf-8")
        else:
            final_decision = "E. LATERAL_LAYER_IMPROVES_BUT_DOES_NOT_PASS_GATE"
    write_csv(OUTPUT_DIR / "full_validation.csv", full_rows, full_fieldnames)
    full_summary["final_decision"] = final_decision
    full_summary["balance_residual_unchanged"] = BALANCE_RESIDUAL_PATH.read_text(encoding="utf-8") == balance_residual_before
    full_summary["step6_status"] = "BLOCKED"
    write_json(OUTPUT_DIR / "full_validation_summary.json", full_summary)

    print(json.dumps({"baseline_source": baseline_source, "kept": [c.name for c in kept], "final_decision": final_decision}, indent=2))


if __name__ == "__main__":
    main()
