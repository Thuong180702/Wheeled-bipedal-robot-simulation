"""Phase B.9 Step 5.13: reset equilibrium repair and baseline revalidation."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase_b9_posture_geometry_inspection import (  # noqa: E402
    body_com,
    contact_forces_by_wheel,
    wheel_bottom_heights,
)
from scripts.phase_b9_step5_lqr_gain_strengthening import (  # noqa: E402
    apply_balanced_root_init,
    create_tuned_controller,
    load_balanced_init_table,
)
from wheeled_biped.controllers.dual_rate_balance_controller import DualRateConfig  # noqa: E402
from wheeled_biped.envs.balance_env import BalanceEnv  # noqa: E402
from wheeled_biped.utils.config import get_model_path  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase_b9_step5_13_reset_equilibrium_fix"
TABLE_PATH = PROJECT_ROOT / "configs" / "controllers" / "b9_balanced_root_init_table.yaml"
BEST_LQR_PATH = PROJECT_ROOT / "outputs" / "phase_b9_lqr_gain_strengthening" / "best_lqr_config.yaml"
CONTROLLER_CONFIG_PATH = PROJECT_ROOT / "configs" / "controllers" / "dual_rate_balance_controller_b9.yaml"
BALANCE_RESIDUAL_PATH = PROJECT_ROOT / "configs" / "training" / "balance_residual.yaml"

SETTLING_STEPS = 50
REVALIDATION_STEPS = 1000
REVALIDATION_EPISODES_PER_HEIGHT = 5


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([roll, pitch, yaw]), b"xyz")
    return quat


def quat_to_rpy(quat: np.ndarray) -> tuple[float, float, float]:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    r = mat.reshape(3, 3)
    roll = math.atan2(r[2, 1], r[2, 2])
    pitch = math.atan2(-r[2, 0], math.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
    yaw = math.atan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


def finite_force(value: float) -> float:
    return 0.0 if math.isnan(value) else float(value)


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def expected_weight(model: mujoco.MjModel) -> float:
    return float(np.sum(model.body_mass) * abs(model.opt.gravity[2]))


def reset_model_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    return data


def joint_targets(init: dict[str, float]) -> np.ndarray:
    return np.array([
        0.0, 0.0, init["hip_pitch"], init["knee"], 0.0,
        0.0, 0.0, init["hip_pitch"], init["knee"], 0.0,
    ], dtype=float)


def apply_cpu_init(model: mujoco.MjModel, data: mujoco.MjData, init: dict[str, float], *, full_root: bool) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, 0.71]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:17] = joint_targets(init)
    if full_root:
        data.qpos[0] = init["root_x"]
        data.qpos[1] = init.get("root_y", data.qpos[1])
        data.qpos[2] = init["root_z"]
        data.qpos[3:7] = rpy_to_quat(init["root_roll"], init["root_pitch"], 0.0)
    mujoco.mj_forward(model, data)


def evaluate_cpu_state(model: mujoco.MjModel, data: mujoco.MjData, *, height: float, mode: str, step: int) -> dict[str, Any]:
    left_clearance, right_clearance = wheel_bottom_heights(model, data)
    left_force_raw, right_force_raw = contact_forces_by_wheel(model, data)
    left_force = finite_force(left_force_raw)
    right_force = finite_force(right_force_raw)
    total_force = left_force + right_force
    force_imbalance = abs(left_force - right_force)
    roll, pitch, yaw = quat_to_rpy(data.qpos[3:7].copy())
    com = body_com(model, data)
    wheel_contact_x = 0.5 * (
        data.geom_xpos[model.geom("l_wheel_collision").id, 0]
        + data.geom_xpos[model.geom("r_wheel_collision").id, 0]
    )
    return {
        "height": height,
        "mode": mode,
        "step": step,
        "time_s": step * float(model.opt.timestep),
        "root_x": float(data.qpos[0]),
        "root_y": float(data.qpos[1]),
        "root_z": float(data.qpos[2]),
        "root_roll_rad": roll,
        "root_pitch_rad": pitch,
        "root_yaw_rad": yaw,
        "root_roll_deg": math.degrees(roll),
        "root_pitch_deg": math.degrees(pitch),
        "left_clearance_m": float(left_clearance),
        "right_clearance_m": float(right_clearance),
        "max_abs_clearance_m": max(abs(float(left_clearance)), abs(float(right_clearance))),
        "left_force_n": left_force,
        "right_force_n": right_force,
        "total_force_n": total_force,
        "force_imbalance_n": force_imbalance,
        "force_imbalance_ratio": force_imbalance / total_force if total_force > 1e-9 else math.inf,
        "left_unloaded": left_force < 0.1,
        "right_unloaded": right_force < 0.1,
        "com_x": float(com[0]),
        "com_y": float(com[1]),
        "com_z": float(com[2]),
        "com_to_wheel_x_m": float(com[0] - wheel_contact_x),
        "roll_rate_rad_s": float(data.qvel[3]) if data.qvel.shape[0] > 3 else 0.0,
        "pitch_rate_rad_s": float(data.qvel[4]) if data.qvel.shape[0] > 4 else 0.0,
    }


def settle_rows(model: mujoco.MjModel, init: dict[str, float], height: float, mode: str, steps: int) -> list[dict[str, Any]]:
    data = reset_model_data(model)
    apply_cpu_init(model, data, init, full_root=True)
    rows = []
    for step in range(steps + 1):
        rows.append(evaluate_cpu_state(model, data, height=height, mode=mode, step=step))
        if step < steps:
            if mode == "passive":
                data.qvel[:] = 0.0
                data.ctrl[:] = 0.0
            elif mode == "pid_hold":
                data.ctrl[:] = joint_targets(init)
            mujoco.mj_step(model, data)
    return rows


def compute_root_z_for_clearance(model: mujoco.MjModel, init: dict[str, float], root_x: float, root_roll: float, root_pitch: float) -> float:
    data = reset_model_data(model)
    data.qpos[0] = root_x
    data.qpos[2] = 1.0
    data.qpos[3:7] = rpy_to_quat(root_roll, root_pitch, 0.0)
    data.qpos[7:17] = joint_targets(init)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    left_bottom, right_bottom = wheel_bottom_heights(model, data)
    return float(1.0 - min(left_bottom, right_bottom))


def score_candidate(model: mujoco.MjModel, init: dict[str, float], height: float, root_x: float, root_roll: float, root_pitch: float) -> dict[str, Any]:
    candidate = {
        "root_x": root_x,
        "root_z": compute_root_z_for_clearance(model, init, root_x, root_roll, root_pitch),
        "root_roll": root_roll,
        "root_pitch": root_pitch,
        "hip_pitch": init["hip_pitch"],
        "knee": init["knee"],
    }

    data = reset_model_data(model)
    apply_cpu_init(model, data, candidate, full_root=True)
    t0 = evaluate_cpu_state(model, data, height=height, mode="candidate_t0", step=0)

    for _ in range(SETTLING_STEPS):
        data.ctrl[:] = joint_targets(candidate)
        mujoco.mj_step(model, data)
    settled = evaluate_cpu_state(model, data, height=height, mode="candidate_pid_hold_100ms", step=SETTLING_STEPS)

    weight = expected_weight(model)
    force_ratio = settled["total_force_n"] / weight if weight > 0 else 0.0
    clearance_cost = 5000.0 * (t0["left_clearance_m"] ** 2 + t0["right_clearance_m"] ** 2)
    force_cost = 10.0 * abs(force_ratio - 1.0)
    unload_cost = 100.0 if settled["left_unloaded"] or settled["right_unloaded"] else 0.0
    tilt_cost = abs(settled["root_roll_rad"]) + abs(settled["root_pitch_rad"])
    rate_cost = 0.1 * (abs(settled["roll_rate_rad_s"]) + abs(settled["pitch_rate_rad_s"]))
    penetration_cost = 1000.0 * max(0.0, abs(min(t0["left_clearance_m"], t0["right_clearance_m"])) - 0.001)

    return {
        **candidate,
        "height": height,
        "score": clearance_cost + force_cost + unload_cost + tilt_cost + rate_cost + penetration_cost,
        "t0_left_clearance_m": t0["left_clearance_m"],
        "t0_right_clearance_m": t0["right_clearance_m"],
        "settled_left_force_n": settled["left_force_n"],
        "settled_right_force_n": settled["right_force_n"],
        "settled_total_force_n": settled["total_force_n"],
        "settled_force_ratio": force_ratio,
        "settled_imbalance_ratio": settled["force_imbalance_ratio"],
        "settled_roll_deg": settled["root_roll_deg"],
        "settled_pitch_deg": settled["root_pitch_deg"],
        "settled_roll_rate_rad_s": settled["roll_rate_rad_s"],
        "settled_pitch_rate_rad_s": settled["pitch_rate_rad_s"],
        "settled_left_unloaded": settled["left_unloaded"],
        "settled_right_unloaded": settled["right_unloaded"],
    }


def generate_candidate_for_height(model: mujoco.MjModel, height: float, init: dict[str, float]) -> dict[str, Any]:
    root_x_centers = [0.0, float(init.get("root_x", 0.0))]
    root_roll_centers = [0.0, float(init.get("root_roll", 0.0))]
    root_pitch_centers = [0.0, float(init.get("root_pitch", 0.0))]
    root_x_values = sorted({round(center + dx, 6) for center in root_x_centers for dx in np.linspace(-0.03, 0.03, 7)})
    root_roll_values = sorted({round(center + dr, 6) for center in root_roll_centers for dr in np.linspace(-0.03, 0.03, 7)})
    root_pitch_values = sorted({round(center + dp, 6) for center in root_pitch_centers for dp in np.linspace(-0.02, 0.02, 5)})

    best: dict[str, Any] | None = None
    for root_x in root_x_values:
        for root_roll in root_roll_values:
            for root_pitch in root_pitch_values:
                candidate = score_candidate(model, init, height, root_x, root_roll, root_pitch)
                if best is None or candidate["score"] < best["score"]:
                    best = candidate
    assert best is not None
    return best


def candidate_passes_gates(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if max(abs(candidate["t0_left_clearance_m"]), abs(candidate["t0_right_clearance_m"])) > 0.001:
        reasons.append("t0_clearance_exceeds_1mm")
    if not (0.25 <= candidate["settled_force_ratio"] <= 2.5):
        reasons.append("settled_total_force_implausible")
    if candidate["settled_left_unloaded"] or candidate["settled_right_unloaded"]:
        reasons.append("wheel_unloaded_in_pid_hold")
    if candidate["settled_imbalance_ratio"] > 0.95:
        reasons.append("force_imbalance_too_high")
    if abs(candidate["settled_roll_deg"]) > 15.0 or abs(candidate["settled_pitch_deg"]) > 15.0:
        reasons.append("settled_tilt_too_high")
    if abs(candidate["settled_roll_rate_rad_s"]) > 10.0 or abs(candidate["settled_pitch_rate_rad_s"]) > 10.0:
        reasons.append("settled_rate_too_high")
    return not reasons, reasons


def build_new_table(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    table = {
        "balanced_root_initialization": {
            "description": "Step 5.13 candidate repaired root poses validated before config replacement",
            "heights": {},
        }
    }
    for candidate in candidates:
        table["balanced_root_initialization"]["heights"][f"{candidate['height']:.2f}"] = {
            "root_x": float(candidate["root_x"]),
            "root_z": float(candidate["root_z"]),
            "root_roll": float(candidate["root_roll"]),
            "root_pitch": float(candidate["root_pitch"]),
            "hip_pitch": float(candidate["hip_pitch"]),
            "knee": float(candidate["knee"]),
            "expected_left_clearance": float(candidate["t0_left_clearance_m"]),
            "expected_right_clearance": float(candidate["t0_right_clearance_m"]),
            "expected_left_force": float(candidate["settled_left_force_n"]),
            "expected_right_force": float(candidate["settled_right_force_n"]),
        }
    return table


def write_table(path: Path, table: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(table, f, sort_keys=False)


def verify_reset_bug(model: mujoco.MjModel, old_table: dict[float, dict[str, float]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for height in sorted(old_table):
        init = old_table[height]
        for full_root in [True, False]:
            data = reset_model_data(model)
            apply_cpu_init(model, data, init, full_root=full_root)
            mode = "old_full_root_after_forward" if full_root else "old_step5_joint_only_after_forward"
            rows.append(evaluate_cpu_state(model, data, height=height, mode=mode, step=0))
            for _ in range(SETTLING_STEPS):
                mujoco.mj_step(model, data)
            rows.append(evaluate_cpu_state(model, data, height=height, mode=mode.replace("after_forward", "after_100ms"), step=SETTLING_STEPS))

    severe_penetration = any(
        row["mode"] == "old_full_root_after_forward" and min(row["left_clearance_m"], row["right_clearance_m"]) < -0.01
        for row in rows
    )
    joint_only_no_contact = any(
        row["mode"] == "old_step5_joint_only_after_forward" and row["total_force_n"] < 0.1
        for row in rows
    )
    summary = {
        "reset_bug_confirmed": severe_penetration and joint_only_no_contact,
        "severe_full_root_penetration_confirmed": severe_penetration,
        "step5_joint_only_no_contact_confirmed": joint_only_no_contact,
        "rows": rows,
    }
    return rows, summary


def write_reset_bug_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase B.9 Step 5.13 reset bug verification",
        "",
        f"Reset bug confirmed: **{summary['reset_bug_confirmed']}**",
        f"Severe full-root penetration confirmed: **{summary['severe_full_root_penetration_confirmed']}**",
        f"Step5 joint-only no-contact confirmed: **{summary['step5_joint_only_no_contact_confirmed']}**",
        "",
        "The Step 5 evaluator bug is that full root pose values in the balanced-root table were not applied before this repair; only hip_pitch/knee were written. The table itself is also revalidated before replacement.",
    ]
    (OUTPUT_DIR / "reset_bug_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_table(model: mujoco.MjModel, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    validation_rows = []
    full_trace_rows = []
    all_pass = True
    for candidate in candidates:
        passed, reasons = candidate_passes_gates(candidate)
        all_pass = all_pass and passed
        validation_rows.append({
            **candidate,
            "passed": passed,
            "failure_reasons": ";".join(reasons),
        })
        data = reset_model_data(model)
        apply_cpu_init(model, data, candidate, full_root=True)
        full_trace_rows.append(evaluate_cpu_state(model, data, height=candidate["height"], mode="new_full_root_after_forward", step=0))
    return validation_rows, full_trace_rows, all_pass


def run_settling_logs(model: mujoco.MjModel, table: dict[float, dict[str, float]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    passive_rows = []
    pid_rows = []
    for height, init in sorted(table.items()):
        passive_rows.extend(settle_rows(model, init, height, "passive", SETTLING_STEPS))
        pid_rows.extend(settle_rows(model, init, height, "pid_hold", SETTLING_STEPS))
    return passive_rows, pid_rows


def load_best_lqr_params() -> dict[str, float]:
    with open(BEST_LQR_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env() -> BalanceEnv:
    return BalanceEnv({
        "low_level_pid": {"enabled": True, "disable_pid_action_bias": True},
        "domain_randomization": {"enabled": False},
        "sensor_noise": {"enabled": False},
    })


def revalidate_step5(model: mujoco.MjModel, init_table: dict[float, dict[str, float]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = DualRateConfig.from_yaml(CONTROLLER_CONFIG_PATH)
    controller = create_tuned_controller(config, load_best_lqr_params(), model)
    env = make_env()
    rows = []

    for height in sorted(init_table):
        nearest_height = min(init_table.keys(), key=lambda h: abs(h - height))
        for episode in range(REVALIDATION_EPISODES_PER_HEIGHT):
            rng = jax.random.PRNGKey(13_000 + episode + int(height * 1000))
            state = env.reset(rng)
            state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, init_table))
            controller.reset()
            controller.target_hip_pitch = init_table[nearest_height]["hip_pitch"]
            controller.target_knee = init_table[nearest_height]["knee"]
            controller.last_stable_hip_pitch = controller.target_hip_pitch
            controller.last_stable_knee = controller.target_knee

            pitch_sq = 0.0
            roll_sq = 0.0
            action_sat_count = 0
            steps = 0
            first_failure_variable = "none"
            contact_left_min = math.inf
            contact_right_min = math.inf
            clearance_abs_max = 0.0

            for _ in range(REVALIDATION_STEPS):
                obs_np = np.array(state.obs)
                action = controller.compute_action(obs_np)
                g_body = obs_np[0:3]
                pitch = float(np.arcsin(np.clip(-g_body[0], -1.0, 1.0)))
                roll = float(np.arcsin(np.clip(g_body[1], -1.0, 1.0)))
                pitch_sq += pitch ** 2
                roll_sq += roll ** 2
                action_sat_count += int(np.max(np.abs(action)) >= 0.99)

                state = env.step(state, jnp.array(action))
                steps += 1

                mj_data = mujoco.MjData(model)
                mj_data.qpos[:] = np.array(state.mjx_data.qpos)
                mj_data.qvel[:] = np.array(state.mjx_data.qvel)
                mujoco.mj_forward(model, mj_data)
                left_clearance, right_clearance = wheel_bottom_heights(model, mj_data)
                left_force, right_force = contact_forces_by_wheel(model, mj_data)
                contact_left_min = min(contact_left_min, finite_force(left_force))
                contact_right_min = min(contact_right_min, finite_force(right_force))
                clearance_abs_max = max(clearance_abs_max, abs(left_clearance), abs(right_clearance))

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
            rows.append({
                "height": height,
                "episode": episode,
                "steps": steps,
                "survival_time_s": steps * env.CONTROL_DT,
                "fell": fell,
                "fall_reason": first_failure_variable if fell else "none",
                "pitch_rms_deg": math.degrees(math.sqrt(pitch_sq / steps)) if steps else 0.0,
                "roll_rms_deg": math.degrees(math.sqrt(roll_sq / steps)) if steps else 0.0,
                "action_saturation_rate": action_sat_count / steps if steps else 0.0,
                "first_failure_variable": first_failure_variable,
                "min_left_contact_force_n": contact_left_min,
                "min_right_contact_force_n": contact_right_min,
                "max_abs_clearance_m": clearance_abs_max,
            })

    survival = [row["survival_time_s"] for row in rows]
    falls = [row["fell"] for row in rows]
    summary = {
        "episodes": len(rows),
        "mean_survival_s": float(np.mean(survival)) if survival else 0.0,
        "fall_rate": float(np.mean(falls)) if falls else 1.0,
        "mean_pitch_rms_deg": float(np.mean([row["pitch_rms_deg"] for row in rows])) if rows else 0.0,
        "mean_roll_rms_deg": float(np.mean([row["roll_rms_deg"] for row in rows])) if rows else 0.0,
        "mean_action_saturation_rate": float(np.mean([row["action_saturation_rate"] for row in rows])) if rows else 0.0,
        "step5_passed": bool(rows) and float(np.mean(falls)) == 0.0 and float(np.mean(survival)) >= 19.9,
    }
    return rows, summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    old_table = load_balanced_init_table()
    old_balance_residual = BALANCE_RESIDUAL_PATH.read_text(encoding="utf-8")

    bug_rows, bug_summary = verify_reset_bug(model, old_table)
    write_csv(OUTPUT_DIR / "reset_bug_verification.csv", bug_rows)
    write_json(OUTPUT_DIR / "reset_bug_verification.json", {k: v for k, v in bug_summary.items() if k != "rows"})
    write_reset_bug_markdown(bug_summary)

    candidates = [generate_candidate_for_height(model, height, init) for height, init in sorted(old_table.items())]
    new_table_yaml = build_new_table(candidates)
    write_table(OUTPUT_DIR / "new_balanced_root_table.yaml", new_table_yaml)

    validation_rows, full_trace_rows, gates_pass = validate_table(model, candidates)
    write_csv(OUTPUT_DIR / "reset_equilibrium_validation.csv", validation_rows)
    write_csv(OUTPUT_DIR / "full_root_application_trace.csv", full_trace_rows)

    candidate_table = {candidate["height"]: candidate for candidate in candidates}
    passive_rows, pid_rows = run_settling_logs(model, candidate_table)
    write_csv(OUTPUT_DIR / "passive_settling.csv", passive_rows)
    write_csv(OUTPUT_DIR / "pid_hold_settling.csv", pid_rows)

    config_replaced = False
    if gates_pass:
        shutil.copy2(TABLE_PATH, OUTPUT_DIR / "b9_balanced_root_init_table.before_step5_13.yaml")
        write_table(TABLE_PATH, new_table_yaml)
        config_replaced = True

    smoke_rows = [
        {
            "height": row["height"],
            "passed": row["passed"],
            "failure_reasons": row["failure_reasons"],
            "root_z": row["root_z"],
            "settled_force_ratio": row["settled_force_ratio"],
            "settled_roll_deg": row["settled_roll_deg"],
        }
        for row in validation_rows
    ]
    write_csv(OUTPUT_DIR / "step5_after_reset_fix_smoke.csv", smoke_rows)

    revalidation_summary: dict[str, Any] | None = None
    if gates_pass:
        repaired_table = load_balanced_init_table()
        revalidation_rows, revalidation_summary = revalidate_step5(model, repaired_table)
        write_csv(OUTPUT_DIR / "step5_revalidation_after_reset_fix.csv", revalidation_rows)
        write_json(OUTPUT_DIR / "step5_revalidation_after_reset_fix_summary.json", revalidation_summary)

    final_decision = "D. RESET_EQUILIBRIUM_FIXED_BASELINE_REVALIDATED"
    if not bug_summary["reset_bug_confirmed"]:
        final_decision = "A. RESET_BUG_NOT_CONFIRMED"
    elif not gates_pass:
        final_decision = "C. FULL_ROOT_INIT_BUG_FIXED_TABLE_STILL_INVALID"
    elif revalidation_summary and revalidation_summary.get("step5_passed"):
        final_decision = "E. RESET_FIXED_AND_STEP5_NOW_PASSES"

    summary = {
        "reset_bug_confirmed": bug_summary["reset_bug_confirmed"],
        "table_validation_passed": gates_pass,
        "config_replaced": config_replaced,
        "revalidation_ran": revalidation_summary is not None,
        "revalidation_summary": revalidation_summary,
        "final_decision": final_decision,
        "step6_status": "BLOCKED" if final_decision != "E. RESET_FIXED_AND_STEP5_NOW_PASSES" else "READY_FOR_REVIEW",
        "balance_residual_unchanged": BALANCE_RESIDUAL_PATH.read_text(encoding="utf-8") == old_balance_residual,
    }
    write_json(OUTPUT_DIR / "reset_equilibrium_summary.json", summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
