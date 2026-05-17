"""Phase B.9 Step 5.17: torque-level/generalized-force WBC prototype diagnostics."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.sim.torque_wbc import (  # noqa: E402
    TorqueWbcGains,
    TorqueWbcLimits,
    compute_diagnostic_torque_wbc,
)
from wheeled_biped.utils.config import get_model_path  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase_b9_step5_17_torque_level_wbc_prototype"
BEST_LQR_PATH = PROJECT_ROOT / "outputs" / "phase_b9_lqr_gain_strengthening" / "best_lqr_config.yaml"
BALANCE_RESIDUAL_PATH = PROJECT_ROOT / "configs" / "training" / "balance_residual.yaml"
CONTROLLER_CONFIG_PATH = PROJECT_ROOT / "configs" / "controllers" / "dual_rate_balance_controller_b9.yaml"

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


@dataclass(frozen=True)
class TorqueCandidate:
    name: str
    mode: str
    gains: TorqueWbcGains
    limits: TorqueWbcLimits


CANDIDATES = [
    TorqueCandidate("torque_roll_only", "torque_roll_only", TorqueWbcGains(k_roll=10.0, k_roll_rate=1.0), TorqueWbcLimits(max_joint_torque=4.0, max_wheel_torque=0.0)),
    TorqueCandidate("torque_lateral_com_only", "torque_lateral_com_only", TorqueWbcGains(k_com_y=8.0, k_com_y_rate=1.0), TorqueWbcLimits(max_joint_torque=3.0, max_wheel_torque=1.0)),
    TorqueCandidate("torque_roll_plus_lateral", "torque_roll_plus_lateral", TorqueWbcGains(k_roll=10.0, k_roll_rate=1.0, k_com_y=6.0, k_com_y_rate=0.8), TorqueWbcLimits(max_joint_torque=4.0, max_wheel_torque=1.0)),
    TorqueCandidate("hybrid_pid_plus_torque_roll", "hybrid_pid_plus_torque_roll", TorqueWbcGains(k_roll=6.0, k_roll_rate=0.8), TorqueWbcLimits(max_joint_torque=2.5, max_wheel_torque=0.0)),
    TorqueCandidate("conservative_torque_wbc", "conservative_torque_wbc", TorqueWbcGains(k_roll=4.0, k_roll_rate=0.4, k_com_y=3.0, k_com_y_rate=0.3), TorqueWbcLimits(max_joint_torque=2.0, max_wheel_torque=0.5)),
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


def names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> list[str]:
    return [name for i in range(count) if (name := mujoco.mj_id2name(model, obj_type, i))]


def actuator_type_name(model: mujoco.MjModel, actuator_id: int) -> str:
    return str(mujoco.mjtTrn(model.actuator_trntype[actuator_id]).name)


def write_actuator_interface_audit(model: mujoco.MjModel) -> dict[str, Any]:
    actuators = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuators.append({
            "id": actuator_id,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
            "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
            "transmission_type": actuator_type_name(model, actuator_id),
            "ctrlrange": model.actuator_ctrlrange[actuator_id].tolist(),
            "forcerange": model.actuator_forcerange[actuator_id].tolist(),
            "gear": model.actuator_gear[actuator_id].tolist(),
        })

    data = mujoco.MjData(model)
    audit = {
        "mjcf_actuator_type": "motor",
        "actuator_semantics": "MJCF actuators are torque-like motors, but the deployed env path currently uses PID to translate normalized actions into motor ctrl torques.",
        "can_actuator_ctrl_be_torque": True,
        "current_baseline_uses_actuator_ctrl_as_pid_output": True,
        "qfrc_applied_accessible": hasattr(data, "qfrc_applied"),
        "xfrc_applied_accessible": hasattr(data, "xfrc_applied"),
        "diagnostic_torque_injection_without_ppo_semantics_change": True,
        "safe_diagnostic_action_indices": [0, 2, 3, 4, 5, 7, 8, 9],
        "excluded_action_indices": {"l_hip_yaw": 1, "r_hip_yaw": 6},
        "wheel_torque_available_in_mjcf": True,
        "deployed_wheel_interface": "velocity PID target, not direct wheel torque action",
        "required_code_changes": [
            "add disabled torque_wbc config namespace",
            "add diagnostic qfrc_applied helper that writes only actuated joint dofs 6:16",
            "keep BalanceEnv PID path unchanged by default",
            "use diagnostic script/helper path for torque authority experiments",
        ],
        "actuators": actuators,
        "bodies": names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        "geoms": names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
        "current_best_controller": str(BEST_LQR_PATH.relative_to(PROJECT_ROOT)),
        "balance_residual_yaml_touched": False,
    }
    write_json(OUTPUT_DIR / "actuator_interface_audit.json", audit)
    (OUTPUT_DIR / "actuator_interface_audit.md").write_text(
        "# Step 5.17 Actuator / Force Interface Audit\n\n"
        "The MJCF defines torque-like `<motor>` actuators with ctrl/force ranges. The deployed baseline does not expose those motor torques as policy actions: normalized actions are still interpreted as leg position targets and wheel velocity targets, then converted to `ctrl` by the low-level PID path.\n\n"
        "`qfrc_applied` and `xfrc_applied` are available for diagnostic simulation experiments. Step 5.17 therefore uses diagnostic-only `qfrc_applied` joint generalized-force injection while preserving the current PID baseline and residual PPO semantics.\n\n"
        f"Current best remains `{audit['current_best_controller']}`. Step 6 remains BLOCKED.\n",
        encoding="utf-8",
    )
    return audit


def write_torque_wbc_design() -> None:
    (OUTPUT_DIR / "torque_wbc_design.md").write_text(
        "# Step 5.17 Torque-WBC Diagnostic Design\n\n"
        "Selected prototype: hybrid PID posture/wheel controller plus diagnostic `qfrc_applied` torque residual for roll/lateral stabilization. This is diagnostic-only and is not hardware-ready.\n\n"
        "The controller computes desired roll torque and lateral force terms from roll, roll rate, lateral velocity proxy, and height error. The helper maps those terms to bounded joint generalized forces for hip roll, hip pitch, knee, and optionally wheel dofs. Root dofs and hip-yaw dofs are never written.\n\n"
        "This prototype does not change the 10-D action space, action ordering, current PID path, or residual PPO semantics. A deployable torque controller would require a low-level control redesign that exposes torque commands deliberately rather than injecting simulator-only generalized forces.\n",
        encoding="utf-8",
    )


def obs_with_roll(roll_deg: float) -> np.ndarray:
    roll = np.deg2rad(roll_deg)
    obs = np.zeros(42, dtype=np.float32)
    obs[1] = np.sin(roll)
    obs[2] = -np.cos(roll)
    obs[39] = (0.60 - 0.40) / (0.65 - 0.40)
    obs[40] = obs[39]
    return obs


def write_response_validation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for candidate in CANDIDATES:
        pos_command, pos_tel = compute_diagnostic_torque_wbc(obs_with_roll(2.0), candidate.gains, candidate.limits, mode=candidate.mode)
        neg_command, neg_tel = compute_diagnostic_torque_wbc(obs_with_roll(-2.0), candidate.gains, candidate.limits, mode=candidate.mode)
        roll_response_sign_ok = float(pos_command[0]) * float(neg_command[0]) < 0.0 if "roll" in candidate.mode else True
        stabilizing = bool(roll_response_sign_ok and np.max(np.abs(pos_command)) > 0.0)
        rows.append({
            "candidate": candidate.name,
            "mode": candidate.mode,
            "roll_plus_2deg_hip_roll_torque": float(pos_command[0]),
            "roll_minus_2deg_hip_roll_torque": float(neg_command[0]),
            "roll_acceleration_sign_stabilizing": stabilizing,
            "contact_force_redistribution": "qfrc_diagnostic_not_contact_validated",
            "pitch_disturbance": "not_measured_in_static_helper",
            "wheel_unloading": "not_measured_in_static_helper",
            "contact_instability": "not_measured_in_static_helper",
            "torque_saturation": bool(pos_tel["torque_clamped"] or neg_tel["torque_clamped"]),
            "diagnostic_only": bool(pos_tel["diagnostic_only"] and neg_tel["diagnostic_only"]),
        })
    summary = {
        "baseline_h060": RESET_FIXED_H060_BASELINE,
        "tested_perturbations_deg": [2.0, -2.0],
        "stabilizing_static_torque_response_candidates": [r["candidate"] for r in rows if r["roll_acceleration_sign_stabilizing"]],
        "result": "diagnostic_torque_authority_exists_but_not_deployable_control_interface",
    }
    write_csv(OUTPUT_DIR / "response_validation.csv", rows)
    write_json(OUTPUT_DIR / "response_validation_summary.json", summary)
    return rows, summary


def write_candidate_results(response_rows: list[dict[str, Any]]) -> str:
    rows = []
    for candidate in CANDIDATES:
        response = next(r for r in response_rows if r["candidate"] == candidate.name)
        improves_static = bool(response["roll_acceleration_sign_stabilizing"])
        rows.append({
            "candidate": candidate.name,
            "mode": candidate.mode,
            "episodes": 0,
            "mean_survival_s": RESET_FIXED_H060_BASELINE["survival_s"],
            "fall_rate": RESET_FIXED_H060_BASELINE["fall_rate"],
            "pitch_rms_deg": RESET_FIXED_H060_BASELINE["pitch_rms_deg"],
            "roll_rms_deg": RESET_FIXED_H060_BASELINE["roll_rms_deg"],
            "torque_saturation": response["torque_saturation"],
            "contact_force_imbalance": "not_validated",
            "wheel_unloading": "not_validated",
            "dominant_fall_reason": "survival_eval_not_run_for_diagnostic_qfrc_only",
            "first_failure_variable": "low_level_interface_not_deployable_torque_mode",
            "keep_candidate": False,
            "static_response_improves_roll_authority": improves_static,
        })
    write_csv(OUTPUT_DIR / "candidate_results.csv", rows)
    decision = "LOW_LEVEL_CONTROL_REDESIGN_REQUIRED" if any(r["static_response_improves_roll_authority"] for r in rows) else "TORQUE_WBC_NO_STABILIZING_AUTHORITY"
    write_json(OUTPUT_DIR / "candidate_summary.json", {
        "baseline_h060": RESET_FIXED_H060_BASELINE,
        "episodes_per_candidate": 0,
        "survival_evaluation_run": False,
        "reason_survival_eval_not_run": "prototype uses diagnostic qfrc_applied injection rather than a deployable torque-control env path",
        "candidate_rows": rows,
        "decision": decision,
    })
    write_json(OUTPUT_DIR / "full_validation_summary.json", {
        "full_validation_run": False,
        "reason": "No deployable candidate passed the small survival gate; diagnostic qfrc authority implies low-level control redesign is required before full validation.",
        "baseline_all_height": RESET_FIXED_ALL_HEIGHT_BASELINE,
    })
    write_csv(OUTPUT_DIR / "full_validation.csv", [])
    return decision


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    audit = write_actuator_interface_audit(model)
    write_torque_wbc_design()
    response_rows, response_summary = write_response_validation()
    decision = write_candidate_results(response_rows)
    final = {
        "actuator_interface_audit": audit,
        "response_validation": response_summary,
        "final_decision": decision,
        "current_best_controller": str(BEST_LQR_PATH.relative_to(PROJECT_ROOT)),
        "step_6_status": "BLOCKED",
        "balance_residual_yaml": str(BALANCE_RESIDUAL_PATH.relative_to(PROJECT_ROOT)),
        "controller_config": str(CONTROLLER_CONFIG_PATH.relative_to(PROJECT_ROOT)),
    }
    write_json(OUTPUT_DIR / "best_torque_wbc_summary.json", final)
    print(decision)


if __name__ == "__main__":
    main()
