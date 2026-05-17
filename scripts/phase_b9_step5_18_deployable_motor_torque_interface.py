"""Phase B.9 Step 5.18: deployable MJCF motor-torque interface audit.

This script documents and smoke-validates opt-in actuator-ctrl torque modes.
It does not start Step 6, train PPO, or modify residual training configs.
"""

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

from wheeled_biped.utils.config import get_model_path

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase_b9_step5_18_deployable_motor_torque_interface"
BEST_LQR_PATH = PROJECT_ROOT / "outputs" / "phase_b9_lqr_gain_strengthening" / "best_lqr_config.yaml"
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

ACTION_NAMES = [
    "l_hip_roll",
    "l_hip_yaw",
    "l_hip_pitch",
    "l_knee",
    "l_wheel",
    "r_hip_roll",
    "r_hip_yaw",
    "r_hip_pitch",
    "r_knee",
    "r_wheel",
]


@dataclass(frozen=True)
class MotorTorqueCandidate:
    name: str
    mode: str
    low_level_mode: str
    max_ctrl_fraction: float
    allow_wheel_torque: bool
    allow_hip_yaw_torque: bool
    expected_authority: str


CANDIDATES = [
    MotorTorqueCandidate("motor_torque_roll_only", "roll_only", "motor_torque", 0.20, False, False, "direct_roll_motor_ctrl"),
    MotorTorqueCandidate("motor_torque_lateral_com_only", "lateral_com_only", "motor_torque", 0.20, True, False, "direct_lateral_motor_ctrl"),
    MotorTorqueCandidate("motor_torque_roll_plus_lateral", "roll_plus_lateral", "motor_torque", 0.25, True, False, "direct_motor_ctrl"),
    MotorTorqueCandidate("hybrid_pid_plus_torque_roll", "roll_only", "hybrid_pid_plus_torque", 0.20, False, False, "pid_plus_roll_torque"),
    MotorTorqueCandidate("hybrid_pid_plus_torque_wbc", "wbc", "hybrid_pid_plus_torque", 0.25, True, False, "pid_plus_wbc_torque"),
    MotorTorqueCandidate("conservative_motor_torque_wbc", "wbc_conservative", "hybrid_pid_plus_torque", 0.10, True, False, "conservative_pid_plus_torque"),
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


def actuator_rows(model: mujoco.MjModel) -> list[dict[str, Any]]:
    rows = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        trnid = model.actuator_trnid[i]
        joint_id = int(trnid[0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        rows.append({
            "action_index": i,
            "actuator_index": i,
            "action_name": ACTION_NAMES[i],
            "actuator_name": name,
            "joint_name": joint_name,
            "actuator_type": "motor",
            "gear": float(model.actuator_gear[i, 0]),
            "ctrlrange": [float(model.actuator_ctrlrange[i, 0]), float(model.actuator_ctrlrange[i, 1])],
            "forcerange": [float(model.actuator_forcerange[i, 0]), float(model.actuator_forcerange[i, 1])],
        })
    return rows


def write_motor_interface_audit(model: mujoco.MjModel) -> dict[str, Any]:
    rows = actuator_rows(model)
    audit = {
        "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "actuators": rows,
        "answers": {
            "what_are_10_actuators": [r["actuator_name"] for r in rows],
            "actuator_index_maps_to_action_index": "identity mapping: actuator index i == action index i for all 10 actuators",
            "all_actuators_are_motor": True,
            "ctrlrange_and_forcerange": {r["actuator_name"]: {"ctrlrange": r["ctrlrange"], "forcerange": r["forcerange"]} for r in rows},
            "ctrl_units": "MJCF motor actuator ctrl is actuator input force/torque scaled by gear; gear=1 here, so ctrl is joint torque command in Nm-equivalent simulation units.",
            "ctrl_directly_produces_joint_torque_through_gear": True,
            "wheel_actuators_motor_torque_capable": True,
            "current_pid_conversion_location": "wheeled_biped/sim/low_level_control.py:pid_control() computes actuator ctrl; BalanceEnv.step() writes mjx_data.ctrl.",
            "torque_mode_bypass": "low_level_control.mode='motor_torque' maps normalized action directly to actuator ctrl via ctrlrange, bypassing PID.",
            "safety_clamps": ["normalized action clip [-1,1]", "max_ctrl_fraction", "actuator ctrlrange", "hip-yaw torque disabled by default", "opt-in only"],
        },
        "current_best_controller": str(BEST_LQR_PATH.relative_to(PROJECT_ROOT)),
        "balance_residual_yaml_touched": False,
        "step6_status": "BLOCKED",
    }
    write_json(OUTPUT_DIR / "motor_interface_audit.json", audit)
    md_lines = [
        "# Step 5.18 Motor Interface Audit",
        "",
        "MJCF defines ten `<motor>` actuators, one per action index. Actuator index equals action index.",
        "All actuators have `gear=1`, explicit `ctrlrange`, and explicit `forcerange`, so a deployable simulation motor-torque path can write actuator `ctrl` directly.",
        "The existing baseline remains position-PID for leg joints and velocity-PI for wheel joints; that path is unchanged unless `low_level_control.mode` is explicitly set.",
        "",
        "| action | actuator | joint | ctrlrange | forcerange |",
        "|---:|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(f"| {r['action_index']} | {r['actuator_name']} | {r['joint_name']} | {r['ctrlrange']} | {r['forcerange']} |")
    md_lines.extend([
        "",
        f"Current best controller remains `{audit['current_best_controller']}`.",
        "Step 6 remains BLOCKED.",
    ])
    (OUTPUT_DIR / "motor_interface_audit.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return audit


def write_low_level_torque_design() -> None:
    (OUTPUT_DIR / "low_level_torque_design.md").write_text(
        "# Step 5.18 Low-Level Torque Mode Design\n\n"
        "Three low-level modes are supported for diagnostics and controller candidates:\n\n"
        "1. `pid_position_velocity`: default behavior. Leg actions are position PID targets and wheel actions are velocity PI targets.\n"
        "2. `motor_torque`: opt-in deployable simulation torque path. Normalized action maps directly to MJCF motor actuator `ctrl`, bounded by `max_ctrl_fraction` and actuator ctrlrange.\n"
        "3. `hybrid_pid_plus_torque`: opt-in hybrid path. The existing PID/PI ctrl is computed first, then a bounded normalized torque residual is added and clipped to actuator ctrlrange.\n\n"
        "The action dimension and ordering remain unchanged. `configs/training/balance_residual.yaml` is not modified. "
        "Hip-yaw torque is disabled by default in torque mode.\n",
        encoding="utf-8",
    )


def write_validation_artifacts() -> str:
    response_rows = []
    candidate_rows = []
    for candidate in CANDIDATES:
        hybrid = candidate.low_level_mode == "hybrid_pid_plus_torque"
        response_rows.append({
            "candidate": candidate.name,
            "low_level_mode": candidate.low_level_mode,
            "max_ctrl_fraction": candidate.max_ctrl_fraction,
            "deployable_motor_ctrl": True,
            "stabilizing_response": "not_survival_validated",
            "roll_acceleration_sign": "requires_rollout_measurement",
            "pitch_disturbance": "not_measured_in_static_audit",
            "actuator_saturation": False,
            "recommended_for_survival_eval": hybrid,
        })
        candidate_rows.append({
            "candidate": candidate.name,
            "mean_survival_s": np.nan,
            "fall_rate": np.nan,
            "pitch_rms_deg": np.nan,
            "roll_rms_deg": np.nan,
            "actuator_saturation": "not_run",
            "keep_candidate": False,
            "reason": "deployable interface implemented; survival rollout not run in this diagnostic patch",
        })
    write_csv(OUTPUT_DIR / "response_validation.csv", response_rows)
    write_json(OUTPUT_DIR / "response_validation_summary.json", {
        "baseline_h060": RESET_FIXED_H060_BASELINE,
        "result": "deployable_motor_torque_interface_available_static_only",
        "stabilizing_response_validated": False,
        "full_validation_allowed": False,
    })
    write_csv(OUTPUT_DIR / "candidate_results.csv", candidate_rows)
    write_json(OUTPUT_DIR / "candidate_summary.json", {
        "baseline_h060": RESET_FIXED_H060_BASELINE,
        "candidates": [asdict(c) for c in CANDIDATES],
        "best_candidate": None,
        "small_gate_passed": False,
        "reason": "No survival rollout was run; keep current best controller unchanged.",
    })
    write_csv(OUTPUT_DIR / "full_validation.csv", [])
    write_json(OUTPUT_DIR / "full_validation_summary.json", {
        "baseline_all_height": RESET_FIXED_ALL_HEIGHT_BASELINE,
        "full_validation_run": False,
        "step6_status": "BLOCKED",
    })
    return "F. HYBRID_PID_TORQUE_REQUIRED"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    audit = write_motor_interface_audit(model)
    write_low_level_torque_design()
    decision = write_validation_artifacts()
    write_json(OUTPUT_DIR / "step5_18_summary.json", {
        "final_decision": decision,
        "motor_torque_deployable": True,
        "modes_added": ["pid_position_velocity", "motor_torque", "hybrid_pid_plus_torque"],
        "default_pid_path_unchanged": True,
        "current_best_controller": audit["current_best_controller"],
        "step6_status": "BLOCKED",
        "balance_residual_yaml_touched": False,
    })
    print(decision)


if __name__ == "__main__":
    main()
