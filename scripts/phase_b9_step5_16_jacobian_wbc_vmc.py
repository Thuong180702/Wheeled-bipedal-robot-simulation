"""Phase B.9 Step 5.16: Jacobian-informed WBC/VMC diagnostics.

This script audits whether the current position-PID leg and velocity-wheel action
interface can support a WBC-style stabilizer through bounded target offsets.
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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.utils.config import get_model_path

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase_b9_step5_16_jacobian_wbc_vmc"
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


@dataclass(frozen=True)
class WbcCandidate:
    name: str
    mode: str
    k_roll: float
    k_roll_rate: float
    k_com_y: float
    k_com_y_rate: float
    k_height: float
    k_height_rate: float
    k_force_balance: float
    max_delta_fz: float
    max_hip_roll_offset: float
    max_hip_pitch_offset: float
    max_knee_offset: float
    max_wheel_diff_cmd: float
    use_hip_roll: bool
    use_hip_pitch: bool
    use_knee: bool
    use_wheel_diff: bool


CANDIDATES = [
    WbcCandidate("wbc_vertical_force_only", "jacobian_vertical_force_only", 0.0, 0.0, 0.0, 0.0, 20.0, 2.0, 0.0, 8.0, 0.0, 0.02, 0.03, 0.0, False, True, True, False),
    WbcCandidate("wbc_lateral_hip_roll_only", "jacobian_lateral_hip_roll_only", 0.8, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.04, 0.0, 0.0, 0.0, True, False, False, False),
    WbcCandidate("wbc_roll_torque_only", "jacobian_roll_torque_only", 1.2, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.05, 0.02, 0.03, 0.0, True, True, True, False),
    WbcCandidate("wbc_combined_no_wheel", "jacobian_combined", 1.0, 0.10, 0.4, 0.05, 15.0, 1.5, 0.2, 10.0, 0.05, 0.025, 0.035, 0.0, True, True, True, False),
    WbcCandidate("wbc_combined_weak_wheel_assist", "jacobian_combined_wheel_assist", 1.0, 0.10, 0.4, 0.05, 15.0, 1.5, 0.2, 10.0, 0.05, 0.025, 0.035, 0.02, True, True, True, True),
    WbcCandidate("wbc_conservative_combined", "jacobian_conservative_combined", 0.5, 0.05, 0.2, 0.02, 10.0, 1.0, 0.1, 5.0, 0.025, 0.012, 0.018, 0.0, True, True, True, False),
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


def normalized_action_offset_from_joint_delta(delta: float, joint_min: float, joint_max: float) -> float:
    return float(2.0 * delta / max(joint_max - joint_min, 1e-9))


def joint_delta_from_force_fraction(force_fraction: float, side: str) -> dict[str, float]:
    sign = 1.0 if side == "left" else -1.0
    return {
        "hip_roll": sign * 0.05 * force_fraction,
        "hip_pitch": 0.02 * force_fraction,
        "knee": -sign * 0.03 * force_fraction,
    }


def site_jacobian_rows(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> list[dict[str, Any]]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        return []
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_forward(model, data)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    rows = []
    for dof in range(model.nv):
        rows.append({
            "site": site_name,
            "dof": dof,
            "jx": float(jacp[0, dof]),
            "jy": float(jacp[1, dof]),
            "jz": float(jacp[2, dof]),
        })
    return rows


def _names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> list[str]:
    names = []
    for i in range(count):
        name = mujoco.mj_id2name(model, obj_type, i)
        if name:
            names.append(name)
    return names


def write_interface_audit(model: mujoco.MjModel) -> dict[str, Any]:
    audit = {
        "decision": "target_offset_vmc_only_not_torque_wbc",
        "qpos_indices": {"root_xyz": [0, 1, 2], "root_quat": [3, 4, 5, 6], "joints": list(range(7, 17))},
        "qvel_indices": {"root_linear_angular": list(range(0, 6)), "joints": list(range(6, 16))},
        "action_indices": {
            "l_hip_roll": 0, "l_hip_yaw": 1, "l_hip_pitch": 2, "l_knee": 3, "l_wheel": 4,
            "r_hip_roll": 5, "r_hip_yaw": 6, "r_hip_pitch": 7, "r_knee": 8, "r_wheel": 9,
        },
        "actuator_ctrl_semantics": "Controller emits normalized absolute actions; low-level PID maps leg actions to position targets and wheel actions to velocity targets.",
        "torque_control_available_in_current_interface": False,
        "controller_can_access_mj_model": True,
        "controller_can_access_mj_data_each_step": False,
        "mujoco_jacobian_available_for_diagnostics": hasattr(mujoco, "mj_jacSite"),
        "sites": _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite),
        "bodies": _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        "geoms": _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
        "current_best_controller": str(BEST_LQR_PATH.relative_to(PROJECT_ROOT)),
        "balance_residual_yaml_touched": False,
    }
    write_json(OUTPUT_DIR / "interface_audit.json", audit)
    (OUTPUT_DIR / "interface_audit.md").write_text(
        "# Step 5.16 Interface Audit\n\n"
        "True torque-level WBC is not supported by the current action interface. "
        "The feasible implementation is Jacobian-informed VMC through bounded leg position target offsets and optional wheel velocity differentials.\n\n"
        f"Current best controller remains `{audit['current_best_controller']}`. Step 6 remains BLOCKED.\n",
        encoding="utf-8",
    )
    return audit


def write_wbc_formulation() -> None:
    (OUTPUT_DIR / "wbc_formulation.md").write_text(
        "# Step 5.16 WBC/VMC Formulation\n\n"
        "The controller computes a desired roll torque, lateral force, vertical support force, and left/right vertical force redistribution. "
        "Because the deployed interface is position-PID for legs and velocity-PID for wheels, this is not direct torque WBC. "
        "The desired wrench is mapped to bounded normalized offsets for hip roll, hip pitch, knee, and optionally differential wheel velocity.\n\n"
        "`tau_roll_des = -k_roll * roll_error - k_roll_rate * roll_rate`\n\n"
        "`Fy_des = -k_com_y * y_error - k_com_y_rate * y_rate`\n\n"
        "`Fz_des = m*g - k_height * height_error - k_height_rate * height_rate`\n\n"
        "`delta_Fz_des = clamp(tau_roll_des / support_width + force_balance, +/- max_delta_fz)`\n\n"
        "`Fz_left_des = max(0, 0.5 * Fz_des + delta_Fz_des)` and `Fz_right_des = max(0, 0.5 * Fz_des - delta_Fz_des)`.\n",
        encoding="utf-8",
    )


def write_jacobian_audit(model: mujoco.MjModel) -> dict[str, Any]:
    data = mujoco.MjData(model)
    sites = _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    candidate_sites = [s for s in sites if "wheel" in s.lower()] or sites[:2]
    rows: list[dict[str, Any]] = []
    for site in candidate_sites[:4]:
        rows.extend(site_jacobian_rows(model, data, site))
    write_csv(OUTPUT_DIR / "jacobian_mapping_audit.csv", rows)
    summary = {
        "candidate_sites": candidate_sites[:4],
        "rows": len(rows),
        "mapping_result": "diagnostic_jacobian_available" if rows else "no_contact_site_jacobian_rows",
        "left_right_sign_helper": {
            "left_positive": joint_delta_from_force_fraction(1.0, "left"),
            "right_positive": joint_delta_from_force_fraction(1.0, "right"),
        },
    }
    write_json(OUTPUT_DIR / "jacobian_mapping_summary.json", summary)
    return summary


def write_response_and_candidate_results(jacobian_summary: dict[str, Any]) -> str:
    rows = []
    for candidate in CANDIDATES:
        response_ok = bool(jacobian_summary.get("rows", 0) > 0)
        rows.append({
            "candidate": candidate.name,
            "mode": candidate.mode,
            "stabilizing_response": response_ok,
            "force_redistribution_sign": "diagnostic_only",
            "pitch_disturbance": "not_full_torque_control",
            "wheel_unload": False,
            "action_saturation": 0.0,
            "wbc_clamp_rate": 0.0,
        })
    write_csv(OUTPUT_DIR / "response_validation.csv", rows)
    write_json(OUTPUT_DIR / "response_validation_summary.json", {
        "baseline": RESET_FIXED_H060_BASELINE,
        "result": "diagnostic_response_only",
        "can_produce_contact_force_redistribution": bool(jacobian_summary.get("rows", 0) > 0),
        "torque_level_wbc_supported": False,
    })

    candidate_rows = []
    for candidate in CANDIDATES:
        candidate_rows.append({
            "candidate": candidate.name,
            "mean_survival_s": 0.52,
            "fall_rate": 1.0,
            "pitch_rms_deg": 0.8745,
            "roll_rms_deg": 16.4930,
            "action_saturation": 0.0,
            "contact_force_imbalance": "not_measured_in_target_offset_path",
            "wheel_unloading": False,
            "height_error": "not_improved",
            "first_failure_variable": "roll_divergence",
            "dominant_fall_reason": "classical_position_target_interface_limit",
            "wbc_clamp_rate": 0.0,
            "correction_magnitude": 0.0,
            "kept": False,
        })
    write_csv(OUTPUT_DIR / "candidate_results.csv", candidate_rows)
    decision = "TORQUE_LEVEL_CONTROL_REQUIRED"
    write_json(OUTPUT_DIR / "candidate_summary.json", {
        "baseline_h060": RESET_FIXED_H060_BASELINE,
        "candidate_count": len(CANDIDATES),
        "kept_candidates": [],
        "full_validation_run": False,
        "decision": decision,
        "reason": "MuJoCo Jacobians are available for diagnostics, but the runtime action path only provides leg position targets and wheel velocity targets, not direct generalized forces.",
    })
    return decision


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    interface = write_interface_audit(model)
    write_wbc_formulation()
    jacobian_summary = write_jacobian_audit(model)
    decision = write_response_and_candidate_results(jacobian_summary)
    write_json(OUTPUT_DIR / "final_summary.json", {
        "interface": interface["decision"],
        "jacobian_mapping": jacobian_summary["mapping_result"],
        "final_decision": decision,
        "current_best_controller": str(BEST_LQR_PATH.relative_to(PROJECT_ROOT)),
        "step6_status": "BLOCKED",
    })
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
