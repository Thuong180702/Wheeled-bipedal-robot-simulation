from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pandas as pd

from scripts.audit_step_e_height_variant_position_hold_v2 import (
    analyze_telemetry as analyze_step_e_telemetry,
    classify_variant_result as classify_step_e_variant_result,
)
from wheeled_biped.utils.config import get_model_path
from wheeled_biped.validation.step_c_height_recovery import (
    StepCThresholds,
    build_step_c_pass_fail_summary,
    evaluate_step_c_case,
    render_step_c_report,
)


DEFAULT_OUTPUT_DIR = Path("outputs/operational_height_envelope_search")
STEP_E_OUTPUT_DIR = Path("outputs/step_e_extreme_height_position_hold")
STEP_C_OUTPUT_DIR = Path("outputs/step_c_extreme_height_recovery")
SIM_OUTPUT_DIR = Path("outputs/hierarchical_controller_sim")
D2_PROFILE = "candidate_D2_wheel_velocity_damping_light"
EXTREME_VARIANTS = ("min_operational_height", "max_operational_height")


@dataclass(frozen=True)
class StaticValidationThresholds:
    height_tolerance_m: float = 0.005
    support_error_preferred_m: float = 0.005
    support_error_max_m: float = 0.010
    pitch_max_abs_rad: float = 0.03
    roll_max_abs_rad: float = 0.03
    yaw_max_abs_rad: float = 0.03
    hip_yaw_max_abs_rad: float = 0.02
    joint_limit_margin_min_rad: float = 0.02
    selection_joint_margin_min_rad: float = 0.05
    controller_min_com_z_m: float = 0.38
    controller_max_com_z_m: float = 0.43
    min_wheel_floor_force_n: float = 1.0
    target_contact_dist_m: float = -5e-4
    root_z_only_joint_delta_min_rad: float = 1e-3


@dataclass
class CandidateStats:
    total_evaluated: int = 0
    passed_contact: int = 0
    passed_height: int = 0
    passed_com_centering: int = 0
    passed_orientation: int = 0
    passed_joint_margin: int = 0
    passed_all: int = 0
    best_by_height: list[float] | None = None
    best_by_com: list[float] | None = None
    top_rejected: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OperationalHeightCandidate:
    variant_name: str
    requested_target_com_z_m: float
    achieved_com_z_m: float
    calibrated_root_z_m: float
    hip_pitch_ref: float
    knee_ref: float
    nominal_hip_pitch_ref: float
    nominal_knee_ref: float
    hip_roll_left: float
    hip_roll_right: float
    hip_yaw_left: float
    hip_yaw_right: float
    support_center_x: float
    support_center_y: float
    com_x_m: float
    com_y_m: float
    com_support_error_x: float
    com_support_error_y: float
    com_support_error_norm_xy: float
    left_wheel_contact: bool
    right_wheel_contact: bool
    wheel_floor_contact_count: int
    non_wheel_floor_contact_count: int
    min_wheel_contact_dist_m: float
    total_wheel_floor_fz: float
    pitch_x_rad: float
    roll_y_rad: float
    yaw_z_rad: float
    joint_limit_margin_min_rad: float
    root_z_only: bool
    setup_valid: bool
    setup_failure_reason: str | None
    equilibrium_joint_pos: list[float] | None = None
    equilibrium_com_pos: list[float] | None = None
    equilibrium_pitch_x: float | None = None
    equilibrium_roll_y: float | None = None
    equilibrium_yaw_z: float | None = None
    posture_search_method: str = "operational_height_multiobjective_search"
    candidate_stats: dict[str, Any] | None = None
    wbc_applied: bool = False
    hidden_torque_norm_max: float = 0.0
    ownership_violation_count_max: int = 0

    @property
    def target_com_z_m(self) -> float:
        return self.requested_target_com_z_m

    @property
    def height_error_m(self) -> float:
        return abs(self.achieved_com_z_m - self.requested_target_com_z_m)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def compute_orientation_from_gravity(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    torso_body_id = model.body("torso").id
    torso_xmat = data.xmat[torso_body_id].reshape(3, 3)
    gravity_body = torso_xmat.T @ np.array([0.0, 0.0, -1.0])
    return float(gravity_body[0]), float(gravity_body[1])


def yaw_from_qpos(data: mujoco.MjData) -> float:
    quat = np.asarray(data.qpos[3:7], dtype=float)
    return float(2.0 * np.arctan2(quat[3], quat[0]))


def compute_support_center(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    l_wheel_body_id = model.body("l_wheel_link").id
    r_wheel_body_id = model.body("r_wheel_link").id
    l_wheel_pos = data.xpos[l_wheel_body_id]
    r_wheel_pos = data.xpos[r_wheel_body_id]
    return float(0.5 * (l_wheel_pos[0] + r_wheel_pos[0])), float(0.5 * (l_wheel_pos[1] + r_wheel_pos[1]))


def classify_floor_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    left_wheel_contact = False
    right_wheel_contact = False
    non_wheel_floor_contacts = 0
    min_dist = None
    total_wheel_floor_fz = 0.0

    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        if floor_geom_id not in (g1, g2):
            continue

        dist = float(contact.dist)
        min_dist = dist if min_dist is None else min(min_dist, dist)
        involves_left = l_wheel_geom_id in (g1, g2)
        involves_right = r_wheel_geom_id in (g1, g2)
        if involves_left or involves_right:
            left_wheel_contact = left_wheel_contact or involves_left
            right_wheel_contact = right_wheel_contact or involves_right
            force_contact = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force_contact)
            frame = np.asarray(contact.frame).reshape(3, 3)
            force_world = frame.T @ force_contact[:3]
            total_wheel_floor_fz += float(abs(force_world[2]))
        else:
            non_wheel_floor_contacts += 1

    return {
        "left_wheel_contact": bool(left_wheel_contact),
        "right_wheel_contact": bool(right_wheel_contact),
        "wheel_floor_contact_count": int(left_wheel_contact) + int(right_wheel_contact),
        "non_wheel_floor_contact_count": int(non_wheel_floor_contacts),
        "min_wheel_contact_dist_m": float(min_dist if min_dist is not None else 0.0),
        "total_wheel_floor_fz": float(total_wheel_floor_fz),
    }


def calibrate_root_z_for_wheel_floor_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    target_dist: float = -5e-4,
    max_iters: int = 30,
) -> float:
    l_wheel_body_id = model.body("l_wheel_link").id
    r_wheel_body_id = model.body("r_wheel_link").id
    wheel_radius = float(model.geom_size[model.geom("l_wheel_collision").id][0])
    lo, hi = 0.25, 0.75
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        data.qpos[2] = mid
        mujoco.mj_forward(model, data)
        l_bottom = float(data.xpos[l_wheel_body_id, 2] - wheel_radius)
        r_bottom = float(data.xpos[r_wheel_body_id, 2] - wheel_radius)
        avg_bottom = 0.5 * (l_bottom + r_bottom)
        if avg_bottom > target_dist:
            hi = mid
        else:
            lo = mid
        if abs(avg_bottom - target_dist) < 1e-5:
            break
    mujoco.mj_forward(model, data)
    return float(data.qpos[2])


def joint_limit_margin(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    margins = []
    for joint_name in ("l_hip_pitch", "l_knee", "r_hip_pitch", "r_knee"):
        joint_id = model.joint(joint_name).id
        qpos_addr = model.jnt_qposadr[joint_id]
        low, high = model.jnt_range[joint_id]
        value = float(data.qpos[qpos_addr])
        margins.append(value - float(low))
        margins.append(float(high) - value)
    return float(min(margins))


def make_data_for_pose(model: mujoco.MjModel, hip_pitch: float, knee: float) -> mujoco.MjData:
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[9] = hip_pitch
    data.qpos[10] = knee
    data.qpos[14] = hip_pitch
    data.qpos[15] = knee
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    calibrate_root_z_for_wheel_floor_contact(model, data)
    mujoco.mj_forward(model, data)
    return data


def build_candidate_from_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    variant_name: str,
    requested_target_com_z_m: float,
    nominal_hip_pitch_ref: float,
    nominal_knee_ref: float,
    candidate_stats: CandidateStats | None = None,
) -> OperationalHeightCandidate:
    torso_id = model.body("torso").id
    com = np.asarray(data.subtree_com[torso_id], dtype=float)
    support_x, support_y = compute_support_center(model, data)
    error_x = float(com[0] - support_x)
    error_y = float(com[1] - support_y)
    pitch_x, roll_y = compute_orientation_from_gravity(model, data)
    contacts = classify_floor_contacts(model, data)
    hip_pitch = float(data.qpos[9])
    knee = float(data.qpos[10])
    root_z_only = (
        abs(hip_pitch - nominal_hip_pitch_ref) < StaticValidationThresholds.root_z_only_joint_delta_min_rad
        and abs(knee - nominal_knee_ref) < StaticValidationThresholds.root_z_only_joint_delta_min_rad
        and abs(float(com[2]) - requested_target_com_z_m) > StaticValidationThresholds.height_tolerance_m
    )

    candidate = OperationalHeightCandidate(
        variant_name=variant_name,
        requested_target_com_z_m=float(requested_target_com_z_m),
        achieved_com_z_m=float(com[2]),
        calibrated_root_z_m=float(data.qpos[2]),
        hip_pitch_ref=hip_pitch,
        knee_ref=knee,
        nominal_hip_pitch_ref=float(nominal_hip_pitch_ref),
        nominal_knee_ref=float(nominal_knee_ref),
        hip_roll_left=float(data.qpos[7]),
        hip_roll_right=float(data.qpos[12]),
        hip_yaw_left=float(data.qpos[8]),
        hip_yaw_right=float(data.qpos[13]),
        support_center_x=support_x,
        support_center_y=support_y,
        com_x_m=float(com[0]),
        com_y_m=float(com[1]),
        com_support_error_x=error_x,
        com_support_error_y=error_y,
        com_support_error_norm_xy=float(math.sqrt(error_x * error_x + error_y * error_y)),
        pitch_x_rad=float(pitch_x),
        roll_y_rad=float(roll_y),
        yaw_z_rad=yaw_from_qpos(data),
        joint_limit_margin_min_rad=joint_limit_margin(model, data),
        root_z_only=bool(root_z_only),
        setup_valid=True,
        setup_failure_reason=None,
        equilibrium_joint_pos=np.asarray(data.qpos[7:17], dtype=float).tolist(),
        equilibrium_com_pos=com.tolist(),
        equilibrium_pitch_x=float(pitch_x),
        equilibrium_roll_y=float(roll_y),
        equilibrium_yaw_z=yaw_from_qpos(data),
        candidate_stats=asdict(candidate_stats) if candidate_stats is not None else None,
        **contacts,
    )
    return validate_operational_height_candidate(candidate)


def validate_operational_height_candidate(
    candidate: OperationalHeightCandidate,
    thresholds: StaticValidationThresholds | None = None,
) -> OperationalHeightCandidate:
    thresholds = thresholds or StaticValidationThresholds()
    failures: list[str] = []
    if candidate.height_error_m > thresholds.height_tolerance_m:
        failures.append(f"height_error={candidate.height_error_m:.6f}m")
    if not (candidate.left_wheel_contact and candidate.right_wheel_contact and candidate.wheel_floor_contact_count >= 2):
        failures.append("missing_wheel_contact")
    if candidate.non_wheel_floor_contact_count > 0:
        failures.append(f"non_wheel_floor_contacts={candidate.non_wheel_floor_contact_count}")
    if candidate.total_wheel_floor_fz < thresholds.min_wheel_floor_force_n:
        failures.append(f"wheel_contact_force_low={candidate.total_wheel_floor_fz:.6f}N")
    if candidate.com_support_error_norm_xy > thresholds.support_error_max_m:
        failures.append(f"support_not_centered={candidate.com_support_error_norm_xy:.6f}m")
    if abs(candidate.pitch_x_rad) > thresholds.pitch_max_abs_rad:
        failures.append(f"pitch_x={candidate.pitch_x_rad:.6f}rad")
    if abs(candidate.roll_y_rad) > thresholds.roll_max_abs_rad:
        failures.append(f"roll_y={candidate.roll_y_rad:.6f}rad")
    if abs(candidate.yaw_z_rad) > thresholds.yaw_max_abs_rad:
        failures.append(f"yaw_z={candidate.yaw_z_rad:.6f}rad")
    if max(abs(candidate.hip_yaw_left), abs(candidate.hip_yaw_right)) > thresholds.hip_yaw_max_abs_rad:
        failures.append("hip_yaw_not_near_reference")
    if candidate.joint_limit_margin_min_rad < thresholds.joint_limit_margin_min_rad:
        failures.append(f"joint_limit_margin={candidate.joint_limit_margin_min_rad:.6f}rad")
    if candidate.achieved_com_z_m < thresholds.controller_min_com_z_m:
        failures.append(f"below_controller_min_com_z={candidate.achieved_com_z_m:.6f}m")
    if candidate.achieved_com_z_m > thresholds.controller_max_com_z_m:
        failures.append(f"above_controller_max_com_z={candidate.achieved_com_z_m:.6f}m")
    if candidate.root_z_only:
        failures.append("root_z_only")
    if candidate.wbc_applied:
        failures.append("wbc_applied")
    if candidate.hidden_torque_norm_max != 0.0:
        failures.append("hidden_torque_nonzero")
    if candidate.ownership_violation_count_max != 0:
        failures.append("ownership_violation")

    candidate.setup_valid = not failures
    candidate.setup_failure_reason = "; ".join(failures) if failures else None
    if not candidate.setup_valid:
        candidate.equilibrium_joint_pos = None
        candidate.equilibrium_com_pos = None
        candidate.equilibrium_pitch_x = None
        candidate.equilibrium_roll_y = None
        candidate.equilibrium_yaw_z = None
    return candidate


def nominal_pose(model: mujoco.MjModel) -> OperationalHeightCandidate:
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    calibrate_root_z_for_wheel_floor_contact(model, data)
    mujoco.mj_forward(model, data)
    torso_id = model.body("torso").id
    nominal_com_z = float(data.subtree_com[torso_id][2])
    return build_candidate_from_data(
        model,
        data,
        variant_name="nominal_reference",
        requested_target_com_z_m=nominal_com_z,
        nominal_hip_pitch_ref=float(data.qpos[9]),
        nominal_knee_ref=float(data.qpos[10]),
    )


def search_height_candidate(
    model: mujoco.MjModel,
    *,
    variant_name: str,
    target_com_z_m: float,
    nominal: OperationalHeightCandidate,
    search_range: float = 0.45,
    search_steps: int = 36,
    thresholds: StaticValidationThresholds | None = None,
) -> OperationalHeightCandidate:
    thresholds = thresholds or StaticValidationThresholds()
    stats = CandidateStats()
    best_valid: tuple[float, OperationalHeightCandidate] | None = None
    best_any: tuple[float, OperationalHeightCandidate] | None = None
    hip_values = np.linspace(nominal.hip_pitch_ref - search_range, nominal.hip_pitch_ref + search_range, search_steps)
    knee_values = np.linspace(nominal.knee_ref - search_range, nominal.knee_ref + search_range, search_steps)

    for hip_pitch in hip_values:
        for knee in knee_values:
            stats.total_evaluated += 1
            try:
                data = make_data_for_pose(model, float(hip_pitch), float(knee))
                candidate = build_candidate_from_data(
                    model,
                    data,
                    variant_name=variant_name,
                    requested_target_com_z_m=target_com_z_m,
                    nominal_hip_pitch_ref=nominal.hip_pitch_ref,
                    nominal_knee_ref=nominal.knee_ref,
                )
            except Exception as exc:
                stats.top_rejected.append({"hip_pitch": float(hip_pitch), "knee": float(knee), "reason": str(exc)})
                continue

            height_error = candidate.height_error_m
            score = (
                100.0 * height_error
                + 50.0 * candidate.com_support_error_norm_xy
                + 10.0 * abs(candidate.pitch_x_rad)
                + 10.0 * abs(candidate.roll_y_rad)
                + abs(candidate.hip_pitch_ref - nominal.hip_pitch_ref)
                + abs(candidate.knee_ref - nominal.knee_ref)
            )
            if best_any is None or score < best_any[0]:
                best_any = (score, candidate)
            if stats.best_by_height is None or height_error < stats.best_by_height[2]:
                stats.best_by_height = [float(hip_pitch), float(knee), float(height_error), float(score)]
            if stats.best_by_com is None or candidate.com_support_error_norm_xy < stats.best_by_com[2]:
                stats.best_by_com = [float(hip_pitch), float(knee), float(candidate.com_support_error_norm_xy), float(score)]

            if candidate.left_wheel_contact and candidate.right_wheel_contact and candidate.non_wheel_floor_contact_count == 0:
                stats.passed_contact += 1
            if height_error <= thresholds.height_tolerance_m:
                stats.passed_height += 1
            if candidate.com_support_error_norm_xy <= thresholds.support_error_max_m:
                stats.passed_com_centering += 1
            if abs(candidate.pitch_x_rad) <= thresholds.pitch_max_abs_rad and abs(candidate.roll_y_rad) <= thresholds.roll_max_abs_rad:
                stats.passed_orientation += 1
            if candidate.joint_limit_margin_min_rad >= thresholds.joint_limit_margin_min_rad:
                stats.passed_joint_margin += 1
            if candidate.setup_valid:
                stats.passed_all += 1
                if best_valid is None or score < best_valid[0]:
                    best_valid = (score, candidate)
            elif len(stats.top_rejected) < 20:
                stats.top_rejected.append({
                    "hip_pitch": float(hip_pitch),
                    "knee": float(knee),
                    "achieved_com_z_m": candidate.achieved_com_z_m,
                    "support_error_norm_xy": candidate.com_support_error_norm_xy,
                    "reason": candidate.setup_failure_reason,
                })

    selected = best_valid[1] if best_valid is not None else best_any[1]
    selected.candidate_stats = asdict(stats)
    return validate_operational_height_candidate(selected, thresholds)


def search_operational_height_envelope(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    step_m: float = 0.005,
    lower_extra_m: float = 0.08,
    upper_extra_m: float = 0.08,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    nominal = nominal_pose(model)
    thresholds = StaticValidationThresholds()

    lower_targets = np.arange(nominal.achieved_com_z_m - lower_extra_m, nominal.achieved_com_z_m, step_m)
    upper_targets = np.arange(nominal.achieved_com_z_m + step_m, nominal.achieved_com_z_m + upper_extra_m + 0.5 * step_m, step_m)
    targets = [float(x) for x in lower_targets] + [nominal.achieved_com_z_m] + [float(x) for x in upper_targets]

    candidates: list[OperationalHeightCandidate] = []
    for target in targets:
        if abs(target - nominal.achieved_com_z_m) < 1e-9:
            candidate = nominal
            candidate.variant_name = "nominal_reference"
        else:
            direction = "low" if target < nominal.achieved_com_z_m else "high"
            candidate = search_height_candidate(
                model,
                variant_name=f"search_{direction}_{target:.3f}m",
                target_com_z_m=target,
                nominal=nominal,
                thresholds=thresholds,
            )
        candidates.append(candidate)

    selected = select_envelope_extrema(candidates, thresholds)
    min_candidate = selected["min_candidate"]
    max_candidate = selected["max_candidate"]
    min_candidate.variant_name = "min_operational_height"
    max_candidate.variant_name = "max_operational_height"

    write_envelope_artifacts(output_dir, candidates, selected, thresholds)
    return {
        "summary": selected,
        "candidates": candidates,
        "thresholds": asdict(thresholds),
        "output_dir": str(output_dir),
    }


def select_envelope_extrema(
    candidates: list[OperationalHeightCandidate],
    thresholds: StaticValidationThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or StaticValidationThresholds()
    validated = []
    for candidate in candidates:
        if candidate.setup_valid:
            validated.append(validate_operational_height_candidate(candidate, thresholds))
        else:
            validated.append(candidate)
    safe = [
        candidate for candidate in validated
        if candidate.setup_valid and candidate.joint_limit_margin_min_rad >= thresholds.selection_joint_margin_min_rad
    ]
    if not safe:
        return {
            "envelope_search_verdict": "INCONCLUSIVE",
            "reason": "no_static_candidate_with_selection_safety_margin",
            "min_candidate": None,
            "max_candidate": None,
            "extrema_are_conservative": True,
        }
    min_candidate = min(safe, key=lambda candidate: candidate.achieved_com_z_m)
    max_candidate = max(safe, key=lambda candidate: candidate.achieved_com_z_m)
    return {
        "envelope_search_verdict": "PASS",
        "min_candidate": min_candidate,
        "max_candidate": max_candidate,
        "min_operational_height_m": min_candidate.achieved_com_z_m,
        "max_operational_height_m": max_candidate.achieved_com_z_m,
        "extrema_are_conservative": True,
        "absolute_mechanical_extrema_claimed": False,
        "selection_joint_margin_min_rad": thresholds.selection_joint_margin_min_rad,
    }


def setup_json_payload(candidate: OperationalHeightCandidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["target_com_z_m"] = candidate.requested_target_com_z_m
    payload["height_error_m"] = candidate.height_error_m
    payload["joint_limit_valid"] = candidate.joint_limit_margin_min_rad >= StaticValidationThresholds().joint_limit_margin_min_rad
    return payload


def write_envelope_artifacts(
    output_dir: Path,
    candidates: list[OperationalHeightCandidate],
    selected: dict[str, Any],
    thresholds: StaticValidationThresholds,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "operational_height_search_grid.csv"
    fields = list(asdict(candidates[0]).keys()) + ["height_error_m", "target_com_z_m"] if candidates else []
    with grid_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            row = asdict(candidate)
            row["height_error_m"] = candidate.height_error_m
            row["target_com_z_m"] = candidate.requested_target_com_z_m
            writer.writerow(row)

    valid_candidates = [setup_json_payload(candidate) for candidate in candidates if candidate.setup_valid]
    (output_dir / "operational_height_valid_candidates.json").write_text(json.dumps(valid_candidates, indent=2), encoding="utf-8")

    min_candidate = selected.get("min_candidate")
    max_candidate = selected.get("max_candidate")
    summary = {
        key: value for key, value in selected.items()
        if key not in {"min_candidate", "max_candidate"}
    }
    summary["min_candidate"] = setup_json_payload(min_candidate) if min_candidate is not None else None
    summary["max_candidate"] = setup_json_payload(max_candidate) if max_candidate is not None else None
    summary["thresholds"] = asdict(thresholds)
    summary["candidate_count"] = len(candidates)
    summary["valid_candidate_count"] = len(valid_candidates)
    (output_dir / "operational_height_envelope_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    static_validation = {
        "static_extreme_validation_verdict": summary.get("envelope_search_verdict", "INCONCLUSIVE"),
        "min_operational_height": summary["min_candidate"],
        "max_operational_height": summary["max_candidate"],
        "checks": [
            "wheel contacts valid",
            "no non-wheel floor contacts",
            "support error within threshold",
            "pitch/roll/yaw safe",
            "hip/knee joint limit margin enforced",
            "not root-z-only",
            "equilibrium references captured",
        ],
    }
    (output_dir / "static_extreme_validation.json").write_text(json.dumps(static_validation, indent=2), encoding="utf-8")

    if min_candidate is not None:
        (output_dir / "min_operational_height_setup.json").write_text(json.dumps(setup_json_payload(min_candidate), indent=2), encoding="utf-8")
    if max_candidate is not None:
        (output_dir / "max_operational_height_setup.json").write_text(json.dumps(setup_json_payload(max_candidate), indent=2), encoding="utf-8")

    report = render_envelope_report(summary)
    (output_dir / "operational_height_envelope_report.md").write_text(report, encoding="utf-8")


def render_envelope_report(summary: dict[str, Any]) -> str:
    min_candidate = summary.get("min_candidate") or {}
    max_candidate = summary.get("max_candidate") or {}
    return "\n".join([
        "# Operational Height Envelope Search",
        "",
        f"- Verdict: **{summary.get('envelope_search_verdict', 'INCONCLUSIVE')}**",
        "- Extrema type: conservative validated extrema, not absolute mechanical limits",
        f"- Min operational height: `{min_candidate.get('achieved_com_z_m', 'n/a')}` m",
        f"- Max operational height: `{max_candidate.get('achieved_com_z_m', 'n/a')}` m",
        "",
        "## Static validity definition",
        "",
        "A candidate is valid only when it changes hip/knee posture, calibrates root_z for wheel contact, keeps both wheels on the floor, avoids non-wheel floor contact, centers CoM over the support region, preserves upright posture, and keeps hip/knee away from joint limits.",
        "",
        "## Selected min candidate",
        "",
        json.dumps(min_candidate, indent=2),
        "",
        "## Selected max candidate",
        "",
        json.dumps(max_candidate, indent=2),
    ])


def snapshot_outputs() -> tuple[set[Path], set[Path]]:
    csvs = set(SIM_OUTPUT_DIR.glob("telemetry_*.csv")) if SIM_OUTPUT_DIR.exists() else set()
    sidecars = set(SIM_OUTPUT_DIR.glob("telemetry_*.summary.json")) if SIM_OUTPUT_DIR.exists() else set()
    return csvs, sidecars


def copy_newest_outputs(case_name: str, output_dir: Path, before_csv: set[Path], before_sidecar: set[Path]) -> Path | None:
    current_csv = set(SIM_OUTPUT_DIR.glob("telemetry_*.csv")) if SIM_OUTPUT_DIR.exists() else set()
    new_csv = current_csv - before_csv
    if not new_csv:
        return None
    source_csv = max(new_csv, key=lambda path: path.stat().st_mtime)
    dest_csv = output_dir / f"{case_name}_telemetry.csv"
    shutil.copy2(source_csv, dest_csv)

    current_sidecars = set(SIM_OUTPUT_DIR.glob("telemetry_*.summary.json")) if SIM_OUTPUT_DIR.exists() else set()
    new_sidecars = current_sidecars - before_sidecar
    if new_sidecars:
        source_sidecar = max(new_sidecars, key=lambda path: path.stat().st_mtime)
        shutil.copy2(source_sidecar, dest_csv.with_suffix(".summary.json"))
    return dest_csv


def build_extreme_simulation_command(setup_path: Path, steps: int) -> list[str]:
    return [
        "python",
        "scripts/simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--height-variant-setup", str(setup_path).replace("\\", "/"),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", "500",
        "--write-run-summary-sidecar",
        "--vd-sagittal-authority-profile", D2_PROFILE,
    ]


def run_extreme_simulation(case_name: str, setup_path: Path, output_dir: Path, steps: int) -> tuple[dict[str, Any], Path | None]:
    before_csv, before_sidecar = snapshot_outputs()
    cmd = build_extreme_simulation_command(setup_path, steps)
    process_error = None
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        process_error = exc
    telemetry_path = copy_newest_outputs(case_name, output_dir, before_csv, before_sidecar)
    return {
        "case_name": case_name,
        "command": cmd,
        "simulation_returncode": None if process_error is None else process_error.returncode,
        "simulation_error": None if process_error is None else str(process_error),
        "telemetry_path": str(telemetry_path) if telemetry_path is not None else None,
    }, telemetry_path


def load_setup(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_step_e_extreme_validation(
    *,
    envelope_dir: Path = DEFAULT_OUTPUT_DIR,
    output_dir: Path = STEP_E_OUTPUT_DIR,
    steps: int = 5000,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("min_operational_height", envelope_dir / "min_operational_height_setup.json"),
        ("max_operational_height", envelope_dir / "max_operational_height_setup.json"),
    ]
    case_matrix = []
    results = []
    metrics = []
    for case_name, setup_path in cases:
        setup = load_setup(setup_path)
        case_matrix.append({"case_name": case_name, "variant_setup_path": str(setup_path), "target_com_z_m": setup["achieved_com_z_m"]})
        result, telemetry_path = run_extreme_simulation(case_name, setup_path, output_dir, steps)
        if telemetry_path is None or not telemetry_path.exists():
            result.update({"verdict": "INCONCLUSIVE", "primary_failure": "telemetry_missing", "failure_classifications": ["telemetry_missing"]})
            results.append(result)
            continue
        metric = analyze_step_e_telemetry(telemetry_path, setup)
        metric["variant_name"] = case_name
        metric["case_name"] = case_name
        metric["candidate_D2_profile_used"] = True
        classified = classify_step_e_variant_result(metric)
        result.update(classified)
        result["variant_name"] = case_name
        results.append(result)
        metrics.append(metric)

    fail_count = sum(1 for result in results if result.get("verdict") == "FAIL")
    inconclusive_count = sum(1 for result in results if result.get("verdict") == "INCONCLUSIVE")
    overall = "PASS" if fail_count == 0 and inconclusive_count == 0 else "FAIL" if fail_count else "INCONCLUSIVE"
    summary = {
        "overall_step_e_extreme_verdict": overall,
        "final_decision": f"STEP_E_EXTREME_HEIGHT_{overall}",
        "candidate_D2_profile_used": True,
        "controller_behavior_changed": False,
        "case_count": len(results),
        "passed_cases": [r["case_name"] for r in results if r.get("verdict") == "PASS"],
        "failed_cases": [r["case_name"] for r in results if r.get("verdict") == "FAIL"],
        "inconclusive_cases": [r["case_name"] for r in results if r.get("verdict") == "INCONCLUSIVE"],
    }
    (output_dir / "step_e_extreme_case_matrix.json").write_text(json.dumps(case_matrix, indent=2), encoding="utf-8")
    (output_dir / "step_e_extreme_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "step_e_extreme_position_hold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "step_e_extreme_position_hold_report.md").write_text(render_step_e_extreme_report(summary, results, metrics), encoding="utf-8")
    return {"summary": summary, "results": results, "metrics": metrics}


def render_step_e_extreme_report(summary: dict[str, Any], results: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> str:
    metric_by_case = {m["case_name"]: m for m in metrics}
    lines = [
        "# Step E Extreme-Height Position Hold Report",
        "",
        f"- Overall verdict: **{summary['overall_step_e_extreme_verdict']}**",
        "- Controller behavior changed: `false`",
        f"- Candidate profile used: `{D2_PROFILE}`",
        "",
        "| Case | Verdict | Support max (m) | HipYaw max (rad) | Pitch max (rad) | Roll max (rad) | Wheel max (rad/s) | Height final error (m) | Contact valid (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        case = result["case_name"]
        metric = metric_by_case.get(case, {})
        support = metric.get("support_position_error_m", {})
        posture = metric.get("posture", {})
        height = metric.get("height", {})
        wheel_contact = metric.get("wheel_contact", {})
        lines.append(
            f"| {case} | {result.get('verdict')} | {support.get('max_abs')} | {posture.get('hip_yaw_abs_max_max_rad')} | {posture.get('pitch_x_max_abs_rad')} | {posture.get('roll_y_max_abs_rad')} | {wheel_contact.get('wheel_vel_mean_max_abs_rad_s')} | {height.get('final_error_vs_achieved_initial_m')} | {wheel_contact.get('contact_valid_percent_raw')} |"
        )
    return "\n".join(lines) + "\n"


def run_step_c_extreme_validation(
    *,
    envelope_dir: Path = DEFAULT_OUTPUT_DIR,
    output_dir: Path = STEP_C_OUTPUT_DIR,
    steps: int = 5000,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = StepCThresholds()
    case_matrix = []
    results = []
    for case_name, setup_file in [
        ("min_operational_height", "min_operational_height_setup.json"),
        ("max_operational_height", "max_operational_height_setup.json"),
    ]:
        setup_path = envelope_dir / setup_file
        setup = load_setup(setup_path)
        case = {
            "case_name": case_name,
            "initialization_method": "operational_height_extreme_static_pose",
            "variant_setup_path": str(setup_path),
            "target_com_z_m": setup["achieved_com_z_m"],
            "achieved_initial_com_z_m": setup["achieved_com_z_m"],
            "calibrated_root_z_m": setup["calibrated_root_z_m"],
            "hip_pitch_ref": setup["hip_pitch_ref"],
            "knee_ref": setup["knee_ref"],
            "setup_valid": setup["setup_valid"],
            "left_wheel_contact": setup["left_wheel_contact"],
            "right_wheel_contact": setup["right_wheel_contact"],
            "non_wheel_floor_contact_count": setup["non_wheel_floor_contact_count"],
            "local_transition_recovery": "NOT_RUN",
        }
        case_matrix.append(case)
        result, telemetry_path = run_extreme_simulation(case_name, setup_path, output_dir, steps)
        if telemetry_path is None or not telemetry_path.exists():
            result.update({
                "verdict": "INCONCLUSIVE",
                "primary_failure": "unclear_requires_more_telemetry",
                "failure_classifications": ["unclear_requires_more_telemetry", "telemetry_missing"],
                "wbc_applied": False,
                "step_e_invariants_preserved": False,
            })
        else:
            df = pd.read_csv(telemetry_path)
            result.update(evaluate_step_c_case(
                df,
                case_name=case_name,
                target_com_z_m=float(setup["achieved_com_z_m"]),
                expected_steps=steps,
                thresholds=thresholds,
            ))
            result["telemetry_path"] = str(telemetry_path)
        result.update(case)
        result["command"] = build_extreme_simulation_command(setup_path, steps)
        results.append(result)

    summary = build_step_c_pass_fail_summary(results, controller_behavior_changed=False)
    artifacts = {
        "case_matrix": output_dir / "step_c_extreme_case_matrix.json",
        "metrics": output_dir / "step_c_extreme_metrics.json",
        "summary": output_dir / "step_c_extreme_pass_fail_summary.json",
        "report": output_dir / "step_c_extreme_height_recovery_report.md",
    }
    artifacts["case_matrix"].write_text(json.dumps(case_matrix, indent=2), encoding="utf-8")
    artifacts["metrics"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary["artifact_paths"] = {key: str(path) for key, path in artifacts.items() if key != "summary"}
    artifacts["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    artifacts["report"].write_text(render_step_c_report(
        case_results=results,
        summary=summary,
        artifact_paths={key: str(path) for key, path in artifacts.items()},
    ), encoding="utf-8")
    return {"summary": summary, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Search and validate operational standing-height extrema")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step-m", type=float, default=0.005)
    parser.add_argument("--lower-extra-m", type=float, default=0.08)
    parser.add_argument("--upper-extra-m", type=float, default=0.08)
    parser.add_argument("--run-step-e", action="store_true")
    parser.add_argument("--run-step-c", action="store_true")
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    search_result = search_operational_height_envelope(
        output_dir=args.output_dir,
        step_m=args.step_m,
        lower_extra_m=args.lower_extra_m,
        upper_extra_m=args.upper_extra_m,
    )
    verdict = search_result["summary"].get("envelope_search_verdict")
    if verdict != "PASS":
        return 1
    if args.run_step_e:
        step_e = run_step_e_extreme_validation(envelope_dir=args.output_dir, steps=args.steps)
        if step_e["summary"]["overall_step_e_extreme_verdict"] != "PASS":
            return 1
    if args.run_step_c:
        step_c = run_step_c_extreme_validation(envelope_dir=args.output_dir, steps=args.steps)
        if step_c["summary"]["overall_step_c_verdict"] != "PASS":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
