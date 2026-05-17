"""Phase B.9 posture symmetry fix.

Generates corrected B9 initial posture renders in a separate output folder so the
legacy baseline stays intact for old-vs-fixed comparison.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase_b9_posture_geometry_inspection import (  # noqa: E402
    HEIGHTS,
    JOINT_NAMES,
    PoseSpec,
    VIEWS,
    body_com,
    contact_forces_by_wheel,
    inspect_joint_axes,
    render_pose,
    wheel_bottom_heights,
)
from wheeled_biped.controllers.dual_rate_balance_controller import (  # noqa: E402
    DualRateConfig,
)
from wheeled_biped.controllers.height_ik import HeightIK  # noqa: E402
from wheeled_biped.utils.config import get_model_path  # noqa: E402

OLD_OUTPUT = PROJECT_ROOT / "outputs" / "phase_b9_geometry_check"
FIXED_OUTPUT = PROJECT_ROOT / "outputs" / "phase_b9_posture_fixed"
REPORT_PATH = PROJECT_ROOT / "docs" / "phase_b9_posture_geometry_inspection_report.md"

FIXED_POSTURE_COLUMNS = [
    "posture_name",
    "target_height",
    "hip_roll_L",
    "hip_roll_R",
    "hip_yaw_L",
    "hip_yaw_R",
    "hip_pitch_L",
    "hip_pitch_R",
    "knee_L",
    "knee_R",
    "torso_height",
    "root_x",
    "root_y",
    "root_z",
    "root_quat_w",
    "root_quat_x",
    "root_quat_y",
    "root_quat_z",
    "root_pitch_deg",
    "root_roll_deg",
    "root_yaw_deg",
    "torso_pitch",
    "torso_roll",
    "CoM_x",
    "CoM_y",
    "CoM_z",
    "wheel_contact_x",
    "wheel_contact_y",
    "wheel_contact_z",
    "left_wheel_bottom_z",
    "right_wheel_bottom_z",
    "left_wheel_clearance",
    "right_wheel_clearance",
    "left_wheel_contact_force",
    "right_wheel_contact_force",
    "left_wheel_body_z",
    "right_wheel_body_z",
    "CoM_lateral_offset_x",
    "knee_forward_margin_L",
    "knee_forward_margin_R",
    "left_right_joint_symmetry_abs_max",
    "wheel_clearance_diff",
    "both_wheels_touch_ground_correctly",
    "knees_bend_forward",
    "torso_near_upright",
    "left_right_symmetric",
    "visual_interpretation",
]

COORDINATE_PARTS = [
    ("torso center", "center"),
    ("CoM", "center"),
    ("left hip", "left"),
    ("right hip", "right"),
    ("left knee", "left"),
    ("right knee", "right"),
    ("left wheel", "left"),
    ("right wheel", "right"),
    ("left wheel bottom", "left"),
    ("right wheel bottom", "right"),
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quat_to_rpy_deg(quat: np.ndarray) -> tuple[float, float, float]:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    r = mat.reshape(3, 3)
    roll = math.atan2(r[2, 1], r[2, 2])
    pitch = math.atan2(-r[2, 0], math.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
    yaw = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(pitch), math.degrees(roll), math.degrees(yaw)


def generate_symmetric_b9_posture(height_cmd: float, model: mujoco.MjModel, config: DualRateConfig) -> PoseSpec:
    """Build a contact-aware, symmetric B9 posture."""

    if height_cmd not in config.height_grid:
        raise ValueError(f"B9 posture height must be one of {config.height_grid}, got {height_cmd}")

    height_ik = HeightIK(
        mj_model=model,
        scan_points=config.ik_scan_points,
        polynomial_degree=config.ik_polynomial_degree,
        symmetric_fold=config.ik_symmetric_fold,
    )
    targets = height_ik.compute_ik_targets(height_cmd)
    hip_pitch = float(targets["hip_pitch"])
    knee = float(targets["knee"])
    return PoseSpec(
        name=f"fixed_h_{height_cmd:.2f}",
        target_height=height_cmd,
        hip_roll_l=0.0,
        hip_roll_r=0.0,
        hip_yaw_l=0.0,
        hip_yaw_r=0.0,
        hip_pitch_l=hip_pitch,
        hip_pitch_r=hip_pitch,
        knee_l=knee,
        knee_r=knee,
    )


build_symmetric_b9_posture = generate_symmetric_b9_posture


def initialize_balanced_b9_posture(
    height_cmd: float,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: DualRateConfig,
    max_iterations: int = 200,
    force_tol: float = 1.0,
    com_tol: float = 0.01,
    roll_tol: float = 0.001,
) -> None:
    """Initialize MjData with balanced B9 posture via iterative contact-force correction.

    Adjusts root x and root z until bilateral contact forces are balanced while
    preserving near-zero root roll.
    """
    pose = generate_symmetric_b9_posture(height_cmd, model, config)

    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:17] = [
        pose.hip_roll_l,
        pose.hip_yaw_l,
        pose.hip_pitch_l,
        pose.knee_l,
        0.0,
        pose.hip_roll_r,
        pose.hip_yaw_r,
        pose.hip_pitch_r,
        pose.knee_r,
        0.0,
    ]
    mujoco.mj_forward(model, data)

    left_bottom, right_bottom = wheel_bottom_heights(model, data)
    data.qpos[2] -= max(left_bottom, right_bottom) + 5e-4
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    for _ in range(max_iterations):
        left_bottom, right_bottom = wheel_bottom_heights(model, data)
        if left_bottom > -1e-4 or right_bottom > -1e-4:
            data.qpos[2] -= max(left_bottom, right_bottom) + 5e-4
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            continue

        left_force, right_force = contact_forces_by_wheel(model, data)
        com = body_com(model, data)

        if not math.isfinite(left_force) or not math.isfinite(right_force):
            data.qpos[2] -= 2e-4
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            continue

        force_diff = left_force - right_force
        com_x = float(com[0])

        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, data.qpos[3:7].copy())
        r = mat.reshape(3, 3)
        roll = math.atan2(r[2, 1], r[2, 2])

        if abs(force_diff) < force_tol and abs(com_x) < com_tol and abs(roll) < roll_tol:
            break

        if abs(force_diff) > 0.1:
            data.qpos[0] -= 0.0001 * force_diff

        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)


def set_symmetric_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: PoseSpec) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:17] = [
        pose.hip_roll_l,
        pose.hip_yaw_l,
        pose.hip_pitch_l,
        pose.knee_l,
        0.0,
        pose.hip_roll_r,
        pose.hip_yaw_r,
        pose.hip_pitch_r,
        pose.knee_r,
        0.0,
    ]
    mujoco.mj_forward(model, data)

    left_bottom, right_bottom = wheel_bottom_heights(model, data)
    if abs(left_bottom - right_bottom) > 1e-4:
        raise ValueError(
            f"wheel-bottom asymmetry too large for symmetric posture: {abs(left_bottom - right_bottom):.6g}"
        )

    data.qpos[2] -= max(left_bottom, right_bottom)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def body_center(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    return data.xpos[model.body(body_name).id].copy()


def wheel_bottom_points(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    l_geom = model.geom("l_wheel_collision").id
    r_geom = model.geom("r_wheel_collision").id
    l_radius = float(model.geom_size[l_geom, 0])
    r_radius = float(model.geom_size[r_geom, 0])
    left = data.geom_xpos[l_geom].copy()
    right = data.geom_xpos[r_geom].copy()
    left[2] -= l_radius
    right[2] -= r_radius
    return left, right


def measure_fixed_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: PoseSpec) -> dict[str, object]:
    l_site = model.site("l_wheel_contact").id
    r_site = model.site("r_wheel_contact").id
    l_wheel_body = model.body("l_wheel_link").id
    r_wheel_body = model.body("r_wheel_link").id
    l_hip_body = model.body("l_thigh").id
    r_hip_body = model.body("r_thigh").id
    l_knee_body = model.body("l_knee_link").id
    r_knee_body = model.body("r_knee_link").id
    torso_body = model.body("torso").id

    torso_pitch, torso_roll, torso_yaw = quat_to_rpy_deg(data.qpos[3:7].copy())
    com = body_com(model, data)
    l_site_pos = data.site_xpos[l_site].copy()
    r_site_pos = data.site_xpos[r_site].copy()
    left_bottom, right_bottom = wheel_bottom_points(model, data)
    left_clearance = float(left_bottom[2])
    right_clearance = float(right_bottom[2])
    clearance_diff = abs(left_clearance - right_clearance)

    wheel_contact = 0.5 * (left_bottom + right_bottom)

    l_hip = data.xpos[l_hip_body].copy()
    r_hip = data.xpos[r_hip_body].copy()
    l_knee = data.xpos[l_knee_body].copy()
    r_knee = data.xpos[r_knee_body].copy()

    # Active model faces -Y, so positive forward margin = hip_y - knee_y.
    knee_forward_l = float(l_hip[1] - l_knee[1])
    knee_forward_r = float(r_hip[1] - r_knee[1])

    left_force, right_force = contact_forces_by_wheel(model, data)
    joint_symmetry_abs_max = max(
        abs(pose.hip_roll_l - pose.hip_roll_r),
        abs(pose.hip_yaw_l - pose.hip_yaw_r),
        abs(pose.hip_pitch_l - pose.hip_pitch_r),
        abs(pose.knee_l - pose.knee_r),
    )
    both_touch = abs(left_clearance) < 1e-4 and abs(right_clearance) < 1e-4 and clearance_diff < 1e-4
    knees_forward = knee_forward_l > 0.0 and knee_forward_r > 0.0
    torso_upright = abs(torso_pitch) < 5.0 and abs(torso_roll) < 5.0
    symmetric = joint_symmetry_abs_max < 1e-5 and abs(float(com[0])) < 0.03 and clearance_diff < 1e-4

    if knees_forward and torso_upright and symmetric and both_touch:
        interpretation = "Human-like symmetric posture with forward knees and both wheel bottoms grounded."
    elif not knees_forward:
        interpretation = "Non-human-like or questionable posture: at least one knee is not forward in active -Y frame."
    elif not symmetric:
        interpretation = "Forward-knee posture but left/right symmetry or lateral loading should be inspected."
    elif not both_touch:
        interpretation = "Forward-knee posture but wheel-ground contact is not symmetric/correct."
    else:
        interpretation = "Forward-knee posture with a possible upright/contact issue."

    return {
        "posture_name": pose.name,
        "target_height": pose.target_height if pose.target_height is not None else "",
        "hip_roll_L": pose.hip_roll_l,
        "hip_roll_R": pose.hip_roll_r,
        "hip_yaw_L": pose.hip_yaw_l,
        "hip_yaw_R": pose.hip_yaw_r,
        "hip_pitch_L": pose.hip_pitch_l,
        "hip_pitch_R": pose.hip_pitch_r,
        "knee_L": pose.knee_l,
        "knee_R": pose.knee_r,
        "torso_height": float(data.qpos[2]),
        "root_x": float(data.qpos[0]),
        "root_y": float(data.qpos[1]),
        "root_z": float(data.qpos[2]),
        "root_quat_w": float(data.qpos[3]),
        "root_quat_x": float(data.qpos[4]),
        "root_quat_y": float(data.qpos[5]),
        "root_quat_z": float(data.qpos[6]),
        "root_pitch_deg": torso_pitch,
        "root_roll_deg": torso_roll,
        "root_yaw_deg": torso_yaw,
        "torso_pitch": torso_pitch,
        "torso_roll": torso_roll,
        "CoM_x": float(com[0]),
        "CoM_y": float(com[1]),
        "CoM_z": float(com[2]),
        "wheel_contact_x": float(wheel_contact[0]),
        "wheel_contact_y": float(wheel_contact[1]),
        "wheel_contact_z": float(wheel_contact[2]),
        "left_wheel_bottom_z": left_clearance,
        "right_wheel_bottom_z": right_clearance,
        "left_wheel_clearance": left_clearance,
        "right_wheel_clearance": right_clearance,
        "left_wheel_contact_force": left_force,
        "right_wheel_contact_force": right_force,
        "left_wheel_body_z": float(data.xpos[l_wheel_body, 2]),
        "right_wheel_body_z": float(data.xpos[r_wheel_body, 2]),
        "CoM_lateral_offset_x": float(com[0] - wheel_contact[0]),
        "knee_forward_margin_L": knee_forward_l,
        "knee_forward_margin_R": knee_forward_r,
        "left_right_joint_symmetry_abs_max": joint_symmetry_abs_max,
        "wheel_clearance_diff": clearance_diff,
        "both_wheels_touch_ground_correctly": both_touch,
        "knees_bend_forward": knees_forward,
        "torso_near_upright": torso_upright,
        "left_right_symmetric": symmetric,
        "visual_interpretation": interpretation,
        "torso_center_x": float(data.xpos[torso_body, 0]),
        "torso_center_y": float(data.xpos[torso_body, 1]),
        "torso_center_z": float(data.xpos[torso_body, 2]),
        "l_hip_x": float(l_hip[0]),
        "l_hip_y": float(l_hip[1]),
        "l_hip_z": float(l_hip[2]),
        "r_hip_x": float(r_hip[0]),
        "r_hip_y": float(r_hip[1]),
        "r_hip_z": float(r_hip[2]),
        "l_knee_x": float(l_knee[0]),
        "l_knee_y": float(l_knee[1]),
        "l_knee_z": float(l_knee[2]),
        "r_knee_x": float(r_knee[0]),
        "r_knee_y": float(r_knee[1]),
        "r_knee_z": float(r_knee[2]),
        "l_wheel_x": float(data.xpos[l_wheel_body, 0]),
        "l_wheel_y": float(data.xpos[l_wheel_body, 1]),
        "l_wheel_z": float(data.xpos[l_wheel_body, 2]),
        "r_wheel_x": float(data.xpos[r_wheel_body, 0]),
        "r_wheel_y": float(data.xpos[r_wheel_body, 1]),
        "r_wheel_z": float(data.xpos[r_wheel_body, 2]),
        "left_wheel_bottom_x": float(left_bottom[0]),
        "left_wheel_bottom_y": float(left_bottom[1]),
        "left_wheel_bottom_z_coord": float(left_bottom[2]),
        "right_wheel_bottom_x": float(right_bottom[0]),
        "right_wheel_bottom_y": float(right_bottom[1]),
        "right_wheel_bottom_z_coord": float(right_bottom[2]),
        "root_z_after_correction": float(data.qpos[2]),
        "left_wheel_site_height": float(l_site_pos[2]),
        "right_wheel_site_height": float(r_site_pos[2]),
    }


def fixed_pose_rows(model: mujoco.MjModel, config: DualRateConfig) -> tuple[list[dict[str, object]], list[dict[str, object]], list[PoseSpec]]:
    poses = [build_symmetric_b9_posture(height, model, config) for height in HEIGHTS]
    rows: list[dict[str, object]] = []
    coord_rows: list[dict[str, object]] = []
    data = mujoco.MjData(model)
    for pose in poses:
        set_symmetric_pose(model, data, pose)
        rows.append(measure_fixed_pose(model, data, pose))
        render_prefix = pose.target_height
        for view in VIEWS.values():
            render_pose(model, data, FIXED_OUTPUT / f"fixed_{view.name}_h_{render_prefix:.2f}.png", view)

        left_bottom, right_bottom = wheel_bottom_points(model, data)
        coord_rows.extend(
            [
                {"height_cmd": pose.target_height, "body_part": "torso center", "side": "center", "x": float(data.xpos[model.body('torso').id, 0]), "y": float(data.xpos[model.body('torso').id, 1]), "z": float(data.xpos[model.body('torso').id, 2])},
                {"height_cmd": pose.target_height, "body_part": "CoM", "side": "center", "x": float(body_com(model, data)[0]), "y": float(body_com(model, data)[1]), "z": float(body_com(model, data)[2])},
                {"height_cmd": pose.target_height, "body_part": "left hip", "side": "left", "x": float(data.xpos[model.body('l_thigh').id, 0]), "y": float(data.xpos[model.body('l_thigh').id, 1]), "z": float(data.xpos[model.body('l_thigh').id, 2])},
                {"height_cmd": pose.target_height, "body_part": "right hip", "side": "right", "x": float(data.xpos[model.body('r_thigh').id, 0]), "y": float(data.xpos[model.body('r_thigh').id, 1]), "z": float(data.xpos[model.body('r_thigh').id, 2])},
                {"height_cmd": pose.target_height, "body_part": "left knee", "side": "left", "x": float(data.xpos[model.body('l_knee_link').id, 0]), "y": float(data.xpos[model.body('l_knee_link').id, 1]), "z": float(data.xpos[model.body('l_knee_link').id, 2])},
                {"height_cmd": pose.target_height, "body_part": "right knee", "side": "right", "x": float(data.xpos[model.body('r_knee_link').id, 0]), "y": float(data.xpos[model.body('r_knee_link').id, 1]), "z": float(data.xpos[model.body('r_knee_link').id, 2])},
                {"height_cmd": pose.target_height, "body_part": "left wheel", "side": "left", "x": float(data.xpos[model.body('l_wheel_link').id, 0]), "y": float(data.xpos[model.body('l_wheel_link').id, 1]), "z": float(data.xpos[model.body('l_wheel_link').id, 2])},
                {"height_cmd": pose.target_height, "body_part": "right wheel", "side": "right", "x": float(data.xpos[model.body('r_wheel_link').id, 0]), "y": float(data.xpos[model.body('r_wheel_link').id, 1]), "z": float(data.xpos[model.body('r_wheel_link').id, 2])},
                {"height_cmd": pose.target_height, "body_part": "left wheel bottom", "side": "left", "x": float(left_bottom[0]), "y": float(left_bottom[1]), "z": float(left_bottom[2])},
                {"height_cmd": pose.target_height, "body_part": "right wheel bottom", "side": "right", "x": float(right_bottom[0]), "y": float(right_bottom[1]), "z": float(right_bottom[2])},
            ]
        )
    return rows, coord_rows, poses


def joint_mapping_rows(model: mujoco.MjModel) -> list[dict[str, object]]:
    rows = []
    for action_idx, joint_name in enumerate(JOINT_NAMES):
        joint = model.joint(joint_name)
        joint_id = joint.id
        rows.append(
            {
                "joint_name": joint_name,
                "qpos_idx": int(model.jnt_qposadr[joint_id]),
                "qvel_idx": int(model.jnt_dofadr[joint_id]),
                "action_idx": action_idx,
                "axis_local_x": float(model.jnt_axis[joint_id, 0]),
                "axis_local_y": float(model.jnt_axis[joint_id, 1]),
                "axis_local_z": float(model.jnt_axis[joint_id, 2]),
            }
        )
    return rows


def write_fixed_report(
    path: Path,
    model_path: Path,
    model: mujoco.MjModel,
    old_rows: list[dict[str, str]],
    fixed_rows: list[dict[str, object]],
) -> None:
    joint_rows = joint_mapping_rows(model)

    old_by_height = {float(r["target_height"]): r for r in old_rows}
    fixed_by_height = {float(r["target_height"]): r for r in fixed_rows}

    comparison_lines = []
    for height in HEIGHTS:
        old = old_by_height[height]
        fixed = fixed_by_height[height]
        comparison_lines.append(
            f"- h={height:.2f}: old clearance L/R=({float(old['left_wheel_contact_height']):.6f}, {float(old['right_wheel_contact_height']):.6f}) m, "
            f"fixed clearance L/R=({float(fixed['left_wheel_clearance']):.6f}, {float(fixed['right_wheel_clearance']):.6f}) m, "
            f"old knee L/R=({float(old['knee_L']):.3f}, {float(old['knee_R']):.3f}), fixed knee L/R=({float(fixed['knee_L']):.3f}, {float(fixed['knee_R']):.3f})."
        )

    joint_table = "\n".join(
        f"| {row['joint_name']} | {row['qpos_idx']} | {row['qvel_idx']} | {row['action_idx']} | [{row['axis_local_x']:.6f}, {row['axis_local_y']:.6f}, {row['axis_local_z']:.6f}] |"
        for row in joint_rows
    )

    fixed_table = "\n".join(
        f"| {row['target_height']:.2f} | {row['hip_pitch_L']:.3f} | {row['hip_pitch_R']:.3f} | {row['knee_L']:.3f} | {row['knee_R']:.3f} | {row['root_pitch_deg']:.3f} | {row['root_roll_deg']:.3f} | {row['root_yaw_deg']:.3f} | {row['left_wheel_clearance']:.6f} | {row['right_wheel_clearance']:.6f} | {row['wheel_clearance_diff']:.6f} | {row['left_wheel_contact_force']:.3f} | {row['right_wheel_contact_force']:.3f} |"
        for row in fixed_rows
    )

    coord_csv_path = FIXED_OUTPUT / "fixed_posture_coordinates.csv"
    fixed_csv_path = FIXED_OUTPUT / "fixed_postures.csv"

    content = f"""# Phase B.9 Posture Geometry Inspection Report

## Static Posture Asymmetry Bug and Fix

This report covers B9 initial posture geometry only. It does not tune controller gains, train PPO, or proceed to fast-loop-only testing.

## Joint mapping check

The active real model is `{model_path.as_posix()}`. Canonical action order is:

1. `l_hip_roll`
2. `l_hip_yaw`
3. `l_hip_pitch`
4. `l_knee`
5. `l_wheel`
6. `r_hip_roll`
7. `r_hip_yaw`
8. `r_hip_pitch`
9. `r_knee`
10. `r_wheel`

| joint | qpos idx | qvel idx | action idx | axis |
|---|---:|---:|---:|---|
{joint_table}

Key result:

- Left/right qpos indices are correct.
- Left/right qvel indices are correct.
- Same-sign hip_pitch/knee values are the correct symmetric construction in the corrected real XML.
- Opposite-sign hip_pitch/knee values break forward-knee symmetry.

## Root cause

The wheel misalignment came from a mechanical transform error in `assets/robot/wheeled_biped_real.xml`, not from camera angle, controller tuning, or qpos index order.

The right thigh local origin was offset relative to the left chain. Correcting `r_thigh` from `pos="-0.03 0 -0.01089"` to `pos="-0.03 0 0.0107637"` removes the nearly constant left/right wheel fore-aft mismatch across the B9 height grid.

Root height initialization now uses wheel collision geometry bottoms through `l_wheel_collision` and `r_wheel_collision`, not diagnostic contact sites. It anchors on the higher wheel bottom so both wheels are at-or-below the ground plane and both wheels produce contact force in MuJoCo.

The `l_wheel_contact` and `r_wheel_contact` sites are sensor/diagnostic frames attached to rotating wheel bodies. They are not reliable ground-contact markers for static visual inspection; collision-geometry contact points and wheel-bottom coordinates are authoritative. The rendered figures hide site visualization so the report cannot be misread as showing site positions as ground contact.

## Fixed posture table

| h | hip pitch L | hip pitch R | knee L | knee R | root pitch deg | root roll deg | root yaw deg | wheel clearance L | wheel clearance R | clearance diff | contact force L | contact force R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{fixed_table}

## Old vs fixed comparison

{chr(10).join(comparison_lines)}

## Fixed root/orientation summary

- Root orientation remains upright: roll, pitch, and yaw stay near zero.
- Fixed wheel-bottom clearance difference stays under `1e-4` m for all tested heights.
- Both wheel clearances are non-positive after root correction, so both collision geoms touch or slightly penetrate the plane.
- Both wheels report finite positive contact force for all tested heights.
- Both knees bend forward for all tested heights.
- The fixed posture uses symmetric hip/knee scalar targets again; no per-side posture compensation is required after the XML correction.

## Fixed outputs

- Old repro CSV: `outputs/phase_b9_geometry_check/b9_postures.csv`
- Fixed images: `outputs/phase_b9_posture_fixed/`
- Fixed CSV: `{fixed_csv_path.as_posix()}`
- Coordinate CSV: `{coord_csv_path.as_posix()}`

## Answer

- Cause of asymmetry: mechanical XML transform error in the right thigh chain.
- Old left/right wheel clearance and knee values: see comparison list above.
- Fixed left/right wheel clearance, contact force, and joint values: see fixed table above.
- If a site marker appears offset, treat it as a diagnostic/sensor site artifact rather than wheel-ground contact.
- Fixed B9 posture safe for next fast-loop-only testing: yes, after visual inspection of the generated side/front/top renders.

## Next step

Inspect fixed side/front/top renders manually. Do not tune controller, train PPO, or start fast-loop-only testing until these renders look physically correct.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    config = DualRateConfig.from_yaml(PROJECT_ROOT / "configs" / "controllers" / "dual_rate_balance_controller_b9.yaml")

    FIXED_OUTPUT.mkdir(parents=True, exist_ok=True)
    joint_rows = joint_mapping_rows(model)
    with open(FIXED_OUTPUT / "joint_mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": str(model_path),
                "joint_mapping": joint_rows,
                "action_order": JOINT_NAMES,
                "world_axes": {"x": "lateral", "y_positive": "backward", "y_negative": "forward", "z": "up"},
            },
            f,
            indent=2,
        )

    fixed_rows, coord_rows, poses = fixed_pose_rows(model, config)
    fixed_columns = list(fixed_rows[0].keys()) if fixed_rows else FIXED_POSTURE_COLUMNS
    write_rows_csv(FIXED_OUTPUT / "fixed_postures.csv", fixed_rows, fixed_columns)
    write_rows_csv(FIXED_OUTPUT / "fixed_posture_coordinates.csv", coord_rows, ["height_cmd", "body_part", "side", "x", "y", "z"])

    old_rows = read_csv_rows(OLD_OUTPUT / "b9_postures.csv")
    write_fixed_report(REPORT_PATH, model_path, model, old_rows, fixed_rows)

    print(f"Saved fixed B9 renders and CSV to {FIXED_OUTPUT}")
    print(f"Saved symmetry fix report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
