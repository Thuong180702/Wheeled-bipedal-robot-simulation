#!/usr/bin/env python3
"""Validate balance-core controller across true standing-height variants.

This script generates true height variants using CoM-calibrated posture search
with explicit CoM centering and static-balance validity gates.

Key features:
1. Nominal variant uses validated keyframe baseline (not HeightIK regeneration)
2. Low/high variants search hip_pitch/knee pairs to achieve target CoM height
3. Each variant must satisfy CoM centering relative to support center
4. Each variant must satisfy static-balance validity gates
5. Each variant captures its own equilibrium references

This is NOT root-z-only perturbation. Each variant has different joint posture.

IMPORTANT: HeightIK uses torso/root height (qpos[2]), NOT CoM height.
Therefore HeightIK cannot be used directly with target_com_z.
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import mujoco
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.utils.config import get_model_path


@dataclass
class HeightVariantSetup:
    """Setup validation result for a height variant with CoM centering metrics."""
    variant_name: str
    target_com_z_m: float
    achieved_com_z_m: float
    height_error_m: float
    calibrated_root_z_m: float

    # Support center
    support_center_x: float
    support_center_y: float

    # CoM position and CoM-support error
    com_x_m: float
    com_y_m: float
    com_z_m: float
    com_support_error_x: float
    com_support_error_y: float
    com_support_error_norm_xy: float

    # Capture point
    cp_x_m: float
    cp_y_m: float
    cp_error_x_m: float
    cp_error_y_m: float

    # Joint references
    hip_pitch_ref_left: float
    hip_pitch_ref_right: float
    knee_ref_left: float
    knee_ref_right: float
    hip_roll_left: float
    hip_roll_right: float
    hip_yaw_left: float
    hip_yaw_right: float

    # Contact
    wheel_floor_contact_count: int
    left_wheel_contact: bool
    right_wheel_contact: bool
    min_wheel_contact_dist_m: float
    non_wheel_floor_contact_count: int

    # Orientation
    pitch_x_rad: float
    roll_y_rad: float
    yaw_z_rad: float

    # Validity
    joint_limit_valid: bool
    setup_valid: bool
    setup_failure_reason: str | None

    # Equilibrium references (captured only if valid)
    equilibrium_joint_pos: list[float] | None
    equilibrium_com_pos: list[float] | None
    equilibrium_pitch_x: float | None
    equilibrium_roll_y: float | None
    equilibrium_yaw_z: float | None

    # Method
    posture_search_method: str  # "keyframe_baseline" or "com_calibrated_search"


@dataclass
class NominalReference:
    """Nominal baseline reference for comparing variants."""
    support_center_x: float
    support_center_y: float
    com_support_error_x: float
    com_support_error_y: float
    com_support_error_norm_xy: float
    cp_x_m: float
    cp_y_m: float
    pitch_x_rad: float
    roll_y_rad: float
    yaw_z_rad: float


def compute_support_center(model, data):
    """Compute support center from wheel contact positions."""
    l_wheel_body_id = model.body("l_wheel_link").id
    r_wheel_body_id = model.body("r_wheel_link").id

    l_wheel_pos = data.xpos[l_wheel_body_id]
    r_wheel_pos = data.xpos[r_wheel_body_id]

    support_center_x = 0.5 * (l_wheel_pos[0] + r_wheel_pos[0])
    support_center_y = 0.5 * (l_wheel_pos[1] + r_wheel_pos[1])

    return float(support_center_x), float(support_center_y)


def compute_com_support_error(com_x, com_y, support_center_x, support_center_y):
    """Compute CoM projection error relative to support center."""
    error_x = com_x - support_center_x
    error_y = com_y - support_center_y
    error_norm = np.sqrt(error_x**2 + error_y**2)
    return float(error_x), float(error_y), float(error_norm)


def compute_capture_point_simple(com_x, com_y, com_z, com_vx, com_vy, g=9.81):
    """Compute capture point. At setup with zero velocity, CP should equal CoM projection."""
    omega = np.sqrt(g / com_z) if com_z > 0 else 0.0
    cp_x = com_x + (com_vx / omega if omega > 0 else 0.0)
    cp_y = com_y + (com_vy / omega if omega > 0 else 0.0)
    return float(cp_x), float(cp_y)


def compute_orientation_from_gravity_simple(model, data):
    """Compute body orientation from gravity vector in body frame."""
    torso_body_id = model.body("torso").id
    torso_xmat = data.xmat[torso_body_id].reshape(3, 3)
    gravity_world = np.array([0.0, 0.0, -1.0])
    gravity_body = torso_xmat.T @ gravity_world
    pitch_x = float(gravity_body[0])
    roll_y = float(gravity_body[1])
    return pitch_x, roll_y


def classify_floor_contacts_simple(model, data):
    """Classify floor contacts into wheel and non-wheel."""
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    left_wheel_contact = False
    right_wheel_contact = False
    non_wheel_floor_contacts = 0
    min_dist = None

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)

        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        if not involves_floor:
            continue

        if g1 == l_wheel_geom_id or g2 == l_wheel_geom_id:
            left_wheel_contact = True
            d = float(c.dist)
            min_dist = d if min_dist is None else min(min_dist, d)
        elif g1 == r_wheel_geom_id or g2 == r_wheel_geom_id:
            right_wheel_contact = True
            d = float(c.dist)
            min_dist = d if min_dist is None else min(min_dist, d)
        else:
            non_wheel_floor_contacts += 1

    return {
        "left_wheel_contact": left_wheel_contact,
        "right_wheel_contact": right_wheel_contact,
        "non_wheel_floor_contacts": non_wheel_floor_contacts,
        "min_wheel_contact_dist_m": min_dist if min_dist is not None else 0.0,
        "wheel_floor_contact_count": int(left_wheel_contact) + int(right_wheel_contact),
    }


def calibrate_root_z_for_wheel_floor_contact(model, data, target_dist=-5e-4, max_iters=20):
    """Calibrate root_z using geometric measurement of wheel bottom heights."""
    l_wheel_body_id = model.body("l_wheel_link").id
    r_wheel_body_id = model.body("r_wheel_link").id
    l_wheel_geom = model.geom("l_wheel_collision")
    wheel_radius = model.geom_size[l_wheel_geom.id][0]

    root_z_min = 0.30
    root_z_max = 0.70

    for iteration in range(max_iters):
        root_z_mid = (root_z_min + root_z_max) / 2.0
        data.qpos[2] = root_z_mid
        mujoco.mj_forward(model, data)

        l_wheel_pos_z = data.xpos[l_wheel_body_id, 2]
        r_wheel_pos_z = data.xpos[r_wheel_body_id, 2]
        l_wheel_bottom = l_wheel_pos_z - wheel_radius
        r_wheel_bottom = r_wheel_pos_z - wheel_radius
        avg_wheel_bottom = (l_wheel_bottom + r_wheel_bottom) / 2.0

        if avg_wheel_bottom > target_dist:
            root_z_max = root_z_mid
        else:
            root_z_min = root_z_mid

        if abs(avg_wheel_bottom - target_dist) < 1e-4:
            break

    return float(data.qpos[2])


def search_com_calibrated_posture_with_ranking(
    model: mujoco.MjModel,
    target_com_z_m: float,
    nominal_hip_pitch: float,
    nominal_knee: float,
    nominal_ref: NominalReference,
    search_range: float = 0.15,
    search_steps: int = 15,
    height_tolerance: float = 0.005,
    com_support_tolerance: float = 0.015,
    orientation_tolerance: float = 0.03,
) -> tuple[float, float, float, dict] | None:
    """Search for hip_pitch/knee achieving target CoM with multi-criteria ranking.

    Ranks candidates by:
    1. setup_valid (all gates passed)
    2. height error
    3. CoM support-centering error relative to nominal
    4. orientation error
    5. closeness to nominal joint posture

    Returns (hip_pitch, knee, achieved_com_z, metrics_dict) or None.
    """
    torso_id = model.body("torso").id
    candidates = []

    hip_pitch_values = np.linspace(
        nominal_hip_pitch - search_range,
        nominal_hip_pitch + search_range,
        search_steps
    )
    knee_values = np.linspace(
        nominal_knee - search_range,
        nominal_knee + search_range,
        search_steps
    )

    for hip_pitch in hip_pitch_values:
        for knee in knee_values:
            if hip_pitch < 0.0 or hip_pitch > 1.5:
                continue
            if knee < 0.0 or knee > 2.5:
                continue

            data = mujoco.MjData(model)
            if model.nkey > 0:
                mujoco.mj_resetDataKeyframe(model, data, 0)

            data.qpos[9] = hip_pitch
            data.qpos[10] = knee
            data.qpos[14] = hip_pitch
            data.qpos[15] = knee
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0

            try:
                calibrate_root_z_for_wheel_floor_contact(model, data)
                mujoco.mj_forward(model, data)
            except:
                continue

            # Measure all metrics
            achieved_com_z = float(data.subtree_com[torso_id][2])
            height_error = abs(achieved_com_z - target_com_z_m)

            support_center_x, support_center_y = compute_support_center(model, data)
            com_x = float(data.subtree_com[torso_id][0])
            com_y = float(data.subtree_com[torso_id][1])
            com_support_error_x, com_support_error_y, com_support_error_norm = compute_com_support_error(
                com_x, com_y, support_center_x, support_center_y
            )

            com_vx = float(data.subtree_linvel[torso_id][0])
            com_vy = float(data.subtree_linvel[torso_id][1])
            cp_x, cp_y = compute_capture_point_simple(com_x, com_y, achieved_com_z, com_vx, com_vy)

            contact_info = classify_floor_contacts_simple(model, data)
            pitch_x, roll_y = compute_orientation_from_gravity_simple(model, data)

            quat = data.qpos[3:7]
            yaw_z = 2.0 * np.arctan2(quat[3], quat[0])

            # Check validity gates
            valid = True
            failure_reasons = []

            if height_error >= height_tolerance:
                valid = False
                failure_reasons.append("height_error")

            if not (contact_info["left_wheel_contact"] and contact_info["right_wheel_contact"]):
                valid = False
                failure_reasons.append("missing_wheel_contact")

            if contact_info["non_wheel_floor_contacts"] > 0:
                valid = False
                failure_reasons.append("non_wheel_floor_contact")

            # CoM centering gate
            com_support_error_x_diff = abs(com_support_error_x - nominal_ref.com_support_error_x)
            com_support_error_y_diff = abs(com_support_error_y - nominal_ref.com_support_error_y)
            if com_support_error_x_diff > com_support_tolerance or com_support_error_y_diff > com_support_tolerance:
                valid = False
                failure_reasons.append("com_not_centered")

            # Orientation gate
            pitch_error = abs(pitch_x - nominal_ref.pitch_x_rad)
            roll_error = abs(roll_y - nominal_ref.roll_y_rad)
            yaw_error = abs(yaw_z - nominal_ref.yaw_z_rad)
            if pitch_error > orientation_tolerance or roll_error > orientation_tolerance or yaw_error > 0.05:
                valid = False
                failure_reasons.append("orientation_not_equilibrium")

            # Hip roll gate
            hip_roll_left = float(data.qpos[7])
            hip_roll_right = float(data.qpos[12])
            if abs(hip_roll_left) > 0.03 or abs(hip_roll_right) > 0.03:
                valid = False
                failure_reasons.append("hip_roll_excessive")

            # Compute ranking scores
            posture_distance = np.sqrt((hip_pitch - nominal_hip_pitch)**2 + (knee - nominal_knee)**2)

            candidates.append({
                "hip_pitch": hip_pitch,
                "knee": knee,
                "achieved_com_z": achieved_com_z,
                "height_error": height_error,
                "com_support_error_x_diff": com_support_error_x_diff,
                "com_support_error_y_diff": com_support_error_y_diff,
                "com_support_error_norm": com_support_error_norm,
                "pitch_error": pitch_error,
                "roll_error": roll_error,
                "yaw_error": yaw_error,
                "posture_distance": posture_distance,
                "valid": valid,
                "failure_reasons": failure_reasons,
                "metrics": {
                    "support_center_x": support_center_x,
                    "support_center_y": support_center_y,
                    "com_x": com_x,
                    "com_y": com_y,
                    "com_support_error_x": com_support_error_x,
                    "com_support_error_y": com_support_error_y,
                    "cp_x": cp_x,
                    "cp_y": cp_y,
                    "pitch_x": pitch_x,
                    "roll_y": roll_y,
                    "yaw_z": yaw_z,
                },
            })

    if not candidates:
        return None

    # Rank by: valid first, then height error, then CoM centering, then orientation, then posture distance
    candidates.sort(key=lambda c: (
        not c["valid"],  # Valid candidates first
        c["height_error"],
        c["com_support_error_x_diff"] + c["com_support_error_y_diff"],
        c["pitch_error"] + c["roll_error"],
        c["posture_distance"],
    ))

    best = candidates[0]
    return (best["hip_pitch"], best["knee"], best["achieved_com_z"], best["metrics"])

print(remaining_functions)
