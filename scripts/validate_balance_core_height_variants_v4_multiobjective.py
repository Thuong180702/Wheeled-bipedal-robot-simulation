#!/usr/bin/env python3
"""Validate balance-core controller across true standing-height variants.

This script generates true height variants using multi-objective CoM-calibrated posture search:
1. Nominal variant uses validated keyframe baseline (not HeightIK regeneration)
2. Low/high variants search hip_pitch/knee pairs with multi-objective scoring
3. Multi-objective scoring considers: height error, CoM centering, orientation, joint distance
4. Each variant calibrates root_z for wheel-floor contact
5. Each variant captures its own equilibrium references
6. Setup validity is checked with comprehensive gates before marking variant as ready

This is NOT root-z-only perturbation. Each variant has different joint posture.

IMPORTANT: HeightIK uses torso/root height (qpos[2]), NOT CoM height.
Therefore HeightIK cannot be used directly with target_com_z.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.utils.config import get_model_path


@dataclass
class CandidateStats:
    """Statistics for posture search candidates."""
    total_evaluated: int = 0
    passed_contact: int = 0
    passed_height: int = 0
    passed_com_centering: int = 0
    passed_orientation: int = 0
    passed_all: int = 0
    best_by_height: tuple[float, float, float, float] | None = None  # (hip_pitch, knee, height_error, score)
    best_by_com: tuple[float, float, float, float] | None = None  # (hip_pitch, knee, com_error, score)
    top_rejected: list[dict] = field(default_factory=list)  # Top 5 rejected candidates with reasons


@dataclass
class HeightVariantSetup:
    """Setup validation result for a height variant."""
    variant_name: str
    target_com_z_m: float
    achieved_com_z_m: float
    height_error_m: float
    calibrated_root_z_m: float
    hip_pitch_ref: float
    knee_ref: float
    hip_roll_left: float
    hip_roll_right: float
    hip_yaw_left: float
    hip_yaw_right: float
    # Support center and CoM centering fields
    support_center_x: float
    support_center_y: float
    com_x_m: float
    com_y_m: float
    com_support_error_x: float
    com_support_error_y: float
    com_support_error_norm_xy: float
    # Capture point fields
    cp_x_m: float
    cp_y_m: float
    cp_error_x_m: float
    cp_error_y_m: float
    # Contact and orientation
    wheel_floor_contact_count: int
    left_wheel_contact: bool
    right_wheel_contact: bool
    min_wheel_contact_dist_m: float
    non_wheel_floor_contact_count: int
    pitch_x_rad: float
    roll_y_rad: float
    yaw_z_rad: float
    joint_limit_valid: bool
    # Validity and failure tracking
    setup_valid: bool
    setup_failure_reason: str | None
    # Equilibrium references (only for valid setups)
    equilibrium_joint_pos: list[float] | None
    equilibrium_com_pos: list[float] | None
    equilibrium_pitch_x: float | None
    equilibrium_roll_y: float | None
    equilibrium_yaw_z: float | None
    # Metadata
    posture_search_method: str
    candidate_stats: CandidateStats | None = None


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


def search_com_calibrated_posture_multiobjective(
    model: mujoco.MjModel,
    target_com_z_m: float,
    nominal_hip_pitch: float,
    nominal_knee: float,
    nominal_com_support_error_x: float,
    nominal_com_support_error_y: float,
    search_range: float = 0.20,
    search_steps: int = 20,
) -> tuple[float, float, float, CandidateStats] | None:
    """Search for hip_pitch/knee using multi-objective scoring.

    Returns (hip_pitch, knee, achieved_com_z, stats) or None if no valid candidate found.

    Multi-objective scoring considers:
    - Height error (target vs achieved CoM Z)
    - CoM centering error (relative to nominal)
    - Orientation error (pitch, roll, yaw)
    - Joint distance from nominal
    """
    torso_id = model.body("torso").id

    # Weights for multi-objective scoring
    w_height = 100.0
    w_com_y = 50.0  # Sagittal CoM centering (most important for balance)
    w_com_x = 20.0  # Lateral CoM centering
    w_pitch = 10.0
    w_roll = 10.0
    w_yaw = 5.0
    w_joint = 1.0

    # Tolerances for gate checks
    height_tolerance = 0.005  # 5mm
    com_centering_tolerance_x = 0.015  # 15mm
    com_centering_tolerance_y = 0.015  # 15mm
    orientation_tolerance = 0.03  # ~1.7 degrees

    # Candidate tracking
    stats = CandidateStats()
    all_candidates = []

    # Search grid around nominal values
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
            stats.total_evaluated += 1

            # Skip if clearly out of reasonable range
            if hip_pitch < 0.0 or hip_pitch > 1.5:
                continue
            if knee < 0.0 or knee > 2.5:
                continue

            data = mujoco.MjData(model)
            if model.nkey > 0:
                mujoco.mj_resetDataKeyframe(model, data, 0)

            # Apply symmetric posture
            data.qpos[9] = hip_pitch
            data.qpos[10] = knee
            data.qpos[14] = hip_pitch
            data.qpos[15] = knee
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0

            # Calibrate root_z
            try:
                calibrate_root_z_for_wheel_floor_contact(model, data)
                mujoco.mj_forward(model, data)
            except:
                continue

            # Check contact validity (hard gate)
            contact_info = classify_floor_contacts_simple(model, data)
            if not (contact_info["left_wheel_contact"] and contact_info["right_wheel_contact"]):
                continue
            if contact_info["non_wheel_floor_contacts"] > 0:
                continue

            stats.passed_contact += 1

            # Measure achieved state
            achieved_com_z = float(data.subtree_com[torso_id][2])
            height_error = abs(achieved_com_z - target_com_z_m)

            # Compute support center and CoM support error
            support_center_x, support_center_y = compute_support_center(model, data)
            com_x = float(data.subtree_com[torso_id][0])
            com_y = float(data.subtree_com[torso_id][1])
            com_support_error_x, com_support_error_y, com_support_error_norm = compute_com_support_error(
                com_x, com_y, support_center_x, support_center_y
            )

            # Compute delta from nominal CoM support error
            delta_com_error_x = abs(com_support_error_x - nominal_com_support_error_x)
            delta_com_error_y = abs(com_support_error_y - nominal_com_support_error_y)

            # Compute orientation
            pitch_x, roll_y = compute_orientation_from_gravity_simple(model, data)
            quat = data.qpos[3:7]
            yaw_z = 2.0 * np.arctan2(quat[3], quat[0])

            # Compute joint distance from nominal
            joint_dist = np.sqrt((hip_pitch - nominal_hip_pitch)**2 + (knee - nominal_knee)**2)

            # Multi-objective score (lower is better)
            score = (
                w_height * height_error +
                w_com_y * delta_com_error_y +
                w_com_x * delta_com_error_x +
                w_pitch * abs(pitch_x) +
                w_roll * abs(roll_y) +
                w_yaw * abs(yaw_z) +
                w_joint * joint_dist
            )

            # Gate checks
            passed_height = height_error < height_tolerance
            passed_com_centering = (delta_com_error_x < com_centering_tolerance_x and
                                   delta_com_error_y < com_centering_tolerance_y)
            passed_orientation = (abs(pitch_x) < orientation_tolerance and
                                 abs(roll_y) < orientation_tolerance)

            if passed_height:
                stats.passed_height += 1
            if passed_com_centering:
                stats.passed_com_centering += 1
            if passed_orientation:
                stats.passed_orientation += 1

            # Track best candidates
            if stats.best_by_height is None or height_error < stats.best_by_height[2]:
                stats.best_by_height = (hip_pitch, knee, height_error, score)

            if stats.best_by_com is None or com_support_error_norm < stats.best_by_com[2]:
                stats.best_by_com = (hip_pitch, knee, com_support_error_norm, score)

            # Store candidate info
            candidate_info = {
                "hip_pitch": float(hip_pitch),
                "knee": float(knee),
                "height_error": float(height_error),
                "delta_com_error_x": float(delta_com_error_x),
                "delta_com_error_y": float(delta_com_error_y),
                "com_support_error_norm": float(com_support_error_norm),
                "pitch_x": float(pitch_x),
                "roll_y": float(roll_y),
                "score": float(score),
                "passed_height": passed_height,
                "passed_com_centering": passed_com_centering,
                "passed_orientation": passed_orientation,
                "passed_all": passed_height and passed_com_centering and passed_orientation,
            }

            all_candidates.append(candidate_info)

            if candidate_info["passed_all"]:
                stats.passed_all += 1

    # Sort candidates by score
    all_candidates.sort(key=lambda c: c["score"])

    # Find best valid candidate (passes all gates)
    valid_candidates = [c for c in all_candidates if c["passed_all"]]

    if valid_candidates:
        best = valid_candidates[0]
        return (best["hip_pitch"], best["knee"], target_com_z_m, stats)

    # No valid candidate found - store top 5 rejected with reasons
    for candidate in all_candidates[:5]:
        reasons = []
        if not candidate["passed_height"]:
            reasons.append(f"height_error={candidate['height_error']:.6f}m")
        if not candidate["passed_com_centering"]:
            reasons.append(f"com_not_centered (dx={candidate['delta_com_error_x']:.6f}m, dy={candidate['delta_com_error_y']:.6f}m)")
        if not candidate["passed_orientation"]:
            reasons.append(f"orientation_error (pitch={candidate['pitch_x']:.4f}, roll={candidate['roll_y']:.4f})")

        stats.top_rejected.append({
            "hip_pitch": candidate["hip_pitch"],
            "knee": candidate["knee"],
            "score": candidate["score"],
            "reasons": "; ".join(reasons),
        })

    return None


def generate_height_variant_setup(
    model: mujoco.MjModel,
    variant_name: str,
    target_com_z_m: float,
    nominal_hip_pitch: float,
    nominal_knee: float,
    nominal_com_support_error_x: float,
    nominal_com_support_error_y: float,
    use_keyframe_baseline: bool = False,
    tolerance_m: float = 0.005,
) -> HeightVariantSetup:
    """Generate and validate setup for a height variant.

    Args:
        use_keyframe_baseline: If True, use keyframe 0 as-is (for nominal).
                               If False, search for posture via multi-objective CoM calibration.
    """
    torso_id = model.body("torso").id

    if use_keyframe_baseline:
        # Nominal: use validated keyframe baseline
        data = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)

        data.qvel[:] = 0.0
        data.qacc[:] = 0.0

        # Calibrate root_z
        calibrated_root_z = calibrate_root_z_for_wheel_floor_contact(model, data)
        mujoco.mj_forward(model, data)

        hip_pitch = float(data.qpos[9])
        knee = float(data.qpos[10])
        posture_method = "keyframe_baseline"
        candidate_stats = None
    else:
        # Variant: search for posture via multi-objective CoM calibration
        search_result = search_com_calibrated_posture_multiobjective(
            model, target_com_z_m, nominal_hip_pitch, nominal_knee,
            nominal_com_support_error_x, nominal_com_support_error_y
        )

        if search_result is None:
            # Failed to find valid posture - return invalid setup with stats
            return HeightVariantSetup(
                variant_name=variant_name,
                target_com_z_m=target_com_z_m,
                achieved_com_z_m=0.0,
                height_error_m=999.0,
                calibrated_root_z_m=0.0,
                hip_pitch_ref=0.0,
                knee_ref=0.0,
                hip_roll_left=0.0,
                hip_roll_right=0.0,
                hip_yaw_left=0.0,
                hip_yaw_right=0.0,
                support_center_x=0.0,
                support_center_y=0.0,
                com_x_m=0.0,
                com_y_m=0.0,
                com_support_error_x=0.0,
                com_support_error_y=0.0,
                com_support_error_norm_xy=0.0,
                cp_x_m=0.0,
                cp_y_m=0.0,
                cp_error_x_m=0.0,
                cp_error_y_m=0.0,
                wheel_floor_contact_count=0,
                left_wheel_contact=False,
                right_wheel_contact=False,
                min_wheel_contact_dist_m=0.0,
                non_wheel_floor_contact_count=0,
                pitch_x_rad=0.0,
                roll_y_rad=0.0,
                yaw_z_rad=0.0,
                joint_limit_valid=False,
                setup_valid=False,
                setup_failure_reason="no_com_centered_solution_found",
                equilibrium_joint_pos=None,
                equilibrium_com_pos=None,
                equilibrium_pitch_x=None,
                equilibrium_roll_y=None,
                equilibrium_yaw_z=None,
                posture_search_method="multiobjective_search_failed",
                candidate_stats=None,
            )

        hip_pitch, knee, _, candidate_stats = search_result

        # Recreate with found posture
        data = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)

        data.qpos[9] = hip_pitch
        data.qpos[10] = knee
        data.qpos[14] = hip_pitch
        data.qpos[15] = knee
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0

        calibrated_root_z = calibrate_root_z_for_wheel_floor_contact(model, data)
        mujoco.mj_forward(model, data)
        posture_method = "multiobjective_com_calibrated_search"

    # Measure achieved state
    achieved_com_z = float(data.subtree_com[torso_id][2])
    height_error = abs(achieved_com_z - target_com_z_m)

    contact_info = classify_floor_contacts_simple(model, data)
    pitch_x, roll_y = compute_orientation_from_gravity_simple(model, data)

    quat = data.qpos[3:7]
    yaw_z = 2.0 * np.arctan2(quat[3], quat[0])

    joint_pos = data.qpos[7:17]
    joint_limit_valid = not np.any(np.abs(joint_pos) > 3.5)

    # Compute support center and CoM support error
    support_center_x, support_center_y = compute_support_center(model, data)
    com_x = float(data.subtree_com[torso_id][0])
    com_y = float(data.subtree_com[torso_id][1])
    com_support_error_x, com_support_error_y, com_support_error_norm = compute_com_support_error(
        com_x, com_y, support_center_x, support_center_y
    )

    # Compute capture point (simplified: CP = CoM for zero velocity)
    cp_x = com_x
    cp_y = com_y
    cp_error_x = com_x - nominal_com_support_error_x
    cp_error_y = com_y - nominal_com_support_error_y

    # Validate setup
    setup_valid = True
    failure_reasons = []

    if height_error >= tolerance_m:
        setup_valid = False
        failure_reasons.append(f"height_error={height_error:.6f}m")

    if not (contact_info["left_wheel_contact"] and contact_info["right_wheel_contact"]):
        setup_valid = False
        failure_reasons.append("missing_wheel_contact")

    if contact_info["non_wheel_floor_contacts"] > 0:
        setup_valid = False
        failure_reasons.append(f"non_wheel_floor_contacts={contact_info['non_wheel_floor_contacts']}")

    if abs(pitch_x) > 0.03 or abs(roll_y) > 0.03:
        setup_valid = False
        failure_reasons.append("orientation_not_equilibrium")

    if abs(data.qpos[7]) > 0.03 or abs(data.qpos[12]) > 0.03:
        setup_valid = False
        failure_reasons.append("hip_roll_not_nominal")

    if not joint_limit_valid:
        setup_valid = False
        failure_reasons.append("joint_limit_violation")

    # Capture equilibrium if valid
    equilibrium_joint_pos = None
    equilibrium_com_pos = None
    equilibrium_pitch_x = None
    equilibrium_roll_y = None
    equilibrium_yaw_z = None

    if setup_valid:
        equilibrium_joint_pos = joint_pos.tolist()
        equilibrium_com_pos = data.subtree_com[torso_id].tolist()
        equilibrium_pitch_x = float(pitch_x)
        equilibrium_roll_y = float(roll_y)
        equilibrium_yaw_z = float(yaw_z)

    return HeightVariantSetup(
        variant_name=variant_name,
        target_com_z_m=target_com_z_m,
        achieved_com_z_m=achieved_com_z,
        height_error_m=height_error,
        calibrated_root_z_m=calibrated_root_z,
        hip_pitch_ref=hip_pitch,
        knee_ref=knee,
        hip_roll_left=float(data.qpos[7]),
        hip_roll_right=float(data.qpos[12]),
        hip_yaw_left=float(data.qpos[8]),
        hip_yaw_right=float(data.qpos[13]),
        support_center_x=support_center_x,
        support_center_y=support_center_y,
        com_x_m=com_x,
        com_y_m=com_y,
        com_support_error_x=com_support_error_x,
        com_support_error_y=com_support_error_y,
        com_support_error_norm_xy=com_support_error_norm,
        cp_x_m=cp_x,
        cp_y_m=cp_y,
        cp_error_x_m=cp_error_x,
        cp_error_y_m=cp_error_y,
        wheel_floor_contact_count=contact_info["wheel_floor_contact_count"],
        left_wheel_contact=contact_info["left_wheel_contact"],
        right_wheel_contact=contact_info["right_wheel_contact"],
        min_wheel_contact_dist_m=contact_info["min_wheel_contact_dist_m"],
        non_wheel_floor_contact_count=contact_info["non_wheel_floor_contacts"],
        pitch_x_rad=float(pitch_x),
        roll_y_rad=float(roll_y),
        yaw_z_rad=float(yaw_z),
        joint_limit_valid=joint_limit_valid,
        setup_valid=setup_valid,
        setup_failure_reason="; ".join(failure_reasons) if failure_reasons else None,
        equilibrium_joint_pos=equilibrium_joint_pos,
        equilibrium_com_pos=equilibrium_com_pos,
        equilibrium_pitch_x=equilibrium_pitch_x,
        equilibrium_roll_y=equilibrium_roll_y,
        equilibrium_yaw_z=equilibrium_yaw_z,
        posture_search_method=posture_method,
        candidate_stats=candidate_stats,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate balance-core across true height variants using multi-objective CoM-calibrated posture search"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/balance_core_true_height_variants",
        help="Output directory for setup reports",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))

    print("=== Balance-Core True Height Variant Setup Validation (Multi-Objective) ===")
    print()

    # Measure nominal from keyframe
    print("--- Generating Nominal Baseline ---")
    data_nominal = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data_nominal, 0)
    data_nominal.qvel[:] = 0.0
    data_nominal.qacc[:] = 0.0
    calibrate_root_z_for_wheel_floor_contact(model, data_nominal)
    mujoco.mj_forward(model, data_nominal)

    torso_id = model.body("torso").id
    nominal_com_z = float(data_nominal.subtree_com[torso_id][2])
    nominal_hip_pitch = float(data_nominal.qpos[9])
    nominal_knee = float(data_nominal.qpos[10])

    print(f"Nominal CoM: {nominal_com_z:.6f} m")
    print(f"Nominal hip_pitch: {nominal_hip_pitch:.4f} rad, knee: {nominal_knee:.4f} rad")
    print()

    # Capture nominal reference values for CoM centering comparison
    nominal_support_center_x, nominal_support_center_y = compute_support_center(model, data_nominal)
    nominal_com_x = float(data_nominal.subtree_com[torso_id][0])
    nominal_com_y = float(data_nominal.subtree_com[torso_id][1])
    nominal_com_support_error_x, nominal_com_support_error_y, nominal_com_support_error_norm = compute_com_support_error(
        nominal_com_x, nominal_com_y, nominal_support_center_x, nominal_support_center_y
    )

    print(f"Nominal support center: ({nominal_support_center_x:.6f}, {nominal_support_center_y:.6f}) m")
    print(f"Nominal CoM support error: ({nominal_com_support_error_x:.6f}, {nominal_com_support_error_y:.6f}) m")
    print(f"Nominal CoM support error norm: {nominal_com_support_error_norm:.6f} m")
    print()

    # Define granular variants
    variants = [
        ("nominal", nominal_com_z, True),
        ("high_tiny", nominal_com_z + 0.005, False),
        ("high_small", nominal_com_z + 0.010, False),
        ("low_tiny", nominal_com_z - 0.005, False),
        ("low_small", nominal_com_z - 0.010, False),
    ]

    setup_results = []
    valid_variants = []
    invalid_variants = []

    # Generate each variant
    for variant_name, target_height, use_keyframe in variants:
        print(f"--- {variant_name} (target={target_height:.6f}m) ---")

        setup = generate_height_variant_setup(
            model, variant_name, target_height,
            nominal_hip_pitch, nominal_knee,
            nominal_com_support_error_x, nominal_com_support_error_y,
            use_keyframe, tolerance_m=0.005
        )

        setup_results.append(setup)

        if setup.setup_valid:
            valid_variants.append(variant_name)
            print(f"  [VALID]")
        else:
            invalid_variants.append(variant_name)
            print(f"  [INVALID]: {setup.setup_failure_reason}")

        print(f"  Achieved: {setup.achieved_com_z_m:.6f} m, error: {setup.height_error_m:.6f} m")
        print(f"  CoM support error: ({setup.com_support_error_x:.6f}, {setup.com_support_error_y:.6f}) m")
        print(f"  Method: {setup.posture_search_method}")

        # Print candidate statistics if available
        if setup.candidate_stats:
            stats = setup.candidate_stats
            print(f"  Candidates evaluated: {stats.total_evaluated}")
            print(f"    Passed contact: {stats.passed_contact}")
            print(f"    Passed height: {stats.passed_height}")
            print(f"    Passed CoM centering: {stats.passed_com_centering}")
            print(f"    Passed orientation: {stats.passed_orientation}")
            print(f"    Passed all gates: {stats.passed_all}")

            if stats.top_rejected:
                print(f"  Top rejected candidates:")
                for i, rej in enumerate(stats.top_rejected[:3], 1):
                    print(f"    {i}. hip_pitch={rej['hip_pitch']:.4f}, knee={rej['knee']:.4f}")
                    print(f"       Reasons: {rej['reasons']}")

        print()

    # Write reports
    height_ik_audit = {
        "height_ik_metric": "torso/root height (qpos[2])",
        "height_ik_metric_is_not_com_height": True,
        "warning": "HeightIK cannot be used directly with target_com_z",
    }

    # Serialize candidate stats for JSON
    def serialize_stats(stats):
        if stats is None:
            return None
        return {
            "total_evaluated": stats.total_evaluated,
            "passed_contact": stats.passed_contact,
            "passed_height": stats.passed_height,
            "passed_com_centering": stats.passed_com_centering,
            "passed_orientation": stats.passed_orientation,
            "passed_all": stats.passed_all,
            "best_by_height": stats.best_by_height,
            "best_by_com": stats.best_by_com,
            "top_rejected": stats.top_rejected,
        }

    json_report = {
        "height_ik_metric_audit": height_ik_audit,
        "search_method": "multiobjective_com_calibrated",
        "search_weights": {
            "w_height": 100.0,
            "w_com_y": 50.0,
            "w_com_x": 20.0,
            "w_pitch": 10.0,
            "w_roll": 10.0,
            "w_yaw": 5.0,
            "w_joint": 1.0,
        },
        "nominal_com_z_m": nominal_com_z,
        "nominal_support_center_x": nominal_support_center_x,
        "nominal_support_center_y": nominal_support_center_y,
        "nominal_com_support_error_x": nominal_com_support_error_x,
        "nominal_com_support_error_y": nominal_com_support_error_y,
        "nominal_com_support_error_norm": nominal_com_support_error_norm,
        "com_centering_gate_enforced": True,
        "static_balance_gate_enforced": True,
        "nominal_reference_comparison_enforced": True,
        "valid_variants": valid_variants,
        "invalid_variants": invalid_variants,
        "setup_results": [
            {
                **{k: v for k, v in vars(s).items() if k != "candidate_stats"},
                "candidate_stats": serialize_stats(s.candidate_stats),
            }
            for s in setup_results
        ],
        "ready_for_b5_b10": len(valid_variants) >= 2,
    }

    json_path = output_dir / "true_height_variant_setup_report.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    # Write markdown report
    md_lines = [
        "# Balance-Core True Height Variant Setup Report (Multi-Objective Search)",
        "",
        "## Search Method",
        "",
        "**Multi-objective CoM-calibrated posture search** with scoring weights:",
        "- Height error: 100.0",
        "- CoM Y centering (sagittal): 50.0",
        "- CoM X centering (lateral): 20.0",
        "- Pitch error: 10.0",
        "- Roll error: 10.0",
        "- Yaw error: 5.0",
        "- Joint distance from nominal: 1.0",
        "",
        "## HeightIK Metric Audit",
        "",
        "**CRITICAL**: HeightIK uses torso/root height (qpos[2]), NOT CoM height.",
        "Cannot be used directly with target_com_z. Use multi-objective CoM-calibrated search instead.",
        "",
        "## Gate Enforcement Status",
        "",
        f"- **CoM centering gate**: {'✓ ENFORCED' if json_report['com_centering_gate_enforced'] else '✗ NOT ENFORCED'}",
        f"- **Static-balance gate**: {'✓ ENFORCED' if json_report['static_balance_gate_enforced'] else '✗ NOT ENFORCED'}",
        f"- **Nominal reference comparison**: {'✓ ENFORCED' if json_report['nominal_reference_comparison_enforced'] else '✗ NOT ENFORCED'}",
        "",
        "## Summary",
        "",
        f"- **Nominal CoM**: {nominal_com_z:.6f} m",
        f"- **Nominal support center**: ({nominal_support_center_x:.6f}, {nominal_support_center_y:.6f}) m",
        f"- **Nominal CoM support error**: ({nominal_com_support_error_x:.6f}, {nominal_com_support_error_y:.6f}) m",
        f"- **Valid variants**: {len(valid_variants)}/{len(variants)} ({', '.join(valid_variants) if valid_variants else 'none'})",
        f"- **Invalid variants**: {len(invalid_variants)}/{len(variants)} ({', '.join(invalid_variants) if invalid_variants else 'none'})",
        f"- **Ready for B5-B10**: {'yes' if json_report['ready_for_b5_b10'] else 'no'}",
        "",
        "## Variant Details",
        "",
    ]

    for s in setup_results:
        status = "✓ VALID" if s.setup_valid else f"✗ INVALID: {s.setup_failure_reason}"
        md_lines.extend([
            f"### {s.variant_name} (target={s.target_com_z_m:.6f}m)",
            "",
            f"**Status**: {status}",
            "",
            "**Height**:",
            f"- Target CoM Z: {s.target_com_z_m:.6f} m",
            f"- Achieved CoM Z: {s.achieved_com_z_m:.6f} m",
            f"- Height error: {s.height_error_m:.6f} m",
            f"- Calibrated root Z: {s.calibrated_root_z_m:.6f} m",
            "",
            "**Posture**:",
            f"- Method: {s.posture_search_method}",
            f"- Hip pitch: {s.hip_pitch_ref:.4f} rad ({s.hip_pitch_ref * 57.3:.1f}°)",
            f"- Knee: {s.knee_ref:.4f} rad ({s.knee_ref * 57.3:.1f}°)",
            f"- Hip roll (L/R): {s.hip_roll_left:.4f} / {s.hip_roll_right:.4f} rad",
            f"- Hip yaw (L/R): {s.hip_yaw_left:.4f} / {s.hip_yaw_right:.4f} rad",
            "",
            "**CoM Centering**:",
            f"- Support center: ({s.support_center_x:.6f}, {s.support_center_y:.6f}) m",
            f"- CoM position: ({s.com_x_m:.6f}, {s.com_y_m:.6f}, {s.achieved_com_z_m:.6f}) m",
            f"- CoM support error: ({s.com_support_error_x:.6f}, {s.com_support_error_y:.6f}) m",
            f"- CoM support error norm: {s.com_support_error_norm_xy:.6f} m",
            "",
            "**Orientation**:",
            f"- Pitch X: {s.pitch_x_rad:.4f} rad ({s.pitch_x_rad * 57.3:.1f}°)",
            f"- Roll Y: {s.roll_y_rad:.4f} rad ({s.roll_y_rad * 57.3:.1f}°)",
            f"- Yaw Z: {s.yaw_z_rad:.4f} rad ({s.yaw_z_rad * 57.3:.1f}°)",
            "",
            "**Contact**:",
            f"- Wheel floor contacts: {s.wheel_floor_contact_count}",
            f"- Left wheel contact: {s.left_wheel_contact}",
            f"- Right wheel contact: {s.right_wheel_contact}",
            f"- Non-wheel floor contacts: {s.non_wheel_floor_contact_count}",
            "",
        ])

        # Add candidate statistics if available
        if s.candidate_stats:
            stats = s.candidate_stats
            md_lines.extend([
                "**Candidate Search Statistics**:",
                f"- Total evaluated: {stats.total_evaluated}",
                f"- Passed contact gate: {stats.passed_contact}",
                f"- Passed height gate: {stats.passed_height}",
                f"- Passed CoM centering gate: {stats.passed_com_centering}",
                f"- Passed orientation gate: {stats.passed_orientation}",
                f"- Passed all gates: {stats.passed_all}",
                "",
            ])

            if stats.best_by_height:
                h = stats.best_by_height
                md_lines.append(f"- Best by height: hip_pitch={h[0]:.4f}, knee={h[1]:.4f}, error={h[2]:.6f}m")

            if stats.best_by_com:
                c = stats.best_by_com
                md_lines.append(f"- Best by CoM centering: hip_pitch={c[0]:.4f}, knee={c[1]:.4f}, error={c[2]:.6f}m")

            if stats.top_rejected:
                md_lines.extend([
                    "",
                    "**Top Rejected Candidates**:",
                ])
                for i, rej in enumerate(stats.top_rejected, 1):
                    md_lines.extend([
                        f"{i}. hip_pitch={rej['hip_pitch']:.4f}, knee={rej['knee']:.4f}, score={rej['score']:.2f}",
                        f"   - Reasons: {rej['reasons']}",
                    ])

            md_lines.append("")

        if s.setup_valid and s.equilibrium_joint_pos is not None:
            md_lines.extend([
                "**Equilibrium References** (captured):",
                f"- Joint pos: {[f'{x:.4f}' for x in s.equilibrium_joint_pos]}",
                f"- CoM pos: [{s.equilibrium_com_pos[0]:.6f}, {s.equilibrium_com_pos[1]:.6f}, {s.equilibrium_com_pos[2]:.6f}] m",
                f"- Pitch X: {s.equilibrium_pitch_x:.4f} rad",
                f"- Roll Y: {s.equilibrium_roll_y:.4f} rad",
                f"- Yaw Z: {s.equilibrium_yaw_z:.4f} rad",
                "",
            ])

    md_path = output_dir / "true_height_variant_setup_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Reports: {json_path}, {md_path}")
    print(f"Valid: {len(valid_variants)}/{len(variants)}")
    print("Ready for review. Do not proceed to B5-B10 until approved.")


if __name__ == "__main__":
    main()
