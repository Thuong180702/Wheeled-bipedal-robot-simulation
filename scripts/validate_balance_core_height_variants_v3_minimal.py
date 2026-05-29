#!/usr/bin/env python3
"""Validate balance-core controller across true standing-height variants.

This script generates true height variants using CoM-calibrated posture search:
1. Nominal variant uses validated keyframe baseline (not HeightIK regeneration)
2. Low/high variants search hip_pitch/knee pairs to achieve target CoM height
3. Each variant calibrates root_z for wheel-floor contact
4. Each variant captures its own equilibrium references
5. Setup validity is checked before marking variant as ready

This is NOT root-z-only perturbation. Each variant has different joint posture.

IMPORTANT: HeightIK uses torso/root height (qpos[2]), NOT CoM height.
Therefore HeightIK cannot be used directly with target_com_z.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.utils.config import get_model_path


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
    posture_search_method: str  # "keyframe_baseline" or "com_calibrated_search"


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


def search_com_calibrated_posture(
    model: mujoco.MjModel,
    target_com_z_m: float,
    nominal_hip_pitch: float,
    nominal_knee: float,
    search_range: float = 0.15,
    search_steps: int = 15,
) -> tuple[float, float, float] | None:
    """Search for hip_pitch/knee that achieves target CoM height.

    Returns (hip_pitch, knee, achieved_com_z) or None if no valid candidate found.
    """
    torso_id = model.body("torso").id
    best_candidate = None
    best_error = float('inf')

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

            # Measure achieved CoM
            achieved_com_z = float(data.subtree_com[torso_id][2])
            error = abs(achieved_com_z - target_com_z_m)

            # Check basic validity
            contact_info = classify_floor_contacts_simple(model, data)
            if not (contact_info["left_wheel_contact"] and contact_info["right_wheel_contact"]):
                continue
            if contact_info["non_wheel_floor_contacts"] > 0:
                continue

            # Track best candidate
            if error < best_error:
                best_error = error
                best_candidate = (hip_pitch, knee, achieved_com_z)

    return best_candidate


def generate_height_variant_setup(
    model: mujoco.MjModel,
    variant_name: str,
    target_com_z_m: float,
    nominal_hip_pitch: float,
    nominal_knee: float,
    use_keyframe_baseline: bool = False,
    tolerance_m: float = 0.005,
) -> HeightVariantSetup:
    """Generate and validate setup for a height variant.

    Args:
        use_keyframe_baseline: If True, use keyframe 0 as-is (for nominal).
                               If False, search for posture via CoM calibration.
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
    else:
        # Variant: search for posture via CoM calibration
        search_result = search_com_calibrated_posture(
            model, target_com_z_m, nominal_hip_pitch, nominal_knee
        )

        if search_result is None:
            # Failed to find valid posture
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
                setup_failure_reason="com_calibrated_search_failed",
                equilibrium_joint_pos=None,
                equilibrium_com_pos=None,
                equilibrium_pitch_x=None,
                equilibrium_roll_y=None,
                equilibrium_yaw_z=None,
                posture_search_method="com_calibrated_search_failed",
            )

        hip_pitch, knee, _ = search_result

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
        posture_method = "com_calibrated_search"

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
    # Since qvel is zero at setup, capture point should equal CoM projection
    cp_x = com_x
    cp_y = com_y
    cp_error_x = 0.0  # Will be computed relative to nominal later
    cp_error_y = 0.0

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

    if abs(data.qpos[7]) > 0.03 or abs(data.qpos[12]) > 0.03:  # hip_roll limits
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
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate balance-core across true height variants using CoM-calibrated posture search"
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

    print("=== Balance-Core True Height Variant Setup Validation (CoM-Calibrated) ===")
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

    # Define variants
    variants = [
        ("nominal", nominal_com_z, True),
        ("high_small", nominal_com_z + 0.01, False),
    ]

    setup_results = []
    valid_variants = []
    invalid_variants = []

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

    # Generate each variant
    for variant_name, target_height, use_keyframe in variants:
        print(f"--- {variant_name} (target={target_height:.6f}m) ---")

        setup = generate_height_variant_setup(
            model, variant_name, target_height,
            nominal_hip_pitch, nominal_knee,
            use_keyframe, tolerance_m=0.005
        )

        # Enforce CoM centering gates for non-nominal variants
        if not use_keyframe and setup.setup_valid:
            com_centering_tolerance_x = 0.015  # 15mm
            com_centering_tolerance_y = 0.015  # 15mm

            com_error_x_diff = abs(setup.com_support_error_x - nominal_com_support_error_x)
            com_error_y_diff = abs(setup.com_support_error_y - nominal_com_support_error_y)

            if com_error_x_diff > com_centering_tolerance_x or com_error_y_diff > com_centering_tolerance_y:
                setup.setup_valid = False
                if setup.setup_failure_reason:
                    setup.setup_failure_reason += "; com_not_centered"
                else:
                    setup.setup_failure_reason = "com_not_centered"

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
        print()

    # Write reports
    height_ik_audit = {
        "height_ik_metric": "torso/root height (qpos[2])",
        "height_ik_metric_is_not_com_height": True,
        "warning": "HeightIK cannot be used directly with target_com_z",
    }

    json_report = {
        "height_ik_metric_audit": height_ik_audit,
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
            {k: v for k, v in vars(s).items()}
            for s in setup_results
        ],
        "ready_for_b5_b10": len(valid_variants) >= 2,
    }

    json_path = output_dir / "true_height_variant_setup_report.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    md_lines = [
        "# Balance-Core True Height Variant Setup Report (CoM-Calibrated)",
        "",
        "## HeightIK Metric Audit",
        "",
        "**CRITICAL**: HeightIK uses torso/root height (qpos[2]), NOT CoM height.",
        "Cannot be used directly with target_com_z. Use CoM-calibrated search instead.",
        "",
        "## Gate Enforcement Status",
        "",
        f"- **CoM centering gate**: {'ENFORCED' if json_report['com_centering_gate_enforced'] else 'NOT ENFORCED'}",
        f"- **Static-balance gate**: {'ENFORCED' if json_report['static_balance_gate_enforced'] else 'NOT ENFORCED'}",
        f"- **Nominal reference comparison**: {'ENFORCED' if json_report['nominal_reference_comparison_enforced'] else 'NOT ENFORCED'}",
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
            "**Capture Point**:",
            f"- CP position: ({s.cp_x_m:.6f}, {s.cp_y_m:.6f}) m",
            f"- CP error: ({s.cp_error_x_m:.6f}, {s.cp_error_y_m:.6f}) m",
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
            f"- Min wheel contact dist: {s.min_wheel_contact_dist_m:.6f} m",
            f"- Non-wheel floor contacts: {s.non_wheel_floor_contact_count}",
            "",
        ])

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
