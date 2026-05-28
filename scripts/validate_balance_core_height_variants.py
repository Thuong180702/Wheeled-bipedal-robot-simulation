#!/usr/bin/env python3
"""Validate balance-core controller across true standing-height variants.

This script generates true height variants by:
1. Using HeightIK to compute hip_pitch/knee for target CoM height
2. Calibrating root_z for wheel-floor contact
3. Capturing per-variant equilibrium references
4. Validating setup before simulation

This is NOT root-z-only perturbation. Each variant has different joint posture.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.controllers.height_ik import HeightIK
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
    wheel_floor_contact_count: int
    left_wheel_contact: bool
    right_wheel_contact: bool
    min_wheel_contact_dist_m: float
    non_wheel_floor_contact_count: int
    pitch_x_rad: float
    roll_y_rad: float
    yaw_z_rad: float
    joint_limit_valid: bool
    setup_valid: bool
    setup_failure_reason: str | None
    equilibrium_joint_pos: list[float] | None
    equilibrium_com_pos: list[float] | None
    equilibrium_pitch_x: float | None
    equilibrium_roll_y: float | None
    equilibrium_yaw_z: float | None


def compute_orientation_from_gravity_simple(model, data):
    """Compute body orientation from gravity vector in body frame."""
    torso_body_id = model.body("torso").id
    # Get rotation matrix for torso body
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
    """Calibrate root_z to achieve target wheel-floor contact distance.

    Uses geometric measurement of wheel bottom heights rather than contact detection.
    """
    # Get wheel body IDs and wheel radius
    l_wheel_body_id = model.body("l_wheel_link").id
    r_wheel_body_id = model.body("r_wheel_link").id
    l_wheel_geom = model.geom("l_wheel_collision")
    wheel_radius = model.geom_size[l_wheel_geom.id][0]

    # Binary search for root_z that puts wheels at target distance from ground
    root_z_min = 0.30
    root_z_max = 0.70

    for iteration in range(max_iters):
        root_z_mid = (root_z_min + root_z_max) / 2.0
        data.qpos[2] = root_z_mid
        mujoco.mj_forward(model, data)

        # Measure wheel bottom heights geometrically
        l_wheel_pos_z = data.xpos[l_wheel_body_id, 2]
        r_wheel_pos_z = data.xpos[r_wheel_body_id, 2]
        l_wheel_bottom = l_wheel_pos_z - wheel_radius
        r_wheel_bottom = r_wheel_pos_z - wheel_radius
        avg_wheel_bottom = (l_wheel_bottom + r_wheel_bottom) / 2.0

        # Target: wheels slightly penetrating ground (target_dist is negative, e.g. -0.0005)
        # avg_wheel_bottom should equal target_dist (negative means below ground)
        if avg_wheel_bottom > target_dist:
            # Wheels too high, lower root_z
            root_z_max = root_z_mid
        else:
            # Wheels too low, raise root_z
            root_z_min = root_z_mid

        # Check convergence
        if abs(avg_wheel_bottom - target_dist) < 1e-4:
            break

    return float(data.qpos[2])


def generate_height_variant_setup(
    model: mujoco.MjModel,
    height_ik: HeightIK,
    variant_name: str,
    target_com_z_m: float,
    tolerance_m: float = 0.005,
) -> HeightVariantSetup:
    """Generate and validate setup for a height variant.

    Returns setup validation result with all diagnostics.
    """
    # Generate posture using HeightIK
    targets = height_ik.compute_ik_targets(target_com_z_m)
    hip_pitch = float(targets["hip_pitch"])
    knee = float(targets["knee"])

    # Create fresh data
    data = mujoco.MjData(model)

    # Initialize from keyframe 0
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    # Apply height-specific posture (symmetric left/right)
    data.qpos[9] = hip_pitch   # l_hip_pitch
    data.qpos[10] = knee        # l_knee
    data.qpos[14] = hip_pitch   # r_hip_pitch
    data.qpos[15] = knee        # r_knee

    # Keep hip_roll and hip_yaw at nominal (near zero from keyframe)
    # Wheels remain at zero velocity target

    # Zero velocities and accelerations
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    # Calibrate root_z for wheel-floor contact
    calibrated_root_z = calibrate_root_z_for_wheel_floor_contact(model, data)

    # Run forward kinematics
    mujoco.mj_forward(model, data)

    # Measure achieved CoM height
    torso_id = model.body("torso").id
    achieved_com_z = float(data.subtree_com[torso_id][2])
    height_error = abs(achieved_com_z - target_com_z_m)

    # Check contacts
    contact_info = classify_floor_contacts_simple(model, data)

    # Check orientation
    pitch_x, roll_y = compute_orientation_from_gravity_simple(model, data)

    # Extract yaw from quaternion (simplified)
    quat = data.qpos[3:7]
    yaw_z = 2.0 * np.arctan2(quat[3], quat[0])  # Simplified yaw extraction

    # Check joint limits (simplified - just check if any joint is clearly out of reasonable range)
    joint_pos = data.qpos[7:17]
    joint_limit_valid = True
    # Basic sanity check: all joints should be within [-3.14, 3.14] rad
    if np.any(np.abs(joint_pos) > 3.5):
        joint_limit_valid = False

    # Validate setup
    setup_valid = True
    failure_reasons = []

    if height_error >= tolerance_m:
        setup_valid = False
        failure_reasons.append(f"height_error={height_error:.6f}m >= tolerance={tolerance_m}m")

    if not (contact_info["left_wheel_contact"] and contact_info["right_wheel_contact"]):
        setup_valid = False
        failure_reasons.append("missing_wheel_contact")

    if contact_info["non_wheel_floor_contacts"] > 0:
        setup_valid = False
        failure_reasons.append(f"non_wheel_floor_contacts={contact_info['non_wheel_floor_contacts']}")

    if abs(pitch_x) > 0.1 or abs(roll_y) > 0.1:
        setup_valid = False
        failure_reasons.append(f"orientation_error: pitch_x={pitch_x:.4f}, roll_y={roll_y:.4f}")

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
        equilibrium_joint_pos = data.qpos[7:17].tolist()
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
    )


def main():
    parser = argparse.ArgumentParser(description="Validate balance-core across true height variants")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/balance_core_true_height_variants",
        help="Output directory for setup reports",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # Initialize HeightIK
    height_ik = HeightIK(model, scan_points=25, polynomial_degree=2, symmetric_fold=True)

    print("=== Balance-Core True Height Variant Setup Validation ===")
    print()
    print(f"HeightIK feasible range: [{height_ik.height_range[0]:.4f}, {height_ik.height_range[1]:.4f}] m")
    print()

    # Define candidate variants
    nominal_com_z = 0.404  # From recent balance-core run

    variants = [
        ("nominal", nominal_com_z),
        ("high_small", nominal_com_z + 0.01),
        ("high_medium", nominal_com_z + 0.02),
    ]

    print(f"Nominal equilibrium CoM height: {nominal_com_z:.6f} m")
    print(f"Candidate variants: {len(variants)}")
    print()

    # Generate and validate each variant
    setup_results = []
    valid_variants = []
    invalid_variants = []

    for variant_name, target_height in variants:
        print(f"--- Variant: {variant_name} (target={target_height:.6f}m) ---")

        setup = generate_height_variant_setup(
            model=model,
            height_ik=height_ik,
            variant_name=variant_name,
            target_com_z_m=target_height,
            tolerance_m=0.005,
        )

        setup_results.append(setup)

        if setup.setup_valid:
            valid_variants.append(variant_name)
            print(f"  [VALID]")
        else:
            invalid_variants.append(variant_name)
            print(f"  [INVALID]: {setup.setup_failure_reason}")

        print(f"  Achieved CoM Z: {setup.achieved_com_z_m:.6f} m (error: {setup.height_error_m:.6f} m)")
        print(f"  Calibrated root Z: {setup.calibrated_root_z_m:.6f} m")
        print(f"  Hip pitch: {setup.hip_pitch_ref:.4f} rad, Knee: {setup.knee_ref:.4f} rad")
        print(f"  Wheel contacts: L={setup.left_wheel_contact}, R={setup.right_wheel_contact}")
        print(f"  Non-wheel floor contacts: {setup.non_wheel_floor_contact_count}")
        print(f"  Orientation: pitch_x={setup.pitch_x_rad:.4f}, roll_y={setup.roll_y_rad:.4f}")
        print()

    # Write JSON report
    json_report = {
        "height_ik_feasible_range": {
            "min_m": height_ik.height_range[0],
            "max_m": height_ik.height_range[1],
        },
        "nominal_equilibrium_com_z_m": nominal_com_z,
        "selected_variants": [{"name": name, "target_com_z_m": height} for name, height in variants],
        "valid_variants": valid_variants,
        "invalid_variants": invalid_variants,
        "setup_results": [
            {
                "variant_name": s.variant_name,
                "target_com_z_m": s.target_com_z_m,
                "achieved_com_z_m": s.achieved_com_z_m,
                "height_error_m": s.height_error_m,
                "calibrated_root_z_m": s.calibrated_root_z_m,
                "hip_pitch_ref": s.hip_pitch_ref,
                "knee_ref": s.knee_ref,
                "hip_roll_left": s.hip_roll_left,
                "hip_roll_right": s.hip_roll_right,
                "hip_yaw_left": s.hip_yaw_left,
                "hip_yaw_right": s.hip_yaw_right,
                "wheel_floor_contact_count": s.wheel_floor_contact_count,
                "left_wheel_contact": s.left_wheel_contact,
                "right_wheel_contact": s.right_wheel_contact,
                "min_wheel_contact_dist_m": s.min_wheel_contact_dist_m,
                "non_wheel_floor_contact_count": s.non_wheel_floor_contact_count,
                "pitch_x_rad": s.pitch_x_rad,
                "roll_y_rad": s.roll_y_rad,
                "yaw_z_rad": s.yaw_z_rad,
                "joint_limit_valid": s.joint_limit_valid,
                "setup_valid": s.setup_valid,
                "setup_failure_reason": s.setup_failure_reason,
                "equilibrium_joint_pos": s.equilibrium_joint_pos,
                "equilibrium_com_pos": s.equilibrium_com_pos,
                "equilibrium_pitch_x": s.equilibrium_pitch_x,
                "equilibrium_roll_y": s.equilibrium_roll_y,
                "equilibrium_yaw_z": s.equilibrium_yaw_z,
            }
            for s in setup_results
        ],
        "infrastructure_sufficient": len(valid_variants) > 0,
        "ready_for_b5_b10": len(valid_variants) >= 2,  # At least nominal + one variant
    }

    json_path = output_dir / "true_height_variant_setup_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    # Write markdown report
    md_lines = [
        "# Balance-Core True Height Variant Setup Report",
        "",
        "## Summary",
        "",
        f"- **HeightIK feasible range**: [{height_ik.height_range[0]:.4f}, {height_ik.height_range[1]:.4f}] m",
        f"- **Nominal equilibrium CoM Z**: {nominal_com_z:.6f} m",
        f"- **Candidate variants**: {len(variants)}",
        f"- **Valid variants**: {len(valid_variants)} ({', '.join(valid_variants) if valid_variants else 'none'})",
        f"- **Invalid variants**: {len(invalid_variants)} ({', '.join(invalid_variants) if invalid_variants else 'none'})",
        f"- **Infrastructure sufficient**: {'yes' if json_report['infrastructure_sufficient'] else 'no'}",
        f"- **Ready for B5-B10**: {'yes' if json_report['ready_for_b5_b10'] else 'no'}",
        "",
        "## Setup Results",
        "",
    ]

    for s in setup_results:
        status = "✓ VALID" if s.setup_valid else f"✗ INVALID: {s.setup_failure_reason}"
        md_lines.extend([
            f"### {s.variant_name} (target={s.target_com_z_m:.6f}m)",
            "",
            f"**Status**: {status}",
            "",
            "**Posture**:",
            f"- Hip pitch: {s.hip_pitch_ref:.4f} rad ({s.hip_pitch_ref * 57.3:.1f}°)",
            f"- Knee: {s.knee_ref:.4f} rad ({s.knee_ref * 57.3:.1f}°)",
            f"- Hip roll (L/R): {s.hip_roll_left:.4f} / {s.hip_roll_right:.4f} rad",
            f"- Hip yaw (L/R): {s.hip_yaw_left:.4f} / {s.hip_yaw_right:.4f} rad",
            "",
            "**Achieved State**:",
            f"- Achieved CoM Z: {s.achieved_com_z_m:.6f} m",
            f"- Height error: {s.height_error_m:.6f} m",
            f"- Calibrated root Z: {s.calibrated_root_z_m:.6f} m",
            "",
            "**Contact**:",
            f"- Wheel floor contacts: {s.wheel_floor_contact_count}",
            f"- Left wheel contact: {s.left_wheel_contact}",
            f"- Right wheel contact: {s.right_wheel_contact}",
            f"- Min wheel contact dist: {s.min_wheel_contact_dist_m:.6f} m",
            f"- Non-wheel floor contacts: {s.non_wheel_floor_contact_count}",
            "",
            "**Orientation**:",
            f"- Pitch X: {s.pitch_x_rad:.4f} rad ({s.pitch_x_rad * 57.3:.1f}°)",
            f"- Roll Y: {s.roll_y_rad:.4f} rad ({s.roll_y_rad * 57.3:.1f}°)",
            f"- Yaw Z: {s.yaw_z_rad:.4f} rad ({s.yaw_z_rad * 57.3:.1f}°)",
            "",
            "**Validity**:",
            f"- Joint limits valid: {s.joint_limit_valid}",
            f"- Setup valid: {s.setup_valid}",
            "",
        ])

        if s.setup_valid and s.equilibrium_joint_pos is not None:
            md_lines.extend([
                "**Equilibrium References**:",
                f"- Joint pos: {[f'{x:.4f}' for x in s.equilibrium_joint_pos]}",
                f"- CoM pos: [{s.equilibrium_com_pos[0]:.6f}, {s.equilibrium_com_pos[1]:.6f}, {s.equilibrium_com_pos[2]:.6f}] m",
                f"- Pitch X: {s.equilibrium_pitch_x:.4f} rad",
                f"- Roll Y: {s.equilibrium_roll_y:.4f} rad",
                f"- Yaw Z: {s.equilibrium_yaw_z:.4f} rad",
                "",
            ])

    md_lines.extend([
        "## Conclusion",
        "",
        f"Setup validation complete. {len(valid_variants)} valid variants ready for simulation.",
        "",
        "**Next steps**:",
        "- Review this setup report",
        "- If approved, proceed to B5-B10: validation protocol, classification, tests",
        "",
    ])

    md_path = output_dir / "true_height_variant_setup_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("=== Setup Validation Complete ===")
    print(f"Valid variants: {len(valid_variants)}/{len(variants)}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print()
    print("Ready for review. Do not proceed to B5-B10 until approved.")


if __name__ == "__main__":
    main()
