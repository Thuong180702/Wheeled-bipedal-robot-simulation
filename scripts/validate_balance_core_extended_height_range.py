#!/usr/bin/env python3
"""Validate balance-core controller across extended height range (±2cm, ±5cm, ±10cm, ±15cm).

This script extends the multi-objective CoM-calibrated posture search to much larger height offsets.
For each target height:
1. Search for valid hip_pitch/knee posture with wider search range
2. Enforce all setup validity gates (height, CoM centering, orientation, contact, joints)
3. Generate detailed candidate search diagnostics
4. Run dynamic validation (500→1000 steps) for valid variants only
5. Support visual simulation for extreme valid variants

This is NOT root-z-only perturbation. Each variant has different joint posture.
"""

import argparse
import json
import subprocess
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
    best_by_height: tuple[float, float, float, float] | None = None
    best_by_com: tuple[float, float, float, float] | None = None
    top_rejected: list[dict] = field(default_factory=list)


@dataclass
class HeightVariantSetup:
    """Setup validation result for a height variant."""
    variant_name: str
    target_com_z_m: float
    offset_from_nominal_m: float
    achieved_com_z_m: float
    height_error_m: float
    calibrated_root_z_m: float
    hip_pitch_ref: float
    knee_ref: float
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
    cp_x_m: float
    cp_y_m: float
    cp_error_x_m: float
    cp_error_y_m: float
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


def search_extended_height_posture(
    model: mujoco.MjModel,
    target_com_z_m: float,
    nominal_hip_pitch: float,
    nominal_knee: float,
    nominal_com_support_error_x: float,
    nominal_com_support_error_y: float,
    offset_magnitude: float,
) -> tuple[float, float, float, CandidateStats] | None:
    """Search for hip_pitch/knee with adaptive range based on offset magnitude.

    For larger offsets (±10cm, ±15cm), use wider search range and more steps.
    """
    torso_id = model.body("torso").id

    # Adaptive search parameters based on offset magnitude
    if abs(offset_magnitude) >= 0.10:
        search_range = 0.50  # ±50cm for ±10cm+ offsets
        search_steps = 30
        height_tolerance = 0.010  # 10mm for extreme offsets
    elif abs(offset_magnitude) >= 0.05:
        search_range = 0.35  # ±35cm for ±5cm offsets
        search_steps = 25
        height_tolerance = 0.007  # 7mm
    else:
        search_range = 0.25  # ±25cm for ±2cm offsets
        search_steps = 20
        height_tolerance = 0.005  # 5mm

    # Weights for multi-objective scoring
    w_height = 100.0
    w_com_y = 50.0
    w_com_x = 20.0
    w_pitch = 10.0
    w_roll = 10.0
    w_yaw = 5.0
    w_joint = 1.0

    # Tolerances for gate checks
    com_centering_tolerance_x = 0.020  # 20mm (relaxed for extreme heights)
    com_centering_tolerance_y = 0.020  # 20mm
    orientation_tolerance = 0.05  # ~2.9 degrees (relaxed for extreme heights)

    stats = CandidateStats()
    all_candidates = []

    hip_pitch_values = np.linspace(
        max(0.0, nominal_hip_pitch - search_range),
        min(1.5, nominal_hip_pitch + search_range),
        search_steps
    )
    knee_values = np.linspace(
        max(0.0, nominal_knee - search_range),
        min(2.5, nominal_knee + search_range),
        search_steps
    )

    for hip_pitch in hip_pitch_values:
        for knee in knee_values:
            stats.total_evaluated += 1

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

            contact_info = classify_floor_contacts_simple(model, data)
            if not (contact_info["left_wheel_contact"] and contact_info["right_wheel_contact"]):
                continue
            if contact_info["non_wheel_floor_contacts"] > 0:
                continue

            stats.passed_contact += 1

            achieved_com_z = float(data.subtree_com[torso_id][2])
            height_error = abs(achieved_com_z - target_com_z_m)

            support_center_x, support_center_y = compute_support_center(model, data)
            com_x = float(data.subtree_com[torso_id][0])
            com_y = float(data.subtree_com[torso_id][1])
            com_support_error_x, com_support_error_y, com_support_error_norm = compute_com_support_error(
                com_x, com_y, support_center_x, support_center_y
            )

            delta_com_error_x = abs(com_support_error_x - nominal_com_support_error_x)
            delta_com_error_y = abs(com_support_error_y - nominal_com_support_error_y)

            pitch_x, roll_y = compute_orientation_from_gravity_simple(model, data)
            quat = data.qpos[3:7]
            yaw_z = 2.0 * np.arctan2(quat[3], quat[0])

            joint_dist = np.sqrt((hip_pitch - nominal_hip_pitch)**2 + (knee - nominal_knee)**2)

            score = (
                w_height * height_error +
                w_com_y * delta_com_error_y +
                w_com_x * delta_com_error_x +
                w_pitch * abs(pitch_x) +
                w_roll * abs(roll_y) +
                w_yaw * abs(yaw_z) +
                w_joint * joint_dist
            )

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

            if stats.best_by_height is None or height_error < stats.best_by_height[2]:
                stats.best_by_height = (hip_pitch, knee, height_error, score)

            if stats.best_by_com is None or com_support_error_norm < stats.best_by_com[2]:
                stats.best_by_com = (hip_pitch, knee, com_support_error_norm, score)

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

    all_candidates.sort(key=lambda c: c["score"])
    valid_candidates = [c for c in all_candidates if c["passed_all"]]

    if valid_candidates:
        best = valid_candidates[0]
        return (best["hip_pitch"], best["knee"], target_com_z_m, stats)

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


def generate_extended_height_variant_setup(
    model: mujoco.MjModel,
    variant_name: str,
    target_com_z_m: float,
    offset_from_nominal_m: float,
    nominal_hip_pitch: float,
    nominal_knee: float,
    nominal_com_support_error_x: float,
    nominal_com_support_error_y: float,
    use_nominal_baseline: bool = False,
) -> HeightVariantSetup:
    """Generate and validate setup for an extended height variant."""
    torso_id = model.body("torso").id

    if use_nominal_baseline:
        data = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        calibrated_root_z = calibrate_root_z_for_wheel_floor_contact(model, data)
        mujoco.mj_forward(model, data)
        hip_pitch = float(data.qpos[9])
        knee = float(data.qpos[10])
        posture_method = "nominal_keyframe_baseline"
        candidate_stats = None
    else:
        search_result = search_extended_height_posture(
            model, target_com_z_m, nominal_hip_pitch, nominal_knee,
            nominal_com_support_error_x, nominal_com_support_error_y,
            offset_from_nominal_m
        )

        if search_result is None:
            return HeightVariantSetup(
                variant_name=variant_name,
                target_com_z_m=target_com_z_m,
                offset_from_nominal_m=offset_from_nominal_m,
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
                setup_failure_reason="no_valid_posture_found_within_search_range",
                equilibrium_joint_pos=None,
                equilibrium_com_pos=None,
                equilibrium_pitch_x=None,
                equilibrium_roll_y=None,
                equilibrium_yaw_z=None,
                posture_search_method="extended_search_failed",
                candidate_stats=search_result[3] if search_result else None,
            )

        hip_pitch, knee, _, candidate_stats = search_result

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
        posture_method = "extended_multiobjective_search"

    achieved_com_z = float(data.subtree_com[torso_id][2])
    height_error = abs(achieved_com_z - target_com_z_m)
    contact_info = classify_floor_contacts_simple(model, data)
    pitch_x, roll_y = compute_orientation_from_gravity_simple(model, data)
    quat = data.qpos[3:7]
    yaw_z = 2.0 * np.arctan2(quat[3], quat[0])
    joint_pos = data.qpos[7:17]
    joint_limit_valid = not np.any(np.abs(joint_pos) > 3.5)

    support_center_x, support_center_y = compute_support_center(model, data)
    com_x = float(data.subtree_com[torso_id][0])
    com_y = float(data.subtree_com[torso_id][1])
    com_support_error_x, com_support_error_y, com_support_error_norm = compute_com_support_error(
        com_x, com_y, support_center_x, support_center_y
    )

    cp_x = com_x
    cp_y = com_y
    cp_error_x = com_x - nominal_com_support_error_x
    cp_error_y = com_y - nominal_com_support_error_y

    # Adaptive tolerance based on offset magnitude
    if abs(offset_from_nominal_m) >= 0.10:
        height_tolerance = 0.010
    elif abs(offset_from_nominal_m) >= 0.05:
        height_tolerance = 0.007
    else:
        height_tolerance = 0.005

    setup_valid = True
    failure_reasons = []

    if height_error >= height_tolerance:
        setup_valid = False
        failure_reasons.append(f"height_error={height_error:.6f}m")

    if not (contact_info["left_wheel_contact"] and contact_info["right_wheel_contact"]):
        setup_valid = False
        failure_reasons.append("missing_wheel_contact")

    if contact_info["non_wheel_floor_contacts"] > 0:
        setup_valid = False
        failure_reasons.append(f"non_wheel_floor_contacts={contact_info['non_wheel_floor_contacts']}")

    if abs(pitch_x) > 0.05 or abs(roll_y) > 0.05:
        setup_valid = False
        failure_reasons.append("orientation_not_equilibrium")

    if abs(data.qpos[7]) > 0.05 or abs(data.qpos[12]) > 0.05:
        setup_valid = False
        failure_reasons.append("hip_roll_not_nominal")

    if not joint_limit_valid:
        setup_valid = False
        failure_reasons.append("joint_limit_violation")

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
        offset_from_nominal_m=offset_from_nominal_m,
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


def run_dynamic_validation(variant_setup: HeightVariantSetup, num_steps: int, output_dir: Path) -> dict:
    """Run full balance-core dynamic validation for a valid variant."""
    variant_name = variant_setup.variant_name
    variant_output_dir = output_dir / f"dynamic_{variant_name}"
    variant_output_dir.mkdir(parents=True, exist_ok=True)

    variant_setup_path = variant_output_dir / "variant_setup.json"
    with open(variant_setup_path, "w") as f:
        json.dump({
            "variant_name": variant_setup.variant_name,
            "target_com_z_m": variant_setup.target_com_z_m,
            "hip_pitch_ref": variant_setup.hip_pitch_ref,
            "knee_ref": variant_setup.knee_ref,
            "hip_roll_left": variant_setup.hip_roll_left,
            "hip_roll_right": variant_setup.hip_roll_right,
            "hip_yaw_left": variant_setup.hip_yaw_left,
            "hip_yaw_right": variant_setup.hip_yaw_right,
            "calibrated_root_z_m": variant_setup.calibrated_root_z_m,
        }, f, indent=2)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--steps", str(num_steps),
        "--height-variant-setup", str(variant_setup_path),
    ]

    try:
        import time
        start_time = time.time()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
        )

        telemetry_path = None
        if result.returncode == 0:
            sim_output_dir = PROJECT_ROOT / "outputs" / "hierarchical_controller_sim"
            if sim_output_dir.exists():
                telemetry_files = list(sim_output_dir.glob("telemetry_*.csv"))
                for tf in sorted(telemetry_files, key=lambda p: p.stat().st_mtime, reverse=True):
                    if tf.stat().st_mtime >= start_time:
                        telemetry_path = str(tf)
                        break

        if result.returncode == 0:
            return {
                "success": True,
                "returncode": result.returncode,
                "telemetry_path": telemetry_path,
            }
        else:
            return {
                "success": False,
                "returncode": result.returncode,
                "telemetry_path": None,
                "error": result.stderr[-1000:] if result.stderr else result.stdout[-1000:] if result.stdout else f"Process exited with code {result.returncode}",
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "telemetry_path": None,
            "error": "Simulation timeout after 300s",
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "telemetry_path": None,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Validate balance-core across extended height range (±2cm, ±5cm, ±10cm, ±15cm)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/balance_core_extended_height_range",
        help="Output directory for validation reports",
    )
    parser.add_argument(
        "--skip-dynamic",
        action="store_true",
        help="Skip dynamic validation (setup only)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))

    print("=== Balance-Core Extended Height Range Validation ===")
    print()

    # Measure nominal baseline
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

    nominal_support_center_x, nominal_support_center_y = compute_support_center(model, data_nominal)
    nominal_com_x = float(data_nominal.subtree_com[torso_id][0])
    nominal_com_y = float(data_nominal.subtree_com[torso_id][1])
    nominal_com_support_error_x, nominal_com_support_error_y, nominal_com_support_error_norm = compute_com_support_error(
        nominal_com_x, nominal_com_y, nominal_support_center_x, nominal_support_center_y
    )

    print(f"Nominal CoM Z: {nominal_com_z:.6f} m")
    print(f"Nominal hip_pitch: {nominal_hip_pitch:.4f} rad, knee: {nominal_knee:.4f} rad")
    print(f"Nominal CoM support error: ({nominal_com_support_error_x:.6f}, {nominal_com_support_error_y:.6f}) m")
    print()

    # Define extended height offsets
    offsets = [0.0, 0.02, -0.02, 0.05, -0.05, 0.10, -0.10, 0.15, -0.15]

    variants = []
    for offset in offsets:
        if offset == 0.0:
            variant_name = "nominal"
            use_baseline = True
        elif offset > 0:
            variant_name = f"high_{int(abs(offset)*100)}cm"
            use_baseline = False
        else:
            variant_name = f"low_{int(abs(offset)*100)}cm"
            use_baseline = False

        variants.append((variant_name, nominal_com_z + offset, offset, use_baseline))

    setup_results = []
    valid_variants = []
    invalid_variants = []

    # Generate each variant setup
    for variant_name, target_height, offset, use_baseline in variants:
        print(f"--- {variant_name} (target={target_height:.6f}m, offset={offset:+.3f}m) ---")

        setup = generate_extended_height_variant_setup(
            model, variant_name, target_height, offset,
            nominal_hip_pitch, nominal_knee,
            nominal_com_support_error_x, nominal_com_support_error_y,
            use_baseline
        )

        setup_results.append(setup)

        if setup.setup_valid:
            valid_variants.append(variant_name)
            print(f"  [VALID]")
        else:
            invalid_variants.append(variant_name)
            print(f"  [INVALID]: {setup.setup_failure_reason}")

        print(f"  Achieved: {setup.achieved_com_z_m:.6f} m, error: {setup.height_error_m:.6f} m")
        print(f"  Hip pitch: {setup.hip_pitch_ref:.4f} rad, knee: {setup.knee_ref:.4f} rad")
        print(f"  CoM support error: ({setup.com_support_error_x:.6f}, {setup.com_support_error_y:.6f}) m")
        print(f"  Method: {setup.posture_search_method}")

        if setup.candidate_stats:
            stats = setup.candidate_stats
            print(f"  Candidates: {stats.total_evaluated} evaluated, {stats.passed_all} passed all gates")
            if stats.top_rejected:
                print(f"  Top rejection: {stats.top_rejected[0]['reasons']}")

        print()

    # Write setup report
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
        "validation_method": "extended_height_range_multiobjective_search",
        "nominal_com_z_m": nominal_com_z,
        "offsets_tested_m": offsets,
        "valid_variants": valid_variants,
        "invalid_variants": invalid_variants,
        "setup_results": [
            {
                **{k: v for k, v in vars(s).items() if k != "candidate_stats"},
                "candidate_stats": serialize_stats(s.candidate_stats),
            }
            for s in setup_results
        ],
    }

    json_path = output_dir / "extended_height_setup_report.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    print(f"Setup report: {json_path}")
    print(f"Valid: {len(valid_variants)}/{len(variants)}")
    print()

    # Dynamic validation for valid variants
    if not args.skip_dynamic and valid_variants:
        print("=== Dynamic Validation ===")
        print()

        dynamic_results = []
        for setup in setup_results:
            if not setup.setup_valid:
                continue

            print(f"--- {setup.variant_name} ---")

            # Progressive validation: 500 → 1000 steps
            for target_steps in [500, 1000]:
                print(f"  Testing {target_steps} steps...")
                result = run_dynamic_validation(setup, target_steps, output_dir)

                dynamic_results.append({
                    "variant_name": setup.variant_name,
                    "target_steps": target_steps,
                    "success": result["success"],
                    "telemetry_path": result.get("telemetry_path"),
                    "error": result.get("error"),
                })

                if not result["success"]:
                    print(f"    FAILED: {result.get('error', 'unknown')}")
                    break
                else:
                    print(f"    PASSED")

            print()

        # Write dynamic summary
        dynamic_json_path = output_dir / "extended_height_dynamic_summary.json"
        with open(dynamic_json_path, "w") as f:
            json.dump({
                "validation_method": "full_balance_core_4_source_controller",
                "dynamic_results": dynamic_results,
            }, f, indent=2)

        print(f"Dynamic summary: {dynamic_json_path}")

    print("Extended height range validation complete.")
    print(f"Valid variants: {', '.join(valid_variants) if valid_variants else 'none'}")
    print(f"Invalid variants: {', '.join(invalid_variants) if invalid_variants else 'none'}")


if __name__ == "__main__":
    main()
