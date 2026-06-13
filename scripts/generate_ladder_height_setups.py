"""Generate intermediate height setup JSONs for Experiment 0 ladder mapping.

Generates setup JSONs for:
- Low side: 0.380m, 0.340m, 0.320m
- High side: 0.430m, 0.465m

Uses the same physical search infrastructure as boundary_height_setups.py.

Usage:
    python scripts/generate_ladder_height_setups.py
"""

import json
import math
from pathlib import Path
from typing import Any, Dict

import mujoco
import numpy as np

from wheeled_biped.utils.config import get_model_path
from wheeled_biped.validation.physical_standing_height_envelope import (
    PhysicalStandingThresholds,
    build_support_segment_geometry,
    compute_robot_com_xy,
    extract_wheel_floor_contact_points,
    evaluate_static_standing_pose,
)
from scripts.search_physical_standing_height_envelope import (
    SearchConfig,
    calibrate_root_z_from_wheel_geometry,
    resolve_standing_joint_addresses,
)


# Ladder heights needed for Experiment 0
LADDER_TARGETS = [
    # Low side
    {"name": "low_0p380", "target_com_z_m": 0.380},
    {"name": "low_0p340", "target_com_z_m": 0.340},
    {"name": "low_0p320", "target_com_z_m": 0.320},
    # High side
    {"name": "high_0p430", "target_com_z_m": 0.430},
    {"name": "high_0p465", "target_com_z_m": 0.465},
]


def _quaternion_to_euler(quat: np.ndarray) -> tuple:
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll_y_rad = float(np.arctan2(sinr_cosp, cosr_cosp))
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch_x_rad = float(np.copysign(np.pi / 2, sinp))
    else:
        pitch_x_rad = float(np.arcsin(sinp))
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw_z_rad = float(np.arctan2(siny_cosp, cosy_cosp))
    return pitch_x_rad, roll_y_rad, yaw_z_rad


def search_height_for_target(
    *,
    model: mujoco.MjModel,
    target_com_z_m: float,
    nominal_hip_pitch: float,
    nominal_knee: float,
    joint_addresses: Dict[str, Any],
    thresholds: PhysicalStandingThresholds,
    config: SearchConfig,
) -> Dict[str, Any] | None:
    """Search for best symmetric hip-pitch/knee posture achieving target CoM height."""
    hip_pitch_range = 0.6
    knee_range = 0.6

    hip_pitch_values = np.linspace(
        nominal_hip_pitch - hip_pitch_range,
        nominal_hip_pitch + hip_pitch_range,
        config.hip_pitch_grid_steps,
    )
    knee_values = np.linspace(
        nominal_knee - knee_range,
        nominal_knee + knee_range,
        config.knee_grid_steps,
    )

    best_candidate = None
    best_score = float("inf")

    for hip_pitch in hip_pitch_values:
        for knee in knee_values:
            data = mujoco.MjData(model)
            data.qpos[:] = 0.0
            data.qpos[3] = 1.0  # quaternion w

            data.qpos[joint_addresses["l_hip_pitch"]] = hip_pitch
            data.qpos[joint_addresses["l_knee"]] = knee
            data.qpos[joint_addresses["r_hip_pitch"]] = hip_pitch
            data.qpos[joint_addresses["r_knee"]] = knee

            try:
                calibrated_root_z = calibrate_root_z_from_wheel_geometry(
                    model, data, target_contact_depth_m=config.target_contact_depth_m
                )
                data.qpos[2] = calibrated_root_z
                mujoco.mj_forward(model, data)
            except Exception:
                continue

            achieved_com_z = float(data.subtree_com[0][2])
            contacts = extract_wheel_floor_contact_points(model, data)
            com_xy = compute_robot_com_xy(model, data)

            quat = data.qpos[3:7]
            pitch_x_rad, roll_y_rad, yaw_z_rad = _quaternion_to_euler(quat)

            joint_margins = []
            for joint_name in ["l_hip_pitch", "l_knee", "r_hip_pitch", "r_knee"]:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_addr = model.jnt_qposadr[joint_id]
                low, high = model.jnt_range[joint_id]
                value = float(data.qpos[qpos_addr])
                joint_margins.append(value - float(low))
                joint_margins.append(float(high) - value)
            joint_limit_margin_rad = float(min(joint_margins))

            candidate_is_root_z_only = (
                abs(hip_pitch - nominal_hip_pitch) < 1e-3
                and abs(knee - nominal_knee) < 1e-3
                and abs(achieved_com_z - target_com_z_m) > 0.005
            )

            support_geom = build_support_segment_geometry(
                left_wheel_contact_xy=contacts.left_wheel_contact_xy,
                right_wheel_contact_xy=contacts.right_wheel_contact_xy,
                com_xy=com_xy,
                thresholds=thresholds,
            )

            result = evaluate_static_standing_pose(
                left_wheel_contact_xy=contacts.left_wheel_contact_xy,
                right_wheel_contact_xy=contacts.right_wheel_contact_xy,
                com_xy=com_xy,
                pitch_x_rad=pitch_x_rad,
                roll_y_rad=roll_y_rad,
                yaw_z_rad=yaw_z_rad,
                left_wheel_contact=contacts.left_wheel_contact,
                right_wheel_contact=contacts.right_wheel_contact,
                non_wheel_floor_contact_count=contacts.non_wheel_floor_contact_count,
                joint_limit_margin_rad=joint_limit_margin_rad,
                thresholds=thresholds,
                candidate_source="ladder_search",
                candidate_is_root_z_only=candidate_is_root_z_only,
            )

            candidate = {
                "requested_target_com_z_m": target_com_z_m,
                "achieved_com_z_m": achieved_com_z,
                "calibrated_root_z_m": calibrated_root_z,
                "hip_pitch_ref": hip_pitch,
                "knee_ref": knee,
                "joint_qpos": {
                    "l_hip_roll": float(data.qpos[joint_addresses["l_hip_roll"]]),
                    "l_hip_yaw": float(data.qpos[joint_addresses["l_hip_yaw"]]),
                    "l_hip_pitch": float(data.qpos[joint_addresses["l_hip_pitch"]]),
                    "l_knee": float(data.qpos[joint_addresses["l_knee"]]),
                    "r_hip_roll": float(data.qpos[joint_addresses["r_hip_roll"]]),
                    "r_hip_yaw": float(data.qpos[joint_addresses["r_hip_yaw"]]),
                    "r_hip_pitch": float(data.qpos[joint_addresses["r_hip_pitch"]]),
                    "r_knee": float(data.qpos[joint_addresses["r_knee"]]),
                },
                "support_geometry": {
                    "left_wheel_xy": list(contacts.left_wheel_contact_xy),
                    "right_wheel_xy": list(contacts.right_wheel_contact_xy),
                    "support_center_xy": list(support_geom.support_center_xy),
                    "support_width_m": support_geom.segment_length_m,
                },
                "contact_metrics": {
                    "left_wheel_contact": contacts.left_wheel_contact,
                    "right_wheel_contact": contacts.right_wheel_contact,
                    "non_wheel_floor_contact_count": contacts.non_wheel_floor_contact_count,
                },
                "joint_limit_margin_rad": joint_limit_margin_rad,
                "candidate_source": "ladder_search",
                "candidate_is_root_z_only": candidate_is_root_z_only,
                "static_feasible": result.static_feasible,
                "rejection_reasons": result.rejection_reasons,
            }

            height_error = abs(achieved_com_z - target_com_z_m)
            com_error = np.linalg.norm(np.array(com_xy) - np.array(support_geom.support_center_xy))
            score = 1000.0 * (0 if result.static_feasible else 1) + 100.0 * height_error + 10.0 * com_error

            if score < best_score:
                best_score = score
                best_candidate = candidate

    return best_candidate


def build_height_variant_setup(
    *,
    variant_name: str,
    target_com_z_m: float,
    candidate: Dict[str, Any],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_addresses: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert a physical search candidate to height-variant-setup format."""
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0  # quaternion w

    for joint_name, value in candidate["joint_qpos"].items():
        data.qpos[joint_addresses[joint_name]] = value

    calibrated_root_z = calibrate_root_z_from_wheel_geometry(
        model, data, target_contact_depth_m=-5e-4
    )
    data.qpos[2] = calibrated_root_z
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    achieved_com_z = float(data.subtree_com[0][2])
    com_xy = compute_robot_com_xy(model, data)
    contacts = extract_wheel_floor_contact_points(model, data)
    quat = data.qpos[3:7]
    pitch_x_rad, roll_y_rad, yaw_z_rad = _quaternion_to_euler(quat)

    joint_margins = []
    for joint_name in ["l_hip_pitch", "l_knee", "r_hip_pitch", "r_knee",
                        "l_hip_yaw", "r_hip_yaw", "l_hip_roll", "r_hip_roll"]:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_addr = model.jnt_qposadr[joint_id]
        low, high = model.jnt_range[joint_id]
        value = float(data.qpos[qpos_addr])
        joint_margins.append(value - float(low))
        joint_margins.append(float(high) - value)
    joint_limit_margin_rad = float(min(joint_margins))

    l_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
    l_wheel_xpos = tuple(float(data.xpos[l_wheel_body_id][i]) for i in range(3))
    r_wheel_xpos = tuple(float(data.xpos[r_wheel_body_id][i]) for i in range(3))
    support_center_x = 0.5 * (l_wheel_xpos[0] + r_wheel_xpos[0])
    support_center_y = 0.5 * (l_wheel_xpos[1] + r_wheel_xpos[1])

    equilibrium_joint_pos = [float(data.qpos[7 + i]) for i in range(10)]
    com_pos = data.subtree_com[0]
    equilibrium_com_pos = [float(com_pos[0]), float(com_pos[1]), float(com_pos[2])]

    com_support_error_x = float(com_xy[0]) - support_center_x
    com_support_error_y = float(com_xy[1]) - support_center_y
    com_support_error_norm = math.sqrt(com_support_error_x**2 + com_support_error_y**2)

    result = evaluate_static_standing_pose(
        left_wheel_contact_xy=contacts.left_wheel_contact_xy,
        right_wheel_contact_xy=contacts.right_wheel_contact_xy,
        com_xy=com_xy,
        pitch_x_rad=pitch_x_rad,
        roll_y_rad=roll_y_rad,
        yaw_z_rad=yaw_z_rad,
        left_wheel_contact=contacts.left_wheel_contact,
        right_wheel_contact=contacts.right_wheel_contact,
        non_wheel_floor_contact_count=contacts.non_wheel_floor_contact_count,
        joint_limit_margin_rad=joint_limit_margin_rad,
        thresholds=PhysicalStandingThresholds(),
        candidate_source="ladder_search",
        candidate_is_root_z_only=False,
    )

    return {
        "variant_name": variant_name,
        "target_com_z_m": target_com_z_m,
        "achieved_com_z_m": achieved_com_z,
        "height_error_m": abs(achieved_com_z - target_com_z_m),
        "calibrated_root_z_m": calibrated_root_z,
        "hip_pitch_ref": candidate["hip_pitch_ref"],
        "knee_ref": candidate["knee_ref"],
        "hip_roll_left": 0.0,
        "hip_roll_right": 0.0,
        "hip_yaw_left": 0.0,
        "hip_yaw_right": 0.0,
        "support_center_x": support_center_x,
        "support_center_y": support_center_y,
        "com_x_m": float(com_xy[0]),
        "com_y_m": float(com_xy[1]),
        "com_support_error_x": com_support_error_x,
        "com_support_error_y": com_support_error_y,
        "com_support_error_norm_xy": com_support_error_norm,
        "wheel_floor_contact_count": int(contacts.left_wheel_contact) + int(contacts.right_wheel_contact),
        "left_wheel_contact": contacts.left_wheel_contact,
        "right_wheel_contact": contacts.right_wheel_contact,
        "non_wheel_floor_contact_count": contacts.non_wheel_floor_contact_count,
        "pitch_x_rad": pitch_x_rad,
        "roll_y_rad": roll_y_rad,
        "yaw_z_rad": yaw_z_rad,
        "joint_limit_valid": joint_limit_margin_rad > 0.05,
        "joint_limit_margin_rad": joint_limit_margin_rad,
        "setup_valid": result.static_feasible,
        "setup_failure_reason": None if result.static_feasible else "; ".join(result.rejection_reasons),
        "static_feasible": result.static_feasible,
        "rejection_reasons": result.rejection_reasons,
        "equilibrium_joint_pos": equilibrium_joint_pos,
        "equilibrium_com_pos": equilibrium_com_pos,
        "equilibrium_pitch_x": pitch_x_rad,
        "equilibrium_roll_y": roll_y_rad,
        "equilibrium_yaw_z": yaw_z_rad,
        "candidate_source": "ladder_search",
        "candidate_is_root_z_only": False,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate intermediate height setup JSONs for Experiment 0 ladder mapping"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/physical_target_height_setups"),
        help="Output directory for setup JSONs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Ladder Height Setup Generator - Experiment 0")
    print("=" * 80)

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    thresholds = PhysicalStandingThresholds()
    config = SearchConfig()
    joint_addresses = resolve_standing_joint_addresses(model)

    data_nominal = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data_nominal, 0)
    data_nominal.qvel[:] = 0.0
    data_nominal.qacc[:] = 0.0

    nominal_root_z = calibrate_root_z_from_wheel_geometry(
        model, data_nominal, target_contact_depth_m=config.target_contact_depth_m
    )
    data_nominal.qpos[2] = nominal_root_z
    mujoco.mj_forward(model, data_nominal)

    nominal_com_z = float(data_nominal.subtree_com[0][2])
    nominal_hip_pitch = float(data_nominal.qpos[joint_addresses["l_hip_pitch"]])
    nominal_knee = float(data_nominal.qpos[joint_addresses["l_knee"]])

    print(f"Nominal CoM z: {nominal_com_z:.6f} m")
    print(f"Nominal hip_pitch: {nominal_hip_pitch:.6f} rad")
    print(f"Nominal knee: {nominal_knee:.6f} rad")

    setups = []
    static_validations = []
    all_valid = True

    for target_info in LADDER_TARGETS:
        name = target_info["name"]
        target_com_z = target_info["target_com_z_m"]

        print(f"\n--- Searching for {name} at {target_com_z:.3f} m ---")

        candidate = search_height_for_target(
            model=model,
            target_com_z_m=target_com_z,
            nominal_hip_pitch=nominal_hip_pitch,
            nominal_knee=nominal_knee,
            joint_addresses=joint_addresses,
            thresholds=thresholds,
            config=config,
        )

        if candidate is None:
            print(f"  FAILED: No candidate found for {target_com_z:.3f} m")
            all_valid = False
            continue

        print(f"  Found candidate: com_z={candidate['achieved_com_z_m']:.6f} m, "
              f"hip_pitch={candidate['hip_pitch_ref']:.4f}, knee={candidate['knee_ref']:.4f}")
        print(f"  Static feasible: {candidate['static_feasible']}")
        print(f"  Joint limit margin: {candidate['joint_limit_margin_rad']:.4f} rad")
        print(f"  Root-z-only: {candidate['candidate_is_root_z_only']}")

        data_setup = mujoco.MjData(model)
        setup = build_height_variant_setup(
            variant_name=name,
            target_com_z_m=target_com_z,
            candidate=candidate,
            model=model,
            data=data_setup,
            joint_addresses=joint_addresses,
        )

        print(f"  Achieved CoM z: {setup['achieved_com_z_m']:.6f} m (error: {setup['height_error_m']:.6f} m)")
        print(f"  Setup valid: {setup['setup_valid']}")
        print(f"  Root z: {setup['calibrated_root_z_m']:.6f} m")
        print(f"  Left wheel contact: {setup['left_wheel_contact']}")
        print(f"  Right wheel contact: {setup['right_wheel_contact']}")
        print(f"  Non-wheel contacts: {setup['non_wheel_floor_contact_count']}")
        print(f"  COM-support error: {setup['com_support_error_norm_xy']:.6f} m")

        setup_path = output_dir / f"{name}_setup.json"
        setup_path.write_text(json.dumps(setup, indent=2), encoding="utf-8")
        print(f"  Written: {setup_path}")

        validation = {
            "target_name": name,
            "target_com_z_m": target_com_z,
            "achieved_com_z_m": setup["achieved_com_z_m"],
            "height_error_m": setup["height_error_m"],
            "setup_valid": setup["setup_valid"],
            "setup_failure_reason": setup["setup_failure_reason"],
            "left_wheel_contact": setup["left_wheel_contact"],
            "right_wheel_contact": setup["right_wheel_contact"],
            "non_wheel_floor_contact_count": setup["non_wheel_floor_contact_count"],
            "com_support_error_norm_xy": setup["com_support_error_norm_xy"],
            "pitch_x_rad": setup["pitch_x_rad"],
            "roll_y_rad": setup["roll_y_rad"],
            "yaw_z_rad": setup["yaw_z_rad"],
            "joint_limit_margin_rad": setup["joint_limit_margin_rad"],
            "candidate_is_root_z_only": setup["candidate_is_root_z_only"],
        }
        static_validations.append(validation)
        setups.append(setup)

        if not setup["setup_valid"]:
            all_valid = False
            print(f"  STATIC VALIDATION FAILED: {setup['setup_failure_reason']}")
        else:
            print(f"  STATIC VALIDATION PASSED")

    # Write ladder validation summary
    validation_path = output_dir / "ladder_setup_validation_summary.json"
    validation_path.write_text(json.dumps(static_validations, indent=2), encoding="utf-8")

    # Write inventory
    inventory = {
        "phase": "Experiment 0 Setup Inventory",
        "heights_generated": [t["target_com_z_m"] for t in LADDER_TARGETS],
        "all_valid": all_valid,
        "validations": static_validations,
    }
    inventory_path = Path("outputs/height_range_extension_experiment_0/setup_inventory.json")
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    # Write markdown inventory
    md_lines = ["# Experiment 0 Setup Inventory\n", "## Generated Heights\n"]
    for v in static_validations:
        status = "PASS" if v["setup_valid"] else "FAIL"
        md_lines.extend([
            f"\n## {v['target_name']}: {status}",
            f"- Target: {v['target_com_z_m']:.3f} m",
            f"- Achieved: {v['achieved_com_z_m']:.6f} m (error: {v['height_error_m']:.6f} m)",
            f"- Left wheel contact: {v['left_wheel_contact']}",
            f"- Right wheel contact: {v['right_wheel_contact']}",
            f"- Non-wheel contacts: {v['non_wheel_floor_contact_count']}",
            f"- COM-support error: {v['com_support_error_norm_xy']:.6f} m",
            f"- Joint limit margin: {v['joint_limit_margin_rad']:.4f} rad",
            f"- Root-z-only: {v['candidate_is_root_z_only']}",
        ])

    md_path = Path("outputs/height_range_extension_experiment_0/setup_inventory.md")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n" + "=" * 80)
    if all_valid:
        print("All ladder height setups PASSED static validation")
    else:
        print("WARNING: Some ladder height setups FAILED static validation")
    print("=" * 80)

    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
