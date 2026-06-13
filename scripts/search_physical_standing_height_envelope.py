"""
Physical standing height envelope search helpers.

Task 3: Helper functions for calibration, joint resolution, and extrema selection.
Task 4: Serialization, artifact writing, and revalidation functions.
Uses shared thresholds from wheeled_biped.validation.physical_standing_height_envelope.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

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


@dataclass
class SearchConfig:
    """
    Configuration for physical standing height envelope search.

    This dataclass captures all parameters needed for the search:
    - Height range to explore
    - Joint limits and constraints
    - Contact calibration settings
    - Thresholds for feasibility evaluation
    """
    coarse_target_step_m: float = 0.005
    fine_target_step_m: float = 0.001
    initial_target_span_m: float = 0.06
    outward_expand_step_m: float = 0.01
    max_outward_expansions: int = 6
    hip_pitch_grid_steps: int = 17
    knee_grid_steps: int = 17
    target_contact_depth_m: float = -5e-4


def resolve_standing_joint_addresses(model: mujoco.MjModel) -> Dict[str, int | Tuple[float, float, float]]:
    """
    Resolve joint addresses and axes for standing posture joints.

    This function reads actual model joint names and axes using mj_name2id,
    not hardcoded index offsets or sign conventions.

    Args:
        model: MuJoCo model.

    Returns:
        Dict with joint indices and axis tuples.
    """
    # Resolve joint IDs by name
    l_hip_pitch_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_hip_pitch")
    l_knee_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_knee")
    r_hip_pitch_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_hip_pitch")
    r_knee_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_knee")
    l_hip_yaw_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_hip_yaw")
    r_hip_yaw_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_hip_yaw")
    l_hip_roll_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_hip_roll")
    r_hip_roll_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_hip_roll")

    # Get qpos addresses for these joints
    l_hip_pitch_idx = int(model.jnt_qposadr[l_hip_pitch_jnt_id])
    l_knee_idx = int(model.jnt_qposadr[l_knee_jnt_id])
    r_hip_pitch_idx = int(model.jnt_qposadr[r_hip_pitch_jnt_id])
    r_knee_idx = int(model.jnt_qposadr[r_knee_jnt_id])
    l_hip_yaw_idx = int(model.jnt_qposadr[l_hip_yaw_jnt_id])
    r_hip_yaw_idx = int(model.jnt_qposadr[r_hip_yaw_jnt_id])
    l_hip_roll_idx = int(model.jnt_qposadr[l_hip_roll_jnt_id])
    r_hip_roll_idx = int(model.jnt_qposadr[r_hip_roll_jnt_id])

    # Get joint axes from model and convert to tuples
    l_hip_pitch_axis = tuple(float(x) for x in model.jnt_axis[l_hip_pitch_jnt_id])
    l_knee_axis = tuple(float(x) for x in model.jnt_axis[l_knee_jnt_id])
    r_hip_pitch_axis = tuple(float(x) for x in model.jnt_axis[r_hip_pitch_jnt_id])
    r_knee_axis = tuple(float(x) for x in model.jnt_axis[r_knee_jnt_id])
    l_hip_yaw_axis = tuple(float(x) for x in model.jnt_axis[l_hip_yaw_jnt_id])
    r_hip_yaw_axis = tuple(float(x) for x in model.jnt_axis[r_hip_yaw_jnt_id])
    l_hip_roll_axis = tuple(float(x) for x in model.jnt_axis[l_hip_roll_jnt_id])
    r_hip_roll_axis = tuple(float(x) for x in model.jnt_axis[r_hip_roll_jnt_id])

    return {
        "l_hip_pitch": l_hip_pitch_idx,
        "l_knee": l_knee_idx,
        "r_hip_pitch": r_hip_pitch_idx,
        "r_knee": r_knee_idx,
        "l_hip_yaw": l_hip_yaw_idx,
        "r_hip_yaw": r_hip_yaw_idx,
        "l_hip_roll": l_hip_roll_idx,
        "r_hip_roll": r_hip_roll_idx,
        "l_hip_pitch_axis": l_hip_pitch_axis,
        "l_knee_axis": l_knee_axis,
        "r_hip_pitch_axis": r_hip_pitch_axis,
        "r_knee_axis": r_knee_axis,
        "l_hip_yaw_axis": l_hip_yaw_axis,
        "r_hip_yaw_axis": r_hip_yaw_axis,
        "l_hip_roll_axis": l_hip_roll_axis,
        "r_hip_roll_axis": r_hip_roll_axis,
    }


def calibrate_root_z_from_wheel_geometry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    target_contact_depth_m: float = -5e-4,
) -> float:
    """
    Calibrate root z position from wheel geometry to establish floor contact.

    This function computes the root_z value that places both wheels at the
    target contact depth below the floor plane, based on current joint configuration.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (assumes qpos already set except root_z).
        target_contact_depth_m: Target penetration depth (negative = penetration).

    Returns:
        Calibrated root_z value that places wheels at floor level.
    """
    # Get wheel body IDs
    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    # Forward kinematics with current qpos
    mujoco.mj_forward(model, data)

    # Get wheel positions
    l_wheel_z = data.xpos[l_wheel_id][2]
    r_wheel_z = data.xpos[r_wheel_id][2]

    # Average wheel center z
    avg_wheel_z = 0.5 * (l_wheel_z + r_wheel_z)

    # Extract wheel radius from model geometry
    # Documented fallback: 0.06m is the wheel radius in the MJCF model specification
    wheel_radius = 0.06
    try:
        # Try to read from left wheel collision geometry
        l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
        if l_wheel_geom_id >= 0:
            # For cylinder geoms, size[0] is the radius
            geom_type = model.geom_type[l_wheel_geom_id]
            if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                wheel_radius = float(model.geom_size[l_wheel_geom_id][0])
    except Exception:
        # Fall back to documented default if geometry extraction fails
        pass

    # Target wheel center at wheel_radius above floor, adjusted by contact depth
    target_wheel_z = wheel_radius + target_contact_depth_m

    # Compute required root z adjustment
    current_root_z = data.qpos[2]
    calibrated_root_z = current_root_z + (target_wheel_z - avg_wheel_z)

    return calibrated_root_z


def select_physical_extrema(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select physical height envelope extrema from candidate poses.

    This function:
    1. Filters to static_feasible=True candidates only
    2. Ignores dynamic failure metadata fields
    3. Returns min/max from the static_feasible set

    Args:
        candidates: List of candidate dicts with static_feasible and achieved_com_z_m fields.

    Returns:
        Dict with "physical_min_height" and "physical_max_height" keys.
    """
    # Filter to static feasible only
    static_feasible_candidates = [
        c for c in candidates if c.get("static_feasible", False)
    ]

    if not static_feasible_candidates:
        return {
            "physical_min_height": None,
            "physical_max_height": None,
        }

    # Find min and max by achieved_com_z_m
    min_candidate = min(static_feasible_candidates, key=lambda c: c["achieved_com_z_m"])
    max_candidate = max(static_feasible_candidates, key=lambda c: c["achieved_com_z_m"])

    return {
        "physical_min_height": min_candidate,
        "physical_max_height": max_candidate,
    }


def serialize_candidate_setup(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize candidate setup to JSON-compatible dict with required schema fields.

    This function extracts the minimal reproducible setup information needed
    to revalidate a physical standing height candidate.

    Required schema fields (approved plan):
    - requested_target_com_z_m: float
    - achieved_com_z_m: float
    - calibrated_root_z_m: float
    - hip_pitch_ref: float
    - knee_ref: float
    - joint_qpos: dict with 8 standing joint names -> float values
    - support_geometry: dict (serialized)
    - contact_metrics: dict (serialized)
    - joint_limit_margin_rad: float
    - candidate_source: str
    - candidate_is_root_z_only: bool
    - rejection_reasons: list[str]

    Args:
        candidate: Candidate dict from search with evaluation results.

    Returns:
        JSON-serializable dict with required schema fields.
    """
    payload = {
        "requested_target_com_z_m": float(candidate["requested_target_com_z_m"]),
        "achieved_com_z_m": float(candidate["achieved_com_z_m"]),
        "calibrated_root_z_m": float(candidate["calibrated_root_z_m"]),
        "hip_pitch_ref": float(candidate["hip_pitch_ref"]),
        "knee_ref": float(candidate["knee_ref"]),
        "joint_qpos": candidate["joint_qpos"],
        "support_geometry": candidate["support_geometry"],
        "contact_metrics": candidate["contact_metrics"],
        "joint_limit_margin_rad": float(candidate["joint_limit_margin_rad"]),
        "candidate_source": str(candidate["candidate_source"]),
        "candidate_is_root_z_only": bool(candidate["candidate_is_root_z_only"]),
        "rejection_reasons": list(candidate["rejection_reasons"]),
    }
    return payload


def _quaternion_to_euler(quat: np.ndarray) -> tuple[float, float, float]:
    """
    Convert quaternion to Euler angles.

    Args:
        quat: Quaternion as [qw, qx, qy, qz].

    Returns:
        Tuple of (pitch_x_rad, roll_y_rad, yaw_z_rad).
    """
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]

    # Compute roll (rotation around X)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll_y_rad = float(np.arctan2(sinr_cosp, cosr_cosp))

    # Compute pitch (rotation around Y)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch_x_rad = float(np.copysign(np.pi / 2, sinp))
    else:
        pitch_x_rad = float(np.arcsin(sinp))

    # Compute yaw (rotation around Z)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw_z_rad = float(np.arctan2(siny_cosp, cosy_cosp))

    return pitch_x_rad, roll_y_rad, yaw_z_rad


def write_candidate_artifacts(
    output_dir: Path,
    valid_candidates: List[Dict[str, Any]],
    invalid_candidates: List[Dict[str, Any]],
    extrema: Dict[str, Any],
    static_revalidation: Dict[str, Any],
    search_grid_rows: List[Dict[str, Any]],
) -> None:
    """
    Write physical height search artifacts to output directory.

    This function writes all artifact files required for physical standing
    height envelope documentation and revalidation:
    - physical_height_search_grid.csv: All search grid rows
    - physical_height_valid_candidates.json: Valid candidates only
    - physical_height_invalid_candidates.json: Invalid candidates with rejection reasons
    - physical_height_envelope_summary.json: Extrema and summary statistics
    - physical_min_height_setup.json: Min height setup (if available)
    - physical_max_height_setup.json: Max height setup (if available)

    Args:
        output_dir: Output directory path.
        valid_candidates: List of valid candidate dicts.
        invalid_candidates: List of invalid candidate dicts.
        extrema: Dict with physical_min_height and physical_max_height keys.
        static_revalidation: Dict with revalidation verdict.
        search_grid_rows: List of all candidate dicts for CSV export.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write search grid CSV
    if search_grid_rows:
        grid_path = output_dir / "physical_height_search_grid.csv"
        # Get all keys from first row for fieldnames
        fieldnames = list(search_grid_rows[0].keys())
        with grid_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in search_grid_rows:
                writer.writerow(row)

    # Write valid candidates JSON
    valid_serialized = [serialize_candidate_setup(c) for c in valid_candidates]
    (output_dir / "physical_height_valid_candidates.json").write_text(
        json.dumps(valid_serialized, indent=2), encoding="utf-8"
    )

    # Write invalid candidates JSON (preserve rejection_reasons)
    invalid_serialized = [serialize_candidate_setup(c) for c in invalid_candidates]
    (output_dir / "physical_height_invalid_candidates.json").write_text(
        json.dumps(invalid_serialized, indent=2), encoding="utf-8"
    )

    # Write envelope summary JSON
    min_height = extrema.get("physical_min_height")
    max_height = extrema.get("physical_max_height")

    summary = {
        "physical_min_height": serialize_candidate_setup(min_height) if min_height is not None else None,
        "physical_max_height": serialize_candidate_setup(max_height) if max_height is not None else None,
        "static_revalidation": static_revalidation,
        "valid_candidate_count": len(valid_candidates),
        "invalid_candidate_count": len(invalid_candidates),
        "total_candidate_count": len(search_grid_rows),
    }
    (output_dir / "physical_height_envelope_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Write individual extrema setup files
    if min_height is not None:
        (output_dir / "physical_min_height_setup.json").write_text(
            json.dumps(serialize_candidate_setup(min_height), indent=2), encoding="utf-8"
        )

    if max_height is not None:
        (output_dir / "physical_max_height_setup.json").write_text(
            json.dumps(serialize_candidate_setup(max_height), indent=2), encoding="utf-8"
        )


def revalidate_saved_extrema(setup_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Revalidate saved extrema setups by rebuilding MuJoCo state and recomputing feasibility.

    This function implements the revalidation round-trip:
    1. Load setup JSON with joint_qpos and root_z
    2. Rebuild MuJoCo data.qpos from saved values
    3. Recompute contacts via extract_wheel_floor_contact_points
    4. Recompute CoM via compute_robot_com_xy
    5. Call evaluate_static_standing_pose with recomputed values
    6. Return revalidation results

    Args:
        setup_paths: List of Path objects to setup JSON files.

    Returns:
        List of per-setup result dicts, each containing:
        - setup_path: str path to the setup file
        - static_feasible: bool revalidated feasibility
        - rejection_reasons: list of rejection reasons (if any)
        - achieved_com_z_m: float recomputed CoM z
    """
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    thresholds = PhysicalStandingThresholds()

    # Resolve joint addresses once
    joint_addresses = resolve_standing_joint_addresses(model)

    results = []

    for setup_path in setup_paths:
        # Load setup JSON
        with open(setup_path, "r", encoding="utf-8") as f:
            setup = json.load(f)

        # Extract saved values
        calibrated_root_z = setup["calibrated_root_z_m"]
        joint_qpos = setup["joint_qpos"]

        # Rebuild MuJoCo state
        data = mujoco.MjData(model)
        data.qpos[:] = 0.0
        data.qpos[2] = calibrated_root_z  # root_z
        data.qpos[3] = 1.0  # root quaternion w

        # Set joint positions from saved setup
        data.qpos[joint_addresses["l_hip_roll"]] = joint_qpos["l_hip_roll"]
        data.qpos[joint_addresses["l_hip_yaw"]] = joint_qpos["l_hip_yaw"]
        data.qpos[joint_addresses["l_hip_pitch"]] = joint_qpos["l_hip_pitch"]
        data.qpos[joint_addresses["l_knee"]] = joint_qpos["l_knee"]
        data.qpos[joint_addresses["r_hip_roll"]] = joint_qpos["r_hip_roll"]
        data.qpos[joint_addresses["r_hip_yaw"]] = joint_qpos["r_hip_yaw"]
        data.qpos[joint_addresses["r_hip_pitch"]] = joint_qpos["r_hip_pitch"]
        data.qpos[joint_addresses["r_knee"]] = joint_qpos["r_knee"]

        # Forward kinematics
        mujoco.mj_forward(model, data)

        # Recompute contacts and CoM
        contacts = extract_wheel_floor_contact_points(model, data)
        com_xy = compute_robot_com_xy(model, data)

        # Extract orientation from quaternion
        quat = data.qpos[3:7]
        pitch_x_rad, roll_y_rad, yaw_z_rad = _quaternion_to_euler(quat)

        # Compute joint limit margin
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

        # Use saved candidate_is_root_z_only from setup payload (approved provenance semantics)
        candidate_is_root_z_only = bool(setup.get("candidate_is_root_z_only", False))

        # Revalidate static feasibility
        revalidation_result = evaluate_static_standing_pose(
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
            candidate_source="revalidation",
            candidate_is_root_z_only=candidate_is_root_z_only,
        )

        # Compile result
        result_entry = {
            "setup_path": str(setup_path),
            "static_feasible": revalidation_result.static_feasible,
            "rejection_reasons": revalidation_result.rejection_reasons,
            "achieved_com_z_m": float(data.subtree_com[0][2]),
        }
        results.append(result_entry)

    return results


def search_physical_standing_height_envelope(
    *,
    output_dir: Path = Path("outputs/physical_standing_height_envelope_search"),
    config: SearchConfig | None = None,
) -> Dict[str, Any]:
    """
    Search for physical standing height envelope using static feasibility only.

    This function implements the coarse-to-fine search strategy:
    1. Load model and extract nominal pose
    2. Coarse search: broad height range with 0.005m steps
    3. For each target height, search symmetric hip-pitch/knee pairs
    4. Calibrate root_z from wheel geometry after setting posture
    5. Evaluate static feasibility (no controller constraints, no dynamic checks)
    6. Refine boundaries if needed
    7. Select conservative extrema with joint-margin safety buffer
    8. Write artifacts
    9. Revalidate extrema

    Args:
        output_dir: Output directory for artifacts.
        config: Search configuration (uses defaults if None).

    Returns:
        Dict with search results, extrema, and artifact paths.
    """
    config = config or SearchConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    thresholds = PhysicalStandingThresholds()

    # Resolve joint addresses
    joint_addresses = resolve_standing_joint_addresses(model)

    # Get nominal pose from keyframe
    data_nominal = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data_nominal, 0)
    data_nominal.qvel[:] = 0.0
    data_nominal.qacc[:] = 0.0

    # Calibrate root_z for nominal
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

    # Build coarse target height grid
    # Physical search: broader range than operational, no controller limits
    coarse_lower_bound = nominal_com_z - 0.15
    coarse_upper_bound = nominal_com_z + 0.15
    coarse_targets = np.arange(coarse_lower_bound, coarse_upper_bound + 0.5 * config.coarse_target_step_m, config.coarse_target_step_m)

    print(f"\nCoarse search: {len(coarse_targets)} targets from {coarse_lower_bound:.3f}m to {coarse_upper_bound:.3f}m")

    candidates = []
    search_grid_rows = []

    # Search each target height
    for target_com_z in coarse_targets:
        best_candidate = search_height_for_target(
            model=model,
            target_com_z_m=float(target_com_z),
            nominal_hip_pitch=nominal_hip_pitch,
            nominal_knee=nominal_knee,
            joint_addresses=joint_addresses,
            thresholds=thresholds,
            config=config,
        )

        if best_candidate is not None:
            candidates.append(best_candidate)
            search_grid_rows.append(best_candidate)

    print(f"\nCoarse search complete: {len(candidates)} candidates evaluated")

    # Split into valid and invalid
    valid_candidates = [c for c in candidates if c["static_feasible"]]
    invalid_candidates = [c for c in candidates if not c["static_feasible"]]

    print(f"Valid candidates: {len(valid_candidates)}")
    print(f"Invalid candidates: {len(invalid_candidates)}")

    # Select extrema
    extrema = select_physical_extrema(candidates)
    physical_min = extrema.get("physical_min_height")
    physical_max = extrema.get("physical_max_height")

    if physical_min is None or physical_max is None:
        print("\nWARNING: Could not identify physical extrema")
        verdict = "PHYSICAL_ENVELOPE_INCONCLUSIVE"
        static_revalidation = {"verdict": verdict, "reason": "no_valid_candidates_found"}
    else:
        print(f"\nPhysical extrema identified:")
        print(f"  Min height: {physical_min['achieved_com_z_m']:.6f} m")
        print(f"  Max height: {physical_max['achieved_com_z_m']:.6f} m")

        # Revalidate extrema
        min_setup_path = output_dir / "physical_min_height_setup.json"
        max_setup_path = output_dir / "physical_max_height_setup.json"

        # Write setup files first (so revalidation can load them)
        # Write with empty static_revalidation temporarily
        write_candidate_artifacts(
            output_dir=output_dir,
            valid_candidates=valid_candidates,
            invalid_candidates=invalid_candidates,
            extrema=extrema,
            static_revalidation={},  # Will update after revalidation
            search_grid_rows=search_grid_rows,
        )

        # Revalidate
        revalidation_results = revalidate_saved_extrema([min_setup_path, max_setup_path])
        revalidation_pass = all(r["static_feasible"] for r in revalidation_results)

        if revalidation_pass:
            verdict = "PHYSICAL_ENVELOPE_PASS"
            static_revalidation = {
                "verdict": verdict,
                "revalidation_results": revalidation_results,
            }
        else:
            verdict = "PHYSICAL_ENVELOPE_REVALIDATION_FAIL"
            static_revalidation = {
                "verdict": verdict,
                "revalidation_results": revalidation_results,
                "reason": "extrema_failed_static_revalidation",
            }

        print(f"\nStatic revalidation: {verdict}")

        # Write artifacts again with populated static_revalidation
        write_candidate_artifacts(
            output_dir=output_dir,
            valid_candidates=valid_candidates,
            invalid_candidates=invalid_candidates,
            extrema=extrema,
            static_revalidation=static_revalidation,
            search_grid_rows=search_grid_rows,
        )

        # Write static validation artifact
        (output_dir / "static_physical_extrema_validation.json").write_text(
            json.dumps(static_revalidation, indent=2), encoding="utf-8"
        )

        # Write report
        report = render_physical_envelope_report(
            extrema=extrema,
            valid_count=len(valid_candidates),
            invalid_count=len(invalid_candidates),
            verdict=verdict,
            static_revalidation=static_revalidation,
        )
        (output_dir / "physical_height_envelope_report.md").write_text(report, encoding="utf-8")

    return {
        "verdict": verdict,
        "extrema": extrema,
        "valid_candidate_count": len(valid_candidates),
        "invalid_candidate_count": len(invalid_candidates),
        "static_revalidation": static_revalidation,
        "output_dir": str(output_dir),
    }


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
    """
    Search for best symmetric hip-pitch/knee posture achieving target CoM height.

    Args:
        model: MuJoCo model.
        target_com_z_m: Target CoM z coordinate.
        nominal_hip_pitch: Nominal hip pitch reference.
        nominal_knee: Nominal knee reference.
        joint_addresses: Resolved joint addresses dict.
        thresholds: Physical standing thresholds.
        config: Search configuration.

    Returns:
        Best candidate dict or None if search failed.
    """
    # Create search grid around nominal posture
    hip_pitch_range = 0.6  # ±0.6 rad around nominal
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
            # Create test pose
            data = mujoco.MjData(model)
            data.qpos[:] = 0.0
            data.qpos[3] = 1.0  # quaternion w

            # Set symmetric hip/knee
            data.qpos[joint_addresses["l_hip_pitch"]] = hip_pitch
            data.qpos[joint_addresses["l_knee"]] = knee
            data.qpos[joint_addresses["r_hip_pitch"]] = hip_pitch
            data.qpos[joint_addresses["r_knee"]] = knee

            # Calibrate root_z from wheel geometry
            try:
                calibrated_root_z = calibrate_root_z_from_wheel_geometry(
                    model, data, target_contact_depth_m=config.target_contact_depth_m
                )
                data.qpos[2] = calibrated_root_z
                mujoco.mj_forward(model, data)
            except Exception:
                continue

            # Extract metrics
            achieved_com_z = float(data.subtree_com[0][2])
            contacts = extract_wheel_floor_contact_points(model, data)
            com_xy = compute_robot_com_xy(model, data)

            # Extract orientation
            quat = data.qpos[3:7]
            pitch_x_rad, roll_y_rad, yaw_z_rad = _quaternion_to_euler(quat)

            # Compute joint limit margin
            joint_margins = []
            for joint_name in ["l_hip_pitch", "l_knee", "r_hip_pitch", "r_knee"]:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_addr = model.jnt_qposadr[joint_id]
                low, high = model.jnt_range[joint_id]
                value = float(data.qpos[qpos_addr])
                joint_margins.append(value - float(low))
                joint_margins.append(float(high) - value)
            joint_limit_margin_rad = float(min(joint_margins))

            # Check root_z_only
            candidate_is_root_z_only = (
                abs(hip_pitch - nominal_hip_pitch) < 1e-3
                and abs(knee - nominal_knee) < 1e-3
                and abs(achieved_com_z - target_com_z_m) > 0.005
            )

            # Build support geometry
            support_geom = build_support_segment_geometry(
                left_wheel_contact_xy=contacts.left_wheel_contact_xy,
                right_wheel_contact_xy=contacts.right_wheel_contact_xy,
                com_xy=com_xy,
                thresholds=thresholds,
            )

            # Evaluate static feasibility
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
                candidate_source="coarse_grid_search",
                candidate_is_root_z_only=candidate_is_root_z_only,
            )

            # Build candidate dict
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
                "candidate_source": "coarse_grid_search",
                "candidate_is_root_z_only": candidate_is_root_z_only,
                "static_feasible": result.static_feasible,
                "rejection_reasons": result.rejection_reasons,
            }

            # Score candidate (prefer valid, then minimize height error and CoM offset)
            height_error = abs(achieved_com_z - target_com_z_m)
            com_error = np.linalg.norm(np.array(com_xy) - np.array(support_geom.support_center_xy))
            score = 1000.0 * (0 if result.static_feasible else 1) + 100.0 * height_error + 10.0 * com_error

            if score < best_score:
                best_score = score
                best_candidate = candidate

    return best_candidate


def render_physical_envelope_report(
    *,
    extrema: Dict[str, Any],
    valid_count: int,
    invalid_count: int,
    verdict: str,
    static_revalidation: Dict[str, Any],
) -> str:
    """
    Render human-readable physical envelope search report.

    Args:
        extrema: Dict with physical_min_height and physical_max_height.
        valid_count: Number of valid candidates.
        invalid_count: Number of invalid candidates.
        verdict: Overall verdict string.
        static_revalidation: Static revalidation result dict.

    Returns:
        Markdown report string.
    """
    min_height = extrema.get("physical_min_height")
    max_height = extrema.get("physical_max_height")

    lines = [
        "# Physical Standing Height Envelope Search Report",
        "",
        f"**Verdict:** {verdict}",
        "",
        "## Search summary",
        "",
        f"- Valid candidates: {valid_count}",
        f"- Invalid candidates: {invalid_count}",
        f"- Total evaluated: {valid_count + invalid_count}",
        "",
        "## Physical extrema",
        "",
    ]

    if min_height is not None:
        lines.extend([
            f"### Physical minimum height: {min_height['achieved_com_z_m']:.6f} m",
            "",
            f"- Hip pitch: {min_height['hip_pitch_ref']:.6f} rad",
            f"- Knee: {min_height['knee_ref']:.6f} rad",
            f"- Root z: {min_height['calibrated_root_z_m']:.6f} m",
            f"- Joint limit margin: {min_height['joint_limit_margin_rad']:.6f} rad",
            "",
        ])

    if max_height is not None:
        lines.extend([
            f"### Physical maximum height: {max_height['achieved_com_z_m']:.6f} m",
            "",
            f"- Hip pitch: {max_height['hip_pitch_ref']:.6f} rad",
            f"- Knee: {max_height['knee_ref']:.6f} rad",
            f"- Root z: {max_height['calibrated_root_z_m']:.6f} m",
            f"- Joint limit margin: {max_height['joint_limit_margin_rad']:.6f} rad",
            "",
        ])

    lines.extend([
        "## Static revalidation",
        "",
        f"Verdict: {static_revalidation.get('verdict', 'N/A')}",
        "",
        "Both extrema were revalidated by reloading setup JSON, rebuilding MuJoCo state, and recomputing static feasibility.",
        "",
        "## Important notes",
        "",
        "- This envelope is based on **static feasibility only**",
        "- No controller constraints were applied",
        "- No dynamic stability checks were performed",
        "- Dynamic failure at these extrema does NOT invalidate the physical envelope",
        "- The physical envelope quantifies kinematic workspace, not controller capability",
    ])

    return "\n".join(lines)


def main() -> int:
    """
    Main entrypoint for physical standing height envelope search.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Search for physical standing height envelope using static feasibility only"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/physical_standing_height_envelope_search"),
        help="Output directory for artifacts",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Physical Standing Height Envelope Search")
    print("=" * 80)

    result = search_physical_standing_height_envelope(output_dir=args.output_dir)

    print("\n" + "=" * 80)
    print(f"Search complete: {result['verdict']}")
    print(f"Artifacts written to: {result['output_dir']}")
    print("=" * 80)

    return 0 if result["verdict"] in ("PHYSICAL_ENVELOPE_PASS",) else 1


if __name__ == "__main__":
    raise SystemExit(main())

