"""
Tests for physical standing height envelope geometry primitives.

Task 1: Support-segment geometry tests only.
Task 2: Contact extraction and static feasibility tests.
"""

import pytest
import math
import mujoco
import numpy as np

from wheeled_biped.validation.physical_standing_height_envelope import (
    PhysicalStandingThresholds,
    SupportSegmentGeometry,
    build_support_segment_geometry,
    ROBOT_COM_CONVENTION,
    WheelContactPoints,
    StaticStandingFeasibilityResult,
    compute_robot_com_xy,
    extract_wheel_floor_contact_points,
    evaluate_static_standing_pose,
)
from wheeled_biped.utils.config import get_model_path
from scripts.search_physical_standing_height_envelope import calibrate_root_z_from_wheel_geometry


def _quaternion_to_euler(quat):
    """
    Convert quaternion to Euler angles (pitch, roll, yaw).

    Test helper to avoid duplicated conversion code.

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


def test_left_right_wheel_segment_contains_centered_com():
    """
    A CoM centered laterally between the two wheels should be
    contained within the left-right support segment.

    Exact example from Task 1 plan:
    left=(-0.10, 0.00), right=(0.10, 0.00), com=(0.00, 0.01)
    Expected: projection_fraction = 0.5, lateral_offset = 0.0, sagittal_offset = 0.01
    """
    left_wheel_contact_xy = (-0.10, 0.00)
    right_wheel_contact_xy = (0.10, 0.00)
    com_xy = (0.00, 0.01)

    thresholds = PhysicalStandingThresholds()

    geom = build_support_segment_geometry(
        left_wheel_contact_xy=left_wheel_contact_xy,
        right_wheel_contact_xy=right_wheel_contact_xy,
        com_xy=com_xy,
        thresholds=thresholds,
    )

    # Exact assertions from plan
    assert abs(geom.com_projection_fraction_on_wheel_segment - 0.5) < 1e-6
    assert abs(geom.com_lateral_offset_from_support_center_m - 0.0) < 1e-6
    assert abs(geom.com_sagittal_offset_from_support_center_m - 0.01) < 1e-6

    # CoM should be contained (projection inside segment)
    assert geom.com_projection_inside_wheel_segment
    assert geom.valid


def test_front_back_wheel_segment_uses_same_projection_method():
    """
    When wheels are positioned front-back (sagittal axis), the same
    projection method should work correctly.

    Exact example from Task 1 plan:
    left=(0.00, -0.12), right=(0.00, 0.12), com=(0.01, 0.00)
    Expected: projection_fraction = 0.5, abs(sagittal_offset) = 0.01
    """
    left_wheel_contact_xy = (0.00, -0.12)
    right_wheel_contact_xy = (0.00, 0.12)
    com_xy = (0.01, 0.00)

    thresholds = PhysicalStandingThresholds()

    geom = build_support_segment_geometry(
        left_wheel_contact_xy=left_wheel_contact_xy,
        right_wheel_contact_xy=right_wheel_contact_xy,
        com_xy=com_xy,
        thresholds=thresholds,
    )

    # Exact assertions from plan
    assert abs(geom.com_projection_fraction_on_wheel_segment - 0.5) < 1e-6
    assert abs(abs(geom.com_sagittal_offset_from_support_center_m) - 0.01) < 1e-6

    # Should have valid geometry
    assert geom.valid
    assert geom.com_projection_inside_wheel_segment


def test_projection_outside_segment_fails_containment():
    """
    A CoM far outside the support segment should not be contained.

    Exact example from Task 1 plan:
    left=(-0.10, 0.00), right=(0.10, 0.00), com=(0.25, 0.00)
    Expected: inside = False, fraction > 1.0
    """
    left_wheel_contact_xy = (-0.10, 0.00)
    right_wheel_contact_xy = (0.10, 0.00)
    com_xy = (0.25, 0.00)

    thresholds = PhysicalStandingThresholds()

    geom = build_support_segment_geometry(
        left_wheel_contact_xy=left_wheel_contact_xy,
        right_wheel_contact_xy=right_wheel_contact_xy,
        com_xy=com_xy,
        thresholds=thresholds,
    )

    # Exact assertions from plan
    assert not geom.com_projection_inside_wheel_segment
    assert geom.com_projection_fraction_on_wheel_segment > 1.0

    # Should not be valid
    assert not geom.valid
    assert "projection_outside_wheel_segment" in geom.rejection_reasons


def test_degenerate_wheel_segment_is_rejected():
    """
    When left and right wheels are at nearly the same position (degenerate),
    the geometry should be marked as invalid with the exact rejection reasons.

    Exact example from Task 1 plan:
    left=(0.00, 0.00), right=(1e-9, 1e-9), com=(0.00, 0.00)
    Expected: valid = False, "degenerate_wheel_support_segment" in rejection_reasons
    """
    left_wheel_contact_xy = (0.00, 0.00)
    right_wheel_contact_xy = (1e-9, 1e-9)
    com_xy = (0.00, 0.00)

    thresholds = PhysicalStandingThresholds()

    geom = build_support_segment_geometry(
        left_wheel_contact_xy=left_wheel_contact_xy,
        right_wheel_contact_xy=right_wheel_contact_xy,
        com_xy=com_xy,
        thresholds=thresholds,
    )

    # Exact assertions from plan
    assert not geom.valid
    assert "degenerate_wheel_support_segment" in geom.rejection_reasons
    assert "support_geometry_invalid" in geom.rejection_reasons


def test_support_segment_geometry_to_dict():
    """
    SupportSegmentGeometry.to_dict() should produce a serializable dict.
    """
    left_wheel_contact_xy = (-0.10, 0.00)
    right_wheel_contact_xy = (0.10, 0.00)
    com_xy = (0.00, 0.01)

    thresholds = PhysicalStandingThresholds()

    geom = build_support_segment_geometry(
        left_wheel_contact_xy=left_wheel_contact_xy,
        right_wheel_contact_xy=right_wheel_contact_xy,
        com_xy=com_xy,
        thresholds=thresholds,
    )

    d = geom.to_dict()

    # Should have expected keys from the planned API
    assert "valid" in d
    assert "rejection_reasons" in d
    assert "support_center_xy" in d
    assert "wheel_line_direction_xy" in d
    assert "support_error_direction_xy" in d
    assert "com_projection_fraction_on_wheel_segment" in d
    assert "com_projection_inside_wheel_segment" in d
    assert "com_lateral_offset_from_support_center_m" in d
    assert "com_sagittal_offset_from_support_center_m" in d
    assert "segment_length_m" in d
    assert "min_endpoint_margin_m" in d

    # Values should be Python-native types
    assert isinstance(d["valid"], bool)
    assert isinstance(d["rejection_reasons"], list)
    assert isinstance(d["support_center_xy"], tuple)
    assert isinstance(d["com_lateral_offset_from_support_center_m"], float)


def test_thresholds_dataclass_creation():
    """PhysicalStandingThresholds should have the planned default values."""
    thresholds = PhysicalStandingThresholds()

    assert thresholds.projection_tolerance == 1e-6
    assert thresholds.preferred_sagittal_offset_m == 0.01
    assert thresholds.max_sagittal_offset_m == 0.02
    assert thresholds.max_pitch_abs_rad == 0.10
    assert thresholds.max_roll_abs_rad == 0.05
    assert thresholds.max_yaw_abs_rad == 0.10
    assert thresholds.min_joint_limit_margin_rad == 0.05
    assert thresholds.degenerate_segment_length_m == 1e-6


def test_thresholds_dataclass_custom_values():
    """PhysicalStandingThresholds should accept custom values."""
    thresholds = PhysicalStandingThresholds(
        projection_tolerance=1e-5,
        preferred_sagittal_offset_m=0.02,
        max_sagittal_offset_m=0.03,
    )

    assert thresholds.projection_tolerance == 1e-5
    assert thresholds.preferred_sagittal_offset_m == 0.02
    assert thresholds.max_sagittal_offset_m == 0.03


# ============================================================================
# Task 2: Contact extraction and static feasibility tests
# ============================================================================


def test_robot_com_convention_is_documented():
    """ROBOT_COM_CONVENTION should be set to whole_robot, not torso-only subtree."""
    assert ROBOT_COM_CONVENTION == "whole_robot"


def test_compute_robot_com_xy_uses_documented_project_convention():
    """
    compute_robot_com_xy should use whole-robot CoM, not torso-only subtree.

    This test verifies that the CoM computation follows the documented
    project convention of using data.subtree_com[0] (whole robot), not
    data.subtree_com[1] (torso-only subtree).
    """
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Set a standing pose
    data.qpos[2] = 0.55  # root_z
    mujoco.mj_forward(model, data)

    # Get whole-robot CoM (body 0 is world root)
    whole_robot_com = data.subtree_com[0]

    # Test function should match whole-robot convention
    com_xy = compute_robot_com_xy(model, data)

    assert abs(com_xy[0] - whole_robot_com[0]) < 1e-9
    assert abs(com_xy[1] - whole_robot_com[1]) < 1e-9


def test_extract_wheel_contacts_from_calibrated_static_pose():
    """
    extract_wheel_floor_contact_points should extract actual MuJoCo contacts
    from a calibrated static pose with wheels on the floor.

    This test validates STRONG contact requirements:
    - left_wheel_contact must be True
    - right_wheel_contact must be True
    - left_wheel_contact_xy must not be None
    - right_wheel_contact_xy must not be None
    - non_wheel_floor_contact_count must be 0
    """
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Set a nominal standing pose with symmetric leg configuration
    # Use smaller angles to avoid leg-floor contacts while maintaining wheel contact
    data.qpos[:] = 0.0  # Zero all joints
    data.qpos[2] = 0.60  # Initial root_z guess
    data.qpos[3] = 1.0   # root quaternion w

    # Set symmetric hip/knee angles for standing
    # Joint order: l_hip_roll(7), l_hip_yaw(8), l_hip_pitch(9), l_knee(10), l_wheel(11)
    #              r_hip_roll(12), r_hip_yaw(13), r_hip_pitch(14), r_knee(15), r_wheel(16)
    # Use smaller angles to keep legs off the floor
    hip_pitch = 0.3
    knee = -0.6

    data.qpos[9] = hip_pitch   # l_hip_pitch
    data.qpos[10] = knee       # l_knee
    data.qpos[14] = hip_pitch  # r_hip_pitch
    data.qpos[15] = knee       # r_knee

    # Calibrate root_z from wheel geometry to place wheels on floor
    calibrated_root_z = calibrate_root_z_from_wheel_geometry(model, data, target_contact_depth_m=-5e-4)
    data.qpos[2] = calibrated_root_z

    # Forward kinematics and contact detection
    mujoco.mj_forward(model, data)

    # Extract wheel-floor contacts
    contacts = extract_wheel_floor_contact_points(model, data)

    # STRONG contact test: both wheels MUST be in contact
    assert contacts.left_wheel_contact, "Left wheel must have floor contact"
    assert contacts.right_wheel_contact, "Right wheel must have floor contact"
    assert contacts.left_wheel_contact_xy is not None, "Left wheel contact XY must not be None"
    assert contacts.right_wheel_contact_xy is not None, "Right wheel contact XY must not be None"
    assert contacts.non_wheel_floor_contact_count == 0, "No non-wheel floor contacts allowed"
    assert len(contacts.rejection_reasons) == 0, "No rejection reasons for valid contact"

    # Contact points should be reasonable
    assert abs(contacts.left_wheel_contact_xy[0]) < 0.5
    assert abs(contacts.left_wheel_contact_xy[1]) < 0.5
    assert abs(contacts.right_wheel_contact_xy[0]) < 0.5
    assert abs(contacts.right_wheel_contact_xy[1]) < 0.5

    # Should be able to serialize
    d = contacts.to_dict()
    assert "left_wheel_contact" in d
    assert "right_wheel_contact" in d
    assert "left_wheel_contact_xy" in d
    assert "right_wheel_contact_xy" in d
    assert "non_wheel_floor_contact_count" in d


def test_missing_wheel_contact_geometry_is_reported():
    """
    extract_wheel_floor_contact_points should report when wheel-floor
    contacts are missing or incomplete.

    This is the separate test for INVALID cases where contacts are missing.
    """
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Set robot high in the air with no floor contact
    data.qpos[:] = 0.0
    data.qpos[2] = 5.0  # 5m above ground - no contact
    data.qpos[3] = 1.0  # root quaternion w

    mujoco.mj_forward(model, data)

    contacts = extract_wheel_floor_contact_points(model, data)

    # Should report missing contacts
    assert not contacts.left_wheel_contact, "Left wheel should not have contact"
    assert not contacts.right_wheel_contact, "Right wheel should not have contact"
    assert "missing_wheel_floor_contact_geometry" in contacts.rejection_reasons


def test_root_z_only_candidates_are_rejected():
    """
    evaluate_static_standing_pose should reject poses that only set root_z
    without proper leg configuration.

    This test verifies the root_z_only_candidate_not_allowed rejection reason.
    """
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    thresholds = PhysicalStandingThresholds()

    # Set only root_z, leave all joints at zero (degenerate pose)
    data.qpos[:] = 0.0
    data.qpos[2] = 0.50
    data.qpos[3] = 1.0  # root quaternion w

    mujoco.mj_forward(model, data)

    # Extract contacts and CoM
    contacts = extract_wheel_floor_contact_points(model, data)
    com_xy = compute_robot_com_xy(model, data)

    # Extract orientation
    quat = data.qpos[3:7]
    pitch_x_rad, roll_y_rad, yaw_z_rad = _quaternion_to_euler(quat)

    # Check if all joints are at zero
    joint_positions = data.qpos[7:]
    candidate_is_root_z_only = bool(np.allclose(joint_positions, 0.0, atol=1e-6))

    # Evaluate with root_z_only flag
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
        joint_limit_margin_rad=1.0,  # Dummy value
        thresholds=thresholds,
        candidate_source="test",
        candidate_is_root_z_only=candidate_is_root_z_only,
    )

    # Should be rejected with the specific rejection reason
    assert not result.static_feasible
    assert "root_z_only_candidate_not_allowed" in result.rejection_reasons


def test_large_pitch_roll_yaw_are_reported():
    """
    evaluate_static_standing_pose should detect and report when pitch, roll,
    or yaw exceed configured thresholds.

    This test verifies the pitch_roll_yaw_out_of_bounds rejection reason.
    """
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Set strict thresholds
    thresholds = PhysicalStandingThresholds(
        max_pitch_abs_rad=0.05,
        max_roll_abs_rad=0.03,
        max_yaw_abs_rad=0.05,
    )

    # Set a standing pose with large pitch via quaternion
    # Pitch rotation around Y axis
    pitch_angle = 0.15  # 15 degrees, exceeds 0.05 rad threshold
    data.qpos[:] = 0.0
    data.qpos[2] = 0.50
    data.qpos[3] = np.cos(pitch_angle / 2)  # qw
    data.qpos[4] = 0.0  # qx
    data.qpos[5] = np.sin(pitch_angle / 2)  # qy (pitch)
    data.qpos[6] = 0.0  # qz

    # Set some leg configuration
    # Joint order: l_hip_roll(7), l_hip_yaw(8), l_hip_pitch(9), l_knee(10), l_wheel(11)
    #              r_hip_roll(12), r_hip_yaw(13), r_hip_pitch(14), r_knee(15), r_wheel(16)
    data.qpos[9] = 0.3   # l_hip_pitch
    data.qpos[10] = -0.6  # l_knee
    data.qpos[14] = 0.3  # r_hip_pitch
    data.qpos[15] = -0.6 # r_knee

    mujoco.mj_forward(model, data)

    # Extract contacts and CoM
    contacts = extract_wheel_floor_contact_points(model, data)
    com_xy = compute_robot_com_xy(model, data)

    # Extract orientation
    quat = data.qpos[3:7]
    pitch_x_rad, roll_y_rad, yaw_z_rad = _quaternion_to_euler(quat)

    # Check if all joints are at zero
    joint_positions = data.qpos[7:]
    candidate_is_root_z_only = bool(np.allclose(joint_positions, 0.0, atol=1e-6))

    # Evaluate
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
        joint_limit_margin_rad=1.0,  # Dummy value
        thresholds=thresholds,
        candidate_source="test",
        candidate_is_root_z_only=candidate_is_root_z_only,
    )

    # Should report orientation violation
    assert "pitch_roll_yaw_out_of_bounds" in result.rejection_reasons

    # Result should be serializable
    d = result.to_dict()
    assert "static_feasible" in d
    assert "rejection_reasons" in d


# ============================================================================
# Task 3: Search helper tests
# ============================================================================


def test_search_script_imports_shared_utility():
    """
    scripts/search_physical_standing_height_envelope.py must import
    build_support_segment_geometry from the shared utility module,
    not define it locally.

    This ensures proper code reuse and avoids duplication.
    """
    # Read the source file text using portable Path-based approach
    from pathlib import Path
    script_path = Path(__file__).parent.parent / "scripts" / "search_physical_standing_height_envelope.py"
    with open(script_path, "r") as f:
        source_text = f.read()

    # Must contain import from shared utility
    assert "from wheeled_biped.validation.physical_standing_height_envelope import" in source_text, \
        "search script must import from shared utility"

    # Must NOT define build_support_segment_geometry locally
    assert "def build_support_segment_geometry" not in source_text, \
        "search script must not define build_support_segment_geometry locally"


def test_resolve_joint_addresses_reads_model_names_not_hardcoded_signs():
    """
    resolve_standing_joint_addresses must read actual model joint names and axes,
    not rely on hardcoded joint index offsets or sign conventions.

    This test verifies that the function reads from model.joint() and model.jnt_axis,
    not from hardcoded constants.
    """
    from scripts.search_physical_standing_height_envelope import resolve_standing_joint_addresses

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))

    addresses = resolve_standing_joint_addresses(model)

    # Should return a dict with joint indices and axis keys
    assert isinstance(addresses, dict), "Must return a dict"

    # Must have all joint index keys
    assert "l_hip_pitch" in addresses, "Must have l_hip_pitch key"
    assert "l_knee" in addresses, "Must have l_knee key"
    assert "r_hip_pitch" in addresses, "Must have r_hip_pitch key"
    assert "r_knee" in addresses, "Must have r_knee key"
    assert "l_hip_yaw" in addresses, "Must have l_hip_yaw key"
    assert "r_hip_yaw" in addresses, "Must have r_hip_yaw key"
    assert "l_hip_roll" in addresses, "Must have l_hip_roll key"
    assert "r_hip_roll" in addresses, "Must have r_hip_roll key"

    # Indices should be valid qpos indices (not action indices)
    assert 0 <= addresses["l_hip_pitch"] < model.nq
    assert 0 <= addresses["l_knee"] < model.nq
    assert 0 <= addresses["r_hip_pitch"] < model.nq
    assert 0 <= addresses["r_knee"] < model.nq

    # Left and right indices should differ
    assert addresses["l_hip_pitch"] != addresses["r_hip_pitch"]
    assert addresses["l_knee"] != addresses["r_knee"]

    # Should also provide axis directions read from model
    assert "l_hip_pitch_axis" in addresses, "Must have l_hip_pitch_axis key"
    assert "l_knee_axis" in addresses, "Must have l_knee_axis key"
    assert "r_hip_pitch_axis" in addresses, "Must have r_hip_pitch_axis key"
    assert "r_knee_axis" in addresses, "Must have r_knee_axis key"
    assert "l_hip_yaw_axis" in addresses, "Must have l_hip_yaw_axis key"
    assert "r_hip_yaw_axis" in addresses, "Must have r_hip_yaw_axis key"
    assert "l_hip_roll_axis" in addresses, "Must have l_hip_roll_axis key"
    assert "r_hip_roll_axis" in addresses, "Must have r_hip_roll_axis key"

    # Axes should be tuples of 3 floats
    assert isinstance(addresses["l_hip_pitch_axis"], tuple)
    assert len(addresses["l_hip_pitch_axis"]) == 3


def test_select_physical_extrema_ignores_dynamic_failure_fields():
    """
    select_physical_extrema must select min/max heights from static_feasible
    candidates only, ignoring any dynamic failure metadata fields.

    This test verifies that the function:
    1. Only considers candidates where static_feasible=True
    2. Ignores candidates with dynamic failure annotations
    3. Returns the minimum and maximum heights from the static_feasible set
    """
    from scripts.search_physical_standing_height_envelope import select_physical_extrema

    # Create mock candidates with mixed static/dynamic validity
    # Structure matches what evaluate_static_standing_pose returns
    candidates = [
        {
            "target_com_z": 0.38,
            "achieved_com_z_m": 0.38,
            "static_feasible": True,
            "dynamic_stable": False,  # Dynamic failure - should be ignored for envelope
            "rejection_reasons": [],
        },
        {
            "target_com_z": 0.40,
            "achieved_com_z_m": 0.40,
            "static_feasible": False,  # Not static feasible - should be excluded
            "dynamic_stable": True,
            "rejection_reasons": ["joint_limit_margin_too_small"],
        },
        {
            "target_com_z": 0.41,
            "achieved_com_z_m": 0.41,
            "static_feasible": True,
            "dynamic_stable": True,
            "rejection_reasons": [],
        },
        {
            "target_com_z": 0.43,
            "achieved_com_z_m": 0.43,
            "static_feasible": True,
            "dynamic_stable": False,  # Dynamic failure - should be ignored for envelope
            "rejection_reasons": [],
        },
    ]

    selected = select_physical_extrema(candidates)

    # Should return a dict with physical_min_height and physical_max_height keys
    assert isinstance(selected, dict), "Must return a dict"
    assert "physical_min_height" in selected, "Must have physical_min_height key"
    assert "physical_max_height" in selected, "Must have physical_max_height key"

    # Should select from static_feasible only: 0.38, 0.41, 0.43
    assert selected["physical_min_height"]["achieved_com_z_m"] == pytest.approx(0.38, abs=0.01), \
        "min_height should be 0.38 (lowest static_feasible)"
    assert selected["physical_max_height"]["achieved_com_z_m"] == pytest.approx(0.43, abs=0.01), \
        "max_height should be 0.43 (highest static_feasible)"


# ============================================================================
# Task 4: Serialization, artifact writing, and revalidation tests
# ============================================================================


def test_serialize_candidate_setup_keeps_required_schema_fields():
    """
    serialize_candidate_setup must produce a JSON-serializable dict with
    the required schema fields for reproducible setup revalidation.

    Required fields from approved plan:
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
    """
    from scripts.search_physical_standing_height_envelope import serialize_candidate_setup

    # Create a mock candidate dict matching approved plan schema
    mock_candidate = {
        "requested_target_com_z_m": 0.50,
        "achieved_com_z_m": 0.495,
        "calibrated_root_z_m": 0.60,
        "hip_pitch_ref": 0.4,
        "knee_ref": -0.8,
        "joint_qpos": {
            "l_hip_roll": 0.0,
            "l_hip_yaw": 0.0,
            "l_hip_pitch": 0.4,
            "l_knee": -0.8,
            "r_hip_roll": 0.0,
            "r_hip_yaw": 0.0,
            "r_hip_pitch": 0.4,
            "r_knee": -0.8,
        },
        "support_geometry": {
            "valid": True,
            "rejection_reasons": [],
            "support_center_xy": (0.0, 0.0),
            "com_lateral_offset_from_support_center_m": 0.001,
            "com_sagittal_offset_from_support_center_m": 0.005,
        },
        "contact_metrics": {
            "left_wheel_contact": True,
            "right_wheel_contact": True,
            "left_wheel_contact_xy": (-0.08, 0.01),
            "right_wheel_contact_xy": (0.08, 0.01),
            "non_wheel_floor_contact_count": 0,
            "rejection_reasons": [],
        },
        "joint_limit_margin_rad": 0.8,
        "candidate_source": "search",
        "candidate_is_root_z_only": False,
        "rejection_reasons": [],
    }

    payload = serialize_candidate_setup(mock_candidate)

    # Must be a dict
    assert isinstance(payload, dict), "Must return a dict"

    # Must have all required schema fields from approved plan
    assert "requested_target_com_z_m" in payload, "Must have requested_target_com_z_m"
    assert "achieved_com_z_m" in payload, "Must have achieved_com_z_m"
    assert "calibrated_root_z_m" in payload, "Must have calibrated_root_z_m"
    assert "hip_pitch_ref" in payload, "Must have hip_pitch_ref"
    assert "knee_ref" in payload, "Must have knee_ref"
    assert "joint_qpos" in payload, "Must have joint_qpos"
    assert "support_geometry" in payload, "Must have support_geometry"
    assert "contact_metrics" in payload, "Must have contact_metrics"
    assert "joint_limit_margin_rad" in payload, "Must have joint_limit_margin_rad"
    assert "candidate_source" in payload, "Must have candidate_source"
    assert "candidate_is_root_z_only" in payload, "Must have candidate_is_root_z_only"
    assert "rejection_reasons" in payload, "Must have rejection_reasons"

    # joint_qpos must have all 8 standing joints
    joint_qpos = payload["joint_qpos"]
    assert isinstance(joint_qpos, dict), "joint_qpos must be a dict"
    required_joints = ["l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee",
                       "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee"]
    for joint_name in required_joints:
        assert joint_name in joint_qpos, f"joint_qpos must have {joint_name}"
        assert isinstance(joint_qpos[joint_name], (int, float)), f"{joint_name} must be numeric"

    # Types must be JSON-serializable
    assert isinstance(payload["requested_target_com_z_m"], (int, float))
    assert isinstance(payload["achieved_com_z_m"], (int, float))
    assert isinstance(payload["calibrated_root_z_m"], (int, float))
    assert isinstance(payload["hip_pitch_ref"], (int, float))
    assert isinstance(payload["knee_ref"], (int, float))
    assert isinstance(payload["joint_limit_margin_rad"], (int, float))
    assert isinstance(payload["candidate_source"], str)
    assert isinstance(payload["candidate_is_root_z_only"], bool)
    assert isinstance(payload["rejection_reasons"], list)


def test_rejection_reasons_are_preserved_in_artifacts():
    """
    write_candidate_artifacts must preserve rejection_reasons in both
    valid and invalid candidate JSON files.

    This test verifies that diagnostic information is not lost during
    serialization, enabling post-search analysis of failure modes.
    """
    from scripts.search_physical_standing_height_envelope import (
        serialize_candidate_setup,
        write_candidate_artifacts,
    )
    from pathlib import Path
    import json
    import tempfile

    # Create valid and invalid mock candidates using approved schema
    valid_candidate = {
        "requested_target_com_z_m": 0.50,
        "achieved_com_z_m": 0.495,
        "calibrated_root_z_m": 0.60,
        "hip_pitch_ref": 0.4,
        "knee_ref": -0.8,
        "joint_qpos": {
            "l_hip_roll": 0.0, "l_hip_yaw": 0.0, "l_hip_pitch": 0.4, "l_knee": -0.8,
            "r_hip_roll": 0.0, "r_hip_yaw": 0.0, "r_hip_pitch": 0.4, "r_knee": -0.8,
        },
        "support_geometry": {"valid": True, "rejection_reasons": []},
        "contact_metrics": {"left_wheel_contact": True, "right_wheel_contact": True,
                          "non_wheel_floor_contact_count": 0, "rejection_reasons": []},
        "joint_limit_margin_rad": 0.8,
        "candidate_source": "search",
        "candidate_is_root_z_only": False,
        "rejection_reasons": [],
    }

    invalid_candidate = {
        "requested_target_com_z_m": 0.35,
        "achieved_com_z_m": 0.348,
        "calibrated_root_z_m": 0.45,
        "hip_pitch_ref": 0.2,
        "knee_ref": -0.4,
        "joint_qpos": {
            "l_hip_roll": 0.0, "l_hip_yaw": 0.0, "l_hip_pitch": 0.2, "l_knee": -0.4,
            "r_hip_roll": 0.0, "r_hip_yaw": 0.0, "r_hip_pitch": 0.2, "r_knee": -0.4,
        },
        "support_geometry": {"valid": True, "rejection_reasons": []},
        "contact_metrics": {"left_wheel_contact": True, "right_wheel_contact": True,
                          "non_wheel_floor_contact_count": 0, "rejection_reasons": []},
        "joint_limit_margin_rad": 0.03,
        "candidate_source": "search",
        "candidate_is_root_z_only": False,
        "rejection_reasons": ["joint_limit_margin_too_small", "pitch_roll_yaw_out_of_bounds"],
    }

    extrema = {
        "physical_min_height": valid_candidate,
        "physical_max_height": valid_candidate,
    }

    static_revalidation = {"verdict": "PASS"}
    search_grid_rows = [valid_candidate, invalid_candidate]

    # Write artifacts to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        write_candidate_artifacts(output_dir, [valid_candidate], [invalid_candidate],
                                 extrema, static_revalidation, search_grid_rows)

        # Check that artifacts were created
        assert (output_dir / "physical_height_valid_candidates.json").exists()
        assert (output_dir / "physical_height_invalid_candidates.json").exists()

        # Load and verify rejection_reasons are preserved
        with open(output_dir / "physical_height_invalid_candidates.json") as f:
            invalid_data = json.load(f)

        assert len(invalid_data) == 1, "Should have 1 invalid candidate"
        assert "rejection_reasons" in invalid_data[0], "Must preserve rejection_reasons"
        assert len(invalid_data[0]["rejection_reasons"]) == 2, "Should have 2 rejection reasons"
        assert "joint_limit_margin_too_small" in invalid_data[0]["rejection_reasons"]
        assert "pitch_roll_yaw_out_of_bounds" in invalid_data[0]["rejection_reasons"]


def test_revalidate_saved_extrema_recomputes_static_feasibility():
    """
    revalidate_saved_extrema must reload setup JSON, rebuild MuJoCo state,
    recompute contacts/CoM, and call evaluate_static_standing_pose to verify
    the saved extrema are still statically feasible.

    This test verifies the revalidation round-trip:
    1. Load setup JSON with joint_qpos and root_z
    2. Rebuild MuJoCo data.qpos from saved values
    3. Recompute contacts via extract_wheel_floor_contact_points
    4. Recompute CoM via compute_robot_com_xy
    5. Call evaluate_static_standing_pose with recomputed values
    6. Return revalidation results as a LIST
    """
    from scripts.search_physical_standing_height_envelope import (
        serialize_candidate_setup,
        revalidate_saved_extrema,
    )
    from pathlib import Path
    import json
    import tempfile

    # Create a valid candidate setup using approved schema
    valid_setup = {
        "requested_target_com_z_m": 0.50,
        "achieved_com_z_m": 0.495,
        "calibrated_root_z_m": 0.60,
        "hip_pitch_ref": 0.4,
        "knee_ref": -0.8,
        "joint_qpos": {
            "l_hip_roll": 0.0, "l_hip_yaw": 0.0, "l_hip_pitch": 0.4, "l_knee": -0.8,
            "r_hip_roll": 0.0, "r_hip_yaw": 0.0, "r_hip_pitch": 0.4, "r_knee": -0.8,
        },
        "support_geometry": {"valid": True, "rejection_reasons": []},
        "contact_metrics": {"left_wheel_contact": True, "right_wheel_contact": True,
                          "non_wheel_floor_contact_count": 0, "rejection_reasons": []},
        "joint_limit_margin_rad": 0.8,
        "candidate_source": "search",
        "candidate_is_root_z_only": False,
        "rejection_reasons": [],
    }

    # Write setup to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_path = Path(tmpdir) / "test_setup.json"
        with open(setup_path, "w") as f:
            json.dump(valid_setup, f, indent=2)

        # Revalidate from saved setup
        results = revalidate_saved_extrema([setup_path])

        # Must return a LIST, not a dict wrapper
        assert isinstance(results, list), "Must return a list"
        assert len(results) == 1, "Should have 1 result"

        result = results[0]
        assert "setup_path" in result, "Must have setup_path"
        assert "static_feasible" in result, "Must have static_feasible"
        assert "rejection_reasons" in result, "Must have rejection_reasons"
        assert "achieved_com_z_m" in result, "Must have achieved_com_z_m"

        # Should have valid types
        assert isinstance(result["static_feasible"], bool)
        assert isinstance(result["rejection_reasons"], list)
        assert isinstance(result["achieved_com_z_m"], (int, float))


# ============================================================================
# Task 6: Guard-rail tests for physical envelope stop point
# ============================================================================


def test_selected_extrema_come_from_static_feasibility_only():
    """
    The saved physical extrema in physical_height_envelope_summary.json
    must have been selected from static feasibility checks only, with no
    dynamic failure metadata influencing the selection.

    This test verifies:
    1. physical_min_height has candidate_is_root_z_only=False
    2. physical_max_height has candidate_is_root_z_only=False
    3. Both extrema have empty rejection_reasons from static checks
    4. static_revalidation section exists and shows PASS verdict
    5. No dynamic failure fields (dynamic_stable, step_e_*, step_c_*) are present
    """
    from pathlib import Path
    import json

    # Load the actual summary artifact
    summary_path = Path(__file__).parent.parent / "outputs" / \
                   "physical_standing_height_envelope_search" / \
                   "physical_height_envelope_summary.json"

    assert summary_path.exists(), \
        f"Summary artifact not found at {summary_path}"

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Verify physical_min_height comes from static feasibility only
    assert "physical_min_height" in summary, "Must have physical_min_height"
    min_height = summary["physical_min_height"]
    assert min_height["candidate_is_root_z_only"] is False, \
        "physical_min_height must not be a root_z_only candidate"
    assert len(min_height["rejection_reasons"]) == 0, \
        "physical_min_height must have no rejection reasons from static checks"

    # Verify physical_max_height comes from static feasibility only
    assert "physical_max_height" in summary, "Must have physical_max_height"
    max_height = summary["physical_max_height"]
    assert max_height["candidate_is_root_z_only"] is False, \
        "physical_max_height must not be a root_z_only candidate"
    assert len(max_height["rejection_reasons"]) == 0, \
        "physical_max_height must have no rejection reasons from static checks"

    # Verify static_revalidation exists and shows PASS
    assert "static_revalidation" in summary, "Must have static_revalidation"
    revalidation = summary["static_revalidation"]
    assert revalidation["verdict"] == "PHYSICAL_ENVELOPE_PASS", \
        "static_revalidation verdict must be PHYSICAL_ENVELOPE_PASS"
    assert "revalidation_results" in revalidation, "Must have revalidation_results"
    assert isinstance(revalidation["revalidation_results"], list), \
        "revalidation_results must be a list"
    assert len(revalidation["revalidation_results"]) == 2, \
        "revalidation_results must have 2 entries (min and max)"

    # Verify all revalidation results show static_feasible=True
    for result in revalidation["revalidation_results"]:
        assert result["static_feasible"] is True, \
            "All revalidation results must have static_feasible=True"
        assert len(result["rejection_reasons"]) == 0, \
            "All revalidation results must have no rejection reasons"

    # Verify NO dynamic failure metadata fields are present in extrema
    forbidden_dynamic_fields = [
        "dynamic_stable", "dynamic_failure", "step_e_result", "step_c_result",
        "controller_stable", "recovery_successful", "torque_feasible",
    ]
    for field in forbidden_dynamic_fields:
        assert field not in min_height, \
            f"physical_min_height must not have dynamic field: {field}"
        assert field not in max_height, \
            f"physical_max_height must not have dynamic field: {field}"


def test_validation_summary_has_no_step_e_or_step_c_fields():
    """
    The physical_height_envelope_summary.json must not contain any Step E
    (height-variant robustness) or Step C (height recovery) execution fields.

    This test enforces the stop point: the physical envelope search performs
    ONLY static feasibility checks, with dynamic validation deferred to later
    stages.

    Forbidden top-level fields:
    - step_e_validation, step_e_results, step_e_summary
    - step_c_validation, step_c_results, step_c_summary
    - dynamic_validation, dynamic_results, dynamic_summary
    - controller_validation, controller_results
    """
    from pathlib import Path
    import json

    # Load the actual summary artifact
    summary_path = Path(__file__).parent.parent / "outputs" / \
                   "physical_standing_height_envelope_search" / \
                   "physical_height_envelope_summary.json"

    assert summary_path.exists(), \
        f"Summary artifact not found at {summary_path}"

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Define forbidden Step E / Step C / dynamic validation fields
    forbidden_top_level_fields = [
        "step_e_validation",
        "step_e_results",
        "step_e_summary",
        "step_c_validation",
        "step_c_results",
        "step_c_summary",
        "dynamic_validation",
        "dynamic_results",
        "dynamic_summary",
        "controller_validation",
        "controller_results",
        "height_variant_robustness",
        "height_recovery_validation",
    ]

    # Verify none of the forbidden fields are present
    summary_keys = set(summary.keys())
    for field in forbidden_top_level_fields:
        assert field not in summary_keys, \
            f"Summary must not contain Step E/Step C field: {field}"

    # Verify only expected static fields are present
    expected_fields = {
        "physical_min_height",
        "physical_max_height",
        "static_revalidation",
        "valid_candidate_count",
        "invalid_candidate_count",
        "total_candidate_count",
    }

    # Allow for minor variations, but core fields must match
    assert expected_fields.issubset(summary_keys), \
        f"Summary must contain all expected static fields. Missing: {expected_fields - summary_keys}"

    # Verify static_revalidation does NOT contain dynamic metadata
    revalidation = summary["static_revalidation"]
    forbidden_revalidation_fields = [
        "step_e_verdict", "step_c_verdict", "dynamic_verdict",
        "controller_verdict", "torque_verdict",
    ]
    revalidation_keys = set(revalidation.keys())
    for field in forbidden_revalidation_fields:
        assert field not in revalidation_keys, \
            f"static_revalidation must not contain dynamic field: {field}"
