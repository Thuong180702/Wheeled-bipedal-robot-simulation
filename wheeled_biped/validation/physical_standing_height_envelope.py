"""
Physical standing height envelope geometry primitives.

Task 1: Support-segment geometry only.
Task 2: Contact extraction and static feasibility validation.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any, Optional
import math
import mujoco
import numpy as np


# ============================================================================
# CoM Convention Documentation
# ============================================================================

ROBOT_COM_CONVENTION = "whole_robot"
"""
CoM convention for this project: use data.subtree_com[0] (whole robot),
NOT data.subtree_com[1] (torso-only subtree).
"""


@dataclass
class PhysicalStandingThresholds:
    """
    Thresholds defining the physical standing height envelope.

    These are the constraints for what constitutes a valid standing posture.
    """
    projection_tolerance: float = 1e-6
    preferred_sagittal_offset_m: float = 0.01
    max_sagittal_offset_m: float = 0.02
    max_pitch_abs_rad: float = 0.10
    max_roll_abs_rad: float = 0.05
    max_yaw_abs_rad: float = 0.10
    min_joint_limit_margin_rad: float = 0.05
    degenerate_segment_length_m: float = 1e-6


# ============================================================================
# Task 1: Support Segment Geometry Dataclass
# ============================================================================


@dataclass
class SupportSegmentGeometry:
    """
    Geometric description of the support segment formed by two wheel contacts.

    The support segment is the line between the left and right wheel contact points.
    This dataclass captures the geometry needed to check CoM containment.
    """
    valid: bool
    rejection_reasons: List[str]
    support_center_xy: Tuple[float, float]
    wheel_line_direction_xy: Tuple[float, float]
    support_error_direction_xy: Tuple[float, float]
    com_projection_fraction_on_wheel_segment: float
    com_projection_inside_wheel_segment: bool
    com_lateral_offset_from_support_center_m: float
    com_sagittal_offset_from_support_center_m: float
    segment_length_m: float
    min_endpoint_margin_m: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a serializable dictionary.

        Returns:
            Dictionary with Python-native types (float, bool, list, tuple).
        """
        return asdict(self)


# ============================================================================
# Task 2: Contact Points and Feasibility Result Dataclasses
# ============================================================================


@dataclass
class WheelContactPoints:
    """
    Wheel-floor contact points extracted from MuJoCo contact geometry.

    This dataclass captures actual wheel-floor contact points from MuJoCo's
    contact detection, or reports when contact geometry is missing.
    """
    left_wheel_contact_xy: Optional[Tuple[float, float]]
    right_wheel_contact_xy: Optional[Tuple[float, float]]
    left_wheel_contact: bool
    right_wheel_contact: bool
    non_wheel_floor_contact_count: int
    wheel_contact_force_z_n: Optional[Tuple[float, float]]
    rejection_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a serializable dictionary.

        Returns:
            Dictionary with Python-native types (float, bool, list, tuple).
        """
        return asdict(self)


@dataclass
class StaticStandingFeasibilityResult:
    """
    Result of static standing pose feasibility evaluation.

    This dataclass captures the full validation result including contact
    geometry, support segment geometry, orientation checks, and overall
    feasibility verdict.
    """
    setup_valid: bool
    static_feasible: bool
    rejection_reasons: List[str]
    candidate_source: str
    candidate_is_root_z_only: bool
    support_geometry: Optional[SupportSegmentGeometry]
    contact_metrics: Optional[WheelContactPoints]
    posture_metrics: Dict[str, float]
    joint_limit_margin_rad: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a serializable dictionary.

        Returns:
            Dictionary with Python-native types (float, bool, list, tuple).
        """
        result = {
            "setup_valid": self.setup_valid,
            "static_feasible": self.static_feasible,
            "rejection_reasons": self.rejection_reasons,
            "candidate_source": self.candidate_source,
            "candidate_is_root_z_only": self.candidate_is_root_z_only,
            "posture_metrics": self.posture_metrics,
            "joint_limit_margin_rad": self.joint_limit_margin_rad,
        }

        if self.contact_metrics is not None:
            result["contact_metrics"] = self.contact_metrics.to_dict()
        else:
            result["contact_metrics"] = None

        if self.support_geometry is not None:
            result["support_geometry"] = self.support_geometry.to_dict()
        else:
            result["support_geometry"] = None

        return result


def build_support_segment_geometry(
    *,
    left_wheel_contact_xy: Tuple[float, float],
    right_wheel_contact_xy: Tuple[float, float],
    com_xy: Tuple[float, float],
    thresholds: PhysicalStandingThresholds,
) -> SupportSegmentGeometry:
    """
    Build support segment geometry from wheel positions and CoM.

    The support segment is the line connecting the left and right wheel contact points.
    This function computes geometric properties needed for containment checking.

    Coordinate Convention:
        All inputs (wheel contacts and CoM) are in the world-XY frame.
        Offset outputs are NOT simple X/Y components. Instead, they are projections
        onto geometry-derived directions:
        - com_lateral_offset_from_support_center_m is the signed component along the
          wheel-line direction (left-to-right wheel vector) from the midpoint
        - com_sagittal_offset_from_support_center_m is the signed component along the
          perpendicular support-error direction (90° rotation of wheel line) from the midpoint
        The wheel line direction and support error direction are computed geometrically from
        the actual wheel positions, without assuming any particular axis alignment.

    Args:
        left_wheel_contact_xy: Left wheel contact position (x, y) in world frame.
        right_wheel_contact_xy: Right wheel contact position (x, y) in world frame.
        com_xy: Center of mass position (x, y) in world frame.
        thresholds: Physical standing thresholds configuration.

    Returns:
        SupportSegmentGeometry describing the support segment and CoM relationship.
    """
    rejection_reasons = []

    # Compute support center (midpoint between wheels)
    support_center_xy = (
        0.5 * (left_wheel_contact_xy[0] + right_wheel_contact_xy[0]),
        0.5 * (left_wheel_contact_xy[1] + right_wheel_contact_xy[1]),
    )

    # Compute wheel line vector (left to right)
    wheel_line_vec = (
        right_wheel_contact_xy[0] - left_wheel_contact_xy[0],
        right_wheel_contact_xy[1] - left_wheel_contact_xy[1],
    )
    segment_length_m = math.sqrt(wheel_line_vec[0]**2 + wheel_line_vec[1]**2)

    # Check for degenerate case (wheels too close)
    if segment_length_m < thresholds.degenerate_segment_length_m:
        rejection_reasons.append("degenerate_wheel_support_segment")
        rejection_reasons.append("support_geometry_invalid")
        return SupportSegmentGeometry(
            valid=False,
            rejection_reasons=rejection_reasons,
            support_center_xy=support_center_xy,
            wheel_line_direction_xy=(0.0, 1.0),  # Default to y-axis
            support_error_direction_xy=(1.0, 0.0),  # Default to x-axis
            com_projection_fraction_on_wheel_segment=0.0,
            com_projection_inside_wheel_segment=False,
            com_lateral_offset_from_support_center_m=0.0,
            com_sagittal_offset_from_support_center_m=0.0,
            segment_length_m=segment_length_m,
            min_endpoint_margin_m=0.0,
        )

    # Compute wheel line direction (unit vector from left to right)
    wheel_line_direction_xy = (
        wheel_line_vec[0] / segment_length_m,
        wheel_line_vec[1] / segment_length_m,
    )

    # Compute perpendicular direction (horizontal perpendicular to wheel line)
    # Rotate wheel_line_direction_xy by 90 degrees counterclockwise
    support_error_direction_xy = (
        -wheel_line_direction_xy[1],
        wheel_line_direction_xy[0],
    )

    # Compute CoM offset from support center
    com_offset = (
        com_xy[0] - support_center_xy[0],
        com_xy[1] - support_center_xy[1],
    )

    # Project CoM offset onto wheel line direction to get lateral component
    com_lateral_offset_from_support_center_m = (
        com_offset[0] * wheel_line_direction_xy[0] +
        com_offset[1] * wheel_line_direction_xy[1]
    )

    # Project CoM offset onto support error direction to get sagittal component
    com_sagittal_offset_from_support_center_m = (
        com_offset[0] * support_error_direction_xy[0] +
        com_offset[1] * support_error_direction_xy[1]
    )

    # Compute projection fraction (0.0 = left wheel, 1.0 = right wheel)
    com_projection_fraction_on_wheel_segment = (
        0.5 + com_lateral_offset_from_support_center_m / segment_length_m
    )

    # Check if projection is inside segment bounds
    com_projection_inside_wheel_segment = (
        abs(com_lateral_offset_from_support_center_m) <= segment_length_m / 2.0 + thresholds.projection_tolerance
    )

    # Compute minimum distance to either endpoint
    min_endpoint_margin_m = segment_length_m / 2.0 - abs(com_lateral_offset_from_support_center_m)

    # Validate geometry
    valid = True
    if not com_projection_inside_wheel_segment:
        rejection_reasons.append("projection_outside_wheel_segment")
        valid = False

    if abs(com_sagittal_offset_from_support_center_m) > thresholds.max_sagittal_offset_m:
        rejection_reasons.append("sagittal_support_offset_too_large")
        valid = False

    return SupportSegmentGeometry(
        valid=valid,
        rejection_reasons=rejection_reasons,
        support_center_xy=support_center_xy,
        wheel_line_direction_xy=wheel_line_direction_xy,
        support_error_direction_xy=support_error_direction_xy,
        com_projection_fraction_on_wheel_segment=com_projection_fraction_on_wheel_segment,
        com_projection_inside_wheel_segment=com_projection_inside_wheel_segment,
        com_lateral_offset_from_support_center_m=com_lateral_offset_from_support_center_m,
        com_sagittal_offset_from_support_center_m=com_sagittal_offset_from_support_center_m,
        segment_length_m=segment_length_m,
        min_endpoint_margin_m=min_endpoint_margin_m,
    )


# ============================================================================
# Task 2: Contact Extraction and Static Feasibility Functions
# ============================================================================


def compute_robot_com_xy(model: mujoco.MjModel, data: mujoco.MjData) -> Tuple[float, float]:
    """
    Compute the whole-robot center of mass (CoM) XY position.

    Follows ROBOT_COM_CONVENTION = "whole_robot": uses data.subtree_com[0],
    NOT data.subtree_com[1] (torso-only subtree).

    Args:
        model: MuJoCo model.
        data: MuJoCo data (assumes mj_forward has been called).

    Returns:
        Tuple (com_x, com_y) in world frame.
    """
    # Body 0 is the world root - subtree_com[0] is the whole-robot CoM
    whole_robot_com = data.subtree_com[0]
    return (float(whole_robot_com[0]), float(whole_robot_com[1]))


def extract_wheel_floor_contact_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> WheelContactPoints:
    """
    Extract wheel-floor contact points from MuJoCo contact geometry.

    This function uses actual MuJoCo contact detection to find where the
    left and right wheels touch the floor. If wheel-floor contacts are
    missing or incomplete, it reports the issue.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (assumes mj_forward has been called).

    Returns:
        WheelContactPoints with contact positions or rejection reasons.
    """
    rejection_reasons = []
    non_wheel_floor_contact_count = 0

    # Get wheel body IDs
    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    # Floor is geom 0 attached to world body (body 0)
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    left_wheel_contact_xy = None
    right_wheel_contact_xy = None
    left_wheel_contact = False
    right_wheel_contact = False
    left_wheel_force_z = None
    right_wheel_force_z = None

    # Scan all active contacts
    for i in range(data.ncon):
        contact = data.contact[i]

        # Get the two geoms in contact
        geom1 = contact.geom1
        geom2 = contact.geom2

        # Get the body IDs for these geoms
        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]

        # Check if one geom is the floor
        is_floor_contact = (geom1 == floor_geom_id) or (geom2 == floor_geom_id)

        if not is_floor_contact:
            continue

        # Check if the other body is a wheel
        is_left_wheel_floor = (body1 == l_wheel_id) or (body2 == l_wheel_id)
        is_right_wheel_floor = (body1 == r_wheel_id) or (body2 == r_wheel_id)

        if is_left_wheel_floor:
            # Extract contact point XY
            left_wheel_contact_xy = (float(contact.pos[0]), float(contact.pos[1]))
            left_wheel_contact = True
            # Extract vertical contact force using MuJoCo's contact force function
            # contactForce returns 6D force: [normal, tangent1, tangent2, torque1, torque2, torque3]
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, c_array)
            # Normal force is the first component, vertical force for horizontal floor
            left_wheel_force_z = float(abs(c_array[0]))
        elif is_right_wheel_floor:
            right_wheel_contact_xy = (float(contact.pos[0]), float(contact.pos[1]))
            right_wheel_contact = True
            # Extract vertical contact force
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, c_array)
            right_wheel_force_z = float(abs(c_array[0]))
        else:
            # Floor contact but not with wheels
            non_wheel_floor_contact_count += 1

    # Check for missing contacts
    if not left_wheel_contact or not right_wheel_contact:
        rejection_reasons.append("missing_wheel_floor_contact_geometry")

    # Prepare wheel contact forces tuple
    wheel_contact_force_z_n = None
    if left_wheel_force_z is not None and right_wheel_force_z is not None:
        wheel_contact_force_z_n = (left_wheel_force_z, right_wheel_force_z)

    return WheelContactPoints(
        left_wheel_contact_xy=left_wheel_contact_xy,
        right_wheel_contact_xy=right_wheel_contact_xy,
        left_wheel_contact=left_wheel_contact,
        right_wheel_contact=right_wheel_contact,
        non_wheel_floor_contact_count=non_wheel_floor_contact_count,
        wheel_contact_force_z_n=wheel_contact_force_z_n,
        rejection_reasons=rejection_reasons,
    )


def evaluate_static_standing_pose(
    *,
    left_wheel_contact_xy: Optional[Tuple[float, float]],
    right_wheel_contact_xy: Optional[Tuple[float, float]],
    com_xy: Tuple[float, float],
    pitch_x_rad: float,
    roll_y_rad: float,
    yaw_z_rad: float,
    left_wheel_contact: bool,
    right_wheel_contact: bool,
    non_wheel_floor_contact_count: int,
    joint_limit_margin_rad: float,
    thresholds: PhysicalStandingThresholds,
    candidate_source: str = "unknown",
    candidate_is_root_z_only: bool = False,
) -> StaticStandingFeasibilityResult:
    """
    Evaluate whether a pose represents a feasible static standing configuration.

    This function performs comprehensive validation:
    - Checks for strong wheel-floor contact (both wheels must be in contact)
    - Rejects root-z-only candidates (if flagged)
    - Checks orientation (pitch, roll, yaw)
    - Reports non-wheel floor contacts
    - Builds support segment geometry if contacts are valid
    - Returns overall feasibility verdict

    Args:
        left_wheel_contact_xy: Left wheel contact position (x, y) or None.
        right_wheel_contact_xy: Right wheel contact position (x, y) or None.
        com_xy: Center of mass position (x, y).
        pitch_x_rad: Body pitch (rotation around X).
        roll_y_rad: Body roll (rotation around Y).
        yaw_z_rad: Body yaw (rotation around Z).
        left_wheel_contact: True if left wheel has floor contact.
        right_wheel_contact: True if right wheel has floor contact.
        non_wheel_floor_contact_count: Number of non-wheel floor contacts.
        joint_limit_margin_rad: Minimum distance to joint limits.
        thresholds: Physical standing thresholds configuration.
        candidate_source: Source identifier for this candidate.
        candidate_is_root_z_only: True if all joints are at zero (root_z only).

    Returns:
        StaticStandingFeasibilityResult with full validation details.
    """
    rejection_reasons = []
    setup_valid = True
    static_feasible = True

    # Build posture metrics
    posture_metrics = {
        "pitch_x_rad": pitch_x_rad,
        "roll_y_rad": roll_y_rad,
        "yaw_z_rad": yaw_z_rad,
    }

    # Check for root-z-only candidates
    if candidate_is_root_z_only:
        rejection_reasons.append("root_z_only_candidate_not_allowed")
        setup_valid = False
        static_feasible = False

    # Check orientation thresholds
    if abs(pitch_x_rad) > thresholds.max_pitch_abs_rad:
        rejection_reasons.append("pitch_roll_yaw_out_of_bounds")
        static_feasible = False

    if abs(roll_y_rad) > thresholds.max_roll_abs_rad:
        rejection_reasons.append("pitch_roll_yaw_out_of_bounds")
        static_feasible = False

    if abs(yaw_z_rad) > thresholds.max_yaw_abs_rad:
        rejection_reasons.append("pitch_roll_yaw_out_of_bounds")
        static_feasible = False

    # Check joint limit margin
    if joint_limit_margin_rad < thresholds.min_joint_limit_margin_rad:
        rejection_reasons.append("joint_limit_margin_too_small")
        static_feasible = False

    # STRONG contact test: both wheels must be in contact
    if not left_wheel_contact or not right_wheel_contact:
        rejection_reasons.append("missing_wheel_floor_contact_geometry")
        setup_valid = False
        static_feasible = False

    # Check for non-wheel floor contacts
    if non_wheel_floor_contact_count > 0:
        rejection_reasons.append("non_wheel_floor_contact")
        static_feasible = False

    # Build contact metrics
    # Note: We still create the WheelContactPoints even if contacts are missing
    # to provide diagnostic information
    contact_metrics = WheelContactPoints(
        left_wheel_contact_xy=left_wheel_contact_xy,
        right_wheel_contact_xy=right_wheel_contact_xy,
        left_wheel_contact=left_wheel_contact,
        right_wheel_contact=right_wheel_contact,
        non_wheel_floor_contact_count=non_wheel_floor_contact_count,
        wheel_contact_force_z_n=None,  # Not provided in this function signature
        rejection_reasons=[],  # Contact-specific rejections already added to main list
    )

    # Build support segment geometry only if we have valid wheel contacts
    support_geometry = None
    if left_wheel_contact and right_wheel_contact and left_wheel_contact_xy is not None and right_wheel_contact_xy is not None:
        support_geometry = build_support_segment_geometry(
            left_wheel_contact_xy=left_wheel_contact_xy,
            right_wheel_contact_xy=right_wheel_contact_xy,
            com_xy=com_xy,
            thresholds=thresholds,
        )

        if not support_geometry.valid:
            rejection_reasons.extend(support_geometry.rejection_reasons)
            static_feasible = False

    return StaticStandingFeasibilityResult(
        setup_valid=setup_valid,
        static_feasible=static_feasible,
        rejection_reasons=rejection_reasons,
        candidate_source=candidate_source,
        candidate_is_root_z_only=candidate_is_root_z_only,
        support_geometry=support_geometry,
        contact_metrics=contact_metrics,
        posture_metrics=posture_metrics,
        joint_limit_margin_rad=joint_limit_margin_rad,
    )
