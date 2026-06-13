from __future__ import annotations

from scripts.search_operational_height_envelope import (
    OperationalHeightCandidate,
    StaticValidationThresholds,
    select_envelope_extrema,
    validate_operational_height_candidate,
)
from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule
from scripts.run_step_c_height_recovery import build_simulation_command


def make_candidate(**overrides):
    values = {
        "variant_name": "candidate",
        "requested_target_com_z_m": 0.40,
        "achieved_com_z_m": 0.4005,
        "calibrated_root_z_m": 0.53,
        "hip_pitch_ref": 0.95,
        "knee_ref": 1.78,
        "nominal_hip_pitch_ref": 0.926052,
        "nominal_knee_ref": 1.748364,
        "hip_roll_left": 0.0,
        "hip_roll_right": 0.0,
        "hip_yaw_left": 0.0,
        "hip_yaw_right": 0.0,
        "support_center_x": 0.0,
        "support_center_y": -0.0134,
        "com_x_m": 0.0005,
        "com_y_m": -0.0130,
        "com_support_error_x": 0.0005,
        "com_support_error_y": 0.0004,
        "com_support_error_norm_xy": 0.00064,
        "left_wheel_contact": True,
        "right_wheel_contact": True,
        "wheel_floor_contact_count": 2,
        "non_wheel_floor_contact_count": 0,
        "min_wheel_contact_dist_m": -0.0005,
        "total_wheel_floor_fz": 68.0,
        "pitch_x_rad": 0.0,
        "roll_y_rad": 0.0,
        "yaw_z_rad": 0.0,
        "joint_limit_margin_min_rad": 0.30,
        "root_z_only": False,
        "setup_valid": True,
        "setup_failure_reason": None,
    }
    values.update(overrides)
    return OperationalHeightCandidate(**values)


def test_operational_height_candidate_rejects_root_z_only_pose_without_leg_adjustment():
    candidate = make_candidate(
        hip_pitch_ref=0.926052,
        knee_ref=1.748364,
        root_z_only=True,
    )

    result = validate_operational_height_candidate(candidate)

    assert result.setup_valid is False
    assert "root_z_only" in result.setup_failure_reason


def test_operational_height_candidate_accepts_valid_symmetric_hip_knee_root_z_pose():
    candidate = make_candidate()

    result = validate_operational_height_candidate(candidate)

    assert result.setup_valid is True
    assert result.setup_failure_reason is None


def test_com_support_centering_threshold_is_enforced():
    candidate = make_candidate(com_support_error_norm_xy=0.011)

    result = validate_operational_height_candidate(candidate)

    assert result.setup_valid is False
    assert "support_not_centered" in result.setup_failure_reason


def test_wheel_contacts_are_required():
    candidate = make_candidate(left_wheel_contact=False, wheel_floor_contact_count=1)

    result = validate_operational_height_candidate(candidate)

    assert result.setup_valid is False
    assert "missing_wheel_contact" in result.setup_failure_reason


def test_non_wheel_contacts_reject_candidate():
    candidate = make_candidate(non_wheel_floor_contact_count=1)

    result = validate_operational_height_candidate(candidate)

    assert result.setup_valid is False
    assert "non_wheel_floor_contacts" in result.setup_failure_reason


def test_joint_limit_margin_is_enforced():
    candidate = make_candidate(joint_limit_margin_min_rad=0.019)

    result = validate_operational_height_candidate(candidate)

    assert result.setup_valid is False
    assert "joint_limit_margin" in result.setup_failure_reason


def test_min_max_envelope_selection_uses_safety_margin_not_first_invalid_point():
    candidates = [
        make_candidate(variant_name="low_invalid", requested_target_com_z_m=0.390, achieved_com_z_m=0.390, setup_valid=False, setup_failure_reason="static_invalid"),
        make_candidate(variant_name="low_boundary", requested_target_com_z_m=0.392, achieved_com_z_m=0.392, joint_limit_margin_min_rad=0.021),
        make_candidate(variant_name="low_safe", requested_target_com_z_m=0.394, achieved_com_z_m=0.394, joint_limit_margin_min_rad=0.08),
        make_candidate(variant_name="high_safe", requested_target_com_z_m=0.416, achieved_com_z_m=0.416, joint_limit_margin_min_rad=0.08),
        make_candidate(variant_name="high_boundary", requested_target_com_z_m=0.418, achieved_com_z_m=0.418, joint_limit_margin_min_rad=0.021),
        make_candidate(variant_name="high_invalid", requested_target_com_z_m=0.420, achieved_com_z_m=0.420, setup_valid=False, setup_failure_reason="static_invalid"),
    ]

    selected = select_envelope_extrema(candidates, StaticValidationThresholds(selection_joint_margin_min_rad=0.05))

    assert selected["min_candidate"].variant_name == "low_safe"
    assert selected["max_candidate"].variant_name == "high_safe"
    assert selected["extrema_are_conservative"] is True


def test_min_envelope_selection_excludes_heights_below_controller_safety_floor():
    candidates = [
        make_candidate(variant_name="below_termination_floor", requested_target_com_z_m=0.34, achieved_com_z_m=0.34, joint_limit_margin_min_rad=0.08),
        make_candidate(variant_name="below_step_c_safety_floor", requested_target_com_z_m=0.37, achieved_com_z_m=0.37, joint_limit_margin_min_rad=0.08),
        make_candidate(variant_name="lowest_controller_ready", requested_target_com_z_m=0.385, achieved_com_z_m=0.385, joint_limit_margin_min_rad=0.08),
        make_candidate(variant_name="high_safe", requested_target_com_z_m=0.416, achieved_com_z_m=0.416, joint_limit_margin_min_rad=0.08),
    ]

    selected = select_envelope_extrema(candidates)

    assert selected["min_candidate"].variant_name == "lowest_controller_ready"


def test_max_envelope_selection_can_apply_dynamic_readiness_ceiling():
    candidates = [
        make_candidate(variant_name="low_safe", requested_target_com_z_m=0.394, achieved_com_z_m=0.394, joint_limit_margin_min_rad=0.08),
        make_candidate(variant_name="highest_controller_ready", requested_target_com_z_m=0.430, achieved_com_z_m=0.430, joint_limit_margin_min_rad=0.08),
        make_candidate(variant_name="static_only_tall_pose", requested_target_com_z_m=0.472, achieved_com_z_m=0.472, joint_limit_margin_min_rad=0.08),
    ]

    selected = select_envelope_extrema(candidates)

    assert selected["max_candidate"].variant_name == "highest_controller_ready"


def test_custom_setup_json_initialization_does_not_change_controller_torque_logic():
    cmd = build_simulation_command(
        steps=5000,
        telemetry_decimation=1,
        failure_window_steps=500,
        height_variant_setup="outputs/operational_height_envelope_search/min_operational_height_setup.json",
        vd_sagittal_authority_profile="candidate_D2_wheel_velocity_damping_light",
    )

    assert "--height-variant-setup" in cmd
    assert "--initial-root-z-perturbation" not in cmd
    assert "--controller-mode" in cmd
    assert cmd[cmd.index("--controller-mode") + 1] == "balance-core"


def test_candidate_d2_profile_can_be_active_for_extreme_cases_when_configured():
    schedule = resolve_sagittal_authority_schedule("candidate_D2_wheel_velocity_damping_light")

    assert schedule.is_active_for_variant("max_operational_height") is True
    assert schedule.is_active_for_variant("min_operational_height") is True
    assert schedule.velocity_damping_scale == 1.10


def test_wbc_remains_off_in_static_validation_contract():
    candidate = make_candidate(wbc_applied=False, hidden_torque_norm_max=0.0, ownership_violation_count_max=0)

    result = validate_operational_height_candidate(candidate)

    assert result.setup_valid is True
    assert result.wbc_applied is False
    assert result.hidden_torque_norm_max == 0.0
    assert result.ownership_violation_count_max == 0
