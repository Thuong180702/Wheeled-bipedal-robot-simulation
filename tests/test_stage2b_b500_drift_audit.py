import pytest

from scripts import simulate_hierarchical_controller as sim


REQUIRED_AUDIT_FIELDS = {
    # State
    "com_z", "com_vz", "pitch_x", "pitch_rate_x", "roll_y", "roll_rate_y", "yaw_z",
    "com_x", "com_y", "cp_x", "cp_y",
    "com_error_x", "com_error_y", "com_error_z",
    "cp_error_x", "cp_error_y",
    "pitch_error", "roll_error", "height_error",
    # Contact
    "left_wheel_floor_contact", "right_wheel_floor_contact", "total_wheel_floor_fz",
    "left_fz_actual", "right_fz_actual", "fz_asymmetry_actual",
    "non_wheel_floor_contacts", "contact_dist_min", "contact_dist_max",
    # WBC command
    "correction_wrench_Fx", "correction_wrench_Fy", "correction_wrench_Fz",
    "correction_wrench_Mx", "correction_wrench_My", "correction_wrench_Mz",
    "correction_Fy_com", "correction_Fy_cp", "correction_Fy_pitch", "correction_My_roll",
    "distributor_f_left", "distributor_f_right", "distributor_fz_sum",
    "tau_hip_roll", "tau_contact", "tau_wbc_correction", "tau_wbc_after_authority_clip",
    # Torque stack
    "tau_static_feedforward", "tau_static_posture", "tau_total_raw", "tau_final",
    "saturation_flags", "rate_limit_flags",
}


def test_b500_audit_field_catalog_exists():
    assert hasattr(sim, "build_stage2b_drift_audit_field_names")


def test_b500_audit_field_catalog_contains_required_fields():
    fields = set(sim.build_stage2b_drift_audit_field_names())
    missing = REQUIRED_AUDIT_FIELDS - fields
    assert not missing, f"Missing B500 audit fields: {sorted(missing)}"
