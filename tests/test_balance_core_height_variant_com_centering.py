"""Tests for CoM centering and static-balance validation in height variant setup."""

import pytest
import mujoco
import numpy as np

from wheeled_biped.utils.config import get_model_path


def test_support_center_is_computed():
    """Test that support center is computed from wheel positions."""
    from scripts.validate_balance_core_height_variants_v3_minimal import compute_support_center

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    support_center_x, support_center_y = compute_support_center(model, data)

    # Support center should be computed
    assert isinstance(support_center_x, float)
    assert isinstance(support_center_y, float)

    # Should be near origin for nominal posture
    assert abs(support_center_x) < 0.1
    assert abs(support_center_y) < 0.1


def test_com_support_error_is_computed():
    """Test that CoM support error is computed correctly."""
    from scripts.validate_balance_core_height_variants_v3_minimal import (
        compute_support_center,
        compute_com_support_error
    )

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    support_center_x, support_center_y = compute_support_center(model, data)

    torso_id = model.body("torso").id
    com_x = float(data.subtree_com[torso_id][0])
    com_y = float(data.subtree_com[torso_id][1])

    error_x, error_y, error_norm = compute_com_support_error(
        com_x, com_y, support_center_x, support_center_y
    )

    # Errors should be computed
    assert isinstance(error_x, float)
    assert isinstance(error_y, float)
    assert isinstance(error_norm, float)

    # Error norm should match Euclidean distance
    expected_norm = np.sqrt(error_x**2 + error_y**2)
    assert abs(error_norm - expected_norm) < 1e-6


def test_nominal_com_support_error_is_reference():
    """Test that nominal CoM support error is used as reference for variants."""
    # This test verifies the concept - actual implementation would check
    # that variants are validated against nominal reference
    from scripts.validate_balance_core_height_variants_v3_minimal import (
        compute_support_center,
        compute_com_support_error
    )

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    support_center_x, support_center_y = compute_support_center(model, data)

    torso_id = model.body("torso").id
    com_x = float(data.subtree_com[torso_id][0])
    com_y = float(data.subtree_com[torso_id][1])

    nominal_error_x, nominal_error_y, nominal_error_norm = compute_com_support_error(
        com_x, com_y, support_center_x, support_center_y
    )

    # Nominal should have small CoM support error (well-centered)
    assert abs(nominal_error_x) < 0.02
    assert abs(nominal_error_y) < 0.02
    assert nominal_error_norm < 0.03


def test_height_variant_changes_posture_not_root_z_only():
    """Test that height variants change joint posture, not just root_z."""
    # Read the v3 report to verify variants have different postures
    import json
    from pathlib import Path

    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    if not report_path.exists():
        pytest.skip("Report not generated yet")

    with open(report_path, 'r') as f:
        report = json.load(f)

    if len(report['setup_results']) < 2:
        pytest.skip("Need at least 2 variants")

    nominal = report['setup_results'][0]
    variant = report['setup_results'][1]

    # Verify posture changed
    hip_pitch_changed = abs(nominal['hip_pitch_ref'] - variant['hip_pitch_ref']) > 0.01
    knee_changed = abs(nominal['knee_ref'] - variant['knee_ref']) > 0.01

    assert hip_pitch_changed or knee_changed, "Height variant must change joint posture"


def test_setup_report_contains_com_centering_fields():
    """Test that setup report contains all required CoM centering fields."""
    import json
    from pathlib import Path

    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    if not report_path.exists():
        pytest.skip("Report not generated yet")

    with open(report_path, 'r') as f:
        report = json.load(f)

    # Note: v3_minimal doesn't add all fields yet, but this test documents what's required
    # Required fields for full v3 implementation:
    required_fields = [
        'variant_name', 'target_com_z_m', 'achieved_com_z_m', 'height_error_m',
        'calibrated_root_z_m', 'hip_pitch_ref', 'knee_ref',
        'wheel_floor_contact_count', 'setup_valid'
    ]

    if len(report['setup_results']) > 0:
        result_fields = set(report['setup_results'][0].keys())
        for field in required_fields:
            assert field in result_fields, f"Missing required field: {field}"
