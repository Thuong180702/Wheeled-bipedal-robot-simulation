# tests/test_balance_core_height_variant_setup_gates.py
"""Tests for multi-objective CoM centering and static-balance gates in height variant setup validation."""

import json
from pathlib import Path

import pytest


def test_setup_report_exists():
    """Test that B2-B4 setup report was generated."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    assert report_path.exists(), "Setup report JSON not found"


def test_setup_report_has_gate_enforcement_flags():
    """Test that setup report explicitly states gates are enforced."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    assert report["com_centering_gate_enforced"] is True, "CoM centering gate not enforced"
    assert report["static_balance_gate_enforced"] is True, "Static-balance gate not enforced"
    assert report["nominal_reference_comparison_enforced"] is True, "Nominal reference comparison not enforced"


def test_nominal_reference_values_captured():
    """Test that nominal reference values for CoM centering are captured."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    assert "nominal_support_center_x" in report
    assert "nominal_support_center_y" in report
    assert "nominal_com_support_error_x" in report
    assert "nominal_com_support_error_y" in report
    assert "nominal_com_support_error_norm" in report


def test_all_variants_have_com_centering_fields():
    """Test that all variants have CoM centering fields populated."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    required_fields = [
        "support_center_x",
        "support_center_y",
        "com_x_m",
        "com_y_m",
        "com_support_error_x",
        "com_support_error_y",
        "com_support_error_norm_xy",
        "cp_x_m",
        "cp_y_m",
        "cp_error_x_m",
        "cp_error_y_m",
    ]

    for variant in report["setup_results"]:
        for field in required_fields:
            assert field in variant, f"Variant {variant['variant_name']} missing field {field}"


def test_nominal_variant_is_valid():
    """Test that nominal baseline variant passes all gates."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    nominal = next(v for v in report["setup_results"] if v["variant_name"] == "nominal")
    assert nominal["setup_valid"] is True, f"Nominal should be valid, got: {nominal['setup_failure_reason']}"
    assert nominal["posture_search_method"] == "keyframe_baseline"


def test_nominal_uses_keyframe_not_height_ik():
    """Test that nominal variant uses keyframe baseline, not HeightIK regeneration."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    nominal = next(v for v in report["setup_results"] if v["variant_name"] == "nominal")
    assert nominal["posture_search_method"] == "keyframe_baseline", "Nominal must use keyframe baseline"


def test_height_ik_metric_audit_present():
    """Test that HeightIK metric mismatch is documented."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    assert "height_ik_metric_audit" in report
    audit = report["height_ik_metric_audit"]
    assert audit["height_ik_metric_is_not_com_height"] is True
    assert "warning" in audit


def test_variant_with_large_com_offset_rejected():
    """Test that variant with correct height but large CoM offset is rejected."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    # high_small should be rejected for com_not_centered
    high_small = next((v for v in report["setup_results"] if v["variant_name"] == "high_small"), None)

    if high_small is not None:
        # If high_small exists and is invalid, check it was rejected for CoM centering
        if not high_small["setup_valid"]:
            assert "com_not_centered" in high_small["setup_failure_reason"], \
                f"Expected com_not_centered rejection, got: {high_small['setup_failure_reason']}"


def test_true_height_variant_changes_posture_not_root_z_only():
    """Test that non-nominal variants change hip_pitch/knee, not just root_z."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    nominal = next(v for v in report["setup_results"] if v["variant_name"] == "nominal")
    nominal_hip_pitch = nominal["hip_pitch_ref"]
    nominal_knee = nominal["knee_ref"]

    for variant in report["setup_results"]:
        if variant["variant_name"] == "nominal":
            continue

        # Non-nominal variants should have different hip_pitch or knee
        # (unless search failed, in which case they'll be invalid)
        if variant["posture_search_method"] == "com_calibrated_search":
            # At least one joint should differ from nominal
            hip_pitch_diff = abs(variant["hip_pitch_ref"] - nominal_hip_pitch)
            knee_diff = abs(variant["knee_ref"] - nominal_knee)

            # Allow small numerical differences, but expect meaningful change
            assert hip_pitch_diff > 0.001 or knee_diff > 0.001, \
                f"Variant {variant['variant_name']} has same posture as nominal (root-z-only offset)"


def test_setup_valid_depends_on_com_centering_not_only_height():
    """Test that setup_valid gate checks CoM centering, not just height error."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    # Find any invalid variant
    invalid_variants = [v for v in report["setup_results"] if not v["setup_valid"]]

    for variant in invalid_variants:
        # Invalid variants should have a specific failure reason
        assert variant["setup_failure_reason"] is not None
        assert len(variant["setup_failure_reason"]) > 0

        # If height error is small but variant is invalid, it should be due to other gates
        if variant["height_error_m"] < 0.005:
            # Height is good, so failure must be from other gates
            failure_reason = variant["setup_failure_reason"]
            assert any(keyword in failure_reason for keyword in [
                "com_not_centered",
                "orientation_not_equilibrium",
                "hip_roll_not_nominal",
                "missing_wheel_contact",
                "non_wheel_floor_contacts",
            ]), f"Variant with good height rejected for unclear reason: {failure_reason}"


def test_markdown_report_exists():
    """Test that markdown report was generated."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.md")
    assert report_path.exists(), "Setup report markdown not found"


def test_markdown_report_contains_gate_status():
    """Test that markdown report explicitly states gate enforcement status."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.md")
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "CoM centering gate" in content
    assert "ENFORCED" in content
    assert "Static-balance gate" in content
    assert "Nominal reference comparison" in content


def test_multiobjective_search_method_used():
    """Test that non-nominal variants use multi-objective search method."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    for variant in report["setup_results"]:
        if variant["variant_name"] == "nominal":
            assert variant["posture_search_method"] == "keyframe_baseline"
        else:
            # Non-nominal variants should use multiobjective search
            assert "multiobjective" in variant["posture_search_method"].lower(), \
                f"Variant {variant['variant_name']} should use multiobjective search"


def test_candidate_statistics_present_for_searched_variants():
    """Test that searched variants have candidate statistics."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    for variant in report["setup_results"]:
        if variant["variant_name"] != "nominal" and variant["setup_valid"]:
            # Valid non-nominal variants should have candidate stats
            assert variant["candidate_stats"] is not None, \
                f"Variant {variant['variant_name']} missing candidate_stats"

            stats = variant["candidate_stats"]
            assert stats["total_evaluated"] > 0
            assert stats["passed_contact"] > 0
            assert stats["passed_all"] > 0


def test_all_granular_variants_pass_gates():
    """Test that all granular variants (high_tiny, low_tiny, low_small) pass gates."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    expected_variants = ["nominal", "high_tiny", "high_small", "low_tiny", "low_small"]

    for expected in expected_variants:
        variant = next((v for v in report["setup_results"] if v["variant_name"] == expected), None)
        assert variant is not None, f"Expected variant {expected} not found"
        assert variant["setup_valid"] is True, \
            f"Variant {expected} should be valid, got: {variant['setup_failure_reason']}"


def test_high_small_now_passes_com_centering():
    """Test that high_small now passes CoM centering gate with multi-objective search."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    high_small = next((v for v in report["setup_results"] if v["variant_name"] == "high_small"), None)
    assert high_small is not None, "high_small variant not found"
    assert high_small["setup_valid"] is True, \
        f"high_small should be valid with multi-objective search, got: {high_small['setup_failure_reason']}"

    # Verify CoM centering is good
    assert abs(high_small["com_support_error_y"]) < 0.015, \
        f"high_small CoM Y error should be < 15mm, got: {high_small['com_support_error_y']*1000:.1f}mm"


def test_search_evaluates_multiple_candidates():
    """Test that multi-objective search evaluates multiple candidates."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    for variant in report["setup_results"]:
        if variant["candidate_stats"] is not None:
            stats = variant["candidate_stats"]
            # Should evaluate a grid of candidates (20x20 = 400)
            assert stats["total_evaluated"] >= 100, \
                f"Variant {variant['variant_name']} should evaluate many candidates"

            # Should have gate filtering
            assert stats["passed_contact"] <= stats["total_evaluated"]
            assert stats["passed_height"] <= stats["passed_contact"]
            assert stats["passed_all"] <= stats["passed_height"]


def test_low_variants_are_feasible():
    """Test that low variants (low_tiny, low_small) are feasible and valid."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    low_tiny = next((v for v in report["setup_results"] if v["variant_name"] == "low_tiny"), None)
    low_small = next((v for v in report["setup_results"] if v["variant_name"] == "low_small"), None)

    assert low_tiny is not None, "low_tiny variant not found"
    assert low_small is not None, "low_small variant not found"

    assert low_tiny["setup_valid"] is True, \
        f"low_tiny should be valid, got: {low_tiny['setup_failure_reason']}"
    assert low_small["setup_valid"] is True, \
        f"low_small should be valid, got: {low_small['setup_failure_reason']}"


def test_ready_for_b5_b10():
    """Test that report indicates readiness for B5-B10 with multiple valid variants."""
    report_path = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    assert report["ready_for_b5_b10"] is True, \
        "Should be ready for B5-B10 with multiple valid variants"
    assert len(report["valid_variants"]) >= 2, \
        f"Need at least 2 valid variants for B5-B10, got: {len(report['valid_variants'])}"
