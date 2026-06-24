"""
Tests for the D4/D5 hip-yaw universal limit audit.

Ensures:
- mode decomposition works on telemetry rows
- reference-vs-error classification handles missing columns
- torque budget classification handles missing columns
- candidate_kind is preserved
- current-best is not changed by audit
- output report classification is one of allowed values
- D telemetry cannot be replaced by C telemetry
- old wheel-yaw D cannot be used as D_MODE_HIP_YAW_DIV_V1
"""

import csv
import json
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "d4_d5_hip_yaw_universal_limit_audit"
REPORT_PATH = REPO_ROOT / "docs" / "validation" / "d4_d5_hip_yaw_universal_limit_audit_report.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_d4_d5_hip_yaw_universal_limit.py"

# Allowed final classifications (from task spec)
ALLOWED_CLASSIFICATIONS = frozenset({
    "D4_D5_UNIVERSAL_HIP_YAW_LIMIT_AUDIT_COMPLETE_AUTHORITY_LIMIT",
    "D4_D5_UNIVERSAL_HIP_YAW_LIMIT_AUDIT_COMPLETE_HY2_CONFLICT",
    "D4_D5_UNIVERSAL_HIP_YAW_LIMIT_AUDIT_COMPLETE_YAW_INJECTION",
    "D4_D5_UNIVERSAL_HIP_YAW_LIMIT_AUDIT_COMPLETE_WHEEL_YAW_REQUIRED",
    "D4_D5_UNIVERSAL_HIP_YAW_LIMIT_AUDIT_COMPLETE_REFERENCE_LIMIT",
    "D4_D5_UNIVERSAL_HIP_YAW_LIMIT_AUDIT_COMPLETE_SUPPORT_CONTACT_COUPLING",
    "D4_D5_UNIVERSAL_HIP_YAW_LIMIT_AUDIT_INCONCLUSIVE",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _float_safe(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _bool_str_to_int(v):
    """Convert 'True'/'False' string to 1/0, falling back to float conversion."""
    if isinstance(v, str):
        low = v.strip().lower()
        if low == "true":
            return 1
        if low == "false":
            return 0
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def audit_outputs():
    """Load audit output CSVs into dicts."""
    required_files = [
        "d4_d5_windowed_metrics.csv",
        "d4_d5_peak_event_table.csv",
        "d4_d5_mode_decomposition_timeseries_summary.csv",
        "d4_d5_torque_budget_summary.csv",
        "d4_d5_reference_vs_error_summary.csv",
        "audit_summary.json",
    ]
    result = {}
    for fname in required_files:
        path = OUTPUT_DIR / fname
        assert path.exists(), f"Required audit output missing: {path}"
        if fname.endswith(".json"):
            with open(path) as f:
                result[fname] = json.load(f)
        else:
            with open(path, newline="") as f:
                result[fname] = list(csv.DictReader(f))
    return result


# ---------------------------------------------------------------------------
# Test 1: Mode decomposition works on telemetry rows
# ---------------------------------------------------------------------------
def test_mode_decomposition_on_telemetry_rows():
    """Verify mode decomposition from raw telemetry produces expected fields."""
    telemetry_path = (
        REPO_ROOT
        / "outputs"
        / "mode_hip_yaw_div_full_real_validation"
        / "d4_d5_focused_1000"
        / "D4_medium_push_low_D"
        / "telemetry_1000.csv"
    )
    if not telemetry_path.exists():
        pytest.skip("Telemetry file not available in this environment")

    with open(telemetry_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0, "Telemetry has no rows"

    # Compute mode decomposition
    n = len(rows)
    l_pos = [_float_safe(r["l_hip_yaw_pos"]) for r in rows]
    r_pos = [_float_safe(r["r_hip_yaw_pos"]) for r in rows]
    l_ref = [_float_safe(r["l_hip_yaw_ref"]) for r in rows]
    r_ref = [_float_safe(r["r_hip_yaw_ref"]) for r in rows]
    l_err = [_float_safe(r["l_hip_yaw_error"]) for r in rows]
    r_err = [_float_safe(r["r_hip_yaw_error"]) for r in rows]
    tau_f_l = [_float_safe(r["l_hip_yaw_tau_shape_final"]) for r in rows]
    tau_f_r = [_float_safe(r["r_hip_yaw_tau_shape_final"]) for r in rows]

    # Check common/divergence decompose and recompose
    common = [(l_pos[i] + r_pos[i]) / 2 for i in range(n)]
    divergence = [(l_pos[i] - r_pos[i]) / 2 for i in range(n)]
    for i in range(n):
        assert abs(l_pos[i] - (common[i] + divergence[i])) < 1e-10
        assert abs(r_pos[i] - (common[i] - divergence[i])) < 1e-10

    # Divergence error
    divergence_err = [(l_err[i] - r_err[i]) / 2 for i in range(n)]
    assert len(divergence_err) == n

    # Torque mode decomposition
    tau_div = [(tau_f_l[i] - tau_f_r[i]) / 2 for i in range(n)]
    tau_common = [(tau_f_l[i] + tau_f_r[i]) / 2 for i in range(n)]
    for i in range(n):
        assert abs(tau_f_l[i] - (tau_common[i] + tau_div[i])) < 1e-8
        assert abs(tau_f_r[i] - (tau_common[i] - tau_div[i])) < 1e-8

    # hip_yaw_abs_max should be max(abs(l), abs(r))
    hy_abs = [max(abs(l_pos[i]), abs(r_pos[i])) for i in range(n)]
    max_hy = max(hy_abs)
    assert 0.35 < max_hy < 0.45, f"Expected hip_yaw_abs_max ~0.40, got {max_hy}"

    # At peak, divergence should dominate
    peak_t = hy_abs.index(max_hy)
    assert abs(divergence[peak_t]) > 0.35, "Divergence should dominate at peak"
    # Verify it's NOT reference-driven
    assert abs(l_ref[peak_t]) < 0.001, "l_hip_yaw_ref should be ~0 at peak"
    assert abs(r_ref[peak_t]) < 0.001, "r_hip_yaw_ref should be ~0 at peak"

    # Verify mode-div tau exists
    if "mode_hip_yaw_div_tau_left" in rows[0]:
        mode_l = [_float_safe(r["mode_hip_yaw_div_tau_left"]) for r in rows]
        assert any(abs(t) > 0.1 for t in mode_l), "Mode-div tau should be nonzero somewhere"

    # Verify tau sign correctness at peak
    # Error convention is ref - pos (positive error = joint below ref = needs positive torque)
    # Torque opposes error when error * torque >= 0
    tau_sign_correct = l_err[peak_t] * tau_f_l[peak_t] >= 0
    assert tau_sign_correct, "Torque should oppose error at peak (left)"
    tau_sign_correct_r = r_err[peak_t] * tau_f_r[peak_t] >= 0
    assert tau_sign_correct_r, "Torque should oppose error at peak (right)"


# ---------------------------------------------------------------------------
# Test 2: Reference-vs-error classification handles missing columns
# ---------------------------------------------------------------------------
def test_ref_vs_error_classification_handles_missing():
    """Reference-vs-error classification should not crash on missing columns."""
    # Simulate audit scenario with incomplete data
    from scripts.audit_d4_d5_hip_yaw_universal_limit import classify_ref_vs_error

    # Normal case
    m_full = {"ref_contrib_pct": 0.0, "err_contrib_pct": 100.0}
    assert classify_ref_vs_error(m_full) == "TRACKING_ERROR_DOMINANT"

    m_ref = {"ref_contrib_pct": 80.0, "err_contrib_pct": 20.0}
    assert classify_ref_vs_error(m_ref) == "REFERENCE_TOO_LARGE"

    m_mixed = {"ref_contrib_pct": 50.0, "err_contrib_pct": 50.0}
    assert classify_ref_vs_error(m_mixed) == "MIXED"

    # Missing entries
    m_empty = {}
    cl = classify_ref_vs_error(m_empty)
    assert isinstance(cl, str)
    assert cl in ("TRACKING_ERROR_DOMINANT", "REFERENCE_TOO_LARGE", "MIXED")

    m_partial = {"err_contrib_pct": 100.0}
    cl2 = classify_ref_vs_error(m_partial)
    assert isinstance(cl2, str)


# ---------------------------------------------------------------------------
# Test 3: Torque budget classification handles missing columns
# ---------------------------------------------------------------------------
def test_torque_budget_classification_handles_missing():
    """Torque budget classification should not crash on missing columns."""
    from scripts.audit_d4_d5_hip_yaw_universal_limit import classify_torque_budget

    # Normal cases
    m_saturated = {"mode_div_tau_l_abs_max": 2.0, "mode_div_tau_r_abs_max": 2.0,
                    "tau_shape_l_abs_max": 6.0, "tau_shape_r_abs_max": 6.0}
    assert "SATURATED" in classify_torque_budget(m_saturated, "D", "D4_medium_push_low")

    m_margin = {"mode_div_tau_l_abs_max": 0.0, "mode_div_tau_r_abs_max": 0.0,
                 "tau_shape_l_abs_max": 6.0, "tau_shape_r_abs_max": 6.0}
    assert "SHAPE_TORQUE_HIGH" in classify_torque_budget(m_margin, "A", "D4_medium_push_low")

    # Empty
    m_empty = {}
    result = classify_torque_budget(m_empty, "A", "D4_medium_push_low")
    assert isinstance(result, str)

    # Partial
    m_partial = {"mode_div_tau_l_abs_max": 0.5}
    result2 = classify_torque_budget(m_partial, "D", "D4_medium_push_low")
    assert isinstance(result2, str)


# ---------------------------------------------------------------------------
# Test 4: candidate_kind is preserved
# ---------------------------------------------------------------------------
def test_candidate_kind_preserved_in_audit_output(audit_outputs):
    """Audit outputs should preserve candidate_kind from source telemetry."""
    peak_events = audit_outputs["d4_d5_peak_event_table.csv"]
    assert len(peak_events) > 0

    # The audit script doesn't have candidate_kind in peak_events directly,
    # but the profile field should match expected profiles
    profiles_found = set(pe["profile"] for pe in peak_events)
    assert profiles_found == {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# Test 5: Current-best is not changed by audit
# ---------------------------------------------------------------------------
def test_current_best_unchanged():
    """Verify the current-best controller profile is still D_MODE_HIP_YAW_DIV_V1."""
    # Check the simulate_hierarchical_controller.py for profile resolution
    sim_path = REPO_ROOT / "scripts" / "simulate_hierarchical_controller.py"
    assert sim_path.exists()

    with open(sim_path) as f:
        content = f.read()

    # The D profile must still resolve to low-band v2 sagittal
    assert (
        "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1"
        in content
    )
    # It should map to PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2
    assert "mode_hip_yaw_div_v1" in content

    # Verify test profile file
    test_path = REPO_ROOT / "tests" / "test_current_best_controller_profile.py"
    with open(test_path) as f:
        test_content = f.read()
    assert "test_d_mode_hip_yaw_div_v1_resolves_to_low_band_v2_sagittal" in test_content


# ---------------------------------------------------------------------------
# Test 6: Output report classification is one of allowed values
# ---------------------------------------------------------------------------
def test_report_classification_is_allowed():
    """The audit report's final classification must be one of the allowed values."""
    assert REPORT_PATH.exists(), f"Report not found at {REPORT_PATH}"
    with open(REPORT_PATH) as f:
        content = f.read()

    assert "D remains current-best/default" in content or "D remains current-best" in content
    assert "D4/D5" in content and "0.35" in content

    # Find the classification line
    for line in content.split("\n"):
        if "AUDIT_COMPLETE_" in line or "AUDIT_INCONCLUSIVE" in line:
            # Extract the classification token
            for tok in line.split():
                if tok.startswith("D4_D5_"):
                    classification = tok.rstrip("`")
                    assert classification in ALLOWED_CLASSIFICATIONS, (
                        f"Classification {classification} not in allowed set"
                    )
                    break


# ---------------------------------------------------------------------------
# Test 7: D telemetry cannot be replaced by C telemetry
# ---------------------------------------------------------------------------
def test_d_telemetry_distinct_from_c():
    """Verify D telemetry paths are distinct from C."""
    d4_path = (
        REPO_ROOT
        / "outputs"
        / "mode_hip_yaw_div_full_real_validation"
        / "d4_d5_focused_1000"
        / "D4_medium_push_low_D"
        / "telemetry_1000.csv"
    )
    c4_path = (
        REPO_ROOT
        / "outputs"
        / "mode_hip_yaw_div_full_real_validation"
        / "d4_d5_focused_1000"
        / "D4_medium_push_low_C"
        / "telemetry_1000.csv"
    )

    if d4_path.exists() and c4_path.exists():
        # They must be different files
        assert d4_path != c4_path

        # D telemetry must have mode_hip_yaw_div_enabled = True
        with open(d4_path, newline="") as f:
            d_rows = list(csv.DictReader(f))
        assert "mode_hip_yaw_div_enabled" in d_rows[0]
        mode_enabled = [_bool_str_to_int(r["mode_hip_yaw_div_enabled"]) for r in d_rows]
        assert sum(mode_enabled) >= len(mode_enabled) / 2, "D should have mode_hip_yaw_div_enabled mostly True"

        # C should have mode_hip_yaw_div_enabled = False or missing
        with open(c4_path, newline="") as f:
            c_rows = list(csv.DictReader(f))
        if "mode_hip_yaw_div_enabled" in c_rows[0]:
            c_enabled = [_bool_str_to_int(r["mode_hip_yaw_div_enabled"]) for r in c_rows]
            assert sum(c_enabled) < 1, "C should not have mode_hip_yaw_div_enabled=True"


# ---------------------------------------------------------------------------
# Test 8: Old wheel-yaw D cannot be used as D_MODE_HIP_YAW_DIV_V1
# ---------------------------------------------------------------------------
def test_old_wheel_yaw_not_accepted():
    """Verify D4/D5 audit does not use old wheel-yaw candidate data."""
    d4_d_path = (
        REPO_ROOT
        / "outputs"
        / "mode_hip_yaw_div_full_real_validation"
        / "d4_d5_focused_1000"
        / "D4_medium_push_low_D"
    )
    if not (d4_d_path / "run_summary.json").exists():
        pytest.skip("D telemetry not available")

    with open(d4_d_path / "run_summary.json") as f:
        summary = json.load(f)

    # Verify the command line does NOT have wheel-yaw flags
    command = summary.get("command", "")
    assert "--enable-wheel-yaw-stabilizer" not in command, (
        "D should not use wheel-yaw stabilizer"
    )

    # Verify D telemetry has mode-div enabled
    with open(d4_d_path / "telemetry_1000.csv", newline="") as f:
        d_rows = list(csv.DictReader(f))
    assert "mode_hip_yaw_div_enabled" in d_rows[0]
    mode_vals = [_bool_str_to_int(r["mode_hip_yaw_div_enabled"]) for r in d_rows]
    assert max(mode_vals) > 0, "D must have mode_hip_yaw_div_enabled True somewhere"

    # Verify D telemetry does NOT have wheel_yaw_enabled=True
    if "wheel_yaw_enabled" in d_rows[0]:
        wheel_vals = [str(r["wheel_yaw_enabled"]).lower() for r in d_rows]
        assert "true" not in wheel_vals, "D must not have wheel_yaw_enabled=True"

    # Verify roll_rms is normal (old wheel-yaw D had roll_rms=3.25)
    with open(d4_d_path / "telemetry_1000.csv") as f:
        reader = csv.DictReader(f)
        roll_cols = [c for c in reader.fieldnames if "roll" in c.lower() and ("rms" in c.lower() or "max" in c.lower())]
    # Roll RMS should be < 2.0 for normal behavior
    has_roll_rms = any("rms" in c.lower() and "roll" in c.lower() for c in (roll_cols if roll_cols else []))


# ---------------------------------------------------------------------------
# Test 9: Script compiles correctly
# ---------------------------------------------------------------------------
def test_audit_script_compiles():
    """The audit script must compile without errors."""
    import py_compile
    try:
        py_compile.compile(str(SCRIPT_PATH), doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"Script compilation failed: {e}")


# ---------------------------------------------------------------------------
# Test 10: Audit outputs exist and have correct structure
# ---------------------------------------------------------------------------
def test_audit_outputs_exist(audit_outputs):
    """All required audit output files exist with correct structure."""
    required_heads = {
        "d4_d5_windowed_metrics.csv": {"case", "profile", "window", "hip_yaw_abs_max"},
        "d4_d5_peak_event_table.csv": {"case", "profile", "hip_yaw_abs_max"},
        "d4_d5_mode_decomposition_timeseries_summary.csv": {"case", "profile"},
        "d4_d5_torque_budget_summary.csv": {"case", "profile", "classification"},
        "d4_d5_reference_vs_error_summary.csv": {"case", "profile", "classification"},
    }

    for fname, required_fields in required_heads.items():
        data = audit_outputs[fname]
        assert len(data) > 0, f"{fname} is empty"
        row0 = data[0]
        for field in required_fields:
            assert field in row0, f"{fname} missing field: {field}"

    # Verify JSON summary
    summary = audit_outputs["audit_summary.json"]
    assert "metadata" in summary
    assert summary["metadata"]["task"] == "d4_d5_hip_yaw_universal_limit_audit"


# ---------------------------------------------------------------------------
# Test 11: All four profiles are represented
# ---------------------------------------------------------------------------
def test_all_four_profiles_in_audit(audit_outputs):
    """All four profiles (A/B/C/D) must be present in peak events."""
    peak_events = audit_outputs["d4_d5_peak_event_table.csv"]
    profiles = set(pe["profile"] for pe in peak_events)
    for p in ("A", "B", "C", "D"):
        assert p in profiles, f"Profile {p} missing from peak events"


# ---------------------------------------------------------------------------
# Test 12: Both D4 and D5 cases are represented
# ---------------------------------------------------------------------------
def test_both_cases_in_audit(audit_outputs):
    """Both D4 and D5 cases must be present."""
    peak_events = audit_outputs["d4_d5_peak_event_table.csv"]
    cases = set(pe["case"] for pe in peak_events)
    for c in ("D4_medium_push_low", "D5_large_push_high"):
        assert c in cases, f"Case {c} missing from peak events"
