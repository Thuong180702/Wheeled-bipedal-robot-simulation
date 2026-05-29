"""Regression tests for removal of failed E0c/E0d runtime branches."""

from pathlib import Path


def test_no_e0c_or_e0d_runtime_control_branches_remain_in_simulation_script():
    script = Path("scripts/simulate_hierarchical_controller.py").read_text(encoding="utf-8")

    forbidden_tokens = [
        "e0c_enabled",
        "e0d_enabled",
        "e0c_telemetry",
        "e0d_telemetry",
        "cp_bias_total_m",
        "cp_bias_final_m",
        "position_reference_y_m",
        "e0d_prev_desired_velocity",
    ]

    for token in forbidden_tokens:
        assert token not in script


def test_failed_e0_reports_are_preserved_as_documentation_only():
    report_dir = Path("outputs/balance_core_position_containment")

    expected_reports = [
        report_dir / "e0b_failure_analysis.md",
        report_dir / "e0c_failure_analysis.md",
        report_dir / "e0d_phase_aware_report.md",
        report_dir / "position_containment_summary.md",
    ]

    for path in expected_reports:
        assert path.exists(), f"Missing preserved report: {path}"
