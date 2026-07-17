"""Phase 3D — Slow tests for three-arm counterfactual evaluation.

Validates completed audit reports: suite coverage, safety gates, classification
consistency, WBC solve rates, and controller integrity.

All tests marked @pytest.mark.slow — only run after report generation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pytest


# ── Paths ────────────────────────────────────────────────────────────────────

REPORT_JSON_PATH = PROJECT_ROOT / "docs" / "validation" / "k2_phase3d_three_arm_counterfactual_audit.json"
JSONL_PATH = PROJECT_ROOT / "outputs" / "phase3d_three_arm_counterfactual_results.jsonl"


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_report():
    """Load the report JSON if it exists."""
    if not REPORT_JSON_PATH.exists():
        return None
    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_entries():
    """Load all JSONL entries."""
    if not JSONL_PATH.exists():
        return []
    entries = []
    seen = set()
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = (entry.get("scenario"), entry.get("arm"), entry.get("suite"))
                if key not in seen:
                    seen.add(key)
                    entries.append(entry)
            except (json.JSONDecodeError, KeyError):
                continue
    return entries


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestPhase3DReportExists:
    """Verify report files exist and are valid."""

    def test_report_json_exists(self):
        """Report JSON file exists."""
        assert REPORT_JSON_PATH.exists(), f"Report not found: {REPORT_JSON_PATH}"

    def test_report_is_valid_json(self):
        """Report JSON parses correctly."""
        report = load_report()
        assert report is not None, "Report could not be loaded"
        assert "phase" in report
        assert report["phase"] == "3D"
        assert "verdict" in report

    def test_jsonl_exists(self):
        """JSONL results file exists."""
        assert JSONL_PATH.exists(), f"JSONL not found: {JSONL_PATH}"


@pytest.mark.slow
class TestSuiteCoverage:
    """Verify suite coverage requirements."""

    def test_standard_deterministic_suite_completed(self):
        """Standard deterministic suite has entries."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")
        tc = report.get("test_suite_coverage", {})
        std = tc.get("standard_deterministic", {})
        # Report should indicate completion status
        assert "completed" in std

    def test_deterministic_push_suite(self):
        """Deterministic push suite coverage in report."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")
        tc = report.get("test_suite_coverage", {})
        push = tc.get("deterministic_single_push", {})
        assert "completed" in push
        assert "num_scenarios" in push

    def test_long_horizon_suite(self):
        """Long-horizon suite coverage in report."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")
        tc = report.get("test_suite_coverage", {})
        lh = tc.get("long_horizon_3000", {})
        assert "completed" in lh
        assert "num_scenarios" in lh

    def test_random_push_suite(self):
        """Random push suite coverage in report."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")
        tc = report.get("test_suite_coverage", {})
        rp = tc.get("random_single_push_mild", {})
        assert "completed" in rp

    def test_legacy_suites_handled(self):
        """Legacy C/D/E suites are represented in report."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")
        tc = report.get("test_suite_coverage", {})
        for suite in ["legacy_c", "legacy_d", "legacy_e"]:
            assert suite in tc, f"Missing {suite} in report"


@pytest.mark.slow
class TestSafetyGates:
    """Verify safety gates are satisfied."""

    def test_wbc_solve_success_rate(self):
        """WBC solve success rate >= 0.99 if data exists."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")

        ca = report.get("counterfactual_audit", {})
        rate = ca.get("wbc_solve_success_rate")
        if rate is not None:
            assert rate >= 0.99, f"WBC solve rate {rate:.3f} < 0.99"

    def test_assist_falls_le_v3_falls(self):
        """Assist falls <= V3 falls."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")

        sc = report.get("safety_comparison", {})
        assist_falls = sc.get("assist_falls", 0)
        v3_falls = sc.get("v3_falls", 0)
        assert assist_falls <= v3_falls, f"Assist falls ({assist_falls}) > V3 falls ({v3_falls})"

    def test_assist_safety_fails_le_v3_safety_fails(self):
        """Assist safety fails <= V3 safety fails."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")

        sc = report.get("safety_comparison", {})
        assist_safety = sc.get("assist_safety_fails", 0)
        v3_safety = sc.get("v3_safety_fails", 0)
        assert assist_safety <= v3_safety, f"Assist safety ({assist_safety}) > V3 safety ({v3_safety})"

    def test_nan_inf_count_zero(self):
        """No NaN/Inf in results."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")

        sc = report.get("safety_comparison", {})
        assert sc.get("nan_inf_count", 0) == 0

    def test_torque_limit_violations_zero(self):
        """No torque limit violations."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")

        sc = report.get("safety_comparison", {})
        assert sc.get("torque_limit_violations", 0) == 0


@pytest.mark.slow
class TestControllerIntegrity:
    """Verify controller files unchanged."""

    def test_controller_modified_false(self):
        """controller_modified is False."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")
        assert report.get("controller_modified", True) is False

    def test_qp_torque_not_injected_realtime(self):
        """QP torque not injected into realtime."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")
        assert report.get("qp_torque_injected_into_realtime", True) is False

    def test_wbc_torque_offline_only(self):
        """WBC torque applied only to offline clones."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")
        assert report.get("wbc_torque_applied_only_to_offline_clones", False) is True

    def test_assist_torque_offline_only(self):
        """Assist torque applied only to offline clones."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")
        assert report.get("assist_torque_applied_only_to_offline_clones", False) is True

    def test_realtime_integration_false(self):
        """No realtime integration."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")
        assert report.get("realtime_integration", True) is False


@pytest.mark.slow
class TestClassificationConsistency:
    """Verify classification counts are consistent."""

    def test_classification_counts_sum_correctly(self):
        """Classification counts sum to number of scenarios."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")

        poc = report.get("physical_outcome_comparison", {})
        wbc = poc.get("wbc_only", {})
        assist = poc.get("assist", {})

        wbc_total = sum(wbc.values())
        assist_total = sum(assist.values())

        # Both should sum to the same total (or 0 if no data)
        assert wbc_total == assist_total, f"WBC total ({wbc_total}) != assist total ({assist_total})"

    def test_best_arm_counts_sum_to_total(self):
        """Best arm counts sum to scenario count."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")

        poc = report.get("physical_outcome_comparison", {})
        best = poc.get("best_arm_counts", {})
        best_total = sum(best.values())

        wbc = poc.get("wbc_only", {})
        wbc_total = sum(wbc.values())

        assert best_total == wbc_total, f"Best arm total ({best_total}) != scenario total ({wbc_total})"


@pytest.mark.slow
class TestReportFields:
    """Verify all required report fields present."""

    REQUIRED_TOP_LEVEL = [
        "phase", "verdict", "constants_version", "timestamp",
        "phase3c_prerequisite", "validation_crosscheck",
        "test_suite_coverage", "counterfactual_audit",
        "safety_comparison", "physical_outcome_comparison",
        "aggregate_ratios", "torque_comparison", "wbc_constraints",
        "performance", "controller_modified",
        "qp_torque_injected_into_realtime",
        "wbc_torque_applied_only_to_offline_clones",
        "assist_torque_applied_only_to_offline_clones",
        "realtime_integration", "limitations",
    ]

    def test_all_required_fields_present(self):
        """All required top-level fields present in report."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")

        for field in self.REQUIRED_TOP_LEVEL:
            assert field in report, f"Missing field: {field}"

    def test_phase3c_prerequisite_fields(self):
        """Phase 3C prerequisite fields present."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")

        pc = report.get("phase3c_prerequisite", {})
        assert "phase3c_ready" in pc
        assert "total_qp_solves_completed" in pc
        assert "hard_constraints_pass" in pc
        assert "controller_modified" in pc

    def test_counterfactual_audit_fields(self):
        """Counterfactual audit fields present."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")

        ca = report.get("counterfactual_audit", {})
        assert "baseline_controller" in ca
        assert "arms" in ca
        assert len(ca.get("arms", [])) == 3

    def test_verdict_is_valid(self):
        """Verdict is one of the valid values."""
        report = load_report()
        if report is None:
            pytest.skip("Report not found")

        valid_verdicts = [
            "READY_FOR_PHASE_3E_GUARDED_WBC_ASSIST_EXPERIMENT",
            "PARTIAL_READY",
            "NOT_READY",
            "PARTIAL_READY_AFTER_STANDARD",
            "PARTIAL_READY_AFTER_DETERMINISTIC_PUSH",
            "PARTIAL_READY_AFTER_RANDOM_PUSH",
            "PARTIAL_READY_AFTER_LONG_HORIZON",
        ]
        assert report["verdict"] in valid_verdicts, f"Invalid verdict: {report['verdict']}"
