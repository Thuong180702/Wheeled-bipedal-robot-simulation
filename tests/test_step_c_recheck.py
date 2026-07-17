"""Tests for Step C regression recheck verification.

This test validates that the step C recheck report exists and contains
the passing verdict for the Low-band v2 profile.
"""

import re
from pathlib import Path

REPORT_PATH = Path(__file__).parents[1] / "docs" / "validation" / "step_c_regression_recheck.md"


class TestStepCRecheckReport:
    """Verify the Step C regression recheck report exists and passes."""

    def test_report_exists(self):
        """The recheck report file must exist."""
        assert REPORT_PATH.is_file(), (
            f"Step C recheck report not found at {REPORT_PATH}"
        )

    def report_text(self):
        return REPORT_PATH.read_text(encoding="utf-8")

    def test_contains_verdict(self):
        """The report must contain the overall verdict marker."""
        text = self.report_text()
        assert "STEP_C_RECHECK_PASS" in text, (
            "Report does not contain STEP_C_RECHECK_PASS verdict"
        )

    def test_verdict_line(self):
        """The overall verdict line must read STEP_C_RECHECK_PASS."""
        text = self.report_text()
        matches = re.findall(r"STEP_C_RECHECK_(PASS|FAIL)", text)
        assert len(matches) >= 1, "No STEP_C_RECHECK verdict found in report"
        assert matches[-1] == "PASS", (
            f"Final verdict is not PASS; found: {matches}"
        )

    def test_summary_table_has_pass(self):
        """The summary table should not contain FAIL entries."""
        text = self.report_text()
        # Look at the Summary Table section
        in_summary = False
        for line in text.splitlines():
            if line.startswith("## Summary Table"):
                in_summary = True
                continue
            if in_summary and line.startswith("## "):
                break
            if in_summary and "|" in line and not line.startswith("|-"):
                if "FAIL" in line and "PASS" not in line:
                    assert False, f"Summary table contains FAIL-only entry: {line.strip()}"
