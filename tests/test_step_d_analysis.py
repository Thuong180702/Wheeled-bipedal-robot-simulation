"""Tests for scripts/analyze_step_d.py."""
import csv
import json
import pathlib
import sys

import pytest

# Paths (repo root is one level up from tests/)
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
ANALYZE_SCRIPT = SCRIPTS_DIR / "analyze_step_d.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_safe_row(case_id: str, profile: str, max_abs: float = 0.10) -> dict:
    """Produce a single CSV row dict that will pass the safety_ok check."""
    return {
        "case_id": case_id,
        "height": "high_0p480",
        "steps": "1000",
        "push_mag_N": "0",
        "push_dur": "5",
        "push_int": "150",
        "profile": profile,
        "fell": "False",
        "term_reason": "",
        "min_drift": "-0.02",
        "max_drift": "0.03",
        "max_abs": str(max_abs),
        "p2p": "0.05",
        "pos_pct": "60.0",
        "neg_pct": "40.0",
        "zero_crossings": "5",
        "out15_pct": "0.0",
        "out25_pct": "0.0",
        "pitch_max_abs_deg": "5.0",
        "roll_rms_deg": "0.5",
        "hip_yaw_abs_max_rad": "0.05",
        "yaw_drift_max_rad": "0.02",
        "wbc_authority_rows": "0",
        "wbc_owner_rows": "0",
        "hidden_torque_max": "0.0",
        "ownership_violation_max": "0",
    }


def _write_dummy_csv(
    path: pathlib.Path,
    rows: list[dict],
) -> pathlib.Path:
    """Write a CSV from a list of dict rows, ensuring all keys are covered."""
    if not rows:
        # Write header only (empty file)
        path.write_text("dummy\n", encoding="utf-8")
        return path
    fieldnames = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_csv_all_cases(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a CSV with all 6 PUSH_CASES for profiles A and C, all safe.

    Profile A rows get max_abs = 0.12; profile C rows get max_abs = 0.10
    so that C <= A + 0.05 holds (0.10 <= 0.12 + 0.05 = 0.17 → True).
    """
    rows: list[dict] = []
    case_ids = [
        "D1_small_push_high",
        "D2_medium_push_high",
        "D3_small_push_low",
        "D4_medium_push_low",
        "D5_large_push_high",
        "D6_random_push_high",
    ]
    for cid in case_ids:
        rows.append(_make_safe_row(cid, "A", max_abs=0.12))
        rows.append(_make_safe_row(cid, "C", max_abs=0.10))
    return _write_dummy_csv(tmp_path / "dummy_metrics.csv", rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnalyzeStepD:
    """Integration tests for scripts/analyze_step_d.py."""

    def test_json_summary_created(self, dummy_csv_all_cases: pathlib.Path) -> None:
        """Run analysis on a full dummy CSV and verify the JSON summary."""
        out_json = dummy_csv_all_cases.with_name("summary.json")
        out_report = dummy_csv_all_cases.with_name("report.md")

        rc, out, err = subprocess_run([
            sys.executable,
            str(ANALYZE_SCRIPT),
            "--metrics-csv", str(dummy_csv_all_cases),
            "--output-json", str(out_json),
            "--output-report", str(out_report),
        ])
        assert rc == 0, (
            f"analyze_step_d.py exited with code {rc}\n"
            f"stdout: {out}\nstderr: {err}"
        )

        assert out_json.is_file(), f"JSON summary not found at {out_json}"
        with open(out_json, encoding="utf-8") as f:
            data = json.load(f)

        assert data["classification"] == "STEP_D_RANDOM_PUSH_PASS", (
            f"Expected STEP_D_RANDOM_PUSH_PASS, got {data['classification']}"
        )
        assert data["must_not_fall_pass"] is True
        assert data["any_hard_fail"] is False
        assert data["c_not_worse_count"] == data["total_cases"]
        assert data["total_cases"] == 6

    def test_report_created(self, dummy_csv_all_cases: pathlib.Path) -> None:
        """Verify the markdown report is written."""
        out_json = dummy_csv_all_cases.with_name("summary2.json")
        out_report = dummy_csv_all_cases.with_name("report.md")

        rc, out, err = subprocess_run([
            sys.executable,
            str(ANALYZE_SCRIPT),
            "--metrics-csv", str(dummy_csv_all_cases),
            "--output-json", str(out_json),
            "--output-report", str(out_report),
        ])
        assert rc == 0, (
            f"analyze_step_d.py exited with code {rc}\n"
            f"stdout: {out}\nstderr: {err}"
        )

        assert out_report.is_file(), f"Report not found at {out_report}"
        text = out_report.read_text(encoding="utf-8")
        assert "Gate Summary" in text
        assert "STEP_D_RANDOM_PUSH_PASS" in text
        assert "B2v2 Baseline" in text
        assert "Low-Band v2 Candidate" in text

    def test_missing_csv_exits(self, tmp_path: pathlib.Path) -> None:
        """Running with a non-existent CSV should exit with code 1."""
        missing = tmp_path / "nonexistent.csv"
        rc, out, err = subprocess_run([
            sys.executable,
            str(ANALYZE_SCRIPT),
            "--metrics-csv", str(missing),
        ])
        assert rc == 1, (
            f"Expected exit code 1 for missing CSV, got {rc}\n"
            f"stdout: {out}\nstderr: {err}"
        )

    def test_empty_csv_fails(self, tmp_path: pathlib.Path) -> None:
        """Running with an empty CSV (header only, no data) should exit non-zero."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("dummy\n", encoding="utf-8")
        rc, out, err = subprocess_run([
            sys.executable,
            str(ANALYZE_SCRIPT),
            "--metrics-csv", str(empty_csv),
        ])
        assert rc == 1, (
            f"Expected exit code 1 for empty CSV, got {rc}\n"
            f"stdout: {out}\nstderr: {err}"
        )


def subprocess_run(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
