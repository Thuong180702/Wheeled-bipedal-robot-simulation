"""Phase 3D.3-C4 — Benchmark output schema validation tests.

Validates the structure and content of the benchmark output JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "phase3d3_incremental_qp"

ALLOWED_VERDICTS = frozenset({
    "INCREMENTAL_QP_CORRECTNESS_PASS",
    "INCREMENTAL_QP_CORRECTNESS_FAIL",
    "PARTIAL_SPEEDUP_NOT_FULLY_UNBLOCKED",
    "CLOSED_LOOP_EVALUATION_UNBLOCKED",
    "REALTIME_CANDIDATE_STRONG",
    "INCREMENTAL_QP_INSUFFICIENT",
})

FORBIDDEN_VERDICTS = frozenset({
    "REALTIME_READY",
    "PRODUCTION_READY",
    "WBC_PROMOTED",
    "DEFAULT_CONTROLLER_UPDATED",
})


# ═══════════════════════════════════════════════════════════════════════════════
# TestIncrementalQPBenchmarkOutputSchema
# ═══════════════════════════════════════════════════════════════════════════════

class TestIncrementalQPBenchmarkOutputSchema:
    """Tests for the benchmark JSON output schema."""

    # ── Fixtures ────────────────────────────────────────────────────────────

    @pytest.fixture
    def benchmark_data(self):
        """Load benchmark JSON, skipping if it hasn't been generated yet."""
        path = OUTPUT_DIR / "incremental_qp_benchmark.json"
        if not path.exists():
            pytest.skip("Benchmark not yet generated — run "
                        "scripts/phase3d3_incremental_qp_benchmark.py first")
        with open(path) as f:
            return json.load(f)

    @pytest.fixture
    def verdict_data(self):
        """Load verdict JSON, skipping if it hasn't been generated yet."""
        path = OUTPUT_DIR / "incremental_qp_verdict.json"
        if not path.exists():
            pytest.skip("Verdict not yet generated — run "
                        "scripts/phase3d3_incremental_qp_benchmark.py first")
        with open(path) as f:
            return json.load(f)

    @pytest.fixture
    def correctness_data(self):
        """Load correctness JSON, skipping if it hasn't been generated yet."""
        path = OUTPUT_DIR / "incremental_qp_correctness.json"
        if not path.exists():
            pytest.skip("Correctness audit not yet generated — run "
                        "scripts/phase3d3_incremental_qp_correctness_audit.py first")
        with open(path) as f:
            return json.load(f)

    # ── Benchmark JSON tests ────────────────────────────────────────────────

    def test_benchmark_has_required_keys(self, benchmark_data):
        """Benchmark JSON must contain the standard top-level keys."""
        for key in [
            "timestamp",
            "config",
            "verdict",
            "full_rebuild_results",
            "incremental_results",
        ]:
            assert key in benchmark_data, f"Missing key: {key}"

    def test_benchmark_config_has_task_and_rolling_mode(self, benchmark_data):
        """Config section must specify task_mode and rolling_mode."""
        config = benchmark_data["config"]
        assert "task_mode" in config
        assert "rolling_mode" in config

    def test_benchmark_verdict_is_allowed(self, benchmark_data):
        """Verdict must be one of the allowed strings."""
        verdict = benchmark_data["verdict"]["verdict"]
        assert verdict in ALLOWED_VERDICTS, (
            f"Unknown verdict: {verdict}"
        )

    def test_benchmark_verdict_is_not_forbidden(self, benchmark_data):
        """Verdict must NOT be any of the forbidden strings."""
        verdict = benchmark_data["verdict"]["verdict"]
        assert verdict not in FORBIDDEN_VERDICTS, (
            f"Forbidden verdict produced: {verdict}"
        )

    def test_benchmark_verdict_has_timing_fields(self, benchmark_data):
        """Verdict dict must contain timing statistics."""
        v = benchmark_data["verdict"]
        for key in [
            "incr_mean_ms",
            "incr_p95_ms",
            "full_mean_ms",
            "speedup_ratio",
            "thresholds_explanation",
        ]:
            assert key in v, f"Missing verdict field: {key}"

    def test_incremental_results_have_path_field(self, benchmark_data):
        """Every incremental result entry must have path='incremental'."""
        incr = benchmark_data["incremental_results"]
        assert len(incr) > 0, "No incremental results — benchmark may be empty"
        for entry in incr:
            assert entry["path"] == "incremental", (
                f"Incremental entry has wrong path: {entry['path']}"
            )

    def test_full_rebuild_results_have_path_field(self, benchmark_data):
        """Every full rebuild result entry must have path='full_rebuild'."""
        full = benchmark_data["full_rebuild_results"]
        assert len(full) > 0, "No full rebuild results — benchmark may be empty"
        for entry in full:
            assert entry["path"] == "full_rebuild", (
                f"Full rebuild entry has wrong path: {entry['path']}"
            )

    def test_all_results_have_time_s_field(self, benchmark_data):
        """Every timing entry must have a non-negative time_s."""
        for entry in (benchmark_data["full_rebuild_results"]
                      + benchmark_data["incremental_results"]):
            assert "time_s" in entry, f"Missing time_s in entry"
            assert entry["time_s"] >= 0, f"Negative time_s: {entry['time_s']}"

    def test_all_results_have_solve_success_field(self, benchmark_data):
        """Every entry must have a boolean solve_success."""
        for entry in (benchmark_data["full_rebuild_results"]
                      + benchmark_data["incremental_results"]):
            assert "solve_success" in entry
            assert isinstance(entry["solve_success"], bool)

    # ── Verdict JSON tests ──────────────────────────────────────────────────

    def test_verdict_json_has_keys(self, verdict_data):
        """Verdict JSON must have timestamp and verdict fields."""
        for key in ["timestamp", "verdict", "incr_mean_ms", "full_mean_ms",
                     "speedup_ratio"]:
            assert key in verdict_data, f"Missing key in verdict JSON: {key}"

    def test_verdict_json_verdict_allowed(self, verdict_data):
        """Verdict string in verdict JSON must be allowed."""
        verdict = verdict_data["verdict"]
        assert verdict in ALLOWED_VERDICTS, f"Unknown verdict: {verdict}"
        assert verdict not in FORBIDDEN_VERDICTS, f"Forbidden verdict: {verdict}"

    def test_verdict_consistent_with_benchmark(self, benchmark_data, verdict_data):
        """Verdict JSON should agree with the benchmark JSON verdict."""
        bv = benchmark_data["verdict"]["verdict"]
        vv = verdict_data["verdict"]
        assert bv == vv, (
            f"Verdict mismatch: benchmark={bv}, verdict_json={vv}"
        )

    # ── Correctness JSON tests ──────────────────────────────────────────────

    def test_correctness_json_has_required_keys(self, correctness_data):
        """Correctness audit JSON must have timestamp, verdict, cases, thresholds."""
        for key in ["timestamp", "verdict", "cases", "thresholds"]:
            assert key in correctness_data, f"Missing key in correctness JSON: {key}"

    def test_correctness_cases_are_non_empty(self, correctness_data):
        """Correctness audit must have at least one test case."""
        cases = correctness_data["cases"]
        assert len(cases) > 0, "No test cases in correctness audit"

    def test_correctness_cases_have_required_fields(self, correctness_data):
        """Each correctness case must have standard fields."""
        for case in correctness_data["cases"]:
            assert "case" in case
            assert "pass" in case
            assert isinstance(case["pass"], bool)

    def test_correctness_thresholds_have_tolerances(self, correctness_data):
        """Thresholds dict must contain tau, P, and A tolerances."""
        thresholds = correctness_data["thresholds"]
        for key in ["tau_tol", "p_stale_tol", "a_stale_tol"]:
            assert key in thresholds, f"Missing threshold: {key}"

    # ── CSV tests ───────────────────────────────────────────────────────────

    @pytest.fixture
    def csv_data(self):
        """Check CSV file existence, skip if not generated."""
        path = OUTPUT_DIR / "incremental_qp_timing.csv"
        if not path.exists():
            pytest.skip("CSV not yet generated")
        return path

    def test_csv_has_header_and_data(self, csv_data):
        """CSV must have a header row and at least one data row."""
        import csv
        with open(csv_data, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) >= 2, "CSV must have header + at least 1 data row"
        assert rows[0] == ["path", "time_s", "time_ms"], (
            f"Unexpected CSV header: {rows[0]}"
        )

    def test_csv_row_count_matches_benchmark(self, benchmark_data, csv_data):
        """CSV row count (excluding header) must match total benchmark entries."""
        import csv
        total_entries = (
            len(benchmark_data["full_rebuild_results"])
            + len(benchmark_data["incremental_results"])
        )
        with open(csv_data, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        data_rows = len(rows) - 1  # exclude header
        assert data_rows == total_entries, (
            f"CSV has {data_rows} data rows but benchmark has {total_entries} entries"
        )
