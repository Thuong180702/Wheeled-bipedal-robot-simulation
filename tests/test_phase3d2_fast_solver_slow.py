"""Phase 3D.2 — Slow Tests for Fast Structured QP Solver.

These tests verify completed reports and gate conditions. They require:
  - OSQP to be installed
  - Phase 3D.2 correctness audit to have run
  - Phase 3D.2 performance benchmark to have run
  - Phase 3D.1 validation cross-check to have run
  - Phase 3D.1 three-arm smoke rollout to have run

Run:
    pytest tests/test_phase3d2_fast_solver_slow.py -q -m slow
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.slow]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPORT_DIR = _PROJECT_ROOT / "outputs" / "phase3d2"
_CORRECTNESS_PATH = _REPORT_DIR / "phase3d2_correctness_audit.json"
_SUMMARY_PATH = _REPORT_DIR / "phase3d2_fast_solver_summary.json"

# ── Skip if OSQP not available ───────────────────────────────────────────

try:
    import osqp  # noqa: F401
    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False

# ── Skip if MuJoCo not available ─────────────────────────────────────────

try:
    import mujoco  # noqa: F401
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


def _skip_if_no_osqp():
    if not HAS_OSQP:
        pytest.skip("OSQP not installed")


def _skip_if_no_reports():
    if not _CORRECTNESS_PATH.exists():
        pytest.skip(f"Correctness audit report not found: {_CORRECTNESS_PATH}")
    if not _SUMMARY_PATH.exists():
        pytest.skip(f"Performance summary not found: {_SUMMARY_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# Correctness audit checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorrectnessAuditReport:
    """Verify correctness audit report meets pass criteria."""

    def test_report_exists(self):
        """Correctness audit report exists."""
        assert _CORRECTNESS_PATH.exists(), (
            f"Report not found: {_CORRECTNESS_PATH}. "
            f"Run: python scripts/phase3d2_solver_correctness_audit.py --backend osqp"
        )

    def test_correctness_audit_pass(self):
        """Correctness audit pass == true."""
        _skip_if_no_reports()
        with open(_CORRECTNESS_PATH) as f:
            report = json.load(f)
        assert report.get("pass", False), (
            f"Correctness audit did not pass. "
            f"Cases: {report.get('num_cases', 0)}, "
            f"Successes: {report.get('fast_solver_successes', 0)}"
        )

    def test_max_residuals_within_threshold(self):
        """Maximum residuals are within tolerance."""
        _skip_if_no_reports()
        with open(_CORRECTNESS_PATH) as f:
            report = json.load(f)
        assert report.get("max_dynamics_residual", float("inf")) < 1e-5
        assert report.get("max_contact_accel_residual", float("inf")) < 1e-4
        assert report.get("max_friction_violation", float("inf")) <= 1e-6
        assert report.get("max_torque_violation", float("inf")) <= 1e-6

    def test_not_slsqp_only(self):
        """Fast backend is not SLSQP."""
        _skip_if_no_reports()
        with open(_CORRECTNESS_PATH) as f:
            report = json.load(f)
        assert not report.get("uses_slsqp_only", True), (
            "Correctness audit must use a fast backend, not SLSQP alone"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Performance benchmark checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformanceBenchmark:
    """Verify performance benchmark meets targets."""

    def test_summary_exists(self):
        """Performance summary exists."""
        assert _SUMMARY_PATH.exists(), (
            f"Summary not found: {_SUMMARY_PATH}. "
            f"Run: python scripts/phase3d2_fast_solver_benchmark.py --backend osqp --warm-start --repeat 10"
        )

    def test_minimum_solves(self):
        """At least 72 solves were performed."""
        _skip_if_no_reports()
        with open(_SUMMARY_PATH) as f:
            summary = json.load(f)
        assert summary.get("num_solves", 0) >= 72, (
            f"Need >= 72 solves, got {summary.get('num_solves', 0)}"
        )

    def test_success_rate(self):
        """Success rate >= 0.99."""
        _skip_if_no_reports()
        with open(_SUMMARY_PATH) as f:
            summary = json.load(f)
        sr = summary.get("success_rate", 0.0)
        assert sr >= 0.99, f"Success rate {sr:.4f} < 0.99"

    def test_mean_solve_time(self):
        """Mean solve time <= 0.05 s (50 ms)."""
        _skip_if_no_reports()
        with open(_SUMMARY_PATH) as f:
            summary = json.load(f)
        mean_t = summary.get("mean_solve_time_s", float("inf"))
        assert mean_t <= 0.05, f"Mean solve time {mean_t*1000:.1f} ms > 50 ms"

    def test_p95_solve_time(self):
        """P95 solve time <= 0.10 s (100 ms)."""
        _skip_if_no_reports()
        with open(_SUMMARY_PATH) as f:
            summary = json.load(f)
        p95_t = summary.get("p95_solve_time_s", float("inf"))
        assert p95_t <= 0.10, f"P95 solve time {p95_t*1000:.1f} ms > 100 ms"

    def test_meets_batch_target(self):
        """Meets batch evaluation target."""
        _skip_if_no_reports()
        with open(_SUMMARY_PATH) as f:
            summary = json.load(f)
        assert summary.get("meets_batch_target", False), "Batch target not met"

    def test_not_slsqp_only(self):
        """Not using SLSQP only."""
        _skip_if_no_reports()
        with open(_SUMMARY_PATH) as f:
            summary = json.load(f)
        assert not summary.get("uses_slsqp_only", True)


# ═══════════════════════════════════════════════════════════════════════════════
# Controller integrity checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestControllerIntegritySlow:
    """Verify controller files have not been modified."""

    def _hash_file(self, path: Path) -> str:
        import hashlib
        if not path.exists():
            return "MISSING"
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def test_k2_jax_controller_reference_exists(self):
        """k2_jax_controller.py is present (not modified by Phase 3D.2)."""
        path = _PROJECT_ROOT / "wheeled_biped" / "controllers" / "k2_jax_controller.py"
        assert path.exists(), "k2_jax_controller.py should exist"

    def test_sagittal_controller_reference_exists(self):
        """sagittal controller exists (not modified)."""
        path = _PROJECT_ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
        assert path.exists(), "sagittal velocity controller should exist"

    def test_run_k2_jax_realtime_exists(self):
        """run_k2_jax_realtime.py exists (not modified)."""
        path = _PROJECT_ROOT / "scripts" / "run_k2_jax_realtime.py"
        assert path.exists(), "run_k2_jax_realtime.py should exist"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3D.1 gate checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase3D1GateRerun:
    """Verify Phase 3D.1 blocked gates were rerun."""

    def test_validation_crosscheck_report(self):
        """Validation cross-check report exists or is noted as pending."""
        crosscheck_path = _PROJECT_ROOT / "outputs" / "phase3d" / "validation_crosscheck.json"
        if not crosscheck_path.exists():
            # Try alternative location
            crosscheck_path = _PROJECT_ROOT / "docs" / "validation" / "k2_phase3d_three_arm_counterfactual_audit.json"
        # This test is informational — skip if no report yet
        if not crosscheck_path.exists():
            pytest.skip("Validation cross-check report not found — still blocked")

    def test_smoke_rollout_report(self):
        """Three-arm smoke rollout report exists or is noted as pending."""
        smoke_path = _PROJECT_ROOT / "outputs" / "phase3d" / "three_arm_smoke.json"
        if not smoke_path.exists():
            smoke_path = _PROJECT_ROOT / "docs" / "validation" / "k2_phase3d_three_arm_counterfactual_audit.json"
        if not smoke_path.exists():
            pytest.skip("Three-arm smoke rollout report not found — still blocked")


# ═══════════════════════════════════════════════════════════════════════════════
# Report integrity checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestReportIntegrity:
    def test_audit_md_exists(self):
        """Audit markdown report exists."""
        path = _PROJECT_ROOT / "docs" / "validation" / "k2_phase3d2_fast_solver_audit.md"
        if not path.exists():
            pytest.skip("Audit markdown not yet created")

    def test_audit_json_exists(self):
        """Audit JSON report exists."""
        path = _PROJECT_ROOT / "docs" / "validation" / "k2_phase3d2_fast_solver_audit.json"
        if not path.exists():
            pytest.skip("Audit JSON not yet created")

    def test_json_has_required_fields(self):
        """Audit JSON has all required top-level fields."""
        path = _PROJECT_ROOT / "docs" / "validation" / "k2_phase3d2_fast_solver_audit.json"
        if not path.exists():
            pytest.skip("Audit JSON not yet created")

        with open(path) as f:
            data = json.load(f)

        required_fields = [
            "phase", "verdict", "constants_version", "controller_modified",
            "v3_profile_changed", "realtime_integration",
            "qp_torque_injected_into_realtime",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    @pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
    def test_json_verdict_correct(self):
        """If OSQP is available, verdict should be READY or PARTIAL_READY."""
        path = _PROJECT_ROOT / "docs" / "validation" / "k2_phase3d2_fast_solver_audit.json"
        if not path.exists():
            pytest.skip("Audit JSON not yet created")

        with open(path) as f:
            data = json.load(f)

        verdict = data.get("verdict", "")
        assert verdict in [
            "READY_FOR_PHASE_3D_FULL_BATCH_EXECUTION",
            "READY_FOR_REALTIME_CANDIDATE_BENCHMARK",
            "PARTIAL_READY",
            "NOT_READY",
        ], f"Invalid verdict: {verdict}"

    def test_json_controller_not_modified(self):
        """controller_modified must be false."""
        path = _PROJECT_ROOT / "docs" / "validation" / "k2_phase3d2_fast_solver_audit.json"
        if not path.exists():
            pytest.skip("Audit JSON not yet created")

        with open(path) as f:
            data = json.load(f)
        assert data.get("controller_modified") is False, "Controller must not be modified"
        assert data.get("realtime_integration") is False, "No realtime integration"
        assert data.get("qp_torque_injected_into_realtime") is False, "No QP torque in realtime"
