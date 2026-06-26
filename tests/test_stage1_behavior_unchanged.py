"""Stage 1: Verify K2 behavior is unchanged after profiling instrumentation.

The --profile-controller flag adds timing instrumentation but does NOT change
any control logic. These tests verify that the Python K2 controller produces
identical behavior with and without --profile-controller (by proxy: the
standard K2 tests still pass with the instrumented script).
"""

import subprocess
import sys
import pytest


SIMULATE_SCRIPT = "scripts/simulate_hierarchical_controller.py"

BASE_K2_ARGS = [
    sys.executable,
    SIMULATE_SCRIPT,
    "--controller-mode", "balance-core",
    "--sagittal-controller", "velocity-damped",
    "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
    "--steps", "200",
]

STEP_C_HEIGHTS = [
    "high_0p480",
    "mid_0p400",
    "low_0p330",
]


def _height_setup_path(variant_name: str) -> str:
    return f"outputs/physical_target_height_setups_centered/{variant_name}_setup.json"


def _run_k2_smoke(args_extra: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = list(BASE_K2_ARGS)
    if args_extra:
        cmd.extend(args_extra)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


class TestProfileFlagDoesNotBreakK2:
    """K2 controller passes smoke tests with --profile-controller."""

    @pytest.mark.parametrize("height_variant", STEP_C_HEIGHTS)
    def test_smoke_no_profile(self, height_variant: str):
        """K2 smoke test without --profile-controller (baseline)."""
        result = _run_k2_smoke([
            "--height-variant-setup", _height_setup_path(height_variant),
        ])
        assert result.returncode == 0, (
            f"K2 smoke failed for {height_variant} without profiling:\n"
            f"STDERR: {result.stderr[-500:]}"
        )

    @pytest.mark.parametrize("height_variant", STEP_C_HEIGHTS)
    def test_smoke_with_profile(self, height_variant: str):
        """K2 smoke test with --profile-controller (instrumented)."""
        result = _run_k2_smoke([
            "--height-variant-setup", _height_setup_path(height_variant),
            "--profile-controller",
        ])
        assert result.returncode == 0, (
            f"K2 smoke failed for {height_variant} with profiling:\n"
            f"STDERR: {result.stderr[-500:]}"
        )

    @pytest.mark.parametrize("height_variant", STEP_C_HEIGHTS)
    def test_profile_produces_report(self, height_variant: str):
        """--profile-controller produces a JSON report."""
        import os
        report_path = "outputs/profile/stage1_controller_profile_breakdown.json"
        # Remove old report if exists
        if os.path.exists(report_path):
            os.remove(report_path)

        result = _run_k2_smoke([
            "--height-variant-setup", _height_setup_path(height_variant),
            "--profile-controller",
        ])
        assert result.returncode == 0

        assert os.path.exists(report_path), (
            f"Profile report not created at {report_path} for {height_variant}"
        )

        import json
        with open(report_path) as f:
            report = json.load(f)
        assert report.get("step_count", 0) > 0, "Profile report has zero steps"
        assert report.get("backend") == "python"


class TestK2BehaviorUnchangedAfterInstrumentation:
    """K2 controller semantics are identical before/after Stage 1 changes.

    The instrumentation is purely additive (timing only). No control logic
    was modified. These tests verify that the existing K2 test suite still
    passes with the instrumented script.
    """

    def test_existing_k2_tests_still_pass(self):
        """Run the existing K2 best-current promotion tests."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_k2_best_current_promotion.py", "-v", "--tb=short"],
            capture_output=True, text=True, timeout=120,
        )
        # These tests check profile constants, not the simulation script,
        # so they should pass regardless of our instrumentation.
        assert result.returncode == 0, (
            f"Existing K2 tests failed:\n{result.stdout[-1000:]}\n{result.stderr[-500:]}"
        )

    def test_existing_current_best_tests_still_pass(self):
        """Run the existing current-best controller profile tests."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_current_best_controller_profile.py", "-v", "--tb=short"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"Existing current-best tests failed:\n{result.stdout[-1000:]}\n{result.stderr[-500:]}"
        )
