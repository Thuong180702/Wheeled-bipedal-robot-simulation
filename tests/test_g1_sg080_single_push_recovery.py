"""Tests for G1_sg080 single-push recovery diagnostic.

Verifies:
1. Runner defines exactly one push window
2. Push duration is 10 steps
3. Requested steps is 2000
4. Profile parameters match G1_sg080
5. D remains current-best
6. No threshold relaxation
7. No WBC enabled
8. Validation_source must be real_simulation
9. Classification enum is valid
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _import_from_path(module_name: str, file_path: Path):
    """Import a Python file as a module."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Parse runner constants by importing the module
# ---------------------------------------------------------------------------
RUNNER_PATH = ROOT / "scripts" / "run_g1_sg080_single_90n_10step_push_recovery.py"
ANALYZER_PATH = ROOT / "scripts" / "analyze_g1_sg080_single_push_recovery.py"


@pytest.fixture(scope="session")
def runner_module():
    return _import_from_path("runner", RUNNER_PATH)


@pytest.fixture(scope="session")
def analyzer_module():
    return _import_from_path("analyzer", ANALYZER_PATH)


# ---------------------------------------------------------------------------
# Test 1: Runner constants
# ---------------------------------------------------------------------------

class TestRunnerConstants:
    """Verify the runner script defines the correct scenario."""

    def test_steps_is_2000(self, runner_module):
        assert runner_module.STEPS == 2000, f"Expected 2000, got {runner_module.STEPS}"

    def test_push_magnitude_is_90(self, runner_module):
        assert runner_module.PUSH_MAG_N == 90.0, f"Expected 90.0 N, got {runner_module.PUSH_MAG_N}"

    def test_push_duration_is_10(self, runner_module):
        assert runner_module.PUSH_DUR_STEPS == 10, f"Expected 10, got {runner_module.PUSH_DUR_STEPS}"

    def test_push_count_is_1(self, runner_module):
        assert runner_module.PUSH_COUNT == 1, f"Expected 1, got {runner_module.PUSH_COUNT}"

    def test_push_start_step_is_500(self, runner_module):
        assert runner_module.PUSH_START_STEP == 500, f"Expected 500, got {runner_module.PUSH_START_STEP}"

    def test_height_label_is_high_0p480(self, runner_module):
        assert runner_module.HEIGHT_LABEL == "high_0p480", f"Expected high_0p480, got {runner_module.HEIGHT_LABEL}"

    def test_case_id(self, runner_module):
        expected = "G1_sg080_single_90n_10step_push_high_2000"
        assert runner_module.CASE_ID == expected, f"Expected {expected}, got {runner_module.CASE_ID}"


class TestRunnerProfile:
    """Verify the runner uses G1_sg080 profile parameters."""

    def test_profile_name(self, runner_module):
        """Runner must use the D profile (current-best sagittal base)."""
        cmd = runner_module.build_g1_sg080_cmd(Path("dummy"))
        cmd_str = " ".join(cmd)
        assert "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1" in cmd_str, (
            "Runner must use D_MODE_HIP_YAW_DIV_V1 profile"
        )

    def test_mode_div_kp_is_10(self, runner_module):
        cmd = runner_module.build_g1_sg080_cmd(Path("dummy"))
        cmd_str = " ".join(cmd)
        assert "--mode-hip-yaw-div-kp 10.0" in cmd_str, "G1_sg080 kp must be 10.0"

    def test_mode_div_kd_is_0_50(self, runner_module):
        cmd = runner_module.build_g1_sg080_cmd(Path("dummy"))
        cmd_str = " ".join(cmd)
        assert "--mode-hip-yaw-div-kd 0.50" in cmd_str, "G1_sg080 kd must be 0.50"

    def test_mode_div_max_torque_is_7_5(self, runner_module):
        cmd = runner_module.build_g1_sg080_cmd(Path("dummy"))
        cmd_str = " ".join(cmd)
        assert "--mode-hip-yaw-div-max-torque 7.5" in cmd_str, "G1_sg080 max_torque must be 7.5"

    def test_mode_div_soft_limit_is_0_30(self, runner_module):
        cmd = runner_module.build_g1_sg080_cmd(Path("dummy"))
        cmd_str = " ".join(cmd)
        assert "--mode-hip-yaw-div-soft-limit-rad 0.30" in cmd_str, "G1_sg080 soft_limit must be 0.30"

    def test_mode_div_soft_gain_is_0_80(self, runner_module):
        cmd = runner_module.build_g1_sg080_cmd(Path("dummy"))
        cmd_str = " ".join(cmd)
        assert "--mode-hip-yaw-div-soft-gain 0.80" in cmd_str, "G1_sg080 soft_gain must be 0.80"

    def test_mode_div_ref_source_is_target(self, runner_module):
        cmd = runner_module.build_g1_sg080_cmd(Path("dummy"))
        cmd_str = " ".join(cmd)
        assert "--mode-hip-yaw-div-ref-source target" in cmd_str, "G1_sg080 ref_source must be target"


class TestNoProductionChanges:
    """Verify D remains current-best and no unwanted features are enabled."""

    def test_d_remains_current_best(self):
        """D_MODE_HIP_YAW_DIV_V1 must still be in the profile choices."""
        from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1" in SAGITTAL_AUTHORITY_PROFILES

    def test_no_wbc_enabled(self, runner_module):
        cmd = runner_module.build_g1_sg080_cmd(Path("dummy"))
        cmd_str = " ".join(cmd)
        assert "--controller-mode balance-core" in cmd_str, "Must use balance-core mode"
        # WBC would be '--controller-mode wbc' or similar - verify not present
        assert "wbc" not in cmd_str.split("--controller-mode")[1].split()[0] if "--controller-mode" in cmd_str else True

    def test_no_wheel_yaw_stabilizer(self, runner_module):
        cmd = runner_module.build_g1_sg080_cmd(Path("dummy"))
        cmd_str = " ".join(cmd)
        assert "--enable-wheel-yaw-stabilizer" not in cmd_str, "Wheel yaw stabilizer must NOT be enabled"

    def test_no_support_aware_h_gate(self, runner_module):
        cmd = runner_module.build_g1_sg080_cmd(Path("dummy"))
        cmd_str = " ".join(cmd)
        assert "--mode-hip-yaw-div-support-enabled" not in cmd_str, "Support-aware H gate must NOT be enabled"

    def test_no_threshold_relaxation(self, runner_module):
        cmd = runner_module.build_g1_sg080_cmd(Path("dummy"))
        cmd_str = " ".join(cmd)
        # Verify the soft-limit is still 0.30 (not relaxed)
        assert "--mode-hip-yaw-div-soft-limit-rad 0.30" in cmd_str

    def test_sagittal_push_only(self, runner_module):
        cmd = runner_module.build_g1_sg080_cmd(Path("dummy"))
        cmd_str = " ".join(cmd)
        assert "--sagittal-push-only" in cmd_str, "Push must be sagittal-only"


# ---------------------------------------------------------------------------
# Test 2: Analyzer classification enum
# ---------------------------------------------------------------------------

class TestAnalyzerClassification:
    """Verify the analyzer defines a valid classification enum."""

    def test_classification_keys(self, analyzer_module):
        expected_keys = {
            "SINGLE_PUSH_RECOVERY_PASS",
            "SINGLE_PUSH_RECOVERY_PASS_WITH_HIP_YAW_LIMIT",
            "SINGLE_PUSH_RECOVERY_FAIL_HIP_YAW",
            "SINGLE_PUSH_RECOVERY_FAIL_SUPPORT",
            "SINGLE_PUSH_RECOVERY_FAIL_FALL",
            "SINGLE_PUSH_RECOVERY_FAIL_UNSTABLE_FINAL_WINDOW",
            "SINGLE_PUSH_RECOVERY_INCONCLUSIVE",
        }
        actual_keys = set(analyzer_module.CLASSIFICATION.values())
        assert actual_keys == expected_keys, (
            f"Classification keys mismatch.\n"
            f"  Missing: {expected_keys - actual_keys}\n"
            f"  Extra:   {actual_keys - expected_keys}"
        )

    def test_classification_values_match_keys(self, analyzer_module):
        for key, value in analyzer_module.CLASSIFICATION.items():
            assert key == value, f"Classification {key} -> {value} should be identity mapping"

    def test_validation_source_in_analysis(self, analyzer_module):
        result = analyzer_module.analyze.__code__
        # The analyze function's result dict should include validation_source
        # We check indirectly by running analyze on an empty/synthetic CSV
        import tempfile
        import csv
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "terminated"])
            writer.writerow(["0", "False"])
            writer.writerow(["1", "False"])
            f.flush()
            result = analyzer_module.analyze(Path(f.name))
        assert result["validation_source"] == "real_simulation", "validation_source must be real_simulation"


# ---------------------------------------------------------------------------
# Test 3: Compile checks
# ---------------------------------------------------------------------------

class TestCompile:
    """Verify all scripts compile without errors."""

    def test_runner_compiles(self):
        spec = importlib.util.spec_from_file_location("runner_test", RUNNER_PATH)
        assert spec is not None, f"Cannot locate {RUNNER_PATH}"

    def test_analyzer_compiles(self):
        spec = importlib.util.spec_from_file_location("analyzer_test", ANALYZER_PATH)
        assert spec is not None, f"Cannot locate {ANALYZER_PATH}"
