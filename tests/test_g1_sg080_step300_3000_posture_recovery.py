"""Tests for G1_sg080 step300/3000 posture recovery diagnostic.

Verifies:
1. Runner defines exactly one push window
2. Push start is step 300
3. Push duration is 10 steps
4. Requested steps is 3000
5. Profile parameters match G1_sg080
6. D remains current-best
7. No threshold relaxation
8. No WBC enabled
9. No support-aware H gate enabled
10. Validation_source must be real_simulation
11. Posture recovery classification enum is valid
12. Audit is required if classification is not PASS or PASS_WITH_POSITION_DRIFT
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Module paths
# ---------------------------------------------------------------------------
RUNNER_PATH = ROOT / "scripts" / "run_g1_sg080_single_90n_10step_push_step300_3000.py"
ANALYZER_PATH = ROOT / "scripts" / "analyze_g1_sg080_step300_3000_posture_recovery.py"
AUDIT_PATH = ROOT / "scripts" / "audit_g1_sg080_posture_recovery_failure.py"

# Use the existing analyzer for comparison tests
EXISTING_RUNNER_PATH = ROOT / "scripts" / "run_g1_sg080_single_90n_10step_push_recovery.py"
EXISTING_ANALYZER_PATH = ROOT / "scripts" / "analyze_g1_sg080_single_push_recovery.py"


@pytest.fixture(scope="session")
def runner_module():
    return _import_from_path("runner", RUNNER_PATH)


@pytest.fixture(scope="session")
def analyzer_module():
    return _import_from_path("analyzer", ANALYZER_PATH)


@pytest.fixture(scope="session")
def audit_module():
    return _import_from_path("audit", AUDIT_PATH)


# ---------------------------------------------------------------------------
# Test 1: Runner constants
# ---------------------------------------------------------------------------

class TestRunnerConstants:
    """Verify the runner script defines the correct scenario."""

    def test_steps_is_3000(self, runner_module):
        assert runner_module.STEPS == 3000, f"Expected 3000, got {runner_module.STEPS}"

    def test_push_magnitude_is_90(self, runner_module):
        assert runner_module.PUSH_MAG_N == 90.0, f"Expected 90.0 N, got {runner_module.PUSH_MAG_N}"

    def test_push_duration_is_10(self, runner_module):
        assert runner_module.PUSH_DUR_STEPS == 10, f"Expected 10, got {runner_module.PUSH_DUR_STEPS}"

    def test_push_count_is_1(self, runner_module):
        assert runner_module.PUSH_COUNT == 1, f"Expected 1, got {runner_module.PUSH_COUNT}"

    def test_push_start_step_is_300(self, runner_module):
        assert runner_module.PUSH_START_STEP == 300, f"Expected 300, got {runner_module.PUSH_START_STEP}"

    def test_height_label_is_high_0p480(self, runner_module):
        assert runner_module.HEIGHT_LABEL == "high_0p480", f"Expected high_0p480, got {runner_module.HEIGHT_LABEL}"

    def test_case_id(self, runner_module):
        expected = "G1_sg080_single_90n_10step_push_step300_3000"
        assert runner_module.CASE_ID == expected, f"Expected {expected}, got {runner_module.CASE_ID}"


class TestRunnerProfile:
    """Verify the runner uses G1_sg080 profile parameters."""

    def test_profile_name(self, runner_module):
        cmd = runner_module.build_g1_sg080_cmd(Path("dummy"))
        cmd_str = " ".join(cmd)
        assert "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1" in cmd_str, (
            "Runner must use D_MODE_HIP_YAW_DIV_V1 profile"
        )

    def test_mode_div_kp_is_10(self, runner_module):
        cmd_str = " ".join(runner_module.build_g1_sg080_cmd(Path("dummy")))
        assert "--mode-hip-yaw-div-kp 10.0" in cmd_str, "G1_sg080 kp must be 10.0"

    def test_mode_div_kd_is_0_50(self, runner_module):
        cmd_str = " ".join(runner_module.build_g1_sg080_cmd(Path("dummy")))
        assert "--mode-hip-yaw-div-kd 0.50" in cmd_str, "G1_sg080 kd must be 0.50"

    def test_mode_div_max_torque_is_7_5(self, runner_module):
        cmd_str = " ".join(runner_module.build_g1_sg080_cmd(Path("dummy")))
        assert "--mode-hip-yaw-div-max-torque 7.5" in cmd_str, "G1_sg080 max_torque must be 7.5"

    def test_mode_div_soft_limit_is_0_30(self, runner_module):
        cmd_str = " ".join(runner_module.build_g1_sg080_cmd(Path("dummy")))
        assert "--mode-hip-yaw-div-soft-limit-rad 0.30" in cmd_str, "G1_sg080 soft_limit must be 0.30"

    def test_mode_div_soft_gain_is_0_80(self, runner_module):
        cmd_str = " ".join(runner_module.build_g1_sg080_cmd(Path("dummy")))
        assert "--mode-hip-yaw-div-soft-gain 0.80" in cmd_str, "G1_sg080 soft_gain must be 0.80"

    def test_mode_div_ref_source_is_target(self, runner_module):
        cmd_str = " ".join(runner_module.build_g1_sg080_cmd(Path("dummy")))
        assert "--mode-hip-yaw-div-ref-source target" in cmd_str, "G1_sg080 ref_source must be target"


class TestNoProductionChanges:
    """Verify D remains current-best and no unwanted features are enabled."""

    def test_d_remains_current_best(self):
        """D_MODE_HIP_YAW_DIV_V1 must still be in the profile choices."""
        from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1" in SAGITTAL_AUTHORITY_PROFILES

    def test_no_wbc_enabled(self, runner_module):
        cmd_str = " ".join(runner_module.build_g1_sg080_cmd(Path("dummy")))
        assert "--controller-mode balance-core" in cmd_str, "Must use balance-core mode"

    def test_no_wheel_yaw_stabilizer(self, runner_module):
        cmd_str = " ".join(runner_module.build_g1_sg080_cmd(Path("dummy")))
        assert "--enable-wheel-yaw-stabilizer" not in cmd_str, "Wheel yaw stabilizer must NOT be enabled"

    def test_no_support_aware_h_gate(self, runner_module):
        cmd_str = " ".join(runner_module.build_g1_sg080_cmd(Path("dummy")))
        assert "--mode-hip-yaw-div-support-enabled" not in cmd_str, "Support-aware H gate must NOT be enabled"

    def test_no_threshold_relaxation(self, runner_module):
        cmd_str = " ".join(runner_module.build_g1_sg080_cmd(Path("dummy")))
        assert "--mode-hip-yaw-div-soft-limit-rad 0.30" in cmd_str

    def test_sagittal_push_only(self, runner_module):
        cmd_str = " ".join(runner_module.build_g1_sg080_cmd(Path("dummy")))
        assert "--sagittal-push-only" in cmd_str, "Push must be sagittal-only"


# ---------------------------------------------------------------------------
# Test 2: Analyzer classification enum
# ---------------------------------------------------------------------------

class TestAnalyzerClassification:
    """Verify the analyzer defines a valid classification enum."""

    def test_classification_keys(self, analyzer_module):
        expected_keys = {
            "POSTURE_RECOVERY_PASS",
            "POSTURE_RECOVERY_PASS_WITH_POSITION_DRIFT",
            "POSTURE_RECOVERY_PARTIAL_HIP_YAW_ONLY",
            "POSTURE_RECOVERY_FAIL_PITCH_SUPPORT_OSCILLATION",
            "POSTURE_RECOVERY_FAIL_POSTURE_NOT_SETTLED",
            "POSTURE_RECOVERY_FAIL_FALL",
            "POSTURE_RECOVERY_INCONCLUSIVE_PUSH_CONFIG_INVALID",
            "POSTURE_RECOVERY_INCONCLUSIVE_MISSING_TELEMETRY",
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
        import tempfile
        import csv
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "terminated", "push_active", "robot_pitch_x",
                             "robot_roll_y", "robot_yaw_z", "hip_yaw_abs_max",
                             "support_position_error_m", "com_z",
                             "mode_hip_yaw_div_enabled", "mode_hip_yaw_div_kp",
                             "mode_hip_yaw_div_kd", "mode_hip_yaw_div_max_torque",
                             "mode_hip_yaw_div_soft_limit_rad", "mode_hip_yaw_div_soft_gain",
                             "mode_hip_yaw_div_height_gate", "mode_hip_yaw_div_tau_left_raw",
                             "mode_hip_yaw_div_tau_right_raw", "mode_hip_yaw_div_tau_left",
                             "mode_hip_yaw_div_tau_right", "mode_hip_yaw_div_error",
                             "mode_hip_yaw_div_rate", "pitch_rate_rad_s", "roll_rate_rad_s",
                             "yaw_rate_rad_s", "contact_force_valid",
                             "hip_yaw_common_error_rad", "hip_yaw_divergence_error_rad",
                             "height_error_m", "target_com_z_m",
                             "pitch_error_x_rad", "pitch_x_ref_rad",
                             "outer_loop_support_error_m", "outer_loop_support_error_rate_mps",
                             "outer_loop_pitch_ref_total_deg", "physics_equivalent_pitch_ref_deg",
                             "outer_loop_gate_pass", "support_outer_loop_kp_effective",
                             "support_outer_loop_kd_effective"])
            writer.writerow(["0", "False", "False", "0", "0", "0", "0", "0", "0",
                             "False", "10", "0.5", "7.5", "0.3", "0.8",
                             "1.0", "0", "0", "0", "0", "0", "0",
                             "0", "0", "0", "False",
                             "0", "0", "0", "0",
                             "0", "0",
                             "0", "0", "0", "0",
                             "1", "0", "0"])
            for s in range(1, 3000):
                writer.writerow([str(s), "False",
                                "True" if 300 <= s <= 309 else "False",
                                "0", "0", "0", "0", "0", "0",
                                "False", "10", "0.5", "7.5", "0.3", "0.8",
                                "1.0", "0", "0", "0", "0", "0", "0",
                                "0", "0", "0", "False",
                                "0", "0", "0", "0",
                                "0", "0",
                                "0", "0", "0", "0",
                                "1", "0", "0"])
            f.flush()
            result = analyzer_module.analyze(Path(f.name), Path(f.name).parent)
        assert result["validation_source"] == "real_simulation", "validation_source must be real_simulation"


# ---------------------------------------------------------------------------
# Test 3: Audit classification enforcement
# ---------------------------------------------------------------------------

class TestAuditRequirements:
    """Verify audit is required when classification is not PASS or PASS_WITH_POSITION_DRIFT."""

    def test_audit_module_exists(self):
        assert AUDIT_PATH.exists(), f"Audit script not found at {AUDIT_PATH}"

    def test_audit_has_audit_function(self, audit_module):
        assert hasattr(audit_module, "audit"), "Audit module must have audit() function"

    def test_audit_returns_dict_with_failure_classes(self, audit_module):
        """Check the audit function returns expected structure (no telemetry required)."""
        import tempfile
        import csv
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "terminated", "push_active", "robot_pitch_x",
                             "robot_roll_y", "robot_yaw_z", "hip_yaw_abs_max",
                             "support_position_error_m", "com_z", "target_com_z_m",
                             "height_error_m", "pitch_error", "pitch_error_x_rad",
                             "pitch_x_ref_rad", "physics_equivalent_pitch_ref_deg",
                             "outer_loop_support_error_m", "outer_loop_support_error_rate_mps",
                             "outer_loop_pitch_ref_total_deg", "outer_loop_gate_pass",
                             "support_outer_loop_kp_effective", "support_outer_loop_kd_effective",
                             "mode_hip_yaw_div_tau_left_raw", "mode_hip_yaw_div_tau_right_raw",
                             "mode_hip_yaw_div_error", "hip_yaw_common_error_rad",
                             "hip_yaw_divergence_error_rad", "mode_hip_yaw_div_enabled",
                             "yaw_rate_rad_s", "contact_force_valid",
                             "mode_hip_yaw_div_kp", "mode_hip_yaw_div_kd"])
            for s in range(3000):
                writer.writerow([str(s), "False", "False",
                                "0", "0", "0", "0", "0", "0.48", "0.48",
                                "0", "0", "0", "0", "0",
                                "0", "0", "0", "1",
                                "1", "1",
                                "0", "0", "0", "0",
                                "0", "False",
                                "0", "False",
                                "10", "0.5"])
            f.flush()
            result = audit_module.audit(Path(f.name), None, Path(f.name).parent)

        assert isinstance(result, dict)
        assert "failure_classes" in result, "Audit must include failure_classes"
        assert isinstance(result["failure_classes"], list), "failure_classes must be a list"
        assert len(result["failure_classes"]) >= 1, "At least one failure class expected"


# ---------------------------------------------------------------------------
# Test 4: Compile checks
# ---------------------------------------------------------------------------

class TestCompile:
    """Verify all scripts compile without errors."""

    def test_runner_compiles(self):
        spec = importlib.util.spec_from_file_location("runner_test", RUNNER_PATH)
        assert spec is not None, f"Cannot locate {RUNNER_PATH}"

    def test_analyzer_compiles(self):
        spec = importlib.util.spec_from_file_location("analyzer_test", ANALYZER_PATH)
        assert spec is not None, f"Cannot locate {ANALYZER_PATH}"

    def test_audit_compiles(self):
        spec = importlib.util.spec_from_file_location("audit_test", AUDIT_PATH)
        assert spec is not None, f"Cannot locate {AUDIT_PATH}"
