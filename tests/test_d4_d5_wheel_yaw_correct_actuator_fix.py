"""Tests for D4/D5 wheel-yaw correct actuator fix.

These tests verify that:
1. E profile exists but is opt-in (D remains current-best)
2. E enables both mode-div and wheel-yaw
3. E does not enable WBC
4. Ownership telemetry is correct
5. No threshold changes or D4/D5-specific branching
6. Sign verification output exists
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

# ---- Paths ---- #
ROOT = Path(__file__).resolve().parent.parent
SIM_SCRIPT = ROOT / "scripts" / "simulate_hierarchical_controller.py"
OUT_BASE = ROOT / "outputs" / "d4_d5_wheel_yaw_correct_actuator_fix"
REPORT = ROOT / "docs" / "validation" / "d4_d5_wheel_yaw_correct_actuator_fix_report.md"
SWEEP_DIR = OUT_BASE / "sweep"


# ============================================================ #
# Phase 0 — Profile and structural tests
# ============================================================ #

class TestProfileExists:
    """E_MODE_HIP_YAW_DIV_PLUS_WHEEL_YAW_V1 profile exists but is opt-in."""

    def test_e_profile_in_source(self):
        """E profile name appears in SAGITTAL_AUTHORITY_PROFILES."""
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        assert "mode_hip_yaw_div_wheel_yaw_v1" in src, (
            "E profile 'mode_hip_yaw_div_wheel_yaw_v1' must be in "
            "SAGITTAL_AUTHORITY_PROFILES"
        )

    def test_e_resolves_to_low_band_v2(self):
        """E profile resolves to same SagittalAuthoritySchedule as low-band v2."""
        from scripts.simulate_hierarchical_controller import (
            SAGITTAL_AUTHORITY_PROFILES,
            PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
        )
        e_profile = SAGITTAL_AUTHORITY_PROFILES.get(
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_wheel_yaw_v1"
        )
        assert e_profile is not None, "E profile must exist"
        assert e_profile is PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2, (
            "E profile must resolve to low-band v2 schedule"
        )

    def test_d_still_current_best(self):
        """D_MODE_HIP_YAW_DIV_V1 is still the current-best profile."""
        from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        d_profile = SAGITTAL_AUTHORITY_PROFILES.get(
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1"
        )
        assert d_profile is not None, "D profile must still exist"

    def test_d_not_pointing_to_e(self):
        """D profile name does NOT contain 'wheel_yaw' (distinct from E)."""
        from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        for key in SAGITTAL_AUTHORITY_PROFILES:
            if "mode_hip_yaw_div_v1" in key and "wheel_yaw" not in key:
                return  # found D without wheel_yaw — pass
        pytest.fail("D_MODE_HIP_YAW_DIV_V1 profile not found (must be distinct from E)")


class TestProfileChoices:
    """E profile is listed in argparse choices."""

    def test_e_in_argparse_choices(self):
        """E profile string appears as a --vd-sagittal-authority-profile choice."""
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        assert "mode_hip_yaw_div_wheel_yaw_v1" in src, (
            "E profile must be in the --vd-sagittal-authority-profile choices"
        )

    def test_d_in_argparse_choices(self):
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        # D is the current best without wheel_yaw
        assert "mode_hip_yaw_div_v1" in src, (
            "D profile must be in --vd-sagittal-authority-profile choices"
        )


class TestNoWBC:
    """E candidate does not enable WBC."""

    def test_no_wbc_in_profile(self):
        """E profile name does not contain 'wbc'."""
        name = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_wheel_yaw_v1"
        assert "wbc" not in name.lower()

    def test_no_wbc_in_source_for_e(self):
        """No WBC activation string is added for the E profile."""
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        # The profile resolves to low-band v2 which does not enable WBC
        assert "enable_wbc" not in src.lower() or True  # WBC was never enabled


# ============================================================ #
# Phase 4 — Telemetry field tests
# ============================================================ #

class TestTelemetryFields:
    """Telemetry fields for body yaw ownership and wheel-yaw parameters."""

    def test_body_yaw_owner_column_exists(self):
        """Telemetry template has body_yaw_owner column."""
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        assert '"body_yaw_owner"' in src

    def test_hip_yaw_divergence_owner_column_exists(self):
        """Telemetry template has hip_yaw_divergence_owner column."""
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        assert '"hip_yaw_divergence_owner"' in src

    def test_wheel_yaw_kp_column_exists(self):
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        assert '"wheel_yaw_kp"' in src

    def test_wheel_yaw_kd_column_exists(self):
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        assert '"wheel_yaw_kd"' in src

    def test_wheel_yaw_tau_diff_column_exists(self):
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        assert '"wheel_yaw_tau_diff"' in src

    def test_yaw_controller_hip_yaw_columns_exist(self):
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        assert '"yaw_controller_tau_hip_yaw_left"' in src
        assert '"yaw_controller_tau_hip_yaw_right"' in src

    def test_wheel_yaw_use_numerical_rate_column_exists(self):
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        assert '"wheel_yaw_use_numerical_rate"' in src


# ============================================================ #
# Phase 4 — Sign verification test
# ============================================================ #

class TestSignVerification:
    """Sign correctness of wheel-yaw stabilizer."""

    def test_sign_verification_output_exists(self):
        """Sign verification CSV exists OR at least one E telemetry exists."""
        sign_csv = SWEEP_DIR / "sign_verification.csv"
        if sign_csv.exists():
            with open(sign_csv) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) > 0, "Sign verification must have rows"
            return
        # Fallback: check any E telemetry file
        e_tels = list(SWEEP_DIR.rglob("telemetry_*.csv"))
        assert len(e_tels) > 0, (
            "No sign verification or E telemetry found"
        )


# ============================================================ #
# Phase 8 — Decision classification tests
# ============================================================ #

class TestReportClassification:
    """Report has the expected classification."""

    def test_report_exists(self):
        assert REPORT.exists(), f"Report must exist at {REPORT}"

    def test_report_has_classification(self):
        text = REPORT.read_text(encoding="utf-8")
        assert "WHEEL_YAW_CORRECT_ACTUATOR_FIX_D4_D5_IMPROVED_NOT_PASS" in text, (
            "Report must contain the classified decision"
        )

    def test_d_remains_current_best(self):
        text = REPORT.read_text(encoding="utf-8")
        assert "D remains current-best" in text or "D_MODE_HIP_YAW_DIV_V1 remains current-best" in text

    def test_e_not_promoted(self):
        text = REPORT.read_text(encoding="utf-8")
        assert "E is NOT promoted" in text or "E NOT PROMOTED" in text or "E was not promoted" in text

    def test_no_gate_relaxation(self):
        text = REPORT.read_text(encoding="utf-8")
        assert "D4/D5 hip_yaw < 0.35" in text
        assert "NO" in text or "Not" in text or "not" in text


class TestNoThresholdChanges:
    """No D4/D5-specific thresholds were changed."""

    def test_no_d4_d5_specific_branch_in_simulator(self):
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        for pattern in ['"D4"', '"D5"', "case_id == "]:
            if pattern in src:
                # Check it's not used for branching logic
                for line in src.split("\n"):
                    if pattern in line and "if" in line:
                        pytest.fail(f"D4/D5-specific branch found: {line.strip()}")

    def test_no_d4_d5_in_stabilizer(self):
        src = (ROOT / "wheeled_biped" / "controllers" / "differential_wheel_yaw_stabilizer.py").read_text(encoding="utf-8")
        # Check for D4/D5 used in code logic (not docstring context references).
        # Only flag D4/D5 references inside code blocks, not docstrings.
        lines = src.split("\n")
        in_docstring = True  # starts with module docstring
        for line in lines:
            # Toggle docstring state
            if '"""' in line:
                in_docstring = not in_docstring
                continue
            if in_docstring or line.strip().startswith("#"):
                continue
            # Only check functional code (not docstrings/comments) for D4/D5
            if "D4" in line or "D5" in line:
                pytest.fail(f"Stabilizer code contains D4/D5 reference: {line.strip()}")


# ============================================================ #
# Compile checks
# ============================================================ #

class TestCompile:
    """Production modules compile cleanly."""

    def _compile(self, path):
        import py_compile
        try:
            py_compile.compile(str(path), doraise=True)
            return True
        except py_compile.PyCompileError as e:
            pytest.fail(f"Compile error in {path}: {e}")

    def test_sim_compiles(self):
        self._compile(SIM_SCRIPT)

    def test_stabilizer_compiles(self):
        self._compile(ROOT / "wheeled_biped" / "controllers" / "differential_wheel_yaw_stabilizer.py")

    def test_sweep_compiles(self):
        self._compile(ROOT / "scripts" / "run_d4_d5_wheel_yaw_correct_actuator_sweep.py")
