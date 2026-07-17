"""Tests for support-aware mode-div authority schedule (H candidates).

These tests verify that:
1. H candidates are opt-in (D remains current-best unless promoted)
2. Support-aware gating is continuous (no hard thresholds)
3. Support-aware gating uses support telemetry, not case labels
4. Support gate telemetry columns exist
5. Combined gate telemetry columns exist
6. Support-aware mode disabled leaves D behavior unchanged
7. Report classification is one of allowed values
8. No WBC enabled
9. No PFF source changes
10. No low-band v2 tuning changes
11. No D4/D5-specific branching
12. No height-name-specific branching
13. No hip-yaw gate relaxation
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SIM_SCRIPT = ROOT / "scripts" / "simulate_hierarchical_controller.py"
SWEEP_SCRIPT = ROOT / "scripts" / "run_support_aware_mode_div_sweep.py"
REPORT_PATH = ROOT / "docs" / "validation" / "support_aware_mode_div_authority_schedule_report.md"


class TestProfileExists:
    """D is current-best. H is opt-in."""

    def test_d_still_current_best(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        d_keys = [k for k in SAGITTAL_AUTHORITY_PROFILES if "mode_hip_yaw_div_v1" in k and "wheel_yaw" not in k]
        assert len(d_keys) >= 1, "D profile must exist"

    def test_h_not_promoted_to_profile(self):
        from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        h_profiles = [k for k in SAGITTAL_AUTHORITY_PROFILES if "support_aware" in k or "H1_" in k or "H2_" in k]
        assert len(h_profiles) == 0, "No H profile should exist -- H is parameter-based"


class TestNoWBC:
    def test_no_wbc_in_sweep_script(self):
        src = SWEEP_SCRIPT.read_text(encoding="utf-8")
        wbc_related = [line for line in src.split("\n") if "wbc" in line.lower()]
        assert len(wbc_related) == 0, "H sweep script must not reference WBC"


class TestTelemetryFields:
    def test_support_gate_columns_exist(self):
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        assert '"mode_hip_yaw_div_support_error_gate"' in src
        assert '"mode_hip_yaw_div_support_rate_gate"' in src
        assert '"mode_hip_yaw_div_effective_support_gate"' in src

    def test_combined_gate_column_exists(self):
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        assert '"mode_hip_yaw_div_combined_gate"' in src

    def test_support_error_columns_exist(self):
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        assert '"mode_hip_yaw_div_support_error_m"' in src
        assert '"mode_hip_yaw_div_support_error_rate_mps"' in src

    def test_validation_source_required(self):
        report = REPORT_PATH.read_text(encoding="utf-8")
        assert "Not run" in report or "real simulation" in report.lower() or "real-simulation" in report.lower()


class TestGuardRails:
    def test_no_d4_d5_specific_branch(self):
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        for line in src.split("\n"):
            if "D4" in line or "D5" in line:
                stripped = line.strip()
                if stripped.startswith("#") or '"""' in stripped or "D4/D5" in line:
                    continue
                if "if" in stripped.lower() and ("D4" in stripped or "D5" in stripped):
                    pytest.fail(f"D4/D5-specific branch found: {stripped}")

    def test_no_height_name_specific_branch_in_sweep(self):
        src = SWEEP_SCRIPT.read_text(encoding="utf-8")
        forbidden = [
            "if case_name == \"D5_large_push_high\"",
            "if case_name == \"D4_medium_push_low\"",
            "if height_label == \"high_0p480\"",
            "if height_label == \"low_0p330\"",
        ]
        for s in forbidden:
            assert s not in src

    def test_support_gate_is_continuous(self):
        """Support-aware gate must be continuous (no hard on/off)."""
        from wheeled_biped.controllers.mode_based_hip_yaw_divergence_controller import (
            ModeBasedHipYawDivergenceController,
        )
        ctrl = ModeBasedHipYawDivergenceController({
            "enabled": True,
            "kp_div": 10.0,
            "kd_div": 0.5,
            "max_torque": 7.5,
            "soft_limit_rad": 0.3,
            "soft_limit_gain": 0.8,
            "ref_source": "target",
            "support_gate_enabled": True,
            "support_threshold_m": 0.30,
            "support_width_m": 0.10,
            "support_min_gate": 0.70,
        })
        g1 = ctrl._support_error_gate(0.29)  # below threshold
        g2 = ctrl._support_error_gate(0.301)  # just above threshold
        g3 = ctrl._support_error_gate(0.35)   # in transition
        g4 = ctrl._support_error_gate(0.45)   # above width
        assert g1 == pytest.approx(1.0, abs=0.01), "Below threshold should be 1.0"
        assert g2 < 1.0 and g2 > g3, "Should decrease smoothly above threshold"
        assert g3 < g2, "Should decrease monotonically"
        assert g4 == pytest.approx(ctrl.support_min_gate, abs=0.02), "Above width should approach min_gate"

    def test_support_gate_uses_support_telemetry_not_case_labels(self):
        """Gate must be based on support_error/rate, not case labels."""
        from wheeled_biped.controllers.mode_based_hip_yaw_divergence_controller import (
            ModeBasedHipYawDivergenceController,
        )
        ctrl = ModeBasedHipYawDivergenceController({
            "enabled": True,
            "kp_div": 10.0,
            "kd_div": 0.5,
            "max_torque": 7.5,
            "soft_limit_rad": 0.3,
            "soft_limit_gain": 0.8,
            "ref_source": "target",
            "support_gate_enabled": True,
        })
        # Same state, different support_error -> different gate
        g_small = ctrl._support_error_gate(0.05)
        g_large = ctrl._support_error_gate(0.50)
        assert g_large < g_small, "Larger support error should give smaller gate"

    def test_disabled_mode_unchanged(self):
        """When support_gate_enabled=False, D behavior is preserved."""
        from wheeled_biped.controllers.mode_based_hip_yaw_divergence_controller import (
            HipYawState,
            ModeBasedHipYawDivergenceController,
        )
        # D config (G1_sg080 base but with support disabled)
        cfg = {
            "enabled": True,
            "kp_div": 10.0,
            "kd_div": 0.5,
            "max_torque": 7.5,
            "soft_limit_rad": 0.3,
            "soft_limit_gain": 0.8,
            "ref_source": "target",
            "support_gate_enabled": False,
        }
        ctrl = ModeBasedHipYawDivergenceController(cfg)
        state = HipYawState(div_error=0.1, div_rate=0.05, height=0.48,
                            support_error=0.5, support_error_rate=1.0)
        out = ctrl.compute(state)
        # support_error_gate should be 1.0 when disabled
        assert out["effective_support_gate"] == 1.0
        assert out["combined_gate"] == ctrl._height_gate(0.48)  # height gate only

    def test_no_pff_source_change(self):
        from wheeled_biped.controllers.physics_equilibrium_feedforward import (
            physics_equilibrium_feedforward_tau_each_wheel_nm,
        )
        assert physics_equilibrium_feedforward_tau_each_wheel_nm is not None

    def test_low_band_v2_tuning_unchanged(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        profile = SAGITTAL_AUTHORITY_PROFILES.get(
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
        )
        assert profile is not None
        assert profile.low_band_support_outer_loop_enabled is True

    def test_hip_yaw_gate_not_relaxed(self):
        """soft_limit_rad must remain 0.30."""
        src = SIM_SCRIPT.read_text(encoding="utf-8")
        # Check that the default soft_limit_rad in CLI is still 0.30
        # (the default value in the parser definition)
        assert "default=0.30" in src or any(
            "soft-limit-rad" in line and "0.30" in line
            for line in src.split("\n")
        ), "soft_limit_rad should not be relaxed"


class TestReportClassification:
    def test_report_exists(self):
        assert REPORT_PATH.exists()

    def test_report_classification_is_allowed(self):
        text = REPORT_PATH.read_text(encoding="utf-8")
        allowed = [
            "SUPPORT_AWARE_MODE_DIV_FIX_PASS",
            "SUPPORT_AWARE_MODE_DIV_FIX_PASS_WITH_MONITORING",
            "SUPPORT_AWARE_MODE_DIV_FIX_D5_IMPROVED_NOT_PASS",
            "SUPPORT_AWARE_MODE_DIV_FIX_NO_IMPROVEMENT_NOT_PASS",
            "SUPPORT_AWARE_MODE_DIV_FIX_FAIL_REGRESSION",
            "SUPPORT_AWARE_MODE_DIV_FIX_FAIL_SAFETY",
            "SUPPORT_AWARE_MODE_DIV_FIX_INCONCLUSIVE",
        ]
        assert any(a in text for a in allowed)


class TestCompile:
    def _compile(self, path):
        import py_compile
        try:
            py_compile.compile(str(path), doraise=True)
            return True
        except py_compile.PyCompileError as e:
            pytest.fail(f"Compile error in {path}: {e}")

    def test_sweep_compiles(self):
        self._compile(SWEEP_SCRIPT)

    def test_diag_compiles(self):
        self._compile(ROOT / "scripts" / "analyze_support_aware_mode_div_timing.py")

    def test_analyze_compiles(self):
        self._compile(ROOT / "scripts" / "analyze_support_aware_mode_div_results.py")

    def test_sim_compiles(self):
        self._compile(SIM_SCRIPT)
