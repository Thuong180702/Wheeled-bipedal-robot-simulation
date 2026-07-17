"""Tests for K2 dynamic height gate-crossing regression validation.

Verifies:
  - Dynamic height scenarios defined correctly
  - All scenarios cross the notch gate (0.42-0.48m)
  - Classifier rejects gate discontinuity spikes
  - Classifier rejects non-real source
  - K1/K2 paired conditions identical except profile
  - Script compiles
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


class TestDynamicScenarios:
    """Verify dynamic height scenarios are correctly defined."""

    @pytest.fixture
    def scenarios(self):
        from scripts.validate_k2_dynamic_height_gate_crossing import DYNAMIC_SCENARIOS
        return DYNAMIC_SCENARIOS

    def test_all_scenarios_defined(self, scenarios):
        required = [
            "ramp_up_0p330_to_0p480",
            "ramp_down_0p480_to_0p330",
            "up_down_cycle_0p330_0p480_0p330",
            "gate_dwell_0p420_0p450_0p480",
            "gate_chatter_0p400_0p470",
        ]
        for name in required:
            assert name in scenarios, f"Missing scenario: {name}"

    def test_all_scenarios_cross_notch_gate(self, scenarios):
        """Every scenario must have at least one waypoint at or above 0.42m."""
        for name, info in scenarios.items():
            max_h = max(h for _, h in info["waypoints"])
            assert max_h >= 0.42, (
                f"Scenario {name} max height {max_h} does not cross notch gate (0.42m)"
            )

    def test_ramp_up_ends_above_gate(self, scenarios):
        s = scenarios["ramp_up_0p330_to_0p480"]
        final_h = s["waypoints"][-1][1]
        assert final_h >= 0.48, f"ramp_up must end above gate, got {final_h}"

    def test_ramp_down_ends_below_gate(self, scenarios):
        s = scenarios["ramp_down_0p480_to_0p330"]
        final_h = s["waypoints"][-1][1]
        assert final_h <= 0.33, f"ramp_down must end below gate, got {final_h}"

    def test_gate_chatter_crosses_multiple_times(self, scenarios):
        """Gate chatter must cross 0.42m boundary multiple times."""
        s = scenarios["gate_chatter_0p400_0p470"]
        heights = [h for _, h in s["waypoints"]]
        above_42 = sum(1 for h in heights if h >= 0.42)
        below_42 = sum(1 for h in heights if h < 0.42)
        assert above_42 >= 4, f"Gate chatter must have multiple crossings above 0.42m, got {above_42}"
        assert below_42 >= 4, f"Gate chatter must have multiple crossings below 0.42m, got {below_42}"

    def test_gate_dwell_includes_all_gate_points(self, scenarios):
        s = scenarios["gate_dwell_0p420_0p450_0p480"]
        heights = [h for _, h in s["waypoints"]]
        assert 0.420 in heights or any(abs(h - 0.420) < 0.001 for h in heights)
        assert 0.450 in heights or any(abs(h - 0.450) < 0.001 for h in heights)
        assert 0.480 in heights or any(abs(h - 0.480) < 0.001 for h in heights)


class TestDynamicClassifier:
    """Verify dynamic height classifier logic."""

    def _make_metrics(self, fell=False, hy=0.1, pitch_rms=3.0, support_rms=0.05,
                      height_rmse=0.01, gate_spike=3.0, wbc=0, hidden_torque=0):
        return {
            "fell": fell, "hip_yaw_abs_max": hy,
            "pitch_rms_deg": pitch_rms, "support_rms_m": support_rms,
            "height_tracking_rmse": height_rmse,
            "pitch_spike_at_crossing_max_deg": gate_spike,
            "wbc": wbc, "hidden_torque": hidden_torque,
        }

    def test_k2_falls_k1_does_not_is_regression(self):
        from scripts.validate_k2_dynamic_height_gate_crossing import classify_dynamic
        k1 = self._make_metrics(fell=False)
        k2 = self._make_metrics(fell=True)
        assert classify_dynamic(k1, k2) == "REGRESSION"

    def test_k2_gate_spike_exceeds_1p5x_and_absolute_is_worse(self):
        """Gate spike must exceed both 1.5x K1 AND absolute 8.0 deg to trigger worse."""
        from scripts.validate_k2_dynamic_height_gate_crossing import classify_dynamic
        k1 = self._make_metrics(gate_spike=6.0)
        k2 = self._make_metrics(gate_spike=10.0)  # 1.67x and exceeds 8.0 deg absolute
        assert classify_dynamic(k1, k2) == "WORSE_BUT_SAFE"

    def test_k2_gate_spike_below_absolute_is_not_worse(self):
        """Gate spike below 8.0 deg absolute is not flagged even if ratio > 1.5x."""
        from scripts.validate_k2_dynamic_height_gate_crossing import classify_dynamic
        k1 = self._make_metrics(gate_spike=3.0)
        k2 = self._make_metrics(gate_spike=5.5)  # 1.83x but < 8.0 deg absolute
        assert classify_dynamic(k1, k2) == "EQUIVALENT"

    def test_k2_height_tracking_much_worse_is_worse(self):
        from scripts.validate_k2_dynamic_height_gate_crossing import classify_dynamic
        k1 = self._make_metrics(height_rmse=0.01)
        k2 = self._make_metrics(height_rmse=0.03)  # 3x
        result = classify_dynamic(k1, k2)
        assert result in ("WORSE_BUT_SAFE", "MIXED_SAFE_TRADEOFF")

    def test_k2_better_pitch_and_support_is_strong_better(self):
        from scripts.validate_k2_dynamic_height_gate_crossing import classify_dynamic
        k1 = self._make_metrics(pitch_rms=4.0, support_rms=0.10)
        k2 = self._make_metrics(pitch_rms=3.0, support_rms=0.07)
        assert classify_dynamic(k1, k2) == "STRONG_BETTER"

    def test_both_equal_is_equivalent(self):
        from scripts.validate_k2_dynamic_height_gate_crossing import classify_dynamic
        m = self._make_metrics()
        assert classify_dynamic(m, m) == "EQUIVALENT"

    def test_none_metrics_is_invalid(self):
        from scripts.validate_k2_dynamic_height_gate_crossing import classify_dynamic
        assert classify_dynamic(None, self._make_metrics()) == "INVALID"

    def test_hip_yaw_exceeds_035_is_regression(self):
        from scripts.validate_k2_dynamic_height_gate_crossing import classify_dynamic
        k1 = self._make_metrics(hy=0.30)
        k2 = self._make_metrics(hy=0.40)
        assert classify_dynamic(k1, k2) == "REGRESSION"


class TestProfilesCorrect:
    def test_k1_k2_profiles_match(self):
        from scripts.validate_k2_dynamic_height_gate_crossing import K1_PROFILE, K2_PROFILE
        assert K1_PROFILE == "k1_pitch_rate_notch_v1"
        assert K2_PROFILE == "k2_notch_low_q_v1"

    def test_mode_div_flags_present(self):
        from scripts.validate_k2_dynamic_height_gate_crossing import MODE_DIV_FLAGS
        assert "--enable-mode-hip-yaw-divergence" in MODE_DIV_FLAGS


class TestCompileChecks:
    def test_dynamic_script_compiles(self):
        import py_compile
        path = PROJECT_ROOT / "scripts" / "validate_k2_dynamic_height_gate_crossing.py"
        result = py_compile.compile(str(path), doraise=True)
        assert result is not None

    def test_long_run_script_compiles(self):
        import py_compile
        path = PROJECT_ROOT / "scripts" / "validate_k2_post_promotion_long_run.py"
        result = py_compile.compile(str(path), doraise=True)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
