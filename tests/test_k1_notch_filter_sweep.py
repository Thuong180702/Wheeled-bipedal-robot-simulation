"""Tests for K1 Notch Filter Parameter and Topology Sweep.

Verifies:
  - K1 baseline profile unchanged
  - Filter sweep profiles are opt-in only
  - No profile auto-promoted
  - Sweep parameter grid generated correctly
  - Scorer rejects WIP regression
  - Scorer rejects hidden torque/WBC
  - Scorer penalizes complexity
  - Topology variants compile
  - No threshold relaxation
  - Report path exists
"""

import json
import math
import os
import sys
from pathlib import Path

import pytest

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONTROLLERS_DIR = PROJECT_ROOT / "wheeled_biped" / "controllers"

# Make scripts importable
sys.path.insert(0, str(SCRIPTS_DIR))


class TestK1BaselineUnchanged:
    """Verify K1 baseline profile is unchanged after sweep infrastructure."""

    def test_k1_gains_unchanged(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            SagittalVelocityDampedBalanceController,
            K1_PITCH_RATE_NOTCH,
        )
        k1 = K1_PITCH_RATE_NOTCH
        ctrl = SagittalVelocityDampedBalanceController(authority_schedule=k1)
        # Controller constructor parameters (canonical K1 gains per CLAUDE.md)
        assert ctrl.kp_pitch == 50.0
        assert ctrl.kd_pitch == 10.0
        assert ctrl.max_position_tau == 3.0
        assert ctrl.max_tau_wheel == 5.0
        # Schedule dataclass fields (nominal values)
        assert k1.k_position_nominal == 40.0
        assert k1.k_velocity_nominal == 15.0
        assert k1.k_wheel_velocity_nominal == 0.5

    def test_k1_filter_params_unchanged(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH,
        )
        k1 = K1_PITCH_RATE_NOTCH
        assert k1.enable_wip_notch_filter is True
        assert k1.wip_notch_center_hz == 2.5
        assert k1.wip_notch_q == 6.0
        assert k1.wip_notch_filter_blend == 1.0
        assert k1.wip_notch_target_signal == "pitch_rate"
        assert k1.wip_notch_height_gate_start_m == 0.42
        assert k1.wip_notch_height_gate_full_m == 0.48
        assert k1.wip_notch_filter_type == "biquad_notch"

    def test_k1_no_wbc(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH,
        )
        assert getattr(K1_PITCH_RATE_NOTCH, "wbc_enabled", False) is False

    def test_k1_no_hidden_torque(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            SagittalVelocityDampedBalanceController,
            K1_PITCH_RATE_NOTCH,
        )
        ctrl = SagittalVelocityDampedBalanceController(authority_schedule=K1_PITCH_RATE_NOTCH)
        suspicious = ["hidden_torque", "secret_gain", "wbc_active", "extra_torque",
                      "hidden_damping", "secret_notch"]
        for attr in suspicious:
            assert not hasattr(ctrl, attr), f"Should not have '{attr}' attribute"


class TestSweepProfilesAreOptIn:
    """Verify all sweep profiles are opt-in only."""

    def test_k1_not_marked_as_sweep(self):
        """K1 must NOT be in the sweep profile dict."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            ALL_K_SWEEP_PROFILES,
        )
        assert "k1_pitch_rate_notch_v1" not in ALL_K_SWEEP_PROFILES, \
            "K1 must not be in ALL_K_SWEEP_PROFILES"

    def test_all_sweep_profiles_have_k_sweep_prefix(self):
        """All sweep profiles must have k_sweep_ prefix."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            ALL_K_SWEEP_PROFILES,
        )
        for name in ALL_K_SWEEP_PROFILES:
            assert name.startswith("k_sweep_"), \
                f"Sweep profile '{name}' must start with 'k_sweep_'"

    def test_no_sweep_profile_uses_k1_name(self):
        """No sweep profile may use 'k1_pitch_rate_notch' naming."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            ALL_K_SWEEP_PROFILES,
        )
        for name in ALL_K_SWEEP_PROFILES:
            assert "k1_pitch_rate_notch_v1" not in name.lower(), \
                f"'{name}' must not impersonate K1"

    def test_sweep_profiles_inherit_from_k1(self):
        """All sweep profiles must inherit K1's sagittal base (non-filter params)."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            ALL_K_SWEEP_PROFILES,
            K1_PITCH_RATE_NOTCH,
        )
        for name, profile in ALL_K_SWEEP_PROFILES.items():
            assert profile.low_band_support_outer_loop_enabled == K1_PITCH_RATE_NOTCH.low_band_support_outer_loop_enabled
            assert profile.kd_pitch_nominal == K1_PITCH_RATE_NOTCH.kd_pitch_nominal
            assert profile.k_position_nominal == K1_PITCH_RATE_NOTCH.k_position_nominal
            assert profile.k_velocity_nominal == K1_PITCH_RATE_NOTCH.k_velocity_nominal
            assert profile.k_wheel_velocity_nominal == K1_PITCH_RATE_NOTCH.k_wheel_velocity_nominal
            assert profile.max_position_tau_nominal == K1_PITCH_RATE_NOTCH.max_position_tau_nominal

    def test_sweep_profiles_count(self):
        """There should be exactly 19 sweep profiles (Groups A-D)."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            ALL_K_SWEEP_PROFILES,
        )
        assert len(ALL_K_SWEEP_PROFILES) == 19, \
            f"Expected 19 sweep profiles, got {len(ALL_K_SWEEP_PROFILES)}"


class TestSweepScriptCompiles:
    """Verify sweep and scoring scripts compile."""

    def test_sweep_script_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sweep_k1_notch_filter_parameters",
            SCRIPTS_DIR / "sweep_k1_notch_filter_parameters.py",
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "run_sweep")

    def test_score_script_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "score_k1_notch_filter_sweep",
            SCRIPTS_DIR / "score_k1_notch_filter_sweep.py",
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "score_all_candidates")


class TestSweepParameterGrid:
    """Verify sweep parameter grid is correct."""

    def test_sweep_script_has_all_groups(self):
        from sweep_k1_notch_filter_parameters import (
            GROUP_A_FC_SWEEP, GROUP_B_Q_SWEEP, GROUP_C_BLEND_SWEEP, GROUP_D_TOPOLOGY,
        )
        assert len(GROUP_A_FC_SWEEP) == 9  # 1.5 to 3.5 in 0.25 steps + K1
        assert len(GROUP_B_Q_SWEEP) == 6   # 2,3,4,6,8,10
        assert len(GROUP_C_BLEND_SWEEP) == 5  # 0, 0.25, 0.5, 0.75, 1.0
        assert len(GROUP_D_TOPOLOGY) == 5   # notch_disabled + 4 LP cutoffs

    def test_sweep_group_a_frequencies(self):
        from sweep_k1_notch_filter_parameters import GROUP_A_FC_SWEEP
        centers = sorted([p["center_hz"] for p in GROUP_A_FC_SWEEP.values()])
        expected = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]
        assert centers == expected, f"Expected {expected}, got {centers}"

    def test_sweep_group_b_q_values(self):
        from sweep_k1_notch_filter_parameters import GROUP_B_Q_SWEEP
        qs = sorted([p["Q"] for p in GROUP_B_Q_SWEEP.values()])
        expected = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
        assert qs == expected, f"Expected {expected}, got {qs}"

    def test_sweep_group_c_blend_values(self):
        from sweep_k1_notch_filter_parameters import GROUP_C_BLEND_SWEEP
        blends = sorted([p["blend"] for p in GROUP_C_BLEND_SWEEP.values()])
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert blends == expected, f"Expected {expected}, got {blends}"

    def test_sweep_group_d_has_all_topologies(self):
        from sweep_k1_notch_filter_parameters import GROUP_D_TOPOLOGY
        filter_types = set(p["filter_type"] for p in GROUP_D_TOPOLOGY.values())
        assert "notch_disabled" in filter_types
        assert "first_order_lowpass" in filter_types
        lp_cutoffs = sorted([p.get("lp_cutoff", 0) for p in GROUP_D_TOPOLOGY.values()
                            if p["filter_type"] == "first_order_lowpass"])
        assert lp_cutoffs == [3.0, 4.0, 5.0, 6.0]


class TestScorerLogic:
    """Verify scorer applies correct penalties and rejections."""

    def test_scorer_weights_summary(self):
        from score_k1_notch_filter_sweep import WEIGHTS
        assert "low_freq_pitch_power" in WEIGHTS
        assert "pitch_notch_coherence" in WEIGHTS
        assert "pitch_rms" in WEIGHTS
        assert "support_rms" in WEIGHTS
        assert "wip_band_power" in WEIGHTS
        assert "safety" in WEIGHTS
        assert "clipping" in WEIGHTS
        assert "complexity" in WEIGHTS

    def test_scorer_hard_reject_thresholds(self):
        from score_k1_notch_filter_sweep import HARD_REJECT
        assert HARD_REJECT["wip_power_ratio_max"] == 1.25
        assert HARD_REJECT["lf_power_ratio_max"] == 1.20
        assert HARD_REJECT["pitch_rms_ratio_max"] == 1.15

    def test_compute_score_rejects_fall(self):
        from score_k1_notch_filter_sweep import compute_candidate_score
        baseline = {"lf_power_0p15_0p55_hz": 0.01, "wip_power_2p0_3p0_hz": 0.001,
                     "pitch_rms_deg": 2.5, "support_rms_m": 0.02,
                     "lf_pitch_notch_coherence": 0.8, "clip_fraction": 0.0}
        candidate = {
            "lf_power_0p15_0p55_hz": 0.005, "wip_power_2p0_3p0_hz": 0.0005,
            "pitch_rms_deg": 1.5, "support_rms_m": 0.01,
            "lf_pitch_notch_coherence": 0.3, "clip_fraction": 0.0,
            "has_fall": True, "has_nan": False,
            "pitch_abs_max_deg": 15.0, "body_height_min_m": 0.45,
            "lf_peak_freq_hz": 0.2,
        }
        params = {"filter_type": "biquad_notch", "center_hz": 2.5, "Q": 4.0, "blend": 0.5}
        result = compute_candidate_score(candidate, baseline, params, "test_candidate")
        assert result["classification"] == "INVALID"
        assert "FALL" in result["hard_reject_reasons"]

    def test_compute_score_rejects_wip_regression(self):
        from score_k1_notch_filter_sweep import compute_candidate_score
        baseline = {"lf_power_0p15_0p55_hz": 0.01, "wip_power_2p0_3p0_hz": 0.001,
                     "pitch_rms_deg": 2.5, "support_rms_m": 0.02,
                     "lf_pitch_notch_coherence": 0.8, "clip_fraction": 0.0}
        candidate = {
            "lf_power_0p15_0p55_hz": 0.005, "wip_power_2p0_3p0_hz": 0.003,
            "pitch_rms_deg": 2.0, "support_rms_m": 0.015,
            "lf_pitch_notch_coherence": 0.3, "clip_fraction": 0.0,
            "has_fall": False, "has_nan": False,
            "pitch_abs_max_deg": 10.0, "body_height_min_m": 0.48,
            "lf_peak_freq_hz": 0.2,
        }
        params = {"filter_type": "biquad_notch"}
        result = compute_candidate_score(candidate, baseline, params, "test_candidate")
        # WIP ratio = 0.003/0.001 = 3.0 > 1.25 -> hard reject
        assert result["classification"] == "INVALID"
        assert any("WIP_POWER_RATIO" in r for r in result["hard_reject_reasons"])

    def test_compute_score_penalizes_complexity(self):
        from score_k1_notch_filter_sweep import compute_candidate_score
        baseline = {"lf_power_0p15_0p55_hz": 0.01, "wip_power_2p0_3p0_hz": 0.001,
                     "pitch_rms_deg": 2.5, "support_rms_m": 0.02,
                     "lf_pitch_notch_coherence": 0.8, "clip_fraction": 0.0}
        candidate = {
            "lf_power_0p15_0p55_hz": 0.01, "wip_power_2p0_3p0_hz": 0.001,
            "pitch_rms_deg": 2.5, "support_rms_m": 0.02,
            "lf_pitch_notch_coherence": 0.8, "clip_fraction": 0.0,
            "has_fall": False, "has_nan": False,
            "pitch_abs_max_deg": 10.0, "body_height_min_m": 0.48,
            "lf_peak_freq_hz": 0.39,
        }
        # First-order lowpass
        params_lp = {"filter_type": "first_order_lowpass", "lp_cutoff": 3.0}
        result_lp = compute_candidate_score(candidate, baseline, params_lp, "test_lp")
        # Biquad notch
        params_notch = {"filter_type": "biquad_notch"}
        result_notch = compute_candidate_score(candidate, baseline, params_notch, "test_notch")
        # LP should have slightly higher score due to complexity penalty
        assert result_lp["score_components"]["complexity_penalty"] > \
               result_notch["score_components"]["complexity_penalty"]

    def test_scorer_classifies_strong_improvement(self):
        from score_k1_notch_filter_sweep import compute_candidate_score
        baseline = {"lf_power_0p15_0p55_hz": 0.01, "wip_power_2p0_3p0_hz": 0.001,
                     "pitch_rms_deg": 2.5, "support_rms_m": 0.02,
                     "lf_pitch_notch_coherence": 0.8, "clip_fraction": 0.0}
        # Candidate with much better metrics
        candidate = {
            "lf_power_0p15_0p55_hz": 0.003,  # 0.3x baseline
            "wip_power_2p0_3p0_hz": 0.0008,  # 0.8x baseline
            "pitch_rms_deg": 1.5, "support_rms_m": 0.01,
            "lf_pitch_notch_coherence": 0.2,  # much lower coherence
            "clip_fraction": 0.0,
            "has_fall": False, "has_nan": False,
            "pitch_abs_max_deg": 10.0, "body_height_min_m": 0.48,
            "lf_peak_freq_hz": 0.15,
        }
        params = {"filter_type": "biquad_notch"}
        result = compute_candidate_score(candidate, baseline, params, "test_better")
        assert result["classification"] == "STRONG_IMPROVEMENT"


class TestTopologyVariants:
    """Verify topology variants are available and compile correctly."""

    def test_first_order_lowpass_exists(self):
        from wheeled_biped.controllers.signal_filters import FirstOrderLowPassFilter
        lp = FirstOrderLowPassFilter(fs_hz=100.0, cutoff_hz=3.0)
        assert lp.fs_hz == 100.0
        assert lp.cutoff_hz == 3.0
        assert 0 < lp.alpha < 1.0

    def test_first_order_lowpass_output_finite(self):
        from wheeled_biped.controllers.signal_filters import FirstOrderLowPassFilter
        lp = FirstOrderLowPassFilter(fs_hz=100.0, cutoff_hz=3.0)
        for x in [0.1, 0.2, 0.15, -0.1, -0.2, 0.05]:
            y = lp.update(x)
            assert math.isfinite(y), f"Output should be finite, got {y}"

    def test_first_order_lowpass_reset(self):
        from wheeled_biped.controllers.signal_filters import FirstOrderLowPassFilter
        lp = FirstOrderLowPassFilter(fs_hz=100.0, cutoff_hz=3.0)
        for x in [0.1, 0.2, 0.15]:
            lp.update(x)
        lp.reset()
        state = lp.get_state()
        assert abs(state[0]) < 1e-15, f"State should be zero after reset, got {state[0]}"

    def test_first_order_lowpass_state_format(self):
        from wheeled_biped.controllers.signal_filters import FirstOrderLowPassFilter
        lp = FirstOrderLowPassFilter(fs_hz=100.0, cutoff_hz=3.0)
        lp.update(0.5)
        state = lp.get_state()
        assert len(state) == 4, "get_state must return 4-tuple for telemetry compatibility"

    def test_biquad_notch_unchanged(self):
        from wheeled_biped.controllers.signal_filters import BiquadNotchFilter
        nf = BiquadNotchFilter(fs_hz=100.0, fc_hz=2.5, Q=6.0)
        assert nf.fc_hz == 2.5
        assert nf.Q == 6.0
        assert nf.fs_hz == 100.0
        # Should produce finite output
        for x in [0.1, 0.2, 0.15]:
            y = nf.update(x)
            assert math.isfinite(y)

    def test_notch_disabled_profile_exists(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K_SWEEP_NOTCH_DISABLED,
        )
        assert K_SWEEP_NOTCH_DISABLED.wip_notch_filter_type == "notch_disabled"
        assert K_SWEEP_NOTCH_DISABLED.enable_wip_notch_filter is True  # K1 base, overridden at runtime

    def test_lowpass_profile_exists(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K_SWEEP_LP_3P0, K_SWEEP_LP_6P0,
        )
        assert K_SWEEP_LP_3P0.wip_notch_filter_type == "first_order_lowpass"
        assert K_SWEEP_LP_3P0.wip_lowpass_cutoff_hz == 3.0
        assert K_SWEEP_LP_6P0.wip_lowpass_cutoff_hz == 6.0


class TestNoThresholdRelaxation:
    """Verify no thresholds are relaxed in sweep profiles."""

    def test_max_position_tau_unchanged(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            ALL_K_SWEEP_PROFILES, K1_PITCH_RATE_NOTCH,
        )
        k1_max = K1_PITCH_RATE_NOTCH.max_position_tau_nominal
        for name, profile in ALL_K_SWEEP_PROFILES.items():
            assert profile.max_position_tau_nominal == k1_max, \
                f"{name}: max_position_tau_nominal changed from {k1_max} to {profile.max_position_tau_nominal}"

    def test_no_torque_clip_relaxation(self):
        """Verify torque clip limits unchanged in sweep profiles."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            SagittalVelocityDampedBalanceController,
            ALL_K_SWEEP_PROFILES, K1_PITCH_RATE_NOTCH,
        )
        k1_ctrl = SagittalVelocityDampedBalanceController(authority_schedule=K1_PITCH_RATE_NOTCH)
        k1_max_tau = k1_ctrl.max_tau_wheel
        k1_max_pos = k1_ctrl.max_position_tau
        for name, profile in ALL_K_SWEEP_PROFILES.items():
            ctrl = SagittalVelocityDampedBalanceController(authority_schedule=profile)
            assert ctrl.max_tau_wheel == k1_max_tau, \
                f"{name}: max_tau_wheel changed from {k1_max_tau} to {ctrl.max_tau_wheel}"
            assert ctrl.max_position_tau == k1_max_pos, \
                f"{name}: max_position_tau changed from {k1_max_pos} to {ctrl.max_position_tau}"

    def test_height_gate_unchanged(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            ALL_K_SWEEP_PROFILES,
        )
        for name, profile in ALL_K_SWEEP_PROFILES.items():
            assert profile.wip_notch_height_gate_start_m == 0.42, \
                f"{name}: height_gate_start changed"
            assert profile.wip_notch_height_gate_full_m == 0.48, \
                f"{name}: height_gate_full changed"


class TestReportPaths:
    """Verify output directory and expected report paths."""

    def test_output_dir_exists(self):
        output_dir = PROJECT_ROOT / "outputs" / "k1_notch_filter_sweep"
        assert output_dir.exists(), f"Output directory {output_dir} must exist"

    def test_sweep_script_imports_succeed(self):
        """Verify all imports resolve."""
        import sweep_k1_notch_filter_parameters
        import score_k1_notch_filter_sweep
        assert sweep_k1_notch_filter_parameters.OUTPUT_DIR.exists()
        assert score_k1_notch_filter_sweep.OUTPUT_DIR.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
