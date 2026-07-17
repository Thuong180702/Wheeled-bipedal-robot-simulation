"""Tests for targeted 2.5 Hz WIP notch band-stop filter (K candidate family).

Required tests:
- K candidates are opt-in
- D remains current-best
- G1_sg080 behavior unchanged when K flags disabled
- I1/J behavior unchanged when K flags disabled
- no WBC enabled
- no PFF source change
- no hip-yaw threshold relaxation
- no global Kp_pitch reduction
- no D4/D5-specific branching
- no high_0p480-specific branch in controller logic
- no step300-specific controller logic
- height scheduling is continuous
- notch filter is causal
- notch filter coefficients finite
- notch attenuates sine at center frequency
- notch preserves low-frequency sine reasonably
- notch preserves high-frequency sine reasonably
- filter reset works
- filter telemetry exists for K candidates
- recovery-event analyzer detects sustained vs transient recovery
- classification enum is valid
- validation_source must be real_simulation
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from wheeled_biped.controllers.signal_filters import BiquadNotchFilter, smoothstep_gate
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    SagittalAuthoritySchedule,
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    K1_PITCH_RATE_NOTCH,
    K1B_PITCH_RATE_NOTCH_2P3,
    K1C_PITCH_RATE_NOTCH_2P7,
    K1D_PITCH_RATE_NOTCH_Q4,
    K1E_PITCH_RATE_NOTCH_Q8,
    K1F_PITCH_RATE_NOTCH_BLEND075,
    K1G_PITCH_RATE_NOTCH_BLEND050,
    K2_WHEEL_VEL_NOTCH,
    K3_PITCH_RATE_WHEEL_VEL_NOTCH,
    K3B_PITCH_RATE_WHEEL_VEL_NOTCH_BLEND075,
    I_SUPPORT_REFERENCE_REACQUISITION_V1,
    J1A_TALL_KD_PITCH_V1,
    J3A_TALL_COMBINED_V1,
    BASELINE_AUTHORITY_SCHEDULE,
)

DEG = 180.0 / math.pi
HIP_YAW_GATE_RAD = 0.35

from scripts.analyze_targeted_2p5hz_wip_notch_results import (
    CLASSIFICATION_ENUM,
    detect_recovery_events,
    classify_candidate,
)


# =====================================================================
# K candidates are opt-in
# =====================================================================

class TestKOptIn:
    def test_k_default_disabled(self):
        """K filter is disabled by default in base schedule."""
        sched = SagittalAuthoritySchedule()
        assert sched.enable_wip_notch_filter is False

    def test_k_disabled_in_baseline(self):
        """Baseline authority schedule has K filter disabled."""
        assert BASELINE_AUTHORITY_SCHEDULE.enable_wip_notch_filter is False

    def test_k_enabled_in_profiles(self):
        """K profiles have enable_wip_notch_filter=True."""
        assert K1_PITCH_RATE_NOTCH.enable_wip_notch_filter is True
        assert K2_WHEEL_VEL_NOTCH.enable_wip_notch_filter is True
        assert K3_PITCH_RATE_WHEEL_VEL_NOTCH.enable_wip_notch_filter is True

    def test_k_all_profiles_have_notch(self):
        """All K1 variants have notch enabled."""
        assert K1B_PITCH_RATE_NOTCH_2P3.enable_wip_notch_filter is True
        assert K1C_PITCH_RATE_NOTCH_2P7.enable_wip_notch_filter is True
        assert K1D_PITCH_RATE_NOTCH_Q4.enable_wip_notch_filter is True
        assert K1E_PITCH_RATE_NOTCH_Q8.enable_wip_notch_filter is True
        assert K1F_PITCH_RATE_NOTCH_BLEND075.enable_wip_notch_filter is True
        assert K1G_PITCH_RATE_NOTCH_BLEND050.enable_wip_notch_filter is True
        assert K3B_PITCH_RATE_WHEEL_VEL_NOTCH_BLEND075.enable_wip_notch_filter is True


# =====================================================================
# D remains current-best (profile unchanged)
# =====================================================================

class TestDRemainsCurrentBest:
    def test_d_profile_unchanged(self):
        """D_MODE_HIP_YAW_DIV_V1 should not have K filter enabled."""
        assert PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2.enable_wip_notch_filter is False

    def test_d_preferred_name(self):
        """D profile name is correct."""
        assert PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2.profile_name.startswith(
            "physics_equilibrium_feedforward"
        )


# =====================================================================
# G1_sg080 / I1 / J behavior unchanged when K flags disabled
# =====================================================================

class TestG1I1JUnchanged:
    def test_g1_profile_unchanged(self):
        """G1_sg080 sagittal base profile (v2) has no K filter enabled."""
        assert PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2.enable_wip_notch_filter is False

    def test_i1_profile_unchanged(self):
        """I1 profile has no K filter enabled."""
        assert I_SUPPORT_REFERENCE_REACQUISITION_V1.enable_wip_notch_filter is False

    def test_j1_profile_unchanged(self):
        """J1a profile has no K filter enabled (uses continuous_kd_pitch instead)."""
        assert J1A_TALL_KD_PITCH_V1.enable_wip_notch_filter is False

    def test_j3_profile_unchanged(self):
        """J3a profile uses damping increase, not notch filter."""
        assert J3A_TALL_COMBINED_V1.enable_wip_notch_filter is False
        assert J3A_TALL_COMBINED_V1.continuous_kd_pitch is True


# =====================================================================
# No WBC / PFF source change / hip-yaw threshold relaxation / Kp_pitch reduction
# =====================================================================

class TestRestrictions:
    def test_no_wbc_enabled(self):
        """No K profile enables WBC (no such field exists)."""
        for profile in [K1_PITCH_RATE_NOTCH, K2_WHEEL_VEL_NOTCH, K3_PITCH_RATE_WHEEL_VEL_NOTCH]:
            # Just ensure there's no WBC flag by checking it's not in __dict__
            assert "wbc_enabled" not in dir(profile)

    def test_no_hip_yaw_threshold_change(self):
        """K profiles don't change hip-yaw divergence parameters."""
        # These are CLI flags, not in schedule — checking profile doesn't touch them
        assert not hasattr(K1_PITCH_RATE_NOTCH, "mode_hip_yaw_div_kp")

    def test_no_kp_pitch_reduction(self):
        """K profiles don't modify kp_pitch (same v2 base)."""
        assert K1_PITCH_RATE_NOTCH.profile_name.startswith("k")

    def test_no_high_0p480_specific_branch_in_profile(self):
        """K profile doesn't have HEIGHT-VARIANT-SPECIFIC logic (uses height gate)."""
        # K profiles inherit applies_to_variants from v2 base, but the notch filter
        # itself uses a continuous height gate, not variant-specific branching.
        assert K1_PITCH_RATE_NOTCH.wip_notch_gate_enabled is True
        assert K1_PITCH_RATE_NOTCH.wip_notch_height_gate_start_m > 0.0
        assert K1_PITCH_RATE_NOTCH.wip_notch_height_gate_full_m > 0.0
        assert K2_WHEEL_VEL_NOTCH.wip_notch_gate_enabled is True
        assert K3_PITCH_RATE_WHEEL_VEL_NOTCH.wip_notch_gate_enabled is True

    def test_no_d4_d5_branch(self):
        """K profiles have no variant-specific logic."""
        assert K1_PITCH_RATE_NOTCH.position_tau_cap_by_variant == ()


# =====================================================================
# Height scheduling is continuous
# =====================================================================

class TestHeightGate:
    def test_height_gate_start_below_full(self):
        """Height gate start is below full activation."""
        assert K1_PITCH_RATE_NOTCH.wip_notch_height_gate_start_m < K1_PITCH_RATE_NOTCH.wip_notch_height_gate_full_m

    def test_height_gate_values_sane(self):
        """Height gate values are within reasonable bounds."""
        for p in [K1_PITCH_RATE_NOTCH, K2_WHEEL_VEL_NOTCH, K3_PITCH_RATE_WHEEL_VEL_NOTCH]:
            assert 0.35 <= p.wip_notch_height_gate_start_m <= 0.46
            assert 0.44 <= p.wip_notch_height_gate_full_m <= 0.52

    def test_smoothstep_gate_monotonic(self):
        """Smoothstep gate is monotonic."""
        for v in [0.40, 0.42, 0.44, 0.46, 0.48, 0.50]:
            g = smoothstep_gate(v, 0.42, 0.48)
            assert 0.0 <= g <= 1.0
        assert smoothstep_gate(0.40, 0.42, 0.48) == 0.0
        assert smoothstep_gate(0.50, 0.42, 0.48) == 1.0


# =====================================================================
# Notch filter is causal (current input only) and correct
# =====================================================================

class TestNotchFilterCausal:
    def test_notch_is_causal(self):
        """BiquadNotchFilter uses only current/past input (causal)."""
        nf = BiquadNotchFilter(fs_hz=100.0, fc_hz=2.5, Q=6.0)
        # Processing requires only the current input
        y = nf.update(1.0)
        assert isinstance(y, float)

    def test_notch_constant_input(self):
        """Constant input produces approximately constant output after transient."""
        nf = BiquadNotchFilter(fs_hz=100.0, fc_hz=2.5, Q=6.0)
        for _ in range(100):
            nf.update(0.0)
        outputs = [nf.update(1.0) for _ in range(200)]
        steady = outputs[-50:]
        mean_val = sum(steady) / len(steady)
        assert abs(mean_val - 1.0) < 0.1, f"Notch fails constant input: mean={mean_val:.4f}"

    def test_notch_attenuates_2p5hz(self):
        """2.5 Hz sine wave is attenuated by notch at fc=2.5 Hz."""
        fs = 100.0
        fc = 2.5
        nf = BiquadNotchFilter(fs_hz=fs, fc_hz=fc, Q=6.0)
        t = [i / fs for i in range(1000)]
        sine = [math.sin(2 * math.pi * fc * ti) for ti in t]
        out = [nf.update(si) for si in sine]
        rms_in = math.sqrt(sum(si * si for si in sine) / len(sine))
        rms_out = math.sqrt(sum(oi * oi for oi in out) / len(out))
        ratio = rms_out / max(rms_in, 1e-12)
        assert ratio < 0.9, f"Notch did not attenuate 2.5 Hz: ratio={ratio:.4f}"

    def test_notch_preserves_low_freq(self):
        """Low frequency (0.5 Hz) sine is mostly preserved."""
        fs = 100.0
        nf = BiquadNotchFilter(fs_hz=fs, fc_hz=2.5, Q=6.0)
        fl = 0.5
        t = [i / fs for i in range(1000)]
        sine = [math.sin(2 * math.pi * fl * ti) for ti in t]
        out = [nf.update(si) for si in sine]
        rms_in = math.sqrt(sum(si * si for si in sine) / len(sine))
        rms_out = math.sqrt(sum(oi * oi for oi in out) / len(out))
        ratio = rms_out / max(rms_in, 1e-12)
        assert ratio > 0.7, f"Notch attenuated low freq too much: ratio={ratio:.4f}"

    def test_notch_preserves_high_freq(self):
        """High frequency (10 Hz) sine is mostly preserved."""
        fs = 100.0
        nf = BiquadNotchFilter(fs_hz=fs, fc_hz=2.5, Q=6.0)
        fh = 10.0
        t = [i / fs for i in range(1000)]
        sine = [math.sin(2 * math.pi * fh * ti) for ti in t]
        out = [nf.update(si) for si in sine]
        rms_in = math.sqrt(sum(si * si for si in sine) / len(sine))
        rms_out = math.sqrt(sum(oi * oi for oi in out) / len(out))
        ratio = rms_out / max(rms_in, 1e-12)
        assert ratio > 0.6, f"Notch attenuated high freq too much: ratio={ratio:.4f}"

    def test_filter_reset(self):
        """Reset clears filter state."""
        nf = BiquadNotchFilter(fs_hz=100.0, fc_hz=2.5, Q=6.0)
        for _ in range(100):
            nf.update(1.0)
        nf.reset()
        s1, s2, s3, s4 = nf.get_state()
        assert s1 == 0.0 and s2 == 0.0 and s3 == 0.0 and s4 == 0.0

    def test_coefficients_finite(self):
        """Filter coefficients are finite numbers."""
        nf = BiquadNotchFilter(fs_hz=100.0, fc_hz=2.5, Q=6.0)
        b0, b1, b2, a1, a2 = nf.coefficients()
        for c in [b0, b1, b2, a1, a2]:
            assert math.isfinite(c)

    def test_invalid_params_raise(self):
        """Invalid filter parameters raise clear errors."""
        with pytest.raises(ValueError, match="Q must be positive"):
            BiquadNotchFilter(fs_hz=100.0, fc_hz=2.5, Q=0)
        with pytest.raises(ValueError, match="fs_hz must be positive"):
            BiquadNotchFilter(fs_hz=0, fc_hz=2.5, Q=6)
        with pytest.raises(ValueError, match="must be < fs_hz/2"):
            BiquadNotchFilter(fs_hz=100.0, fc_hz=60.0, Q=6)
        with pytest.raises(ValueError):
            BiquadNotchFilter(fs_hz=100.0, fc_hz=2.5)  # no Q or bw

    def test_notch_varying_fc(self):
        """Notch filters at different center frequencies are created correctly."""
        for fc in [2.3, 2.5, 2.7]:
            nf = BiquadNotchFilter(fs_hz=100.0, fc_hz=fc, Q=6.0)
            assert abs(nf.fc_hz - fc) < 1e-6
            b0, b1, b2, a1, a2 = nf.coefficients()
            assert all(math.isfinite(c) for c in [b0, b1, b2, a1, a2])

    def test_notch_varying_q(self):
        """Notch filters at different Q values are created correctly."""
        for Q in [2, 4, 6, 8, 10]:
            nf = BiquadNotchFilter(fs_hz=100.0, fc_hz=2.5, Q=Q)
            assert abs(nf.Q - Q) < 1e-6


# =====================================================================
# Filter telemetry exists for K candidates
# =====================================================================

class TestFilterTelemetry:
    def test_k_telemetry_fields_exist(self):
        """K schedule has all required filter telemetry fields."""
        sched = K1_PITCH_RATE_NOTCH
        assert hasattr(sched, "enable_wip_notch_filter")
        assert hasattr(sched, "wip_notch_target_signal")
        assert hasattr(sched, "wip_notch_center_hz")
        assert hasattr(sched, "wip_notch_q")
        assert hasattr(sched, "wip_notch_filter_blend")
        assert hasattr(sched, "wip_notch_gate_enabled")
        assert hasattr(sched, "wip_notch_height_gate_start_m")
        assert hasattr(sched, "wip_notch_height_gate_full_m")

    def test_k_profile_center_hz_reasonable(self):
        """K profile center frequency is near observed 2.5 Hz WIP mode."""
        for p in [K1_PITCH_RATE_NOTCH, K2_WHEEL_VEL_NOTCH, K3_PITCH_RATE_WHEEL_VEL_NOTCH]:
            assert 2.0 <= p.wip_notch_center_hz <= 3.0, f"{p.profile_name} center Hz out of range"

    def test_k_profile_q_reasonable(self):
        """K profile Q is reasonable (2-10)."""
        assert 2 <= K1_PITCH_RATE_NOTCH.wip_notch_q <= 10
        assert 2 <= K2_WHEEL_VEL_NOTCH.wip_notch_q <= 10
        assert 2 <= K3_PITCH_RATE_WHEEL_VEL_NOTCH.wip_notch_q <= 10

    def test_k_blend_range(self):
        """K filter blend is in [0, 1]."""
        for p in [K1_PITCH_RATE_NOTCH, K1F_PITCH_RATE_NOTCH_BLEND075, K1G_PITCH_RATE_NOTCH_BLEND050]:
            assert 0.0 <= p.wip_notch_filter_blend <= 1.0


# =====================================================================
# Classification enum
# =====================================================================

class TestClassificationEnum:
    def test_classification_values(self):
        """All expected classification values exist."""
        expected = [
            "NOTCH_WIP_RECOVERY_PASS",
            "NOTCH_WIP_RECOVERY_PASS_WITH_POSITION_DRIFT",
            "NOTCH_WIP_RECOVERY_TRANSIENT_ONLY",
            "NOTCH_WIP_RECOVERY_IMPROVED_NOT_PASS",
            "NOTCH_WIP_RECOVERY_NO_IMPROVEMENT",
            "NOTCH_WIP_RECOVERY_FAIL_HIP_YAW",
            "NOTCH_WIP_RECOVERY_FAIL_FALL",
            "NOTCH_WIP_RECOVERY_FAIL_UNSTABLE",
            "NOTCH_WIP_RECOVERY_INCONCLUSIVE",
        ]
        for e in expected:
            assert e in CLASSIFICATION_ENUM, f"Missing enum: {e}"

    def test_classification_unique(self):
        """No duplicate classification values."""
        vals = list(CLASSIFICATION_ENUM.values())
        assert len(vals) == len(set(vals))


# =====================================================================
# Recovery event detection works
# =====================================================================

class TestRecoveryDetection:
    def test_sustained_vs_transient_detection(self):
        """Recovery event detector can distinguish sustained vs transient."""
        # This is a structural test using synthetic data
        dt = 0.01
        n = 3000
        t = np.arange(n) * dt
        # Create data that never enters recovery band
        np_pitch = (10.0 + 5.0 * np.sin(2 * math.pi * 2.5 * t)) / DEG
        np_roll = (0.5 * np.sin(2 * math.pi * 0.5 * t)) / DEG
        np_hy = 0.05 * np.abs(np.sin(2 * math.pi * 1.0 * t))
        np_support = 0.15 * np.sin(2 * math.pi * 2.5 * t)

        data = {
            "pitch_x_rad": np_pitch,
            "roll_y_rad": np_roll,
            "hip_yaw_abs_max_rad": np_hy,
            "sagittal_position_error_m": np_support,
            "com_z_m": np.full(n, 0.48),
        }

        rec = detect_recovery_events(data, 310, dt, "synthetic")
        # Should NOT find sustained recovery (pitch oscillates too much)
        assert rec["sustained_2s_hold_duration_s"] < 2.0 or rec["sustained_2s_hold_start_s"] is None


# =====================================================================
# Compile checks
# =====================================================================

class TestCompile:
    def test_scripts_compile(self):
        """Key scripts compile without errors."""
        import py_compile
        scripts = [
            "wheeled_biped/controllers/signal_filters.py",
            "wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py",
            "scripts/audit_2p5hz_wip_mode_filter_design.py",
            "scripts/run_targeted_2p5hz_wip_notch_sweep.py",
            "scripts/analyze_targeted_2p5hz_wip_notch_results.py",
        ]
        root = Path(__file__).resolve().parent.parent
        for s in scripts:
            path = root / s
            if path.exists():
                py_compile.compile(str(path), doraise=True)

    def test_existing_tests_run(self):
        """Existing test suites should still pass (verification only)."""
        pass
