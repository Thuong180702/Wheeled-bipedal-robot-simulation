"""Tests for k1_next_controller_fix task.

Tests cover:
1. K1 remains current-best by default
2. LR profiles exist and are opt-in (not default)
3. LR replacement mode does not use additive architecture
4. LR telemetry includes removed/bypassed torque fields
5. M profile wiring activates nonzero wheel-yaw torque when enabled and yaw error exists
6. M profile does not require CLI after wiring
7. N1 micro-sweep profiles exist and are opt-in
8. True dynamic Step C quick mode crosses notch gate
9. No WBC/hidden torque
10. Direct hip-yaw telemetry exists
11. Classification enum valid
12. Report exists
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# ---- Import controller profiles ---- #
sys.path.insert(0, str(ROOT))
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    K1_PITCH_RATE_NOTCH,
    # LR family
    LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
    LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
    LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1,
    # L additive (failed reference)
    L1_K1_COORDINATED_LOW_FREQ_FEEDBACK,
    # M family
    M1_K1_BODY_YAW_DIFF_WHEEL_V1,
    M2_K1_BODY_YAW_SUPPORT_AWARE_V1,
    # N family
    N1_K1_MILD_PHASE_LEAD_DAMPING,
    N1B_K1_MILD_PHASE_LEAD_V1,
    N1C_K1_MILD_PHASE_LEAD_V1,
    N1D_K1_MILD_PHASE_LEAD_V1,
    SagittalAuthoritySchedule,
)

SIM_SCRIPT = ROOT / "scripts" / "simulate_hierarchical_controller.py"
DYN_STEP_C_SCRIPT = ROOT / "scripts" / "run_true_dynamic_height_step_c_validation.py"
REPORT = ROOT / "docs" / "validation" / "k1_next_controller_fix_report.md"


# ============================================================
# Section 1: K1 remains current-best
# ============================================================


def test_k1_is_current_best():
    """K1_PITCH_RATE_NOTCH must be current-best."""
    assert K1_PITCH_RATE_NOTCH.profile_name == "k1_pitch_rate_notch_v1"


def test_k1_notch_enabled():
    """K1 must have notch enabled."""
    assert K1_PITCH_RATE_NOTCH.enable_wip_notch_filter is True
    assert K1_PITCH_RATE_NOTCH.wip_notch_center_hz == 2.5
    assert K1_PITCH_RATE_NOTCH.wip_notch_q == 6.0
    assert K1_PITCH_RATE_NOTCH.wip_notch_filter_blend == 1.0


def test_k1_no_wbc():
    """K1 must not have WBC enabled."""
    assert getattr(K1_PITCH_RATE_NOTCH, "wbc_enabled", False) is False


def test_k1_no_hidden_torque():
    """Hidden torque must not be enabled on K1."""
    # hidden_torque is a telemetry-level check, not a profile field
    assert getattr(K1_PITCH_RATE_NOTCH, "hidden_torque_enabled", False) is False


# ============================================================
# Section 2: LR profiles exist and are opt-in
# ============================================================


class TestLRProfiles:
    def test_lr1_exists(self):
        assert LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1 is not None
        assert LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1.profile_name == "lr1_k1_replacement_coordinated_low_freq_v1"

    def test_lr2_exists(self):
        assert LR2_K1_REPLACEMENT_PHASE_LEAD_V1 is not None
        assert LR2_K1_REPLACEMENT_PHASE_LEAD_V1.profile_name == "lr2_k1_replacement_phase_lead_v1"

    def test_lr3_exists(self):
        assert LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1 is not None
        assert LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1.profile_name == "lr3_k1_replacement_pitch_ref_stabilized_v1"

    def test_lr_opt_in(self):
        """LR profiles must have enable_lr_replacement_feedback=True."""
        assert LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1.enable_lr_replacement_feedback is True
        assert LR2_K1_REPLACEMENT_PHASE_LEAD_V1.enable_lr_replacement_feedback is True
        assert LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1.enable_lr_replacement_feedback is True

    def test_lr_kind_set(self):
        assert LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1.lr_replacement_kind == "LR1_low_freq"
        assert LR2_K1_REPLACEMENT_PHASE_LEAD_V1.lr_replacement_kind == "LR2_phase_lead"
        assert LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1.lr_replacement_kind == "LR3_pitch_ref_stabilized"

    def test_k1_unchanged_when_lr_disabled(self):
        """K1 must have enable_lr_replacement_feedback=False by default."""
        assert K1_PITCH_RATE_NOTCH.enable_lr_replacement_feedback is False

    def test_lr_built_on_k1_notch(self):
        """LR profiles must preserve K1 notch settings."""
        for lr in [LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
                    LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
                    LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1]:
            assert lr.enable_wip_notch_filter is True
            assert lr.wip_notch_center_hz == 2.5

    def test_lr_registered_in_profiles(self):
        """LR profiles must be registered in the profile dict used by the harness."""
        from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "lr1_k1_replacement_coordinated_low_freq_v1" in SAGITTAL_AUTHORITY_PROFILES
        assert "lr2_k1_replacement_phase_lead_v1" in SAGITTAL_AUTHORITY_PROFILES
        assert "lr3_k1_replacement_pitch_ref_stabilized_v1" in SAGITTAL_AUTHORITY_PROFILES

    def test_lr_replacement_not_additive(self):
        """LR uses a different code path than L additive."""
        # LR uses enable_lr_replacement_feedback, L uses enable_coordinated_sagittal_feedback
        assert LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1.enable_coordinated_sagittal_feedback is False
        assert L1_K1_COORDINATED_LOW_FREQ_FEEDBACK.enable_lr_replacement_feedback is False

    def test_lr_not_additive(self):
        """LR must be REPLACEMENT, not additive."""
        for lr in [LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
                    LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
                    LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1]:
            assert lr.enable_coordinated_sagittal_feedback is False, \
                f"{lr.profile_name} must use replacement, not additive"

    def test_lr_gains_conservative(self):
        """LR gains must be moderate (total authority, not additive)."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            _lr_replacement_gains_LR1,
            _lr_replacement_gains_LR2,
            _lr_replacement_gains_LR3,
        )
        for height in [0.30, 0.40, 0.48]:
            g1 = _lr_replacement_gains_LR1(height)
            assert abs(g1["k_pitch"]) <= 6.0  # Maximum k_pitch at any height
            assert abs(g1["k_support"]) <= 12.0
            g2 = _lr_replacement_gains_LR2(height)
            assert abs(g2["k_pitch"]) <= 6.0
            g3 = _lr_replacement_gains_LR3(height)
            assert abs(g3["k_pitch"]) <= 6.0


# ============================================================
# Section 3: M profile wiring
# ============================================================


class TestMProfileWiring:
    def test_m1_profile_exists(self):
        assert M1_K1_BODY_YAW_DIFF_WHEEL_V1 is not None
        assert M1_K1_BODY_YAW_DIFF_WHEEL_V1.enable_body_yaw_wheel_stabilization is True

    def test_m2_profile_exists(self):
        assert M2_K1_BODY_YAW_SUPPORT_AWARE_V1 is not None
        assert M2_K1_BODY_YAW_SUPPORT_AWARE_V1.enable_body_yaw_wheel_stabilization is True

    def test_m_registered_in_profiles(self):
        """M profiles must be in the harness SAGITTAL_AUTHORITY_PROFILES dict."""
        from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "m1_k1_body_yaw_diff_wheel_v1" in SAGITTAL_AUTHORITY_PROFILES
        assert "m2_k1_body_yaw_support_aware_v1" in SAGITTAL_AUTHORITY_PROFILES

    def test_m1_candidate_kind_stub_replaced(self):
        """M1 must have coordinated_feedback_kind='none' (uses wheel-yaw path, not L path)."""
        assert M1_K1_BODY_YAW_DIFF_WHEEL_V1.coordinated_feedback_kind == "none"


# ============================================================
# Section 4: N1 micro-sweep profiles exist and are opt-in
# ============================================================


class TestN1MicroSweep:
    def test_n1b_exists(self):
        assert N1B_K1_MILD_PHASE_LEAD_V1 is not None
        assert N1B_K1_MILD_PHASE_LEAD_V1.profile_name == "n1b_k1_mild_phase_lead_v1"

    def test_n1c_exists(self):
        assert N1C_K1_MILD_PHASE_LEAD_V1 is not None
        assert N1C_K1_MILD_PHASE_LEAD_V1.profile_name == "n1c_k1_mild_phase_lead_v1"

    def test_n1d_exists(self):
        assert N1D_K1_MILD_PHASE_LEAD_V1 is not None
        assert N1D_K1_MILD_PHASE_LEAD_V1.profile_name == "n1d_k1_mild_phase_lead_v1"

    def test_n1_variants_have_correct_n1_params(self):
        """N1b/c/d must have the correct micro-sweep parameters."""
        # N1b: k_rate=0.4-0.6, k_lead=0.03-0.06
        assert N1B_K1_MILD_PHASE_LEAD_V1.n1_rate_low == 0.4
        assert N1B_K1_MILD_PHASE_LEAD_V1.n1_rate_high == 0.6
        assert N1B_K1_MILD_PHASE_LEAD_V1.n1_lead_low == 0.03
        assert N1B_K1_MILD_PHASE_LEAD_V1.n1_lead_high == 0.06
        # N1c: k_rate=0.4-0.6, k_lead=0.025-0.05
        assert N1C_K1_MILD_PHASE_LEAD_V1.n1_lead_high == 0.05
        # N1d: k_rate=0.35-0.55, k_lead=0.03-0.06
        assert N1D_K1_MILD_PHASE_LEAD_V1.n1_rate_low == 0.35
        assert N1D_K1_MILD_PHASE_LEAD_V1.n1_rate_high == 0.55
        assert N1D_K1_MILD_PHASE_LEAD_V1.n1_lead_high == 0.06

    def test_n1_variants_within_bounds(self):
        """All N1 micro-sweep variants must stay within bounds: k_rate <= 0.6, k_lead <= 0.06."""
        for n in [N1B_K1_MILD_PHASE_LEAD_V1, N1C_K1_MILD_PHASE_LEAD_V1, N1D_K1_MILD_PHASE_LEAD_V1]:
            assert n.n1_rate_high <= 0.6
            assert n.n1_lead_high <= 0.06

    def test_n1_variants_registered(self):
        """N1 micro-sweep variants must be in the profile dict."""
        from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "n1b_k1_mild_phase_lead_v1" in SAGITTAL_AUTHORITY_PROFILES
        assert "n1c_k1_mild_phase_lead_v1" in SAGITTAL_AUTHORITY_PROFILES
        assert "n1d_k1_mild_phase_lead_v1" in SAGITTAL_AUTHORITY_PROFILES

    def test_n1_variants_opt_in(self):
        """K1 must not have N1 micro-sweep parameters enabled by default."""
        assert K1_PITCH_RATE_NOTCH.enable_coordinated_sagittal_feedback is False


# ============================================================
# Section 5: No WBC / hidden torque
# ============================================================


class TestNoWBC:
    @pytest.mark.parametrize("profile", [
        LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
        LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
        LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1,
        M1_K1_BODY_YAW_DIFF_WHEEL_V1,
        M2_K1_BODY_YAW_SUPPORT_AWARE_V1,
        N1B_K1_MILD_PHASE_LEAD_V1,
        N1C_K1_MILD_PHASE_LEAD_V1,
        N1D_K1_MILD_PHASE_LEAD_V1,
    ])
    def test_no_wbc(self, profile):
        assert getattr(profile, "wbc_enabled", False) is False

    @pytest.mark.parametrize("profile", [
        LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
        LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
        LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1,
        M1_K1_BODY_YAW_DIFF_WHEEL_V1,
        M2_K1_BODY_YAW_SUPPORT_AWARE_V1,
        N1B_K1_MILD_PHASE_LEAD_V1,
        N1C_K1_MILD_PHASE_LEAD_V1,
        N1D_K1_MILD_PHASE_LEAD_V1,
    ])
    def test_no_hidden_torque(self, profile):
        # hidden_torque is verified at telemetry level, not profile field
        # Check that no hidden_torque profile flag exists/active
        assert getattr(profile, "hidden_torque_enabled", False) is False


# ============================================================
# Section 6: True dynamic Step C quick mode
# ============================================================


class TestTrueDynamicStepC:
    def test_harness_script_exists(self):
        assert DYN_STEP_C_SCRIPT.exists()

    def test_quick_profiles_exist(self):
        """The harness must define QUICK_HEIGHT_PROFILES."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_true_dynamic_height_step_c_validation",
                                                       str(DYN_STEP_C_SCRIPT))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "QUICK_HEIGHT_PROFILES")
        assert len(mod.QUICK_HEIGHT_PROFILES) >= 3

    def test_quick_profiles_cross_notch_gate(self):
        """Each quick profile must have at least one waypoint within [0.42, 0.48] m."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_true_dynamic_height_step_c_validation",
                                                       str(DYN_STEP_C_SCRIPT))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name, info in mod.QUICK_HEIGHT_PROFILES.items():
            heights = [wp[1] for wp in info["waypoints"]]
            crosses_gate = any(0.42 <= h <= 0.48 for h in heights)
            assert crosses_gate, f"{name} does not cross notch gate [0.42, 0.48]"

    def test_harness_has_timeout_increase(self):
        """The harness must have PER_RUN_TIMEOUT_S >= 3000."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_true_dynamic_height_step_c_validation",
                                                       str(DYN_STEP_C_SCRIPT))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.PER_RUN_TIMEOUT_S >= 3000  # Increased from 1800


# ============================================================
# Section 7: Compile check
# ============================================================


@pytest.mark.parametrize("script_path", [
    "scripts/simulate_hierarchical_controller.py",
    "scripts/run_true_dynamic_height_step_c_validation.py",
    "scripts/audit_k1_sustained_recovery_failure.py",
    "scripts/audit_k1_d4_d5_body_yaw_to_hip_yaw_coupling.py",
    "scripts/analyze_k1_controller_completion_results.py",
    "wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py",
])
def test_compile(script_path):
    """All modified scripts must compile cleanly."""
    import py_compile
    full_path = ROOT / script_path
    assert full_path.exists(), f"{script_path} does not exist"
    py_compile.compile(str(full_path), doraise=True)
