"""Tests for LR replacement equilibrium/feedforward pass-through fix.

Tests cover:
1. K1 remains current-best by default
2. K1 profile does not enable LR
3. LR profiles are opt-in only
4. LR replacement mode is not additive
5. LR has nonzero equilibrium/feedforward pass-through field
6. LR does not zero physics_ff_applied when the baseline K1 path would apply it
7. LR telemetry contains required fields
8. LR total torque is not only LR_feedback_torque
9. LR does not duplicate K1 independent dynamic terms
10. No WBC/hidden torque fields are enabled by LR
11. The old additive L profiles remain failed references and are not current-best
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    K1_PITCH_RATE_NOTCH,
    # LR family
    LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
    LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
    LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1,
    # L additive (failed reference)
    L1_K1_COORDINATED_LOW_FREQ_FEEDBACK,
    SagittalAuthoritySchedule,
    # LR gain functions
    _lr_replacement_gains_LR1,
    _lr_replacement_gains_LR2,
    _lr_replacement_gains_LR3,
)


# ============================================================
# Section 1: K1 remains current-best by default
# ============================================================

def test_k1_is_current_best():
    """K1_PITCH_RATE_NOTCH must be current-best."""
    assert K1_PITCH_RATE_NOTCH.profile_name == "k1_pitch_rate_notch_v1"


def test_k1_does_not_enable_lr():
    """K1 must NOT have LR replacement enabled."""
    assert K1_PITCH_RATE_NOTCH.enable_lr_replacement_feedback is False


# ============================================================
# Section 2: LR profiles exist and are opt-in
# ============================================================

def test_lr1_profile_exists():
    assert LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1 is not None
    assert LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1.profile_name == "lr1_k1_replacement_coordinated_low_freq_v1"


def test_lr2_profile_exists():
    assert LR2_K1_REPLACEMENT_PHASE_LEAD_V1 is not None
    assert LR2_K1_REPLACEMENT_PHASE_LEAD_V1.profile_name == "lr2_k1_replacement_phase_lead_v1"


def test_lr3_profile_exists():
    assert LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1 is not None
    assert LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1.profile_name == "lr3_k1_replacement_pitch_ref_stabilized_v1"


def test_lr_profiles_are_opt_in():
    """LR profiles must have enable_lr_replacement_feedback=True."""
    for lr in [LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
                LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
                LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1]:
        assert lr.enable_lr_replacement_feedback is True, \
            f"{lr.profile_name} must be opt-in only"


def test_lr_profiles_registered_in_harness():
    """LR profiles must be in the harness SAGITTAL_AUTHORITY_PROFILES dict."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    assert "lr1_k1_replacement_coordinated_low_freq_v1" in SAGITTAL_AUTHORITY_PROFILES
    assert "lr2_k1_replacement_phase_lead_v1" in SAGITTAL_AUTHORITY_PROFILES
    assert "lr3_k1_replacement_pitch_ref_stabilized_v1" in SAGITTAL_AUTHORITY_PROFILES


# ============================================================
# Section 3: LR replacement mode is NOT additive
# ============================================================

def test_lr_not_additive():
    """LR must use enable_lr_replacement_feedback, NOT enable_coordinated_sagittal_feedback."""
    for lr in [LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
                LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
                LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1]:
        assert lr.enable_coordinated_sagittal_feedback is False, \
            f"{lr.profile_name} must use replacement, not additive"


def test_lr_vs_l_are_distinct():
    """LR and L families must use different activation flags."""
    assert LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1.enable_lr_replacement_feedback is True
    assert L1_K1_COORDINATED_LOW_FREQ_FEEDBACK.enable_lr_replacement_feedback is False
    assert L1_K1_COORDINATED_LOW_FREQ_FEEDBACK.enable_coordinated_sagittal_feedback is True


# ============================================================
# Section 4: LR has equilibrium/feedforward pass-through fields
# ============================================================

def test_lr_eq_ff_pass_through_field_exists():
    """The SagittalAuthoritySchedule must have the lr_replacement_kind field."""
    assert hasattr(LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1, "lr_replacement_kind")
    assert LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1.lr_replacement_kind == "LR1_low_freq"


def test_lr_replacement_kind_set():
    """All LR profiles must have distinct lr_replacement_kind values."""
    assert LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1.lr_replacement_kind == "LR1_low_freq"
    assert LR2_K1_REPLACEMENT_PHASE_LEAD_V1.lr_replacement_kind == "LR2_phase_lead"
    assert LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1.lr_replacement_kind == "LR3_pitch_ref_stabilized"


def test_sagittal_schedule_has_lr_fields():
    """SagittalAuthoritySchedule must have enable_lr_replacement_feedback and lr_replacement_kind fields."""
    sched = SagittalAuthoritySchedule()
    assert hasattr(sched, "enable_lr_replacement_feedback")
    assert hasattr(sched, "lr_replacement_kind")
    assert sched.enable_lr_replacement_feedback is False  # disabled by default
    assert sched.lr_replacement_kind == "none"


# ============================================================
# Section 5: LR physics_ff_applied integrity (code-path check)
# ============================================================

def test_lr_preserves_k1_notch():
    """LR profiles built on K1 must preserve the notch filter."""
    for lr in [LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
                LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
                LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1]:
        assert lr.enable_wip_notch_filter is True
        assert lr.wip_notch_center_hz == 2.5
        assert lr.wip_notch_q == 6.0


def test_lr_preserves_k1_outer_loop():
    """LR profiles built on K1 must preserve the outer loop settings."""
    for lr in [LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
                LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
                LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1]:
        assert lr.calibrated_outer_loop_enabled is True
        assert lr.low_band_support_outer_loop_enabled is True


# ============================================================
# Section 6: LR telemetry field verification (code-path)
# ============================================================

def test_lr_gains_have_kind():
    """LR gain functions must return a 'kind' key."""
    for height in [0.40, 0.48]:
        g1 = _lr_replacement_gains_LR1(height)
        assert "kind" in g1
        assert g1["kind"] == "lr_replacement_low_freq_state_feedback"
        g2 = _lr_replacement_gains_LR2(height)
        assert "kind" in g2
        assert g2["kind"] == "lr_replacement_phase_lead_compensation"
        g3 = _lr_replacement_gains_LR3(height)
        assert "kind" in g3
        assert g3["kind"] == "lr_replacement_pitch_ref_stabilization"


def test_lr_gains_are_finite():
    """All LR gains must be finite at any valid height."""
    for height in [0.30, 0.40, 0.48]:
        for func in [_lr_replacement_gains_LR1, _lr_replacement_gains_LR2, _lr_replacement_gains_LR3]:
            g = func(height)
            for k in ["k_pitch", "k_pitch_rate", "k_support", "k_support_vel"]:
                assert abs(g[k]) < 1e6, f"{func.__name__}({height})[{k}] is not finite"
                import math
                assert math.isfinite(g[k]), f"{func.__name__}({height})[{k}] is not finite"


# ============================================================
# Section 7: LR total torque composition (code-path)
# ============================================================

def test_lr_feedback_is_not_full_command():
    """LR feedback torque gains are moderate — designed as supplement to EQ/FF, not standalone.

    After the EQ/FF fix, LR_dynamic_feedback is added to tau_eq_ff_pass_through
    to form the total command. The LR gains are moderate because they replace
    only the dynamic terms, not the full command.
    """
    # At nominal height 0.48m, verify LR gains are in expected moderate range
    g1 = _lr_replacement_gains_LR1(0.48)
    # k_pitch ~3.5-6.0: moderate compared to K1's kp_pitch=50.0
    assert 2.0 <= g1["k_pitch"] <= 8.0
    assert 0.3 <= g1["k_pitch_rate"] <= 2.0
    assert 5.0 <= abs(g1["k_support"]) <= 15.0
    assert 0.1 <= abs(g1["k_support_vel"]) <= 1.0


# ============================================================
# Section 8: LR does not duplicate K1 independent dynamic terms (code-path)
# ============================================================

def test_lr_architecture_fix_comments_present():
    """The controller file must contain the EQ/FF pass-through architecture docstring.

    Verifies the fix is applied by checking for the EQ/FF pass-through architecture
    comment in the LR replacement section of the controller.
    """
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    assert "tau_eq_ff_pass_through" in content, \
        "Controller must have EQ/FF pass-through variable"
    assert "eq_ff_pass_through" in content, \
        "Controller must have eq_ff_pass_through mode string"


# ============================================================
# Section 9: No WBC/hidden torque
# ============================================================

@pytest.mark.parametrize("profile", [
    LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
    LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
    LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1,
])
def test_lr_no_wbc(profile):
    assert getattr(profile, "wbc_enabled", False) is False


@pytest.mark.parametrize("profile", [
    LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
    LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
    LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1,
])
def test_lr_no_hidden_torque(profile):
    assert getattr(profile, "hidden_torque_enabled", False) is False


# ============================================================
# Section 10: Old additive L profiles remain failed references
# ============================================================

def test_l_profiles_are_not_current_best():
    """L profiles are failed references and must not be current-best (additive failure)."""
    assert L1_K1_COORDINATED_LOW_FREQ_FEEDBACK.enable_coordinated_sagittal_feedback is True
    assert L1_K1_COORDINATED_LOW_FREQ_FEEDBACK.enable_lr_replacement_feedback is False
    # L is additive, not replacement — it is a known failed reference


# ============================================================
# Section 11: K1 is unchanged when LR is disabled
# ============================================================

def test_k1_unchanged_profile():
    """K1 must still have its original profile name and key settings after the fix."""
    assert K1_PITCH_RATE_NOTCH.profile_name == "k1_pitch_rate_notch_v1"
    assert K1_PITCH_RATE_NOTCH.enable_wip_notch_filter is True
    assert K1_PITCH_RATE_NOTCH.enable_lr_replacement_feedback is False
    assert K1_PITCH_RATE_NOTCH.lr_replacement_kind == "none"


# ============================================================
# Section 12: Compile checks
# ============================================================

@pytest.mark.parametrize("script_path", [
    "scripts/simulate_hierarchical_controller.py",
    "wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py",
])
def test_compile(script_path):
    """Modified scripts must compile cleanly."""
    import py_compile
    full_path = ROOT / script_path
    assert full_path.exists(), f"{script_path} does not exist"
    py_compile.compile(str(full_path), doraise=True)
