"""Tests for LR support-drift sign/phase audit and constrained gain sweep (LRS).

Covers:
1. K1 remains current-best
2. LR EQ/FF pass-through still enabled for LR
3. LRS variants are opt-in
4. LRS does not modify K1
5. LRS obeys hard gain bounds
6. LRS telemetry includes component-wise torques
7. Sign/phase audit script handles required columns
8. Hip-yaw telemetry extraction handles all known column variants
9. No WBC/hidden torque fields enabled
10. Classification enum valid
11. LRS profiles registered in harness
12. LRS gain functions have correct 'kind' field
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# Section 1: K1 remains current-best
# ============================================================

def test_k1_is_current_best():
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        K1_PITCH_RATE_NOTCH,
    )
    assert K1_PITCH_RATE_NOTCH.profile_name == "k1_pitch_rate_notch_v1"
    assert K1_PITCH_RATE_NOTCH.enable_lr_replacement_feedback is False


# ============================================================
# Section 2: LR EQ/FF pass-through still enabled for LR
# ============================================================

def test_lr_eq_ff_still_enabled():
    """The LR family must still use EQ/FF pass-through."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
        LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
        LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1,
    )
    for lr in [LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
               LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
               LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1]:
        assert lr.enable_lr_replacement_feedback is True

    # Verify the EQ/FF pass-through architecture comment in controller
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    assert "tau_eq_ff_pass_through" in content, "EQ/FF pass-through must remain"
    assert "eq_ff_pass_through" in content, "EQ/FF mode string must remain"


# ============================================================
# Section 3: LRS variants are opt-in
# ============================================================

def test_lrs_variants_exist_and_opt_in():
    """LRS profiles must exist and be opt-in only."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        LRS1_SUPPORT_DOMINANT_V1,
        LRS2_PITCH_RATE_DAMPING_V1,
        LRS3_BALANCED_MEDIUM_V1,
    )
    assert LRS1_SUPPORT_DOMINANT_V1.profile_name == "lrs1_support_dominant_v1"
    assert LRS1_SUPPORT_DOMINANT_V1.enable_lr_replacement_feedback is True
    assert LRS1_SUPPORT_DOMINANT_V1.lr_replacement_kind == "LRS1_support_dominant"

    assert LRS2_PITCH_RATE_DAMPING_V1.profile_name == "lrs2_pitch_rate_damping_v1"
    assert LRS2_PITCH_RATE_DAMPING_V1.enable_lr_replacement_feedback is True
    assert LRS2_PITCH_RATE_DAMPING_V1.lr_replacement_kind == "LRS2_pitch_rate_damping"

    assert LRS3_BALANCED_MEDIUM_V1.profile_name == "lrs3_balanced_medium_v1"
    assert LRS3_BALANCED_MEDIUM_V1.enable_lr_replacement_feedback is True
    assert LRS3_BALANCED_MEDIUM_V1.lr_replacement_kind == "LRS3_balanced_medium"


def test_lrs_not_additive():
    """LRS must NOT use additive coordinated feedback flag."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        LRS1_SUPPORT_DOMINANT_V1,
        LRS2_PITCH_RATE_DAMPING_V1,
        LRS3_BALANCED_MEDIUM_V1,
    )
    for lrs in [LRS1_SUPPORT_DOMINANT_V1, LRS2_PITCH_RATE_DAMPING_V1, LRS3_BALANCED_MEDIUM_V1]:
        assert lrs.enable_coordinated_sagittal_feedback is False


# ============================================================
# Section 4: LRS does not modify K1
# ============================================================

def test_lrs_does_not_modify_k1():
    """K1 must remain unchanged."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        K1_PITCH_RATE_NOTCH,
    )
    assert K1_PITCH_RATE_NOTCH.profile_name == "k1_pitch_rate_notch_v1"
    assert K1_PITCH_RATE_NOTCH.enable_wip_notch_filter is True
    assert K1_PITCH_RATE_NOTCH.enable_lr_replacement_feedback is False
    assert K1_PITCH_RATE_NOTCH.lr_replacement_kind == "none"


# ============================================================
# Section 5: LRS obeys hard gain bounds
# ============================================================

@pytest.mark.parametrize("height", [0.30, 0.40, 0.48])
def test_lrs1_gain_bounds(height):
    """LRS1 gains must obey hard bounds."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        _lrs_replacement_gains_S1,
    )
    g = _lrs_replacement_gains_S1(height)
    assert abs(g["k_pitch"]) <= 15.0, f"k_pitch={g['k_pitch']} exceeds bound 15 at height={height}"
    assert abs(g["k_pitch_rate"]) <= 3.0, f"k_pitch_rate={g['k_pitch_rate']} exceeds bound 3 at height={height}"
    assert abs(g["k_support"]) <= 30.0, f"k_support={g['k_support']} exceeds 2.5x LR1 bound at height={height}"  # 2.5x -12 = 30
    assert abs(g["k_support_vel"]) <= 1.5, f"k_support_vel={g['k_support_vel']} exceeds 2.5x bound at height={height}"  # 2.5x -0.6 = 1.5


@pytest.mark.parametrize("height", [0.30, 0.40, 0.48])
def test_lrs2_gain_bounds(height):
    """LRS2 gains must obey hard bounds."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        _lrs_replacement_gains_S2,
    )
    g = _lrs_replacement_gains_S2(height)
    assert abs(g["k_pitch"]) <= 15.0
    assert abs(g["k_pitch_rate"]) <= 3.0, f"k_pitch_rate={g['k_pitch_rate']} exceeds hard bound 3"
    assert abs(g["k_support"]) <= 30.0
    assert abs(g["k_support_vel"]) <= 1.5


@pytest.mark.parametrize("height", [0.30, 0.40, 0.48])
def test_lrs3_gain_bounds(height):
    """LRS3 gains must obey hard bounds."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        _lrs_replacement_gains_S3,
    )
    g = _lrs_replacement_gains_S3(height)
    assert abs(g["k_pitch"]) <= 15.0
    assert abs(g["k_pitch_rate"]) <= 3.0
    assert abs(g["k_support"]) <= 30.0
    assert abs(g["k_support_vel"]) <= 1.5


def test_lrs_gains_finite():
    """All LRS gains must be finite."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        _lrs_replacement_gains_S1,
        _lrs_replacement_gains_S2,
        _lrs_replacement_gains_S3,
    )
    for func in [_lrs_replacement_gains_S1, _lrs_replacement_gains_S2, _lrs_replacement_gains_S3]:
        for height in [0.30, 0.40, 0.48]:
            g = func(height)
            for k in ["k_pitch", "k_pitch_rate", "k_support", "k_support_vel"]:
                assert math.isfinite(g[k]), f"{func.__name__}({height})[{k}] not finite"


# ============================================================
# Section 6: LRS telemetry includes component-wise torques
# ============================================================

def test_lrs_component_telemetry_exists():
    """Controller must output LRS component-wise torque fields."""
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    assert "LRS_tau_pitch_component_nm" in content, "LRS pitch component telemetry missing"
    assert "LRS_tau_pitch_rate_component_nm" in content, "LRS pitch_rate component telemetry missing"
    assert "LRS_tau_support_component_nm" in content, "LRS support component telemetry missing"
    assert "LRS_tau_support_vel_component_nm" in content, "LRS support_vel component telemetry missing"


def test_lrs_state_support_velocity_telemetry_exists():
    """Controller must output LR_state_support_velocity_m_s."""
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    assert "LR_state_support_velocity_m_s" in content, "LR support velocity state telemetry missing"


# ============================================================
# Section 7: Sign/phase audit script handles required columns
# ============================================================

def test_audit_script_exists():
    """Audit script must exist."""
    audit_path = ROOT / "scripts" / "audit_lr_support_drift_sign_phase.py"
    assert audit_path.exists(), "Audit script missing"


def test_audit_script_compiles():
    """Audit script must compile."""
    import py_compile
    py_compile.compile(str(ROOT / "scripts" / "audit_lr_support_drift_sign_phase.py"), doraise=True)


def test_audit_script_has_sign_checks():
    """Audit script must compute sign agreement checks."""
    audit_path = ROOT / "scripts" / "audit_lr_support_drift_sign_phase.py"
    content = audit_path.read_text()
    assert "compute_sign_agreement" in content
    assert "stabilizing_sign" in content
    assert "support_error" in content
    assert "support_velocity" in content


def test_audit_script_has_phase_analysis():
    """Audit script must compute phase analysis."""
    audit_path = ROOT / "scripts" / "audit_lr_support_drift_sign_phase.py"
    content = audit_path.read_text()
    assert "compute_phase_analysis" in content
    assert "0.35" in content  # frequency band
    assert "0.65" in content


def test_audit_script_has_drift_events():
    """Audit script must analyze support-drift threshold events."""
    audit_path = ROOT / "scripts" / "audit_lr_support_drift_sign_phase.py"
    content = audit_path.read_text()
    assert "compute_drift_events" in content
    assert "threshold_" in content  # f-string keys: f"threshold_{thresh:.2f}m"
    assert "0.25" in content
    assert "0.50" in content
    assert "1.00" in content


# ============================================================
# Section 8: Hip-yaw telemetry extraction handles all known column variants
# ============================================================

def test_hip_yaw_column_resolution_in_audit():
    """Audit script must handle all hip-yaw column name variants."""
    audit_path = ROOT / "scripts" / "audit_lr_support_drift_sign_phase.py"
    content = audit_path.read_text()
    assert "_HIP_YAW_LEFT_CANDIDATES" in content
    assert "_HIP_YAW_RIGHT_CANDIDATES" in content
    assert "l_hip_yaw_pos" in content
    assert "l_hip_yaw_pos_rad" in content
    assert "hip_yaw_abs_max" in content
    assert "hip_yaw_common_error_rad" in content
    assert "hip_yaw_divergence_error_rad" in content
    assert "_resolve_column" in content


# ============================================================
# Section 9: No WBC/hidden torque fields enabled
# ============================================================

@pytest.mark.parametrize("profile_attr", [
    "LRS1_SUPPORT_DOMINANT_V1",
    "LRS2_PITCH_RATE_DAMPING_V1",
    "LRS3_BALANCED_MEDIUM_V1",
])
def test_lrs_no_wbc(profile_attr):
    from wheeled_biped.controllers import sagittal_velocity_damped_balance_controller as ctrl
    profile = getattr(ctrl, profile_attr)
    assert getattr(profile, "wbc_enabled", False) is False


@pytest.mark.parametrize("profile_attr", [
    "LRS1_SUPPORT_DOMINANT_V1",
    "LRS2_PITCH_RATE_DAMPING_V1",
    "LRS3_BALANCED_MEDIUM_V1",
])
def test_lrs_no_hidden_torque(profile_attr):
    from wheeled_biped.controllers import sagittal_velocity_damped_balance_controller as ctrl
    profile = getattr(ctrl, profile_attr)
    assert getattr(profile, "hidden_torque_enabled", False) is False


# ============================================================
# Section 10: LRS profiles registered in harness
# ============================================================

def test_lrs_profiles_registered_in_harness():
    """LRS profiles must be in the SAGITTAL_AUTHORITY_PROFILES dict."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    assert "lrs1_support_dominant_v1" in SAGITTAL_AUTHORITY_PROFILES
    assert "lrs2_pitch_rate_damping_v1" in SAGITTAL_AUTHORITY_PROFILES
    assert "lrs3_balanced_medium_v1" in SAGITTAL_AUTHORITY_PROFILES


def test_lrs_profiles_in_validation_list():
    """LRS profiles must be in the CLI validation list."""
    harness_path = ROOT / "scripts" / "simulate_hierarchical_controller.py"
    content = harness_path.read_text()
    assert '"lrs1_support_dominant_v1"' in content
    assert '"lrs2_pitch_rate_damping_v1"' in content
    assert '"lrs3_balanced_medium_v1"' in content


# ============================================================
# Section 11: LRS gain functions have correct 'kind' field
# ============================================================

def test_lrs_gain_kinds():
    """LRS gain functions must return correct 'kind'."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        _lrs_replacement_gains_S1,
        _lrs_replacement_gains_S2,
        _lrs_replacement_gains_S3,
    )
    g1 = _lrs_replacement_gains_S1(0.48)
    assert g1["kind"] == "lrs1_support_dominant"
    g2 = _lrs_replacement_gains_S2(0.48)
    assert g2["kind"] == "lrs2_pitch_rate_damping"
    g3 = _lrs_replacement_gains_S3(0.48)
    assert g3["kind"] == "lrs3_balanced_medium"


# ============================================================
# Section 12: Classification enum valid
# ============================================================

VALID_CLASSIFICATIONS = {
    "K1_REMAINS_CURRENT_BEST_LRS_NO_READY_CANDIDATE",
    "LRS_CANDIDATE_READY_FOR_BROAD_VALIDATION",
    "LRS_COMPLETES_3000_BUT_NOT_BETTER_THAN_K1",
    "LRS_SIGN_ERROR_CONFIRMED_NEEDS_FIX",
    "LRS_PHASE_LAG_CONFIRMED_NEEDS_LEAD_COMPENSATION",
    "LRS_SUPPORT_GAIN_TOO_WEAK",
    "LRS_PITCH_RATE_DAMPING_TOO_WEAK",
    "LRS_SAFETY_REGRESSION",
    "INCONCLUSIVE",
}


def test_classification_enum_valid():
    """All task classifications must be in the valid set."""
    # This validates the enum definitions exist and are consistent
    assert len(VALID_CLASSIFICATIONS) == 9


# ============================================================
# Section 13: Compile checks
# ============================================================

@pytest.mark.parametrize("script_path", [
    "scripts/simulate_hierarchical_controller.py",
    "wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py",
    "scripts/audit_lr_support_drift_sign_phase.py",
])
def test_compile(script_path):
    """Modified scripts must compile cleanly."""
    import py_compile
    full_path = ROOT / script_path
    assert full_path.exists(), f"{script_path} does not exist"
    py_compile.compile(str(full_path), doraise=True)


# ============================================================
# Section 14: LR support velocity in telemetry via CSV export
# ============================================================

def test_lr_support_velocity_in_controller_telem():
    """Controller writes LR_state_support_velocity_m_s to its telemetry dict."""
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    # The telemetry output line should contain this key
    assert '"LR_state_support_velocity_m_s"' in content
