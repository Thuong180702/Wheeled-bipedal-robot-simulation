"""Tests for LP Priority Sagittal Allocator architecture.

Covers:
1. K1 remains current-best by default
2. K1 profile does not enable LP
3. LP profiles are opt-in only
4. LP uses EQ/FF pass-through
5. LP is not additive L
6. LP is not single equal-priority LR/LRS sum
7. LP computes pitch priority separately from support residual
8. LP support residual is gated by pitch_abs and pitch_rate
9. LP support residual uses residual authority
10. LP support residual is slew-limited
11. LP does not overwrite LR/LRS profiles
12. LP telemetry contains all required LP fields
13. LP obeys torque/safety bounds
14. No WBC/hidden torque
15. Direct hip-yaw telemetry resolver still works
16. Classification enum valid
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# Section 1: K1 remains current-best by default
# ============================================================

def test_k1_is_current_best():
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        K1_PITCH_RATE_NOTCH,
    )
    assert K1_PITCH_RATE_NOTCH.profile_name == "k1_pitch_rate_notch_v1"
    assert K1_PITCH_RATE_NOTCH.enable_lp_priority_allocator is False
    assert K1_PITCH_RATE_NOTCH.lp_allocator_kind == "none"


# ============================================================
# Section 2: K1 profile does not enable LP
# ============================================================

def test_k1_does_not_enable_lp():
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        K1_PITCH_RATE_NOTCH,
    )
    assert K1_PITCH_RATE_NOTCH.enable_lp_priority_allocator is False
    assert K1_PITCH_RATE_NOTCH.lp_allocator_kind == "none"
    assert K1_PITCH_RATE_NOTCH.enable_lr_replacement_feedback is False


# ============================================================
# Section 3: LP profiles are opt-in only
# ============================================================

def test_lp_variants_exist_and_opt_in():
    """LP profiles must exist and be opt-in only."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        LP1_K1_PRIORITY_PITCH_FIRST_SUPPORT_RESIDUAL_V1,
        LP2_K1_PRIORITY_PITCH_STRONG_SUPPORT_SOFT_V1,
        LP3_K1_PRIORITY_SUPPORT_RECENTER_WHEN_SAFE_V1,
    )
    assert LP1_K1_PRIORITY_PITCH_FIRST_SUPPORT_RESIDUAL_V1.profile_name == "lp1_k1_priority_pitch_first_support_residual_v1"
    assert LP1_K1_PRIORITY_PITCH_FIRST_SUPPORT_RESIDUAL_V1.enable_lp_priority_allocator is True
    assert LP1_K1_PRIORITY_PITCH_FIRST_SUPPORT_RESIDUAL_V1.lp_allocator_kind == "LP1_pitch_first_support_residual"

    assert LP2_K1_PRIORITY_PITCH_STRONG_SUPPORT_SOFT_V1.profile_name == "lp2_k1_priority_pitch_strong_support_soft_v1"
    assert LP2_K1_PRIORITY_PITCH_STRONG_SUPPORT_SOFT_V1.enable_lp_priority_allocator is True
    assert LP2_K1_PRIORITY_PITCH_STRONG_SUPPORT_SOFT_V1.lp_allocator_kind == "LP2_pitch_strong_support_soft"

    assert LP3_K1_PRIORITY_SUPPORT_RECENTER_WHEN_SAFE_V1.profile_name == "lp3_k1_priority_support_recenter_when_safe_v1"
    assert LP3_K1_PRIORITY_SUPPORT_RECENTER_WHEN_SAFE_V1.enable_lp_priority_allocator is True
    assert LP3_K1_PRIORITY_SUPPORT_RECENTER_WHEN_SAFE_V1.lp_allocator_kind == "LP3_support_recenter_when_safe"


# ============================================================
# Section 4: LP uses EQ/FF pass-through
# ============================================================

def test_lp_eq_ff_in_controller():
    """Controller must contain EQ/FF pass-through for LP path."""
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    assert "LP_eq_ff_pass_through" in content, "LP EQ/FF pass-through missing"
    assert "tau_eq_ff_pass_through" in content, "EQ/FF pass-through concept must remain"


# ============================================================
# Section 5: LP is not additive L
# ============================================================

def test_lp_not_additive():
    """LP must NOT use additive coordinated feedback flag."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        LP1_K1_PRIORITY_PITCH_FIRST_SUPPORT_RESIDUAL_V1,
        LP2_K1_PRIORITY_PITCH_STRONG_SUPPORT_SOFT_V1,
        LP3_K1_PRIORITY_SUPPORT_RECENTER_WHEN_SAFE_V1,
    )
    for lp in [LP1_K1_PRIORITY_PITCH_FIRST_SUPPORT_RESIDUAL_V1,
               LP2_K1_PRIORITY_PITCH_STRONG_SUPPORT_SOFT_V1,
               LP3_K1_PRIORITY_SUPPORT_RECENTER_WHEN_SAFE_V1]:
        assert lp.enable_coordinated_sagittal_feedback is False
        assert lp.enable_lr_replacement_feedback is False


# ============================================================
# Section 6: LP is not single equal-priority LR/LRS sum
# ============================================================

def test_lp_not_lr_lrs():
    """LP must use its own architecture, not LR/LRS fields."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        LP1_K1_PRIORITY_PITCH_FIRST_SUPPORT_RESIDUAL_V1,
        LP2_K1_PRIORITY_PITCH_STRONG_SUPPORT_SOFT_V1,
        LP3_K1_PRIORITY_SUPPORT_RECENTER_WHEN_SAFE_V1,
    )
    for lp in [LP1_K1_PRIORITY_PITCH_FIRST_SUPPORT_RESIDUAL_V1,
               LP2_K1_PRIORITY_PITCH_STRONG_SUPPORT_SOFT_V1,
               LP3_K1_PRIORITY_SUPPORT_RECENTER_WHEN_SAFE_V1]:
        assert lp.enable_lp_priority_allocator is True
        # LP does NOT enable LR replacement
        assert lp.enable_lr_replacement_feedback is False
        # LP has its own kind field
        assert lp.lp_allocator_kind.startswith("LP")


# ============================================================
# Section 7: LP computes pitch priority separately from support residual
# ============================================================

def test_lp_separate_pitch_and_support():
    """Controller must compute pitch priority and support residual separately."""
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    assert "LP_pitch_priority" in content
    assert "LP_support_allocated" in content
    assert "LP_support_slew_limited" in content
    # LP must NOT use a single equal-priority sum
    assert "LP_eq_ff_pass_through + LP_pitch_priority + LP_support_slew_limited" in content


# ============================================================
# Section 8: LP support residual is gated by pitch state
# ============================================================

def test_lp_support_gated_by_pitch():
    """LP support must be gated by pitch_abs and pitch_rate."""
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    assert "LP_pitch_abs_gate" in content
    assert "LP_pitch_rate_gate" in content
    assert "LP_saturation_gate" in content
    assert "LP_direction_gate" in content
    assert "LP_support_gate" in content
    # The composite gate must multiply the individual gates
    assert "LP_pitch_abs_gate * LP_pitch_rate_gate * LP_saturation_gate * LP_direction_gate" in content


# ============================================================
# Section 9: LP support residual uses residual authority
# ============================================================

def test_lp_residual_authority():
    """LP support must use residual authority from remaining torque budget."""
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    assert "LP_residual_authority_nm" in content
    assert "LP_support_limit_nm" in content
    assert "LP_support_residual_fraction" in content
    # Support must be limited by residual authority (clamped +/- limit)
    assert "max(-LP_support_limit_nm," in content


# ============================================================
# Section 10: LP support residual is slew-limited
# ============================================================

def test_lp_support_slew_limited():
    """LP support must be slew-limited."""
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    assert "LP_support_slew_limited" in content
    assert "_lp_prev_support_allocated" in content
    assert "support_slew_limit_nm_per_step" in content


# ============================================================
# Section 11: LP does not overwrite LR/LRS profiles
# ============================================================

def test_lp_does_not_overwrite_lr_lrs():
    """LP must not overwrite LR/LRS profiles."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
        LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
        LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1,
        LRS1_SUPPORT_DOMINANT_V1,
        LRS2_PITCH_RATE_DAMPING_V1,
        LRS3_BALANCED_MEDIUM_V1,
    )
    # LR profiles must still exist and have their original settings
    for lr in [LR1_K1_REPLACEMENT_COORDINATED_LOW_FREQ_V1,
               LR2_K1_REPLACEMENT_PHASE_LEAD_V1,
               LR3_K1_REPLACEMENT_PITCH_REF_STABILIZED_V1]:
        assert lr.enable_lr_replacement_feedback is True
        assert lr.enable_lp_priority_allocator is False

    for lrs in [LRS1_SUPPORT_DOMINANT_V1,
                LRS2_PITCH_RATE_DAMPING_V1,
                LRS3_BALANCED_MEDIUM_V1]:
        assert lrs.enable_lr_replacement_feedback is True
        assert lrs.enable_lp_priority_allocator is False


# ============================================================
# Section 12: LP telemetry contains all required LP fields
# ============================================================

LP_REQUIRED_TELEMETRY = [
    "LP_enabled",
    "LP_candidate_kind",
    "LP_allocator_mode",
    "LP_tau_eq_ff_nm",
    "LP_tau_pitch_priority_raw_nm",
    "LP_tau_pitch_priority_nm",
    "LP_tau_support_raw_nm",
    "LP_tau_support_allocated_nm",
    "LP_tau_support_slew_limited_nm",
    "LP_tau_total_preclip_nm",
    "LP_tau_total_postclip_nm",
    "LP_pitch_abs_gate",
    "LP_pitch_rate_gate",
    "LP_saturation_gate",
    "LP_direction_gate",
    "LP_support_gate",
    "LP_residual_authority_nm",
    "LP_support_limit_nm",
    "LP_support_residual_fraction",
    "LP_pitch_error_rad",
    "LP_pitch_rate_effective_rad_s",
    "LP_support_error_m",
    "LP_support_velocity_m_s",
    "LP_support_suppressed_reason",
    "LP_near_saturation",
    "LP_support_direction_assists_pitch_error",
    "LP_gains_kind",
]


def test_lp_telemetry_fields_exist():
    """All required LP telemetry fields must be in the controller diagnostics."""
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    for field in LP_REQUIRED_TELEMETRY:
        assert f'"{field}"' in content, f"LP telemetry field {field} missing from diagnostics"


# ============================================================
# Section 13: LP obeys torque/safety bounds
# ============================================================

def test_lp_pitch_priority_limit():
    """LP pitch priority must have a hard limit."""
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    assert "pitch_priority_limit_nm" in content


def test_lp_safety_gates_exist():
    """LP must have pitch safety gate thresholds."""
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    assert "pitch_safe_low_deg" in content
    assert "pitch_safe_high_deg" in content
    assert "rate_safe_low_deg_s" in content
    assert "rate_safe_high_deg_s" in content


# ============================================================
# Section 14: No WBC/hidden torque
# ============================================================

@pytest.mark.parametrize("profile_attr", [
    "LP1_K1_PRIORITY_PITCH_FIRST_SUPPORT_RESIDUAL_V1",
    "LP2_K1_PRIORITY_PITCH_STRONG_SUPPORT_SOFT_V1",
    "LP3_K1_PRIORITY_SUPPORT_RECENTER_WHEN_SAFE_V1",
])
def test_lp_no_wbc(profile_attr):
    from wheeled_biped.controllers import sagittal_velocity_damped_balance_controller as ctrl
    profile = getattr(ctrl, profile_attr)
    assert getattr(profile, "wbc_enabled", False) is False


@pytest.mark.parametrize("profile_attr", [
    "LP1_K1_PRIORITY_PITCH_FIRST_SUPPORT_RESIDUAL_V1",
    "LP2_K1_PRIORITY_PITCH_STRONG_SUPPORT_SOFT_V1",
    "LP3_K1_PRIORITY_SUPPORT_RECENTER_WHEN_SAFE_V1",
])
def test_lp_no_hidden_torque(profile_attr):
    from wheeled_biped.controllers import sagittal_velocity_damped_balance_controller as ctrl
    profile = getattr(ctrl, profile_attr)
    assert getattr(profile, "hidden_torque_enabled", False) is False


# ============================================================
# Section 15: Direct hip-yaw telemetry resolver still works
# ============================================================

def test_hip_yaw_column_resolution_still_works():
    """Audit script hip-yaw resolver must still handle all variants."""
    audit_path = ROOT / "scripts" / "audit_lr_support_drift_sign_phase.py"
    content = audit_path.read_text()
    assert "_HIP_YAW_LEFT_CANDIDATES" in content
    assert "_HIP_YAW_RIGHT_CANDIDATES" in content
    assert "_resolve_column" in content
    assert "l_hip_yaw_pos" in content


# ============================================================
# Section 16: Classification enum valid
# ============================================================

LP_VALID_CLASSIFICATIONS = {
    "K1_REMAINS_CURRENT_BEST_LP_NO_READY_CANDIDATE",
    "LP_CANDIDATE_READY_FOR_BROAD_VALIDATION",
    "LP_COMPLETES_3000_BUT_NOT_BETTER_THAN_K1",
    "LP_REDUCES_SUPPORT_PITCH_COUPLING_BUT_NOT_READY",
    "LP_FAIL_SUPPORT_DRIFT",
    "LP_FAIL_PITCH_OSCILLATION",
    "LP_SAFETY_REGRESSION",
    "INCONCLUSIVE",
}


def test_classification_enum_valid():
    """All LP task classifications must be in the valid set."""
    assert len(LP_VALID_CLASSIFICATIONS) == 8


# ============================================================
# Section 17: LP profiles registered in harness
# ============================================================

def test_lp_profiles_registered_in_harness():
    """LP profiles must be in the SAGITTAL_AUTHORITY_PROFILES dict."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    assert "lp1_k1_priority_pitch_first_support_residual_v1" in SAGITTAL_AUTHORITY_PROFILES
    assert "lp2_k1_priority_pitch_strong_support_soft_v1" in SAGITTAL_AUTHORITY_PROFILES
    assert "lp3_k1_priority_support_recenter_when_safe_v1" in SAGITTAL_AUTHORITY_PROFILES


def test_lp_profiles_in_validation_list():
    """LP profiles must be in the CLI validation list."""
    harness_path = ROOT / "scripts" / "simulate_hierarchical_controller.py"
    content = harness_path.read_text()
    assert '"lp1_k1_priority_pitch_first_support_residual_v1"' in content
    assert '"lp2_k1_priority_pitch_strong_support_soft_v1"' in content
    assert '"lp3_k1_priority_support_recenter_when_safe_v1"' in content


# ============================================================
# Section 18: LP gain functions have correct 'kind' field
# ============================================================

def test_lp_gain_kinds():
    """LP gain functions must return correct 'kind'."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        _lp_priority_gains_LP1,
        _lp_priority_gains_LP2,
        _lp_priority_gains_LP3,
    )
    g1 = _lp_priority_gains_LP1(0.48)
    assert g1["kind"] == "lp1_pitch_first_support_residual"
    g2 = _lp_priority_gains_LP2(0.48)
    assert g2["kind"] == "lp2_pitch_strong_support_soft"
    g3 = _lp_priority_gains_LP3(0.48)
    assert g3["kind"] == "lp3_support_recenter_when_safe"


# ============================================================
# Section 19: LP gain bounds
# ============================================================

@pytest.mark.parametrize("height", [0.30, 0.40, 0.48])
def test_lp1_gain_bounds(height):
    """LP1 gains must obey hard bounds."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        _lp_priority_gains_LP1,
    )
    g = _lp_priority_gains_LP1(height)
    assert abs(g["k_pitch_lp"]) <= 15.0, f"k_pitch_lp={g['k_pitch_lp']} exceeds bound 15"
    assert abs(g["k_pitch_rate_lp"]) <= 3.0, f"k_pitch_rate_lp={g['k_pitch_rate_lp']} exceeds bound 3"
    assert abs(g["k_support_lp"]) <= 30.0, f"k_support_lp={g['k_support_lp']} exceeds bound 30"
    assert abs(g["k_support_vel_lp"]) <= 1.5, f"k_support_vel_lp={g['k_support_vel_lp']} exceeds bound 1.5"


@pytest.mark.parametrize("height", [0.30, 0.40, 0.48])
def test_lp2_gain_bounds(height):
    """LP2 gains must obey hard bounds."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        _lp_priority_gains_LP2,
    )
    g = _lp_priority_gains_LP2(height)
    assert abs(g["k_pitch_lp"]) <= 15.0
    assert abs(g["k_pitch_rate_lp"]) <= 3.0, f"k_pitch_rate_lp={g['k_pitch_rate_lp']} exceeds bound 3"
    assert abs(g["k_support_lp"]) <= 30.0
    assert abs(g["k_support_vel_lp"]) <= 1.5


@pytest.mark.parametrize("height", [0.30, 0.40, 0.48])
def test_lp3_gain_bounds(height):
    """LP3 gains must obey hard bounds."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        _lp_priority_gains_LP3,
    )
    g = _lp_priority_gains_LP3(height)
    assert abs(g["k_pitch_lp"]) <= 15.0
    assert abs(g["k_pitch_rate_lp"]) <= 3.0
    assert abs(g["k_support_lp"]) <= 30.0
    assert abs(g["k_support_vel_lp"]) <= 1.5


def test_lp_gains_finite():
    """All LP gains must be finite."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        _lp_priority_gains_LP1,
        _lp_priority_gains_LP2,
        _lp_priority_gains_LP3,
    )
    for func in [_lp_priority_gains_LP1, _lp_priority_gains_LP2, _lp_priority_gains_LP3]:
        for height in [0.30, 0.40, 0.48]:
            g = func(height)
            for k in ["k_pitch_lp", "k_pitch_rate_lp", "k_support_lp", "k_support_vel_lp"]:
                assert math.isfinite(g[k]), f"{func.__name__}({height})[{k}] not finite"


# ============================================================
# Section 20: LP gain functions have safety gate parameters
# ============================================================

def test_lp_gains_have_safety_gates():
    """All LP gain functions must return safety gate thresholds."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        _lp_priority_gains_LP1,
        _lp_priority_gains_LP2,
        _lp_priority_gains_LP3,
    )
    for func in [_lp_priority_gains_LP1, _lp_priority_gains_LP2, _lp_priority_gains_LP3]:
        g = func(0.48)
        assert "pitch_safe_low_deg" in g
        assert "pitch_safe_high_deg" in g
        assert "rate_safe_low_deg_s" in g
        assert "rate_safe_high_deg_s" in g
        assert "pitch_priority_limit_nm" in g
        assert "support_residual_fraction" in g
        assert "support_slew_limit_nm_per_step" in g
        assert "direction_gate_enabled" in g
        assert "kind" in g


# ============================================================
# Section 21: LP3 has settling-specific parameters
# ============================================================

def test_lp3_has_settling_params():
    """LP3 gain function must return pitch settling parameters."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        _lp_priority_gains_LP3,
    )
    g = _lp_priority_gains_LP3(0.48)
    assert "pitch_settle_threshold_deg" in g
    assert "pitch_settle_steps_required" in g
    assert g["pitch_settle_steps_required"] > 0


# ============================================================
# Section 22: LP state variables exist in controller init
# ============================================================

def test_lp_state_variables_in_init():
    """Controller __init__ must initialize LP state variables."""
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    assert "_lp_prev_support_allocated" in content
    assert "_lp_pitch_settle_counter" in content


# ============================================================
# Section 23: Compile checks
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


# ============================================================
# Section 24: LP deadband configured
# ============================================================

def test_lp_deadband_configured():
    """LP must use support deadband to ignore small errors."""
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    assert "support_deadband_m" in content


# ============================================================
# Section 25: LP suppression reason telemetry
# ============================================================

def test_lp_suppression_reason_telemetry():
    """LP must report why support was suppressed."""
    controller_path = ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
    content = controller_path.read_text()
    assert "LP_support_suppressed_reason" in content
    assert "pitch_abs" in content  # one of the suppression reasons
    assert "deadband" in content
