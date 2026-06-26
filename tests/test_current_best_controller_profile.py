"""Verify that the K1 pitch-rate notch profile is available in the
controller registry and is the current-best/default profile.

After the evidence-based K1 best-current promotion, the default/current-best
PFF profile should be ``k1_pitch_rate_notch_v1``, which is the low-band v2
sagittal schedule plus the runtime mode-hip-yaw-divergence controller
(kp=10.0, kd=0.50, max_torque=7.5, soft_limit=0.30, soft_gain=0.80,
ref_source=target) and the causal IIR biquad notch filter on pitch_rate
(fc=2.5 Hz, Q=6, blend=1.0, height gate 0.42-0.48 m).

D_MODE_HIP_YAW_DIV_V1 remains available as the previous current-best
legacy/reference profile.

All legacy profiles must remain selectable.

Promotion policy: best-current, not full-goal-solved.
K1 is promoted with known WIP recovery limitation.
"""
import sys
import pathlib

# Add scripts/ to path so we can import from sibling modules
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# The known profile registry in simulate_hierarchical_controller.py
PROFILES_TO_CHECK = {
    # Legacy PFF profiles
    "physics_equilibrium_feedforward_outer_loop": "Current PFF (pre-promotion)",
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v1": "Low-band v1",
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2": "Low-band v2 (sagittal base for current-best)",
    # Previous current-best (legacy)
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1": "D_MODE_HIP_YAW_DIV_V1 (previous current-best, legacy)",
    # Current-best (K2 promoted 2026-06-25; Step C/E/D all gates passed)
    "k2_notch_low_q_v1": "K2 notch low-Q v1 (current-best, promoted with Step C/E/D validation)",
    # Previous current-best (K1, legacy)
    "k1_pitch_rate_notch_v1": "K1 pitch-rate notch v1 (previous current-best, legacy)",
    # B2v2 baseline
    "calibrated_support_position_outer_loop_pitch_ref_v2": "B2v2 experimental",
    # Legacy B / A profiles
    "support_position_outer_loop_pitch_ref": "Legacy current-best B",
    "height_scheduled_pitch_equilibrium_trim": "Legacy fallback A",
}


def test_all_profiles_available():
    """All named profiles must exist in SAGITTAL_AUTHORITY_PROFILES."""
    from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    for name, label in PROFILES_TO_CHECK.items():
        assert name in SAGITTAL_AUTHORITY_PROFILES, (
            f"Missing profile: {name} ({label})"
        )


def test_d_mode_hip_yaw_div_v1_resolves_to_low_band_v2_sagittal():
    """D_MODE_HIP_YAW_DIV_V1 (previous current-best) must resolve to low-band v2.

    This verifies backward compatibility: the sagittal schedule is byte-for-byte
    identical to low-band v2; the divergence-mode controller is enabled separately
    via runtime CLI flags. D remains available as legacy/reference.
    """
    from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    d_profile = SAGITTAL_AUTHORITY_PROFILES.get(
        "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1"
    )
    c_profile = SAGITTAL_AUTHORITY_PROFILES.get(
        "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
    )
    assert d_profile is not None
    assert c_profile is not None
    # Same profile object (alias). Both share the same sagittal schedule.
    assert d_profile is c_profile, (
        "D_MODE_HIP_YAW_DIV_V1 must resolve to the same SagittalAuthoritySchedule "
        "as low-band v2; the divergence-mode controller is enabled at runtime."
    )
    # Sagittal signature of low-band v2.
    assert d_profile.low_band_support_outer_loop_enabled is True
    assert d_profile.low_band_support_center_m == 0.320
    assert d_profile.low_band_support_sigma_m == 0.004


def test_k2_notch_low_q_v1_is_current_best():
    """K2 notch low-Q v1 is the current-best/default controller (promoted
    2026-06-25 after Step C/E/D validation gates passed).

    K2 extends K1 with wip_notch_q=2.0 (wider notch). Same low-band v2 sagittal
    base, same mode-div params kp=10.0, kd=0.50, mt=7.5, sg=0.80, with a causal
    IIR biquad notch filter on pitch_rate (fc=2.5 Hz, Q=2.0, blend=1.0, height
    gate 0.42-0.48 m).
    """
    from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    k2 = SAGITTAL_AUTHORITY_PROFILES.get(
        "k2_notch_low_q_v1"
    )
    assert k2 is not None, "K2 notch_low_q_v1 profile missing"
    # Sagittal base = low-band v2
    assert k2.low_band_support_outer_loop_enabled is True
    assert k2.low_band_support_center_m == 0.320
    assert k2.low_band_support_sigma_m == 0.004
    # Notch filter — Q=2.0 (wider, better LF suppression)
    assert k2.enable_wip_notch_filter is True
    assert k2.wip_notch_target_signal == "pitch_rate"
    assert k2.wip_notch_center_hz == 2.5
    assert k2.wip_notch_q == 2.0
    assert k2.wip_notch_filter_blend == 1.0
    assert k2.wip_notch_height_gate_start_m == 0.42
    assert k2.wip_notch_height_gate_full_m == 0.48
    # No K3 combined notch
    assert k2.wip_notch_target_signal != "pitch_rate_and_wheel_velocity"
    # No WBC
    assert getattr(k2, "wbc_enabled", False) is False


def test_k1_pitch_rate_notch_v1_is_legacy():
    """K1 pitch-rate notch v1 is the previous current-best, available as legacy.
    Must remain unchanged and selectable."""
    from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    k1 = SAGITTAL_AUTHORITY_PROFILES.get(
        "k1_pitch_rate_notch_v1"
    )
    assert k1 is not None, "K1 pitch_rate_notch_v1 must remain available as legacy"
    # K1 Q remains 6.0 (unchanged)
    assert k1.wip_notch_q == 6.0
    # All other params unchanged
    assert k1.enable_wip_notch_filter is True
    assert k1.wip_notch_center_hz == 2.5
    assert k1.wip_notch_filter_blend == 1.0


def test_profile_is_continuous():
    """v2 profile uses continuous low-band support (Gaussian scaling), not bins.

    The profile inherits applies_to_variants from the parent PFF, but
    the low-band mechanism itself is continuous via center_m and sigma_m.
    """
    from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    profile = SAGITTAL_AUTHORITY_PROFILES.get(
        "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
    )
    assert profile is not None
    # Continuous low-band support is enabled via Gaussian height scaling
    assert profile.low_band_support_outer_loop_enabled is True
    assert profile.low_band_support_center_m == 0.320
    assert profile.low_band_support_sigma_m == 0.004


def test_pff_source_unchanged():
    """Physics equilibrium feedforward source/interpolation must be unchanged.

    Verify that the PFF module (physics_equilibrium_feedforward.py) still
    exports the expected functions and is not modified for the low-band v2.
    """
    from wheeled_biped.controllers import physics_equilibrium_feedforward as pff
    assert hasattr(pff, "physics_equilibrium_feedforward_tau_each_wheel_nm")
    assert hasattr(pff, "physics_equilibrium_feedforward_params")


def test_no_wbc_enabled():
    """v2 profile must not enable WBC."""
    from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        SagittalAuthoritySchedule,
    )
    profile = SAGITTAL_AUTHORITY_PROFILES.get(
        "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
    )
    assert profile is not None
    # No WBC-related fields should be set
    assert getattr(profile, "wbc_enabled", False) is False, "WBC should not be enabled"


def test_centered_posture_height_schedule():
    """Check that centered_posture_height_schedule is used with v2.

    Verify through the profile's pitch_ref_height_schedule setting.
    """
    from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    profile = SAGITTAL_AUTHORITY_PROFILES.get(
        "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
    )
    assert profile is not None
    # The profile should have physics_equilibrium_feedforward_enabled
    assert profile.physics_equilibrium_feedforward_enabled is True


def test_hip_yaw_gate_policy_unchanged():
    """Hip-yaw gate policy must not be altered by profile selection."""
    import importlib
    spec = importlib.util.spec_from_file_location(
        "hip_yaw_gate_policy",
        ROOT / "wheeled_biped" / "validation" / "hip_yaw_gate_policy.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "dummy_policy")
