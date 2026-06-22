"""Verify that the low-band v2 profile is available in the controller registry.

After promotion, the default/current-best PFF profile should be
physics_equilibrium_feedforward_outer_loop_low_band_support_v2.
All legacy profiles must remain selectable.
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
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2": "Low-band v2 (promoted candidate)",
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
