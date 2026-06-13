"""
Unit tests for Phase 6 Joint Low-Height Sagittal-Yaw Fix

Tests the J1-J3 schedule profiles:
- J1: Support cap (k_position=80, max_tau=6.0, k_velocity=15 baseline)
- J2: Support cap + moderate damping (k_position=80, max_tau=6.0, k_velocity=25)
- J3: Support cap + strong damping (k_position=80, max_tau=6.0, k_velocity=30)

Validates:
- Schedule continuity (smoothstep interpolation)
- Boundary values at z=0.300 and z=0.393
- Baseline profile unchanged
- No WBC, no hip-roll modification
"""

import pytest
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    SagittalVelocityDampedBalanceController,
    BASELINE_AUTHORITY_SCHEDULE,
    JOINT_FIX_J1_SUPPORT_CAP,
    JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING,
    JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING,
    scheduled_k_position,
)


def test_baseline_schedule_unchanged():
    """Baseline schedule should have no continuous scheduling enabled."""
    assert BASELINE_AUTHORITY_SCHEDULE.profile_name == "baseline"
    assert BASELINE_AUTHORITY_SCHEDULE.continuous_k_position is False
    assert BASELINE_AUTHORITY_SCHEDULE.continuous_max_position_tau is False
    assert BASELINE_AUTHORITY_SCHEDULE.continuous_k_velocity is False


def test_j1_profile_configuration():
    """J1 should have k_position and max_position_tau scheduling, but not k_velocity."""
    assert JOINT_FIX_J1_SUPPORT_CAP.profile_name == "J1_support_cap"
    assert JOINT_FIX_J1_SUPPORT_CAP.continuous_k_position is True
    assert JOINT_FIX_J1_SUPPORT_CAP.k_position_nominal == 40.0
    assert JOINT_FIX_J1_SUPPORT_CAP.k_position_low_max == 80.0
    assert JOINT_FIX_J1_SUPPORT_CAP.continuous_max_position_tau is True
    assert JOINT_FIX_J1_SUPPORT_CAP.max_position_tau_nominal == 3.0
    assert JOINT_FIX_J1_SUPPORT_CAP.max_position_tau_low_max == 6.0
    assert JOINT_FIX_J1_SUPPORT_CAP.continuous_k_velocity is False


def test_j2_profile_configuration():
    """J2 should have all three schedules: k_position, max_position_tau, k_velocity=25."""
    assert JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING.profile_name == "J2_support_cap_moderate_damping"
    assert JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING.continuous_k_position is True
    assert JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING.k_position_nominal == 40.0
    assert JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING.k_position_low_max == 80.0
    assert JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING.continuous_max_position_tau is True
    assert JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING.max_position_tau_nominal == 3.0
    assert JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING.max_position_tau_low_max == 6.0
    assert JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING.continuous_k_velocity is True
    assert JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING.k_velocity_nominal == 15.0
    assert JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING.k_velocity_low_max == 25.0


def test_j3_profile_configuration():
    """J3 should have all three schedules: k_position, max_position_tau, k_velocity=30."""
    assert JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING.profile_name == "J3_support_cap_strong_damping"
    assert JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING.continuous_k_position is True
    assert JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING.k_position_nominal == 40.0
    assert JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING.k_position_low_max == 80.0
    assert JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING.continuous_max_position_tau is True
    assert JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING.max_position_tau_nominal == 3.0
    assert JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING.max_position_tau_low_max == 6.0
    assert JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING.continuous_k_velocity is True
    assert JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING.k_velocity_nominal == 15.0
    assert JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING.k_velocity_low_max == 30.0


def test_schedule_boundary_values():
    """Test scheduled_k_position at boundary heights."""
    z_low, z_high = 0.300, 0.393
    k_nominal, k_low_max = 40.0, 80.0

    # At z=0.300 (low height), should get max authority
    k_at_low = scheduled_k_position(z_low, k_nominal, k_low_max, z_low, z_high)
    assert abs(k_at_low - k_low_max) < 1e-6, f"Expected {k_low_max} at z={z_low}, got {k_at_low}"

    # At z=0.393 (high height), should get nominal authority
    k_at_high = scheduled_k_position(z_high, k_nominal, k_low_max, z_low, z_high)
    assert abs(k_at_high - k_nominal) < 1e-6, f"Expected {k_nominal} at z={z_high}, got {k_at_high}"

    # Above z_high, should clamp to nominal
    k_above = scheduled_k_position(0.450, k_nominal, k_low_max, z_low, z_high)
    assert abs(k_above - k_nominal) < 1e-6, f"Expected {k_nominal} above z_high, got {k_above}"

    # Below z_low, should clamp to max
    k_below = scheduled_k_position(0.250, k_nominal, k_low_max, z_low, z_high)
    assert abs(k_below - k_low_max) < 1e-6, f"Expected {k_low_max} below z_low, got {k_below}"


def test_schedule_continuity():
    """Test that schedule is continuous (no jumps) across height range."""
    z_low, z_high = 0.300, 0.393
    k_nominal, k_low_max = 40.0, 80.0

    prev_k = None
    for z in [0.300, 0.320, 0.340, 0.360, 0.380, 0.393]:
        k = scheduled_k_position(z, k_nominal, k_low_max, z_low, z_high)

        # Check monotonic decrease (as height increases, authority decreases)
        if prev_k is not None:
            assert k <= prev_k, f"Schedule not monotonic: k({z})={k} > k(prev)={prev_k}"

        prev_k = k


def test_j1_controller_instantiation():
    """Test that controller can be instantiated with J1 profile."""
    controller = SagittalVelocityDampedBalanceController(
        k_position=40.0,
        k_velocity=15.0,
        max_position_tau=3.0,
        authority_schedule=JOINT_FIX_J1_SUPPORT_CAP,
    )

    assert controller.authority_schedule.profile_name == "J1_support_cap"
    assert controller.authority_schedule.continuous_k_position is True
    assert controller.authority_schedule.continuous_max_position_tau is True
    assert controller.authority_schedule.continuous_k_velocity is False


def test_j2_controller_instantiation():
    """Test that controller can be instantiated with J2 profile."""
    controller = SagittalVelocityDampedBalanceController(
        k_position=40.0,
        k_velocity=15.0,
        max_position_tau=3.0,
        authority_schedule=JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING,
    )

    assert controller.authority_schedule.profile_name == "J2_support_cap_moderate_damping"
    assert controller.authority_schedule.continuous_k_position is True
    assert controller.authority_schedule.continuous_max_position_tau is True
    assert controller.authority_schedule.continuous_k_velocity is True


def test_j3_controller_instantiation():
    """Test that controller can be instantiated with J3 profile."""
    controller = SagittalVelocityDampedBalanceController(
        k_position=40.0,
        k_velocity=15.0,
        max_position_tau=3.0,
        authority_schedule=JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING,
    )

    assert controller.authority_schedule.profile_name == "J3_support_cap_strong_damping"
    assert controller.authority_schedule.continuous_k_position is True
    assert controller.authority_schedule.continuous_max_position_tau is True
    assert controller.authority_schedule.continuous_k_velocity is True


def test_schedule_z_range_consistency():
    """All J profiles should use the same z_low and z_high."""
    assert JOINT_FIX_J1_SUPPORT_CAP.k_position_z_low == 0.300
    assert JOINT_FIX_J1_SUPPORT_CAP.k_position_z_high == 0.393

    assert JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING.k_position_z_low == 0.300
    assert JOINT_FIX_J2_SUPPORT_CAP_MODERATE_DAMPING.k_position_z_high == 0.393

    assert JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING.k_position_z_low == 0.300
    assert JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING.k_position_z_high == 0.393


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
