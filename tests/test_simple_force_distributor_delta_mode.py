"""Unit tests for delta distribution mode in SimpleForceDistributor.

Tests verify that delta mode correctly handles correction wrenches with Fz ≈ 0,
enabling roll correction through vertical force asymmetry.
"""

import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.simple_force_distributor import SimpleForceDistributor


@pytest.fixture
def distributor():
    """Create force distributor with standard config."""
    return SimpleForceDistributor(
        tau_hip_roll_max=15.0,
        max_force_asymmetry=40.0,
        min_wheel_force=10.0,
    )


@pytest.fixture
def wheel_positions():
    """Standard wheel positions relative to CoM."""
    wheel_pos_left = jnp.array([0.0, 0.15, 0.0])  # 0.3m track width
    wheel_pos_right = jnp.array([0.0, -0.15, 0.0])
    return wheel_pos_left, wheel_pos_right


def test_delta_mode_zero_correction_gives_zero_force(distributor, wheel_positions):
    """Zero correction wrench in delta mode produces zero forces."""
    wheel_pos_left, wheel_pos_right = wheel_positions

    wrench = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        desired_wrench=wrench,
        left_contact=True,
        right_contact=True,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        distribution_mode="delta",
    )

    assert jnp.allclose(f_left, jnp.zeros(3)), "Zero correction should produce zero left force"
    assert jnp.allclose(f_right, jnp.zeros(3)), "Zero correction should produce zero right force"
    assert jnp.allclose(tau_hip_roll, jnp.zeros(2)), "Zero correction should produce zero hip-roll torque"
    assert diagnostics["reason"] == "zero_correction_delta"


def test_delta_mode_my_with_zero_fz_produces_force_asymmetry(distributor, wheel_positions):
    """Nonzero My with Fz=0 in delta mode produces vertical force asymmetry."""
    wheel_pos_left, wheel_pos_right = wheel_positions

    # Test positive My
    wrench_pos = jnp.array([0.0, 0.0, 0.0, 0.0, +10.0, 0.0])

    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        desired_wrench=wrench_pos,
        left_contact=True,
        right_contact=True,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        distribution_mode="delta",
    )

    fz_diff_pos = float(f_left[2] - f_right[2])

    assert abs(fz_diff_pos) > 0.1, f"My=+10 Nm should produce nonzero fz_diff, got {fz_diff_pos:.3f}"
    assert diagnostics["reason"] == "delta_double_contact"

    # Test negative My
    wrench_neg = jnp.array([0.0, 0.0, 0.0, 0.0, -10.0, 0.0])

    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        desired_wrench=wrench_neg,
        left_contact=True,
        right_contact=True,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        distribution_mode="delta",
    )

    fz_diff_neg = float(f_left[2] - f_right[2])

    assert abs(fz_diff_neg) > 0.1, f"My=-10 Nm should produce nonzero fz_diff, got {fz_diff_neg:.3f}"
    assert fz_diff_pos * fz_diff_neg < 0, "Opposite My should produce opposite fz_diff signs"


def test_delta_mode_my_sign_consistency(distributor, wheel_positions):
    """Achieved My has the same sign as requested My in delta mode."""
    wheel_pos_left, wheel_pos_right = wheel_positions

    test_cases = [
        (+5.0, "positive"),
        (-5.0, "negative"),
        (+10.0, "positive"),
        (-10.0, "negative"),
    ]

    for my_cmd, expected_sign in test_cases:
        wrench = jnp.array([0.0, 0.0, 0.0, 0.0, my_cmd, 0.0])

        f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
            desired_wrench=wrench,
            left_contact=True,
            right_contact=True,
            wheel_pos_left=wheel_pos_left,
            wheel_pos_right=wheel_pos_right,
            hip_roll_authority_scale=1.0,
            distribution_mode="delta",
        )

        # Compute achieved My from force asymmetry
        # My = y_l * fz_l + y_r * fz_r (lateral positions generate roll moment)
        y_l = float(wheel_pos_left[1])
        y_r = float(wheel_pos_right[1])
        fz_l = float(f_left[2])
        fz_r = float(f_right[2])
        achieved_my = y_l * fz_l + y_r * fz_r

        if expected_sign == "positive":
            assert achieved_my > 0, f"My={my_cmd} should produce positive achieved_my, got {achieved_my:.3f}"
        else:
            assert achieved_my < 0, f"My={my_cmd} should produce negative achieved_my, got {achieved_my:.3f}"

        # Check sign consistency
        assert my_cmd * achieved_my > 0, f"My={my_cmd} and achieved_my={achieved_my:.3f} have opposite signs"


def test_delta_mode_single_contact_zero_force_on_non_contact_wheel(distributor, wheel_positions):
    """Non-contact wheel receives zero force in delta mode."""
    wheel_pos_left, wheel_pos_right = wheel_positions

    wrench = jnp.array([0.0, 0.0, 10.0, 0.0, 5.0, 0.0])

    # Left contact only
    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        desired_wrench=wrench,
        left_contact=True,
        right_contact=False,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        distribution_mode="delta",
    )

    assert jnp.allclose(f_right, jnp.zeros(3)), "Non-contact right wheel should have zero force"
    assert not jnp.allclose(f_left, jnp.zeros(3)), "Contact left wheel should have nonzero force"
    assert diagnostics["reason"] == "delta_single_contact"

    # Right contact only
    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        desired_wrench=wrench,
        left_contact=False,
        right_contact=True,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        distribution_mode="delta",
    )

    assert jnp.allclose(f_left, jnp.zeros(3)), "Non-contact left wheel should have zero force"
    assert not jnp.allclose(f_right, jnp.zeros(3)), "Contact right wheel should have nonzero force"
    assert diagnostics["reason"] == "delta_single_contact"


def test_delta_mode_no_contact_preserves_hip_roll_torques(distributor, wheel_positions):
    """No contact preserves hip-roll torques but zeros wheel forces in delta mode."""
    wheel_pos_left, wheel_pos_right = wheel_positions

    wrench = jnp.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0])

    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        desired_wrench=wrench,
        left_contact=False,
        right_contact=False,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        distribution_mode="delta",
    )

    assert jnp.allclose(f_left, jnp.zeros(3)), "No contact should produce zero left force"
    assert jnp.allclose(f_right, jnp.zeros(3)), "No contact should produce zero right force"
    assert not jnp.allclose(tau_hip_roll, jnp.zeros(2)), "Hip-roll torques should be preserved"
    assert diagnostics["reason"] == "delta_no_contact"


def test_absolute_mode_behavior_unchanged(distributor, wheel_positions):
    """Absolute mode behavior remains unchanged after adding delta mode."""
    wheel_pos_left, wheel_pos_right = wheel_positions

    # Test with typical absolute wrench (gravity compensation + correction)
    robot_mass = 8.1
    g = 9.81
    wrench = jnp.array([0.0, 0.0, robot_mass * g, 0.0, 5.0, 0.0])

    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        desired_wrench=wrench,
        left_contact=True,
        right_contact=True,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        distribution_mode="absolute",
    )

    # Check that forces are positive (absolute mode)
    assert float(f_left[2]) > 0, "Absolute mode should produce positive left force"
    assert float(f_right[2]) > 0, "Absolute mode should produce positive right force"

    # Check that total vertical force matches input
    total_fz = float(f_left[2] + f_right[2])
    assert abs(total_fz - robot_mass * g) < 1.0, f"Total Fz should match input, got {total_fz:.1f} vs {robot_mass * g:.1f}"

    # Check that force asymmetry is limited by liftoff threshold
    fz_diff = abs(float(f_left[2] - f_right[2]))
    assert fz_diff <= distributor.max_force_asymmetry, f"Force asymmetry {fz_diff:.1f} exceeds max {distributor.max_force_asymmetry}"


def test_delta_mode_clips_excessive_delta_forces(distributor, wheel_positions):
    """Delta mode clips excessive delta forces by max_delta_fz."""
    wheel_pos_left, wheel_pos_right = wheel_positions

    # Request very large My that would produce excessive delta forces
    wrench = jnp.array([0.0, 0.0, 0.0, 0.0, 100.0, 0.0])

    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        desired_wrench=wrench,
        left_contact=True,
        right_contact=True,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        distribution_mode="delta",
        max_delta_fz=30.0,
    )

    # Check that delta forces are clipped
    assert abs(float(f_left[2])) <= 30.0, f"Left delta force should be clipped to 30 N, got {f_left[2]:.1f}"
    assert abs(float(f_right[2])) <= 30.0, f"Right delta force should be clipped to 30 N, got {f_right[2]:.1f}"


def test_delta_mode_allows_negative_delta_forces(distributor, wheel_positions):
    """Delta mode allows negative delta forces (reducing baseline contact load)."""
    wheel_pos_left, wheel_pos_right = wheel_positions

    # Request My that produces negative delta force on one side
    wrench = jnp.array([0.0, 0.0, 0.0, 0.0, -10.0, 0.0])

    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        desired_wrench=wrench,
        left_contact=True,
        right_contact=True,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        distribution_mode="delta",
    )

    # At least one side should have negative delta force
    has_negative = float(f_left[2]) < 0 or float(f_right[2]) < 0
    assert has_negative, "Delta mode should allow negative delta forces"
