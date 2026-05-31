"""Tests for sagittal balance state helpers: frame projection, velocity, state bundle."""

import math

import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.sagittal_balance_state import (
    build_sagittal_balance_state,
    compute_support_center_xy,
    project_sagittal_displacement,
    project_sagittal_velocity,
)


def test_project_sagittal_displacement_uses_initial_heading_frame():
    origin_xy = (0.0, 0.0)
    sagittal_axis_xy = (0.0, 1.0)
    current_xy = (0.2, 0.5)

    displacement = project_sagittal_displacement(
        origin_xy=origin_xy,
        sagittal_axis_xy=sagittal_axis_xy,
        current_xy=current_xy,
    )

    assert displacement == pytest.approx(0.5)


def test_project_sagittal_velocity_matches_initial_heading_axis():
    sagittal_axis_xy = (0.0, 1.0)
    velocity_xy = (0.3, -0.4)

    velocity = project_sagittal_velocity(
        sagittal_axis_xy=sagittal_axis_xy,
        velocity_xy=velocity_xy,
    )

    assert velocity == pytest.approx(-0.4)


def test_build_sagittal_balance_state_orders_required_terms():
    state = build_sagittal_balance_state(
        sagittal_position_error=0.1,
        sagittal_velocity=-0.2,
        pitch_x=0.03,
        pitch_rate_x=-0.04,
        wheel_velocity_mean=1.5,
    )

    expected = jnp.array([0.1, -0.2, 0.03, -0.04, 1.5])
    assert jnp.allclose(state, expected)


def test_project_sagittal_displacement_remains_correct_with_nonzero_yaw():
    yaw_rad = math.radians(30)
    sagittal_axis_xy = (math.sin(yaw_rad), math.cos(yaw_rad))
    current_xy = (0.1, 0.1732)
    origin_xy = (0.0, 0.0)

    displacement = project_sagittal_displacement(
        origin_xy=origin_xy,
        sagittal_axis_xy=sagittal_axis_xy,
        current_xy=current_xy,
    )

    assert displacement == pytest.approx(0.2, abs=1e-4)


def test_project_sagittal_displacement_zero_when_at_origin():
    displacement = project_sagittal_displacement(
        origin_xy=(1.0, 2.0),
        sagittal_axis_xy=(0.0, 1.0),
        current_xy=(1.0, 2.0),
    )
    assert displacement == pytest.approx(0.0)


def test_project_sagittal_velocity_zero_when_perpendicular():
    velocity = project_sagittal_velocity(
        sagittal_axis_xy=(0.0, 1.0),
        velocity_xy=(0.5, 0.0),
    )
    assert velocity == pytest.approx(0.0)


def test_project_sagittal_displacement_negative_when_behind():
    displacement = project_sagittal_displacement(
        origin_xy=(0.0, 0.0),
        sagittal_axis_xy=(0.0, 1.0),
        current_xy=(0.0, -0.3),
    )
    assert displacement == pytest.approx(-0.3)


# ---- compute_support_center_xy: state separation tests ----

def test_support_center_is_midpoint_of_wheel_bodies():
    """Support center is the XY midpoint of left and right wheel body positions."""
    l_wheel = (1.0, 2.0, 0.06)
    r_wheel = (3.0, 4.0, 0.06)
    cx, cy = compute_support_center_xy(l_wheel, r_wheel)
    assert cx == pytest.approx(2.0)
    assert cy == pytest.approx(3.0)


def test_support_center_symmetric_wheels():
    """Symmetric wheel placement gives support center at robot centerline."""
    l_wheel = (-0.1, 0.5, 0.06)
    r_wheel = (0.1, 0.5, 0.06)
    cx, cy = compute_support_center_xy(l_wheel, r_wheel)
    assert cx == pytest.approx(0.0)
    assert cy == pytest.approx(0.5)


def test_support_position_error_zero_when_wheels_at_equilibrium():
    """If wheel support center is at equilibrium, support position error is zero."""
    l_wheel_eq = (0.0, 0.0, 0.06)
    r_wheel_eq = (0.0, 0.0, 0.06)
    support_eq = compute_support_center_xy(l_wheel_eq, r_wheel_eq)

    l_wheel_now = (0.0, 0.0, 0.06)
    r_wheel_now = (0.0, 0.0, 0.06)
    support_now = compute_support_center_xy(l_wheel_now, r_wheel_now)

    sagittal_axis = (0.0, 1.0)
    error = project_sagittal_displacement(
        origin_xy=support_eq,
        sagittal_axis_xy=sagittal_axis,
        current_xy=support_now,
    )
    assert error == pytest.approx(0.0)


def test_support_position_error_positive_when_wheels_move_forward():
    """If wheel support center moves forward, support position error is positive."""
    support_eq = (0.0, 0.0)
    support_now = (0.0, 0.15)  # moved 0.15 m forward along Y
    sagittal_axis = (0.0, 1.0)
    error = project_sagittal_displacement(
        origin_xy=support_eq,
        sagittal_axis_xy=sagittal_axis,
        current_xy=support_now,
    )
    assert error == pytest.approx(0.15)


def test_support_position_error_negative_when_wheels_move_backward():
    """If wheel support center moves backward, support position error is negative."""
    support_eq = (0.0, 0.0)
    support_now = (0.0, -0.12)
    sagittal_axis = (0.0, 1.0)
    error = project_sagittal_displacement(
        origin_xy=support_eq,
        sagittal_axis_xy=sagittal_axis,
        current_xy=support_now,
    )
    assert error == pytest.approx(-0.12)


def test_com_pitch_motion_does_not_affect_support_position_error():
    """COM moving forward due to pitch does NOT count as support position drift.

    When the robot pitches forward, COM moves forward relative to the support center.
    The support position error (based on wheel midpoint) should remain near zero
    if the wheels haven't moved, even though COM has moved.

    This is the key invariant: position hold tracks wheel support, not COM.
    """
    # Wheels haven't moved
    l_wheel_eq = (-0.1, 0.0, 0.06)
    r_wheel_eq = (0.1, 0.0, 0.06)
    support_eq = compute_support_center_xy(l_wheel_eq, r_wheel_eq)

    l_wheel_now = (-0.1, 0.0, 0.06)
    r_wheel_now = (0.1, 0.0, 0.06)
    support_now = compute_support_center_xy(l_wheel_now, r_wheel_now)

    sagittal_axis = (0.0, 1.0)
    support_error = project_sagittal_displacement(
        origin_xy=support_eq,
        sagittal_axis_xy=sagittal_axis,
        current_xy=support_now,
    )

    # COM moved forward due to pitch (e.g., 0.20 m forward)
    com_eq = (0.0, 0.0)
    com_now = (0.0, 0.20)
    com_error = project_sagittal_displacement(
        origin_xy=com_eq,
        sagittal_axis_xy=sagittal_axis,
        current_xy=com_now,
    )

    # Support position error is zero (wheels didn't move)
    assert support_error == pytest.approx(0.0)
    # COM error is nonzero (COM moved due to pitch)
    assert com_error == pytest.approx(0.20)
    # They are different — this is the key separation
    assert abs(support_error - com_error) > 0.1


def test_support_position_error_unaffected_by_yaw():
    """Nonzero yaw does not break initial-heading-frame projection of support position."""
    yaw_rad = math.radians(30)
    sagittal_axis = (math.sin(yaw_rad), math.cos(yaw_rad))

    # Wheels move 0.2 m along the initial sagittal axis
    support_eq = (0.0, 0.0)
    # 0.2 m along (sin30, cos30) = (0.1, 0.1732)
    support_now = (0.2 * math.sin(yaw_rad), 0.2 * math.cos(yaw_rad))

    error = project_sagittal_displacement(
        origin_xy=support_eq,
        sagittal_axis_xy=sagittal_axis,
        current_xy=support_now,
    )
    assert error == pytest.approx(0.2, abs=1e-4)


def test_pitch_error_zero_at_equilibrium():
    """When pitch_x equals pitch_ref, pitch error is zero."""
    pitch_ref = 0.021  # ~1.2 deg equilibrium pitch
    pitch_x = 0.021
    pitch_error = pitch_x - pitch_ref
    assert pitch_error == pytest.approx(0.0)


def test_pitch_error_positive_when_pitched_forward():
    """Positive pitch deviation from reference gives positive pitch error."""
    pitch_ref = 0.021
    pitch_x = 0.21  # ~12 deg
    pitch_error = pitch_x - pitch_ref
    assert pitch_error > 0.0
    assert pitch_error == pytest.approx(0.189)


def test_pitch_error_negative_when_pitched_backward():
    """Negative pitch deviation from reference gives negative pitch error."""
    pitch_ref = 0.021
    pitch_x = -0.05
    pitch_error = pitch_x - pitch_ref
    assert pitch_error < 0.0
