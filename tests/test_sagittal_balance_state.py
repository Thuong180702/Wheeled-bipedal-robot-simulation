"""Tests for sagittal balance state helpers: frame projection, velocity, state bundle."""

import math

import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.sagittal_balance_state import (
    build_sagittal_balance_state,
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
