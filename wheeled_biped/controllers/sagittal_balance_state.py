"""Sagittal balance state helpers.

Provides pure functions for constructing sagittal balance state from
telemetry/state inputs. All sagittal quantities are expressed in the
initial-heading frame.

Axis convention (project standard):
  X: lateral
  Y: sagittal
  Z: vertical
"""

import jax.numpy as jnp


def project_sagittal_displacement(
    origin_xy: tuple[float, float],
    sagittal_axis_xy: tuple[float, float],
    current_xy: tuple[float, float],
) -> float:
    """Project planar position onto the initial-heading sagittal axis.

    Parameters
    ----------
    origin_xy : (x, y) world position of equilibrium reference.
    sagittal_axis_xy : (x, y) unit vector of initial sagittal heading in world frame.
    current_xy : (x, y) current planar CoM position in world frame.

    Returns
    -------
    float : signed sagittal displacement along the initial-heading axis.
    """
    dx = current_xy[0] - origin_xy[0]
    dy = current_xy[1] - origin_xy[1]
    return dx * sagittal_axis_xy[0] + dy * sagittal_axis_xy[1]


def project_sagittal_velocity(
    sagittal_axis_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
) -> float:
    """Project planar velocity onto the initial-heading sagittal axis.

    Parameters
    ----------
    sagittal_axis_xy : (x, y) unit vector of initial sagittal heading in world frame.
    velocity_xy : (vx, vy) current planar CoM velocity in world frame.

    Returns
    -------
    float : signed sagittal velocity along the initial-heading axis.
    """
    return velocity_xy[0] * sagittal_axis_xy[0] + velocity_xy[1] * sagittal_axis_xy[1]


def build_sagittal_balance_state(
    sagittal_position_error: float,
    sagittal_velocity: float,
    pitch_x: float,
    pitch_rate_x: float,
    wheel_velocity_mean: float,
) -> jnp.ndarray:
    """Build the 5-element sagittal balance state vector.

    Parameters
    ----------
    sagittal_position_error : signed displacement from equilibrium (m).
    sagittal_velocity : signed CoM velocity along initial-heading axis (m/s).
    pitch_x : robot-frame sagittal tilt (rad).
    pitch_rate_x : sagittal angular velocity (rad/s).
    wheel_velocity_mean : mean of left/right wheel velocities (rad/s).

    Returns
    -------
    jnp.ndarray : shape (5,) sagittal balance state vector.
    """
    return jnp.array([
        sagittal_position_error,
        sagittal_velocity,
        pitch_x,
        pitch_rate_x,
        wheel_velocity_mean,
    ])
