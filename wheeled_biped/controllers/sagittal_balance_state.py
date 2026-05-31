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


def compute_support_center_xy(
    l_wheel_body_xpos: tuple[float, float, float],
    r_wheel_body_xpos: tuple[float, float, float],
) -> tuple[float, float]:
    """Compute the XY position of the wheel support center (midpoint).

    For a wheeled biped, the support center is the midpoint between the two
    wheel contact points. This is the controlled position for standing-in-place,
    NOT the COM position (which is allowed to move relative to the support center
    during pitch balance).

    Parameters
    ----------
    l_wheel_body_xpos : (x, y, z) world position of left wheel body.
    r_wheel_body_xpos : (x, y, z) world position of right wheel body.

    Returns
    -------
    (x, y) : world XY position of the support center.
    """
    support_x = 0.5 * (l_wheel_body_xpos[0] + r_wheel_body_xpos[0])
    support_y = 0.5 * (l_wheel_body_xpos[1] + r_wheel_body_xpos[1])
    return (support_x, support_y)


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
