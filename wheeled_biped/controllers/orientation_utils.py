"""Unified orientation utilities for generic and robot-frame angle extraction."""

import jax.numpy as jnp
import numpy as np
from jax import Array


def compute_orientation_from_gravity(gravity_body: Array) -> tuple[float, float]:
    """Compute generic roll/pitch from gravity vector in body frame."""
    body_pitch_x, body_roll_y = compute_robot_frame_orientation_from_gravity(gravity_body)
    return body_pitch_x, body_roll_y


def compute_orientation_from_quaternion(quat: np.ndarray) -> tuple[float, float, float]:
    """Compute generic roll, pitch, yaw from quaternion [w, x, y, z]."""
    w, x, y, z = quat

    roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    return roll, pitch, yaw


def compute_robot_frame_orientation_from_gravity(gravity_body: Array) -> tuple[float, float]:
    """Compute robot-frame pitch_x and roll_y from gravity vector in body frame."""
    gx, gy, gz = gravity_body[0], gravity_body[1], gravity_body[2]
    body_pitch_x = jnp.arctan2(-gy, -gz)
    body_roll_y = jnp.arctan2(gx, -gz)
    return body_pitch_x, body_roll_y


def compute_robot_frame_orientation_from_quaternion(quat: np.ndarray) -> tuple[float, float, float]:
    """Compute robot-frame (pitch_x, roll_y, yaw_z) from quaternion [w, x, y, z]."""
    roll, pitch, yaw = compute_orientation_from_quaternion(quat)
    body_pitch_x = roll
    body_roll_y = pitch
    body_yaw_z = yaw
    return body_pitch_x, body_roll_y, body_yaw_z
