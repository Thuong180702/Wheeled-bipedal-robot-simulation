"""Unified orientation computation utilities for consistent roll/pitch/yaw extraction.

Provides single source of truth for orientation computation from gravity vector
and quaternion, ensuring consistency across all controller components.
"""

import jax.numpy as jnp
import numpy as np
from jax import Array


def compute_orientation_from_gravity(gravity_body: Array) -> tuple[float, float]:
    """Compute roll and pitch from gravity vector in body frame."""
    gx, gy, gz = gravity_body[0], gravity_body[1], gravity_body[2]
    roll = jnp.arctan2(-gy, -gz)
    pitch = jnp.arctan2(gx, -gz)
    return roll, pitch


def compute_orientation_from_quaternion(quat: np.ndarray) -> tuple[float, float, float]:
    """Compute roll, pitch, yaw from quaternion [w, x, y, z]."""
    w, x, y, z = quat

    roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    return roll, pitch, yaw
