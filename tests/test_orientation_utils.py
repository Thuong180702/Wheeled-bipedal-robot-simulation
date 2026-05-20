import math

import jax.numpy as jnp
import numpy as np

from wheeled_biped.controllers.orientation_utils import (
    compute_orientation_from_gravity,
    compute_orientation_from_quaternion,
)


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    return np.array([
        math.cos(half),
        axis[0] * math.sin(half),
        axis[1] * math.sin(half),
        axis[2] * math.sin(half),
    ])


def gravity_body_from_quat(quat):
    w, x, y, z = quat
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return rot.T @ np.array([0.0, 0.0, -9.81])


def test_identity_quaternion_is_level():
    roll, pitch, yaw = compute_orientation_from_quaternion(np.array([1.0, 0.0, 0.0, 0.0]))
    assert abs(roll) < 1e-9
    assert abs(pitch) < 1e-9
    assert abs(yaw) < 1e-9


def test_x_axis_rotation_is_roll_only():
    quat = quat_from_axis_angle([1.0, 0.0, 0.0], 0.2)
    roll, pitch, yaw = compute_orientation_from_quaternion(quat)
    assert abs(roll - 0.2) < 1e-6
    assert abs(pitch) < 1e-6
    assert abs(yaw) < 1e-6


def test_y_axis_rotation_is_pitch_only():
    quat = quat_from_axis_angle([0.0, 1.0, 0.0], -0.15)
    roll, pitch, yaw = compute_orientation_from_quaternion(quat)
    assert abs(roll) < 1e-6
    assert abs(pitch + 0.15) < 1e-6
    assert abs(yaw) < 1e-6


def test_gravity_and_quaternion_paths_agree_for_small_angles():
    quat = quat_from_axis_angle([1.0, 0.0, 0.0], 0.08)
    gravity_body = gravity_body_from_quat(quat)
    roll_q, pitch_q, _ = compute_orientation_from_quaternion(quat)
    roll_g, pitch_g = compute_orientation_from_gravity(jnp.array(gravity_body))
    assert abs(float(roll_g) - roll_q) < 1e-3
    assert abs(float(pitch_g) - pitch_q) < 1e-3
