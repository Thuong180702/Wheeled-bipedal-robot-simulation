import math

import jax.numpy as jnp
import numpy as np

from wheeled_biped.controllers.orientation_utils import (
    compute_orientation_from_gravity,
    compute_orientation_from_quaternion,
    compute_robot_frame_orientation_from_gravity,
    compute_robot_frame_orientation_from_quaternion,
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


def test_robot_frame_x_rotation_is_body_pitch_x_only():
    quat = quat_from_axis_angle([1.0, 0.0, 0.0], 0.2)
    body_pitch_x, body_roll_y, body_yaw_z = compute_robot_frame_orientation_from_quaternion(quat)
    assert abs(body_pitch_x - 0.2) < 1e-6
    assert abs(body_roll_y) < 1e-6
    assert abs(body_yaw_z) < 1e-6


def test_robot_frame_y_rotation_is_body_roll_y_only():
    quat = quat_from_axis_angle([0.0, 1.0, 0.0], -0.15)
    body_pitch_x, body_roll_y = compute_robot_frame_orientation_from_gravity(jnp.array(gravity_body_from_quat(quat)))
    assert abs(float(body_pitch_x)) < 1e-3
    assert abs(float(body_roll_y) + 0.15) < 1e-3


def test_robot_frame_gravity_and_quaternion_paths_agree():
    quat = quat_from_axis_angle([1.0, 0.0, 0.0], 0.08)
    gravity_body = gravity_body_from_quat(quat)
    pitch_x_q, roll_y_q, _ = compute_robot_frame_orientation_from_quaternion(quat)
    pitch_x_g, roll_y_g = compute_robot_frame_orientation_from_gravity(jnp.array(gravity_body))
    assert abs(float(pitch_x_g) - pitch_x_q) < 1e-3
    assert abs(float(roll_y_g) - roll_y_q) < 1e-3
