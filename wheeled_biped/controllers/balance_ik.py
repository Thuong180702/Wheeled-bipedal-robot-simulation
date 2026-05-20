"""Balance-aware inverse kinematics for wheeled biped.

Computes joint angles that achieve desired height while maintaining
CoM centered over wheel contact points for stable standing.
"""

import jax.numpy as jnp
from jax import Array


def compute_balanced_leg_angles(
    height_cmd: float,
    hip_to_knee_length: float = 0.26,
    knee_to_wheel_length: float = 0.28,
    wheel_forward_offset: float = 0.1019,  # Actual wheel forward offset from MuJoCo model
) -> tuple[float, float]:
    """Compute hip_pitch and knee angles for balanced standing at desired height.

    Uses inverse kinematics with constraints:
    1. Wheels touch ground (contact constraint)
    2. CoM centered over wheels (balance constraint)
    3. Torso vertical (posture constraint)

    Args:
        height_cmd: Desired CoM height in meters
        hip_to_knee_length: Length from hip_pitch joint to knee joint
        knee_to_wheel_length: Length from knee joint to wheel center
        wheel_forward_offset: Forward distance from torso to wheels

    Returns:
        Tuple of (hip_pitch, knee) angles in radians that achieve stable standing
    """
    # Robot geometry constants
    hip_height_offset = 0.05  # Hip joint height above CoM
    wheel_radius = 0.060  # Wheel radius

    # Total leg length available
    total_leg_length = hip_to_knee_length + knee_to_wheel_length

    # Clamp height to achievable range
    min_height = 0.35  # Fully crouched
    max_height = total_leg_length * 0.95  # Nearly straight
    height_cmd = jnp.clip(height_cmd, min_height, max_height)

    # Hip position in world frame (assuming torso at origin)
    hip_x = 0.0
    hip_z = height_cmd + hip_height_offset

    # Wheel contact point in world frame
    wheel_x = wheel_forward_offset
    wheel_z = wheel_radius

    # Required leg length to reach from hip to wheel contact
    dx = wheel_x - hip_x
    dz = hip_z - wheel_z
    required_leg_length = jnp.sqrt(dx**2 + dz**2)

    # Angle from hip to target (leg angle from vertical)
    leg_angle = jnp.arctan2(dx, dz)

    # Use law of cosines to find knee angle (reachable case)
    # knee angle is the interior angle at the knee joint
    # cos(knee) = (L1^2 + L2^2 - required^2) / (2*L1*L2)
    cos_knee = (hip_to_knee_length**2 + knee_to_wheel_length**2 - required_leg_length**2) / \
               (2 * hip_to_knee_length * knee_to_wheel_length)
    cos_knee = jnp.clip(cos_knee, -1.0, 1.0)
    knee_reachable = jnp.arccos(cos_knee)

    # Use law of sines to find angle at hip (reachable case)
    # sin(alpha) / L2 = sin(knee) / required
    sin_alpha = (knee_to_wheel_length * jnp.sin(knee_reachable)) / required_leg_length
    sin_alpha = jnp.clip(sin_alpha, -1.0, 1.0)
    alpha = jnp.arcsin(sin_alpha)

    # Hip pitch is leg angle PLUS alpha (thigh angles further forward from hip-to-wheel line)
    hip_pitch_reachable = leg_angle + alpha

    # Handle unreachable case: use maximum extension (JAX-compatible)
    unreachable = required_leg_length > total_leg_length
    hip_pitch = jnp.where(unreachable, leg_angle, hip_pitch_reachable)
    knee = jnp.where(unreachable, 0.1, knee_reachable)

    return hip_pitch, knee


def compute_target_posture_from_height(height_cmd: float) -> Array:
    """Compute full 10-DOF target posture for balanced standing.

    Args:
        height_cmd: Desired CoM height in meters

    Returns:
        Target joint positions (10,) with balanced leg configuration
    """
    hip_pitch, knee = compute_balanced_leg_angles(height_cmd)

    target_pos = jnp.array([
        0.0,  # l_hip_roll - neutral
        0.0,  # l_hip_yaw - neutral
        hip_pitch,  # l_hip_pitch - forward lean for balance
        knee,  # l_knee - computed for height + balance
        0.0,  # l_wheel - no target
        0.0,  # r_hip_roll - neutral
        0.0,  # r_hip_yaw - neutral
        hip_pitch,  # r_hip_pitch - forward lean for balance
        knee,  # r_knee - computed for height + balance
        0.0,  # r_wheel - no target
    ])

    return target_pos
