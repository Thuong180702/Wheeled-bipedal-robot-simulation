"""Calculate base_z for keyframe given joint angles to ensure wheels touch ground."""

import numpy as np

# Robot dimensions from MJCF
HIP_OFFSET_Z = -0.0295  # Hip is 0.0295m below torso origin
THIGH_LENGTH = 0.26
SHIN_LENGTH = 0.28
WHEEL_RADIUS = 0.06

# Additional offsets from MJCF body positions
HIP_YAW_TO_THIGH_Z = -0.030024  # From l_hip_yaw_link to l_thigh
KNEE_TO_WHEEL_Y = -0.0205  # Shin to wheel offset in y
WHEEL_CENTER_OFFSET_X = -0.038  # Wheel center offset from shin axis

def calculate_wheel_ground_contact_z(hip_pitch_rad, knee_rad):
    """Calculate the z-position of wheel ground contact point relative to torso origin.

    Args:
        hip_pitch_rad: Hip pitch angle in radians (positive = forward)
        knee_rad: Knee angle in radians (positive = bent forward)

    Returns:
        z_wheel_contact: Z-position of wheel ground contact relative to torso origin (negative = below)
    """
    # Start from torso origin
    z = 0.0

    # Hip offset
    z += HIP_OFFSET_Z

    # Hip yaw to thigh offset
    z += HIP_YAW_TO_THIGH_Z

    # Thigh segment (vertical component)
    # Hip pitch rotates in sagittal plane, positive = forward
    z -= THIGH_LENGTH * np.cos(hip_pitch_rad)

    # Shin segment (vertical component)
    # Knee angle is relative to thigh
    # Total angle from vertical = hip_pitch - knee
    shin_angle_from_vertical = hip_pitch_rad - knee_rad
    z -= SHIN_LENGTH * np.cos(shin_angle_from_vertical)

    # Wheel radius (from wheel center to ground contact)
    z -= WHEEL_RADIUS

    return z

def calculate_base_z_for_ground_contact(hip_pitch_rad, knee_rad):
    """Calculate base_z needed to place wheels on ground.

    Args:
        hip_pitch_rad: Hip pitch angle in radians
        knee_rad: Knee angle in radians

    Returns:
        base_z: Required base_z for keyframe
    """
    z_wheel_contact = calculate_wheel_ground_contact_z(hip_pitch_rad, knee_rad)
    # base_z is the torso origin height above ground
    # If wheel contact is at z_wheel_contact (negative), then base_z = -z_wheel_contact
    base_z = -z_wheel_contact
    return base_z

# Test configurations
configs = [
    # (hip_pitch_deg, knee_deg, description)
    (20, 90, "Straighter legs (20° hip, 90° knee)"),
    (25, 110, "Previous Run 26 (25° hip, 110° knee)"),
    (30, 100, "Moderate (30° hip, 100° knee)"),
    (15, 80, "Very straight (15° hip, 80° knee)"),
]

print("=" * 80)
print("Base Z Calculation for Keyframe Configurations")
print("=" * 80)
print()

for hip_pitch_deg, knee_deg, description in configs:
    hip_pitch_rad = np.deg2rad(hip_pitch_deg)
    knee_rad = np.deg2rad(knee_deg)

    base_z = calculate_base_z_for_ground_contact(hip_pitch_rad, knee_rad)

    print(f"{description}")
    print(f"  Hip pitch: {hip_pitch_deg}° ({hip_pitch_rad:.4f} rad)")
    print(f"  Knee: {knee_deg}° ({knee_rad:.4f} rad)")
    print(f"  Required base_z: {base_z:.4f} m")
    print()

# Calculate for the current keyframe
print("=" * 80)
print("Current keyframe analysis:")
print("=" * 80)
current_hip_pitch = 0.349  # rad
current_knee = 1.571  # rad
current_base_z = 0.460  # m

calculated_base_z = calculate_base_z_for_ground_contact(current_hip_pitch, current_knee)
print(f"Current keyframe: base_z={current_base_z:.4f}m, hip_pitch={current_hip_pitch:.4f}rad, knee={current_knee:.4f}rad")
print(f"Calculated required base_z: {calculated_base_z:.4f}m")
print(f"Difference: {current_base_z - calculated_base_z:.4f}m")
print()

if abs(current_base_z - calculated_base_z) > 0.01:
    print(f"WARNING: Current base_z is {current_base_z - calculated_base_z:.4f}m off!")
    print(f"Recommended: Update base_z to {calculated_base_z:.4f}m")
else:
    print("Current base_z is correct for ground contact.")
