"""Calculate required motor torques for all joints with safety factor.

This script calculates the maximum torque requirements for each joint type
based on robot dynamics, geometry, and operational requirements.
"""

import numpy as np
import mujoco
from pathlib import Path


def calculate_motor_requirements():
    """Calculate required motor torques for all joint types."""

    # Robot parameters
    total_mass = 8.1  # kg
    g = 9.81  # m/s^2
    weight = total_mass * g  # N

    # Geometry from XML
    hip_separation = 0.230  # m (distance between left and right hips)
    thigh_length = 0.26  # m
    shin_length = 0.28  # m
    wheel_radius = 0.06  # m

    # Operational requirements
    max_push_force = 200  # N (from CLAUDE.md requirement)
    safety_factor = 1.5

    print("="*80)
    print("MOTOR TORQUE REQUIREMENT CALCULATION")
    print("="*80)
    print(f"\nRobot parameters:")
    print(f"  Total mass: {total_mass} kg")
    print(f"  Weight: {weight:.2f} N")
    print(f"  Hip separation: {hip_separation*1000:.1f} mm")
    print(f"  Thigh length: {thigh_length*1000:.1f} mm")
    print(f"  Shin length: {shin_length*1000:.1f} mm")
    print(f"  Wheel radius: {wheel_radius*1000:.1f} mm")
    print(f"  Safety factor: {safety_factor}")

    # ========================================================================
    # 1. HIP ROLL MOTORS
    # ========================================================================
    print("\n" + "="*80)
    print("1. HIP ROLL MOTORS (l_hip_roll, r_hip_roll)")
    print("="*80)

    # From telemetry analysis: max desired roll moment = 62 Nm
    max_desired_roll_moment = 62.0  # Nm

    # Hip roll joint creates roll moment by tilting the legs laterally
    # The moment arm depends on the leg configuration
    # Approximate: hip roll torque creates lateral force at wheel contact
    # Moment arm ≈ leg_length (vertical distance from hip to wheel)

    # At standing height (h=0.71m), leg length ≈ 0.71 - 0.06 = 0.65 m
    # At squat height (h=0.40m), leg length ≈ 0.40 - 0.06 = 0.34 m

    leg_length_standing = 0.65  # m
    leg_length_squat = 0.34  # m

    # Roll moment = (tau_hip_roll_left - tau_hip_roll_right) × (leg_length / hip_separation)
    # This is approximate - actual calculation requires full kinematics

    # Conservative estimate: each hip roll must generate half the roll moment
    # with the shorter leg length (worst case = squat)
    required_hip_roll_torque = max_desired_roll_moment / 2.0

    # But we also need to consider lateral stabilization during push recovery
    # A 200N lateral push creates additional roll moment
    lateral_push_moment = max_push_force * 0.4  # assume CoM height = 0.4m
    required_hip_roll_push = lateral_push_moment / 2.0

    required_hip_roll = max(required_hip_roll_torque, required_hip_roll_push)
    required_hip_roll_with_safety = required_hip_roll * safety_factor

    print(f"\nRoll moment requirement:")
    print(f"  Max desired roll moment (from telemetry): {max_desired_roll_moment:.1f} Nm")
    print(f"  Required per hip roll motor: {required_hip_roll:.1f} Nm")
    print(f"  Lateral push moment (200N × 0.4m): {lateral_push_moment:.1f} Nm")
    print(f"  Required per hip roll (push recovery): {required_hip_roll_push:.1f} Nm")
    print(f"\n  >> Required hip roll torque: {required_hip_roll:.1f} Nm")
    print(f"  >> With safety factor {safety_factor}: {required_hip_roll_with_safety:.1f} Nm")
    print(f"\n  Current motor limit: 30 Nm")
    print(f"  Status: {'[X] INSUFFICIENT' if required_hip_roll_with_safety > 30 else '[OK] SUFFICIENT'}")

    # ========================================================================
    # 2. HIP YAW MOTORS
    # ========================================================================
    print("\n" + "="*80)
    print("2. HIP YAW MOTORS (l_hip_yaw, r_hip_yaw)")
    print("="*80)

    # Hip yaw is used for:
    # - Yaw stabilization (heading control)
    # - Differential wheel velocity compensation

    # Yaw moment requirement is typically lower than roll/pitch
    # Estimate: 20% of body weight × hip separation
    required_hip_yaw = 0.2 * weight * (hip_separation / 2)
    required_hip_yaw_with_safety = required_hip_yaw * safety_factor

    print(f"\nYaw moment requirement:")
    print(f"  Estimated requirement: {required_hip_yaw:.1f} Nm")
    print(f"  With safety factor {safety_factor}: {required_hip_yaw_with_safety:.1f} Nm")
    print(f"\n  Current motor limit: 30 Nm")
    print(f"  Status: {'[X] INSUFFICIENT' if required_hip_yaw_with_safety > 30 else '[OK] SUFFICIENT'}")

    # ========================================================================
    # 3. HIP PITCH MOTORS
    # ========================================================================
    print("\n" + "="*80)
    print("3. HIP PITCH MOTORS (l_hip_pitch, r_hip_pitch)")
    print("="*80)

    # Hip pitch supports body weight and controls height
    # Worst case: deep squat with hip pitch = 55° (0.96 rad)

    hip_pitch_angle = 0.968  # rad (from keyframe)
    knee_angle = 1.698  # rad (from keyframe)

    # Simplified static analysis:
    # At squat, hip pitch torque ≈ (weight/2) × (CoM_horizontal_offset)
    # CoM is forward of hip by ~0.1-0.2m in squat
    com_offset_squat = 0.15  # m (conservative estimate)

    required_hip_pitch_static = (weight / 2) * com_offset_squat

    # Dynamic requirement: standing up from squat
    # Need to accelerate body upward: F = m × a
    # Assume max acceleration = 2 m/s^2 (moderate speed)
    max_accel = 2.0  # m/s^2
    dynamic_force = total_mass * max_accel
    required_hip_pitch_dynamic = (dynamic_force / 2) * com_offset_squat

    # Push recovery: 200N forward push at CoM height 0.4m
    # Hip pitch must resist forward rotation
    push_moment_per_leg = (max_push_force * 0.4) / 2

    required_hip_pitch = max(required_hip_pitch_static,
                             required_hip_pitch_dynamic,
                             push_moment_per_leg)
    required_hip_pitch_with_safety = required_hip_pitch * safety_factor

    print(f"\nHip pitch torque requirement:")
    print(f"  Static (squat): {required_hip_pitch_static:.1f} Nm")
    print(f"  Dynamic (stand-up, a={max_accel} m/s^2): {required_hip_pitch_dynamic:.1f} Nm")
    print(f"  Push recovery (200N forward): {push_moment_per_leg:.1f} Nm")
    print(f"\n  >> Required hip pitch torque: {required_hip_pitch:.1f} Nm")
    print(f"  >> With safety factor {safety_factor}: {required_hip_pitch_with_safety:.1f} Nm")
    print(f"\n  Current motor limit: 150 Nm")
    print(f"  Status: {'[X] INSUFFICIENT' if required_hip_pitch_with_safety > 150 else '[OK] SUFFICIENT'}")

    # ========================================================================
    # 4. KNEE MOTORS
    # ========================================================================
    print("\n" + "="*80)
    print("4. KNEE MOTORS (l_knee, r_knee)")
    print("="*80)

    # Knee torque is similar to hip pitch but with different moment arm
    # At deep squat, knee angle = 97° (1.69 rad)

    # Knee torque ≈ hip pitch torque × (thigh_length / shin_length)
    # This is approximate - actual calculation requires full inverse dynamics

    required_knee = required_hip_pitch * (thigh_length / shin_length)
    required_knee_with_safety = required_knee * safety_factor

    print(f"\nKnee torque requirement:")
    print(f"  Estimated from hip pitch: {required_knee:.1f} Nm")
    print(f"  With safety factor {safety_factor}: {required_knee_with_safety:.1f} Nm")
    print(f"\n  Current motor limit: 150 Nm")
    print(f"  Status: {'[X] INSUFFICIENT' if required_knee_with_safety > 150 else '[OK] SUFFICIENT'}")

    # ========================================================================
    # 5. WHEEL MOTORS
    # ========================================================================
    print("\n" + "="*80)
    print("5. WHEEL MOTORS (l_wheel, r_wheel)")
    print("="*80)

    # Wheel torque requirements:
    # 1. Balance: accelerate/decelerate to maintain CoM over wheels
    # 2. Push recovery: accelerate to move under falling CoM

    # Max wheel acceleration for balance
    # Assume need to accelerate at 3 m/s^2 (aggressive balancing)
    max_wheel_accel = 3.0  # m/s^2
    wheel_angular_accel = max_wheel_accel / wheel_radius  # rad/s^2

    # Wheel inertia (from XML): 0.00012247 kg⋅m²
    wheel_inertia = 0.00012247  # kg⋅m^2

    # Torque = I × α + friction
    required_wheel_accel = wheel_inertia * wheel_angular_accel

    # Ground friction force for push recovery
    # Need to generate 200N horizontal force to recover from push
    # Friction force = wheel_torque / wheel_radius
    # wheel_torque = push_force × wheel_radius
    required_wheel_push = max_push_force * wheel_radius

    # Rolling resistance and friction
    # Assume coefficient of rolling resistance = 0.01
    rolling_resistance = 0.01 * (weight / 2) * wheel_radius

    required_wheel = max(required_wheel_accel, required_wheel_push) + rolling_resistance
    required_wheel_with_safety = required_wheel * safety_factor

    print(f"\nWheel torque requirement:")
    print(f"  Acceleration (3 m/s^2 linear): {required_wheel_accel:.2f} Nm")
    print(f"  Push recovery (200N x {wheel_radius}m): {required_wheel_push:.1f} Nm")
    print(f"  Rolling resistance: {rolling_resistance:.2f} Nm")
    print(f"\n  >> Required wheel torque: {required_wheel:.1f} Nm")
    print(f"  >> With safety factor {safety_factor}: {required_wheel_with_safety:.1f} Nm")
    print(f"\n  Current motor limit: 30 Nm")
    print(f"  Status: {'[X] INSUFFICIENT' if required_wheel_with_safety > 30 else '[OK] SUFFICIENT'}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY - REQUIRED MOTOR TORQUES (with 1.5× safety factor)")
    print("="*80)

    results = [
        ("Hip Roll", required_hip_roll_with_safety, 30, 2),
        ("Hip Yaw", required_hip_yaw_with_safety, 30, 2),
        ("Hip Pitch", required_hip_pitch_with_safety, 150, 2),
        ("Knee", required_knee_with_safety, 150, 2),
        ("Wheel", required_wheel_with_safety, 30, 2),
    ]

    print(f"\n{'Joint Type':<15} {'Required (Nm)':<15} {'Current (Nm)':<15} {'Quantity':<10} {'Status':<10}")
    print("-" * 80)

    for joint_type, required, current, qty in results:
        status = "[OK]" if required <= current else "[X] UPGRADE"
        print(f"{joint_type:<15} {required:>14.1f} {current:>14.1f} {qty:>9} {status:<10}")

    print("\n" + "="*80)
    print("RECOMMENDED MOTOR SPECIFICATIONS")
    print("="*80)

    print(f"\n1. Hip Roll Motors (x2):")
    print(f"   Current: 30 Nm")
    print(f"   Required: {required_hip_roll_with_safety:.1f} Nm")
    if required_hip_roll_with_safety > 30:
        recommended = np.ceil(required_hip_roll_with_safety / 5) * 5
        print(f"   >> UPGRADE to: {recommended:.0f} Nm motors")
    else:
        print(f"   >> Current motors are SUFFICIENT")

    print(f"\n2. Hip Yaw Motors (x2):")
    print(f"   Current: 30 Nm")
    print(f"   Required: {required_hip_yaw_with_safety:.1f} Nm")
    print(f"   >> Current motors are SUFFICIENT")

    print(f"\n3. Hip Pitch Motors (x2):")
    print(f"   Current: 150 Nm")
    print(f"   Required: {required_hip_pitch_with_safety:.1f} Nm")
    print(f"   >> Current motors are SUFFICIENT")

    print(f"\n4. Knee Motors (x2):")
    print(f"   Current: 150 Nm")
    print(f"   Required: {required_knee_with_safety:.1f} Nm")
    print(f"   >> Current motors are SUFFICIENT")

    print(f"\n5. Wheel Motors (x2):")
    print(f"   Current: 30 Nm")
    print(f"   Required: {required_wheel_with_safety:.1f} Nm")
    if required_wheel_with_safety > 30:
        recommended = np.ceil(required_wheel_with_safety / 5) * 5
        print(f"   >> UPGRADE to: {recommended:.0f} Nm motors")
    else:
        print(f"   >> Current motors are SUFFICIENT")

    print("\n" + "="*80)


if __name__ == "__main__":
    calculate_motor_requirements()
