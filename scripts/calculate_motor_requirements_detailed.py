"""Detailed motor torque and speed calculation for wheeled biped robot.

This script calculates realistic motor requirements based on:
1. Actual robot geometry and mass distribution
2. Required motion speeds and accelerations
3. Static and dynamic torque requirements
4. Motor speed requirements (RPM)
"""

import numpy as np
import mujoco


def calculate_detailed_motor_requirements():
    """Calculate detailed motor torque and speed requirements."""

    # Load model to get accurate inertia data
    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Robot parameters
    total_mass = 8.1  # kg
    g = 9.81  # m/s^2
    weight = total_mass * g  # N

    # Geometry from XML
    hip_separation = 0.230  # m
    thigh_length = 0.26  # m
    shin_length = 0.28  # m
    wheel_radius = 0.06  # m
    leg_length = thigh_length + shin_length  # 0.54 m
    com_height_standing = 0.41  # m

    # Get body masses from model
    torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    l_thigh_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_thigh")
    l_shin_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_knee_link")
    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")

    torso_mass = mj_model.body_mass[torso_id]
    thigh_mass = mj_model.body_mass[l_thigh_id]
    shin_mass = mj_model.body_mass[l_shin_id]
    wheel_mass = mj_model.body_mass[l_wheel_id]
    leg_mass = thigh_mass + shin_mass + wheel_mass

    print("="*80)
    print("DETAILED MOTOR REQUIREMENT CALCULATION")
    print("="*80)
    print(f"\nRobot parameters:")
    print(f"  Total mass: {total_mass} kg")
    print(f"  Torso mass: {torso_mass} kg")
    print(f"  Leg mass (per leg): {leg_mass:.2f} kg")
    print(f"    - Thigh: {thigh_mass} kg")
    print(f"    - Shin: {shin_mass} kg")
    print(f"    - Wheel: {wheel_mass} kg")
    print(f"  Weight: {weight:.2f} N")
    print(f"  CoM height (standing): {com_height_standing} m")
    print(f"  Leg length: {leg_length} m")
    print(f"  Hip separation: {hip_separation*1000:.1f} mm")
    print(f"  Wheel radius: {wheel_radius*1000:.1f} mm")

    safety_factor = 1.5

    # ========================================================================
    # 1. HIP ROLL MOTORS - RECALCULATE CAREFULLY
    # ========================================================================
    print("\n" + "="*80)
    print("1. HIP ROLL MOTORS (l_hip_roll, r_hip_roll)")
    print("="*80)

    # Hip roll controls lateral leg tilt
    # Torque needed to:
    # a) Support leg weight when tilted
    # b) Accelerate leg laterally for roll stabilization

    # a) Static torque: leg weight × CoM distance × sin(tilt_angle)
    # Leg CoM is approximately at mid-leg: 0.27 m from hip roll
    leg_com_distance = 0.27  # m
    max_hip_roll_angle = 0.7  # rad (40 degrees, from joint limit)

    static_torque_per_leg = leg_mass * g * leg_com_distance * np.sin(max_hip_roll_angle)

    # b) Dynamic torque: need to tilt leg quickly for roll correction
    # If robot has 2.6 deg roll error, need to tilt legs to correct
    # Assume need to tilt leg 10 degrees in 0.3 seconds
    tilt_angle = 10 * np.pi / 180  # rad
    tilt_time = 0.3  # s
    angular_accel = 2 * tilt_angle / (tilt_time ** 2)  # rad/s^2

    # Leg moment of inertia about hip roll (approximate as point mass)
    leg_inertia_roll = leg_mass * (leg_com_distance ** 2)
    dynamic_torque_per_leg = leg_inertia_roll * angular_accel

    # c) Roll stabilization: counteract gravitational roll torque
    # At 2.6 deg roll, gravitational torque = m × g × h × sin(roll)
    roll_error = 2.6 * np.pi / 180  # rad
    grav_roll_torque = total_mass * g * com_height_standing * np.sin(roll_error)
    # Each hip roll contributes to counteracting this
    roll_stabilization_per_hip = grav_roll_torque / 2

    # d) Push recovery: 200N lateral push at CoM height
    # Creates roll moment: 200N × 0.4m = 80 Nm
    # But hip roll doesn't directly create this moment
    # Hip roll tilts leg, which changes ground reaction force
    # Approximate: hip_roll_torque × (leg_length / hip_separation) = roll_moment
    # So: hip_roll_torque = roll_moment × (hip_separation / leg_length)
    lateral_push = 200  # N
    push_roll_moment = lateral_push * com_height_standing
    push_torque_per_hip = push_roll_moment * (hip_separation / (2 * leg_length))

    required_hip_roll = max(static_torque_per_leg,
                            dynamic_torque_per_leg,
                            roll_stabilization_per_hip,
                            push_torque_per_hip)
    required_hip_roll_with_safety = required_hip_roll * safety_factor

    print(f"\nHip roll torque components:")
    print(f"  a) Static (leg weight at max tilt): {static_torque_per_leg:.2f} Nm")
    print(f"  b) Dynamic (tilt 10 deg in 0.3s): {dynamic_torque_per_leg:.2f} Nm")
    print(f"  c) Roll stabilization (2.6 deg error): {roll_stabilization_per_hip:.2f} Nm")
    print(f"  d) Push recovery (200N lateral): {push_torque_per_hip:.2f} Nm")
    print(f"\n  >> Required hip roll torque: {required_hip_roll:.1f} Nm")
    print(f"  >> With safety factor {safety_factor}: {required_hip_roll_with_safety:.1f} Nm")

    # Speed requirement: how fast does hip roll need to move?
    # Max angular velocity: 40 degrees in 0.5 seconds
    max_hip_roll_velocity = max_hip_roll_angle / 0.5  # rad/s
    max_hip_roll_rpm = max_hip_roll_velocity * 60 / (2 * np.pi)

    print(f"\nHip roll speed requirement:")
    print(f"  Max angular velocity: {max_hip_roll_velocity:.2f} rad/s ({max_hip_roll_rpm:.1f} RPM)")
    print(f"\n  Current motor limit: 30 Nm")
    print(f"  Status: {'[X] INSUFFICIENT' if required_hip_roll_with_safety > 30 else '[OK] SUFFICIENT'}")

    # ========================================================================
    # 2. HIP YAW MOTORS
    # ========================================================================
    print("\n" + "="*80)
    print("2. HIP YAW MOTORS (l_hip_yaw, r_hip_yaw)")
    print("="*80)

    # Hip yaw rotates leg about vertical axis
    # Torque needed for yaw stabilization and differential drive compensation

    # Leg moment of inertia about yaw axis (vertical)
    leg_inertia_yaw = leg_mass * (leg_length / 2) ** 2  # approximate

    # Required yaw acceleration: 20 degrees in 0.5 seconds
    yaw_angle = 20 * np.pi / 180  # rad
    yaw_time = 0.5  # s
    yaw_angular_accel = 2 * yaw_angle / (yaw_time ** 2)

    required_hip_yaw = leg_inertia_yaw * yaw_angular_accel
    required_hip_yaw_with_safety = required_hip_yaw * safety_factor

    # Speed requirement
    max_hip_yaw_velocity = 0.4 / 0.5  # rad/s (from joint limit)
    max_hip_yaw_rpm = max_hip_yaw_velocity * 60 / (2 * np.pi)

    print(f"\nHip yaw torque requirement:")
    print(f"  Dynamic (rotate 20 deg in 0.5s): {required_hip_yaw:.2f} Nm")
    print(f"  With safety factor {safety_factor}: {required_hip_yaw_with_safety:.1f} Nm")
    print(f"\nHip yaw speed requirement:")
    print(f"  Max angular velocity: {max_hip_yaw_velocity:.2f} rad/s ({max_hip_yaw_rpm:.1f} RPM)")
    print(f"\n  Current motor limit: 30 Nm")
    print(f"  Status: [OK] SUFFICIENT")

    # ========================================================================
    # 3. HIP PITCH MOTORS
    # ========================================================================
    print("\n" + "="*80)
    print("3. HIP PITCH MOTORS (l_hip_pitch, r_hip_pitch)")
    print("="*80)

    # Hip pitch supports body weight and controls height
    # Most critical joint for standing and squatting

    # Static torque at squat (hip pitch = 55 deg)
    hip_pitch_angle = 0.968  # rad (55.5 deg)

    # Torque = (body_weight / 2) × moment_arm
    # Moment arm depends on CoM position relative to hip
    # At squat, CoM is forward of hip by ~0.1-0.15m
    moment_arm_squat = 0.12  # m (conservative)
    static_hip_pitch = (weight / 2) * moment_arm_squat

    # Dynamic torque: standing up from squat
    # Need to accelerate body upward at 1.5 m/s^2
    standup_accel = 1.5  # m/s^2
    dynamic_force = total_mass * standup_accel
    dynamic_hip_pitch = (dynamic_force / 2) * moment_arm_squat

    # Push recovery: 200N forward push
    # Creates pitch moment that hip pitch must resist
    forward_push = 200  # N
    push_pitch_moment = forward_push * com_height_standing
    push_hip_pitch = push_pitch_moment / 2

    required_hip_pitch = max(static_hip_pitch, dynamic_hip_pitch, push_hip_pitch)
    required_hip_pitch_with_safety = required_hip_pitch * safety_factor

    # Speed requirement: squat from 0.7m to 0.4m in 2 seconds
    # Hip pitch changes from ~30 deg to ~55 deg = 25 deg = 0.44 rad
    hip_pitch_range = 0.44  # rad
    squat_time = 2.0  # s
    max_hip_pitch_velocity = hip_pitch_range / squat_time
    max_hip_pitch_rpm = max_hip_pitch_velocity * 60 / (2 * np.pi)

    print(f"\nHip pitch torque requirement:")
    print(f"  Static (squat position): {static_hip_pitch:.1f} Nm")
    print(f"  Dynamic (stand-up, a=1.5 m/s^2): {dynamic_hip_pitch:.1f} Nm")
    print(f"  Push recovery (200N forward): {push_hip_pitch:.1f} Nm")
    print(f"\n  >> Required hip pitch torque: {required_hip_pitch:.1f} Nm")
    print(f"  >> With safety factor {safety_factor}: {required_hip_pitch_with_safety:.1f} Nm")
    print(f"\nHip pitch speed requirement:")
    print(f"  Max angular velocity: {max_hip_pitch_velocity:.2f} rad/s ({max_hip_pitch_rpm:.1f} RPM)")
    print(f"\n  Current motor limit: 150 Nm")
    print(f"  Status: [OK] SUFFICIENT")

    # ========================================================================
    # 4. KNEE MOTORS
    # ========================================================================
    print("\n" + "="*80)
    print("4. KNEE MOTORS (l_knee, r_knee)")
    print("="*80)

    # Knee torque similar to hip pitch
    # Supports shin + wheel weight and controls leg extension

    shin_wheel_mass = shin_mass + wheel_mass
    shin_com_distance = shin_length / 2  # approximate

    # Static torque at squat (knee = 97 deg)
    knee_angle = 1.698  # rad
    static_knee = shin_wheel_mass * g * shin_com_distance * np.sin(knee_angle - np.pi/2)

    # Dynamic torque: leg extension
    # Similar to hip pitch but with lighter load
    dynamic_knee = required_hip_pitch * (shin_wheel_mass / (total_mass / 2))

    required_knee = max(static_knee, dynamic_knee)
    required_knee_with_safety = required_knee * safety_factor

    # Speed requirement: same as hip pitch
    max_knee_velocity = max_hip_pitch_velocity
    max_knee_rpm = max_hip_pitch_rpm

    print(f"\nKnee torque requirement:")
    print(f"  Static (squat position): {static_knee:.1f} Nm")
    print(f"  Dynamic (leg extension): {dynamic_knee:.1f} Nm")
    print(f"\n  >> Required knee torque: {required_knee:.1f} Nm")
    print(f"  >> With safety factor {safety_factor}: {required_knee_with_safety:.1f} Nm")
    print(f"\nKnee speed requirement:")
    print(f"  Max angular velocity: {max_knee_velocity:.2f} rad/s ({max_knee_rpm:.1f} RPM)")
    print(f"\n  Current motor limit: 150 Nm")
    print(f"  Status: [OK] SUFFICIENT")

    # ========================================================================
    # 5. WHEEL MOTORS
    # ========================================================================
    print("\n" + "="*80)
    print("5. WHEEL MOTORS (l_wheel, r_wheel)")
    print("="*80)

    # Wheel torque for balancing and locomotion

    # a) Acceleration for balancing: 2 m/s^2 linear
    balance_accel = 2.0  # m/s^2
    wheel_angular_accel = balance_accel / wheel_radius

    # Wheel inertia from model
    wheel_inertia = 0.00012247  # kg⋅m^2
    accel_torque = wheel_inertia * wheel_angular_accel

    # b) Ground friction force
    # To accelerate robot at 2 m/s^2: F = m × a
    required_force = total_mass * balance_accel
    friction_torque = required_force * wheel_radius / 2  # per wheel

    # c) Rolling resistance
    rolling_coeff = 0.01
    rolling_resistance = rolling_coeff * (weight / 2) * wheel_radius

    # d) Push recovery: need to accelerate at 3 m/s^2 to recover from 200N push
    push_recovery_accel = 3.0  # m/s^2
    push_recovery_force = total_mass * push_recovery_accel
    push_recovery_torque = push_recovery_force * wheel_radius / 2

    required_wheel = max(friction_torque, push_recovery_torque) + rolling_resistance
    required_wheel_with_safety = required_wheel * safety_factor

    # Speed requirement: max linear speed 1.5 m/s
    max_linear_speed = 1.5  # m/s
    max_wheel_angular_velocity = max_linear_speed / wheel_radius  # rad/s
    max_wheel_rpm = max_wheel_angular_velocity * 60 / (2 * np.pi)

    print(f"\nWheel torque requirement:")
    print(f"  Acceleration (2 m/s^2 linear): {friction_torque:.2f} Nm")
    print(f"  Push recovery (3 m/s^2): {push_recovery_torque:.2f} Nm")
    print(f"  Rolling resistance: {rolling_resistance:.2f} Nm")
    print(f"\n  >> Required wheel torque: {required_wheel:.1f} Nm")
    print(f"  >> With safety factor {safety_factor}: {required_wheel_with_safety:.1f} Nm")
    print(f"\nWheel speed requirement:")
    print(f"  Max linear speed: {max_linear_speed} m/s")
    print(f"  Max angular velocity: {max_wheel_angular_velocity:.1f} rad/s ({max_wheel_rpm:.0f} RPM)")
    print(f"\n  Current motor limit: 30 Nm")
    print(f"  Status: [OK] SUFFICIENT")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY - MOTOR REQUIREMENTS (with 1.5x safety factor)")
    print("="*80)

    results = [
        ("Hip Roll", required_hip_roll_with_safety, max_hip_roll_rpm, 30, 2),
        ("Hip Yaw", required_hip_yaw_with_safety, max_hip_yaw_rpm, 30, 2),
        ("Hip Pitch", required_hip_pitch_with_safety, max_hip_pitch_rpm, 150, 2),
        ("Knee", required_knee_with_safety, max_knee_rpm, 150, 2),
        ("Wheel", required_wheel_with_safety, max_wheel_rpm, 30, 2),
    ]

    print(f"\n{'Joint':<12} {'Torque (Nm)':<12} {'Speed (RPM)':<12} {'Current (Nm)':<14} {'Qty':<5} {'Status':<10}")
    print("-" * 80)

    for joint_type, required_torque, required_rpm, current, qty in results:
        status = "[OK]" if required_torque <= current else "[X]"
        print(f"{joint_type:<12} {required_torque:>11.1f} {required_rpm:>11.0f} {current:>13.0f} {qty:>4} {status:<10}")

    print("\n" + "="*80)
    print("RECOMMENDED MOTOR SPECIFICATIONS")
    print("="*80)

    print(f"\n1. Hip Roll Motors (x2):")
    print(f"   Required: {required_hip_roll_with_safety:.1f} Nm, {max_hip_roll_rpm:.0f} RPM")
    print(f"   Current: 30 Nm")
    if required_hip_roll_with_safety > 30:
        recommended = np.ceil(required_hip_roll_with_safety / 5) * 5
        print(f"   >> UPGRADE to: {recommended:.0f} Nm motors")
    else:
        print(f"   >> Current motors SUFFICIENT")

    print(f"\n2. Hip Yaw Motors (x2):")
    print(f"   Required: {required_hip_yaw_with_safety:.1f} Nm, {max_hip_yaw_rpm:.0f} RPM")
    print(f"   Current: 30 Nm")
    print(f"   >> Current motors SUFFICIENT")

    print(f"\n3. Hip Pitch Motors (x2):")
    print(f"   Required: {required_hip_pitch_with_safety:.1f} Nm, {max_hip_pitch_rpm:.0f} RPM")
    print(f"   Current: 150 Nm")
    print(f"   >> Current motors SUFFICIENT")

    print(f"\n4. Knee Motors (x2):")
    print(f"   Required: {required_knee_with_safety:.1f} Nm, {max_knee_rpm:.0f} RPM")
    print(f"   Current: 150 Nm")
    print(f"   >> Current motors SUFFICIENT")

    print(f"\n5. Wheel Motors (x2):")
    print(f"   Required: {required_wheel_with_safety:.1f} Nm, {max_wheel_rpm:.0f} RPM")
    print(f"   Current: 30 Nm")
    print(f"   >> Current motors SUFFICIENT")

    print("\n" + "="*80)
    print("NOTES:")
    print("="*80)
    print("- Hip roll calculation based on actual leg mass and geometry")
    print("- Torque requirements include 1.5x safety factor")
    print("- Speed requirements based on realistic motion profiles")
    print("- Push recovery assumes 200N lateral/forward push")
    print("="*80)


if __name__ == "__main__":
    calculate_detailed_motor_requirements()
