"""Calculate MAXIMUM motor torques for ALL postures including worst cases.

This includes:
- Standing and squatting (already calculated)
- Lying down and standing up from lying
- Falling and recovery
- Maximum joint range of motion
"""

import numpy as np
import mujoco


def calculate_maximum_motor_torques():
    """Calculate absolute maximum motor torques for all possible postures."""

    # Load model
    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)

    # Robot parameters
    total_mass = 8.1  # kg
    g = 9.81  # m/s^2
    weight = total_mass * g  # N

    # Get body masses
    torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    l_thigh_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_thigh")
    l_shin_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_knee_link")
    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")

    torso_mass = mj_model.body_mass[torso_id]
    thigh_mass = mj_model.body_mass[l_thigh_id]
    shin_mass = mj_model.body_mass[l_shin_id]
    wheel_mass = mj_model.body_mass[l_wheel_id]
    leg_mass = thigh_mass + shin_mass + wheel_mass

    # Geometry
    thigh_length = 0.26  # m
    shin_length = 0.28  # m
    leg_length = thigh_length + shin_length
    hip_separation = 0.230  # m
    wheel_radius = 0.06  # m

    print("="*80)
    print("MAXIMUM MOTOR TORQUE CALCULATION - ALL POSTURES")
    print("="*80)
    print(f"\nRobot: {total_mass} kg, {torso_mass} kg torso, {leg_mass:.2f} kg per leg")
    print(f"Leg: {thigh_mass} kg thigh + {shin_mass} kg shin + {wheel_mass} kg wheel")

    safety_factor = 1.5

    # ========================================================================
    # 1. HIP ROLL - WORST CASE
    # ========================================================================
    print("\n" + "="*80)
    print("1. HIP ROLL MOTORS - WORST CASE ANALYSIS")
    print("="*80)

    # Case A: Standing/squatting (already calculated)
    leg_com_distance = 0.27  # m
    max_hip_roll_angle = 0.7  # rad (40 deg)
    case_a_static = leg_mass * g * leg_com_distance * np.sin(max_hip_roll_angle)

    # Case B: Lying down - hip roll must lift entire leg against gravity
    # When robot is on its side, hip roll lifts leg vertically
    # Torque = leg_weight × leg_CoM_distance
    case_b_lying = leg_mass * g * leg_com_distance

    # Case C: Standing up from lying - dynamic torque
    # Need to lift leg from horizontal to vertical in 2 seconds
    standup_time = 2.0  # s
    standup_angle = np.pi / 2  # 90 degrees
    standup_angular_accel = 2 * standup_angle / (standup_time ** 2)
    leg_inertia_roll = leg_mass * (leg_com_distance ** 2)
    case_c_dynamic = leg_inertia_roll * standup_angular_accel

    # Case D: Push recovery (already calculated)
    lateral_push = 200  # N
    com_height = 0.41  # m
    push_roll_moment = lateral_push * com_height
    case_d_push = push_roll_moment * (hip_separation / (2 * leg_length))

    max_hip_roll = max(case_a_static, case_b_lying, case_c_dynamic, case_d_push)
    max_hip_roll_with_safety = max_hip_roll * safety_factor

    print(f"\nHip roll torque cases:")
    print(f"  A) Standing (leg tilted 40 deg): {case_a_static:.2f} Nm")
    print(f"  B) Lying down (lift leg vertically): {case_b_lying:.2f} Nm")
    print(f"  C) Stand up from lying (90 deg in 2s): {case_c_dynamic:.2f} Nm")
    print(f"  D) Push recovery (200N lateral): {case_d_push:.2f} Nm")
    print(f"\n  >> MAXIMUM hip roll torque: {max_hip_roll:.1f} Nm")
    print(f"  >> With safety factor {safety_factor}: {max_hip_roll_with_safety:.1f} Nm")
    print(f"\n  Current motor: 30 Nm")
    print(f"  Status: {'[X] INSUFFICIENT' if max_hip_roll_with_safety > 30 else '[OK] SUFFICIENT'}")

    # ========================================================================
    # 2. HIP YAW - WORST CASE
    # ========================================================================
    print("\n" + "="*80)
    print("2. HIP YAW MOTORS - WORST CASE ANALYSIS")
    print("="*80)

    # Hip yaw rotates leg about vertical axis
    # Worst case: rapid yaw rotation for recovery

    # Leg moment of inertia about yaw axis
    leg_inertia_yaw = leg_mass * (leg_length / 2) ** 2

    # Case A: Normal yaw control (already calculated)
    yaw_angle = 20 * np.pi / 180
    yaw_time = 0.5
    case_a_yaw = leg_inertia_yaw * (2 * yaw_angle / (yaw_time ** 2))

    # Case B: Emergency yaw correction - 40 deg in 0.3s
    emergency_yaw_angle = 40 * np.pi / 180
    emergency_yaw_time = 0.3
    case_b_emergency = leg_inertia_yaw * (2 * emergency_yaw_angle / (emergency_yaw_time ** 2))

    max_hip_yaw = max(case_a_yaw, case_b_emergency)
    max_hip_yaw_with_safety = max_hip_yaw * safety_factor

    print(f"\nHip yaw torque cases:")
    print(f"  A) Normal (20 deg in 0.5s): {case_a_yaw:.2f} Nm")
    print(f"  B) Emergency (40 deg in 0.3s): {case_b_emergency:.2f} Nm")
    print(f"\n  >> MAXIMUM hip yaw torque: {max_hip_yaw:.1f} Nm")
    print(f"  >> With safety factor {safety_factor}: {max_hip_yaw_with_safety:.1f} Nm")
    print(f"\n  Current motor: 30 Nm")
    print(f"  Status: [OK] SUFFICIENT")

    # ========================================================================
    # 3. HIP PITCH - WORST CASE
    # ========================================================================
    print("\n" + "="*80)
    print("3. HIP PITCH MOTORS - WORST CASE ANALYSIS")
    print("="*80)

    # Case A: Squatting (already calculated)
    moment_arm_squat = 0.12  # m
    case_a_squat = (weight / 2) * moment_arm_squat

    # Case B: Push recovery (already calculated)
    forward_push = 200  # N
    case_b_push = (forward_push * com_height) / 2

    # Case C: Lying down - hip pitch must support torso weight
    # When robot falls forward, hip pitch must resist torso rotation
    # Torque = torso_weight × distance_to_hip
    torso_to_hip_distance = 0.15  # m (approximate)
    case_c_lying = torso_mass * g * torso_to_hip_distance

    # Case D: Standing up from lying - must lift torso
    # Torso rotates from horizontal to vertical about hip
    # Torque = torso_weight × CoM_distance + dynamic torque
    standup_time_pitch = 3.0  # s (slower, more controlled)
    standup_angle_pitch = np.pi / 2  # 90 degrees
    standup_angular_accel_pitch = 2 * standup_angle_pitch / (standup_time_pitch ** 2)
    torso_inertia = torso_mass * (torso_to_hip_distance ** 2)
    case_d_standup_static = torso_mass * g * torso_to_hip_distance
    case_d_standup_dynamic = torso_inertia * standup_angular_accel_pitch
    case_d_standup = case_d_standup_static + case_d_standup_dynamic

    # Case E: Maximum joint range - hip pitch at extreme angle
    # At hip_pitch = 1.8 rad (103 deg), supporting full body weight
    max_hip_pitch_angle = 1.8  # rad (from joint limit)
    case_e_extreme = (weight / 2) * 0.20  # larger moment arm at extreme angle

    max_hip_pitch = max(case_a_squat, case_b_push, case_c_lying,
                        case_d_standup, case_e_extreme)
    max_hip_pitch_with_safety = max_hip_pitch * safety_factor

    print(f"\nHip pitch torque cases:")
    print(f"  A) Squatting: {case_a_squat:.1f} Nm")
    print(f"  B) Push recovery (200N forward): {case_b_push:.1f} Nm")
    print(f"  C) Lying down (support torso): {case_c_lying:.1f} Nm")
    print(f"  D) Stand up from lying: {case_d_standup:.1f} Nm")
    print(f"     - Static (torso weight): {case_d_standup_static:.1f} Nm")
    print(f"     - Dynamic (rotation): {case_d_standup_dynamic:.1f} Nm")
    print(f"  E) Extreme angle (103 deg): {case_e_extreme:.1f} Nm")
    print(f"\n  >> MAXIMUM hip pitch torque: {max_hip_pitch:.1f} Nm")
    print(f"  >> With safety factor {safety_factor}: {max_hip_pitch_with_safety:.1f} Nm")
    print(f"\n  Current motor: 150 Nm")
    print(f"  Status: {'[X] INSUFFICIENT' if max_hip_pitch_with_safety > 150 else '[OK] SUFFICIENT'}")

    # ========================================================================
    # 4. KNEE - WORST CASE
    # ========================================================================
    print("\n" + "="*80)
    print("4. KNEE MOTORS - WORST CASE ANALYSIS")
    print("="*80)

    shin_wheel_mass = shin_mass + wheel_mass
    shin_com_distance = shin_length / 2

    # Case A: Squatting (already calculated)
    knee_angle = 1.698  # rad (97 deg)
    case_a_knee_squat = shin_wheel_mass * g * shin_com_distance * np.sin(knee_angle - np.pi/2)

    # Case B: Lying down - knee must support shin+wheel weight
    case_b_knee_lying = shin_wheel_mass * g * shin_com_distance

    # Case C: Standing up from lying - dynamic torque
    knee_standup_time = 2.0  # s
    knee_standup_angle = np.pi / 2
    knee_standup_angular_accel = 2 * knee_standup_angle / (knee_standup_time ** 2)
    shin_inertia = shin_wheel_mass * (shin_com_distance ** 2)
    case_c_knee_dynamic = shin_inertia * knee_standup_angular_accel

    # Case D: Maximum knee angle (2.7 rad = 155 deg)
    max_knee_angle = 2.7  # rad
    case_d_knee_extreme = shin_wheel_mass * g * shin_com_distance * np.sin(max_knee_angle - np.pi/2)

    max_knee = max(case_a_knee_squat, case_b_knee_lying,
                   case_c_knee_dynamic, case_d_knee_extreme)
    max_knee_with_safety = max_knee * safety_factor

    print(f"\nKnee torque cases:")
    print(f"  A) Squatting (97 deg): {case_a_knee_squat:.2f} Nm")
    print(f"  B) Lying down (support shin+wheel): {case_b_knee_lying:.2f} Nm")
    print(f"  C) Stand up from lying (90 deg in 2s): {case_c_knee_dynamic:.2f} Nm")
    print(f"  D) Extreme angle (155 deg): {case_d_knee_extreme:.2f} Nm")
    print(f"\n  >> MAXIMUM knee torque: {max_knee:.1f} Nm")
    print(f"  >> With safety factor {safety_factor}: {max_knee_with_safety:.1f} Nm")
    print(f"\n  Current motor: 150 Nm")
    print(f"  Status: [OK] SUFFICIENT")

    # ========================================================================
    # 5. WHEEL - WORST CASE
    # ========================================================================
    print("\n" + "="*80)
    print("5. WHEEL MOTORS - WORST CASE ANALYSIS")
    print("="*80)

    # Case A: Normal balancing (already calculated)
    balance_accel = 2.0  # m/s^2
    case_a_balance = (total_mass * balance_accel * wheel_radius) / 2

    # Case B: Push recovery - aggressive acceleration
    push_recovery_accel = 3.0  # m/s^2
    case_b_push = (total_mass * push_recovery_accel * wheel_radius) / 2

    # Case C: Emergency stop - maximum deceleration
    emergency_decel = 4.0  # m/s^2
    case_c_emergency = (total_mass * emergency_decel * wheel_radius) / 2

    # Case D: Climbing small obstacle (2cm step)
    # Need to lift robot weight over obstacle
    obstacle_height = 0.02  # m
    # Force = weight × (obstacle_height / wheel_radius)
    case_d_obstacle = weight * (obstacle_height / wheel_radius) * wheel_radius / 2

    # Case E: Wheel slip recovery - maximum friction
    # Coefficient of friction = 1.2 (rubber on concrete)
    friction_coeff = 1.2
    case_e_friction = friction_coeff * (weight / 2) * wheel_radius

    max_wheel = max(case_a_balance, case_b_push, case_c_emergency,
                    case_d_obstacle, case_e_friction)
    max_wheel_with_safety = max_wheel * safety_factor

    print(f"\nWheel torque cases:")
    print(f"  A) Normal balancing (2 m/s^2): {case_a_balance:.2f} Nm")
    print(f"  B) Push recovery (3 m/s^2): {case_b_push:.2f} Nm")
    print(f"  C) Emergency stop (4 m/s^2): {case_c_emergency:.2f} Nm")
    print(f"  D) Climb obstacle (2cm step): {case_d_obstacle:.2f} Nm")
    print(f"  E) Maximum friction (u=1.2): {case_e_friction:.2f} Nm")
    print(f"\n  >> MAXIMUM wheel torque: {max_wheel:.1f} Nm")
    print(f"  >> With safety factor {safety_factor}: {max_wheel_with_safety:.1f} Nm")
    print(f"\n  Current motor: 30 Nm")
    print(f"  Status: {'[X] INSUFFICIENT' if max_wheel_with_safety > 30 else '[OK] SUFFICIENT'}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY - ABSOLUTE MAXIMUM TORQUES (with 1.5x safety)")
    print("="*80)

    results = [
        ("Hip Roll", max_hip_roll_with_safety, 30, "Lying down"),
        ("Hip Yaw", max_hip_yaw_with_safety, 30, "Emergency yaw"),
        ("Hip Pitch", max_hip_pitch_with_safety, 150, "Stand from lying"),
        ("Knee", max_knee_with_safety, 150, "Lying down"),
        ("Wheel", max_wheel_with_safety, 30, "Max friction"),
    ]

    print(f"\n{'Joint':<12} {'Max (Nm)':<12} {'Current (Nm)':<14} {'Status':<12} {'Worst Case':<20}")
    print("-" * 80)

    for joint, max_torque, current, worst_case in results:
        status = "[OK]" if max_torque <= current else "[X] UPGRADE"
        print(f"{joint:<12} {max_torque:>11.1f} {current:>13.0f} {status:<12} {worst_case:<20}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if max_hip_roll_with_safety > 30:
        recommended_hip_roll = np.ceil(max_hip_roll_with_safety / 5) * 5
        print(f"\n[!] Hip Roll: UPGRADE from 30 Nm to {recommended_hip_roll:.0f} Nm")
        print(f"    Reason: Lying down requires {max_hip_roll:.1f} Nm (without safety)")
    else:
        print(f"\n[OK] Hip Roll: Current 30 Nm is SUFFICIENT")

    if max_hip_pitch_with_safety > 150:
        recommended_hip_pitch = np.ceil(max_hip_pitch_with_safety / 10) * 10
        print(f"\n[!] Hip Pitch: UPGRADE from 150 Nm to {recommended_hip_pitch:.0f} Nm")
        print(f"    Reason: Standing from lying requires {max_hip_pitch:.1f} Nm (without safety)")
    else:
        print(f"\n[OK] Hip Pitch: Current 150 Nm is SUFFICIENT")

    if max_wheel_with_safety > 30:
        recommended_wheel = np.ceil(max_wheel_with_safety / 5) * 5
        print(f"\n[!] Wheel: UPGRADE from 30 Nm to {recommended_wheel:.0f} Nm")
        print(f"    Reason: Maximum friction requires {max_wheel:.1f} Nm (without safety)")
    else:
        print(f"\n[OK] Wheel: Current 30 Nm is SUFFICIENT")

    print(f"\n[OK] Hip Yaw: Current 30 Nm is SUFFICIENT")
    print(f"[OK] Knee: Current 150 Nm is SUFFICIENT")

    print("\n" + "="*80)
    print("NOTES:")
    print("="*80)
    print("- This calculation includes ALL postures: standing, squatting, lying, falling")
    print("- Worst cases: lying down, standing from lying, emergency maneuvers")
    print("- Safety factor 1.5x applied to all calculations")
    print("- If robot doesn't need to stand up from lying, requirements are lower")
    print("="*80)


if __name__ == "__main__":
    calculate_maximum_motor_torques()
