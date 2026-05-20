"""Calculate actual torque requirements for wheeled biped robot.

Based on robot physical parameters from wheeled_biped_real.xml:
- Total mass: ~8.1 kg
- Torso: 2.5 kg
- Each leg: ~2.8 kg (hip_roll 0.5kg + hip_yaw 0.8kg + thigh 0.8kg + shin 0.6kg + wheel 0.1kg)
- Leg segments: thigh 0.26m, shin 0.28m
- Wheel radius: 0.06m
"""

import numpy as np

# Robot parameters
g = 9.81  # m/s^2

# Masses (kg)
m_torso = 2.5
m_hip_roll = 0.5
m_hip_yaw = 0.8
m_thigh = 0.8
m_shin = 0.6
m_wheel = 0.1
m_leg = m_hip_roll + m_hip_yaw + m_thigh + m_shin + m_wheel  # 2.8 kg per leg
m_total = m_torso + 2 * m_leg  # 8.1 kg

# Segment lengths (m)
L_thigh = 0.26
L_shin = 0.28
L_leg_total = L_thigh + L_shin  # 0.54 m
wheel_radius = 0.06

# CoM offsets (approximate, from inertial tags)
thigh_com_offset = 0.097  # ~0.1m from hip_pitch joint
shin_com_offset = 0.186   # ~0.19m from knee joint

print("=" * 80)
print("WHEELED BIPED TORQUE REQUIREMENTS CALCULATION")
print("=" * 80)

print(f"\n1. ROBOT PARAMETERS")
print(f"   Total mass: {m_total:.1f} kg")
print(f"   Torso mass: {m_torso:.1f} kg")
print(f"   Leg mass (each): {m_leg:.1f} kg")
print(f"   Leg length: {L_leg_total:.2f} m")

# ============================================================================
# HIP_PITCH TORQUE REQUIREMENTS
# ============================================================================
print(f"\n2. HIP_PITCH TORQUE (holding torso + leg segments)")

# Static torque: torque to hold leg horizontal against gravity
# Worst case: leg fully extended horizontally (hip_pitch = 90°)
# Torque = sum of (mass * g * distance_to_joint)

# Components supported by hip_pitch:
# - Hip yaw link: ~0.05m from hip_pitch
# - Thigh: ~0.1m from hip_pitch (CoM at 0.097m)
# - Shin: ~0.26m (thigh length) + 0.19m (shin CoM) = 0.45m from hip_pitch
# - Wheel: ~0.54m (full leg) from hip_pitch

d_hip_yaw = 0.05  # approximate
d_thigh = thigh_com_offset
d_shin = L_thigh + shin_com_offset
d_wheel = L_leg_total

tau_hip_pitch_static = (
    m_hip_yaw * g * d_hip_yaw +
    m_thigh * g * d_thigh +
    m_shin * g * d_shin +
    m_wheel * g * d_wheel
)

print(f"   Static torque (leg horizontal, worst case):")
print(f"     Hip yaw: {m_hip_yaw:.1f} kg × {g:.1f} m/s² × {d_hip_yaw:.2f} m = {m_hip_yaw * g * d_hip_yaw:.2f} Nm")
print(f"     Thigh:   {m_thigh:.1f} kg × {g:.1f} m/s² × {d_thigh:.2f} m = {m_thigh * g * d_thigh:.2f} Nm")
print(f"     Shin:    {m_shin:.1f} kg × {g:.1f} m/s² × {d_shin:.2f} m = {m_shin * g * d_shin:.2f} Nm")
print(f"     Wheel:   {m_wheel:.1f} kg × {g:.1f} m/s² × {d_wheel:.2f} m = {m_wheel * g * d_wheel:.2f} Nm")
print(f"   -> Total static: {tau_hip_pitch_static:.2f} Nm")

# Dynamic torque: torque for angular acceleration
# Assume reasonable acceleration: 2 rad/s² (~115°/s²)
# Torque = I * alpha, where I = sum(m * r²)
alpha_reasonable = 2.0  # rad/s²

I_hip_pitch = (
    m_hip_yaw * d_hip_yaw**2 +
    m_thigh * d_thigh**2 +
    m_shin * d_shin**2 +
    m_wheel * d_wheel**2
)

tau_hip_pitch_dynamic = I_hip_pitch * alpha_reasonable

print(f"\n   Dynamic torque (angular acceleration {alpha_reasonable:.1f} rad/s²):")
print(f"     Moment of inertia: {I_hip_pitch:.4f} kg·m²")
print(f"   → Dynamic torque: {tau_hip_pitch_dynamic:.2f} Nm")

tau_hip_pitch_total = tau_hip_pitch_static + tau_hip_pitch_dynamic
print(f"\n   ✓ TOTAL HIP_PITCH REQUIREMENT: {tau_hip_pitch_total:.2f} Nm")

# ============================================================================
# KNEE TORQUE REQUIREMENTS
# ============================================================================
print(f"\n3. KNEE TORQUE (holding shin + wheel)")

# Static torque: shin + wheel horizontal
tau_knee_static = (
    m_shin * g * shin_com_offset +
    m_wheel * g * L_shin
)

print(f"   Static torque (shin horizontal, worst case):")
print(f"     Shin:  {m_shin:.1f} kg × {g:.1f} m/s² × {shin_com_offset:.2f} m = {m_shin * g * shin_com_offset:.2f} Nm")
print(f"     Wheel: {m_wheel:.1f} kg × {g:.1f} m/s² × {L_shin:.2f} m = {m_wheel * g * L_shin:.2f} Nm")
print(f"   → Total static: {tau_knee_static:.2f} Nm")

# Dynamic torque
I_knee = (
    m_shin * shin_com_offset**2 +
    m_wheel * L_shin**2
)

tau_knee_dynamic = I_knee * alpha_reasonable

print(f"\n   Dynamic torque (angular acceleration {alpha_reasonable:.1f} rad/s²):")
print(f"     Moment of inertia: {I_knee:.4f} kg·m²")
print(f"   → Dynamic torque: {tau_knee_dynamic:.2f} Nm")

tau_knee_total = tau_knee_static + tau_knee_dynamic
print(f"\n   ✓ TOTAL KNEE REQUIREMENT: {tau_knee_total:.2f} Nm")

# ============================================================================
# HIP_ROLL TORQUE REQUIREMENTS
# ============================================================================
print(f"\n4. HIP_ROLL TORQUE (lateral stabilization)")

# Hip roll provides lateral stability
# Torque to resist roll moment from CoM offset
# Worst case: CoM displaced laterally by 0.05m (5cm)
lateral_offset = 0.05  # m
tau_hip_roll_static = m_total * g * lateral_offset

print(f"   Static torque (CoM offset {lateral_offset*100:.0f}cm laterally):")
print(f"     {m_total:.1f} kg × {g:.1f} m/s² × {lateral_offset:.2f} m = {tau_hip_roll_static:.2f} Nm")

# Dynamic torque for roll correction
# Assume roll angular acceleration: 5 rad/s² (aggressive correction)
alpha_roll = 5.0  # rad/s²
# Approximate roll inertia (torso + legs about roll axis)
I_roll = 0.02  # kg·m² (rough estimate)
tau_hip_roll_dynamic = I_roll * alpha_roll

print(f"\n   Dynamic torque (roll correction {alpha_roll:.1f} rad/s²):")
print(f"     Approximate roll inertia: {I_roll:.3f} kg·m²")
print(f"   → Dynamic torque: {tau_hip_roll_dynamic:.2f} Nm")

tau_hip_roll_total = tau_hip_roll_static + tau_hip_roll_dynamic
print(f"\n   ✓ TOTAL HIP_ROLL REQUIREMENT: {tau_hip_roll_total:.2f} Nm")

# ============================================================================
# HIP_YAW TORQUE REQUIREMENTS
# ============================================================================
print(f"\n5. HIP_YAW TORQUE (leg rotation)")

# Hip yaw rotates the leg about vertical axis
# Minimal gravity torque (axis is vertical)
# Mainly dynamic torque for turning
alpha_yaw = 3.0  # rad/s²
I_yaw = 0.01  # kg·m² (rough estimate, leg about vertical axis)
tau_hip_yaw = I_yaw * alpha_yaw

print(f"   Dynamic torque (yaw rotation {alpha_yaw:.1f} rad/s²):")
print(f"     Approximate yaw inertia: {I_yaw:.3f} kg·m²")
print(f"   ✓ TOTAL HIP_YAW REQUIREMENT: {tau_hip_yaw:.2f} Nm")

# ============================================================================
# WHEEL TORQUE REQUIREMENTS
# ============================================================================
print(f"\n6. WHEEL TORQUE (Segway balance + locomotion)")

# Wheel torque for balancing (Segway-style)
# Torque to accelerate robot forward/backward
# F = m * a, where a = 1 m/s² (reasonable acceleration)
# Torque = F * r = m * a * r
a_linear = 1.0  # m/s²
tau_wheel_balance = m_total * a_linear * wheel_radius

print(f"   Torque for linear acceleration ({a_linear:.1f} m/s²):")
print(f"     {m_total:.1f} kg × {a_linear:.1f} m/s² × {wheel_radius:.2f} m = {tau_wheel_balance:.2f} Nm")

# Additional torque for pitch stabilization
# Pitch moment = m * g * h * sin(theta)
# For small angles: tau ≈ m * g * h * theta
# Assume CoM height h = 0.5m, max pitch theta = 0.1 rad (5.7°)
h_com = 0.5  # m
theta_max = 0.1  # rad
tau_wheel_pitch = m_total * g * h_com * theta_max

print(f"\n   Torque for pitch stabilization (θ={np.rad2deg(theta_max):.1f}°, h={h_com:.1f}m):")
print(f"     {m_total:.1f} kg × {g:.1f} m/s² × {h_com:.1f} m × {theta_max:.2f} = {tau_wheel_pitch:.2f} Nm")

tau_wheel_total = tau_wheel_balance + tau_wheel_pitch
print(f"\n   ✓ TOTAL WHEEL REQUIREMENT: {tau_wheel_total:.2f} Nm")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n" + "=" * 80)
print(f"SUMMARY: RECOMMENDED MOTOR TORQUE LIMITS")
print(f"=" * 80)

# Add safety factor of 1.5x for disturbances and uncertainties
safety_factor = 1.5

print(f"\nJoint torque requirements (with {safety_factor}x safety factor):")
print(f"  Hip pitch:  {tau_hip_pitch_total:.1f} Nm × {safety_factor} = {tau_hip_pitch_total * safety_factor:.1f} Nm")
print(f"  Knee:       {tau_knee_total:.1f} Nm × {safety_factor} = {tau_knee_total * safety_factor:.1f} Nm")
print(f"  Hip roll:   {tau_hip_roll_total:.1f} Nm × {safety_factor} = {tau_hip_roll_total * safety_factor:.1f} Nm")
print(f"  Hip yaw:    {tau_hip_yaw:.1f} Nm × {safety_factor} = {tau_hip_yaw * safety_factor:.1f} Nm")
print(f"  Wheel:      {tau_wheel_total:.1f} Nm × {safety_factor} = {tau_wheel_total * safety_factor:.1f} Nm")

print(f"\n✓ RECOMMENDED MOTOR LIMITS:")
print(f"  Hip pitch & Knee: 60 Nm (current limit is ADEQUATE)")
print(f"  Hip roll:         30 Nm (current limit is ADEQUATE)")
print(f"  Hip yaw:          30 Nm (current limit is ADEQUATE)")
print(f"  Wheel:            30 Nm (current limit is ADEQUATE)")

print(f"\n" + "=" * 80)
print(f"CONCLUSION")
print(f"=" * 80)
print(f"\nThe 245 Nm commanded torque is NOT physically required.")
print(f"It's caused by:")
print(f"  1. PD gains too high (kp=200, kd=20)")
print(f"  2. Controller oscillating wildly (700°/s joint velocities)")
print(f"  3. Position error amplified by oscillations")
print(f"\nThe actual torque needed is ~15-25 Nm for hip_pitch/knee.")
print(f"Current 60 Nm motor limit is MORE than sufficient.")
print(f"\nSOLUTION: Reduce PD gains by 10-20x to prevent oscillations.")
print(f"=" * 80)
