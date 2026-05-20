"""Analyze initial contact geometry to understand force asymmetry.

Even with perfect orientation (gravity=[0,0,-9.81]), contact forces are asymmetric.
This script investigates the geometric cause.
"""

import mujoco
import numpy as np

# Load model
model_path = "assets/robot/wheeled_biped_real.xml"
mj_model = mujoco.MjModel.from_xml_path(model_path)
mj_data = mujoco.MjData(mj_model)

# Initialize from keyframe
if mj_model.nkey > 0:
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    print("[OK] Robot initialized using keyframe 0")

# Forward kinematics
mujoco.mj_forward(mj_model, mj_data)

# Zero velocities explicitly
mj_data.qvel[:] = 0.0
mj_data.qacc[:] = 0.0

# Recompute with zero velocities
mujoco.mj_forward(mj_model, mj_data)

print("=" * 80)
print("Initial Contact Geometry Analysis")
print("=" * 80)

# Get body IDs
torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

# Get positions
torso_pos = mj_data.xpos[torso_id]
l_wheel_pos = mj_data.xpos[l_wheel_id]
r_wheel_pos = mj_data.xpos[r_wheel_id]

print("\n1. BODY POSITIONS")
print("-" * 80)
print(f"Torso:       [{torso_pos[0]:8.6f}, {torso_pos[1]:8.6f}, {torso_pos[2]:8.6f}]")
print(f"Left wheel:  [{l_wheel_pos[0]:8.6f}, {l_wheel_pos[1]:8.6f}, {l_wheel_pos[2]:8.6f}]")
print(f"Right wheel: [{r_wheel_pos[0]:8.6f}, {r_wheel_pos[1]:8.6f}, {r_wheel_pos[2]:8.6f}]")

# Compute CoM
total_mass = 0.0
com_pos = np.zeros(3)
for i in range(mj_model.nbody):
    body_mass = mj_model.body_mass[i]
    body_pos = mj_data.xpos[i]
    com_pos += body_mass * body_pos
    total_mass += body_mass
com_pos /= total_mass

print(f"\nCoM:         [{com_pos[0]:8.6f}, {com_pos[1]:8.6f}, {com_pos[2]:8.6f}]")
print(f"Total mass:  {total_mass:.3f} kg")

# Wheel positions relative to CoM
l_wheel_rel = l_wheel_pos - com_pos
r_wheel_rel = r_wheel_pos - com_pos

print("\n2. WHEEL POSITIONS RELATIVE TO CoM")
print("-" * 80)
print(f"Left wheel:  [{l_wheel_rel[0]:8.6f}, {l_wheel_rel[1]:8.6f}, {l_wheel_rel[2]:8.6f}]")
print(f"Right wheel: [{r_wheel_rel[0]:8.6f}, {r_wheel_rel[1]:8.6f}, {r_wheel_rel[2]:8.6f}]")
print(f"\nWheel spacing (x-axis): {abs(l_wheel_rel[0] - r_wheel_rel[0]):.6f} m")
print(f"Wheel y-offset difference: {abs(l_wheel_rel[1] - r_wheel_rel[1]):.6f} m")
print(f"Wheel z-offset difference: {abs(l_wheel_rel[2] - r_wheel_rel[2]):.6f} m")

# Check if CoM is centered between wheels
com_x_offset = com_pos[0] - (l_wheel_pos[0] + r_wheel_pos[0]) / 2.0
com_y_offset = com_pos[1] - (l_wheel_pos[1] + r_wheel_pos[1]) / 2.0

print(f"\nCoM offset from wheel centerline:")
print(f"  X-offset: {com_x_offset:8.6f} m (should be ~0)")
print(f"  Y-offset: {com_y_offset:8.6f} m (should be ~0)")

# Analyze contacts
print("\n3. CONTACT ANALYSIS")
print("-" * 80)
print(f"Number of contacts: {mj_data.ncon}")

left_contacts = []
right_contacts = []

for i in range(mj_data.ncon):
    contact = mj_data.contact[i]
    geom1_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
    geom2_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

    # Get contact force
    if i < len(mj_data.efc_force):
        force = mj_data.efc_force[i]
    else:
        force = 0.0

    # Get contact position
    contact_pos = contact.pos

    print(f"\nContact {i}:")
    print(f"  Geoms: {geom1_name} - {geom2_name}")
    print(f"  Position: [{contact_pos[0]:8.6f}, {contact_pos[1]:8.6f}, {contact_pos[2]:8.6f}]")
    print(f"  Force: {force:.2f} N")
    print(f"  Penetration depth: {contact.dist:.6f} m")

    if "l_wheel" in geom1_name or "l_wheel" in geom2_name:
        left_contacts.append((contact_pos, force, contact.dist))
    if "r_wheel" in geom1_name or "r_wheel" in geom2_name:
        right_contacts.append((contact_pos, force, contact.dist))

# Summarize contact forces
left_total = sum(f for _, f, _ in left_contacts)
right_total = sum(f for _, f, _ in right_contacts)

print("\n4. CONTACT FORCE SUMMARY")
print("-" * 80)
print(f"Left wheel contacts:  {len(left_contacts)}")
print(f"Right wheel contacts: {len(right_contacts)}")
print(f"Left total force:     {left_total:.2f} N")
print(f"Right total force:    {right_total:.2f} N")
print(f"Force asymmetry:      {abs(left_total - right_total):.2f} N")
print(f"Expected per wheel:   {total_mass * 9.81 / 2.0:.2f} N")

# Analyze penetration depths
if left_contacts:
    left_avg_depth = np.mean([d for _, _, d in left_contacts])
    print(f"\nLeft wheel avg penetration:  {left_avg_depth:.6f} m")
if right_contacts:
    right_avg_depth = np.mean([d for _, _, d in right_contacts])
    print(f"Right wheel avg penetration: {right_avg_depth:.6f} m")

if left_contacts and right_contacts:
    depth_diff = abs(left_avg_depth - right_avg_depth)
    print(f"Penetration depth difference: {depth_diff:.6f} m")

# Check joint positions
print("\n5. JOINT CONFIGURATION")
print("-" * 80)
joint_names = [
    "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
    "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"
]

for i, name in enumerate(joint_names):
    joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qpos_addr = mj_model.jnt_qposadr[joint_id]
    qpos = mj_data.qpos[qpos_addr]
    print(f"{name:15s}: {qpos:8.6f} rad")

# Check for asymmetry
l_hip_roll = mj_data.qpos[mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "l_hip_roll")]]
r_hip_roll = mj_data.qpos[mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "r_hip_roll")]]
l_hip_pitch = mj_data.qpos[mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "l_hip_pitch")]]
r_hip_pitch = mj_data.qpos[mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "r_hip_pitch")]]
l_knee = mj_data.qpos[mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "l_knee")]]
r_knee = mj_data.qpos[mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "r_knee")]]

print(f"\nJoint symmetry check:")
print(f"  Hip roll difference:  {abs(l_hip_roll - r_hip_roll):.6f} rad")
print(f"  Hip pitch difference: {abs(l_hip_pitch - r_hip_pitch):.6f} rad")
print(f"  Knee difference:      {abs(l_knee - r_knee):.6f} rad")

print("\n6. DIAGNOSIS")
print("=" * 80)
if abs(left_total - right_total) > 1.0:
    print("FINDING: Significant contact force asymmetry detected!")
    print(f"  Left: {left_total:.2f} N, Right: {right_total:.2f} N")
    print(f"  Asymmetry: {abs(left_total - right_total):.2f} N")
    print()

    if abs(com_x_offset) > 0.001:
        print(f"CAUSE 1: CoM is offset from wheel centerline by {com_x_offset:.6f} m in X")
        print("  This creates a roll moment that requires asymmetric contact forces")

    if abs(com_y_offset) > 0.001:
        print(f"CAUSE 2: CoM is offset from wheel centerline by {com_y_offset:.6f} m in Y")
        print("  This creates a pitch moment that affects contact distribution")

    if left_contacts and right_contacts:
        if abs(left_avg_depth - right_avg_depth) > 0.0001:
            print(f"CAUSE 3: Unequal wheel penetration depths")
            print(f"  Left: {left_avg_depth:.6f} m, Right: {right_avg_depth:.6f} m")
            print("  This indicates wheels are not at the same height")

    if abs(l_hip_roll - r_hip_roll) > 0.001:
        print(f"CAUSE 4: Hip roll joints are not symmetric")
        print(f"  Left: {l_hip_roll:.6f} rad, Right: {r_hip_roll:.6f} rad")

    print()
    print("RECOMMENDATION:")
    print("  The keyframe needs to be adjusted to achieve symmetric contact forces.")
    print("  Options:")
    print("  1. Adjust root position to center CoM over wheel centerline")
    print("  2. Adjust hip roll joints to compensate for geometric asymmetry")
    print("  3. Adjust root height to equalize wheel penetration depths")
else:
    print("Contact forces are symmetric - no adjustment needed")

print("=" * 80)
