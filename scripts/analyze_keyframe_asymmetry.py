"""Analyze existing keyframe to diagnose contact force asymmetry.

Loads the keyframe, computes contact forces, and suggests adjustments
to balance the robot's center of mass.
"""

import mujoco
import numpy as np


def analyze_keyframe(model_path: str, keyframe_id: int = 0):
    """Analyze keyframe contact forces and CoM position.

    Args:
        model_path: Path to MuJoCo XML model
        keyframe_id: Keyframe index to analyze
    """
    # Load model and data
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Load keyframe
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, keyframe_id)
    mujoco.mj_forward(mj_model, mj_data)

    print("=" * 80)
    print(f"Keyframe {keyframe_id} Analysis")
    print("=" * 80)

    # Root state
    root_pos = mj_data.qpos[:3]
    root_quat = mj_data.qpos[3:7]
    joint_pos = mj_data.qpos[7:]

    print(f"\nRoot state:")
    print(f"  Position: {root_pos}")
    print(f"  Quaternion: {root_quat}")
    print(f"  Joint positions: {joint_pos}")

    # CoM position
    com_pos = mj_data.subtree_com[1]  # Robot subtree CoM
    print(f"\nCenter of Mass:")
    print(f"  Position: {com_pos}")
    print(f"  Height: {com_pos[2]:.4f} m")

    # Get wheel body positions
    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    l_wheel_pos = mj_data.xpos[l_wheel_id]
    r_wheel_pos = mj_data.xpos[r_wheel_id]

    print(f"\nWheel positions:")
    print(f"  Left wheel:  {l_wheel_pos}")
    print(f"  Right wheel: {r_wheel_pos}")
    print(f"  Wheel separation: {np.linalg.norm(r_wheel_pos - l_wheel_pos):.4f} m")

    # Wheel center (support polygon center)
    wheel_center = (l_wheel_pos + r_wheel_pos) / 2.0
    print(f"  Wheel center: {wheel_center}")

    # CoM offset from wheel center
    com_offset = com_pos - wheel_center
    print(f"\nCoM offset from wheel center:")
    print(f"  Lateral (x): {com_offset[0]*1000:.2f} mm")
    print(f"  Sagittal (y): {com_offset[1]*1000:.2f} mm")
    print(f"  Vertical (z): {com_offset[2]*1000:.2f} mm")

    # Contact forces
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    left_wheel_force = 0.0
    right_wheel_force = 0.0

    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        # Compute contact force in world frame
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)
        frame = np.array(contact.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]

        if geom1 == l_wheel_geom_id or geom2 == l_wheel_geom_id:
            left_wheel_force += force_world[2]
        if geom1 == r_wheel_geom_id or geom2 == r_wheel_geom_id:
            right_wheel_force += force_world[2]

    total_force = left_wheel_force + right_wheel_force
    robot_mass = np.sum(mj_model.body_mass)
    expected_weight = robot_mass * 9.81

    print(f"\nContact forces:")
    print(f"  Left wheel:  {left_wheel_force:.2f} N")
    print(f"  Right wheel: {right_wheel_force:.2f} N")
    print(f"  Total:       {total_force:.2f} N")
    print(f"  Expected:    {expected_weight:.2f} N")
    print(f"  Asymmetry:   {abs(left_wheel_force - right_wheel_force):.2f} N")

    if total_force > 1.0:
        left_pct = (left_wheel_force / total_force) * 100.0
        right_pct = (right_wheel_force / total_force) * 100.0
        print(f"  Left %:      {left_pct:.1f}%")
        print(f"  Right %:     {right_pct:.1f}%")

    # Diagnose asymmetry
    print(f"\n" + "=" * 80)
    print("Diagnosis:")
    print("=" * 80)

    if abs(left_wheel_force - right_wheel_force) > 5.0:
        print(f"[PROBLEM] Large contact force asymmetry detected!")

        if left_wheel_force < right_wheel_force:
            print(f"  -> Right wheel has more load ({right_wheel_force:.2f} N vs {left_wheel_force:.2f} N)")
            print(f"  -> CoM is shifted toward RIGHT wheel")
            print(f"  -> Need to shift CoM LEFT to balance")

            # Suggest adjustments
            print(f"\nSuggested adjustments:")
            print(f"  1. Increase left hip roll (shift CoM left)")
            print(f"  2. Decrease right hip roll (shift CoM left)")
            print(f"  3. Check root position lateral offset")

        else:
            print(f"  -> Left wheel has more load ({left_wheel_force:.2f} N vs {right_wheel_force:.2f} N)")
            print(f"  -> CoM is shifted toward LEFT wheel")
            print(f"  -> Need to shift CoM RIGHT to balance")

            print(f"\nSuggested adjustments:")
            print(f"  1. Decrease left hip roll (shift CoM right)")
            print(f"  2. Increase right hip roll (shift CoM right)")
            print(f"  3. Check root position lateral offset")
    else:
        print(f"[OK] Contact forces are reasonably balanced")

    # Check sagittal balance
    if abs(com_offset[1]) > 0.01:  # More than 10mm sagittal offset
        print(f"\n[WARNING] CoM has sagittal offset: {com_offset[1]*1000:.2f} mm")
        if com_offset[1] > 0:
            print(f"  -> CoM is FORWARD of wheel center")
            print(f"  -> Robot will tend to pitch forward")
        else:
            print(f"  -> CoM is BACKWARD of wheel center")
            print(f"  -> Robot will tend to pitch backward")

    return {
        "com_pos": com_pos,
        "wheel_center": wheel_center,
        "com_offset": com_offset,
        "left_force": left_wheel_force,
        "right_force": right_wheel_force,
        "asymmetry": abs(left_wheel_force - right_wheel_force),
    }


if __name__ == "__main__":
    model_path = "assets/robot/wheeled_biped_real.xml"
    analyze_keyframe(model_path, keyframe_id=0)
