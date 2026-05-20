"""Adjust keyframe to move CoM forward by changing leg configuration.

Reduces hip pitch angles to lean torso forward, moving CoM toward wheel center.
"""

import mujoco
import numpy as np


def adjust_com_forward(
    model_path: str,
    keyframe_id: int = 0,
    target_sagittal_offset: float = 0.0,
    hip_pitch_adjustment_step: float = 0.05,  # radians per iteration
    max_iterations: int = 30,
):
    """Adjust hip pitch to move CoM forward.

    Args:
        model_path: Path to MuJoCo XML model
        keyframe_id: Keyframe index to adjust
        target_sagittal_offset: Target CoM offset in Y direction (meters)
        hip_pitch_adjustment_step: Hip pitch change per iteration (radians)
        max_iterations: Maximum adjustment iterations

    Returns:
        Adjusted qpos and ctrl arrays
    """
    # Load model and data
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Load initial keyframe
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, keyframe_id)
    mujoco.mj_forward(mj_model, mj_data)

    print("=" * 80)
    print("Adjusting leg configuration to move CoM forward")
    print("=" * 80)

    # Get initial state
    qpos = mj_data.qpos.copy()

    # Get wheel body IDs
    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    # Initial measurements
    com_pos = mj_data.subtree_com[1]
    l_wheel_pos = mj_data.xpos[l_wheel_id]
    r_wheel_pos = mj_data.xpos[r_wheel_id]
    wheel_center = (l_wheel_pos + r_wheel_pos) / 2.0
    initial_offset = com_pos[1] - wheel_center[1]

    print(f"\nInitial state:")
    print(f"  CoM sagittal offset: {initial_offset*1000:.2f} mm")
    print(f"  CoM height: {com_pos[2]:.4f} m")
    print(f"  Hip pitch: L={qpos[9]:.4f} rad ({np.degrees(qpos[9]):.2f} deg), R={qpos[14]:.4f} rad ({np.degrees(qpos[14]):.2f} deg)")
    print(f"  Target offset: {target_sagittal_offset*1000:.2f} mm")
    print(f"  Adjustment needed: {(target_sagittal_offset - initial_offset)*1000:.2f} mm")

    # Hip pitch joint indices in qpos: 9 (left), 14 (right)
    # qpos layout: [root_pos(3), root_quat(4), joints(10)]
    # joints: [l_hip_roll, l_hip_yaw, l_hip_pitch, l_knee, l_wheel, r_hip_roll, r_hip_yaw, r_hip_pitch, r_knee, r_wheel]
    l_hip_pitch_idx = 9
    r_hip_pitch_idx = 14

    # Iteratively adjust hip pitch
    for iteration in range(max_iterations):
        # Compute current offset
        mj_data.qpos[:] = qpos
        mujoco.mj_forward(mj_model, mj_data)

        com_pos = mj_data.subtree_com[1]
        l_wheel_pos = mj_data.xpos[l_wheel_id]
        r_wheel_pos = mj_data.xpos[r_wheel_id]
        wheel_center = (l_wheel_pos + r_wheel_pos) / 2.0

        current_offset = com_pos[1] - wheel_center[1]
        error = target_sagittal_offset - current_offset

        print(f"\nIteration {iteration}:")
        print(f"  Current offset: {current_offset*1000:.2f} mm")
        print(f"  Error: {error*1000:.2f} mm")
        print(f"  CoM height: {com_pos[2]:.4f} m")
        print(f"  Hip pitch: L={qpos[l_hip_pitch_idx]:.4f} rad ({np.degrees(qpos[l_hip_pitch_idx]):.2f} deg)")

        # Check convergence
        if abs(error) < 0.010:  # 10mm tolerance
            print(f"  [OK] Converged!")
            break

        # Determine adjustment direction
        # CORRECTED: Increasing hip pitch (more forward lean) moves CoM forward
        if error > 0:  # Need to move CoM forward
            adjustment = hip_pitch_adjustment_step  # Increase hip pitch
        else:  # Need to move CoM backward
            adjustment = -hip_pitch_adjustment_step  # Decrease hip pitch

        # Apply adjustment to both hips symmetrically
        qpos[l_hip_pitch_idx] += adjustment
        qpos[r_hip_pitch_idx] += adjustment

        print(f"  Hip pitch adjustment: {adjustment:.4f} rad ({np.degrees(adjustment):.2f} deg)")

    # Final measurements
    mj_data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, mj_data)

    com_pos = mj_data.subtree_com[1]
    l_wheel_pos = mj_data.xpos[l_wheel_id]
    r_wheel_pos = mj_data.xpos[r_wheel_id]
    wheel_center = (l_wheel_pos + r_wheel_pos) / 2.0
    final_offset = com_pos[1] - wheel_center[1]

    # Measure contact forces
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    left_force = 0.0
    right_force = 0.0

    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        force_contact = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)
        frame = np.array(contact.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]

        if geom1 == l_wheel_geom_id or geom2 == l_wheel_geom_id:
            left_force += force_world[2]
        if geom1 == r_wheel_geom_id or geom2 == r_wheel_geom_id:
            right_force += force_world[2]

    print(f"\n" + "=" * 80)
    print("Final adjusted keyframe:")
    print("=" * 80)
    print(f"  CoM sagittal offset: {final_offset*1000:.2f} mm (change: {(final_offset - initial_offset)*1000:.2f} mm)")
    print(f"  CoM height: {com_pos[2]:.4f} m")
    print(f"  Hip pitch: L={qpos[l_hip_pitch_idx]:.4f} rad ({np.degrees(qpos[l_hip_pitch_idx]):.2f} deg)")
    print(f"  Hip pitch: R={qpos[r_hip_pitch_idx]:.4f} rad ({np.degrees(qpos[r_hip_pitch_idx]):.2f} deg)")
    print(f"  Left contact: {left_force:.2f} N")
    print(f"  Right contact: {right_force:.2f} N")
    print(f"  Asymmetry: {abs(left_force - right_force):.2f} N")

    # Generate control targets (zero for passive standing)
    ctrl = np.zeros(mj_model.nu)

    return qpos, ctrl


def format_keyframe_xml(qpos: np.ndarray, ctrl: np.ndarray) -> str:
    """Format adjusted configuration as MuJoCo keyframe XML."""
    root_pos = qpos[:3]
    root_quat = qpos[3:7]
    joint_pos = qpos[7:]

    xml = f"""  <keyframe>
    <key name="standing"
         qpos="{root_pos[0]:.6f} {root_pos[1]:.6f} {root_pos[2]:.6f}
               {root_quat[0]:.6f} {root_quat[1]:.6f} {root_quat[2]:.6f} {root_quat[3]:.6f}
               {joint_pos[0]:.6f} {joint_pos[1]:.6f} {joint_pos[2]:.6f} {joint_pos[3]:.6f} {joint_pos[4]:.6f}
               {joint_pos[5]:.6f} {joint_pos[6]:.6f} {joint_pos[7]:.6f} {joint_pos[8]:.6f} {joint_pos[9]:.6f}"
         ctrl="{ctrl[0]:.6f} {ctrl[1]:.6f} {ctrl[2]:.6f} {ctrl[3]:.6f} {ctrl[4]:.6f}
               {ctrl[5]:.6f} {ctrl[6]:.6f} {ctrl[7]:.6f} {ctrl[8]:.6f} {ctrl[9]:.6f}"/>
  </keyframe>"""

    return xml


if __name__ == "__main__":
    model_path = "assets/robot/wheeled_biped_real.xml"

    # Adjust keyframe to center CoM over wheel base
    qpos_adjusted, ctrl_adjusted = adjust_com_forward(
        model_path,
        keyframe_id=0,
        target_sagittal_offset=0.0,  # Center CoM over wheels
        hip_pitch_adjustment_step=0.05,  # 2.86 degrees per step
        max_iterations=30,
    )

    print("\n" + "=" * 80)
    print("Adjusted keyframe XML:")
    print("=" * 80)
    print(format_keyframe_xml(qpos_adjusted, ctrl_adjusted))
