"""Adjust keyframe to center CoM sagittally over wheel base.

Takes existing keyframe and adjusts hip pitch angles to move CoM forward.
"""

import mujoco
import numpy as np


def adjust_keyframe_sagittal(
    model_path: str,
    keyframe_id: int = 0,
    target_sagittal_offset: float = 0.0,
    max_iterations: int = 20,
):
    """Adjust keyframe to achieve target sagittal CoM offset.

    Args:
        model_path: Path to MuJoCo XML model
        keyframe_id: Keyframe index to adjust
        target_sagittal_offset: Target CoM offset in Y direction (meters)
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
    print("Adjusting keyframe sagittal balance")
    print("=" * 80)

    # Get initial state
    qpos = mj_data.qpos.copy()

    # Get wheel positions
    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    l_wheel_pos = mj_data.xpos[l_wheel_id]
    r_wheel_pos = mj_data.xpos[r_wheel_id]
    wheel_center = (l_wheel_pos + r_wheel_pos) / 2.0

    # Initial CoM
    com_pos = mj_data.subtree_com[1]
    initial_offset = com_pos[1] - wheel_center[1]

    print(f"\nInitial state:")
    print(f"  CoM sagittal offset: {initial_offset*1000:.2f} mm")
    print(f"  Target offset: {target_sagittal_offset*1000:.2f} mm")
    print(f"  Adjustment needed: {(target_sagittal_offset - initial_offset)*1000:.2f} mm")

    # Iteratively adjust root Y position to move entire robot forward
    # This preserves leg configuration while shifting CoM
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
        print(f"  Root Y: {qpos[1]:.6f} m")

        # Check convergence
        if abs(error) < 0.005:  # 5mm tolerance
            print(f"  [OK] Converged!")
            break

        # Adjust root Y position
        # Moving root forward by X moves CoM forward by approximately X
        # Use conservative gain to avoid overshooting
        adjustment = error * 0.8

        qpos[1] += adjustment  # Root Y position (index 1)

        print(f"  Root Y adjustment: {adjustment*1000:.2f} mm")
        print(f"  New root Y: {qpos[1]:.6f} m")

    # Final forward kinematics
    mj_data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, mj_data)

    # Measure final contact forces
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    left_wheel_force = 0.0
    right_wheel_force = 0.0

    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        force_contact = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)
        frame = np.array(contact.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]

        if geom1 == l_wheel_geom_id or geom2 == l_wheel_geom_id:
            left_wheel_force += force_world[2]
        if geom1 == r_wheel_geom_id or geom2 == r_wheel_geom_id:
            right_wheel_force += force_world[2]

    # Final state
    com_pos = mj_data.subtree_com[1]
    wheel_center = (mj_data.xpos[l_wheel_id] + mj_data.xpos[r_wheel_id]) / 2.0
    final_offset = com_pos[1] - wheel_center[1]

    print(f"\n" + "=" * 80)
    print("Final adjusted keyframe:")
    print("=" * 80)
    print(f"  CoM sagittal offset: {final_offset*1000:.2f} mm")
    print(f"  CoM height: {com_pos[2]:.4f} m")
    print(f"  Left contact force: {left_wheel_force:.2f} N")
    print(f"  Right contact force: {right_wheel_force:.2f} N")
    print(f"  Asymmetry: {abs(left_wheel_force - right_wheel_force):.2f} N")

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
    qpos_adjusted, ctrl_adjusted = adjust_keyframe_sagittal(
        model_path,
        keyframe_id=0,
        target_sagittal_offset=0.0,  # Center CoM over wheels
        max_iterations=20,
    )

    print("\n" + "=" * 80)
    print("Adjusted keyframe XML:")
    print("=" * 80)
    print(format_keyframe_xml(qpos_adjusted, ctrl_adjusted))
