"""Adjust keyframe to eliminate initial roll error after mj_forward.

After adjusting hip pitch and root Z, the keyframe develops 2.6° roll after mj_forward.
This script adjusts the keyframe to achieve 0° roll after dynamics settle.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def adjust_roll_error(
    model_path: str,
    keyframe_id: int = 0,
    target_roll_deg: float = 0.0,
    max_iterations: int = 50,
    tolerance_deg: float = 0.1,
):
    """Adjust keyframe to eliminate roll error after mj_forward.

    Strategy: Adjust root quaternion to pre-compensate for roll that appears
    after mj_forward due to dynamics.

    Args:
        model_path: Path to MuJoCo XML model
        keyframe_id: Keyframe index to adjust
        target_roll_deg: Target roll angle in degrees
        max_iterations: Maximum adjustment iterations
        tolerance_deg: Acceptable roll error in degrees

    Returns:
        Adjusted qpos and ctrl arrays
    """
    # Load model and data
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Load initial keyframe
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, keyframe_id)

    print("=" * 80)
    print("Adjusting keyframe to eliminate roll error after mj_forward")
    print("=" * 80)

    # Get initial state
    qpos = mj_data.qpos.copy()
    initial_quat = qpos[3:7].copy()

    # Get wheel body IDs for CoM measurement
    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    # Get wheel geom IDs for contact force measurement
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    def measure_state():
        """Measure roll, pitch, and contact forces after mj_forward."""
        mujoco.mj_forward(mj_model, mj_data)

        # Extract roll and pitch from quaternion
        quat = mj_data.qpos[3:7]
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # xyzw format
        euler = rot.as_euler('xyz', degrees=True)
        roll_deg = euler[0]
        pitch_deg = euler[1]

        # Measure CoM position
        com_pos = mj_data.subtree_com[1]
        l_wheel_pos = mj_data.xpos[l_wheel_id]
        r_wheel_pos = mj_data.xpos[r_wheel_id]
        wheel_center = (l_wheel_pos + r_wheel_pos) / 2.0
        sagittal_offset = com_pos[1] - wheel_center[1]
        lateral_offset = com_pos[0] - wheel_center[0]

        # Measure contact forces
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

        total_force = left_force + right_force
        asymmetry = abs(left_force - right_force)

        return roll_deg, pitch_deg, sagittal_offset, lateral_offset, left_force, right_force, total_force, asymmetry

    # Initial measurements
    print(f"\nInitial state:")
    print(f"  Root quaternion: [{initial_quat[0]:.6f}, {initial_quat[1]:.6f}, {initial_quat[2]:.6f}, {initial_quat[3]:.6f}]")

    mj_data.qpos[:] = qpos
    roll_deg, pitch_deg, sagittal_offset, lateral_offset, left_force, right_force, total_force, asymmetry = measure_state()

    print(f"  Roll after mj_forward: {roll_deg:.2f}°")
    print(f"  Pitch after mj_forward: {pitch_deg:.2f}°")
    print(f"  CoM sagittal offset: {sagittal_offset*1000:.2f} mm")
    print(f"  CoM lateral offset: {lateral_offset*1000:.2f} mm")
    print(f"  Left contact: {left_force:.2f} N")
    print(f"  Right contact: {right_force:.2f} N")
    print(f"  Total force: {total_force:.2f} N")
    print(f"  Asymmetry: {asymmetry:.2f} N")
    print(f"  Target roll: {target_roll_deg:.2f}°")
    print(f"  Roll error: {roll_deg - target_roll_deg:.2f}°")

    # Iteratively adjust root quaternion to compensate for roll error
    print(f"\nAdjusting root quaternion to eliminate roll error...")

    best_quat = initial_quat.copy()
    best_error = abs(roll_deg - target_roll_deg)

    for iteration in range(max_iterations):
        # Measure current roll
        mj_data.qpos[:] = qpos
        roll_deg, pitch_deg, sagittal_offset, lateral_offset, left_force, right_force, total_force, asymmetry = measure_state()

        roll_error = roll_deg - target_roll_deg

        print(f"\nIteration {iteration}:")
        print(f"  Current roll: {roll_deg:.2f}°")
        print(f"  Roll error: {roll_error:.2f}°")
        print(f"  Pitch: {pitch_deg:.2f}°")
        print(f"  CoM lateral offset: {lateral_offset*1000:.2f} mm")
        print(f"  Contact asymmetry: {asymmetry:.2f} N")

        # Update best if improved
        if abs(roll_error) < best_error:
            best_error = abs(roll_error)
            best_quat = qpos[3:7].copy()
            print(f"  [NEW BEST] Roll error improved to {best_error:.2f}°")

        # Check convergence
        if abs(roll_error) < tolerance_deg:
            print(f"  [OK] Target roll achieved!")
            break

        # Compute roll correction
        # Negative feedback: if roll is positive, apply negative roll correction
        correction_deg = -roll_error * 0.5  # Damped correction to avoid overshoot

        # Convert current quaternion to rotation
        current_quat = qpos[3:7]
        current_rot = Rotation.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])

        # Apply roll correction
        correction_rot = Rotation.from_euler('x', correction_deg, degrees=True)
        new_rot = correction_rot * current_rot

        # Convert back to quaternion (wxyz format for MuJoCo)
        new_quat_xyzw = new_rot.as_quat()
        qpos[3:7] = [new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]]

        print(f"  Applied roll correction: {correction_deg:.2f}°")

    # Use best found quaternion
    qpos[3:7] = best_quat
    mj_data.qpos[:] = qpos
    roll_deg, pitch_deg, sagittal_offset, lateral_offset, left_force, right_force, total_force, asymmetry = measure_state()

    com_height = mj_data.subtree_com[1][2]

    print(f"\n" + "=" * 80)
    print("Final adjusted keyframe:")
    print("=" * 80)
    print(f"  Root quaternion: [{best_quat[0]:.6f}, {best_quat[1]:.6f}, {best_quat[2]:.6f}, {best_quat[3]:.6f}]")
    print(f"  Roll after mj_forward: {roll_deg:.2f}° (target: {target_roll_deg:.2f}°)")
    print(f"  Pitch after mj_forward: {pitch_deg:.2f}°")
    print(f"  CoM height: {com_height:.4f} m")
    print(f"  CoM sagittal offset: {sagittal_offset*1000:.2f} mm")
    print(f"  CoM lateral offset: {lateral_offset*1000:.2f} mm")
    print(f"  Left contact: {left_force:.2f} N")
    print(f"  Right contact: {right_force:.2f} N")
    print(f"  Total force: {total_force:.2f} N")
    print(f"  Asymmetry: {asymmetry:.2f} N")

    if abs(roll_deg - target_roll_deg) < tolerance_deg:
        print(f"\n[OK] Roll error within tolerance ({abs(roll_deg - target_roll_deg):.2f}° < {tolerance_deg:.2f}°)")
    else:
        print(f"\n[WARNING] Could not achieve target roll (best error: {best_error:.2f}°)")

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

    # Adjust keyframe to eliminate roll error
    qpos_adjusted, ctrl_adjusted = adjust_roll_error(
        model_path,
        keyframe_id=0,
        target_roll_deg=0.0,
        max_iterations=50,
        tolerance_deg=0.1,
    )

    print("\n" + "=" * 80)
    print("Adjusted keyframe XML:")
    print("=" * 80)
    print(format_keyframe_xml(qpos_adjusted, ctrl_adjusted))
