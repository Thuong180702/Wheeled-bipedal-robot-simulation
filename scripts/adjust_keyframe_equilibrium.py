"""Adjust keyframe to achieve equilibrium: zero roll and correct contact forces.

This script iteratively adjusts both root quaternion (for roll) and root Z (for contact forces)
to achieve a balanced equilibrium configuration.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def adjust_equilibrium(
    model_path: str,
    keyframe_id: int = 0,
    target_roll_deg: float = 0.0,
    target_total_force: float = 79.5,
    roll_tolerance_deg: float = 0.1,
    force_tolerance: float = 2.0,
    max_outer_iterations: int = 10,
    max_roll_iterations: int = 20,
    max_force_iterations: int = 20,
):
    """Adjust keyframe to achieve equilibrium with zero roll and correct contact forces.

    Args:
        model_path: Path to MuJoCo XML model
        keyframe_id: Keyframe index to adjust
        target_roll_deg: Target roll angle in degrees
        target_total_force: Target total contact force in N
        roll_tolerance_deg: Acceptable roll error in degrees
        force_tolerance: Acceptable force error in N
        max_outer_iterations: Maximum outer loop iterations
        max_roll_iterations: Maximum roll adjustment iterations per outer loop
        max_force_iterations: Maximum force adjustment iterations per outer loop

    Returns:
        Adjusted qpos and ctrl arrays
    """
    # Load model and data
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Load initial keyframe
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, keyframe_id)

    print("=" * 80)
    print("Adjusting keyframe to achieve equilibrium")
    print("=" * 80)

    # Get initial state
    qpos = mj_data.qpos.copy()

    # Get wheel body IDs
    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    # Get wheel geom IDs
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    def measure_state():
        """Measure roll, contact forces, and CoM after mj_forward."""
        mujoco.mj_forward(mj_model, mj_data)

        # Extract roll from quaternion
        quat = mj_data.qpos[3:7]
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = rot.as_euler('xyz', degrees=True)
        roll_deg = euler[0]
        pitch_deg = euler[1]

        # Measure CoM
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
    mj_data.qpos[:] = qpos
    roll_deg, pitch_deg, sagittal_offset, lateral_offset, left_force, right_force, total_force, asymmetry = measure_state()

    print(f"  Root Z: {qpos[2]:.6f} m")
    print(f"  Root quaternion: [{qpos[3]:.6f}, {qpos[4]:.6f}, {qpos[5]:.6f}, {qpos[6]:.6f}]")
    print(f"  Roll: {roll_deg:.2f}° (target: {target_roll_deg:.2f}°)")
    print(f"  Pitch: {pitch_deg:.2f}°")
    print(f"  Total force: {total_force:.2f} N (target: {target_total_force:.2f} N)")
    print(f"  Asymmetry: {asymmetry:.2f} N")
    print(f"  CoM sagittal offset: {sagittal_offset*1000:.2f} mm")

    # Outer loop: alternate between roll and force adjustment
    for outer_iter in range(max_outer_iterations):
        print(f"\n{'='*80}")
        print(f"Outer iteration {outer_iter}")
        print(f"{'='*80}")

        # Step 1: Adjust roll
        print(f"\nStep 1: Adjusting roll...")
        for roll_iter in range(max_roll_iterations):
            mj_data.qpos[:] = qpos
            roll_deg, pitch_deg, sagittal_offset, lateral_offset, left_force, right_force, total_force, asymmetry = measure_state()

            roll_error = roll_deg - target_roll_deg

            if roll_iter % 5 == 0:
                print(f"  Roll iter {roll_iter}: roll={roll_deg:.2f}°, error={roll_error:.2f}°")

            if abs(roll_error) < roll_tolerance_deg:
                print(f"  [OK] Roll converged: {roll_deg:.2f}°")
                break

            # Apply damped roll correction
            correction_deg = -roll_error * 0.5
            current_quat = qpos[3:7]
            current_rot = Rotation.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
            correction_rot = Rotation.from_euler('x', correction_deg, degrees=True)
            new_rot = correction_rot * current_rot
            new_quat_xyzw = new_rot.as_quat()
            qpos[3:7] = [new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]]

        # Step 2: Adjust contact force via root Z
        print(f"\nStep 2: Adjusting contact force...")
        z_min = qpos[2] - 0.05
        z_max = qpos[2] + 0.05

        for force_iter in range(max_force_iterations):
            z_test = (z_min + z_max) / 2.0
            qpos[2] = z_test
            mj_data.qpos[:] = qpos
            roll_deg, pitch_deg, sagittal_offset, lateral_offset, left_force, right_force, total_force, asymmetry = measure_state()

            force_error = total_force - target_total_force

            if force_iter % 5 == 0:
                print(f"  Force iter {force_iter}: Z={z_test:.6f}m, force={total_force:.2f}N, error={force_error:.2f}N")

            if abs(force_error) < force_tolerance:
                print(f"  [OK] Force converged: {total_force:.2f} N")
                break

            if force_error > 0:
                z_min = z_test
            else:
                z_max = z_test

            if abs(z_max - z_min) < 0.0001:
                print(f"  [CONVERGED] Search range too small")
                break

        # Check overall convergence
        mj_data.qpos[:] = qpos
        roll_deg, pitch_deg, sagittal_offset, lateral_offset, left_force, right_force, total_force, asymmetry = measure_state()

        roll_error = abs(roll_deg - target_roll_deg)
        force_error = abs(total_force - target_total_force)

        print(f"\nOuter iteration {outer_iter} result:")
        print(f"  Roll: {roll_deg:.2f}° (error: {roll_error:.2f}°)")
        print(f"  Total force: {total_force:.2f} N (error: {force_error:.2f} N)")

        if roll_error < roll_tolerance_deg and force_error < force_tolerance:
            print(f"  [OK] Both roll and force converged!")
            break

    # Final measurements
    mj_data.qpos[:] = qpos
    roll_deg, pitch_deg, sagittal_offset, lateral_offset, left_force, right_force, total_force, asymmetry = measure_state()
    com_height = mj_data.subtree_com[1][2]

    print(f"\n{'='*80}")
    print("Final equilibrium keyframe:")
    print(f"{'='*80}")
    print(f"  Root Z: {qpos[2]:.6f} m")
    print(f"  Root quaternion: [{qpos[3]:.6f}, {qpos[4]:.6f}, {qpos[5]:.6f}, {qpos[6]:.6f}]")
    print(f"  Roll: {roll_deg:.2f}° (target: {target_roll_deg:.2f}°)")
    print(f"  Pitch: {pitch_deg:.2f}°")
    print(f"  CoM height: {com_height:.4f} m")
    print(f"  CoM sagittal offset: {sagittal_offset*1000:.2f} mm")
    print(f"  CoM lateral offset: {lateral_offset*1000:.2f} mm")
    print(f"  Left contact: {left_force:.2f} N")
    print(f"  Right contact: {right_force:.2f} N")
    print(f"  Total force: {total_force:.2f} N (target: {target_total_force:.2f} N)")
    print(f"  Asymmetry: {asymmetry:.2f} N")

    roll_ok = abs(roll_deg - target_roll_deg) < roll_tolerance_deg
    force_ok = abs(total_force - target_total_force) < force_tolerance

    if roll_ok and force_ok:
        print(f"\n[OK] Equilibrium achieved!")
    else:
        if not roll_ok:
            print(f"\n[WARNING] Roll not converged (error: {abs(roll_deg - target_roll_deg):.2f}°)")
        if not force_ok:
            print(f"\n[WARNING] Force not converged (error: {abs(total_force - target_total_force):.2f} N)")

    # Generate control targets
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

    # Adjust keyframe to achieve equilibrium
    qpos_adjusted, ctrl_adjusted = adjust_equilibrium(
        model_path,
        keyframe_id=0,
        target_roll_deg=0.0,
        target_total_force=79.5,
        roll_tolerance_deg=0.1,
        force_tolerance=2.0,
        max_outer_iterations=10,
        max_roll_iterations=20,
        max_force_iterations=20,
    )

    print("\n" + "=" * 80)
    print("Adjusted keyframe XML:")
    print("=" * 80)
    print(format_keyframe_xml(qpos_adjusted, ctrl_adjusted))
