"""Adjust keyframe root Z to achieve correct contact force magnitude.

After adjusting hip pitch, the robot geometry changed and wheels penetrate ground.
This script raises root Z until contact forces match expected weight.
"""

import mujoco
import numpy as np


def adjust_contact_force(
    model_path: str,
    keyframe_id: int = 0,
    target_total_force: float = 79.5,  # Robot mass 8.10 kg × 9.81 m/s²
    tolerance: float = 2.0,  # ±2N tolerance
    max_iterations: int = 50,
):
    """Adjust root Z to achieve target total contact force.

    Args:
        model_path: Path to MuJoCo XML model
        keyframe_id: Keyframe index to adjust
        target_total_force: Target total contact force in N
        tolerance: Acceptable force error in N
        max_iterations: Maximum adjustment iterations

    Returns:
        Adjusted qpos and ctrl arrays
    """
    # Load model and data
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Load initial keyframe
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, keyframe_id)

    print("=" * 80)
    print("Adjusting keyframe height to achieve correct contact force")
    print("=" * 80)

    # Get initial state
    qpos = mj_data.qpos.copy()
    initial_root_z = qpos[2]

    # Get wheel geom IDs
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    def measure_contact_forces():
        """Measure total contact force after mj_forward."""
        mujoco.mj_forward(mj_model, mj_data)

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
        return left_force, right_force, total_force, asymmetry

    print(f"\nInitial state:")
    print(f"  Root Z: {initial_root_z:.6f} m")
    print(f"  Target total force: {target_total_force:.2f} N")

    # Measure initial forces
    mj_data.qpos[:] = qpos
    left_force, right_force, total_force, asymmetry = measure_contact_forces()

    print(f"  Left contact: {left_force:.2f} N")
    print(f"  Right contact: {right_force:.2f} N")
    print(f"  Total force: {total_force:.2f} N")
    print(f"  Asymmetry: {asymmetry:.2f} N")
    print(f"  Force error: {total_force - target_total_force:.2f} N")

    # Binary search for optimal height
    # If total force > target, robot is penetrating → raise it
    # If total force < target, robot is floating → lower it
    z_min = initial_root_z - 0.05  # 50mm below
    z_max = initial_root_z + 0.05  # 50mm above

    best_z = initial_root_z
    best_error = abs(total_force - target_total_force)

    print(f"\nSearching for optimal height...")
    print(f"  Search range: [{z_min:.6f}, {z_max:.6f}] m")

    for iteration in range(max_iterations):
        # Try midpoint
        z_test = (z_min + z_max) / 2.0

        qpos[2] = z_test
        mj_data.qpos[:] = qpos
        left_force, right_force, total_force, asymmetry = measure_contact_forces()

        force_error = total_force - target_total_force

        print(f"\nIteration {iteration}:")
        print(f"  Test Z: {z_test:.6f} m")
        print(f"  Left: {left_force:.2f} N, Right: {right_force:.2f} N")
        print(f"  Total: {total_force:.2f} N")
        print(f"  Force error: {force_error:.2f} N")
        print(f"  Asymmetry: {asymmetry:.2f} N")

        # Update best if improved
        if abs(force_error) < best_error:
            best_error = abs(force_error)
            best_z = z_test
            print(f"  [NEW BEST] Error improved to {best_error:.2f} N")

        # Check convergence
        if abs(force_error) < tolerance:
            print(f"  [OK] Target force achieved!")
            break

        # Adjust search range
        if force_error > 0:
            # Total force too high → robot penetrating → raise it
            z_min = z_test
        else:
            # Total force too low → robot floating → lower it
            z_max = z_test

        # Check if search range is too small
        if abs(z_max - z_min) < 0.0001:  # 0.1mm
            print(f"  [CONVERGED] Search range too small")
            break

    # Use best found height
    qpos[2] = best_z
    mj_data.qpos[:] = qpos
    left_force, right_force, total_force, asymmetry = measure_contact_forces()

    com_height = mj_data.subtree_com[1][2]
    com_pos = mj_data.subtree_com[1]

    # Get wheel positions for sagittal offset
    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
    l_wheel_pos = mj_data.xpos[l_wheel_id]
    r_wheel_pos = mj_data.xpos[r_wheel_id]
    wheel_center = (l_wheel_pos + r_wheel_pos) / 2.0
    sagittal_offset = com_pos[1] - wheel_center[1]

    print(f"\n" + "=" * 80)
    print("Final adjusted keyframe:")
    print("=" * 80)
    print(f"  Root Z: {best_z:.6f} m (change: {(best_z - initial_root_z)*1000:.2f} mm)")
    print(f"  CoM height: {com_height:.4f} m")
    print(f"  CoM sagittal offset: {sagittal_offset*1000:.2f} mm")
    print(f"  Left contact: {left_force:.2f} N")
    print(f"  Right contact: {right_force:.2f} N")
    print(f"  Total force: {total_force:.2f} N (target: {target_total_force:.2f} N)")
    print(f"  Force error: {abs(total_force - target_total_force):.2f} N")
    print(f"  Asymmetry: {asymmetry:.2f} N")

    if abs(total_force - target_total_force) < tolerance:
        print(f"\n[OK] Contact force within target ({abs(total_force - target_total_force):.2f} N < {tolerance:.2f} N)")
    else:
        print(f"\n[WARNING] Could not achieve target force (best error: {best_error:.2f} N)")

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

    # Adjust keyframe to achieve correct contact force
    qpos_adjusted, ctrl_adjusted = adjust_contact_force(
        model_path,
        keyframe_id=0,
        target_total_force=79.5,  # 8.10 kg × 9.81 m/s²
        tolerance=2.0,  # ±2N
        max_iterations=50,
    )

    print("\n" + "=" * 80)
    print("Adjusted keyframe XML:")
    print("=" * 80)
    print(format_keyframe_xml(qpos_adjusted, ctrl_adjusted))
