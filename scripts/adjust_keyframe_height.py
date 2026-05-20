"""Adjust keyframe root Z height to eliminate contact penetration artifacts.

The 9.23 N contact asymmetry appears after mj_forward due to initial penetration.
This script adjusts root Z to achieve proper contact without penetration.
"""

import mujoco
import numpy as np


def find_proper_contact_height(
    model_path: str,
    keyframe_id: int = 0,
    target_contact_asymmetry: float = 1.0,  # Target < 1N asymmetry
    max_iterations: int = 50,
):
    """Find root Z height that minimizes contact asymmetry.

    Args:
        model_path: Path to MuJoCo XML model
        keyframe_id: Keyframe index to adjust
        target_contact_asymmetry: Target maximum asymmetry in N
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
    print("Adjusting keyframe height to eliminate contact asymmetry")
    print("=" * 80)

    # Get initial state
    qpos = mj_data.qpos.copy()
    initial_root_z = qpos[2]

    # Get wheel geom IDs
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    def measure_contact_asymmetry():
        """Measure contact force asymmetry after mj_forward."""
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

        return left_force, right_force, abs(left_force - right_force)

    print(f"\nInitial state:")
    print(f"  Root Z: {initial_root_z:.6f} m")

    # Measure initial asymmetry
    mj_data.qpos[:] = qpos
    left_force, right_force, asymmetry = measure_contact_asymmetry()

    print(f"  Left contact: {left_force:.2f} N")
    print(f"  Right contact: {right_force:.2f} N")
    print(f"  Asymmetry: {asymmetry:.2f} N")

    # Binary search for optimal height
    # Start with a range around current height
    z_min = initial_root_z - 0.01  # 10mm below
    z_max = initial_root_z + 0.01  # 10mm above

    best_z = initial_root_z
    best_asymmetry = asymmetry

    print(f"\nSearching for optimal height...")
    print(f"  Search range: [{z_min:.6f}, {z_max:.6f}] m")

    for iteration in range(max_iterations):
        # Try midpoint
        z_test = (z_min + z_max) / 2.0

        qpos[2] = z_test
        mj_data.qpos[:] = qpos
        left_force, right_force, asymmetry = measure_contact_asymmetry()

        print(f"\nIteration {iteration}:")
        print(f"  Test Z: {z_test:.6f} m")
        print(f"  Left: {left_force:.2f} N, Right: {right_force:.2f} N")
        print(f"  Asymmetry: {asymmetry:.2f} N")

        # Update best if improved
        if asymmetry < best_asymmetry:
            best_asymmetry = asymmetry
            best_z = z_test
            print(f"  [NEW BEST] Asymmetry improved to {best_asymmetry:.2f} N")

        # Check convergence
        if asymmetry < target_contact_asymmetry:
            print(f"  [OK] Target asymmetry achieved!")
            break

        # Adjust search range based on which wheel has more force
        if left_force > right_force:
            # Left wheel has more force, try raising robot slightly
            z_min = z_test
        else:
            # Right wheel has more force, try lowering robot slightly
            z_max = z_test

        # Check if search range is too small
        if abs(z_max - z_min) < 0.0001:  # 0.1mm
            print(f"  [CONVERGED] Search range too small")
            break

    # Use best found height
    qpos[2] = best_z
    mj_data.qpos[:] = qpos
    left_force, right_force, asymmetry = measure_contact_asymmetry()

    com_height = mj_data.subtree_com[1][2]

    print(f"\n" + "=" * 80)
    print("Final adjusted keyframe:")
    print("=" * 80)
    print(f"  Root Z: {best_z:.6f} m (change: {(best_z - initial_root_z)*1000:.2f} mm)")
    print(f"  CoM height: {com_height:.4f} m")
    print(f"  Left contact: {left_force:.2f} N")
    print(f"  Right contact: {right_force:.2f} N")
    print(f"  Asymmetry: {asymmetry:.2f} N")

    if asymmetry < target_contact_asymmetry:
        print(f"\n[OK] Contact asymmetry within target ({asymmetry:.2f} N < {target_contact_asymmetry:.2f} N)")
    else:
        print(f"\n[WARNING] Could not achieve target asymmetry (best: {best_asymmetry:.2f} N)")

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

    # Adjust keyframe height to minimize contact asymmetry
    qpos_adjusted, ctrl_adjusted = find_proper_contact_height(
        model_path,
        keyframe_id=0,
        target_contact_asymmetry=1.0,  # Target < 1N
        max_iterations=50,
    )

    print("\n" + "=" * 80)
    print("Adjusted keyframe XML:")
    print("=" * 80)
    print(format_keyframe_xml(qpos_adjusted, ctrl_adjusted))
