"""Find static equilibrium configuration for wheeled biped robot.

The goal is to find joint angles (hip_pitch, knee) that result in:
1. CoM directly above wheel contact line (CoM_y ≈ 0)
2. Contact forces equal to robot weight (not compressed)
3. Zero initial roll moment
"""

import mujoco
import numpy as np
from scipy.optimize import minimize


def evaluate_config(joint_angles, mj_model, mj_data, l_wheel_geom_id, r_wheel_geom_id):
    """Evaluate how close a configuration is to static equilibrium.

    Args:
        joint_angles: [hip_pitch, knee] for both legs (symmetric)
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        l_wheel_geom_id: Left wheel geom ID
        r_wheel_geom_id: Right wheel geom ID

    Returns:
        cost: Lower is better (0 = perfect equilibrium)
    """
    hip_pitch, knee = joint_angles

    # Set symmetric joint configuration
    # Joint order: l_hip_roll, l_hip_yaw, l_hip_pitch, l_knee, l_wheel,
    #              r_hip_roll, r_hip_yaw, r_hip_pitch, r_knee, r_wheel
    mj_data.qpos[7:17] = [0, 0, hip_pitch, knee, 0,  # Left leg
                          0, 0, hip_pitch, knee, 0]  # Right leg

    # Forward kinematics
    mujoco.mj_forward(mj_model, mj_data)

    # Get CoM position
    com_pos = mj_data.subtree_com[1]
    com_height = com_pos[2]
    com_y_offset = com_pos[1]

    # Get contact forces
    total_fz = 0.0
    left_fz = 0.0
    right_fz = 0.0

    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        force_contact = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)
        frame = np.array(contact.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        fz = force_world[2]

        total_fz += fz

        # Identify wheel geoms
        if geom1 == l_wheel_geom_id or geom2 == l_wheel_geom_id:
            left_fz += fz
        if geom1 == r_wheel_geom_id or geom2 == r_wheel_geom_id:
            right_fz += fz

    # Compute cost components
    robot_mass = sum(mj_model.body_mass)
    weight = robot_mass * 9.81

    # 1. CoM should be centered laterally (y ≈ 0)
    lateral_error = com_y_offset ** 2

    # 2. Total contact force should equal weight (not compressed)
    force_error = ((total_fz - weight) / weight) ** 2

    # 3. Contact forces should be symmetric
    force_asymmetry = ((left_fz - right_fz) / weight) ** 2

    # Total cost (weighted sum)
    # Focus on lateral balance and force symmetry
    cost = (
        10000.0 * lateral_error +    # Lateral balance is CRITICAL
        100.0 * force_error +        # Force magnitude matters
        1000.0 * force_asymmetry     # Symmetry is very important
    )

    return cost


def find_equilibrium():
    """Find equilibrium configuration using optimization."""
    # Load model
    mj_model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
    mj_data = mujoco.MjData(mj_model)

    # Get wheel geom IDs
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'l_wheel_collision')
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'r_wheel_collision')
    print(f"Wheel geom IDs: left={l_wheel_geom_id}, right={r_wheel_geom_id}")

    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

    # Initial guess from keyframe
    initial_guess = [0.95, 1.70]  # hip_pitch, knee

    print("Finding equilibrium configuration...")
    print("=" * 80)
    print(f"Initial guess: hip_pitch={initial_guess[0]:.3f}, knee={initial_guess[1]:.3f}")

    # Evaluate initial configuration
    initial_cost = evaluate_config(initial_guess, mj_model, mj_data, l_wheel_geom_id, r_wheel_geom_id)
    print(f"Initial cost: {initial_cost:.6f}")
    print()

    # Optimize
    result = minimize(
        evaluate_config,
        initial_guess,
        args=(mj_model, mj_data, l_wheel_geom_id, r_wheel_geom_id),
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
    )

    if result.success:
        print("Optimization successful!")
        print(f"Final cost: {result.fun:.6f}")
        print(f"Optimal joint angles:")
        print(f"  hip_pitch = {result.x[0]:.6f} rad ({np.degrees(result.x[0]):.2f} deg)")
        print(f"  knee = {result.x[1]:.6f} rad ({np.degrees(result.x[1]):.2f} deg)")
        print()

        # Evaluate final configuration
        mj_data.qpos[7:17] = [0, 0, result.x[0], result.x[1], 0,
                              0, 0, result.x[0], result.x[1], 0]
        mujoco.mj_forward(mj_model, mj_data)

        com_pos = mj_data.subtree_com[1]
        robot_mass = sum(mj_model.body_mass)
        weight = robot_mass * 9.81

        total_fz = 0.0
        left_fz = 0.0
        right_fz = 0.0

        for i in range(mj_data.ncon):
            contact = mj_data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            force_contact = np.zeros(6)
            mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)
            frame = np.array(contact.frame).reshape(3, 3)
            force_world = frame.T @ force_contact[:3]
            fz = force_world[2]

            total_fz += fz
            if geom1 == l_wheel_geom_id or geom2 == l_wheel_geom_id:
                left_fz += fz
            if geom1 == r_wheel_geom_id or geom2 == r_wheel_geom_id:
                right_fz += fz

        print("Final configuration:")
        print(f"  CoM position: [{com_pos[0]:.6f}, {com_pos[1]:.6f}, {com_pos[2]:.6f}] m")
        print(f"  CoM height: {com_pos[2]:.4f} m")
        print(f"  CoM y-offset: {com_pos[1]:.6f} m (should be ~0)")
        print(f"  Robot weight: {weight:.2f} N")
        print(f"  Total contact force: {total_fz:.2f} N")
        print(f"  Left wheel force: {left_fz:.2f} N")
        print(f"  Right wheel force: {right_fz:.2f} N")
        print(f"  Force asymmetry: {left_fz - right_fz:.2f} N")
        print(f"  Force ratio: {total_fz / weight:.3f} (should be ~1.0)")
        print()

        print("Updated keyframe for wheeled_biped_real.xml:")
        print(f'    <key name="standing"')
        print(f'         qpos="0 0 0.545')
        print(f'               1 0 0 0')
        print(f'               0 0 {result.x[0]:.6f} {result.x[1]:.6f} 0')
        print(f'               0 0 {result.x[0]:.6f} {result.x[1]:.6f} 0"')
        print(f'         ctrl="0 0 0 0 0')
        print(f'               0 0 0 0 0"/>')

    else:
        print("Optimization failed!")
        print(result.message)


if __name__ == "__main__":
    find_equilibrium()
