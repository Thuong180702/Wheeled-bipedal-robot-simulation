"""Roll/Pitch Axis Contract Audit.

Establishes ground truth for axis conventions across:
- centroidal_state_estimator.py
- centroidal_wrench_computer.py
- simple_force_distributor.py
- contact_jacobian.py

Suspected mismatches:
1. state.pitch_x/roll_y may use world-frame Euler instead of robot-frame
2. Mx/My moment channels may be swapped
3. Hip-roll torque may contribute to wrong moment channel
4. Vertical force asymmetry may use wrong coordinate for My
"""

import jax.numpy as jnp
import mujoco
import numpy as np
from pathlib import Path

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.simple_force_distributor import SimpleForceDistributor
from wheeled_biped.controllers.contact_jacobian import ContactJacobian
from wheeled_biped.controllers.orientation_utils import (
    compute_orientation_from_quaternion,
    compute_robot_frame_orientation_from_quaternion,
)


def load_model():
    """Load MuJoCo model."""
    xml_path = Path("assets/robot/wheeled_biped_real.xml")
    if not xml_path.exists():
        raise FileNotFoundError(f"Model file not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def calibrate_equilibrium(model, data):
    """Find equilibrium configuration."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    return data.qpos.copy(), data.qvel.copy()


def apply_roll_perturbation(qpos, angle_rad):
    """Apply roll perturbation (rotation about X-axis)."""
    quat = qpos[3:7].copy()
    # Roll quaternion: rotation about X-axis
    half_angle = angle_rad / 2.0
    roll_quat = np.array([np.cos(half_angle), np.sin(half_angle), 0.0, 0.0])
    # Multiply quaternions
    w1, x1, y1, z1 = roll_quat
    w2, x2, y2, z2 = quat
    quat_new = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    quat_new = quat_new / np.linalg.norm(quat_new)
    qpos_new = qpos.copy()
    qpos_new[3:7] = quat_new
    return qpos_new


def apply_pitch_perturbation(qpos, angle_rad):
    """Apply pitch perturbation (rotation about Y-axis)."""
    quat = qpos[3:7].copy()
    # Pitch quaternion: rotation about Y-axis
    half_angle = angle_rad / 2.0
    pitch_quat = np.array([np.cos(half_angle), 0.0, np.sin(half_angle), 0.0])
    # Multiply quaternions
    w1, x1, y1, z1 = pitch_quat
    w2, x2, y2, z2 = quat
    quat_new = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    quat_new = quat_new / np.linalg.norm(quat_new)
    qpos_new = qpos.copy()
    qpos_new[3:7] = quat_new
    return qpos_new


def part_a_orientation_mapping_audit(model, data, state_estimator):
    """Part A: Orientation mapping audit."""
    print("=" * 80)
    print("PART A: Orientation Mapping Audit")
    print("=" * 80)

    qpos_eq, qvel_eq = calibrate_equilibrium(model, data)

    test_cases = [
        ("Equilibrium", qpos_eq, 0.0, "none"),
        ("+Roll (X-axis)", apply_roll_perturbation(qpos_eq, +0.1), +0.1, "roll"),
        ("-Roll (X-axis)", apply_roll_perturbation(qpos_eq, -0.1), -0.1, "roll"),
        ("+Pitch (Y-axis)", apply_pitch_perturbation(qpos_eq, +0.1), +0.1, "pitch"),
        ("-Pitch (Y-axis)", apply_pitch_perturbation(qpos_eq, -0.1), -0.1, "pitch"),
    ]

    print("\nTest | Quat | Euler(roll,pitch,yaw) | body_pitch_x | body_roll_y | state.pitch_x | state.roll_y")
    print("-" * 120)

    for name, qpos_test, expected_angle, axis_type in test_cases:
        data.qpos[:] = qpos_test
        data.qvel[:] = qvel_eq
        mujoco.mj_forward(model, data)

        quat = data.qpos[3:7]

        # Euler angles (world frame)
        roll_euler, pitch_euler, yaw_euler = compute_orientation_from_quaternion(quat)

        # Robot frame
        body_pitch_x, body_roll_y, body_yaw_z = compute_robot_frame_orientation_from_quaternion(quat)

        # State estimator
        state, _ = state_estimator.estimate(jnp.zeros(42), data, None)

        print(f"{name:20s} | [{quat[0]:.3f},{quat[1]:.3f},{quat[2]:.3f},{quat[3]:.3f}] | "
              f"({roll_euler*180/np.pi:+6.2f}, {pitch_euler*180/np.pi:+6.2f}, {yaw_euler*180/np.pi:+6.2f}) | "
              f"{body_pitch_x*180/np.pi:+6.2f} | {body_roll_y*180/np.pi:+6.2f} | "
              f"{state.pitch_x*180/np.pi:+6.2f} | {state.roll_y*180/np.pi:+6.2f}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("\nExpected behavior:")
    print("  - +Roll (X-axis) should increase roll_y")
    print("  - +Pitch (Y-axis) should increase pitch_x")
    print("\nCurrent mapping:")
    print("  - state.pitch_x = ?")
    print("  - state.roll_y = ?")
    print("\nIf state.pitch_x/roll_y use world-frame Euler instead of robot-frame:")
    print("  FIX: centroidal_state_estimator.py should use body_pitch_x/body_roll_y")


def part_b_moment_channel_truth_table(model, data, distributor, contact_jacobian):
    """Part B: Moment channel truth table."""
    print("\n" + "=" * 80)
    print("PART B: Moment Channel Truth Table")
    print("=" * 80)

    qpos_eq, qvel_eq = calibrate_equilibrium(model, data)
    data.qpos[:] = qpos_eq
    data.qvel[:] = qvel_eq
    mujoco.mj_forward(model, data)

    com_pos = data.subtree_com[1]

    # Get wheel positions
    l_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel")
    r_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel")
    l_wheel_pos = data.xpos[l_wheel_body_id]
    r_wheel_pos = data.xpos[r_wheel_body_id]
    wheel_pos_left = jnp.array(l_wheel_pos - com_pos)
    wheel_pos_right = jnp.array(r_wheel_pos - com_pos)

    test_wrenches = [
        ("Mx=+5", jnp.array([0.0, 0.0, 80.0, +5.0, 0.0, 0.0])),
        ("Mx=-5", jnp.array([0.0, 0.0, 80.0, -5.0, 0.0, 0.0])),
        ("My=+5", jnp.array([0.0, 0.0, 80.0, 0.0, +5.0, 0.0])),
        ("My=-5", jnp.array([0.0, 0.0, 80.0, 0.0, -5.0, 0.0])),
    ]

    print("\nTest | Input(Mx,My) | f_left[2] | f_right[2] | delta_fz | tau_hip_roll | Achieved(Mx,My)")
    print("-" * 100)

    for name, wrench in test_wrenches:
        Fx, Fy, Fz, Mx, My, Mz = wrench

        # Distribute wrench
        f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
            desired_wrench=wrench,
            left_contact=True,
            right_contact=True,
            wheel_pos_left=wheel_pos_left,
            wheel_pos_right=wheel_pos_right,
            hip_roll_authority_scale=1.0,
            distribution_mode="absolute",
        )

        # Compute achieved wrench
        solution = jnp.concatenate([f_left, f_right, tau_hip_roll])
        A_wrench = contact_jacobian.build_wrench_matrix(data, wheel_pos_left, wheel_pos_right)
        achieved_wrench = A_wrench @ solution

        delta_fz = float(f_left[2] - f_right[2])
        achieved_mx = float(achieved_wrench[3])
        achieved_my = float(achieved_wrench[4])

        print(f"{name:10s} | ({Mx:+.1f}, {My:+.1f}) | {f_left[2]:+6.1f} | {f_right[2]:+6.1f} | "
              f"{delta_fz:+6.1f} | [{tau_hip_roll[0]:+.1f}, {tau_hip_roll[1]:+.1f}] | "
              f"({achieved_mx:+.2f}, {achieved_my:+.2f})")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("\nExpected:")
    print("  - Mx command should produce Mx response (pitch control)")
    print("  - My command should produce My response (roll control)")
    print("\nIf Mx/My channels are swapped:")
    print("  FIX: Swap Mx<->My in centroidal_wrench_computer or force distributor")


def part_c_direct_hip_roll_torque_truth_table(model, data):
    """Part C: Direct hip-roll torque truth table."""
    print("\n" + "=" * 80)
    print("PART C: Direct Hip-Roll Torque Truth Table")
    print("=" * 80)

    qpos_eq, qvel_eq = calibrate_equilibrium(model, data)

    # Get hip-roll joint indices
    l_hip_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "l_hip_roll")
    r_hip_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "r_hip_roll")

    test_cases = [
        ("[+5, -5]", +5.0, -5.0),
        ("[-5, +5]", -5.0, +5.0),
        ("[+10, -10]", +10.0, -10.0),
        ("[-10, +10]", -10.0, +10.0),
    ]

    print("\nTest | tau_hip_roll | pitch_x_before | pitch_x_after | delta_pitch_x | roll_y_before | roll_y_after | delta_roll_y")
    print("-" * 120)

    for name, tau_left, tau_right in test_cases:
        data.qpos[:] = qpos_eq
        data.qvel[:] = qvel_eq
        mujoco.mj_forward(model, data)

        quat_before = data.qpos[3:7].copy()
        roll_before, pitch_before, _ = compute_orientation_from_quaternion(quat_before)

        # Apply torques
        data.ctrl[l_hip_roll_id] = tau_left
        data.ctrl[r_hip_roll_id] = tau_right

        # Step once
        mujoco.mj_step(model, data)

        quat_after = data.qpos[3:7].copy()
        roll_after, pitch_after, _ = compute_orientation_from_quaternion(quat_after)

        delta_pitch = pitch_after - pitch_before
        delta_roll = roll_after - roll_before

        print(f"{name:12s} | [{tau_left:+.1f}, {tau_right:+.1f}] | "
              f"{pitch_before*180/np.pi:+6.3f} | {pitch_after*180/np.pi:+6.3f} | {delta_pitch*180/np.pi:+7.4f} | "
              f"{roll_before*180/np.pi:+6.3f} | {roll_after*180/np.pi:+6.3f} | {delta_roll*180/np.pi:+7.4f}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("\nExpected:")
    print("  - Hip-roll torque should primarily affect roll_y (not pitch_x)")
    print("\nIf hip-roll torque affects pitch_x instead:")
    print("  FIX: contact_jacobian.py may have wrong moment channel for hip-roll")


def part_d_delta_force_asymmetry_truth_table(model, data, contact_jacobian):
    """Part D: Delta force asymmetry truth table."""
    print("\n" + "=" * 80)
    print("PART D: Delta Force Asymmetry Truth Table")
    print("=" * 80)

    qpos_eq, qvel_eq = calibrate_equilibrium(model, data)
    data.qpos[:] = qpos_eq
    data.qvel[:] = qvel_eq
    mujoco.mj_forward(model, data)

    com_pos = data.subtree_com[1]

    # Get wheel positions
    l_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel")
    r_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel")
    l_wheel_pos = data.xpos[l_wheel_body_id]
    r_wheel_pos = data.xpos[r_wheel_body_id]
    wheel_pos_left = jnp.array(l_wheel_pos - com_pos)
    wheel_pos_right = jnp.array(r_wheel_pos - com_pos)

    print(f"\nWheel positions relative to CoM:")
    print(f"  Left:  x={wheel_pos_left[0]:+.3f}, y={wheel_pos_left[1]:+.3f}, z={wheel_pos_left[2]:+.3f}")
    print(f"  Right: x={wheel_pos_right[0]:+.3f}, y={wheel_pos_right[1]:+.3f}, z={wheel_pos_right[2]:+.3f}")

    test_cases = [
        ("delta_fz=[+10, -10]", jnp.array([0.0, 0.0, +10.0]), jnp.array([0.0, 0.0, -10.0])),
        ("delta_fz=[-10, +10]", jnp.array([0.0, 0.0, -10.0]), jnp.array([0.0, 0.0, +10.0])),
    ]

    print("\nTest | delta_fz_left | delta_fz_right | Achieved Mx | Achieved My")
    print("-" * 80)

    for name, f_left, f_right in test_cases:
        # Compute achieved wrench from force asymmetry
        tau_hip_roll = jnp.zeros(2)
        solution = jnp.concatenate([f_left, f_right, tau_hip_roll])
        A_wrench = contact_jacobian.build_wrench_matrix(data, wheel_pos_left, wheel_pos_right)
        achieved_wrench = A_wrench @ solution

        achieved_mx = float(achieved_wrench[3])
        achieved_my = float(achieved_wrench[4])

        print(f"{name:20s} | {f_left[2]:+6.1f} | {f_right[2]:+6.1f} | {achieved_mx:+8.3f} | {achieved_my:+8.3f}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("\nExpected from r x F:")
    print("  My = z*Fx - x*Fz  (vertical force asymmetry across x-position)")
    print("  Mx = y*Fz - z*Fy  (vertical force asymmetry across y-position)")
    print("\nFor side-by-side wheels (y != 0, x ~= 0):")
    print("  - Vertical Fz asymmetry should produce Mx (pitch moment)")
    print("  - NOT My (roll moment)")
    print("\nIf implementation uses y-coordinate for My:")
    print("  FIX: simple_force_distributor.py delta mode uses wrong coordinate")


def run_axis_contract_audit():
    """Run complete axis contract audit."""
    print("=" * 80)
    print("ROLL/PITCH AXIS CONTRACT AUDIT")
    print("=" * 80)

    model, data = load_model()

    # Create components
    state_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=8.1,
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )

    distributor = SimpleForceDistributor(
        tau_hip_roll_max=15.0,
        max_force_asymmetry=40.0,
        min_wheel_force=10.0,
    )

    contact_jacobian = ContactJacobian(model)

    # Run audit parts
    part_a_orientation_mapping_audit(model, data, state_estimator)
    part_b_moment_channel_truth_table(model, data, distributor, contact_jacobian)
    part_c_direct_hip_roll_torque_truth_table(model, data)
    part_d_delta_force_asymmetry_truth_table(model, data, contact_jacobian)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review all four parts to establish ground truth")
    print("2. Fix axis contract mismatches across all files")
    print("3. Add regression tests for correct axis mapping")
    print("4. Only then rerun B0/B100/B500")


if __name__ == "__main__":
    run_axis_contract_audit()
