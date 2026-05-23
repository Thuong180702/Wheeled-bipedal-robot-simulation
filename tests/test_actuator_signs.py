"""Actuator sign and authority validation tests.

Verifies that each actuator produces force in the expected direction
and that support joints have sufficient authority.
"""

import numpy as np
import mujoco
import pytest


@pytest.fixture
def robot_at_keyframe():
    """Load robot at calibrated standing keyframe.

    Returns:
        Tuple of (mj_model, mj_data)
    """
    model_path = "assets/robot/wheeled_biped_real.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Reset to keyframe 0 if available
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    return model, data


def measure_contact_fz(mj_model, mj_data):
    """Measure total vertical contact force from MuJoCo contact solver.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data

    Returns:
        float: Total vertical contact force (Fz) in Newtons
    """
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    total_fz = 0.0

    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        involves_wheel = g1 in {l_wheel_geom_id, r_wheel_geom_id} or g2 in {l_wheel_geom_id, r_wheel_geom_id}

        if not (involves_floor and involves_wheel):
            continue

        # Use mj_contactForce to get the contact force in the contact frame
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)

        # Transform to world frame using contact frame
        frame = np.array(c.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        total_fz += float(force_world[2])

    return total_fz


def test_actuator_sign_consistency(robot_at_keyframe):
    """Test 2.1: Verify each actuator produces expected acceleration direction.

    For each joint, apply +1.0 Nm and -1.0 Nm torques and verify that
    the resulting accelerations have opposite signs.
    """
    mj_model, mj_data = robot_at_keyframe

    joint_names = [
        "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
        "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"
    ]

    print("\n" + "=" * 80)
    print("Test 2.1: Actuator Sign Consistency")
    print("=" * 80)
    print(f"{'Joint':<15} {'qacc(+1Nm)':<15} {'qacc(-1Nm)':<15} {'Consistent':<12}")
    print("-" * 80)

    for joint_idx in range(10):
        # Test positive torque
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mj_data.ctrl[:] = 0.0  # Zero all controls
        mj_data.ctrl[joint_idx] = 1.0
        mujoco.mj_step(mj_model, mj_data)
        qacc_pos = mj_data.qacc[6 + joint_idx]

        # Test negative torque
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mj_data.ctrl[:] = 0.0  # Zero all controls
        mj_data.ctrl[joint_idx] = -1.0
        mujoco.mj_step(mj_model, mj_data)
        qacc_neg = mj_data.qacc[6 + joint_idx]

        # Verify opposite signs
        sign_consistent = np.sign(qacc_pos) == -np.sign(qacc_neg)

        print(f"{joint_names[joint_idx]:<15} {qacc_pos:>14.6f} {qacc_neg:>14.6f} {'PASS' if sign_consistent else 'FAIL':<12}")

        assert sign_consistent, f"Joint {joint_names[joint_idx]} does not have consistent sign response"

    print("=" * 80)


def test_support_joint_authority(robot_at_keyframe):
    """Test 2.2: Verify support joints can influence contact force.

    For each support joint (hip_pitch, knee), apply torques and measure
    the change in vertical contact force. This is diagnostic only.
    """
    mj_model, mj_data = robot_at_keyframe

    support_joints = [2, 3, 7, 8]  # l_hip_pitch, l_knee, r_hip_pitch, r_knee
    joint_names = ["l_hip_pitch", "l_knee", "r_hip_pitch", "r_knee"]

    print("\n" + "=" * 100)
    print("Test 2.2: Support Joint Authority (Diagnostic)")
    print("=" * 100)
    print(f"{'Joint':<15} {'Fz_base(N)':<12} {'Fz_pos(N)':<12} {'Fz_neg(N)':<12} {'dFz_pos':<10} {'dFz_neg':<10} {'Helpful':<10}")
    print("-" * 100)

    for i, joint_idx in enumerate(support_joints):
        # Baseline: zero torque
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mj_data.ctrl[:] = 0.0  # Zero all controls
        mujoco.mj_step(mj_model, mj_data)
        fz_baseline = measure_contact_fz(mj_model, mj_data)

        # Positive torque
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mj_data.ctrl[:] = 0.0  # Zero all controls
        mj_data.ctrl[joint_idx] = 10.0
        mujoco.mj_step(mj_model, mj_data)
        fz_pos = measure_contact_fz(mj_model, mj_data)
        qacc_pos = mj_data.qacc[6 + joint_idx]

        # Negative torque
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mj_data.ctrl[:] = 0.0  # Zero all controls
        mj_data.ctrl[joint_idx] = -10.0
        mujoco.mj_step(mj_model, mj_data)
        fz_neg = measure_contact_fz(mj_model, mj_data)
        qacc_neg = mj_data.qacc[6 + joint_idx]

        # Compute deltas
        delta_fz_pos = fz_pos - fz_baseline
        delta_fz_neg = fz_neg - fz_baseline

        # Determine helpful sign
        helpful_sign = '+' if delta_fz_pos > delta_fz_neg else '-'

        print(f"{joint_names[i]:<15} {fz_baseline:>11.3f} {fz_pos:>11.3f} {fz_neg:>11.3f} "
              f"{delta_fz_pos:>9.3f} {delta_fz_neg:>9.3f} {helpful_sign:>9}")

    print("=" * 100)


def test_left_right_symmetry(robot_at_keyframe):
    """Test 2.3: Verify left/right joint pairs have symmetric response.

    For each joint pair, apply the same torque and verify that the
    acceleration magnitudes are within 10% of each other.
    """
    mj_model, mj_data = robot_at_keyframe

    joint_pairs = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
    pair_names = [
        ("l_hip_roll", "r_hip_roll"),
        ("l_hip_yaw", "r_hip_yaw"),
        ("l_hip_pitch", "r_hip_pitch"),
        ("l_knee", "r_knee"),
        ("l_wheel", "r_wheel")
    ]

    print("\n" + "=" * 90)
    print("Test 2.3: Left-Right Symmetry")
    print("=" * 90)
    print(f"{'Joint Pair':<30} {'|qacc_left|':<15} {'|qacc_right|':<15} {'Ratio':<10} {'Symmetric':<12}")
    print("-" * 90)

    for i, (left_idx, right_idx) in enumerate(joint_pairs):
        # Test left joint
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mj_data.ctrl[:] = 0.0  # Zero all controls
        mj_data.ctrl[left_idx] = 5.0
        mujoco.mj_step(mj_model, mj_data)
        qacc_left = abs(mj_data.qacc[6 + left_idx])

        # Test right joint
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mj_data.ctrl[:] = 0.0  # Zero all controls
        mj_data.ctrl[right_idx] = 5.0
        mujoco.mj_step(mj_model, mj_data)
        qacc_right = abs(mj_data.qacc[6 + right_idx])

        # Compute ratio
        ratio = qacc_left / max(qacc_right, 1e-6)

        # Check symmetry (within 10%)
        symmetric = 0.9 <= ratio <= 1.1

        pair_name = f"{pair_names[i][0]} / {pair_names[i][1]}"
        print(f"{pair_name:<30} {qacc_left:>14.6f} {qacc_right:>14.6f} {ratio:>9.3f} {'PASS' if symmetric else 'FAIL':<12}")

        assert symmetric, f"Joint pair {pair_name} does not have symmetric response (ratio={ratio:.3f})"

    print("=" * 90)
