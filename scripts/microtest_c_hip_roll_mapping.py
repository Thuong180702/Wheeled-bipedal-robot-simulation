"""Micro-test C: Hip-roll torque mapping direction.

Verify whether direct hip-roll torque produces expected roll_y acceleration.

Setup:
- Calibrated equilibrium
- Stage 2B feedforward + static posture active
- Disable WBC correction
- Apply direct hip-roll test torques for one step

Test cases:
1. tau_hip_roll = [+5, -5] Nm
2. tau_hip_roll = [-5, +5] Nm
3. tau_hip_roll = [+10, -10] Nm
4. tau_hip_roll = [-10, +10] Nm

Acceptance:
- Determine which hip-roll torque pattern reduces positive roll_y
- Determine which hip-roll torque pattern reduces negative roll_y
"""

import mujoco
import numpy as np
from pathlib import Path


def load_model():
    """Load MuJoCo model."""
    xml_path = Path("assets/robot/wheeled_biped_real.xml")
    if not xml_path.exists():
        raise FileNotFoundError(f"Model file not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def calibrate_equilibrium(model, data, target_height=0.50):
    """Find equilibrium configuration at target height."""
    # Reset to default pose
    mujoco.mj_resetData(model, data)

    # Set initial base height slightly above target to account for CoM offset
    # The base is below the CoM, so we need to start higher
    data.qpos[2] = target_height + 0.1

    # Let robot settle under gravity with zero control
    data.ctrl[:] = 0.0
    for _ in range(2000):
        mujoco.mj_step(model, data)

    return data.qpos.copy(), data.qvel.copy()


def apply_hip_roll_torque_one_step(model, data, tau_left, tau_right):
    """Apply hip-roll torques for one step and measure response.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        tau_left: Left hip-roll torque (Nm)
        tau_right: Right hip-roll torque (Nm)

    Returns:
        dict with before/after state
    """
    # Get hip-roll joint indices
    l_hip_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_hip_roll")
    r_hip_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_hip_roll")

    # Record before state
    roll_y_before = data.qpos[4]  # Base roll (quat converted)
    roll_rate_y_before = data.qvel[3]  # Base roll rate
    pitch_x_before = data.qpos[5]
    com_z_before = data.subtree_com[1][2]

    # Get base quaternion and convert to roll
    quat = data.qpos[3:7]
    # Simple roll extraction (assumes small angles)
    roll_y_before = 2.0 * np.arctan2(quat[1], quat[0])

    # Apply torques
    data.ctrl[l_hip_roll_id] = tau_left
    data.ctrl[r_hip_roll_id] = tau_right

    # Step once
    mujoco.mj_step(model, data)

    # Record after state
    quat_after = data.qpos[3:7]
    roll_y_after = 2.0 * np.arctan2(quat_after[1], quat_after[0])
    roll_rate_y_after = data.qvel[3]
    pitch_x_after = data.qpos[5]
    com_z_after = data.subtree_com[1][2]

    # Compute deltas
    delta_roll_y = roll_y_after - roll_y_before
    delta_roll_rate_y = roll_rate_y_after - roll_rate_y_before

    # Get contact forces
    left_wheel_fz = 0.0
    right_wheel_fz = 0.0
    left_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    right_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        force_contact = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force_contact)
        frame = np.array(contact.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]

        if geom1 == left_wheel_geom_id or geom2 == left_wheel_geom_id:
            left_wheel_fz += force_world[2]
        if geom1 == right_wheel_geom_id or geom2 == right_wheel_geom_id:
            right_wheel_fz += force_world[2]

    return {
        "tau_left": tau_left,
        "tau_right": tau_right,
        "roll_y_before": roll_y_before,
        "roll_y_after": roll_y_after,
        "delta_roll_y": delta_roll_y,
        "roll_rate_y_before": roll_rate_y_before,
        "roll_rate_y_after": roll_rate_y_after,
        "delta_roll_rate_y": delta_roll_rate_y,
        "pitch_x_before": pitch_x_before,
        "pitch_x_after": pitch_x_after,
        "com_z_before": com_z_before,
        "com_z_after": com_z_after,
        "left_wheel_fz": left_wheel_fz,
        "right_wheel_fz": right_wheel_fz,
    }


def run_microtest_c():
    """Run micro-test C: Hip-roll torque mapping direction."""
    print("=" * 80)
    print("MICRO-TEST C: Hip-roll torque mapping direction")
    print("=" * 80)

    # Load model
    model, data = load_model()

    # Calibrate equilibrium
    print("\nCalibrating equilibrium at h=0.50m...")
    qpos_eq, qvel_eq = calibrate_equilibrium(model, data, target_height=0.50)
    print(f"Equilibrium: com_z={data.subtree_com[1][2]:.3f}m")

    # Test cases
    test_cases = [
        (+5.0, -5.0),
        (-5.0, +5.0),
        (+10.0, -10.0),
        (-10.0, +10.0),
    ]

    results = []

    for tau_left, tau_right in test_cases:
        # Reset to equilibrium
        data.qpos[:] = qpos_eq
        data.qvel[:] = qvel_eq
        mujoco.mj_forward(model, data)

        # Apply torque and measure
        result = apply_hip_roll_torque_one_step(model, data, tau_left, tau_right)
        results.append(result)

        print(f"\nTest: tau_hip_roll = [{tau_left:+.1f}, {tau_right:+.1f}] Nm")
        print(f"  roll_y: {result['roll_y_before']:+.4f} -> {result['roll_y_after']:+.4f} rad (delta={result['delta_roll_y']:+.6f})")
        print(f"  roll_rate_y: {result['roll_rate_y_before']:+.4f} -> {result['roll_rate_y_after']:+.4f} rad/s (delta={result['delta_roll_rate_y']:+.6f})")
        print(f"  pitch_x: {result['pitch_x_before']:+.4f} -> {result['pitch_x_after']:+.4f} rad")
        print(f"  com_z: {result['com_z_before']:.3f} -> {result['com_z_after']:.3f} m")
        print(f"  wheel_fz: L={result['left_wheel_fz']:.1f}N, R={result['right_wheel_fz']:.1f}N")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print("\nRoll acceleration vs torque pattern:")
    for i, result in enumerate(results):
        tau_pattern = f"[{result['tau_left']:+.1f}, {result['tau_right']:+.1f}]"
        delta_roll_rate = result['delta_roll_rate_y']
        direction = "LEFT" if delta_roll_rate < 0 else "RIGHT"
        print(f"  {tau_pattern} Nm -> delta_roll_rate={delta_roll_rate:+.6f} rad/s ({direction})")

    print("\nInterpretation:")
    print("  - Positive delta_roll_rate_y = rolling RIGHT (positive roll_y)")
    print("  - Negative delta_roll_rate_y = rolling LEFT (negative roll_y)")
    print("  - To correct positive roll_y (leaning right), need negative delta_roll_rate_y")
    print("  - To correct negative roll_y (leaning left), need positive delta_roll_rate_y")

    # Determine correct mapping
    print("\nMapping verification:")
    for i, result in enumerate(results):
        tau_pattern = f"[{result['tau_left']:+.1f}, {result['tau_right']:+.1f}]"
        delta_roll_rate = result['delta_roll_rate_y']

        if result['tau_left'] > 0 and result['tau_right'] < 0:
            # Left positive, right negative
            if delta_roll_rate < 0:
                print(f"  OK {tau_pattern} produces LEFT roll (correct for right-lean correction)")
            else:
                print(f"  FAIL {tau_pattern} produces RIGHT roll (WRONG for right-lean correction)")
        elif result['tau_left'] < 0 and result['tau_right'] > 0:
            # Left negative, right positive
            if delta_roll_rate > 0:
                print(f"  OK {tau_pattern} produces RIGHT roll (correct for left-lean correction)")
            else:
                print(f"  FAIL {tau_pattern} produces LEFT roll (WRONG for left-lean correction)")

    print("\n" + "=" * 80)
    print("MICRO-TEST C COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_microtest_c()
