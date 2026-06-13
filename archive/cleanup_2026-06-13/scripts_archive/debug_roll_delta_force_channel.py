"""Diagnostic to prove roll delta-force channel bug in correction-only mode.

In correction-only mode, My correction cannot generate vertical force asymmetry
because split_fz_from_my treats total_fz as absolute contact force, not delta.
With correction_Fz ≈ 0, max_safe_diff becomes 0, blocking the main roll channel.
"""

import jax.numpy as jnp
import mujoco
import numpy as np
from pathlib import Path

from wheeled_biped.controllers.simple_force_distributor import SimpleForceDistributor
from wheeled_biped.controllers.contact_jacobian import ContactJacobian


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
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Settle for 2 seconds
    for _ in range(2000):
        mujoco.mj_step(model, data)

    return data.qpos.copy(), data.qvel.copy()


def get_wheel_positions_relative_to_com(data, com_pos):
    """Get wheel positions relative to CoM."""
    # Get wheel body IDs
    l_wheel_body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel")
    r_wheel_body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel")

    # Get wheel positions in world frame
    l_wheel_pos = data.xpos[l_wheel_body_id]
    r_wheel_pos = data.xpos[r_wheel_body_id]

    # Compute relative to CoM
    wheel_pos_left = jnp.array(l_wheel_pos - com_pos)
    wheel_pos_right = jnp.array(r_wheel_pos - com_pos)

    return wheel_pos_left, wheel_pos_right


def test_roll_correction_channel():
    """Test whether My correction can generate vertical force asymmetry."""
    print("=" * 80)
    print("DIAGNOSTIC: Roll delta-force channel bug")
    print("=" * 80)

    # Load model
    model, data = load_model()

    # Calibrate equilibrium
    print("\nCalibrating equilibrium at h=0.50m...")
    qpos_eq, qvel_eq = calibrate_equilibrium(model, data, target_height=0.50)
    com_pos = data.subtree_com[1]
    print(f"Equilibrium: com_z={com_pos[2]:.3f}m")

    # Create force distributor
    distributor = SimpleForceDistributor(
        tau_hip_roll_max=15.0,
        max_force_asymmetry=40.0,
        min_wheel_force=10.0,
    )

    # Create contact Jacobian
    contact_jacobian = ContactJacobian(model)

    # Get wheel positions
    wheel_pos_left, wheel_pos_right = get_wheel_positions_relative_to_com(data, com_pos)

    print(f"\nWheel positions relative to CoM:")
    print(f"  Left:  [{wheel_pos_left[0]:+.3f}, {wheel_pos_left[1]:+.3f}, {wheel_pos_left[2]:+.3f}]")
    print(f"  Right: [{wheel_pos_right[0]:+.3f}, {wheel_pos_right[1]:+.3f}, {wheel_pos_right[2]:+.3f}]")

    # Test correction wrenches (correction-only mode: Fz ≈ 0)
    test_wrenches = [
        jnp.array([0.0, 0.0, 0.0, 0.0, +5.0, 0.0]),
        jnp.array([0.0, 0.0, 0.0, 0.0, -5.0, 0.0]),
        jnp.array([0.0, 0.0, 0.0, 0.0, +10.0, 0.0]),
        jnp.array([0.0, 0.0, 0.0, 0.0, -10.0, 0.0]),
    ]

    print("\n" + "=" * 80)
    print("TEST: Correction-only mode (Fz ~= 0)")
    print("=" * 80)

    for i, wrench in enumerate(test_wrenches):
        Fx, Fy, Fz, Mx, My, Mz = wrench

        print(f"\nTest {i+1}: My={My:+.1f} Nm, Fz={Fz:.1f} N")

        # Distribute wrench
        f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
            desired_wrench=wrench,
            left_contact=True,
            right_contact=True,
            wheel_pos_left=wheel_pos_left,
            wheel_pos_right=wheel_pos_right,
            hip_roll_authority_scale=1.0,
            recovery_mode=False,
        )

        # Compute achieved wrench
        solution = jnp.concatenate([f_left, f_right, tau_hip_roll])
        A_wrench = contact_jacobian.build_wrench_matrix(data, wheel_pos_left, wheel_pos_right)
        achieved_wrench = A_wrench @ solution

        # Extract components
        fz_left = float(f_left[2])
        fz_right = float(f_right[2])
        fz_diff = fz_left - fz_right
        achieved_My = float(achieved_wrench[4])
        my_error = float(My - achieved_My)

        print(f"  Input:")
        print(f"    My:  {My:+.2f} Nm")
        print(f"    Fz:  {Fz:+.2f} N")
        print(f"  Output:")
        print(f"    f_left[2]:  {fz_left:+.2f} N")
        print(f"    f_right[2]: {fz_right:+.2f} N")
        print(f"    fz_diff:    {fz_diff:+.2f} N")
        print(f"    tau_hip_roll: [{tau_hip_roll[0]:+.2f}, {tau_hip_roll[1]:+.2f}] Nm")
        print(f"  Achieved:")
        print(f"    My:         {achieved_My:+.2f} Nm")
        print(f"    My error:   {my_error:+.2f} Nm")

        # Check for bug
        if abs(My) > 0.1 and abs(fz_diff) < 0.1:
            print(f"  BUG DETECTED: My={My:+.1f} Nm but fz_diff={fz_diff:+.2f} N (near zero)")
            print(f"    Roll correction channel is blocked!")
        elif abs(My) > 0.1 and abs(achieved_My) < abs(My) * 0.5:
            print(f"  BUG DETECTED: My={My:+.1f} Nm but achieved_My={achieved_My:+.2f} Nm")
            print(f"    Roll correction is severely attenuated!")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("\nExpected bug symptoms:")
    print("  - fz_diff ~= 0 even when My != 0")
    print("  - achieved My from contact-force asymmetry is near zero")
    print("  - only direct hip-roll torque remains")
    print("\nRoot cause:")
    print("  split_fz_from_my treats total_fz as absolute contact force")
    print("  With correction_Fz ~= 0, max_safe_diff becomes 0")
    print("  This blocks vertical force asymmetry generation")
    print("\nFix:")
    print("  Implement delta distribution mode for correction-only WBC")
    print("  Treat wrench as delta/correction, not absolute force")
    print("  Allow negative delta forces (reducing baseline contact load)")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_roll_correction_channel()
