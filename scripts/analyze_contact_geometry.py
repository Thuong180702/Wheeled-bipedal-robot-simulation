"""Analyze contact geometry and wrench matrix conditioning.

Checks if the wheeled biped morphology makes desired wrenches feasible.
"""

import jax.numpy as jnp
import mujoco
import numpy as np

from wheeled_biped.controllers.contact_jacobian import ContactJacobian


def analyze_wrench_matrix_conditioning(mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
    """Analyze wrench matrix conditioning and controllability."""

    # Initialize contact Jacobian
    contact_jac = ContactJacobian(mj_model)

    # Get CoM position
    com_pos = jnp.array(mj_data.subtree_com[1])  # Body 1 is torso

    # Get wheel positions
    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    l_wheel_pos = jnp.array(mj_data.xpos[l_wheel_id])
    r_wheel_pos = jnp.array(mj_data.xpos[r_wheel_id])

    # Compute relative positions
    wheel_pos_left = l_wheel_pos - com_pos
    wheel_pos_right = r_wheel_pos - com_pos

    print("=" * 80)
    print("CONTACT GEOMETRY ANALYSIS")
    print("=" * 80)
    print(f"\nCoM position: [{com_pos[0]:.4f}, {com_pos[1]:.4f}, {com_pos[2]:.4f}] m")
    print(f"Left wheel:   [{l_wheel_pos[0]:.4f}, {l_wheel_pos[1]:.4f}, {l_wheel_pos[2]:.4f}] m")
    print(f"Right wheel:  [{r_wheel_pos[0]:.4f}, {r_wheel_pos[1]:.4f}, {r_wheel_pos[2]:.4f}] m")

    print(f"\nWheel positions relative to CoM:")
    print(f"Left:  [{wheel_pos_left[0]:.4f}, {wheel_pos_left[1]:.4f}, {wheel_pos_left[2]:.4f}] m")
    print(f"Right: [{wheel_pos_right[0]:.4f}, {wheel_pos_right[1]:.4f}, {wheel_pos_right[2]:.4f}] m")

    # Compute separation
    lateral_sep = abs(wheel_pos_left[1] - wheel_pos_right[1])
    sagittal_sep = abs(wheel_pos_left[0] - wheel_pos_right[0])
    vertical_sep = abs(wheel_pos_left[2] - wheel_pos_right[2])

    print(f"\nWheel separation:")
    print(f"Lateral (y):   {lateral_sep:.4f} m")
    print(f"Sagittal (x):  {sagittal_sep:.4f} m")
    print(f"Vertical (z):  {vertical_sep:.4f} m")

    # Build wrench matrix
    A_wrench = contact_jac.build_wrench_matrix(mj_data, wheel_pos_left, wheel_pos_right)

    print("\n" + "=" * 80)
    print("WRENCH MATRIX ANALYSIS")
    print("=" * 80)
    print(f"\nA_wrench shape: {A_wrench.shape}")
    print(f"Maps 8D decision vars [f_left(3), f_right(3), tau_hip_roll(2)] to 6D wrench")

    # Compute conditioning
    A_np = np.array(A_wrench)

    # Singular value decomposition
    U, s, Vt = np.linalg.svd(A_np)

    print(f"\nSingular values:")
    for i, sv in enumerate(s):
        print(f"  s{i+1} = {sv:.6f}")

    condition_number = s[0] / s[-1] if s[-1] > 1e-10 else np.inf
    print(f"\nCondition number: {condition_number:.2e}")

    if condition_number > 1e6:
        print("[WARNING] Matrix is ill-conditioned (kappa > 1e6)")
    elif condition_number > 1e3:
        print("[CAUTION] Matrix is poorly conditioned (kappa > 1e3)")
    else:
        print("[OK] Matrix is well-conditioned")

    # Check rank
    rank = np.linalg.matrix_rank(A_np, tol=1e-6)
    print(f"\nRank: {rank}/6")

    if rank < 6:
        print(f"[WARNING] Matrix is rank-deficient (rank {rank} < 6)")
        print("Some wrench components are not independently controllable")
    else:
        print("[OK] Matrix is full rank - all wrench components controllable")

    # Analyze null space
    if A_np.shape[1] > A_np.shape[0]:
        # Overdetermined system - check null space of A^T
        null_space_dim = A_np.shape[1] - rank
        print(f"\nNull space dimension: {null_space_dim}")

        if null_space_dim > 0:
            print(f"System has {null_space_dim} degrees of freedom")
            print("Multiple force distributions can achieve same wrench")

    # Check wrench component controllability
    print("\n" + "=" * 80)
    print("WRENCH COMPONENT CONTROLLABILITY")
    print("=" * 80)

    # Test each wrench component independently
    wrench_names = ["Fx", "Fy", "Fz", "Mx (roll)", "My (pitch)", "Mz (yaw)"]

    for i, name in enumerate(wrench_names):
        # Create unit wrench in this direction
        desired_wrench = np.zeros(6)
        desired_wrench[i] = 1.0

        # Try to solve for forces (least squares)
        try:
            solution, residual, rank_i, s_i = np.linalg.lstsq(A_np, desired_wrench, rcond=None)

            # Check if solution is feasible (compressive forces)
            f_left_z = solution[2]
            f_right_z = solution[5]

            feasible = f_left_z >= -1e-6 and f_right_z >= -1e-6

            if len(residual) > 0:
                error = np.sqrt(residual[0])
            else:
                error = np.linalg.norm(A_np @ solution - desired_wrench)

            status = "[OK]" if feasible and error < 0.01 else "[WARN]"
            print(f"{status} {name:12s}: error={error:.6f}, fz_L={f_left_z:.3f}, fz_R={f_right_z:.3f}")

            if not feasible:
                print(f"   -> Requires tensile forces (not feasible)")
            if error > 0.01:
                print(f"   -> Large tracking error (wrench not achievable)")

        except np.linalg.LinAlgError:
            print(f"[WARN] {name:12s}: Singular matrix - cannot solve")

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if lateral_sep < 0.01:
        print("\n[CRITICAL] Zero lateral wheel separation")
        print("   -> Lateral forces/moments have limited authority")
        print("   -> Roll moment primarily controlled by hip roll torques")
        print("   -> Lateral CoM/CP tracking may be infeasible")

    if condition_number > 1e3:
        print("\n[WARNING] Poorly conditioned wrench matrix")
        print("   -> Small wrench errors amplified to large force errors")
        print("   -> QP solver may struggle to track desired wrench")
        print("   -> Consider relaxing wrench tracking requirements")

    if rank < 6:
        print(f"\n[WARNING] Rank-deficient wrench matrix (rank {rank})")
        print("   -> Some wrench components are coupled")
        print("   -> Independent control of all 6 DOF not possible")


def main():
    """Run contact geometry analysis."""

    # Load robot model
    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Reset to keyframe 0 (standing pose)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)

    # Analyze
    analyze_wrench_matrix_conditioning(mj_model, mj_data)

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("\n1. If lateral separation is zero:")
    print("   -> Reduce lateral CoM/CP tracking gains")
    print("   -> Rely more on hip roll torques for roll stabilization")
    print("   -> Accept limited lateral force authority")

    print("\n2. If matrix is ill-conditioned:")
    print("   -> Use soft constraints instead of hard equality")
    print("   -> Increase w_wrench weight for better tracking")
    print("   -> Add regularization to QP cost function")

    print("\n3. If rank-deficient:")
    print("   -> Identify which wrench components are controllable")
    print("   -> Adjust controller to only command feasible wrenches")
    print("   -> Consider adding more contact points")


if __name__ == "__main__":
    main()
