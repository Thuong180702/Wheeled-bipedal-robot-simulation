"""Pre-implementation diagnostics for WBC correction-only approach.

Runs three critical diagnostics before implementing correction-only WBC:
A. Zero correction equilibrium check
B. Distributor zero-input check
C. Passive/static contact feasibility

Usage:
    python scripts/debug_wbc_correction_only_diagnostics.py
"""

import numpy as np
import mujoco
import jax.numpy as jnp

from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.orientation_utils import (
    compute_robot_frame_orientation_from_quaternion,
)
from wheeled_biped.controllers.robot_model_utils import (
    get_total_robot_mass,
    get_robot_weight,
)
from scripts.simulate_hierarchical_controller import (
    calibrate_root_z_for_wheel_floor_contact,
)


MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


def measure_total_contact_force(model, data):
    """Measure total vertical contact force using proper MuJoCo API."""
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision"
    )
    r_wheel_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision"
    )

    total_fz = 0.0
    wheel_geom_ids = {l_wheel_geom_id, r_wheel_geom_id}
    contact_count = 0
    min_dist = None

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        involves_wheel = g1 in wheel_geom_ids or g2 in wheel_geom_ids
        if not (involves_floor and involves_wheel):
            continue

        contact_count += 1
        d = float(c.dist)
        min_dist = d if min_dist is None else min(min_dist, d)

        force_contact = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force_contact)
        frame = np.array(c.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        total_fz += float(force_world[2])

    return total_fz, contact_count, min_dist


def setup_calibrated_equilibrium():
    """Setup robot at calibrated equilibrium state.

    Returns:
        Tuple of (model, data, robot_mass, model_weight, equilibrium_com_z)
    """
    # Load model and reset to keyframe
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Calibrate root_z for -5e-4 contact penetration
    mujoco.mj_forward(model, data)
    calibrate_root_z_for_wheel_floor_contact(model, data, target_dist=-5e-4)

    # Zero velocities
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    # Get model-derived mass
    robot_mass = get_total_robot_mass(model)
    model_weight = get_robot_weight(model)

    # Capture equilibrium CoM z
    equilibrium_com_z = float(data.subtree_com[1, 2])

    return model, data, robot_mass, model_weight, equilibrium_com_z


def diagnostic_a_zero_correction_equilibrium():
    """Diagnostic A: Zero correction equilibrium check.

    Verifies that at calibrated equilibrium with height_cmd = equilibrium_com_z,
    correction wrench is near zero.
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC A: ZERO CORRECTION EQUILIBRIUM CHECK")
    print("=" * 80)

    # Setup calibrated equilibrium
    model, data, robot_mass, model_weight, equilibrium_com_z = (
        setup_calibrated_equilibrium()
    )

    print(f"\n[SETUP]")
    print(f"  Robot mass: {robot_mass:.4f} kg")
    print(f"  Model weight: {model_weight:.4f} N")
    print(f"  Equilibrium CoM z: {equilibrium_com_z:.6f} m")
    print(f"  Root z: {float(data.qpos[2]):.6f} m")

    # Initialize controllers
    gravity = 9.81
    centroidal_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=robot_mass, torso_inertia=jnp.array([0.1, 0.1, 0.05])
        ),
        mj_model=model,
    )

    capture_estimator = CapturePointEstimator(
        CapturePointEstimatorConfig(gravity=gravity, min_height=0.35)
    )

    wbc_controller = IntegratedWBC(
        model,
        k_roll=60.0,
        k_roll_rate=12.0,
        k_roll_integral=0.0,
        k_pitch=300.0,
        k_pitch_rate=15.0,
        k_com_lateral=15.0,
        k_com_lateral_damping=3.0,
        k_com_sagittal=50.0,
        k_com_sagittal_damping=6.0,
        k_cp_lateral=50.0,
        k_cp_sagittal=100.0,
        k_height=50.0,
        k_height_damping=0.0,
        robot_mass=robot_mass,
        gravity=gravity,
        max_roll_moment=25.0,
        wbc_authority_budget=0.95,
        max_actuator_torque=60.0,
        force_feedback_gain=0.2,
        force_feedback_warmup_steps=5,
    )

    # Build observation with height_cmd = equilibrium_com_z (zero height error)
    height_cmd = equilibrium_com_z
    obs = jnp.zeros(42)

    # CRITICAL: Set gravity in body frame for upright equilibrium orientation
    # At equilibrium, body frame aligns with world frame, so gravity_body = [0, 0, -g]
    obs = obs.at[0:3].set(jnp.array([0.0, 0.0, -gravity]))

    obs = obs.at[36].set(height_cmd)
    obs = obs.at[37].set(equilibrium_com_z)

    # Estimate state
    centroidal_state, _ = centroidal_estimator.estimate(obs, data, None)
    centroidal_state = capture_estimator.update(centroidal_state)

    # CRITICAL: Set equilibrium reference before computing corrections
    quat = data.qpos[3:7]
    from wheeled_biped.controllers.orientation_utils import (
        compute_robot_frame_orientation_from_quaternion,
    )
    pitch_x, roll_y, yaw_z = compute_robot_frame_orientation_from_quaternion(quat)

    wbc_controller.wrench_computer.set_equilibrium_reference(
        com_pos=centroidal_state.com_pos,
        com_z=equilibrium_com_z,
        pitch_x=pitch_x,
        roll_y=roll_y,
        capture_point=centroidal_state.capture_point,
        joint_pos=data.qpos[7:17].copy(),
    )

    print(f"\n[EQUILIBRIUM REFERENCE SET]")
    print(f"  CoM position: [{float(centroidal_state.com_pos[0]):.6f}, "
          f"{float(centroidal_state.com_pos[1]):.6f}, "
          f"{float(centroidal_state.com_pos[2]):.6f}] m")
    print(f"  Pitch: {pitch_x * 180 / np.pi:.4f}°")
    print(f"  Roll: {roll_y * 180 / np.pi:.4f}°")
    print(f"  Capture point: [{float(centroidal_state.capture_point[0]):.6f}, "
          f"{float(centroidal_state.capture_point[1]):.6f}] m")

    # Compute current WBC wrench with breakdown
    desired_force, desired_moment, correction_breakdown = (
        wbc_controller.wrench_computer.compute_desired_wrench_with_breakdown(
            obs, centroidal_state, height_cmd, roll_integral=0.0
        )
    )
    current_wrench = jnp.concatenate([desired_force, desired_moment])

    # Compute proposed baseline wrench (diagnostic only)
    baseline_wrench = jnp.array([0.0, 0.0, model_weight, 0.0, 0.0, 0.0])

    # Compute proposed correction wrench (control output)
    # This is what correction-only WBC would produce
    correction_wrench = current_wrench - baseline_wrench

    print(f"\n[WRENCH COMPUTATION]")
    print(f"  Current WBC wrench (baseline + correction):")
    print(f"    Fx: {float(current_wrench[0]):7.3f} N")
    print(f"    Fy: {float(current_wrench[1]):7.3f} N")
    print(f"    Fz: {float(current_wrench[2]):7.3f} N")
    print(f"    Mx: {float(current_wrench[3]):7.3f} Nm")
    print(f"    My: {float(current_wrench[4]):7.3f} Nm")
    print(f"    Mz: {float(current_wrench[5]):7.3f} Nm")

    print(f"\n  Proposed baseline wrench (diagnostic only, NOT mapped through J^T f):")
    print(f"    Fz: {float(baseline_wrench[2]):7.3f} N (should equal model_weight)")

    print(f"\n  Proposed correction wrench (control output, mapped through J^T f):")
    print(f"    Fx: {float(correction_wrench[0]):7.3f} N")
    print(f"    Fy: {float(correction_wrench[1]):7.3f} N")
    print(f"    Fz: {float(correction_wrench[2]):7.3f} N")
    print(f"    Mx: {float(correction_wrench[3]):7.3f} Nm")
    print(f"    My: {float(correction_wrench[4]):7.3f} Nm")
    print(f"    Mz: {float(correction_wrench[5]):7.3f} Nm")

    print(f"\n  Correction breakdown from wrench computer:")
    print(f"    com_error_y: {correction_breakdown['com_error_y']:.6f} m")
    print(f"    cp_error_y: {correction_breakdown['cp_error_y']:.6f} m")
    print(f"    pitch_error: {correction_breakdown['pitch_error']:.6f} rad")
    print(f"    correction_Fy_com: {correction_breakdown['correction_Fy_com']:.3f} N")
    print(f"    correction_Fy_cp: {correction_breakdown['correction_Fy_cp']:.3f} N")
    print(f"    correction_Fy_pitch: {correction_breakdown['correction_Fy_pitch']:.3f} N")
    print(f"    correction_wrench_Fy (total): {correction_breakdown['correction_wrench_Fy']:.3f} N")
    print(f"    correction_wrench_norm: {correction_breakdown['correction_wrench_norm']:.3f} N")

    # Compute metrics
    correction_wrench_norm = float(jnp.linalg.norm(correction_wrench))
    correction_fz = float(correction_wrench[2])
    correction_fx = float(correction_wrench[0])
    correction_fy = float(correction_wrench[1])
    correction_my = float(correction_wrench[4])

    threshold_10pct = 0.10 * model_weight
    threshold_5pct = 0.05 * model_weight

    print(f"\n[VERIFICATION]")
    print(f"  Correction wrench norm: {correction_wrench_norm:.3f} N")
    print(f"  Threshold (10% model weight): {threshold_10pct:.3f} N")
    print(f"  Pass: {correction_wrench_norm < threshold_10pct}")

    print(f"\n  Correction Fz: {correction_fz:.3f} N")
    print(f"  Threshold (5% model weight): {threshold_5pct:.3f} N")
    print(f"  Pass: {abs(correction_fz) < threshold_5pct}")

    print(f"\n  Correction Fx: {correction_fx:.3f} N (should be near zero)")
    print(f"  Correction Fy: {correction_fy:.3f} N (should be near zero)")
    print(f"  Correction My: {correction_my:.3f} Nm (should be near zero)")

    # State diagnostics
    quat = data.qpos[3:7]
    pitch_x, roll_y, yaw_z = compute_robot_frame_orientation_from_quaternion(quat)

    print(f"\n[STATE DIAGNOSTICS]")
    print(f"  Pitch: {pitch_x * 180 / np.pi:.4f}°")
    print(f"  Roll: {roll_y * 180 / np.pi:.4f}°")
    print(f"  Yaw: {yaw_z * 180 / np.pi:.4f}°")
    print(f"  CoM position: [{float(centroidal_state.com_pos[0]):.6f}, "
          f"{float(centroidal_state.com_pos[1]):.6f}, "
          f"{float(centroidal_state.com_pos[2]):.6f}] m")
    print(f"  CoM velocity: [{float(centroidal_state.com_vel[0]):.6f}, "
          f"{float(centroidal_state.com_vel[1]):.6f}, "
          f"{float(centroidal_state.com_vel[2]):.6f}] m/s")
    print(f"  Capture point: [{float(centroidal_state.capture_point[0]):.6f}, "
          f"{float(centroidal_state.capture_point[1]):.6f}] m")

    # Overall verdict
    pass_norm = correction_wrench_norm < threshold_10pct
    pass_fz = abs(correction_fz) < threshold_5pct

    print(f"\n[DIAGNOSTIC A VERDICT]")
    if pass_norm and pass_fz:
        print("  [PASS] Correction wrench is near zero at calibrated equilibrium")
        print("  Safe to proceed with correction-only WBC implementation")
    else:
        print("  [FAIL] Correction wrench is NOT near zero at equilibrium")
        print("  Investigate equilibrium reference computation before implementation")

    return pass_norm and pass_fz


def diagnostic_b_distributor_zero_input():
    """Diagnostic B: Distributor zero-input check.

    Verifies that SimpleForceDistributor produces zero force when
    correction_wrench = 0.
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC B: DISTRIBUTOR ZERO-INPUT CHECK")
    print("=" * 80)

    # Setup calibrated equilibrium
    model, data, robot_mass, model_weight, equilibrium_com_z = (
        setup_calibrated_equilibrium()
    )

    print(f"\n[SETUP]")
    print(f"  Robot mass: {robot_mass:.4f} kg")
    print(f"  Model weight: {model_weight:.4f} N")

    # Initialize WBC controller to access force distributor
    gravity = 9.81
    wbc_controller = IntegratedWBC(
        model,
        robot_mass=robot_mass,
        gravity=gravity,
    )

    # Compute wheel positions relative to CoM
    com_pos = jnp.array(data.subtree_com[1])
    wheel_pos_left, wheel_pos_right = wbc_controller._compute_wheel_positions_relative_to_com(
        data, com_pos
    )

    print(f"\n[WHEEL POSITIONS RELATIVE TO COM]")
    print(f"  Left wheel: [{float(wheel_pos_left[0]):.6f}, "
          f"{float(wheel_pos_left[1]):.6f}, {float(wheel_pos_left[2]):.6f}] m")
    print(f"  Right wheel: [{float(wheel_pos_right[0]):.6f}, "
          f"{float(wheel_pos_right[1]):.6f}, {float(wheel_pos_right[2]):.6f}] m")

    # Test zero correction wrench
    correction_wrench = jnp.zeros(6)

    print(f"\n[TEST 1: DOUBLE CONTACT]")
    f_left, f_right, tau_hip_roll, diag = (
        wbc_controller.force_distributor.distribute_wrench_contact_aware(
            correction_wrench,
            left_contact=True,
            right_contact=True,
            wheel_pos_left=wheel_pos_left,
            wheel_pos_right=wheel_pos_right,
            hip_roll_authority_scale=1.0,
        )
    )

    f_left_z = float(f_left[2])
    f_right_z = float(f_right[2])
    f_total_z = f_left_z + f_right_z

    print(f"  Left wheel Fz: {f_left_z:.3f} N")
    print(f"  Right wheel Fz: {f_right_z:.3f} N")
    print(f"  Total Fz: {f_total_z:.3f} N")
    print(f"  Pass (< 1.0 N): {abs(f_total_z) < 1.0}")

    pass_double = abs(f_total_z) < 1.0

    print(f"\n[TEST 2: SINGLE CONTACT (LEFT ONLY)]")
    f_left_single, f_right_single, _, _ = (
        wbc_controller.force_distributor.distribute_wrench_contact_aware(
            correction_wrench,
            left_contact=True,
            right_contact=False,
            wheel_pos_left=wheel_pos_left,
            wheel_pos_right=wheel_pos_right,
            hip_roll_authority_scale=1.0,
        )
    )

    f_left_single_z = float(f_left_single[2])
    f_right_single_z = float(f_right_single[2])

    print(f"  Left wheel Fz: {f_left_single_z:.3f} N")
    print(f"  Right wheel Fz (non-contact): {f_right_single_z:.3f} N")
    print(f"  Pass left (< 1.0 N): {abs(f_left_single_z) < 1.0}")
    print(f"  Pass right (< 0.1 N): {abs(f_right_single_z) < 0.1}")

    pass_single = abs(f_left_single_z) < 1.0 and abs(f_right_single_z) < 0.1

    print(f"\n[TEST 3: NO CONTACT]")
    f_left_none, f_right_none, _, _ = (
        wbc_controller.force_distributor.distribute_wrench_contact_aware(
            correction_wrench,
            left_contact=False,
            right_contact=False,
            wheel_pos_left=wheel_pos_left,
            wheel_pos_right=wheel_pos_right,
            hip_roll_authority_scale=1.0,
        )
    )

    f_left_none_z = float(f_left_none[2])
    f_right_none_z = float(f_right_none[2])
    f_total_none_z = f_left_none_z + f_right_none_z

    print(f"  Left wheel Fz: {f_left_none_z:.3f} N")
    print(f"  Right wheel Fz: {f_right_none_z:.3f} N")
    print(f"  Total Fz: {f_total_none_z:.3f} N")
    print(f"  Pass (< 0.1 N): {abs(f_total_none_z) < 0.1}")

    pass_none = abs(f_total_none_z) < 0.1

    # Overall verdict
    print(f"\n[DIAGNOSTIC B VERDICT]")
    if pass_double and pass_single and pass_none:
        print("  [PASS] SimpleForceDistributor produces zero force for zero correction")
        print("  No force floor injection detected")
        print("  Safe to proceed with correction-only WBC implementation")
    else:
        print("  [FAIL] SimpleForceDistributor injects force even with zero correction")
        print("  Likely cause: min_recovery_force or min_wheel_force behavior")
        print("  MUST fix distributor before implementing correction-only WBC")

    return pass_double and pass_single and pass_none


def diagnostic_c_passive_contact_feasibility():
    """Diagnostic C: Passive/static contact feasibility.

    Verifies whether contact constraints alone can support robot weight
    without actuator torques.
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC C: PASSIVE/STATIC CONTACT FEASIBILITY")
    print("=" * 80)

    # Setup calibrated equilibrium
    model, data, robot_mass, model_weight, equilibrium_com_z = (
        setup_calibrated_equilibrium()
    )

    print(f"\n[SETUP]")
    print(f"  Robot mass: {robot_mass:.4f} kg")
    print(f"  Model weight: {model_weight:.4f} N")
    print(f"  Equilibrium CoM z: {equilibrium_com_z:.6f} m")
    print(f"  Root z: {float(data.qpos[2]):.6f} m")

    # Disable all controllers: tau = 0
    data.ctrl[:] = 0.0

    print(f"\n[RUNNING 20 STEPS WITH TAU = 0]")
    print(f"{'Step':>4s} {'Contact Fz':>12s} {'CoM z':>10s} {'CoM vz':>10s} "
          f"{'Pitch':>10s} {'Roll':>10s} {'Contacts':>8s} {'Min Dist':>10s}")
    print("-" * 80)

    max_com_z_drift = 0.0
    max_com_vz = 0.0
    max_pitch = 0.0
    max_roll = 0.0
    min_contact_fz = float('inf')
    max_contact_fz = 0.0

    for step in range(20):
        mujoco.mj_step(model, data)

        # Measure contact forces
        contact_fz, contact_count, min_dist = measure_total_contact_force(model, data)

        # Measure state
        com_z = float(data.subtree_com[1, 2])
        com_vz = float(data.qvel[2])
        quat = data.qpos[3:7]
        pitch_x, roll_y, _ = compute_robot_frame_orientation_from_quaternion(quat)

        # Track extremes
        com_z_drift = abs(com_z - equilibrium_com_z)
        max_com_z_drift = max(max_com_z_drift, com_z_drift)
        max_com_vz = max(max_com_vz, abs(com_vz))
        max_pitch = max(max_pitch, abs(pitch_x))
        max_roll = max(max_roll, abs(roll_y))
        min_contact_fz = min(min_contact_fz, contact_fz)
        max_contact_fz = max(max_contact_fz, contact_fz)

        print(f"{step:4d} {contact_fz:12.2f} N {com_z:10.6f} m {com_vz:10.6f} m/s "
              f"{pitch_x * 180 / np.pi:10.4f}° {roll_y * 180 / np.pi:10.4f}° "
              f"{contact_count:8d} {min_dist if min_dist is not None else 0.0:10.6f} m")

    # Verification
    contact_fz_error_pct = abs(max_contact_fz - model_weight) / model_weight * 100
    pass_contact_fz = contact_fz_error_pct < 20.0
    pass_com_z_drift = max_com_z_drift < 0.02
    pass_com_vz = max_com_vz < 0.05
    pass_pitch = max_pitch < 0.087  # 5 degrees
    pass_roll = max_roll < 0.087  # 5 degrees

    print(f"\n[VERIFICATION]")
    print(f"  Max CoM z drift: {max_com_z_drift:.6f} m (threshold: 0.02 m)")
    print(f"  Pass: {pass_com_z_drift}")

    print(f"\n  Max CoM vz: {max_com_vz:.6f} m/s (threshold: 0.05 m/s)")
    print(f"  Pass: {pass_com_vz}")

    print(f"\n  Max pitch: {max_pitch * 180 / np.pi:.4f}° (threshold: 5.0°)")
    print(f"  Pass: {pass_pitch}")

    print(f"\n  Max roll: {max_roll * 180 / np.pi:.4f}° (threshold: 5.0°)")
    print(f"  Pass: {pass_roll}")

    print(f"\n  Contact Fz range: [{min_contact_fz:.2f}, {max_contact_fz:.2f}] N")
    print(f"  Model weight: {model_weight:.2f} N")
    print(f"  Max error: {contact_fz_error_pct:.1f}% (threshold: 20%)")
    print(f"  Pass: {pass_contact_fz}")

    # Overall verdict
    pass_all = (
        pass_contact_fz and pass_com_z_drift and pass_com_vz and pass_pitch and pass_roll
    )

    print(f"\n[DIAGNOSTIC C VERDICT]")
    if pass_all:
        print("  [PASS] Contact constraints provide stable baseline support")
        print("  Robot remains stable with tau=0 for 20 steps")
        print("  Correction-only WBC is physically sound for this robot/simulator")
    elif pass_contact_fz and not (pass_com_z_drift and pass_com_vz):
        print("  [PARTIAL] Contact forces stable but robot slowly drifts/sags")
        print("  Contact constraints provide partial support")
        print("  A separate posture/static holding controller IS needed")
        print("  alongside correction-only WBC")
    else:
        print("  [FAIL] Contact constraints do NOT provide baseline support")
        print("  Robot collapses or contact forces unstable")
        print("  Investigate:")
        print("    - Calibration producing proper wheel-floor contact?")
        print("    - Contact solver parameters correct?")
        print("    - Floor/wheel collision geoms active?")
        print("    - Contact stiffness/damping reasonable?")

    return pass_all


def main():
    """Run all three pre-implementation diagnostics."""
    print("\n" + "=" * 80)
    print("WBC CORRECTION-ONLY PRE-IMPLEMENTATION DIAGNOSTICS")
    print("=" * 80)
    print("\nThese diagnostics verify the physics assumptions before implementing")
    print("correction-only WBC. All three must pass or clearly identify blockers.")

    # Run diagnostics
    pass_a = diagnostic_a_zero_correction_equilibrium()
    pass_b = diagnostic_b_distributor_zero_input()
    pass_c = diagnostic_c_passive_contact_feasibility()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\n[DIAGNOSTIC RESULTS]")
    print(f"  A. Zero correction equilibrium: {'PASS' if pass_a else 'FAIL'}")
    print(f"  B. Distributor zero-input: {'PASS' if pass_b else 'FAIL'}")
    print(f"  C. Passive contact feasibility: {'PASS' if pass_c else 'FAIL'}")

    print(f"\n[NEXT STEPS]")
    if pass_a and pass_b and pass_c:
        print("  [OK] All diagnostics passed")
        print("  [OK] Safe to write implementation plan for correction-only WBC")
        print("  [OK] Physics assumptions validated")
        print("  [OK] No distributor fixes needed")
        print("  [OK] Contact constraints provide full baseline support")
    elif not pass_a:
        print("  [FAIL] Diagnostic A failed: correction wrench NOT near zero at equilibrium")
        print("  -> Fix equilibrium reference computation before implementation")
        print("  -> Verify height_cmd = equilibrium_com_z")
        print("  -> Check that all velocities are zero")
        print("  -> Verify pitch/roll near zero after calibration")
    elif not pass_b:
        print("  [FAIL] Diagnostic B failed: distributor injects force with zero correction")
        print("  -> Fix SimpleForceDistributor before implementation")
        print("  -> Remove/gate min_recovery_force behavior")
        print("  -> Ensure zero correction -> zero distributed force")
        print("  -> Ensure non-contact wheels receive zero force")
    elif not pass_c:
        print("  [WARN] Diagnostic C failed: contact constraints alone insufficient")
        print("  -> This does NOT invalidate correction-only WBC")
        print("  -> It means a separate posture/static holding controller IS needed")
        print("  -> Correction-only WBC handles perturbations, posture controller")
        print("     handles internal joint gravity compensation")
        print("  -> Safe to proceed with implementation, but note this requirement")
    else:
        print("  [WARN] Multiple diagnostics failed")
        print("  -> Address failures in order: A, then B, then C")
        print("  -> Do not implement correction-only WBC until A and B pass")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
