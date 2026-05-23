"""Static support parity test with StaticBalanceController wrapper.

Compares wrapped WBC behavior against inverse dynamics baseline.
"""

import numpy as np
import mujoco

SUPPORT_JOINTS = [2, 3, 7, 8]
MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


def measure_total_contact_force(model, data):
    """Measure total vertical contact force using proper MuJoCo API."""
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    total_fz = 0.0
    wheel_geom_ids = {l_wheel_geom_id, r_wheel_geom_id}

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        involves_wheel = g1 in wheel_geom_ids or g2 in wheel_geom_ids
        if not (involves_floor and involves_wheel):
            continue

        # Use proper MuJoCo API: mj_contactForce + contact.frame
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force_contact)
        frame = np.array(c.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        total_fz += float(force_world[2])

    return total_fz


def run_case_b(enable_wrapper=False):
    """Run Case B: Current pipeline with optional wrapper."""
    from wheeled_biped.controllers.static_balance_controller import StaticBalanceController
    from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
    from wheeled_biped.controllers.centroidal_state_estimator import (
        CentroidalStateEstimator,
        CentroidalStateEstimatorConfig,
    )
    from wheeled_biped.controllers.capture_point_estimator import (
        CapturePointEstimator,
        CapturePointEstimatorConfig,
    )
    from wheeled_biped.controllers.orientation_utils import compute_robot_frame_orientation_from_quaternion
    import jax.numpy as jnp

    # Load model and data
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Initialize controllers
    robot_mass = 15.0
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

    # Initialize wrapper if enabled
    static_balance_controller = None
    if enable_wrapper:
        static_balance_controller = StaticBalanceController(
            model, data, wbc_controller, calibration_config={'target_contact_dist': -5e-4}
        )

    # Build observation
    height_cmd = 0.534
    obs = jnp.zeros(42)
    obs = obs.at[36].set(height_cmd)
    obs = obs.at[37].set(data.subtree_com[1, 2])

    # Estimate state
    centroidal_state, _ = centroidal_estimator.estimate(obs, data, None)
    centroidal_state = capture_estimator.update(centroidal_state)

    # Compute WBC torque
    tau_wbc, _ = wbc_controller.compute_wbc_torque_with_diagnostics(
        data, obs, centroidal_state, height_cmd
    )

    # Apply wrapper if enabled
    if static_balance_controller is not None:
        quat = data.qpos[3:7]
        pitch_x, roll_y, _ = compute_robot_frame_orientation_from_quaternion(quat)
        current_state = {
            'com_z': float(data.subtree_com[1, 2]),
            'pitch_x': float(pitch_x),
            'roll_y': float(roll_y),
            'joint_pos': data.qpos[7:17].copy(),
            'com_vel': data.qvel[0:3].copy(),
            'angular_vel': data.qvel[3:6].copy(),
        }
        tau_wbc_wrapped, _ = static_balance_controller.wrap(np.array(tau_wbc), current_state)
        tau_wbc = tau_wbc_wrapped

    # Apply and step
    data.ctrl[:] = np.array(tau_wbc)
    mujoco.mj_step(model, data)

    contact_fz = measure_total_contact_force(model, data)

    return np.array(tau_wbc), contact_fz


def run_case_d():
    """Run Case D: Inverse dynamics baseline."""
    from scripts.simulate_hierarchical_controller import calibrate_root_z_for_wheel_floor_contact

    # Load model and data
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Calibrate (reuse existing helper)
    calibrate_root_z_for_wheel_floor_contact(model, data)

    # Zero velocities
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    # Forward then inverse dynamics
    mujoco.mj_forward(model, data)
    mujoco.mj_inverse(model, data)

    tau_id = data.qfrc_inverse[6:16].copy()

    # Apply and step
    data.ctrl[:] = tau_id
    mujoco.mj_step(model, data)

    contact_fz = measure_total_contact_force(model, data)

    return tau_id, contact_fz


def main():
    """Run static support parity comparison."""
    print("\n[STATIC SUPPORT PARITY TEST V2]\n")

    # Case B: Old WBC (wrapper disabled)
    print("Running Case B (old WBC)...")
    tau_old_wbc, contact_fz_old = run_case_b(enable_wrapper=False)
    print(f"  tau[{SUPPORT_JOINTS}]: {tau_old_wbc[SUPPORT_JOINTS]}")
    print(f"  contact_fz: {contact_fz_old:.1f} N")

    # Case B': Wrapped WBC (wrapper enabled)
    print("\nRunning Case B' (wrapped WBC)...")
    tau_wrapped, contact_fz_wrapped = run_case_b(enable_wrapper=True)
    print(f"  tau[{SUPPORT_JOINTS}]: {tau_wrapped[SUPPORT_JOINTS]}")
    print(f"  contact_fz: {contact_fz_wrapped:.1f} N")

    # Case D: Inverse dynamics baseline
    print("\nRunning Case D (inverse dynamics)...")
    tau_id, contact_fz_id = run_case_d()
    print(f"  tau[{SUPPORT_JOINTS}]: {tau_id[SUPPORT_JOINTS]}")
    print(f"  contact_fz: {contact_fz_id:.1f} N")

    # Comparison
    print("\n[COMPARISON]")
    error_old = np.linalg.norm(tau_old_wbc[SUPPORT_JOINTS] - tau_id[SUPPORT_JOINTS])
    error_wrapped = np.linalg.norm(tau_wrapped[SUPPORT_JOINTS] - tau_id[SUPPORT_JOINTS])

    print(f"Torque RMSE (old WBC vs inverse dynamics): {error_old:.2f} Nm")
    print(f"Torque RMSE (wrapped WBC vs inverse dynamics): {error_wrapped:.2f} Nm")

    if error_old > 0:
        improvement = (error_old - error_wrapped) / error_old * 100
        print(f"Improvement: {improvement:.1f}%")

    fz_error_old = abs(contact_fz_old - 79.5)
    fz_error_wrapped = abs(contact_fz_wrapped - 79.5)

    print(f"\nContact force error (old WBC): {fz_error_old:.1f} N")
    print(f"Contact force error (wrapped WBC): {fz_error_wrapped:.1f} N")

    if fz_error_old > 0:
        fz_improvement = (fz_error_old - fz_error_wrapped) / fz_error_old * 100
        print(f"Improvement: {fz_improvement:.1f}%")

    # Verdict
    print("\n[VERDICT]")
    if error_wrapped < error_old and fz_error_wrapped < fz_error_old:
        print("✅ Wrapped WBC is closer to inverse dynamics than old WBC")
    elif error_wrapped < error_old or fz_error_wrapped < fz_error_old:
        print("⚠️ Wrapped WBC shows partial improvement over old WBC")
    else:
        print("❌ Wrapped WBC did not improve over old WBC")


if __name__ == "__main__":
    main()

