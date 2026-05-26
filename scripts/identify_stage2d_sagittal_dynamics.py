#!/usr/bin/env python3
"""
Phase 1: Sagittal dynamics identification for Stage2D LQR controller.

Identifies discrete-time linear dynamics around equilibrium:
    x_{t+1} = A x_t + B u_t

State vector: x = [pitch_x, pitch_rate_x, cp_error_y, com_vy, wheel_vel_mean]
Input: u = common wheel torque (Nm)

Uses small perturbations around calibrated equilibrium with:
- Static posture holding enabled
- Static feedforward enabled
- WBC disabled
- Roll controller disabled
- Stage2B/2C sagittal controllers disabled

Output: A matrix, B vector, controllability analysis, model validation.
"""

import argparse
import numpy as np
from pathlib import Path
import mujoco
from scipy import linalg

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.orientation_utils import compute_orientation_from_gravity
from wheeled_biped.controllers.static_posture_holding_controller import StaticPostureHoldingController
from wheeled_biped.controllers.static_feedforward_controller import (
    StaticFeedforwardController,
    load_empirical_feedforward_from_telemetry,
)


STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD = np.array([
    0.0, 0.0, 0.0, -15.5, 0.0,
    0.0, 0.0, 0.0, -15.8, 0.0,
], dtype=np.float64)


def extract_state(mj_data, centroidal_state, equilibrium_cp_y):
    """Extract 5-dimensional state vector.

    State: [pitch_x, pitch_rate_x, cp_error_y, com_vy, wheel_vel_mean]

    Note: Does NOT include absolute com_y, as Phase 0 analysis showed
    it couples badly and destabilizes the controller.
    """
    # Robot-frame pitch from gravity vector
    quat = mj_data.qpos[3:7]
    R = np.zeros((3, 3))
    mujoco.mju_quat2Mat(R.ravel(), quat)
    gravity_body = R.T @ np.array([0, 0, -1])
    pitch_x = np.arctan2(-gravity_body[1], gravity_body[2])

    # Pitch rate from body angular velocity
    pitch_rate_x = mj_data.qvel[3]

    # Capture point error (sagittal Y axis)
    cp_y = centroidal_state.capture_point[1]
    cp_error_y = cp_y - equilibrium_cp_y

    # CoM velocity (sagittal Y axis)
    com_vy = centroidal_state.com_vel[1]

    # Mean wheel velocity
    wheel_vel_left = mj_data.qvel[10]  # l_wheel in qvel
    wheel_vel_right = mj_data.qvel[15]  # r_wheel in qvel
    wheel_vel_mean = 0.5 * (wheel_vel_left + wheel_vel_right)

    return np.array([pitch_x, pitch_rate_x, cp_error_y, com_vy, wheel_vel_mean])


def run_perturbation_experiment(
    mj_model,
    mj_data_init,
    static_posture_controller,
    static_feedforward_controller,
    centroidal_estimator,
    capture_estimator,
    equilibrium_cp_y,
    wheel_torque_perturbation,
    steps=10,
    control_dt=0.002,
):
    """Run a single perturbation experiment with constant wheel torque.

    Returns:
        states: (steps+1, 5) array of state vectors
        inputs: (steps,) array of applied wheel torques
        valid: bool, whether the experiment stayed within valid region
    """
    # Copy initial state
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = mj_data_init.qpos
    mj_data.qvel[:] = mj_data_init.qvel
    mujoco.mj_forward(mj_model, mj_data)

    # Storage
    states = []
    inputs = []

    # Extract initial state
    obs = np.concatenate([mj_data.qpos, mj_data.qvel])
    centroidal_state = centroidal_estimator.estimate(obs, mj_data, None)[0]
    centroidal_state = capture_estimator.update(centroidal_state)

    x0 = extract_state(mj_data, centroidal_state, equilibrium_cp_y)
    states.append(x0)

    valid = True

    for step in range(steps):
        # Get current state
        joint_pos = mj_data.qpos[7:17]
        joint_vel = mj_data.qvel[6:16]

        # Compute base control (static posture + feedforward)
        tau_static_posture = static_posture_controller.compute_torques(joint_pos, joint_vel)
        tau_static_feedforward = static_feedforward_controller.compute_feedforward(step)

        # Add wheel perturbation
        tau_total = tau_static_posture + tau_static_feedforward
        tau_total[4] += wheel_torque_perturbation  # l_wheel
        tau_total[9] += wheel_torque_perturbation  # r_wheel

        # Clip to actuator limits
        torque_limit = mj_model.actuator_ctrlrange[:, 1]
        tau_clipped = np.clip(tau_total, -torque_limit, torque_limit)

        # Apply control
        mj_data.ctrl[:] = tau_clipped

        # Step simulation (multiple substeps for control_dt)
        substeps = int(control_dt / mj_model.opt.timestep)
        for _ in range(substeps):
            mujoco.mj_step(mj_model, mj_data)

        # Extract next state
        obs = np.concatenate([mj_data.qpos, mj_data.qvel])
        centroidal_state = centroidal_estimator.estimate(obs, mj_data, None)[0]
        centroidal_state = capture_estimator.update(centroidal_state)

        x_next = extract_state(mj_data, centroidal_state, equilibrium_cp_y)
        states.append(x_next)
        inputs.append(wheel_torque_perturbation)

        # Check validity (reject if too far from equilibrium or contact lost)
        com_z = centroidal_state.com_pos[2]
        if abs(x_next[0]) > 0.3 or com_z < 0.45:  # pitch > 17 deg or height too low
            valid = False
            break

    return np.array(states), np.array(inputs), valid


def identify_linear_model(states_list, inputs_list):
    """Identify discrete linear model from collected data.

    Model: x_{t+1} = A x_t + B u_t

    Args:
        states_list: list of (T+1, 5) state arrays
        inputs_list: list of (T,) input arrays

    Returns:
        A: (5, 5) state transition matrix
        B: (5,) input matrix
        residual: prediction error norm
    """
    # Stack all transitions
    X = []  # Current states
    U = []  # Inputs
    Y = []  # Next states

    for states, inputs in zip(states_list, inputs_list):
        for t in range(len(inputs)):
            X.append(states[t])
            U.append(inputs[t])
            Y.append(states[t+1])

    X = np.array(X)  # (N, 5)
    U = np.array(U).reshape(-1, 1)  # (N, 1)
    Y = np.array(Y)  # (N, 5)

    # Least squares: Y = X @ A.T + U @ B.T
    # Stack [X, U] and solve for [A.T; B.T]
    Z = np.hstack([X, U])  # (N, 6)

    # Solve for each output dimension
    AB = np.linalg.lstsq(Z, Y, rcond=None)[0]  # (6, 5)

    A = AB[:5, :].T  # (5, 5)
    B = AB[5, :]     # (5,)

    # Compute residual
    Y_pred = X @ A.T + U @ B.reshape(1, -1)
    residual = np.linalg.norm(Y - Y_pred) / np.sqrt(len(Y))

    return A, B, residual


def check_controllability(A, B):
    """Check controllability of (A, B) system.

    Returns:
        rank: rank of controllability matrix
        is_controllable: whether system is fully controllable
    """
    n = A.shape[0]

    # Build controllability matrix: [B, AB, A^2B, ..., A^{n-1}B]
    C = np.zeros((n, n))
    AB = B.copy()

    for i in range(n):
        C[:, i] = AB
        AB = A @ AB

    rank = np.linalg.matrix_rank(C)
    is_controllable = (rank == n)

    return rank, is_controllable


def main():
    parser = argparse.ArgumentParser(description="Identify sagittal dynamics for Stage2D LQR")
    parser.add_argument("--steps-per-experiment", type=int, default=10, help="Steps per perturbation")
    parser.add_argument("--control-dt", type=float, default=0.002, help="Control timestep (s)")
    parser.add_argument("--output-dir", type=str, default="outputs/stage2d_sysid", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Stage2D Sagittal Dynamics Identification")
    print("="*60)

    # Load model
    model_path = "assets/robot/wheeled_biped_real.xml"
    print(f"\nLoading model: {model_path}")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Initialize to keyframe 0
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

    mujoco.mj_forward(mj_model, mj_data)

    # Create controllers (equilibrium configuration)
    static_posture_controller = StaticPostureHoldingController(
        kp_hip_roll=5.0, kd_hip_roll=1.0,
        kp_hip_yaw=5.0, kd_hip_yaw=1.0,
        kp_hip_pitch=30.0, kd_hip_pitch=4.0,
        kp_knee=40.0, kd_knee=5.0,
        max_torque_hip_roll=15.0, max_torque_hip_yaw=15.0,
        max_torque_hip_pitch=30.0, max_torque_knee=30.0,
    )

    static_feedforward_controller = StaticFeedforwardController(
        empirical_feedforward=STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD,
        scale=0.5,
        joint_group='knee',
        ramp_mode='instant',
        sign='positive',
    )

    # Set equilibrium reference
    equilibrium_joint_pos = mj_data.qpos[7:17].copy()
    static_posture_controller.set_equilibrium_reference(equilibrium_joint_pos)

    # Create state estimators
    centroidal_config = CentroidalStateEstimatorConfig()
    centroidal_estimator = CentroidalStateEstimator(mj_model, centroidal_config)

    capture_config = CapturePointEstimatorConfig()
    capture_estimator = CapturePointEstimator(capture_config)

    # Get equilibrium capture point
    obs = np.concatenate([mj_data.qpos, mj_data.qvel])
    centroidal_state_eq = centroidal_estimator.estimate(obs, mj_data, None)[0]
    centroidal_state_eq = capture_estimator.update(centroidal_state_eq)
    equilibrium_cp_y = centroidal_state_eq.capture_point[1]

    print(f"\nEquilibrium capture point Y: {equilibrium_cp_y:.6f} m")

    # Run perturbation experiments
    perturbations = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    print(f"\nRunning perturbation experiments:")
    print(f"  Perturbations: {perturbations} Nm")
    print(f"  Steps per experiment: {args.steps_per_experiment}")
    print(f"  Control dt: {args.control_dt} s")

    states_list = []
    inputs_list = []

    for u_pert in perturbations:
        print(f"\n  Running u = {u_pert:+.1f} Nm...", end=" ")

        states, inputs, valid = run_perturbation_experiment(
            mj_model=mj_model,
            mj_data_init=mj_data,
            static_posture_controller=static_posture_controller,
            static_feedforward_controller=static_feedforward_controller,
            centroidal_estimator=centroidal_estimator,
            capture_estimator=capture_estimator,
            equilibrium_cp_y=equilibrium_cp_y,
            wheel_torque_perturbation=u_pert,
            steps=args.steps_per_experiment,
            control_dt=args.control_dt,
        )

        if valid:
            states_list.append(states)
            inputs_list.append(inputs)
            print(f"OK ({len(states)-1} steps)")
        else:
            print(f"REJECTED (contact lost or large deviation)")

    if len(states_list) < 3:
        print("\n[ERROR] Not enough valid experiments for identification")
        return

    print(f"\nCollected {len(states_list)} valid experiments")

    # Identify linear model
    print("\nIdentifying linear model...")
    A, B, residual = identify_linear_model(states_list, inputs_list)

    print("\n" + "="*60)
    print("IDENTIFIED MODEL")
    print("="*60)

    print("\nState vector: x = [pitch_x, pitch_rate_x, cp_error_y, com_vy, wheel_vel_mean]")
    print("Input: u = wheel_torque (Nm)")
    print("\nDiscrete dynamics: x_{t+1} = A x_t + B u_t")

    print("\nA matrix (5x5):")
    print(A)

    print("\nB vector (5,):")
    print(B)

    print(f"\nPrediction residual (RMS): {residual:.6f}")

    # Check controllability
    rank, is_controllable = check_controllability(A, B)
    print(f"\nControllability:")
    print(f"  Rank: {rank}/5")
    print(f"  Fully controllable: {is_controllable}")

    # Check B sign consistency
    print(f"\nB vector sign analysis:")
    print(f"  B[0] (pitch_x):       {B[0]:+.6f}  {'✓ positive' if B[0] > 0 else '✗ negative'}")
    print(f"  B[1] (pitch_rate_x):  {B[1]:+.6f}")
    print(f"  B[2] (cp_error_y):    {B[2]:+.6f}")
    print(f"  B[3] (com_vy):        {B[3]:+.6f}")
    print(f"  B[4] (wheel_vel_mean):{B[4]:+.6f}")

    print("\nExpected sign convention:")
    print("  Positive wheel torque → robot moves backward (+Y)")
    print("  Positive pitch_x → falling forward (-Y)")
    print("  → B[0] should be POSITIVE (torque opposes pitch)")

    if B[0] > 0:
        print("\n✓ B[0] sign is CORRECT")
    else:
        print("\n✗ WARNING: B[0] sign is NEGATIVE (unexpected)")

    # Save results
    results_file = output_dir / "identified_model.npz"
    np.savez(
        results_file,
        A=A,
        B=B,
        residual=residual,
        controllability_rank=rank,
        equilibrium_cp_y=equilibrium_cp_y,
        perturbations=perturbations,
    )
    print(f"\nResults saved to: {results_file}")

    # Validation: predict on held-out short rollouts
    print("\n" + "="*60)
    print("MODEL VALIDATION")
    print("="*60)

    print("\nTesting prediction on collected data:")
    for i, (states, inputs) in enumerate(zip(states_list, inputs_list)):
        u = inputs[0]  # Constant input

        # Predict using identified model
        x_pred = [states[0]]
        for t in range(len(inputs)):
            x_next_pred = A @ x_pred[-1] + B * u
            x_pred.append(x_next_pred)

        x_pred = np.array(x_pred)

        # Compute error
        error = np.linalg.norm(states - x_pred, axis=1)
        max_error = np.max(error)
        final_error = error[-1]

        print(f"  u={u:+.1f} Nm: max_error={max_error:.4f}, final_error={final_error:.4f}")

    print("\n" + "="*60)
    print("ACCEPTANCE CRITERIA")
    print("="*60)

    acceptance = []

    if B[0] > 0:
        acceptance.append("✓ B[0] has consistent positive sign")
    else:
        acceptance.append("✗ B[0] sign is negative (unexpected)")

    if residual < 0.01:
        acceptance.append(f"✓ Prediction residual is low ({residual:.6f})")
    elif residual < 0.05:
        acceptance.append(f"⚠ Prediction residual is moderate ({residual:.6f})")
    else:
        acceptance.append(f"✗ Prediction residual is high ({residual:.6f})")

    if is_controllable:
        acceptance.append(f"✓ System is fully controllable (rank={rank})")
    else:
        acceptance.append(f"✗ System is not fully controllable (rank={rank}/5)")

    for item in acceptance:
        print(f"  {item}")

    if all('✓' in item for item in acceptance):
        print("\n✓ MODEL ACCEPTED - Ready for Phase 2 (LQR design)")
    else:
        print("\n⚠ MODEL NEEDS REVIEW - Check warnings before proceeding")


if __name__ == "__main__":
    main()
