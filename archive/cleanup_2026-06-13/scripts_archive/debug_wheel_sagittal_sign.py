"""Wheel sagittal sign truth diagnostic.

Determines the correct sign convention for wheel torques in sagittal (pitch) control.
Tests isolated wheel torques to establish which sign moves the robot forward/backward
and which sign reduces positive/negative pitch errors.
"""

import jax.numpy as jnp
import mujoco
import numpy as np
from pathlib import Path

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.static_posture_holding_controller import StaticPostureHoldingController
from wheeled_biped.controllers.static_feedforward_controller import StaticFeedforwardController
from wheeled_biped.controllers.orientation_utils import compute_orientation_from_gravity


STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD = np.array([
    0.0, 0.0, 0.0, -15.5, 0.0,
    0.0, 0.0, 0.0, -15.8, 0.0,
], dtype=np.float64)


def calibrate_root_z(model, data, target_dist=-5e-4, max_iters=5):
    """Calibrate root z position for wheel-floor contact."""
    for _ in range(max_iters):
        mujoco.mj_forward(model, data)

        floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
        r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

        min_dist = None
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
            involves_wheel = g1 in {l_wheel_geom_id, r_wheel_geom_id} or g2 in {l_wheel_geom_id, r_wheel_geom_id}

            if involves_floor and involves_wheel:
                if min_dist is None or c.dist < min_dist:
                    min_dist = c.dist

        if min_dist is None:
            break

        error = min_dist - target_dist
        if abs(error) < 1e-5:
            break

        data.qpos[2] -= error * 0.8

    mujoco.mj_forward(model, data)


def test_wheel_sign_convention():
    """Test wheel torque sign convention for sagittal control."""
    print("=" * 80)
    print("WHEEL SAGITTAL SIGN TRUTH DIAGNOSTIC")
    print("=" * 80)

    # Load model
    xml_path = Path("assets/robot/wheeled_biped_real.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Initialize and calibrate
    mujoco.mj_resetDataKeyframe(model, data, 0)
    calibrate_root_z(model, data)

    robot_mass = float(np.sum(model.body_mass))
    gravity = float(abs(model.opt.gravity[2]))

    # Create controllers
    state_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=robot_mass,
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )

    capture_estimator = CapturePointEstimator(
        CapturePointEstimatorConfig(gravity=gravity, min_height=0.35)
    )

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
    mujoco.mj_forward(model, data)
    equilibrium_joint_pos = jnp.array(data.qpos[7:17])
    static_posture_controller.set_equilibrium_reference(equilibrium_joint_pos)

    centroidal_state_eq, _ = state_estimator.estimate(jnp.zeros(42), data, None)
    centroidal_state_eq = capture_estimator.update(centroidal_state_eq)

    base_body_id = 1
    R_eq = np.array(data.xmat[base_body_id]).reshape(3, 3)
    gravity_world = np.array([0.0, 0.0, -gravity])
    gravity_body_eq = R_eq.T @ gravity_world
    pitch_x_eq, roll_y_eq = compute_orientation_from_gravity(jnp.array(gravity_body_eq))

    print(f"\n[Equilibrium State]")
    print(f"  CoM: [{float(centroidal_state_eq.com_pos[0]):.6f}, {float(centroidal_state_eq.com_pos[1]):.6f}, {float(centroidal_state_eq.com_pos[2]):.6f}] m")
    print(f"  Pitch: {float(pitch_x_eq)*57.3:.2f} deg, Roll: {float(roll_y_eq)*57.3:.2f} deg")
    print(f"  Robot front direction: -Y axis")

    # Test cases: [tau_left_wheel, tau_right_wheel]
    test_cases = [
        ([+1.0, +1.0], "positive_small"),
        ([-1.0, -1.0], "negative_small"),
        ([+3.0, +3.0], "positive_large"),
        ([-3.0, -3.0], "negative_large"),
    ]

    control_dt = 0.01
    n_substeps = int(model.opt.timestep / control_dt) if model.opt.timestep < control_dt else 1

    for tau_wheel_cmd, test_name in test_cases:
        print(f"\n{'=' * 80}")
        print(f"[Test: {test_name}] tau_wheel = {tau_wheel_cmd}")
        print(f"{'=' * 80}")

        # Reset to equilibrium
        mujoco.mj_resetDataKeyframe(model, data, 0)
        calibrate_root_z(model, data)

        # Run for 1, 5, 20 steps
        for n_steps in [1, 5, 20]:
            # Reset to equilibrium
            mujoco.mj_resetDataKeyframe(model, data, 0)
            calibrate_root_z(model, data)

            results = []

            for step in range(n_steps):
                # Get current state
                joint_pos = jnp.array(data.qpos[7:17])
                joint_vel = jnp.array(data.qvel[6:16])

                # Compute support torques
                tau_static_posture, _ = static_posture_controller.compute_posture_holding_torque(
                    joint_pos, joint_vel
                )
                tau_static_feedforward = jnp.array(static_feedforward_controller.compute_feedforward())

                # Add test wheel torque
                tau_wheel_test = jnp.zeros(10)
                tau_wheel_test = tau_wheel_test.at[4].set(tau_wheel_cmd[0])  # l_wheel
                tau_wheel_test = tau_wheel_test.at[9].set(tau_wheel_cmd[1])  # r_wheel

                tau_total = tau_static_feedforward + tau_static_posture + tau_wheel_test

                # Apply torques
                data.ctrl[:] = np.array(tau_total)

                # Step simulation
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)

                # Log state
                centroidal_state, _ = state_estimator.estimate(jnp.zeros(42), data, None)
                centroidal_state = capture_estimator.update(centroidal_state)

                R = np.array(data.xmat[base_body_id]).reshape(3, 3)
                gravity_body = R.T @ gravity_world
                pitch_x, roll_y = compute_orientation_from_gravity(jnp.array(gravity_body))

                wheel_vel_left = float(data.qvel[10])  # l_wheel joint velocity
                wheel_vel_right = float(data.qvel[15])  # r_wheel joint velocity

                results.append({
                    'step': step,
                    'wheel_vel_left': wheel_vel_left,
                    'wheel_vel_right': wheel_vel_right,
                    'com_y': float(centroidal_state.com_pos[1]),
                    'com_vy': float(centroidal_state.com_vel[1]),
                    'cp_y': float(centroidal_state.capture_point[1]),
                    'pitch_x': float(pitch_x),
                    'pitch_rate_x': float(centroidal_state.body_pitch_rate_x),
                })

            # Report results
            final = results[-1]
            initial_com_y = float(centroidal_state_eq.com_pos[1])
            delta_com_y = final['com_y'] - initial_com_y

            print(f"\n  After {n_steps} steps:")
            print(f"    Wheel velocities: L={final['wheel_vel_left']:+.3f}, R={final['wheel_vel_right']:+.3f} rad/s")
            print(f"    CoM Y: {final['com_y']:+.6f} m (delta={delta_com_y:+.6f} m)")
            print(f"    CoM Vy: {final['com_vy']:+.6f} m/s")
            print(f"    CP Y: {final['cp_y']:+.6f} m")
            print(f"    Pitch X: {final['pitch_x']*57.3:+.2f} deg")
            print(f"    Pitch rate X: {final['pitch_rate_x']*57.3:+.2f} deg/s")

            # Interpret direction
            if delta_com_y < -0.001:
                direction = "FORWARD (-Y)"
            elif delta_com_y > 0.001:
                direction = "BACKWARD (+Y)"
            else:
                direction = "STATIONARY"

            print(f"    -> Robot moved: {direction}")

    print(f"\n{'=' * 80}")
    print(f"[Conclusion]")
    print(f"{'=' * 80}")
    print(f"Review the results above to determine:")
    print(f"  1. Which wheel torque sign moves the robot forward (-Y direction)")
    print(f"  2. Which wheel torque sign moves the robot backward (+Y direction)")
    print(f"  3. For positive pitch_x (falling forward), which torque sign opposes it")
    print(f"  4. For negative pitch_x (falling backward), which torque sign opposes it")
    print(f"\nUse these findings to set the correct sign in Stage2BSagittalWheelController.")


if __name__ == "__main__":
    test_wheel_sign_convention()
