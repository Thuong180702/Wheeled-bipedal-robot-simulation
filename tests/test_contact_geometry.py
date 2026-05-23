"""Test contact geometry fixes: actual contact points vs body origins."""

import numpy as np
import jax.numpy as jnp
import mujoco

from wheeled_biped.controllers.contact_jacobian import ContactJacobian
from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)

MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


def test_contact_jacobian_uses_contact_point_not_body_origin():
    """Verify ContactJacobian uses actual contact point, not wheel body origin."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    contact_jac = ContactJacobian(model)
    l_contact_point, r_contact_point = contact_jac.get_wheel_contact_points(data)

    # Get wheel body origins
    l_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
    l_body_origin = np.array(data.xpos[l_wheel_body_id])
    r_body_origin = np.array(data.xpos[r_wheel_body_id])

    # Get wheel collision geom centers
    l_geom_center = np.array(data.geom_xpos[contact_jac.l_wheel_geom_id])
    r_geom_center = np.array(data.geom_xpos[contact_jac.r_wheel_geom_id])

    # Contact point should be closer to geom center than body origin
    dist_l_to_geom = np.linalg.norm(np.array(l_contact_point) - l_geom_center)
    dist_l_to_body = np.linalg.norm(np.array(l_contact_point) - l_body_origin)

    dist_r_to_geom = np.linalg.norm(np.array(r_contact_point) - r_geom_center)
    dist_r_to_body = np.linalg.norm(np.array(r_contact_point) - r_body_origin)

    assert dist_l_to_geom < dist_l_to_body, (
        f"Left contact point should be closer to geom ({dist_l_to_geom:.4f}) "
        f"than body origin ({dist_l_to_body:.4f})"
    )
    assert dist_r_to_geom < dist_r_to_body, (
        f"Right contact point should be closer to geom ({dist_r_to_geom:.4f}) "
        f"than body origin ({dist_r_to_body:.4f})"
    )

    # Contact point z should be near floor (z ≈ 0)
    assert abs(float(l_contact_point[2])) < 0.01, f"Left contact z should be near floor: {l_contact_point[2]}"
    assert abs(float(r_contact_point[2])) < 0.01, f"Right contact z should be near floor: {r_contact_point[2]}"


def test_wrench_matrix_uses_same_contact_point_as_jacobian():
    """Verify IntegratedWBC wrench matrix uses same contact point as ContactJacobian."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    robot_mass = float(np.sum(model.body_mass))
    gravity = float(abs(model.opt.gravity[2]))

    wbc = IntegratedWBC(
        model,
        robot_mass=robot_mass,
        gravity=gravity,
        k_height=50.0,
        k_height_damping=0.0,
    )

    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=robot_mass,
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )
    capture_estimator = CapturePointEstimator(
        CapturePointEstimatorConfig(gravity=gravity, min_height=0.35)
    )

    state, _ = estimator.estimate(jnp.zeros(42), data, None)
    state = capture_estimator.update(state)

    # Get contact points from ContactJacobian
    l_contact_jac, r_contact_jac = wbc.contact_jacobian.get_wheel_contact_points(data)

    # Get wheel positions from IntegratedWBC (used for wrench matrix)
    wheel_pos_left, wheel_pos_right = wbc._compute_wheel_positions_relative_to_com(
        data, state.com_pos
    )

    # Reconstruct absolute positions
    l_contact_wbc = np.array(wheel_pos_left) + np.array(state.com_pos)
    r_contact_wbc = np.array(wheel_pos_right) + np.array(state.com_pos)

    # Should match within numerical precision
    assert np.allclose(l_contact_jac, l_contact_wbc, atol=1e-6), (
        f"Left contact point mismatch: Jacobian {l_contact_jac} vs WBC {l_contact_wbc}"
    )
    assert np.allclose(r_contact_jac, r_contact_wbc, atol=1e-6), (
        f"Right contact point mismatch: Jacobian {r_contact_jac} vs WBC {r_contact_wbc}"
    )


def test_force_distributor_uses_actual_wheel_positions():
    """Verify SimpleForceDistributor uses actual wheel x positions, not hard-coded 0.135."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    robot_mass = float(np.sum(model.body_mass))
    gravity = float(abs(model.opt.gravity[2]))

    wbc = IntegratedWBC(
        model,
        robot_mass=robot_mass,
        gravity=gravity,
        k_height=50.0,
        k_height_damping=0.0,
    )

    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=robot_mass,
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )
    capture_estimator = CapturePointEstimator(
        CapturePointEstimatorConfig(gravity=gravity, min_height=0.35)
    )

    state, _ = estimator.estimate(jnp.zeros(42), data, None)
    state = capture_estimator.update(state)

    wheel_pos_left, wheel_pos_right = wbc._compute_wheel_positions_relative_to_com(
        data, state.com_pos
    )

    # Actual wheel x positions should NOT be exactly ±0.135
    # (wheel collision geom has offset pos="-0.038 0 0")
    x_left = float(wheel_pos_left[0])
    x_right = float(wheel_pos_right[0])

    # Should be asymmetric due to geom offset
    assert abs(x_left) != abs(x_right), (
        f"Wheel x positions should be asymmetric due to geom offset: "
        f"left={x_left:.4f}, right={x_right:.4f}"
    )

    # Neither should be exactly 0.135
    assert abs(abs(x_left) - 0.135) > 0.01, f"Left x should not be hard-coded 0.135: {x_left:.4f}"
    assert abs(abs(x_right) - 0.135) > 0.01, f"Right x should not be hard-coded 0.135: {x_right:.4f}"


def test_height_damping_affects_vertical_force():
    """Verify k_height_damping increases Fz when com_vel[2] < 0 (falling)."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    robot_mass = float(np.sum(model.body_mass))
    gravity = float(abs(model.opt.gravity[2]))

    # Create two WBC instances: one without damping, one with
    wbc_no_damp = IntegratedWBC(
        model,
        robot_mass=robot_mass,
        gravity=gravity,
        k_height=50.0,
        k_height_damping=0.0,
    )

    wbc_with_damp = IntegratedWBC(
        model,
        robot_mass=robot_mass,
        gravity=gravity,
        k_height=50.0,
        k_height_damping=40.0,
    )

    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=robot_mass,
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )
    capture_estimator = CapturePointEstimator(
        CapturePointEstimatorConfig(gravity=gravity, min_height=0.35)
    )

    state, _ = estimator.estimate(jnp.zeros(42), data, None)
    state = capture_estimator.update(state)

    # Simulate falling: set com_vel[2] < 0
    state = state.replace(com_vel=jnp.array([0.0, 0.0, -0.5]))

    obs = jnp.zeros(42)
    obs = obs.at[36].set(0.40)  # height_cmd
    obs = obs.at[37].set(float(state.com_pos[2]))

    # Compute desired wrench for both
    desired_force_no_damp, _ = wbc_no_damp.wrench_computer.compute_desired_wrench(
        obs, state, 0.40, 0.0
    )
    desired_force_with_damp, _ = wbc_with_damp.wrench_computer.compute_desired_wrench(
        obs, state, 0.40, 0.0
    )

    fz_no_damp = float(desired_force_no_damp[2])
    fz_with_damp = float(desired_force_with_damp[2])

    # With negative com_vel[2], damping should INCREASE Fz
    # Fz = ... + k_height * height_error - k_height_damping * com_vel[2]
    # com_vel[2] = -0.5 → -k_height_damping * (-0.5) = +20 N increase
    expected_increase = 40.0 * 0.5  # k_height_damping * abs(com_vel[2])

    assert fz_with_damp > fz_no_damp, (
        f"Damping should increase Fz when falling: "
        f"no_damp={fz_no_damp:.2f}, with_damp={fz_with_damp:.2f}"
    )

    actual_increase = fz_with_damp - fz_no_damp
    assert abs(actual_increase - expected_increase) < 1.0, (
        f"Fz increase should be ~{expected_increase:.2f} N, got {actual_increase:.2f} N"
    )
