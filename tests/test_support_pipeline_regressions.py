import jax.numpy as jnp
import mujoco

from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.integrated_wbc import IntegratedWBC

MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


def test_per_actuator_authority_clipping_respects_xml_ctrlrange():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    controller = IntegratedWBC(
        model,
        wbc_authority_budget=1.0,
        use_per_actuator_authority=True,
    )

    tau = jnp.array([200.0] * 10)
    clipped = controller.clip_to_authority_budget(tau)

    actuator_limits = jnp.array(model.actuator_ctrlrange[:, 1])
    assert jnp.allclose(clipped, actuator_limits)
    assert clipped[2] == 150.0
    assert clipped[3] == 150.0
    assert clipped[7] == 150.0
    assert clipped[8] == 150.0
    assert clipped[0] == 30.0
    assert clipped[4] == 30.0


def test_scalar_authority_clipping_scales_unrelated_joints_together():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    controller = IntegratedWBC(
        model,
        wbc_authority_budget=1.0,
        max_actuator_torque=60.0,
        use_per_actuator_authority=False,
    )

    tau = jnp.array([30.0, 30.0, 120.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0])
    clipped = controller.clip_to_authority_budget(tau)

    expected = tau * 0.5
    assert jnp.allclose(clipped, expected)
    assert clipped[2] == 60.0
    assert clipped[0] == 15.0
    assert clipped[3] == 15.0


def test_step0_preload_eliminates_rate_limit_ramp_from_zero():
    control_dt = 0.01
    max_torque_rate = 400.0
    tau_total_clipped = jnp.array([0.0, 0.0, -20.0, 10.0, 0.0, 0.0, 0.0, -20.0, 10.0, 0.0])

    tau_prev_zero = jnp.zeros(10)
    rate_from_zero = (tau_total_clipped - tau_prev_zero) / control_dt
    rate_from_zero = jnp.clip(rate_from_zero, -max_torque_rate, max_torque_rate)
    tau_smooth_from_zero = tau_prev_zero + rate_from_zero * control_dt

    tau_prev_preloaded = tau_total_clipped
    rate_from_preload = (tau_total_clipped - tau_prev_preloaded) / control_dt
    rate_from_preload = jnp.clip(rate_from_preload, -max_torque_rate, max_torque_rate)
    tau_smooth_from_preload = tau_prev_preloaded + rate_from_preload * control_dt

    assert tau_smooth_from_zero[2] == -4.0
    assert tau_smooth_from_zero[3] == 4.0
    assert tau_smooth_from_zero[7] == -4.0
    assert tau_smooth_from_zero[8] == 4.0

    assert jnp.allclose(tau_smooth_from_preload, tau_total_clipped)


def test_contact_force_is_zero_pre_step_and_nonzero_post_step_for_logging_order():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=float(sum(model.body_mass)),
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )
    capture_estimator = CapturePointEstimator(CapturePointEstimatorConfig())

    state_pre, com_pre = estimator.estimate(jnp.zeros(42), data, None)
    state_pre = capture_estimator.update(state_pre)

    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)

    state_post, _ = estimator.estimate(jnp.zeros(42), data, com_pre)
    state_post = capture_estimator.update(state_post)

    assert state_pre.left_wheel_contact and state_pre.right_wheel_contact
    assert abs(float(state_pre.total_contact_force_z)) < 1e-6
    assert abs(float(state_post.total_contact_force_z)) > 1e-3
