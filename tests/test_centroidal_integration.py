"""Integration tests for centroidal state estimation pipeline."""

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)


def test_no_nan_rollout_100_steps():
    """Test 100-step rollout produces no NaNs in centroidal state estimation."""
    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mjx_model = mjx.put_model(mj_model)

    data = mjx.make_data(mjx_model)
    data = mjx.forward(mjx_model, data)

    centroidal_config = CentroidalStateEstimatorConfig(
        robot_mass=15.0,
        torso_inertia=jnp.array([0.1, 0.1, 0.05])
    )
    centroidal_estimator = CentroidalStateEstimator(centroidal_config)

    capture_config = CapturePointEstimatorConfig(gravity=9.81, min_height=0.35)
    capture_estimator = CapturePointEstimator(capture_config)

    prev_com_pos = None

    for step in range(100):
        obs = jnp.zeros(42)

        centroidal_state, new_com_pos = centroidal_estimator.estimate(
            obs, data, prev_com_pos
        )
        prev_com_pos = new_com_pos

        centroidal_state = capture_estimator.update(centroidal_state)

        assert not jnp.any(jnp.isnan(centroidal_state.com_pos))
        assert not jnp.any(jnp.isnan(centroidal_state.com_vel))
        assert not jnp.any(jnp.isnan(centroidal_state.capture_point))
        assert not jnp.any(jnp.isnan(centroidal_state.divergence))
        assert not jnp.any(jnp.isnan(centroidal_state.linear_momentum))
        assert not jnp.any(jnp.isnan(centroidal_state.angular_momentum))
        assert not jnp.isnan(centroidal_state.left_wheel_force)
        assert not jnp.isnan(centroidal_state.right_wheel_force)

        data = mjx.step(mjx_model, data)
