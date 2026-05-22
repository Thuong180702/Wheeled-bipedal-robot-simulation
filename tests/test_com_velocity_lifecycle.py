import jax.numpy as jnp
import mujoco

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)

MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


def _build_estimator(model: mujoco.MjModel) -> CentroidalStateEstimator:
    return CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=float(sum(model.body_mass)),
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )


def test_buggy_lifecycle_can_zero_next_control_com_velocity():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    estimator = _build_estimator(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    # Control estimate at t0.
    state0, com0 = estimator.estimate(jnp.zeros(42), data, None)

    # Advance one control interval with zero control.
    control_dt = 0.01
    n_substeps = int(control_dt / model.opt.timestep)
    for _ in range(n_substeps):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)

    # Logging estimate at t1.
    state_log_1, com1 = estimator.estimate(jnp.zeros(42), data, com0)

    # Buggy next-control estimate: previous set to current post-step com.
    state_buggy, _ = estimator.estimate(jnp.zeros(42), data, com1)

    # Correct next-control estimate should not use com1 as previous in same state sample.
    state_correct, _ = estimator.estimate(jnp.zeros(42), data, com0)

    assert jnp.linalg.norm(state_correct.com_vel) > 1e-6
    assert jnp.linalg.norm(state_buggy.com_vel) < 1e-9
    assert jnp.linalg.norm(state_log_1.com_vel) > 1e-6


def test_simulation_script_keeps_separate_prev_control_com_state():
    with open("scripts/simulate_hierarchical_controller.py", encoding="utf-8") as f:
        src = f.read()

    assert "prev_control_com_pos" in src
    assert "prev_control_com_pos = logged_com_pos" not in src
