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


def test_integrated_wbc_reports_contact_aware_diagnostics():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=sum(model.body_mass),
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )
    state, _ = estimator.estimate(jnp.zeros(42), data, None)
    state = CapturePointEstimator(CapturePointEstimatorConfig()).update(state)

    controller = IntegratedWBC(
        model,
        robot_mass=sum(model.body_mass),
        gravity=abs(model.opt.gravity[2]),
    )
    tau, diagnostics = controller.compute_wbc_torque_with_diagnostics(
        data, jnp.zeros(42), state, float(state.com_pos[2])
    )

    assert tau.shape == (10,)
    assert diagnostics["left_contact_active"]
    assert diagnostics["right_contact_active"]
    assert diagnostics["force_distribution_feasible"]
    assert diagnostics["desired_wrench_Fz"] > 0.0
    assert diagnostics["total_contact_force_z"] > 0.0
    assert diagnostics["distributed_left_fz"] > 0.0
    assert diagnostics["distributed_right_fz"] > 0.0


def test_wbc_torque_leg_joints_are_zero():
    """WBC must not command hip_pitch or knee joints.

    These joints are posture-controlled separately.  If WBC drives them,
    the posture controller and WBC fight each other and the robot collapses.
    """
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=sum(model.body_mass),
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )
    state, _ = estimator.estimate(jnp.zeros(42), data, None)
    state = CapturePointEstimator(CapturePointEstimatorConfig()).update(state)

    controller = IntegratedWBC(
        model,
        robot_mass=sum(model.body_mass),
        gravity=abs(model.opt.gravity[2]),
    )
    tau, _ = controller.compute_wbc_torque_with_diagnostics(
        data, jnp.zeros(42), state, float(state.com_pos[2])
    )

    # Joint indices for hip_pitch and knee (left and right)
    leg_joint_indices = [2, 3, 7, 8]  # l_hip_pitch, l_knee, r_hip_pitch, r_knee

    for idx in leg_joint_indices:
        assert abs(float(tau[idx])) < 1e-6, (
            f"WBC torque on joint {idx} should be zero but is {tau[idx]:.6f}"
        )
