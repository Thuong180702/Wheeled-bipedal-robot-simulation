"""Tests for controlled Step E hip-yaw posture authority fix."""

import jax.numpy as jnp
import numpy as np

from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer
from wheeled_biped.controllers.balance_core_types import ACTION_DIM
from wheeled_biped.controllers.shape_posture_controller import (
    BALANCE_CORE_HIP_YAW_AUTHORITY,
    ShapePostureController,
)
from scripts.evaluate_step_e_hip_yaw_authority import HIP_YAW_AUTHORITY_CANDIDATES


def _hip_yaw_error_state(error_rad: float = 0.1):
    q_ref = jnp.zeros(ACTION_DIM)
    joint_pos = jnp.zeros(ACTION_DIM).at[1].set(-error_rad).at[6].set(-error_rad)
    joint_vel = jnp.zeros(ACTION_DIM)
    return q_ref, joint_pos, joint_vel


def test_shape_posture_hip_yaw_torque_sign_remains_correct():
    controller = ShapePostureController(kp_hip_yaw=10.0, kd_hip_yaw=2.0)
    q_ref, joint_pos, joint_vel = _hip_yaw_error_state(0.1)

    tau, _ = controller.compute(q_ref, joint_pos, joint_vel)

    assert float(tau[1]) > 0.0
    assert float(tau[6]) > 0.0


def test_selected_authority_profile_increases_hip_yaw_torque_magnitude():
    baseline = ShapePostureController(kp_hip_yaw=5.0, kd_hip_yaw=1.0)
    selected = ShapePostureController(
        kp_hip_yaw=BALANCE_CORE_HIP_YAW_AUTHORITY.kp_hip_yaw,
        kd_hip_yaw=BALANCE_CORE_HIP_YAW_AUTHORITY.kd_hip_yaw,
    )
    q_ref, joint_pos, joint_vel = _hip_yaw_error_state(0.1)

    tau_baseline, _ = baseline.compute(q_ref, joint_pos, joint_vel)
    tau_selected, _ = selected.compute(q_ref, joint_pos, joint_vel)

    assert abs(float(tau_selected[1])) > abs(float(tau_baseline[1]))
    assert abs(float(tau_selected[6])) > abs(float(tau_baseline[6]))


def test_shape_posture_still_outputs_zero_on_hip_roll_and_wheels():
    controller = ShapePostureController(kp_hip_yaw=20.0, kd_hip_yaw=4.0)
    q_ref = jnp.ones(ACTION_DIM) * 0.2
    joint_pos = jnp.zeros(ACTION_DIM)
    joint_vel = jnp.zeros(ACTION_DIM)

    tau, _ = controller.compute(q_ref, joint_pos, joint_vel)

    np.testing.assert_allclose(np.asarray(tau)[[0, 4, 5, 9]], np.zeros(4), atol=1e-9)


def test_balance_core_ownership_remains_unchanged_with_higher_hip_yaw_authority():
    controller = ShapePostureController(kp_hip_yaw=20.0, kd_hip_yaw=4.0)
    q_ref, joint_pos, joint_vel = _hip_yaw_error_state(0.1)
    tau_shape, _ = controller.compute(q_ref, joint_pos, joint_vel)
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.array([30.0] * ACTION_DIM),
        max_torque_rate=jnp.array([1000.0] * ACTION_DIM),
        control_dt=0.01,
    )

    result = composer.compose(
        tau_shape_posture=tau_shape,
        tau_support_feedforward=jnp.zeros(ACTION_DIM),
        tau_sagittal_wheel_balance=jnp.zeros(ACTION_DIM),
        tau_lateral_roll_balance=jnp.zeros(ACTION_DIM),
        tau_prev=jnp.zeros(ACTION_DIM),
    )

    assert result.ownership_violation_count == 0
    assert result.active_torque_owner_per_joint[1] == "tau_shape_posture"
    assert result.active_torque_owner_per_joint[6] == "tau_shape_posture"


def test_balance_core_authority_profile_matches_selected_candidate():
    assert BALANCE_CORE_HIP_YAW_AUTHORITY.kp_hip_yaw == 15.0
    assert BALANCE_CORE_HIP_YAW_AUTHORITY.kd_hip_yaw == 3.0


