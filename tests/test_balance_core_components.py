"""Tests for balance-core controller components."""

import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.balance_core_types import ACTION_DIM, SUPPORT_SHAPE_INDICES
from wheeled_biped.controllers.shape_posture_controller import ShapePostureController


def test_shape_posture_outputs_only_on_support_shape_joints():
    controller = ShapePostureController()
    q_ref = jnp.zeros(ACTION_DIM)
    joint_pos = jnp.array([0.2, -0.1, 0.3, -0.2, 0.5, -0.4, 0.1, -0.25, 0.15, -0.3])
    joint_vel = jnp.zeros(ACTION_DIM)

    tau, diagnostics = controller.compute(q_ref, joint_pos, joint_vel)

    assert tau.shape == (ACTION_DIM,)
    for idx in range(ACTION_DIM):
        if idx in SUPPORT_SHAPE_INDICES.tolist():
            assert tau[idx] != 0.0
        else:
            assert tau[idx] == 0.0

    assert tau[0] == 0.0
    assert tau[5] == 0.0
    assert tau[4] == 0.0
    assert tau[9] == 0.0
    assert "posture_error_norm" in diagnostics
    assert "torque_norm" in diagnostics


def test_shape_posture_softens_with_posture_weight_and_contact_scale():
    controller = ShapePostureController()
    q_ref = jnp.zeros(ACTION_DIM)
    joint_pos = jnp.array([0.0, 0.3, -0.2, 0.1, 0.0, 0.0, -0.15, 0.25, -0.35, 0.0])
    joint_vel = jnp.zeros(ACTION_DIM)

    tau_full, _ = controller.compute(
        q_ref,
        joint_pos,
        joint_vel,
        posture_weight=1.0,
        contact_degraded_scale=1.0,
    )
    tau_soft, _ = controller.compute(
        q_ref,
        joint_pos,
        joint_vel,
        posture_weight=0.5,
        contact_degraded_scale=0.4,
    )

    expected_scale = 0.5 * 0.4
    assert jnp.allclose(tau_soft, tau_full * expected_scale)


def test_shape_posture_zero_error_returns_zero_torque_and_norms():
    controller = ShapePostureController()
    q_ref = jnp.zeros(ACTION_DIM)
    joint_pos = jnp.zeros(ACTION_DIM)
    joint_vel = jnp.zeros(ACTION_DIM)

    tau, diagnostics = controller.compute(q_ref, joint_pos, joint_vel)

    assert jnp.allclose(tau, jnp.zeros(ACTION_DIM))
    assert diagnostics["posture_error_norm"] == 0.0
    assert diagnostics["torque_norm"] == 0.0


def test_support_feedforward_outputs_only_on_support_feedforward_joints():
    from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController
    from wheeled_biped.controllers.balance_core_types import SUPPORT_FEEDFORWARD_INDICES

    # Indices: 0=l_hip_roll, 1=l_hip_yaw, 2=l_hip_pitch, 3=l_knee, 4=l_wheel,
    #          5=r_hip_roll, 6=r_hip_yaw, 7=r_hip_pitch, 8=r_knee, 9=r_wheel
    support_vector = jnp.array([0.0, 0.0, 5.0, -3.0, 0.0, 0.0, 0.0, 4.0, -2.0, 0.0])
    controller = SupportFeedforwardController(support_vector=support_vector, joint_group="hip_pitch_knee")

    tau, diagnostics = controller.compute()

    assert tau.shape == (ACTION_DIM,)
    for idx in range(ACTION_DIM):
        if idx in SUPPORT_FEEDFORWARD_INDICES.tolist():
            pass  # Can be nonzero
        else:
            assert tau[idx] == 0.0

    # Verify hip_roll, hip_yaw, wheels are zero
    assert tau[0] == 0.0  # l_hip_roll
    assert tau[1] == 0.0  # l_hip_yaw
    assert tau[4] == 0.0  # l_wheel
    assert tau[5] == 0.0  # r_hip_roll
    assert tau[6] == 0.0  # r_hip_yaw
    assert tau[9] == 0.0  # r_wheel

    # Verify hip_pitch and knee can be nonzero
    assert tau[2] != 0.0  # l_hip_pitch
    assert tau[3] != 0.0  # l_knee
    assert tau[7] != 0.0  # r_hip_pitch
    assert tau[8] != 0.0  # r_knee

    assert "support_feedforward_joint_group" in diagnostics
    assert diagnostics["support_feedforward_joint_group"] == "hip_pitch_knee"
    assert "tau_support_feedforward_norm" in diagnostics


def test_support_feedforward_knee_only_outputs_on_knee_joints():
    from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController

    # Indices: 0=l_hip_roll, 1=l_hip_yaw, 2=l_hip_pitch, 3=l_knee, 4=l_wheel,
    #          5=r_hip_roll, 6=r_hip_yaw, 7=r_hip_pitch, 8=r_knee, 9=r_wheel
    support_vector = jnp.array([0.0, 0.0, 5.0, -3.0, 0.0, 0.0, 0.0, 4.0, -2.0, 0.0])
    controller = SupportFeedforwardController(support_vector=support_vector, joint_group="knee")

    tau, diagnostics = controller.compute()

    # Only knee joints [3, 8] should be nonzero
    assert tau[3] != 0.0  # l_knee
    assert tau[8] != 0.0  # r_knee

    # All others should be zero
    assert tau[0] == 0.0
    assert tau[1] == 0.0
    assert tau[2] == 0.0  # l_hip_pitch should be zero for knee-only
    assert tau[4] == 0.0
    assert tau[5] == 0.0
    assert tau[6] == 0.0
    assert tau[7] == 0.0  # r_hip_pitch should be zero for knee-only
    assert tau[9] == 0.0

    assert diagnostics["support_feedforward_joint_group"] == "knee"


def test_support_feedforward_hip_pitch_only_outputs_on_hip_pitch_joints():
    from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController

    # Indices: 0=l_hip_roll, 1=l_hip_yaw, 2=l_hip_pitch, 3=l_knee, 4=l_wheel,
    #          5=r_hip_roll, 6=r_hip_yaw, 7=r_hip_pitch, 8=r_knee, 9=r_wheel
    support_vector = jnp.array([0.0, 0.0, 5.0, -3.0, 0.0, 0.0, 0.0, 4.0, -2.0, 0.0])
    controller = SupportFeedforwardController(support_vector=support_vector, joint_group="hip_pitch")

    tau, diagnostics = controller.compute()

    # Only hip_pitch joints [2, 7] should be nonzero
    assert tau[2] != 0.0  # l_hip_pitch
    assert tau[7] != 0.0  # r_hip_pitch

    # All others should be zero
    assert tau[0] == 0.0
    assert tau[1] == 0.0
    assert tau[3] == 0.0  # l_knee should be zero for hip_pitch-only
    assert tau[4] == 0.0
    assert tau[5] == 0.0
    assert tau[6] == 0.0
    assert tau[8] == 0.0  # r_knee should be zero for hip_pitch-only
    assert tau[9] == 0.0

    assert diagnostics["support_feedforward_joint_group"] == "hip_pitch"


def test_support_feedforward_scales_with_scale_parameter():
    from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController

    # Indices: 0=l_hip_roll, 1=l_hip_yaw, 2=l_hip_pitch, 3=l_knee, 4=l_wheel,
    #          5=r_hip_roll, 6=r_hip_yaw, 7=r_hip_pitch, 8=r_knee, 9=r_wheel
    support_vector = jnp.array([0.0, 0.0, 5.0, -3.0, 0.0, 0.0, 0.0, 4.0, -2.0, 0.0])

    controller_full = SupportFeedforwardController(support_vector=support_vector, joint_group="hip_pitch_knee", scale=1.0)
    tau_full, _ = controller_full.compute()

    controller_half = SupportFeedforwardController(support_vector=support_vector, joint_group="hip_pitch_knee", scale=0.5)
    tau_half, _ = controller_half.compute()

    assert jnp.allclose(tau_half, tau_full * 0.5)


def test_support_feedforward_zero_vector_returns_zero_torque():
    from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController

    support_vector = jnp.zeros(ACTION_DIM)
    controller = SupportFeedforwardController(support_vector=support_vector, joint_group="hip_pitch_knee")

    tau, diagnostics = controller.compute()

    assert jnp.allclose(tau, jnp.zeros(ACTION_DIM))
    assert diagnostics["tau_support_feedforward_norm"] == 0.0


def test_support_feedforward_jit_compatibility():
    """Verify support feedforward controller works under JIT compilation."""
    import jax
    from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController

    support_vector = jnp.array([0.0, 0.0, 5.0, -3.0, 0.0, 0.0, 0.0, 4.0, -2.0, 0.0])
    controller = SupportFeedforwardController(support_vector=support_vector, joint_group="hip_pitch_knee", scale=1.5)

    # JIT-compile the compute method
    @jax.jit
    def jit_compute():
        tau, _ = controller.compute()
        return tau

    # Call JIT-compiled version
    tau_jit = jit_compute()

    # Call non-JIT version for comparison
    tau_eager, _ = controller.compute()

    # Results should match
    assert jnp.allclose(tau_jit, tau_eager)
    assert tau_jit.shape == (ACTION_DIM,)

    # Verify expected values
    assert tau_jit[2] == 1.5 * 5.0  # l_hip_pitch
    assert tau_jit[3] == 1.5 * (-3.0)  # l_knee
    assert tau_jit[7] == 1.5 * 4.0  # r_hip_pitch
    assert tau_jit[8] == 1.5 * (-2.0)  # r_knee


def test_sagittal_wheel_balance_ignores_position_containment_arguments():
    """Position-containment compatibility args must not change balance-core wheel output."""
    from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController

    controller = SagittalWheelBalanceController()

    base_tau, base_diag = controller.compute(
        pitch_x_rad=0.05,
        pitch_rate_x_rad_s=0.01,
        cp_error_y_m=0.02,
        com_vy_m_s=0.03,
        wheel_vel_left_rad_s=0.4,
        wheel_vel_right_rad_s=-0.1,
        outer_position_bias=0.0,
    )
    compat_tau, compat_diag = controller.compute(
        pitch_x_rad=0.05,
        pitch_rate_x_rad_s=0.01,
        cp_error_y_m=0.02,
        com_vy_m_s=0.03,
        wheel_vel_left_rad_s=0.4,
        wheel_vel_right_rad_s=-0.1,
        outer_position_bias=7.5,
        position_y_m=1.2,
        roll_y_rad=0.4,
    )

    assert jnp.allclose(compat_tau, base_tau)
    assert compat_diag == base_diag
    assert "wheel_vel_mean_rad_s" in compat_diag
    assert "term_wheel_velocity_damping" in compat_diag
    assert "wheel_torque_sign" in compat_diag
    assert "sign_convention" in compat_diag


def test_sagittal_wheel_balance_sign_convention():
    """Verify wheel_torque_sign parameter controls sign convention."""
    from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController

    pitch_x_rad = 0.1
    pitch_rate_x_rad_s = 0.05
    cp_error_y_m = 0.02
    com_vy_m_s = 0.01
    wheel_vel_left_rad_s = 0.0
    wheel_vel_right_rad_s = 0.0
    outer_position_bias = 0.0

    controller_pos = SagittalWheelBalanceController(wheel_torque_sign=1.0)
    tau_pos, diag_pos = controller_pos.compute(
        pitch_x_rad,
        pitch_rate_x_rad_s,
        cp_error_y_m,
        com_vy_m_s,
        wheel_vel_left_rad_s,
        wheel_vel_right_rad_s,
        outer_position_bias,
    )

    controller_neg = SagittalWheelBalanceController(wheel_torque_sign=-1.0)
    tau_neg, diag_neg = controller_neg.compute(
        pitch_x_rad,
        pitch_rate_x_rad_s,
        cp_error_y_m,
        com_vy_m_s,
        wheel_vel_left_rad_s,
        wheel_vel_right_rad_s,
        outer_position_bias,
    )

    # Opposite signs should produce opposite torques
    assert jnp.allclose(tau_neg, -tau_pos)
    assert diag_pos["wheel_torque_sign"] == 1.0
    assert diag_neg["wheel_torque_sign"] == -1.0


def test_sagittal_wheel_balance_velocity_damping_opposes_motion():
    """Verify positive wheel velocity damping opposes wheel motion."""
    from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController

    controller = SagittalWheelBalanceController(wheel_torque_sign=1.0)

    # Zero all balance terms, only test damping
    pitch_x_rad = 0.0
    pitch_rate_x_rad_s = 0.0
    cp_error_y_m = 0.0
    com_vy_m_s = 0.0
    outer_position_bias = 0.0

    # Left wheel spinning forward (positive velocity)
    wheel_vel_left_rad_s = 2.0
    wheel_vel_right_rad_s = 0.0

    tau, diagnostics = controller.compute(
        pitch_x_rad,
        pitch_rate_x_rad_s,
        cp_error_y_m,
        com_vy_m_s,
        wheel_vel_left_rad_s,
        wheel_vel_right_rad_s,
        outer_position_bias,
    )

    # Damping should oppose motion: positive velocity -> negative torque
    assert tau[4] < 0.0  # l_wheel torque should be negative
    assert tau[9] == 0.0  # r_wheel should be zero (no velocity)

    # Right wheel spinning backward (negative velocity)
    wheel_vel_left_rad_s = 0.0
    wheel_vel_right_rad_s = -2.0

    tau2, diagnostics2 = controller.compute(
        pitch_x_rad,
        pitch_rate_x_rad_s,
        cp_error_y_m,
        com_vy_m_s,
        wheel_vel_left_rad_s,
        wheel_vel_right_rad_s,
        outer_position_bias,
    )

    # Damping should oppose motion: negative velocity -> positive torque
    assert tau2[4] == 0.0  # l_wheel should be zero (no velocity)
    assert tau2[9] > 0.0  # r_wheel torque should be positive


def test_sagittal_wheel_balance_positive_pitch_commands_backward_wheel_torque():
    """Positive forward pitch should command positive wheel torque to drive the base backward."""
    from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController

    controller = SagittalWheelBalanceController(wheel_torque_sign=1.0)

    tau, diagnostics = controller.compute(
        pitch_x_rad=0.1,
        pitch_rate_x_rad_s=0.05,
        cp_error_y_m=0.02,
        com_vy_m_s=0.01,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        outer_position_bias=0.0,
    )

    assert tau[4] > 0.0
    assert tau[9] > 0.0


def test_sagittal_wheel_balance_negative_cp_error_commands_backward_wheel_torque():
    """Negative capture-point error during a forward fall should drive the base backward."""
    from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController

    controller = SagittalWheelBalanceController(wheel_torque_sign=1.0)

    tau, diagnostics = controller.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        cp_error_y_m=-0.1,
        com_vy_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        outer_position_bias=0.0,
    )

    assert tau[4] > 0.0
    assert tau[9] > 0.0


def test_sagittal_wheel_balance_negative_forward_velocity_commands_backward_wheel_torque():
    """Negative forward CoM velocity during a forward fall should drive the base backward."""
    from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController

    controller = SagittalWheelBalanceController(wheel_torque_sign=1.0)

    tau, diagnostics = controller.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        cp_error_y_m=0.0,
        com_vy_m_s=-0.1,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        outer_position_bias=0.0,
    )

    assert tau[4] > 0.0
    assert tau[9] > 0.0



def test_lateral_roll_balance_outputs_only_on_hip_roll():
    """Verify lateral roll balance controller outputs only on hip roll indices [0, 5]."""
    from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController

    controller = LateralRollBalanceController()
    roll_y_rad = 0.1
    roll_rate_y_rad_s = 0.05

    tau, diagnostics = controller.compute(roll_y_rad, roll_rate_y_rad_s)

    assert tau.shape == (ACTION_DIM,)
    # Only hip roll indices [0, 5] should be nonzero
    for idx in range(ACTION_DIM):
        if idx in [0, 5]:
            pass  # Can be nonzero
        else:
            assert tau[idx] == 0.0

    # Verify all non-hip-roll joints are zero
    assert tau[1] == 0.0  # l_hip_yaw
    assert tau[2] == 0.0  # l_hip_pitch
    assert tau[3] == 0.0  # l_knee
    assert tau[4] == 0.0  # l_wheel
    assert tau[6] == 0.0  # r_hip_yaw
    assert tau[7] == 0.0  # r_hip_pitch
    assert tau[8] == 0.0  # r_knee
    assert tau[9] == 0.0  # r_wheel

    assert "m_roll_cmd" in diagnostics
    assert "m_roll_clipped" in diagnostics
    assert "hip_roll_torque_sign" in diagnostics
    assert "sign_convention" in diagnostics


def test_lateral_roll_balance_sign_convention():
    """Verify hip_roll_torque_sign parameter controls sign convention.

    Positive roll_y (body tilted right) should produce restoring torques:
    - With sign=1.0: tau_left > 0 (push left side down), tau_right < 0 (pull right side up)
    - With sign=-1.0: opposite torques
    """
    from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController

    roll_y_rad = 0.1  # Positive roll (body tilted right)
    roll_rate_y_rad_s = 0.0

    controller_pos = LateralRollBalanceController(hip_roll_torque_sign=1.0)
    tau_pos, diag_pos = controller_pos.compute(roll_y_rad, roll_rate_y_rad_s)

    controller_neg = LateralRollBalanceController(hip_roll_torque_sign=-1.0)
    tau_neg, diag_neg = controller_neg.compute(roll_y_rad, roll_rate_y_rad_s)

    # Opposite signs should produce opposite torques
    assert jnp.allclose(tau_neg, -tau_pos)
    assert diag_pos["hip_roll_torque_sign"] == 1.0
    assert diag_neg["hip_roll_torque_sign"] == -1.0

    # With sign=1.0, positive roll should produce: tau_left > 0, tau_right < 0
    assert tau_pos[0] > 0.0  # l_hip_roll
    assert tau_pos[5] < 0.0  # r_hip_roll


def test_lateral_roll_balance_stance_regularization_restores_nominal_hip_roll():
    from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController

    controller = LateralRollBalanceController(
        kp_stance=5.0,
        kd_stance=1.0,
        max_stance_torque=5.0,
        stance_weight=0.4,
    )

    tau, diagnostics = controller.compute(
        roll_y_rad=0.0,
        roll_rate_y_rad_s=0.0,
        hip_roll_pos=(-0.4, -0.45),
        hip_roll_vel=(0.0, 0.0),
        hip_roll_ref=(0.0, 0.0),
    )

    assert tau[0] > 0.0
    assert tau[5] > 0.0
    assert diagnostics["stance_error_left"] > 0.0
    assert diagnostics["stance_error_right"] > 0.0
    assert diagnostics["stance_torque_left"] > 0.0
    assert diagnostics["stance_torque_right"] > 0.0


def test_lateral_roll_balance_stance_regularization_zero_at_reference():
    from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController

    controller = LateralRollBalanceController()

    tau, diagnostics = controller.compute(
        roll_y_rad=0.0,
        roll_rate_y_rad_s=0.0,
        hip_roll_pos=(0.0, 0.0),
        hip_roll_vel=(0.0, 0.0),
        hip_roll_ref=(0.0, 0.0),
    )

    assert tau[0] == 0.0
    assert tau[5] == 0.0
    assert diagnostics["stance_torque_left"] == 0.0
    assert diagnostics["stance_torque_right"] == 0.0


def test_lateral_roll_balance_stance_regularization_is_clipped():
    from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController

    controller = LateralRollBalanceController(
        kp_stance=20.0,
        kd_stance=0.0,
        max_stance_torque=3.0,
        stance_weight=0.4,
    )

    tau, diagnostics = controller.compute(
        roll_y_rad=0.0,
        roll_rate_y_rad_s=0.0,
        hip_roll_pos=(-1.0, -1.0),
        hip_roll_vel=(0.0, 0.0),
        hip_roll_ref=(0.0, 0.0),
    )

    assert diagnostics["stance_torque_left"] == 3.0
    assert diagnostics["stance_torque_right"] == 3.0
    # The weighted torque is compared with a tolerance, not for exact equality:
    # it comes back through a float32 intermediate, so it lands 4e-8 from the
    # float64 product — one float32 epsilon. Exact equality here passed or failed
    # purely on whether some earlier test in the session had already switched JAX
    # to x64, which made this the only order-dependent test in the suite.
    assert tau[0] == pytest.approx(0.4 * 3.0, rel=1e-6)
    assert tau[5] == pytest.approx(0.4 * 3.0, rel=1e-6)


def test_lateral_roll_balance_backward_compatibility_without_stance_inputs():
    from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController

    controller = LateralRollBalanceController()

    tau, diagnostics = controller.compute(0.1, 0.05)

    assert diagnostics["stance_error_left"] is None
    assert diagnostics["stance_error_right"] is None
    assert diagnostics["stance_torque_left"] == 0.0
    assert diagnostics["stance_torque_right"] == 0.0
    assert tau[0] != 0.0
    assert tau[5] != 0.0


def test_lateral_roll_balance_large_roll_keeps_roll_balance_dominant():
    from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController

    controller = LateralRollBalanceController(
        kp_roll=40.0,
        kd_roll=8.0,
        kp_stance=5.0,
        kd_stance=1.0,
        max_stance_torque=5.0,
        stance_weight=0.4,
    )

    tau, diagnostics = controller.compute(
        roll_y_rad=0.2,
        roll_rate_y_rad_s=0.0,
        hip_roll_pos=(-0.4, -0.45),
        hip_roll_vel=(0.0, 0.0),
        hip_roll_ref=(0.0, 0.0),
    )

    assert diagnostics["tau_roll_left"] > diagnostics["stance_torque_left"]
    assert abs(float(tau[0])) > diagnostics["stance_weight"] * diagnostics["stance_torque_left"]


def test_lateral_roll_balance_jit_compatibility():
    """Verify lateral roll balance controller works under JIT compilation."""
    import jax
    from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController

    controller = LateralRollBalanceController(hip_roll_torque_sign=1.0)
    roll_y_rad = 0.1
    roll_rate_y_rad_s = 0.05

    # JIT-compile the compute method
    @jax.jit
    def jit_compute(roll_y, roll_rate_y):
        tau, _ = controller.compute(roll_y, roll_rate_y)
        return tau

    # Call JIT-compiled version
    tau_jit = jit_compute(roll_y_rad, roll_rate_y_rad_s)

    # Call non-JIT version for comparison
    tau_eager, _ = controller.compute(roll_y_rad, roll_rate_y_rad_s)

    # Results should match
    assert jnp.allclose(tau_jit, tau_eager)
    assert tau_jit.shape == (ACTION_DIM,)

    # Verify only hip roll indices are nonzero
    assert tau_jit[0] != 0.0  # l_hip_roll
    assert tau_jit[5] != 0.0  # r_hip_roll
    for idx in [1, 2, 3, 4, 6, 7, 8, 9]:
        assert tau_jit[idx] == 0.0


def test_balance_core_components_export_from_controllers_package():
    """Verify all balance-core components are exported from wheeled_biped.controllers package."""
    from wheeled_biped.controllers import (
        BalanceCoreTorqueComposer,
        ContactSupervisor,
        LateralRollBalanceController,
        SagittalWheelBalanceController,
        ShapePostureController,
        SupportFeedforwardController,
        TorqueOwnershipValidator,
    )

    # Verify all classes are importable and are classes
    assert BalanceCoreTorqueComposer is not None
    assert ContactSupervisor is not None
    assert LateralRollBalanceController is not None
    assert SagittalWheelBalanceController is not None
    assert ShapePostureController is not None
    assert SupportFeedforwardController is not None
    assert TorqueOwnershipValidator is not None


def test_balance_core_component_outputs_are_finite_for_nominal_inputs():
    """Verify all balance-core components produce finite outputs for nominal inputs."""
    from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController
    from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController
    from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController

    shape = ShapePostureController()
    support = SupportFeedforwardController(support_vector=jnp.zeros(10), joint_group="knee")
    sagittal = SagittalWheelBalanceController()
    lateral = LateralRollBalanceController()

    tau_shape, _ = shape.compute(jnp.zeros(10), jnp.ones(10) * 0.01, jnp.zeros(10))
    tau_support, _ = support.compute()
    tau_sagittal, _ = sagittal.compute(0.01, 0.02, 0.01, 0.02, 0.1, 0.1, 0.0)
    tau_lateral, _ = lateral.compute(0.01, 0.02)

    for tau in [tau_shape, tau_support, tau_sagittal, tau_lateral]:
        assert jnp.all(jnp.isfinite(tau))
