import jax.numpy as jnp

from scripts.simulate_hierarchical_controller import (
    build_step1_telemetry_template,
    build_step3_wbc_joint_scale,
    build_step6_wbc_joint_scale,
    compute_step6_hip_roll_authority_scale,
    compute_step1_joint_diagnostics,
    compute_step2_torque_components,
    compute_step4_hip_roll_centering,
    compute_step5_wheel_balance,
    compute_step6_control_mode,
)
from wheeled_biped.controllers.leg_position_controller import LegPositionController


def test_step1_telemetry_template_includes_new_fields():
    telemetry = build_step1_telemetry_template()

    required = {
        "tau_wbc_per_joint",
        "tau_posture_per_joint",
        "tau_leg_position_per_joint",
        "tau_wheel_balance_per_joint",
        "tau_total_per_joint",
        "hip_roll_abs_max",
        "hip_yaw_abs_max",
        "hip_pitch_error_max",
        "knee_error_max",
        "wheel_balance_torque",
        "control_mode",
    }

    assert required.issubset(telemetry.keys())


def test_step1_joint_diagnostics_are_zeroed_and_mode_is_upright():
    joint_pos = jnp.array([0.3, -0.2, 0.9, 1.7, 0.0, -0.4, 0.2, 0.8, 1.6, 0.0])
    joint_pos_error = jnp.array([0.0, 0.1, -0.05, 0.02, 0.0, 0.0, -0.2, 0.08, -0.03, 0.0])

    diagnostics = compute_step1_joint_diagnostics(joint_pos, joint_pos_error)

    assert diagnostics["control_mode"] == "upright"
    assert diagnostics["wheel_balance_torque"] == 0.0
    assert jnp.isclose(diagnostics["hip_roll_abs_max"], 0.4)
    assert jnp.isclose(diagnostics["hip_yaw_abs_max"], 0.2)
    assert jnp.isclose(diagnostics["hip_pitch_error_max"], 0.08)
    assert jnp.isclose(diagnostics["knee_error_max"], 0.03)


def test_step2_torque_components_include_active_leg_position_torque():
    controller = LegPositionController(
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=20.0,
        kd_hip_pitch=2.0,
        kp_knee=30.0,
        kd_knee=3.0,
        max_torque=40.0,
    )
    joint_pos = jnp.array([0.2, -0.2, 0.8, 1.6, 1.0, -0.2, 0.2, 1.0, 1.9, -1.0])
    joint_vel = jnp.zeros(10)
    target_joint_pos = jnp.array([0.0, -0.1, 1.0, 1.7, 0.0, 0.0, 0.1, 0.9, 1.8, 0.0])
    tau_wbc = jnp.ones(10)
    tau_posture = jnp.full(10, 0.5)
    tau_wheel_secondary = jnp.zeros(10)
    tau_inverse_dynamics = jnp.zeros(10)

    components = compute_step2_torque_components(
        controller,
        joint_pos,
        joint_vel,
        target_joint_pos,
        tau_wbc,
        tau_posture,
        tau_wheel_secondary,
        tau_inverse_dynamics,
    )

    expected_leg = jnp.array([0.0, 0.5, 4.0, 3.0, 0.0, 0.0, -0.5, -2.0, -3.0, 0.0])
    assert jnp.allclose(components["tau_leg_position"], expected_leg)
    assert jnp.allclose(components["tau_wheel_balance"], jnp.zeros(10))
    assert jnp.allclose(components["tau_total_raw"], tau_wbc + tau_posture + expected_leg)
    assert jnp.isclose(components["tau_leg_position"][4], 0.0)
    assert jnp.isclose(components["tau_leg_position"][9], 0.0)


def test_step3_default_wbc_joint_scale_preserves_support_joints():
    scale = build_step3_wbc_joint_scale()

    assert jnp.allclose(scale, jnp.array([1.0, 0.3, 0.75, 0.75, 1.0, 1.0, 0.3, 0.75, 0.75, 1.0]))


def test_step3_torque_components_scale_wbc_leg_authority():
    controller = LegPositionController(
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=20.0,
        kd_hip_pitch=2.0,
        kp_knee=30.0,
        kd_knee=3.0,
        max_torque=40.0,
    )
    joint_pos = jnp.zeros(10)
    joint_vel = jnp.zeros(10)
    target_joint_pos = jnp.zeros(10)
    tau_wbc = jnp.arange(1.0, 11.0)
    tau_posture = jnp.zeros(10)
    tau_wheel_secondary = jnp.zeros(10)
    tau_inverse_dynamics = jnp.zeros(10)
    wbc_joint_scale = jnp.array([1.0, 0.3, 0.25, 0.25, 1.0, 1.0, 0.3, 0.25, 0.25, 1.0])

    components = compute_step2_torque_components(
        controller,
        joint_pos,
        joint_vel,
        target_joint_pos,
        tau_wbc,
        tau_posture,
        tau_wheel_secondary,
        tau_inverse_dynamics,
        wbc_joint_scale=wbc_joint_scale,
    )

    expected_wbc_scaled = tau_wbc * wbc_joint_scale
    assert jnp.allclose(components["tau_wbc_scaled"], expected_wbc_scaled)
    assert jnp.allclose(components["tau_total_raw"], expected_wbc_scaled)


def test_step3_telemetry_template_includes_scaled_wbc_field():
    telemetry = build_step1_telemetry_template()

    assert "tau_wbc_scaled_per_joint" in telemetry


def test_step4_hip_roll_centering_uses_deadband_and_restoring_sign():
    joint_pos = jnp.array([0.10, 0.0, 0.0, 0.0, 0.0, -0.50, 0.0, 0.0, 0.0, 0.0])
    joint_vel = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0])

    tau = compute_step4_hip_roll_centering(
        joint_pos,
        joint_vel,
        deadband=0.25,
        kp=12.0,
        kd=1.0,
        max_torque=4.0,
    )

    assert jnp.isclose(tau[0], 0.0)
    assert jnp.isclose(tau[5], 4.0)
    assert jnp.allclose(tau[jnp.array([1, 2, 3, 4, 6, 7, 8, 9])], jnp.zeros(8))


def test_step4_default_hip_roll_centering_is_strong_near_large_spread():
    joint_pos = jnp.array([0.70, 0.0, 0.0, 0.0, 0.0, -0.70, 0.0, 0.0, 0.0, 0.0])
    joint_vel = jnp.zeros(10)

    tau = compute_step4_hip_roll_centering(joint_pos, joint_vel)

    assert jnp.allclose(tau, jnp.array([-9.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0]))


def test_step4_torque_components_include_hip_roll_centering():
    controller = LegPositionController(
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=20.0,
        kd_hip_pitch=2.0,
        kp_knee=30.0,
        kd_knee=3.0,
        max_torque=40.0,
    )
    joint_pos = jnp.array([0.50, 0.0, 0.0, 0.0, 0.0, -0.50, 0.0, 0.0, 0.0, 0.0])
    joint_vel = jnp.zeros(10)
    target_joint_pos = jnp.zeros(10)
    tau_wbc = jnp.zeros(10)
    tau_posture = jnp.zeros(10)
    tau_wheel_secondary = jnp.zeros(10)
    tau_inverse_dynamics = jnp.zeros(10)

    components = compute_step2_torque_components(
        controller,
        joint_pos,
        joint_vel,
        target_joint_pos,
        tau_wbc,
        tau_posture,
        tau_wheel_secondary,
        tau_inverse_dynamics,
        tau_hip_roll_centering=compute_step4_hip_roll_centering(joint_pos, joint_vel),
    )

    assert jnp.allclose(components["tau_hip_roll_centering"], jnp.array([-5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0]))
    assert jnp.allclose(components["tau_total_raw"], components["tau_hip_roll_centering"])


def test_step4_telemetry_template_includes_hip_roll_centering_field():
    telemetry = build_step1_telemetry_template()

    assert "tau_hip_roll_centering_per_joint" in telemetry


def test_step5_wheel_balance_uses_same_wheel_torque_with_verified_signs():
    tau_positive = compute_step5_wheel_balance(
        pitch_rad=0.10,
        pitch_rate_rad_s=0.20,
        capture_point_error_y=0.05,
        kp_pitch=10.0,
        kd_pitch=2.0,
        k_cp=4.0,
        max_torque=5.0,
    )
    tau_negative = compute_step5_wheel_balance(
        pitch_rad=-0.10,
        pitch_rate_rad_s=-0.20,
        capture_point_error_y=-0.05,
        kp_pitch=10.0,
        kd_pitch=2.0,
        k_cp=4.0,
        max_torque=5.0,
    )

    assert jnp.allclose(tau_positive, jnp.array([0.0, 0.0, 0.0, 0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 1.6]))
    assert jnp.allclose(tau_negative, jnp.array([0.0, 0.0, 0.0, 0.0, -1.6, 0.0, 0.0, 0.0, 0.0, -1.6]))


def test_step5_wheel_balance_is_bounded_to_recovery_authority():
    tau = compute_step5_wheel_balance(
        pitch_rad=1.0,
        pitch_rate_rad_s=1.0,
        capture_point_error_y=1.0,
        max_torque=4.0,
    )

    assert jnp.allclose(tau, jnp.array([0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0]))


def test_step6_control_mode_uses_upright_and_recovery_thresholds():
    assert compute_step6_control_mode(roll_rad=0.10, pitch_rad=0.10) == "upright"
    assert compute_step6_control_mode(roll_rad=0.31, pitch_rad=0.00) == "recovery"
    assert compute_step6_control_mode(roll_rad=0.00, pitch_rad=-0.26) == "recovery"
    assert compute_step6_control_mode(roll_rad=0.24, pitch_rad=0.00) == "transition"
def test_step6_recovery_mode_keeps_post_distribution_wbc_scale_constant():
    upright_scale = build_step6_wbc_joint_scale("upright")
    recovery_scale = build_step6_wbc_joint_scale("recovery")

    assert jnp.allclose(upright_scale, build_step3_wbc_joint_scale())
    assert jnp.allclose(recovery_scale, build_step3_wbc_joint_scale())


def test_step6_transition_reduces_hip_roll_authority_but_recovery_preserves_support_authority():
    assert compute_step6_hip_roll_authority_scale("upright") == 1.0
    assert compute_step6_hip_roll_authority_scale("transition") == 0.5
    assert compute_step6_hip_roll_authority_scale("recovery") == 1.0


def test_step6_transition_mode_keeps_step3_scale():
    assert jnp.allclose(build_step6_wbc_joint_scale("transition"), build_step3_wbc_joint_scale())


def test_step5_torque_components_include_wheel_balance():
    controller = LegPositionController(
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=20.0,
        kd_hip_pitch=2.0,
        kp_knee=30.0,
        kd_knee=3.0,
        max_torque=40.0,
    )
    zero = jnp.zeros(10)
    tau_wheel_balance = jnp.array([0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 3.0])

    components = compute_step2_torque_components(
        controller,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        tau_wheel_balance=tau_wheel_balance,
    )

    assert jnp.allclose(components["tau_wheel_balance"], tau_wheel_balance)
    assert jnp.allclose(components["tau_total_raw"], tau_wheel_balance)
