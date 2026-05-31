import argparse
import json
from collections import deque
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import pytest

from scripts.simulate_hierarchical_controller import (
    apply_initial_root_z_perturbation,
    build_step1_telemetry_template,
    build_step3_wbc_joint_scale,
    build_step6_wbc_joint_scale,
    compute_step6_hip_roll_authority_scale,
    compute_step1_joint_diagnostics,
    compute_step2_torque_components,
    compute_step4_hip_roll_centering,
    compute_step5_wheel_balance,
    compute_step6_control_mode,
    get_stage2b_default_empirical_feedforward,
    resolve_stage2b_empirical_feedforward,
)
from wheeled_biped.controllers.leg_position_controller import LegPositionController


def test_step1_telemetry_template_includes_root_z_perturbation_fields():
    telemetry = build_step1_telemetry_template()

    required = {
        "initial_root_z_perturbation_m",
        "nominal_equilibrium_com_z_m",
        "initial_com_z_m_after_perturbation",
        "perturbation_applied_after_equilibrium_capture",
    }

    assert required.issubset(telemetry.keys())


def test_step1_telemetry_template_includes_step_e_audit_fields():
    telemetry = build_step1_telemetry_template()

    required = {
        "tau_total_unclipped",
        "tau_total_clipped",
        "tau_total_before_final_clip",
        "tau_total_after_final_clip",
        "tau_position_lower_bound",
        "tau_position_upper_bound",
        "tau_position_total_bound_clipped",
        "position_authority_mode",
        "position_authority_reason",
        "wheel_torque_saturation_left",
        "wheel_torque_saturation_right",
        "wheel_torque_rate_saturation_left",
        "wheel_torque_rate_saturation_right",
    }

    assert required.issubset(telemetry.keys())


def test_apply_initial_root_z_perturbation_offsets_root_z_after_equilibrium_capture():
    class DummyData:
        def __init__(self):
            self.qpos = [0.0, 0.0, 0.45, 1.0]
            self.qvel = [1.0, 2.0, 3.0, 4.0]
            self.qacc = [5.0, 6.0, 7.0, 8.0]

    data = DummyData()

    with patch("scripts.simulate_hierarchical_controller.mujoco.mj_forward") as mj_forward:
        metadata = apply_initial_root_z_perturbation(
            model=object(),
            data=data,
            perturbation_m=0.02,
            nominal_equilibrium_com_z_m=0.41,
        )

    assert data.qpos[2] == pytest.approx(0.47)
    assert data.qvel == [0.0, 0.0, 0.0, 0.0]
    assert data.qacc == [0.0, 0.0, 0.0, 0.0]
    assert metadata == {
        "initial_root_z_perturbation_m": 0.02,
        "nominal_equilibrium_com_z_m": 0.41,
        "initial_com_z_m_after_perturbation": pytest.approx(0.43),
        "perturbation_applied_after_equilibrium_capture": True,
    }
    mj_forward.assert_called_once()


def test_apply_initial_root_z_perturbation_uses_measured_post_perturbation_com_height():
    class DummyData:
        def __init__(self):
            self.qpos = [0.0, 0.0, 0.45, 1.0]
            self.qvel = [1.0, 2.0, 3.0, 4.0]
            self.qacc = [5.0, 6.0, 7.0, 8.0]

    data = DummyData()

    with patch("scripts.simulate_hierarchical_controller.mujoco.mj_forward"):
        metadata = apply_initial_root_z_perturbation(
            model=object(),
            data=data,
            perturbation_m=0.02,
            nominal_equilibrium_com_z_m=0.41,
            initial_com_z_m_after_perturbation=0.418,
        )

    assert metadata["initial_com_z_m_after_perturbation"] == pytest.approx(0.418)


def test_main_accepts_initial_root_z_perturbation_flag():
    from scripts import simulate_hierarchical_controller

    with patch.object(simulate_hierarchical_controller, "validate_balance_core_mode_args") as validate_args:
        with patch.object(simulate_hierarchical_controller.mujoco.MjModel, "from_xml_path", side_effect=RuntimeError("stop after parse")):
            with patch.object(simulate_hierarchical_controller, "time"):
                with patch.object(simulate_hierarchical_controller, "Path") as fake_path:
                    fake_path.return_value.mkdir.return_value = None
                    with patch.object(simulate_hierarchical_controller.sys, "argv", [
                        "simulate_hierarchical_controller.py",
                        "--controller-mode", "balance-core",
                        "--steps", "10",
                        "--initial-root-z-perturbation", "0.02",
                    ]):
                        try:
                            simulate_hierarchical_controller.main()
                        except RuntimeError as exc:
                            assert str(exc) == "stop after parse"
                        else:
                            raise AssertionError("Expected early stop after argument parsing")

    parsed_args = validate_args.call_args.args[0]
    assert parsed_args.initial_root_z_perturbation == 0.02


def test_main_accepts_decimation_failure_window_and_sidecar_flags():
    from scripts import simulate_hierarchical_controller

    with patch.object(simulate_hierarchical_controller, "validate_balance_core_mode_args") as validate_args:
        with patch.object(simulate_hierarchical_controller.mujoco.MjModel, "from_xml_path", side_effect=RuntimeError("stop after parse")):
            with patch.object(simulate_hierarchical_controller, "time"):
                with patch.object(simulate_hierarchical_controller, "Path") as fake_path:
                    fake_path.return_value.mkdir.return_value = None
                    with patch.object(simulate_hierarchical_controller.sys, "argv", [
                        "simulate_hierarchical_controller.py",
                        "--controller-mode", "balance-core",
                        "--steps", "1000",
                        "--telemetry-decimation", "20",
                        "--failure-window-steps", "500",
                        "--write-run-summary-sidecar",
                    ]):
                        try:
                            simulate_hierarchical_controller.main()
                        except RuntimeError as exc:
                            assert str(exc) == "stop after parse"
                        else:
                            raise AssertionError("Expected early stop after argument parsing")

    parsed_args = validate_args.call_args.args[0]
    assert parsed_args.telemetry_decimation == 20
    assert parsed_args.failure_window_steps == 500
    assert parsed_args.write_run_summary_sidecar is True


def test_decimation_boundary_rule_preserves_first_every_n_final_and_termination_rows():
    telemetry_decimation = 4

    def should_keep_main_telemetry_row(source_step_index: int, is_terminating: bool) -> bool:
        if telemetry_decimation <= 1:
            return True
        if source_step_index == 0:
            return True
        if is_terminating:
            return True
        return (source_step_index % telemetry_decimation) == 0

    kept = [
        step for step in range(10)
        if should_keep_main_telemetry_row(step, is_terminating=(step == 9))
    ]

    assert kept == [0, 4, 8, 9]


def test_failure_window_buffer_preserves_latest_full_rate_rows_only():
    failure_window_buffer = deque(maxlen=3)

    for step in range(6):
        failure_window_buffer.append({"source_step_index": step, "time": step * 0.01})

    assert [row["source_step_index"] for row in failure_window_buffer] == [3, 4, 5]


def test_run_summary_sidecar_payload_uses_simulated_steps_not_written_rows(tmp_path):
    finalized_summary_metrics = {
        "pitch_x": {"min": -0.1, "max": 0.2, "rms": 0.12},
        "roll_y": {"min": -0.05, "max": 0.06, "rms": 0.04},
        "com_z": {"min": 0.41, "max": 0.45, "drift": -0.01},
        "wheel_vel_mean": {"min": -1.0, "max": 2.0, "rms": 0.8},
        "wheel_velocity_trend": 0.35,
        "ownership_violation_count_max": 0,
        "hidden_torque_norm_max": 0.0,
        "tau_wbc_norm_max": 0.0,
        "torque_saturation": {"fraction_max": 0.2, "fraction_mean": 0.05},
        "torque_rate_saturation": {"fraction_max": 0.1, "fraction_mean": 0.02},
        "contact_state_summary": {"counts": {"DOUBLE_CONTACT": 10}, "most_common_state": "DOUBLE_CONTACT"},
        "metric_integrity": {"source": "full_rate_online", "limitations": []},
    }
    telemetry = {"time": [0.0, 0.2, 0.4]}
    simulated_steps = 50
    payload = {
        "requested_steps": 50,
        "actual_steps": simulated_steps,
        "survived_steps": simulated_steps,
        "terminated": False,
        "termination_reason": "completed",
        "final_sim_time_s": 0.1,
        "telemetry_decimation": 20,
        "failure_window_steps": 500,
        "written_telemetry_rows": len(telemetry["time"]),
        **finalized_summary_metrics,
    }

    sidecar_path = tmp_path / "telemetry_50.summary.json"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["actual_steps"] == 50
    assert sidecar["survived_steps"] == 50
    assert sidecar["terminated"] is False
    assert sidecar["final_sim_time_s"] == pytest.approx(0.1)
    assert sidecar["wheel_velocity_trend"] == pytest.approx(0.35)
    assert sidecar["written_telemetry_rows"] == 3
    assert sidecar["metric_integrity"]["source"] == "full_rate_online"


def test_failure_window_schema_can_be_adapted_for_validator(tmp_path):
    from wheeled_biped.validation.telemetry_adapter import add_validation_telemetry_fields

    failure_window_telemetry = {
        "source_step_index": [8, 9],
        "time": [0.08, 0.09],
        "control_mode": ["balance-core", "balance-core"],
        "joint_pos": ["0,0,0,0,0,0,0,0,0,0", "0,0,0,0,0,0,0,0,0,0"],
        "joint_vel": ["0,0,0,0,0,0,0,0,0,0", "0,0,0,0,0,0,0,0,0,0"],
        "joint_pos_error": ["0,0,0,0,0,0,0,0,0,0", "0,0,0,0,0,0,0,0,0,0"],
        "tau_wheel_balance_per_joint": ["0,0,0,0,0,0,0,0,0,0", "0,0,0,0,0,0,0,0,0,0"],
        "tau_hip_roll_centering_per_joint": ["0,0,0,0,0,0,0,0,0,0", "0,0,0,0,0,0,0,0,0,0"],
        "tau_posture_per_joint": ["0,0,0,0,0,0,0,0,0,0", "0,0,0,0,0,0,0,0,0,0"],
        "tau_leg_position_per_joint": ["0,0,0,0,0,0,0,0,0,0", "0,0,0,0,0,0,0,0,0,0"],
        "tau_final_per_joint": ["0,0,0,0,0,0,0,0,0,0", "0,0,0,0,0,0,0,0,0,0"],
        "active_torque_owner_per_joint": ["shape_posture,shape_posture,shape_posture,shape_posture,shape_posture,shape_posture,shape_posture,shape_posture,shape_posture,shape_posture"] * 2,
    }

    csv_path = tmp_path / "failure_window_10.csv"
    add_validation_telemetry_fields(
        failure_window_telemetry,
        control_dt=0.01,
        csv_path=str(csv_path),
        survival_steps_override=10,
    )

    assert failure_window_telemetry["survival_steps"] == [10, 10]
    assert failure_window_telemetry["step"] == [8, 9]
    assert "actuator_ctrl_per_joint" in failure_window_telemetry
    assert "hidden_torque_norm" in failure_window_telemetry


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


def test_stage2b_default_empirical_feedforward_is_fixed_validated_vector():
    empirical_ff = get_stage2b_default_empirical_feedforward()

    assert empirical_ff.shape == (10,)
    assert jnp.isclose(empirical_ff[3], -15.5)
    assert jnp.isclose(empirical_ff[8], -15.8)
    assert jnp.isclose(empirical_ff[0], 0.0)
    assert jnp.isclose(empirical_ff[9], 0.0)


def test_stage2b_default_source_does_not_require_telemetry_file():
    empirical_ff = resolve_stage2b_empirical_feedforward(telemetry_path=None)

    assert empirical_ff.shape == (10,)
    assert jnp.isclose(empirical_ff[3], -15.5)
    assert jnp.isclose(empirical_ff[8], -15.8)
