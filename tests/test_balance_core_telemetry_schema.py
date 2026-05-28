import jax.numpy as jnp
from types import SimpleNamespace

from wheeled_biped.controllers.balance_core_types import (
    BALANCE_CORE_REQUIRED_STATE_TELEMETRY,
    BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY,
    ContactSupervisorOutput,
    ContactSupervisorState,
    make_balance_core_telemetry_columns,
)
from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer
from scripts.simulate_hierarchical_controller import append_balance_core_telemetry


def test_balance_core_required_state_telemetry_names_match_spec():
    expected = {
        "pitch_x_rad",
        "roll_y_rad",
        "yaw_z_rad",
        "pitch_rate_x_rad_s",
        "roll_rate_y_rad_s",
        "yaw_rate_z_rad_s",
        "com_x_m",
        "com_y_m",
        "com_z_m",
        "com_vx_m_s",
        "com_vy_m_s",
        "com_vz_m_s",
        "cp_x_m",
        "cp_y_m",
        "cp_error_y_m",
        "wheel_vel_left_rad_s",
        "wheel_vel_right_rad_s",
        "wheel_vel_mean_rad_s",
        "wheel_acc_left_rad_s2",
        "wheel_acc_right_rad_s2",
        "wheel_acc_mean_rad_s2",
        "left_wheel_contact",
        "right_wheel_contact",
        "contact_supervisor_state",
        "contact_previous_state",
        "contact_duration_s",
        "contact_transition_event",
        "contact_force_valid",
        "contact_recovery_hook_fields",
        "hip_roll_left_rad",
        "hip_roll_right_rad",
        "hip_roll_common_component_rad",
        "hip_roll_symmetric_component_rad",
        "hip_roll_abs_max_rad",
        "hip_roll_ref_left_rad",
        "hip_roll_ref_right_rad",
        "hip_roll_error_left_rad",
        "hip_roll_error_right_rad",
    }
    assert set(BALANCE_CORE_REQUIRED_STATE_TELEMETRY) == expected


def test_balance_core_required_torque_telemetry_names_match_spec():
    expected = {
        "tau_shape_posture_per_joint",
        "tau_support_feedforward_per_joint",
        "tau_sagittal_wheel_balance_per_joint",
        "tau_lateral_roll_balance_per_joint",
        "tau_total_raw_per_joint",
        "tau_total_clipped_per_joint",
        "tau_final_per_joint",
        "active_torque_owner_per_joint",
        "ownership_violation_count",
        "torque_saturation_mask_per_joint",
        "torque_rate_saturation_mask_per_joint",
    }
    assert set(BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY) == expected


def test_make_balance_core_telemetry_columns_initializes_all_required_lists():
    columns = make_balance_core_telemetry_columns()
    for name in BALANCE_CORE_REQUIRED_STATE_TELEMETRY + BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY:
        assert name in columns
        assert columns[name] == []


def test_append_balance_core_telemetry_populates_required_fields():
    """Test that append_balance_core_telemetry populates all required state and torque fields."""
    telemetry = make_balance_core_telemetry_columns()

    # Create a mock composer with required parameters
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.ones(10) * 10.0,
        max_torque_rate=jnp.ones(10) * 100.0,
        control_dt=0.02,
    )

    result = composer.compose(
        tau_shape_posture=jnp.zeros(10),
        tau_support_feedforward=jnp.zeros(10),
        tau_sagittal_wheel_balance=jnp.zeros(10),
        tau_lateral_roll_balance=jnp.zeros(10),
        tau_prev=jnp.zeros(10),
    )

    centroidal_state = SimpleNamespace(
        body_pitch_x=0.1,
        body_roll_y=0.2,
        body_yaw_z=0.3,
        body_pitch_rate_x=0.4,
        body_roll_rate_y=0.5,
        body_yaw_rate_z=0.6,
        com_pos=jnp.array([1.0, 2.0, 3.0]),
        com_vel=jnp.array([4.0, 5.0, 6.0]),
        capture_point=jnp.array([7.0, 8.0]),
    )

    contact = ContactSupervisorOutput(
        state=ContactSupervisorState.DOUBLE_CONTACT,
        previous_state=None,
        left_wheel_contact=True,
        right_wheel_contact=True,
        contact_force_valid=True,
        left_normal_force_n=40.0,
        right_normal_force_n=41.0,
        contact_duration_s=0.0,
        transition_event="initial_double_contact",
        recovery_hook_fields={},
    )

    append_balance_core_telemetry(
        telemetry,
        result,
        centroidal_state,
        contact,
        cp_error_y_m=0.7,
        wheel_vel_left_rad_s=1.1,
        wheel_vel_right_rad_s=1.3,
        wheel_acc_left_rad_s2=2.1,
        wheel_acc_right_rad_s2=2.3,
        hip_roll_pos=(-0.2, -0.3),
        hip_roll_ref=(0.0, 0.0),
    )

    # Verify state fields
    assert telemetry["pitch_x_rad"] == [0.1]
    assert telemetry["roll_y_rad"] == [0.2]
    assert telemetry["yaw_z_rad"] == [0.3]
    assert telemetry["contact_supervisor_state"] == ["double_contact"]
    assert telemetry["contact_previous_state"] == ["none"]
    assert telemetry["contact_transition_event"] == ["initial_double_contact"]
    assert telemetry["hip_roll_left_rad"] == [-0.2]
    assert telemetry["hip_roll_right_rad"] == [-0.3]
    assert abs(telemetry["hip_roll_common_component_rad"][0] - (-0.25)) < 1e-9
    assert abs(telemetry["hip_roll_symmetric_component_rad"][0] - 0.05) < 1e-9
    assert abs(telemetry["wheel_vel_mean_rad_s"][0] - 1.2) < 1e-9
    assert abs(telemetry["wheel_acc_mean_rad_s2"][0] - 2.2) < 1e-9

    # Verify torque fields
    assert len(telemetry["tau_final_per_joint"]) == 1
    assert len(telemetry["tau_shape_posture_per_joint"]) == 1
    assert len(telemetry["ownership_violation_count"]) == 1


def test_balance_core_required_telemetry_fields_have_equal_lengths_after_append():
    """Test that all required balance-core telemetry fields have equal lengths after append."""
    telemetry = make_balance_core_telemetry_columns()
    result = BalanceCoreTorqueComposer(
        torque_limit=jnp.ones(10) * 10.0,
        max_torque_rate=jnp.ones(10) * 400.0,
        control_dt=0.02,
    ).compose(
        tau_shape_posture=jnp.zeros(10),
        tau_support_feedforward=jnp.zeros(10),
        tau_sagittal_wheel_balance=jnp.zeros(10),
        tau_lateral_roll_balance=jnp.zeros(10),
        tau_prev=jnp.zeros(10),
    )
    centroidal_state = SimpleNamespace(
        body_pitch_x=0.0,
        body_roll_y=0.0,
        body_yaw_z=0.0,
        body_pitch_rate_x=0.0,
        body_roll_rate_y=0.0,
        body_yaw_rate_z=0.0,
        com_pos=jnp.zeros(3),
        com_vel=jnp.zeros(3),
        capture_point=jnp.zeros(2),
    )
    contact = ContactSupervisorOutput(
        state=ContactSupervisorState.FLIGHT_OR_NO_CONTACT,
        previous_state=None,
        left_wheel_contact=False,
        right_wheel_contact=False,
        contact_force_valid=False,
        left_normal_force_n=0.0,
        right_normal_force_n=0.0,
        contact_duration_s=0.0,
        transition_event="initial_flight_or_no_contact",
        recovery_hook_fields={},
    )

    append_balance_core_telemetry(telemetry, result, centroidal_state, contact, 0.0, 0.0, 0.0, 0.0, 0.0)

    for name in BALANCE_CORE_REQUIRED_STATE_TELEMETRY + BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY:
        assert len(telemetry[name]) == 1, f"Field {name} has length {len(telemetry[name])}, expected 1"
