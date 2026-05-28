"""Tests for balance-core type definitions and constants."""

import pytest


def test_balance_core_types_module_exists():
    from wheeled_biped.controllers import balance_core_types
    assert balance_core_types is not None


from wheeled_biped.controllers.balance_core_types import (
    TorqueSource,
    ContactSupervisorState,
    LEG_POSITION_INDICES,
    WHEEL_VELOCITY_INDICES,
    HIP_ROLL_INDICES,
    BALANCE_CORE_REQUIRED_STATE_TELEMETRY,
    BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY,
)


def test_torque_source_enum_has_four_functional_sources():
    assert len(TorqueSource) == 4
    assert TorqueSource.SHAPE_POSTURE.value == "shape_posture"
    assert TorqueSource.SUPPORT_FEEDFORWARD.value == "support_feedforward"
    assert TorqueSource.SAGITTAL_WHEEL_BALANCE.value == "sagittal_wheel_balance"
    assert TorqueSource.LATERAL_ROLL_BALANCE.value == "lateral_roll_balance"


def test_contact_supervisor_state_enum_has_four_states():
    assert len(ContactSupervisorState) == 4
    assert ContactSupervisorState.DOUBLE_CONTACT.value == "double_contact"
    assert ContactSupervisorState.LEFT_ONLY.value == "left_only"
    assert ContactSupervisorState.RIGHT_ONLY.value == "right_only"
    assert ContactSupervisorState.FLIGHT_OR_NO_CONTACT.value == "flight_or_no_contact"


def test_joint_indices_cover_all_10_actuators():
    import jax.numpy as jnp
    all_indices = jnp.concatenate([LEG_POSITION_INDICES, WHEEL_VELOCITY_INDICES])
    assert len(jnp.unique(all_indices)) == 10
    assert jnp.all(all_indices >= 0)
    assert jnp.all(all_indices < 10)


def test_hip_roll_indices_are_symmetric():
    assert len(HIP_ROLL_INDICES) == 2
    assert 0 in HIP_ROLL_INDICES
    assert 5 in HIP_ROLL_INDICES


def test_required_state_telemetry_uses_robot_frame_explicit_names():
    assert "pitch_x_rad" in BALANCE_CORE_REQUIRED_STATE_TELEMETRY
    assert "roll_y_rad" in BALANCE_CORE_REQUIRED_STATE_TELEMETRY
    assert "yaw_z_rad" in BALANCE_CORE_REQUIRED_STATE_TELEMETRY
    assert "hip_roll_common_component_rad" in BALANCE_CORE_REQUIRED_STATE_TELEMETRY
    assert "hip_roll_error_left_rad" in BALANCE_CORE_REQUIRED_STATE_TELEMETRY
    assert "body_pitch" not in BALANCE_CORE_REQUIRED_STATE_TELEMETRY
    assert "body_roll" not in BALANCE_CORE_REQUIRED_STATE_TELEMETRY


def test_required_torque_telemetry_includes_four_sources_and_ownership():
    assert "tau_shape_posture_per_joint" in BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY
    assert "tau_support_feedforward_per_joint" in BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY
    assert "tau_sagittal_wheel_balance_per_joint" in BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY
    assert "tau_lateral_roll_balance_per_joint" in BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY
    assert "tau_final_per_joint" in BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY
    assert "active_torque_owner_per_joint" in BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY
    assert "ownership_violation_count" in BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY
