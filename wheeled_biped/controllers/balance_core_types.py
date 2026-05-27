"""Balance-core controller type definitions and constants.

Defines enums, constants, and telemetry schemas for the functional four-source
torque stack: shape/posture + support feedforward + sagittal wheel balance + lateral roll balance.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import jax.numpy as jnp


ACTION_DIM = 10


class TorqueSource(Enum):
    """Functional torque source identification."""
    SHAPE_POSTURE = "shape_posture"
    SUPPORT_FEEDFORWARD = "support_feedforward"
    SAGITTAL_WHEEL_BALANCE = "sagittal_wheel_balance"
    LATERAL_ROLL_BALANCE = "lateral_roll_balance"


class ContactSupervisorState(Enum):
    """Contact state for recovery and force validation."""
    DOUBLE_CONTACT = "double_contact"
    LEFT_ONLY = "left_only"
    RIGHT_ONLY = "right_only"
    FLIGHT_OR_NO_CONTACT = "flight_or_no_contact"


@dataclass(frozen=True)
class ContactSupervisorOutput:
    """Read-only contact classification output for telemetry and recovery hooks."""

    state: ContactSupervisorState
    previous_state: ContactSupervisorState | None
    left_wheel_contact: bool
    right_wheel_contact: bool
    contact_force_valid: bool
    left_normal_force_n: float
    right_normal_force_n: float
    contact_duration_s: float
    transition_event: str
    recovery_hook_fields: Mapping[str, object]


# Joint indices (10-actuated robot: 5 per leg)
LEG_POSITION_INDICES = jnp.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=jnp.int32)
WHEEL_VELOCITY_INDICES = jnp.array([4, 9], dtype=jnp.int32)
WHEEL_INDICES = jnp.array([4, 9], dtype=jnp.int32)  # Alias for clarity
HIP_ROLL_INDICES = jnp.array([0, 5], dtype=jnp.int32)
HIP_YAW_INDICES = jnp.array([1, 6], dtype=jnp.int32)
HIP_PITCH_KNEE_INDICES = jnp.array([2, 3, 7, 8], dtype=jnp.int32)
HIP_PITCH_INDICES = jnp.array([2, 7], dtype=jnp.int32)
KNEE_INDICES = jnp.array([3, 8], dtype=jnp.int32)
SUPPORT_SHAPE_INDICES = jnp.array([1, 2, 3, 6, 7, 8], dtype=jnp.int32)
SUPPORT_FEEDFORWARD_INDICES = jnp.array([2, 3, 7, 8], dtype=jnp.int32)


def zeros_action() -> jnp.ndarray:
    """Return zero action/torque vector with canonical action dimension."""
    return jnp.zeros(ACTION_DIM)

# Robot-frame explicit telemetry field names
BALANCE_CORE_REQUIRED_STATE_TELEMETRY = (
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
)

BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY = (
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
)


def make_balance_core_telemetry_columns() -> dict[str, list]:
    """Initialize telemetry columns with empty lists for all required fields."""
    return {
        name: []
        for name in BALANCE_CORE_REQUIRED_STATE_TELEMETRY + BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY
    }


@dataclass(frozen=True)
class BalanceCoreTorqueResult:
    """Complete torque composition result with telemetry for four approved sources."""

    # Four approved source torques
    tau_shape_posture: jnp.ndarray
    tau_support_feedforward: jnp.ndarray
    tau_sagittal_wheel_balance: jnp.ndarray
    tau_lateral_roll_balance: jnp.ndarray

    # Composition stages
    tau_total_raw: jnp.ndarray
    tau_total_clipped: jnp.ndarray
    tau_final: jnp.ndarray

    # Ownership validation results
    active_torque_owner_per_joint: list
    ownership_violation_count: int
    violations: list

    # Saturation telemetry
    saturation_mask: jnp.ndarray
    rate_saturation_mask: jnp.ndarray

    def _to_float_tuple(self, arr: jnp.ndarray) -> tuple:
        """Convert JAX array to tuple of Python floats for telemetry."""
        return tuple(float(x) for x in arr)

    @property
    def telemetry(self) -> dict:
        """Return telemetry dictionary matching BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY schema."""
        return {
            "tau_shape_posture_per_joint": self._to_float_tuple(self.tau_shape_posture),
            "tau_support_feedforward_per_joint": self._to_float_tuple(self.tau_support_feedforward),
            "tau_sagittal_wheel_balance_per_joint": self._to_float_tuple(self.tau_sagittal_wheel_balance),
            "tau_lateral_roll_balance_per_joint": self._to_float_tuple(self.tau_lateral_roll_balance),
            "tau_total_raw_per_joint": self._to_float_tuple(self.tau_total_raw),
            "tau_total_clipped_per_joint": self._to_float_tuple(self.tau_total_clipped),
            "tau_final_per_joint": self._to_float_tuple(self.tau_final),
            "active_torque_owner_per_joint": tuple(self.active_torque_owner_per_joint),
            "ownership_violation_count": int(self.ownership_violation_count),
            "torque_saturation_mask_per_joint": tuple(bool(x) for x in self.saturation_mask),
            "torque_rate_saturation_mask_per_joint": tuple(bool(x) for x in self.rate_saturation_mask),
        }
