"""Sagittal wheel balance controller for balance-core architecture."""

import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.balance_core_types import (
    ACTION_DIM,
    WHEEL_INDICES,
    zeros_action,
)


class SagittalWheelBalanceController:
    """Wheel-based sagittal balance controller.

    Outputs nonzero torque only on wheel joints [4, 9].
    Provides pitch stabilization, capture point tracking, and velocity damping.
    """

    def __init__(
        self,
        kp_pitch: float = 50.0,
        kd_pitch: float = 10.0,
        kp_cp: float = 30.0,
        kd_com_vy: float = 5.0,
        kd_wheel_vel: float = 0.5,
        wheel_torque_sign: float = 1.0,
        enable_position_containment: bool = False,
        kp_position: float = 8.0,
        kd_position_velocity: float = 3.0,
        position_deadband_m: float = 0.08,
        position_soft_limit_m: float = 0.25,
        position_hard_limit_m: float = 0.45,
        max_position_bias: float = 15.0,
        pitch_gate_threshold_rad: float = 0.15,
        roll_gate_threshold_rad: float = 0.15,
    ):
        """Initialize sagittal wheel balance controller.

        Args:
            kp_pitch: Proportional gain for pitch error
            kd_pitch: Derivative gain for pitch rate
            kp_cp: Proportional gain for capture point error
            kd_com_vy: Derivative gain for CoM forward velocity
            kd_wheel_vel: Damping gain for wheel velocity
            wheel_torque_sign: Sign convention (+1.0 or -1.0)
            enable_position_containment: FAILED EXPERIMENT - DO NOT USE. E0b multi-zone direct torque
                position containment failed validation (15.98 m drift vs 35.22 m baseline, still
                unacceptable). Direct wheel torque position correction fights balance. Kept for
                research documentation only. Must remain False.
            kp_position: Proportional gain for position drift correction in soft/hard zones (Nm/m)
            kd_position_velocity: Derivative gain for position velocity damping (Nm/(m/s))
            position_deadband_m: Inner deadband radius - no correction inside (m)
            position_soft_limit_m: Soft zone boundary - weak correction between deadband and soft limit (m)
            position_hard_limit_m: Hard zone boundary - flag violation if exceeded (m)
            max_position_bias: Maximum position correction torque (Nm)
            pitch_gate_threshold_rad: Pitch threshold for balance priority gating (rad)
            roll_gate_threshold_rad: Roll threshold for balance priority gating (rad)
        """
        if wheel_torque_sign not in [1.0, -1.0]:
            raise ValueError(f"wheel_torque_sign must be +1.0 or -1.0, got {wheel_torque_sign}")

        self.kp_pitch = kp_pitch
        self.kd_pitch = kd_pitch
        self.kp_cp = kp_cp
        self.kd_com_vy = kd_com_vy
        self.kd_wheel_vel = kd_wheel_vel
        self.wheel_torque_sign = wheel_torque_sign
        self.enable_position_containment = enable_position_containment
        self.kp_position = kp_position
        self.kd_position_velocity = kd_position_velocity
        self.position_deadband_m = position_deadband_m
        self.position_soft_limit_m = position_soft_limit_m
        self.position_hard_limit_m = position_hard_limit_m
        self.max_position_bias = max_position_bias
        self.pitch_gate_threshold_rad = pitch_gate_threshold_rad
        self.roll_gate_threshold_rad = roll_gate_threshold_rad

    def compute(
        self,
        pitch_x_rad: float,
        pitch_rate_x_rad_s: float,
        cp_error_y_m: float,
        com_vy_m_s: float,
        wheel_vel_left_rad_s: float,
        wheel_vel_right_rad_s: float,
        outer_position_bias: float,
        position_y_m: float = 0.0,
        roll_y_rad: float = 0.0,
    ) -> tuple[Array, dict]:
        """Compute sagittal wheel balance torque and diagnostics.

        Produces nonzero torque only on wheel joints [4, 9].

        Args:
            pitch_x_rad: Body pitch angle (rad)
            pitch_rate_x_rad_s: Body pitch rate (rad/s)
            cp_error_y_m: Capture point error in forward direction (m)
            com_vy_m_s: CoM forward velocity (m/s)
            wheel_vel_left_rad_s: Left wheel velocity (rad/s)
            wheel_vel_right_rad_s: Right wheel velocity (rad/s)
            outer_position_bias: Position bias from outer controller (unused, for interface compatibility)
            position_y_m: Body position Y coordinate for drift containment (m)
            roll_y_rad: Body roll angle for balance priority gating (rad)

        Returns:
            tau: Torque vector (10,) with nonzero values only at wheel indices [4, 9]
            diagnostics: Dictionary of diagnostic values
        """
        # Compute balance terms
        term_pitch = self.kp_pitch * pitch_x_rad
        term_pitch_rate = self.kd_pitch * pitch_rate_x_rad_s
        term_cp = -self.kp_cp * cp_error_y_m
        term_com_vy = -self.kd_com_vy * com_vy_m_s

        # E0b: Multi-zone position drift containment
        # FAILED EXPERIMENT - DO NOT USE
        # Failed validation: 15.98 m drift (better than 35.22 m baseline but still unacceptable)
        # Root cause: direct wheel torque position correction fights balance controller
        # Kept for research documentation only. Must remain disabled.
        if self.enable_position_containment:
            # Zone structure: deadband -> soft zone -> hard zone
            position_error_abs = jnp.abs(position_y_m)

            # Determine which zone we're in
            in_deadband = position_error_abs <= self.position_deadband_m
            in_soft_zone = jnp.logical_and(
                position_error_abs > self.position_deadband_m,
                position_error_abs <= self.position_soft_limit_m
            )
            in_hard_zone = jnp.logical_and(
                position_error_abs > self.position_soft_limit_m,
                position_error_abs <= self.position_hard_limit_m
            )
            containment_violation = position_error_abs > self.position_hard_limit_m

            # Compute effective error for each zone
            # Deadband: no position correction, only velocity damping
            deadband_error = 0.0

            # Soft zone: weak correction proportional to distance beyond deadband
            soft_error = jnp.where(
                in_soft_zone,
                position_y_m - jnp.sign(position_y_m) * self.position_deadband_m,
                0.0
            )

            # Hard zone: stronger correction with full gain
            hard_error = jnp.where(
                in_hard_zone,
                position_y_m - jnp.sign(position_y_m) * self.position_deadband_m,
                0.0
            )

            # Combine zone corrections with different gains
            # Soft zone uses 0.5x gain, hard zone uses 1.0x gain
            soft_gain_factor = 0.5
            position_correction_proportional = (
                -self.kp_position * soft_gain_factor * soft_error
                - self.kp_position * hard_error
            )

            # Add velocity damping (active in all zones to prevent drift accumulation)
            position_correction_velocity = -self.kd_position_velocity * com_vy_m_s

            # Balance priority gate: reduce position correction when pitch/roll are large
            # Uses exponential decay: exp(-((pitch/threshold)^2 + (roll/threshold)^2))
            pitch_normalized = pitch_x_rad / self.pitch_gate_threshold_rad
            roll_normalized = roll_y_rad / self.roll_gate_threshold_rad
            balance_priority_gate = jnp.exp(-(pitch_normalized**2 + roll_normalized**2))

            # Gate is active when < 0.95 (pitch/roll approaching safety threshold)
            balance_priority_gate_active = balance_priority_gate < 0.95

            # Apply gating to position correction
            position_correction_raw = (
                (position_correction_proportional + position_correction_velocity)
                * balance_priority_gate
            )

            # Clip to max authority
            position_bias = jnp.clip(
                position_correction_raw,
                -self.max_position_bias,
                self.max_position_bias
            )
        else:
            # E0b disabled - no position containment
            position_error_abs = jnp.abs(position_y_m)
            in_deadband = False
            in_soft_zone = False
            in_hard_zone = False
            containment_violation = False
            position_correction_proportional = 0.0
            position_correction_velocity = 0.0
            position_correction_raw = 0.0
            position_bias = 0.0
            balance_priority_gate = 1.0
            balance_priority_gate_active = False

        # Compute velocity damping (opposes wheel motion)
        term_wheel_vel_left = -self.kd_wheel_vel * wheel_vel_left_rad_s
        term_wheel_vel_right = -self.kd_wheel_vel * wheel_vel_right_rad_s

        # Sign convention verified by debug_wheel_sagittal_sign_simple.py:
        # - Positive wheel torque (ctrl>0) moves robot backward (+Y) and accelerates wheel positively
        # - Positive pitch_x (forward tilt) requires backward motion to recover → positive wheel torque
        # - Negative cp_error_y / negative forward velocity likewise require backward recovery torque
        # - Therefore corrective terms are summed with their measured physics signs and passed through wheel_torque_sign
        # Position bias is added to balance torque before sign convention application
        balance_torque = term_pitch + term_pitch_rate + term_cp + term_com_vy + position_bias

        # Apply sign convention and add per-wheel damping
        tau_left = self.wheel_torque_sign * balance_torque + term_wheel_vel_left
        tau_right = self.wheel_torque_sign * balance_torque + term_wheel_vel_right

        # Build output vector (zeros except at wheel indices)
        tau = zeros_action()
        tau = tau.at[4].set(tau_left)   # l_wheel
        tau = tau.at[9].set(tau_right)  # r_wheel

        # Diagnostics
        wheel_vel_mean = (wheel_vel_left_rad_s + wheel_vel_right_rad_s) / 2.0
        term_wheel_velocity_damping = (term_wheel_vel_left + term_wheel_vel_right) / 2.0
        planar_drift_m = jnp.sqrt(position_y_m**2)  # Sagittal drift magnitude

        diagnostics = {
            "term_pitch": float(term_pitch),
            "term_pitch_rate": float(term_pitch_rate),
            "term_cp": float(term_cp),
            "term_com_vy": float(term_com_vy),
            "term_wheel_vel_left": float(term_wheel_vel_left),
            "term_wheel_vel_right": float(term_wheel_vel_right),
            "balance_torque_raw": float(balance_torque),
            "tau_left": float(tau_left),
            "tau_right": float(tau_right),
            "wheel_vel_mean_rad_s": wheel_vel_mean,
            "term_wheel_velocity_damping": term_wheel_velocity_damping,
            "wheel_torque_sign": self.wheel_torque_sign,
            "sign_convention": "positive" if self.wheel_torque_sign > 0 else "negative",
            # E0b multi-zone drift containment telemetry
            "position_containment_enabled": bool(position_y_m != 0.0),
            "position_y_m": float(position_y_m),
            "position_error_abs": float(position_error_abs),
            "planar_drift_m": float(planar_drift_m),
            "sagittal_position_velocity_m_s": float(com_vy_m_s),
            "position_deadband_m": float(self.position_deadband_m),
            "position_soft_limit_m": float(self.position_soft_limit_m),
            "position_hard_limit_m": float(self.position_hard_limit_m),
            "in_deadband": bool(in_deadband),
            "in_soft_zone": bool(in_soft_zone),
            "in_hard_zone": bool(in_hard_zone),
            "containment_violation": bool(containment_violation),
            "position_correction_proportional": float(position_correction_proportional),
            "position_correction_velocity": float(position_correction_velocity),
            "position_correction_raw": float(position_correction_raw),
            "position_bias": float(position_bias),
            "balance_priority_gate": float(balance_priority_gate),
            "balance_priority_gate_active": bool(balance_priority_gate_active),
        }

        return tau, diagnostics
