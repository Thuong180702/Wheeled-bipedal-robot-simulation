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
    ):
        """Initialize sagittal wheel balance controller.

        Args:
            kp_pitch: Proportional gain for pitch error
            kd_pitch: Derivative gain for pitch rate
            kp_cp: Proportional gain for capture point error
            kd_com_vy: Derivative gain for CoM forward velocity
            kd_wheel_vel: Damping gain for wheel velocity
            wheel_torque_sign: Sign convention (+1.0 or -1.0)
        """
        if wheel_torque_sign not in [1.0, -1.0]:
            raise ValueError(f"wheel_torque_sign must be +1.0 or -1.0, got {wheel_torque_sign}")

        self.kp_pitch = kp_pitch
        self.kd_pitch = kd_pitch
        self.kp_cp = kp_cp
        self.kd_com_vy = kd_com_vy
        self.kd_wheel_vel = kd_wheel_vel
        self.wheel_torque_sign = wheel_torque_sign

    def compute(
        self,
        pitch_x_rad: float,
        pitch_rate_x_rad_s: float,
        cp_error_y_m: float,
        com_vy_m_s: float,
        wheel_vel_left_rad_s: float,
        wheel_vel_right_rad_s: float,
        outer_position_bias: float,
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

        Returns:
            tau: Torque vector (10,) with nonzero values only at wheel indices [4, 9]
            diagnostics: Dictionary of diagnostic values
        """
        # Compute balance terms
        term_pitch = self.kp_pitch * pitch_x_rad
        term_pitch_rate = self.kd_pitch * pitch_rate_x_rad_s
        term_cp = self.kp_cp * cp_error_y_m
        term_com_vy = self.kd_com_vy * com_vy_m_s

        # Compute velocity damping (opposes wheel motion)
        term_wheel_vel_left = -self.kd_wheel_vel * wheel_vel_left_rad_s
        term_wheel_vel_right = -self.kd_wheel_vel * wheel_vel_right_rad_s

        # Combine balance terms (same for both wheels)
        balance_torque = term_pitch + term_pitch_rate + term_cp + term_com_vy

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

        diagnostics = {
            "wheel_vel_mean_rad_s": wheel_vel_mean,
            "term_wheel_velocity_damping": term_wheel_velocity_damping,
            "wheel_torque_sign": self.wheel_torque_sign,
            "sign_convention": "positive" if self.wheel_torque_sign > 0 else "negative",
        }

        return tau, diagnostics
