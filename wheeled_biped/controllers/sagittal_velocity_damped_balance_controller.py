"""Sagittal velocity-damped balance controller for balance-core architecture.

Analytic controller-by-construction approach: explicit state-feedback terms for
pitch balance, sagittal velocity damping, wheel velocity damping, and optional
weak position return. Built after the LQR/sysid path failed Gate 4 identification
(one-step R²=1.0 but 20-step rollout R²=-1.15e10, dominant eigenvalue λ=1.96).

This controller replaces SagittalWheelBalanceController when selected via
--sagittal-controller velocity-damped. Both controllers are mutually exclusive.

Output: nonzero torque only on wheel joints [4, 9].
"""

import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.balance_core_types import (
    ACTION_DIM,
    WHEEL_INDICES,
    zeros_action,
)


class SagittalVelocityDampedBalanceController:
    """Wheel-based sagittal balance with explicit velocity and position damping.

    Control law:
        tau = k_pitch * pitch_x
            + k_pitch_rate * pitch_rate_x
            + k_velocity * (-sagittal_velocity)
            + k_wheel_velocity * (-wheel_velocity_mean)
            + k_position * (-sagittal_position_error)

    Signs verified by unit tests:
        - positive pitch → restoring torque (opposes tilt)
        - positive pitch_rate → damping torque (opposes angular velocity)
        - positive sagittal_velocity → torque reducing forward velocity
        - positive wheel_velocity_mean → opposing torque
        - positive sagittal_position_error → weak return tendency
    """

    def __init__(
        self,
        kp_pitch: float = 50.0,
        kd_pitch: float = 10.0,
        kp_cp: float = 30.0,
        kd_com_vy: float = 5.0,
        k_velocity: float = 0.0,
        k_wheel_velocity: float = 0.5,
        k_position: float = 0.0,
        wheel_torque_sign: float = 1.0,
        max_tau_wheel: float = 5.0,
    ):
        if wheel_torque_sign not in [1.0, -1.0]:
            raise ValueError(f"wheel_torque_sign must be +1.0 or -1.0, got {wheel_torque_sign}")

        self.kp_pitch = kp_pitch
        self.kd_pitch = kd_pitch
        self.kp_cp = kp_cp
        self.kd_com_vy = kd_com_vy
        self.k_velocity = k_velocity
        self.k_wheel_velocity = k_wheel_velocity
        self.k_position = k_position
        self.wheel_torque_sign = wheel_torque_sign
        self.max_tau_wheel = max_tau_wheel

    def compute(
        self,
        pitch_x_rad: float,
        pitch_rate_x_rad_s: float,
        sagittal_velocity_m_s: float,
        wheel_vel_left_rad_s: float,
        wheel_vel_right_rad_s: float,
        sagittal_position_error_m: float = 0.0,
    ) -> tuple[Array, dict]:
        """Compute sagittal velocity-damped balance torque and diagnostics.

        Args:
            pitch_x_rad: Body pitch angle in robot frame (rad).
            pitch_rate_x_rad_s: Body pitch rate in robot frame (rad/s).
            sagittal_velocity_m_s: CoM velocity projected onto initial-heading
                sagittal axis (m/s). Positive = forward.
            wheel_vel_left_rad_s: Left wheel velocity (rad/s).
            wheel_vel_right_rad_s: Right wheel velocity (rad/s).
            sagittal_position_error_m: Sagittal displacement from equilibrium
                along initial-heading axis (m). Positive = forward of reference.

        Returns:
            tau: Torque vector (10,) with nonzero values only at wheel indices [4, 9].
            diagnostics: Dictionary with per-term decomposition and saturation info.
        """
        wheel_vel_mean = 0.5 * (wheel_vel_left_rad_s + wheel_vel_right_rad_s)

        # Per-wheel damping (separate for each wheel)
        tau_wheel_vel_left = -self.k_wheel_velocity * wheel_vel_left_rad_s
        tau_wheel_vel_right = -self.k_wheel_velocity * wheel_vel_right_rad_s

        # Common balance terms
        tau_pitch = self.kp_pitch * pitch_x_rad
        tau_pitch_rate = self.kd_pitch * pitch_rate_x_rad_s
        tau_sagittal_velocity = -self.k_velocity * sagittal_velocity_m_s
        tau_position = -self.k_position * sagittal_position_error_m

        # Capture-point-like term matching baseline controller's cp/com_vy contributions
        # Uses sagittal_position_error as proxy for cp_error and sagittal_velocity as proxy for com_vy
        # when running in initial-heading frame mode. Disabled (0.0) by default to avoid
        # fighting the separate k_position/k_velocity terms.
        tau_cp = -self.kp_cp * sagittal_position_error_m
        tau_com_vy = -self.kd_com_vy * sagittal_velocity_m_s

        # Common scalar command (before per-wheel damping)
        # No internal clipping - let the composer handle torque limits like baseline does
        tau_common_unclipped = tau_pitch + tau_pitch_rate + tau_sagittal_velocity + tau_position + tau_cp + tau_com_vy
        tau_common = self.wheel_torque_sign * tau_common_unclipped

        # Per-wheel torque with common command + individual wheel damping
        tau_left = tau_common + tau_wheel_vel_left
        tau_right = tau_common + tau_wheel_vel_right

        # Build output vector
        tau = zeros_action()
        tau = tau.at[4].set(tau_left)
        tau = tau.at[9].set(tau_right)

        saturated = bool(
            abs(float(tau_common)) >= self.max_tau_wheel * 0.99
            or abs(float(tau_left)) >= self.max_tau_wheel * 0.99
            or abs(float(tau_right)) >= self.max_tau_wheel * 0.99
        )

        diagnostics = {
            "tau_pitch": float(tau_pitch),
            "tau_pitch_rate": float(tau_pitch_rate),
            "tau_cp": float(tau_cp),
            "tau_com_vy": float(tau_com_vy),
            "tau_sagittal_velocity": float(tau_sagittal_velocity),
            "tau_wheel_velocity_left": float(tau_wheel_vel_left),
            "tau_wheel_velocity_right": float(tau_wheel_vel_right),
            "tau_position": float(tau_position),
            "tau_common_unclipped": float(tau_common_unclipped),
            "tau_common_clipped": float(tau_common),
            "tau_left": float(tau_left),
            "tau_right": float(tau_right),
            "tau_total_unclipped": float(tau_common_unclipped + (tau_wheel_vel_left + tau_wheel_vel_right) / 2.0),
            "tau_total_clipped": float(0.5 * (tau_left + tau_right)),
            "saturated": saturated,
            "wheel_vel_mean_rad_s": float(wheel_vel_mean),
            "sagittal_position_error_m": float(sagittal_position_error_m),
            "sagittal_velocity_m_s": float(sagittal_velocity_m_s),
            "pitch_x_rad": float(pitch_x_rad),
            "pitch_rate_x_rad_s": float(pitch_rate_x_rad_s),
            "wheel_torque_sign": self.wheel_torque_sign,
        }

        return tau, diagnostics
