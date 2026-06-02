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
from dataclasses import dataclass
from typing import Optional

from wheeled_biped.controllers.balance_core_types import (
    ACTION_DIM,
    WHEEL_INDICES,
    zeros_action,
)
from wheeled_biped.controllers.position_hold_capture_gate import (
    PositionHoldCaptureGate,
    CaptureGateDiagnostics,
)


@dataclass(frozen=True)
class SagittalAuthoritySchedule:
    profile_name: str = "baseline"
    applies_to_variants: tuple[str, ...] = ()
    position_tau_cap_scale: float = 1.0
    position_tau_cap_by_variant: tuple[tuple[str, float], ...] = ()
    pitch_tau_scale: float = 1.0
    pitch_tau_cap_nm: float | None = None
    velocity_damping_scale: float = 1.0
    support_velocity_scale: float = 1.0
    support_velocity_gain: float | None = None

    def is_active_for_variant(self, variant_name: str | None) -> bool:
        return variant_name is not None and variant_name in self.applies_to_variants

    def max_position_tau_for_variant(self, variant_name: str | None, baseline_max_position_tau: float) -> float:
        if not self.is_active_for_variant(variant_name):
            return baseline_max_position_tau
        for candidate_name, max_position_tau in self.position_tau_cap_by_variant:
            if candidate_name == variant_name:
                return float(max_position_tau)
        return baseline_max_position_tau * self.position_tau_cap_scale


BASELINE_AUTHORITY_SCHEDULE = SagittalAuthoritySchedule()


class SagittalVelocityDampedBalanceController:
    """Wheel-based sagittal balance with explicit velocity and position damping.

    Control law:
        tau = k_pitch * pitch_x
            + k_pitch_rate * pitch_rate_x
            + k_velocity * (-sagittal_velocity)
            + k_wheel_velocity * (-wheel_velocity_mean)
            + k_position * (-sagittal_position_error)
            + k_support_velocity * (-support_position_velocity)

    Signs verified by unit tests:
        - positive pitch → restoring torque (opposes tilt)
        - positive pitch_rate → damping torque (opposes angular velocity)
        - positive sagittal_velocity → torque reducing forward velocity
        - positive wheel_velocity_mean → opposing torque
        - positive sagittal_position_error → weak return tendency
        - positive support_position_velocity → damping torque (opposes support drift)
    """

    def __init__(
        self,
        kp_pitch: float = 50.0,
        kd_pitch: float = 10.0,
        kp_cp: float = 0.0,
        kd_com_vy: float = 5.0,
        k_velocity: float = 0.0,
        k_wheel_velocity: float = 0.5,
        k_position: float = 0.0,
        k_support_velocity: float = 0.0,
        max_position_tau: float = 3.0,
        wheel_torque_sign: float = 1.0,
        max_tau_wheel: float = 5.0,
        enable_capture_gate: bool = False,
        capture_gate_config: Optional[dict] = None,
        dt: float = 0.01,
        enable_torque_budget_aware_position: bool = False,
        position_tau_budget_cap: float = 7.0,
        enable_position_integral: bool = False,
        ki_position_integral: float = 0.0,
        integral_max_abs: float = 1.0,
        integral_pitch_error_threshold_rad: float = 0.03,
        integral_roll_error_threshold_rad: float = 0.05,
        integral_pitch_rate_threshold_rad_s: float = 0.05,
        integral_support_velocity_threshold_m_s: float = 0.03,
        integral_wheel_velocity_threshold_rad_s: float = 1.0,
        integral_min_com_z_m: float = 0.38,
        integral_max_com_z_m: float = 0.43,
        authority_schedule: SagittalAuthoritySchedule | None = None,
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
        self.k_support_velocity = k_support_velocity
        self.max_position_tau = max_position_tau
        self.wheel_torque_sign = wheel_torque_sign
        self.max_tau_wheel = max_tau_wheel
        self.enable_capture_gate = enable_capture_gate
        self.dt = dt
        self.enable_torque_budget_aware_position = enable_torque_budget_aware_position
        self.position_tau_budget_cap = position_tau_budget_cap
        self.enable_position_integral = enable_position_integral
        self.ki_position_integral = ki_position_integral
        self.integral_max_abs = integral_max_abs
        self.integral_pitch_error_threshold_rad = integral_pitch_error_threshold_rad
        self.integral_roll_error_threshold_rad = integral_roll_error_threshold_rad
        self.integral_pitch_rate_threshold_rad_s = integral_pitch_rate_threshold_rad_s
        self.integral_support_velocity_threshold_m_s = integral_support_velocity_threshold_m_s
        self.integral_wheel_velocity_threshold_rad_s = integral_wheel_velocity_threshold_rad_s
        self.integral_min_com_z_m = integral_min_com_z_m
        self.integral_max_com_z_m = integral_max_com_z_m
        self.authority_schedule = authority_schedule or BASELINE_AUTHORITY_SCHEDULE
        self.position_integral_error = 0.0

        # State for support position velocity computation
        self.prev_support_position_error_m = 0.0

        # Initialize capture gate if enabled
        if self.enable_capture_gate:
            gate_config = capture_gate_config or {}
            self.capture_gate = PositionHoldCaptureGate(**gate_config)
        else:
            self.capture_gate = None

    def compute(
        self,
        pitch_x_rad: float,
        pitch_rate_x_rad_s: float,
        sagittal_velocity_m_s: float,
        wheel_vel_left_rad_s: float,
        wheel_vel_right_rad_s: float,
        sagittal_position_error_m: float = 0.0,
        com_y_m: float = 0.0,
        com_vy_m_s: float = 0.0,
        support_center_y_m: float = 0.0,
        com_z_m: float = 0.4,
        roll_y_rad: float = 0.0,
        contact_valid: bool = True,
        height_variant_name: str | None = None,
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
            com_y_m: CoM position in sagittal direction (m). Required for capture gate.
            com_vy_m_s: CoM velocity in sagittal direction (m/s). Required for capture gate.
            support_center_y_m: Support center position in sagittal direction (m). Required for capture gate.
            com_z_m: CoM height (m). Required for capture gate.

        Returns:
            tau: Torque vector (10,) with nonzero values only at wheel indices [4, 9].
            diagnostics: Dictionary with per-term decomposition and saturation info.
        """
        wheel_vel_mean = 0.5 * (wheel_vel_left_rad_s + wheel_vel_right_rad_s)
        schedule_active = self.authority_schedule.is_active_for_variant(height_variant_name)
        effective_max_position_tau = self.authority_schedule.max_position_tau_for_variant(
            height_variant_name,
            self.max_position_tau,
        )
        effective_pitch_scale = self.authority_schedule.pitch_tau_scale if schedule_active else 1.0
        effective_pitch_tau_cap = self.authority_schedule.pitch_tau_cap_nm if schedule_active else None
        effective_velocity_damping_scale = self.authority_schedule.velocity_damping_scale if schedule_active else 1.0
        effective_support_velocity_scale = self.authority_schedule.support_velocity_scale if schedule_active else 1.0
        effective_support_velocity_gain = (
            self.authority_schedule.support_velocity_gain
            if schedule_active and self.authority_schedule.support_velocity_gain is not None
            else self.k_support_velocity
        )

        # Compute support position velocity (numerical derivative)
        # This is the rate of change of support-center position error in initial-heading frame
        support_position_velocity_m_s = (sagittal_position_error_m - self.prev_support_position_error_m) / self.dt
        self.prev_support_position_error_m = sagittal_position_error_m

        # Per-wheel damping (separate for each wheel)
        tau_wheel_vel_left = -self.k_wheel_velocity * wheel_vel_left_rad_s
        tau_wheel_vel_right = -self.k_wheel_velocity * wheel_vel_right_rad_s

        # Common balance terms
        tau_pitch_raw = self.kp_pitch * pitch_x_rad
        tau_pitch_scheduled = tau_pitch_raw * effective_pitch_scale
        if effective_pitch_tau_cap is None:
            tau_pitch = tau_pitch_scheduled
            tau_pitch_clipped = tau_pitch_scheduled
        else:
            tau_pitch = float(jnp.clip(tau_pitch_scheduled, -effective_pitch_tau_cap, effective_pitch_tau_cap))
            tau_pitch_clipped = tau_pitch
        tau_pitch_rate = self.kd_pitch * pitch_rate_x_rad_s
        tau_sagittal_velocity = -self.k_velocity * effective_velocity_damping_scale * sagittal_velocity_m_s

        # Support position velocity damping term
        # Directly opposes support-center drift velocity to prevent transient position excursions
        tau_support_velocity = -effective_support_velocity_gain * effective_support_velocity_scale * support_position_velocity_m_s

        # Capture-point-like term matching baseline controller's cp/com_vy contributions
        # Uses sagittal_position_error as proxy for cp_error and sagittal_velocity as proxy for com_vy
        # when running in initial-heading frame mode. Disabled (0.0) by default to avoid
        # fighting the separate k_position/k_velocity terms.
        tau_cp = -self.kp_cp * sagittal_position_error_m
        tau_com_vy = -self.kd_com_vy * sagittal_velocity_m_s

        # Position hold term with optional smart capture gating
        tau_position_p = -self.k_position * sagittal_position_error_m
        tau_position_raw = tau_position_p

        integral_active = False
        integral_gate_reason = "disabled"
        integral_saturation_flag = False
        if self.enable_position_integral:
            abs_pitch_error = abs(float(pitch_x_rad))
            abs_pitch_rate = abs(float(pitch_rate_x_rad_s))
            abs_support_velocity = abs(float(support_position_velocity_m_s))
            abs_wheel_velocity_mean = abs(float(wheel_vel_mean))
            com_z_safe = self.integral_min_com_z_m <= float(com_z_m) <= self.integral_max_com_z_m

            if abs_pitch_error > self.integral_pitch_error_threshold_rad:
                integral_gate_reason = "pitch_error_large"
            elif abs_pitch_rate > self.integral_pitch_rate_threshold_rad_s:
                integral_gate_reason = "pitch_rate_large"
            elif abs_support_velocity > self.integral_support_velocity_threshold_m_s:
                integral_gate_reason = "support_velocity_large"
            elif abs_wheel_velocity_mean > self.integral_wheel_velocity_threshold_rad_s:
                integral_gate_reason = "wheel_velocity_large"
            elif not com_z_safe:
                integral_gate_reason = "height_unsafe"
            elif abs(float(roll_y_rad)) > self.integral_roll_error_threshold_rad:
                integral_gate_reason = "roll_error_large"
            elif not contact_valid:
                integral_gate_reason = "contact_invalid"
            else:
                integral_active = True
                integral_gate_reason = "safe_steady_state"

            if integral_active:
                self.position_integral_error += float(sagittal_position_error_m) * self.dt
            else:
                self.position_integral_error = 0.0

        tau_position_integral_unclipped = -self.ki_position_integral * self.position_integral_error
        tau_position_integral = float(max(
            -self.integral_max_abs,
            min(self.integral_max_abs, float(tau_position_integral_unclipped)),
        ))
        integral_saturation_flag = abs(tau_position_integral - float(tau_position_integral_unclipped)) > 1e-9
        if not integral_active:
            tau_position_integral = 0.0
            integral_saturation_flag = False

        tau_position_raw = tau_position_p + tau_position_integral

        # Apply capture gate if enabled
        capture_gate_diagnostics = None
        if self.enable_capture_gate and self.capture_gate is not None:
            tau_position_before_clip, capture_gate_diagnostics = self.capture_gate.apply_gate(
                tau_position_raw=tau_position_raw,
                pitch_x_rad=pitch_x_rad,
                pitch_rate_x_rad_s=pitch_rate_x_rad_s,
                com_y_m=com_y_m,
                com_vy_m_s=com_vy_m_s,
                support_center_y_m=support_center_y_m,
                com_z_m=com_z_m,
            )
        else:
            tau_position_before_clip = tau_position_raw

        # Torque-budget-aware position authority allocation
        if self.enable_torque_budget_aware_position:
            # Compute balance torque before position (all terms except position)
            tau_balance_before_position = (
                tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
                tau_support_velocity + tau_cp + tau_com_vy +
                0.5 * (tau_wheel_vel_left + tau_wheel_vel_right)
            )

            tau_position_lower_bound = -self.max_tau_wheel - float(tau_balance_before_position)
            tau_position_upper_bound = self.max_tau_wheel - float(tau_balance_before_position)

            tau_balance_before_position_log = float(tau_balance_before_position)

            tau_position = float(
                jnp.clip(
                    tau_position_before_clip,
                    tau_position_lower_bound,
                    tau_position_upper_bound,
                )
            )
            tau_position_total_bound_clipped = abs(tau_position - float(tau_position_before_clip)) > 1e-9
            tau_pitch_reserve_applied = 0.0

            if abs(tau_position_before_clip) < 1e-9:
                tau_position_saturation_reason = "none"
                position_authority_reason = "within_bounds"
            elif tau_position <= tau_position_lower_bound + 1e-9 and tau_position_before_clip < tau_position_lower_bound:
                tau_position_saturation_reason = "lower_bound"
                position_authority_reason = "lower_bound"
            elif tau_position >= tau_position_upper_bound - 1e-9 and tau_position_before_clip > tau_position_upper_bound:
                tau_position_saturation_reason = "upper_bound"
                position_authority_reason = "upper_bound"
            else:
                tau_position_saturation_reason = "none"
                position_authority_reason = "within_bounds"

            tau_position_saturated = tau_position_total_bound_clipped

            tau_position_budget_available = float(
                tau_position_upper_bound if tau_position_before_clip >= 0.0 else -tau_position_lower_bound
            )
            tau_position_budget_allowed = float(
                max(0.0, tau_position_upper_bound)
                if tau_position_before_clip >= 0.0
                else max(0.0, -tau_position_lower_bound)
            )
            tau_position_budget_cap = float(self.position_tau_budget_cap)
            pitch_reserve_tau_log = 0.0
            position_authority_mode = "total_torque_bound"

        else:
            # Legacy fixed-cap clipping
            tau_position = float(jnp.clip(tau_position_before_clip, -effective_max_position_tau, effective_max_position_tau))
            tau_position_total_bound_clipped = False
            tau_position_saturated = abs(tau_position_before_clip) >= effective_max_position_tau * 0.99
            tau_position_saturation_reason = "fixed_cap" if tau_position_saturated else "none"
            position_authority_reason = tau_position_saturation_reason

            # Telemetry placeholders
            tau_balance_before_position_log = 0.0
            tau_position_budget_available = 0.0
            tau_position_budget_allowed = 0.0
            tau_position_budget_cap = float(effective_max_position_tau)
            pitch_reserve_tau_log = 0.0
            tau_pitch_reserve_applied = 0.0
            tau_position_lower_bound = -float(effective_max_position_tau)
            tau_position_upper_bound = float(effective_max_position_tau)
            position_authority_mode = "scheduled_fixed_cap" if schedule_active else "fixed_cap"

        # Common scalar command (before per-wheel damping)
        # No internal clipping - let the composer handle torque limits like baseline does
        tau_common_unclipped = (
            tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
            tau_support_velocity + tau_position + tau_cp + tau_com_vy
        )
        tau_total_before_final_clip = float(tau_common_unclipped + (tau_wheel_vel_left + tau_wheel_vel_right) / 2.0)
        tau_common = self.wheel_torque_sign * tau_common_unclipped

        # Per-wheel torque with common command + individual wheel damping
        tau_left = tau_common + tau_wheel_vel_left
        tau_right = tau_common + tau_wheel_vel_right
        tau_total_after_final_clip = float(0.5 * (tau_left + tau_right))

        # Compute final wheel torque margin
        final_wheel_torque_max = max(abs(float(tau_left)), abs(float(tau_right)))
        final_wheel_torque_margin = self.max_tau_wheel - final_wheel_torque_max

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
            "tau_pitch_raw": float(tau_pitch_raw),
            "tau_pitch_scheduled": float(tau_pitch_scheduled),
            "tau_pitch_clipped": float(tau_pitch_clipped),
            "tau_pitch_rate": float(tau_pitch_rate),
            "tau_cp": float(tau_cp),
            "tau_com_vy": float(tau_com_vy),
            "tau_sagittal_velocity": float(tau_sagittal_velocity),
            "tau_support_velocity": float(tau_support_velocity),
            "tau_wheel_velocity_left": float(tau_wheel_vel_left),
            "tau_wheel_velocity_right": float(tau_wheel_vel_right),
            "tau_position_raw": float(tau_position_raw),
            "tau_position_p": float(tau_position_p),
            "tau_position_i": float(tau_position_integral),
            "tau_position_integral": float(tau_position_integral),
            "tau_position_total": float(tau_position_raw),
            "position_integral_error": float(self.position_integral_error),
            "integral_active": bool(integral_active),
            "integral_gate_reason": integral_gate_reason,
            "integral_saturation_flag": bool(integral_saturation_flag),
            "pitch_error_x_rad": float(pitch_x_rad),
            "wheel_velocity_mean_rad_s": float(wheel_vel_mean),
            "com_z_m": float(com_z_m),
            "tau_position_before_clip": float(tau_position_before_clip),
            "tau_position": float(tau_position),
            "tau_position_clipped": float(tau_position),
            "tau_position_lower_bound": float(tau_position_lower_bound),
            "tau_position_upper_bound": float(tau_position_upper_bound),
            "tau_position_total_bound_clipped": bool(tau_position_total_bound_clipped),
            "position_authority_mode": position_authority_mode,
            "position_authority_reason": position_authority_reason,
            "max_position_tau": float(self.max_position_tau),
            "sagittal_schedule_profile": self.authority_schedule.profile_name,
            "high_height_schedule_active": bool(schedule_active),
            "effective_max_position_tau": float(effective_max_position_tau),
            "effective_pitch_scale": float(effective_pitch_scale),
            "effective_pitch_tau_cap": "none" if effective_pitch_tau_cap is None else float(effective_pitch_tau_cap),
            "effective_velocity_damping_scale": float(effective_velocity_damping_scale),
            "effective_support_velocity_scale": float(effective_support_velocity_scale),
            "effective_support_velocity_gain": float(effective_support_velocity_gain),
            "tau_pitch_to_position_ratio": float(abs(tau_pitch) / max(abs(tau_position), 1e-9)),
            "height_variant_name": height_variant_name or "none",
            "tau_position_saturation_flag": bool(tau_position_saturated),
            "tau_position_saturation_reason": tau_position_saturation_reason,
            "tau_balance_before_position": tau_balance_before_position_log,
            "tau_position_budget_available": tau_position_budget_available,
            "tau_position_budget_allowed": tau_position_budget_allowed,
            "tau_position_budget_cap": tau_position_budget_cap,
            "pitch_reserve_tau": pitch_reserve_tau_log,
            "tau_pitch_reserve_applied": float(tau_pitch_reserve_applied),
            "enable_torque_budget_aware_position": self.enable_torque_budget_aware_position,
            "tau_common_unclipped": float(tau_common_unclipped),
            "tau_common_clipped": float(tau_common),
            "tau_left": float(tau_left),
            "tau_right": float(tau_right),
            "tau_total_before_final_clip": tau_total_before_final_clip,
            "tau_total_after_final_clip": tau_total_after_final_clip,
            "tau_total_unclipped": tau_total_before_final_clip,
            "tau_total_clipped": tau_total_after_final_clip,
            "final_wheel_torque_margin": final_wheel_torque_margin,
            "saturated": saturated,
            "wheel_vel_mean_rad_s": float(wheel_vel_mean),
            "sagittal_position_error_m": float(sagittal_position_error_m),
            "sagittal_velocity_m_s": float(sagittal_velocity_m_s),
            "support_position_velocity_m_s": float(support_position_velocity_m_s),
            "pitch_x_rad": float(pitch_x_rad),
            "pitch_rate_x_rad_s": float(pitch_rate_x_rad_s),
            "wheel_torque_sign": self.wheel_torque_sign,
            "k_support_velocity": self.k_support_velocity,
        }

        # Add capture gate diagnostics if enabled
        if capture_gate_diagnostics is not None:
            diagnostics.update({
                "capture_gate_enabled": True,
                "capture_gate_required_direction": capture_gate_diagnostics.required_capture_direction,
                "capture_gate_tau_position_direction": capture_gate_diagnostics.tau_position_direction,
                "capture_gate_position_opposes_capture": capture_gate_diagnostics.position_opposes_capture,
                "capture_gate_factor": capture_gate_diagnostics.gate_factor,
                "capture_gate_active": capture_gate_diagnostics.gate_active,
                "capture_gate_reason": capture_gate_diagnostics.gate_reason,
                "capture_gate_pitch_reversal": capture_gate_diagnostics.pitch_reversal_detected,
                "capture_gate_capture_recovery": capture_gate_diagnostics.capture_recovery_detected,
                "capture_gate_tau_position_gated": capture_gate_diagnostics.tau_position_gated,
                "capture_gate_cp_relative_to_support_m": capture_gate_diagnostics.capture_point_relative_to_support_m,
                "capture_gate_com_support_error_m": capture_gate_diagnostics.com_support_error_y_m,
            })
        else:
            diagnostics["capture_gate_enabled"] = False

        return tau, diagnostics
