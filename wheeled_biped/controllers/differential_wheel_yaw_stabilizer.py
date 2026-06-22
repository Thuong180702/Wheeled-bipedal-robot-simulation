"""Differential wheel yaw stabilizer for balance-core architecture.

Moves body-yaw correction from hip-yaw joints [1, 6] to differential
wheel torque [4, 9]. This addresses the BODY_YAW_WRONG_ACTUATOR issue
where body yaw stabilization through hip-yaw joints causes hip_yaw > 0.35 rad
under large push disturbances (D4/D5).

When this stabilizer is enabled, the existing YawController's gains are reduced
(shared authority mode) so that wheels handle the bulk of yaw correction while
hip-yaw provides fine correction without exceeding the 0.35 rad gate.

Key design: wheel yaw torque is added AFTER the torque composer to avoid
competing with the sagittal balance torque budget.

Opt-in only — disabled by default. No existing profile is changed.
"""

import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.balance_core_types import zeros_action


def compute_height_gate(
    height_m: float,
    z_low: float = 0.340,
    z_high: float = 0.420,
) -> float:
    """Smooth height gate for wheel yaw authority.

    Returns 0.0 at z <= z_low (no wheel yaw at very low heights)
    Returns 1.0 at z >= z_high (full wheel yaw at nominal/high heights)
    Smoothstep transition in between (continuous C1).

    The height gate prevents wheel yaw from destabilizing sagittal balance
    at extreme low heights where the robot has limited stability margin.
    """
    u = (z_high - height_m) / (z_high - z_low)
    u_clamped = max(0.0, min(1.0, u))
    # smoothstep: C1 continuous
    gate = 1.0 - u_clamped * u_clamped * (3.0 - 2.0 * u_clamped)
    return gate


class DifferentialWheelYawStabilizer:
    """Antisymmetric wheel torque for body yaw stabilization.

    Control law (per step):
        tau_yaw_raw = kp_wheel_yaw * yaw_error - kd_wheel_yaw * yaw_rate
        tau[4] = tau_yaw_raw        # left wheel
        tau[9] = -tau_yaw_raw       # right wheel

    Height-gated: authority scales from 0 at low_0p340 to full at 0.420 m.

    Sign convention (matches existing YawController on hip-yaw):
        - Positive yaw_error (robot yawed CW, needs CCW correction):
          tau_yaw_raw > 0 → left wheel forward, right wheel backward → CCW moment
        - Positive yaw_rate (CCW rotation):
          tau_yaw_raw < 0 → left wheel backward, right wheel forward → CW damping

    Output: nonzero torque only on wheel indices [4, 9].
    """

    def __init__(
        self,
        kp_yaw: float = 5.0,
        kd_yaw: float = 1.5,
        max_yaw_torque: float = 5.0,
        lowpass_alpha: float = 0.3,
        height_gate_low: float = 0.280,
        height_gate_high: float = 0.350,
    ):
        """Initialize wheel yaw stabilizer.

        Args:
            kp_yaw: Proportional gain on yaw error [Nm/rad]
            kd_yaw: Derivative gain on yaw rate [Nm/(rad/s)]
            max_yaw_torque: Maximum antisymmetric torque per wheel [Nm]
            lowpass_alpha: First-order lowpass on output tau [0,1].
                1.0 = no filtering. 0.0 = holds previous value.
            height_gate_low: Height below which wheel yaw is zero [m]
            height_gate_high: Height above which full wheel yaw applies [m]
        """
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.max_yaw_torque = max_yaw_torque
        self.lowpass_alpha = lowpass_alpha
        self.height_gate_low = height_gate_low
        self.height_gate_high = height_gate_high
        self._prev_tau_yaw_left = 0.0
        self._prev_tau_yaw_right = 0.0

    def compute(
        self,
        yaw_error: float,
        yaw_rate: float,
        current_height_m: float = 0.45,
    ) -> tuple[Array, dict]:
        """Compute antisymmetric wheel torque for yaw stabilization.

        Args:
            yaw_error: Yaw error (reference - current) [rad]
            yaw_rate: Body yaw angular velocity [rad/s]
            current_height_m: Current CoM height [m] for height gating

        Returns:
            tau: Joint torque command [10], nonzero only at wheel indices [4, 9]
            diagnostics: Dict with wheel yaw control metrics
        """
        # Height gate: reduce authority at low heights
        height_gate = compute_height_gate(
            current_height_m,
            z_low=self.height_gate_low,
            z_high=self.height_gate_high,
        )

        # PD control law
        tau_yaw_raw = self.kp_yaw * yaw_error - self.kd_yaw * yaw_rate

        # Apply height gate
        tau_yaw_gated = tau_yaw_raw * height_gate

        # Clip to actuator limits
        tau_yaw_clipped = jnp.clip(tau_yaw_gated, -self.max_yaw_torque, self.max_yaw_torque)

        # Apply antisymmetrically to wheel joints
        tau_wheel_left_target = float(tau_yaw_clipped)
        tau_wheel_right_target = float(-tau_yaw_clipped)

        # Optional lowpass filtering
        alpha = jnp.clip(self.lowpass_alpha, 0.0, 1.0)
        tau_wheel_left = (1.0 - alpha) * self._prev_tau_yaw_left + alpha * tau_wheel_left_target
        tau_wheel_right = (1.0 - alpha) * self._prev_tau_yaw_right + alpha * tau_wheel_right_target

        # Store for next step
        self._prev_tau_yaw_left = float(tau_wheel_left)
        self._prev_tau_yaw_right = float(tau_wheel_right)

        # Build output
        tau = zeros_action()
        tau = tau.at[4].set(float(tau_wheel_left))
        tau = tau.at[9].set(float(tau_wheel_right))

        diagnostics = {
            "wheel_yaw_error": float(yaw_error),
            "wheel_yaw_rate": float(yaw_rate),
            "wheel_yaw_tau_raw": float(tau_yaw_raw),
            "wheel_yaw_tau_gated": float(tau_yaw_gated),
            "wheel_yaw_tau_clipped": float(tau_yaw_clipped),
            "wheel_yaw_height_gate": float(height_gate),
            "wheel_yaw_height_m": float(current_height_m),
            "wheel_yaw_tau_left": float(tau_wheel_left),
            "wheel_yaw_tau_right": float(tau_wheel_right),
            "wheel_yaw_saturated": bool(abs(float(tau_yaw_gated)) > self.max_yaw_torque),
            "wheel_yaw_kp": float(self.kp_yaw),
            "wheel_yaw_kd": float(self.kd_yaw),
            "wheel_yaw_max_torque": float(self.max_yaw_torque),
            "wheel_yaw_lowpass_alpha": float(self.lowpass_alpha),
        }

        return tau, diagnostics

    def reset(self):
        """Reset internal state. Call on episode reset."""
        self._prev_tau_yaw_left = 0.0
        self._prev_tau_yaw_right = 0.0
