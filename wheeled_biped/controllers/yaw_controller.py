"""Yaw stabilization controller for balance-core architecture.

Provides antisymmetric hip-yaw torque to stabilize body yaw rotation.
Uses hip-yaw joints [1, 6] to generate yaw moments, complementing the
symmetric posture control from ShapePostureController.
"""

import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.balance_core_types import zeros_action


class YawController:
    """Antisymmetric hip-yaw control for body yaw stabilization."""

    def __init__(
        self,
        kp_yaw: float = 5.0,
        kd_yaw: float = 1.0,
        max_yaw_torque: float = 3.0,
    ):
        """Initialize yaw controller.

        Args:
            kp_yaw: Proportional gain on yaw error [Nm/rad]
            kd_yaw: Derivative gain on yaw rate [Nm/(rad/s)]
            max_yaw_torque: Maximum antisymmetric torque per joint [Nm]
        """
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.max_yaw_torque = max_yaw_torque

    def compute(
        self,
        yaw_error: float,
        yaw_rate: float,
    ) -> tuple[Array, dict]:
        """Compute antisymmetric hip-yaw torque for yaw stabilization.

        Args:
            yaw_error: Yaw error (reference - current) [rad]
            yaw_rate: Body yaw angular velocity [rad/s]

        Returns:
            tau: Joint torque command [10], nonzero only at hip-yaw joints [1, 6]
            diagnostics: Dict with yaw control metrics
        """
        # Antisymmetric PD control (damping term has negative sign)
        tau_antisym_raw = self.kp_yaw * yaw_error - self.kd_yaw * yaw_rate

        # Clip to actuator limits
        tau_antisym = jnp.clip(tau_antisym_raw, -self.max_yaw_torque, self.max_yaw_torque)

        # Apply antisymmetrically to hip-yaw joints
        # Positive yaw moment: left negative, right positive
        tau = zeros_action()
        tau = tau.at[1].set(-tau_antisym)  # left hip-yaw
        tau = tau.at[6].set(tau_antisym)   # right hip-yaw

        diagnostics = {
            "yaw_error": float(yaw_error),
            "yaw_rate": float(yaw_rate),
            "tau_yaw_antisym_raw": float(tau_antisym_raw),
            "tau_yaw_antisym": float(tau_antisym),
            "tau_yaw_left": float(-tau_antisym),
            "tau_yaw_right": float(tau_antisym),
            "yaw_saturated": bool(abs(tau_antisym_raw) > self.max_yaw_torque),
            "kp_yaw": float(self.kp_yaw),
            "kd_yaw": float(self.kd_yaw),
            "max_yaw_torque": float(self.max_yaw_torque),
        }

        return tau, diagnostics
