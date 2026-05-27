"""Lateral roll balance controller for balance-core architecture."""

import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.balance_core_types import (
    ACTION_DIM,
    HIP_ROLL_INDICES,
    zeros_action,
)


class LateralRollBalanceController:
    """Hip-roll-based lateral balance controller.

    Outputs nonzero torque only on hip roll joints [0, 5].
    Provides roll stabilization through moment-based control.
    """

    def __init__(
        self,
        kp_roll: float = 40.0,
        kd_roll: float = 8.0,
        max_roll_moment: float = 50.0,
        hip_roll_torque_sign: float = 1.0,
    ):
        """Initialize lateral roll balance controller.

        Args:
            kp_roll: Proportional gain for roll error
            kd_roll: Derivative gain for roll rate
            max_roll_moment: Maximum roll moment magnitude (Nm)
            hip_roll_torque_sign: Sign convention (+1.0 or -1.0)
        """
        if hip_roll_torque_sign not in [1.0, -1.0]:
            raise ValueError(f"hip_roll_torque_sign must be +1.0 or -1.0, got {hip_roll_torque_sign}")

        self.kp_roll = kp_roll
        self.kd_roll = kd_roll
        self.max_roll_moment = max_roll_moment
        self.hip_roll_torque_sign = hip_roll_torque_sign

    def compute(
        self,
        roll_y_rad: float,
        roll_rate_y_rad_s: float,
    ) -> tuple[Array, dict]:
        """Compute lateral roll balance torque and diagnostics.

        Produces nonzero torque only on hip roll joints [0, 5].

        Positive roll_y (body tilted right) produces restoring torques:
        - With sign=1.0: tau_left > 0 (push left side down), tau_right < 0 (pull right side up)
        - With sign=-1.0: opposite torques

        Args:
            roll_y_rad: Body roll angle (rad)
            roll_rate_y_rad_s: Body roll rate (rad/s)

        Returns:
            tau: Torque vector (10,) with nonzero values only at hip roll indices [0, 5]
            diagnostics: Dictionary of diagnostic values
        """
        # Compute roll moment command
        m_roll_cmd = self.kp_roll * roll_y_rad + self.kd_roll * roll_rate_y_rad_s

        # Clip to maximum moment
        m_roll_clipped = jnp.clip(m_roll_cmd, -self.max_roll_moment, self.max_roll_moment)

        # Split moment to left/right hip roll torques
        # Positive roll moment (restoring right tilt) requires:
        # - Left hip roll: positive torque (push left side down)
        # - Right hip roll: negative torque (pull right side up)
        tau_left = self.hip_roll_torque_sign * m_roll_clipped
        tau_right = -self.hip_roll_torque_sign * m_roll_clipped

        # Build output vector (zeros except at hip roll indices)
        tau = zeros_action()
        tau = tau.at[0].set(tau_left)   # l_hip_roll
        tau = tau.at[5].set(tau_right)  # r_hip_roll

        # Diagnostics
        diagnostics = {
            "m_roll_cmd": m_roll_cmd,
            "m_roll_clipped": m_roll_clipped,
            "hip_roll_torque_sign": self.hip_roll_torque_sign,
            "sign_convention": "positive" if self.hip_roll_torque_sign > 0 else "negative",
        }

        return tau, diagnostics
