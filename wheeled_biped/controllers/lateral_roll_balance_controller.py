"""Lateral roll balance controller for balance-core architecture.

Provides two complementary functions:
1. Roll stabilization: PD control on body roll error using hip-roll moment arm.
2. Hip-roll stance regularization: low-priority PD control toward equilibrium
   hip-roll references to prevent nominal stance drift.

Both terms live inside this controller — hip-roll joints [0, 5] remain owned
exclusively by this source. Stance regularization is a soft bias; roll
balance always takes priority when body roll is large.
"""

import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.balance_core_types import zeros_action


class LateralRollBalanceController:
    """Hip-roll-based lateral balance and stance controller."""

    def __init__(
        self,
        kp_roll: float = 40.0,
        kd_roll: float = 8.0,
        max_roll_moment: float = 50.0,
        hip_roll_torque_sign: float = 1.0,
        kp_stance: float = 5.0,
        kd_stance: float = 1.0,
        max_stance_torque: float = 5.0,
        stance_weight: float = 0.4,
    ):
        if hip_roll_torque_sign not in [1.0, -1.0]:
            raise ValueError(f"hip_roll_torque_sign must be +1.0 or -1.0, got {hip_roll_torque_sign}")

        self.kp_roll = kp_roll
        self.kd_roll = kd_roll
        self.max_roll_moment = max_roll_moment
        self.hip_roll_torque_sign = hip_roll_torque_sign
        self.kp_stance = kp_stance
        self.kd_stance = kd_stance
        self.max_stance_torque = max_stance_torque
        self.stance_weight = stance_weight

    def compute(
        self,
        roll_y_rad: float,
        roll_rate_y_rad_s: float,
        hip_roll_pos: tuple[float, float] | Array | None = None,
        hip_roll_vel: tuple[float, float] | Array | None = None,
        hip_roll_ref: tuple[float, float] | Array | None = None,
    ) -> tuple[Array, dict]:
        """Compute lateral roll balance torque and diagnostics.

        Optional stance inputs preserve backward-compatible roll-only behavior when
        omitted. When provided, the stance term is clipped and blended as a soft,
        low-priority bias on top of the primary roll-balance torque.
        """
        m_roll_cmd = self.kp_roll * roll_y_rad + self.kd_roll * roll_rate_y_rad_s
        m_roll_clipped = jnp.clip(m_roll_cmd, -self.max_roll_moment, self.max_roll_moment)

        tau_roll_left = self.hip_roll_torque_sign * m_roll_clipped
        tau_roll_right = -self.hip_roll_torque_sign * m_roll_clipped

        stance_active = (
            hip_roll_pos is not None
            and hip_roll_vel is not None
            and hip_roll_ref is not None
        )

        if stance_active:
            hip_roll_pos_arr = jnp.asarray(hip_roll_pos, dtype=jnp.float32)
            hip_roll_vel_arr = jnp.asarray(hip_roll_vel, dtype=jnp.float32)
            hip_roll_ref_arr = jnp.asarray(hip_roll_ref, dtype=jnp.float32)

            if hip_roll_pos_arr.shape != (2,):
                raise ValueError(f"hip_roll_pos must have shape (2,), got {hip_roll_pos_arr.shape}")
            if hip_roll_vel_arr.shape != (2,):
                raise ValueError(f"hip_roll_vel must have shape (2,), got {hip_roll_vel_arr.shape}")
            if hip_roll_ref_arr.shape != (2,):
                raise ValueError(f"hip_roll_ref must have shape (2,), got {hip_roll_ref_arr.shape}")

            stance_error = hip_roll_ref_arr - hip_roll_pos_arr
            stance_raw = self.kp_stance * stance_error - self.kd_stance * hip_roll_vel_arr
            stance_tau = jnp.clip(stance_raw, -self.max_stance_torque, self.max_stance_torque)
            stance_tau_left = stance_tau[0]
            stance_tau_right = stance_tau[1]
        else:
            hip_roll_pos_arr = None
            hip_roll_vel_arr = None
            hip_roll_ref_arr = None
            stance_error = None
            stance_tau_left = 0.0
            stance_tau_right = 0.0

        tau_left = tau_roll_left + self.stance_weight * stance_tau_left
        tau_right = tau_roll_right + self.stance_weight * stance_tau_right

        tau = zeros_action()
        tau = tau.at[0].set(tau_left)
        tau = tau.at[5].set(tau_right)

        diagnostics = {
            "m_roll_cmd": m_roll_cmd,
            "m_roll_clipped": m_roll_clipped,
            "hip_roll_torque_sign": self.hip_roll_torque_sign,
            "sign_convention": "positive" if self.hip_roll_torque_sign > 0 else "negative",
            "tau_roll_left": tau_roll_left,
            "tau_roll_right": tau_roll_right,
            "kp_stance": self.kp_stance,
            "kd_stance": self.kd_stance,
            "max_stance_torque": self.max_stance_torque,
            "stance_weight": self.stance_weight,
            "stance_error_left": None if stance_error is None else stance_error[0],
            "stance_error_right": None if stance_error is None else stance_error[1],
            "stance_torque_left": stance_tau_left,
            "stance_torque_right": stance_tau_right,
            "stance_torque_norm": 0.0 if stance_error is None else jnp.linalg.norm(jnp.array([stance_tau_left, stance_tau_right])),
            "hip_roll_ref_left": None if hip_roll_ref_arr is None else hip_roll_ref_arr[0],
            "hip_roll_ref_right": None if hip_roll_ref_arr is None else hip_roll_ref_arr[1],
            "hip_roll_pos_left": None if hip_roll_pos_arr is None else hip_roll_pos_arr[0],
            "hip_roll_pos_right": None if hip_roll_pos_arr is None else hip_roll_pos_arr[1],
        }

        return tau, diagnostics

