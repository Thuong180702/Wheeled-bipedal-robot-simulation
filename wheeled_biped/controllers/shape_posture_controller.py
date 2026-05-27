"""Shape-posture controller for balance-core architecture."""

import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.balance_core_types import (
    ACTION_DIM,
    SUPPORT_SHAPE_INDICES,
    zeros_action,
)


class ShapePostureController:
    """Joint-space PD posture controller on support-shape joints only."""

    def __init__(
        self,
        kp_hip_yaw: float = 5.0,
        kd_hip_yaw: float = 1.0,
        kp_hip_pitch: float = 30.0,
        kd_hip_pitch: float = 4.0,
        kp_knee: float = 40.0,
        kd_knee: float = 5.0,
    ):
        self.kp_hip_yaw = kp_hip_yaw
        self.kd_hip_yaw = kd_hip_yaw
        self.kp_hip_pitch = kp_hip_pitch
        self.kd_hip_pitch = kd_hip_pitch
        self.kp_knee = kp_knee
        self.kd_knee = kd_knee

    def compute(
        self,
        q_ref: Array,
        joint_pos: Array,
        joint_vel: Array,
        posture_weight: float = 1.0,
        contact_degraded_scale: float = 1.0,
    ) -> tuple[Array, dict]:
        """Compute posture torque and diagnostics.

        Produces nonzero torque only on support-shape joints:
        hip-yaw [1,6], hip-pitch [2,7], knee [3,8].
        """
        if q_ref.shape != (ACTION_DIM,):
            raise ValueError(f"q_ref must be shape ({ACTION_DIM},), got {q_ref.shape}")
        if joint_pos.shape != (ACTION_DIM,):
            raise ValueError(
                f"joint_pos must be shape ({ACTION_DIM},), got {joint_pos.shape}"
            )
        if joint_vel.shape != (ACTION_DIM,):
            raise ValueError(
                f"joint_vel must be shape ({ACTION_DIM},), got {joint_vel.shape}"
            )

        authority_scale = posture_weight * contact_degraded_scale
        posture_error = q_ref - joint_pos

        tau = zeros_action()

        for idx in [1, 6]:
            tau_raw = self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx]
            tau = tau.at[idx].set(authority_scale * tau_raw)

        for idx in [2, 7]:
            tau_raw = self.kp_hip_pitch * posture_error[idx] - self.kd_hip_pitch * joint_vel[idx]
            tau = tau.at[idx].set(authority_scale * tau_raw)

        for idx in [3, 8]:
            tau_raw = self.kp_knee * posture_error[idx] - self.kd_knee * joint_vel[idx]
            tau = tau.at[idx].set(authority_scale * tau_raw)

        diagnostics = {
            "posture_error_norm": float(jnp.linalg.norm(posture_error[SUPPORT_SHAPE_INDICES])),
            "torque_norm": float(jnp.linalg.norm(tau[SUPPORT_SHAPE_INDICES])),
        }

        return tau, diagnostics
