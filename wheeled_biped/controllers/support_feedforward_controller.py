"""Support feedforward controller for balance-core architecture."""

import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.balance_core_types import (
    ACTION_DIM,
    HIP_PITCH_INDICES,
    KNEE_INDICES,
    SUPPORT_FEEDFORWARD_INDICES,
    zeros_action,
)


class SupportFeedforwardController:
    """Empirical support feedforward torque on hip-pitch and/or knee joints."""

    def __init__(
        self,
        support_vector: Array,
        joint_group: str = "knee",
        scale: float = 1.0,
    ):
        """Initialize support feedforward controller.

        Args:
            support_vector: 10-element empirical support torque vector
            joint_group: Which joints to apply feedforward to.
                Options: "knee", "hip_pitch", "hip_pitch_knee"
            scale: Scaling factor for feedforward torque
        """
        if support_vector.shape != (ACTION_DIM,):
            raise ValueError(
                f"support_vector must be shape ({ACTION_DIM},), got {support_vector.shape}"
            )
        if joint_group not in ["knee", "hip_pitch", "hip_pitch_knee"]:
            raise ValueError(
                f"joint_group must be 'knee', 'hip_pitch', or 'hip_pitch_knee', got {joint_group}"
            )

        self.support_vector = support_vector
        self.joint_group = joint_group
        self.scale = scale

        # Determine which indices to use based on joint_group
        if joint_group == "knee":
            self.active_indices = KNEE_INDICES
        elif joint_group == "hip_pitch":
            self.active_indices = HIP_PITCH_INDICES
        else:  # hip_pitch_knee
            self.active_indices = SUPPORT_FEEDFORWARD_INDICES

    def compute(self) -> tuple[Array, dict]:
        """Compute support feedforward torque and diagnostics.

        Returns:
            tau: Torque output (nonzero only on selected joint group)
            diagnostics: Telemetry including joint_group and norm
        """
        tau = zeros_action()

        # Apply scaled support vector only on active indices (vectorized for JAX compatibility)
        tau = tau.at[self.active_indices].set(
            self.scale * self.support_vector[self.active_indices]
        )

        diagnostics = {
            "support_feedforward_joint_group": self.joint_group,
            "tau_support_feedforward_norm": jnp.linalg.norm(tau),
        }

        return tau, diagnostics
