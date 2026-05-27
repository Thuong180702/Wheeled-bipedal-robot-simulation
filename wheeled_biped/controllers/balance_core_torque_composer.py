"""Balance-core torque composer with four approved sources and ownership validation."""

from typing import Sequence

import jax.numpy as jnp
import numpy as np

from wheeled_biped.controllers.balance_core_types import (
    ACTION_DIM,
    BalanceCoreTorqueResult,
    HIP_PITCH_KNEE_INDICES,
    HIP_ROLL_INDICES,
    SUPPORT_SHAPE_INDICES,
    WHEEL_INDICES,
)
from wheeled_biped.controllers.torque_ownership_validator import (
    TorqueOwnershipValidator,
    TorqueSourceEntry,
)


class BalanceCoreTorqueComposer:
    """Composes four approved torque sources with clipping, rate limiting, and ownership validation."""

    def __init__(
        self,
        torque_limit: jnp.ndarray,
        max_torque_rate: jnp.ndarray,
        control_dt: float,
    ):
        """Initialize composer with actuator limits.

        Args:
            torque_limit: Per-joint torque limits [Nm], shape (ACTION_DIM,)
            max_torque_rate: Per-joint max torque rate [Nm/s], shape (ACTION_DIM,)
            control_dt: Control timestep [s]
        """
        if torque_limit.shape != (ACTION_DIM,):
            raise ValueError(f"torque_limit must have shape ({ACTION_DIM},), got {torque_limit.shape}")
        if max_torque_rate.shape != (ACTION_DIM,):
            raise ValueError(f"max_torque_rate must have shape ({ACTION_DIM},), got {max_torque_rate.shape}")
        if control_dt <= 0:
            raise ValueError(f"control_dt must be positive, got {control_dt}")

        self.torque_limit = torque_limit
        self.max_torque_rate = max_torque_rate
        self.control_dt = control_dt
        self.validator = TorqueOwnershipValidator()

    def compose(
        self,
        tau_shape_posture: jnp.ndarray,
        tau_support_feedforward: jnp.ndarray,
        tau_sagittal_wheel_balance: jnp.ndarray,
        tau_lateral_roll_balance: jnp.ndarray,
        tau_prev: jnp.ndarray,
        validate_ownership: bool = True,
    ) -> BalanceCoreTorqueResult:
        """Compose four approved torque sources with clipping and rate limiting.

        Args:
            tau_shape_posture: Shape/posture torque [Nm], shape (ACTION_DIM,)
            tau_support_feedforward: Support feedforward torque [Nm], shape (ACTION_DIM,)
            tau_sagittal_wheel_balance: Sagittal wheel balance torque [Nm], shape (ACTION_DIM,)
            tau_lateral_roll_balance: Lateral roll balance torque [Nm], shape (ACTION_DIM,)
            tau_prev: Previous final torque [Nm], shape (ACTION_DIM,)
            validate_ownership: If True, validate ownership rules (not JAX-compatible).
                Set to False for JIT/vmap usage.

        Returns:
            BalanceCoreTorqueResult with all telemetry fields
        """
        # Validate input shapes
        for name, tau in [
            ("tau_shape_posture", tau_shape_posture),
            ("tau_support_feedforward", tau_support_feedforward),
            ("tau_sagittal_wheel_balance", tau_sagittal_wheel_balance),
            ("tau_lateral_roll_balance", tau_lateral_roll_balance),
            ("tau_prev", tau_prev),
        ]:
            if tau.shape != (ACTION_DIM,):
                raise ValueError(f"{name} must have shape ({ACTION_DIM},), got {tau.shape}")

        # Sum all four sources
        tau_total_raw = (
            tau_shape_posture
            + tau_support_feedforward
            + tau_sagittal_wheel_balance
            + tau_lateral_roll_balance
        )

        # Apply actuator clipping
        tau_total_clipped = jnp.clip(tau_total_raw, -self.torque_limit, self.torque_limit)

        # Apply rate limiting
        # tau_final = tau_prev + clip((tau_total_clipped - tau_prev) / dt, -max_rate, max_rate) * dt
        delta_desired = tau_total_clipped - tau_prev
        delta_rate = delta_desired / self.control_dt
        delta_rate_limited = jnp.clip(delta_rate, -self.max_torque_rate, self.max_torque_rate)
        tau_final = tau_prev + delta_rate_limited * self.control_dt

        # Detect saturation (where clipping changed the value)
        saturation_mask = jnp.abs(tau_total_raw - tau_total_clipped) > 1e-9
        rate_saturation_mask = jnp.abs(delta_rate - delta_rate_limited) > 1e-9

        # Validate ownership (only if requested, not JAX-compatible)
        if validate_ownership:
            # Convert JAX arrays to numpy for validator
            sources = [
                TorqueSourceEntry(
                    name="tau_shape_posture",
                    tau=np.asarray(tau_shape_posture),
                    owned_indices=np.asarray(SUPPORT_SHAPE_INDICES).tolist(),
                ),
                TorqueSourceEntry(
                    name="tau_support_feedforward",
                    tau=np.asarray(tau_support_feedforward),
                    owned_indices=np.asarray(HIP_PITCH_KNEE_INDICES).tolist(),
                ),
                TorqueSourceEntry(
                    name="tau_sagittal_wheel_balance",
                    tau=np.asarray(tau_sagittal_wheel_balance),
                    owned_indices=np.asarray(WHEEL_INDICES).tolist(),
                ),
                TorqueSourceEntry(
                    name="tau_lateral_roll_balance",
                    tau=np.asarray(tau_lateral_roll_balance),
                    owned_indices=np.asarray(HIP_ROLL_INDICES).tolist(),
                ),
            ]

            validation_result = self.validator.validate(sources)
            active_torque_owner_per_joint = validation_result.active_torque_owner_per_joint
            ownership_violation_count = validation_result.ownership_violation_count
            violations = validation_result.violations
        else:
            # Skip validation for JAX compatibility
            active_torque_owner_per_joint = ["not_validated"] * ACTION_DIM
            ownership_violation_count = 0
            violations = []

        return BalanceCoreTorqueResult(
            tau_shape_posture=tau_shape_posture,
            tau_support_feedforward=tau_support_feedforward,
            tau_sagittal_wheel_balance=tau_sagittal_wheel_balance,
            tau_lateral_roll_balance=tau_lateral_roll_balance,
            tau_total_raw=tau_total_raw,
            tau_total_clipped=tau_total_clipped,
            tau_final=tau_final,
            active_torque_owner_per_joint=active_torque_owner_per_joint,
            ownership_violation_count=ownership_violation_count,
            violations=violations,
            saturation_mask=saturation_mask,
            rate_saturation_mask=rate_saturation_mask,
        )
