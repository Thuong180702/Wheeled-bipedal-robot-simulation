"""Momentum coordinator for Level 2 stabilization.

Provides reactive momentum damping, proactive feedforward compensation,
and contact-aware recovery with 20% authority budget.
"""

import chex
import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


@chex.dataclass(frozen=True)
class MomentumCoordinatorConfig:
    """Configuration for momentum coordinator."""
    # Momentum damping
    k_momentum_lateral: float = 0.8
    k_momentum_sagittal: float = 1.2
    k_angular_roll: float = 1.5

    # Feedforward compensation
    k_feedforward: float = 5.0
    k_feedforward_hip: float = 2.0
    height_transition_threshold: float = 0.05  # m/s

    # Contact-aware recovery
    k_contact_recovery: float = 10.0
    k_contact_wheel_diff: float = 4.0
    unloading_threshold: float = 0.3  # 30% force asymmetry

    # Deadbands
    momentum_deadband_linear: float = 0.5  # kg*m/s
    momentum_deadband_angular: float = 0.2  # kg*m^2/s

    # Authority budget
    momentum_authority_budget: float = 0.2  # 20% of actuator range


class MomentumCoordinator:
    """Momentum coordinator for Level 2 stabilization."""

    def __init__(self, config: MomentumCoordinatorConfig):
        """Initialize momentum coordinator.

        Args:
            config: MomentumCoordinatorConfig with gains and thresholds
        """
        self.config = config

    def compute_momentum_damping_torque(self, state: CentroidalState) -> Array:
        """Compute momentum damping torque to prevent oscillation buildup.

        Args:
            state: CentroidalState with linear and angular momentum

        Returns:
            Damping torque array (10,) opposing unwanted momentum
        """
        tau = jnp.zeros(10)

        # Linear momentum damping (lateral and sagittal)
        linear_momentum_mag = jnp.sqrt(
            state.linear_momentum[0]**2 + state.linear_momentum[1]**2
        )

        # JAX-compatible deadband using jnp.where
        linear_active = jnp.where(
            linear_momentum_mag > self.config.momentum_deadband_linear,
            1.0,
            0.0,
        )

        # Lateral momentum → hip roll damping
        lateral_momentum = state.linear_momentum[1]
        tau_lateral = -self.config.k_momentum_lateral * lateral_momentum * linear_active
        tau = tau.at[0].set(tau_lateral)  # left hip roll
        tau = tau.at[5].set(-tau_lateral)  # right hip roll (opposite sign for differential roll moment)

        # Sagittal momentum → wheel damping
        sagittal_momentum = state.linear_momentum[0]
        tau_sagittal = -self.config.k_momentum_sagittal * sagittal_momentum * linear_active
        tau = tau.at[4].set(tau_sagittal)  # left wheel
        tau = tau.at[9].set(tau_sagittal)  # right wheel

        # Angular momentum damping (roll axis most critical)
        angular_momentum_mag = jnp.abs(state.angular_momentum[0])

        angular_active = jnp.where(
            angular_momentum_mag > self.config.momentum_deadband_angular,
            1.0,
            0.0,
        )

        # Roll momentum → differential hip roll
        roll_momentum = state.angular_momentum[0]
        tau_angular_left = -self.config.k_angular_roll * roll_momentum * angular_active
        tau_angular_right = self.config.k_angular_roll * roll_momentum * angular_active

        tau = tau.at[0].add(tau_angular_left)  # left hip roll (add to existing)
        tau = tau.at[5].add(tau_angular_right)  # right hip roll (opposite sign)

        return tau

    def compute_feedforward_compensation_torque(self, obs: Array, state: CentroidalState) -> Array:
        """Compute feedforward compensation for height transitions.

        Args:
            obs: Observation array with height_cmd at index 39
            state: CentroidalState with current height and velocity

        Returns:
            Feedforward torque array (10,) for proactive compensation
        """
        tau = jnp.zeros(10)

        # Extract height command and current state
        height_cmd = obs[39]
        height_current = state.com_pos[2]
        height_vel = state.com_vel[2]

        # Detect height transition
        height_error = height_cmd - height_current
        transition_active = jnp.where(
            jnp.abs(height_vel) > self.config.height_transition_threshold,
            1.0,
            0.0,
        )

        # Feedforward compensation based on height velocity direction
        # Rising (positive vel) → anticipate backward pitch, apply forward wheel torque
        # Squatting (negative vel) → anticipate forward pitch, apply backward wheel torque
        tau_wheel_ff = self.config.k_feedforward * height_vel * transition_active
        tau_hip_ff = -self.config.k_feedforward_hip * height_vel * transition_active

        # Apply to wheels (common mode)
        tau = tau.at[4].set(tau_wheel_ff)  # left wheel
        tau = tau.at[9].set(tau_wheel_ff)  # right wheel

        # Apply to hip pitch (both legs)
        tau = tau.at[2].set(tau_hip_ff)  # left hip pitch
        tau = tau.at[7].set(tau_hip_ff)  # right hip pitch

        return tau

    def compute_contact_aware_recovery_torque(self, state: CentroidalState) -> Array:
        """Compute contact-aware recovery torque for asymmetric support.

        Args:
            state: CentroidalState with wheel contact forces

        Returns:
            Recovery torque array (10,) for contact-based redistribution
        """
        tau = jnp.zeros(10)

        # Compute force imbalance
        total_force = state.left_wheel_force + state.right_wheel_force

        # Avoid division by zero
        total_force_safe = jnp.where(total_force > 1.0, total_force, 1.0)

        force_ratio_left = state.left_wheel_force / total_force_safe
        force_ratio_right = state.right_wheel_force / total_force_safe

        # Detect unloading (force asymmetry exceeds threshold)
        force_imbalance = jnp.abs(force_ratio_left - force_ratio_right)
        unloading_active = jnp.where(
            force_imbalance > self.config.unloading_threshold,
            1.0,
            0.0,
        )

        # Recovery direction: shift toward loaded wheel
        # If left wheel has less force, apply positive hip roll (shift right)
        # If right wheel has less force, apply negative hip roll (shift left)
        recovery_direction = force_ratio_right - force_ratio_left

        # Hip roll recovery (symmetric - both legs same direction)
        tau_hip_roll = self.config.k_contact_recovery * recovery_direction * unloading_active
        tau = tau.at[0].set(tau_hip_roll)  # left hip roll
        tau = tau.at[5].set(tau_hip_roll)  # right hip roll

        # Wheel differential recovery
        tau_wheel_diff = self.config.k_contact_wheel_diff * recovery_direction * unloading_active
        tau = tau.at[4].set(tau_wheel_diff)  # left wheel
        tau = tau.at[9].set(-tau_wheel_diff)  # right wheel (opposite)

        return tau

    def clip_to_authority_budget(self, tau: Array) -> Array:
        """Clip torque to momentum coordinator authority budget.

        Args:
            tau: Desired torque array (10,)

        Returns:
            Clipped torque array (10,) within 20% authority budget
        """
        # Maximum actuator torque (hardcoded as per Phase 2)
        max_actuator_torque = 30.0

        # Compute budget limit
        budget_limit = self.config.momentum_authority_budget * max_actuator_torque

        # Find maximum absolute torque
        max_tau = jnp.max(jnp.abs(tau))

        # JAX-compatible conditional scaling
        scale_factor = jnp.where(max_tau <= budget_limit, 1.0, budget_limit / max_tau)
        tau_clipped = tau * scale_factor

        return tau_clipped
