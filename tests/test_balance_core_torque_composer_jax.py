"""JAX compatibility test for balance core torque composer."""

import jax
import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.balance_core_types import ACTION_DIM
from wheeled_biped.controllers.balance_core_torque_composer import (
    BalanceCoreTorqueComposer,
)


def test_jax_jit_compatibility():
    """Test that composer works with JAX JIT compilation."""
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.array([10.0] * ACTION_DIM),
        max_torque_rate=jnp.array([100.0] * ACTION_DIM),
        control_dt=0.01,
    )

    # Create a JIT-compiled wrapper
    @jax.jit
    def jit_compose(tau_shape, tau_support, tau_sagittal, tau_lateral, tau_prev):
        result = composer.compose(
            tau_shape_posture=tau_shape,
            tau_support_feedforward=tau_support,
            tau_sagittal_wheel_balance=tau_sagittal,
            tau_lateral_roll_balance=tau_lateral,
            tau_prev=tau_prev,
            validate_ownership=False,  # Disable validation for JAX compatibility
        )
        return result.tau_final, result.tau_total_raw

    # Test inputs
    tau_shape_posture = jnp.array([0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0])
    tau_support_feedforward = jnp.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.5, 2.0, 0.0])
    tau_sagittal_wheel_balance = jnp.array([0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 6.0])
    tau_lateral_roll_balance = jnp.array([7.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0])
    tau_prev = jnp.zeros(ACTION_DIM)

    # This should not raise an error
    tau_final, tau_raw = jit_compose(
        tau_shape_posture,
        tau_support_feedforward,
        tau_sagittal_wheel_balance,
        tau_lateral_roll_balance,
        tau_prev,
    )

    # Verify outputs are JAX arrays
    assert isinstance(tau_final, jnp.ndarray)
    assert isinstance(tau_raw, jnp.ndarray)
    assert tau_final.shape == (ACTION_DIM,)
    assert tau_raw.shape == (ACTION_DIM,)


def test_vectorized_batch_processing():
    """Test that composer can handle batched inputs via vmap."""
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.array([10.0] * ACTION_DIM),
        max_torque_rate=jnp.array([100.0] * ACTION_DIM),
        control_dt=0.01,
    )

    def compose_wrapper(tau_shape, tau_support, tau_sagittal, tau_lateral, tau_prev):
        result = composer.compose(
            tau_shape_posture=tau_shape,
            tau_support_feedforward=tau_support,
            tau_sagittal_wheel_balance=tau_sagittal,
            tau_lateral_roll_balance=tau_lateral,
            tau_prev=tau_prev,
            validate_ownership=False,  # Disable validation for JAX compatibility
        )
        return result.tau_final

    # Create batch of 4 inputs
    batch_size = 4
    tau_shape_batch = jnp.zeros((batch_size, ACTION_DIM))
    tau_support_batch = jnp.zeros((batch_size, ACTION_DIM))
    tau_sagittal_batch = jnp.zeros((batch_size, ACTION_DIM))
    tau_lateral_batch = jnp.zeros((batch_size, ACTION_DIM))
    tau_prev_batch = jnp.zeros((batch_size, ACTION_DIM))

    # Apply vmap
    batched_compose = jax.vmap(compose_wrapper)

    # This should not raise an error
    tau_final_batch = batched_compose(
        tau_shape_batch,
        tau_support_batch,
        tau_sagittal_batch,
        tau_lateral_batch,
        tau_prev_batch,
    )

    # Verify output shape
    assert tau_final_batch.shape == (batch_size, ACTION_DIM)
