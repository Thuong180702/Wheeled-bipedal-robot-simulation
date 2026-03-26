"""
Push Disturbance helper — reusable across tasks.

Wraps the ``apply_external_force`` / ``clear_external_force`` primitives from
``sim.domain_randomization`` with a periodic-window dispatch that decides
*when* to push based on ``step_count``.  The dispatch logic is JAX-jittable
and vmappable, making it drop-in for any env that wants push recovery training.

Typical usage inside an env's ``step()``::

    from wheeled_biped.sim.push_disturbance import apply_push_disturbance

    mjx_data, new_push_rng = apply_push_disturbance(
        mjx_data,
        push_rng,
        body_id=self._torso_id,
        step_count=state.step_count,
        push_interval=self._push_interval,
        push_duration=self._push_duration,
        push_magnitude=self._push_magnitude,
        push_enabled=self._push_enabled,
    )
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx

from wheeled_biped.sim.domain_randomization import (
    apply_external_force,
    clear_external_force,
)


def apply_push_disturbance(
    mjx_data: mjx.Data,
    rng: jax.Array,
    body_id: int,
    step_count: jnp.ndarray,
    *,
    push_interval: int = 200,
    push_duration: int = 5,
    push_magnitude: float = 20.0,
    push_enabled: bool = True,
) -> tuple[mjx.Data, jax.Array]:
    """Apply a periodic random push disturbance to a body.

    The push is applied for ``push_duration`` consecutive steps every
    ``push_interval`` steps.  Outside the window the external force is cleared
    so it does not accumulate.

    This is a pure function (no side-effects).  Safe inside ``jax.jit`` and
    ``jax.vmap``.

    Args:
        mjx_data: Current MJX simulation data.
        rng: JAX random key used to sample push direction and magnitude.
        body_id: MuJoCo body index to push (e.g. torso).
        step_count: Current episode step (jnp scalar int32).
        push_interval: Steps between push windows.
        push_duration: Number of steps the push is held within each window.
        push_magnitude: Peak force magnitude in Newtons.
        push_enabled: If False the function is a no-op (returns data unchanged
            with a refreshed rng).

    Returns:
        Tuple ``(mjx_data_out, new_rng)`` where ``new_rng`` is a split key
        so callers can continue using the stream.
    """
    rng, push_rng = jax.random.split(rng)

    # Determine whether we are inside a push window
    is_push_active: jnp.ndarray = (step_count % push_interval) < push_duration

    mjx_data_pushed, _ = apply_external_force(
        mjx_data,
        push_rng,
        body_id=body_id,
        magnitude=push_magnitude,
    )
    mjx_data_cleared = clear_external_force(mjx_data)

    # Use jax.lax.cond so the dispatch is traceable inside jit/vmap
    mjx_data_out = jax.lax.cond(
        push_enabled & is_push_active,
        lambda: mjx_data_pushed,
        lambda: mjx_data_cleared,
    )

    return mjx_data_out, rng
