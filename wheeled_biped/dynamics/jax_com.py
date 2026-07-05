"""JAX-compatible COM (centre of mass) computation for the K2 robot.

Uses body world positions from ``jax_forward_kinematics``, body inertial
offsets (``body_ipos``), and body masses to compute the whole-robot COM.

All functions use only JAX operations and are ``jax.jit``-compatible.

MuJoCo convention:
    COM_world = sum(m_b * (xpos_b + R_b @ body_ipos_b)) / sum(m_b)
where R_b is the world-frame rotation matrix of body b and body_ipos_b is
the inertial-frame origin in the body-local frame.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from wheeled_biped.dynamics.jax_kinematics import _quat_rotate


def jax_compute_body_com_positions(
    body_pos_world: Array,
    body_quat_world: Array,
    body_ipos: Array,
) -> Array:
    """Compute the COM position of each body in world frame.

    Args:
        body_pos_world: shape (nbody, 3) — body origin positions.
        body_quat_world: shape (nbody, 4) — body quaternions (w,x,y,z).
        body_ipos: shape (nbody, 3) — inertial-position offsets in body frame.

    Returns:
        shape (nbody, 3) — world-frame COM position of each body.
    """
    nbody = body_pos_world.shape[0]
    com_positions = jnp.zeros((nbody, 3), dtype=jnp.float32)
    for b in range(nbody):
        com_positions = com_positions.at[b].set(
            body_pos_world[b] + _quat_rotate(body_quat_world[b], body_ipos[b])
        )
    return com_positions


def jax_compute_com(
    body_pos_world: Array,
    body_quat_world: Array,
    body_ipos: Array,
    body_mass: Array,
) -> Array:
    """Compute whole-robot COM from body kinematics and masses.

    Uses body COM positions (body origin + inertial offset rotated to world
    frame), weighted by body masses.

    Args:
        body_pos_world: shape (nbody, 3).
        body_quat_world: shape (nbody, 4) — (w,x,y,z).
        body_ipos: shape (nbody, 3) — inertial position in body-local frame.
        body_mass: shape (nbody,).

    Returns:
        shape (3,) — COM position in world frame.
    """
    body_com = jax_compute_body_com_positions(body_pos_world, body_quat_world, body_ipos)
    total_mass = jnp.sum(body_mass)
    weighted_sum = jnp.sum(body_mass[:, jnp.newaxis] * body_com, axis=0)
    com = jnp.where(total_mass > 0, weighted_sum / total_mass, jnp.zeros(3, dtype=jnp.float32))
    return com


def jax_compute_subtree_or_total_com(
    qpos: Array,
    constants: dict[str, Any],
) -> dict[str, Array]:
    """Run JAX FK then compute total COM.

    Convenience wrapper combining FK + COM into one callable.

    Args:
        qpos: shape (nq,).
        constants: dict from ``build_kinematic_tree_constants``.

    Returns:
        dict with keys:
            body_pos_world: shape (nbody, 3).
            body_quat_world: shape (nbody, 4).
            com: shape (3,).
            total_mass: float scalar.
    """
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics

    fk = jax_forward_kinematics(qpos, constants)
    com = jax_compute_com(
        fk["body_pos_world"],
        fk["body_quat_world"],
        constants["body_ipos"],
        constants["body_mass"],
    )
    total_mass = jnp.sum(constants["body_mass"])

    return {
        "body_pos_world": fk["body_pos_world"],
        "body_quat_world": fk["body_quat_world"],
        "com": com,
        "total_mass": total_mass,
    }
