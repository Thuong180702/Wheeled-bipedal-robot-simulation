"""
Domain Randomization (DR) cho sim-to-real transfer.

Ngẫu nhiên hóa các thông số vật lý trong MuJoCo model để policy
học được sự bất định, tăng khả năng chuyển sang robot thực.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


def randomize_model(
    mj_model: mujoco.MjModel,
    rng: jax.Array,
    config: dict[str, Any],
) -> tuple[mujoco.MjModel, jax.Array]:
    """Tạo bản sao model với thông số ngẫu nhiên hóa.

    Thay đổi: khối lượng, ma sát, damping, armature.

    Args:
        mj_model: MuJoCo model gốc.
        rng: JAX random key.
        config: dict cấu hình DR từ config file.

    Returns:
        Tuple (model đã randomize, rng mới).
    """
    import numpy as np

    # Tạo bản sao
    model = mj_model.__copy__()

    mass_range = config.get("mass_range", [1.0, 1.0])
    friction_range = config.get("friction_range", [1.0, 1.0])
    damping_range = config.get("joint_damping_range", [1.0, 1.0])

    rng, k1, k2, k3 = jax.random.split(rng, 4)

    # Ngẫu nhiên khối lượng
    n_bodies = model.nbody
    mass_scales = np.array(
        jax.random.uniform(k1, shape=(n_bodies,), minval=mass_range[0], maxval=mass_range[1])
    )
    model.body_mass[:] = mj_model.body_mass * mass_scales

    # Ngẫu nhiên ma sát (chỉ geom có tiếp xúc)
    n_geoms = model.ngeom
    friction_scales = np.array(
        jax.random.uniform(
            k2, shape=(n_geoms, 1), minval=friction_range[0], maxval=friction_range[1]
        )
    )
    model.geom_friction[:] = mj_model.geom_friction * friction_scales

    # Ngẫu nhiên damping khớp
    n_joints = model.njnt
    damping_scales = np.array(
        jax.random.uniform(k3, shape=(n_joints,), minval=damping_range[0], maxval=damping_range[1])
    )
    model.dof_damping[:n_joints] = mj_model.dof_damping[:n_joints] * damping_scales

    return model, rng


def randomize_mjx_model(
    mjx_model: mjx.Model,
    rng: jax.Array,
    config: dict[str, Any],
) -> tuple[mjx.Model, jax.Array]:
    """Ngẫu nhiên hóa MJX model trực tiếp trên GPU.

    Phiên bản GPU-friendly, hoạt động trong jax.jit.

    Args:
        mjx_model: MJX model (trên GPU).
        rng: JAX random key.
        config: cấu hình DR.

    Returns:
        Tuple (mjx_model randomized, rng mới).
    """
    mass_range = config.get("mass_range", [1.0, 1.0])
    friction_range = config.get("friction_range", [1.0, 1.0])
    damping_range = config.get("joint_damping_range", [1.0, 1.0])

    rng, k1, k2, k3 = jax.random.split(rng, 4)

    # Ngẫu nhiên khối lượng
    mass_scale = jax.random.uniform(
        k1,
        shape=mjx_model.body_mass.shape,
        minval=mass_range[0],
        maxval=mass_range[1],
    )
    new_mass = mjx_model.body_mass * mass_scale

    # Ngẫu nhiên ma sát
    friction_scale = jax.random.uniform(
        k2,
        shape=(mjx_model.geom_friction.shape[0], 1),
        minval=friction_range[0],
        maxval=friction_range[1],
    )
    new_friction = mjx_model.geom_friction * friction_scale

    # Ngẫu nhiên damping
    damping_scale = jax.random.uniform(
        k3,
        shape=mjx_model.dof_damping.shape,
        minval=damping_range[0],
        maxval=damping_range[1],
    )
    new_damping = mjx_model.dof_damping * damping_scale

    mjx_model = mjx_model.replace(
        body_mass=new_mass,
        geom_friction=new_friction,
        dof_damping=new_damping,
    )

    return mjx_model, rng


def apply_external_force(
    mjx_data: mjx.Data,
    rng: jax.Array,
    body_id: int = 1,
    magnitude: float = 30.0,
) -> tuple[mjx.Data, jax.Array]:
    """Áp dụng lực đẩy ngẫu nhiên lên thân robot.

    Dùng để kiểm tra khả năng phục hồi cân bằng.

    Args:
        mjx_data: MJX data.
        rng: random key.
        body_id: ID body cần đẩy (1 = torso thường).
        magnitude: cường độ lực (N).

    Returns:
        Tuple (mjx_data với lực, rng mới).
    """
    rng, k1, k2 = jax.random.split(rng, 3)

    # Hướng ngẫu nhiên trong mặt phẳng xy
    angle = jax.random.uniform(k1, minval=0.0, maxval=2 * jnp.pi)
    force_scale = jax.random.uniform(k2, minval=0.5, maxval=1.0) * magnitude

    fx = force_scale * jnp.cos(angle)
    fy = force_scale * jnp.sin(angle)
    fz = 0.0

    # xfrc_applied shape: (nbody, 6) - [fx, fy, fz, tx, ty, tz]
    new_xfrc = mjx_data.xfrc_applied.at[body_id, :3].set(jnp.array([fx, fy, fz]))
    mjx_data = mjx_data.replace(xfrc_applied=new_xfrc)

    return mjx_data, rng


def clear_external_force(mjx_data: mjx.Data) -> mjx.Data:
    """Xóa tất cả lực ngoài."""
    return mjx_data.replace(xfrc_applied=jnp.zeros_like(mjx_data.xfrc_applied))
