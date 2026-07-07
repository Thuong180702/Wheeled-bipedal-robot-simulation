"""Phase 3D.3-E — JAX Dynamics Cache.

Precompiles and caches JAX dynamics/Jacobian functions so that
prepare_phase3b_snapshot_cached() does not trace/recompile on every call.

All jax.jit and jax.jacfwd construction happens ONCE in
initialize_jax_dynamics_cache().  The per-step hot path only calls
already-compiled functions with array inputs.

Design:
  - Extract array tuples (fk_arrays, mm_arrays, bias_arrays) once
  - Build jitted functions as closures over those arrays
  - Warm up all functions with dummy calls
  - Expose prepare_phase3b_snapshot_cached() as drop-in replacement
  - Keep Python contact parsing outside JIT
  - Use fixed-shape padded contact arrays (max_contacts=4)

All functions are offline only. No realtime integration.
No controller coupling. No torque injection.
"""

from __future__ import annotations

from typing import Any, Callable
from dataclasses import dataclass, field
import time
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

# ── Constants ───────────────────────────────────────────────────────────

DEFAULT_MAX_CONTACTS = 4
CACHE_VERSION = "phase3d3e_jax_dynamics_cache_v1"


# ═══════════════════════════════════════════════════════════════════════════
# JAXDynamicsCache
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class JAXDynamicsCache:
    """Precompiled JAX dynamics and Jacobian functions.

    All jax.jit / jax.jacfwd construction happens once during
    ``initialize_jax_dynamics_cache()``.  The per-step hot path
    only calls already-compiled functions.
    """

    # ── Pre-extracted array constants ──────────────────────────────────
    fk_arrays: tuple = field(default_factory=tuple)
    mm_arrays: tuple = field(default_factory=tuple)
    bias_arrays: tuple = field(default_factory=tuple)

    # Model constants (non-JAX — used for body_mass, body_ipos lookups)
    body_mass: np.ndarray | None = None
    body_ipos: np.ndarray | None = None
    torso_body_id: int = 1
    nv: int = 16
    nu: int = 10
    nq: int = 17
    max_contacts: int = DEFAULT_MAX_CONTACTS
    dtype_str: str = "float64"

    # ── Jitted functions (set during initialize) ───────────────────────
    mass_matrix_jit: Callable | None = None
    bias_forces_jit: Callable | None = None
    contact_jacobian_batch_jit: Callable | None = None
    com_jacobian_jit: Callable | None = None
    com_jdot_qdot_jit: Callable | None = None
    torso_ang_vel_jacobian_jit: Callable | None = None
    torso_jdotw_qdot_jit: Callable | None = None
    torso_orientation_error_jit: Callable | None = None
    contact_jdot_qdot_batch_jit: Callable | None = None

    # ── Diagnostics ────────────────────────────────────────────────────
    compile_time_s: float = 0.0
    warmup_time_s: float = 0.0
    call_count: int = 0
    recompile_count: int = 0
    fallback_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    initialized: bool = False

    # ── Environment ────────────────────────────────────────────────────
    jax_platform: str = ""
    jax_backend: str = ""
    jax_enable_x64: bool = False
    device_count: int = 0
    device_kind: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# FK-array variants of COM/torso functions (JIT-compatible)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_com_fk_arrays(
    qpos: Array,
    fk_arrays: tuple,
    body_mass_jax: Array,
    body_ipos_jax: Array,
) -> Array:
    """Compute COM position (3,) from qpos using FK arrays only.

    JIT-compatible: all arguments are JAX arrays or array tuples.
    """
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics_fk_arrays
    from wheeled_biped.dynamics.jax_com import jax_compute_com

    fk = jax_forward_kinematics_fk_arrays(qpos, fk_arrays)
    return jax_compute_com(
        fk["body_pos_world"],
        fk["body_quat_world"],
        body_ipos_jax,
        body_mass_jax,
    )


def _get_torso_quat_fk_arrays(
    qpos: Array,
    fk_arrays: tuple,
    torso_body_id: int,
) -> Array:
    """Return torso body quaternion (4,) from FK arrays only.

    JIT-compatible: all arguments are JAX arrays or array tuples.
    """
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics_fk_arrays
    fk = jax_forward_kinematics_fk_arrays(qpos, fk_arrays)
    return fk["body_quat_world"][torso_body_id]


def _torso_orientation_error_jax(
    qpos: Array,
    fk_arrays: tuple,
    torso_body_id: int,
    roll_target: float,
    pitch_target: float,
) -> dict:
    """Compute torso orientation error using only JAX operations.

    JIT-compatible: all arguments are JAX arrays/scalars.
    Returns dict with e_R (3,), R_torso (3,3), R_target (3,3), current_rpy (3,).
    """
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics_fk_arrays

    fk = jax_forward_kinematics_fk_arrays(qpos, fk_arrays)
    torso_quat = fk["body_quat_world"][torso_body_id]  # (w,x,y,z)

    # Quaternion to rotation matrix (JAX)
    w, x, y, z = torso_quat[0], torso_quat[1], torso_quat[2], torso_quat[3]
    R_torso = jnp.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])

    # Current RPY from rotation matrix
    roll = jnp.arctan2(R_torso[2, 1], R_torso[2, 2])
    pitch = jnp.arctan2(-R_torso[2, 0], jnp.sqrt(R_torso[2, 1]**2 + R_torso[2, 2]**2))
    yaw = jnp.arctan2(R_torso[1, 0], R_torso[0, 0])

    # Build target rotation: yaw-preserving upright
    cr = jnp.cos(roll_target)
    sr = jnp.sin(roll_target)
    cp = jnp.cos(pitch_target)
    sp = jnp.sin(pitch_target)
    cy = jnp.cos(yaw)
    sy = jnp.sin(yaw)

    R_target = jnp.array([
        [cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
        [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
        [-sp,   sr*cp,            cr*cp],
    ])

    # Orientation error: log_SO3(R_target^T @ R_torso)
    R_err = R_target.T @ R_torso
    cos_theta = jnp.clip((jnp.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)

    # log_SO3 with small-angle safety
    skew = jnp.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])

    small_angle = jnp.abs(theta) < 1e-10
    coef = jnp.where(small_angle, 0.5, theta / (2.0 * jnp.sin(theta)))
    e_R = coef * skew

    return {
        "e_R": e_R,
        "R_torso": R_torso,
        "R_target": R_target,
        "current_rpy": jnp.array([roll, pitch, yaw]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════════════════

def initialize_jax_dynamics_cache(
    model,
    constants: dict[str, Any],
    *,
    max_contacts: int = DEFAULT_MAX_CONTACTS,
    dtype: str = "float64",
    warmup: bool = True,
) -> JAXDynamicsCache:
    """Build and warm up all JAX dynamics/Jacobian functions once.

    Args:
        model: CPU MuJoCo MjModel instance.
        constants: dict from ``build_qp_wbc_constants`` (must have
                   _mass_matrix_constants, _dynamics_constants,
                   _contact_constants, _kinematics_constants).
        max_contacts: maximum contact count for padding.
        dtype: output dtype for snapshot arrays ("float64" or "float32").
        warmup: if True, run a dummy call through all jitted functions.

    Returns:
        JAXDynamicsCache with all functions precompiled and diagnostics populated.
    """
    t0 = time.perf_counter()

    cache = JAXDynamicsCache(max_contacts=max_contacts, dtype_str=dtype)

    # ── Record environment ───────────────────────────────────────────
    try:
        cache.jax_platform = str(jax.default_backend())
        try:
            from jax.extend.backend import get_backend
            cache.jax_backend = str(get_backend().platform)
        except ImportError:
            cache.jax_backend = str(jax.lib.xla_bridge.get_backend().platform)
        cache.jax_enable_x64 = bool(jax.config.read("jax_enable_x64"))
        cache.device_count = jax.device_count()
        cache.device_kind = str(jax.devices()[0].device_kind) if jax.device_count() > 0 else "none"
    except Exception:
        pass

    # ── Ensure constants are ready ───────────────────────────────────
    from wheeled_biped.wbc.offline_qp_wbc import (
        _ensure_dynamics_constants, _ensure_contact_constants,
    )
    _ensure_dynamics_constants(constants)
    _ensure_contact_constants(constants)

    from wheeled_biped.wbc.offline_task_stack import _ensure_kinematics_constants_for_tasks
    _ensure_kinematics_constants_for_tasks(constants)

    # ── Extract array tuples (once) ──────────────────────────────────
    from wheeled_biped.dynamics.jax_kinematics import extract_jax_fk_arrays
    from wheeled_biped.dynamics.jax_mass_matrix import extract_jax_mm_arrays
    from wheeled_biped.dynamics.jax_bias_forces import extract_jax_bias_arrays

    mass_c = constants["_mass_matrix_constants"]
    bias_c = constants["_dynamics_constants"]

    cache.fk_arrays = extract_jax_fk_arrays(mass_c)
    cache.mm_arrays = extract_jax_mm_arrays(mass_c)[1:]  # skip fk_arrays element
    cache.bias_arrays = extract_jax_bias_arrays(bias_c)[1:]  # skip fk_arrays element

    # Extract body mass and ipos for COM computations
    cache.body_mass = np.array(mass_c.get("body_mass", np.ones(1)), dtype=np.float32)
    cache.body_ipos = np.array(mass_c.get("body_ipos", np.zeros((1, 3))), dtype=np.float32)

    # Extract torso body ID
    from wheeled_biped.wbc.offline_task_stack import _get_torso_body_id
    cache.torso_body_id = _get_torso_body_id(constants.get("_kinematics_constants", mass_c))

    # ── Build JIT functions as closures over extracted arrays ────────
    # Each closure captures the array tuples so JIT sees stable array args.

    fk_a = cache.fk_arrays
    mm_a = cache.mm_arrays
    bias_a = cache.bias_arrays
    bm_jax = jnp.array(cache.body_mass)
    bipos_jax = jnp.array(cache.body_ipos)
    torso_id = cache.torso_body_id

    # Mass matrix
    from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix_fk_arrays

    @functools.partial(jax.jit, static_argnums=())
    def _mass_matrix_jit(qpos_arr):
        return jax_mass_matrix_fk_arrays(qpos_arr, fk_a, mm_a)

    cache.mass_matrix_jit = _mass_matrix_jit

    # Bias forces
    from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces_fk_arrays

    @functools.partial(jax.jit, static_argnums=())
    def _bias_forces_jit(qpos_arr, qvel_arr):
        return jax_bias_forces_fk_arrays(qpos_arr, qvel_arr, fk_a, bias_a)

    cache.bias_forces_jit = _bias_forces_jit

    # COM Jacobian = jacfwd of COM position (constructed ONCE outside jit)
    _com_jac_fn = jax.jacfwd(_compute_com_fk_arrays, argnums=0)
    cache.com_jacobian_jit = jax.jit(
        lambda qpos_arr: _com_jac_fn(qpos_arr, fk_a, bm_jax, bipos_jax)
    )

    # COM Jdot*qdot via FD — placeholder (Stage E3)
    cache.com_jdot_qdot_jit = None

    # Torso angular velocity Jacobian via jacfwd (constructed ONCE here)
    _torso_quat_jac_fn = jax.jacfwd(_get_torso_quat_fk_arrays, argnums=0)
    cache._torso_quat_jac_fn = _torso_quat_jac_fn  # store for Jdot*qdot reuse
    cache.torso_ang_vel_jacobian_jit = jax.jit(
        lambda qpos_arr: _torso_quat_jac_fn(qpos_arr, fk_a, torso_id)
    )

    # Torso Jdotw*qdot via FD — placeholder (Stage E3)
    cache.torso_jdotw_qdot_jit = None

    # Torso orientation error
    cache.torso_orientation_error_jit = jax.jit(
        lambda qpos_arr: _torso_orientation_error_jax(
            qpos_arr, fk_a, torso_id, 0.0, 0.0,
        )
    )

    # Contact Jacobian batch — placeholder (Stage E4)
    cache.contact_jacobian_batch_jit = None

    # Contact Jdot*qdot batch — placeholder (Stage E4)
    cache.contact_jdot_qdot_batch_jit = None

    compile_time = time.perf_counter() - t0
    cache.compile_time_s = compile_time

    # ── Warmup ───────────────────────────────────────────────────────
    if warmup:
        t_warm = time.perf_counter()
        _warmup_cache(cache)
        cache.warmup_time_s = time.perf_counter() - t_warm

    cache.initialized = True
    return cache


def _warmup_cache(cache: JAXDynamicsCache) -> None:
    """Run one dummy call through each jitted function to trigger compilation."""
    dummy_qpos = jnp.zeros(cache.nq, dtype=jnp.float32)
    dummy_qvel = jnp.zeros(cache.nv, dtype=jnp.float32)

    if cache.mass_matrix_jit is not None:
        _ = cache.mass_matrix_jit(dummy_qpos)

    if cache.bias_forces_jit is not None:
        _ = cache.bias_forces_jit(dummy_qpos, dummy_qvel)

    if cache.com_jacobian_jit is not None:
        _ = cache.com_jacobian_jit(dummy_qpos)

    if cache.torso_ang_vel_jacobian_jit is not None:
        _ = cache.torso_ang_vel_jacobian_jit(dummy_qpos)

    if cache.torso_orientation_error_jit is not None:
        _ = cache.torso_orientation_error_jit(dummy_qpos)

    # Contact and Jdot functions warmed up when implemented

    # Force JAX to finish async dispatch
    _ = jax.block_until_ready(dummy_qpos)
