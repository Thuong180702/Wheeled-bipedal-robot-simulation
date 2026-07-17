# Phase 3D.3-E — JAX Dynamics Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the untraced JAX eager-mode dynamics/Jacobian calls in `prepare_phase3b_snapshot()` with precompiled jitted functions, eliminating ~300s/state JAX tracing overhead on CPU-only Windows.

**Architecture:** Individual JIT-compiled array-only functions (mass matrix, bias forces, COM Jacobian, torso Jacobian, contact Jacobian) pre-built once in `JAXDynamicsCache`, warmed up during initialization, then called from `prepare_phase3b_snapshot_cached()`. Python handles contact parsing/padding outside JIT; JAX only sees fixed-shape arrays.

**Tech Stack:** JAX (jit, jacfwd, vmap), NumPy, existing `_fk_arrays` infrastructure, existing `Phase3BSnapshot` dataclass

## Global Constraints

- Do NOT modify K2 V3 controller (`k2_jax_controller.py`, `sagittal_velocity_damped_balance_controller.py`, controller profiles, controller gains, `configs/controllers/*`)
- Do NOT promote WBC (no `REALTIME_READY`, `PRODUCTION_READY`, `WBC_PROMOTED`, `DEFAULT_CONTROLLER_UPDATED`, `HARDWARE_SAFE`)
- Do NOT JIT the entire `prepare_phase3b_snapshot()` — too many Python dict/list/contact-object boundaries
- No `jax.jit` or `jax.jacfwd` construction inside per-step code — all construction happens once in `initialize_jax_dynamics_cache()`
- No Python contact lists or dicts inside JIT — use fixed-shape padded arrays
- Allowed verdicts: `JAX_DYNAMICS_CACHE_CORRECTNESS_PASS`, `JAX_DYNAMICS_CACHE_CORRECTNESS_FAIL`, `JAX_DYNAMICS_BOTTLENECK_UNRESOLVED`, `JAX_DYNAMICS_PARTIAL_SPEEDUP`, `CLOSED_LOOP_EVALUATION_UNBLOCKED`, `REALTIME_CANDIDATE_INFRASTRUCTURE`, `PARTIAL_JAX_DYNAMICS_CACHE`
- Forbidden verdicts: `REALTIME_READY`, `PRODUCTION_READY`, `WBC_PROMOTED`, `DEFAULT_CONTROLLER_UPDATED`, `HARDWARE_SAFE`
- Default path unchanged without `--use-jax-dynamics-cache` flag
- Compile time and post-warmup runtime reported separately
- Correctness threshold: `max_abs_diff <= 1e-6` for all dynamics/Jacobian and downstream QP matrices
- Original `prepare_phase3b_snapshot()` preserved unchanged
- No GPU assumptions — record `jax_platform`, `jax_backend`, `jax_enable_x64`, `device_count`

---

## File Structure

### Files to CREATE

| File | Responsibility |
|---|---|
| `wheeled_biped/wbc/phase3d3e_jax_dynamics_cache.py` | `JAXDynamicsCache` class, `initialize_jax_dynamics_cache()`, `prepare_phase3b_snapshot_cached()`, all jitted function construction, array-only FK variants for COM/torso |
| `scripts/phase3d3e_jax_dynamics_diagnostic.py` | Measure per-function timing, detect recompilation, separate compile/first-call/warm-call timings |
| `scripts/phase3d3e_jax_dynamics_correctness_audit.py` | Compare original vs cached snapshot across 8 scenarios, including downstream QP matrix equivalence |
| `scripts/phase3d3e_jax_dynamics_benchmark.py` | Benchmark original vs cached snapshot, incremental QP + cached dynamics |
| `tests/test_phase3d3e_jax_dynamics_cache.py` | Unit tests: cache init, warmup, shape stability, recompile guards, fallback, correctness |
| `tests/test_phase3d3e_jax_dynamics_benchmark_schema.py` | Schema validation for diagnostic/benchmark/correctness JSON outputs |

### Files to MODIFY

| File | Change |
|---|---|
| `wheeled_biped/wbc/phase3d3_incremental_qp.py` | Add `jax_dynamics_cache` parameter to `initialize_incremental_qp_workspace()` and `update_incremental_qp_workspace()`; optional cached snapshot path |
| `scripts/phase3d_full_batch_execution.py` | Add `--use-jax-dynamics-cache`, `--jax-dynamics-cache-max-contacts`, `--jax-dynamics-warmup`, `--jax-dynamics-diagnostic` CLI flags |

### Files to CREATE (docs)

| File | Responsibility |
|---|---|
| `docs/validation/k2_phase3d3e_jax_dynamics_report.md` | Final validation report with executive summary, benchmark tables, verdict |

---

## Stage Overview

```
E1: Diagnostic script → measure current bottleneck, prove re-tracing
E2: M + h JIT → jit mass matrix + bias forces via _fk_arrays
E3: COM + torso JIT → jit jacfwd-based Jacobians
E4: Contact JIT → batched contact Jacobian + Jdot*qdot (riskiest stage)
E5: Full cached snapshot → assemble prepare_phase3b_snapshot_cached()
E6: Integration → opt-in --use-jax-dynamics-cache in incremental QP + full-batch
E7: Benchmark + report → end-to-end timing, verdict, validation doc
```

Stop after any stage if correctness fails. Report `PARTIAL_JAX_DYNAMICS_CACHE` if E4 is unsafe.

---

### Task 1: Diagnostic Script — Environment and Baseline Timing

**Files:**
- Create: `scripts/phase3d3e_jax_dynamics_diagnostic.py`

**Interfaces:**
- Produces: JSON file at `outputs/phase3d3e_jax_dynamics/jax_dynamics_diagnostic.json`
- Consumes: `prepare_phase3b_snapshot` from `wheeled_biped.wbc.phase3b_cached_stack`, scenario utilities from `scripts.phase3d_full_batch_execution` (or equivalent scenario-loading code)

- [ ] **Step 1: Write the script skeleton with environment diagnostics**

```python
#!/usr/bin/env python3
"""Phase 3D.3-E1 — JAX Dynamics Diagnostic.

Measures:
  1. JAX environment (platform, backend, device count, x64 status)
  2. Per-function timing inside prepare_phase3b_snapshot()
  3. Whether repeated calls with same-shape inputs recompile
  4. Compile vs first-call vs warm-call timing separation
"""
from __future__ import annotations

import json, time, sys, os
from pathlib import Path
import numpy as np
import jax

# Add repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "outputs" / "phase3d3e_jax_dynamics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def record_jax_environment() -> dict:
    """Record JAX platform/backend/device information."""
    return {
        "jax_version": jax.__version__,
        "jax_platform": str(jax.default_backend()),
        "jax_backend": str(jax.lib.xla_bridge.get_backend().platform),
        "jax_enable_x64": jax.config.read("jax_enable_x64"),
        "device_count": jax.device_count(),
        "devices": [str(d) for d in jax.devices()],
        "device_kind": str(jax.devices()[0].device_kind) if jax.device_count() > 0 else "none",
        "jax_process_index": jax.process_index(),
    }


def load_test_scenario(name: str) -> dict:
    """Load or construct a test scenario (qpos, qvel, contacts, constants).

    For the diagnostic, we use a minimal inline scenario builder
    that does NOT depend on the full batch execution pipeline.
    """
    # Use the same model and constants as the Phase 3D.3 pipeline
    from wheeled_biped.utils.config import get_model_path
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))

    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
    constants = build_qp_wbc_constants(model)

    # Ensure dynamics/contact constants are built
    from wheeled_biped.wbc.offline_qp_wbc import (
        _ensure_dynamics_constants, _ensure_contact_constants,
    )
    _ensure_dynamics_constants(constants)
    _ensure_contact_constants(constants)

    # Ensure kinematics constants
    from wheeled_biped.wbc.offline_task_stack import _ensure_kinematics_constants_for_tasks
    _ensure_kinematics_constants_for_tasks(constants)

    # Default standing qpos
    qpos0 = np.array(model.keyframe("default").qpos, dtype=np.float64)
    qvel0 = np.zeros(model.nv, dtype=np.float64)

    # Extract active contacts by running one simulation step
    import mujoco
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)
    # Inline contact extraction (mirrors phase3d_full_batch_execution.py:368-396)
    contact_constants = constants["_contact_constants"]
    wheel_body_ids = contact_constants.get("wheel_body_ids", {})
    wheel_ids_set = set(int(v) for v in wheel_body_ids.values() if v >= 0)
    contacts = []
    for contact_id in range(data.ncon):
        c = data.contact[contact_id]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        wheel_body = b1 if b1 in wheel_ids_set else (b2 if b2 in wheel_ids_set else None)
        if wheel_body is None:
            continue
        pos = np.array(c.pos, dtype=np.float64)
        frame = np.array(c.frame, dtype=np.float64).reshape(3, 3)
        body_xpos = np.array(data.xpos[wheel_body], dtype=np.float64)
        body_xmat = np.array(data.xmat[wheel_body], dtype=np.float64).reshape(3, 3)
        local_point = body_xmat.T @ (pos - body_xpos)
        contacts.append({
            "body_id": int(wheel_body),
            "position": pos,
            "frame": frame,
            "local_point": local_point,
            "distance": float(c.dist),
        })

    return {
        "name": name,
        "qpos": qpos0,
        "qvel": qvel0,
        "contacts": contacts,
        "constants": constants,
        "model": model,
    }


def time_function_call(fn, *args, **kwargs) -> dict:
    """Time a single function call. Returns {time_s, success, error}."""
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        return {"time_s": elapsed, "success": True, "error": None, "result": result}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"time_s": elapsed, "success": False, "error": str(e), "result": None}


def run_diagnostic():
    """Main diagnostic routine."""
    print("=" * 70)
    print("Phase 3D.3-E1: JAX Dynamics Diagnostic")
    print("=" * 70)

    # ── 1. Environment ──────────────────────────────────────────────────
    print("\n[1/7] Recording JAX environment...")
    env_info = record_jax_environment()
    for k, v in env_info.items():
        print(f"  {k}: {v}")

    # ── 2. Load scenario ────────────────────────────────────────────────
    print("\n[2/7] Loading test scenario...")
    scenario = load_test_scenario("diagnostic_default")
    print(f"  qpos shape: {scenario['qpos'].shape}")
    print(f"  qvel shape: {scenario['qvel'].shape}")
    print(f"  contacts: {len(scenario['contacts'])}")

    # ── 3. Time individual sub-operations ───────────────────────────────
    print("\n[3/7] Timing individual dynamics operations...")

    from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
    from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix
    from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
    import jax.numpy as jnp

    qp_c = scenario["constants"]
    qpos = scenario["qpos"]
    qvel = scenario["qvel"]
    contacts = scenario["contacts"]

    mass_constants = qp_c["_mass_matrix_constants"]
    bias_constants = qp_c["_dynamics_constants"]
    contact_constants = qp_c["_contact_constants"]

    qpos_jax = jnp.array(qpos, dtype=jnp.float32)
    qvel_jax = jnp.array(qvel, dtype=jnp.float32)

    sub_timings = {}

    # Mass matrix
    r = time_function_call(jax_mass_matrix, qpos_jax, mass_constants)
    sub_timings["mass_matrix"] = r
    print(f"  mass_matrix: {r['time_s']:.3f}s {'OK' if r['success'] else 'FAIL: '+str(r['error'])}")

    # Bias forces
    r = time_function_call(jax_bias_forces, qpos_jax, qvel_jax, bias_constants)
    sub_timings["bias_forces"] = r
    print(f"  bias_forces: {r['time_s']:.3f}s {'OK' if r['success'] else 'FAIL: '+str(r['error'])}")

    # Contact Jacobian (per contact)
    from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian
    contact_jac_timings = []
    for i, c in enumerate(contacts):
        bid = int(c["body_id"])
        lp = jnp.array(c["local_point"], dtype=jnp.float32)
        r = time_function_call(contact_point_translational_jacobian, qpos_jax, bid, lp, contact_constants)
        contact_jac_timings.append({"contact_idx": i, **r})
    sub_timings["contact_jacobian_per_contact"] = contact_jac_timings
    total_cj = sum(t["time_s"] for t in contact_jac_timings)
    print(f"  contact_jacobian ({len(contacts)} contacts): {total_cj:.3f}s total")

    # Jdot_qdot
    from wheeled_biped.wbc.offline_qp_wbc import compute_contact_jdot_qdot
    r = time_function_call(compute_contact_jdot_qdot, qpos, qvel, contacts, contact_constants)
    sub_timings["contact_jdot_qdot"] = r
    print(f"  contact_jdot_qdot: {r['time_s']:.3f}s {'OK' if r['success'] else 'FAIL: '+str(r['error'])}")

    # COM Jacobian
    kc = qp_c["_kinematics_constants"]
    from wheeled_biped.wbc.offline_task_stack import compute_com_jacobian
    r = time_function_call(compute_com_jacobian, qpos, kc)
    sub_timings["com_jacobian"] = r
    print(f"  com_jacobian: {r['time_s']:.3f}s {'OK' if r['success'] else 'FAIL: '+str(r['error'])}")

    # COM Jdot_qdot
    from wheeled_biped.wbc.offline_task_stack import compute_com_jdot_qdot
    r = time_function_call(compute_com_jdot_qdot, qpos, qvel, kc)
    sub_timings["com_jdot_qdot"] = r
    print(f"  com_jdot_qdot: {r['time_s']:.3f}s {'OK' if r['success'] else 'FAIL: '+str(r['error'])}")

    # Torso angular velocity Jacobian
    from wheeled_biped.wbc.offline_task_stack import compute_torso_angular_velocity_jacobian
    r = time_function_call(compute_torso_angular_velocity_jacobian, qpos, kc)
    sub_timings["torso_ang_vel_jacobian"] = r
    print(f"  torso_ang_vel_jacobian: {r['time_s']:.3f}s {'OK' if r['success'] else 'FAIL: '+str(r['error'])}")

    # Torso Jdotw_qdot
    from wheeled_biped.wbc.offline_task_stack import compute_torso_jdotw_qdot
    r = time_function_call(compute_torso_jdotw_qdot, qpos, qvel, kc)
    sub_timings["torso_jdotw_qdot"] = r
    print(f"  torso_jdotw_qdot: {r['time_s']:.3f}s {'OK' if r['success'] else 'FAIL: '+str(r['error'])}")

    # Torso orientation error
    from wheeled_biped.wbc.offline_task_stack import compute_torso_orientation_error
    r = time_function_call(compute_torso_orientation_error, qpos, kc)
    sub_timings["torso_orientation_error"] = r
    print(f"  torso_orientation_error: {r['time_s']:.3f}s {'OK' if r['success'] else 'FAIL: '+str(r['error'])}")

    # ── 4. Time full prepare_phase3b_snapshot ────────────────────────────
    print("\n[4/7] Timing full prepare_phase3b_snapshot (original)...")
    r = time_function_call(prepare_phase3b_snapshot, "diag", qpos, qvel, contacts, qp_c)
    sub_timings["full_snapshot"] = r
    print(f"  full_snapshot: {r['time_s']:.3f}s {'OK' if r['success'] else 'FAIL: '+str(r['error'])}")

    # ── 5. Repeated-call test: same state 3 times ──────────────────────
    print("\n[5/7] Repeated-call test: same state 3 times...")
    repeat_timings = []
    for trial in range(3):
        r = time_function_call(prepare_phase3b_snapshot, f"repeat_{trial}", qpos, qvel, contacts, qp_c)
        repeat_timings.append({"trial": trial, **r})
        print(f"  trial {trial}: {r['time_s']:.3f}s {'OK' if r['success'] else 'FAIL'}")

    # ── 6. Same-shape different qpos test ──────────────────────────────
    print("\n[6/7] Same-shape test: perturbed qpos 3 times...")
    rng = np.random.RandomState(42)
    perturbed_timings = []
    for trial in range(3):
        qpos_p = qpos.copy()
        qpos_p[7:17] += rng.randn(10) * 0.01  # small joint perturbation
        r = time_function_call(prepare_phase3b_snapshot, f"perturbed_{trial}", qpos_p, qvel, contacts, qp_c)
        perturbed_timings.append({"trial": trial, **r})
        print(f"  trial {trial}: {r['time_s']:.3f}s {'OK' if r['success'] else 'FAIL'}")

    # ── 7. Assemble and save report ────────────────────────────────────
    print("\n[7/7] Saving diagnostic report...")
    report = {
        "phase": "3D.3-E1",
        "environment": env_info,
        "sub_operation_timings_s": sub_timings,
        "repeated_same_state_timings_s": repeat_timings,
        "perturbed_qpos_timings_s": perturbed_timings,
        "summary": {
            "full_snapshot_first_s": sub_timings["full_snapshot"]["time_s"],
            "repeat_trial_1_s": repeat_timings[0]["time_s"] if len(repeat_timings) > 0 else None,
            "repeat_trial_2_s": repeat_timings[1]["time_s"] if len(repeat_timings) > 1 else None,
            "repeat_trial_3_s": repeat_timings[2]["time_s"] if len(repeat_timings) > 2 else None,
            "mass_matrix_s": sub_timings["mass_matrix"]["time_s"],
            "bias_forces_s": sub_timings["bias_forces"]["time_s"],
            "total_contact_jac_s": total_cj,
            "contact_jdot_qdot_s": sub_timings["contact_jdot_qdot"]["time_s"],
            "com_jacobian_s": sub_timings["com_jacobian"]["time_s"],
            "torso_jacobian_s": sub_timings["torso_ang_vel_jacobian"]["time_s"],
        },
    }

    output_path = OUTPUT_DIR / "jax_dynamics_diagnostic.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDiagnostic report saved to: {output_path}")
    print("Done.")
    return report


if __name__ == "__main__":
    run_diagnostic()
```

- [ ] **Step 2: Run the diagnostic script**

```bash
python scripts/phase3d3e_jax_dynamics_diagnostic.py
```

Expected: Script runs and produces `outputs/phase3d3e_jax_dynamics/jax_dynamics_diagnostic.json`.

- [ ] **Step 3: Verify diagnostic output has required fields**

Check that `jax_dynamics_diagnostic.json` contains: `environment.jax_platform`, `environment.jax_backend`, `sub_operation_timings_s` with all keys, `repeated_same_state_timings_s` with 3 entries, `perturbed_qpos_timings_s` with 3 entries.

- [ ] **Step 4: Commit**

```bash
git add scripts/phase3d3e_jax_dynamics_diagnostic.py
git commit -m "feat(phase3d3e): add JAX dynamics diagnostic script

Measures per-function timing in prepare_phase3b_snapshot(),
repeated-call behavior, and JAX environment info.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: JAXDynamicsCache Skeleton + Environment Diagnostics

**Files:**
- Create: `wheeled_biped/wbc/phase3d3e_jax_dynamics_cache.py` (skeleton)
- Create: `tests/test_phase3d3e_jax_dynamics_cache.py` (initial tests)

**Interfaces:**
- Produces: `JAXDynamicsCache` dataclass, `initialize_jax_dynamics_cache()` function
- Consumes: `extract_jax_fk_arrays` from `wheeled_biped.dynamics.jax_kinematics`, `extract_jax_mm_arrays` from `wheeled_biped.dynamics.jax_mass_matrix`, `extract_jax_bias_arrays` from `wheeled_biped.dynamics.jax_bias_forces`

- [ ] **Step 1: Write the cache skeleton module**

```python
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
    body_mass: Array,
    body_ipos: Array,
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
        body_ipos,
        body_mass,
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
    kc = constants.get("_kinematics_constants", mass_c)
    target_ids = kc.get("target_body_ids", {})
    cache.torso_body_id = int(target_ids.get("torso", target_ids.get("base", target_ids.get("trunk", 1))))

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

    # COM Jacobian = jacfwd of COM position
    _com_jac_fn = jax.jacfwd(_compute_com_fk_arrays, argnums=0)
    cache.com_jacobian_jit = jax.jit(lambda qpos_arr: _com_jac_fn(qpos_arr, fk_a, bm_jax, bipos_jax))

    # COM Jdot*qdot via FD — jacfwd constructed ONCE outside jit (Stage E3)
    cache.com_jdot_qdot_jit = None  # populated in Task 4

    # Torso angular velocity Jacobian via jacfwd (constructed ONCE here)
    _torso_quat_jac_fn = jax.jacfwd(_get_torso_quat_fk_arrays, argnums=0)
    cache._torso_quat_jac_fn = _torso_quat_jac_fn  # store for Jdot*qdot reuse
    cache.torso_ang_vel_jacobian_jit = jax.jit(
        lambda qpos_arr: _torso_quat_jac_fn(qpos_arr, fk_a, torso_id)
    )

    # Torso Jdotw*qdot via FD — jacfwd constructed ONCE outside jit (Stage E3)
    cache.torso_jdotw_qdot_jit = None  # populated in Task 4

    # Torso orientation error
    cache.torso_orientation_error_jit = jax.jit(
        lambda qpos_arr: _torso_orientation_error_jax(qpos_arr, fk_a, torso_id, 0.0, 0.0)
    )

    # Contact Jacobian batch (Stage E4 will implement; placeholder for now)
    cache.contact_jacobian_batch_jit = None  # populated in Task 5

    # Contact Jdot*qdot batch (Stage E4 will implement; placeholder for now)
    cache.contact_jdot_qdot_batch_jit = None  # populated in Task 5

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
```

- [ ] **Step 2: Write initial tests for cache initialization**

```python
"""Tests for Phase 3D.3-E JAX Dynamics Cache."""
import pytest
import numpy as np
import jax

from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import (
    JAXDynamicsCache,
    initialize_jax_dynamics_cache,
    DEFAULT_MAX_CONTACTS,
)


@pytest.fixture(scope="module")
def test_model_and_constants():
    """Load model and build constants once for all cache tests."""
    from wheeled_biped.utils.config import get_model_path
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))

    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
    constants = build_qp_wbc_constants(model)

    return model, constants


class TestJAXDynamicsCacheInit:
    """Tests for cache initialization."""

    def test_cache_initializes(self, test_model_and_constants):
        model, constants = test_model_and_constants
        cache = initialize_jax_dynamics_cache(model, constants, warmup=False)
        assert cache.initialized
        assert cache.max_contacts == DEFAULT_MAX_CONTACTS

    def test_cache_warmup_records_compile_time(self, test_model_and_constants):
        model, constants = test_model_and_constants
        cache = initialize_jax_dynamics_cache(model, constants, warmup=True)
        assert cache.compile_time_s > 0
        assert cache.warmup_time_s >= 0

    def test_cache_records_environment(self, test_model_and_constants):
        model, constants = test_model_and_constants
        cache = initialize_jax_dynamics_cache(model, constants, warmup=False)
        assert cache.jax_platform != ""
        assert cache.jax_backend != ""

    def test_mass_matrix_jit_compiled(self, test_model_and_constants):
        model, constants = test_model_and_constants
        cache = initialize_jax_dynamics_cache(model, constants, warmup=True)
        assert cache.mass_matrix_jit is not None

        import jax.numpy as jnp
        qpos = jnp.zeros(17, dtype=jnp.float32)
        M = cache.mass_matrix_jit(qpos)
        assert M.shape == (16, 16)
        assert np.all(np.isfinite(np.array(M)))

    def test_bias_forces_jit_compiled(self, test_model_and_constants):
        model, constants = test_model_and_constants
        cache = initialize_jax_dynamics_cache(model, constants, warmup=True)
        assert cache.bias_forces_jit is not None

        import jax.numpy as jnp
        qpos = jnp.zeros(17, dtype=jnp.float32)
        qvel = jnp.zeros(16, dtype=jnp.float32)
        h = cache.bias_forces_jit(qpos, qvel)
        assert h.shape == (16,)
        assert np.all(np.isfinite(np.array(h)))

    def test_com_jacobian_jit_compiled(self, test_model_and_constants):
        model, constants = test_model_and_constants
        cache = initialize_jax_dynamics_cache(model, constants, warmup=True)
        assert cache.com_jacobian_jit is not None

        import jax.numpy as jnp
        qpos = jnp.zeros(17, dtype=jnp.float32)
        Jcom_qpos = cache.com_jacobian_jit(qpos)
        # Returns qpos-space Jacobian (3, 17)
        assert Jcom_qpos.shape == (3, 17)
        assert np.all(np.isfinite(np.array(Jcom_qpos)))

    def test_torso_ang_vel_jacobian_jit_compiled(self, test_model_and_constants):
        model, constants = test_model_and_constants
        cache = initialize_jax_dynamics_cache(model, constants, warmup=True)
        assert cache.torso_ang_vel_jacobian_jit is not None

        import jax.numpy as jnp
        qpos = jnp.zeros(17, dtype=jnp.float32)
        Jquat_qpos = cache.torso_ang_vel_jacobian_jit(qpos)
        # Returns torso quaternion qpos-space Jacobian (4, 17)
        assert Jquat_qpos.shape == (4, 17)
        assert np.all(np.isfinite(np.array(Jquat_qpos)))

    def test_torso_orientation_error_jit_compiled(self, test_model_and_constants):
        model, constants = test_model_and_constants
        cache = initialize_jax_dynamics_cache(model, constants, warmup=True)
        assert cache.torso_orientation_error_jit is not None

        import jax.numpy as jnp
        qpos = jnp.zeros(17, dtype=jnp.float32)
        result = cache.torso_orientation_error_jit(qpos)
        assert "e_R" in result
        assert result["e_R"].shape == (3,)
        assert result["current_rpy"].shape == (3,)

    def test_fk_arrays_are_tuples(self, test_model_and_constants):
        model, constants = test_model_and_constants
        cache = initialize_jax_dynamics_cache(model, constants, warmup=False)
        assert isinstance(cache.fk_arrays, tuple)
        assert isinstance(cache.mm_arrays, tuple)
        assert isinstance(cache.bias_arrays, tuple)
        assert len(cache.fk_arrays) > 0
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_phase3d3e_jax_dynamics_cache.py -v
```

Expected: All 8 tests pass.

- [ ] **Step 4: Commit**

```bash
git add wheeled_biped/wbc/phase3d3e_jax_dynamics_cache.py tests/test_phase3d3e_jax_dynamics_cache.py
git commit -m "feat(phase3d3e): add JAXDynamicsCache skeleton with M, h, COM, torso JIT

Initializes jitted mass_matrix, bias_forces, com_jacobian,
torso_ang_vel_jacobian, and torso_orientation_error functions.
Contact and Jdot*qdot functions are placeholders for later stages.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Mass Matrix + Bias Force Correctness Audit

**Files:**
- Modify: `tests/test_phase3d3e_jax_dynamics_cache.py` (add correctness tests)

**Interfaces:**
- Consumes: `JAXDynamicsCache.mass_matrix_jit`, `JAXDynamicsCache.bias_forces_jit`
- Compares against: original `jax_mass_matrix(qpos, constants)`, `jax_bias_forces(qpos, qvel, constants)`

- [ ] **Step 1: Add M + h correctness tests**

Append to `tests/test_phase3d3e_jax_dynamics_cache.py`:

```python
class TestMassMatrixBiasForcesCorrectness:
    """Verify jitted M and h match the original non-jitted functions."""

    @pytest.fixture(scope="module")
    def cache_and_refs(self, test_model_and_constants):
        model, constants = test_model_and_constants
        cache = initialize_jax_dynamics_cache(model, constants, warmup=True)
        return cache, constants

    def _get_default_qpos_qvel(self, test_model_and_constants):
        model, _ = test_model_and_constants
        qpos0 = np.array(model.keyframe("default").qpos, dtype=np.float64)
        qvel0 = np.zeros(model.nv, dtype=np.float64)
        return qpos0, qvel0

    def test_mass_matrix_matches_original_default_pose(self, cache_and_refs, test_model_and_constants):
        cache, constants = cache_and_refs
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)

        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix

        # Original
        M_orig = np.array(
            jax_mass_matrix(jnp.array(qpos0, dtype=jnp.float32), constants["_mass_matrix_constants"]),
            dtype=np.float64,
        )
        # Cached
        M_cache = np.array(
            cache.mass_matrix_jit(jnp.array(qpos0, dtype=jnp.float32)),
            dtype=np.float64,
        )

        max_diff = np.max(np.abs(M_orig - M_cache))
        assert max_diff < 1e-6, f"Mass matrix max diff: {max_diff}"
        assert M_orig.shape == M_cache.shape

    def test_bias_forces_matches_original_default_pose(self, cache_and_refs, test_model_and_constants):
        cache, constants = cache_and_refs
        qpos0, qvel0 = self._get_default_qpos_qvel(test_model_and_constants)

        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        # Original
        h_orig = np.array(
            jax_bias_forces(
                jnp.array(qpos0, dtype=jnp.float32),
                jnp.array(qvel0, dtype=jnp.float32),
                constants["_dynamics_constants"],
            ),
            dtype=np.float64,
        )
        # Cached
        h_cache = np.array(
            cache.bias_forces_jit(
                jnp.array(qpos0, dtype=jnp.float32),
                jnp.array(qvel0, dtype=jnp.float32),
            ),
            dtype=np.float64,
        )

        max_diff = np.max(np.abs(h_orig - h_cache))
        assert max_diff < 1e-6, f"Bias forces max diff: {max_diff}"
        assert h_orig.shape == h_cache.shape

    def test_mass_matrix_matches_original_perturbed_pose(self, cache_and_refs, test_model_and_constants):
        cache, constants = cache_and_refs
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)

        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix

        rng = np.random.RandomState(42)
        for trial in range(5):
            qpos_p = qpos0.copy()
            qpos_p[7:17] += rng.randn(10) * 0.05

            M_orig = np.array(
                jax_mass_matrix(jnp.array(qpos_p, dtype=jnp.float32), constants["_mass_matrix_constants"]),
                dtype=np.float64,
            )
            M_cache = np.array(
                cache.mass_matrix_jit(jnp.array(qpos_p, dtype=jnp.float32)),
                dtype=np.float64,
            )
            max_diff = np.max(np.abs(M_orig - M_cache))
            assert max_diff < 1e-6, f"Trial {trial}: mass matrix max diff: {max_diff}"

    def test_bias_forces_matches_original_nonzero_qvel(self, cache_and_refs, test_model_and_constants):
        cache, constants = cache_and_refs
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)

        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        rng = np.random.RandomState(42)
        for trial in range(5):
            qvel_p = rng.randn(16) * 0.1

            h_orig = np.array(
                jax_bias_forces(
                    jnp.array(qpos0, dtype=jnp.float32),
                    jnp.array(qvel_p, dtype=jnp.float32),
                    constants["_dynamics_constants"],
                ),
                dtype=np.float64,
            )
            h_cache = np.array(
                cache.bias_forces_jit(
                    jnp.array(qpos0, dtype=jnp.float32),
                    jnp.array(qvel_p, dtype=jnp.float32),
                ),
                dtype=np.float64,
            )
            max_diff = np.max(np.abs(h_orig - h_cache))
            assert max_diff < 1e-6, f"Trial {trial}: bias forces max diff: {max_diff}"

    def test_no_recompilation_on_same_shape(self, cache_and_refs, test_model_and_constants):
        """Verify that repeated calls with different values but same shape do not trigger reinit."""
        cache, _ = cache_and_refs
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)

        import jax.numpy as jnp
        import jax

        call_count_before = cache.call_count

        # Call 10 times with slightly different qpos
        rng = np.random.RandomState(99)
        for _ in range(10):
            qpos_p = qpos0.copy()
            qpos_p[7:17] += rng.randn(10) * 0.001
            _ = cache.mass_matrix_jit(jnp.array(qpos_p, dtype=jnp.float32))

        # No reinit should have been triggered
        assert cache.recompile_count == 0
        assert cache.fallback_count == 0
```

- [ ] **Step 2: Run correctness tests**

```bash
python -m pytest tests/test_phase3d3e_jax_dynamics_cache.py::TestMassMatrixBiasForcesCorrectness -v
```

Expected: All 5 tests pass with `max_diff < 1e-6`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase3d3e_jax_dynamics_cache.py
git commit -m "test(phase3d3e): add M + h correctness and recompile-guard tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: COM + Torso JIT with Correctness Audit

**Files:**
- Modify: `wheeled_biped/wbc/phase3d3e_jax_dynamics_cache.py` (add COM Jdot*qdot and torso Jdotw*qdot jitted functions)
- Modify: `tests/test_phase3d3e_jax_dynamics_cache.py` (add COM + torso correctness tests)

- [ ] **Step 1: Add COM Jdot*qdot and torso Jdotw*qdot jitted functions to the cache**

In `initialize_jax_dynamics_cache()`, after the existing COM Jacobian setup, replace the placeholder lines with:

```python
    # ── Build jacfwd functions ONCE (outside any jit) ──────────────
    # These are captured by closure in the jitted FD functions below.
    # CRITICAL: Do NOT create jax.jacfwd inside a jitted function body.
    _com_jac_fn = jax.jacfwd(_compute_com_fk_arrays, argnums=0)
    _torso_quat_jac_fn_local = jax.jacfwd(_get_torso_quat_fk_arrays, argnums=0)

    # ── Helper: qvel → dq/dt (qpos time derivative) ───────────────
    # qvel (16,) → dq_dt (17,): [v_world(3); dquat/dt(4); qvel_hinge(10)]
    # dquat/dt = 0.5 * G(q) @ omega, where G is the 4×3 quaternion rate matrix
    def _qvel_to_dqdt(qpos_arr, qvel_arr):
        """Convert qvel (16,) to qpos time derivative dq/dt (17,)."""
        w, x, y, z = qpos_arr[3], qpos_arr[4], qpos_arr[5], qpos_arr[6]
        G = jnp.array([
            [-x, -y, -z],
            [ w, -z,  y],
            [ z,  w, -x],
            [-y,  x,  w],
        ])
        dquat_dt = 0.5 * G @ qvel_arr[3:6]
        dq_dt = jnp.concatenate([
            qvel_arr[0:3],       # world-frame linear velocity
            dquat_dt,             # quaternion derivative
            qvel_arr[6:16],       # hinge joint velocities
        ])
        return dq_dt  # (17,)

    # ── qpos integration helper (JAX, for FD) ─────────────────────
    def _integrate_qpos_jax(qpos_arr, qvel_arr, dt):
        """Integrate qpos by qvel * dt. JAX-compatible."""
        dq_dt = _qvel_to_dqdt(qpos_arr, qvel_arr)
        return qpos_arr + dq_dt * dt

    # ── COM Jdot*qdot via FD (jacfwd captured from outer scope) ───
    @functools.partial(jax.jit, static_argnums=())
    def _com_jdot_qdot_jit(qpos_arr, qvel_arr, eps=1e-5):
        """Jdot_com @ qvel via central FD of COM Jacobian.

        Uses qpos-space Jacobians and dq/dt for correct dimension matching:
          (J(q+eps*dqdt) - J(q-eps*dqdt)) @ dqdt / (2*eps)
        where dqdt = qvel_to_dqdt(qpos, qvel).
        """
        dq_dt = _qvel_to_dqdt(qpos_arr, qvel_arr)  # (17,)
        q_plus = qpos_arr + dq_dt * eps
        q_minus = qpos_arr - dq_dt * eps

        # _com_jac_fn is the pre-constructed jacfwd (captured from outer scope)
        J_plus = _com_jac_fn(q_plus, fk_a, bm_jax, bipos_jax)    # (3, 17)
        J_minus = _com_jac_fn(q_minus, fk_a, bm_jax, bipos_jax)  # (3, 17)

        return (J_plus - J_minus) @ dq_dt / (2.0 * eps)  # (3,)

    cache.com_jdot_qdot_jit = _com_jdot_qdot_jit

    # ── Torso Jdotw*qdot via FD (jacfwd captured from outer scope) ──
    @functools.partial(jax.jit, static_argnums=())
    def _torso_jdotw_qdot_jit(qpos_arr, qvel_arr, eps=1e-5):
        """Jdot_w_torso @ qvel via central FD of torso quaternion Jacobian.

        Returns torso quaternion-space Jdot*qdot (4,).
        Convert to angular acceleration via: alpha = 2*G(q)^T @ result.
        """
        dq_dt = _qvel_to_dqdt(qpos_arr, qvel_arr)  # (17,)
        q_plus = qpos_arr + dq_dt * eps
        q_minus = qpos_arr - dq_dt * eps

        # _torso_quat_jac_fn_local is pre-constructed jacfwd (captured from outer scope)
        J_plus = _torso_quat_jac_fn_local(q_plus, fk_a, torso_id)    # (4, 17)
        J_minus = _torso_quat_jac_fn_local(q_minus, fk_a, torso_id)  # (4, 17)

        return (J_plus - J_minus) @ dq_dt / (2.0 * eps)  # (4,)

    cache.torso_jdotw_qdot_jit = _torso_jdotw_qdot_jit
```

Also update `_warmup_cache()`:

```python
    if cache.com_jdot_qdot_jit is not None:
        _ = cache.com_jdot_qdot_jit(dummy_qpos, dummy_qvel)

    if cache.torso_jdotw_qdot_jit is not None:
        _ = cache.torso_jdotw_qdot_jit(dummy_qpos, dummy_qvel)
```

- [ ] **Step 2: Add COM + torso correctness tests**

Append to `tests/test_phase3d3e_jax_dynamics_cache.py`:

```python
class TestCOMTorsoCorrectness:
    """Verify jitted COM and torso functions match originals."""

    @pytest.fixture(scope="module")
    def cache_and_refs(self, test_model_and_constants):
        model, constants = test_model_and_constants
        cache = initialize_jax_dynamics_cache(model, constants, warmup=True)
        return cache, constants

    def _get_default_qpos_qvel(self, test_model_and_constants):
        model, _ = test_model_and_constants
        qpos0 = np.array(model.keyframe("default").qpos, dtype=np.float64)
        qvel0 = np.zeros(model.nv, dtype=np.float64)
        return qpos0, qvel0

    def _qpos_jac_to_qvel_jac(self, J_qpos, qpos):
        """Convert (3,17) or (4,17) qpos-space Jacobian to (3,16) or (4,16) qvel-space."""
        rows = J_qpos.shape[0]
        J_qvel = np.zeros((rows, 16), dtype=np.float64)
        # Position columns: identity
        J_qvel[:, 0:3] = J_qpos[:, 0:3]
        # Quaternion columns → angular velocity via G matrix
        q_torso = qpos[3:7]
        w, x, y, z = q_torso[0], q_torso[1], q_torso[2], q_torso[3]
        G = np.array([
            [-x, -y, -z],
            [w, -z,  y],
            [z,  w, -x],
            [-y,  x,  w],
        ])
        # J_qvel[:, 3:6] = J_qpos[:, 3:7] @ (0.5 * G)
        J_qvel[:, 3:6] = J_qpos[:, 3:7] @ (0.5 * G)
        # Hinge joint columns: identity
        J_qvel[:, 6:16] = J_qpos[:, 7:17]
        return J_qvel

    def test_com_jacobian_matches_original(self, cache_and_refs, test_model_and_constants):
        cache, constants = cache_and_refs
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)

        import jax.numpy as jnp
        from wheeled_biped.wbc.offline_task_stack import compute_com_jacobian

        kc = constants["_kinematics_constants"]

        Jcom_orig = compute_com_jacobian(qpos0, kc)  # (3, 16)
        Jcom_qpos = np.array(
            cache.com_jacobian_jit(jnp.array(qpos0, dtype=jnp.float32)),
            dtype=np.float64,
        )  # (3, 17)
        Jcom_cache = self._qpos_jac_to_qvel_jac(Jcom_qpos, qpos0)  # (3, 16)

        max_diff = np.max(np.abs(Jcom_orig - Jcom_cache))
        assert max_diff < 1e-6, f"COM Jacobian max diff: {max_diff}"

    def test_torso_ang_vel_jacobian_matches_original(self, cache_and_refs, test_model_and_constants):
        cache, constants = cache_and_refs
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)

        import jax.numpy as jnp
        from wheeled_biped.wbc.offline_task_stack import compute_torso_angular_velocity_jacobian

        kc = constants["_kinematics_constants"]

        Jr_orig = compute_torso_angular_velocity_jacobian(qpos0, kc)  # (3, 16)

        Jquat_qpos = np.array(
            cache.torso_ang_vel_jacobian_jit(jnp.array(qpos0, dtype=jnp.float32)),
            dtype=np.float64,
        )  # (4, 17)

        # Convert to angular velocity Jacobian
        q_torso = qpos0[3:7]
        w, x, y, z = q_torso[0], q_torso[1], q_torso[2], q_torso[3]
        G = np.array([
            [-x, -y, -z],
            [w, -z,  y],
            [z,  w, -x],
            [-y,  x,  w],
        ])
        Jquat_qvel = self._qpos_jac_to_qvel_jac(Jquat_qpos, qpos0)  # (4, 16)
        Jr_cache = 2.0 * G.T @ Jquat_qvel  # (3, 16)

        max_diff = np.max(np.abs(Jr_orig - Jr_cache))
        assert max_diff < 1e-6, f"Torso ang vel Jacobian max diff: {max_diff}"

    def test_torso_orientation_error_matches_original(self, cache_and_refs, test_model_and_constants):
        cache, constants = cache_and_refs
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)

        import jax.numpy as jnp
        from wheeled_biped.wbc.offline_task_stack import compute_torso_orientation_error

        kc = constants["_kinematics_constants"]

        orig = compute_torso_orientation_error(qpos0, kc)
        cached = cache.torso_orientation_error_jit(jnp.array(qpos0, dtype=jnp.float32))

        e_R_orig = orig["e_R"]
        e_R_cache = np.array(cached["e_R"], dtype=np.float64)
        max_diff_eR = np.max(np.abs(e_R_orig - e_R_cache))
        assert max_diff_eR < 1e-6, f"e_R max diff: {max_diff_eR}"

        rpy_orig = orig["current_rpy"]
        rpy_cache = np.array(cached["current_rpy"], dtype=np.float64)
        max_diff_rpy = np.max(np.abs(rpy_orig - rpy_cache))
        assert max_diff_rpy < 1e-5, f"rpy max diff: {max_diff_rpy}"

    def test_com_jdot_qdot_matches_original(self, cache_and_refs, test_model_and_constants):
        cache, constants = cache_and_refs
        qpos0, qvel0 = self._get_default_qpos_qvel(test_model_and_constants)

        import jax.numpy as jnp
        from wheeled_biped.wbc.offline_task_stack import compute_com_jdot_qdot

        # Need nonzero qvel for meaningful Jdot*qdot
        rng = np.random.RandomState(42)
        qvel_p = rng.randn(16) * 0.1

        kc = constants["_kinematics_constants"]

        jdq_orig = compute_com_jdot_qdot(qpos0, qvel_p, kc)  # (3,)
        # Jitted returns (3,) — uses dq/dt internally for correct dimension matching
        jdq_cache = np.array(
            cache.com_jdot_qdot_jit(
                jnp.array(qpos0, dtype=jnp.float32),
                jnp.array(qvel_p, dtype=jnp.float32),
            ),
            dtype=np.float64,
        )
        assert jdq_cache.shape == (3,), f"Expected (3,), got {jdq_cache.shape}"

        max_diff = np.max(np.abs(jdq_orig - jdq_cache))
        assert max_diff < 1e-6, f"COM Jdot*qdot max diff: {max_diff}"

    def test_torso_jdotw_qdot_matches_original(self, cache_and_refs, test_model_and_constants):
        cache, constants = cache_and_refs
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)

        import jax.numpy as jnp
        from wheeled_biped.wbc.offline_task_stack import compute_torso_jdotw_qdot

        rng = np.random.RandomState(42)
        qvel_p = rng.randn(16) * 0.1

        kc = constants["_kinematics_constants"]

        jdw_orig = compute_torso_jdotw_qdot(qpos0, qvel_p, kc)  # (3,)
        # Jitted returns torso quat-space Jdot*qdot (4,)
        jdw_cache_quat = np.array(
            cache.torso_jdotw_qdot_jit(
                jnp.array(qpos0, dtype=jnp.float32),
                jnp.array(qvel_p, dtype=jnp.float32),
            ),
            dtype=np.float64,
        )
        assert jdw_cache_quat.shape == (4,), f"Expected (4,), got {jdw_cache_quat.shape}"

        # Convert to angular acceleration: alpha = 2 * G(q)^T @ jdw_quat
        q_torso = qpos0[3:7]
        w, x, y, z = q_torso[0], q_torso[1], q_torso[2], q_torso[3]
        G = np.array([[-x,-y,-z],[w,-z,y],[z,w,-x],[-y,x,w]], dtype=np.float64)
        jdw_cache_ang = 2.0 * G.T @ jdw_cache_quat  # (3,)

        max_diff = np.max(np.abs(jdw_orig - jdw_cache_ang))
        assert max_diff < 1e-6, f"Torso Jdotw*qdot max diff: {max_diff}"
```

- [ ] **Step 3: Run COM + torso tests**

```bash
python -m pytest tests/test_phase3d3e_jax_dynamics_cache.py::TestCOMTorsoCorrectness -v
```

Expected: All 6 tests pass with `max_diff < 1e-6`.

- [ ] **Step 4: Commit**

```bash
git add wheeled_biped/wbc/phase3d3e_jax_dynamics_cache.py tests/test_phase3d3e_jax_dynamics_cache.py
git commit -m "feat(phase3d3e): add COM + torso JIT functions with correctness audit

Adds jitted COM Jdot*qdot and torso Jdotw*qdot via central FD.
All COM/torso functions verified against original implementations.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Contact JIT (Batched Padded Representation)

**Files:**
- Modify: `wheeled_biped/wbc/phase3d3e_jax_dynamics_cache.py` (add contact Jacobian batch and contact Jdot*qdot batch jitted functions, add padded contact helpers)
- Modify: `tests/test_phase3d3e_jax_dynamics_cache.py` (add contact correctness tests)

**Risk note:** This is the riskiest stage. The contact Jacobian path involves variable contact counts, body IDs, local points, and finite differences. If JIT of this path proves fragile or incorrect, stop and report `PARTIAL_JAX_DYNAMICS_CACHE`. Do not force a broken implementation.

- [ ] **Step 1: Add padded contact conversion helper and batched contact Jacobian**

Add to `wheeled_biped/wbc/phase3d3e_jax_dynamics_cache.py`, before `initialize_jax_dynamics_cache()`:

```python
# ═══════════════════════════════════════════════════════════════════════════
# Padded contact array helpers
# ═══════════════════════════════════════════════════════════════════════════

def contacts_to_padded_arrays(
    contacts: list[dict[str, Any]],
    max_contacts: int = DEFAULT_MAX_CONTACTS,
) -> dict[str, np.ndarray]:
    """Convert a list of contact dicts to fixed-shape padded arrays.

    Args:
        contacts: list of active contact dicts (body_id, local_point, frame, position).
        max_contacts: maximum contacts to pad to.

    Returns:
        dict with keys: active (max_contacts,), body_id (max_contacts,),
        local_point (max_contacts, 3), frame (max_contacts, 3, 3),
        position (max_contacts, 3), num_contacts (int).

    Raises:
        ValueError: if len(contacts) > max_contacts.
    """
    m = len(contacts)
    if m > max_contacts:
        raise ValueError(
            f"Contact count {m} exceeds max_contacts {max_contacts}. "
            f"Increase max_contacts or fix scenario."
        )

    active = np.zeros(max_contacts, dtype=np.int32)
    body_id = np.zeros(max_contacts, dtype=np.int32)
    local_point = np.zeros((max_contacts, 3), dtype=np.float64)
    frame = np.zeros((max_contacts, 3, 3), dtype=np.float64)
    position = np.zeros((max_contacts, 3), dtype=np.float64)

    for i in range(m):
        c = contacts[i]
        active[i] = 1
        body_id[i] = int(c["body_id"])
        local_point[i, :] = np.array(c["local_point"], dtype=np.float64)
        frame[i, :, :] = np.array(c["frame"], dtype=np.float64)
        position[i, :] = np.array(c["position"], dtype=np.float64)

    return {
        "active": active,
        "body_id": body_id,
        "local_point": local_point,
        "frame": frame,
        "position": position,
        "num_contacts": m,
    }
```

Then, in `initialize_jax_dynamics_cache()`, after the torso setup, add:

```python
    # ── Contact Jacobian batch (vmap over padded contact slots) ──────
    from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian

    # Build a single-contact JAX function that takes arrays
    # contact_point_translational_jacobian needs: qpos, body_id, local_point, constants
    # We need a version that takes FK arrays instead of constants dict.
    # For Stage E4, we wrap the existing function and accept the overhead
    # of the constants dict as a static capture via closure.

    # NOTE: contact_point_translational_jacobian internally uses FK with
    # the constants dict. For a fully-JIT-compatible version, we would need
    # an _fk_arrays variant. For now, we build a minimal array-only wrapper.

    contact_c = constants["_contact_constants"]

    def _single_contact_jacobian(qpos_arr, body_id_scalar, local_point_arr):
        """Compute Jp (3, 16) for a single contact. Wraps existing function."""
        # Cast body_id to int (static across vmap calls for same contact slot)
        return contact_point_translational_jacobian(
            qpos_arr, int(body_id_scalar), local_point_arr, contact_c,
        )

    # We CANNOT vmap over body_id because the existing function takes int, not array.
    # Instead, we use a Python loop over padded contacts, calling a jitted
    # per-contact function. This is still a big improvement over the original
    # because the per-contact function is jitted.

    @functools.partial(jax.jit, static_argnums=())
    def _contact_jacobian_single_jit(qpos_arr, body_id_scalar, local_point_arr):
        """Jitted single-contact translational Jacobian."""
        return contact_point_translational_jacobian(
            qpos_arr, body_id_scalar, local_point_arr, contact_c,
        )

    cache._contact_jacobian_single_jit = _contact_jacobian_single_jit

    # Contact Jdot*qdot via FD — per-contact, jitted
    @functools.partial(jax.jit, static_argnums=())
    def _contact_jdot_qdot_single_jit(qpos_arr, qvel_arr, body_id_scalar, local_point_arr, eps=1e-5):
        """Jitted single-contact Jdot*qdot via central FD."""
        def _integrate(q, v, dt):
            q_out = q.copy()
            q_out = q_out.at[0:3].set(q[0:3] + v[0:3] * dt)
            omega = v[3:6] * dt
            angle = jnp.sqrt(jnp.sum(omega**2))
            safe_angle = jnp.where(angle > 1e-15, angle, 1.0)
            axis = jnp.where(angle > 1e-15, omega / safe_angle, jnp.array([1.0, 0.0, 0.0]))
            half = 0.5 * safe_angle
            s = jnp.sin(half)
            dq = jnp.array([jnp.cos(half), axis[0]*s, axis[1]*s, axis[2]*s])
            w0, x0, y0, z0 = q[3], q[4], q[5], q[6]
            w1, x1, y1, z1 = dq[0], dq[1], dq[2], dq[3]
            q_out = q_out.at[3].set(w0*w1 - x0*x1 - y0*y1 - z0*z1)
            q_out = q_out.at[4].set(w0*x1 + x0*w1 + y0*z1 - z0*y1)
            q_out = q_out.at[5].set(w0*y1 - x0*z1 + y0*w1 + z0*x1)
            q_out = q_out.at[6].set(w0*z1 + x0*y1 - y0*x1 + z0*w1)
            q_out = q_out.at[7:17].set(q[7:17] + v[6:16] * dt)
            return q_out

        q_plus = _integrate(qpos_arr, qvel_arr, eps)
        q_minus = _integrate(qpos_arr, qvel_arr, -eps)

        Jp_plus = contact_point_translational_jacobian(q_plus, body_id_scalar, local_point_arr, contact_c)
        Jp_minus = contact_point_translational_jacobian(q_minus, body_id_scalar, local_point_arr, contact_c)

        return (Jp_plus - Jp_minus) @ qvel_arr / (2.0 * eps)

    cache._contact_jdot_qdot_single_jit = _contact_jdot_qdot_single_jit

    # Store contact constants for per-step use
    cache._contact_constants = contact_c
```

- [ ] **Step 2: Add contact correctness tests**

Append to test file:

```python
class TestContactJacobianCorrectness:
    """Verify jitted contact Jacobian and Jdot*qdot match originals."""

    @pytest.fixture(scope="module")
    def cache_and_refs(self, test_model_and_constants):
        model, constants = test_model_and_constants
        import mujoco
        cache = initialize_jax_dynamics_cache(model, constants, warmup=True)
        # Inline contact extraction
        qpos0 = np.array(model.keyframe("default").qpos, dtype=np.float64)
        data = mujoco.MjData(model)
        data.qpos[:] = qpos0
        mujoco.mj_forward(model, data)
        contact_constants = constants["_contact_constants"]
        wheel_body_ids = contact_constants.get("wheel_body_ids", {})
        wheel_ids_set = set(int(v) for v in wheel_body_ids.values() if v >= 0)
        contacts = []
        for contact_id in range(data.ncon):
            c = data.contact[contact_id]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            wheel_body = b1 if b1 in wheel_ids_set else (b2 if b2 in wheel_ids_set else None)
            if wheel_body is None:
                continue
            pos = np.array(c.pos, dtype=np.float64)
            frame = np.array(c.frame, dtype=np.float64).reshape(3, 3)
            body_xpos = np.array(data.xpos[wheel_body], dtype=np.float64)
            body_xmat = np.array(data.xmat[wheel_body], dtype=np.float64).reshape(3, 3)
            local_point = body_xmat.T @ (pos - body_xpos)
            contacts.append({
                "body_id": int(wheel_body),
                "position": pos, "frame": frame, "local_point": local_point,
            })
        return cache, constants, model, contacts

    def test_contact_jacobian_single_matches_original(self, cache_and_refs):
        cache, constants, model, contacts = cache_and_refs
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian

        qpos0 = np.array(model.keyframe("default").qpos, dtype=np.float64)
        qpos_jax = jnp.array(qpos0, dtype=jnp.float32)
        contact_c = constants["_contact_constants"]

        for i, c in enumerate(contacts[:2]):  # test first 2 contacts
            bid = int(c["body_id"])
            lp = jnp.array(c["local_point"], dtype=jnp.float32)

            Jp_orig = np.array(
                contact_point_translational_jacobian(qpos_jax, bid, lp, contact_c),
                dtype=np.float64,
            )
            Jp_cache = np.array(
                cache._contact_jacobian_single_jit(qpos_jax, bid, lp),
                dtype=np.float64,
            )

            max_diff = np.max(np.abs(Jp_orig - Jp_cache))
            assert max_diff < 1e-6, f"Contact {i}: Jp max diff: {max_diff}"
            assert Jp_orig.shape == (3, 16)

    def test_contact_jdot_qdot_single_matches_original(self, cache_and_refs):
        cache, constants, model, contacts = cache_and_refs
        import jax.numpy as jnp
        from wheeled_biped.wbc.offline_qp_wbc import compute_contact_jdot_qdot

        qpos0 = np.array(model.keyframe("default").qpos, dtype=np.float64)
        qvel0 = np.zeros(model.nv, dtype=np.float64)
        contact_c = constants["_contact_constants"]

        # Need nonzero qvel
        rng = np.random.RandomState(42)
        qvel_p = rng.randn(16) * 0.1

        # Original: computes all contacts
        jdq_all_orig = compute_contact_jdot_qdot(qpos0, qvel_p, contacts, contact_c)

        qpos_jax = jnp.array(qpos0, dtype=jnp.float32)
        qvel_jax = jnp.array(qvel_p, dtype=jnp.float32)

        for i, c in enumerate(contacts[:2]):
            bid = int(c["body_id"])
            lp = jnp.array(c["local_point"], dtype=jnp.float32)

            jdq_cache = np.array(
                cache._contact_jdot_qdot_single_jit(qpos_jax, qvel_jax, bid, lp),
                dtype=np.float64,
            )
            jdq_orig = jdq_all_orig[3*i:3*i+3]

            max_diff = np.max(np.abs(jdq_orig - jdq_cache))
            assert max_diff < 1e-6, f"Contact {i}: Jdot*qdot max diff: {max_diff}"

    def test_padded_contact_array_shape(self):
        """Verify contacts_to_padded_arrays produces correct shapes."""
        from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import contacts_to_padded_arrays

        # Empty contacts
        empty = contacts_to_padded_arrays([], max_contacts=4)
        assert empty["active"].shape == (4,)
        assert empty["body_id"].shape == (4,)
        assert empty["local_point"].shape == (4, 3)
        assert empty["frame"].shape == (4, 3, 3)
        assert empty["position"].shape == (4, 3)
        assert empty["num_contacts"] == 0
        assert np.all(empty["active"] == 0)

        # 2 contacts
        contacts_2 = [
            {"body_id": 5, "local_point": [1, 2, 3], "frame": np.eye(3), "position": [0, 0, 0]},
            {"body_id": 8, "local_point": [4, 5, 6], "frame": np.eye(3), "position": [1, 1, 1]},
        ]
        padded = contacts_to_padded_arrays(contacts_2, max_contacts=4)
        assert padded["num_contacts"] == 2
        assert np.all(padded["active"][:2] == 1)
        assert np.all(padded["active"][2:] == 0)
        assert padded["body_id"][0] == 5
        assert padded["body_id"][1] == 8

    def test_too_many_contacts_raises(self):
        """Verify ValueError when contact count exceeds max_contacts."""
        from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import contacts_to_padded_arrays
        import pytest
        contacts_5 = [
            {"body_id": i, "local_point": [0,0,0], "frame": np.eye(3), "position": [0,0,0]}
            for i in range(5)
        ]
        with pytest.raises(ValueError, match="exceeds max_contacts"):
            contacts_to_padded_arrays(contacts_5, max_contacts=4)
```

- [ ] **Step 3: Run contact tests**

```bash
python -m pytest tests/test_phase3d3e_jax_dynamics_cache.py::TestContactJacobianCorrectness -v
```

Expected: All 4 tests pass. If contact Jacobian tests fail with JIT errors, do NOT force a fix. Report the specific error and move to Task 6 with a `PARTIAL_JAX_DYNAMICS_CACHE` note.

- [ ] **Step 4: Commit (or note partial)**

If tests pass:
```bash
git add wheeled_biped/wbc/phase3d3e_jax_dynamics_cache.py tests/test_phase3d3e_jax_dynamics_cache.py
git commit -m "feat(phase3d3e): add batched contact Jacobian + Jdot*qdot JIT

Adds per-contact jitted Jacobian and Jdot*qdot functions with
padded contact array helpers. Verified against original implementations.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

If tests fail: document in a note file and proceed to Task 6 with contact JIT disabled.

---

### Task 6: Full `prepare_phase3b_snapshot_cached()`

**Files:**
- Modify: `wheeled_biped/wbc/phase3d3e_jax_dynamics_cache.py` (add `prepare_phase3b_snapshot_cached()`)
- Modify: `tests/test_phase3d3e_jax_dynamics_cache.py` (add full snapshot comparison tests)

- [ ] **Step 1: Implement `prepare_phase3b_snapshot_cached()`**

Add to the cache module:

```python
def prepare_phase3b_snapshot_cached(
    cache: JAXDynamicsCache,
    scenario_name: str,
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    constants: dict[str, Any],
    *,
    max_contacts: int = DEFAULT_MAX_CONTACTS,
) -> "Phase3BSnapshot":
    """Drop-in cached/precompiled replacement for prepare_phase3b_snapshot().

    Uses precompiled JAX functions from ``cache`` instead of re-tracing
    JAX operations on every call.

    Args:
        cache: initialized ``JAXDynamicsCache``.
        scenario_name: human-readable scenario identifier.
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        constants: dict from ``build_qp_wbc_constants``.
        max_contacts: max contacts for padded stack.

    Returns:
        ``Phase3BSnapshot`` with all precomputed data (same structure as original).
    """
    import time
    from wheeled_biped.wbc.phase3b_cached_stack import (
        Phase3BSnapshot, PaddedContactStack, MAX_CONTACTS,
        build_padded_contact_stack,
    )

    t0 = time.perf_counter()

    nv = cache.nv
    nu = cache.nu
    m = len(contacts)

    # ── Convert contacts to padded arrays (Python, outside JIT) ────
    padded_c = contacts_to_padded_arrays(contacts, max_contacts=max_contacts)

    qpos_jax = jnp.array(qpos, dtype=jnp.float32)
    qvel_jax = jnp.array(qvel, dtype=jnp.float32)

    # ── Mass matrix and bias forces (jitted) ──────────────────────
    M_jax = cache.mass_matrix_jit(qpos_jax)
    h_jax = cache.bias_forces_jit(qpos_jax, qvel_jax)
    M = np.array(M_jax, dtype=np.float64)
    h = np.array(h_jax, dtype=np.float64)

    # ── Actuator selection matrix ─────────────────────────────────
    S_np = np.array(constants.get("S",
        np.zeros((nv, nu), dtype=np.float64)), dtype=np.float64)
    if S_np.shape != (nv, nu):
        from wheeled_biped.wbc.offline_qp_wbc import build_actuator_selection_matrix_from_dims
        S_np = build_actuator_selection_matrix_from_dims(nv, nu)

    # ── Padded contact stack ──────────────────────────────────────
    if m > 0:
        # Use original build_padded_contact_stack for contact stack
        # (Contact Jacobian per-contact loop in Python; each call is jitted)
        contact_c = constants.get("_contact_constants", cache._contact_constants)

        # Build contact stack manually using jitted per-contact functions
        Jp = np.zeros((max_contacts, 3, nv), dtype=np.float64)
        Jr = np.zeros((max_contacts, 3, nv), dtype=np.float64)
        JcT = np.zeros((nv, 3 * max_contacts), dtype=np.float64)
        frames = np.zeros((max_contacts, 3, 3), dtype=np.float64)
        local_points = np.zeros((max_contacts, 3), dtype=np.float64)
        body_ids = np.zeros(max_contacts, dtype=np.int32)
        normals = np.zeros((max_contacts, 3), dtype=np.float64)
        positions_world = np.zeros((max_contacts, 3), dtype=np.float64)
        active_mask = np.zeros(max_contacts, dtype=bool)

        for i in range(m):
            c = contacts[i]
            bid = int(c["body_id"])
            lp_jax = jnp.array(c["local_point"], dtype=jnp.float32)
            fr = np.array(c["frame"], dtype=np.float64)
            pos = np.array(c["position"], dtype=np.float64)

            # Jitted single-contact Jacobian
            Jp_i = np.array(
                cache._contact_jacobian_single_jit(qpos_jax, bid, lp_jax),
                dtype=np.float64,
            )  # (3, 16)

            n_world = fr[:, 0].copy()
            JcT_i = Jp_i.T @ fr

            Jp[i, :, :] = Jp_i
            JcT[:, 3*i:3*i+3] = JcT_i
            frames[i, :, :] = fr
            local_points[i, :] = np.array(c["local_point"], dtype=np.float64)
            body_ids[i] = bid
            normals[i, :] = n_world
            positions_world[i, :] = pos
            active_mask[i] = True

        contact_stack = PaddedContactStack(
            Jp=Jp, Jr=Jr, JcT=JcT, frame=frames,
            local_point=local_points, body_id=body_ids,
            normal=normals, position_world=positions_world,
            active_mask=active_mask, num_contacts=m,
        )
    else:
        contact_stack = PaddedContactStack(
            Jp=np.zeros((max_contacts, 3, nv), dtype=np.float64),
            Jr=np.zeros((max_contacts, 3, nv), dtype=np.float64),
            JcT=np.zeros((nv, 3 * max_contacts), dtype=np.float64),
            frame=np.zeros((max_contacts, 3, 3), dtype=np.float64),
            local_point=np.zeros((max_contacts, 3), dtype=np.float64),
            body_id=np.zeros(max_contacts, dtype=np.int32),
            normal=np.zeros((max_contacts, 3), dtype=np.float64),
            position_world=np.zeros((max_contacts, 3), dtype=np.float64),
            active_mask=np.zeros(max_contacts, dtype=bool),
            num_contacts=0,
        )

    # ── Jdot_qdot for contact normal acceleration ─────────────────
    if m > 0 and cache._contact_jdot_qdot_single_jit is not None:
        jdot_qdot = np.zeros(3 * max_contacts, dtype=np.float64)
        for i in range(m):
            c = contacts[i]
            bid = int(c["body_id"])
            lp_jax = jnp.array(c["local_point"], dtype=jnp.float32)
            jdq_i = np.array(
                cache._contact_jdot_qdot_single_jit(qpos_jax, qvel_jax, bid, lp_jax),
                dtype=np.float64,
            )
            jdot_qdot[3*i:3*i+3] = jdq_i
    elif m > 0:
        # Fallback to original function
        from wheeled_biped.wbc.offline_qp_wbc import compute_contact_jdot_qdot
        contact_c = constants.get("_contact_constants")
        if contact_c is None:
            from wheeled_biped.wbc.offline_qp_wbc import _ensure_contact_constants
            _ensure_contact_constants(constants)
            contact_c = constants["_contact_constants"]
        jdot_qdot_raw = compute_contact_jdot_qdot(qpos, qvel, contacts, contact_c)
        jdot_qdot = np.zeros(3 * max_contacts, dtype=np.float64)
        jdot_qdot[:3*m] = jdot_qdot_raw
        cache.fallback_count += 1
    else:
        jdot_qdot = np.zeros(3 * max_contacts, dtype=np.float64)

    # ── COM Jacobian and Jdot_qdot ────────────────────────────────
    kc = constants.get("_kinematics_constants")

    Jcom_qpos = np.array(cache.com_jacobian_jit(qpos_jax), dtype=np.float64)  # (3, 17)
    Jcom = _qpos_jac_to_qvel_jac_static(Jcom_qpos, qpos)  # (3, 16)

    if cache.com_jdot_qdot_jit is not None:
        # Returns (3,) — uses dq/dt internally for correct dimension matching
        jdq_com = np.array(
            cache.com_jdot_qdot_jit(qpos_jax, qvel_jax),
            dtype=np.float64,
        )
    else:
        from wheeled_biped.wbc.offline_task_stack import compute_com_jdot_qdot
        jdq_com = compute_com_jdot_qdot(qpos, qvel, kc)
        cache.fallback_count += 1

    # ── COM current position ──────────────────────────────────────
    from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import _compute_com_fk_arrays
    com_pos = np.array(
        _compute_com_fk_arrays(
            qpos_jax, cache.fk_arrays,
            jnp.array(cache.body_mass), jnp.array(cache.body_ipos),
        ),
        dtype=np.float64,
    )

    # ── Torso orientation ─────────────────────────────────────────
    Jquat_qpos = np.array(
        cache.torso_ang_vel_jacobian_jit(qpos_jax), dtype=np.float64
    )  # (4, 17)
    Jquat_qvel = _qpos_jac_to_qvel_jac_static(Jquat_qpos, qpos)  # (4, 16)

    q_torso = qpos[3:7]
    w, x, y, z = q_torso[0], q_torso[1], q_torso[2], q_torso[3]
    G = np.array([[-x,-y,-z],[w,-z,y],[z,w,-x],[-y,x,w]], dtype=np.float64)
    Jr = 2.0 * G.T @ Jquat_qvel  # (3, 16)

    if cache.torso_jdotw_qdot_jit is not None:
        # Returns torso quat-space Jdot*qdot (4,)
        # Convert to angular acceleration: alpha = 2 * G(q)^T @ result
        jdw_torso_quat = np.array(
            cache.torso_jdotw_qdot_jit(qpos_jax, qvel_jax),
            dtype=np.float64,
        )
        jdw_torso = 2.0 * G.T @ jdw_torso_quat  # (3,)
    else:
        from wheeled_biped.wbc.offline_task_stack import compute_torso_jdotw_qdot
        jdw_torso = compute_torso_jdotw_qdot(qpos, qvel, kc)
        cache.fallback_count += 1

    orient_result = cache.torso_orientation_error_jit(qpos_jax)
    e_R = np.array(orient_result["e_R"], dtype=np.float64)
    current_rpy = np.array(orient_result["current_rpy"], dtype=np.float64)

    qvel_np = np.array(qvel, dtype=np.float64)
    omega_current = Jr @ qvel_np

    # ── Torque limits ────────────────────────────────────────────
    tau_min = np.array(constants.get("tau_min", np.full(nu, -60.0)), dtype=np.float64)
    tau_max = np.array(constants.get("tau_max", np.full(nu, 60.0)), dtype=np.float64)

    # ── Robot mass info ──────────────────────────────────────────
    body_mass_arr = constants.get("body_mass", np.ones(1, dtype=np.float32))
    total_mass = float(np.sum(np.array(body_mass_arr)))
    g_val = float(np.array(constants.get("gravity",
        jnp.array([0, 0, -9.81], dtype=jnp.float32))[2]))
    robot_weight = total_mass * abs(g_val)

    # ── Friction coefficient ─────────────────────────────────────
    mu = float(constants.get("mu", 0.8))

    snapshot_time = time.perf_counter() - t0

    cache.call_count += 1

    return Phase3BSnapshot(
        scenario_name=scenario_name,
        qpos=qpos.copy(),
        qvel=qvel.copy(),
        M=M, h=h, S=S_np,
        contact_stack=contact_stack, jdot_qdot=jdot_qdot, mu=mu,
        Jcom=Jcom, jdq_com=jdq_com, com_position=com_pos,
        Jr=Jr, jdw_torso=jdw_torso,
        e_R=e_R, omega_current=omega_current, current_rpy=current_rpy,
        tau_min=tau_min, tau_max=tau_max,
        total_mass=total_mass, robot_weight=robot_weight,
        snapshot_time_s=snapshot_time,
    )


def _qpos_jac_to_qvel_jac_static(J_qpos: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    """Convert qpos-space Jacobian to qvel-space. NumPy, outside JIT."""
    rows = J_qpos.shape[0]
    J_qvel = np.zeros((rows, 16), dtype=np.float64)
    J_qvel[:, 0:3] = J_qpos[:, 0:3]
    q_torso = qpos[3:7]
    w, x, y, z = q_torso[0], q_torso[1], q_torso[2], q_torso[3]
    G = np.array([[-x,-y,-z],[w,-z,y],[z,w,-x],[-y,x,w]], dtype=np.float64)
    J_qvel[:, 3:6] = J_qpos[:, 3:7] @ (0.5 * G)
    J_qvel[:, 6:16] = J_qpos[:, 7:17]
    return J_qvel
```

- [ ] **Step 2: Add full snapshot comparison test**

```python
class TestFullCachedSnapshot:
    """Verify prepare_phase3b_snapshot_cached matches original."""

    @pytest.fixture(scope="module")
    def cache_and_refs(self, test_model_and_constants):
        model, constants = test_model_and_constants
        import mujoco
        cache = initialize_jax_dynamics_cache(model, constants, warmup=True)
        qpos0 = np.array(model.keyframe("default").qpos, dtype=np.float64)
        qvel0 = np.zeros(model.nv, dtype=np.float64)
        data = mujoco.MjData(model)
        data.qpos[:] = qpos0
        mujoco.mj_forward(model, data)
        contact_constants = constants["_contact_constants"]
        wheel_body_ids = contact_constants.get("wheel_body_ids", {})
        wheel_ids_set = set(int(v) for v in wheel_body_ids.values() if v >= 0)
        contacts = []
        for contact_id in range(data.ncon):
            c = data.contact[contact_id]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            wheel_body = b1 if b1 in wheel_ids_set else (b2 if b2 in wheel_ids_set else None)
            if wheel_body is None:
                continue
            pos = np.array(c.pos, dtype=np.float64)
            frame = np.array(c.frame, dtype=np.float64).reshape(3, 3)
            body_xpos = np.array(data.xpos[wheel_body], dtype=np.float64)
            body_xmat = np.array(data.xmat[wheel_body], dtype=np.float64).reshape(3, 3)
            local_point = body_xmat.T @ (pos - body_xpos)
            contacts.append({
                "body_id": int(wheel_body),
                "position": pos, "frame": frame, "local_point": local_point,
            })
        return cache, constants, model, qpos0, qvel0, contacts

    def test_full_snapshot_matches_original(self, cache_and_refs):
        cache, constants, model, qpos0, qvel0, contacts = cache_and_refs

        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import prepare_phase3b_snapshot_cached

        snap_orig = prepare_phase3b_snapshot("test", qpos0, qvel0, contacts, constants)
        snap_cache = prepare_phase3b_snapshot_cached(
            cache, "test", qpos0, qvel0, contacts, constants,
        )

        # Compare all numeric fields
        assert np.max(np.abs(snap_orig.M - snap_cache.M)) < 1e-6, "M mismatch"
        assert np.max(np.abs(snap_orig.h - snap_cache.h)) < 1e-6, "h mismatch"
        assert np.max(np.abs(snap_orig.Jcom - snap_cache.Jcom)) < 1e-6, "Jcom mismatch"
        assert np.max(np.abs(snap_orig.jdq_com - snap_cache.jdq_com)) < 1e-5, "jdq_com mismatch"
        assert np.max(np.abs(snap_orig.Jr - snap_cache.Jr)) < 1e-6, "Jr mismatch"
        assert np.max(np.abs(snap_orig.e_R - snap_cache.e_R)) < 1e-6, "e_R mismatch"
        assert np.max(np.abs(snap_orig.current_rpy - snap_cache.current_rpy)) < 1e-5, "rpy mismatch"
        assert snap_orig.m == snap_cache.m, "contact count mismatch"
        assert snap_orig.total_mass == snap_cache.total_mass, "total_mass mismatch"
```

- [ ] **Step 3: Run full snapshot test**

```bash
python -m pytest tests/test_phase3d3e_jax_dynamics_cache.py::TestFullCachedSnapshot -v
```

Expected: All assertions pass with `max_diff < 1e-6`.

- [ ] **Step 4: Commit**

```bash
git add wheeled_biped/wbc/phase3d3e_jax_dynamics_cache.py tests/test_phase3d3e_jax_dynamics_cache.py
git commit -m "feat(phase3d3e): add prepare_phase3b_snapshot_cached()

Full drop-in replacement using precompiled JAX functions.
Verified against original snapshot across all numeric fields.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Correctness Audit Script (8 Scenarios + QP Equivalence)

**Files:**
- Create: `scripts/phase3d3e_jax_dynamics_correctness_audit.py`

- [ ] **Step 1: Write the correctness audit script**

```python
#!/usr/bin/env python3
"""Phase 3D.3-E — JAX Dynamics Cache Correctness Audit.

Compares original prepare_phase3b_snapshot() against
prepare_phase3b_snapshot_cached() across 8 scenarios.

Also builds downstream QP matrices from both snapshots and
compares H, g, A_eq, b_eq, A_friction, b_friction.
"""
from __future__ import annotations

import json, sys, os, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "outputs" / "phase3d3e_jax_dynamics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_constants():
    from wheeled_biped.utils.config import get_model_path
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))

    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
    constants = build_qp_wbc_constants(model)

    from wheeled_biped.wbc.offline_qp_wbc import (
        _ensure_dynamics_constants, _ensure_contact_constants,
    )
    _ensure_dynamics_constants(constants)
    _ensure_contact_constants(constants)

    from wheeled_biped.wbc.offline_task_stack import _ensure_kinematics_constants_for_tasks
    _ensure_kinematics_constants_for_tasks(constants)

    # Inline contact extraction at default pose
    from wheeled_biped.wbc.offline_qp_wbc import _ensure_contact_constants
    _ensure_contact_constants(constants)
    contact_c = constants["_contact_constants"]
    qpos0 = np.array(model.keyframe("default").qpos, dtype=np.float64)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)
    wheel_body_ids = contact_c.get("wheel_body_ids", {})
    wheel_ids_set = set(int(v) for v in wheel_body_ids.values() if v >= 0)
    contacts_default = []
    for contact_id in range(data.ncon):
        c = data.contact[contact_id]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        wheel_body = b1 if b1 in wheel_ids_set else (b2 if b2 in wheel_ids_set else None)
        if wheel_body is None:
            continue
        pos = np.array(c.pos, dtype=np.float64)
        frame = np.array(c.frame, dtype=np.float64).reshape(3, 3)
        body_xpos = np.array(data.xpos[wheel_body], dtype=np.float64)
        body_xmat = np.array(data.xmat[wheel_body], dtype=np.float64).reshape(3, 3)
        local_point = body_xmat.T @ (pos - body_xpos)
        contacts_default.append({
            "body_id": int(wheel_body),
            "position": pos, "frame": frame, "local_point": local_point,
        })

    return model, constants, contacts_default


def build_scenarios(model, contacts_default):
    """Build 8 test scenarios."""
    qpos0 = np.array(model.keyframe("default").qpos, dtype=np.float64)
    qvel0 = np.zeros(model.nv, dtype=np.float64)
    rng = np.random.RandomState(42)

    scenarios = [
        {
            "name": "keyframe_static",
            "qpos": qpos0.copy(),
            "qvel": qvel0.copy(),
            "contacts": contacts_default,
        },
        {
            "name": "small_forward_velocity",
            "qpos": qpos0.copy(),
            "qvel": np.array([0.05, 0.0, 0.0] + [0.0]*13, dtype=np.float64),
            "contacts": contacts_default,
        },
        {
            "name": "small_lateral_velocity",
            "qpos": qpos0.copy(),
            "qvel": np.array([0.0, 0.05, 0.0] + [0.0]*13, dtype=np.float64),
            "contacts": contacts_default,
        },
        {
            "name": "small_yaw_rate",
            "qpos": qpos0.copy(),
            "qvel": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1] + [0.0]*10, dtype=np.float64),
            "contacts": contacts_default,
        },
        {
            "name": "small_roll_tilt",
            "qpos": _perturb_orientation(qpos0, roll=0.05),
            "qvel": qvel0.copy(),
            "contacts": contacts_default,
        },
        {
            "name": "small_pitch_tilt",
            "qpos": _perturb_orientation(qpos0, pitch=0.05),
            "qvel": qvel0.copy(),
            "contacts": contacts_default,
        },
        {
            "name": "deterministic_push_state",
            "qpos": qpos0.copy(),
            "qvel": np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0]*10, dtype=np.float64),
            "contacts": contacts_default,
        },
        {
            "name": "random_push_state",
            "qpos": qpos0.copy(),
            "qvel": np.array(list(rng.randn(6) * 0.2) + [0.0]*10, dtype=np.float64),
            "contacts": contacts_default,
        },
    ]
    return scenarios


def _perturb_orientation(qpos, roll=0.0, pitch=0.0):
    """Apply small roll/pitch perturbation to the torso quaternion."""
    import jax.numpy as jnp
    qpos_p = qpos.copy()
    # Simple: directly modify quaternion via small rotation
    from wheeled_biped.wbc.offline_task_stack import _quat_rotate
    # Use axis-angle to compute perturbation quaternion
    import numpy as np
    # This is approximate — rotate the existing quaternion
    q_torso = qpos_p[3:7]
    # Apply rotation about x (roll) and y (pitch)
    # For small angles: dq ≈ (1, roll/2, pitch/2, 0)
    half_roll = roll / 2.0
    half_pitch = pitch / 2.0
    # Hamilton product: q_new = q * dq
    w0, x0, y0, z0 = q_torso[0], q_torso[1], q_torso[2], q_torso[3]
    w1, x1, y1, z1 = 1.0, half_roll, half_pitch, 0.0
    norm = np.sqrt(w1*w1 + x1*x1 + y1*y1 + z1*z1)
    w1, x1, y1, z1 = w1/norm, x1/norm, y1/norm, z1/norm
    qpos_p[3] = w0*w1 - x0*x1 - y0*y1 - z0*z1
    qpos_p[4] = w0*x1 + x0*w1 + y0*z1 - z0*y1
    qpos_p[5] = w0*y1 - x0*z1 + y0*w1 + z0*x1
    qpos_p[6] = w0*z1 + x0*y1 - y0*x1 + z0*w1
    return qpos_p


def compare_snapshots(snap_orig, snap_cache, scenario_name):
    """Compare all numeric fields between two snapshots. Returns per-field diffs."""
    fields_to_compare = [
        ("M", snap_orig.M, snap_cache.M),
        ("h", snap_orig.h, snap_cache.h),
        ("S", snap_orig.S, snap_cache.S),
        ("Jcom", snap_orig.Jcom, snap_cache.Jcom),
        ("jdq_com", snap_orig.jdq_com, snap_cache.jdq_com),
        ("Jr", snap_orig.Jr, snap_cache.Jr),
        ("jdw_torso", snap_orig.jdw_torso, snap_cache.jdw_torso),
        ("e_R", snap_orig.e_R, snap_cache.e_R),
        ("current_rpy", snap_orig.current_rpy, snap_cache.current_rpy),
        ("jdot_qdot", snap_orig.jdot_qdot, snap_cache.jdot_qdot),
        ("com_position", snap_orig.com_position, snap_cache.com_position),
        ("omega_current", snap_orig.omega_current, snap_cache.omega_current),
    ]

    results = {}
    for name, a, b in fields_to_compare:
        if a.shape != b.shape:
            results[name] = {"match": False, "max_abs_diff": float("inf"),
                            "error": f"shape mismatch: {a.shape} vs {b.shape}"}
        else:
            max_diff = float(np.max(np.abs(a - b)))
            results[name] = {
                "match": max_diff < 1e-6,
                "max_abs_diff": max_diff,
                "shape": str(a.shape),
            }
    return results


def compare_qp_matrices(snap_orig, snap_cache, constants):
    """Build downstream QP matrices from both snapshots and compare."""
    from wheeled_biped.wbc.phase3b_cached_stack import build_phase3b_qp_from_snapshot

    qp_orig = build_phase3b_qp_from_snapshot(snap_orig, "balanced_default", constants)
    qp_cache = build_phase3b_qp_from_snapshot(snap_cache, "balanced_default", constants)

    qp_fields = [
        ("H", qp_orig.get("H"), qp_cache.get("H")),
        ("g", qp_orig.get("g"), qp_cache.get("g")),
        ("A_eq", qp_orig.get("A_eq"), qp_cache.get("A_eq")),
        ("b_eq", qp_orig.get("b_eq"), qp_cache.get("b_eq")),
    ]
    if qp_orig.get("A_friction") is not None:
        qp_fields.append(("A_friction", qp_orig["A_friction"], qp_cache.get("A_friction")))
        qp_fields.append(("b_friction", qp_orig["b_friction"], qp_cache.get("b_friction")))

    results = {}
    for name, a, b in qp_fields:
        if a is None or b is None:
            results[name] = {"match": a is b, "max_abs_diff": 0.0 if a is b else float("inf")}
        elif a.shape != b.shape:
            results[name] = {"match": False, "max_abs_diff": float("inf"),
                            "error": f"shape: {a.shape} vs {b.shape}"}
        else:
            max_diff = float(np.max(np.abs(a - b)))
            results[name] = {"match": max_diff < 1e-6, "max_abs_diff": max_diff, "shape": str(a.shape)}
    return results


def main():
    print("=" * 70)
    print("Phase 3D.3-E: JAX Dynamics Cache Correctness Audit")
    print("=" * 70)

    # ── Setup ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading model and constants...")
    model, constants, contacts_default = load_model_and_constants()

    from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import (
        initialize_jax_dynamics_cache,
        prepare_phase3b_snapshot_cached,
    )
    from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot

    print("[2/4] Initializing JAX dynamics cache...")
    t0 = time.perf_counter()
    cache = initialize_jax_dynamics_cache(model, constants, warmup=True)
    cache_init_time = time.perf_counter() - t0
    print(f"  Cache initialized in {cache_init_time:.1f}s")
    print(f"  JAX platform: {cache.jax_platform}")

    # ── Scenarios ──────────────────────────────────────────────────────
    scenarios = build_scenarios(model, contacts_default)
    print(f"\n[3/4] Running {len(scenarios)} scenarios...")

    all_results = []
    all_pass = True

    for i, sc in enumerate(scenarios):
        print(f"\n  Scenario {i+1}/{len(scenarios)}: {sc['name']}")

        snap_orig = prepare_phase3b_snapshot(
            sc["name"], sc["qpos"], sc["qvel"], sc["contacts"], constants,
        )
        snap_cache = prepare_phase3b_snapshot_cached(
            cache, sc["name"], sc["qpos"], sc["qvel"], sc["contacts"], constants,
        )

        field_diffs = compare_snapshots(snap_orig, snap_cache, sc["name"])
        qp_diffs = compare_qp_matrices(snap_orig, snap_cache, constants)

        scenario_pass = all(v["match"] for v in field_diffs.values()) and \
                        all(v["match"] for v in qp_diffs.values())

        if not scenario_pass:
            all_pass = False
            for name, d in field_diffs.items():
                if not d["match"]:
                    print(f"    FAIL {name}: diff={d.get('max_abs_diff', 'N/A')}")
            for name, d in qp_diffs.items():
                if not d["match"]:
                    print(f"    FAIL QP {name}: diff={d.get('max_abs_diff', 'N/A')}")
        else:
            print(f"    PASS")

        all_results.append({
            "scenario": sc["name"],
            "pass": scenario_pass,
            "field_diffs": {k: v["max_abs_diff"] for k, v in field_diffs.items()},
            "qp_diffs": {k: v["max_abs_diff"] for k, v in qp_diffs.items()},
            "max_snapshot_diff": max(
                v["max_abs_diff"] for v in field_diffs.values()
                if v["max_abs_diff"] != float("inf")
            ),
            "max_qp_diff": max(
                v["max_abs_diff"] for v in qp_diffs.values()
                if v["max_abs_diff"] != float("inf")
            ),
        })

    # ── Verdict ────────────────────────────────────────────────────────
    print(f"\n[4/4] Audit complete.")
    verdict = "JAX_DYNAMICS_CACHE_CORRECTNESS_PASS" if all_pass else "JAX_DYNAMICS_CACHE_CORRECTNESS_FAIL"
    print(f"  Verdict: {verdict}")

    report = {
        "phase": "3D.3-E",
        "verdict": verdict,
        "cache_init_time_s": cache_init_time,
        "jax_platform": cache.jax_platform,
        "jax_backend": cache.jax_backend,
        "scenarios": all_results,
        "summary": {
            "total": len(scenarios),
            "passed": sum(1 for r in all_results if r["pass"]),
            "failed": sum(1 for r in all_results if not r["pass"]),
        },
    }

    output_path = OUTPUT_DIR / "jax_dynamics_correctness.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report: {output_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the correctness audit**

```bash
python scripts/phase3d3e_jax_dynamics_correctness_audit.py
```

Expected: `JAX_DYNAMICS_CACHE_CORRECTNESS_PASS` with 8/8 scenarios passing.

- [ ] **Step 3: Commit**

```bash
git add scripts/phase3d3e_jax_dynamics_correctness_audit.py
git commit -m "feat(phase3d3e): add correctness audit script (8 scenarios + QP equivalence)

Compares original vs cached snapshot across 8 scenarios and
validates downstream QP matrix equivalence.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Benchmark Script

**Files:**
- Create: `scripts/phase3d3e_jax_dynamics_benchmark.py`
- Create: `tests/test_phase3d3e_jax_dynamics_benchmark_schema.py`

- [ ] **Step 1: Write schema validation tests**

```python
"""Schema validation for Phase 3D.3-E benchmark/diagnostic outputs."""
import json
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "phase3d3e_jax_dynamics"


def _load_json(path):
    with open(path) as f:
        return json.load(f)


class TestDiagnosticOutputSchema:
    """Validate jax_dynamics_diagnostic.json schema."""

    @pytest.fixture(scope="class")
    def diagnostic(self):
        path = OUTPUT_DIR / "jax_dynamics_diagnostic.json"
        if not path.exists():
            pytest.skip(f"Diagnostic output not found at {path}. Run diagnostic script first.")
        return _load_json(path)

    def test_has_environment(self, diagnostic):
        env = diagnostic.get("environment", {})
        assert "jax_platform" in env
        assert "jax_backend" in env
        assert "jax_enable_x64" in env
        assert "device_count" in env

    def test_has_sub_operation_timings(self, diagnostic):
        timings = diagnostic.get("sub_operation_timings_s", {})
        required = ["mass_matrix", "bias_forces", "contact_jacobian_per_contact",
                     "contact_jdot_qdot", "com_jacobian", "com_jdot_qdot",
                     "torso_ang_vel_jacobian", "torso_jdotw_qdot",
                     "torso_orientation_error", "full_snapshot"]
        for key in required:
            assert key in timings, f"Missing timing key: {key}"
            assert "time_s" in timings[key]

    def test_has_repeated_timings(self, diagnostic):
        repeated = diagnostic.get("repeated_same_state_timings_s", [])
        assert len(repeated) == 3

    def test_has_perturbed_timings(self, diagnostic):
        perturbed = diagnostic.get("perturbed_qpos_timings_s", [])
        assert len(perturbed) == 3


class TestCorrectnessOutputSchema:
    """Validate jax_dynamics_correctness.json schema."""

    @pytest.fixture(scope="class")
    def correctness(self):
        path = OUTPUT_DIR / "jax_dynamics_correctness.json"
        if not path.exists():
            pytest.skip(f"Correctness output not found at {path}. Run correctness audit first.")
        return _load_json(path)

    def test_has_verdict(self, correctness):
        assert "verdict" in correctness
        assert correctness["verdict"] in [
            "JAX_DYNAMICS_CACHE_CORRECTNESS_PASS",
            "JAX_DYNAMICS_CACHE_CORRECTNESS_FAIL",
        ]

    def test_has_scenarios(self, correctness):
        scenarios = correctness.get("scenarios", [])
        assert len(scenarios) >= 8

    def test_each_scenario_has_required_fields(self, correctness):
        for sc in correctness.get("scenarios", []):
            assert "scenario" in sc
            assert "pass" in sc
            assert "field_diffs" in sc
            assert "qp_diffs" in sc
            assert "max_snapshot_diff" in sc
            assert "max_qp_diff" in sc


class TestBenchmarkOutputSchema:
    """Validate jax_dynamics_benchmark.json schema."""

    @pytest.fixture(scope="class")
    def benchmark(self):
        path = OUTPUT_DIR / "jax_dynamics_benchmark.json"
        if not path.exists():
            pytest.skip(f"Benchmark output not found at {path}. Run benchmark script first.")
        return _load_json(path)

    def test_has_verdict(self, benchmark):
        assert "verdict" in benchmark

    def test_has_timing_sections(self, benchmark):
        assert "compile_time_s" in benchmark or "compile_time_ms" in benchmark
        assert "post_warmup" in benchmark

    def test_post_warmup_has_statistics(self, benchmark):
        pw = benchmark.get("post_warmup", {})
        if pw:
            assert "mean_ms" in pw or "mean_s" in pw

    def test_has_speedup(self, benchmark):
        assert "speedup_vs_original" in benchmark

    def test_has_environment(self, benchmark):
        env = benchmark.get("environment", {})
        assert "jax_platform" in env or "platform" in env
```

- [ ] **Step 2: Write benchmark script**

The benchmark script loads the model, initializes the cache, runs N scenarios × M steps, and reports timing with compile/ warm/post-warmup separation. Full implementation follows the same pattern as the diagnostic and audit scripts, with these additions:

```python
# Key benchmark measurements:
# - compile_time_s: time to initialize_jax_dynamics_cache(warmup=False)
# - warmup_time_s: time to run _warmup_cache()
# - first_call_time_s: first call to prepare_phase3b_snapshot_cached()
# - post_warmup_times: list of times for subsequent calls
# - original_times: list of times for prepare_phase3b_snapshot() calls
# - speedup = mean(original_times) / mean(post_warmup_times)
```

- [ ] **Step 3: Run benchmark schema tests (will skip if benchmark not yet run)**

```bash
python -m pytest tests/test_phase3d3e_jax_dynamics_benchmark_schema.py -v
```

Expected: Tests skip for missing benchmark output (not yet generated). Diagnostic and correctness tests run if outputs exist.

- [ ] **Step 4: Commit**

```bash
git add scripts/phase3d3e_jax_dynamics_benchmark.py tests/test_phase3d3e_jax_dynamics_benchmark_schema.py
git commit -m "feat(phase3d3e): add benchmark script and schema validation tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Integration — Incremental QP + Full-Batch Flags

**Files:**
- Modify: `wheeled_biped/wbc/phase3d3_incremental_qp.py`
- Modify: `scripts/phase3d_full_batch_execution.py`

- [ ] **Step 1: Add cached snapshot support to incremental QP**

In `wheeled_biped/wbc/phase3d3_incremental_qp.py`:

```python
# Add parameter to initialize_incremental_qp_workspace:
#   jax_dynamics_cache: Any = None  # optional JAXDynamicsCache

# In the snapshot-building section, replace:
#   snapshot = prepare_phase3b_snapshot(...)
# With:
#   if jax_dynamics_cache is not None:
#       from .phase3d3e_jax_dynamics_cache import prepare_phase3b_snapshot_cached
#       snapshot = prepare_phase3b_snapshot_cached(
#           jax_dynamics_cache, "wbc_init", qpos0, qvel0, contacts0, qp_c,
#           max_contacts=max_contacts,
#       )
#   else:
#       snapshot = prepare_phase3b_snapshot(...)

# Same pattern in update_incremental_qp_workspace().
# Add jax_dynamics_cache to IncrementalQPWorkspace dataclass as optional field.
```

- [ ] **Step 2: Add CLI flags to full-batch executor**

In `scripts/phase3d_full_batch_execution.py`:

```python
# Add argparse flags:
parser.add_argument("--use-jax-dynamics-cache", action="store_true",
    help="Use precompiled JAX dynamics cache for snapshot preparation")
parser.add_argument("--jax-dynamics-cache-max-contacts", type=int, default=4,
    help="Max contacts for JAX dynamics cache padding")
parser.add_argument("--jax-dynamics-warmup", action="store_true", default=True,
    help="Warm up JAX dynamics functions during cache init")
parser.add_argument("--jax-dynamics-diagnostic", action="store_true",
    help="Print JAX dynamics cache diagnostics after run")
```

Wire the cache into the full-batch pipeline:
```python
if args.use_jax_dynamics_cache:
    from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import initialize_jax_dynamics_cache
    jax_dynamics_cache = initialize_jax_dynamics_cache(
        model, constants,
        max_contacts=args.jax_dynamics_cache_max_contacts,
        warmup=args.jax_dynamics_warmup,
    )
    # Pass to incremental QP init and update calls
else:
    jax_dynamics_cache = None
```

- [ ] **Step 3: Verify default path unchanged**

```bash
python scripts/phase3d_full_batch_execution.py --quick
```

Expected: Runs as before, no JAX dynamics cache involved.

- [ ] **Step 4: Test with cache enabled**

```bash
python scripts/phase3d_full_batch_execution.py --use-incremental-qp --use-jax-dynamics-cache --quick
```

Expected: Runs with cached dynamics, metadata records `jax_dynamics_cache_enabled: true`.

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/wbc/phase3d3_incremental_qp.py scripts/phase3d_full_batch_execution.py
git commit -m "feat(phase3d3e): add --use-jax-dynamics-cache to incremental QP + full-batch

Opt-in only. Default path unchanged. Metadata records cache usage.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: End-to-End Benchmark Run + Report

**Files:**
- Create: `docs/validation/k2_phase3d3e_jax_dynamics_report.md`

- [ ] **Step 1: Run the full benchmark**

```bash
python scripts/phase3d3e_jax_dynamics_benchmark.py --states 8 --steps 20
```

- [ ] **Step 2: Run full-batch with cache + incremental QP**

```bash
python scripts/phase3d_full_batch_execution.py --use-incremental-qp --use-jax-dynamics-cache --quick
```

- [ ] **Step 3: Run baseline truth check**

```bash
python scripts/phase3d_v3_baseline_truth_check.py
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/test_phase3d3e_jax_dynamics_cache.py tests/test_phase3d3e_jax_dynamics_benchmark_schema.py -v
```

- [ ] **Step 5: Write validation report**

Create `docs/validation/k2_phase3d3e_jax_dynamics_report.md` with:
1. Executive summary
2. Exact branch and commit SHA
3. Files changed
4. Root cause of JAX slowness
5. JAXDynamicsCache architecture
6. Correctness audit results
7. Performance benchmark (compile time, post-warmup mean/p95, speedup)
8. Recompile count, fallback count
9. Integration status
10. Controller integrity confirmation
11. What this means / does not mean
12. Recommended next phase
13. Final verdict summary

- [ ] **Step 6: Commit**

```bash
git add docs/validation/k2_phase3d3e_jax_dynamics_report.md
git commit -m "docs(phase3d3e): add JAX dynamics cache validation report

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Pass/Fail Criteria Per Stage

| Stage | Pass Criteria | Fail Action |
|---|---|---|
| E1 | Diagnostic produces valid JSON, identifies slow functions | Fix script, re-run |
| E2 | All 8 cache-init tests pass, M/h correctness < 1e-6 | Debug JIT closure; check array dtypes |
| E3 | All 6 COM/torso correctness tests pass, diff < 1e-6 | Debug jacfwd closure; check qpos→qvel conversion |
| E4 | All 4 contact tests pass, diff < 1e-6 | If JIT errors: document, set `PARTIAL_JAX_DYNAMICS_CACHE`, continue with contact fallback |
| E5 | Full snapshot comparison passes, diff < 1e-6 | Debug field-by-field; check coordinate conventions |
| E6 | Default path unchanged, `--use-jax-dynamics-cache` works | Fix flag gating |
| E7 | All tests pass, benchmark produces valid report, baseline truth check passes | Fix report; re-run |

## Fallback Behavior

- If contact JIT (Stage E4) fails: contact Jacobian and Jdot*qdot fall back to original functions. Cache records `fallback_count`. Report `PARTIAL_JAX_DYNAMICS_CACHE`.
- If `prepare_phase3b_snapshot_cached()` encounters > 4 contacts: falls back to original `prepare_phase3b_snapshot()`. Cache records `fallback_count`.
- If correctness fails at any stage: STOP. Do not proceed to next stage. Report `JAX_DYNAMICS_CACHE_CORRECTNESS_FAIL`.

## Non-Goals

- JITting the full `prepare_phase3b_snapshot()` as one function
- GPU optimization
- Realtime integration
- Controller integration (K2 V3)
- WBC promotion
- Removing the original `prepare_phase3b_snapshot()`
- Making cached dynamics the default path
