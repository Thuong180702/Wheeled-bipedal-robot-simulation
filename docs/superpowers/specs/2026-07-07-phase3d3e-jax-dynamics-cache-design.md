# Phase 3D.3-E — JAX Dynamics Cache Design

Date: 2026-07-07
Branch: repo-cleanup-t6j
Commit: 6b6feba

## 1. Problem Statement

`prepare_phase3b_snapshot()` takes ~300s per novel state on CPU-only Windows.
This dominates both full-rebuild and incremental QP paths, masking QP-level
improvements from Phase 3D.3.

**Root cause**: None of the JAX dynamics/Jacobian functions called inside
`prepare_phase3b_snapshot()` are `@jax.jit` decorated. Each call executes in
JAX eager/traced mode, which traces every operation from scratch. The
`jax.jacfwd` calls (used for COM and torso Jacobians) are especially expensive
because each creates a new forward-mode AD trace over non-jitted FK functions.

The codebase already has `_fk_arrays` variants designed for JIT
(`extract_jax_fk_arrays`, `jax_mass_matrix_fk_arrays`,
`jax_bias_forces_fk_arrays`) but these were never actually jitted.

## 2. Approach

**Approach A — JIT individual array-only dynamics/Jacobian functions.**

Do NOT JIT the entire `prepare_phase3b_snapshot()` — it takes Python dicts,
lists, and contact objects that would break JIT. Instead, JIT each dynamics
function individually, using array-only inputs.

### 2.1 Boundary

```text
Python outside JIT:
  - parse contacts
  - map body IDs
  - pad contacts to max_contacts
  - build fixed-shape arrays
  - select active contact slots

JAX inside JIT:
  - qpos (nq,) array
  - qvel (nv,) array
  - fixed-shape contact arrays
  - fixed-shape constants arrays (fk_arrays tuples)
```

### 2.2 Fixed contact representation

```text
max_contacts = 4

contact_active:     (4,)      bool or int
contact_body_id:    (4,)      int32
contact_point:      (4, 3)    float64
contact_normal:     (4, 3)    float64
contact_frame:      (4, 3, 3) float64
```

Inactive slots are zeroed and skipped via `contact_active`.

## 3. Architecture

### 3.1 New module: `wheeled_biped/wbc/phase3d3e_jax_dynamics_cache.py`

```python
class JAXDynamicsCache:
    """Owns pre-extracted array constants and precompiled JAX functions."""

    # Pre-extracted array tuples
    fk_arrays: tuple
    mm_arrays: tuple
    bias_arrays: tuple
    contact_arrays: tuple

    # Jitted dynamics functions
    mass_matrix_jit: Callable
    bias_forces_jit: Callable
    contact_jacobian_batch_jit: Callable
    com_jacobian_jit: Callable
    com_jdot_qdot_jit: Callable
    torso_ang_vel_jacobian_jit: Callable
    torso_jdotw_qdot_jit: Callable
    torso_orientation_error_jit: Callable
    contact_jdot_qdot_batch_jit: Callable

    # Diagnostics
    compile_time_s: float
    warmup_time_s: float
    call_count: int
    recompile_count: int

def initialize_jax_dynamics_cache(
    model, constants, *,
    max_contacts=4, dtype="float64", warmup=True,
) -> JAXDynamicsCache: ...

def prepare_phase3b_snapshot_cached(
    cache, scenario_name, qpos, qvel, contacts, constants, *,
    max_contacts=4,
) -> Phase3BSnapshot: ...
```

### 3.2 Jitted functions

| Function | Construction | Inputs |
|---|---|---|
| `mass_matrix_jit` | `jax.jit(jax_mass_matrix_fk_arrays)` | qpos, fk_arrays, mm_arrays |
| `bias_forces_jit` | `jax.jit(jax_bias_forces_fk_arrays)` | qpos, qvel, fk_arrays, bias_arrays |
| `com_jacobian_jit` | `jax.jit(jax.jacfwd(com_position_fn))` | qpos, fk_arrays, body_mass, body_ipos |
| `torso_ang_vel_jacobian_jit` | `jax.jit(jax.jacfwd(torso_angvel_fn))` | qpos, fk_arrays |
| `torso_orientation_error_jit` | `jax.jit(torso_orientation_error_fn)` | qpos, fk_arrays |
| `contact_jacobian_batch_jit` | `jax.jit(jax.vmap(contact_jacobian_fn))` | qpos, body_ids, local_points, contact_arrays, active_mask |
| `com_jdot_qdot_jit` | `jax.jit(com_jdot_qdot_fn)` | qpos, qvel, fk_arrays, body_mass, body_ipos, eps |
| `torso_jdotw_qdot_jit` | `jax.jit(torso_jdotw_qdot_fn)` | qpos, qvel, fk_arrays, eps |
| `contact_jdot_qdot_batch_jit` | `jax.jit(jax.vmap(contact_jdot_qdot_fn))` | qpos, qvel, body_ids, local_points, contact_arrays, active_mask, eps |

All `jax.jit` and `jax.jacfwd` construction happens ONCE in `initialize_jax_dynamics_cache()`.

### 3.3 `prepare_phase3b_snapshot_cached()` flow

```text
1. Convert contacts list → fixed-shape padded arrays (Python, outside JIT)
2. Call mass_matrix_jit(qpos) → M
3. Call bias_forces_jit(qpos, qvel) → h
4. Call contact_jacobian_batch_jit(qpos, padded_contacts) → Jp, JcT
5. Call contact_jdot_qdot_batch_jit(qpos, qvel, padded_contacts) → jdq
6. Call com_jacobian_jit(qpos) → Jcom
7. Call com_jdot_qdot_jit(qpos, qvel) → jdq_com
8. Call torso_ang_vel_jacobian_jit(qpos) → Jr
9. Call torso_jdotw_qdot_jit(qpos, qvel) → jdw_torso
10. Call torso_orientation_error_jit(qpos) → e_R, current_rpy
11. Build Phase3BSnapshot (Python, outside JIT)
12. Return snapshot
```

## 4. Implementation Stages

### Stage 3D.3-E1: Diagnostic Script
- `scripts/phase3d3e_jax_dynamics_diagnostic.py`
- Measure per-function timing, detect recompilation
- Output: `outputs/phase3d3e_jax_dynamics/jax_dynamics_diagnostic.json`

### Stage 3D.3-E2: Mass Matrix + Bias Force JIT
- JIT `jax_mass_matrix_fk_arrays` and `jax_bias_forces_fk_arrays`
- Correctness audit for M and h only

### Stage 3D.3-E3: COM + Torso Jacobian JIT
- Pre-build `jax.jacfwd` functions in cache init
- JIT the Jacobian functions
- Correctness audit for Jcom, Jr, orientation error

### Stage 3D.3-E4: Contact Jacobian + Jdot*qdot JIT
- Batched contact Jacobian via `jax.vmap`
- Batched Jdot*qdot
- Correctness audit for contact arrays

### Stage 3D.3-E5: Full `prepare_phase3b_snapshot_cached()`
- Assemble all jitted calls
- Return identical `Phase3BSnapshot` structure
- Full correctness audit

### Stage 3D.3-E6: Incremental QP Integration
- Add `--use-jax-dynamics-cache` flag
- Wire cached snapshot into `IncrementalQPWorkspace`

### Stage 3D.3-E7: End-to-End Benchmark
- Benchmark incremental QP + cached dynamics
- Full-batch quick run

## 5. Correctness Requirements

Audit script: `scripts/phase3d3e_jax_dynamics_correctness_audit.py`

8 test cases:
1. keyframe_static
2. small_forward_velocity
3. small_lateral_velocity
4. small_yaw_rate
5. small_roll_tilt
6. small_pitch_tilt
7. deterministic_push_state
8. random_push_state

Compare all numeric ndarray fields in `Phase3BSnapshot` recursively.
Pass threshold: `max_abs_diff <= 1e-6`.

## 6. Performance Targets

| Target | Threshold |
|---|---|
| Strong | post_warmup_mean < 30ms, p95 < 50ms → `REALTIME_CANDIDATE_INFRASTRUCTURE` |
| Acceptable | post_warmup_mean < 100ms → `CLOSED_LOOP_EVALUATION_UNBLOCKED` |
| Partial | speedup >= 20× → `JAX_DYNAMICS_PARTIAL_SPEEDUP` |
| Failure | recompile > 0 or speedup < 20× → `JAX_DYNAMICS_BOTTLENECK_UNRESOLVED` |

## 7. Constraints

- Do NOT modify K2 V3 controller
- Do NOT promote WBC
- Do NOT use `REALTIME_READY` verdict
- Original `prepare_phase3b_snapshot()` preserved unchanged
- Cache path opt-in only via `--use-jax-dynamics-cache` flag
- Report compile time and runtime separately

## 8. Files

| File | Action |
|---|---|
| `wheeled_biped/wbc/phase3d3e_jax_dynamics_cache.py` | CREATE |
| `scripts/phase3d3e_jax_dynamics_diagnostic.py` | CREATE |
| `scripts/phase3d3e_jax_dynamics_correctness_audit.py` | CREATE |
| `scripts/phase3d3e_jax_dynamics_benchmark.py` | CREATE |
| `tests/test_phase3d3e_jax_dynamics_cache.py` | CREATE |
| `tests/test_phase3d3e_jax_dynamics_benchmark_schema.py` | CREATE |
| `docs/validation/k2_phase3d3e_jax_dynamics_report.md` | CREATE |
| `wheeled_biped/wbc/phase3d3_incremental_qp.py` | MODIFY (add flag) |
| `scripts/phase3d_full_batch_execution.py` | MODIFY (add flag) |

## 9. Test Commands

```bash
python -m pytest tests/test_phase3d3e_jax_dynamics_cache.py -v
python -m pytest tests/test_phase3d3e_jax_dynamics_benchmark_schema.py -v
python scripts/phase3d3e_jax_dynamics_diagnostic.py
python scripts/phase3d3e_jax_dynamics_correctness_audit.py
python scripts/phase3d3e_jax_dynamics_benchmark.py --states 8 --steps 20
python scripts/phase3d_full_batch_execution.py --use-incremental-qp --use-jax-dynamics-cache --quick
python scripts/phase3d_v3_baseline_truth_check.py
```
