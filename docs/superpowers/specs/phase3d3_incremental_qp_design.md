# Phase 3D.3 — Incremental QP / Persistent OSQP Workspace Design

**Date:** 2026-07-06
**Branch:** `repo-cleanup-t6j`
**Commit (base):** `c2f4b19`
**Status:** Design approved, pending implementation

---

## 1. Problem Statement

Phase 3D FULL_BATCH_EXECUTION was blocked with verdict `FULL_BATCH_BLOCKED_QP_BUILD_BOTTLENECK`. The three-arm counterfactual evaluation (V3_BASELINE vs WBC_ONLY vs V3_PLUS_WBC_ASSIST) cannot be executed because each simulation step takes ~17 seconds, making 5,000-step rollouts infeasible (~24 hours per scenario, ~224 days for the full 225-scenario batch).

## 2. Root Cause

The bottleneck is **not** the OSQP solver. OSQP solve time is ~0.16 ms.

The bottleneck is the upstream QP construction pipeline, which rebuilds the full QP problem from scratch every simulation step:

| Component | Mean Time |
|-----------|-----------|
| `prepare_phase3b_snapshot()` | ~8,000 ms |
| `build_phase3b_qp_from_snapshot()` (×2 calls) | ~4,000 ms |
| `build_phase3c_qp_from_snapshot()` | ~2,000 ms |
| `build_structured_qp_from_phase3c_snapshot()` | ~2,000 ms |
| OSQP solve | 0.16 ms |
| **Total** | **~16,200 ms** |

Additionally, `build_structured_qp_from_phase3c_snapshot()` calls `build_phase3b_qp_from_snapshot()` **twice** — once for the objective (H/g) and once for constraints (A_eq/b_eq/A_friction) — doubling the cost of that component.

Furthermore, the existing `OSQPSolverBackend._update_numeric()` only updates `q`, `l`, `u` — it does **not** update `P` or `A` numeric values. When problem dimensions stay the same (as they do with padded contacts), P and A matrices inside OSQP become **stale** after the first solve.

## 3. Current Call Chain

```
compute_wbc_torque_for_state()                    [offline_three_arm_counterfactual.py:514]
  └─ prepare_phase3b_snapshot()                   [phase3b_cached_stack.py]
  └─ build_phase3c_qp_from_snapshot()             [phase3c_rolling_qp.py:40]
  └─ solve_phase3c_fast()                         [phase3d2_fast_solver.py:53]
       └─ build_structured_qp_from_phase3c_snapshot()  [structured_qp_problem.py:91]
            └─ build_phase3b_qp_from_snapshot()    [CALLED TWICE: lines 212, 268]
            └─ _build_sparse_objective()           [P, q construction]
            └─ _build_unified_constraints()        [A, l, u construction]
            └─ _build_variable_bounds()            [lb, ub construction]
       └─ solve_structured_qp()                   [qp_solver_backends.py:639]
            └─ OSQPSolverBackend.solve()           [setup or _update_numeric + solve]
```

Every step rebuilds: dynamics linearization, contact Jacobians, dense H/g/A_eq/A_friction matrices, sparse P/q/A/l/u/lb/ub, and the dense-to-CSC conversion.

## 4. Why OSQP Solve Time Is NOT the Bottleneck

- OSQP solve: 0.16 ms mean, 0.16 ms P95
- QP build: 16,200 ms mean, 21,000 ms P95
- Ratio: solve is 0.001% of total step time

The Phase 3D.2 fast solver works correctly. The problem is upstream.

## 5. Why Current q/l/u-Only Update Is Insufficient

`OSQPSolverBackend._update_numeric()` calls:
```python
self._solver.update(q=problem.q, l=l_clipped, u=u_clipped)
```

It does **not** update `Px` or `Ax`. Since `P` contains dynamics-dependent terms (M(q), task costs) and `A` contains Jacobian-dependent terms (contact, friction, rolling constraints), the OSQP workspace holds **stale P and A matrices** after the first solve when dimensions don't change.

This means:
- Dynamics changes (different qpos/qvel) are not reflected in the KKT system's P and A blocks
- Only q (linear cost) and l/u (bounds) track the current state
- The QP is solving a fundamentally wrong problem that mixes old dynamics with new bounds

## 6. Target Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   IncrementalQPWorkspace                      │
│                                                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │ Cached           │  │ QPBlockMetadata  │  │ Persistent   │ │
│  │ StructuredQP     │  │                  │  │ OSQP Backend │ │
│  │ Problem          │  │ P_blocks:        │  │              │ │
│  │                  │  │  dynamics, task, │  │ setup()      │ │
│  │ P (CSC, fixed)   │  │  rolling, reg    │  │ update(q,l,u,│ │
│  │ q, l, u, lb, ub  │  │                  │  │   Px, Ax)    │ │
│  │                  │  │ A_blocks:        │  │ warm_start() │ │
│  │ variable_slices  │  │  dynamics,       │  │ solve()      │ │
│  │ constraint_slices│  │  contact_normal, │  │              │ │
│  │                  │  │  friction,       │  │ diagnostics  │ │
│  └─────────────────┘  │  rolling_hard    │  └─────────────┘ │
│                        └──────────────────┘                  │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Warm-start: x_prev (nx,), y_prev (nc,)                   ││
│  │ State tracking: prev_qpos, prev_qvel, prev_contacts      ││
│  │ Counters: setup, update, solve, reinit, fallback          ││
│  │ Timing accumulators per sub-component                    ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

**Initialization (once):**
```
snapshot = prepare_phase3b_snapshot(qpos0, qvel0, contacts0)
structured_qp, block_metadata = build_structured_qp_from_phase3c_snapshot(
    snapshot, ..., return_block_metadata=True)
persistent_backend.setup(structured_qp)
x_warm = zeros(nx), y_warm = zeros(nc)
```

**Per-step (fast path):**
```
snapshot = prepare_phase3b_snapshot(qpos, qvel, contacts)   [Stage B: once only]
patch_P_blocks(cached_P.data, block_metadata, snapshot)     [Stage C: direct CSC]
patch_A_blocks(cached_A.data, block_metadata, snapshot)     [Stage C: direct CSC]
patch_q(cached_q, block_metadata, snapshot)
patch_l_u(cached_l, cached_u, block_metadata, snapshot)
persistent_backend.update(q, l, u, Px=P_data, Ax=A_data)
persistent_backend.warm_start(x=x_warm, y=y_warm)
result = persistent_backend.solve()
x_warm = result.x; y_warm = result.y
```

## 7. PersistentOSQPBackend Design

**New file:** `wheeled_biped/wbc/persistent_osqp_backend.py`

Separate from existing `OSQPSolverBackend` to keep Phase 3D.2 stable.

```python
class PersistentOSQPBackend:
    """Persistent OSQP solver with full numeric update capability.

    Key differences from OSQPSolverBackend:
    - setup() is explicit, counted, and tracked
    - update() accepts ALL 5 numeric arrays: q, l, u, Px, Ax
    - warm_start() accepts dual (y) in addition to primal (x)
    - Diagnostics track: setup_count, update_count, solve_count, reinit_count
    - Exposes whether last solve used fresh_setup, numeric_update, warm_start
    """

    def __init__(self, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000,
                 polish=True, warm_starting=True, adaptive_rho=True,
                 verbose=False):
        self._solver: osqp.OSQP | None = None
        self._setup_count = 0
        self._update_count = 0
        self._solve_count = 0
        self._reinit_count = 0
        self._last_nx = -1
        self._last_nc = -1
        self._last_solve_used_fresh_setup = False
        self._last_solve_used_warm_start_primal = False
        self._last_solve_used_warm_start_dual = False
        self._last_update_had_Px = False
        self._last_update_had_Ax = False
        # Timing accumulators
        self._cumulative_setup_time_s = 0.0
        self._cumulative_update_time_s = 0.0
        self._cumulative_solve_time_s = 0.0

    def setup(self, problem: StructuredQPProblem) -> float:
        """Full OSQP initialization. Called once unless structure changes.
        Returns setup wall time in seconds.
        """

    def update(self, *, q=None, l=None, u=None, Px=None, Ax=None) -> float:
        """Update numeric values in persistent workspace.
        All args optional — only provided arrays are updated.
        OSQP sparsity pattern is reused.
        Px/Ax updates may trigger internal numeric refactorization,
        but the setup/analyze phase is avoided.
        Returns update wall time in seconds.
        """

    def warm_start(self, *, x=None, y=None) -> None:
        """Set primal and/or dual warm-start vectors."""

    def solve(self) -> QPSolution:
        """Solve with current workspace state. Returns QPSolution."""

    def needs_reinit(self, problem: StructuredQPProblem) -> bool:
        """Check if problem dimensions changed, requiring full re-setup."""

    @property
    def diagnostics(self) -> dict:
        """Return full diagnostics snapshot."""

    def close(self) -> None:
        """Clean up solver resources."""
```

**Required diagnostics:**
```python
{
    "setup_count": int,
    "update_count": int,
    "solve_count": int,
    "reinit_count": int,
    "last_solve_used_fresh_setup": bool,
    "last_solve_used_warm_start_primal": bool,
    "last_solve_used_warm_start_dual": bool,
    "last_update_had_Px": bool,
    "last_update_had_Ax": bool,
    "last_solve_status": str,
    "cumulative_setup_time_s": float,
    "cumulative_update_time_s": float,
    "cumulative_solve_time_s": float,
    "last_setup_time_s": float,
    "last_update_time_s": float,
    "last_solve_time_s": float,
}
```

**OSQP refactorization note:** Updating `Px` and `Ax` may cause OSQP to perform internal numeric refactorization. This is still far cheaper than rebuilding the entire QP structure in Python, but we do not claim "no refactorization." We claim: "OSQP workspace is persistent; the sparsity pattern is reused; the OSQP setup/analyze phase is avoided."

## 8. QPBlockMetadata Design

**Added to:** `wheeled_biped/wbc/structured_qp_problem.py`

```python
@dataclass
class QPBlockMetadata:
    """Records where each semantic block lands in the CSC data arrays.

    Enables direct per-step patching without rebuilding the full QP.
    """
    # Per-block CSC data indices for P
    P_blocks: dict[str, dict]
    # Per-block CSC data indices for A
    A_blocks: dict[str, dict]
    # Per-block indices for q vector
    q_indices: dict[str, slice]
    # Per-block indices for l/u vectors
    l_indices: dict[str, slice]
    u_indices: dict[str, slice]
    # Problem dimensions
    nx: int
    nc: int
    nv: int
    nu: int
    n_lambda: int
    k_slack: int
    max_contacts: int
    # Sparsity signatures (for structure-change detection)
    p_nnz: int
    a_nnz: int
    p_indptr_shape: tuple
    a_indptr_shape: tuple
```

**Method for extracting block metadata:**

`build_structured_qp_from_phase3c_snapshot()` is extended with a `return_block_metadata: bool = False` parameter. When True, it returns `(StructuredQPProblem, QPBlockMetadata)`.

The metadata is constructed **during** the original QP build, using known row/column coordinate ranges for each semantic block (dynamics, contact, friction, rolling, tasks, regularization). CSC data indices are tracked by recording the start/end positions in the data array as each block is inserted.

**This is NOT probe-based.** Block metadata comes from the construction process itself — each block's row range, column range, and CSC insertion coordinates are known at build time.

**If the existing builder does not expose this metadata natively:** Stage 3D.3-B implements a helper that constructs the CSC matrices from known coordinate triples (row, col, value) grouped by block, recording insertion positions for each block. This is more robust than post-hoc probing.

## 9. IncrementalQPWorkspace Design

**New file:** `wheeled_biped/wbc/phase3d3_incremental_qp.py`

```python
@dataclass
class IncrementalQPWorkspace:
    """Persistent QP workspace for repeated WBC solves across timesteps."""

    # Cached QP structure (built once)
    structured_qp: StructuredQPProblem
    block_metadata: QPBlockMetadata

    # Persistent solver
    backend: PersistentOSQPBackend

    # Warm-start state
    x_warm: np.ndarray | None = None   # primal (nx,)
    y_warm: np.ndarray | None = None   # dual (nc,)

    # State tracking (for delta computation)
    previous_qpos: np.ndarray | None = None
    previous_qvel: np.ndarray | None = None
    previous_contacts: list | None = None

    # Configuration (immutable after init)
    max_contacts: int = 4
    task_mode: str = "balanced_default"
    rolling_mode: str = "full_rolling_soft"
    constants: dict | None = None
    model: Any = None

    # Diagnostics — counters
    setup_count: int = 0
    update_count: int = 0
    solve_count: int = 0
    reinit_count: int = 0
    fallback_full_rebuild_count: int = 0
    workspace_reinit_required: bool = False

    # Diagnostics — state
    last_active_contact_slots: int = 0
    last_update_mode: str = "none"
    structure_signature: dict | None = None
    p_sparsity_signature: str = ""
    a_sparsity_signature: str = ""

    # Diagnostics — timing accumulators (seconds)
    cumulative_snapshot_time_s: float = 0.0
    cumulative_block_update_time_s: float = 0.0
    cumulative_csc_patch_time_s: float = 0.0
    cumulative_osqp_update_time_s: float = 0.0
    cumulative_osqp_solve_time_s: float = 0.0
```

**Key functions:**

```python
def initialize_incremental_qp_workspace(
    model, qpos0, qvel0, contacts0,
    task_mode: str, rolling_mode: str, constants: dict,
    *, max_contacts: int = 4, backend_name: str = "osqp",
    eps_abs: float = 1e-5, eps_rel: float = 1e-5, max_iter: int = 4000,
) -> IncrementalQPWorkspace:
    """Build the full QP structure once and initialize persistent solver.
    
    Steps:
    1. Build Phase3B snapshot from initial state
    2. Build StructuredQPProblem + QPBlockMetadata (full build, once)
    3. Initialize PersistentOSQPBackend with full P, q, A, l, u
    4. Store warm-start vectors (zeros)
    5. Record structure signatures for change detection
    """

def update_incremental_qp_workspace(
    workspace: IncrementalQPWorkspace,
    qpos: np.ndarray, qvel: np.ndarray, contacts: list,
) -> dict[str, Any]:
    """Per-step update: rebuild snapshot, patch blocks, update backend.
    
    Steps:
    1. Check contact topology vs max_contacts padding
    2. Prepare Phase3B snapshot (may be full rebuild in Stage A/B,
       incremental in Stage C)
    3. Update dense blocks for qpos/qvel-dependent terms
    4. Patch CSC data arrays at precomputed indices
    5. Update q, l, u vectors
    6. Call backend.update(Px=P_data, Ax=A_data, q=q, l=l, u=u)
    7. Return timing diagnostics
    
    Does NOT call build_structured_qp_from_phase3c_snapshot()
    in the hot path.
    """

def solve_incremental_qp(
    workspace: IncrementalQPWorkspace,
    *, warm_start: bool = True,
) -> dict[str, Any]:
    """Solve using persistent workspace.
    
    Steps:
    1. If warm_start and x_warm/y_warm exist, call backend.warm_start()
    2. Call backend.solve()
    3. Store solution as next warm-start
    4. Compute residuals
    5. Return tau, qdd, lambda, residuals, status, timing
    """

def compute_wbc_torque_incremental_for_state(
    mj_data, model,
    workspace: IncrementalQPWorkspace,
    constants: dict,
    controller_context: dict,
) -> dict[str, Any]:
    """Drop-in replacement for compute_wbc_torque_for_state().
    
    Uses the incremental QP path instead of full rebuild.
    Preserves the same return dict structure for compatibility
    with existing three-arm infrastructure.
    
    Does NOT modify compute_wbc_torque_for_state().
    """
```

## 10. Staged Implementation Plan

### Stage 3D.3-A — Persistent OSQP Backend + Stale P/A Fix

**Goal:** Create `PersistentOSQPBackend` with full q/l/u/Px/Ax update support. Prove that P and A updates work correctly.

**Deliverables:**
- `wheeled_biped/wbc/persistent_osqp_backend.py` — new backend class
- Tests in `tests/test_phase3d3_incremental_qp.py`:
  - `test_persistent_backend_setup_and_solve`
  - `test_persistent_backend_update_q_l_u`
  - `test_persistent_backend_update_Px_Ax`
  - `test_persistent_backend_warm_start_primal_and_dual`
  - `test_persistent_backend_needs_reinit_dimension_change`
  - `test_persistent_backend_diagnostics_counters`
  - `test_stale_PA_detection_fails_with_old_backend`
  - `test_stale_PA_detection_passes_with_new_backend`

**Validation criteria:**
- `PersistentOSQPBackend` can setup/solve/update q/l/u/Px/Ax
- Warm-start primal and dual are supported
- Diagnostics counters are correct
- Stale P/A detection test fails on q/l/u-only path and passes on Px/Ax path
- Existing Phase 3D.2 backend unchanged; `test_phase3d2` tests still pass

### Stage 3D.3-B — Deduplicate Phase 3B QP Build + Extract Block Metadata

**Goal:** Call `build_phase3b_qp_from_snapshot()` once instead of twice inside `build_structured_qp_from_phase3c_snapshot()`. Add `QPBlockMetadata` extraction. Add `return_block_metadata` flag.

**Deliverables:**
- Modify `structured_qp_problem.py`:
  - Add `QPBlockMetadata` dataclass
  - Add `build_phase3b_qp_once()` helper
  - Refactor `_build_sparse_objective()` and `_build_unified_constraints()` to accept pre-built `qp_3b`
  - Extend `build_structured_qp_from_phase3c_snapshot()` with `return_block_metadata`
- Tests:
  - `test_build_phase3b_qp_called_once_not_twice`
  - `test_block_metadata_extraction_valid`
  - `test_block_metadata_covers_all_blocks`
  - `test_full_rebuild_with_metadata_equals_without_metadata`
  - `test_phase3d2_regression` (all Phase 3D.2 tests still pass)

**Validation:**
- `build_phase3b_qp_from_snapshot()` is called once, not twice
- Existing full rebuild results remain numerically equivalent
- `QPBlockMetadata` correctly maps all semantic blocks to CSC indices
- Phase 3D.2 tests still pass

### Stage 3D.3-C — IncrementalQPWorkspace + Block-Level CSC Patching

**Goal:** Implement the full `IncrementalQPWorkspace` with block-level CSC patching. Achieve correctness parity with full rebuild path.

**Deliverables:**
- `wheeled_biped/wbc/phase3d3_incremental_qp.py` — full workspace implementation
- Correctness audit script: `scripts/phase3d3_incremental_qp_correctness_audit.py`
- Benchmark script: `scripts/phase3d3_incremental_qp_benchmark.py`
- Tests:
  - `test_incremental_workspace_initializes`
  - `test_fixed_sparsity_pattern_preserved`
  - `test_osqp_workspace_not_recreated_every_step`
  - `test_warm_start_used_after_first_step`
  - `test_incremental_update_returns_finite_tau`
  - `test_incremental_and_full_rebuild_tau_match`
  - `test_workspace_reinit_recorded_if_required`
  - `test_benchmark_output_schema_valid`
  - `test_PA_not_stale_after_qpos_change`
  - `test_controller_integrity_unchanged`

**Validation:**
- `IncrementalQPWorkspace` produces same tau/qdd/lambda as full rebuild within 1e-4 tolerance
- P/A/q/l/u are updated — no stale matrix data
- Timing improves measurably (at minimum: duplicate QP build removed)
- Controller integrity checks pass

### Stage 3D.3-D — Full Batch Runner Integration

**Goal:** Integrate incremental QP path into `phase3d_full_batch_execution.py` behind `--use-incremental-qp` flag.

**Deliverables:**
- Modified `scripts/phase3d_full_batch_execution.py` with new CLI flags
- Integration test: `test_full_batch_runner_with_incremental_qp_flag`
- Integration test: `test_full_batch_runner_default_unchanged`

**CLI flags:**
```bash
--use-incremental-qp
--incremental-qp-max-contacts 4
--incremental-qp-backend osqp
--incremental-qp-reinit-on-topology-change
--benchmark-incremental-qp
```

**Validation:**
- `phase3d_full_batch_execution.py --use-incremental-qp --quick` runs successfully
- Default path (without `--use-incremental-qp`) remains unchanged
- Output config records `incremental_qp_enabled`, `persistent_osqp_workspace`, `updates_Px_Ax`, etc.

## 11. Correctness Audit Plan

**Script:** `scripts/phase3d3_incremental_qp_correctness_audit.py`

**Test cases:**
1. `keyframe_static` — equilibrium state, no perturbation
2. `small_forward_velocity` — qvel perturbation
3. `small_lateral_velocity` — qvel perturbation
4. `small_yaw_rate` — qvel perturbation
5. `small_roll_tilt` — qpos orientation perturbation
6. `small_pitch_tilt` — qpos orientation perturbation
7. `deterministic_push_state` — known external force
8. `random_push_state` — randomized perturbation

**For each case, compare:**

| Metric | Full Rebuild | Incremental | Tolerance |
|--------|-------------|-------------|-----------|
| `tau_wbc` | `tau_full` | `tau_incr` | max_abs_diff ≤ 1e-4 Nm |
| `qdd_wbc` | `qdd_full` | `qdd_incr` | max_abs_diff ≤ 1e-4 |
| `lambda` | `lam_full` | `lam_incr` | max_abs_diff ≤ 1e-4 |
| Dynamics residual | | | ≤ 1e-4 |
| Contact residual | | | ≤ 1e-4 |
| Friction violation | | | ≤ 1e-4 |
| Rolling residual | | | ≤ 1e-4 |
| Solver status | | | "solved" or "solved inaccurate" |
| P data staleness | | | max_abs_diff ≤ 1e-6 |
| A data staleness | | | max_abs_diff ≤ 1e-6 |

**Stale P/A detection test (explicit):**
After a deliberate qpos/qvel change large enough to change dynamics/Jacobians, verify that `incremental_P_data` matches `full_rebuild_P_data` within tolerance, and same for A. If only q/l/u are updated and P/A remain stale to old values, the test must fail.

**Verdict on failure:**
```
INCREMENTAL_QP_CORRECTNESS_FAIL
```
Do not proceed to performance benchmark if correctness fails.

## 12. Performance Benchmark Plan

**Script:** `scripts/phase3d3_incremental_qp_benchmark.py`

**Benchmark matrix:**
- States: ≥ 8 distinct states
- Steps per state: ≥ 20 consecutive steps
- Total incremental steps: ≥ 160

**Measured per path (full rebuild vs incremental):**

| Metric | Description |
|--------|-------------|
| `snapshot_prepare_time_ms` | Phase 3B snapshot preparation |
| `qp_structure_build_time_ms` | Full structured QP build (full) or block update (incr) |
| `dense_to_csc_time_ms` | Dense→CSC conversion (full only) |
| `csc_patch_time_ms` | Direct CSC data patching (incr only) |
| `osqp_setup_time_ms` | Full OSQP setup (first call or reinit) |
| `osqp_update_time_ms` | OSQP numeric update |
| `osqp_solve_time_ms` | OSQP solve |
| `full_wbc_step_time_ms` | End-to-end wall time |
| `solver_success_rate` | Fraction of solves returning "solved" |
| `workspace_reinit_count` | Number of forced reinitializations |
| `speedup_vs_full_rebuild` | `full_time / incremental_time` |

**Output files:**
```
outputs/phase3d3_incremental_qp/
  incremental_qp_correctness.json
  incremental_qp_benchmark.json
  incremental_qp_timing.csv
  incremental_qp_residuals.csv
  incremental_qp_verdict.json
```

**Verdict logic:**
```python
if not correctness_pass:
    verdict = "INCREMENTAL_QP_CORRECTNESS_FAIL"
elif full_wbc_step_time_mean < 30 and full_wbc_step_time_p95 < 50:
    verdict = "REALTIME_CANDIDATE_STRONG"
elif full_wbc_step_time_mean < 120:
    verdict = "CLOSED_LOOP_EVALUATION_UNBLOCKED"
elif speedup >= 50:
    verdict = "PARTIAL_SPEEDUP_NOT_FULLY_UNBLOCKED"
else:
    verdict = "INCREMENTAL_QP_INSUFFICIENT"
```

## 13. Full-Batch Integration Plan

Add to `scripts/phase3d_full_batch_execution.py`:

```python
# New CLI flags
parser.add_argument("--use-incremental-qp", action="store_true",
    help="Use incremental QP path instead of full rebuild each step")
parser.add_argument("--incremental-qp-max-contacts", type=int, default=4)
parser.add_argument("--incremental-qp-backend", type=str, default="osqp")
parser.add_argument("--incremental-qp-reinit-on-topology-change", action="store_true")
parser.add_argument("--benchmark-incremental-qp", action="store_true",
    help="Run incremental QP benchmark alongside three-arm evaluation")
```

**Selection logic (does not modify existing functions):**
```python
if args.use_incremental_qp:
    wbc_torque_fn = compute_wbc_torque_incremental_for_state
    workspace = initialize_incremental_qp_workspace(...)
else:
    wbc_torque_fn = compute_wbc_torque_for_state  # original, untouched
    workspace = None
```

**Output metadata when incremental path is active:**
```json
{
    "incremental_qp_enabled": true,
    "backend": "osqp",
    "persistent_osqp_workspace": true,
    "updates_Px_Ax": true,
    "warm_start_primal": true,
    "warm_start_dual": true,
    "max_contacts": 4,
    "workspace_reinit_count": 0,
    "fallback_full_rebuild_count": 0
}
```

## 14. Files Created/Modified

| File | Action | Stage |
|------|--------|-------|
| `wheeled_biped/wbc/persistent_osqp_backend.py` | **Create** | 3D.3-A |
| `wheeled_biped/wbc/phase3d3_incremental_qp.py` | **Create** | 3D.3-C |
| `wheeled_biped/wbc/structured_qp_problem.py` | **Modify** | 3D.3-B |
| `scripts/phase3d3_incremental_qp_correctness_audit.py` | **Create** | 3D.3-C |
| `scripts/phase3d3_incremental_qp_benchmark.py` | **Create** | 3D.3-C |
| `scripts/phase3d_full_batch_execution.py` | **Modify** | 3D.3-D |
| `tests/test_phase3d3_incremental_qp.py` | **Create** | 3D.3-A |
| `tests/test_phase3d3_incremental_qp_benchmark_schema.py` | **Create** | 3D.3-C |
| `docs/validation/k2_phase3d3_incremental_qp_report.md` | **Create** | Final |

## 15. Files Explicitly NOT Modified

| File | Reason |
|------|--------|
| `wheeled_biped/wbc/qp_solver_backends.py` | Phase 3D.2 backend preserved; new backend goes in separate file |
| `wheeled_biped/wbc/phase3d2_fast_solver.py` | Phase 3D.2 fast solver preserved |
| `wheeled_biped/wbc/offline_three_arm_counterfactual.py` | `compute_wbc_torque_for_state()` preserved; new function added separately |
| `wheeled_biped/wbc/phase3c_rolling_qp.py` | Phase 3C preserved |
| `wheeled_biped/wbc/offline_qp_wbc.py` | Legacy QP preserved |
| `wheeled_biped/controllers/k2_jax_controller.py` | V3 controller NOT touched |
| `wheeled_biped/controllers/*` | No controller files modified |
| `configs/controllers/*` | No profile/gain changes |

## 16. Verdict Rules

**Allowed verdicts:**
- `INCREMENTAL_QP_CORRECTNESS_PASS` — incremental matches full rebuild within tolerance
- `INCREMENTAL_QP_CORRECTNESS_FAIL` — incremental does not match full rebuild
- `PARTIAL_SPEEDUP_NOT_FULLY_UNBLOCKED` — correct but ≥50× speedup, still too slow for batch
- `CLOSED_LOOP_EVALUATION_UNBLOCKED` — correct and <120 ms/step
- `REALTIME_CANDIDATE_STRONG` — correct and <30 ms mean, <50 ms P95
- `INCREMENTAL_QP_INSUFFICIENT` — correct but no meaningful speedup

**Forbidden verdicts (do not use):**
- `REALTIME_READY` — this phase does not validate hardware realtime
- `PRODUCTION_READY` — this phase is evaluation infrastructure only
- `WBC_PROMOTED` — WBC is not promoted to production
- `DEFAULT_CONTROLLER_UPDATED` — V3 is not modified

## 17. Non-Goals

This phase explicitly does NOT:
- Modify K2 V3 controller or its gains
- Promote WBC to production/realtime controller
- Make incremental QP the default path
- Solve height variant state generation (deferred to Phase 3D.4)
- Achieve hardware realtime readiness
- Claim sim-to-real validity
- Run full 225-scenario batch (that's after Phase 3D.3 succeeds)
- Integrate WBC into the realtime controller loop

## 18. Risks and Fallback Plan

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CSC index mapping is fragile | Medium | Use construction-time metadata, not post-hoc probing; fall back to full rebuild if mapping is incomplete |
| Px/Ax update triggers expensive OSQP refactorization | Medium | Measure separately; even with refactorization it should be cheaper than full Python rebuild |
| Contact topology changes too frequently | Low | Padded contacts keep structure fixed; reinit only on genuine topology change |
| Snapshot generation still dominates after Stage B | Medium | Stage C implements direct block patching; Stage 3D.4 can address snapshot caching separately |
| Correctness audit fails (tol > 1e-4) | Low-Medium | Numeric differences may come from floating-point order; investigate and tighten tolerance if justified, but report honestly |
| Speedup insufficient for batch execution | Medium | Even 10-50× speedup is progress; report `PARTIAL_SPEEDUP_NOT_FULLY_UNBLOCKED` honestly |

**Fallback plan:** If incremental QP cannot meet correctness thresholds, report `INCREMENTAL_QP_CORRECTNESS_FAIL` with detailed diagnostics. The full rebuild path remains available. Alternative evaluation strategies (single-step WBC at sampled states, offline trajectory optimization) may be explored.

---

## References

- Phase 3D Full Batch Execution Report: `docs/validation/k2_phase3d_full_batch_execution_report.md`
- Phase 3D.2 Fast Solver: `wheeled_biped/wbc/phase3d2_fast_solver.py`
- Structured QP Problem: `wheeled_biped/wbc/structured_qp_problem.py`
- QP Solver Backends: `wheeled_biped/wbc/qp_solver_backends.py`
- Three-Arm Counterfactual: `wheeled_biped/wbc/offline_three_arm_counterfactual.py`
- Full Batch Runner: `scripts/phase3d_full_batch_execution.py`
- CLAUDE.md: project-level architecture and invariants
