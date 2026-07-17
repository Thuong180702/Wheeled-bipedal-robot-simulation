# K2 Phase 3B.1 — Full Ablation Audit + Compilation Hardening

**Verdict:** `READY_FOR_PHASE_3C_OFFLINE_ROLLING_CONSTRAINTS_AND_TASK_REFINEMENT`
**Timestamp:** 2026-07-04T06:33:04.361814+00:00
**Constants version:** phase3b1_full_ablation

## 1. Executive Summary

- **Total QP solves:** 60/60 (expected 60)
- **Balanced default solved:** 12/12
- **Feasibility only solved:** N/A
- **Modes >= 10/12:** 5/5
- **Hard constraints:** PASS
- **Task residuals finite:** True
- **Solution sanity:** PASS
- **Audit complete:** True
- **Report logic fix:** completed=true ONLY when 60 >= 60

## 2. Controller Integrity Statement

- **Controller modified:** False
- **QP torque injected:** False
- **Realtime integration:** False
- **K2_JAX_DEDICATED_DEFAULT_V3 unchanged:** True

## 3. Report Logic Fix (Task 0)

### Previous Bug
- `ablation_completion.completed` was `true` despite only 5/60 solves.
- `total_qp_solves_expected` varied by run mode (5 for quick, 60 for full).
- `audit_completed` used runtime `total_expected` instead of fixed `FULL_EXPECTED_SOLVES=60`.

### Fix Applied
- `FULL_EXPECTED_SOLVES = 60` constant (12 scenarios × 5 modes).
- `total_qp_solves_expected` is ALWAYS 60 regardless of run mode.
- `completed` is `true` ONLY when unique solved entries across JSONL >= 60.
- `completed` is `false` for quick audit, resumed partial audit, or incomplete full audit.
- Report merges historical JSONL entries with current run entries for accurate counts.
- READY verdict is impossible if `completed` is false.

## 4. Changed Files

- `wheeled_biped/wbc/phase3b_cached_stack.py` (new)
- `scripts/phase3b1_full_ablation_audit.py` (updated — Task 0 fix + memory hardening)
- `scripts/phase3b1_compile_profile.py` (new)
- `tests/test_phase3b_offline_task_stack.py` (updated — quick tests)
- `tests/test_phase3b1_full_ablation_slow.py` (new — slow tests)
- `tests/test_phase3b1_compile_hardening.py` (new)
- `docs/validation/k2_phase3b1_full_ablation_audit.md` (updated)
- `docs/validation/k2_phase3b1_full_ablation_audit.json` (updated)

## 5. Phase 3B Partial-Readiness Recap

- Phase 3B verdict was PARTIAL_READY
- 42/52 tests passed, 10 timed out on JAX XLA compilation
- JSON had placeholder zeros for max residuals/magnitudes
- ablation_results was empty
- Root cause: repeated jax.jacfwd per scenario×mode

## 6. Compile-Time Root-Cause Analysis

- **Root cause:** Repeated JAX jacfwd compilation for COM Jacobian, torso Jacobian, and contact Jacobians
- **Evidence:** Each `compute_com_jacobian()` call creates new jax.jacfwd closure → JAX tracing
- **Impact:** 60 calls (12 scenarios × 5 modes) each triggering JAX compilation
- **Contact shapes:** Vary between 2 and 4 contacts per scenario → recompilation

## 7. Compilation Hardening Changes

### Shape-Stable Contacts
- `PaddedContactStack` with `max_contacts=4`
- All contact tensors have fixed shapes regardless of active contact count
- Inactive contacts masked via `active_mask`

### Snapshot Caching
- `prepare_phase3b_snapshot()` computes M, h, S, contact stack, COM Jacobian, torso Jacobian, Jdot_qdot ONCE per scenario
- `build_phase3b_qp_from_snapshot()` uses cached data only — no JAX calls
- Jacobians reused across all 5 task modes

### Quick/Slow Test Split
- Quick tests: shape validation, single QP solve, no controller imports
- Slow tests: full 12×5 audit validation (marked @pytest.mark.slow)

## 8. Full 12×5 Ablation Results

| Mode | Solved | Failed | Max Dyn Res | Max Contact Accel | Max Friction | Max Torque |
|------|--------|--------|-------------|-------------------|--------------|------------|
| feasibility_only | 12 | 0 | 2.82e-14 | 5.33e-15 | 7.74e-12 | 0.00e+00 |
| balanced_default | 12 | 0 | 2.44e-14 | 4.75e-15 | 2.45e-11 | 0.00e+00 |
| posture_priority | 12 | 0 | 4.61e-14 | 8.88e-15 | 1.45e-09 | 0.00e+00 |
| torso_priority | 12 | 0 | 2.41e-14 | 3.55e-15 | 6.08e-11 | 0.00e+00 |
| com_priority | 12 | 0 | 3.35e-14 | 3.55e-15 | 6.84e-11 | 0.00e+00 |

## 9. Feasibility-Only Regression vs Phase 3

- **Scenarios solved:** 12/12
- **Scenarios failed:** 0
- **Matches Phase 3 gates:** Yes
- **Max dynamics residual:** 2.82e-14
- **Max contact accel residual:** 5.33e-15
- **Max friction violation:** 7.74e-12
- **Max torque violation:** 0.0
- **Max |qdd|:** 57.25
- **Max |tau|:** 2.34
- **Max |lambda|:** 1.01

Feasibility-only regression passes against all Phase 3 hard constraint gates.

## 10. Balanced-Default Validation

- **Scenarios solved:** 12/12
- **Max dynamics residual:** 2.4358655023312095e-14
- **Max contact accel residual:** 4.746203430272544e-15
- **Max friction violation:** 2.452917972515837e-11
- **Max torque violation:** 0.0
- **Max |qdd|:** 57.44638560498425
- **Max |tau|:** 2.383760506241904
- **Max |lambda|:** 1.9276323025320978

## 11. Per-Mode Ablation Summary

### feasibility_only
- Solved: 12, Failed: 0

### balanced_default
- Solved: 12, Failed: 0

### posture_priority
- Solved: 12, Failed: 0

### torso_priority
- Solved: 12, Failed: 0

### com_priority
- Solved: 12, Failed: 0

## 12. Hard-Constraint Residual Validation

- **Max dynamics residual:** 2.436e-14 (gate: 1e-5)
- **Max contact accel residual:** 4.746e-15 (gate: 1e-4)
- **Max friction violation:** 2.453e-11 (gate: 1e-6)
- **Max torque violation:** 0.000e+00 (gate: 1e-6)
- **All PASS:** True

## 13. Task Residual Validation

- **Max COM task residual:** 10.442325656668869
- **Max torso task residual:** 14.438202835518727
- **Max posture task residual:** 81.40981692905616
- **Max wheel accel residual:** 0.05148832606417966
- **Max force regularization residual:** 1.5185320099539987
- **All finite:** True

## 14. Solution Magnitude Sanity

- **Max |qdd|:** 57.44638560498425 (sanity gate: 100.0)
- **Max |tau|:** 2.383760506241904
- **Max |lambda|:** 1.9276323025320978 (sanity gate: 500.0)

## 15. Timing/Performance Summary

- **Method:** Snapshot caching (Jacobians computed once per scenario)
- **Max contacts (padded):** 4
- **Estimated full audit time:** 60 QP solves with cached snapshots

## 16. Limitations

- SLSQP fallback used (OSQP not available)
- Jdot qdot still uses finite difference (but cached in snapshot)
- No analytical COM/torso Jacobians (JAX jacfwd, but cached)
- No tangential rolling constraint (deferred to Phase 3C)
- Offline only — no realtime integration
- No slack variables (soft tasks via costs only)

## 17. Phase 3C Readiness Verdict

**Verdict:** `READY_FOR_PHASE_3C_OFFLINE_ROLLING_CONSTRAINTS_AND_TASK_REFINEMENT`

Proceed to Phase 3C — Offline Rolling Constraints and Task Refinement.
