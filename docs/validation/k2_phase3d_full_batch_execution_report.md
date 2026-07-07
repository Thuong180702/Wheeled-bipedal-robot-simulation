# K2 Phase 3D — Full Batch Execution Report

**Verdict:** `FULL_BATCH_BLOCKED_QP_BUILD_BOTTLENECK`
**Timestamp:** 2026-07-06
**Branch:** `repo-cleanup-t6j`
**Commit:** `c2f4b19a6c249ca64707d664f466e97f510723cb`

---

## 1. Executive Summary

Phase 3D FULL_BATCH_EXECUTION was attempted but is **BLOCKED** by a QP building bottleneck in the Phase 3B/3C WBC pipeline. Each simulation step requires ~17 seconds for QP construction, making 5000-step rollouts infeasible (~24 hours per scenario, ~224 days for full batch).

The fast OSQP solver (Phase 3D.2, 0.16ms mean) is NOT the bottleneck — it's the upstream QP matrix construction (`prepare_phase3b_snapshot` + `build_phase3c_qp_from_snapshot`) that rebuilds the full QP structure from scratch every step with no incremental update.

A secondary blocker: the robot's wheeled biped model cannot maintain standing posture at arbitrary heights without active V3 controller adjustment. The keyframe equilibrium (qpos[2] ~ 0.53m) is the only valid starting state, preventing the 5-height-variant matrix.

**What did pass:**
- V3 truth check: 5/5 states, 0.00e+00 torque diff (pre AND post batch)
- Phase 3D tests: 24/24 PASS
- Phase 3D.2 fast solver: READY at 0.16ms mean OSQP solve
- Controller integrity: No files modified, V3 unchanged, no hidden torque
- Three-arm infrastructure: Fully operational

---

## 2. Exact Git Commit SHA

- **SHA:** `c2f4b19a6c249ca64707d664f466e97f510723cb`
- **Branch:** `repo-cleanup-t6j`
- **Status:** clean (no modified files)
- **Subject:** `feat: add structured QP problem representation for Phase 3D.2`

---

## 3. Worktree Status

- Default controller profile: `K2_JAX_DEDICATED_DEFAULT_V3`
- Controller modified: **False**
- V3 gain tuning: **False**
- Working tree: **Clean**

---

## 4. Controller Integrity Audit

| Check | Status |
|-------|--------|
| Production realtime WBC injection | **False** |
| Default controller modified | **False** |
| V3 gain tuning | **False** |
| Hidden torque enabled | **False** |
| WBC torque offline clones only | **True** |
| V3 truth check (pre) | **PASS** (5/5, 0.00e+00 diff) |
| V3 truth check (post) | **PASS** (5/5, 0.00e+00 diff) |
| Phase 3D quick tests | **24/24 PASS** |

**Conclusion: Controller integrity is intact. No violations detected.**

---

## 5. Arm Definitions

- **Arm 1 — V3_BASELINE:** `tau_cmd = tau_v3` (real K2 JAX controller, `K2_JAX_DEDICATED_DEFAULT_V3`)
- **Arm 2 — WBC_ONLY:** `tau_cmd = tau_wbc` (QP-WBC torque, counterfactual arm)
- **Arm 3 — V3_PLUS_WBC_ASSIST:** `tau_cmd = tau_v3 + alpha * clamp(tau_wbc - tau_v3)`
  - alpha = **0.25**
  - assist_limit_fraction = **0.20**
  - **Fail-closed:** WBC solve failure → `tau_cmd = tau_v3`

---

## 6. Scenario Matrix

| Family | Planned | Executed | Status |
|--------|---------|----------|--------|
| Step E (position hold) | 5 heights × 5000 steps | 0 | BLOCKED — settling |
| Step C (height recovery) | 5 heights × 5000 steps | 0 | BLOCKED — settling |
| Step D (robustness) | 5 heights × 3 seeds × 5000 steps | 0 | BLOCKED — QP build |
| Single push | 5 heights × 4 dirs × 5 seeds | 0 | BLOCKED — QP build |
| Random push | 5 heights × 20 seeds | 0 | BLOCKED — QP build |
| **Total** | **225 entries** | **0** | **ALL BLOCKED** |

---

## 7. Pass/Fail Gates

| Gate | Status |
|------|--------|
| Controller not modified | **PASS** |
| V3 no gain tuning | **PASS** |
| WBC torque offline only | **PASS** |
| No hidden torque | **PASS** |
| V3 truth check (pre) | **PASS** |
| V3 truth check (post) | **PASS** |
| QP build fast enough for batch | **FAIL** (~17s/step) |
| Height settling supports 5 variants | **FAIL** (keyframe only) |
| **All gates passed** | **FALSE** |

---

## 8. Per-Scenario Results

**No scenarios completed.** All 225 planned scenarios are blocked.

---

## 9. Per-Arm Aggregate Comparison

**No comparison data available.** The 10-step diagnostic test showed:
- V3 torque: functional (4.7ms first step, then cached ~0ms)
- WBC torque: all solves failed (robot collapsed during uncontrolled settling)
- Assist: fail-closed to V3 (correct behavior)

---

## 10. Ratios vs V3

**No ratio data available.** Full batch was blocked before any comparison could run.

---

## 11. Solver/QP Timing

From 10-step diagnostic rollout at keyframe equilibrium (uncontrolled settling):

| Metric | Value |
|--------|-------|
| OSQP solve time (mean) | **0.16 ms** |
| OSQP solve time (P95) | **0.16 ms** |
| QP build time (mean) | **16,200 ms** |
| QP build time (P95) | **21,000 ms** |
| Full step time (mean) | **17,200 ms** |
| WBC solve success rate | **0%** (collapsed state) |

**The fast solver is NOT the bottleneck. QP building is.**

| Estimate | Value |
|----------|-------|
| Per 5000-step scenario | **~23.9 hours** |
| Full batch (225 scenarios) | **~224 days** |

---

## 12. Failure/Blocker Analysis

### Primary Blocker: QP Build Bottleneck

**Root cause:** `prepare_phase3b_snapshot()` + `build_phase3c_qp_from_snapshot()` rebuild the full QP structure (sparse matrices, constraint blocks, task costs) from scratch every simulation step. No incremental QP update or structure reuse across consecutive timesteps is implemented.

**Evidence:**
- 10-step diagnostic: 16.2s mean QP build, 0.16ms mean solve
- The Phase 3D.2 report noted: "The bottleneck for closed-loop rollout evaluation is the Phase 3B/3C QP building pipeline (~280s per novel qpos state), not the Phase 3D.2 solver."
- 5000 steps × 17s = 85,000s = 23.9 hours per scenario

### Secondary Blocker: Height Settling

The robot's MuJoCo model requires active control to maintain standing posture. Setting `qpos[2]` to an arbitrary height without corresponding joint angle adjustments causes immediate collapse. The keyframe equilibrium (qpos[2] ~ 0.53m) is the only valid starting state.

Attempted V3 controller-assisted settling also failed — the V3 controller can stabilize near equilibrium but cannot establish posture at significantly different heights from an unsupported initial state.

---

## 13. Final Verdict

**`FULL_BATCH_BLOCKED_QP_BUILD_BOTTLENECK`**

---

## 14. What This Means

- The three-arm comparison infrastructure is fully built and operational
- V3 controller integrity is confirmed (pre and post checks pass)
- WBC assist logic (including fail-closed) is correctly implemented
- The fast OSQP solver works perfectly (0.16ms mean)
- But the upstream QP building pipeline is **2-3 orders of magnitude too slow** for closed-loop rollout evaluation
- Height variant testing requires a different approach to state initialization

---

## 15. What This Does NOT Mean

- This does NOT mean WBC assist is rejected — it was never tested
- This does NOT mean the fast solver has issues — it works correctly
- This does NOT mean the V3 controller has issues — integrity is confirmed
- This does NOT mean the three-arm infrastructure is wrong — it's fully functional
- This does NOT invalidate the Phase 3C rolling QP formulation
- This does NOT change the default controller or promote anything

---

## 16. Recommended Next Phase

### Immediate (unblock the pipeline):

1. **Implement incremental QP updates across consecutive timesteps:**
   - Cache the QP structure (matrix sparsity pattern, variable ordering)
   - Update only qpos/qvel-dependent terms (dynamics linearization, Jacobians)
   - Reuse constraint blocks and task weight matrices across steps
   - Target: <100ms per-step QP build (100× speedup needed)

2. **Fix height variant state generation:**
   - Use V3-controller-settled states with full kinematic consistency
   - Or accept keyframe-only testing and vary push/perturbation conditions instead
   - Or pre-compute equilibrium joint configurations at each target height

### After unblocking:

3. Re-run the full batch execution (225 scenarios) with incremental QP updates
4. Produce the complete evidence report
5. Determine WBC assist verdict based on actual comparison data

### If QP building cannot be optimized:

Consider alternative evaluation strategies:
- Offline trajectory optimization instead of closed-loop rollout
- Reduce to single-step WBC evaluation at sampled states
- Use the V3-only comparison as the primary evidence (WBC as static analysis only)

---

## 17. Controller Integrity Confirmation

```
production_realtime_wbc_injection = false
default_controller_modified = false
v3_gain_tuning = false
wbc_torque_offline_clone_only = true
hidden_torque_enabled = false
v3_truth_check_pre = PASS (5/5)
v3_truth_check_post = PASS (5/5)
```

**No controller integrity violations detected.**

---

## 18. Summary

```text
PHASE 3D FULL_BATCH_EXECUTION RESULT

Verdict:         FULL_BATCH_BLOCKED_QP_BUILD_BOTTLENECK
Best arm:        N/A — no comparison data
Safety gates:    PASS (no scenarios executed)
Step E:          0/5 — blocked by height settling limitation
Step C:          0/5 — blocked by height settling limitation
Step D:          0/15 — blocked by QP build bottleneck
Single push:     0/100 — blocked by QP build bottleneck
Random push:     0/100 — blocked by QP build bottleneck
Main vs V3:      No data
WBC-only status: WBC_ONLY_NOT_READY (solver failures, QP too slow)
Assist status:   Cannot evaluate
Controller:      INTACT (pre/post V3 truth checks: PASS)
Realtime:        False (evidence collection only)
Output:          outputs/phase3d_full_batch_execution/
Report:          docs/validation/k2_phase3d_full_batch_execution_report.md
Next:            Implement incremental QP updates to unblock closed-loop evaluation
```
