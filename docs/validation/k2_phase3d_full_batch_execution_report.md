# K2 Phase 3D — Full Batch Execution Report

**Verdict:** `PARTIAL_EVIDENCE_ONLY`
**Timestamp:** 2026-07-15T14:54:06.040089+00:00
**Batch Type:** FULL_BATCH_EXECUTION

## 1. Executive Summary
- Scenarios evaluated: 1
- Failures: 1
- Blocked: 0
- Solver success rate: 1.0
- Verdict: **PARTIAL_EVIDENCE_ONLY**

## 2. Git Commit SHA
- SHA: `c2f4b19a6c249ca64707d664f466e97f510723cb`
- Branch: `repo-cleanup-t6j`
- Status: clean

## 3. Worktree Status
- Default controller profile: `K2_JAX_DEDICATED_DEFAULT_V3`
- Controller modified: False
- V3 gain tuning: False

## 4. Controller Integrity Audit
- Production realtime WBC injection: False
- Default controller modified: False
- V3 gain tuning: False
- Hidden torque enabled: False
- WBC torque offline clones only: True
- V3 truth check (pre): SKIPPED
- V3 truth check (post): SKIPPED

## 5. Arm Definitions
- **Arm 1 — V3_BASELINE:** `tau_cmd = tau_v3` (real K2 JAX controller)
- **Arm 2 — WBC_ONLY:** `tau_cmd = tau_wbc` (QP-WBC torque, counterfactual)
- **Arm 3 — V3_PLUS_WBC_ASSIST:** `tau_cmd = tau_v3 + alpha * clamp(tau_wbc - tau_v3)`
  - alpha = 0.25
  - assist_limit_fraction = 0.2

## 6. Scenario Matrix
- Height variants: ['nominal']
- Total entries: 1
- Total blocked: 0

## 7. Pass/Fail Gates
- assist_falls_le_v3: **PASS**
- assist_safety_le_v3: **PASS**
- torque_limit_violations_zero: **PASS**
- nan_inf_zero: **PASS**
- controller_not_modified: **PASS**
- wbc_torque_offline_only: **PASS**
- no_hidden_torque: **PASS**
- v3_no_gain_tuning: **PASS**
- **All gates passed: True**

## 8. Per-Scenario Results
| Scenario | Suite | V3 Falls | WBC Falls | Assist Falls | Best Arm |
|----------|-------|----------|-----------|-------------|----------|
| step_e_nominal | step_e | 1 | 0 | 0 | V3_BASELINE |

## 9. Per-Arm Aggregate Comparison
### Safety
- V3 falls: 1, safety fails: 0
- WBC-only falls: 0, safety fails: 0
- Assist falls: 0, safety fails: 0
### Classification
- WBC-only: {'improved': 0, 'equivalent': 0, 'mixed': 0, 'regressed': 0, 'safety_fail': 0}
- Assist: {'improved': 0, 'equivalent': 0, 'mixed': 0, 'regressed': 0, 'safety_fail': 0}
- Best arm counts: {'V3_BASELINE': 1, 'WBC_ONLY': 0, 'V3_PLUS_WBC_ASSIST': 0, 'INCONCLUSIVE': 0}

## 10. Ratios vs V3
- Height error: {'wbc_only_over_v3': None, 'assist_over_v3': None}
- Posture error: {'wbc_only_over_v3': None, 'assist_over_v3': None}
- Drift: {'wbc_only_over_v3': None, 'assist_over_v3': None}
- Yaw error: {'wbc_only_over_v3': None, 'assist_over_v3': None}

## 11. Solver/QP Timing
- Solve time: N/A


- Full step time mean: 0.172 ms
- Solver success rate: 1.0
- WBC solve failures: 0
- Assist WBC failures: 0

## 12. Failure/Blocker Analysis
- Total failures: 1
- Total blocked: 0

## 13. Final Verdict
**PARTIAL_EVIDENCE_ONLY**

## 14. What This Means

## 15. What This Does Not Mean
- NOT hardware-safe or production-ready
- NOT promoted as default controller
- NOT realtime-ready (full pipeline not benchmarked for realtime)
- NOT a replacement for K2 V3

## 16. Recommended Next Phase
Collect more evidence. Address blockers before promotion consideration.
