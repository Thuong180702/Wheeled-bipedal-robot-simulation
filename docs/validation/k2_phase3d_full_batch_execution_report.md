# K2 Phase 3D — Full Batch Execution Report

**Verdict:** `WBC_ONLY_NOT_READY`
**Timestamp:** 2026-07-18T06:29:38.844765+00:00
**Batch Type:** FULL_BATCH_EXECUTION

## 1. Executive Summary
- Scenarios evaluated: 4
- Failures: 4
- Blocked: 1
- Solver success rate: 0.85845
- Verdict: **WBC_ONLY_NOT_READY**

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
- V3 truth check (pre): True
- V3 truth check (post): True

## 5. Arm Definitions
- **Arm 1 — V3_BASELINE:** `tau_cmd = tau_v3` (real K2 JAX controller)
- **Arm 2 — WBC_ONLY:** `tau_cmd = tau_wbc` (QP-WBC torque, counterfactual)
- **Arm 3 — V3_PLUS_WBC_ASSIST:** `tau_cmd = tau_v3 + alpha * clamp(tau_wbc - tau_v3)`
  - alpha = 0.25
  - assist_limit_fraction = 0.2

## 6. Scenario Matrix
- Height variants: ['nominal', 'low_tiny', 'high_tiny', 'low_small', 'high_small']
- Total entries: 4
- Total blocked: 1

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
| step_e_nominal | step_e | 0 | 4824 | 0 | INCONCLUSIVE |
| step_e_low_tiny | step_e | 0 | 4858 | 0 | INCONCLUSIVE |
| step_e_high_tiny | step_e | 0 | 4876 | 0 | INCONCLUSIVE |
| step_e_low_small | step_e | 0 | 155 | 0 | V3_BASELINE |

## 9. Per-Arm Aggregate Comparison
### Safety
- V3 falls: 0, safety fails: 0
- WBC-only falls: 14713, safety fails: 14832
- Assist falls: 0, safety fails: 0
### Classification
- WBC-only: {'improved': 0, 'equivalent': 0, 'mixed': 0, 'regressed': 0, 'safety_fail': 4}
- Assist: {'improved': 0, 'equivalent': 1, 'mixed': 3, 'regressed': 0, 'safety_fail': 0}
- Best arm counts: {'V3_BASELINE': 1, 'WBC_ONLY': 0, 'V3_PLUS_WBC_ASSIST': 0, 'INCONCLUSIVE': 3}

## 10. Ratios vs V3
- Height error: {'wbc_only_over_v3': 0.30282369088935646, 'assist_over_v3': 1.0029718971909436}
- Posture error: {'wbc_only_over_v3': 8.052951511548171, 'assist_over_v3': 1.009747320754153}
- Drift: {'wbc_only_over_v3': 2.3735547442345597, 'assist_over_v3': 0.9984063229383764}
- Yaw error: {'wbc_only_over_v3': 15.427040934081502, 'assist_over_v3': 1.100532723381086}

## 11. Solver/QP Timing
- Solve time mean: 0.452 ms
- Solve time P95: 2.941 ms
- QP build time mean: 105.849 ms
- Full step time mean: 257.960 ms
- Solver success rate: 0.85845
- WBC solve failures: 2831
- Assist WBC failures: 164

## 12. Failure/Blocker Analysis
- Total failures: 4
- Total blocked: 1

## 13. Final Verdict
**WBC_ONLY_NOT_READY**

## 14. What This Means

## 15. What This Does Not Mean
- NOT hardware-safe or production-ready
- NOT promoted as default controller
- NOT realtime-ready (full pipeline not benchmarked for realtime)
- NOT a replacement for K2 V3

## 16. Recommended Next Phase
Collect more evidence. Address blockers before promotion consideration.
