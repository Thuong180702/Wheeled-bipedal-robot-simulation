# K2 Phase 3D — Three-Arm Closed-Loop Counterfactual Robustness Evaluation

**Verdict:** `PARTIAL_READY`
**Timestamp:** 2026-07-05T19:23:12.562655+00:00

## 1. Executive Summary
- Scenarios evaluated: 1
- Phase 3C prerequisite: READY
- Controller modified: False
- WBC torque only in offline clones: True

## 2. Controller Integrity Statement
No production controller files modified. No V3 gain tuning. No promotion.
WBC torque and assist torque applied only to cloned offline evaluation simulations.
No realtime integration. No modification of `K2_JAX_DEDICATED_DEFAULT_V3`.

## 3. Phase 3C Prerequisite Recap
- Phase 3C ready: True
- QP solves completed: 120
- Hard constraints pass: True

## 4. Validation Cross-Check
- Attempted: False
- Cases passed: 0/0

## 5. Three-Arm Evaluation Design
- Arm 1: V3_BASELINE — tau_cmd = tau_v3
- Arm 2: WBC_ONLY — tau_cmd = tau_wbc
- Arm 3: V3_PLUS_WBC_ASSIST — tau_cmd = tau_v3 + alpha * clamp(tau_wbc - tau_v3)

## 6. Assist Formulation
- alpha: 0.25
- assist_limit_fraction: 0.2

## 7. Test Suite Coverage
- legacy_c: {'available': False, 'completed': False, 'num_scenarios': 0}
- legacy_d: {'available': False, 'completed': False, 'num_scenarios': 0}
- legacy_e: {'available': False, 'completed': False, 'num_scenarios': 0}
- standard_deterministic: {'completed': False, 'num_scenarios': 8, 'steps_per_scenario': 1000}
- deterministic_single_push: {'completed': False, 'num_scenarios': 13, 'push_envelopes': ['mild', 'nominal']}
- random_single_push_mild: {'completed': False, 'num_seeds': 0, 'seeds': []}
- random_single_push_harsh_diagnostic: {'completed': False, 'required_for_ready': False, 'seeds': [101, 102, 103, 104, 105]}
- long_horizon_3000: {'completed': False, 'num_scenarios': 8, 'steps_per_scenario': 3000}

## 8. Safety Comparison
- V3 falls: 0
- WBC-only falls: 0
- Assist falls: 0
- V3 safety fails: 0
- WBC-only safety fails: 0
- Assist safety fails: 0
- NaN/Inf: 0
- Torque limit violations: 0

## 9. Physical Outcome Comparison
- WBC-only: {'improved': 0, 'equivalent': 0, 'mixed': 0, 'regressed': 1, 'safety_fail': 0}
- Assist: {'improved': 1, 'equivalent': 0, 'mixed': 0, 'regressed': 0, 'safety_fail': 0}
- Best arm counts: {'V3_BASELINE': 0, 'WBC_ONLY': 0, 'V3_PLUS_WBC_ASSIST': 1, 'INCONCLUSIVE': 0}
- Recommended next path: WBC_ASSIST_PATH

## 10. Limitations
- V3 torque uses real JAX controller path (K2_JAX_DEDICATED_DEFAULT_V3)
- Legacy C/D/E suites not found in repository
- Full validation cross-check pending execution

## 11. Phase 3E Readiness Verdict
**PARTIAL_READY**
