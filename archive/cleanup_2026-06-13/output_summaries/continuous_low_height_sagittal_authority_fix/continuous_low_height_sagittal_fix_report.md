# Continuous Low-Height Sagittal Authority Fix - Final Report

**Date:** 2026-06-03  
**Evaluation Status:** COMPLETE  
**Final Decision:** `LOW_HEIGHT_SAGITTAL_FIX_REQUIRED`

---

## Executive Summary

All continuous k_position schedule candidates (E1, E2, E3) **FAILED** to fix the low_0p300 boundary failure. While increasing k_position from 40 to 100 at extreme flexion provides **marginal improvement** (~16% reduction in support drift), the improvement is **insufficient** to pass acceptance gates.

**Critical failure modes persist:**
- Support position error remains **35% above threshold** (0.203m vs 0.150m)
- Hip yaw drift remains **catastrophic** (>0.20 rad vs 0.07 rad threshold)

---

## Root Cause Verification

✅ **Confirmed:** Insufficient sagittal position authority at extreme flexion (z=0.300m)  
❌ **Fix inadequate:** Continuous k_position scheduling alone does NOT resolve the failure

---

## Implementation Summary

### Files Changed
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
  - Added `smoothstep01()` and `scheduled_k_position()` functions
  - Added `SagittalAuthoritySchedule` continuous k_position fields
  - Integrated height-based k_position scheduling in `compute()`
  - Added telemetry for schedule diagnostics

- `tests/test_sagittal_velocity_damped_balance_controller.py`
  - Added tests for `smoothstep01()` and `scheduled_k_position()`
  - Added tests for continuous schedule profiles

- `scripts/check_schedule_continuity.py`
  - Generated 181-point dense sweep + clamp checks
  - Verified monotonic decrease, no discontinuity, correct clamps

- `scripts/evaluate_continuous_low_height_sagittal_authority_fix.py`
  - Full evaluation pipeline with multi-phase gates
  - Setup generation for intermediate heights (0.330, 0.360, 0.450)
  - Acceptance gate checking and telemetry analysis

- `outputs/physical_target_height_setups/`
  - Generated: `low_0p330_setup.json`, `low_0p360_setup.json`, `high_0p450_setup.json`

### Tests Run
```
pytest tests/test_sagittal_velocity_damped_balance_controller.py -v  ✅ PASS
pytest tests/test_physical_standing_height_envelope.py -v            ✅ PASS
python scripts/check_schedule_continuity.py                         ✅ PASS
python scripts/evaluate_continuous_low_height_sagittal_authority_fix.py  ✅ COMPLETE
```

---

## Schedule Continuity Verification

✅ **All 3 candidates passed continuity checks:**
- `candidate_E1_k60_continuous`: max_abs_delta=0.323, monotonic=True, clamp_verified=True
- `candidate_E2_k80_continuous`: max_abs_delta=0.645, monotonic=True, clamp_verified=True
- `candidate_E3_k100_continuous`: max_abs_delta=0.968, monotonic=True, clamp_verified=True

Formula implemented:
```python
def scheduled_k_position(z_ref, k_nominal, k_low_max, z_low, z_high):
    u = (z_high - z_ref) / (z_high - z_low)
    s = smoothstep01(clamp(u, 0, 1))
    return k_nominal + (k_low_max - k_nominal) * s
```

---

## Evaluation Results

### Candidate Comparison (low_0p300 @ 1000 steps)

| Candidate   | k_position | Support Err | Hip Yaw   | Pitch     | Roll      | Status |
|-------------|------------|-------------|-----------|-----------|-----------|--------|
| **baseline**    | 40         | 0.2430 m    | 0.2136 rad | 0.0951 rad | 0.0150 rad | **FAIL** |
| **E1_k60**      | 40→60      | 0.2216 m    | 0.2113 rad | 0.0940 rad | 0.0150 rad | **FAIL** |
| **E2_k80**      | 40→80      | 0.2094 m    | 0.2081 rad | 0.0983 rad | 0.0150 rad | **FAIL** |
| **E3_k100**     | 40→100     | 0.2031 m    | 0.2054 rad | 0.1001 rad | 0.0150 rad | **FAIL** |
| **Thresholds**  |            | **≤0.1500 m** | **≤0.0700 rad** | **≤0.1000 rad** | **≤0.0500 rad** | |

### Failure Analysis

**Baseline → E3 improvement:**
- Support error: 0.243m → 0.203m (**16% reduction**, still **35% above threshold**)
- Hip yaw: 0.214 rad → 0.205 rad (**4% reduction**, still **193% above threshold**)

**E3 (k100) new failure:**
- Pitch exceeded threshold: 0.1001 rad > 0.100 rad (marginal exceedance)

### Key Observations

1. **Marginal improvement trend:** Increasing k_position reduces support drift slightly
2. **Insufficient magnitude:** 150% increase (40→100) yields only 16% drift reduction
3. **Hip yaw coupling:** Hip yaw drift remains catastrophic across all candidates
4. **Diminishing returns:** E3 causes pitch exceedance without sufficient benefit

---

## Gate-by-Gate Results

### Phase 1: low_0p300 @ 1000 steps
- baseline: **FAIL** (support>0.243, yaw>0.214)
- E1_k60: **FAIL** (support>0.222, yaw>0.211)
- E2_k80: **FAIL** (support>0.209, yaw>0.208)
- E3_k100: **FAIL** (support>0.203, yaw>0.205, pitch>0.100)

**All candidates failed Phase 1 → remaining phases skipped per evaluation protocol**

### Phases Not Run
- Phase 2: low_0p300 @ 5000 steps (skipped)
- Phase 2b: high_0p480 @ 5000 steps (skipped)
- Phase 3: Step C low_0p300/high_0p480 (skipped)
- Phase 4: Step E height grid (skipped)
- Phase 5: Step C height grid (skipped)
- Phase 6: Five-variant regression (skipped)

---

## Acceptance Criteria Check

| Criterion | Threshold | baseline | E1 | E2 | E3 | Status |
|-----------|-----------|----------|----|----|----|----|
| support_position_error | ≤ 0.15 m | 0.243 | 0.222 | 0.209 | 0.203 | ❌ |
| hip_yaw_abs_max | ≤ 0.07 rad | 0.214 | 0.211 | 0.208 | 0.205 | ❌ |
| pitch_x_max_abs | ≤ 0.10 rad | 0.095 | 0.094 | 0.098 | **0.100** | ⚠️ |
| roll_y_max_abs | ≤ 0.05 rad | 0.015 | 0.015 | 0.015 | 0.015 | ✅ |
| height_error_final_abs | ≤ 0.02 m | - | - | - | - | (not checked) |
| non_wheel_floor_contacts | = 0 | - | - | - | - | (not checked) |
| contact_valid_rate | ≥ 99.9% | - | - | - | - | (not checked) |
| wbc_applied | = false | true | true | true | true | ✅ |
| hidden_torque_max | = 0 | 0 | 0 | 0 | 0 | ✅ |
| ownership_violations | = 0 | 0 | 0 | 0 | 0 | ✅ |

---

## Selected Candidate

**NONE** - all candidates failed

---

## Final Decision

### Decision: `LOW_HEIGHT_SAGITTAL_FIX_REQUIRED`

**Rationale:**
1. ✅ Root cause correctly identified: insufficient sagittal position authority at extreme flexion
2. ❌ Proposed fix inadequate: continuous k_position scheduling does NOT resolve the failure
3. ❌ Support drift remains 35% above threshold even with k_position=100
4. ❌ Hip yaw drift remains catastrophic (>0.20 rad) across all candidates

### What Was Achieved
- Continuous k_position scheduling implemented correctly
- Schedule verified continuous, monotonic, with correct clamps
- Marginal improvement demonstrated (16% support drift reduction)
- Hip yaw coupling to support drift confirmed

### Why It Failed
- **Insufficient authority magnitude:** 150% k_position increase → only 16% drift reduction
- **Hip yaw instability persists:** Sagittal drift couples to hip yaw, which remains uncontrolled
- **Fundamental limitation:** Velocity-damped controller at extreme flexion lacks sufficient authority

### Recommended Next Steps

**Option 1: Velocity damping tuning**
- Increase `k_velocity` from 15 to 25-30 at low heights
- May provide additional damping to prevent drift

**Option 2: Max torque increase**
- Increase `max_position_tau` from 3.0 to 5.0-6.0 Nm at low heights
- May allow controller to apply stronger corrective torque

**Option 3: Hybrid approach**
- Combine k_position + k_velocity + max_tau scheduling
- Schedule all 3 parameters together based on height

**Option 4: Controller redesign**
- Consider switching to full LQR or MPC at extreme flexion
- Velocity-damped controller may be fundamentally inadequate below z=0.35m

**Option 5: Operational envelope restriction**
- Accept z_min = 0.35m as physical standing envelope lower bound
- Do not claim Step D readiness for heights below 0.35m

---

## Step D Readiness

**Status:** ❌ **NOT READY**

**Blockers:**
- low_0p300 Step E failure (all candidates)
- low_0p300 Step C not attempted (prerequisite failed)
- high_0p480 Step E/C not verified (evaluation stopped early)
- Practical height grid not validated
- Five-variant regression not verified

**Do NOT start Step D until:**
1. A candidate passes all Phase 1-6 gates, OR
2. Operational envelope is formally restricted to z ≥ 0.35m

---

## Telemetry Evidence

### Schedule Diagnostics (E2_k80 example)
- `schedule_height_source`: "commanded_height_ref_m"
- `schedule_height_reference_m`: 0.300
- `effective_k_position`: 80.0 (at z=0.300m)
- `k_position_schedule_smoothstep`: 1.0 (full low-height boost)
- `low_height_sagittal_schedule_active`: true

### WBC Status
- `wbc_applied`: false (balance-core mode, as expected)
- `hidden_torque_max`: 0.0 (no ownership violations)
- `ownership_violation_count_max`: 0

---

## Summary

| Metric | Result |
|--------|--------|
| **Implementation** | ✅ Complete, correct, tested |
| **Schedule continuity** | ✅ Verified (181 points, clamps checked) |
| **Baseline low_0p300** | ❌ FAIL (support=0.243m, yaw=0.214rad) |
| **E1_k60 low_0p300** | ❌ FAIL (support=0.222m, yaw=0.211rad) |
| **E2_k80 low_0p300** | ❌ FAIL (support=0.209m, yaw=0.208rad) |
| **E3_k100 low_0p300** | ❌ FAIL (support=0.203m, yaw=0.205rad, pitch=0.100rad) |
| **Selected candidate** | NONE |
| **Final decision** | `LOW_HEIGHT_SAGITTAL_FIX_REQUIRED` |
| **Step D readiness** | ❌ NOT READY |

---

**Conclusion:** The continuous low-height sagittal authority fix was correctly implemented and provides marginal improvement, but does NOT solve the boundary failure. A different fix approach is required before Step D can proceed.
