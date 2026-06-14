# APCR1k 1000-step Validation Final Report

## Executive Summary

**Result: FAIL - Do NOT proceed to 2000-step validation**

APCR1k 1000-step simulation completed successfully (robot survived 1000 steps), but drift metrics **worsened** compared to APCR1j:
- APCR1j max_e: 0.1826 m
- APCR1k max_e: 0.2315 m (+26.8% worse)
- Target: < 0.15 m

## Phase 6: Torque and Transmission Audit

### Findings

| Metric | APCR1j | APCR1k | Change |
|--------|--------|--------|--------|
| APCR command range | [-2.0, 2.0] Nm | [-2.0, 2.0] Nm | Same |
| Final wheel tau range | [-1.6, 1.6] Nm | [-1.6, 1.1] Nm | Worse |
| Transmission loss | ~18% | **44%** | +26 pp worse |

### Torque Direction Analysis

**Critical: Sign inversion detected**

| Direction | APCR Command | Expected Final | Actual Final | Result |
|-----------|--------------|-----------------|---------------|--------|
| Positive (forward) | up to +2.0 Nm | Move backward | **+0.19 Nm** | WRONG |
| Negative (backward) | down to -2.0 Nm | Move forward | **-0.44 Nm** | Partial |

When the robot drifted forward (e > 0):
- APCR command was NEGATIVE (correct: wants to move backward)
- But `final_wheel_tau_with_apc` was POSITIVE (wrong: moves forward)
- **The APCR correction is ACCELERATING the drift, not correcting it!**

## Phase 7: Drift Metrics

### Target: max |e| < 0.15 m

| Metric | APCR1j | APCR1k | Target | Status |
|--------|--------|--------|--------|--------|
| max \|e\| | 0.1826 m | 0.2315 m | < 0.15 m | FAIL |
| 50th percentile | - | 0.091 m | - | - |
| 90th percentile | - | 0.173 m | - | - |
| 99th percentile | - | 0.231 m | - | - |

APCR1k entered RECENTER earlier (666/1000 steps in RECENTER state), but the torque was NOT reaching the wheels in the correct direction.

## Phase 8: Episode Audit

| Metric | Value |
|--------|-------|
| Final status | SURVIVED (no fall) |
| Total steps | 1000 |
| Termination reason | None |
| Fall during simulation | No |

## Phase 9: Stability Audit

| Metric | Value | Assessment |
|--------|-------|------------|
| Pitch range | [-0.08, 0.14] deg | Good |
| Pitch RMS | 0.05 deg | Excellent |
| Height range | [0.287, 0.295] m | Good for 0.30m target |
| Max pitch | 8.0 deg | Acceptable |

Robot is stable in terms of pitch and height, but support drift is excessive.

## Phase 10: Classification

```
APCR1K_1000_FAIL_DO_NOT_PROCEED
```

### Reasons for Classification

1. **max_e increased from 0.1826 m to 0.2315 m** (+26.8% worse)
2. **Target max_e < 0.15 m NOT met**
3. **Torque transmission is INVERTED** for positive direction
4. **APCR corrections are accelerating drift**, not correcting it

## Root Cause Analysis

### The Problem: Sign Inversion in APCR Torque Path

The telemetry shows a **sign inversion** in the APCR torque transmission:

```
Step 392 (max error):
  Support error e = +0.2315 m (drifted FORWARD)
  APCR command = -1.998 Nm (correct: wants to move BACKWARD)
  final_wheel_tau_with_apc = +0.187 Nm (WRONG: moves FORWARD)
  final_wheel_tau_without_apc = +2.185 Nm (WITHOUT APC = CORRECT direction)
```

**Analysis:**
- `final_wheel_tau_without_apc` is POSITIVE (correct for moving backward after forward drift)
- `final_wheel_tau_with_apc` is ALSO POSITIVE (should be NEGATIVE to add to the correction)
- The APCR contribution is ADDING to the wrong direction!

### Why APCR1k Made Things Worse

APCR1k lowers the entry threshold from 0.08 m to 0.05 m, causing:
- More time in RECENTER state (666/1000 steps vs less for APCR1j)
- More APCR torque commands issued
- **More torque applied in the WRONG direction**

Early entry without correct torque direction = faster accumulation of error.

## APCR Hysteresis State Analysis

| State | Steps | Support Error Range | APCR Command Range |
|-------|-------|---------------------|---------------------|
| RECENTER_FROM_POSITIVE | 666 | [0.030, 0.232] m | [-2.0, +0.02] Nm |
| RECENTER_FROM_NEGATIVE | 49 | [-0.071, -0.031] m | [+0.15, +2.0] Nm |
| NEUTRAL | 285 | [-0.049, +0.050] m | [-1.8, +1.8] Nm |

**Key observation:**
- In RECENTER_FROM_POSITIVE (666 steps), APCR command is mostly NEGATIVE (correct direction)
- But the final wheel torque is POSITIVE (wrong direction)
- This indicates the sign inversion happens AFTER the APCR tau is computed

## What APCR1k Does NOT Fix

1. **Sign inversion in torque transmission** - This is a bug in the APCR integration
2. **Torque authority** - APCR still limited by wheel_torque_sign and clipping
3. **Gate behavior** - Safety gates remain unchanged
4. **Torque magnitude** - Maximum torque still limited

## Recommendations for Next Steps

### Immediate: Fix Sign Inversion

The root cause is NOT the entry threshold - it's the sign inversion in the torque transmission path. Investigate:

1. Where is the sign inversion introduced?
   - Is `wheel_torque_sign` applied correctly?
   - Is the APCR tau being added/subtracted with correct sign?
   - Is there a sign flip in the recenter/hysteresis logic?

2. Fix the sign issue first before changing thresholds

### Alternative Approach: Fix APCR Logic

Before lowering thresholds again, fix the underlying APCR torque path:

1. Verify `apc_tau_clipped` sign matches the intended corrective direction
2. Verify `final_wheel_tau_with_apc` has correct sign relative to `final_wheel_tau_without_apc`
3. Test with a simple case: positive error should produce negative final torque

### Profile Recommendation

If the sign inversion is fixed, APCR1k's lower entry threshold (0.05 m vs 0.08 m) may still be beneficial because:
- It catches drift earlier
- It matches the user's stated requirement: "e > +0.05 should start RECENTER_FROM_POSITIVE"
- With correct torque direction, earlier entry should reduce max_e

## Files Modified

- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` - Added APCR1k profile
- `scripts/simulate_hierarchical_controller.py` - Added APCR1k profile and option
- `tests/test_sagittal_velocity_damped_balance_controller.py` - Added APCR1k tests (13 tests, all passing)

## Test Results

APCR1k unit tests pass:
- Profile exists and is opt-in only
- Early entry threshold verified (0.05 m < 0.08 m)
- Same torque authority as APCR1j
- All safety gates preserved
- All other APCR profiles unchanged

## Conclusion

**APCR1k 1000-step validation FAILED.**

The simulation survived, but drift metrics worsened due to a **sign inversion bug** in the APCR torque transmission path. Lowering the entry threshold amplified the problem by applying more torque in the wrong direction.

**Do NOT proceed to 2000-step validation until:**
1. The sign inversion is identified and fixed
2. APCR1k 1000-step shows max_e < 0.15 m with correct torque direction

## Next Steps

1. **Root cause investigation**: Trace the sign inversion through the APCR torque path
2. **Fix sign issue**: Ensure APCR tau sign matches corrective direction
3. **Re-run APCR1k 1000-step**: Verify fix works
4. **Re-evaluate APCR1k**: If sign fix works, APCR1k may still be beneficial with early entry

---
*Report generated: 2026-06-10*
*APCR1k profile: APCR1k_support_hysteresis_early_entry*
*Simulation: low_0p300_1000*