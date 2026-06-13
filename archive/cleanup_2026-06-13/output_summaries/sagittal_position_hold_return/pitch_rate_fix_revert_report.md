# Pitch Rate Consistency Fix Revert Report

**Date:** 2026-05-31
**Status:** REVERTED (disabled by default)

## Summary

The pitch rate consistency fix has been disabled in active control. The estimator remains available for diagnostic telemetry only.

## What Was Reverted

1. **Active control path**: `pitch_rate_estimate.pitch_rate_corrected` is no longer used by `SagittalVelocityDampedBalanceController`
2. **Default behavior**: Raw measured pitch rate (`centroidal_state_control.body_pitch_rate_x`) is now used
3. **Optional re-enable**: `--vd-enable-pitch-rate-correction` flag added for testing

## Why It Was Reverted

The pitch rate consistency fix was implemented based on a hypothesis that a pitch rate measurement artifact at step 1236 was causing the transient peak. However:

1. **Transient peak unchanged**: 0.595 m → 0.5951 m (no improvement)
2. **Height variant regressions**: Both `high_5cm` and `low_5cm` fell with the fix enabled
3. **Root cause incorrect**: The pitch rate artifact was a symptom, not the cause

## Verification

After revert, the following was confirmed:

| Check | Status |
|-------|--------|
| No pitch-rate correction in active control | PASS |
| No corrected_pitch_rate used by controller | PASS |
| WBC remains OFF | PASS |
| E0b/E0c/E0d remain absent | PASS |
| Torque ownership unchanged | PASS |
| kp_cp remains disabled (0.0) | PASS |

## Code Changes

**File:** `scripts/simulate_hierarchical_controller.py`

1. Added `--vd-enable-pitch-rate-correction` flag (default: False)
2. Modified velocity-damped controller path to use raw pitch rate by default
3. Estimator still called for diagnostic telemetry

## Baseline Reproduction After Revert

| Metric | Expected | Actual | Match |
|--------|----------|--------|-------|
| Max support_position_error | 0.595 m | 0.5950 m | YES |
| Max error step | 1360 | 1360 | YES |
| Final support_position_error | 0.053 m | 0.0527 m | YES |
| Steady-state mean | 0.0527 m | 0.0527 m | YES |
| Survived 5000 steps | YES | YES | YES |

## Files Modified

- `scripts/simulate_hierarchical_controller.py`: Disabled pitch rate correction by default
- `wheeled_biped/controllers/pitch_rate_consistency_estimator.py`: Unchanged (kept for diagnostics)
- `tests/test_pitch_rate_consistency_estimator.py`: Unchanged (tests diagnostic helper)

## Recommendation

The pitch rate consistency estimator should remain as a diagnostic tool only. It should NOT be used in active control unless a specific use case is identified where the filter lag is acceptable.
