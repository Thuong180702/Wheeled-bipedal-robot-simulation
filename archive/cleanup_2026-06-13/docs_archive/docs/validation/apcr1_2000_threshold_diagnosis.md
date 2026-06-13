# APCR1 2000-Step Threshold Diagnosis

## Classification: APCR_DRIFT_FROM_LATE_RELEASE

## Summary

APCR1 significantly improves positive bias (98.3% → 72.7%) but introduces oscillation that causes more frequent band violations (4.8% → 12.2%). The oscillation pattern suggests APCR is holding CROSS_FROM_POSITIVE too long before releasing, causing overshoot.

## Diagnosis

### Evidence for LATE RELEASE

1. **APCR oscillation pattern**: APCR1 alternates between positive and negative drift, with 19 zero crossings vs D2's 5 crossings.

2. **Window analysis shows late release pattern**:
   - Steps 500-1000: APCR1 positive% = 70.2%, outside band = 18.6%
   - Steps 1000-1500: APCR1 positive% = 71.0%, outside band = 9.2%
   - Steps 1500-2000: APCR1 positive% = 70.2%, outside band = 7.2%
   
   The oscillation persists across all windows, suggesting APCR repeatedly enters CROSS_FROM_POSITIVE, holds too long, then overshoots negative.

3. **APCR1 successfully reduces bias but creates overshoot**: 
   - Mean: 0.0646 → 0.0616 (4.6% improvement)
   - Positive%: 98.3% → 72.7% (26% improvement)
   - Outside band: 4.8% → 12.2% (worse)

4. **Final signed error**:
   - D2: 0.0979 (stays positive, drifts away from zero)
   - APCR1: 0.0047 (very close to zero)

### Root Cause Analysis

The APCR1 parameters are:
- `outer_enter_m = 0.10`: Activates when signed error > 0.10 m
- `inner_exit_m = 0.05`: Exits when signed error enters within 0.05 m of zero
- `opposite_overshoot_m = 0.01`: Allows slight overshoot to opposite side

**Problem**: The exit threshold (`inner_exit_m = 0.05`) is too strict. When APCR is in CROSS_FROM_POSITIVE and the signed error decreases toward zero, it waits until the error is ≤ 0.05 before releasing. By that time, momentum carries it past zero into negative territory.

### Fix Candidates

#### Option A: Raise inner_exit_m (earlier release)
Change `inner_exit_m = 0.05` to `inner_exit_m = 0.07`:
- Allows exit when signed error is still positive (0.07 m)
- Reduces overshoot
- May reduce effectiveness if exit too early

#### Option B: Reduce opposite_overshoot_m (stricter symmetric exit)
Change `opposite_overshoot_m = 0.01` to `opposite_overshoot_m = 0.00`:
- Forces symmetric exit (exit when signed error crosses zero)
- Prevents accumulation of negative overshoot
- May cause premature oscillation if combined with early release

#### Option C: Combine (APCR1b_early_release)
```yaml
inner_exit_m: 0.07  # was 0.05
opposite_overshoot_m: 0.00  # was 0.01
```

This combination:
1. Releases earlier when signed error is still moderately positive
2. Prevents accumulating negative overshoot
3. Should reduce oscillation magnitude while maintaining bias improvement

### Why Not Insufficient Torque?

If APCR had insufficient torque, we would see:
- APCR activating but drift not reversing
- Pitch rate not changing direction after APCR entry
- Drift continuing in the same direction despite APCR

Instead, we see:
- Pitch rate DOES reverse (negative signed error achieved)
- Drift DOES cross to negative (27.2% negative time)
- The problem is REVERSAL TOO STRONG / HOLDING TOO LONG

### Recommendation

**APCR1b_early_release**: Raise inner_exit_m to 0.07 and set opposite_overshoot_m to 0.00.

This should reduce oscillation magnitude while maintaining the bias improvement. Run 500-step validation to verify before proceeding to 2000-step.
