# E1_support_integral Corrected Metric Comparison

## Summary

**Classification**: `E1_AFTER_FIX_NO_EFFECT_ON_OFFICIAL_SUPPORT_METRIC`

After correcting the E1 analyzer to use the official Step E support metric (`support_position_error_m`), the comparison shows:
- E1 and D2 have **identical** support position error profiles
- The integral fix increases integral activity but has no measurable effect on support drift
- D2 baseline itself violates the 0.15 m gate at step 91

## Root Cause of Contradiction

**Original E1 analyzer** used `abs(cp_x)` instead of `support_position_error_m`:
- `cp_x` = capture point x position (~0.005 m magnitude)
- `support_position_error_m` = Euclidean distance of support_center error (~0.176 m max)

**Fix applied**: Changed line 33 in `scripts/analyze_e1_500_before_fix.py` to use `support_position_error_m`.

## Corrected 500-Step Comparison

| Metric | D2 | E1_before | E1_after | Delta (after vs D2) |
|--------|-----|-----------|----------|---------------------|
| support_position_error max (m) | 0.175687 | 0.175687 | 0.175687 | 0.0 |
| support_position_error mean (m) | 0.082715 | 0.082716 | 0.082714 | -0.000001 |
| support_position_error final (m) | 0.057986 | 0.057982 | 0.057870 | -0.000116 |
| first crossing > 0.15m | step 91 | step 91 | step 91 | 0 |
| crossings > 0.15 count | 96 | 96 | 96 | 0 |

### Other Metrics

| Metric | D2 | E1_before | E1_after |
|--------|-----|-----------|----------|
| hip_yaw_abs_max (rad) | 0.101795 | 0.101796 | 0.101796 |
| wheel_vel_mean_max (rad/s) | 4.388715 | 4.388719 | 4.388719 |
| contact_valid_percent | 99.8% | 99.8% | 99.8% |
| height_error_max (m) | 0.006418 | 0.006418 | 0.006418 |
| roll_max (rad) | 0.013343 | 0.013343 | 0.013343 |
| pitch_max (rad) | 0.111053 | 0.111053 | 0.111053 |

## E1 Integral Diagnostics

| Field | E1_before | E1_after | Delta |
|-------|-----------|----------|-------|
| integral_active count | 22 (4.4%) | 39 (7.8%) | +17 steps |
| tau_position_integral max (Nm) | 0.001001 | 0.030342 | +0.029341 |
| tau_position_integral mean (Nm) | 0.000017 | 0.000540 | +0.000523 |
| tau_position_raw max (Nm) | 7.027467 | 7.027467 | 0 |

### Gate Reason Counts

| Reason | E1_before | E1_after |
|--------|-----------|----------|
| pitch_error_large | 349 | 0 (eliminated) |
| pitch_rate_large | 106 | 303 |
| support_velocity_large | 22 | 157 |
| safe_steady_state | 22 | 39 |
| contact_invalid | 1 | 1 |

## Interpretation

The fix successfully:
1. **Eliminated** pitch_error_large gate blocking (349 → 0 steps)
2. **Increased** integral activation (22 → 39 steps, +77%)
3. **Increased** integral magnitude (0.001 → 0.030 Nm, +30x)

However, the support_position_error metric is **identical** across all runs:
- D2, E1_before, and E1_after all have max 0.175687 m
- All cross 0.15 m at step 91
- All have 96 crossings > 0.15 m

This means:
1. The integral magnitude (max 0.030 Nm) is too small relative to other torques (~7 Nm)
2. OR support drift at low_0p300 is dominated by factors the integral cannot address
3. The D2 baseline itself has significant support drift (violates 0.15 m gate)

## Critical Finding: D2 Baseline Violates Support Gate

The D2 baseline itself crosses the 0.15 m support gate at step 91 and has 96 crossings > 0.15 m out of 500 steps. This is a **baseline issue**, not an E1 issue.

## Phase 5 Decision

**Classification**: `E1_AFTER_FIX_NO_EFFECT_ON_OFFICIAL_SUPPORT_METRIC`

**Recommendation**: Stop E1 tuning. The integral has no measurable effect on the official support metric. Consider:
1. Addressing the D2 baseline support drift issue directly
2. Increasing integral gain significantly (ki=10.0 or higher)
3. Running longer simulations to see if integral effect accumulates
