# E1_support_integral 500-Step Gate Fix Report

## Summary

| Phase | Classification | Key Finding |
|-------|---------------|-------------|
| E1 500-step before-fix | `E1_500_NO_EFFECT` | Integral blocked 95.6% of steps by pitch_error_large gate |
| E1 wiring/gate audit | `E1_WIRING_OK_GATE_BLOCKS_EFFECT` | Wiring correct, gate threshold 0.03 rad too restrictive |
| E1 500-step after-fix | `E1_500_AFTER_FIX_NO_EFFECT` | Integral activates more (39 vs 22 steps) but support unchanged |
| **Support Metric Provenance Audit** | `E1_ANALYZER_USED_WRONG_SUPPORT_METRIC` | **E1 analyzer used abs(cp_x) instead of support_position_error_m** |

## CRITICAL CORRECTION

**The original E1 analysis used the WRONG support metric**, causing a 37x discrepancy with the official Step E report.

| Source | Metric Used | Value (first 500 rows) |
|--------|-------------|------------------------|
| Official D2 report | `support_position_error_m` (Euclidean distance) | **0.176 m** |
| Original E1 analyzer | `abs(cp_x)` (raw capture point x) | **0.0047 m** |

**Root cause**: `scripts/analyze_e1_500_before_fix.py` line 33 used `abs(cp_x)` instead of `support_position_error_m`.

**Fix applied**: Changed to use `support_position_error_m` column.

## E1 500-Step Before-Fix Result (Corrected)

- **Simulation**: 500 steps completed, survived
- **Classification**: `E1_500_NO_EFFECT`
- **Evidence**: E1 and D2 telemetry nearly identical using correct metric

### Key Metrics (E1 before vs D2 first 500, using CORRECTED metric)

| Metric | E1 Before | D2 | Delta |
|--------|-----------|-----|-------|
| support_position_error max (m) | **0.175687** | **0.175687** | 0.0 |
| support_position_error mean (m) | **0.082716** | **0.082715** | +0.000001 |
| first crossing > 0.15m | step 91 | step 91 | 0 |
| crossings > 0.15 count | 96 | 96 | 0 |
| hip_yaw_abs_max (rad) | 0.101796 | 0.101795 | +0.000001 |
| pitch_x_max (rad) | 0.111053 | 0.111053 | 0.0 |
| contact_valid% | 99.8% | 99.8% | 0.0% |

### E1 Integral Diagnostics (Before Fix)

| Field | Value |
|-------|-------|
| integral_active count | 22/500 (4.4%) |
| tau_position_integral max (Nm) | 0.001001 |
| tau_position_integral mean (Nm) | 0.000017 |

### E1 Gate Reasons (Before Fix)

| Reason | Count | Percent |
|--------|-------|---------|
| pitch_error_large | 349 | 69.8% |
| pitch_rate_large | 106 | 21.2% |
| safe_steady_state | 22 | 4.4% |
| support_velocity_large | 22 | 4.4% |
| contact_invalid | 1 | 0.2% |

## Root Cause Analysis

The **pitch_error_large gate** (threshold 0.03 rad) blocked the integral for 349/500 steps (69.8%).

At low_0p300:
- Pitch oscillates with max **0.111 rad (6.4 deg)**
- The 0.03 rad threshold is exceeded for 69.8% of steps
- The integral was never given a chance to accumulate

This is a **gate design flaw**, not a wiring flaw. The wiring was verified correct:
- E1 profile exists and sets `enable_position_integral=True`
- Profile parameters (ki=2.0, max=1.0) are passed to controller
- `tau_position_integral` is added to `tau_position_raw`
- Telemetry reports integral fields correctly

## Fix Applied

**Change**: Raised `integral_pitch_error_threshold_rad` from 0.03 to 0.12 rad in E1_support_integral profile.

**Location**: `scripts/simulate_hierarchical_controller.py`, line 204.

**Rationale**:
- 0.12 rad (6.9 deg) allows integral to activate during normal low_0p300 pitch oscillations
- Still protects against extreme pitch events indicating fall risk
- Smallest safe fix addressing the root cause

## Tests After Fix

All required tests passed (90 total):
- `test_sagittal_velocity_damped_balance_controller.py`: 51 passed
- `test_step_e_wbc_gate_validator.py`: 4 passed
- `test_balance_core_height_variant_setup*.py`: 26 passed
- `test_shape_posture_hip_yaw_sign.py`: 9 passed

## E1 500-Step After-Fix Result (Corrected)

### Integral Diagnostics (After Fix vs Before)

| Field | Before | After | Delta |
|-------|--------|-------|-------|
| integral_active count | 22 (4.4%) | 39 (7.8%) | +17 steps |
| tau_position_integral max (Nm) | 0.001001 | 0.030342 | +0.029341 |
| tau_position_integral mean (Nm) | 0.000017 | 0.000540 | +0.000523 |

### E1 Gate Reasons (After Fix)

| Reason | Count | Percent |
|--------|-------|---------|
| pitch_rate_large | 303 | 60.6% |
| support_velocity_large | 157 | 31.4% |
| safe_steady_state | 39 | 7.8% |
| contact_invalid | 1 | 0.2% |

**Note**: `pitch_error_large` gate reason is **eliminated** (0 steps) because 0.12 rad threshold is now above max pitch of 0.111 rad.

### Support Position Error (After vs Before, Corrected Metric)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| max (m) | 0.175687 | 0.175687 | 0.0 |
| mean (m) | 0.082716 | 0.082714 | -0.000002 |
| first crossing > 0.15m | step 91 | step 91 | 0 |
| crossings > 0.15 count | 96 | 96 | 0 |

The support_position_error is **identical** across all runs using the corrected metric.

## Critical Finding: D2 Baseline Violates Support Gate

**The D2 baseline itself crosses the 0.15 m support gate at step 91** and has 96 crossings > 0.15 m out of 500 steps. This is a **baseline issue**, not an E1 issue.

## Interpretation

The fix successfully:
1. **Eliminated** pitch_error_large gate blocking (349 → 0 steps)
2. **Increased** integral activation (22 → 39 steps, +77%)
3. **Increased** integral magnitude (0.001 → 0.030 Nm, +30x)

However, support_position_error is **identical** across all runs:
- D2, E1_before, and E1_after all have max 0.175687 m
- All cross 0.15 m at step 91
- All have 96 crossings > 0.15 m

This means:
1. The integral magnitude (max 0.030 Nm) is too small relative to other torques (~7 Nm)
2. OR support drift at low_0p300 is dominated by factors the integral cannot address
3. The D2 baseline itself has significant support drift (violates 0.15 m gate)

## Final Decision

**Classification**: `E1_AFTER_FIX_NO_EFFECT_ON_OFFICIAL_SUPPORT_METRIC`

**Reason**: E1 with raised pitch threshold produces more integral activity, but this does not translate to measurable support_position_error improvement at 500 steps using the official metric.

**Recommendation**: Stop E1 tuning. The integral has no measurable effect on the official support metric. Consider:
1. Addressing the D2 baseline support drift issue directly
2. Increasing integral gain significantly (ki=10.0 or higher)
3. Running longer simulations to see if integral effect accumulates

## Next Steps

1. **Do NOT proceed to 2000-step validation yet** - the fix is correct but effect is marginal at 500 steps
2. Consider increasing `ki_position_integral` to 5.0 or 10.0 for the next iteration
3. Alternatively, run E1 at 2000 steps to see if integral effect accumulates over time
4. Do NOT commit until effect is demonstrated at reasonable horizon

## Files Created/Modified

| File | Action |
|------|--------|
| `scripts/simulate_hierarchical_controller.py` | Modified (line 204) |
| `scripts/analyze_e1_500_before_fix.py` | Modified (line 33 - fixed support metric) |
| `docs/validation/support_metric_provenance_audit.md` | Created |
| `outputs/step_e_extreme_support_fix_eval/support_metric_provenance_audit.json` | Created |
| `scripts/corrected_e1_500_comparison.py` | Created |
| `docs/validation/e1_support_integral_500_step_corrected_metric_comparison.md` | Created |
| `outputs/step_e_extreme_support_fix_eval/e1_500_corrected_metric_comparison.json` | Created |
| `outputs/step_e_extreme_support_fix_eval/e1_support_metric_provenance_final_summary.json` | Created |
