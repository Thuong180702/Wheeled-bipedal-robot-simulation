# APCR1 2000-Step Validation Final Report

## Final Decision: APCR1_2000_IMPROVES_BUT_DRIFT_STILL_FAR

## Executive Summary

APCR1 **significantly improves the primary goal** (positive bias reduction from 98.3% to 72.7%) but introduces oscillation that causes more frequent band violations (4.8% to 12.2% outside ±0.15 m). The diagnosis indicates LATE RELEASE is the root cause, and APCR1b with earlier release thresholds is recommended.

## Key Findings

### 1. Does APCR1 continue to keep drift inside ±0.15 better than D2?

**NO - APCR1 has more band violations than D2 at 2000 steps.**

| Metric | D2 | APCR1 | Change |
|--------|-----|-------|--------|
| Outside ±0.15 | 4.8% | 12.2% | Worse (+154%) |

However, D2's "good" band containment is misleading - D2 stays within band by consistently drifting positive. APCR1 oscillates but ends much closer to zero (final: 0.0047 vs 0.0979).

### 2. Does positive bias continue to decrease?

**YES - APCR1 significantly reduces positive bias.**

| Metric | D2 | APCR1 | Change |
|--------|-----|-------|--------|
| Positive % | 98.3% | 72.7% | **-26%** |
| Mean signed error | 0.0646 | 0.0616 | -4.6% |
| Final signed error | 0.0979 | 0.0047 | **Much closer to zero** |

### 3. Does drift accumulate again after 500 steps?

**NO - APCR1 maintains consistent oscillation pattern across all windows.**

Window analysis shows:
- Steps 0-500: APCR1 positive% = 79.4%, outside band = 13.8%
- Steps 500-1000: APCR1 positive% = 70.2%, outside band = 18.6%
- Steps 1000-1500: APCR1 positive% = 71.0%, outside band = 9.2%
- Steps 1500-2000: APCR1 positive% = 70.2%, outside band = 7.2%

The bias reduction is maintained throughout the run. The oscillation pattern is consistent.

### 4. Does pitch/hip-yaw/wheel velocity blow up?

**NO - All within acceptable bounds.**

| Metric | D2 | APCR1 | Assessment |
|--------|-----|-------|------------|
| Pitch RMS (deg) | 0.0562 | 0.0701 | Slightly worse (+25%) |
| Roll RMS (deg) | 0.0058 | 0.0061 | Similar |
| Wheel velocity RMS (rad/s) | 1.69 | 3.13 | Higher (expected for oscillation) |

### 5. Are contact/height/roll stable?

**YES - Both survive 2000 steps with stable metrics.**

| Metric | D2 | APCR1 |
|--------|-----|-------|
| Survived | 2000 steps | 2000 steps |
| Height min (m) | 0.2816 | 0.2787 |
| Height mean (m) | 0.2874 | 0.2869 |

## Root Cause Diagnosis

### Classification: APCR_DRIFT_FROM_LATE_RELEASE

The oscillation problem is caused by **late release**. The APCR holds CROSS_FROM_POSITIVE too long before exiting, resulting in negative overshoot.

**Evidence:**
1. APCR1 has 19 zero crossings vs D2's 5 crossings - more oscillation
2. APCR1 min signed error (-0.0805) exceeds the opposite threshold more often
3. Window analysis shows consistent negative time (27-30%) across all windows

**APCR1 current parameters causing the issue:**
- `inner_exit_m = 0.05`: Too strict, waits too long before releasing
- `opposite_overshoot_m = 0.01`: Allows negative accumulation

## APCR1b Design Recommendation

### Parameters to Change

| Parameter | APCR1 | APCR1b |
|-----------|-------|--------|
| `inner_exit_m` | 0.05 | **0.07** |
| `opposite_overshoot_m` | 0.01 | **0.00** |

### Expected Outcome

| Metric | D2 | APCR1 | APCR1b Target |
|--------|-----|-------|---------------|
| Positive % | 98.3% | 72.7% | <75% |
| Outside ±0.15 | 4.8% | 12.2% | <8% |
| Zero crossings | 5 | 19 | 8-12 |

## Recommendations

1. **Do NOT run 5000-step validation** with APCR1 yet - the oscillation causes excessive band violations.

2. **Implement APCR1b** with earlier release (inner_exit_m=0.07, opposite_overshoot_m=0.00).

3. **Run APCR1b 500-step validation** to verify the fix reduces oscillation while maintaining bias improvement.

4. **If APCR1b 500-step shows improvement**, run APCR1b 2000-step validation.

5. **Do NOT change D2 baseline** - it remains the reference.

## Files Generated

- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1_2000_comparison.json`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1_2000_window_metrics.csv`
- `docs/validation/apcr1_2000_threshold_diagnosis.md`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1_2000_threshold_diagnosis.json`
- `docs/validation/apcr1b_threshold_candidate_design.md`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1b_threshold_candidate_design.json`
- `docs/validation/apcr1_2000_validation_final_report.md` (this file)
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1_2000_validation_summary.json` (following)
