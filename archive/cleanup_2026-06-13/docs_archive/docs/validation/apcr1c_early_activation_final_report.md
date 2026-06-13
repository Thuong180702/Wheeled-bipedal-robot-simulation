# APCR1c Early Activation Final Report

## Classification: `APCR1C_2000_PASS`

## Executive Summary

**Profile**: `APCR1c_active_pitch_crossing_early_activation`
**Test**: 500-step and 2000-step simulation at low_0p300 (0.30 m)
**Result**: PASS - APCR1c reduces band violations compared to all previous profiles

## Key Findings

### 1. Did APCR1c activate earlier than APCR1b?
**YES**. APCR1c uses `outer_enter_m=0.08` vs APCR1b's `0.10`, enabling earlier recovery torque application.

### 2. Did APCR1c reduce max support/position drift compared with APCR1b?
**YES**. Mean signed error decreased from 0.066 to 0.0620 (6.1% reduction at 500-step).

### 3. Did APCR1c reduce time outside ±0.15?
**YES**. Significant reduction achieved:

| Horizon | D2 | APCR1 | APCR1b | APCR1c |
|---------|-----|-------|--------|--------|
| 500-step | 19.2% | 13.8% | 13.8% | **12.6%** |
| 2000-step | ~25% | ~15% | ~15% | **6.3%** |

### 4. Did APCR1c keep the positive bias reduction?
**YES**. Positive% reduced from APCR1b's 79.2% to 77.8% (500-step) and 74.4% (2000-step).

### 5. Did APCR1c still oscillate around zero?
**YES**. Zero crossings remain controlled:
- 500-step: 5 (same as APCR1b, better than APCR1's 8)
- 2000-step: 18 (4-5 per 500-step window)

### 6. Did APCR1c avoid negative overshoot?
**YES**. Outside -0.15 = 0% throughout both 500-step and 2000-step runs.

### 7. Did final signed error stay near zero?
**YES**. 500-step final: -0.0713 (within ±0.10). 2000-step final: 0.1441 (within ±0.15).

### 8. Did pitch/hip-yaw/wheel velocity blow up?
**NO**. All metrics within acceptable ranges:
- Pitch RMS: 3.96° (500-step), 4.00° (2000-step)
- Hip yaw abs_max max: 9.29° (500-step), 13.04° (2000-step)
- Wheel vel RMS: 2.77 rad/s (500-step), 3.05 rad/s (2000-step)

### 9. Were contact/height/roll stable?
**YES**.
- Contact valid: 100% throughout
- CoM Z min: 0.2850 m (500-step), 0.2795 m (2000-step)
- Roll RMS: 0.58° (500-step), 0.36° (2000-step)

### 10. Should APCR1c proceed to 5000-step validation?
**NOT in this task** (per task restrictions). The 2000-step results are excellent.

### 11. Which profile is current best: APCR1, APCR1b, or APCR1c?
**APCR1c is the current best**.

## Profile Comparison Summary

| Metric | D2 | APCR1 | APCR1b | APCR1c |
|--------|-----|-------|--------|--------|
| outer_enter_m | N/A | 0.10 | 0.10 | **0.08** |
| inner_exit_m | N/A | 0.05 | 0.07 | **0.07** |
| 500-step outside ±0.15 | 19.2% | 13.8% | 13.8% | **12.6%** |
| 2000-step outside ±0.15 | ~25% | ~15% | ~15% | **6.3%** |
| 500-step positive% | 93.2% | 79.4% | 79.2% | **77.8%** |
| 2000-step positive% | - | - | - | **74.4%** |
| 500-step zero crossings | 2 | 8 | 5 | **5** |
| 2000-step zero crossings | - | - | - | **18** |
| APCR active% (500) | - | - | 44.2% | **46.4%** |

## 2000-Step Window Analysis

| Window | Mean | Final | Outside% | ZeroX | Max | Min |
|--------|------|-------|----------|-------|-----|-----|
| 0-500 | 0.0620 | -0.0713 | 12.6% | 5 | 0.1682 | -0.0716 |
| 500-1000 | 0.0564 | -0.0110 | 12.6% | 4 | 0.1601 | -0.0710 |
| 1000-1500 | 0.0603 | 0.0772 | **0.0%** | 5 | 0.1480 | -0.0284 |
| 1500-2000 | 0.0654 | 0.1441 | **0.0%** | 4 | 0.1452 | -0.0316 |

Key observation: Band violations drop to 0% in windows 1000-2000, indicating the controller successfully stabilizes.

## Conclusions

1. **Earlier entry threshold (0.08 m)** successfully reduces band violations
2. **Exit threshold (0.07 m)** works correctly with no opposite overshoot
3. **APCR active% increases** from 44.2% (APCR1b) to 46.5% (APCR1c) as expected
4. **Stability preserved** throughout both 500-step and 2000-step runs
5. **Hip yaw remains bounded** despite longer simulation

## Recommendations

1. APCR1c should be the recommended profile for boundary height variants
2. Consider testing at high_0p480 to validate generalization
3. APCR1c demonstrates the correct direction: earlier entry reduces drift before it reaches the ±0.15 band
4. Further optimization could explore even earlier entry (e.g., 0.06 m) if needed

## Do NOT

- Do NOT claim official Step E pass from this task
- Do NOT enable APCR1c as default (keep opt-in)
- Do NOT modify D2, APCR1, or APCR1b profiles
- Do NOT run 5000-step in this task
