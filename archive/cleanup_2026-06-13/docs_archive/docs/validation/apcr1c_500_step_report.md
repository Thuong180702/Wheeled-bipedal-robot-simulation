# APCR1c 500-Step Validation Report

## Summary

**Profile**: `APCR1c_active_pitch_crossing_early_activation`
**Test**: 500-step simulation at low_0p300 (0.30 m)
**Result**: PASS - Proceed to 2000-step validation

## Classification: `APCR1C_500_PASS_PROCEED_TO_2000`

## Key Findings

1. **APCR1c activates earlier than APCR1b** (outer_enter_m=0.08 vs 0.10)
2. **Outside ±0.15 reduced to 12.6%** (vs APCR1b's 13.8%, D2's 19.2%)
3. **Positive% reduced to 77.8%** (vs APCR1b's 79.2%, D2's 93.2%)
4. **Mean signed error reduced to 0.0620** (vs APCR1b's 0.066, D2's 0.0824)
5. **Final signed error is -0.0713**, within ±0.10 target
6. **Zero crossings not excessive** (5, same as APCR1b, fewer than APCR1's 8)
7. **No negative overshoot** (outside -0.15 = 0%)
8. **Contact/height/roll stable throughout** (100% contact, min CoM 0.285 m)

## APCR1c Profile Parameters

| Parameter | Value |
|-----------|-------|
| `outer_enter_m` | 0.08 (CHANGED from 0.10) |
| `inner_exit_m` | 0.07 (same as APCR1b) |
| `opposite_overshoot_m` | 0.00 (same as APCR1b) |
| `max_cross_tau` | 1.0 Nm (same as APCR1b) |
| `max_rate_per_step` | 0.4 Nm/step (same as APCR1b) |
| `recovery_gate_mode` | True |

## Signed Error Analysis

| Metric | D2 | APCR1 | APCR1b | APCR1c |
|--------|-----|-------|--------|--------|
| Mean | 0.0824 | 0.0674 | 0.066 | 0.0620 |
| Median | - | - | - | 0.0654 |
| Final | - | - | - | -0.0713 |
| Min | - | - | - | -0.0716 |
| Max | - | - | - | 0.1682 |
| RMS | - | - | - | 0.0915 |
| MAE | - | - | - | 0.0746 |
| Positive% | 93.2% | 79.4% | 79.2% | 77.8% |
| Negative% | - | - | - | 22.0% |
| Zero crossings | 2 | 8 | 5 | 5 |

## Band Violations

| Metric | D2 | APCR1 | APCR1b | APCR1c |
|--------|-----|-------|--------|--------|
| Outside +0.15 | - | - | - | 12.6% |
| Outside -0.15 | - | - | - | 0.0% |
| **Outside ±0.15** | **19.2%** | **13.8%** | **13.8%** | **12.6%** |

APCR1c achieves the lowest band violation rate among all profiles.

## APCR Behavior

| Metric | APCR1 | APCR1b | APCR1c |
|--------|-------|--------|--------|
| Active% | - | 44.2% | 46.4% |
| NEUTRAL% | - | - | 53.6% |
| CROSS_FROM_POSITIVE% | - | - | 46.4% |
| CROSS_FROM_NEGATIVE% | - | - | 0.0% |
| State entries | - | - | 2 |
| State exits | - | - | 2 |

APCR1c is more active than APCR1b (46.4% vs 44.2%), which is expected given earlier entry threshold.

## Stability Metrics

| Metric | Value |
|--------|-------|
| Pitch mean | 0.0466 rad (2.67°) |
| Pitch min | -0.0618 rad (-3.54°) |
| Pitch max | 0.1209 rad (6.93°) |
| Pitch RMS | 0.0691 rad (3.96°) |
| Roll RMS | 0.0101 rad (0.58°) |
| Hip yaw abs_max max | 0.1621 rad (9.29°) |
| Wheel vel RMS | 2.7732 rad/s |
| Contact valid% | 100.0% |
| CoM Z min | 0.2850 m |
| Height error mean | 0.0034 m |

All stability metrics are within acceptable ranges.

## Comparison vs APCR1b

| Metric | APCR1b | APCR1c | Change |
|--------|--------|--------|--------|
| outer_enter_m | 0.10 | 0.08 | -20% |
| Mean signed error | 0.066 | 0.0620 | -6.1% |
| Positive% | 79.2% | 77.8% | -1.4pp |
| **Outside ±0.15** | **13.8%** | **12.6%** | **-1.2pp** |
| Zero crossings | 5 | 5 | same |
| APCR active% | 44.2% | 46.4% | +2.2pp |

The earlier entry threshold (0.08 vs 0.10) enables:
- 6.1% reduction in mean signed error
- 1.2pp reduction in band violations
- 2.2pp increase in APCR activation

## Conclusion

APCR1c successfully reduces band violations compared to APCR1b by entering earlier (0.08 m vs 0.10 m). The key improvements are:

1. **Band violations**: 12.6% vs APCR1b's 13.8% (-1.2pp)
2. **Positive bias**: 77.8% vs APCR1b's 79.2% (-1.4pp)
3. **Mean signed error**: 0.0620 vs APCR1b's 0.066 (-6.1%)
4. **Stability preserved**: 100% contact, pitch/roll stable

APCR1c achieves the best metrics across all profiles tested (D2, APCR1, APCR1b).

## Next Step

Proceed to Phase 6: Run APCR1c 2000-step validation.