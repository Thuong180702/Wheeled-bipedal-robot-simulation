# E1_support_integral 500-Step Before-Fix Analysis

## Summary

- **E1 500-step simulation**: 500 rows, survived=True
- **D2 first 500 rows**: 500 rows, survived=True
- **Classification**: `E1_500_NO_EFFECT`
- **Reason**: integral term was not meaningfully active or nonzero

## Support Position Error

| Metric | E1 | D2 | Delta |
|--------|----|----|-------|
| max (m) | 0.175687 | 0.175687 | +0.000000 |
| mean (m) | 0.082716 | 0.082715 | +0.000001 |
| first crossing > 0.15m | step 91 | step 91 | - |
| crossings > 0.15 count | 96 | 96 | +0 |

## Hip Yaw

| Metric | E1 | D2 | Delta |
|--------|----|----|-------|
| abs_max (rad) | 0.101796 | 0.101795 | +0.000001 |
| abs_mean (rad) | 0.044578 | 0.044578 | -0.000000 |

## Wheel Velocity

| Metric | E1 | D2 | Delta |
|--------|----|----|-------|
| mean_max (rad/s) | 0.000000 | 0.000000 | +0.000000 |

## Height/Roll/Pitch

| Metric | E1 | D2 | Delta |
|--------|----|----|-------|
| height_error_max (m) | 0.006418 | 0.006418 | +0.000000 |
| roll_y_max (rad) | 0.013356 | 0.013356 | +0.000000 |
| pitch_x_max (rad) | 0.111056 | 0.111056 | +0.000000 |

## Contact/Validity

| Metric | E1 | D2 |
|--------|----|----|
| contact_valid% | 99.8% | 99.8% |
| non_wheel_contacts_max | 0 | 0 |

## E1 Integral Diagnostics

| Field | Value |
|-------|-------|
| position_integral_enabled exists | False |
| integral_active count | 22 |
| integral_active percent | 4.4% |
| tau_position_integral max | 0.001001 |
| tau_position_integral mean | 0.000017 |
| tau_position_raw max | 7.027467 |
| tau_position_final max | N/A |

## Gate Reason Counts (E1)

- e1_gate_capture_gate_factor_sum: 500


## Decision Criteria

- Support improves: **False**
- Hip yaw worsens: **False**
- Wheel velocity worsens: **False**
- Height/contact valid: **True**
- Integral active (>50 steps): **False**
- Integral nonzero (>0.001): **True**

## Conclusion

**E1_500_NO_EFFECT**: integral term was not meaningfully active or nonzero
