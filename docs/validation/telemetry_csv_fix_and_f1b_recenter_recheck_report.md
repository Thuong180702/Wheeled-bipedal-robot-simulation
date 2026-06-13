# Telemetry CSV Fix and F1b Recenter Recheck Report

## Executive Summary

**Decision: CSV_FIX_PASS_F1B_RECENTER_IMPROVES_BUT_NOT_ENOUGH**

The CSV writing bug has been fixed. F1b shows meaningful signed drift improvement vs D2:
- Positive percentage decreased from 93.0% to 82.8%
- Total time outside ±0.15 decreased from 96 to 81 steps
- Zero crossings increased from 4 to 5
- Recenter activates 65.8% of the time with proper tau direction

However, F1b still shows residual one-sidedness (positive% = 82.8%) and the recenter does not fully center the drift. F1b should NOT proceed to 2000-step validation yet.

## CSV Bug Fix Summary

### Root Cause
The CSV writing code used `min(len(values) for values in telemetry.values())` to determine row count. When any column was empty, `n_rows` became 0, causing all data rows to be skipped. Several telemetry columns (pitch_rate_estimator fields) were conditionally populated and left empty in balance-core mode.

### Fix Applied
Changed the n_rows calculation to use only non-empty columns:
```python
non_empty_cols = [v for v in telemetry.values() if len(v) > 0]
n_rows = min(len(v) for v in non_empty_cols) if non_empty_cols else 0
```

Also changed the row writing loop to handle columns with fewer entries than `n_rows`:
```python
for i in range(n_rows):
    row = []
    for k in telemetry.keys():
        if len(telemetry[k]) > i:
            row.append(telemetry[k][i])
        else:
            row.append(None)
    writer.writerow(row)
```

### Files Changed
- `scripts/simulate_hierarchical_controller.py`: CSV writing fix

### Tests Added
- `tests/test_simulation_telemetry_csv_writer.py`: 8 tests for telemetry CSV writing

## D2 vs F1b Comparison Results

### Simulation Results
| Metric | D2 | F1b |
|--------|-----|-----|
| Survived steps | 500 | 500 |
| Contact state | double_contact | double_contact |
| Hidden torque max | 0.0000 | 0.0000 |
| Ownership violations | 0 | 0 |

### Signed Support Drift (Primary Metric)
Source: `hip_yaw_comp_support_error_m`

| Metric | D2 | F1b | Change |
|--------|-----|-----|--------|
| Mean | 0.082333 | 0.076410 | -7.2% |
| Min | -0.003528 | -0.033859 | More negative |
| Max | 0.175687 | 0.169489 | -3.5% |
| Final | 0.059282 | -0.032706 | N/A |
| RMS | 0.100668 | 0.100412 | -0.3% |
| MAE | 0.082599 | 0.083156 | +0.7% |
| Positive % | 93.0% | 82.8% | -10.2 pp |
| Negative % | 6.6% | 16.8% | +10.2 pp |
| Zero crossings | 4 | 5 | +1 |
| Outside +0.15 | 96 | 81 | -15 steps |
| Outside -0.15 | 0 | 0 | No change |
| Outside total | 96 | 81 | -15 steps |
| Longest positive interval | 316 | 256 | -60 steps |
| Longest negative interval | 28 | 66 | +38 steps |

### Phase Recenter Behavior (F1b Specific)
| Metric | Value |
|--------|-------|
| Recenter active % | 65.8% |
| Recenter tau max | 0.9999 |
| Recenter tau mean | 0.4181 |
| Recenter tau final | 0.1151 |
| Recenter signed error mean | 0.076340 |

### Support Position Error (Magnitude)
| Metric | D2 | F1b |
|--------|-----|-----|
| Abs max | 0.175687 | 0.169489 |
| Abs mean | 0.082715 | 0.083226 |
| Crossings >0.15 | 96 | 81 |

### Stability (Monitor)
| Metric | D2 | F1b |
|--------|-----|-----|
| Pitch max deg | 6.36 | 6.32 |
| Pitch RMS deg | 3.60 | 4.04 |
| Pitch final deg | 2.72 | -1.40 |
| Roll max deg | 0.77 | 0.75 |
| Roll RMS deg | 0.51 | 0.55 |

### Hip Yaw (Monitor)
| Metric | D2 | F1b |
|--------|-----|-----|
| Abs max | 0.1018 | 0.1376 |
| Abs final | 0.0847 | 0.1376 |

### Wheel Velocity (Monitor)
| Metric | D2 | F1b |
|--------|-----|-----|
| Abs max | 4.3887 | 5.0049 |
| Abs mean | 1.7073 | 2.4779 |

## Analysis

### What F1b Improves
1. **Signed bias reduced**: Positive percentage dropped from 93.0% to 82.8% (10.2 pp improvement)
2. **Boundary crossings reduced**: Total time outside ±0.15 dropped from 96 to 81 steps
3. **More negative excursions**: Longest negative interval increased from 28 to 66 steps
4. **Recenter active**: 65.8% of steps have recenter active with tau in correct direction

### What F1b Does NOT Fix
1. **Still one-sided**: 82.8% positive is still far from 50%
2. **Wheel velocity increased**: Max 5.0 vs 4.4 rad/s, mean 2.5 vs 1.7 rad/s
3. **Hip yaw increased**: Max 0.138 vs 0.102 rad
4. **Pitch RMS increased**: 4.04 vs 3.60 deg (stability degraded slightly)

### Root Cause Assessment
The recenter correctly activates (65.8%) but its effect is insufficient to center the drift. Possible reasons:
1. Recenter torque (`tau_max=1.0`) is too small relative to the support drift rate
2. Recenter is gated off when pitch is dangerous (correct safety behavior)
3. The one-sidedness originates from dynamics, not just from lack of counter-torque

## Classification

### F1B_RECENTER_IMPROVES_BUT_NOT_ENOUGH

**Reasoning:**
- Signed bias improves (93% → 82.8% positive)
- Recenter activates correctly (65.8%)
- Contact/height/roll remain valid
- WBC/hidden/ownership remain clean
- BUT: 82.8% positive is still too one-sided
- BUT: Wheel velocity increased significantly (5.0 vs 4.4 max)
- BUT: Hip yaw increased (0.138 vs 0.102 max)
- F1b should NOT proceed to 2000-step validation yet

## Next Steps

1. **Do NOT run 2000-step F1b** - not ready
2. **Do NOT tune recenter gains** - per task restrictions
3. **Do NOT modify D2 baseline** - protected
4. **Document findings** - recenter logic works but is insufficient
5. **Consider**: F1b needs more aggressive recenter authority, but task prohibits tuning

## Files Created

### Scripts
- `scripts/compare_d2_f1b_telemetry.py` - D2 vs F1b comparison tool

### Tests
- `tests/test_simulation_telemetry_csv_writer.py` - Telemetry CSV writing tests

### Documentation
- `docs/validation/telemetry_csv_writing_bug_audit.md` - Bug audit
- `outputs/step_e_extreme_support_fix_eval/telemetry_csv_bug_fix/csv_writing_bug_audit.json` - JSON audit

### Outputs
- `outputs/step_e_extreme_support_fix_eval/d2_low_0p300_500_after_csv_fix/d2_low_0p300_500_telemetry.csv` - D2 CSV
- `outputs/step_e_extreme_support_fix_eval/f1b_low_0p300_500_after_csv_fix/f1b_low_0p300_500_telemetry.csv` - F1b CSV
- `outputs/step_e_extreme_support_fix_eval/f1b_low_0p300_500_after_csv_fix_comparison.json` - Comparison JSON