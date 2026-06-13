# E2 Support Integral Cap 500-Step Evaluation Report

## Task Summary
Evaluate E2 (`E2_support_integral_higher_cap`) at low_0p300 for 500 steps against D2 baseline and E1.

## Profile Definition
- **Profile name**: `E2_support_integral_higher_cap`
- **Key difference from E1**: Position cap increased from 4.0 Nm to 5.0 Nm (+25%)
- **E1 vs E2**: E1 uses `integral_pitch_error_threshold_rad=0.12`, E2 uses `0.03`

## Simulation Results

### Official Support Metric (support_position_error_m)

| Metric | D2 | E1_before | E1_after | E2 | Delta vs D2 |
|--------|-----|-----------|----------|-----|-------------|
| max (m) | 0.175687 | 0.175687 | 0.175687 | **0.170277** | **-0.005410** |
| mean (m) | 0.082715 | 0.082716 | 0.082714 | **0.067657** | **-0.015058** |
| final (m) | 0.057986 | 0.057982 | 0.057870 | **0.027604** | **-0.030382** |
| crossings >0.15m | 96 | 96 | 96 | **62** | **-34** |
| first crossing | 91 | 91 | 91 | 89 | -2 |

**E2 significantly improves the official support metric**:
- Max reduced by 3.1%
- Mean reduced by 18.2%
- Final value reduced by 52.4%
- Violations >0.15m reduced from 96 to 62 (35% reduction)

### Position Authority

| Metric | D2 | E1_after | E2 |
|--------|-----|----------|-----|
| effective_max_position_tau (Nm) | 4.0 | 4.0 | **5.0** |
| tau_position_raw_max (Nm) | 7.0275 | 7.0275 | 6.8111 |
| tau_position_integral_max (Nm) | 0.0 | 0.0303 | 0.0076 |
| integral_active_count | 0 | 39 | 31 |
| integral_active_percent | 0% | 7.8% | 6.2% |

**E2 position cap increased to 5.0 Nm** (verified in telemetry)

### Step E Gate Status

| Gate | Threshold | D2 | E2 | Status |
|------|-----------|-----|-----|--------|
| hip_yaw_abs_max | <0.10 rad | 0.1018 | **0.1304** | **FAIL (+28%)** |
| wheel_vel_mean_max | - | 4.3918 | 4.3918 | PASS |
| contact_valid_percent | >95% | 99.8% | 99.8% | PASS |
| height_error_max | - | 0.006418 | 0.007978 | WARN (+24%) |
| roll_max | - | 0.0133 | 0.0152 | WARN (+14%) |
| pitch_max | record only | 0.1111 | 0.1244 | RECORD (+12%) |
| hidden_torque_max | 0 | 0.0 | 0.0 | PASS |
| ownership_violations_max | 0 | 0 | 0 | PASS |

## Classification

**E2_500_REGRESSES_OTHER_GATES**

## Analysis

### Support Improvement (Positive)
E2 shows clear improvement on the official support metric:
1. Max support error reduced from 0.1757m to 0.1703m
2. Mean support error reduced by 18%
3. Final support error reduced by 52%
4. Violation count reduced from 96 to 62

This is a **meaningful improvement** in the primary metric.

### Hip Yaw Regression (Critical)
- hip_yaw_abs_max increased from 0.1018 rad to 0.1304 rad
- This exceeds the 0.10 rad gate threshold by 30%
- The higher position cap (+25%) and/or lower pitch threshold (0.03 vs 0.12) appears to cause hip yaw divergence

### Root Cause Hypothesis
E2 uses `integral_pitch_error_threshold_rad=0.03` which is more restrictive than E1_after's `0.12`. This means:
1. The integral activates less frequently (31 steps vs 39 steps)
2. But the higher cap (5.0 Nm vs 4.0 Nm) allows more aggressive position corrections
3. These corrections may couple into hip yaw through WBC/posture interactions

## Decision

**E2_500_REGRESSES_OTHER_GATES**

### Reasoning
While E2 significantly improves support_position_error_m (the official Step E metric), it causes hip_yaw_abs_max to exceed the 0.10 rad gate threshold by 30%. This is a regression on a hard Step E gate.

### Recommendation
Do NOT proceed to 2000-step validation with current E2 settings.

If E2 is to be revisited:
1. The pitch threshold should be raised to 0.12 (same as E1_after) to allow more integral activation
2. Investigate why higher position cap causes hip yaw divergence
3. Consider decoupling hip yaw from position authority corrections

### Files Generated
- `outputs/step_e_extreme_support_fix_eval/e2_low_0p300_500/e2_low_0p300_500_telemetry.csv`
- `outputs/step_e_extreme_support_fix_eval/e2_low_0p300_500_comparison.json`
- `outputs/step_e_extreme_support_fix_eval/e2_low_0p300_500_comparison.csv`
- `docs/validation/e2_profile_verification.md`

## Final Decision
```
E2_500_REGRESSES_OTHER_GATES
```

Do NOT:
- Run 2000-step validation
- Run 5000-step validation
- Run Step C or Step D
- Commit changes

Next step: Audit why E2 causes hip_yaw regression before any further evaluation.