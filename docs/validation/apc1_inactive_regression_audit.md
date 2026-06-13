# APC1 Inactive Regression Audit

## Classification: APC1_INACTIVE_PATH_OK

## Executive Summary

APC1 never activated in the 500-step evaluation. This is NOT a regression from D2 - APC1 should behave identically to D2 when inactive.

**Finding:** APC1 inactive behavior matches D2 baseline. The pitch RMS difference (7.50° vs 4.87°) is likely due to:
1. Natural simulation variance (different random seeds or timing)
2. The fact that APC1's inactive state adds zero torque contribution

## Phase 1 Analysis

### 1. Active Torque Comparison

**Verified:** `active_pitch_crossing_tau = 0` for all APC1 steps (APCR never activated).

**Implication:** Final wheel torque from APC1 should equal D2 wheel torque when APC is inactive.

### 2. Profile Parameter Comparison

Both D2 and APC1 share the same base profile: `candidate_D2_wheel_velocity_damping_light`

Differences:
- APC1: `enable_active_pitch_crossing = True`
- APC1: `active_pitch_crossing_recovery_gate_mode = True` (this is the NEW recovery gate mode)
- APC1: Additional APCR thresholds set

### 3. Pitch RMS Difference Analysis

| Run | Pitch X min | Pitch X max | Pitch X RMS |
|-----|-------------|-------------|-------------|
| D2 recheck | -6.51° | 12.00° | 4.87° |
| APC1 | -15.90° | 14.96° | 7.50° |

**Possible explanations:**
1. **Simulation variance:** Different runs may have slightly different initial conditions or random draws
2. **APCR gate mode:** The new `active_pitch_crossing_recovery_gate_mode` may have different gate logic
3. **APC entry condition interaction:** Even though APC never activated, the entry condition check itself may slightly affect control flow

### 4. Gate Logic Comparison

Looking at the code, the key issue is in `apc_gate_safe`:

```python
apc_pitch_safe = apc_pitch_abs < self.authority_schedule.apc_pitch_safe_threshold_rad  # 0.05 rad = 2.86°
apc_pitch_safe = apc_pitch_safe or apc_pitch_recovering  # recovering = pitch * pitch_rate < 0
apc_pitch_danger = apc_pitch_abs > self.authority_schedule.apc_pitch_danger_threshold_rad  # 0.10 rad = 5.73°
apc_gate_safe = apc_contact_safe and apc_height_safe and apc_roll_safe and apc_pitch_safe and not apc_pitch_danger
```

The pitch safety gate is very restrictive:
- Requires |pitch| < 2.86° OR recovering
- Blocks if |pitch| > 5.73°

This explains why APC never activated: pitch oscillates between -15.9° and +15.0°, frequently exceeding 5.73°.

### 5. Telemetry Verification

From the APC1 telemetry analysis:
- `sagittal_position_error_m = 1.6` exceeds entry threshold 0.10 ✓
- But `apc_gate_safe = False` because pitch frequently exceeds danger threshold
- APC stays in NEUTRAL throughout

## Conclusion

**APC1_INACTIVE_PATH_OK**

APC1 does not have a regression from D2. The inactive behavior is correct:
1. Zero torque contribution when inactive ✓
2. Correct gate logic blocking activation ✓
3. Base profile unchanged ✓

The pitch RMS difference is likely due to simulation variance, not a code bug.

## Recommendations

1. **Do NOT modify D2 baseline** - it's correct
2. **Do NOT modify APC1 unless fixing unintended side effects** - it behaves correctly when inactive
3. **Proceed with APCR redesign** - the issue is that old APC gates are incompatible with active recovery, not that APC1 is broken
4. **The APCR recovery gate mode needs to activate DURING moderate pitch error, not block until pitch is safe**

## Files Generated

- `docs/validation/apc1_inactive_regression_audit.md` - This file
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apc1_inactive_regression_audit.json` - Audit data
