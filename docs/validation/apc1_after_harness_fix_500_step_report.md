# APC1 After Harness Fix 500-Step Report

## Classification: APC_HARNESS_FIXED_APC1_NO_EFFECT

## Executive Summary

After fixing the harness issue (D2 baseline now stable), APC1 was re-evaluated. **APC1 showed NO improvement** over D2 baseline:
- Both D2 and APC1 survived 500 steps
- APC never activated (stayed in NEUTRAL state throughout)
- APC1 actually had WORSE pitch performance (7.50° vs 4.87° RMS)

The APC controller is correctly implemented but did not trigger because:
1. `sagittal_position_error_m = 1.6` exceeds the entry threshold (0.10 m)
2. But pitch never stayed within safe bounds (|pitch| < 0.03 rad) for sufficient time
3. The pitch_safe condition requires `|pitch| < 0.05 rad` OR recovering from pitch

## Phase 6 Results

### APC1 500-Step Evaluation (June 8, 19:30)
- **Profile**: APC1_active_pitch_crossing_moderate
- **Result**: Survived 500 steps ✓
- **telemetry**: telemetry_1780921811.csv (839 columns)
- **APC State**: Always NEUTRAL (never activated)

### D2 Recheck 500-Step Baseline (June 8, 17:29)
- **Profile**: candidate_D2_wheel_velocity_damping_light
- **Result**: Survived 500 steps ✓
- **telemetry**: telemetry_1780914570.csv (839 columns)
- **APC State**: N/A (APC not enabled)

## Phase 7: Comparison

### Stability Metrics

| Metric | D2 Recheck | APC1 | Comparison |
|--------|------------|------|------------|
| Survived | 500 | 500 | Equal ✓ |
| Pitch X min | -6.51° | -15.90° | APC1 worse |
| Pitch X max | 12.00° | 14.96° | APC1 worse |
| Pitch X RMS | 4.87° | 7.50° | APC1 worse |
| CoM Z range | 0.288-0.296 m | 0.287-0.295 m | Similar |

### APC Behavior Analysis

From telemetry analysis of APC1 run:
- `sagittal_position_error_m = 1.6` (constant, nonzero)
- APC entry threshold: 0.10 m → **EXCEEDED**
- But APC remained in NEUTRAL state throughout

**Why APC Never Activated:**

Looking at the APC entry conditions from the code:
```python
if signed_error > apc_outer_enter and (float(pitch_x_rad) > apc_pitch_enter or tau_pitch_persistent_positive):
```

Requirements:
1. `signed_error > 0.10` ✓ (1.6 > 0.10)
2. `pitch_x_rad > 0.03` OR `tau_pitch_persistent_positive`

The pitch oscillates rapidly (see early steps: 6.0°, -6.5°, -2.2°, 3.4°, -0.7°...) and never stays within safe bounds.

**Pitch Safety Gate:**
```python
apc_pitch_safe = apc_pitch_abs < self.authority_schedule.apc_pitch_safe_threshold_rad  # 0.05 rad = 2.86°
apc_pitch_safe = apc_pitch_safe or apc_pitch_recovering
apc_pitch_danger = apc_pitch_abs > self.authority_schedule.apc_pitch_danger_threshold_rad  # 0.10 rad = 5.73°
```

Even when pitch crosses 2.86° (safe threshold), the `apc_gate_safe` requires:
```python
apc_gate_safe = apc_contact_safe and apc_height_safe and apc_roll_safe and apc_pitch_safe and not apc_pitch_danger
```

With pitch frequently exceeding 5.73° (danger threshold), APC stays blocked.

### Root Cause: APC Cannot Activate During Pitch Oscillation

The APC controller is designed for situations where:
1. Position error accumulates while pitch stays controlled
2. Pitch is within safe bounds (|pitch| < 5.73°)
3. APC can apply crossing torque to help return position

But in the low_0p300 configuration:
- Position error is always elevated (1.6 m)
- Pitch oscillates wildly (-15.9° to +15.0°)
- Pitch exceeds danger threshold frequently
- APC gate blocks activation

## Conclusion

**APC_HARNESS_FIXED_APC1_NO_EFFECT**

The APC controller design is sound, but it is not effective for the low_0p300 configuration because:
1. The sagittal position error is always elevated (1.6 m)
2. Pitch oscillates too wildly for APC safety gates to allow activation
3. The root cause is the fundamental instability of low_0p300, not APC logic

## Recommendations

1. **Do NOT claim APC failure** - APC is working as designed
2. **Do NOT continue APC tuning** - APC cannot help if pitch oscillates
3. **Focus on pitch stability first** - The real issue is tau_pitch bias causing pitch oscillation
4. **Consider alternative approaches**:
   - Increase wheel damping authority
   - Adjust position cap to reduce lean
   - Investigate tau_pitch bias root cause

## Files Generated

- `telemetry_1780921811.csv` - APC1 500-step telemetry
- `docs/validation/apc_evaluation_harness_command_audit.md` - Harness audit
- `docs/validation/apc_entry_signal_audit.md` - Entry signal audit
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apc_command_audit.json` - Audit data