# APC Entry Signal Audit

## Classification: APC_ENTRY_SIGNAL_OK

## Executive Summary

The APC controller correctly uses `sagittal_position_error_m` as its entry signal. The issue is NOT with the APC entry signal itself, but with the fact that the D2 baseline run produced `sagittal_position_error_m = 0.0` due to a D2 baseline regression.

**The APC is NOT broken - the baseline is broken.**

## APC Entry Signal Analysis

### APC Entry Condition

From `sagittal_velocity_damped_balance_controller.py` line 1178-1184:

```python
if apc_enabled and apc_gate_safe:
    # Enter CROSS_FROM_POSITIVE: signed_error > outer AND (pitch > threshold OR tau_pitch persistent positive)
    if signed_error > apc_outer_enter and (float(pitch_x_rad) > apc_pitch_enter or tau_pitch_persistent_positive):
        apc_state = "CROSS_FROM_POSITIVE"
    # Enter CROSS_FROM_NEGATIVE: signed_error < -outer AND (pitch < -threshold OR tau_pitch persistent negative)
    elif signed_error < -apc_outer_enter and (float(pitch_x_rad) < -apc_pitch_enter or tau_pitch_persistent_negative):
        apc_state = "CROSS_FROM_NEGATIVE"
```

Where `signed_error = float(sagittal_position_error_m)` (line 834).

### APC1 Configuration

- `apc_outer_enter_m = 0.10` (must exceed 0.10 m to enter)
- `apc_pitch_enter_rad = 0.03` (pitch must exceed 0.03 rad, or tau_pitch persistent)

### Signal Comparison

| Signal | D2 Recheck | D2 Baseline | APC1 |
|--------|------------|-------------|------|
| sagittal_position_error_m | 1.6 | 0.0 | 0.0 |
| Allows APC entry? | YES (1.6 > 0.10) | NO (0.0 < 0.10) | NO (0.0 < 0.10) |
| hip_yaw_comp_support_error_m | Oscillating | 0.0 | Large (up to 12+) |

### Why sagittal_position_error_m is Used

The APC uses `sagittal_position_error_m` because:

1. **It's the yaw-aware compensated error**: The controller receives `sagittal_position_error_m` which is already the yaw-aware compensated sagittal position error from `boundary_fix.apply_yaw_aware_position_compensation()`.

2. **Consistent with other recenter strategies**: F1, F2, and G1 all use the same `signed_error = float(sagittal_position_error_m)` pattern.

3. **Captures true drift**: By using yaw-aware compensated error, APC responds to actual drift rather than apparent drift from hip-yaw rotation.

### Why hip_yaw_comp_support_error_m is NOT Used

The `hip_yaw_comp_support_error_m` signal from shape_posture_controller is a different signal:

- **hip_yaw_comp_support_error_m**: Support position error from shape posture controller (raw, not yaw-compensated)
- **sagittal_position_error_m**: Sagittal position error from sagittal controller (yaw-compensated)

Using different signals for different controllers would create inconsistency.

## Root Cause: D2 Baseline Regression

The D2 baseline telemetry shows `sagittal_position_error_m = 0.0` throughout, but this is a **D2 baseline issue**, not an APC issue.

**Evidence**:
1. The same command with the same D2 profile now produces `sagittal_position_error_m = 1.6` (D2 recheck at 17:29)
2. The D2 baseline (17:19) had fewer telemetry columns (729 vs 839), suggesting a code or configuration difference
3. The D2 baseline failed at step 18 with `height_too_low` - same as APC1

## Conclusion

**APC_ENTRY_SIGNAL_OK**

The APC controller correctly uses `sagittal_position_error_m` as its entry signal. The APC never activated in the D2 baseline and APC1 runs because `sagittal_position_error_m` was 0.0, which is a D2 baseline regression issue.

The APC design is sound. The APC will work correctly when:
1. The D2 baseline is stable (producing nonzero `sagittal_position_error_m`)
2. The signed drift exceeds the APC entry threshold (0.10 m)
3. The pitch is within safe bounds

## Recommendations

1. **Do NOT change APC entry signal** - it's correctly designed
2. **Investigate D2 baseline regression** - understand why D2 baseline (17:19) produced `sagittal_position_error_m = 0.0`
3. **Rerun APC1 with verified D2 baseline** - once D2 baseline is stable, re-evaluate APC1

## Files Referenced

- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` - APC implementation
- `scripts/simulate_hierarchical_controller.py` - sagittal_position_error_m computation