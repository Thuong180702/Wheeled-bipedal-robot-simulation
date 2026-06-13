# APC1 500-Step Evaluation Report (Phase 7)

## Executive Summary

**APC1 failed to show improvement** over the D2 baseline at low_0p300 configuration. Both APC1 and D2 baseline simulations terminated at step 18 (0.2 seconds) with identical `height_too_low` termination. This indicates that the low_0p300 configuration has fundamental instability issues that neither D2 nor APC1 can overcome in the current simulation setup.

## Test Configuration

- **Profile:** `APC1_active_pitch_crossing_moderate`
- **Height variant:** `low_0p300` (target CoM Z = 0.300 m)
- **Simulation:** 500 steps requested (5.0 seconds)
- **Actual survival:** 18 steps (0.2 seconds)
- **Baseline for comparison:** `candidate_D2_wheel_velocity_damping_light`

## Key Findings

### 1. Identical Failure Mode
Both APC1 and D2 baseline show:
- Same CoM height trajectory (0.295 → 0.240 m)
- Same pitch evolution (-0.0° → -42.9° → -87.0°)
- Same contact loss timing (active_wheels=0 at step 10)
- Same termination reason: `height_too_low`

### 2. Sagittal Position Error = 0
**Critical observation:** `sagittal_position_error_m` is 0.0000 for all steps in both simulations. This means:
- APC's entry condition (signed_error > 0.10 m) is **never met**
- APC never enters CROSS_FROM_POSITIVE state
- APC torque contribution is 0 throughout the simulation

### 3. Root Cause: Position Error Not Tracked
The `sagittal_position_error_m` being zero suggests that either:
1. The position error tracking is not enabled/configured in this simulation mode
2. The signed_support_error computation path is not active
3. The low_0p300 configuration uses a different error tracking mechanism

## APC Telemetry Evidence

From telemetry analysis (telemetry_1780913786.csv):
- `sagittal_position_error_m`: 0.0000 throughout
- APC fields not visible in telemetry columns (may need explicit APC telemetry export)

## Comparison Table

| Metric | APC1 | D2 Baseline |
|--------|------|-------------|
| Steps survived | 18 | 18 |
| Simulation time | 0.2s | 0.2s |
| Final CoM height | 0.240 m | 0.240 m |
| Final pitch | -87.0° | -87.0° |
| Contact state | lost | lost |
| Termination | height_too_low | height_too_low |

## Diagnosis: Why APC Didn't Activate

### Entry Condition Analysis
APC requires:
1. `signed_error > outer_enter_m` (default: 0.10 m)
2. `pitch_x > pitch_enter_rad` (default: 0.03 rad)

At step 0-9:
- pitch_x evolves from -0.0° to -16.8°
- signed_error remains at 0.0000 m

At step 10+:
- pitch_x exceeds danger threshold (0.10 rad ≈ 5.7°)
- APC safety gates block activation due to pitch_danger

### Timing Issue
The signed_error never accumulates because:
1. The position tracking/feedback loop may not be active
2. The sagittal controller operates in a way that keeps position error at 0
3. Or the error tracking mechanism needs different configuration

## Recommendations

### For Low_0p300 Stability
The fundamental issue is **not** lack of APC - it's that the robot falls before APC can activate. The low_0p300 configuration needs:
1. Better initial posture stability
2. Higher wheel torque authority at startup
3. Different pitch recovery mechanism

### For APC to Work
APC requires a configuration where:
1. Position error can accumulate beyond 0.10 m
2. Pitch stays within safe bounds (|pitch| < 0.10 rad) for sufficient time
3. The sagittal controller allows drift before intervention

### Next Steps
1. **Evaluate APC at higher heights** (e.g., high_0p480) where stability allows error accumulation
2. **Check position error tracking** in simulate_hierarchical_controller.py to understand why signed_error = 0
3. **Consider F1/F1b/G1a/G1b** which have different intervention mechanisms
4. **Re-run D2/F1b/G1a/G1b** at 500 steps for comparison as planned in Phase 8

## Files Generated

- Telemetry: `outputs/hierarchical_controller_sim/telemetry_1780913786.csv` (APC1)
- Telemetry: `outputs/hierarchical_controller_sim/telemetry_1780913944.csv` (D2 baseline)
- Analysis: `analyze_apc_failure.py`
