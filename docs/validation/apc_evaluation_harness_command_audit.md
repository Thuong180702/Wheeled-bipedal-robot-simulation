# APC Evaluation Harness Command Audit

## Classification: D2_BASELINE_REGRESSION

## Executive Summary

**D2 baseline REGRESSED** - the same command that previously survived 500 steps now fails at step 18. The APC1 result is NOT valid for comparison because the baseline itself is broken.

**Phase 1 conclusion**: D2 recheck PASSED 500 steps. D2 baseline (17:19) FAILED at step 18. APC1 (17:16) FAILED at step 18. Both failures share the same `sagittal_position_error_m = 0.0` signal.

## Phase 1 Results

### D2 Known-Good Recheck (June 8, 17:29) - PASSED ✓
- **Command used**: `python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --sagittal-controller velocity-damped --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json --steps 500 --telemetry-decimation 1 --failure-window-steps 500 --write-run-summary-sidecar`
- **Result**: Survived 500 steps ✓
- **telemetry**: telemetry_1780914570.csv (839 columns)
- **sagittal_position_error_m**: 1.6 (constant, nonzero) - allows APC activation

### D2 Baseline (June 8, 17:19) - FAILED ✗
- **Command**: Appears identical to recheck
- **Result**: Survived only 18 steps, terminated with `height_too_low`
- **telemetry**: telemetry_1780913944.csv (729 columns - 110 fewer columns!)
- **sagittal_position_error_m**: 0.0 (zero) - blocks APC activation

### APC1 (June 8, 17:16) - FAILED ✗
- **Result**: Survived only 18 steps, terminated with `height_too_low`
- **telemetry**: telemetry_1780913786.csv
- **sagittal_position_error_m**: 0.0 (zero) - blocks APC activation

## Phase 2: Command Comparison

### Command Analysis

Both D2 runs (baseline and recheck) used the same command:
```
--controller-mode balance-core
--sagittal-controller velocity-damped
--vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light
--height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json
--steps 500
```

The commands are IDENTICAL. The difference is in the execution environment or code state.

## Phase 1 Results

### D2 Known-Good Recheck (June 8, 17:29)
- **Command used**: `python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --sagittal-controller velocity-damped --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json --steps 500 --telemetry-decimation 1 --failure-window-steps 500 --write-run-summary-sidecar`
- **Result**: Survived 500 steps ✓ (telemetry_1780914570.csv)
- **Status**: PASSED

Wait - the recheck PASSED 500 steps! But the original D2 baseline (telemetry_1780913944.csv from 17:19) only survived 18 steps.

### Comparing D2 Runs

| Run | Timestamp | Steps | Result | sagittal_position_error_m |
|-----|-----------|-------|--------|---------------------------|
| D2 recheck (telemetry_1780914570) | 17:29 | 500 | Survived ✓ | 1.6 (constant, nonzero) |
| D2 baseline (telemetry_1780913944) | 17:19 | 18 | Failed height_too_low | 0.0000 (zero) |
| APC1 (telemetry_1780913786) | 17:16 | 18 | Failed height_too_low | 0.0000 (zero) |

## Key Findings

### Finding 1: Different sagittal_position_error_m Values

The recheck run has `sagittal_position_error_m = 1.6` throughout, while the baseline and APC1 runs have `sagittal_position_error_m = 0.0000`.

This is the SAME signal used for APC entry check. APC's signed_error is derived from `sagittal_position_error_m`.

### Finding 2: hip_yaw_comp_support_error_m Differs

| Run | hip_yaw_comp_support_error_m at step 1 | Pattern |
|-----|----------------------------------------|---------|
| D2 recheck | 1.11 (nonzero) | Oscillates then stabilizes |
| D2 baseline | 0.0 (zero) | All zeros |
| APC1 | 0.36 (nonzero) | Large values (up to 12+) |

### Finding 3: Same Height Setup Used

Both runs used the same height variant setup file:
- `outputs/physical_target_height_setups/low_0p300_setup.json`

### Finding 4: Both D2 Runs Used Same Command

The commands appear identical:
- `--controller-mode balance-core`
- `--sagittal-controller velocity-damped`
- `--vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light`
- `--height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json`

## Classification: D2_BASELINE_REGRESSION

The D2 baseline itself has regressed. The recheck at 17:29 survived 500 steps, but the earlier run at 17:19 only survived 18 steps with the same command.

## Root Cause Hypothesis

The difference in `sagittal_position_error_m` values (1.6 vs 0.0) suggests:

1. **yaw_aware_position_compensation_active**: The recheck has yaw-aware compensation ACTIVE, while the baseline has it INACTIVE (causing sagittal_position_error to stay at 0.0)

2. Or there was a code change between 17:16-17:19 and 17:29 that affected position tracking

3. Or there was a transient MuJoCo state issue

## Recommendations

1. **Do not claim APC failed or passed yet** - need to understand why D2 baseline had sagittal_position_error=0

2. **Compare D2 baseline vs recheck code**: Check what changed between the two runs

3. **Verify yaw_aware_position_compensation**: The recheck has `yaw_aware_position_compensation_active = True`, which should produce nonzero `sagittal_position_error_m`

4. **Check boundary_yaw_position_profile**: The default is "baseline", which should enable yaw-aware compensation for boundary variants

## Files Referenced

- `telemetry_1780914570.csv` - D2 recheck (PASSED 500 steps)
- `telemetry_1780913944.csv` - D2 baseline (FAILED at step 18)
- `telemetry_1780913786.csv` - APC1 (FAILED at step 18)
- `docs/validation/low_0p300_initial_condition_fix_final_report.md` - Previous init fix report
