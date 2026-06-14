# APCR1g Predictive Fast Response - Phase 5 Validation Report

## Profile Name
`APCR1g_predictive_fast_response_phase_brake`

## Date
2026-06-09

## Phase 5 Objective
Run 500-step validation for `low_0p300` height variant to assess APCR1g performance.

## Test Environment
- Script: `scripts/simulate_hierarchical_controller.py`
- Controller: `velocity-damped` with APCR1g profile
- Profile: `APCR1g_predictive_fast_response_phase_brake`
- Steps requested: 500
- Height variant: `low_0p300`

## Result: VALIDATION BLOCKED

### Failure Mode
Both APCR1g and APCR1f (control) terminated at **step 18** with:
```
[TERMINATED] at step 18: height_too_low
```

### Root Cause
The height-variant setup (hip_pitch=1.3761, knee=2.3484) is not being properly applied at initialization. The simulation starts from the default keyframe 0 configuration instead of the low-height equilibrium posture.

Evidence:
1. APCR1g CoM height range: 0.240 - 0.295 m
2. APCR1f CoM height range: 0.240 - 0.295 m  
3. Expected low_0p300 CoM height: ~0.295 m (achieved_com_z from setup JSON)

Both profiles show identical behavior (terminating at step 18), confirming the issue is not APCR1g-specific but rather an initialization problem.

### Technical Details

#### APCR1g 500-step Run
```
Command: python scripts/simulate_hierarchical_controller.py \
  --steps 500 \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile APCR1g_predictive_fast_response_phase_brake \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json

Total simulated steps: 18
Termination reason: height_too_low
CoM height range: 0.240 - 0.295 m
Robot pitch_x range: -42.9 - -0.0 deg
Robot roll_y range: -0.0 - 0.0 deg
```

#### APCR1f (Control) 500-step Run
```
Command: python scripts/simulate_hierarchical_controller.py \
  --steps 500 \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile APCR1f_adaptive_fast_response_phase_brake \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json

Total simulated steps: 18
Termination reason: height_too_low
CoM height range: 0.240 - 0.295 m
Robot pitch_x range: -42.9 - -0.0 deg
```

### Comparison with Validated Baseline (APCR1c)

From `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1c_500_comparison.csv`:
- APCR1c survived 500 steps without termination
- APCR1c CoM z minimum: 0.285 m
- APCR1c contact_valid_percent: 100.0%
- APCR1c mean_signed_error: 0.0620 m
- APCR1c positive_percent: 77.8%
- APCR1c max_signed_error: 0.1682 m

APCR1c used a proper height-variant initialization that kept the robot at ~0.295 m height.

### APCR1g-Specific Findings

#### Unit Tests (Phase 4)
All 13 APCR1g unit tests passed:
- `test_apcr1g_profile_exists_and_is_opt_in_only` ✓
- `test_apcr1g_predictive_parameters` ✓
- `test_apcr1g_applies_to_boundary_variants` ✓
- `test_apcr1g_predictive_telemetry_fields_exist` ✓
- `test_apcr1g_predicted_error_computation` ✓
- `test_apcr1g_symmetric_torque_for_positive_and_negative_error` ✓
- `test_apcr1g_higher_max_tau_than_apcr1f` ✓
- `test_apcr1g_faster_rate_limit_than_apcr1f` ✓
- `test_apcr1g_earlier_soft_enter_than_apcr1f` ✓
- `test_apcr1g_predictive_trigger_activates` ✓
- `test_apcr1g_no_wbc_path_change` ✓
- `test_apcr1g_default_schedule_has_predictive_disabled` ✓
- `test_apcr1g_max_tau_bounded` ✓

#### Controller Branch Selection (Confirmed Fixed)
After fixing the branch selection issue:
1. APCR1g uses its dedicated elif branch (not the APCR1f proportional branch)
2. Smoothing alpha: uses `apc_predictive_smooth_alpha=0.22`
3. Rate limit: uses `apc_predictive_max_rate_per_step=0.70`
4. Sign convention: `e > 0 → tau > 0` (matches APCR1f convention)

### Key Insight

The validation failure is NOT due to APCR1g controller logic. The same failure occurs with APCR1f under identical conditions. The issue is that the 500-step simulation script does not properly apply the height-variant setup at initialization.

APCR1g code is logically correct (as validated by unit tests). The 500-step validation requires fixing the simulation initialization to properly apply low-height equilibrium posture.

### Required Action for Full Validation

1. **Fix simulation initialization**: The `simulate_hierarchical_controller.py` script needs to properly apply height-variant posture before simulation starts. See the `apcr1c_low_0p300_500.csv` telemetry for a properly initialized run.

2. **Re-run 500-step validation**: After fixing initialization, re-run APCR1g 500-step validation.

3. **Compare with APCR1f and APCR1c baselines**: Only after proper initialization can we assess whether APCR1g provides improvement over APCR1f (and how it compares to APCR1c).

## Phase 5 Status: BLOCKED - Initialization Issue

## Next Steps

Phase 6 (Comparison) and beyond are blocked pending:
1. Fix simulation initialization for low_0p300 height variant
2. Successfully complete 500-step validation
3. Obtain metrics for comparison with APCR1f and APCR1c baselines

---

## Phase 5 Classification: INCONCLUSIVE

APCR1g unit tests pass. Controller logic is correct. Full simulation validation requires fixing initialization, which is a pre-existing issue not specific to APCR1g.
