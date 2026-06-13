# Step E Extreme Height Support Fix Implementation Report

## Summary

Implemented opt-in sagittal authority profiles (E1, E2, E3) for fixing support_position_error and high-height wheel_velocity failures at extreme heights (low_0p300, high_0p480) without modifying the protected D2 baseline.

## Files Changed

### Modified Files

1. **`wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`**
   - Added `scheduled_k_wheel_velocity()` function for continuous high-height wheel damping
   - Added fields to `SagittalAuthoritySchedule`:
     - `continuous_k_wheel_velocity`, `k_wheel_velocity_nominal`, `k_wheel_velocity_high_max`, `k_wheel_velocity_z_low`, `k_wheel_velocity_z_high`
     - `enable_position_integral`, `ki_position_integral`, `integral_max_abs` (with 0.0 defaults)
     - Integral threshold fields: `integral_pitch_error_threshold_rad`, `integral_support_velocity_threshold_m_s`, etc.
   - Updated `max_position_tau_for_variant()` to return `max_position_tau_low_max` when continuous scheduling is enabled
   - Implemented continuous `k_wheel_velocity` scheduling in `compute()`
   - Updated wheel damping to use `effective_k_wheel_velocity`
   - Added telemetry fields for high-height wheel damping

2. **`scripts/simulate_hierarchical_controller.py`**
   - Added E1, E2, E3 profiles to `SAGITTAL_AUTHORITY_PROFILES`:
     - `E1_support_integral`: Position integral only
     - `E2_support_integral_higher_cap`: Integral + 5.0 Nm position cap
     - `E3_support_integral_cap_wheel_damping`: Integral + cap + high-height wheel damping
   - Added E1/E2/E3 to CLI choices for `--vd-sagittal-authority-profile`
   - Updated controller instantiation to extract integral parameters from profile when enabled

3. **`tests/test_sagittal_velocity_damped_balance_controller.py`**
   - Added 11 new tests for E1/E2/E3 profiles:
     - `test_extreme_height_profiles_have_integral_fields`
     - `test_default_schedule_has_no_integral`
     - `test_position_integral_anti_windup`
     - `test_position_integral_gate_deactivates_on_large_pitch`
     - `test_position_integral_reset_on_gate_failure`
     - `test_high_height_wheel_damping_schedule`
     - `test_high_height_wheel_damping_reduces_wheel_torque`
     - `test_continuous_k_wheel_velocity_not_active_at_nominal`
     - `test_e3_profile_telemetry_fields_exist`
     - `test_e1_e2_e3_profiles_are_extreme_variant_only`
     - `test_e2_position_cap_increases_at_boundary`

### New Files

1. **`docs/validation/step_e_extreme_height_support_fix_design.md`**
   - Design document with candidate definitions, formulas, and pass/fail gates

2. **`scripts/evaluate_step_e_extreme_support_fix_candidates.py`**
   - Staged evaluation script (smoke → validation → screening → official)
   - Supports regression testing on five standard variants

## Candidate Definitions

### Baseline: D2 (protected, unchanged)
```yaml
profile_name: candidate_D2_wheel_velocity_damping_light
applies_to_variants: D2_HEIGHT_VARIANTS  # includes low_0p300, high_0p480
position_tau_cap_by_variant: 4.0 Nm
velocity_damping_scale: 1.10
```

### E1: Support Integral (minimal risk)
```yaml
profile_name: E1_support_integral
applies_to_variants: BOUNDARY_HEIGHT_VARIANTS  # low_0p300, high_0p480 only
enable_position_integral: True
ki_position_integral: 2.0
integral_max_abs: 1.0 Nm (anti-windup)
max_position_tau: 4.0 Nm (unchanged)
velocity_damping_scale: 1.10
```

### E2: Support Integral + Higher Cap
```yaml
profile_name: E2_support_integral_higher_cap
applies_to_variants: BOUNDARY_HEIGHT_VARIANTS
enable_position_integral: True
ki_position_integral: 2.0
integral_max_abs: 1.0 Nm
continuous_max_position_tau: True
max_position_tau_nominal: 4.0 Nm
max_position_tau_low_max: 5.0 Nm (25% increase)
velocity_damping_scale: 1.10
```

### E3: Integral + Cap + High-Height Wheel Damping
```yaml
profile_name: E3_support_integral_cap_wheel_damping
applies_to_variants: BOUNDARY_HEIGHT_VARIANTS
enable_position_integral: True
ki_position_integral: 2.0
integral_max_abs: 1.0 Nm
continuous_max_position_tau: True
max_position_tau_nominal: 4.0 Nm
max_position_tau_low_max: 5.0 Nm
continuous_k_wheel_velocity: True
k_wheel_velocity_nominal: 0.5
k_wheel_velocity_high_max: 0.75 (50% increase at high heights)
k_wheel_velocity_z_low: 0.45 m
k_wheel_velocity_z_high: 0.52 m
```

## Key Design Decisions

1. **Integral Anti-Windup**: Position integral only accumulates when ALL gates pass:
   - Pitch error < 0.03 rad
   - Pitch rate < 0.05 rad/s
   - Support velocity < 0.03 m/s
   - Wheel velocity < 1.0 rad/s
   - Height in [0.28, 0.50] m
   - Contact valid
   - Roll error < 0.05 rad

2. **Continuous Scheduling**: k_wheel_velocity increases smoothly at HIGH heights (inverse of k_position scheduling):
   - z ≤ 0.45m: k_wheel_velocity = 0.5 (nominal)
   - z ≥ 0.52m: k_wheel_velocity = 0.75 (max)
   - Smooth interpolation in between

3. **No Variant Patches**: E1/E2/E3 use continuous formulas, not variant-name logic

4. **Profile Extraction**: When a profile has `enable_position_integral=True`, the script extracts integral parameters from the profile rather than using CLI defaults

## Tests Run

All 90 tests pass:
- 51 tests in `test_sagittal_velocity_damped_balance_controller.py`
- 4 tests in `test_step_e_wbc_gate_validator.py`
- 26 tests in `test_balance_core_height_variant_setup.py` + `test_balance_core_height_variant_setup_gates.py`
- 9 tests in `test_shape_posture_hip_yaw_sign.py`

## Smoke Test Results

E1 profile at low_0p300 (100 steps):
- `[EXTREME HEIGHT PROFILE] E1_support_integral: integral enabled (ki=2.0)` - confirmed
- Simulation completed without crash
- `[BALANCE-CORE] Sagittal controller: velocity-damped` - confirmed

## Usage

```bash
# Stage 1: 100-step smoke
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile E1_support_integral \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 100

# Staged evaluation
python scripts/evaluate_step_e_extreme_support_fix_candidates.py --stage 1
python scripts/evaluate_step_e_extreme_support_fix_candidates.py --stage 2
python scripts/evaluate_step_e_extreme_support_fix_candidates.py --stage 3
python scripts/evaluate_step_e_extreme_support_fix_candidates.py --stage 4
python scripts/evaluate_step_e_extreme_support_fix_candidates.py --stage regression
```

## Pass/Fail Gates

### Stage 1 (100-step smoke)
- survived: True
- contact_valid_percent_raw >= 99.9%
- non_wheel_floor_contacts = 0
- hidden_torque = 0
- ownership = 0
- WBC gate pass
- no roll collapse
- no large height collapse

### Stage 4 (5000-step official)
- support_position_error_max_abs < 0.15 m
- wheel_vel_mean_max_abs < 5.0 rad/s
- hip_yaw_abs_max < 0.10 rad
- All structural gates pass

## Restrictions Enforced

- ✅ Did NOT modify candidate_D2_wheel_velocity_damping_light
- ✅ Did NOT change default controller behavior
- ✅ Did NOT enable HY2-DIV by default
- ✅ Did NOT add WBC changes
- ✅ Did NOT relax Step E gates
- ✅ Did NOT enable legacy WBC

## Next Steps (Pending)

1. Run Stage 1 smoke tests for all candidates (D2, E1, E2, E3) at both heights
2. Run Stage 2 validation for candidates passing smoke
3. Run Stage 3 screening for candidates passing validation
4. Run Stage 4 official Step E for candidates passing screening
5. Run regression tests on five standard variants
6. If support/wheel pass but hip_yaw remains: classify hip-yaw as independent issue, write next plan

## Decision Points

- **EXTREME_SUPPORT_FIX_PASS**: All gates pass for both heights
- **EXTREME_SUPPORT_FIX_PARTIAL_HIP_YAW_REMAINS**: Support/wheel pass, hip_yaw fails
- **EXTREME_SUPPORT_FIX_PARTIAL_WHEEL_REMAINS**: Support passes, wheel or hip_yaw fails
- **EXTREME_SUPPORT_FIX_FAILED**: Support fails
- **EXTREME_SUPPORT_FIX_REGRESSES_BASELINE**: Standard variants regress
- **EXTREME_SUPPORT_FIX_REQUIRES_HIP_YAW_NEXT**: Support/wheel pass, hip_yaw needs separate fix