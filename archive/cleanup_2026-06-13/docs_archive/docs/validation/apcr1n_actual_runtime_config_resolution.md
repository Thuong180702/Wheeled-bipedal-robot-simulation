# APCR1n Actual Runtime Config Resolution

**Date:** 2026-06-11  
**Profile:** APCR1n_recenter_priority_torque_boost  
**Classification:** APCR1N_RUNTIME_CONFIG_FEATURE_CODE_NOT_PRESENT

## Executive Summary

The APCR1n successful runs (1000-step, 2000-step, 5000-step) all used **controller code WITHOUT the APCR1n feature implementation**. The config values exist in `simulate_hierarchical_controller.py`, but the runtime logic in `sagittal_velocity_damped_balance_controller.py` was added AFTER the runs were executed.

## Evidence

### 1. Telemetry Column Analysis

**Expected APCR1n columns (from current code):**
- `apcr1n_recenter_priority_active`
- `apcr1n_startup_guard_active`
- `apcr1n_wheel_damping_override_active`
- `apcr1n_wheel_damping_scale`
- `apcr1n_wheel_damping_before`
- `apcr1n_wheel_damping_after`
- `apcr1n_wheel_damping_fights_drift`
- `apcr1n_position_cap_boost_active`
- `apcr1n_position_cap_current`
- `apcr1n_tau_position_raw`
- `apcr1n_tau_position_after_cap`
- `apcr1n_position_saturated`
- `apcr1n_safety_gate_pass`
- `apcr1n_final_torque_direction_correct`
- `apcr1n_final_torque_fights_drift`
- `apcr1n_physical_drift_column_used`

**Actual columns in telemetry.csv:**
- **NONE** of the above columns exist

### 2. Runtime Config Values

**Actual runtime values from telemetry.csv (step 100):**
- `effective_max_position_tau` = 3.0 (default, not 4.0 from APCR1n config)
- `effective_velocity_damping_scale` = 1.0 (default, not 1.10 from APCR1n config)

**Expected values from APCR1n config (lines 1210-1212):**
- `continuous_max_position_tau` = True
- `max_position_tau_nominal` = 4.0
- `velocity_damping_scale` = 1.10

### 3. Git Status

**APCR1n feature code location:**
```
git diff wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py
```

Shows APCR1n code (lines 1601-1730, 3177-3192) is in **uncommitted changes** (marked with `+`).

**APCR1n run timestamp:**
```
outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1n_low_0p300_1000/telemetry.csv
Modified: Jun 11 09:41
```

The feature implementation was added AFTER the run.

### 4. Feature Activation Dependency

**Current APCR1n activation logic (line 1639):**
```python
apcr1n_recenter_priority_active = self._apc_drift_priority_active
```

This depends on `_apc_drift_priority_active`, which is set only when:
```python
if abs_error > 0.08 and moving_away_drift:
    self._apc_drift_priority_active = True
```

**Telemetry analysis of APCR1n 1000-step run:**
- Steps with |e| > 0.08: 626 / 1000 (62.6%)
- Steps with moving_away: 485 / 1000 (48.5%)
- Steps eligible (|e| > 0.08 AND moving_away): 280 / 1000 (28.0%)

If the feature code had been present, drift_priority should have activated for 280 steps.

## Timeline Reconstruction

1. **APCR1n config added** to `simulate_hierarchical_controller.py` (lines 1204-1269) - **committed or before June 11 09:00**
2. **APCR1n 1000-step run executed** - June 11 09:41 - used OLD controller code
3. **APCR1n 2000-step run executed** - June 11 18:03 - used OLD controller code
4. **APCR1n 5000-step run executed** - June 11 18:19 - used OLD controller code
5. **APCR1n feature implementation added** to `sagittal_velocity_damped_balance_controller.py` - **after June 11 18:19, uncommitted**

## Config Inconsistency Resolution

The documentation inconsistency from prior reports is now explained:

**Report claimed APCR1n uses:**
- `continuous_max_position_tau` = True
- `max_position_tau_nominal` = 4.0
- `velocity_damping_scale` = 1.10
- `position_cap_normal_nm` = 4.0

**Actual runtime used:**
- `effective_max_position_tau` = 3.0 (default from base class)
- `effective_velocity_damping_scale` = 1.0 (default from base class)
- Position cap = 3.0 (default)

**Why the mismatch:**
The config values exist in `SagittalAuthoritySchedule`, but the controller code that READS those values was not present at runtime.

## What Actually Ran

The APCR1n runs used:
- **APCR1h-equivalent soft-band proportional mode** (no hysteresis, no drift_priority telemetry)
- **Default D2 baseline parameters:** max_position_tau=3.0, velocity_damping_scale=1.0
- **NO recenter priority features:** no wheel damping override, no position cap boost
- **Standard APCR soft-band behavior with drift_priority logic** (if it existed in that version)

The profile name "APCR1n_recenter_priority_torque_boost" was passed via `--vd-sagittal-authority-profile`, but only the base APCR1h-style parameters were used because the new feature code didn't exist.

## Implications

1. **APCR1n 5000-step success is valid** but it succeeded WITHOUT the new features
2. **Feature activation audit is now unnecessary** - features never existed at runtime
3. **Ablation study must be redesigned** - need to test:
   - Current APCR1n baseline (no features, runs as APCR1h-lite)
   - APCR1n with new feature code (once features are actually wired)
4. **Config parameters (lines 1210-1212, 1262) need runtime wiring**

## Classification

**APCR1N_RUNTIME_CONFIG_FEATURE_CODE_NOT_PRESENT**

The successful APCR1n runs used an older controller version that lacked the APCR1n feature implementation entirely. The config exists but was never consumed.

## Recommendations

**Do NOT re-run APCR1n 5000 until feature wiring is verified.**

Instead:
1. Commit current APCR1n feature code
2. Add tests for APCR1n feature activation
3. Run smoke test (100 steps) with telemetry validation
4. Verify APCR1n telemetry columns exist
5. Verify continuous_max_position_tau, max_position_tau_nominal, velocity_damping_scale are consumed
6. Only then proceed to ablation study

## Next Phase

**Skip Phase 2 (activation audit)** - features were never present.

**Proceed directly to Phase 1b:** Verify current APCR1n feature code runtime behavior before designing ablation.
