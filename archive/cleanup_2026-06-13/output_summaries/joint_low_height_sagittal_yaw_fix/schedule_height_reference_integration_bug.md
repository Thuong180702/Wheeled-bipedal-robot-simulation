# Joint Profile Schedule Integration Bug

**Date:** 2026-06-05  
**Status:** BLOCKER - CANNOT PROCEED WITH PHASE 6  
**Classification:** SCHEDULE_HEIGHT_REFERENCE_BUG

## Executive Summary

J0-J3 profile evaluation is **blocked** because the sagittal controller schedule is receiving the wrong height reference (`commanded_height_ref_m=0.4m` instead of `achieved_com_z=0.293m`), causing all J1-J3 profiles to be inactive at low_0p300.

## Evidence

All profiles (J0, J1, J2, J3) show **identical metrics** at low_0p300:

```json
{
  "effective_k_position": 40.0,       // Should be 80 for J1-J3
  "effective_k_velocity": 15.0,       // Should be 25/30 for J2/J3
  "effective_max_position_tau": 0.0,  // Should be 6.0 for J1-J3
  "schedule_active": false,           // Should be true at z=0.293m
  "schedule_height_ref": 0.4,         // WRONG - should be 0.293m
  "schedule_smoothstep": 0.0,         // Should be ~1.0 at z=0.293m
  "com_z": 0.293                      // Actual CoM height
}
```

## Root Cause Analysis

The sagittal controller's `compute()` method receives `commanded_height_ref_m` parameter, which is being passed as the **nominal height command (0.4m)** instead of the **achieved target height from height_variant_setup (0.293m)**.

From [wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py:301-311]:

```python
# Determine scheduling height source
if commanded_height_ref_m is not None:
    schedule_height_ref = commanded_height_ref_m  # <-- Uses commanded, not achieved
    schedule_height_source = "target_reference"
else:
    # Fallback: use first-order filtered current com_z
    alpha_filter = 0.9
    self._filtered_com_z = alpha_filter * self._filtered_com_z + (1.0 - alpha_filter) * float(com_z_m)
    schedule_height_ref = self._filtered_com_z
    schedule_height_source = "filtered_current_fallback"
```

When `schedule_height_ref=0.4m` and `z_high=0.393m`:
- Smoothstep `u = (0.393 - 0.4) / (0.393 - 0.300) = -0.075` → clamped to 0
- Result: schedule **inactive** even though robot is at z=0.293m

## Impact

1. **J1-J3 profiles are NOT active** - all scheduled parameters remain at nominal values
2. **No meaningful comparison** - J0-J3 behave identically because schedules are off
3. **Phase 6 evaluation is invalid** - cannot assess if J1-J3 fix sagittal-yaw coupling

## Where the Bug Is

Need to trace where `commanded_height_ref_m` is passed to sagittal controller in [scripts/simulate_hierarchical_controller.py]. The simulation should pass:

- **For height variants**: `height_variant_setup["achieved_com_z_m"]` (0.293m for low_0p300)
- **For nominal**: `height_cmd` (0.4m)

Instead, it appears to always pass `height_cmd` (0.4m) regardless of variant.

## Expected Behavior

At low_0p300 (z=0.293m) with J1 profile:
- `schedule_height_ref` should be **0.293m** (achieved from setup)
- Smoothstep `u = (0.393 - 0.293) / (0.393 - 0.300) = 1.075` → clamped to 1.0
- `smoothstep = 1.0 * 1.0 * (3 - 2*1.0) = 1.0`
- `effective_k_position = 40 + (80 - 40) * 1.0 = 80.0`
- `effective_max_position_tau = 3.0 + (6.0 - 3.0) * 1.0 = 6.0`
- `schedule_active = true`

## Fix Required

Locate where sagittal controller is called in balance-core mode and ensure:

```python
# Determine height reference for schedule
if height_variant_setup is not None:
    # Use achieved height from variant setup
    height_ref_for_schedule = height_variant_setup["achieved_com_z_m"]
else:
    # Use nominal command height
    height_ref_for_schedule = height_cmd

# Pass to sagittal controller
tau, diag = sagittal_controller.compute(
    ...,
    commanded_height_ref_m=height_ref_for_schedule,  # <-- Fix here
)
```

## Verification After Fix

Rerun smoke tests and verify:

**J0 (baseline):**
- `effective_k_position = 40.0`
- `schedule_active = false` (baseline has no schedule)

**J1:**
- `effective_k_position ≈ 80.0` (at z=0.293m)
- `effective_max_position_tau ≈ 6.0`
- `effective_k_velocity = 15.0` (unchanged)
- `schedule_active = true`
- `schedule_smoothstep ≈ 1.0`

**J2:**
- `effective_k_position ≈ 80.0`
- `effective_max_position_tau ≈ 6.0`
- `effective_k_velocity ≈ 25.0` (scheduled)
- `schedule_active = true`

**J3:**
- `effective_k_position ≈ 80.0`
- `effective_max_position_tau ≈ 6.0`
- `effective_k_velocity ≈ 30.0` (scheduled)
- `schedule_active = true`

## Status

**BLOCKER** - Phase 6 candidate evaluation cannot proceed until this integration bug is fixed.

The schedule height reference must use `achieved_com_z_m` for height variants, not the nominal `height_cmd`.
