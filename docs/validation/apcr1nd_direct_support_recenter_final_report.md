# APCR1nD Direct Support Drift Trigger - Final Report

**Date:** 2026-06-11
**Profile:** `APCR1nD_direct_support_recenter_features`
**Classification:** `APCR1N_FEATURES_BLOCKED_BY_UNNECESSARY_APC_DEPENDENCY`

## Executive Summary

APCR1nD was created to fix the issue where APCR1n features never activated because they depended on `_apc_drift_priority_active`, which requires `enable_active_pitch_crossing=True`. The APCR1n profile does NOT enable APC, so the features were permanently blocked.

APCR1nD introduces a **direct support drift trigger** that activates features based on direct drift conditions, without requiring APC.

## Problem Statement

APCR1n features did not activate in 2000-step evaluation because:
1. Features depend on `_apc_drift_priority_active`
2. `_apc_drift_priority_active` is set only within the proportional soft band branch
3. Proportional branch requires `apc_enabled=True`
4. APCR1n profile does NOT set `enable_active_pitch_crossing=True`

Evidence from APCR1n 5000-step run:
- `apcr1n_recenter_priority_active_count: 0`
- `apcr1n_position_cap_boost_active_count: 0`
- Drift stayed bounded (ratio 1.10 < 1.5 threshold)
- No features needed to activate because drift stayed below hard safety threshold

## Solution

APCR1nD uses a **direct support drift trigger** that:
- Does NOT require `enable_active_pitch_crossing=True`
- Does NOT enter the proportional soft band branch
- Directly checks drift conditions and safety gates
- Sets `apcr1nd_direct_recenter_priority_active` flag independently

## Key Differences

| Aspect | APCR1n | APCR1nD |
|--------|--------|---------|
| Trigger | `_apc_drift_priority_active` | Direct drift conditions |
| Requires APC | Yes (but disabled) | No |
| Feature activation check | Line 1644 | New block before line 1630 |
| Telemetry prefix | `apcr1n_*` | `apcr1nd_*` |

## Implementation

### Profile Configuration

```yaml
APCR1nD_direct_support_recenter_features:
  profile_name: "APCR1nD_direct_support_recenter_features"
  applies_to_variants: ("low_0p300", "low_0p330", "low_0p360", "extreme_height")
  
  # APCR1h base (copied)
  continuous_max_position_tau: True
  max_position_tau_nominal: 4.0
  velocity_damping_scale: 1.10
  
  # APCR1n recenter priority features (base)
  recenter_priority_enabled: True
  recenter_priority_startup_guard_steps: 100
  vd_wheel_damping_recenter_override_enabled: True
  vd_wheel_damping_recenter_scale: 0.30
  vd_wheel_damping_recenter_min_abs_nm: 0.50
  vd_wheel_damping_preserve_if_opposes_drift: True
  position_cap_recenter_boost_enabled: True
  position_cap_normal_nm: 4.0
  position_cap_recenter_nm: 5.0
  position_cap_emergency_nm: 6.0
  position_cap_ramp_steps: 50
  recenter_priority_safe_min_com_z: 0.27
  recenter_priority_safe_roll_rad: 0.15
  recenter_priority_safe_pitch_rad: 0.15
  
  # APCR1nD: Direct support drift trigger (KEY DIFFERENCE)
  recenter_priority_direct_enabled: True
  recenter_priority_direct_enter_m: 0.08
  recenter_priority_direct_emergency_m: 0.12
  recenter_priority_direct_hard_m: 0.15
  recenter_priority_direct_exit_m: 0.02
```

### Direct Trigger Logic

The direct recenter trigger is evaluated BEFORE the proportional soft band branch:

```python
if self.authority_schedule.recenter_priority_direct_enabled:
    # Startup guard (100 steps)
    if current_step < startup_guard_steps:
        apcr1nd_direct_recenter_block_reason = "startup_guard"
    else:
        # Compute drift conditions
        signed_error = float(sagittal_position_error_m)
        abs_error = abs(signed_error)
        e_dot = signed_error - self._apcr1nd_prev_error
        self._apcr1nd_prev_error = signed_error
        moving_away = signed_error * e_dot > 0.0
        
        # Safety gates
        com_z_safe = com_z_m >= 0.27
        roll_safe = abs_roll <= 0.15
        pitch_safe = abs_pitch <= 0.15
        
        # Threshold checks
        if not contact_valid:
            block_reason = "contact_invalid"
        elif not com_z_safe:
            block_reason = "height_unsafe"
        elif abs_error < exit_thresh:  # 0.02 m
            eligible = True
        elif abs_error > enter_thresh and moving_away:  # 0.08 m
            active = True
        else:
            eligible = True if abs_error > enter_thresh else False
```

## Test Results

All 15 APCR1nD tests pass:

```
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_profile_exists_and_is_opt_in_only PASSED
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_based_on_apcr1n PASSED
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_direct_trigger_parameters PASSED
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_applies_to_boundary_variants PASSED
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_startup_guard_blocks_activation PASSED
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_direct_trigger_activates_on_eligible_drift PASSED
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_direct_trigger_inactive_when_converging PASSED
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_direct_trigger_blocked_by_contact PASSED
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_direct_trigger_blocked_by_height PASSED
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_direct_trigger_blocked_by_roll PASSED
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_direct_trigger_blocked_by_pitch PASSED
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_telemetry_fields_exist PASSED
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_no_wbc_path_change PASSED
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_profile_in_registry PASSED
tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1nd_does_not_require_apc PASSED

15 passed, 270 deselected
```

## Files Modified

1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
   - Added `APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES` profile
   - Added `recenter_priority_direct_enabled` and related fields to `SagittalAuthoritySchedule`
   - Added direct trigger logic block BEFORE existing APCR1n block
   - Added APCR1nD-specific telemetry
   - Updated feature activation to use direct trigger

2. `tests/test_sagittal_velocity_damped_balance_controller.py`
   - Added 15 APCR1nD-specific tests

3. `scripts/simulate_hierarchical_controller.py`
   - Added APCR1nD to `SAGITTAL_AUTHORITY_PROFILES`
   - Added APCR1nD to `--vd-sagittal-authority-profile` choices

## Decision

**Recommendation:** Keep APCR1nD as an opt-in profile but do NOT make it default or merge into APCR1n.

**Rationale:**
1. APCR1n already survived 5000 steps without features activating (drift stayed bounded)
2. APCR1nD's features would only help if drift exceeds 0.08m while moving away
3. Adding unnecessary torque boosts could cause instability
4. APCR1nD is available for evaluation if needed

**When to use APCR1nD:**
- If APCR1n fails on a specific variant and drift exceeds 0.08m
- If wheel damping override is needed to fight drift recovery
- If position cap boost is needed for safe RECENTER

**Do NOT:**
- Claim Step E pass with APCR1nD
- Make APCR1nD the default profile
- Enable APCR1nD for all variants
- Commit APCR1nD to mainline (per restrictions)

## Compliance with Restrictions

- ✅ Do NOT modify D2 baseline
- ✅ Do NOT modify APCR1h or existing APCR1n behavior
- ✅ Do NOT make APCR1nD default
- ✅ Do NOT enable HY2-DIV
- ✅ Do NOT add WBC
- ✅ Do NOT enable legacy WBC
- ✅ Do NOT relax official Step E gates
- ✅ Do NOT claim Step E pass
- ✅ Do NOT run high_0p480
- ✅ Do NOT run Step C
- ✅ Do NOT run Step D
- ✅ Do NOT run 5000-step
- ✅ Do NOT commit