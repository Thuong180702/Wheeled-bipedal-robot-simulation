# APCR1nD Direct Support Recenter Features Design

**Date:** 2026-06-11
**Profile Name:** `APCR1nD_direct_support_recenter_features`
**Based On:** APCR1n (APCR1N_RECENTER_PRIORITY_TORQUE_BOOST)
**Classification:** `APCR1N_FEATURES_BLOCKED_BY_UNNECESSARY_APC_DEPENDENCY`

## Problem Statement

APCR1n features did not activate in 2000-step evaluation because:
1. Features depend on `_apc_drift_priority_active`
2. `_apc_drift_priority_active` is set only in proportional soft band branch
3. Proportional branch requires `apc_enabled=True`
4. APCR1n profile does NOT set `enable_active_pitch_crossing=True`

## Solution

Create APCR1nD with **direct support drift trigger** that:
- Does NOT require `enable_active_pitch_crossing=True`
- Does NOT enter the proportional soft band branch
- Directly checks drift conditions and safety gates
- Sets `_apcr1nd_direct_recenter_priority_active` flag independently

## Design

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
  
  # APCR1nD: Direct support drift trigger
  # Key difference: activates based on DIRECT drift conditions, NOT via APC
  recenter_priority_direct_enabled: True  # NEW
  recenter_priority_direct_enter_m: 0.08  # m - enter threshold
  recenter_priority_direct_emergency_m: 0.12  # m - emergency threshold
  recenter_priority_direct_hard_m: 0.15  # m - hard safety threshold
  recenter_priority_direct_exit_m: 0.02  # m - exit threshold (for hysteresis)
```

### Direct Trigger Logic

The direct recenter trigger is evaluated **BEFORE** the proportional soft band branch:

```python
# New block: APCR1nD Direct Support Drift Trigger
# Evaluated independently of APC state
apcr1nd_direct_recenter_priority_active = False
apcr1nd_direct_recenter_eligible = False
apcr1nd_direct_recenter_block_reason = "none"

if self.authority_schedule.recenter_priority_direct_enabled:
    # Track steps for startup guard
    if not hasattr(self, '_apcr1nd_step_counter'):
        self._apcr1nd_step_counter = 0
    current_step = self._apcr1nd_step_counter
    self._apcr1nd_step_counter += 1
    
    # Startup guard
    startup_guard_steps = self.authority_schedule.recenter_priority_startup_guard_steps
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
        abs_pitch = abs(float(pitch_x_rad))
        abs_roll = abs(float(roll_y_rad)) if roll_y_rad is not None else 0.0
        com_z_safe = float(com_z_m) >= self.authority_schedule.recenter_priority_safe_min_com_z
        roll_safe = abs_roll <= self.authority_schedule.recenter_priority_safe_roll_rad
        pitch_safe = abs_pitch <= self.authority_schedule.recenter_priority_safe_pitch_rad
        
        # Check eligibility
        enter_thresh = self.authority_schedule.recenter_priority_direct_enter_m
        emergency_thresh = self.authority_schedule.recenter_priority_direct_emergency_m
        hard_thresh = self.authority_schedule.recenter_priority_direct_hard_m
        exit_thresh = self.authority_schedule.recenter_priority_direct_exit_m
        
        if not contact_valid:
            apcr1nd_direct_recenter_block_reason = "contact_invalid"
        elif not com_z_safe:
            apcr1nd_direct_recenter_block_reason = "height_unsafe"
        elif not roll_safe:
            apcr1nd_direct_recenter_block_reason = "roll_unsafe"
        elif not pitch_safe:
            apcr1nd_direct_recenter_block_reason = "pitch_unsafe"
        elif abs_error > hard_thresh:
            apcr1nd_direct_recenter_block_reason = "hard_safety"
        elif abs_error < exit_thresh:
            apcr1nd_direct_recenter_block_reason = "within_exit_band"
            apcr1nd_direct_recenter_eligible = True
        elif abs_error > enter_thresh and moving_away:
            apcr1nd_direct_recenter_eligible = True
            apcr1nd_direct_recenter_priority_active = True
        elif abs_error > enter_thresh:
            # Error is large but moving toward zero - still eligible
            apcr1nd_direct_recenter_eligible = True
        else:
            apcr1nd_direct_recenter_block_reason = "below_enter_threshold"
```

### Feature Activation

With `apcr1nd_direct_recenter_priority_active` as the trigger instead of `_apc_drift_priority_active`:

```python
# Feature 1: Direct recenter priority (replaces apcr1n_recenter_priority_active)
apcr1nd_recenter_priority_active = apcr1nd_direct_recenter_priority_active

# Feature 2: Wheel damping override
if apcr1nd_direct_recenter_priority_active and apcr1nd_wheel_damping_fights_drift:
    # Apply wheel damping override
    ...

# Feature 3: Position cap boost
if apcr1nd_direct_recenter_priority_active and apcr1nd_safety_gate_pass:
    # Apply position cap boost
    ...
```

### Telemetry Fields

New APCR1nD telemetry columns:

```python
"apcr1nd_direct_recenter_priority_active": bool,
"apcr1nd_direct_recenter_eligible": bool,
"apcr1nd_direct_recenter_block_reason": str,
"apcr1nd_moving_away": bool,
"apcr1nd_abs_error": float,
"apcr1nd_error_rate": float,
"apcr1nd_wheel_damping_override_active": bool,
"apcr1nd_position_cap_boost_active": bool,
"apcr1nd_position_cap_current": float,
"apcr1nd_final_torque_direction_correct": bool,
"apcr1nd_final_torque_fights_drift": bool,
```

## Safety Gates

Same as APCR1n:
- `contact_valid`: Required
- `com_z >= 0.27`: Required
- `roll <= 0.15 rad`: Required
- `pitch <= 0.15 rad`: Required
- `abs_error <= 0.15 m`: Hard safety threshold

## Exit Conditions

For hysteresis behavior:
- Exit when `abs_error <= exit_thresh` (0.02 m)
- Hysteresis prevents rapid toggle near threshold

## Difference from APCR1n

| Aspect | APCR1n | APCR1nD |
|--------|--------|---------|
| Trigger | `_apc_drift_priority_active` | Direct drift conditions |
| Requires APC | Yes (but disabled) | No |
| Activation check | Line 1644 | New block before line 1630 |
| Telemetry | `apcr1n_*` | `apcr1nd_*` |

## Implementation Plan

1. Add `APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES` profile
2. Add `recenter_priority_direct_enabled` and related fields to `SagittalAuthoritySchedule`
3. Add direct trigger logic block BEFORE existing APCR1n block
4. Add APCR1nD-specific telemetry
5. Update feature activation to use direct trigger
6. Add tests

## Testing

1. Verify direct trigger activates when `abs(e) > 0.08` and `moving_away`
2. Verify direct trigger does NOT activate during startup guard
3. Verify direct trigger blocked by contact/height/roll/pitch unsafe
4. Verify wheel damping override activates when eligible
5. Verify position cap boost activates when eligible
6. Verify 2000-step run completes without crash
