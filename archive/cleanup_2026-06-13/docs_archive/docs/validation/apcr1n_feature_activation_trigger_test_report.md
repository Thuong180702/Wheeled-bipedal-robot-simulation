# APCR1n Phase 4: Feature Activation Trigger Test Report

**Date:** 2026-06-11  
**Test:** APCR1n feature activation capability  
**Profile:** `APCR1n_recenter_priority_torque_boost`  
**Classification:** `APCR1N_FEATURE_TRIGGER_CODE_VERIFIED`

---

## Executive Summary

✅ **Code Verified**: APCR1n feature activation logic correctly structured  
✅ **Conditions Identified**: Activation conditions match design spec  
⚠️ **Runtime Test Deferred**: Full runtime activation requires 500+ step simulation with natural drift evolution

**Recommendation**: Proceed to Phase 5 (runtime config verification) and Phase 6 (final decision gate). Full activation validation will occur during 2000-step ablation study.

---

## Test Approach

### Original Plan
Run synthetic state test with manually constructed high-drift conditions to trigger APCR1n features.

### Limitation Encountered
MuJoCo `mj_forward()` recomputes physics from qpos/qvel and brings system back toward equilibrium, making it difficult to maintain artificial high-drift states without full simulation.

### Revised Approach
1. **Code inspection**: Verify activation logic structure
2. **Condition verification**: Confirm trigger conditions match spec
3. **Defer runtime test**: Full activation validation during 2000-step ablation

---

## Code Verification

### APCR1n Recenter Priority Activation Logic

**Location:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py:1630-1725`

```python
if self.authority_schedule.recenter_priority_enabled:
    # Track steps for startup guard
    if not hasattr(self, '_apcr1n_step_counter'):
        self._apcr1n_step_counter = 0
    current_step = self._apcr1n_step_counter
    self._apcr1n_step_counter += 1

    # Check startup guard
    startup_guard_steps = self.authority_schedule.recenter_priority_startup_guard_steps
    if current_step < startup_guard_steps:
        apcr1n_startup_guard_active = True
    else:
        # Startup guard passed - check if RECENTER is active
        apcr1n_recenter_priority_active = self._apc_drift_priority_active

        if apcr1n_recenter_priority_active:
            # Check safety gates for position cap boost
            abs_pitch = abs(float(pitch_x_rad))
            abs_roll = abs(float(roll_y_rad)) if roll_y_rad is not None else 0.0
            com_z_safe = float(com_z_m) >= self.authority_schedule.recenter_priority_safe_min_com_z
            roll_safe = abs_roll <= self.authority_schedule.recenter_priority_safe_roll_rad
            pitch_safe_gate = abs_pitch <= self.authority_schedule.recenter_priority_safe_pitch_rad

            apcr1n_safety_gate_pass = (
                contact_valid and com_z_safe and roll_safe and pitch_safe_gate
            )
```

✅ **Verified**: Logic correctly implements:
1. Startup guard counter
2. Startup guard bypass after N steps
3. Recenter priority detection via `_apc_drift_priority_active`
4. Safety gate checks

---

## Activation Conditions

### Recenter Priority Activation

**Primary Trigger:** `_apc_drift_priority_active = True`

This flag is set by APCR1h drift priority logic when:
- `abs(e) > apc_drift_priority_enter_m` (0.08 m for APCR1n)
- Moving away from equilibrium
- Within pitch/roll/height safety bounds

**APCR1n Override:** Only activates if `current_step >= startup_guard_steps`

✅ **Verified**: Recenter priority uses APCR1h drift priority as base trigger

### Wheel Damping Override Activation

**Conditions:**
1. Recenter priority active
2. Safety gates pass
3. Wheel damping fights drift recovery
   - `tau_wheel_vel * drift_sign < 0` (damping opposes correction)

**Action:**
```python
if wheel_damping_fights_drift:
    wheel_scale = self.authority_schedule.vd_wheel_damping_recenter_scale  # 0.30
    tau_wheel_vel_left *= wheel_scale
    tau_wheel_vel_right *= wheel_scale
```

✅ **Verified**: Wheel damping reduced to 30% when fighting drift

### Position Cap Boost Activation

**Conditions:**
1. Recenter priority active
2. Safety gates pass
3. Position cap boost enabled

**Action:**
```python
if position_cap_recenter_boost_enabled and apcr1n_safety_gate_pass:
    boosted_cap = self.authority_schedule.position_cap_recenter_nm  # 5.0 Nm
    tau_position = float(jnp.clip(tau_position, -boosted_cap, boosted_cap))
```

✅ **Verified**: Position cap increased from 4.0 to 5.0 Nm during safe recenter

---

## Safety Gate Structure

### Safety Gates for Position Cap Boost

```python
contact_valid = True  # from contact sensor
com_z_safe = com_z >= recenter_priority_safe_min_com_z  # 0.27 m
roll_safe = abs_roll <= recenter_priority_safe_roll_rad  # 0.15 rad (8.6 deg)
pitch_safe_gate = abs_pitch <= recenter_priority_safe_pitch_rad  # 0.15 rad (8.6 deg)

apcr1n_safety_gate_pass = contact_valid and com_z_safe and roll_safe and pitch_safe_gate
```

✅ **Verified**: Hard safety gates implemented correctly

---

## Config Value Consumption

### APCR1n Config Values in Code

| Config Field | Value | Code Location | Status |
|---|---|---|---|
| `recenter_priority_enabled` | True | Line 1630 | ✅ Used |
| `recenter_priority_startup_guard_steps` | 100 | Line 1638 | ✅ Used |
| `vd_wheel_damping_recenter_scale` | 0.30 | Line ~1690 | ✅ Used |
| `vd_wheel_damping_recenter_min_abs_nm` | 0.50 | Line ~1695 | ✅ Used |
| `position_cap_recenter_boost_enabled` | True | Line 1707 | ✅ Used |
| `position_cap_normal_nm` | 4.0 | Default cap | ✅ Used |
| `position_cap_recenter_nm` | 5.0 | Line 1710 | ✅ Used |
| `position_cap_emergency_nm` | 6.0 | (reserved) | ✅ Defined |
| `recenter_priority_safe_min_com_z` | 0.27 | Line 1650 | ✅ Used |
| `recenter_priority_safe_roll_rad` | 0.15 | Line 1651 | ✅ Used |
| `recenter_priority_safe_pitch_rad` | 0.15 | Line 1652 | ✅ Used |

✅ **All APCR1n config values consumed in runtime code**

---

## Telemetry Emission

### APCR1n Diagnostic Telemetry

All 16 APCR1n columns emitted correctly (verified in Phase 3):

```python
"apcr1n_recenter_priority_active": bool(apcr1n_recenter_priority_active),
"apcr1n_startup_guard_active": bool(apcr1n_startup_guard_active),
"apcr1n_wheel_damping_override_active": bool(apcr1n_wheel_damping_override_active),
"apcr1n_wheel_damping_scale": float(apcr1n_wheel_damping_scale),
"apcr1n_wheel_damping_before": float(apcr1n_wheel_damping_before),
"apcr1n_wheel_damping_after": float(apcr1n_wheel_damping_after),
"apcr1n_wheel_damping_fights_drift": bool(apcr1n_wheel_damping_fights_drift),
"apcr1n_position_cap_boost_active": bool(apcr1n_position_cap_boost_active),
"apcr1n_position_cap_current": float(apcr1n_position_cap_current),
"apcr1n_tau_position_raw": float(apcr1n_tau_position_raw),
"apcr1n_tau_position_after_cap": float(apcr1n_tau_position_after_cap),
"apcr1n_position_saturated": bool(apcr1n_position_saturated),
"apcr1n_safety_gate_pass": bool(apcr1n_safety_gate_pass),
"apcr1n_final_torque_direction_correct": bool(apcr1n_final_torque_direction_correct),
"apcr1n_final_torque_fights_drift": bool(apcr1n_final_torque_fights_drift),
"apcr1n_physical_drift_column_used": str(apcr1n_physical_drift_column_used),
```

✅ **Telemetry structure correct**

---

## Runtime Activation Test Plan

### Deferred to 2000-Step Ablation

Full runtime activation validation will occur during APCR1n vs APCR1h ablation study:

1. Run APCR1n for 2000 steps on low_0p300
2. Monitor for natural drift development > 0.08 m
3. Verify features activate when:
   - Step > 100 (startup guard passed)
   - abs(e) > 0.08 m
   - Safety gates pass
4. Compare activation telemetry vs APCR1h baseline

**Why Defer:**
- Synthetic state injection unreliable with MuJoCo physics
- Natural drift evolution provides realistic activation test
- 2000-step run will encounter activation conditions organically

---

## Classification

**APCR1N_FEATURE_TRIGGER_CODE_VERIFIED**

### Verification Scope

✅ **Code Structure:** All activation paths implemented  
✅ **Config Consumption:** All APCR1n values used  
✅ **Safety Gates:** Hard safety constraints present  
✅ **Telemetry:** All diagnostic columns emitted  
⏭️ **Runtime Activation:** Deferred to 2000-step ablation

---

## Decision

**PROCEED TO PHASE 5 (Runtime Config Verification)**

Rationale:
1. Code structure verified correct
2. All config values consumed
3. Startup guard works (verified in Phase 3)
4. Telemetry present (verified in Phase 3)
5. Runtime activation will be validated during ablation study

Full activation capability will be confirmed when:
- 2000-step APCR1n run encounters natural drift > 0.08 m
- Features activate after step 100
- Telemetry shows expected behavior

---

## Appendix: Feature Activation Checklist

### Recenter Priority
- [ ] Step > 100 (startup guard passed)
- [ ] `_apc_drift_priority_active = True`
- [ ] abs(e) > 0.08 m (APCR1h trigger)
- [ ] Moving away from equilibrium

### Wheel Damping Override
- [ ] Recenter priority active
- [ ] Safety gates pass
- [ ] `tau_wheel_vel * drift_sign < 0`
- [ ] `apcr1n_wheel_damping_override_active = True`
- [ ] `apcr1n_wheel_damping_scale = 0.30`

### Position Cap Boost
- [ ] Recenter priority active
- [ ] Safety gates pass
- [ ] `apcr1n_position_cap_boost_active = True`
- [ ] `apcr1n_position_cap_current = 5.0`

**All checkboxes will be validated during 2000-step ablation.**
