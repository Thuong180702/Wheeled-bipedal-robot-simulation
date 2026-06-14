# APCR1n APC Dependency Audit

**Date:** 2026-06-11
**Profile:** APCR1n (APCR1N_RECENTER_PRIORITY_TORQUE_BOOST)
**Issue:** Features did not activate in 2000-step evaluation despite drift occurring

## Classification

**APCR1N_FEATURES_BLOCKED_BY_UNNECESSARY_APC_DEPENDENCY**

## Evidence

### Telemetry from Phase 2 evaluation
- `active_pitch_crossing_active = 0/2000`
- `active_pitch_crossing_gate_reason = "disabled"` for all steps
- `recenters_priority_active = 0/2000`
- `wheel_damping_override_active = 0/2000`
- `position_cap_boost_active = 0/2000`

### Drift conditions DID occur
- `|e| > 0.08` for 716/1900 post-startup steps
- `moving_away` for 899/1900 post-startup steps
- both conditions for 337/1900 steps

## Root Cause Analysis

### Code Path

APCR1n features depend on `_apc_drift_priority_active` (line 1644):

```python
# Line 1630-1645
if self.authority_schedule.recenter_priority_enabled:
    if current_step < startup_guard_steps:
        apcr1n_startup_guard_active = True
    else:
        # RECENTER is active if drift priority is active (APCR1h uses drift_priority)
        apcr1n_recenter_priority_active = self._apc_drift_priority_active
```

### Where `_apc_drift_priority_active` is set

The `_apc_drift_priority_active` flag is set ONLY within the APCR1h drift priority block at line 2278-2331:

```python
# Line 2278
if drift_priority_enabled:
    ...
    if abs_error > emergency_threshold and moving_away_drift:
        self._apc_drift_priority_emergency_active = True
        self._apc_drift_priority_active = True
        ...
    elif abs_error > drift_priority_enter and moving_away_drift:
        self._apc_drift_priority_active = True
        ...
```

### Why `_apc_drift_priority_active` is never set for APCR1n

The drift priority block at line 2278 is nested inside the **proportional soft band branch** (line 2134):

```python
# Line 2134
if apc_proportional_mode and apc_enabled and apc_gate_safe and not predictive_enabled and not hysteresis_enabled:
    ...
    # Line 2278: Drift priority block
    if drift_priority_enabled:
        ...
        self._apc_drift_priority_active = True  # Only reachable if apc_enabled=True
```

For APCR1n, `apc_proportional_mode=True` BUT `apc_enabled=False` because:
- APCR1n profile does NOT set `enable_active_pitch_crossing=True`
- Default value for `enable_active_pitch_crossing` is `False` (line 204)

Therefore:
- Line 2134 condition: `apc_proportional_mode and apc_enabled` → `True and False` → **skipped**
- Drift priority block at line 2278 is **never reached**
- `_apc_drift_priority_active` is **never set to True**
- APCR1n features never activate

### APC gate reason for all steps: "disabled"

Even if we somehow reached the drift priority logic, line 2837-2838 confirms why APC telemetry shows "disabled":

```python
if not apc_enabled:
    apc_gate_reason = "disabled"
```

### APCR1n profile analysis

APCR1n profile (lines 936-1001):
- `apc_drift_priority_enabled=True` ✓
- `apc_proportional_soft_band_mode=True` ✓
- **MISSING:** `enable_active_pitch_crossing=True`

The APCR1n design intended to use APCR1h drift priority WITHOUT enabling full APC. But the code structure requires `apc_enabled=True` to enter the proportional branch where drift priority lives.

### Design intent vs. implementation

| Component | Intent | Reality |
|-----------|--------|---------|
| APCR1n features | Activate based on drift | Blocked by APC |
| `apc_drift_priority_enabled` | Enable drift logic | Part of proportional branch |
| `apc_proportional_soft_band_mode` | Enable proportional torque | Needs `apc_enabled=True` |
| `enable_active_pitch_crossing` | Enable APC | Not set for APCR1n |

### Was APC dependency necessary?

**NO.** The drift priority logic (abs_error > threshold AND moving_away) does not require APC state machine. It only needs:
1. Drift threshold check
2. Moving away detection
3. Safety gates

APCR1n features should activate based on these conditions directly, not via APC state.

## Solution

Create APCR1nD with **direct support drift trigger** that:
1. Does NOT require `enable_active_pitch_crossing=True`
2. Does NOT enter the proportional soft band branch
3. Directly checks drift conditions and safety gates
4. Sets `_apcr1nd_direct_recenter_priority_active` flag independently

## Files to Modify

1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
   - Add `APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES` profile
   - Add direct drift trigger logic BEFORE the proportional branch
   - Add telemetry fields for APCR1nD

2. `tests/test_sagittal_velocity_damped_balance_controller.py`
   - Add tests for APCR1nD profile
   - Add tests for direct trigger logic

## Verification Plan

1. Run 2000-step with APCR1nD
2. Verify `apcr1nd_direct_recenter_priority_active` activates when:
   - step >= 100
   - abs(e) > 0.08
   - moving_away = True
   - safety gates pass
3. Verify `apcr1nd_wheel_damping_override_active` activates when:
   - direct recenter active AND damping fights drift
4. Verify `apcr1nd_position_cap_boost_active` activates when:
   - direct recenter active AND safety gates pass

## References

- APCR1n feature activation trigger test: `docs/validation/apcr1n_feature_activation_trigger_test_report.md`
- APCR1n Phase 2 runtime feature activation audit: `docs/validation/apcr1n_phase2_runtime_feature_activation_audit.md`
