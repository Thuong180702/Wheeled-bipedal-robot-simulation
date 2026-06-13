# APCR1h Drift Priority Wiring Fix Report

## Date
2026-06-10

## Classification
**APCR1H_DRIFT_PRIORITY_WIRING_FIXED**

## Summary

The APCR1h drift priority feature had a wiring bug where line 1719 overwrote the drift priority tau limit with the lower APCR1f value. This has been fixed.

---

## Root Cause

In `sagittal_velocity_damped_balance_controller.py`, the drift priority override correctly set `adaptive_max_tau` to the higher drift priority limit, but line 1719 unconditionally overwrote it:

```python
# Lines 1702-1711: Drift priority sets adaptive_max_tau correctly
if self._apc_drift_priority_active:
    self._apc_drift_priority_tau_limit = selected_tau_limit
    # ... rate limiting ...
    adaptive_max_tau = self._apc_drift_priority_prev_tau + rate_limited_delta

# Line 1719 (BEFORE FIX): OVERWRITES drift priority
adaptive_max_tau = self._apc_fast_response_adaptive_tau_limit  # ← This line negates drift priority
```

---

## Fix Applied

Moved the APCR1f tau assignment into an `else` branch so it only applies when drift priority is NOT active:

```python
# Phase brake is disabled when drift priority is active
if self._apc_drift_priority_active:
    self._apc_fast_response_phase_brake_active = False
else:
    # Use APCR1f max tau only when drift priority is NOT active
    adaptive_max_tau = self._apc_fast_response_adaptive_tau_limit
```

---

## Validation Results

### 500-Step Simulation After Fix

| Metric | Before Fix (Buggy) | After Fix | Delta |
|--------|-------------------|-----------|-------|
| Max APCR tau | 1.253 Nm | **1.491 Nm** | +0.238 Nm |
| Max error | 0.1572 m | **0.1568 m** | -0.0004 m |
| Outside ±0.15 | 36 steps | **34 steps** | -2 steps |
| P2P | - | **0.1801 m** | - |

### Drift Priority Activation

When drift priority conditions are met (|error| > 0.08 AND moving away):
- 89 candidate steps in 500-step run
- APCR tau at candidates: mean = -1.24 Nm, max = -1.41 Nm
- This is higher than the old 1.253 Nm cap, confirming drift priority is working

### Comparison with APCR1f Baseline

| Metric | APCR1f (baseline) | APCR1h Fixed | Delta |
|--------|-------------------|--------------|-------|
| Max error | 0.1572 m | 0.1568 m | -0.0004 m |
| Max APCR tau | 1.253 Nm | 1.491 Nm | +0.238 Nm |
| Outside ±0.15 | 2.2% (2000-step) | 6.8% (500-step) | Mixed |

---

## Findings

1. **Fix confirmed working**: APCR tau increased by +0.238 Nm when drift priority activates
2. **Modest drift improvement**: Max error reduced by 0.4 mm, outside ±0.15 reduced by 2 steps
3. **Rate limiting still constrains**: The higher authority takes multiple steps to ramp up due to rate limits
4. **APCR1i design needed**: The modest improvement suggests APCR1h still doesn't fully satisfy the user's principle

---

## Next Steps

1. **Phase 4**: Design APCR1i with proper hysteresis state machine
2. **Phase 5**: Implement APCR1i
3. **Phase 7**: Run APCR1i 500-step validation

---

## Files Modified

- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
  - Line 1717-1719: Added `else` branch to prevent drift priority overwrite
