# APCR1h Drift Priority Activation Audit - Final Report

## Date
2026-06-10

## Classification
**APCR1H_DRIFT_PRIORITY_WIRING_FIXED_AND_APCR1I_IMPLEMENTED**

## Summary

APCR1h drift priority wiring was fixed in a previous session. APCR1i (support hysteresis recenter) has been implemented but NOT yet added to CLI. The current task is to add APCR1i to CLI, add tests, and run validation.

---

## Phase 1: APCR1h Drift Priority Audit (Previous Session)

### Root Cause Found and Fixed

APCR1h had a wiring bug where `adaptive_max_tau` was overwritten immediately after being set by drift priority:

```python
# Lines 1775-1783: Drift priority sets adaptive_max_tau correctly
if self._apc_drift_priority_active:
    self._apc_drift_priority_tau_limit = selected_tau_limit
    tau_delta = selected_tau_limit - self._apc_drift_priority_prev_tau
    rate_limited_delta = max(-selected_rate_limit, min(selected_rate_limit, tau_delta))
    adaptive_max_tau = self._apc_drift_priority_prev_tau + rate_limited_delta

# Line 1791 (BEFORE FIX): OVERWRITES drift priority
adaptive_max_tau = self._apc_fast_response_adaptive_tau_limit  # ← Bug

# FIX: Moved to else branch
else:
    adaptive_max_tau = self._apc_fast_response_adaptive_tau_limit
```

### Post-Fix Results

| Metric | Before Fix (Buggy) | After Fix | Delta |
|--------|-------------------|-----------|-------|
| Max APCR tau | 1.253 Nm | **1.491 Nm** | +0.238 Nm |
| Max error | 0.1572 m | **0.1568 m** | -0.0004 m |

---

## Phase 2: Support Recenter Principle Verification

APCR1h (after wiring fix) still does not fully satisfy the user's desired principle:

**User's Principle:**
> If support drift moves far away from zero:
> - wheels may move faster and reverse direction if needed
> - support drift recovery must be prioritized even if pitch is near balanced
> - the controller must keep driving support back toward zero
> - once the drift direction reverses, do not immediately switch back
> - hold the recenter phase until the support error reaches near zero or crosses slightly to the opposite side
> - then switch/release according to symmetric hysteresis

**APCR1h Behavior:**
- Uses proportional soft band control
- Phase brake reduces torque when error starts returning toward zero
- Exits early when pitch becomes balanced or pitch sign changes
- Does NOT hold recenter through zero crossing
- Does NOT have symmetric hysteresis state machine

**Conclusion: APCR1H_DOES_NOT_SATISFY_RECENTER_PRINCIPLE**

---

## Phase 4: APCR1i Design

APCR1i was designed to implement the user's principle with a symmetric hysteresis state machine.

### State Machine States

| State | Description |
|-------|-------------|
| NEUTRAL | No recenter active, error near zero |
| RECENTER_FROM_POSITIVE | Positive drift, driving backward |
| RECENTER_FROM_NEGATIVE | Negative drift, driving forward |
| HOLD_THROUGH_ZERO | Error crossing zero, holding direction |

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| outer_enter_m | 0.08 | Enter recenter when |e| > this |
| inner_exit_m | 0.03 | Exit recenter when |e| <= this |
| opposite_release_m | 0.03 | Allow small overshoot into opposite direction |
| emergency_m | 0.12 | Emergency clamp activates |
| hard_m | 0.15 | Hard safety activates |
| recenter_max_tau | 1.75 Nm | Max during recenter |
| emergency_max_tau | 2.00 Nm | Max during emergency |
| hold_max_tau | 1.50 Nm | Max during hold-through-zero |
| recenter_rate | 0.90 Nm/step | Rate during recenter |
| emergency_rate | 1.00 Nm/step | Rate during emergency |

### State Transitions

```
NEUTRAL → RECENTER_FROM_POSITIVE: e > +0.08
NEUTRAL → RECENTER_FROM_NEGATIVE: e < -0.08
RECENTER_FROM_POSITIVE → NEUTRAL: e <= 0.03 AND e_dot < 0
RECENTER_FROM_POSITIVE → RECENTER_FROM_NEGATIVE: e < -0.03 (overshoot)
RECENTER_FROM_NEGATIVE → NEUTRAL: e >= -0.03 AND e_dot > 0
RECENTER_FROM_NEGATIVE → RECENTER_FROM_POSITIVE: e > +0.03 (overshoot)
```

### Phase Brake

- **DISABLED** while outside inner band (|e| > 0.03)
- **ENABLED** only when |e| <= 0.03 and moving toward zero

---

## Phase 5: APCR1i Implementation Status

### Implementation Complete

APCR1i is implemented in `sagittal_velocity_damped_balance_controller.py`:
- Profile defined at lines 639-676
- State machine implemented at lines 1815-1966
- Telemetry added at lines 2601+

### Missing: CLI Integration

APCR1i profile is defined but NOT added to CLI choices in `simulate_hierarchical_controller.py`.

---

## Phase 6: Required Actions

1. **Add APCR1i to CLI choices** in `simulate_hierarchical_controller.py`
2. **Add tests for APCR1i** in `test_sagittal_velocity_damped_balance_controller.py`
3. **Run APCR1i 500-step validation**
4. **Compare APCR1i vs APCR1h vs D2**

---

## Classification: APCR1H_DRIFT_PRIORITY_WIRING_FIXED_AND_APCR1I_IMPLEMENTED