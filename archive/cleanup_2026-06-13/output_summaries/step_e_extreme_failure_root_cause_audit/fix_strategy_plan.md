# Step E Extreme Height Root Cause — Fix Strategy Plan

**Date:** 2026-06-07
**Baseline:** D2 official
**Scope:** Validator fix only (controller modifications explicitly out of scope)

---

## Executive Summary

This audit identified **one validator gate bug** and **two structural controller limitations** as root causes of Step E failures at 0.300m and 0.480m heights.

| Gate | Issue | Fix Required |
|------|-------|--------------|
| `wbc_applied` | **FALSE POSITIVE** — fires on QP structural artifact | Validator change |
| `support_position_error` | Position authority saturated, no integral | Controller change (out of scope) |
| `wheel_velocity` | Transient spike at high height | Controller change (out of scope) |
| `hip_yaw` | HY2-DIV disabled, no corrective torque | Controller change (out of scope) |

**Scope constraint:** Controller modifications (position integral, position authority, HY2-DIV, WBC) are explicitly prohibited by audit scope. Only the validator gate fix can be implemented.

---

## Gate Fix: wbc_applied

### Current Definition (INCORRECT)

```python
wbc_applied = tau_wbc_norm > 0.001
```

### Problem

`tau_wbc` is the **output of the whole-body controller QP solver**, which maps contact forces to joint torques via Jacobian transpose:

```python
tau_wbc = J_contact^T * contact_forces
```

This is **structural physics**, not active control. The QP always produces non-zero tau_wbc during contact, regardless of whether active WBC is used.

### Evidence

| Metric | low_0p300 | high_0p480 |
|--------|-----------|------------|
| `tau_wbc_norm` | 14.49 Nm | 20.13 Nm |
| `per_actuator_wbc_authority_enabled` | False | False |
| `active_torque_owner` | support_feedforward | support_feedforward |

The actual torque owners show **"support_feedforward"** — this is the force distribution QP, not active WBC.

### Recommended Fix

```python
# Option 1: Check per-actuator authority flag (RECOMMENDED)
wbc_applied = per_actuator_wbc_authority_enabled == True

# Option 2: Check torque ownership
# If all owners are 'support_feedforward' or 'none', no active WBC
active_wbc = any(owner not in ['support_feedforward', 'none', 'shape_posture'] 
                for owner in active_torque_owner_per_joint)
```

### Impact After Fix

| Case | Current | After Fix |
|------|---------|-----------|
| low_0p300 | FAIL | **PASS** |
| high_0p480 | FAIL | **PASS** |

---

## Controller Fixes (OUT OF SCOPE — Documented for Future)

These fixes are **documented but not implementable** under the current audit scope.

### Fix 1: Enable Position Integral

**Root Cause:** `integral_active_ratio = 0.0` — position integral never activates.

**Recommended Change:**
```yaml
# In D2 profile or controller config
position_integral:
  enabled: true
  ki: 0.5  # Current value assumed to be 0
  integral_limit: 2.0  # Nm
```

**Evidence:** `tau_position_integral_max = 0.0 Nm` for both heights.

**Impact:** Would enable steady-state drift correction.

### Fix 2: Increase Position Authority

**Root Cause:** `tau_position` saturates at 4.0 Nm immediately.

**Recommended Change:**
```yaml
# In D2 profile or controller config
position_authority:
  saturation_limit: 6.0  # Nm (current assumed 4.0 Nm)
```

**Evidence:** `tau_position_max = 4.0 Nm` saturates within first 110 steps.

**Impact:** Would provide more position correction authority before saturation.

### Fix 3: Enable HY2-DIV (Documented, Not Recommended for D2)

**Note:** HY2-DIV is explicitly out of scope. The root cause is documented for completeness.

**Root Cause:** `hip_yaw_div_enabled = 0` steps, no corrective differential yaw torque.

**Evidence:** Hip yaw failures are in **divergence mode** (legs rotating opposite), indicating asymmetric loading without correction.

**If Enabled:** Would provide corrective yaw torque to reduce leg divergence.

---

## Remaining Failures After Validator Fix

After fixing the `wbc_applied` gate:

| Case | Remaining Failures |
|------|-------------------|
| low_0p300 | support_position_error, hip_yaw |
| high_0p480 | wheel_velocity, support_position_error, hip_yaw |

These failures require **controller modifications** (out of scope).

---

## Gap Analysis

| Fix Category | Implementable? | Notes |
|--------------|---------------|-------|
| wbc_applied gate | **YES** | Validator change only |
| Position integral | NO | Controller modification |
| Position authority | NO | Controller modification |
| HY2-DIV | NO | Explicitly prohibited |
| WBC | NO | Explicitly prohibited |

**Recommendation:** Implement the validator gate fix, then proceed to Step C/D with a D2+controller_fixes profile that includes position integral and increased authority.

---

## Implementation Risk

| Fix | Risk Level | Mitigation |
|-----|-----------|------------|
| wbc_applied gate | **NONE** | Aligns gate with actual WBC usage |
| Position integral | LOW | Add with proper anti-windup |
| Position authority | LOW | Add with saturation monitoring |

---

## Files to Modify

### Validator Gate Fix

```
wheeled_biped/controllers/step_e_validator.py
```

Change the `wbc_applied` gate definition from:
```python
wbc_applied = tau_wbc_norm > 0.001
```

To:
```python
wbc_applied = per_actuator_wbc_authority_enabled == True
```

Or more conservatively:
```python
wbc_applied = any(owner not in ['support_feedforward', 'none', 'shape_posture'] 
                  for owner in active_torque_owner_per_joint)
```

---

## Verification Plan

After implementing the validator fix:

1. Run Step E official check for low_0p300
2. Verify wbc_applied gate now PASSES
3. Run Step E official check for high_0p480
4. Verify wbc_applied gate now PASSES

Expected remaining failures (documented):
- low_0p300: support_position_error, hip_yaw
- high_0p480: wheel_velocity, support_position_error, hip_yaw
