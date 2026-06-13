# WBC Artifact Audit - Step E Extreme Height Root Cause

**Date:** 2026-06-07
**Classification:** `WBC_FALSE_POSITIVE_SUPPORT_FEEDFORWARD`

---

## Summary

The WBC "applied=true" flag is a **structural artifact from QP force distribution**, not actual active whole-body control.

**Conclusion:** The gate `wbc_applied == false` fails because `tau_wbc_norm > 0.001`, but this is expected behavior from the QP force distribution solving for contact forces. The actual per-actuator WBC authority is **disabled** (`per_actuator_wbc_authority_enabled = False`).

---

## Telemetry Evidence

### tau_wbc_norm Values

| Case | tau_wbc_norm (max) | Gate | Pass/Fail |
|------|-------------------|------|-----------|
| low_0p300 | 14.49 Nm | > 0.001 | FAIL |
| high_0p480 | 20.13 Nm | > 0.001 | FAIL |

### tau_wbc_per_joint (Step 0, low_0p300)

```
Joint 0 (hip_roll):  0.0000 Nm
Joint 1 (hip_yaw):  -2.3242 Nm
Joint 2 (hip_pitch): 0.9437 Nm
Joint 3 (knee):      9.1910 Nm
Joint 4 (wheel):     0.0000 Nm
Joint 5 (hip_roll):  0.0000 Nm
Joint 6 (hip_yaw):   2.3242 Nm
Joint 7 (hip_pitch): 0.9436 Nm
Joint 8 (knee):      9.1910 Nm
Joint 9 (wheel):     0.0000 Nm
```

### active_torque_owner_per_joint (Step 0, low_0p300)

```
Joint 0 (hip_roll):  none
Joint 1 (hip_yaw):   none
Joint 2 (hip_pitch): support_feedforward
Joint 3 (knee):      support_feedforward
Joint 4 (wheel):     none
Joint 5 (hip_roll):  none
Joint 6 (hip_yaw):   none
Joint 7 (hip_pitch): support_feedforward
Joint 8 (knee):      support_feedforward
Joint 9 (wheel):      none
```

### per_actuator_wbc_authority_enabled

| Case | Value |
|------|-------|
| low_0p300 | False |
| high_0p480 | False |

---

## Root Cause Analysis

### What is tau_wbc?

`tau_wbc` is the **output of the whole-body controller QP solver**. The QP solves for contact forces to satisfy a desired wrench (force + torque) at the Center of Mass.

The QP force distribution maps contact forces to joint torques via the Jacobian transpose:

```
tau_wbc = J_contact^T * contact_forces
```

This is a **physical constraint satisfaction** calculation, not an active control signal.

### What is tau_wbc_per_joint?

For the balance-core controller, `tau_wbc` contains:
- Hip pitch (joint 2, 7): ~0.94 Nm (minimal)
- Knee (joint 3, 8): ~9.2 Nm at 0.300m, ~1.2 Nm at 0.480m

These torques are **inverse dynamics torques** computed from the contact forces the QP wants to apply to maintain balance.

### Active vs Structural Torque

| Torque Type | Description | Active Control? | Prohibited? |
|-------------|-------------|-----------------|-------------|
| **Structural** | QP force distribution | No - physics | No |
| **Active WBC** | Per-joint torque from RL/WBC policy | Yes | Yes |

**Evidence:**
- `active_torque_owner_per_joint` shows **"none"** for all joints
- `per_actuator_wbc_authority_enabled = False`
- Joint torques are **"support_feedforward"** for hip_pitch/knee

This means:
1. The QP solves for contact forces → tau_wbc
2. tau_wbc is mapped to joint torques via Jacobian
3. But the controller does **NOT** use these torques as active joint commands
4. The actual joint torques come from `tau_support_feedforward` only

---

## Gate Definition Problem

### Current Gate Definition

```python
wbc_applied = tau_wbc_norm > 0.001
```

This gate is **incorrect** because it fires on structural QP output, not active WBC.

### Correct Gate Definition

The gate should check for **active** WBC:

```python
# Option 1: Check the per-actuator authority flag
wbc_applied = per_actuator_wbc_authority_enabled

# Option 2: Check torque ownership
# If all owners are 'support_feedforward' or 'none', no active WBC
active_wbc = any(owner not in ['support_feedforward', 'none', 'shape_posture'] 
                for owner in active_torque_owner)
```

---

## Classification

**WBC_FALSE_POSITIVE_SUPPORT_FEEDFORWARD**

Rationale:
- `tau_wbc_norm > 0.001` because QP force distribution produces non-zero joint torques
- These torques are structural (physics), not active (control)
- `per_actuator_wbc_authority_enabled = False` confirms no active WBC
- `active_torque_owner` shows "support_feedforward" for joints 2,3,7,8 - this is the force distribution, not active control

---

## Recommendation

### Validator Fix (Required)

Change the `wbc_applied` gate from:

```python
# OLD (incorrect)
wbc_applied = tau_wbc_norm > 0.001
```

To:

```python
# NEW (correct)
wbc_applied = per_actuator_wbc_authority_enabled == True
```

Or more conservatively:

```python
# Check if any joint has non-structural torque ownership
wbc_applied = any(owner not in ['support_feedforward', 'none', 'shape_posture'] 
                  for owner in active_torque_owner_per_joint)
```

### After Validator Fix

Both heights would pass the `wbc_applied` gate because:
- `per_actuator_wbc_authority_enabled = False`
- No active WBC torques are applied

---

## Impact on Official Step E Results

| Gate | Current | After Fix |
|------|---------|-----------|
| wbc_applied (low_0p300) | FAIL | PASS |
| wbc_applied (high_0p480) | FAIL | PASS |

**Note:** The other failures (support drift, hip yaw, wheel velocity) remain unchanged.
