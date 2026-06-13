# Step E Extreme Height Root Cause Audit — Final Report

**Date:** 2026-06-07
**Baseline:** D2 official
**Cases:** low_0p300 (0.300m), high_0p480 (0.480m)
**Scope:** Validator fix only (no controller modifications)

---

## Executive Summary

This audit investigated why the D2 baseline fails Step E official gates at extreme heights (0.300m and 0.480m). The investigation identified:

| Finding | Count | Fixable? |
|---------|-------|----------|
| **TRUE FAILURES** | 6 gate failures | No (controller changes required) |
| **FALSE POSITIVE** | 2 gate failures | **YES (validator bug)** |

### Primary Finding

> **The `wbc_applied` gate is a FALSE POSITIVE.** It fires on `tau_wbc_norm > 0.001`, but `tau_wbc` is structural QP force distribution output, not active WBC control. Both heights should **PASS** this gate since `per_actuator_wbc_authority_enabled = False`.

---

## Gate Results Summary

| Gate | low_0p300 (0.300m) | high_0p480 (0.480m) | Fixable? |
|------|--------------------|--------------------|----------|
| `wbc_applied` | ❌ FALSE POSITIVE | ❌ FALSE POSITIVE | **YES** |
| `support_position_error < 0.15m` | ❌ FAIL (0.176m) | ❌ FAIL (0.173m) | No |
| `wheel_velocity < 5.0 rad/s` | ✅ PASS (4.39) | ❌ FAIL (5.26) | No |
| `hip_yaw < 0.10 rad` | ❌ FAIL (0.313) | ❌ FAIL (0.275) | No |

---

## Event Order

### low_0p300 (0.300m): SUPPORT_DRIFT_PRIMARY

| Rank | Event | Step | Time (s) | Value |
|------|-------|------|----------|-------|
| 1 | support_drift_gate | 91 | 0.91 | 0.176m |
| 2 | hip_yaw_gate | 328 | 3.28 | 0.313 rad |

**Causal chain:** Support drift first → hip yaw divergence second

### high_0p480 (0.480m): WHEEL_VELOCITY_PRIMARY

| Rank | Event | Step | Time (s) | Value |
|------|-------|------|----------|-------|
| 1 | wheel_velocity_gate | 73 | 0.73 | 5.26 rad/s |
| 2 | support_drift_gate | 108 | 1.08 | 0.173m |
| 3 | hip_yaw_gate | 2426 | 24.26 | 0.275 rad |

**Causal chain:** Wheel velocity spike → support drift → hip yaw divergence

---

## Root Cause Analysis

### 1. wbc_applied: FALSE POSITIVE (FIXABLE)

**Classification:** `WBC_FALSE_POSITIVE_SUPPORT_FEEDFORWARD`

**Evidence:**
- `tau_wbc_norm_low_0p300 = 14.49 Nm` — fires gate
- `tau_wbc_norm_high_0p480 = 20.13 Nm` — fires gate
- `per_actuator_wbc_authority_enabled = False` — **no active WBC**
- `active_torque_owner = support_feedforward` — QP force distribution

**Mechanism:**
```
Gate definition: wbc_applied = tau_wbc_norm > 0.001

tau_wbc = J_contact^T × contact_forces  ← QP force distribution (structural)
       ≠ active WBC torques

Reality: per_actuator_wbc_authority_enabled = False
```

**Fix:**
```python
# Current (incorrect)
wbc_applied = tau_wbc_norm > 0.001

# Fixed
wbc_applied = per_actuator_wbc_authority_enabled == True
```

---

### 2. support_position_error: POSITION_AUTHORITY_SATURATED

**Classification:** `NO_POSITION_INTEGRAL`, `POSITION_AUTHORITY_SATURATED`, `OSCILLATORY_DRIFT`

**Evidence:**
- `tau_position_max = 4.0 Nm` (saturated at limit)
- `tau_position_integral_max = 0.0 Nm` (integral disabled)
- `integral_active_ratio = 0.0` (never activates)
- `drift_characteristic = OSCILLATORY`

**Mechanism:**
```
tau_position = Kp × position_error + Ki × integral_error (DISABLED)
     ↓
No position integral → cannot correct steady-state drift
Position authority saturated at 4.0 Nm → no remaining authority
     ↓
CoM drifts away from wheel support center
```

---

### 3. wheel_velocity: TRANSIENT_EXCEEDANCE (high_0p480 only)

**Classification:** `TRANSIENT_EXCEEDANCE`, `WHEEL_VEL_LEADS_SUPPORT`

**Evidence:**
- Peak: 5.26 rad/s at step 83
- Time above gate: only **0.34%** (17 steps out of 5000)
- `wheel_vel_leads_support = True`
- `sagittal_term_pitch = 3.287 Nm` at peak

**Mechanism:**
```
High height → more top-heavy → larger pitch response
     ↓
Sagittal term pitch spike → wheel velocity spike
     ↓
CoM momentum buildup → support drift
```

---

### 4. hip_yaw: DIVERGENCE_MODE (HY2_DIV_DISABLED)

**Classification:** `DIVERGENCE_PRIMARY`, `HY2_DIV_DISABLED`, `SECONDARY_TO_SUPPORT_DRIFT`

**Evidence:**
- `hip_yaw_divergence_max = 0.312 rad` (low), `0.275 rad` (high)
- `hip_yaw_common_mode_max = 0.010 rad` (low), `0.020 rad` (high)
- `hip_yaw_div_torques_used = False` (HY2-DIV disabled)
- `hip_yaw_comp_active_ratio = 0.0`

**Mechanism:**
```
Support drift → asymmetric loading on legs
     ↓
No corrective differential yaw torque (HY2-DIV disabled)
     ↓
Legs rotate outward (diverge), not together (common mode)
```

---

## Structural Root Causes (Out of Scope)

These findings identify systemic controller limitations that require modifications outside the validator scope:

### A. Position Integral Disabled

| Evidence | Value |
|----------|-------|
| `integral_active_ratio` | 0.0 |
| `integral_active_count` | 0 |
| `tau_position_integral_max` | 0.0 Nm |

**Impact:** Cannot correct steady-state position drift.

### B. Position Authority Saturated

| Evidence | Value |
|----------|-------|
| `tau_position_max` | 4.0 Nm |
| `saturated_at_step` | ~108 |

**Impact:** Controller exhausts position authority immediately.

### C. HY2-DIV Disabled

| Evidence | Value |
|----------|-------|
| `hip_yaw_div_enabled_count` | 0 |
| `hip_yaw_div_active_count` | 0 |

**Impact:** No corrective yaw differential torque.

---

## Recommendations

### Immediate (Validator Fix)

| Action | File | Change |
|--------|------|--------|
| Fix wbc_applied gate | `wheeled_biped/controllers/step_e_validator.py` | `wbc_applied = per_actuator_wbc_authority_enabled == True` |

### For Step C/D (Controller Changes)

| Priority | Action | Rationale |
|----------|--------|-----------|
| 1 | Enable position integral | Correct steady-state drift |
| 2 | Increase position authority limit | Prevent immediate saturation |
| 3 | Consider HY2-DIV | Correct hip yaw divergence |

### Not Recommended

| Action | Why |
|--------|-----|
| Relax gates | Hides real issues |
| Add WBC | Gate bug is the issue, not missing WBC |
| Gain tuning | No root cause identified for tuning targets |

---

## Files Produced

| Phase | File | Status |
|-------|------|--------|
| Phase 0 | `artifact_inventory.json`, `artifact_inventory.md` | ✅ |
| Phase 1 | `wbc_artifact_audit.json`, `wbc_artifact_audit.md` | ✅ |
| Phase 2 | `event_order/event_order_*.csv`, `event_order_summary.json` | ✅ |
| Phase 3 | `support_drift/support_drift_audit.json` | ✅ |
| Phase 4 | `hip_yaw/hip_yaw_audit.json` | ✅ |
| Phase 5 | `wheel_velocity/wheel_velocity_audit.json` | ✅ |
| Phase 6 | `causal_map.json`, `causal_map.md` | ✅ |
| Phase 7 | `fix_strategy_plan.json`, `fix_strategy_plan.md` | ✅ |
| Phase 8 | `step_e_extreme_height_root_cause_summary.json`, `step_e_extreme_height_root_cause_summary.md` | ✅ |

---

## Conclusion

The Step E failures at 0.300m and 0.480m have **one fixable gate bug** (wbc_applied false positive) and **three structural controller limitations** (position integral disabled, position authority saturated, HY2-DIV disabled).

**Immediate action:** Fix the wbc_applied gate in the validator to check `per_actuator_wbc_authority_enabled` instead of `tau_wbc_norm > 0.001`.

**Remaining failures:** After the validator fix, both heights will still fail `support_position_error`, `hip_yaw`, and `high_0p480` will also fail `wheel_velocity`. These require controller modifications (position integral, position authority, HY2-DIV) which are documented in the fix strategy plan.
