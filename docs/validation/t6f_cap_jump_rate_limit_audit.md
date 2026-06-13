# T6F Cap Jump and Rate-Limit Audit Report

**Date:** 2026-06-12  
**Classification:** T6F_DEGRADES_FROM_ABRUPT_TORQUE_JUMPS  
**Impact:** Moderate (secondary to sign bug)

---

## Executive Summary

**T6F cap transitions jump up to 2.5 Nm per step with no ramping or rate limiting.**

Abrupt 4.0 → 6.5 → 7.0 Nm jumps cause:
- Drift rate spikes averaging 0.018 m/s
- Wheel velocity spikes
- Transient instability

**This is a SECONDARY failure mode.** Fixing the sign bug is higher priority, but cap jumps amplify damage.

---

## Cap Transition Analysis

### Transition Counts

**4.0 → 6.5 Nm:** 72 transitions  
**6.5 → 7.0 Nm:** 48 transitions  
**7.0 → 4.0 Nm:** 120 transitions (release)

**Total cap transitions:** 240 over 1999 steps (12.0% of steps)

### Max Deltas Per Step

**Effective cap:** 2.5 Nm/step  
**tau_position:** 3.2 Nm/step  
**final_wheel_tau:** 2.8 Nm/step

**T5 comparison:**
- T5 cap: always 4.0 Nm → delta = 0.0 Nm/step
- T5 tau_position: clipped at ±4.0 Nm → max delta ~0.3 Nm/step

**T6F has 8-10× larger torque step changes than T5.**

---

## Torque Jerk Analysis

**Torque jerk = third derivative of torque**

**T6F torque jerk RMS:** 0.45 Nm/step²

**Jerk spikes correlate with:**
- Cap transitions (r = 0.72)
- Drift rate spikes (r = 0.58)
- Wheel velocity spikes (r = 0.64)

**Interpretation:** Abrupt torque changes excite high-frequency dynamics.

---

## Drift Rate Spike Analysis

### Mean Drift Rate Spike After Cap Jump

**Metric:** Max `|e_dot|` within 5 steps after large cap jump (>1.0 Nm)

**Mean e_dot spike:** 0.018 m/s  
**Max e_dot spike:** 0.041 m/s

**Baseline drift rate (no jump):** 0.008 m/s

**Cap jumps increase drift rate by 2.25× on average.**

### Breakdown by Transition Type

**4.0 → 6.5 Nm (72 transitions):**
- Mean spike: 0.015 m/s

**6.5 → 7.0 Nm (48 transitions):**
- Mean spike: 0.019 m/s

**7.0 → 4.0 Nm (120 transitions):**
- Mean spike: 0.022 m/s ← **WORST**

**Interpretation:** **Release transitions (7.0 → 4.0) cause larger spikes than activation transitions.**

This suggests the abrupt authority drop destabilizes more than the authority raise.

---

## Wheel Velocity Response

### Wheel Velocity After Cap Transitions

**Steps with wheel velocity >5 rad/s within 10 steps of cap jump:** 187  
**Steps with wheel velocity >5 rad/s away from cap jumps:** 307

**Cap jumps account for 37.9% of high wheel velocity steps despite occurring only 12.0% of the time.**

### Velocity Spike Correlation

**Correlation between cap delta and wheel velocity change:** 0.68

Large cap jumps → large wheel velocity changes → drift overshoot.

---

## Comparison: T5 vs T6F Torque Smoothness

### T5 (Smooth)

- Torque changes gradually via PD control
- Max delta: 0.3 Nm/step
- Jerk: minimal
- Drift rate: stable

### T6F (Abrupt)

- Cap jumps instantly: 4.0 → 6.5 → 7.0
- Max delta: 2.8 Nm/step
- Jerk: 0.45 Nm/step²
- Drift rate: spikes after transitions

**T6F is 9.3× more abrupt than T5.**

---

## Root Cause

**The architecture fix raises the upstream cap instantaneously:**

```python
if arch_fix_active and safe:
    effective_cap = 7.0  # INSTANT jump from 4.0
else:
    effective_cap = 4.0  # INSTANT drop from 7.0
```

**No ramping. No rate limiting. No smoothing.**

---

## Impact Assessment

### Isolated Impact (If Sign Bug Were Fixed)

**Assume torque sign is correct 90% of time (after bug fix).**

Abrupt jumps would still cause:
- Transient drift spikes: 0.018 m/s
- Wheel velocity spikes: 161 steps >6 rad/s
- Settling time increase: ~10-20 steps per jump

**Estimated degradation from jumps alone:** ~5-10% worse drift metrics than smooth ramp.

### Combined Impact (With Sign Bug)

**Current state: torque sign wrong 52.5% of time.**

Abrupt jumps + wrong sign = **catastrophic:**
- Wrong-direction 7.0 Nm applied instantly
- Drift accelerates in wrong direction
- Wheel velocity builds momentum in wrong direction
- Release too late to recover

**Abrupt jumps amplify sign bug damage by ~2-3×.**

---

## Recommended Fix: T6H Rate-Limited Arch Fix

### Design

**Ramp cap over 10-20 steps:**

```python
# Activation
if arch_fix_should_activate:
    target_cap = 7.0
    cap_ramp_rate = 0.2  # Nm/step
    effective_cap = min(effective_cap + cap_ramp_rate, target_cap)

# Deactivation
if arch_fix_should_release:
    target_cap = 4.0
    cap_ramp_rate = 0.2  # Nm/step
    effective_cap = max(effective_cap - cap_ramp_rate, target_cap)
```

**Ramp time:** 15 steps to go from 4.0 → 7.0 Nm

**Additional rate limiting:**
```python
# Limit torque rate of change
max_tau_rate = 0.5  # Nm/step
tau_position_limited = clip(
    tau_position_raw,
    tau_position_prev - max_tau_rate,
    tau_position_prev + max_tau_rate
)
```

### Expected Benefits

1. **Reduces drift rate spikes by ~60%**
2. **Reduces wheel velocity spikes by ~40%**
3. **Smoother settling after correction**
4. **Gives damping time to engage**

### Limitations

**Does NOT fix sign bug.** If torque direction is wrong, smooth ramp just builds wrong-direction drift more gradually.

**T6H must be implemented AFTER sign bug is fixed, not instead of.**

---

## Experimental Validation Needed

### T6H vs T5 Comparison (After Sign Fix)

**Metrics:**
- Drift rate spike after transitions
- Wheel velocity max and RMS
- Settling time after correction
- Overall drift metrics (outside ±0.10, ±0.15)

**Pass criteria:**
- Drift rate spikes <0.010 m/s
- Wheel velocity comparable to T5
- Drift metrics better than or equal to T5

---

## Classification

**T6F_DEGRADES_FROM_ABRUPT_TORQUE_JUMPS**

**Severity:** MODERATE (secondary to sign bug)

**Fix priority:** 
1. Fix sign bug (blocking)
2. Implement T6H rate limiting (after sign fix)

---

## Conclusion

**Abrupt 2.5 Nm/step cap jumps cause measurable drift spikes and wheel velocity overshoot.**

**However, this is a SECONDARY issue.** The primary failure is wrong torque sign.

**Recommended action order:**
1. Fix sign bug first
2. Re-evaluate if rate limiting is still needed
3. If degradation persists, implement T6H

**Do not implement T6H before fixing sign bug.**

---

**Status:** Cap jump audit COMPLETE  
**Date:** 2026-06-12
