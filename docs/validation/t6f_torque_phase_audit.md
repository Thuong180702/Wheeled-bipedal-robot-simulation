# T6F Torque Phase Audit Report

**Date:** 2026-06-12  
**Classification:** T6F_DEGRADES_FROM_WRONG_TORQUE_SIGN (primary)  
**Supporting:** PHASE_LAG_OVERSHOOT (minor), HIGH_TORQUE_HELD_TOO_LONG (moderate)

---

## Executive Summary

**T6F final wheel torque opposes drift error only 47.5% of time — essentially random.**

This is the PRIMARY root cause of drift degradation. Raised authority from 4.0 Nm to 7.0 Nm amplifies wrong-direction torque, making drift worse instead of better.

---

## Torque Direction Analysis

### Direction Correctness

**Metric:** Does `sign(final_wheel_tau)` oppose `sign(drift_error)`?

**Expected:** Torque should oppose drift >80% of time  
**Actual T6F:** Torque opposes drift **47.5%** of time

**This indicates a latent sign convention bug in the torque composition path.**

### Breakdown by Regime

Analysis of when torque direction is correct vs incorrect:

**When arch_fix is active (913 steps):**
- Correct direction: 43.2%
- Wrong direction: 56.8%

**When arch_fix is inactive (1086 steps):**
- Correct direction: 50.8%
- Wrong direction: 49.2%

**Interpretation:** Wrong-direction torque is present in BOTH normal (4.0 Nm) and raised (7.0 Nm) authority regimes, but **becomes more damaging when amplified to 7.0 Nm**.

---

## Phase Lag Analysis

### Activation to Reversal Delay

**Mean delay from arch_fix activation to e_dot sign reversal:** 15.2 steps

This is moderate. For comparison:
- 10 steps = acceptable control response
- 20 steps = slow but tolerable
- >30 steps = control lag problem

15.2 steps is within acceptable range but not optimal.

### Reversal to Deactivation Delay

**Mean delay from e_dot reversal to arch_fix deactivation:** 28.4 steps

This is the more concerning metric. After error starts converging, high torque remains active for ~28 steps on average.

**Impact:** Overpowered correction continues during convergence phase, preventing settling.

---

## Overshoot Analysis

### Detected Overshoot Events

**Count:** 0 significant overshoot events in 2000-step window

**Overshoot detection criteria:**
- Error reverses direction after e_dot sign change
- `|e_after| > |e_before|` within 20 steps
- Error crosses zero

**Why no overshoot detected despite degradation?**

T6F error trajectory is **monotonic drift in negative direction**, not oscillatory overshoot:
- T5: oscillates around zero, crosses ±0.08 m band
- T6F: drifts progressively negative, reaches -0.212 m

**Interpretation:** Wrong-sign torque causes cumulative drift, not overshoot oscillation.

---

## Convergence Phase Behavior

### High Torque While Converging

**High torque active while `e * e_dot < 0` (converging):** 421 steps (21.1% of episode)

**Breakdown:**
- Error moving away (`e * e_dot > 0`): 1013 steps → arch_fix active 90.1%
- Error converging (`e * e_dot < 0`): 986 steps → arch_fix active 42.7%

**Key finding:** Architecture fix does not decay quickly enough when error starts converging.

### Impact on Drift

When high torque is held during convergence:
- If torque sign is CORRECT: may overshoot but will recover
- If torque sign is WRONG: prevents convergence and amplifies drift

**T6F has wrong sign 52.5% of time, so held high torque during convergence is destructive.**

---

## Comparison: T5 vs T6F Torque Behavior

### T5 (4.0 Nm cap)

- Torque opposes drift: ~85% of time (estimated from drift recovery)
- Max torque: 4.0 Nm clipped
- Drift amplitude: oscillates ±0.08 to ±0.15 m
- Recovery: error returns toward zero after excursions

### T6F (7.0 Nm raised cap)

- Torque opposes drift: **47.5%** of time
- Max torque: 7.0 Nm transmitted
- Drift amplitude: monotonic negative drift to -0.212 m
- No recovery: error grows progressively

**Interpretation:** T5's 4.0 Nm clip masks the sign bug by limiting damage. T6F's 7.0 Nm exposes and amplifies the bug.

---

## Root-Cause Hypothesis

### Latent Sign Convention Bug

**Hypothesis:** The torque composition path has a sign flip that becomes dominant at high torque magnitude.

**Possible locations:**
1. **`apcr1n_tau_position_after_cap` composition:** Position torque may have wrong sign when raised cap is applied
2. **Wheel velocity damping sign:** Damping term may oppose correction instead of assisting at high velocity
3. **Support drift error sign:** Sign convention may flip when error exceeds certain threshold
4. **Yaw compensation torque:** Hip yaw feedforward may have wrong sign at high magnitude
5. **Final wheel torque composition with APC:** APC torque addition may have wrong sign

### Why Bug Was Not Detected Before

- **T5 at 4.0 Nm:** Bug present but damage limited by cap
- **Phase 7 (1200 steps):** Too short to accumulate severe degradation
- **Normal-height tests:** Bug may be less severe at lower heights where drift is smaller

### Evidence Supporting Sign Bug

1. **Random torque direction:** 47.5% correct is near 50% coin flip
2. **Present in both regimes:** Wrong direction in normal AND raised authority
3. **No gain mismatch:** Phase 4 found response ratio = 0.97, so sign is the issue
4. **Monotonic drift:** Pattern consistent with amplified wrong-direction torque, not overshoot

---

## Diagnostic Test

### Recommended Immediate Test

Add step-by-step sign audit to telemetry:

```python
# In controller compute step:
sign_drift_error = np.sign(active_pitch_crossing_signed_error_m)
sign_final_tau = np.sign(final_wheel_tau_with_apc)
sign_correct = (sign_drift_error * sign_final_tau < 0)  # Opposite signs

# Log:
telemetry["torque_sign_correct"] = sign_correct
telemetry["sign_drift"] = sign_drift_error
telemetry["sign_tau"] = sign_final_tau
```

Run 500-step diagnostic on T6F and report:
- `sign_correct` percentage by band state
- Steps where sign is wrong and `|tau| > 5.0 Nm`
- Correlation between wrong sign and drift growth rate

**Expected if sign bug exists:**
- Wrong sign concentrated at high torque magnitude
- Drift rate spikes when wrong sign + high torque

---

## Classification

**Primary:** T6F_DEGRADES_FROM_WRONG_TORQUE_SIGN  
**Supporting:** HIGH_TORQUE_HELD_TOO_LONG

**Phase lag and overshoot are NOT primary causes.**

---

## Next Steps

1. **Fix torque sign bug** (blocking priority)
2. **After sign fix:** Re-run T6F 2000-step screening
3. **If sign fix solves degradation:** T6F may pass without T6H/T6I
4. **If degradation persists:** Proceed to T6H/T6I candidates

**Do not implement T6G/T6H/T6I until sign bug is proven fixed.**

---

**Status:** Torque phase audit COMPLETE  
**Date:** 2026-06-12
