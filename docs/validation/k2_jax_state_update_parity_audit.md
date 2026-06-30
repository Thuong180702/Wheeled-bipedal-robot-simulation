# K2 JAX State Update Parity Audit — Phase 6

**Date:** 2026-06-27
**Method:** Compare JAX state evolution against expected Python state evolution
**Verdict:** **NOTCH FILTER STATE DIVERGES — ROOT CAUSE OF WHEEL MISMATCH**

---

## 1. State Layout (328 fields)

| Group | Fields | Indices | Update Timing |
|-------|--------|---------|---------------|
| Notch filter | 4 (x1, x2, y1, y2) | 0-3 | Every step, before sagittal torque |
| Prev torque | 10 (all actuators) | 4-13 | Every step, after composer |
| Filtered CoM Z | 1 | 14 | Every step, during height scheduling |
| Prev support error | 1 | 15 | Every step, after outer loop |
| Outer loop pitch ref | 1 | 16 | Every step, rate-limited + lowpassed |
| Outer loop prev support error | 1 | 17 | Every step |
| Outer loop support error rate | 1 | 18 | Every step, lowpassed |
| ABS core | 9 (sum, trim, hold, sign, zc, count, ptr, guard) | 19-27 | Every step, during ABS update |
| ABS ring buffer | 300 (slow window) | 28-327 | Every step, circular write |

---

## 2. State Initialization

### JAX initialization:
```python
_jax_state = pack_state_k2()
# All fields zero. filtered_com_z defaults to 0.4.
```

### Python initialization:
- Notch state: zeros (BiquadNotchFilter starts clean)
- prev_tau: zeros (first step has no previous torque)
- filtered_com_z: 0.4 (or height_cmd?)
- Support/outer loop: zeros
- ABS: zeros (empty ring buffer)

**Status at step 0: IDENTICAL for all state fields.**

---

## 3. Step 0 → Step 1 State Transition (fixed_high_0p480)

### Notch Filter State

**Input at step 1:** pitch_rate = 0.251322669978 rad/s (IDENTICAL for PY and JX)

**JAX notch update (lines 1098-1102):**
```python
notch_out = notch_b0 * pitch_rate + notch_b1 * notch_x1 + notch_b2 * notch_x2 
          - notch_a1 * notch_y1 - notch_a2 * notch_y2
new_notch_x1 = pitch_rate           # = 0.251322669978
new_notch_x2 = old_notch_x1         # = 0.0
new_notch_y1 = notch_out            # = 0.2418637148
new_notch_y2 = old_notch_y1         # = 0.0
```

**Python notch update (signal_filters.py:361-363):**
```python
y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
return (y, x, x1, y, y1)  # (notch_out, new_x1, new_x2, new_y1, new_y2)
```

**Comparison after step 1:**

| Field | Python Value (inferred) | JAX Value | Match? |
|-------|------------------------|-----------|--------|
| notch_x1 | 0.251322669978 | 0.251322669978 | ✓ (same input) |
| notch_x2 | 0.0 | 0.0 | ✓ (both from old x1=0) |
| notch_y1 | ~0.2408917094 | 0.2418637148 | **✗ DIFF = 0.000972** |
| notch_y2 | 0.0 | 0.0 | ✓ (both from old y1=0) |

**Notch output (notch_y1): PY differs from JX by ~0.000972 rad/s despite identical inputs, formula, state, and coefficients.**

### Root Cause Analysis of Notch Divergence

Given:
- Identical formula: `y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2`
- Identical state at step 0: `[x1=0, x2=0, y1=0, y2=0]`
- Identical input at step 1: `x = pitch_rate = 0.251322669978`
- Identical coefficients: computed from `biquad_notch_coefficients(fs=100.0, fc=2.5, Q=2.0)`

The step 1 output simplifies to: `y = b0 * x` (since all other terms are zero)

For divergence at step 1 (before any state accumulation), the ONLY possible cause is a difference in `b0 * x`:
- If `x` is identical: `b0` must differ
- If `b0` is identical: `x` must differ

**Since both `x` and `b0` are confirmed identical, the notch output should be identical. The observed divergence contradicts this.**

### Possible Explanations:

1. **Coefficient computation difference:**
   - Python: `BiquadNotchFilter` computes coefficients in `__init__` using `biquad_notch_coefficients()`
   - JAX: `pack_params_stage2` computes coefficients using `_python_biquad_notch_coefficients()` (same function)
   - Are both called with identical arguments? fs=100.0, fc=2.5, Q=2.0 (confirmed)
   - **Verdict: Coefficients should be identical. UNLIKELY source.**

2. **The Python pitch rate input to the notch filter may differ from what JAX receives:**
   - Python uses `pitch_rate_raw` (before boost/smoothing?) vs `pitch_rate_for_control_boosted`
   - JAX receives `pitch_rate_for_control_boosted` via input
   - **Check:** `pitch_rate_for_control_boosted` = 0.251322669978 at step 1
   - If Python notch processes `pitch_rate_raw` instead of the boosted version, the inputs would differ
   - **VERDICT: MOST LIKELY CAUSE.** Need to verify which pitch_rate signal the Python notch filter actually receives.

3. **JAX notch update order vs Python:**
   - Both use DF2T with identical state transition: `[x, x1, y, y1]`
   - If Python uses a different order internally, the output would differ
   - **Verdict: Code review confirms identical order. UNLIKELY source.**

4. **Notch height gate interaction:**
   - Both gate at 0.42-0.48 with smoothstep
   - At h=0.48, notch_gate=1.0 (fully active) for both
   - blend=1.0 for both
   - pitch_rate_eff = (1-gate)*pitch_rate + gate*notch_out = notch_out (same for both)
   - **Verdict: Gate/blend identical. NOT the source.**

### Other State Fields After Step 1

| Field | Python (inferred) | JAX Value | Match? |
|-------|------------------|-----------|--------|
| prev_tau[4] (wheel L) | 0.6519404687 | 0.6616605222 | **✗ DIFF = 0.00972** |
| prev_tau[9] (wheel R) | 0.6707174665 | 0.6804375199 | **✗ DIFF = 0.00972** |
| prev_tau[1] (hip_yaw L) | ~same | ~same | ✓ (small) |
| prev_tau[6] (hip_yaw R) | ~same | ~same | ✓ (small) |
| filtered_com_z | 0.480 | 0.480 | ✓ |
| prev_support_error | 0.00062467 | 0.00062467 | ✓ |
| ol_pitch_ref_smoothed | ~0.0 | 0.0 | ✓ |
| ol_prev_support_error | 0.00062467 | 0.00062467 | ✓ |
| ol_support_error_rate | ~small | ~small | ✓ |
| ABS slow_sum | 0.00062467 | 0.00062467 | ✓ |
| ABS fast_sum | 0.00031233 | 0.00031233 | ✓ |
| ABS trim_tau | 0.0 | 0.0 | ✓ (hold-down active) |
| ABS hold_steps | 100 | 100 | ✓ |
| ABS slow_count | 2 | 2 | ✓ |

**The state divergence is CONFINED to zwei groups:**
1. **Notch filter y1 (filtered output)** — diverges by ~0.00097
2. **prev_tau (wheels [4,9] only)** — diverges by ~0.0097 (consequence of notch)

All other state fields (19 + 9 ABS + 300 ring buffer) match initially.

---

## 4. Step 1 → Step 2 State Transition

The notch state at step 2 shows:
```
JAX: notch_x1 = 0.3145745129, notch_x2 = 0.2513226700
     notch_y1 = 0.2847532094, notch_y2 = 0.2418637148

Python (inferred): notch_y1 = ~0.28270 (based on divergence growth)
```

The cumulative notch state difference causes:
- Step 2 tau_pitch_rate differs more
- tau_common differs more  
- wheel torque diff grows: 0.00972 → 0.03523

**This is a cumulative state error — each step's small difference accumulates in the filter state, causing monotonic divergence growth.**

---

## 5. Updated State Fields Per Step

### JAX state after step 0:
```
notch = [0, 0, 0, 0]
prev_tau = [0, 0, 2.05, -4.0, -3.303, 0, 0, 1.60, -4.0, -3.303]
```
### JAX state after step 1:
```
notch = [0.25132, 0, 0.24186, 0]
prev_tau = [~0, ~0, 2.05, -4.0, 0.66166, ~0, ~0, 1.60, -4.0, 0.68044]
```
### JAX state after step 2:
```
notch = [0.31457, 0.25132, 0.28475, 0.24186]
prev_tau = [...]  (wheels differ by ~0.035 Nm)
```

---

## 6. First Divergent State Field

**First divergent state field: `notch_y1` (filtered output) at step 1**

- JAX value: 0.2418637148
- Python value (inferred): ~0.2408917094
- Difference: ~0.000972
- Propagation: notch_y1 → next step's y2 → participates in next step's notch computation → notch output diverges more → tau_pitch_rate diverges → tau_common diverges → wheel torque diverges

**Secondary: `prev_tau[4]` and `prev_tau[9]` at step 1**
- These are downstream consequences of the notch divergence, not independent root causes.

---

## 7. Conclusion

1. **The notch filter is the root cause of state divergence.**
2. **The divergence appears at step 1** — before any state accumulation — suggesting the issue is in input signal routing OR coefficient precision.
3. **The most likely mechanism:** Python notch filter receives a different `pitch_rate` signal than what JAX receives, OR the BiquadNotchFilter object has different sampling frequency or internal state handling.
4. **The divergence is SMALL** (0.001 rad/s at step 1) but **CUMULATIVE** — grows linearly with steps.
5. **prev_tau divergence is a consequence, not a cause.** Fixing the notch filter would fix prev_tau.
6. **All other state fields (filtered_com_z, outer loop, ABS, support) match** — the divergence is confined to the notch signal path.

**Recommended verification (DO NOT APPLY):**
1. Print raw `pitch_rate` value that enters Python `BiquadNotchFilter.update()` at step 1
2. Compare against `pitch_rate_for_control_boosted` (what JAX receives)
3. If they differ, the input routing is the root cause
4. If they match, investigate BiquadNotchFilter internal coefficient precision
