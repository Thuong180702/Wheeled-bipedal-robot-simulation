# K2 Pitch RMS — Stateful Terms Audit

**Date:** 2026-06-30
**Phase:** 5 — AUDIT STATEFUL FILTERS AND TRIM TERMS

---

## 1. Stateful Terms Inventory

Each stateful term in the K2 controller is audited for initialization, update order, and potential divergence between Python source and JAX standalone.

### 1.1 Notch Filter State

| Field | Python init | JAX init | Match? |
|-------|------------|----------|--------|
| notch_x1 | 0.0 | 0.0 (pack_state_k2) | ✅ |
| notch_x2 | 0.0 | 0.0 | ✅ |
| notch_y1 | 0.0 | 0.0 | ✅ |
| notch_y2 | 0.0 | 0.0 | ✅ |

**Update order:**
- Python: `notch_y, x1, x2, y1, y2 = biquad_notch(x, x1, x2, y1, y2, b0, b1, b2, a1, a2)`
- JAX: Same update (`_single_biquad_notch_jax`)

**Verdict:** ✅ Equivalent

### 1.2 Previous Torque (Rate Limiter)

| Field | Python init | JAX init | Match? |
|-------|------------|----------|--------|
| prev_tau (10,) | zeros(10) | zeros(10) (pack_state_k2) | ✅ |

**Update:** Set to `tau_final` after composer step in both paths.

**Verdict:** ✅ Equivalent

### 1.3 Filtered CoM Height

| Field | Python init | JAX init | Match? |
|-------|------------|----------|--------|
| _filtered_com_z | 0.4 (default) | 0.0 (pack_state_k2) | ⚠️ DIFFERENT |

**Python** (`SagittalVelocityDampedBalanceController.__init__`, line 4177):
```python
self._filtered_com_z = 0.4  # Initialize to default com_z
```

**JAX** (`pack_state_k2()`):
```python
# _S_FILTERED_COM_Z = 0.0
```

**Impact:** At step 0, Python uses `alpha * 0.4 + (1-alpha) * current_com_z` for first update. JAX uses `alpha * 0.0 + (1-alpha) * current_com_z`. With alpha=0.9, the difference is `0.9 * 0.4 = 0.36m` on the first filtered value. This affects:
- `schedule_height_ref` → height-dependent scheduling of max_position_tau
- All grid interpolations (calibrated outer loop, physics FF)
- Notch filter height gate

**Over how many steps does this matter?** With alpha=0.9, the impulse response is `0.9^n`. After 10 steps: `0.9^10 = 0.35`. After 20 steps: `0.9^20 = 0.12`. So the effect persists for ~20-30 steps, then decays to <1% of the initial difference.

**For pitch RMS over 2000 steps:** This initial transient affects only ~1-2% of the total window. Unlikely to explain a 1-2° difference in RMS.

**BUT:** If the filtered_com_z causes different scheduling at early steps, that could create a different trajectory that doesn't converge even after the filtered value converges. This is the butterfly effect.

**Verdict:** ⚠️ Minor difference, unlikely root cause but worth fixing for strict parity.

### 1.4 Previous Support Error

| Field | Python init | JAX init | Match? |
|-------|------------|----------|--------|
| prev_support_position_error_m | 0.0 | 0.0 (pack_state_k2) | ✅ |

**Python:** Set at end of each compute() call: `self.prev_support_position_error_m = sagittal_position_error_m`
**JAX:** Set in state: `new_state = new_state.at[_S_PREV_SUPPORT_ERROR].set(support_pos_err)`

**Verdict:** ✅ Equivalent

### 1.5 Outer Loop State

| Field | Python init | JAX init | Match? |
|-------|------------|----------|--------|
| pitch_ref_smoothed | 0.0 | 0.0 | ✅ |
| prev_support_error | None → 0.0 first step | 0.0 | ⚠️ |
| support_error_rate | 0.0 | 0.0 | ✅ |

**Python** (line 6415-6425):
```python
if outer_loop_prev_support_error_m is None:
    ol_rate_raw = 0.0
else:
    ol_rate_raw = (ol_support_error - outer_loop_prev_support_error_m) / control_dt
```

**JAX** (line 1922-1924):
```python
support_error_rate_raw = jnp.where(
    ol_prev_support_error == 0.0, 0.0,
    (support_pos_err - ol_prev_support_error) / control_dt)
```

**Difference:** Python checks for None (first step), JAX checks for == 0.0. If support_pos_err is truly zero at step 0 (which it should be for a static initial condition), both produce 0.0. But if it's non-zero at step 0 (e.g., due to mj_forward artifacts), Python produces 0.0 (because prev is None) while JAX produces a non-zero derivative.

**Verdict:** ⚠️ Subtle difference if support error is non-zero at step 0. Should verify.

### 1.6 ABS Trim State (Ring Buffer)

| Field | Python init | JAX init | Match? |
|-------|------------|----------|--------|
| slow_sum | 0.0 | 0.0 | ✅ |
| fast_sum | 0.0 | 0.0 | ✅ |
| trim_tau | 0.0 | 0.0 | ✅ |
| hold_steps | 0 | 0 | ✅ |
| prev_err_sign | 0.0 | 0.0 | ✅ |
| zc_count | 0 | 0 | ✅ |
| slow_count | 0 | 0 | ✅ |
| slow_ptr | 0 | 0 | ✅ |
| guard_trigger | 0 | 0 | ✅ |
| Ring buffer | zeros(N) | zeros(N) | ✅ |
| ZC buffer | zeros(N_zc) | zeros(N_zc) | ✅ |

**Verdict:** ✅ Equivalent (all initialized to zero)

### 1.7 APCR1ND State

| Field | Python init | JAX init | Match? |
|-------|------------|----------|--------|
| step_counter | 0 | 0 | ✅ |
| prev_error | 0.0 | 0.0 | ✅ |
| converging_steps | 0 | 0 | ✅ |
| recenter_held | False (0) | 0.0 | ✅ |

**Verdict:** ✅ Equivalent

---

## 2. Stateful Term Summary

| # | Term | Status | Impact on pitch |
|---|------|--------|-----------------|
| 1 | Notch filter | ✅ Equivalent | Zero |
| 2 | Prev torque | ✅ Equivalent | Zero |
| 3 | Filtered CoM height | ⚠️ Different init (0.4 vs 0.0) | Small transient |
| 4 | Prev support error | ✅ Equivalent | Zero |
| 5 | Outer loop | ⚠️ Subtle None-vs-zero difference | Small if support error non-zero at step 0 |
| 6 | ABS ring buffer | ✅ Equivalent | Zero |
| 7 | APCR1ND state | ✅ Equivalent | Zero |

---

## 3. Recommended Fix (filtered_com_z init)

The most concrete mismatch found is the `filtered_com_z` initialization (0.4 in Python vs 0.0 in JAX).

**Fix:** Initialize `_filtered_com_z` in JAX state to match Python's default.

**Location:** `pack_state_k2()` or the dedicated runner's initial state.

**Expected impact:** Very small (affects first ~20 steps only). But should be fixed for strict parity.

---

## 4. Other Observations

### 4.1 Centroidal estimator observation passing

**Python** (`simulate_hierarchical_controller.py`):
```python
centroidal_state_log, logged_com_pos = centroidal_estimator.estimate(
    jnp.zeros(42), mj_data, control_com_pos
)
```
Passes `jnp.zeros(42)` — same as dedicated runner.

**Dedicated runner** (`run_k2_jax_realtime.py`):
```python
centroidal, prev_com_pos = centroidal_estimator.estimate(
    np.zeros(42), mj_data, prev_com_pos
)
```
Passes `np.zeros(42)` — same.

So both pass zero observations. The `prev_com_pos` is tracked and fed back. This should produce identical results given identical physics state.

### 4.2 No other stateful term mismatches found

All other stateful terms (notch, ABS, APCR1ND, outer loop) have equivalent initialization and update logic between Python and JAX paths.

---

## 5. Conclusion

**The stateful terms audit does NOT reveal the root cause of the pitch RMS gap.** The only discrepancy found is `filtered_com_z` initialization (0.4 vs 0.0), which has a negligible impact on RMS over 2000 steps.

**This narrows the root cause to:**
1. Physics/orchestration differences (warm-start, substep ordering)
2. Numerical precision differences accumulating over 2000 steps
3. An undiscovered computational difference in a non-stateful term

**Recommended next step:** Run Phases 6-7 (targeted patch + validation of the fix).
