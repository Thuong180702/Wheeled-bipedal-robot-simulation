# K2 JAX — ABS Trim Gap Matrix (Phase 2)

**Date:** 2026-06-28
**Based on:** Phase 0 baseline + Phase 1 Python source trace

---

## Gap Matrix: Python vs JAX

| # | Python scalar | JAX scalar | Match? | Gap type | Torque impact | Fix required |
|---|--------------|-----------|--------|----------|--------------|-------------|
| 1 | `adaptive_bias_trim_enabled` (True) | `_sch.adaptive_bias_trim_enabled` (True from profile) | ✅ PASS | — | — | — |
| 2 | `adaptive_bias_trim_replace_t6j` (True) | Not checked (T6J block already disabled for K2) | ✅ PASS | — | — | — |
| 3 | `adaptive_bias_window_steps` (300) | `_ABS_SLOW_WINDOW` (300) | ✅ PASS | — | — | — |
| 4 | `adaptive_bias_fast_window_steps` (100) | `_ABS_FAST_WINDOW` (100) | ✅ PASS | — | — | — |
| 5 | `adaptive_bias_enter_threshold_m` (0.035) | `_sch.adaptive_bias_enter_threshold_m` | ✅ PASS | — | — | — |
| 6 | `adaptive_bias_exit_threshold_m` (0.012) | `_sch.adaptive_bias_exit_threshold_m` | ✅ PASS | — | — | — |
| 7 | `adaptive_bias_relief_hysteresis_m` (0.005) | `_sch.adaptive_bias_relief_hysteresis_m` | ✅ PASS | — | — | — |
| 8 | `adaptive_bias_k_tau_per_m` (5.0) | `_sch.adaptive_bias_k_tau_per_m` | ✅ PASS | — | — | — |
| 9 | `adaptive_bias_max_tau_low_nm` (0.35) | `_sch.adaptive_bias_max_tau_low_nm` | ✅ PASS | — | — | — |
| 10 | `adaptive_bias_max_tau_high_nm` (0.50) | `_sch.adaptive_bias_max_tau_high_nm` | ✅ PASS | — | — | — |
| 11 | `adaptive_bias_max_tau_extreme_nm` (0.55) | `_sch.adaptive_bias_max_tau_extreme_nm` | ✅ PASS | — | — | — |
| 12 | `adaptive_bias_height_low_m` (0.38) | `_sch.adaptive_bias_height_low_m` | ✅ PASS | — | — | — |
| 13 | `adaptive_bias_height_high_m` (0.48) | `_sch.adaptive_bias_height_high_m` | ✅ PASS | — | — | — |
| 14 | `adaptive_bias_height_extreme_m` (0.52) | `_sch.adaptive_bias_height_extreme_m` | ✅ PASS | — | — | — |
| 15 | `adaptive_bias_rate_nm_per_step` (0.006) | `_sch.adaptive_bias_rate_nm_per_step` | ✅ PASS | — | — | — |
| 16 | `adaptive_bias_fast_rate_nm_per_step` (0.012) | Not used | ✅ PASS | DIAGNOSTIC_ONLY | 0 — unused in both | — |
| 17 | `adaptive_bias_decay_rate_nm_per_step` (0.018) | `_sch.adaptive_bias_decay_rate_nm_per_step` | ✅ PASS | — | — | — |
| 18 | `adaptive_bias_only_when_upright` (True) | `_sch.adaptive_bias_only_when_upright` | ✅ PASS | — | — | — |
| 19 | `adaptive_bias_only_when_contact_stable` (True) | JAX hardcodes `_contact_ok = True` (line 1724) | ❌ **WRONG** | WRONG_SAFETY_GATE | **~0.98 Nm push_fwd** | Fix contact_ok to use actual contact_valid |
| 20 | `adaptive_bias_disable_if_pitch_gt_deg` (12.0) | `_sch.adaptive_bias_disable_if_pitch_gt_deg` | ✅ PASS | — | — | — |
| 21 | `adaptive_bias_disable_if_roll_gt_deg` (5.0) | `_sch.adaptive_bias_disable_if_roll_gt_deg` | ✅ PASS | — | — | — |
| 22 | `adaptive_bias_disable_if_abs_error_gt_m` (0.24) | `_sch.adaptive_bias_disable_if_abs_error_gt_m` | ✅ PASS | — | — | — |
| 23 | `adaptive_bias_disable_if_hip_yaw_gt_rad` (0.25) | `_sch.adaptive_bias_disable_if_hip_yaw_gt_rad` | ✅ PASS | — | — | — |
| 24 | `adaptive_bias_zero_crossing_guard_enabled` (True) | `_sch.adaptive_bias_zero_crossing_guard_enabled` | ✅ PASS | — | — | — |
| 25 | `adaptive_bias_zero_crossing_window_steps` (500) | JAX limited to ring buffer (300) | ❌ **WRONG** | WRONG_ZC_WINDOW | **~0.16-1.51 Nm ramp/gate** | Expand ZC window to 500 or add separate ZC buffer |
| 26 | `adaptive_bias_zero_crossing_limit` (8) | `_sch.adaptive_bias_zero_crossing_limit` | ✅ PASS | — | — | — |
| 27 | `adaptive_bias_zero_crossing_max_scale` (0.5) | `_sch.adaptive_bias_zero_crossing_max_scale` | ✅ PASS | — | — | — |
| 28 | `adaptive_bias_sign_reversal_hold_steps` (100) | `_sch.adaptive_bias_sign_reversal_hold_steps` | ✅ PASS | — | — | — |
| 29 | `_adaptive_bias_trim_tau` | `_ABS_TRIM_TAU` | ✅ PASS | — | — | — |
| 30 | `_adaptive_bias_trim_target_tau` | Not tracked | ⚠️ MISSING | DIAGNOSTIC_ONLY | 0 — diagnostic | Add diagnostic field |
| 31 | `_adaptive_bias_slow_error_history` (list, max 300) | Ring buffer (300, oldest→ptr→wrap) | ✅ PASS | — | — | — |
| 32 | `_adaptive_bias_fast_error_history` (list, max 100) | Derived from ring buffer (most recent 100) | ✅ PASS | — | — | — |
| 33 | `_adaptive_bias_zero_crossing_history` (list, max 500) | Derived from ring buffer (max 300) | ❌ **WRONG** | WRONG_ZC_WINDOW | **~0.16-1.51 Nm** | See #25 |
| 34 | `_adaptive_bias_guard_trigger_count` (cycling: +=1, >=3→0, else→0) | JAX: +=1 always, =0 on else (line 2155) | ❌ **WRONG** | WRONG_GUARD_RESET | Diagnostic only | Fix guard_trigger ≥3 reset |
| 35 | `_adaptive_bias_prev_error_sign` (updated in elif hold>0 too) | JAX: only updated on err_sign_changed (line 2171) | ⚠️ **PARTIAL** | WRONG_PREV_SIGN_UPDATE | Minimal (<0.01 Nm) | Fix prev_sign update when hold>0 |
| 36 | `_adaptive_bias_hold_steps` | `_ABS_HOLD_STEPS` | ✅ PASS | — | — | — |
| 37 | `contact_valid` (Python: from MuJoCo) | JAX: hardcoded True | ❌ **WRONG** | MISSING_INPUT | **~0.98 Nm** | Add contact_valid to JAX inputs |
| 38 | `hip_yaw_abs_max_tracking` (Python: running max) | JAX: `max(|q_hy_l - qref_hy_l|, |q_hy_r - qref_hy_r|)` (instantaneous) | ⚠️ **PARTIAL** | WRONG_SIGNAL_SOURCE | Varies | Use running max or add tracking state |
| 39 | Height scheduling formula | JAX `t_h` interpolation + `jnp.where` | ✅ PASS | — | — | — |
| 40 | Hysteresis/proportional target formula | JAX `jnp.where` chain | ✅ PASS | — | — | — |
| 41 | Asymmetric rate limiting | JAX `jnp.where(is_decay, ...)` | ✅ PASS | — | — | — |
| 42 | Safety gate composition | JAX `_contact_ok & _upright_ok & _abs_error_ok & _hip_yaw_ok` | ❌ see #19,#37 | — | — | Fix #19 |
| 43 | Trim applied to tau_position | JAX `external_position_trim` in sagittal assembly | ✅ PASS | — | — | — |

---

## Gap Summary

| Status | Count | Items |
|--------|-------|-------|
| ✅ PASS | 36 | Formula, params, ring buffer, rate limiting all match |
| ⚠️ PARTIAL | 3 | #30 (diagnostic), #35 (prev_sign edge case), #38 (hip_yaw source) |
| ❌ WRONG | 4 | #19 (contact gate), #25 (ZC window), #33 (ZC buffer), #34 (guard reset), #37 (contact_valid input) |
| UNUSED | 1 | #16 (fast_rate) |

## Root Cause Analysis

### Gap #19/#37: Contact Safety Gate — Explains push_fwd 0.98 Nm

**Python (line 5660):**
```python
contact_ok = (not sch.adaptive_bias_only_when_contact_stable) or bool(contact_valid)
```
With `only_when_contact_stable=True`: `contact_ok = contact_valid`

**JAX (line 1724):**
```python
_contact_ok = True  # contact_valid always True from JAX perspective
```

During push (90N forward at step 100), the robot's wheels may momentarily lose contact with the ground. Python detects this via MuJoCo contact forces and sets `contact_valid = False`, which disables ABS trim via `safety_pass = False`. JAX hardcodes `True` and continues applying trim. The trim at this point is at or near max_tau (~0.50 Nm at 0.48m height), which directly adds ~0.50 Nm to tau_position, which flows to wheel torque. Over several steps of push recovery, this accumulates to the observed ~0.98 Nm diff.

**Fix:** Add `contact_valid` to `K2_JAX_INPUT_FIELDS`, pass it from Python in both-synced mode, and use it in the safety gate.

### Gap #25/#33: ZC Window Size — Explains ramp_up/gate_chatter degradation

**Python:** `adaptive_bias_zero_crossing_window_steps = 500`. ZC history is a separate list that grows to 500 entries.

**JAX:** ZC count is derived from the ring buffer, which holds at most 300 entries.

After 300+ steps, Python counts sign changes over 500 entries while JAX counts over 300. With more entries, Python's ZC count can be higher, triggering the ZC guard more often. When the guard activates (`zc_count > 8`), `guard_scale = 0.5`, halving `max_tau_g`. This difference cascades:
- Different max_tau → different clip ceiling
- Different clip ceiling → different trim_tau
- Different trim_tau → different tau_position → different wheel torque

The ramp_up scenario crosses height scheduling gates (0.33→0.48m), which changes kpos and other gains, amplifying the ZC mismatch. Gate_chatter oscillates through the gate repeatedly, causing maximum divergence.

**Fix:** Add a separate ZC ring buffer of 500 entries, or expand the existing buffer to 500 entries.

### Gap #34: Guard Trigger Reset — Diagnostic Only

JAX increments `guard_trigger` without the ≥3 reset. This affects ONLY the diagnostic value — `guard_scale` is computed from `zc_guard_active`, which is correct.

### Gap #35: Prev Sign Update — Minimal Impact

Python updates `prev_error_sign` in the `elif hold_steps > 0` branch. JAX only updates on `err_sign_changed`. This affects when `err_sign_changed` fires after hold expiration, but the impact is minimal because:
1. During hold, sign is typically stable
2. Hold expiration means trim is already decaying to zero

---

## Torque Impact Estimates

| Scenario | Gap | Estimated torque impact | Mechanism |
|----------|-----|------------------------|-----------|
| push_fwd_90N | #19 (contact gate) | ~0.50 Nm per step × ~2-3 steps of contact loss ≈ **0.98-1.50 Nm** | trim applied when it shouldn't be |
| ramp_up | #25 (ZC window) | ~0.05-0.25 Nm accumulating over 350 steps post-gate ≈ **0.16 Nm** | different guard activation → different max_tau |
| gate_chatter | #25 (ZC window) | ~0.25-0.55 Nm amplified by repeated gate crossing ≈ **1.51 Nm** | repeated ZC mismatch at each gate crossing |
| push_bwd_90N | #19 (contact gate) | Similar to push_fwd but backward push may break contact differently | Robot falls (both fail similarly) |

---

## Fix Priority

1. **HIGH — Fix #19/#37 (contact_valid input + contact_ok gate)**: Directly causes push_fwd 0.98 Nm diff. Requires adding 1 bool field to inputs and 1 line change in safety gate.

2. **HIGH — Fix #25/#33 (ZC window 300→500)**: Causes ramp_up and gate_chatter diffs. Requires expanding ring buffer or adding separate ZC buffer. Needs careful state layout change.

3. **LOW — Fix #34 (guard_trigger reset)**: Diagnostic only. Fix for completeness.

4. **LOW — Fix #35 (prev_sign update)**: Minimal control impact. Fix for exact parity.

5. **MEDIUM — Fix #38 (hip_yaw tracking)**: Could cause safety gate divergence in scenarios with large hip yaw transients, but not observed in current baseline.
