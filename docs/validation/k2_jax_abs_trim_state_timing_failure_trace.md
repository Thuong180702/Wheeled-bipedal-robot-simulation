# K2 JAX ABS Trim State/Timing Failure Trace — Phase 0

**Date:** 2026-06-28
**Branch:** `repo-cleanup-t6j`
**Trace script:** `scripts/phase0_abs_trim_trace.py` (via subprocess of simulate_hierarchical_controller.py)

---

## Executive Summary

**Finding: The ABS trim state/timing is NOT divergent. The prior diagnosis was based on a diagnostic bug.**

The previous classification that `_ABS_TRIM_TAU` diverges from `_adaptive_bias_trim_tau` was incorrect. The diagnostic code at `simulate_hierarchical_controller.py:6739` was reading from `_jax_state` (the incremental JAX state, initialized to zeros at simulation start) instead of `_jax_state_synced` (the correctly packed state used for the both-synced computation). When reading from the correct variable, all ABS trim values — state packing, computation intermediates, and final trim — match Python exactly to machine precision.

The actual torque divergence at wheel actuators [4,9] (0.028 Nm at step 140, growing) has a **different root cause** not in ABS trim.

---

## Reproduction Setup

- **Scenario:** ramp_up (0.330→0.480 m trajectory, 200 simulation steps)
- **Controller:** K2_NOTCH_LOW_Q_V1, both-synced mode
- **Command:**
  ```bash
  python scripts/simulate_hierarchical_controller.py \
    --controller-mode balance-core \
    --sagittal-controller velocity-damped \
    --vd-sagittal-authority-profile k2_notch_low_q_v1 \
    --controller-backend both-synced \
    --steps 200 \
    --height-variant-setup outputs/.../low_0p330_setup.json \
    --dynamic-height-trajectory outputs/.../ramp_up_0p330_to_0p480.json
  ```

---

## Trace Results

### A. ABS State Values (Step 140)

| Field | Python (pre-compute) | JAX (pre-compute, synced) | Match? |
|-------|---------------------|--------------------------|--------|
| `_adaptive_bias_trim_tau` / `_ABS_TRIM_TAU` | -0.0540000000 | -0.0540000000 | ✓ |
| `_adaptive_bias_hold_steps` | 72 | 72 | ✓ |
| `_adaptive_bias_prev_error_sign` | 1 | 1 | ✓ |
| `_adaptive_bias_crossing_count` | 0 | 0 | ✓ |
| `_adaptive_bias_guard_trigger_count` | 0 | 0 | ✓ |
| Slow history length | 140 | 140 (count) | ✓ |
| Slow sum | 3.386754 | 3.386754 | ✓ |
| ZC buffer count | N/A | 140 | ✓ |

**Verdict: STATE PACKING IS CORRECT. All pre-compute ABS state values match Python exactly.**

### B. ABS Compute Intermediates (Step 140)

| Intermediate | Python (trace) | JAX (diag) | Match? |
|-------------|---------------|-----------|--------|
| `signed_error` | 1.3695093489e-01 | 1.3695093489e-01 | ✓ |
| `mean_err` / `slow_mean` | 2.4990814973e-02 | 2.4990814973e-02 | ✓ |
| `fast_mean_err` | 4.4069329968e-02 | 4.4069329968e-02 | ✓ |
| `sign_err` | 1.0 | 1.0 | ✓ |
| `max_tau_current` | 3.5000000000e-01 | N/A | — |
| `max_tau_g` (guarded) | 3.5000000000e-01 | N/A | — |
| `guard_scale` | 1.0000 | N/A | — |
| `raw_target` | -6.4954074864e-02 | -6.4954074864e-02 | ✓ |
| `clipped_target` | -6.4954074864e-02 | -6.4954074864e-02 | ✓ |
| `is_decay` | False | 0.0 | ✓ |
| `rate_used` | 6.0000000000e-03 | 6.0000000000e-03 | ✓ |
| `trim_delta` | -6.0000000000e-03 | -6.0000000000e-03 | ✓ |
| `new_trim` / updated trim | -6.0000000000e-02 | -6.0000000000e-02 | ✓ |
| `safety_pass` | True | 1.0 | ✓ |
| `trim_to_apply` | -6.0000000000e-02 | -6.0000000000e-02 | ✓ |
| `hold_steps` (post) | 71 | 71.0 | ✓ |
| `err_sign_changed` | False | — | — |
| `sign_rev_blocked` | False | — | — |
| `near_zero` | False | — | — |
| `in_hysteresis` | False | — | — |
| `zc_guard_active` | False | — | — |

**Verdict: COMPUTATION IS CORRECT. All 12 ABS intermediates exported via the JAX diag match Python exactly. No first divergent scalar exists in the ABS trim subsystem.**

### C. Position Torque Path (Step 140)

| Field | Python | JAX | Diff |
|-------|--------|-----|------|
| `tau_position` (pre-trim, Python) vs `tau_position` (post-trim, JAX) | -5.478037 | -5.538037 | 0.060000 |
| ABS trim to apply | -0.060000 | -0.060000 | 0.0 — MATCH ✓ |
| Effective max position tau | N/A | N/A | — |
| `tau_wheel[4]` (final) | 0.261997 | 0.233879 | **0.028118** |
| `tau_wheel[9]` (final) | 0.264721 | 0.236603 | **0.028118** |

**Key observation:** The `tau_position` difference of 0.060 equals the ABS trim value exactly. This is expected because Python's `tau_position_before_trim` is captured BEFORE trim addition, while JAX's `tau_position` (from diag) is AFTER trim addition (includes `external_position_trim`). The field names represent different points in the computation pipeline.

The final wheel torque difference (0.028 Nm) does NOT match the trim value (0.060 Nm), confirming the torque divergence has a separate root cause.

### D. State Timing

| Timing Event | Status |
|-------------|--------|
| Pre-Python-compute capture (`_both_synced_capture`) | ✓ Captured at lines 5932-5965 |
| `abs_trim_tau` captured from `_sag._adaptive_bias_trim_tau` | ✓ Pre-compute value |
| `pack_state_from_python_k2(abs_trim_tau=...)` receives capture value | ✓ Verified via `_jax_state_synced[21]` read |
| JAX reads `_ABS_TRIM_TAU` at index 21 | ✓ Correct |
| JAX writes updated trim to index 21 | ✓ Verified via `_jax_new_state[21]` read (`jx_trim_post`) |
| No post-python-compute mutation of captured state | ✓ State captured before `compute()` |

**Verdict: CAPTURE TIMING IS CORRECT for ABS trim.**

---

## Diagnostic Bug Analysis

### Bug Location

**File:** `scripts/simulate_hierarchical_controller.py`
**Lines:** 6736-6739 (original, before Phase 0 fix)

```python
_jx_zc = float(_jax_state[24]) if _jax_state.shape[0] > 24 else 0.0      # BUG
_jx_guard = float(_jax_state[27]) if _jax_state.shape[0] > 27 else 0.0    # BUG
_jx_zc_buf_count = float(_jax_state[328]) if _jax_state.shape[0] > 328 else 0.0  # BUG
_jx_abs_trim = float(_jax_state[21]) if _jax_state.shape[0] > 21 else 0.0  # BUG
```

### Root Cause

`_jax_state` is initialized ONCE at simulation start via `pack_state_k2()` (line 5336), which creates an all-zeros state array of shape (834,). In both-synced mode, this variable is NEVER updated — the packed state used for computation is `_jax_state_synced` (created fresh each step from the Python snapshot).

Reading `_jax_state[21]` always returns 0.0 because it's the zero-initialized array. **This caused the false diagnosis that JAX ABS trim = 0.0 while Python had nonzero trim.**

### Fix

Changed lines 6736-6739 to read from `_jax_state_synced` (the correctly packed state) instead of `_jax_state`:

```python
_jx_zc = float(_jax_state_synced[24]) if _jax_state_synced.shape[0] > 24 else 0.0
_jx_guard = float(_jax_state_synced[27]) if _jax_state_synced.shape[0] > 27 else 0.0
_jx_zc_buf_count = float(_jax_state_synced[328]) if _jax_state_synced.shape[0] > 328 else 0.0
_jx_abs_trim = float(_jax_state_synced[21]) if _jax_state_synced.shape[0] > 21 else 0.0
```

Also added `_jax_new_state` read for post-compute trim:
```python
_jx_abs_trim_post = float(_jax_new_state[21]) if _jax_new_state.shape[0] > 21 else 0.0
```

### Secondary Issue: JAX Diag Missing ABS Fields

The JAX diag (`K2_JAX_DIAG_FIELDS`) previously lacked ABS intermediate fields. Diagnostics reading `_diag_d.get('abs_slow_mean', 0)` always returned default 0.0. Added 12 ABS intermediate fields (indices 32-43) to the diag for proper comparison.

---

## Ruling Out ABS Trim Divergence Causes

| Cause | Investigation | Result |
|-------|--------------|--------|
| 1. Wrong state capture timing | Verified capture before Python compute, packing with correct value | **RULED OUT** — capture timing is correct |
| 2. Wrong state packing index | Verified `_ABS_TRIM_TAU = 21`, field name `abs_trim_tau`, pack writes to index 21 | **RULED OUT** — index is correct |
| 3. Wrong JAX state read index | Verified `state_flat[_ABS_TRIM_TAU]` reads index 21 in `_k2_jax_adaptive_bias_trim` | **RULED OUT** — read index is correct |
| 4. Wrong JAX update timing | Verified trim update before `trim_to_apply` computation | **RULED OUT** — update order matches Python |
| 5. Wrong formula | Step-by-step comparison of all 12 intermediates — all match | **RULED OUT** — formula is correct |
| 6. Wrong safety gate | `_safety` gate matches Python's `safety_pass` (both True at step 140) | **RULED OUT** — safety gate matches |
| 7. Wrong ring chronological order | Slow buffer sum (3.386754) and count (140) match Python history | **RULED OUT** — chronology is correct |
| 8. Wrong reset/default | Both start from same zero state | **RULED OUT** — reset/default matches |

---

## Actual Torque Divergence Source (Preliminary)

At step 140, the wheel torque [4,9] differs by 0.028 Nm (symmetric left/right). This grows linearly (0.028→0.057→0.083→0.104 over steps 140-143).

All verified matching components:
- ✅ tau_pitch: identical
- ✅ tau_pitch_rate: identical
- ✅ tau_sagittal_velocity: identical
- ✅ tau_support_velocity: identical
- ✅ tau_wheel_velocity_left: identical
- ✅ tau_wheel_velocity_right: identical
- ✅ tau_position components (pitch position, integral): same pre-trim
- ✅ tau_cp: 0.0 for both (kp_cp=0)
- ✅ MODE_DIV: identical
- ✅ ABS trim: identical

Unverified components (candidates for the divergence):
- `tau_com_vy` — not exported in either diag for direct comparison. Python uses `kd_com_vy=5.0`. JAX passes `kd_com_vy=5.0` hard-coded. Needs verification.
- APCR1ND position cap boost effects on composer
- Composer clip/rate-limit asymmetry

**This is NOT an ABS trim issue. Root cause investigation continues in Phase 3.**

---

## Acceptance Checklist

| Criterion | Status |
|-----------|--------|
| Identified whether divergence from wrong state capture timing | ✓ RULED OUT — timing is correct |
| Identified whether divergence from wrong state packing index | ✓ RULED OUT — index 21 is correct |
| Identified whether divergence from wrong JAX state read index | ✓ RULED OUT — reads index 21 |
| Identified whether divergence from wrong JAX update timing | ✓ RULED OUT — order matches |
| Identified whether divergence from wrong formula | ✓ RULED OUT — all intermediates match |
| Identified whether divergence from wrong safety gate | ✓ RULED OUT — gates match |
| Identified whether divergence from wrong ring chronological order | ✓ RULED OUT — buffer contents match |
| Identified whether divergence from wrong reset/default | ✓ RULED OUT — initialization matches |
| First divergent scalar identified | N/A — ABS trim does NOT diverge |
| No code changes made before diagnostic validated | ✓ — only diagnostic fixes, no control logic changes |

---

## Key Files Modified for Diagnostic Tracing

1. `scripts/simulate_hierarchical_controller.py`:
   - Fixed `_jax_state[21]` → `_jax_state_synced[21]` (lines 6736-6739)
   - Added `_jax_new_state` read for post-compute trim
   - Added `_jx_slow_sum_pre`, `_jx_slow_count_pre`, `_jx_slow_ptr_pre` reads
   - Added ABS_TRACE diagnostic block (lines 6744-6762)

2. `wheeled_biped/controllers/k2_jax_controller.py`:
   - Added 12 ABS intermediates to `K2_JAX_DIAG_FIELDS` (indices 32-43)
   - Modified `_k2_jax_adaptive_bias_trim` to return `abs_diag` array
   - Added diag writes for all ABS intermediates

3. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`:
   - Added `_py_abs_trim_trace` dict population after ABS trim computation (after line 5764)

---

## Next Phase

**Phase 1 — State Layout Audit**: Now that ABS trim is ruled out, audit the complete 834-field state layout for any remaining issues. The actual torque divergence source lies outside ABS trim and must be investigated in Phase 3 (compute timing audit beyond ABS).

**Reclassification needed:** The previous classification `K2_JAX_PORT_INCOMPLETE_WITH_EXACT_BLOCKER` citing ABS trim as the blocker was based on a diagnostic bug. The ABS trim subsystem exhibits PERFECT PARITY. The actual torque divergence at wheels requires root-cause analysis of non-ABS components (tau_com_vy, APCR1ND composer effects, or WBC/torque redistribution).
