# K2 JAX ABS Trim Both-Synced Capture Timing Audit — Phase 2

**Date:** 2026-06-28
**Branch:** `repo-cleanup-t6j`

## Capture Timing Architecture

### Python Side

The both-synced capture happens at `simulate_hierarchical_controller.py:5932-5965`, BEFORE the Python `sagittal_controller.compute()` call:

```
Step n control loop:
  1. _both_synced_capture = snapshot of ALL Python K2 state  ← LINE 5932-5965
     - notch filter state (_x1, _x2, _y1, _y2)                   ← PRE-MUTATION snapshot
     - prev_tau (10-vector)
     - filtered_com_z
     - prev_support_error
     - outer_loop state (pitch_ref, prev_err, rate)
     - ABS trim state (trim_tau, hold, sign, zc, guard, histories)
     - ABS ZC history
     - APCR1ND state (step, prev_err, converging, held)
  
  2. Python sagittal_controller.compute() runs                    ← MUTATES internal state
     - Updates _adaptive_bias_trim_tau (inline)
     - Updates error histories (append)
     - Updates hold_steps, prev_err_sign
     - Applies trim to tau_position
  
  3. JAX state packed from _both_synced_capture                   ← LINE 6598-6625
     pack_state_from_python_k2(
         abs_trim_tau=_py_state_snap["abs_trim_tau"],             ← SAME value as step 1
         ...
     )
  
  4. JAX controller_step(_jax_state_synced, _jax_input, _jax_params)
     - Reads _ABS_TRIM_TAU from packed state (index 21)
     - Computes ABS trim from this state
     - Updates _ABS_TRIM_TAU
  
  5. Compare: Python post-compute output vs JAX post-compute output
```

### Verification

At step 140, ramp_up scenario:
- Python pre-compute `_adaptive_bias_trim_tau` = -0.0540000000 (captured in snapshot)
- JAX pre-compute `_ABS_TRIM_TAU` = -0.0540000000 (read from `_jax_state_synced[21]`)
- **MATCH** — state capture and packing is correct

At step 140, post-compute:
- Python `_adaptive_bias_trim_tau` after compute = -0.0600000000
- JAX `_ABS_TRIM_TAU` after step = -0.0600000000 (read from `_jax_new_state[21]`)
- **MATCH** — JAX computes same updated trim as Python

### Notch Filter Pre-Snapshot

**Precedent:** Notch filter state must be snapshotted BEFORE Python compute because `compute()` mutates the notch filter's internal state (`_x1`, `_x2`, `_y1`, `_y2`) via `biquad_notch_update()`. The both-synced capture snapshots these as float scalars (`notch_x1`, etc.) before compute runs. This is correct.

### ABS Trim State — No Reference Mutation Issue

Unlike the notch filter, the ABS trim state values are captured as Python scalars (`float`, `int`, `list`) — these are value copies, not references. The capture is:
```python
"abs_trim_tau": float(_sag._adaptive_bias_trim_tau),
"abs_hold_steps": int(_sag._adaptive_bias_hold_steps),
...
"abs_slow_error_history": list(_sag._adaptive_bias_slow_error_history),
```

Python's `compute()` later mutates `_sag._adaptive_bias_trim_tau` (the instance attribute), but the captured float is already a VALUE copy. No reference-mutation issue exists for ABS state.

### Error History Capture

The error histories are captured as `list()` copies:
```python
"abs_slow_error_history": list(_sag._adaptive_bias_slow_error_history),
```

This creates a shallow copy of the list. Since the list contains floats (immutable), the copy is safe. Python's `compute()` will `.append()` new errors to the original list, but the captured list snapshot is unaffected.

## Timing Semantics Decision

**Synchronization semantics:** JAX uses Python pre-compute ABS state and computes the SAME step. This is the correct semantics because:

1. The pre-compute state represents the controller's state at the start of step n, accumulated from steps 0..n-1.
2. Both Python and JAX then compute step n from this identical starting state.
3. Post-compute values (new trim, new hold, etc.) are compared to verify formula parity.

**This is NOT a "JAX uses Python post-compute state for next step" model.** JAX always starts from Python's pre-compute state for the current step, computes independently, and the results are compared. JAX's post-compute state is discarded; it does not carry over to the next step.

## Acceptance

| Check | Status |
|-------|--------|
| Python state captured before Python compute | ✓ VERIFIED (line 5932-5965) |
| `_adaptive_bias_trim_tau` captured as value copy | ✓ VERIFIED (`float()` call) |
| Captured value is pre-compute (not post-compute) | ✓ VERIFIED |
| JAX receives Python state before same step | ✓ VERIFIED (same `_both_synced_capture`) |
| Notch/APCR state snapshot precedent handled correctly | ✓ VERIFIED (pre-snapshot floats) |
| `abs_trim_tau` in `_py_state_snap` is a float snapshot | ✓ VERIFIED |
| `pack_state_from_python_k2(abs_trim_tau=...)` receives same value as logged | ✓ VERIFIED (trace shows jx_trim_pre = py_trim) |
| No post-Python-compute mutation changes captured state before JAX packing | ✓ VERIFIED (value copies) |
| Synchronization semantics defined and correct | ✓ VERIFIED (pre-compute → same-step compute) |

## Classification

**K2_JAX_ABS_TRIM_BOTH_SYNCED_CAPTURE_TIMING_CORRECT** — Capture timing is correct for all ABS trim state fields. No pre/post mixing. No reference mutation issues. Correct pre-compute → same-step semantics.
