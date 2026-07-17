# K2 JAX — ABS Trim JAX Subsystem Design (Phase 3)

**Date:** 2026-06-28
**Classification target:** Remove PARTIAL status for ABS trim

---

## A. Parameter Changes

**No new params added.** All ABS trim parameters are already read from the K2 profile (`K2_NOTCH_LOW_Q_V1`) at runtime via `from ... import K2_NOTCH_LOW_Q_V1 as _sch`. The `adaptive_bias_zero_crossing_window_steps = 500` value is accessed as `_sch.adaptive_bias_zero_crossing_window_steps`.

Add a module-level constant:
```python
_ABS_ZC_WINDOW = 500  # from K2 profile: adaptive_bias_zero_crossing_window_steps
```

No changes to `pack_params_stage2()` or params_flat needed.

---

## B. Input Changes

Add `contact_valid` (float64, 0.0 or 1.0) to `K2_JAX_INPUT_FIELDS`:

```python
K2_JAX_INPUT_FIELDS = (..., "contact_valid")  # index 41
K2_JAX_INPUT_SIZE = 42  # was 41
```

New index constant:
```python
_I_CONTACT_VALID = 41
```

Update `pack_input_k2()` to accept `contact_valid` parameter.

---

## C. State Changes

### C.1 New ZC Ring Buffer

Add a separate 500-entry ring buffer for zero-crossing detection, matching Python's separate `_adaptive_bias_zero_crossing_history`.

New state layout (834 entries total = old 332 + 2 header + 500 buffer):

```
0-3:   notch state (4)
4-13:  prev_tau (10)
14:    filtered_com_z
15:    prev_support_error
16-18: outer_loop state (3)
19-27: ABS core (9) — slow_sum, fast_sum, trim_tau, hold_steps, prev_err_sign, zc_count, slow_count, slow_ptr, guard_trigger
28-327: abs_slow_buffer (300)
328:     abs_zc_buffer_count (NEW)
329:     abs_zc_buffer_ptr (NEW)
330-829: abs_zc_buffer (500) (NEW)
830:     apcr1nd_step_counter (shifted from 328)
831:     apcr1nd_prev_error (shifted from 329)
832:     apcr1nd_converging_steps (shifted from 330)
833:     apcr1nd_recenter_held (shifted from 331)
```

### C.2 New Indices

```python
_ABS_ZC_BUF_COUNT = 328
_ABS_ZC_BUF_PTR = 329
_ABS_ZC_BUF_START = 330
_ABS_ZC_BUF_END = 830  # 330 + 500

_S_APCR1ND_STEP_COUNTER = 830     # shifted from 328
_S_APCR1ND_PREV_ERROR = 831       # shifted from 329
_S_APCR1ND_CONVERGING_STEPS = 832 # shifted from 330
_S_APCR1ND_RECENTER_HELD = 833    # shifted from 331
```

### C.3 Updated Field Lists

```python
_ABS_ZC_FIELDS = ("abs_zc_count_buf", "abs_zc_ptr_buf")
_ABS_ZC_BUF_FIELDS = tuple(f"abs_zc_buf_{i}" for i in range(500))

# Rebuild state fields (replacing APCR1ND append)
K2_JAX_STATE_FIELDS = K2_JAX_STATE_FIELDS + _ABS_CORE_FIELDS + _ABS_RING_FIELDS + _ABS_ZC_FIELDS + _ABS_ZC_BUF_FIELDS + _APCR1ND_STATE_FIELDS
K2_JAX_STATE_SIZE = 834
```

---

## D. New/Modified Functions

### D.1 ZC Buffer Update

```python
def _abs_update_zc_buffer(state_flat, error_signed):
    """Push error into ZC ring buffer (500 entries, separate from slow/fast)."""
    ptr = state_flat[_ABS_ZC_BUF_PTR].astype(jnp.int32)
    count = state_flat[_ABS_ZC_BUF_COUNT]
    
    new_state = state_flat.at[_ABS_ZC_BUF_START + ptr].set(error_signed)
    new_state = new_state.at[_ABS_ZC_BUF_PTR].set((ptr + 1) % _ABS_ZC_WINDOW)
    new_count = jnp.where(count >= _ABS_ZC_WINDOW, count, count + 1.0)
    new_state = new_state.at[_ABS_ZC_BUF_COUNT].set(new_count)
    
    return new_state
```

### D.2 ZC Counting from ZC Buffer

```python
def _abs_count_zero_crossings_from_zc(state_flat):
    """Count sign changes in ZC ring buffer (JIT-compatible, 500 entries)."""
    count = state_flat[_ABS_ZC_BUF_COUNT]
    ptr = state_flat[_ABS_ZC_BUF_PTR].astype(jnp.int32)
    buf = state_flat[_ABS_ZC_BUF_START:_ABS_ZC_BUF_END]
    
    i_range = jnp.arange(_ABS_ZC_WINDOW)
    reverse_indices = (ptr - 1 - i_range + _ABS_ZC_WINDOW) % _ABS_ZC_WINDOW
    vals = buf[reverse_indices]
    
    vals_prev = jnp.roll(vals, shift=1)
    vals_prev = vals_prev.at[0].set(vals[0])
    
    valid_curr = i_range < count
    valid_prev = jnp.roll(valid_curr, shift=1)
    both_valid = valid_curr & valid_prev
    
    sign_change = (vals < 0) != (vals_prev < 0)
    zc = jnp.sum(jnp.where(sign_change & both_valid, 1, 0))
    return zc
```

### D.3 Fixed Guard Trigger Reset

```python
# OLD (wrong):
guard_trigger = jnp.where(zc_guard, guard_trigger + 1.0,
                 jnp.where(guard_trigger >= 3.0, 0.0, 0.0))

# NEW (correct — matches Python svdbc.py:5633-5641):
guard_trigger = jnp.where(
    zc_guard,
    jnp.where(guard_trigger + 1.0 >= 3.0, 0.0, guard_trigger + 1.0),  # inc, reset at 3
    0.0,  # hard reset when guard inactive
)
```

### D.4 Fixed Prev Sign Update

```python
# OLD (line 2171):
new_prev_sign = jnp.where(err_sign_changed, sign_err, prev_err_sign)

# NEW (matches Python elif hold_steps > 0 branch):
new_prev_sign = jnp.where(err_sign_changed | (new_hold > 0.0), sign_err, prev_err_sign)
```

Python updates `prev_error_sign` in both the `if err_sign_changed` branch AND the `elif hold_steps > 0` branch. The JAX equivalent is: update when `err_sign_changed` OR when `new_hold > 0`.

### D.5 Fixed Contact Safety Gate

```python
# OLD:
_contact_ok = True

# NEW:
_contact_ok = (contact_valid > 0.5)  # float64 → bool
```

Where `contact_valid` comes from input flat at `_I_CONTACT_VALID`.

---

## E. Updated `_k2_jax_adaptive_bias_trim` Signature

Add `contact_valid` parameter:
```python
def _k2_jax_adaptive_bias_trim(
    signed_error, state_flat,
    schedule_h, pitch_x, safety_pass_in,
    contact_valid,  # NEW
):
```

Update ZC counting:
```python
# OLD:
state_flat = _abs_update_ring_buffer(state_flat, signed_error)
zc_count = _abs_count_zero_crossings(state_flat)

# NEW:
state_flat = _abs_update_ring_buffer(state_flat, signed_error)
state_flat = _abs_update_zc_buffer(state_flat, signed_error)  # NEW
zc_count = _abs_count_zero_crossings_from_zc(state_flat)     # uses ZC buffer
```

---

## F. Updated Caller in `k2_jax_controller_step`

```python
# OLD:
_safety = _contact_ok & _upright_ok & _abs_error_ok & _hip_yaw_ok

# NEW:
_contact_ok_input = input_flat[_I_CONTACT_VALID] > 0.5
_contact_ok = jnp.where(
    float(_sch.adaptive_bias_only_when_contact_stable) > 0.5,
    _contact_ok_input,
    True,
)
_safety = _contact_ok & _upright_ok & _abs_error_ok & _hip_yaw_ok
```

---

## G. State Packing Changes

### G.1 `pack_state_k2` — No signature change, just zeros the new fields

### G.2 `pack_state_from_python_k2` — Add `abs_zc_error_history` param

```python
def pack_state_from_python_k2(
    ...,
    abs_zc_error_history=None,  # NEW: list[float], max 500 entries
):
```

Pack ZC buffer:
```python
if abs_zc_error_history is not None and len(abs_zc_error_history) > 0:
    n_entries = len(abs_zc_error_history)
    write_ptr = n_entries % _ABS_ZC_WINDOW
    for i, val in enumerate(abs_zc_error_history):
        buf_idx = (write_ptr + i) % _ABS_ZC_WINDOW
        s = s.at[_ABS_ZC_BUF_START + buf_idx].set(float(val))
    s = s.at[_ABS_ZC_BUF_COUNT].set(float(min(n_entries, _ABS_ZC_WINDOW)))
    s = s.at[_ABS_ZC_BUF_PTR].set(float(write_ptr))
```

### G.3 `pack_state_k2_final` — Add ZC init

Add `abs_zc_error_history=None` parameter and pack it. Or add separate `abs_zc_buf_*` fields.

---

## H. Both-Synced Changes

In `simulate_hierarchical_controller.py`:

1. Pass `contact_valid` from Python physics to JAX input:
   ```python
   contact_valid=float(_contact_valid_for_feedback),
   ```

2. Pass `abs_zc_error_history` to state packing:
   ```python
   abs_zc_error_history=list(_sag._adaptive_bias_zero_crossing_history),
   ```

3. Update input size reference from 41 to 42.

---

## I. Backward Compatibility

- `pack_state_k2()`: No signature change needed (new fields default to zero)
- `pack_input_k2()`: Added `contact_valid` with default `1.0` (backward-compatible)
- `pack_params_stage2()`: No change needed
- Test files: Need minimum updates for state/input size constants

---

## J. Diagnostics

Add to both-synced output:
- `py_contact_valid` / `jx_contact_valid`
- `py_zc_count` / `jx_zc_count` (from ZC buffer vs ring buffer)
- `py_zc_buffer_entries` / `jx_zc_buffer_count`

---

## K. Files Changed

1. `wheeled_biped/controllers/k2_jax_controller.py` — main implementation
2. `scripts/simulate_hierarchical_controller.py` — both-synced state/input packing
3. `tests/test_k2_jax_step_parity.py` — update state/input size references
4. `tests/test_k2_jax_component_parity.py` — update pack/unpack tests

---

## L. Acceptance Criteria

- State size: 332 → 834
- Input size: 41 → 42
- ZC buffer: 500 entries, separate from slow/fast buffer
- Contact gate: uses actual contact_valid, not hardcoded True
- Guard trigger: correctly resets at ≥3 when ZC guard active
- Prev sign: updated on hold>0 (matching Python elif branch)
- Fixed-height parity: preserved (<1e-5)
- Dynamic parity: restored (<1e-5 for ramp_up, gate_chatter)
- Push parity: restored (<1e-5 for push_fwd_90N, push_bwd_90N)
- All tests pass
- No Python behavior changed
- No gains tuned
- No thresholds relaxed
