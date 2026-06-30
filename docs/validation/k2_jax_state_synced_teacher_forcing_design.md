# K2 JAX State-Synced Teacher-Forcing Design

**Date:** 2026-06-27
**Classification:** `K2_JAX_STATE_SYNCED_DESIGN_APPROACH_A`

---

## 1. Problem Statement

The existing `--controller-backend both` mode compares Python and JAX torque while each backend maintains **independent internal controller state**:

- Python has `tau_prev`, `BiquadNotchFilter` (x1/x2/y1/y2), `_filtered_com_z`, `prev_support_error`, outer-loop state (pitch_ref_smoothed, prev_support_error, support_error_rate), and ABS state (ring buffer, trim, hold, signs, counts, guard).
- JAX has a 328-field flat state array with the same fields laid out contiguously.
- A tiny step-0 torque difference (4.77e-08 Nm) causes `prev_tau` divergence, then rate limiting amplifies it from step 1 onward.
- Therefore ordinary `both`-mode cannot prove strict parity after step 0.

## 2. Design Goal

Prove that, given **the same physical inputs and the same internal controller state**, JAX computes the **same torque** as Python K2.

The goal is **not** to run JAX as a real controller in this mode — it is to validate formula/coefficient parity with bit-accuracy.

## 3. Approach A — Python State → JAX State Packer (SELECTED)

### 3.1 Capture Point: Before Python Compute

At each control step, **before** Python computes:

1. **Save** a snapshot of Python K2 internal state.
2. Python computes → produces `python_tau` and updates its own state for next step.
3. **Pack** the saved snapshot into JAX 328-field state format.
4. Run JAX `k2_jax_controller_step()` with the synced state + same `input_flat`.
5. Compare `python_tau` vs `jax_tau`.
6. **Discard** JAX state — do not persist across steps.

Both backends start from identical state at the same logical compute point, so any torque difference must be a formula or parameter mismatch.

### 3.2 State Mapping: Python → JAX

| # | Python Source | Python Variable/Attribute | JAX Index | JAX Field | Size |
|---|--------------|--------------------------|-----------|-----------|------|
| 0 | Notch filter | `notch_filter.x1` | 0 | `notch_x1` | 1 |
| 1 | Notch filter | `notch_filter.x2` | 1 | `notch_x2` | 1 |
| 2 | Notch filter | `notch_filter.y1` | 2 | `notch_y1` | 1 |
| 3 | Notch filter | `notch_filter.y2` | 3 | `notch_y2` | 1 |
| 4-13 | Sim loop | `tau_prev[0:10]` | 4-13 | `prev_tau_0..9` | 10 |
| 14 | Sagittal ctrl | `self._filtered_com_z` | 14 | `filtered_com_z` | 1 |
| 15 | Sim loop | `prev_support_error` | 15 | `prev_support_error` | 1 |
| 16 | Sim loop | `outer_loop_pitch_ref_smoothed_deg` | 16 | `ol_pitch_ref_smoothed` | 1 |
| 17 | Sim loop | `outer_loop_prev_support_error_m` | 17 | `ol_prev_support_error` | 1 |
| 18 | Sim loop | `outer_loop_support_error_rate_smoothed` | 18 | `ol_support_error_rate` | 1 |
| 19 | ABS state | `_adaptive_bias_ring_buffer[0:300]` sum | 19 | `abs_slow_sum` | 1 |
| 20 | ABS state | Computed (fast mean not stored) | 20 | `abs_fast_sum` | 1 |
| 21 | ABS state | `self._adaptive_bias_trim_tau` | 21 | `abs_trim_tau` | 1 |
| 22 | ABS state | `self._adaptive_bias_hold_steps` | 22 | `abs_hold_steps` | 1 |
| 23 | ABS state | `self._adaptive_bias_prev_err_sign` | 23 | `abs_prev_err_sign` | 1 |
| 24 | ABS state | `self._adaptive_bias_zc_count` | 24 | `abs_zc_count` | 1 |
| 25 | ABS state | Ring buffer valid count | 25 | `abs_slow_count` | 1 |
| 26 | ABS state | Ring buffer write pointer | 26 | `abs_slow_ptr` | 1 |
| 27 | ABS state | `self._adaptive_bias_guard_trigger_count` | 27 | `abs_guard_trigger` | 1 |
| 28-327 | ABS state | `self._adaptive_bias_ring_buffer[0:300]` | 28-327 | `abs_buf_0..299` | 300 |

**Total: 328 fields**

### 3.3 Python State Access Methods

#### 3.3.1 Notch Filter State

The notch filter is accessed via the sagittal controller's internal `_notch_filter` attribute. The `BiquadNotchFilter` class stores state as `x1`, `x2`, `y1`, `y2`.

Access: `balance_core_controllers["sagittal"]._notch_filter.x1` (x2, y1, y2)

#### 3.3.2 Sim Loop Nonlocal Variables

These are Python nonlocal variables in the control step closure:

- `tau_prev` — 10-element JAX array (previous final torque)
- `prev_support_error` — float (previous support position error)
- `outer_loop_pitch_ref_smoothed_deg` — float
- `outer_loop_prev_support_error_m` — float
- `outer_loop_support_error_rate_smoothed` — float

#### 3.3.3 Filtered CoM Z

Access: `balance_core_controllers["sagittal"]._filtered_com_z` — float

#### 3.3.4 ABS State

The sagittal controller (`SagittalVelocityDampedBalanceController`) maintains:

- `self._adaptive_bias_trim_tau` — float
- `self._adaptive_bias_hold_steps` — int
- `self._adaptive_bias_prev_err_sign` — int (-1, 0, 1)
- `self._adaptive_bias_zc_count` — int
- `self._adaptive_bias_guard_trigger_count` — int
- `self._adaptive_bias_ring_buffer` — list[int], max 300 entries (circular buffer)
- `self._adaptive_bias_slow_count` — int (valid entries in ring buffer)
- `self._adaptive_bias_slow_ptr` — int (write pointer)

Running sums (`abs_slow_sum`, `abs_fast_sum`) are computed lazily in Python but stored explicitly in JAX.

### 3.4 State Capture Timing

The critical invariant: state is captured **before** Python computes, because that is the state Python uses to compute the current step's torque.

```
For each control step:
  ┌─ SAVE snapshot of Python state (before compute)
  ├─ Python computes → python_tau, updates Python state
  ├─ PACK snapshot into JAX state
  ├─ JAX computes with synced state → jax_tau
  ├─ COMPARE python_tau vs jax_tau
  └─ DISCARD jax_tau, jax_new_state (not used for physics or next step)
```

### 3.5 Input Parity

The `input_flat` (41 fields) must be identical between Python and JAX. Both read from the same physics state at the same step, so input is guaranteed identical. Verification:

- Print input_flat hash at each step
- Print per-field diff if any field differs by > 1e-12

### 3.6 Normal Backend Behavior Preservation

- `--controller-backend python` — unchanged
- `--controller-backend jax` — unchanged
- `--controller-backend both` — unchanged (existing independent-state comparison)
- **New:** `--controller-backend both-synced` — state-synced teacher-forcing

The new mode:
- Uses Python torque for physics (like `both`)
- Extracts Python state before compute
- Packs into JAX state
- Compares JAX output from synced state
- Logs detailed diagnostics

## 4. Acceptance Criteria

### 4.1 State Packing Completeness
- [ ] All 328 JAX state fields are populated from Python state or explicitly marked as zero/unused
- [ ] No UNKNOWN state fields
- [ ] No hidden independent JAX state in synced comparison
- [ ] State capture timing is documented and correct (before Python compute)

### 4.2 Parity Thresholds
- [ ] Full 10-dim final tau max_abs_diff < 1e-5 for all checked steps
- [ ] Component torque terms < 1e-8 where available
- [ ] Input fields < 1e-12 (guaranteed by shared physics)
- [ ] State fields after packing < 1e-12 vs Python source
- [ ] No systematic growth in diff

### 4.3 Backward Compatibility
- [ ] Normal `python` backend unchanged
- [ ] Normal `jax` backend unchanged
- [ ] Normal `both` backend unchanged
- [ ] All existing tests still pass

## 5. Implementation Plan

### 5.1 New Function: `pack_state_from_python_k2()`

Location: `wheeled_biped/controllers/k2_jax_controller.py`

```python
def pack_state_from_python_k2(
    notch_filter,        # BiquadNotchFilter instance
    tau_prev,            # np.ndarray or jnp.ndarray, shape (10,)
    filtered_com_z,      # float
    prev_support_error,  # float
    ol_pitch_ref_smoothed,    # float
    ol_prev_support_error,    # float
    ol_support_error_rate,    # float
    abs_trim_tau,        # float
    abs_hold_steps,      # int
    abs_prev_err_sign,   # int
    abs_zc_count,        # int
    abs_guard_trigger,   # int
    abs_ring_buffer,     # list[float], max 300 entries
    abs_slow_count,      # int
    abs_slow_ptr,        # int
    abs_slow_sum,        # float
    abs_fast_sum,        # float
) -> jnp.ndarray:
    """Pack Python K2 internal state into JAX 328-field state array."""
```

### 5.2 New CLI Flag

Add `--controller-backend both-synced` to `simulate_hierarchical_controller.py`.

### 5.3 New Comparison Block

In the balance-core control step, after Python computes, check if `both-synced` mode:

1. Pack Python state snapshot → JAX state
2. Call JAX step function with synced state
3. Compare torques
4. Log detailed diagnostics

### 5.4 State Mapping Document

Create: `docs/validation/k2_jax_python_state_to_jax_state_mapping.md`

## 6. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| ABS ring buffer not accessible from Python | Add getter method to sagittal controller |
| Notch filter state not accessible from sim loop | Add accessor or read directly from sagittal controller |
| Off-by-one in state timing | Document capture point explicitly; verify by checking step-0 diff = 0 |
| JIT compilation overhead for synced mode | Acceptable — this is validation, not production |
| Python ABS ring buffer format differs from JAX | Verify entry-by-entry mapping in tests |

## 7. Classification

**`K2_JAX_STATE_SYNCED_DESIGN_APPROACH_A`**

Approach A (Python state → JAX state packer, capture before compute) is the preferred design. It provides the clearest parity proof with minimal code changes and zero risk to existing backend behavior.
