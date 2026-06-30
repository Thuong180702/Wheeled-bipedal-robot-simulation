# K2 JAX Python State → JAX State Mapping

**Date:** 2026-06-27
**Purpose:** Exact field-by-field mapping from Python K2 controller internal state to JAX 328-field flat state array.

---

## 1. State Timing

State is captured **BEFORE** Python computes the current step. This state reflects the accumulated result of all previous steps (0..n-1). Both Python and JAX compute step n from this identical starting state.

## 2. Complete State Mapping

### 2.1 Notch Filter State (Indices 0-3)

| JAX Index | JAX Field | Python Source | Python Access | Type |
|-----------|-----------|---------------|---------------|------|
| 0 | `notch_x1` | `BiquadNotchFilter.x1` | `sagittal._wip_notch_pitch_rate.x1` | float64 |
| 1 | `notch_x2` | `BiquadNotchFilter.x2` | `sagittal._wip_notch_pitch_rate.x2` | float64 |
| 2 | `notch_y1` | `BiquadNotchFilter.y1` | `sagittal._wip_notch_pitch_rate.y1` | float64 |
| 3 | `notch_y2` | `BiquadNotchFilter.y2` | `sagittal._wip_notch_pitch_rate.y2` | float64 |

**Notes:**
- K2 profile (`k2_notch_low_q_v1`) has `enable_wip_notch_filter=True`
- If notch filter is `None` (not yet initialized or disabled), all four fields = 0.0
- Python BiquadNotchFilter stores state as float; JAX stores as float64

### 2.2 Previous Torque (Indices 4-13)

| JAX Index | JAX Field | Python Source | Python Access |
|-----------|-----------|---------------|---------------|
| 4 | `prev_tau_0` (l_hip_roll) | `tau_prev[0]` | sim loop nonlocal `tau_prev` |
| 5 | `prev_tau_1` (l_hip_yaw) | `tau_prev[1]` | sim loop nonlocal `tau_prev` |
| 6 | `prev_tau_2` (l_hip_pitch) | `tau_prev[2]` | sim loop nonlocal `tau_prev` |
| 7 | `prev_tau_3` (l_knee) | `tau_prev[3]` | sim loop nonlocal `tau_prev` |
| 8 | `prev_tau_4` (l_wheel) | `tau_prev[4]` | sim loop nonlocal `tau_prev` |
| 9 | `prev_tau_5` (r_hip_roll) | `tau_prev[5]` | sim loop nonlocal `tau_prev` |
| 10 | `prev_tau_6` (r_hip_yaw) | `tau_prev[6]` | sim loop nonlocal `tau_prev` |
| 11 | `prev_tau_7` (r_hip_pitch) | `tau_prev[7]` | sim loop nonlocal `tau_prev` |
| 12 | `prev_tau_8` (r_knee) | `tau_prev[8]` | sim loop nonlocal `tau_prev` |
| 13 | `prev_tau_9` (r_wheel) | `tau_prev[9]` | sim loop nonlocal `tau_prev` |

**Notes:**
- This is the **final** torque from the previous step (after rate limiting, clipping, composer)
- Joint order matches the canonical 10-DOF order

### 2.3 Height Scheduling State (Index 14)

| JAX Index | JAX Field | Python Source | Python Access |
|-----------|-----------|---------------|---------------|
| 14 | `filtered_com_z` | `_filtered_com_z` | `sagittal._filtered_com_z` (float) |

**Notes:**
- Updated inside `SagittalVelocityDampedBalanceController.compute()`
- Initial value: 0.4 (default CoM height)

### 2.4 Support Error State (Index 15)

| JAX Index | JAX Field | Python Source | Python Access |
|-----------|-----------|---------------|---------------|
| 15 | `prev_support_error` | `prev_support_error` | sim loop nonlocal `prev_support_error` (float) |

**Notes:**
- Updated at end of each control step: `prev_support_error = sagittal_diag.get("support_position_error_m", 0.0)`

### 2.5 Outer Loop State (Indices 16-18)

| JAX Index | JAX Field | Python Source | Python Access |
|-----------|-----------|---------------|---------------|
| 16 | `ol_pitch_ref_smoothed` | `outer_loop_pitch_ref_smoothed_deg` | sim loop nonlocal (float, degrees) |
| 17 | `ol_prev_support_error` | `outer_loop_prev_support_error_m` | sim loop nonlocal (float, meters) |
| 18 | `ol_support_error_rate` | `outer_loop_support_error_rate_smoothed` | sim loop nonlocal (float, m/s) |

**Notes:**
- Outer loop is active for K2 profile (calibrated v2 PCHIP gains)
- `ol_prev_support_error_m` starts as `None`, initialized to current support error on first step

### 2.6 ABS Core State (Indices 19-27)

| JAX Index | JAX Field | Python Source | Python Access |
|-----------|-----------|---------------|---------------|
| 19 | `abs_slow_sum` | sum of slow error history | Computed: `sum(slow_history[-300:])` |
| 20 | `abs_fast_sum` | sum of fast error history | Computed: `sum(fast_history[-100:])` |
| 21 | `abs_trim_tau` | `_adaptive_bias_trim_tau` | `sagittal._adaptive_bias_trim_tau` (float) |
| 22 | `abs_hold_steps` | `_adaptive_bias_hold_steps` | `sagittal._adaptive_bias_hold_steps` (int) |
| 23 | `abs_prev_err_sign` | `_adaptive_bias_prev_error_sign` | `sagittal._adaptive_bias_prev_error_sign` (int: -1/0/1) |
| 24 | `abs_zc_count` | `_adaptive_bias_crossing_count` | `sagittal._adaptive_bias_crossing_count` (int) |
| 25 | `abs_slow_count` | len(slow error history) | `len(sagittal._adaptive_bias_slow_error_history)` |
| 26 | `abs_slow_ptr` | write pointer | Computed: `len(slow_history) % 300` |
| 27 | `abs_guard_trigger` | `_adaptive_bias_guard_trigger_count` | `sagittal._adaptive_bias_guard_trigger_count` (int) |

**Notes:**
- Python uses Python lists with `append`/`pop(0)` for sliding windows
- JAX uses a fixed-size ring buffer (300 elements) with a write pointer
- Conversion: iterate Python list (oldest first), write to ring buffer at sequential positions starting from `ptr`
- `slow_sum` and `fast_sum` are computed from the Python lists at capture time

### 2.7 ABS Ring Buffer (Indices 28-327)

| JAX Index Range | JAX Field | Python Source | Python Access |
|----------------|-----------|---------------|---------------|
| 28-327 | `abs_buf_0` .. `abs_buf_299` | `_adaptive_bias_slow_error_history` | `sagittal._adaptive_bias_slow_error_history` (list[float]) |

**Conversion logic:**
```python
n_entries = len(slow_error_history)  # typically grows to 300, then stays at 300
write_ptr = n_entries % 300
for i, val in enumerate(slow_error_history):
    buf_idx = (write_ptr + i) % 300
    state[buf_idx] = val
```

**Notes:**
- Python list is oldest-first (index 0 = oldest entry)
- JAX ring buffer wraps at 300 entries
- After 300 entries, the buffer is full and oldest entries are overwritten
- ABS ring buffer is only used when `adaptive_bias_trim_enabled=True` (K2 profile: True)

## 3. Fields Explicitly Zeroed (No Python Equivalent)

The following JAX state fields do not have a direct Python equivalent and are initialized to zero:

- `abs_slow_count` (index 25): Set to 0 when history is empty
- `abs_slow_ptr` (index 26): Set to 0 when history is empty
- `abs_slow_sum` (index 19): Set to 0 when history is empty
- `abs_fast_sum` (index 20): Set to 0 when fast history is empty

## 4. State Size Summary

| Group | Fields | Indices |
|-------|--------|---------|
| Notch filter | 4 | 0-3 |
| Previous torque | 10 | 4-13 |
| Height scheduling | 1 | 14 |
| Support error | 1 | 15 |
| Outer loop | 3 | 16-18 |
| ABS core | 9 | 19-27 |
| ABS ring buffer | 300 | 28-327 |
| **Total** | **328** | **0-327** |

## 5. Verification Checklist

- [ ] All 328 JAX state fields populated from Python state or explicitly zeroed
- [ ] No UNKNOWN state fields
- [ ] State capture timing is before Python compute (state reflects end of step n-1)
- [ ] Notch filter correctly accessed from sagittal._wip_notch_pitch_rate
- [ ] ABS ring buffer correctly converted from Python list to JAX circular buffer
- [ ] Input_flat packing verified identical for both backends
- [ ] Normal backend behavior preserved (python, jax, both unchanged)
