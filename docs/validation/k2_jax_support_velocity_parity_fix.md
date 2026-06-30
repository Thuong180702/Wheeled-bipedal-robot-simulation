# K2 JAX Support Velocity Parity Fix

**Date:** 2026-06-27
**Classification:** `K2_JAX_SUPPORT_VELOCITY_FIX_APPLIED`

---

## 1. Bug

JAX input packed `support_velocity_m_s=0.0` hardcoded. Python K2 computes dynamic `support_position_velocity_m_s` as the numerical derivative of support position error.

## 2. Root Cause

In `simulate_hierarchical_controller.py:6539`, the `pack_input_k2()` call hardcoded `support_velocity_m_s=0.0`.

Python computes at `sagittal_velocity_damped_balance_controller.py:4625`:
```python
support_position_velocity_m_s = (sagittal_position_error_m - self.prev_support_position_error_m) / self.dt
```

The computed value is available in `sagittal_diag["support_position_velocity_m_s"]` after Python compute.

## 3. Fix

**File:** `scripts/simulate_hierarchical_controller.py`
**Line:** 6539
**Change:**
```python
# Before:
support_velocity_m_s=0.0,

# After:
support_velocity_m_s=float(sagittal_diag.get("support_position_velocity_m_s", 0.0)),
```

## 4. Impact on Torque

**K2 profile has `k_support_velocity=0.0`** (confirmed in `k2_notch_low_q_v1_create_and_validate_report.md`).

The `effective_support_velocity_gain` is 0.0, so `tau_support_velocity = -0.0 * 1.0 * support_velocity_m_s = 0`.

**Therefore:** This fix does NOT change JAX torque output. The ~0.17 Nm wheel diff at high_0p480 is caused by another mechanism. Per-term sagittal diagnostics have been added to identify the real source.

## 5. Diagnostics Added

In `both-synced` mode, per-step diagnostics now report:
- `py_support_velocity_m_s` — Python-computed value
- `jax_input_support_velocity_m_s` — JAX input value
- `support_velocity_diff` — difference
- `py_tau_sv` — Python tau_support_velocity
- `jax_tau_sv` — JAX tau_support_velocity
- `gain` — effective_support_velocity_gain

Per-term sagittal torque comparison:
- tau_pitch, tau_pitch_rate, tau_sagittal_velocity, tau_support_velocity
- tau_cp, tau_com_vy, tau_wheel_vel_left, tau_wheel_vel_right, tau_position

## 6. Classification

**`K2_JAX_SUPPORT_VELOCITY_FIX_APPLIED`**

Input parity fixed. Torque unaffected (gain=0.0). Diagnostics in place for tracing real wheel diff source.
