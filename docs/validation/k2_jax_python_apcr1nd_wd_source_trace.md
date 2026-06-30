# K2 JAX Python APCR1ND Wheel Damping Source-of-Truth Trace

**Date:** 2026-06-29
**Phase:** 1
**Source file:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`

---

## A. Drift Priority State Machine (lines 6386-6526)

### State Variables

| Variable | Line | Type | Persists |
|----------|------|------|----------|
| `_apcr1nd_step_counter` | 4256 | int | Across steps |
| `_apcr1nd_prev_error` | 4257 | float | Across steps |
| `_apcr1nd_tuned_converging_steps` | 4263 | int | Across steps |
| `_apcr1nd_tuned_recenter_held` | 4264 | bool | Across steps |
| `apcr1nd_direct_recenter_priority_active` | 6379 | LOCAL | One compute() call |

### Execution Order (within `compute()`)

1. **Step counter increment** (line 6388-6389): `current_step = self._apcr1nd_step_counter; self._apcr1nd_step_counter += 1`
2. **Startup guard** (line 6393): Block if `current_step < startup_guard_steps`
3. **Drift detection** (lines 6397-6401): e_dot = signed_error - prev_error; moving_away = drift * e_dot > 0
4. **`prev_error` update** (line 6400): ALWAYS updates `self._apcr1nd_prev_error = signed_error` (even during startup guard)
5. **Safety gates** (lines 6410-6414): abs_pitch, abs_roll, com_z_safe, roll_safe, pitch_safe

### Tuned Variant Logic (lines 6417-6498, K2 profile active)

**Converging steps update** (lines 6427-6430):
- Only inside `else` branch (when safety gates PASS)
- `if converging: steps += 1; else: steps = 0`
- When safety FAILS: converging_steps UNCHANGED from previous value

**Safety check** (line 6433):
```python
safety_pass = contact_valid and com_z_safe and roll_safe and pitch_safe
```

**On `not safety_pass`** (lines 6435-6446):
- Sets `apcr1nd_direct_recenter_priority_active = False`
- Sets `self._apcr1nd_tuned_recenter_held = False`
- Sets block_reason
- Sets `eligible = False`

**On `safety_pass`** (lines 6448-6498):
- `prev_active = self._apcr1nd_tuned_recenter_held`
- Entry conditions: soft_entry, direct_entry, emergency_entry
- Hold condition: `prev_active and abs_error > release_inner_m`
- Release conditions: release_by_inner_band, release_by_converging
- Decision (if/elif chain):
  1. `release` → active=False, held=False
  2. `emergency | hold_outside` → active=True, held=True
  3. `direct | soft` → active=True, held=True
  4. `hold_condition` → active=True, held=True
  5. `else` → active=False, held=False, "below_threshold"

---

## B. APCR1n Recenter Priority → Wheel Damping Override (lines 6528-6670)

### Gate selection (lines 6565-6572)

```python
if self.authority_schedule.recenter_priority_direct_enabled:
    # APCR1nD: Use the direct support drift trigger
    apcr1n_recenter_priority_active = apcr1nd_direct_recenter_priority_active  # LOCAL var
else:
    # Original APCR1n: depends on _apc_drift_priority_active
    apcr1n_recenter_priority_active = self._apc_drift_priority_active
```

**CRITICAL:** The K2 profile has `recenter_priority_direct_enabled=True`, so the gate comes from `apcr1nd_direct_recenter_priority_active` (LOCAL, computed above), NOT from `_apc_drift_priority_active`.

### Wheel damping override gating (lines 6608-6670)

**Activation gate** (lines 6643-6647): Only applies if:
- Tuned variant: `wheel_scale < 1.0`
- Non-tuned: `apcr1n_wheel_damping_fights_drift`

Also requires `apcr1n_recenter_priority_active=True` (the enclosing `if` at line 6574).

**Band-based scale** (lines 6618-6628):
- abs_error >= emergency_band_m (0.12) → damping_scale_emergency (0.10)
- abs_error >= hard_band_m (0.10) → damping_scale_hard (0.20)
- abs_error >= desired_band_m (0.08) → damping_scale_desired (0.40)
- abs_error >= soft_enter_m (0.05) → damping_scale_soft (0.70)
- else → damping_scale_normal (1.0)

**Preserve-if-helps** (lines 6631-6635): If damping opposes drift, keep scale=1.0

**Min clamp** (lines 6659-6663): Ensure minimum damping preserved (`vd_wheel_damping_recenter_min_abs_nm`)

---

## C. State Timing

| When | What |
|------|------|
| Before compute() | `_apcr1nd_step_counter` from previous step |
| During APCR1nD section | `_apcr1nd_step_counter += 1`, `_apcr1nd_prev_error` updated, gate computed, `_apcr1nd_tuned_recenter_held` set |
| During APCR1n section | `apcr1n_recenter_priority_active` set from APCR1nD result, wheel damping override applied |
| **Both-synced capture** | Captures BEFORE compute() — reads **POST-state from previous step** |
| After compute() | State variables mutated, ready for next step's capture |

### Both-Synced Capture Timing Issue

The capture at line 5995-5996:
```python
"py_wd_override_active": float(1.0 if _sag._apc_drift_priority_active else -1.0),
```

This captures `_apc_drift_priority_active` (the APC-based gate, NOT the APCR1ND gate). When `recenter_priority_direct_enabled=True`:
- Python controller uses `apcr1nd_direct_recenter_priority_active` (line 6569)
- Not `_apc_drift_priority_active`

This means `py_wd_override_active` is the WRONG gate for APCR1ND wheel damping override when direct mode is active. However, this captured value is not used for actual gating in JAX — JAX computes its own `recenter_active` from `k2_jax_apcr1nd_compute_gate`.

---

## D. Position Cap Boost Safety (lines 6576-6584)

```python
apcr1n_safety_gate_pass = (
    contact_valid and com_z_safe and roll_safe and pitch_safe_gate
)
```

Used for `position_cap_recenter_boost_enabled` — when safety passes, raises max_position_tau from normal cap up to emergency cap.

---

## E. Acceptance

- [x] Every APCR1ND scalar mapped to source line
- [x] No UNKNOWN fields
- [x] Python wheel damping override uses POST-update APCR1ND state (computed in same compute() call, lines 6373-6498, used at line 6569)
- [x] `_apc_drift_priority_active` is a SEPARATE gate from `apcr1nd_direct_recenter_priority_active`
- [x] Both-synced capture of `py_wd_override_active` captures the WRONG gate (APC not APCR1ND)
