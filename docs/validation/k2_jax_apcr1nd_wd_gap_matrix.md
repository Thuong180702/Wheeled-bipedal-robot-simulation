# K2 JAX APCR1ND Wheel Damping — Gap Matrix

**Date:** 2026-06-29
**Phase:** 2

---

## Scalar Gap Matrix

| Python scalar | JAX scalar | Match? | Gap Type | First Divergent Step | Impact |
|--------------|------------|--------|----------|----------------------|--------|
| `_apcr1nd_step_counter` | `new_step_counter` | **PASS** | — | — | — |
| `_apcr1nd_prev_error` | `new_prev_error` | **PASS** | — | — | — |
| `_apcr1nd_tuned_converging_steps` | `new_converging_steps` | **MISMATCH** | STATE_TIMING_MISMATCH | When safety fails | State divergence accumulation |
| `_apcr1nd_tuned_recenter_held` | `new_recenter_held` | **MISMATCH** | PRE_POST_STATE_MISMATCH | When safety fails | Gate persistence persists |
| `apcr1nd_direct_recenter_priority_active` | `recenter_active` | **MISMATCH** | FORMULA_MISMATCH | When safety fails or contact_valid=0 | Wheel damping override fires differently |
| `safety_pass` (APCR1ND gate) | `safety_pass` (JAX gate) | **MISMATCH** | FORMULA_MISMATCH | When contact_valid=0 | Root cause of all above |
| `apcr1n_safety_gate_pass` (cap boost) | `_cap_safety` (JAX cap boost) | **MISMATCH** | FORMULA_MISMATCH | When contact_valid=0 | Position cap differs |
| `wheel_scale` (tuned) | `wheel_scale` (JAX) | **PASS** | — | — | Band thresholds match |
| `apply_override` gate | `apply_override` (JAX) | **PASS*** | — | — | Formula matches; gated by recenter_active |
| `tau_wheel_vel_left/right` | `_new_tau_wvl/wvr` | **PASS*** | — | — | Formula matches; depends on above |
| `min_damping` clamp | min clamp (JAX) | **PASS** | — | — | `vd_wheel_damping_recenter_min_abs_nm` |
| `boosted_cap` | `_boosted_cap` | **MISMATCH** | FORMULA_MISMATCH | When contact_valid=0 | Safety check missing contact_valid |
| `effective_max_position_tau` | `effective_max_pos_tau` | **PASS** | — | — | Captured from Python via `effective_max_pos_tau_py` |
| Two-clip order | Two-clip order | **PASS** | — | — | Both: first clip to max, second clip to boosted_cap |

*Note: Wheel damping formula PASSES but its OUTPUT depends on `recenter_active` gating, which diverges.

---

## Detailed Gap Analysis

### Gap 1: `safety_pass` — Missing `contact_valid`

**Python** (`svdbc.py:6433`):
```python
safety_pass = contact_valid and com_z_safe and roll_safe and pitch_safe
```

**JAX** (`k2_jax_controller.py:726`):
```python
safety_pass = com_z_safe & roll_safe & pitch_safe
```

**Gap type:** FORMULA_MISMATCH

**Impact:** When `contact_valid=False` (wheel lift during push), Python marks safety as failed and disables APCR1ND. JAX sees safety as passing and keeps APCR1ND active. This is the ROOT CAUSE of all downstream mismatches.

**Verification:** The `contact_valid` input IS available in JAX (`_contact_valid_val` at line 1811), it's just not passed to `k2_jax_apcr1nd_compute_gate`.

### Gap 2: Converging Steps Update Timing

**Python** (`svdbc.py:6427-6430`):
```python
if not safety_pass:
    # converging_steps NOT updated
else:
    if converging:
        self._apcr1nd_tuned_converging_steps += 1
    else:
        self._apcr1nd_tuned_converging_steps = 0
```

**JAX** (`k2_jax_controller.py:729-733`):
```python
new_converging_steps = jnp.where(
    after_guard & converging,
    converging_steps + 1.0,
    jnp.where(after_guard, 0.0, converging_steps),
)
```

**Gap type:** STATE_TIMING_MISMATCH

**Impact:** When safety fails, Python keeps old converging_steps; JAX unconditionally updates. This causes state divergence on safety gate flapping.

### Gap 3: `recenter_held` Reset on Safety Fail

**Python** (`svdbc.py:6445-6446`):
```python
if not safety_pass:
    apcr1nd_direct_recenter_priority_active = False
    self._apcr1nd_tuned_recenter_held = False  # RESET
```

**JAX** (`k2_jax_controller.py:761-764`):
```python
gated = after_guard & safety_pass
release = gated & (release_by_inner_band | release_by_converging)
activate = gated & (emergency_entry | hold_outside_band | direct_entry | soft_entry | hold_condition)
new_recenter_held = jnp.where(
    release, 0.0,
    jnp.where(activate, 1.0, recenter_held),  # KEEPS old value when !gated
)
```

**Gap type:** PRE_POST_STATE_MISMATCH

**Impact:** When safety fails, Python resets `recenter_held` → APCR1ND gate closes. JAX preserves `recenter_held` → APCR1ND gate stays open. This is the PRIMARY cause of the `recenter_active` divergence at step 279.

### Gap 4: Position Cap Boost Safety

**Python** (`svdbc.py:6582-6584`):
```python
apcr1n_safety_gate_pass = contact_valid and com_z_safe and roll_safe and pitch_safe_gate
```

**JAX** (`k2_jax_controller.py:1855-1857`):
```python
_cap_safety = com_z >= _apcr1nd_safe_com_z
_cap_safety = _cap_safety & (jnp.abs(roll_y) <= _apcr1nd_safe_roll)
_cap_safety = _cap_safety & (jnp.abs(pitch_x) <= _apcr1nd_safe_pitch)
```

**Gap type:** FORMULA_MISMATCH

**Impact:** Same missing `contact_valid` as Gap 1. When contact_valid=0 during push, Python skips position cap boost; JAX may apply it. This causes the 4.79 Nm tau_position divergence at push onset.

### Gap 5: Both-Synced Capture `py_wd_override_active`

**Python capture** (`simulate_hierarchical_controller.py:5995-5996`):
```python
"py_wd_override_active": float(1.0 if _sag._apc_drift_priority_active else -1.0),
```

**Gap type:** DIAGNOSTIC_ONLY

**Impact:** This captures the APC gate (`_apc_drift_priority_active`), not the APCR1ND gate (`apcr1nd_direct_recenter_priority_active`). With `recenter_priority_direct_enabled=True`, the ACTUAL gate in Python is the APCR1ND direct trigger (line 6569), NOT the APC gate. This value is NOT used for JAX gating (JAX computes its own gate), so it's diagnostic-only. But it may confuse debugging.

---

## Summary

**Root cause:** Three formula/state-timing mismatches in `k2_jax_apcr1nd_compute_gate`:
1. Missing `contact_valid` in safety_pass
2. Converging steps update unconditional (should be conditional on safety)
3. `recenter_held` not reset on safety fail

**Plus:** Missing `contact_valid` in position cap boost safety check (separate instance of same pattern).

**Not mismatched:** Wheel damping scale formula, min clamp, band thresholds, params — all match.

**First divergent torque path:** Step 20 push onset → tau_position diverges (position cap safety mismatch) → propagates through tau_common → wheel torque final.

---

## Acceptance

- [x] First divergent APCR1ND scalar identified: `safety_pass` (missing contact_valid)
- [x] 0.34-0.47 Nm final torque divergence explained through: safety_pass → recenter_held → recenter_active → wheel damping override → tau_wheel_vel → final tau
- [x] All gap types classified: 4 FORMULA_MISMATCH, 1 STATE_TIMING_MISMATCH, 1 PRE_POST_STATE_MISMATCH, 1 DIAGNOSTIC_ONLY
