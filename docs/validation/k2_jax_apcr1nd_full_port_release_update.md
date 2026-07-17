# K2 JAX APCR1ND Full Port Release Update — Phase 8

**Date:** 2026-06-28
**Branch:** repo-cleanup-t6j

## 1. Previous Classification

**K2_JAX_RELEASE_LOCK_PASS_DYNAMIC_PARITY_BLOCKED**

Known limitation: Dynamic height and push both-synced parity diverge at wheel indices [4,9].

## 2. APCR1ND Source Trace Summary

APCR1ND (Adaptive Position Centering Recenter with Direct drift trigger) is ACTIVE for K2_NOTCH_LOW_Q_V1. The `recenter_priority_enabled`, `recenter_priority_direct_enabled`, `apcr1nd_tuned_enabled`, and `vd_wheel_damping_recenter_override_enabled` fields are all True, inherited through the profile chain.

The APCR1ND wheel damping override applies band-based scaling to wheel velocity damping torque when the sagittal position error exceeds thresholds, with hysteresis gating based on drift direction, safety checks, and startup guard.

See: [k2_jax_python_apcr1nd_source_trace.md](k2_jax_python_apcr1nd_source_trace.md)

## 3. Exact Root Causes

### Root Cause: Priority Order Inversion in JAX Gate Function

**File:** `wheeled_biped/controllers/k2_jax_controller.py`, `k2_jax_apcr1nd_compute_gate()`

The JAX `k2_jax_apcr1nd_compute_gate()` function used:
```python
new_recenter_held = jnp.where(
    activate, 1.0,          # ACTIVATE took priority
    jnp.where(release, 0.0, recenter_held),
)
```

But Python's if/elif chain (line 6437-6461 of `sagittal_velocity_damped_balance_controller.py`) checks RELEASE first:
```python
if release_by_inner_band or release_by_converging:
    active = False  # RELEASE takes priority
elif ...:
    active = True
elif hold_condition:
    active = True  # Only reached if NOT releasing
```

When position error is slowly converging above the release band, BOTH `release_by_converging` and `hold_condition` can be True. Python correctly deactivates (release priority). JAX incorrectly activated (activate priority). This caused JAX to apply the wheel damping override when Python didn't, producing ~50% wheel damping difference.

## 4. Exact JAX Implementation

### New Function: `k2_jax_apcr1nd_compute_gate()`
- Pure JAX implementation of Python's APCR1nD Tuned Variants Logic (lines 6349-6490)
- Computes: startup guard → drift detection → safety gates → entry/hold/release conditions
- Returns: `recenter_active` boolean + 4 updated state values

### State Additions (4 fields, 328 → 332):
| Index | Field | Python Source |
|-------|-------|---------------|
| 328 | `apcr1nd_step_counter` | `_apcr1nd_step_counter` (line 4256) |
| 329 | `apcr1nd_prev_error` | `_apcr1nd_prev_error` (line 4257) |
| 330 | `apcr1nd_tuned_converging_steps` | `_apcr1nd_tuned_converging_steps` (line 4263) |
| 331 | `apcr1nd_tuned_recenter_held` | `_apcr1nd_tuned_recenter_held` (line 4264) |

### Params Additions (8 fields, 33 → 41):
| Index | Field | K2 Value | Python Source |
|-------|-------|----------|---------------|
| 33 | `apcr1nd_startup_guard_steps` | 40.0 | `recenter_priority_startup_guard_steps` |
| 34 | `apcr1nd_safe_min_com_z` | 0.25 | `recenter_priority_safe_min_com_z` |
| 35 | `apcr1nd_safe_roll_rad` | 0.30 | `recenter_priority_safe_roll_rad` |
| 36 | `apcr1nd_safe_pitch_rad` | 0.30 | `recenter_priority_safe_pitch_rad` |
| 37 | `apcr1nd_direct_enter_m` | 0.06 | `apcr1nd_direct_enter_m` |
| 38 | `apcr1nd_release_inner_m` | 0.03 | `apcr1nd_release_inner_m` |
| 39 | `apcr1nd_hold_outside_band` | 1.0 (True) | `apcr1nd_hold_outside_band` |
| 40 | `apcr1nd_converging_release_steps` | 15.0 | `apcr1nd_converging_release_steps` |

## 5. Files/Lines Changed

| File | Lines | Change |
|------|-------|--------|
| `wheeled_biped/controllers/k2_jax_controller.py` | 139-140 | Params fields extended (33→41) |
| `wheeled_biped/controllers/k2_jax_controller.py` | 163-174 | Params index constants added |
| `wheeled_biped/controllers/k2_jax_controller.py` | 177-262 | `pack_params_stage2` accepts 8 new APCR1ND params |
| `wheeled_biped/controllers/k2_jax_controller.py` | 243-275 | `unpack_params_stage2` reads APCR1ND params |
| `wheeled_biped/controllers/k2_jax_controller.py` | 628-726 | NEW: `k2_jax_apcr1nd_compute_gate()` function |
| `wheeled_biped/controllers/k2_jax_controller.py` | 765 | `apply_override` now includes `recenter_active` gate |
| `wheeled_biped/controllers/k2_jax_controller.py` | 1045-1062 | State fields extended (328→332) |
| `wheeled_biped/controllers/k2_jax_controller.py` | 1065-1071 | APCR1ND state index constants |
| `wheeled_biped/controllers/k2_jax_controller.py` | 1218-1227 | `pack_state_k2` accepts APCR1ND state |
| `wheeled_biped/controllers/k2_jax_controller.py` | 1386-1439 | `pack_state_from_python_k2` accepts APCR1ND state |
| `wheeled_biped/controllers/k2_jax_controller.py` | 1477-1509 | Controller step unpacks APCR1ND state |
| `wheeled_biped/controllers/k2_jax_controller.py` | 1567-1576 | Controller step unpacks APCR1ND params |
| `wheeled_biped/controllers/k2_jax_controller.py` | 1694-1719 | APCR1ND gate computed before sagittal assembly |
| `wheeled_biped/controllers/k2_jax_controller.py` | 1732 | Override call passes `recenter_active` gate |
| `wheeled_biped/controllers/k2_jax_controller.py` | 1843-1847 | APCR1ND state packed into new_state |
| `scripts/simulate_hierarchical_controller.py` | 5305-5310 | `_auth_sched` variable defined for APCR1ND params |
| `scripts/simulate_hierarchical_controller.py` | 5326-5333 | APCR1ND params passed to `pack_params_stage2` |
| `scripts/simulate_hierarchical_controller.py` | 5949-5953 | APCR1ND state captured for both-synced |
| `scripts/simulate_hierarchical_controller.py` | 6616-6621 | APCR1ND state passed to `pack_state_from_python_k2` |
| `scripts/simulate_hierarchical_controller.py` | 6707-6716 | APCR1ND diagnostics in both-synced output |
| `tests/test_k2_jax_step_parity.py` | 160-165 | APCR1ND state sources added |

## 6. State/Input/Params Layout Changes

| Layout | Before | After | Change |
|--------|--------|-------|--------|
| State | 328 | **332** | +4 (APCR1ND gating) |
| Params (Stage 2) | 33 | **41** | +8 (APCR1ND gating) |
| Input | 41 | 41 | No change |

## 7. Both-Synced Dynamic Parity Result

| Scenario | Pre-Fix | Post-Fix | Verdict |
|----------|---------|----------|---------|
| fixed_high_0p480 | 9.54e-08 | 9.54e-08 | PASS (unchanged) |
| fixed_low_0p330 | 9.54e-08 | 9.54e-08 | PASS (unchanged) |
| **ramp_up** | **7.88e-01** | **9.54e-08** | **PASS (FIXED)** |
| ramp_down | 9.54e-08 | 9.54e-08 | PASS (unchanged) |
| **gate_chatter** | **7.92e-01** | **9.54e-08** | **PASS (FIXED)** |
| push_fwd_90N | 3.00e+00 | 3.00e+00 | FAIL (pre-existing, separate issue) |
| push_bwd_90N | 3.30e+00 | 3.30e+00 | FAIL (pre-existing, separate issue) |

Dynamic height both-synced parity: **2/2 FIXED, 3/3 SCENARIOS PASS** (ramp_up, ramp_down, gate_chatter)

## 8. Functional Validation

No functional regressions expected. The APCR1ND gating fix only changes when the wheel damping override is applied — it now matches Python's decision exactly. At fixed heights, the override was already not applying (due to small position errors), so behavior is unchanged. At dynamic heights, JAX's override now applies at the same steps as Python's.

JAX backend remains opt-in. Python remains default.

## 9. Test Status

**131/131 tests pass.** No xfail, no skip. No regressions.

## 10. Final Classification

**K2_JAX_APCR1ND_DYNAMIC_HEIGHT_PARITY_PASS_PUSH_REMAINS_BLOCKED**

### What was fixed:
1. APCR1ND gating fully ported to JAX (`k2_jax_apcr1nd_compute_gate`)
2. Priority order bug fixed (release before activate, matching Python)
3. APCR1ND state fields added and synced in both-synced mode
4. Dynamic height parity: ramp_up and gate_chatter now pass at <1e-5

### What remains blocked:
1. Push parity (push_fwd_90N, push_bwd_90N) — **separate issue**, pre-existing before APCR1ND changes. 3.0 Nm max difference at wheel [4,9], consistent with composer max_tau_wheel clipping. Likely root cause: torque composer or sign convention difference during large push-induced errors.

### Hard constraints maintained:
- No gain tuning ✓
- No threshold relaxation ✓
- No empirical correction factors ✓
- No K2 control principle changes ✓
- No JAX default change ✓
- No Python K2 behavior modification ✓
- No fixed-height parity regression ✓
- No release lock gate breakage ✓
- Python K2 remains source of truth ✓
