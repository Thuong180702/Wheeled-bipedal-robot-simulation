# K2 JAX Post-Fix Test Regression Report

**Date:** 2026-06-27
**Classification:** `K2_JAX_ALL_TESTS_PASS`

---

## 1. Summary

All 131 tests pass after fixing test reference to match D12 (calibrated outer loop v1→v2). Python backend unchanged. JAX backend remains opt-in.

---

## 2. Test Suite Results

```
tests/test_k2_jax_step_parity.py         17/17 PASS
tests/test_k2_jax_backend_cli.py         14/14 PASS
tests/test_k2_jax_component_parity.py    94/94 PASS (2 initially failed, FIXED)
tests/test_k2_jax_branch_activity_audit.py 6/6 PASS
```

**Total: 131/131 PASS (0 xfail, 0 skip)**

---

## 3. Initial Failures and Resolution

### Failure 1: `test_kp_grid_error`
- **Error:** Kp grid max error: 5.25e-01 (threshold 1e-06)
- **Root cause:** `pchip_refs` fixture imported from `calibrated_outer_loop_functions` (v1), but `build_calibrated_grid_params()` now uses v2 (D12 fix). v2 Kp at 0.465m=1.000 vs v1=1.350 (diff=0.350), at 0.480m=1.050 vs v1=1.575 (diff=0.525).
- **Fix:** Updated `pchip_refs` fixture to import from `calibrated_outer_loop_functions_v2` (matching D12).
- **File:** [tests/test_k2_jax_component_parity.py:682](tests/test_k2_jax_component_parity.py#L682)

### Failure 2: `test_kd_grid_error`
- **Error:** Kd grid max error: 5.00e-02 (threshold 1e-06)
- **Root cause:** Same as above. v2 Kd at 0.480m=0.000 vs v1=0.050 (diff=0.050).
- **Fix:** Same as above — updated reference to v2.

---

## 4. Key Test Groups

### Step Parity Tests (17/17)
All JIT compilation, state evolution, output bounds, and multi-step tests pass.

### Backend CLI Tests (14/14)
Backend flag parsing, Python/JAX smoke, Stage7 benchmark, and Python-backend-unchanged tests pass.

### Component Parity Tests (94/94)
- Notch coefficient parity: 36/36 (D1 fixed — bit-identical coefficients)
- Notch update parity: 5/5 (10K random, impulse, stream tests)
- Smoothstep gate parity: 3/3
- Torque composer parity: 5/5
- State pack/unpack: 2+3=5
- Params pack/unpack: 4 (including D2/D3 new fields)
- Height scheduling: 11/11
- Outer loop: 3/3 (basic, deadband, saturation)
- Calibrated outer loop: 6/6 (FIXED — D12 v2 reference)
- Physics FF: 2/2
- Low-band support: 3/3
- Shape posture PD: 1/1
- Lateral roll: 2/2
- Yaw controller: 1/1
- Mode div: 3/3 (D2/D3 confirmed)
- Sagittal assembly: 2/2
- Rate limit/lowpass: 2/2
- Index constants: 2/2

### Branch Activity Audit Tests (6/6 — Phase 7 pre-check)
- `test_disabled_strategies_inactive` PASS — all disabled mechanisms confirmed zero
- `test_enabled_strategies_active` PASS — all enabled mechanisms active
- `test_k2_notch_params_correct` PASS
- `test_no_wbc_enabled` PASS — no WBC leakage
- `test_no_hidden_torque_flags` PASS — no hidden torque
- `test_branch_audit_classification` PASS — 0 UNEXPECTED_ACTIVE

---

## 5. Python Backend Verification

`test_python_backend_completes` PASS — Python K2 behavior unchanged.

---

## 6. JAX Backend Opt-In Status

JAX remains opt-in:
- `--controller-backend` defaults to `python`
- `test_backend_default_is_python` PASS
- No default changed

---

## 7. Files Modified for Phase 2

| File | Change |
|------|--------|
| [tests/test_k2_jax_component_parity.py:682](tests/test_k2_jax_component_parity.py#L682) | Updated `pchip_refs` fixture: v1→v2 import (D12 fix reference update) |

---

## 8. Classification

**`K2_JAX_ALL_TESTS_PASS`**

All 131 K2 JAX tests pass. No xfail added. No tolerance relaxed. Python backend unchanged. JAX backend remains opt-in.
