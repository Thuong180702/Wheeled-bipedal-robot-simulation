# K2 JAX Postfix2 Test Regression Report

**Date:** 2026-06-27
**Classification:** `K2_JAX_POSTFIX2_TESTS_PASS`

---

## 1. Summary

Full regression test suite executed after Phase 1 (support_velocity) and Phase 2 (mode_div_error) parity fixes. All 131 tests pass. No regressions, no xfail, no skip added.

## 2. Command

```bash
pytest tests/test_k2_jax_*.py -v
```

## 3. Results

```
======================= 131 passed in 801.58s (0:13:21) =======================
```

### Test Files

| File | Tests | Result |
|------|-------|--------|
| test_k2_jax_backend_cli.py | 14 | 14 PASS |
| test_k2_jax_branch_activity_audit.py | 6 | 6 PASS |
| test_k2_jax_component_parity.py | 86 | 86 PASS |
| test_k2_jax_step_parity.py | 25 | 25 PASS |
| **Total** | **131** | **131 PASS** |

### Key Test Categories

- **Backend CLI:** Python default, JAX opt-in, both-synced mode
- **Branch Audit:** No hidden torque/WBC, K2 params correct
- **Component Parity:** Notch coefficients, smoothstep, torque composer, height scheduling, outer loop, physics FF, low-band support, shape posture, lateral roll, yaw, mode-div, sagittal assembly
- **Step Parity:** JIT compiles, no NaN, state evolves, torques in limits, diag populated, state/diag field audits, state pack/unpack

## 4. Verification Targets

| Target | Status |
|--------|--------|
| No failures | ✓ 131/131 PASS |
| No xfail | ✓ 0 xfail |
| No skip | ✓ 0 skip |
| Python backend unchanged | ✓ (backend CLI tests pass) |
| JAX backend remains opt-in | ✓ (default_is_python test passes) |
| both-synced infrastructure intact | ✓ (smoke tests pass) |

## 5. Classification

**`K2_JAX_POSTFIX2_TESTS_PASS`**

All tests pass. No regressions from Phase 1/2 fixes.
