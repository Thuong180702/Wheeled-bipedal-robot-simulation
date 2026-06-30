# K2 JAX Final Parity Test Regression Report

**Date:** 2026-06-27
**Classification:** `K2_JAX_FINAL_PARITY_TESTS_PASS`

---

## 1. Summary

Full regression test suite executed after three parity fixes:
1. Notch filter state capture fix (Phase 1-2)
2. Sagittal velocity damping scale fix (Phase 3-4)
3. Support velocity input fix (previous session)

All 131 tests pass. No regressions, no xfail, no skip.

## 2. Command

```bash
pytest tests/test_k2_jax_*.py -v
```

## 3. Results

```
======================= 131 passed in 571.73s (0:09:31) ========================
```

### Test Files

| File | Tests | Result |
|------|-------|--------|
| test_k2_jax_backend_cli.py | 14 | 14 PASS |
| test_k2_jax_branch_activity_audit.py | 6 | 6 PASS |
| test_k2_jax_component_parity.py | 86 | 86 PASS |
| test_k2_jax_step_parity.py | 25 | 25 PASS |
| **Total** | **131** | **131 PASS** |

### Key Test Categories Verified

- **Backend CLI:** Python default, JAX opt-in, both-synced mode
- **Branch Audit:** No hidden torque/WBC, K2 params correct
- **Component Parity:** Notch coefficients (25 param combinations), smoothstep, torque composer (5 tests), height scheduling, outer loop, physics FF, low-band support, shape posture, lateral roll, yaw, mode-div, sagittal assembly
- **Step Parity:** JIT compiles, no NaN, state evolves, torques in limits, diag populated, state/diag field audits, state pack/unpack

## 4. New/Updated Tests

No new tests added. Existing component parity tests verify that:
- `test_k2_default_params` — params now include `k_velocity` and `velocity_damping_scale` (size 33, was 31)
- `test_params_size_consistent` — passes with updated params layout
- `test_params_fields_unique` — passes

## 5. Verification Targets

| Target | Status |
|--------|--------|
| No failures | ✓ 131/131 PASS |
| No xfail | ✓ 0 xfail |
| No skip | ✓ 0 skip |
| Python backend unchanged | ✓ (backend CLI tests pass) |
| JAX backend remains opt-in | ✓ (default_is_python test passes) |
| both-synced infrastructure intact | ✓ (smoke tests pass) |
| Params layout updated | ✓ (size 33, k_velocity, velocity_damping_scale) |

## 6. Classification

**`K2_JAX_FINAL_PARITY_TESTS_PASS`**

All 131 tests pass. No regressions. Params layout updated to 33 fields without breaking existing tests.
