# K2 JAX Post-Strict-Clone Test Regression

## Date: 2026-06-28

## Classification: **K2_JAX_POST_STRICT_CLONE_TEST_REGRESSION_PASS**

---

## 1. Run

```bash
pytest tests/test_k2_jax_*.py -v
```

```
============================= test session starts =============================
platform win32 -- Python 3.10.2, pytest-9.0.2
collected 131 items

tests/test_k2_jax_backend_cli.py .............. [ 14/131]
tests/test_k2_jax_branch_activity_audit.py ...... [ 20/131]
tests/test_k2_jax_component_parity.py .........................................
  ............................................................................ [ 86/131]
tests/test_k2_jax_step_parity.py ....................................... [131/131]

======================= 131 passed in 694.07s (0:11:34) =======================
```

## 2. Summary

| Metric | Value |
|--------|-------|
| Total tests | 131 |
| Passed | 131 |
| Failed | 0 |
| Xfail | 0 |
| Skip | 0 |
| Duration | 694.07 s |

## 3. Verification Gates

| Gate | Result |
|------|--------|
| 131/131 pass | ✓ |
| No xfail | ✓ |
| No skip | ✓ |
| Python backend default | ✓ (tests confirm default is "python") |
| JAX backend opt-in | ✓ (tests confirm JAX requires explicit flag) |
| Both-synced mode available | ✓ |
| State-synced parity tests pass | ✓ |
| Component parity tests pass | ✓ (notch, smoothstep, composer, shape, yaw, mode-div, sagittal, lateral) |
| Step parity tests pass | ✓ (JIT compile, no NaN, state evolves, torque within limits, diag populated) |
| Backend CLI tests pass | ✓ (help, python, jax, default) |
| Branch activity audit pass | ✓ (disabled inactive, enabled active, no WBC, no hidden torque) |

## 4. Classification

### K2_JAX_POST_STRICT_CLONE_TEST_REGRESSION_PASS

All 131 tests pass with zero failures, zero xfail, zero skip. No regression introduced by the two hip-yaw fixes. Python remains default backend. JAX remains opt-in.
