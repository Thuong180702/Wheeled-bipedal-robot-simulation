# K2 JAX Postfix2 Functional Smoke Report

**Date:** 2026-06-27
**Classification:** `K2_JAX_FUNCTIONAL_SMOKE_POSTFIX2_PASS`

---

## 1. Summary

Functional smoke recheck after Phase 1 (support_velocity) and Phase 2 (mode_div_error) parity fixes. All functional gates pass. No behavior regressions.

## 2. Fixed-Height Smoke

| Height | Steps | Backend | Fell | Status |
|--------|-------|---------|------|--------|
| high_0p480 | 6000 | JAX | No | PASS |
| low_0p330 | 6000 | JAX | No | PASS |

Both heights pass long-run-equivalent validation.

## 3. Long-Run Extended Validation

Executed as Phase 4 — 5 heights × 6000 steps, all JAX backend:

| Height | Steps | Fell | Status |
|--------|-------|------|--------|
| low_0p330 | 6000/6000 | No | PASS |
| mid_0p400 | 6000/6000 | No | PASS |
| high_0p430 | 6000/6000 | No | PASS |
| high_0p450 | 6000/6000 | No | PASS |
| high_0p480 | 6000/6000 | No | PASS |

## 4. Regression Tests

131/131 tests pass. 0 xfail, 0 skip.

## 5. Previously Validated (Unchanged by Phase 1/2 Fixes)

| Test | Result | Source |
|------|--------|--------|
| Fixed-height 17/17 | PASS | Prior 9-phase audit (Stage 6H) |
| Push recovery 4/4 | PASS | Prior validation (Stage 6I) |
| Dynamic height 5/5 | PASS | Prior validation (Stage 6K) |
| Branch/torque audit 6/6 | PASS | Prior audit |
| Performance hot-step 0.273ms | PASS | Prior benchmark (Stage 7) |

## 6. Acceptance Criteria

| Criterion | Status |
|-----------|--------|
| No falls | ✓ |
| No NaN | ✓ |
| No actuator violations | ✓ (torque < 16 Nm per actuator) |
| No hidden torque/WBC | ✓ |
| Metrics not worse than previous JAX functional pass | ✓ |
| Python backend unchanged | ✓ (131/131 tests) |
| JAX backend remains opt-in | ✓ |

## 7. Classification

**`K2_JAX_FUNCTIONAL_SMOKE_POSTFIX2_PASS`**

All functional gates pass. No regressions from Phase 1/2 fixes. JAX backend validated across 5 heights × 6000 steps.
