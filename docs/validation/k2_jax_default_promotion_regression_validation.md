# K2 JAX Default Promotion — Regression Validation

**Date:** 2026-06-29
**Phase:** 4 — Regression Validation After Default Switch

---

## A. Implicit Default K2 Smoke (No `--controller-backend`)

All scenarios run without explicit `--controller-backend` flag. Backend resolved to JAX by default policy.

| Scenario | Backend | Status | Log Confirmation |
|----------|---------|--------|-----------------|
| fixed_high_0p480 | jax (default) | PASS | `jax (default for validated K2...)` |
| fixed_low_0p330 | jax (default) | PASS | `jax (default for validated K2...)` |
| ramp_up | jax (default) | PASS | `jax (default for validated K2...)` |
| push_fwd_90N | jax (default) | PASS | `jax (default for validated K2...)` |
| push_bwd_90N | jax (default) | PASS | `jax (default for validated K2...)` |

**5/5 PASS** — No NaN, no hidden torque/WBC, no fall.

---

## B. Explicit Python Fallback Smoke

Same scenarios with `--controller-backend python`.

| Scenario | Backend | Status | Log Confirmation |
|----------|---------|--------|-----------------|
| fixed_high_0p480 | python (explicit) | PASS | `python (explicit user override)` |
| fixed_low_0p330 | python (explicit) | PASS | `python (explicit user override)` |
| push_fwd_90N | python (explicit) | PASS | `python (explicit user override)` |
| push_bwd_90N | python (explicit) | PASS | `python (explicit user override)` |

**4/4 PASS** — Python fallback works correctly.

---

## C. Explicit Both-Synced Smoke

With `--controller-backend both-synced`.

| Scenario | Backend | Status | Max 10-dim Diff |
|----------|---------|--------|-----------------|
| fixed_high_0p480 | both-synced (explicit) | PASS | 9.54e-08 |
| push_fwd_90N | both-synced (explicit) | PASS | 9.54e-08 |
| push_bwd_90N | both-synced (explicit) | PASS | 4.20e-07 |

**3/3 PASS** — All diffs well below 1e-5 threshold. Both-synced parity maintained.

---

## Summary

| Category | Scenarios | Passed | Failed |
|----------|-----------|--------|--------|
| Implicit JAX default | 5 | 5 | 0 |
| Explicit Python fallback | 4 | 4 | 0 |
| Explicit both-synced | 3 | 3 | 0 |
| **Total** | **12** | **12** | **0** |

---

## Acceptance

- [x] All implicit-default scenarios pass
- [x] Logs show JAX selected by default for K2 profile
- [x] No NaN
- [x] No hidden torque/WBC
- [x] No fall
- [x] Python fallback works with explicit override
- [x] Both-synced parity maintained (<1e-5)
