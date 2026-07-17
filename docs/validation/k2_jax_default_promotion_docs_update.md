# K2 JAX Default Promotion — Documentation Update

**Date:** 2026-06-29
**Phase:** 6 — Documentation and CLI Help Update

---

## 1. Changes Made

### 1.1 CLI Help Text (simulate_hierarchical_controller.py)

Updated `--controller-backend` help text to describe new default policy:

```
Controller backend. Default (when not specified): JAX for validated K2
balance-core + velocity-damped + k2_notch_low_q_v1 profile; Python for
all other profiles. Choices: python (reference), jax (JIT-accelerated),
both (teacher-forcing with independent state), both-synced
(teacher-forcing with state-synced JAX). JAX backend requires
balance-core mode. Python backend is always available. Use
--controller-backend python to force Python backend.
```

### 1.2 Backend Decision Logging

Script now prints backend decision at startup:

```
[BACKEND] Controller backend: jax (default for validated K2 balance-core + velocity-damped + k2_notch_low_q_v1 profile)
[BACKEND] Controller backend: python (explicit user override)
[BACKEND] Controller backend: python (default fallback for non-validated profile)
```

### 1.3 README — No Changes Required

The README does not contain specific controller backend default instructions. The CLI help text is the primary user-facing documentation for backend selection.

---

## 2. Default Promotion Scope (User-Facing Summary)

- **JAX is now the default backend** for `--controller-mode balance-core --sagittal-controller velocity-damped --vd-sagittal-authority-profile k2_notch_low_q_v1`
- **Python remains available** via `--controller-backend python`
- **both-synced remains available** via `--controller-backend both-synced` (for parity/debug)
- **Non-K2 profiles are NOT promoted** — they continue to default to Python
- **Legacy mode is NOT promoted** — continues to default to Python

---

## 3. Validation Evidence Summary

| Evidence | Detail |
|----------|--------|
| Both-synced parity | 9/9 scenarios, all < 1e-5 |
| Functional validation | 12/12 scenarios pass |
| Long-run validation | 5 heights pass |
| Tests | 164/164 pass |
| Hot-step performance | ~0.185 ms (<< 10 ms) |
| APCR1ND parity | PASS |
| ABS trim parity | PASS |
| MODE_DIV parity | PASS |

---

## 4. Rollback Instructions

To revert to Python backend default:

1. **Per-invocation:** Add `--controller-backend python` to any command
2. **Permanent:** Revert `--controller-backend` default in `scripts/simulate_hierarchical_controller.py` from `None` (auto-resolve) back to `"python"`

No code removal needed — Python backend is fully preserved.

---

## 5. Acceptance

- [x] CLI help accurate (describes default promotion policy)
- [x] Backend decision logged with source
- [x] Fallback instructions explicit
- [x] No claim that JAX is universally default outside validated K2 scope
- [x] Python backend availability documented
- [x] both-synced availability documented
