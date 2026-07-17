# K2 JAX Default Promotion — Scope Lock

**Date:** 2026-06-29
**Previous Classification:** K2_JAX_FULL_SEMANTIC_PORT_RELEASE_HARDENING_PASS
**This Document:** Promotion scope lock before default switch

---

## 1. Previous Classification Summary

K2_JAX_FULL_SEMANTIC_PORT_RELEASE_HARDENING_PASS was achieved with:

- 9/9 both-synced parity (max diff 4.70e-07, all below 1e-5 threshold)
- push_fwd_90N: 9.54e-08 PASS
- push_bwd_90N: 4.70e-07 PASS
- APCR1ND wheel damping override parity: PASS
- ABS trim parity: PASS
- MODE_DIV parity: PASS
- APCR1ND/T6I/contact_valid semantics: PASS
- Functional validation: 12/12 PASS
- Long-run validation: PASS (5 heights)
- 147/147 tests PASS
- Hot-step performance: ~0.185 ms (well below 10 ms)
- No hidden torque/WBC
- No NaN
- No threshold relaxation
- No gain tuning
- Python K2 remains source of truth

---

## 2. Validation Evidence Summary

| Evidence | Result |
|----------|--------|
| 9-scenario both-synced parity | 9/9 PASS |
| push_fwd_90N parity | 9.54e-08 |
| push_bwd_90N parity | 4.70e-07 |
| APCR1ND WD override parity | PASS |
| ABS trim parity | PASS |
| MODE_DIV state parity | PASS |
| Functional validation (fixed/dynamic/push) | 12/12 PASS |
| Long-run validation (5 heights) | PASS |
| Test suite | 147/147 PASS |
| Hot-step performance | ~0.185 ms |
| 6 semantic mismatches fixed | All resolved |

---

## 3. Promotion Scope

### INCLUDED — JAX becomes default for:

| Axis | Value |
|------|-------|
| Controller mode | `balance-core` ONLY |
| Sagittal controller | `velocity-damped` ONLY |
| Authority profile (CLI) | `k2_notch_low_q_v1` |
| Authority profile (code) | `K2_NOTCH_LOW_Q_V1` |
| WBC/hidden torque path | NONE (balance-core is WBC-free) |

### EXCLUDED — JAX is NOT default for:

| Axis | Rationale |
|------|-----------|
| Controller mode `legacy` | Not validated for JAX |
| Controller mode `standing-balance` | Not validated for JAX |
| Sagittal controller `baseline` | Not validated for JAX |
| Non-K2 authority profiles (K1, J*, L*, M*, N*, unified, etc.) | Not validated for JAX |
| K2 profiles other than `k2_notch_low_q_v1` (e.g., `k2_wheel_vel_notch_v1`, `k3_*`) | Not validated for JAX |
| Any profile with WBC/hidden torque path | balance-core is WBC-free |
| Hardware/sim-to-real | Not validated |

---

## 4. Fallback Policy

- Python backend remains fully available via `--controller-backend python`
- Python backend code is NOT deleted or modified
- If user explicitly sets `--controller-backend python`, Python is used regardless of profile
- If the profile/controller combination is outside validated K2 scope, backend defaults to Python
- `both-synced` backend remains available for parity/debug via `--controller-backend both-synced`

---

## 5. Rollback Policy

To roll back the default to Python:

1. Change the default value of `--controller-backend` in `scripts/simulate_hierarchical_controller.py` back to `"python"`
2. Or: users can always pass `--controller-backend python` explicitly
3. No code needs to be restored — Python backend is preserved

---

## 6. Acceptance

- [x] Promotion scope is explicit
- [x] Python fallback policy is explicit
- [x] No unvalidated path included
- [x] Rollback policy is documented
