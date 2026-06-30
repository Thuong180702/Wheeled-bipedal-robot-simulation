# K2 JAX Default Promotion — Final Report

**Date:** 2026-06-29
**Final Classification:** **K2_JAX_DEFAULT_PROMOTION_PASS**

---

## 1. Previous Classification

**K2_JAX_FULL_SEMANTIC_PORT_RELEASE_HARDENING_PASS** (2026-06-29)

Achieved with:
- 9/9 both-synced parity (max diff 4.70e-07)
- 6 semantic mismatches fixed (contact_valid, converging steps, recenter_held, position cap boost, contact_valid in position cap, JAX contact_valid input definition)
- 147/147 tests PASS
- Functional validation 12/12 PASS
- Long-run validation PASS
- Hot-step ~0.185 ms

---

## 2. Promotion Scope

### JAX is now default for:

- **Controller mode:** `balance-core` only
- **Sagittal controller:** `velocity-damped` only
- **Authority profile:** `k2_notch_low_q_v1` / `K2_NOTCH_LOW_Q_V1` only
- **No WBC/hidden torque path** (balance-core is WBC-free)

### JAX is NOT default for:

- Controller mode `legacy` or `standing-balance`
- Sagittal controller `baseline`
- Non-K2 profiles (K1, J*, L*, M*, N*, unified, etc.)
- Any profile with WBC/hidden torque

---

## 3. Code Changes

### File: `scripts/simulate_hierarchical_controller.py`

1. **Added scope constants and detection functions** (after line 2097):
   - `_K2_JAX_DEFAULT_SAGITTAL_CONTROLLER = "velocity-damped"`
   - `_K2_JAX_DEFAULT_AUTHORITY_PROFILE = "k2_notch_low_q_v1"`
   - `_is_validated_k2_jax_scope(args)` — checks if config is in validated K2 scope
   - `resolve_controller_backend(args)` — resolves backend with promotion policy

2. **Changed argparse `--controller-backend` default:** `"python"` → `None` (resolved post-parse)

3. **Updated help text** to describe new default policy

4. **Added backend resolution** after `parse_args()` with logging

5. **Updated runtime fallback** to safety-net pattern

### File: `tests/test_k2_jax_backend_cli.py`

- Replaced `test_backend_default_is_python` with 7 new tests covering default promotion behavior
- Added `test_non_k2_velocity_damped_other_profile_defaults_to_python`

### File: `tests/test_stage1_behavior_unchanged.py`

- Added `--controller-backend python` to `BASE_K2_ARGS` (test explicitly tests Python controller)

---

## 4. Backend Selection Behavior

| User Input | Profile | Result | Log |
|-----------|---------|--------|-----|
| (none) | K2 validated | `jax` | "default for validated K2..." |
| (none) | Non-K2 | `python` | "default fallback for non-validated profile" |
| `--controller-backend python` | Any | `python` | "explicit user override" |
| `--controller-backend jax` | Any | `jax` | "explicit user override" |
| `--controller-backend both-synced` | Any | `both-synced` | "explicit user override" |
| `--controller-backend both` | Any | `both` | "explicit user override" |

---

## 5. Explicit Override Behavior

- `--controller-backend python` → Python backend, regardless of profile
- `--controller-backend jax` → JAX backend, requires balance-core mode
- `--controller-backend both-synced` → both-synced parity mode, requires balance-core mode
- Invalid backend values → argparse rejects with error

---

## 6. Full Test Results

| Suite | Tests | Passed | Failed |
|-------|-------|--------|--------|
| test_k2_jax_backend_cli.py | 20 | 20 | 0 |
| test_k2_jax_component_parity.py | 85 | 85 | 0 |
| test_k2_jax_step_parity.py | 55 | 55 | 0 |
| test_stage1_behavior_unchanged.py | 4 | 4 | 0 |
| **Total** | **164** | **164** | **0** |

---

## 7. Implicit-Default Validation

| Scenario | Backend | Status |
|----------|---------|--------|
| fixed_high_0p480 | jax (default) | PASS |
| fixed_low_0p330 | jax (default) | PASS |
| ramp_up | jax (default) | PASS |
| push_fwd_90N | jax (default) | PASS |
| push_bwd_90N | jax (default) | PASS |

**5/5 PASS** — No NaN, no fall, no hidden torque/WBC.

---

## 8. Explicit Python Fallback Validation

| Scenario | Status |
|----------|--------|
| fixed_high_0p480 | PASS |
| fixed_low_0p330 | PASS |
| push_fwd_90N | PASS |
| push_bwd_90N | PASS |

**4/4 PASS**

---

## 9. Both-Synced Validation

| Scenario | Max Diff | Status |
|----------|----------|--------|
| fixed_high_0p480 | 9.54e-08 | PASS |
| push_fwd_90N | 9.54e-08 | PASS |
| push_bwd_90N | 4.20e-07 | PASS |

**3/3 PASS** — All below 1e-5 strict parity threshold.

---

## 10. Performance Sanity

- Hot-step timing: ~0.185 ms (well below 10 ms target)
- JIT compilation: ~1.4 s (one-time, per session)
- No repeated recompilation
- No performance regression

---

## 11. Documentation Updated

- [x] CLI help text updated (describes promotion policy)
- [x] Backend decision logged at startup
- [x] Scope lock document: `docs/validation/k2_jax_default_promotion_scope_lock.md`
- [x] Selection audit: `docs/validation/k2_jax_default_selection_audit.md`
- [x] Implementation report: `docs/validation/k2_jax_default_switch_implementation_report.md`
- [x] Test report: `docs/validation/k2_jax_default_promotion_test_report.md`
- [x] Regression validation: `docs/validation/k2_jax_default_promotion_regression_validation.md`
- [x] Release guard: `docs/validation/k2_jax_default_promotion_release_guard.md`
- [x] Docs update: `docs/validation/k2_jax_default_promotion_docs_update.md`
- [x] This final report

---

## 12. Rollback Plan

**Method 1 (per-invocation):**
```bash
--controller-backend python
```

**Method 2 (permanent):**
In `scripts/simulate_hierarchical_controller.py`, change:
- `default=None` → `default="python"` for `--controller-backend`
- Remove or disable `resolve_controller_backend()` call after `parse_args()`

No Python backend code was removed — rollback is trivial.

---

## 13. Final Classification

### **K2_JAX_DEFAULT_PROMOTION_PASS**

**Justification:**

| Gate | Status |
|------|--------|
| JAX is default for validated K2 scope | YES |
| Python fallback works | YES |
| both-synced still works | YES |
| Unvalidated profiles NOT silently promoted | YES |
| Full tests pass (164/164) | YES |
| Release guard passes | YES |
| Docs updated | YES |
| Rollback plan documented | YES |
| Explicit Python fallback passes | YES |
| Both-synced parity maintained | YES |

**Verification:** All non-negotiable gates met.

**Correct claim:** JAX is semantically equivalent to Python K2 in validated scenarios (max parity diff < 1e-5 across all 9 scenarios) and faster at the hot-step level (~0.185 ms vs Python).

---

## Appendix: Deliverables Checklist

| Phase | Document | Status |
|-------|----------|--------|
| 0 | `k2_jax_default_promotion_scope_lock.md` | Created |
| 1 | `k2_jax_default_selection_audit.md` | Created |
| 2 | `k2_jax_default_switch_implementation_report.md` | Created |
| 3 | `k2_jax_default_promotion_test_report.md` | Created |
| 4 | `k2_jax_default_promotion_regression_validation.md` | Created |
| 5 | `k2_jax_default_promotion_release_guard.md` | Created |
| 6 | `k2_jax_default_promotion_docs_update.md` | Created |
| 7 | `k2_jax_default_promotion_final_report.md` | This document |
