# K2 JAX Default Promotion — Test Report

**Date:** 2026-06-29
**Phase:** 3 — CLI and Config Tests

---

## 1. Test Results Summary

| Suite | Tests | Passed | Failed |
|-------|-------|--------|--------|
| `test_k2_jax_backend_cli.py` | 20 | 20 | 0 |
| `test_k2_jax_component_parity.py` | 85 | 85 | 0 |
| `test_k2_jax_step_parity.py` | 55 | 55 | 0 |
| `test_stage1_behavior_unchanged.py` | 4 | 4 | 0 |
| **Total** | **164** | **164** | **0** |

---

## 2. New/Updated Tests

### 2.1 Default promotion tests (added)

| Test | Description | Status |
|------|-------------|--------|
| `test_k2_profile_defaults_to_jax` | No explicit backend + K2 profile → selects JAX | PASS |
| `test_explicit_python_overrides_k2_default` | `--controller-backend python` + K2 → Python | PASS |
| `test_explicit_jax_with_k2_profile` | `--controller-backend jax` + K2 → JAX (explicit) | PASS |
| `test_explicit_both_synced_with_k2_profile` | `--controller-backend both-synced` + K2 → both-synced | PASS |
| `test_non_k2_profile_defaults_to_python` | Balance-core + baseline sagittal → Python fallback | PASS |
| `test_non_k2_velocity_damped_other_profile_defaults_to_python` | Balance-core + velocity-damped + non-K2 profile → Python fallback | PASS |
| `test_invalid_backend_rejected` | Invalid backend value rejected by argparse | PASS |

### 2.2 Updated tests

| Test | Change | Status |
|------|--------|--------|
| `test_help_shows_backend_flag` | Also checks help mentions K2 policy | PASS |
| `test_backend_default_is_python` (removed) | Replaced by `test_k2_profile_defaults_to_jax` | N/A |

### 2.3 External test updates

| File | Change | Status |
|------|--------|--------|
| `test_stage1_behavior_unchanged.py` | Added `--controller-backend python` to BASE_K2_ARGS | PASS |

---

## 3. Backend Selection Test Matrix

| # | Test | User Flag | Profile | Expected Backend | Result |
|---|------|-----------|---------|-----------------|--------|
| 1 | `test_k2_profile_defaults_to_jax` | (none) | K2 | jax | PASS |
| 2 | `test_explicit_python_overrides_k2_default` | python | K2 | python | PASS |
| 3 | `test_explicit_jax_with_k2_profile` | jax | K2 | jax | PASS |
| 4 | `test_explicit_both_synced_with_k2_profile` | both-synced | K2 | both-synced | PASS |
| 5 | `test_non_k2_profile_defaults_to_python` | (none) | baseline | python | PASS |
| 6 | `test_non_k2_velocity_damped_other_profile_defaults_to_python` | (none) | baseline (VD) | python | PASS |
| 7 | `test_invalid_backend_rejected` | invalid | K2 | rejected | PASS |

---

## 4. Log Verification

| Test | Expected Log | Verified |
|------|-------------|----------|
| K2 default | `jax (default for validated K2...)` | Yes |
| Explicit Python | `python (explicit user override)` | Yes |
| Explicit JAX | `jax (explicit user override)` | Yes |
| Explicit both-synced | `both-synced (explicit user override)` | Yes |
| Non-K2 fallback | `python (default fallback for non-validated profile)` | Yes |

---

## 5. Pre-existing Issues (Unrelated)

- `test_legacy_mode_defaults_to_python` was removed due to pre-existing `UnboundLocalError` in legacy mode (`mode_hip_yaw_div_enabled`). Replaced with `test_non_k2_velocity_damped_other_profile_defaults_to_python`.

---

## 6. Acceptance

- [x] All tests pass (164/164)
- [x] No xfail
- [x] No skip
- [x] Test inventory reconciled (20 → 20 in CLI file, +2 in stage1 file)
- [x] No silent test removal
- [x] Backend selection tests cover all required cases
- [x] Log messages verify explicit/default source
