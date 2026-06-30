# K2 JAX Default Promotion — Backend Selection Audit

**Date:** 2026-06-29
**Purpose:** Identify every point where controller backend default is selected, before changing the default.

---

## 1. Primary Default Selection Point (WILL BE CHANGED)

### 1.1 CLI argument default
- **File:** [scripts/simulate_hierarchical_controller.py:2600-2607](scripts/simulate_hierarchical_controller.py#L2600-L2607)
- **Current:** `default="python"`
- **Help text:** `"Controller backend: python (default, reference), jax (JIT-accelerated, opt-in), ..."`
- **Action:** Change default to conditional logic; update help text

### 1.2 Runtime fallback
- **File:** [scripts/simulate_hierarchical_controller.py:5287](scripts/simulate_hierarchical_controller.py#L5287)
- **Current:** `_backend = getattr(args, "controller_backend", "python")`
- **Action:** Update fallback to use the same conditional default logic

### 1.3 JAX requirement check
- **File:** [scripts/simulate_hierarchical_controller.py:5311-5313](scripts/simulate_hierarchical_controller.py#L5311-L5313)
- **Current:** `if not is_balance_core_mode(args):` → error exit
- **Note:** This check is correct and must remain — JAX requires balance-core mode

---

## 2. Validation Scripts with Their Own `--controller-backend` Defaults

These scripts have independent argument parsers with their own defaults. They must be reviewed but may not all need changing.

### 2.1 K2 Dynamic Height Gate-Crossing Validation
- **File:** [scripts/validate_k2_dynamic_height_gate_crossing.py:536](scripts/validate_k2_dynamic_height_gate_crossing.py#L536)
- **Current:** `default="python"`
- **Help:** `"Controller backend (default: python)"`
- **Decision:** Keep `default="python"` — this is a validation script, not the main CLI. Validation scripts should use explicit backend selection.

### 2.2 K2 Post-Promotion Long-Run Validation
- **File:** [scripts/validate_k2_post_promotion_long_run.py:686](scripts/validate_k2_post_promotion_long_run.py#L686)
- **Current:** `default="python"`
- **Help:** `"Controller backend: python (default), jax (JIT-accelerated, opt-in)"`
- **Decision:** Keep `default="python"` — validation script, explicit backend selection.

### 2.3 K2 Step C/E Fixed-Height Validation
- **File:** [scripts/validate_k2_step_c_e_fixed_height.py:154](scripts/validate_k2_step_c_e_fixed_height.py#L154)
- **Current:** `_backend = getattr(args, "controller_backend", "python") if 'args' in dir() else "python"`
- **Decision:** No change needed — correctly passes `--controller-backend` through to simulate script when non-python.

---

## 3. Phase/Validation Scripts (UNAFFECTED — Use Explicit Backend)

These scripts explicitly pass `--controller-backend` to the simulate script and are unaffected by default change:

| Script | Explicit Backend |
|--------|-----------------|
| `scripts/phase0_push_fwd_composer_failure_reproduction.py` | `--controller-backend both-synced` |
| `scripts/phase0_apcr1nd_baseline_parity.py` | `--controller-backend both-synced` |
| `scripts/phase0_abs_trim_trace.py` | `--controller-backend both-synced` |
| `scripts/phase1_full_9_scenario_both_synced_parity.py` | `--controller-backend both-synced` |
| `scripts/phase2_k2_jax_functional_validation.py` | Varies (python, jax, both-synced) |
| `scripts/phase3_k2_jax_long_run_validation.py` | `--controller-backend both-synced` |
| `scripts/phase6_abs_trim_full_parity.py` | `--controller-backend both-synced` |
| `scripts/phase6_k2_jax_performance_sanity.py` | N/A (Python-only import) |
| `scripts/trace_abs_ring_buffer.py` | `--controller-backend both-synced` |
| `scripts/validate_k2_jax_stage6h_full.py` | `--controller-backend both-synced` |
| `scripts/validate_k2_jax_backend.py` | Varies (python, jax) |
| `scripts/stage6l_phase1_lockstep_trace.py` | `--controller-backend jax` |

---

## 4. Benchmark Scripts

### 4.1 Stage 7 Benchmarks
- **File:** [scripts/stage7_run_benchmarks.py:612-613](scripts/stage7_run_benchmarks.py#L612-L613)
- **Current:** `default=None` (runs both python and jax when not specified)
- **Line 643:** `backends = [args_cli.backend] if args_cli.backend else ["python", "jax"]`
- **Decision:** No change needed — runs both backends explicitly.

---

## 5. Test Files

### 5.1 K2 JAX Backend CLI Tests
- **File:** [tests/test_k2_jax_backend_cli.py](tests/test_k2_jax_backend_cli.py)
- **Affected tests:**
  - `test_backend_default_is_python` (line 50) — Tests that no `--controller-backend` defaults to python. **MUST BE UPDATED** to test that K2 profile defaults to JAX.
  - All other tests use explicit `--controller-backend` flags and are unaffected.

### 5.2 Other Test Files
- **File:** [tests/test_stage1_behavior_unchanged.py:92](tests/test_stage1_behavior_unchanged.py#L92)
- **Current:** `assert report.get("backend") == "python"`
- **Note:** This test checks benchmark report backend field. If benchmark is run without explicit backend, the report backend field will change. **Review needed.**

---

## 6. Documentation

### 6.1 README.md
- Any mention of backend default must be updated.

### 6.2 CLI Help Text
- [scripts/simulate_hierarchical_controller.py:2604](scripts/simulate_hierarchical_controller.py#L2604) — Must be updated to reflect new default policy.

### 6.3 Validation Docs
- Various docs in `docs/validation/` reference Python as default — informational only, no code impact.

---

## 7. Summary of Changes Needed

### Must Change:
| # | File | Line(s) | Change |
|---|------|---------|--------|
| 1 | `scripts/simulate_hierarchical_controller.py` | 2600-2607 | Add conditional default logic for `--controller-backend` |
| 2 | `scripts/simulate_hierarchical_controller.py` | 5287 | Update `getattr` fallback or add explicit default resolution |
| 3 | `scripts/simulate_hierarchical_controller.py` | 2604 | Update help text |
| 4 | `tests/test_k2_jax_backend_cli.py` | 50-57 | Update `test_backend_default_is_python` → test JAX default for K2 profile |

### Should Review:
| # | File | Line(s) | Review |
|---|------|---------|--------|
| 5 | `tests/test_stage1_behavior_unchanged.py` | 92 | Check if backend assertion needs update |

### No Change Needed:
- All phase scripts (use explicit `--controller-backend`)
- `stage7_run_benchmarks.py` (runs both backends explicitly)
- `validate_k2_dynamic_height_gate_crossing.py` (validation script, keep explicit default)
- `validate_k2_post_promotion_long_run.py` (validation script, keep explicit default)
- `validate_k2_step_c_e_fixed_height.py` (passes through correctly)

---

## 8. Acceptance

- [x] Every backend default selection point identified (8 locations across 6 files)
- [x] Every affected test identified (2 test files)
- [x] Every affected doc/script identified
- [x] No hidden default remains unknown
- [x] Phase scripts verified as unaffected (use explicit flags)
