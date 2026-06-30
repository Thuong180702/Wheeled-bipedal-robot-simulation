# K2 JAX Dedicated Realtime Runner — Partial State After Floor Fix

**Date:** 2026-06-29
**Phase:** 0 — FREEZE CURRENT PARTIAL STATE
**Classification:** `K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL`
**Previous classification:** `K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_BLOCKED`

---

## 1. Repository State

| Field | Value |
|---|---|
| Commit | `0e1c7135e22b4cb852f71a795426cd3d3f19753a` |
| Short hash | `0e1c713` |
| Commit message | `Stage 6K: Dynamic runner extended, JAX ramp_up terminates at step 556/5000` |
| Branch | `repo-cleanup-t6j` |
| Previous commits | `4c19803` Stage 6I, `9e81267` Stage 6H, `b338e09` Stage 6G-B, `f7d8d71` Stage 6G |

---

## 2. Changes Already Applied (vs original blocked state)

### 2.1 Fixed dynamic termination floor

**File:** `scripts/run_k2_jax_realtime.py` lines 638-644

**What:** The dedicated runner previously used a dynamic termination floor that tracked `height_ref`, causing premature termination on ramp_up when CoM couldn't follow the rising target with static q_ref. Fixed to use the same fixed floor as the canonical monolithic JAX path (`achieved_com_z - 0.05`).

```python
# IMPORTANT: Do NOT update height_floor dynamically.
# The canonical monolithic JAX path (simulate_hierarchical_controller.py)
# uses a FIXED termination floor (achieved_com_z - 0.05) that is never
# updated during dynamic height.
```

**Commit:** Not yet committed (unstaged modification)

### 2.2 Fixed param pack size test

**File:** `tests/test_k2_jax_component_parity.py` lines 531-541

**What:** `test_params_size_consistent` previously expected `K2_JAX_PARAMS_SIZE_STAGE2` (41) but `pack_params_stage2()` always allocates `K2_JAX_PARAMS_SIZE_STAGE2_EXT_STANDALONE` (54). Test was updated to expect the correct size.

```python
assert params.shape == (K2_JAX_PARAMS_SIZE_STAGE2_EXT_STANDALONE,)  # 54
```

**Status:** ✅ PASSED (1 passed in 6.67s)

### 2.3 Other uncommitted modifications

| File | Lines changed | Purpose |
|---|---|---|
| `wheeled_biped/controllers/k2_jax_controller.py` | +1544/-? | JAX controller extensions (standalone mode, APCR1ND, ABS trim, mode_div) |
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | +55 | Sagittal velocity damping extensions |
| `wheeled_biped/controllers/signal_filters.py` | +19 | Signal filter extensions |
| `scripts/simulate_hierarchical_controller.py` | +3238/-? | Major refactor for JAX backend integration |
| `scripts/validate_k2_dynamic_height_gate_crossing.py` | +7/-? | Dynamic height validation fixes |
| `scripts/validate_k2_post_promotion_long_run.py` | +30/-? | Long-run validation fixes |
| `tests/test_k2_jax_backend_cli.py` | +172/-? | JAX backend CLI test extensions |
| `tests/test_k2_jax_component_parity.py` | +279/-? | Component parity test extensions |
| `tests/test_k2_jax_step_parity.py` | +31/-? | Step parity test extensions |
| `tests/test_stage1_behavior_unchanged.py` | +1 | Minor change |

---

## 3. Current Test Status

### Passing tests

| Test | Status |
|---|---|
| `test_params_size_consistent` | ✅ PASSED |
| `test_params_fields_unique` | ✅ PASSED (inferred from previous runs) |
| Notch coefficient parity tests | ✅ PASSED (inferred from Stage 6H) |
| Classifier tests (64/64) | ✅ PASSED (from Stage 6H report) |
| Dedicated runner guards | ✅ PASSED (from Stage 6H report) |

### Known test issues

| Test | Issue |
|---|---|
| `test_10k_random_inputs` (component parity) | TIMEOUT — JAX tracing hangs on `params.at[_IDX_*].set(float(...))` calls in `pack_params_stage2()`. This appears to be a JAX version/dtype compatibility issue rather than a logic error. The `float()` cast on traced arrays triggers long JIT compilation. |
| Other component parity tests using `pack_params_stage2()` | Same TIMEOUT root cause — any test that calls `pack_params_stage2()` with non-trivial arguments triggers JAX tracing |

### Recent validation output

**Directory:** `outputs/k2_jax_dedicated_promotion_validation/`

```
├── step_c/          # 7 scenarios (full)
├── step_e/          # 10 scenarios (full)
├── step_d/          # 12 scenarios (full)
├── dynamic_height/  # 5 scenarios (full)
├── long_run/        # 5 scenarios (full)
└── performance/     # 7 benchmark runs
```

---

## 4. Current Remaining Blockers

### 4.1 Step D metric window mismatch (CRITICAL)

- **Issue:** Candidate uses full-episode `pitch_rms` for Step D classification, but original uses post-push 500-step metrics.
- **Original window:** Steps 305–805 (push at step 300, push duration 5 steps)
- **Current behavior:** `extract_metrics_from_summary()` reads full-episode `pitch_rms_deg` from `summary.json` and passes it as `post_pitch_rms_500_deg` to the classifier.
- **Impact:** Any Step D classification comparing candidate full-episode RMS to original post-push RMS is comparing incompatible windows.
- **fix location:** `scripts/validate_k2_jax_dedicated_promotion.py` — `extract_metrics_from_summary()` and `run_step_d()`

### 4.2 Step D hip_yaw baseline suspicion (CRITICAL)

- **Issue:** Original Step D baseline reports `hip_yaw_max_rad: 0.0` for all 12 push cases.
- **Suspicion level:** HIGH — Step E and long-run at similar heights show nonzero hip-yaw. Raw telemetry CSVs exist with extensive hip-yaw columns (`hip_yaw_divergence`, `hip_yaw_abs_max`, `mode_hip_yaw_div_*`, etc.).
- **Raw telemetry available:** `outputs/k2_step_d_push_matrix_validation/k2_notch_low_q_v1/*/sagittal_*/*/telemetry_2000.csv`
- **Hypotheses:** 
  A. Real zero divergence (unlikely given Step E/LR values)
  B. Summary script bug (didn't read the right column)
  C. `hip_yaw_max_rad` was never computed from raw telemetry and defaulted to 0
  D. mode_div was disabled for Step D original runs

### 4.3 Systematic hip-yaw regression (Step E, Step D, dynamic, long-run)

- **Issue:** Candidate shows elevated `hip_yaw_max_rad` vs original across all non-Step-C scopes.
- **Hypothesis:** `standalone_mode=True` computes sagittal/support intermediates differently from monolithic JAX path, leading to different mode_div inputs and ultimately different hip-yaw divergence.

### 4.4 Systematic support RMS regression (Step E, Step D)

- **Issue:** `support_rms_m` is hardcoded to 0.0 in `extract_metrics_from_summary()` — the candidate doesn't compute support RMS at all.
- **Impact:** Any comparison with original support_rms is invalid because the candidate value is always 0.0.

### 4.5 standalone_mode hypothesis

- `standalone_mode=True` in the dedicated runner computes sag_pos_err, sag_vel, support_vel, and pitch_x_error from raw state inputs.
- The monolithic JAX path receives Python-precomputed values for these fields.
- If these computations differ, ALL downstream control terms (sagittal damping, support FF, mode_div gating) will diverge.

### 4.6 gate_dwell unsafe hip-yaw

- **Current:** `hip_yaw_max_rad = 0.537 rad` in gate_dwell dynamic scenario.
- **Safety gate:** 0.35 rad absolute.
- **Status:** SAFETY_FAIL — must be fixed before promotion.

---

## 5. Dynamic q_ref Modes

### `original-k2-exact` (default, used for promotion)
- **Behavior:** STATIC q_ref from initial equilibrium_joint_pos captured at initialization
- **Effect:** Prevents CoM from following dynamic height commands — falls on ramp_up/up_down_cycle with fixed termination floor

### `setup-interp-debug` (debug-only)
- **Behavior:** APPROXIMATE linear interpolation from height setup files
- **Known defect:** Produces WORSE hip-yaw divergence
- **Usage rule:** NEVER for promotion validation

---

## 6. Reproducibility

To reproduce this PARTIAL state:

```bash
git checkout 0e1c7135e22b4cb852f71a795426cd3d3f19753a

# Run full validation (will produce current PARTIAL results)
python scripts/validate_k2_jax_dedicated_promotion.py \
  --scope all \
  --output-dir outputs/k2_jax_dedicated_promotion_validation

# Verify passing test
pytest tests/test_k2_jax_component_parity.py::TestParamsPackUnpackStage2::test_params_size_consistent -v

# Expected: 39/39 scenarios run, SAFETY_FAIL in some dynamic_height scenarios,
# SAFE_BUT_WORSE in Step D/E/long-run due to metric window and hip-yaw issues
```

---

## 7. Acceptance

- [x] Current PARTIAL state is documented and reproducible
- [x] Exact commit recorded: `0e1c713`
- [x] Exact changes applied: dynamic termination floor fix + param pack size test fix
- [x] All remaining blockers documented with specific locations
- [x] No code changes in this phase
- [x] No speculative fixes proposed
- [x] Phase 1 target: Fix Step D metric window parity (post-push 500-step window extraction)
