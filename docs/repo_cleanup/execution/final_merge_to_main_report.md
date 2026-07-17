# Final Merge to Main — Repo Cleanup + Semantic Controller Names

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j` → `main`
**Merge commit SHA:** `6f6c845`
**Classification:** MAIN_CLEANUP_MERGE_PASS

---

## 1. Semantic names introduced

| Semantic constant | `profile_name` | CLI value |
|---|---|---|
| `SUPPORT_CENTERING_BIAS_TRIM` | `support_centering_bias_trim` | `--vd-sagittal-authority-profile support_centering_bias_trim` |
| `PHASE_AWARE_AUTHORITY_RELEASE` | `phase_aware_authority_release` | `--vd-sagittal-authority-profile phase_aware_authority_release` |
| `EMERGENCY_BUDGET_CAP_RAISE` | `emergency_budget_cap_raise` | `--vd-sagittal-authority-profile emergency_budget_cap_raise` |
| `BAND_LIMITED_SUPPORT_RECENTER` | `band_limited_support_recenter` | `--vd-sagittal-authority-profile band_limited_support_recenter` |

## 2. Legacy aliases preserved

| Legacy constant | Maps to |
|---|---|
| `T6J_CENTERING_BIAS_TRIM` | `SUPPORT_CENTERING_BIAS_TRIM` |
| `T6I_PHASE_AWARE_RELEASE` | `PHASE_AWARE_AUTHORITY_RELEASE` |
| `T6F_BUDGET_CAP_RAISE` | `EMERGENCY_BUDGET_CAP_RAISE` |
| `APCR1ND_T5_BAND_LIMITED_BALANCED` | `BAND_LIMITED_SUPPORT_RECENTER` |

CLI string keys for legacy names (`T6J_centering_bias_trim`, etc.) remain in `JOINT_FIX_PROFILES` and `SAGITTAL_AUTHORITY_PROFILES`, pointing to the same `SagittalAuthoritySchedule` objects. Behavior is unchanged.

## 3. Files renamed

| Old | New | Type |
|---|---|---|
| `scripts/run_t6j_height_ladder.py` | `scripts/run_support_centering_height_ladder.py` | git mv + deprecation wrapper |
| `scripts/analyze_t6j_height_ladder.py` | `scripts/analyze_support_centering_height_ladder.py` | git mv + deprecation wrapper |
| `tests/test_t6j_centering_bias_trim.py` | (kept) | Updated in-place |
| — | `tests/test_support_centering_bias_trim.py` | New file |

## 4. Tests passed

| Suite | Result |
|---|---|
| `test_support_centering_bias_trim.py` (new) | 24 passed |
| `test_t6j_centering_bias_trim.py` (updated) | 26 passed |
| `test_t6h_t6i_variants.py` (updated) | 38 passed |
| `test_t6_high_height_variants.py` | 46 passed |
| `test_sagittal_velocity_damped_balance_controller.py` | 207 passed |
| `test_simulation_telemetry_csv_writer.py` | 34 passed |
| `test_low_height_setup_initialization.py` | 13 passed (1 skipped) |
| `test_step_e_wbc_gate_validator.py` | 42 passed |
| **Total** | **430 passed, 1 skipped** |

## 5. New semantic CLI smoke

```
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile support_centering_bias_trim \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 100 --telemetry-decimation 1 --failure-window-steps 100 \
  --write-run-summary-sidecar
```
**Result:** `[OK] Completed full simulation without falling` — 100/100 steps, Terminated: False, Exit: 0

## 6. Legacy alias CLI smoke

```
python scripts/simulate_hierarchical_controller.py \
  --vd-sagittal-authority-profile T6J_centering_bias_trim \
  [same args]
```
**Result:** `[OK] Completed full simulation without falling` — 100/100 steps, Terminated: False, Exit: 0

Both resolve to the same `SUPPORT_CENTERING_BIAS_TRIM` profile object. No behavior change.

## 7. Cleanup branch push result

```
6858fc1 Rename validation profiles to semantic controller names
4a15542 Finalize deep output cleanup safety guards
e4fa308 Document deep output cleanup after T6J validation
c20a6a1 Clean repository around T6J validation profile
→ origin/repo-cleanup-t6j  ✓
```

## 8. Main merge commit SHA

```
6f6c845 Merge repo cleanup and semantic controller names
```

Merge made by `ort` strategy — no conflicts. All 14 files from the cleanup branch merged cleanly into `main`.

## 9. Main push result

```
ef532bb..6f6c845  main -> main  ✓
```

## 10. Remaining cleanup risks

**None.** All risks from the original cleanup report were resolved:
- 12 `balance_core_*` dirs deleted in a second pass (resolved the over-refusal)
- Output size reduced from 8.2 G → 276 M (~7.9 G freed)
- Legacy aliases confirmed to map to same profile objects
- T6J smoke confirms runtime behavior unchanged

## Classification

**MAIN_CLEANUP_MERGE_PASS**

All phases completed in order, all tests pass, both semantic and legacy CLI values work, merge clean with no conflicts, main pushed successfully.