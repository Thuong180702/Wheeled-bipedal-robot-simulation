# Controller Semantic Rename — Verification

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Classification:** CONTROLLER_SEMANTIC_RENAME_PASS

---

## Semantic names introduced

| Semantic constant | profile_name | CLI key |
|---|---|---|
| `SUPPORT_CENTERING_BIAS_TRIM` | `support_centering_bias_trim` | `support_centering_bias_trim` |
| `PHASE_AWARE_AUTHORITY_RELEASE` | `phase_aware_authority_release` | `phase_aware_authority_release` |
| `EMERGENCY_BUDGET_CAP_RAISE` | `emergency_budget_cap_raise` | `emergency_budget_cap_raise` |
| `BAND_LIMITED_SUPPORT_RECENTER` | `band_limited_support_recenter` | `band_limited_support_recenter` |

## Legacy aliases preserved

| Legacy constant | Points to |
|---|---|
| `T6J_CENTERING_BIAS_TRIM` | `SUPPORT_CENTERING_BIAS_TRIM` |
| `T6I_PHASE_AWARE_RELEASE` | `PHASE_AWARE_AUTHORITY_RELEASE` |
| `T6F_BUDGET_CAP_RAISE` | `EMERGENCY_BUDGET_CAP_RAISE` |
| `APCR1ND_T5_BAND_LIMITED_BALANCED` | `BAND_LIMITED_SUPPORT_RECENTER` |

Both constants and string-key aliases map to the same `SagittalAuthoritySchedule` objects. No behavior change.

## Files changed

| File | Change |
|---|---|
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | Primary constants renamed, semantic aliases added, `JOINT_FIX_PROFILES` keyed with both semantic + legacy strings |
| `scripts/simulate_hierarchical_controller.py` | Imports extended, profile dict keyed with both names, CLI choices include all 4 semantic + 4 legacy strings |
| `scripts/run_t6j_height_ladder.py` → `scripts/run_support_centering_height_ladder.py` | git mv; deprecation wrapper created |
| `scripts/analyze_t6j_height_ladder.py` → `scripts/analyze_support_centering_height_ladder.py` | git mv; deprecation wrapper created |
| `tests/test_support_centering_bias_trim.py` | New file, all semantic-name tests, alias identity tests, original behavior tests |
| `tests/test_t6j_centering_bias_trim.py` | Profile name assertions updated to canonical semantic name |
| `tests/test_t6h_t6i_variants.py` | T6I profile_name assertion updated to canonical semantic name |
| `docs/validation/t6j_centering_bias_trim_final_report.md` | Title + header updated to semantic name |
| `docs/validation/t6j_height_ladder_validation_report.md` | Title + header updated to semantic name |
| `docs/repo_cleanup/execution/deep_outputs_cleanup_report.md` | Legacy label naming note added |

## Tests

| Test suite | Result |
|---|---|
| `test_support_centering_bias_trim.py` (24 tests) | **24 passed** |
| `test_t6j_centering_bias_trim.py` (26 tests) | **26 passed** (legacy aliases still work) |
| `test_t6h_t6i_variants.py` (38 tests) | **38 passed** |
| `test_t6_high_height_variants.py` (46 tests) | **46 passed** |
| `test_sagittal_velocity_damped_balance_controller.py` (207 tests) | **207 passed** |
| `test_simulation_telemetry_csv_writer.py` (34 tests) | **34 passed** |
| `test_low_height_setup_initialization.py` (13 tests) | **13 passed** (1 skipped) |
| `test_step_e_wbc_gate_validator.py` (42 tests) | **42 passed** |
| **Total** | **430 passed, 1 skipped, 0 failed** |

## Compile check

`wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`,
`scripts/simulate_hierarchical_controller.py`,
`scripts/run_support_centering_height_ladder.py`,
`scripts/analyze_support_centering_height_ladder.py` → **COMPILE OK**

## CLI smoke tests

### Semantic name
```
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile support_centering_bias_trim \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 100 --telemetry-decimation 1 --failure-window-steps 100 \
  --write-run-summary-sidecar
```
**Result: [OK] Completed full simulation without falling (100/100 steps). Terminated: False. Exit: 0**

### Legacy alias
```
python scripts/simulate_hierarchical_controller.py \
  --vd-sagittal-authority-profile T6J_centering_bias_trim \
  [same args]
```
**Result: [OK] Completed full simulation without falling (100/100 steps). Terminated: False. Exit: 0**

Both launch successfully, complete all steps, and use the same profile object.

## Classification

**CONTROLLER_SEMANTIC_RENAME_PASS**

All 4 semantic names work as CLI values, legacy names map to the same objects, no numeric parameters changed, no behavior changed, all tests pass.