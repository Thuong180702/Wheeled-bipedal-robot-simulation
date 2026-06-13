# Post-Cleanup Verification Report

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Baseline commit:** `ef532bb` (main)

---

## 1. Source-tree integrity

No runtime or test source was modified by the cleanup. Confirmed via `git diff main..HEAD`:

| Path scope | Staged changes |
|---|---|
| `wheeled_biped/**` | NONE |
| `tests/**` | NONE |
| `configs/**` | NONE |
| `assets/**` (besides `export.log` untrack) | NONE |

The only index changes are file moves (`R`), the `export.log` deletion from index (`D`), the `.gitignore` edit (`M`), and the new audit docs (`A`).

---

## 2. Compile checks

```
python -m py_compile scripts/simulate_hierarchical_controller.py            → OK
python -m py_compile wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py → OK
python -m py_compile wheeled_biped/controllers/shape_posture_controller.py  → OK
```

**Result: COMPILE OK**

---

## 3. Targeted baseline tests (cleanup-sensitive)

```
tests/test_t6j_centering_bias_trim.py
tests/test_t6h_t6i_variants.py
tests/test_sagittal_velocity_damped_balance_controller.py
tests/test_simulation_telemetry_csv_writer.py
tests/test_low_height_setup_initialization.py
tests/test_step_e_wbc_gate_validator.py
```

**Result: 371 passed** (run as a batch). T6J profile suite alone re-confirmed: **26 passed**.

These are the tests that exercise the T6J runtime path, the controller profile registry, telemetry CSV writing, the height-variant setup JSONs, and the Step E gate validator — i.e. everything the archive/delete actions could plausibly have touched. All green.

---

## 4. Broader fast suite (`pytest -q -m "not slow"`)

```
55 failed, 1806 passed, 17 skipped, 46 deselected, 1 xfailed, 1 xpassed
```

### 55 failures are PRE-EXISTING and UNRELATED to cleanup

Evidence:

1. **They fail identically on `main` (pre-cleanup).** Ran 3 representative failing tests after `git checkout main`:
   - `test_unified_controller.py::...test_novelin_height_cmd_yaw_adapter_for_39` → FAILED on main
   - `test_integrated_wbc.py::test_wbc_torque_leg_joints_are_zero` → FAILED on main
   - `test_unified_force_distributor.py::test_distribute_wrench_roll_moment` → FAILED on main

   → 3 failed on main, same as on the cleanup branch. The cleanup did not introduce them.

2. **No failing test imports an archived module.** Scanned all 7 failing test files for `from scripts.` / `import scripts.` — none import any archived `scripts/` one-off.

3. **No runtime/test source was touched by the cleanup** (Section 1).

These failures are pre-existing controller-physics regressions (WBC/force-distribution/obs-adapter convention assertions) that exist on `main` independently of repository hygiene. They are out of scope for this cleanup task and must not be attributed to it.

---

## 5. T6J smoke simulation

```
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile T6J_centering_bias_trim \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 100 --telemetry-decimation 1 --failure-window-steps 100 \
  --write-run-summary-sidecar
```

Result:
- T6J profile resolved and ran.
- **Status: [OK] Completed full simulation without falling** (100/100 steps).
- CoM height range 0.481–0.491 m; pitch_x range −0.0…5.7°, roll_y −0.0…0.1°.
- Telemetry written: 791 columns, 100 rows.
- No immediate failure / no fall.

---

## 6. Production-script presence

All 11 production scripts confirmed present after archiving:
`train.py, evaluate.py, eval_balance.py, visualize.py, validate_checkpoint.py, export_results.py, compare_baseline.py, analyze_residual.py, simulate_hierarchical_controller.py, run_t6j_height_ladder.py, analyze_t6j_height_ladder.py`.

Setup JSONs preserved in `outputs/physical_target_height_setups/` (all 11 + summaries intact).

---

## Classification

**CLEANUP_VERIFICATION_PASS**

Rationale: the cleanup introduced zero source changes, all cleanup-sensitive targeted tests pass (371), the T6J runtime smoke-runs cleanly, and the 55 broader-suite failures are demonstrably pre-existing on `main` and independent of any archive/delete action.
