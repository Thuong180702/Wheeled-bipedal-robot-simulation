# Repository Cleanup — Final Audit Report

**Date:** 2026-06-13
**Repo:** Thuong180702/Wheeled-bipedal-robot-simulation
**Branch:** main @ `ef532bb` (Add t6j)
**Scope:** Inspection and planning only. No files deleted, moved, modified, or committed.

This report consolidates the six prior audit phases:

- [repo_audit_baseline.md](repo_audit_baseline.md) — Phase 0
- [repo_inventory.md](repo_inventory.md) / [repo_inventory.json](repo_inventory.json) — Phase 1
- [dependency_usage_audit.md](dependency_usage_audit.md) / [dependency_usage_audit.json](dependency_usage_audit.json) — Phase 2
- [gitignore_generated_artifact_audit.md](gitignore_generated_artifact_audit.md) — Phase 3
- [cleanup_manifest.md](cleanup_manifest.md) / [cleanup_manifest.json](cleanup_manifest.json) — Phase 4
- [proposed_clean_repo_structure.md](proposed_clean_repo_structure.md) — Phase 5

---

## Baseline facts

| Metric | Value |
|---|---|
| Tracked files (`git ls-files`) | 1072 |
| Root-level tracked `.py` (loose scripts) | 58 |
| `scripts/*.py` | 336 |
| `tests/*.py` | 130 |
| `wheeled_biped/**/*.py` | 95 |
| `docs/**/*.md` (excl. worktrees) | 339 |
| Working tree | clean (only new untracked audit outputs) |

Two large noise sources dominate the repo: **the `scripts/` + root loose-script sprawl (~394 `.py`)** and **the `docs/validation/` experiment-report sprawl (~280 `.md`)**, both products of the long APCR → T6 controller-tuning campaign.

---

## The 15 required answers

### 1. What is the current best controller path?

`T6J_centering_bias_trim` is the current best validated profile. Runtime path:

```
scripts/simulate_hierarchical_controller.py
  → SagittalVelocityDampedBalanceController(authority_schedule=T6J_CENTERING_BIAS_TRIM)
     (wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py)
  → balance_core_types.py + position_hold_capture_gate.py
  → validation/telemetry_adapter.py (telemetry fields)
scripts/run_t6j_height_ladder.py     → drives the above per height variant
scripts/analyze_t6j_height_ladder.py → post-processes ladder output
tests/test_t6j_centering_bias_trim.py → validates the profile
```

`T6J_CENTERING_BIAS_TRIM` is a `SagittalAuthoritySchedule` literal at
`sagittal_velocity_damped_balance_controller.py:1593`, registered in
`JOINT_FIX_PROFILES` at line ~1740.

### 2. Which files are required for T6J?

**Runtime (must keep):**
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` (holds T6J + its inheritance chain)
- `wheeled_biped/controllers/balance_core_types.py`
- `wheeled_biped/controllers/position_hold_capture_gate.py`
- `wheeled_biped/validation/telemetry_adapter.py`
- The full hierarchical-controller stack imported by `simulate_hierarchical_controller.py` (centroidal estimator, capture point, integrated WBC, momentum coordinator, posture regularizer, leg position, stage2b/c/d, contact jacobian, balance_core_torque_composer, contact supervisor, lateral roll, sagittal wheel, shape posture, yaw controller, static feedforward/holding)
- `scripts/simulate_hierarchical_controller.py`, `scripts/run_t6j_height_ladder.py`, `scripts/analyze_t6j_height_ladder.py`

**Tests:** `tests/test_t6j_centering_bias_trim.py` (+ the broader controller test suite as regression cover).

**Docs:** `docs/validation/t6j_centering_bias_trim_final_report.md`, `..._design.md`, `..._implementation_tests_report.md`, `t6j_height_ladder_validation_report.md`, `t6j_high_0p480_5000_validation_report.md`.

### 3. Which older controllers are safe to remove?

No controller profile is **code-safe to delete** today — all live in one file and several are still imported by `simulate_hierarchical_controller.py` (`T6F_SIGN_CORRECTED`, `T6H_SOFT_BLEND_ARCH_FIX`, `APCR1ND_T1–T5`, `T6A–T6E`). Removing them requires editing both the controller module and the sim script, which is out of scope for this no-modify audit.

Safe to **archive (docs + standalone one-off scripts only)**, with no runtime impact:
- `T6F_sign_corrected` — failed design (docs + `analyze_t6f_sign_*`, `audit_t6f_*`, `debug_t6f_sign_zero_torque.py`)
- `T6H_soft_blend_arch_fix` — failed design, superseded by T6I (docs + `analyze_t6h_t6i_500.py`)
- All `APCR1*` and `T1–T5` variant docs and their one-off analysis/audit scripts.

### 4. Which older controllers must be kept because T6J depends on them?

- **`T6I_phase_aware_release`** — direct base; T6J inherits its `apcr1nd_*` and `t6i_*` fields (asserted in `test_t6j_inherits_t6i_settings`). **Keep.**
- **`T6F_budget_cap_raise`** — supplies the `arch_fix`/`budget_cap_raise` configuration inherited up the chain. **Keep the code profile.**
- **`APCR1ND_T5` (and the T1–T4 it builds on)** — still imported by `simulate_hierarchical_controller.py`; the T6 line descends conceptually from APCR1ND. **Keep the code profiles** until a refactor decouples them.

### 5. Which tests are essential?

- `tests/test_t6j_centering_bias_trim.py` (T6J correctness + inheritance guard)
- `tests/test_t6h_t6i_variants.py` (guards the T6F/T6H/T6I chain T6J depends on)
- Core stack: `test_action_codec`, `test_residual_balance_env`, `test_ppo_trainer`, `test_curriculum`, `test_benchmark`, `test_unified_controller`, `test_model`, `test_rewards`, `test_sim_helpers`, `test_noise_and_dr`, `test_standing_quality`, `test_baseline`, `test_eval_balance`, `test_telemetry`, `test_validate_checkpoint`
- Controller-stack regression tests (LQR/IK, centroidal, WBC, momentum, posture, contact jacobian, stage2*, etc.)

### 6. Which tests are obsolete?

**None are recommended for deletion.** The 130 test files form the regression net for the controller evolution; tests for superseded profiles still pin behavior of shared code. `test_env.py` and `test_smoke_train.py` remain valid but are slow — run on demand, not in fast CI. Recommendation: **keep all tests.**

### 7. Which reports should be kept?

Current truth only:
- `docs/validation/t6j_*` (5 files)
- `docs/validation/t6i_full_staged_validation_final_report.md`, `t6i_height_ladder_validation_summary.md`, `t6i_high_0p480_full_validation_summary.md`
- `docs/validation/current_balance_core_roadmap_status.md`
- `docs/validation/step_c_height_recovery_done.md`, `step_e_height_variant_robustness_done.md`, `step_e_done_2026-06-01.md`
- `docs/validation/protected_d2_baseline_freeze.md`
- Root: `README.md`, `CLAUDE.md`; `paper/` manuscript; `docs/baseline_workflow.md`

### 8. Which reports should be archived?

~280 `.md` under `docs/`, none on the T6J truth path:
- `docs/validation/apcr1*.md` (~80), `t5_*.md`, `t6f_sign_corrected*.md` / `t6f_sign_fix*.md`, `t6h_*.md`
- `e1_*`, `e2_*`, `f1_*`, `f2_*`, `g1_*` experiment reports
- `docs/phase_b*.md`, `docs/task_11_*.md`, `docs/upstream_controller_debug_plan.md`
- `docs/superpowers/**` (plans/specs/reports — development history)

Archive to `archive/cleanup_2026-06-13/docs_archive/` preserving relative paths.

### 9. Which generated outputs should be deleted?

All gitignored, none tracked — safe to delete from disk to reclaim space:
- Root run logs: `t6a_run.log`, `t6b_5000_run.log`, `t6b_run.log`, `t6c_run.log`, `t6d_run.log`, `t6e_run.log`, `t6h_t6i_500_T{5,6F,6H,6I}_run.log`
- `MUJOCO_LOG.TXT`, `paper/main.log`
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `wheeled_biped_sim.egg-info/`
- `outputs/`, `outputs_residual_smoke/`, `outputs_residual_test/` (regenerable artifacts)
- Audit temporaries: `repo_tracked_files.txt`, `repo_status_ignored.txt`

One **tracked** generated file should be untracked (`git rm --cached`): `assets/robot-urdf/export.log`.

### 10. Which training files must be kept?

- `scripts/train.py`, `scripts/evaluate.py`, `scripts/eval_balance.py`, `scripts/validate_checkpoint.py`, `scripts/export_results.py`, `scripts/compare_baseline.py`, `scripts/visualize.py`, `scripts/analyze_residual.py`
- `wheeled_biped/training/**`, `wheeled_biped/envs/**`, `wheeled_biped/rewards/**`, `wheeled_biped/eval/**`
- All `configs/` (training stages, curriculum, controllers, robot, baseline_lqr)
- `assets/` (robot model + meshes)

### 11. Which training files are obsolete?

No *training pipeline* code is obsolete. The obsolete bulk is the **controller-tuning experiment harness**, not RL training: ~220 one-off `scripts/` (`analyze_*`, `audit_*`, `debug_*`, `diagnose_*`, `check_*`, `compute_*`, `find_*`, `phase_b9_*`, `screening_*`, `tune_*`, `validate_balance_core_*` v2/v3/v4 duplicates) plus the ~58 loose root `.py` (all `analyze_/audit_/check_/debug_/compare_/verify_*` plus `test_init.py`, `tmp_b9_audit_gate_gen.py`, `verify_residual_ckpt.py`). None are imported by any module (verified: zero import references).

### 12. What should be added to .gitignore?

Current `.gitignore` already covers caches, venvs, outputs, `*.log`, `MUJOCO_LOG.TXT`. Gaps:
- `*.csv` is **not** ignored → `assets/robot-urdf/urdf/HOANTHIEN_TEST.csv` is tracked (likely intentional source data — review, don't blanket-ignore).
- Add explicit `*.egg-info/` (currently only `*.egg-info/` via build block — confirm `wheeled_biped_sim.egg-info/` is matched; it is via `*.egg-info/`).
- Consider ignoring stray root `*_run.log` (already matched by `*.log`).
- Add `archive/` if the backup archive is created inside the repo and should not be tracked.
- `export.log` is tracked despite `*.log` — needs `git rm --cached` (gitignore alone won't untrack).

### 13. What is the exact cleanup manifest?

See [cleanup_manifest.md](cleanup_manifest.md) / [cleanup_manifest.json](cleanup_manifest.json). Summary:

| Class | Count (approx) | Action |
|---|---|---|
| KEEP | ~200 tracked | unchanged |
| BACKUP_THEN_REMOVE | ~500 (220 scripts + 280 docs) | archive → remove from tree |
| DELETE | ~40 (generated/temp + obsolete loose analysis) | delete (gitignored) / `git rm --cached` (export.log) |

Expected tracked count after cleanup: **~832** (from 1072), pending manual review confirmation.

### 14. What are the risks?

- **Low:** deleting gitignored generated artifacts (logs, caches, outputs) — fully regenerable.
- **Low:** keeping all controller code and tests — zero behavior change.
- **Medium:** archiving ~220 scripts / ~280 docs — must back up first; some "obsolete" docs may hold rationale referenced by the paper. Verify against `paper/main.tex` citations before removing.
- **Medium:** removing loose root analysis `.py` — confirmed unimported, but a few may be invoked manually by ad-hoc workflows; archive rather than hard-delete.
- **None high:** no production/runtime path is touched by any proposed action.

### 15. What should be reviewed manually before deletion?

1. `assets/robot-urdf/urdf/HOANTHIEN_TEST.csv` and `assets/robot-urdf/export.log` — confirm source-vs-generated classification.
2. `paper/main.tex` — cross-check that no archived `docs/validation/*.md` is cited/needed for the manuscript.
3. The decision to **archive vs delete** the ~58 loose root `.py` and ~220 `scripts/` one-offs (recommend archive).
4. Whether to **refactor** `simulate_hierarchical_controller.py` to drop unused profile imports (`T6F_SIGN_CORRECTED`, `T6H`, `T6A–T6E`) — enables later code-level removal but is a *modification*, deferred.
5. Confirm `outputs/physical_target_height_setups/*_setup.json` (consumed by `run_t6j_height_ladder.py`) are regenerable before clearing `outputs/`.

---

## Final classification

**REPO_CLEANUP_AUDIT_NEEDS_MANUAL_REVIEW**

Rationale: The plan is fully actionable and runtime-safe, but execution requires human sign-off on (a) archive-vs-delete policy for ~220 scripts and ~280 docs, (b) the CSV/export.log asset classification, (c) paper-citation cross-check, and (d) confirmation that `outputs/` setup JSONs are regenerable. No destructive step should run until items 1–5 above are confirmed.

*No files were deleted, moved, modified, or committed during this audit.*
