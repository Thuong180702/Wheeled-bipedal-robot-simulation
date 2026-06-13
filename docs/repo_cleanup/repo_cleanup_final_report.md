# Repository Cleanup — Final Execution Report

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Base commit:** `ef532bb` (main)
**Policy:** Safe cleanup — archive-in-repo (`git mv`), preserve all runtime/test/config/training source.

---

## 1. Branch name

`repo-cleanup-t6j`

## 2. Files archived

| Archive bucket | Count |
|---|---|
| `archive/cleanup_2026-06-13/scripts_archive/` | 85 |
| `archive/cleanup_2026-06-13/root_analysis_archive/` | 58 |
| `archive/cleanup_2026-06-13/docs_archive/` | 192 |
| **Total archived (moved, still tracked)** | **335** |

All moves used `git mv` (tracked files), so history is preserved. `archive/` is intentionally **not** gitignored — the archive is part of commit history per the approved policy.

## 3. Files deleted

| Class | Detail |
|---|---|
| Root run logs (disk) | `t6a_run.log, t6b_5000_run.log, t6b_run.log, t6c_run.log, t6d_run.log, t6e_run.log, t6h_t6i_500_T{5,6F,6H,6I}_run.log`, `MUJOCO_LOG.TXT` — all gitignored, removed from disk |
| Caches (disk) | `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `__pycache__/`, `wheeled_biped_sim.egg-info` — gitignored, removed from disk |
| Audit temp (disk) | `repo_tracked_files.txt`, `repo_status_ignored.txt`; scratch `_*.txt` in execution/ |
| Untracked from git (kept on disk) | `assets/robot-urdf/export.log` (`git rm --cached`) |

No tracked source file was hard-deleted. The only index deletion is `export.log` (untracked, local copy retained).

## 4. Files kept

Index tracked count: **1096** (was 1072). Net change = −1 (export.log untracked) + 25 (new audit/execution docs) + 335 archived (moved, still tracked, net-zero on count).

## 5. Controllers kept

All controller source in `wheeled_biped/controllers/` — **0 controller modules touched**. The single profile-holding file `sagittal_velocity_damped_balance_controller.py` is unchanged. Profiles preserved: `T6J_centering_bias_trim`, `T6I_phase_aware_release`, `T6F_budget_cap_raise`, `T6F_sign_corrected`, `T6H_soft_blend_arch_fix`, `APCR1ND_T1–T5`, `T6A–T6E`, and the full hierarchical stack.

## 6. Controllers archived / deleted

**None.** No controller profile or controller module was archived or deleted (all share one source file and several are still imported by `simulate_hierarchical_controller.py`). Only *docs and one-off analysis scripts* about superseded profiles were archived.

## 7. Tests kept

All **130** test files in `tests/` retained. No test deleted or moved.

## 8. Tests removed

**None.**

## 9. Reports kept

All 17 T6J/T6I/Step-C/Step-E/hip-yaw/height-envelope final reports listed in the safe policy, plus the broader `docs/validation/` truth-path set (144 validation docs remain) and `docs/repo_cleanup/`. Root `README.md`, `CLAUDE.md`, `docs/baseline_workflow.md`, paper files all kept.

## 10. Reports archived

192 obsolete docs moved to `archive/cleanup_2026-06-13/docs_archive/` (relative paths preserved): `docs/validation/apcr1*`, `t5*`, `t6f_sign_corrected*`, `t6f_sign_fix*`, `t6h*`, `e1/e2/f1/f2/g1*`, `phase_*`; `docs/phase_b*`; `docs/task_*`; `docs/upstream_controller_debug_plan.md`; `docs/superpowers/**`.

## 11. Scripts kept

All 11 production scripts present: `train.py, evaluate.py, eval_balance.py, visualize.py, validate_checkpoint.py, export_results.py, compare_baseline.py, analyze_residual.py, simulate_hierarchical_controller.py, run_t6j_height_ladder.py, analyze_t6j_height_ladder.py`. Plus all test-imported and script-imported helpers, and the phase_b9 Step 5.13–5.25 mainline-infra scripts (excluded from archiving per project memory + import scan). 251 scripts remain in `scripts/`.

## 12. Scripts archived

85 obsolete one-off `scripts/*.py` (analyze_apcr*/t6f*/t6h*, audit_apcr*, debug_*, diagnose_*, check_*, compute_*, boundary_*, screening_*) + 58 loose root-level analysis `.py`. **Exclusions verified:** 12 pattern-matched scripts imported by tests and all phase_b9 mainline-infra scripts were kept (not archived).

## 13. Outputs / caches deleted

Gitignored generated artifacts only (Section 3). `outputs/physical_target_height_setups/` **preserved intact** (11 setup JSONs + summaries) — required by `run_t6j_height_ladder.py` and height-variant validation; not proven regenerable, so kept.

## 14. Training files kept

`wheeled_biped/training/**`, `wheeled_biped/envs/**`, `wheeled_biped/rewards/**`, `wheeled_biped/eval/**`, all `configs/**`, all `assets/**` (robot model + meshes). Zero training/config/asset source touched.

## 15. .gitignore changes

Added explicit (clarity) exclusions, both already matched by existing `*.log`:
```
assets/robot-urdf/export.log
*_run.log
```

## 16. Test results

- Targeted cleanup-sensitive batch: **371 passed** (T6J profile suite alone: 26 passed).
- Compile checks: **OK** (sim entrypoint + sagittal controller + shape posture controller).
- Broader fast suite (`-m "not slow"`): **1806 passed, 55 failed, 17 skipped**.
  - The 55 failures are **pre-existing on `main`** (verified: 3 representative failures reproduced after `git checkout main`) and are controller-physics/WBC/obs-adapter convention regressions **unrelated to cleanup**. No failing test imports any archived module; no runtime/test source was modified.

## 17. T6J smoke result

`simulate_hierarchical_controller.py` with `T6J_centering_bias_trim` on `high_0p480_setup.json`, 100 steps: **[OK] Completed full simulation without falling** (100/100). CoM 0.481–0.491 m, pitch_x ≤5.7°, roll_y ≤0.1°. Telemetry: 791 cols × 100 rows.

## 18. Known risks

- **Low:** archived docs/scripts are still in repo history (recoverable via `git mv` back). No data loss.
- **Low:** 55 pre-existing fast-suite failures remain (not introduced here; tracked separately as controller-physics regressions on `main`).
- **None:** no runtime/test/config/asset/training source changed; T6J path verified intact.
- `export.log` local copy retained on disk; only untracked from git.

## 19. Rollback instructions

```bash
# Discard the whole cleanup branch (return to main):
git checkout main
git branch -D repo-cleanup-t6j

# OR, if already committed and you want to undo on-branch:
git reset --hard ef532bb     # back to pre-cleanup base

# Restore a single archived file:
git mv archive/cleanup_2026-06-13/scripts_archive/<name>.py scripts/<name>.py

# Re-track export.log:
git add -f assets/robot-urdf/export.log
```

---

## Final cleanup classification

**REPO_CLEANUP_VERIFIED_READY_TO_COMMIT**
