# Approved Training/Output Cleanup — Final Report

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Scope:** Executed ONLY the user-approved low-risk subset from
[training_cleanup_audit.md](training_cleanup_audit.md) (which classified the broader
cleanup as `TRAINING_CLEANUP_NEEDS_MANUAL_REVIEW`).

---

## 1. Deleted folders / files

| Path | Type | Size freed | Git status |
|---|---|---|---|
| `outputs_residual_smoke/` | gitignored output dir | 13 M | untracked |
| `outputs_residual_test/` | gitignored output dir | 37 M | untracked |
| `outputs/step_e_height_variant_sagittal_schedule_fix_damping/` | unreferenced diagnostic dir | 417 M | untracked |
| `__pycache__/` (12 dirs) | cache | — | untracked |
| `.pytest_cache/` | cache | — | untracked |
| root run logs (`t6*_run.log`, `MUJOCO_LOG.TXT`) | generated logs | — | already absent (removed in prior cleanup) |

**No tracked file was deleted.** `git status --short` shows only the 4 new audit
docs under `docs/repo_cleanup/execution/` as changes; zero `D` entries in the tracked set.

---

## 2. Freed disk estimate

`outputs/` went **8.6 G → 8.2 G** (≈417 M from the damping dir).
Plus `outputs_residual_smoke/` (13 M) + `outputs_residual_test/` (37 M) deleted whole.

**Total freed: ≈467 M** (plus cache dirs, regenerable).

---

## 3. Protected checkpoint verification (post-deletion)

| Asset | Status |
|---|---|
| `outputs/balance/rl/seed42/checkpoints/final/checkpoint.pkl` | OK |
| `outputs/balance/rl/seed113/checkpoints/final/checkpoint.pkl` | OK |
| `outputs/balance/rl/seed999/checkpoints/final/checkpoint.pkl` | OK |
| per-seed `*_metrics.jsonl` (all 3) | OK |
| per-seed `tb/` event dirs (all 3) | OK |

Seed sizes unchanged after cleanup: seed42 119 M, seed113 94 M, seed999 62 M.

---

## 4. `physical_target_height_setups/` verification

Intact (56 K) — all 11 setup JSONs + summaries + report present and untouched.

---

## 5. `backup_checkpoints/` preserved

Present (19 M, 24 tracked files). **Not** `git rm`'d, not deleted — per policy.

---

## 6. Tests / compile results

- `py_compile`: `simulate_hierarchical_controller.py`, `train.py`, `evaluate.py`,
  `sagittal_velocity_damped_balance_controller.py` → **COMPILE OK**
- Key tests: `test_t6j_centering_bias_trim`, `test_simulation_telemetry_csv_writer`,
  `test_low_height_setup_initialization`, `test_step_e_wbc_gate_validator`
  → **48 passed**
- CLI smoke: `validate_checkpoint.py --help`, `evaluate.py --help`,
  `train.py --help` → all OK

---

## 7. Git status

```
?? docs/repo_cleanup/execution/approved_training_cleanup_after.txt
?? docs/repo_cleanup/execution/approved_training_cleanup_before.txt
?? docs/repo_cleanup/execution/approved_training_cleanup_delete_log.md
?? docs/repo_cleanup/execution/training_cleanup_audit.md
```

All deletions were of gitignored / untracked artifacts, so the tracked tree is
unchanged except for the new audit docs.

---

## 8. Risks remaining

- **None introduced by this cleanup.** Deleted items were gitignored, regenerable,
  and (for the damping dir) confirmed unreferenced by any kept doc/script.
- The broader `outputs/` (still 8.2 G) remains under `TRAINING_CLEANUP_NEEDS_MANUAL_REVIEW`
  — the large referenced diagnostic dirs (3.4 G `step_e_extreme_support_fix_eval/`, etc.)
  and `backup_checkpoints/` still await per-item sign-off.

---

## Classification

**APPROVED_TRAINING_CLEANUP_PASS**

All protected assets verified present post-deletion, compile + key tests green,
no tracked file removed, only the approved subset deleted.

*Only the user-approved subset was deleted. No checkpoints, no `backup_checkpoints/`,
no training code or configs were touched.*
