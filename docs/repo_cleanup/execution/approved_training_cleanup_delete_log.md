# Approved Training Cleanup — Delete Log

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Scope:** User-approved low-risk subset only. No checkpoints, seeds, setup JSONs, or `backup_checkpoints/` touched.

---

## Deleted paths

| Path | Type | Before size | Git status | Reason |
|---|---|---|---|---|
| `outputs_residual_smoke/` | output dir | 13 M | untracked (gitignored) | residual smoke artifacts, regenerable |
| `outputs_residual_test/` | output dir | 37 M | untracked (gitignored) | residual test artifacts, regenerable |
| `outputs/step_e_height_variant_sagittal_schedule_fix_damping/` | output dir | 417 M | untracked (gitignored) | unreferenced by any kept doc/script |
| `__pycache__/` (all, 12 dirs) | cache | — | untracked (gitignored) | regenerable bytecode |
| `.pytest_cache/` | cache | — | untracked (gitignored) | regenerable |

Approx disk freed: **~467 M**.

## Run logs / MUJOCO_LOG.TXT

Not present on disk — already removed during the prior `repo-cleanup-t6j` repository cleanup. No action needed.

## Explicitly NOT deleted (preserved)

- `outputs/balance/rl/seed42`, `seed113`, `seed999` (checkpoints, metrics JSONL, tb/)
- `outputs/physical_target_height_setups/`
- `backup_checkpoints/` (tracked — not removed, not `git rm`)
- `outputs/step_e_extreme_support_fix_eval/`, `outputs/hierarchical_controller_sim/`,
  `outputs/step_e_height_variant_position_hold_audit_v2/`, `outputs/step_e_height_variant_position_hold_final/`
- all other `outputs/` dirs referenced by kept docs/scripts
- all training code and configs (untouched)
