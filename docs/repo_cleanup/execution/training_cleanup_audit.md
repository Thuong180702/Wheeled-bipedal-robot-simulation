# Training Cleanup Safety Audit

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Scope:** Inspection only. No training code, configs, checkpoints, or outputs were deleted during this audit.

---

## 1. Training output folders and sizes

| Folder | Size | Git status | .gitignore |
|---|---|---|---|
| `outputs/` | **8.6 G** | 0 tracked | ignored |
| `outputs_residual_smoke/` | 13 M | 0 tracked | ignored |
| `outputs_residual_test/` | 37 M | 0 tracked | ignored |
| `backup_checkpoints/` | 19 M | **24 files TRACKED** | not ignored |

`outputs/` holds **173 sub-directories**. Largest disposable experiment-diagnostic dirs:

| Dir | Size | Referenced by kept docs/scripts? |
|---|---|---|
| `outputs/step_e_extreme_support_fix_eval/` | 3.4 G | **YES** (3 validation reports) |
| `outputs/hierarchical_controller_sim/` | 714 M | **YES** (3 validation reports) |
| `outputs/step_e_height_variant_position_hold_audit_v2/` | 465 M | **YES** (kept script) |
| `outputs/step_e_height_variant_sagittal_schedule_fix_damping/` | 417 M | no reference found |
| `outputs/step_e_height_variant_position_hold_final/` | 417 M | **YES** (2 kept docs) |
| `outputs/step_c_sagittal_schedule_fix/` | 273 M | **YES** (kept script) |
| `outputs/hip_yaw_divergence_after_sign_fix_audit/` | 257 M | **YES** (2 docs + 1 script) |
| `outputs/balance_core_validation/` | 173 M | **YES** (kept script) |
| `outputs/posture_feasibility/` | 172 M | **YES** (kept script) |
| `outputs/hip_yaw_disturbance_rejection_audit/` | 169 M | **YES** (2 docs + 1 script) |

The vast majority of the large output dirs are still referenced by **kept** validation docs and kept scripts (output-path arguments). Deleting them wouldn't break Python imports, but would orphan doc references and remove regenerable-but-expensive artifacts.

---

## 2. Training-stage checkpoints

Only the **`balance`** stage has trained runs on disk. `balance_robust`, `stand_up`, etc. are mentioned in README but have **no** `outputs/.../rl` runs (consistent with CLAUDE.md "untrained stages").

| Stage | Seed | Size | `checkpoints/final/checkpoint.pkl` | metrics JSONL |
|---|---|---|---|---|
| balance | seed42 | 119 M | OK (2.6 M) | OK |
| balance | seed113 | 94 M | OK (2.6 M) | OK |
| balance | seed999 | 62 M | OK (2.6 M) | OK |

Latest per-seed step checkpoints: seed42 → `step_9846784`; seed113 → `step_18071552`; seed999 → `step_19087360`. `final/` exists for all three.

`backup_checkpoints/` (tracked, 19 M): Day1/Day2 snapshots at steps 1654784 / 3293184 / 4931584 (Day1) and 1654784 / 3293184 / 5390336 (Day2), each with `checkpoint.pkl` + eval/validation JSON + telemetry PNG.

---

## 3. Reference analysis (README / scripts / paper / docs / configs)

- **README heavily references** `outputs/balance/rl/seed{42,113,999}/checkpoints/final/checkpoint.pkl`, the per-seed `*_metrics.jsonl`, and `tb/` event dirs — for training, resume, eval, paper-eval, baseline freeze, and visualization commands. **These are load-bearing.**
- **The README-referenced checkpoints live in gitignored `outputs/`** — they are NOT in git, so deleting them is **irreversible without an ~18M-step retrain**.
- `backup_checkpoints/` is **tracked but unreferenced** by README/configs/paper — its only consumer would be manual rollback.
- Many large diagnostic output dirs are referenced by kept validation docs and kept scripts (Section 1).
- `outputs/physical_target_height_setups/` present and required by T6J validation — **must not delete** (confirmed intact).
- `outputs_residual_smoke/` / `outputs_residual_test/` appear only as command-example output dirs in CLAUDE.md — regenerable, but currently the only residual-pipeline artifacts on disk.

---

## 4. Safety determination

| Question | Answer |
|---|---|
| Are all candidate outputs purely generated + regenerable + unreferenced? | **No** |
| Do README/docs/scripts reference output paths slated for deletion? | **Yes** (extensively) |
| Are README-referenced checkpoints recoverable if deleted? | **No** (gitignored, not in git; need retrain) |
| Is any deletion target a *tracked* file? | **Yes** (`backup_checkpoints/`, 24 files) |
| Setup JSONs / T6J validation assets safe? | Preserved, untouched |

Deleting generated training outputs here is **not** a no-consequence operation:
1. The only trained policy checkpoints (3 seeds) are the paper's main results and are not in version control — deletion is irreversible.
2. ~Half of the large output dirs are cited by kept validation reports.
3. `backup_checkpoints/` deletion would be a tracked-file removal needing explicit sign-off and a different (git rm) workflow.

---

## Classification

**TRAINING_CLEANUP_NEEDS_MANUAL_REVIEW**

No training outputs, checkpoints, or `backup_checkpoints/` were deleted. Per policy, deletion proceeds only under `TRAINING_CLEANUP_SAFE_GENERATED_ONLY`, which does not hold.

### Recommended next decisions (require user sign-off)

1. **Preserve always:** `outputs/balance/rl/seed{42,113,999}` (final + metrics + tb), `outputs/physical_target_height_setups/`.
2. **Safe-ish to prune** (regenerable, no kept reference found): `outputs/step_e_height_variant_sagittal_schedule_fix_damping/` (417 M) and any other dir confirmed unreferenced by a kept doc/script — review one-by-one, not bulk.
3. **Do NOT bulk-delete** the referenced 3.4 G / 714 M / 465 M dirs without first deciding whether their validation reports are still authoritative.
4. **`backup_checkpoints/` (tracked):** decide keep-as-rollback vs `git rm` + external archive — separate from this gitignored-output cleanup.
5. If disk is the driver, the single biggest win is `outputs/step_e_extreme_support_fix_eval/` (3.4 G) — but it's referenced by 3 kept reports; confirm those are superseded before removing.

*No files were deleted, moved, or modified during this training audit.*
