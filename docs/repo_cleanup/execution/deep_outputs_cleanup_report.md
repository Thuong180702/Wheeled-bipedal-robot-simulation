# Deep Outputs Cleanup — Final Report

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Scope:** Deep clean of the remaining `outputs/` bulk (~8.2 G) with mandatory summary extraction before deletion. No tracked files, training code, configs, checkpoints, or T6J setup assets were touched.

---

## 1–3. Disk impact

| Metric | Value |
|---|---|
| `outputs/` before | **8.2 G** |
| `outputs/` after | **~651 M** |
| **Disk freed** | **~7.51 G** (8,066,814,481 bytes) |
| Summary archive size | 81 M (1,009 small artifacts) |

`outputs/` is gitignored — none of these deletions affect version control.

---

## 4. Protected checkpoint verification

All re-confirmed present **after** cleanup and **after** the T6J smoke run:

| Asset | Status |
|---|---|
| `outputs/balance/rl/seed42/checkpoints/final/checkpoint.pkl` | OK |
| `outputs/balance/rl/seed113/checkpoints/final/checkpoint.pkl` | OK |
| `outputs/balance/rl/seed999/checkpoints/final/checkpoint.pkl` | OK |
| per-seed `*_metrics.jsonl` (×3) | OK |
| per-seed `tb/` event dirs (×3) | OK |

`outputs/balance` (275 M) preserved whole.

## 5. physical_target_height_setups verification

`outputs/physical_target_height_setups/` (56 K, all 11 setups + summaries) — **intact**.

## 6. backup_checkpoints verification

`backup_checkpoints/` (19 M, 24 tracked files) — **untouched, not `git rm`'d**.

---

## 7. Directories summarized then deleted

**146 of 154** `SUMMARIZE_THEN_DELETE` dirs were summary-extracted and deleted (includes the 3.3 G `step_e_extreme_support_fix_eval`, 713 M `hierarchical_controller_sim`, 464 M `position_hold_audit_v2`, 417 M `position_hold_final`, etc.). Small JSON/CSV/MD summaries for each were copied to `archive/cleanup_2026-06-13/output_summaries/<dir>/` with a per-dir `manifest.json`.

## 8. Directories deleted directly

**12 of 16** `DELETE_DIRECT` dirs (no summaries, no kept reference, or empty) deleted.

Total deleted: **158 directories**, 0 failures.

## 9. Directories preserved

- `outputs/balance` (PROTECT)
- `outputs/physical_target_height_setups` (PROTECT)
- **12 `balance_core_*` dirs over-refused by the safety guard** (~377 M) — see Remaining Risks.

## 10. Summary archive location

`archive/cleanup_2026-06-13/output_summaries/` — 154 per-dir folders, global manifest at `docs/repo_cleanup/execution/deep_outputs_summary_extraction_manifest.json`.

---

## 11. Tests / compile

Compile OK (5 modules). Key tests: **47 passed, 1 skipped**.

## 12. Smoke simulation

T6J @ high_0p480, 100/100 steps, **no fall**. Telemetry regenerated cleanly into a fresh `hierarchical_controller_sim/`.

## 13. Git status

Only new/untracked docs under `docs/repo_cleanup/execution/`, the two cleanup scripts, and the `archive/.../output_summaries/` tree. **No tracked file deleted or modified** by the output cleanup (deleted dirs were all gitignored).

---

## 14. Remaining risks / review items

1. **12 `balance_core_*` dirs (~377 M) were conservatively NOT deleted.** The `clean_outputs_bulk.py` guard refuses any path containing `outputs/balance`, which over-matches `outputs/balance_core_*`. This is a safe false-positive (kept more than required). Their summaries were already extracted, so a follow-up could delete them explicitly if the disk is needed. The largest are `balance_core_validation` (173 M) and `balance_core_position_containment` (86 M).
2. A few raw top-level `.txt`/`.log` files remain directly under `outputs/` (e.g. `stage2b_*_with_ownership_mask.log`) — small, left in place.
3. The deleted diagnostic dirs are regenerable but expensive; their authoritative numbers now live only in the extracted summaries + kept validation reports. Acceptable per the approved policy.

---

## Final classification

**DEEP_OUTPUTS_CLEANUP_PASS_WITH_REMAINING_REVIEW_ITEMS**

Rationale: ~7.51 G reclaimed, all protected checkpoints/setups/backups verified intact, runtime verification PASS, summaries archived. One benign over-refusal (12 `balance_core_*` dirs kept) remains as an optional follow-up — hence PASS_WITH_REMAINING_REVIEW_ITEMS rather than plain PASS.
