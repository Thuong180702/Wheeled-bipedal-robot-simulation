# Deep Outputs Cleanup — Final Report

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Scope:** Deep clean of the remaining `outputs/` bulk (~8.2 G) with mandatory summary extraction before deletion. No tracked files, training code, configs, checkpoints, or T6J setup assets were touched.

---

## 1–3. Disk impact (updated after balance_core cleanup)

| Metric | Value |
|---|---|
| `outputs/` before any cleanup | **8.2 G** |
| `outputs/` after first deep pass | **~651 M** |
| `outputs/` after balance_core cleanup | **~276 M** |
| **Total disk freed** | **~7.9 G** (8,459,212,490 bytes) |
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

- `outputs/balance` (PROTECT) — trained seed trees intact
- `outputs/physical_target_height_setups` (PROTECT)
- `backup_checkpoints/` (PROTECT, tracked, not `git rm`'d)
- 7 small top-level `.txt`/`.log` files directly under `outputs/` (a few KB each — left in place)

## 10. Summary archive location

`archive/cleanup_2026-06-13/output_summaries/` — **166** per-dir folders (154 first-pass + 12 balance_core), global manifest at `docs/repo_cleanup/execution/deep_outputs_summary_extraction_manifest.json`.

---

## 11. Tests / compile

Compile OK (5 modules). Key tests: **47 passed, 1 skipped**.

## 12. Smoke simulation

T6J @ high_0p480, 100/100 steps, **no fall**. Telemetry regenerated cleanly into a fresh `hierarchical_controller_sim/`.

## 13. Git status

Only new/untracked docs under `docs/repo_cleanup/execution/`, the two cleanup scripts, and the `archive/.../output_summaries/` tree. **No tracked file deleted or modified** by the output cleanup (deleted dirs were all gitignored).

---

## 14. Remaining risks / review items

**None.** The `balance_core_*` over-refusal was resolved in a second pass: the guard was fixed to exact-path parent containment, all 12 dirs passed self-check + manifest check + tracked-file check + protected-path check, and all 12 were deleted with 0 failures.

1. The deleted diagnostic dirs are regenerable but expensive; their authoritative numbers live in the extracted summaries + kept validation reports. Acceptable per the approved policy.
2. The summary archive at `archive/cleanup_2026-06-13/output_summaries/` (81 M) is gitignored — confirm your backup strategy covers it if needed.

---

## Final classification

**DEEP_OUTPUTS_CLEANUP_PASS**

Total freed: ~7.9 G across two passes (8,459,212,490 bytes). All protected checkpoints/setups/backups verified intact. 170 raw diagnostic dirs deleted after summary extraction. 1,009 summary artifacts archived. Runtime verification PASS. No tracked file modified.
