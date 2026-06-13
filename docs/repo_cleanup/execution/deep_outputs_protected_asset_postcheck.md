# Deep Outputs Cleanup — Protected Asset Post-Check

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Phase:** 5 (post-deletion verification)

---

## Protected assets after bulk deletion

| Asset | Status |
|---|---|
| `outputs/balance/rl/seed42/checkpoints/final/checkpoint.pkl` | OK |
| `outputs/balance/rl/seed113/checkpoints/final/checkpoint.pkl` | OK |
| `outputs/balance/rl/seed999/checkpoints/final/checkpoint.pkl` | OK |
| `outputs/balance/rl/seed42/tb` | OK |
| `outputs/balance/rl/seed113/tb` | OK |
| `outputs/balance/rl/seed999/tb` | OK |
| `outputs/balance/rl/seed42/balance_seed42_metrics.jsonl` | OK |
| `outputs/balance/rl/seed113/balance_seed113_metrics.jsonl` | OK |
| `outputs/balance/rl/seed999/balance_seed999_metrics.jsonl` | OK |
| `outputs/physical_target_height_setups/` | OK |
| `backup_checkpoints/` | OK |

**FAIL count: 0** — all protected assets survived the deletion.

---

## Size before / after

| Metric | Value |
|---|---|
| `outputs/` before deep cleanup | 8.2 G |
| `outputs/` after deep cleanup | ~651 M |
| Freed | ~7.5 G |

Note: `outputs/hierarchical_controller_sim/` was deleted in Phase 4 (713 M raw), then the Phase 6 T6J smoke regenerated a fresh ~1.1 M telemetry file in it — expected and harmless.

---

## `clean_outputs_bulk.py` safety refusals

The deletion script independently refused 12 `balance_core_*` directories because their path prefix matched the protected `outputs/balance` guard substring. These were **not** the trained-policy `outputs/balance/rl` dirs — they are separate diagnostic dirs (`balance_core_validation`, `balance_core_position_containment`, etc.) — but the script's conservative substring guard kept them rather than risk a false delete. They remain on disk (~378 M) and are carried into the Phase 7 report as a remaining-review item.

**Classification: protected assets intact — proceed.**
