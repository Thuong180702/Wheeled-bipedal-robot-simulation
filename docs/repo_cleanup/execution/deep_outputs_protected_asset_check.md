# Deep Outputs — Protected Asset Pre-Check

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Phase:** 1 (before any deletion)

---

## Checkpoint files

| Path | Status |
|---|---|
| `outputs/balance/rl/seed42/checkpoints/final/checkpoint.pkl` | OK |
| `outputs/balance/rl/seed113/checkpoints/final/checkpoint.pkl` | OK |
| `outputs/balance/rl/seed999/checkpoints/final/checkpoint.pkl` | OK |

## Metrics JSONL

| Path | Status |
|---|---|
| `outputs/balance/rl/seed42/balance_seed42_metrics.jsonl` | OK |
| `outputs/balance/rl/seed113/balance_seed113_metrics.jsonl` | OK |
| `outputs/balance/rl/seed999/balance_seed999_metrics.jsonl` | OK |

## TensorBoard event dirs

| Path | Status |
|---|---|
| `outputs/balance/rl/seed42/tb` | OK |
| `outputs/balance/rl/seed113/tb` | OK |
| `outputs/balance/rl/seed999/tb` | OK |

## Other protected

| Path | Status |
|---|---|
| `outputs/physical_target_height_setups` | OK |
| `backup_checkpoints/` | OK |

---

**FAIL count: 0** — all protected assets present. Cleanup may proceed to Phase 2 (audit).
