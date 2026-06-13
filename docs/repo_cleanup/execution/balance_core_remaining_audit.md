# Balance Core Dirs — Remaining Audit

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Phase:** 2 (re-audit after fixed guard)

---

## Sizes and classification

| Directory | Size | Classification | Manifest |
|---|---|---|---|
| `outputs/balance_core_validation` | 173 M | SUMMARIZE_THEN_DELETE | YES |
| `outputs/balance_core_position_containment` | 86 M | SUMMARIZE_THEN_DELETE | YES |
| `outputs/balance_core_position_containment_e0b` | 40 M | SUMMARIZE_THEN_DELETE | YES |
| `outputs/balance_core_position_aware_precheck_5000` | 32 M | DELETE_DIRECT | NO |
| `outputs/balance_core_e0_cleanup_validation_5000` | 32 M | DELETE_DIRECT | NO |
| `outputs/balance_core_e0_cleanup_validation` | 6.4 M | DELETE_DIRECT | NO |
| `outputs/balance_core_position_aware_precheck_1000` | 6.4 M | DELETE_DIRECT | NO |
| `outputs/balance_core_true_height_variants` | 84 K | SUMMARIZE_THEN_DELETE | YES |
| `outputs/balance_core_extended_height_range` | 58 K | SUMMARIZE_THEN_DELETE | YES |
| `outputs/balance_core_longevity_height_sweep` | 25 K | SUMMARIZE_THEN_DELETE | YES |
| `outputs/balance_core_extended_longevity` | 8 K | SUMMARIZE_THEN_DELETE | YES |
| `outputs/balance_core_height_recovery` | 4 K | DELETE_DIRECT | NO |
| **Total** | **~415 M** | 12 dirs | **7 have manifests** |

## Tracked status check

`git ls-files outputs/balance_core*` → **0 tracked files**. All are untracked/gitignored. Safe to delete.

## Protected-path guard (fixed)

The updated `clean_outputs_bulk.py` now uses resolved-path parent containment, not substring matching:

| Path | is_protected |
|---|---|
| `outputs/balance` | **True** |
| `outputs/balance/rl/seed42` | **True** |
| `outputs/balance_core_validation` | **False** ✓ |
| `outputs/balance_core_position_containment` | **False** ✓ |

Self-check: `SELF_CHECK_PASS`.

## Decision

All 12 `balance_core_*` dirs are untracked, gitignored, and classified `SUMMARIZE_THEN_DELETE` (7) or `DELETE_DIRECT` (5). Their summaries were either already extracted (7) or need extraction for the 5 without manifests before deletion.