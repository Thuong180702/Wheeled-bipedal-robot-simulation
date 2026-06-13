# Balance Core Cleanup — Post-Check

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Phase:** 5

---

## Protected assets after balance_core deletion

| Asset | Status |
|---|---|
| `outputs/balance/rl/seed42/checkpoints/final/checkpoint.pkl` | OK |
| `outputs/balance/rl/seed113/checkpoints/final/checkpoint.pkl` | OK |
| `outputs/balance/rl/seed999/checkpoints/final/checkpoint.pkl` | OK |
| `outputs/balance/rl/seed42/tb` | OK |
| `outputs/balance/rl/seed113/tb` | OK |
| `outputs/balance/rl/seed999/tb` | OK |
| `outputs/physical_target_height_setups/` | OK |
| `backup_checkpoints/` | OK |

**FAIL count: 0** — all protected assets survived the balance_core deletion.

---

## Size impact

| Metric | Value |
|---|---|
| `outputs/` before balance_core cleanup | ~651 M |
| `outputs/` after balance_core cleanup | **~276 M** |
| Additional freed | **~375 M** (12 dirs × 392,398,009 bytes) |

`outputs/` now contains only the preserved `balance/` seed trees (275 M) and a tiny `hierarchical_controller_sim/` (~1.1 M from the smoke run). All 12 `balance_core_*` diagnostic dirs are gone.

---

## Guard verification

`is_protected()` self-check passed: `outputs/balance` and `outputs/balance/rl/seed42` are protected; `outputs/balance_core_validation` is not. All 12 dirs passed the tracked-file check, manifest check, and path check before deletion. 0 failures.