# K2 JAX Production Print Suppression

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Phase:** 2

## Print sites audited and guarded

All per-step `print()` calls in `scripts/simulate_hierarchical_controller.py` were audited and classified:

### Per-step progress print (was bottleneck #1)
**Location:** `simulation_step()`, line ~8670
**Original:** Every 10 steps (headless) or 100 steps (visual)
**Fixed:** 
- Guarded by `if not _quiet:`
- In non-quiet production JAX: interval changed from 10 to 500 steps
- In `--quiet` mode: suppressed entirely

### B0-AUDIT diagnostic prints
**Location:** `simulation_step()`, line ~8592
**Original:** Every step < 20 in headless mode (6× prints per step)
**Fixed:** Added `and not _quiet` guard

### WBC diagnostic step 0
**Location:** `simulation_step()`, line ~5845
**Original:** Step 0 only, verbose diagnostic
**Fixed:** Added `and not _quiet` guard. Already guarded by `not _jax_fast_path` in production JAX.

### Wrapper telemetry prints
**Location:** `simulation_step()`, line ~5839
**Original:** Steps 0-19, `log_wrapper_telemetry()`
**Fixed:** Added `and not _quiet` guard

### STAGE 2 diagnostic prints
**Location:** `simulation_step()`, line ~5946
**Original:** Steps 0-9, not in `_jax_fast_path`
**Fixed:** Added `and not _quiet` guard

### Early support prints
**Location:** `simulation_step()`, line ~7494
**Original:** Steps 0-9, not in visual mode
**Fixed:** Added `and not _quiet` guard

### LIFECYCLE prints
**Location:** `simulation_step()`, line ~7599
**Original:** Steps 0-19, two prints per step
**Fixed:** Added `and not _quiet` guard

### Both-synced trace prints
**Location:** `simulation_step()`, lines ~6889-7088
**Original:** Only when `--synced-trace-steps` flag set
**Fixed:** No change needed — these are opt-in debug only

## Before/after

| Mode | Per-step prints | Terminal I/O |
|------|----------------|-------------|
| Python fallback (default) | Every 10 steps | ~3-4 ms amortized |
| Python fallback `--quiet` | 0 per step | ~0 ms |
| Production JAX (auto-quiet) | 0 per step | ~0 ms |
| Production JAX `--verbose` | Every 500 steps | ~0.1 ms amortized |

## Acceptance

- [x] Per-step print count = 0 in production JAX quiet mode
- [x] Terminal I/O time ~0 ms/step in quiet mode
- [x] Progress print retained at 500-step intervals when non-quiet
- [x] All debug/diagnostic prints guarded by `not _quiet`
- [x] Termination print always shown (critical event)
