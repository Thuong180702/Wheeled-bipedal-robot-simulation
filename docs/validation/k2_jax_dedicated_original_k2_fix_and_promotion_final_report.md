# K2 JAX Dedicated Realtime Runner — Fix and Promotion Final Report

**Date:** 2026-06-29
**Commit (baseline):** `0e1c713` — "Stage 6K: Dynamic runner extended"
**Commit (post-fix):** Working tree on `repo-cleanup-t6j`

## Final Classification

**K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL**

PARTIAL because hip-yaw divergence >0.35 rad at 3 lowest heights (pre-existing K2 limit, canonical JAX matches). NOT BLOCKED because all dedicated-runner-specific failures are fixed.

---

## Root Causes Found & Fixed

### 1. Physics substep mismatch (CRITICAL — caused ALL dynamic height failures)
- Dedicated runner: `mujoco.mj_step()` **1×** per step
- Canonical path: `mujoco.mj_step()` **5×** per step (physics_dt=0.002, control_dt=0.01)
- **Result:** Physics at 1/5 rate → dynamic trajectories 5× too fast → all fail
- **Fix:** Added `for _ in range(_n_substeps): mj_step()` loop

### 2. Parameter source-of-truth (MAINTAINABILITY)
- Removed hardcoded `K2_PROFILE` dict, import from canonical `K2_NOTCH_LOW_Q_V1`
- Added `--dump-k2-params` flag
- 0 parameter mismatches

## Promotion Matrix

| | Before | After |
|---|---|---|
| ramp_up | ❌ Fall @ 2989 | ✅ 5000/5000 |
| ramp_down | ❌ Fall @ 4471 | ✅ 5000/5000 |
| gate_chatter | ❌ Fall @ 2288 | ✅ 5000/5000 |
| Step E (10 heights) | Not tested | ✅ 10/10 no falls |
| low_0p300 hip-yaw | 0.666 rad | 0.412 rad (−38%) |
| Push 90N | Survived (artificially weak) | Falls (matches canonical) |

## Tests: 44/44 PASS

## Deliverables in `docs/validation/`
- `k2_jax_dedicated_fix_baseline.md`
- `k2_jax_dedicated_param_source_fix.md`
- `k2_jax_dedicated_orchestration_alignment.md`
- `k2_jax_dedicated_post_fix_promotion_scope.md`
- `k2_jax_dedicated_original_k2_fix_and_promotion_final_report.md`
