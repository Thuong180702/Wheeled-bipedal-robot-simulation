# Stage 7B: Dynamic/Push Stability Fix Report

**Date:** 2026-06-27
**Classification:** `STAGE7B_PARTIAL_PASS_WITH_BLOCKERS` — 2/5 fixed, 3/5 need deeper teacher-forcing

## Root Cause Analysis

### Bug #1 (FIXED): Knee torque bypasses composer clipping
The JAX post-hoc support feedforward (-7.75/-7.9 Nm on knees) was applied AFTER the composer, bypassing `max_position_tau` clipping. Python's support FF goes through the composer and is clipped to -4.0 Nm.

**Fix:** Moved `k2_jax_empirical_support_ff()` into `tau_sum` before the JAX composer.
**Impact:** ramp_up PASS (was FAIL@3067), gate_chatter PASS (was FAIL@1549)

### Bug #2 (FIXED): Extra hip-yaw support FF has no Python equivalent
`k2_jax_support_feedforward_compute` applies height-gated hip_yaw torque based on support position error. Python has no equivalent mechanism.

**Fix:** Removed `tau_support_ff` from `tau_sum`.
**Impact:** No regression; gate_chatter still passes.

### Remaining Issue: Subtle internal state divergence
ramp_down, push_fwd, and push_bwd still fail. The root cause is NOT in the initial torque (verified: step 0 torques match, contact forces match at 163.456 N). The divergence accumulates over many steps as the JAX controller's internal state (notch filter, outer loop, adaptive bias trim ring buffer) evolves differently from Python.

## Results

| Scenario | Before | After Fix 1 | After Fix 2 |
|----------|--------|------------|------------|
| ramp_up 0.33→0.48 | FAIL@3067 | **PASS** | -- |
| gate_chatter 0.40-0.47 | FAIL@1549 | **PASS** | -- |
| ramp_down 0.48→0.33 | FAIL@4159 | FAIL@1789 ⚠️ | FAIL@1789 |
| push_fwd_90N | FAIL@699 | FAIL@699 | FAIL@699 |
| push_bwd_90N | FAIL@756 | FAIL@756 | FAIL@756 |

⚠️ ramp_down got WORSE after Fix 1 — knee torque reduction from -7.75 to -4.0 Nm reduces descent control authority.

## Files Changed

### `wheeled_biped/controllers/k2_jax_controller.py`
1. **Added** `_K2_EMPIRICAL_SUPPORT_FF` constant and `k2_jax_empirical_support_ff()` helper
2. **Modified** `tau_sum` composition: added empirical FF, removed hip-yaw support FF
3. Diff: ~15 lines added, ~5 lines removed

### `scripts/simulate_hierarchical_controller.py`
1. **Removed** post-hoc empirical support FF (7 lines): hip_pitch/knee additions
2. Diff: ~7 lines removed

## Tests

All 31 tests pass (step parity + CLI + benchmark smoke). No regressions.

## What's Needed to Fix Remaining 3 Scenarios

The remaining failures require **full per-component teacher-forcing** (Phase 4):
1. Run Python with full telemetry for ramp_down, push_fwd, push_bwd
2. Compute JAX torque from identical physics states
3. Compare all 10 torque components per step
4. Identify which components diverge (notch? outer loop? ABS? lateral roll?)
5. Fix only the proven cause

The external torque (empirical FF) is now correct. The internal state handling (notch filter, outer loop state, ABS ring buffer) likely has subtle differences in how it accumulates over hundreds of steps, causing gradual height loss during descent and push recovery.

## Verdict

**STAGE7B_PARTIAL_PASS_WITH_BLOCKERS** — 2/5 dynamic/push scenarios fixed. The controller compute is realtime-ready (JAX hot-step < 0.3 ms). The remaining 3 scenarios need deeper investigation of internal state divergence, which is recommended for a separate investigation stage.
