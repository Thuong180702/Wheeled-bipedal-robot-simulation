# K2 JAX Visual Runtime Audit — Phase 1: Headless Baseline Timing

**Date:** 2026-06-29
**Status:** ROOT CAUSE IDENTIFIED

## Commands Tested

1. Implicit default (no --controller-backend): resolves to `jax` ✓
2. Explicit `--controller-backend jax`: resolves to `jax` ✓

## Phase 0: Backend Resolution

- Both implicit and explicit correctly select JAX backend
- No Python fallback
- No both-synced unless explicitly requested

## Phase 1: Headless Timing

### run-with --profile-controller (100 steps, --wbc-quiet)

| Component | Mean time (ms/step) |
|-----------|---------------------|
| centroidal_control | 6.10 |
| capture_control | 1.36 |
| **balance_core_block** | **48.48** |
| centroidal_log | 4.35 |
| capture_log | 1.03 |
| telemetry | 10.99 |
| **total_per_step** | **130.02** |

Achieved: ~7.7 Hz (100 steps in 17.7s)

### Run with --stage7-benchmark (200 steps, warmup=10, measured=100)

| Metric | Value |
|--------|-------|
| **JAX hot-step mean** | **0.284 ms** |
| JAX hot-step p95 | 0.378 ms |
| JAX JIT compile | 1.45s (one-time) |
| **Total step mean** | **105.411 ms** |
| Achieved sim rate | ~9.5 Hz |
| Realtime factor | 0.04 |

## Root Cause

**The Python WBC + BALANCE-CORE controller pipeline runs EVERY step in JAX mode.**

JAX itself takes 0.284 ms per step. But the full Python pipeline (WBC QP solve → sagittal controller → lateral controller → composer → telemetry) takes ~105 ms per step — **370x overhead**.

The Python controller output is then **discarded** at line 7100 where `tau_smooth = _jax_tau` replaces it with JAX output.

No code path exists to skip the Python controller when `_jax_enabled` is True.

## Key Findings

1. **JAX compile:** One-time, 1.45s. No repeated recompilation. ✓
2. **JAX step speed:** 0.284 ms (3500+ Hz). JAX controller is NOT the bottleneck. ✓
3. **Python WBC QP solve:** ~106 ms at step 0, remains expensive throughout. ✗
4. **No conditional skip:** Python controller always runs, even in pure JAX mode. ✗
5. **Wasted work:** Python controller output discarded at line 7100 when JAX overrides tau_smooth. ✗
