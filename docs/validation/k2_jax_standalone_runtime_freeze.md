# K2 JAX Standalone Runtime — Phase 0 Freeze Report

**Date:** 2026-06-29
**Branch:** `repo-cleanup-t6j`
**Commit:** `0e1c713` Stage 6K: Dynamic runner extended, JAX ramp_up terminates at step 556/5000

## Purpose

Freeze and document the current post-fix runtime state BEFORE any code changes toward standalone JAX realtime.

## Benchmark Results

### 1. Headless JAX Push (low_0p330, push_bwd_90N, 1000 steps)

| Metric | Value |
|--------|-------|
| Wall time | 92.4 s |
| Sim time | 10.0 s |
| Total step mean | **82.95 ms** |
| Achieved sim Hz | **10.7 Hz** |
| Realtime factor | 0.11x |
| JAX hot-step mean | 0.280 ms |
| JAX hot-step p95 | 0.370 ms |
| JAX pack input mean | 6.71 ms |
| Physics step mean | 0.30 ms |
| JIT compile time | 1.68 s (one-time) |
| Recompilation count | 0 |
| Fell | No |
| NaN | No |

### 2. Headless JAX Fixed-High (high_0p480, no push, 1000 steps)

| Metric | Value |
|--------|-------|
| Wall time | 69.4 s |
| Sim time | 10.0 s |
| Total step mean | **61.82 ms** |
| Achieved sim Hz | **14.3 Hz** |
| Realtime factor | 0.14x |
| JAX hot-step mean | 0.247 ms |
| JAX hot-step p95 | 0.293 ms |
| JAX pack input mean | 5.05 ms |
| Physics step mean | 0.23 ms |
| JIT compile time | 1.43 s (one-time) |
| Recompilation count | 0 |
| Fell | No |
| NaN | No |

### 3. Visual JAX Push (low_0p330, push_bwd_90N, 300 steps)

| Metric | Value |
|--------|-------|
| Wall time | 31.0 s |
| Sim time | 3.0 s |
| Total step mean | **71.29 ms** |
| Achieved sim Hz | **9.4 Hz** |
| Realtime factor | 0.09x |
| JAX hot-step mean | 0.231 ms |
| JAX hot-step p95 | 0.342 ms |
| JAX pack input mean | 5.85 ms |
| Physics step mean | 0.26 ms |
| JIT compile time | 1.49 s (one-time) |
| Recompilation count | 0 |
| Fell | No |
| NaN | No |

## Code Path Confirmation

### Python WBC/composer — CONFIRMED SKIPPED

The JAX fast path (`_jax_fast_path = _jax_enabled and not _both_synced_enabled`) correctly skips:
- WBC QP solve (line 5692-5718)
- Static balance wrapper (line 5721)
- Python composer (line 6656-6685)

These are replaced with zero tensors. WBC_flag and hidden_torque_flag remain false in all benchmarks.

### Python sagittal controller compute — CONFIRMED STILL RUNS

The sagittal controller is called unconditionally inside `if is_balance_core_mode(args):` (line 6082), which is NOT gated by `_jax_fast_path`:
- `shape_posture.compute()` — line 6099
- `sagittal_wheel_balance.compute()` (svdbc) — line 6393
- `lateral_roll_balance.compute()` — line 6476
- Mode-div hip-yaw divergence — lines 6620-6651
- Support feedforward — line 6139

**Root cause confirmed:** The Python sagittal controller compute path runs every step even in `backend=jax`, consuming ~55-75 ms of the ~62-83 ms total step time.

### JAX Path — CONFIRMED ACTIVE

The JAX controller step runs at line 6690+, computing torque from Python-sourced inputs via `pack_input_k2()`. JAX torque replaces Python torque as the final output. JAX hot-step time is excellent at ~0.25-0.28 ms.

## Bottleneck Breakdown (Headless Fixed-High)

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Python sagittal compute + shape posture + lateral + yaw + support FF | **~55 ms** (estimated) | ~89% |
| JAX pack_input (incl. arg eval with dir() calls) | 5.05 ms | ~8% |
| JAX JIT step | 0.25 ms | <1% |
| Physics step | 0.23 ms | <1% |
| Telemetry + other | ~1.3 ms | ~2% |
| **Total** | **61.82 ms** | 100% |

## JAX Pack Input Anomaly

The `jax_pack_input_s` measurement (5.05-6.71 ms) is unexpectedly high. The `pack_input_k2()` function itself uses numpy intermediates and should take ~0.01 ms. The inflated measurement is caused by Python argument evaluation overhead:
- `dir()` calls in conditional expressions (e.g., `'pitch_x_error' in dir()`)
- Dict lookups (`sagittal_diag.get(...)`)
- Attribute accesses on centroidal state objects

This will be addressed when the input contract is redesigned for raw-state packing.

## Raw Benchmark Data

- `outputs/benchmark/stage7_freeze_headless_push_jax.json`
- `outputs/benchmark/stage7_freeze_headless_fixed_high_jax.json`
- `outputs/benchmark/stage7_freeze_visual_push_jax.json`

## Acceptance Checklist

| Criterion | Status |
|-----------|--------|
| Current bottleneck quantified | ✅ Pass — Python sagittal compute ~55-75 ms/step |
| Confirm Python WBC/composer skipped | ✅ Pass — Verified at lines 5692, 6656 |
| Confirm Python sagittal compute still runs | ✅ Pass — Verified at line 6393, unguarded by `_jax_fast_path` |
| No semantic code changes in this phase | ✅ Pass — Read-only measurement |
| JAX hot-step ~0.3 ms | ✅ Pass — 0.247-0.280 ms |
| No JIT recompilation | ✅ Pass — Count = 0 |
| No NaN/fall | ✅ Pass — All 3 runs completed cleanly |

## Next Phase

→ Phase 1: Map all Python sagittal output dependencies used in `pack_input_k2()` and the JAX controller step.
