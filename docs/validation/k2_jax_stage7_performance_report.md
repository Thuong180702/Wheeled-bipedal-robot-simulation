# Stage 7: JAX K2 Controller Performance Benchmark Report

**Date:** 2026-06-27
**Classification:** PENDING -- see summary below

## Environment

- **Python:** 3.10.2
- **JAX:** 0.6.2
- **jaxlib:** 0.6.2
- **MuJoCo:** 3.6.0
- **Platform:** Windows-10-10.0.26200-SP0
- **CPU:** Intel64 Family 6 Model 142 Stepping 12, GenuineIntel
- **JAX x64:** True

## Benchmark Configuration

- **Warmup steps:** 100
- **Measured steps:** 1000
- **Control dt:** 0.01 s (100 Hz)
- **Controller mode:** balance-core
- **Sagittal profile:** k2_notch_low_q_v1

## Headless Benchmark Results

| Scenario | Py Total Mean (ms) | Py Total p95 (ms) | JX Total Mean (ms) | JX Total p95 (ms) | JX Hot-Step Mean (ms) | JX Hot-Step p95 (ms) | Speedup |
|----------|-------------------|------------------|-------------------|------------------|----------------------|---------------------|---------|
| fixed_high_0p480 | 150.831076 | 197.0025 | 110.964937 | 132.9229 | 0.273329 | 0.3449 | 1.36 |

**Note:** 'Hot-Step' = JIT execution only (with `block_until_ready()`). Compile time excluded.

## JAX Path Overhead Breakdown

| Scenario | Pack Input (ms) | JIT Step (ms) | Support FF (ms) | Diag Map (ms) | JAX Total (ms) |
|----------|----------------|--------------|----------------|--------------|---------------|
| fixed_high_0p480 | 17.199669 | 0.273329 | 0.0 | 0.001071 | 17.474069 |

## JIT Compilation

| Scenario | Compile Time (s) | Recompilations |
|----------|-----------------|----------------|
| fixed_high_0p480 | 1.0675 | 0 |

## Validation During Benchmark

| Scenario | Py Fell | JX Fell | Py NaN | JX NaN | Py HipYaw (rad) | JX HipYaw (rad) |
|----------|---------|---------|--------|--------|----------------|----------------|
| fixed_high_0p480 | False | False | False | False | 0.015362253412604332 | 0.02058192023683415 |

## Recompilation Audit

- **Input flat shape:** [41]
- **State flat shape:** [328]
- **Params shape:** [31]
- **Diag flat shape:** [30]
- **Static args:** none (params, state, input all passed as traced arrays)
- **Dynamic height recompiles:** none expected — target height passed in input_flat each step
- **Telemetry mode recompiles:** none — telemetry decimation does not affect JIT shapes
- **Headless vs visual:** unchanged — same input/state/params shapes in both modes
- **Recompilation count:** 0

## Bottleneck Analysis

### Python Backend Bottlenecks

- **Python mean total step:** 150.8 ms
- **JAX mean total step:** 111.0 ms
- **JAX hot-step mean:** 0.273 ms (JIT only, with block_until_ready)

### Primary bottleneck: Python balance-core computation + telemetry (~100+ ms)
The JAX JIT step itself is extremely fast (<1 ms), but the total step time is dominated by:
1. Python balance-core controller computation (runs in both backends for telemetry)
2. Input packing (Python -> JAX device transfer, ~20 ms)
3. Telemetry dict construction per step
4. Duplicate state estimation (control + log phases)
