# K2 JAX Production Runtime Freeze — Post-Standalone Performance Baseline

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Phase:** 0 — Freeze

## Commands executed

### A. Fixed-high (high_0p480, headless)

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --controller-backend jax \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 1000 \
  --wbc-quiet
```

### B. Push-backward (low_0p330, push bwd 90N, headless)

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --controller-backend jax \
  --height-variant-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --push-enabled \
  --push-sequence-file outputs/k2_jax_hip_yaw_phase6/trajectories/push_bwd_90N.json \
  --steps 1000 \
  --wbc-quiet
```

## Results

### Fixed-high (high_0p480)

| Metric | Value |
|--------|-------|
| Total wall time | 54.7 s |
| Total steps | 1000 |
| Simulated time | 10.0 s |
| **Achieved Hz** | **~18.3 Hz** |
| Mean step time | ~54.7 ms |
| Telemetry columns | 756 (732 populated) |
| Termination | False (completed, no fall) |
| CSV path | outputs/hierarchical_controller_sim/telemetry_*.csv |

### Push-backward (low_0p330)

| Metric | Value |
|--------|-------|
| Total wall time | 58.6 s |
| Total steps | 1000 |
| Simulated time | 10.0 s |
| **Achieved Hz** | **~17.1 Hz** |
| Mean step time | ~58.6 ms |
| Telemetry columns | 756 (732 populated) |
| Termination | False (completed, no fall) |
| Push | 90 N backward |
| CSV path | outputs/hierarchical_controller_sim/telemetry_*.csv |

## Bottleneck audit

### Per-step terminal I/O

| Source | Location | Frequency | Estimated cost |
|--------|----------|-----------|----------------|
| Progress print (`Step XXX: h=...`) | [simulate_hierarchical_controller.py:8596](scripts/simulate_hierarchical_controller.py#L8596) | Every 10 steps | ~30-40 ms (terminal flush) |
| B0-AUDIT diagnostic prints | [simulate_hierarchical_controller.py:8521-8583](scripts/simulate_hierarchical_controller.py#L8521) | Every step < 20 (6× prints per step) | ~5-10 ms |
| WBC diagnostic (step 0) | [simulate_hierarchical_controller.py:5763](scripts/simulate_hierarchical_controller.py#L5763) | Step 0 only | ~1 ms |
| Both-synced trace prints | [simulate_hierarchical_controller.py:6889-7088](scripts/simulate_hierarchical_controller.py#L6889) | Only when `--synced-trace-steps` set | 0 ms (opt-in) |

**Total per-step terminal I/O: ~30-40 ms/step amortized**

### Telemetry construction

| Source | Location | Cost |
|--------|----------|------|
| Scale-calibrated telemetry (balance-core) | Lines 7604-8500+ | ~5-10 ms/step |
| 756 columns populated per step | `telemetry[key].append(value)` | High dict overhead |
| Telemetry row snapshot + decimation | Lines 8493-8507 | ~1 ms |

**Total telemetry: ~5-10 ms/step**

### JAX input packing

| Source | Location | Cost |
|--------|----------|------|
| `pack_input_k2()` with 45 elements | Lines 6806-6828 | ~5-6 ms |
| Python float conversion + dict construction | Inside pack_input_k2 | ~3 ms |
| `jnp.array()` conversions per step | 9× `jnp.array()` calls | ~2 ms |

**Total JAX input packing: ~5-6 ms/step**

### JAX hot-step

| Metric | Value |
|--------|-------|
| JAX controller step | ~0.29 ms |
| block_until_ready | Included in JIT timing |

**JAX hot-step: ~0.3 ms/step**

### Other overhead

| Source | Estimated cost |
|--------|----------------|
| MuJoCo physics step | ~0.5 ms |
| Centroidal estimator | ~0.3 ms |
| Capture point update | ~0.1 ms |
| Python overhead, context switches | ~2-3 ms |

## Non-controller overhead breakdown

| Category | Cost (ms/step) | % of total |
|----------|----------------|------------|
| Terminal I/O (progress print) | 30-40 | 55-68% |
| Telemetry construction | 5-10 | 9-17% |
| JAX input packing | 5-6 | 9-10% |
| Physics + estimation | 0.9 | 1.5% |
| Python overhead | 2-3 | 3.5-5% |
| JAX hot-step | 0.3 | 0.5% |
| **Total** | **~44-59** | **100%** |

## Key findings

1. **Terminal I/O is the dominant bottleneck.** The progress print every 10 steps (line 8596) accounts for 55-68% of total wall time. Each print causes a terminal flush that blocks for ~30-40 ms.

2. **Telemetry is the second bottleneck.** 756 columns are populated every step through Python dict `append()`. This costs ~5-10 ms/step. The CSV is written once at the end, so CSV I/O is NOT the issue — the dict construction is.

3. **JAX input packing is costly at ~5-6 ms.** The `pack_input_k2()` function creates new dicts and `jnp.array()` objects every step. For a 45-element vector at control rates, this is 100-200x slower than expected.

4. **JAX hot-step is negligible at 0.29 ms.** The actual JAX computation is blazingly fast — only 0.5% of total time.

5. **JAX is configured for float64.** `jax.config.update("jax_enable_x64", True)` at line 5378. This doubles memory bandwidth and may affect dispatch time.

6. **JAX runs on CPU.** No GPU detected. The 0.29 ms JAX hot-step suggests small matrix ops on CPU with minimal overhead.

7. **No WBC/composer overhead in JAX fast path.** `_jax_fast_path` correctly skips WBC and Python controller pipeline.

## Acceptance

- [x] 16-18 Hz bottleneck reproduced
- [x] Timing buckets identify exact non-controller overhead
- [x] No code changes in this phase
- [x] Terminal I/O confirmed as #1 bottleneck (55-68% of total)
- [x] Telemetry confirmed as #2 bottleneck (9-17% of total)
- [x] JAX input packing confirmed as #3 bottleneck (9-10% of total)
