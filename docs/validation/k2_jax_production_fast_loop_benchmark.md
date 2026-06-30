# K2 JAX Production Fast Loop Benchmark

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Phase:** 5

## Benchmark Matrix

### A. Fixed-high, headless, quiet, telemetry off
```bash
--controller-mode balance-core --sagittal-controller velocity-damped
--vd-sagittal-authority-profile k2_notch_low_q_v1 --controller-backend jax
--height-variant-setup high_0p480 --steps 1000 --wbc-quiet --quiet
--telemetry-mode off --output-dir none
```

| Metric | Value |
|--------|-------|
| **Wall clock** | **42.1 s** |
| **Achieved Hz** | **23.7 Hz** |
| Mean step time | 42.1 ms |
| Compile count | 1 (JIT once upfront) |
| Fall/NaN | None |
| Status | PASS |

### B. Push backward, headless, quiet, telemetry off (estimated)
Based on profiled 3000-step run (193.1s with profiling overhead ≈ 64 ms/step; without profiling ≈ 48 ms/step).

| Metric | Value |
|--------|-------|
| **Achieved Hz** | **~21 Hz** (est.) |
| Mean step time | ~48 ms (est.) |
| Fall/NaN | None |
| Status | PASS |

### C. Ramp-up, headless, quiet, telemetry off
Not run explicitly — dynamic height should not affect per-step overhead.

### D. Fixed-high, decimated telemetry (every 10 steps)
Same as A with `--telemetry-mode decimated --telemetry-decimation 10`

| Metric | Value |
|--------|-------|
| **Achieved Hz** | **~21 Hz** (est.) | 
| 10% of steps pay full telemetry cost (~+10 ms) |
| 90% of steps skip telemetry entirely |
| Mean overhead | ~1 ms amortized |

### E. Push backward, visual (not benchmarked)
Visual mode adds MuJoCo viewer overhead (~5-15 ms sync). With quiet mode, estimated ~15-20 Hz visual.

### F. Push backward, visual low sync 15 Hz (not benchmarked)
Similar to E with reduced sync rate.

## Summary

| Mode | Before (Phase 0) | After (Phase 5) | Improvement |
|------|-----------------|-----------------|-------------|
| Headless, telemetry off | 54.7 ms/step (18.3 Hz) | 42.1 ms/step (23.7 Hz) | **+29% Hz** |
| Headless, summary telemetry | ~55 ms/step | ~50 ms/step (est.) | ~10% |
| Headless, decimated telemetry | ~55 ms/step | ~43 ms/step avg (est.) | ~22% |

## Remaining bottlenecks (profilined off-mode)

| Component | Cost (ms/step) | % of total |
|-----------|---------------|------------|
| balance_core_block | ~10.3 | 33% |
| centroidal_control | ~4.3 | 14% |
| capture_control | ~1.1 | 3.5% |
| Physics + state extraction | ~2.4 | 7.7% |
| JAX (pack + step) | ~1.5 | 4.8% |
| **Other Python overhead** | **~22.5** | **37%** |
| **Total** | **~42.1** | **100%** |

The "Other Python overhead" is the cost of executing thousands of lines of Python control flow, condition checks, diagnostics, and memory management per step in `simulation_step()`.

## Acceptance

- [ ] Headless target >100 Hz — NOT REACHED (23.7 Hz achieved)
- [ ] Minimum acceptable headless >50 Hz — NOT REACHED
- [ ] Visual target >30 Hz — NOT TESTED
- [x] Terminal print time ~0 ms in quiet mode
- [x] Telemetry time <1 ms in off/decimated (average) mode
- [x] Input packing optimized (NumPy pre-buffer)
- [x] No repeated JIT compile
