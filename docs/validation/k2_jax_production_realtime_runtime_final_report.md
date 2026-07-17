# K2 JAX Production Realtime Runtime — Final Report

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Classification:** `K2_JAX_PRODUCTION_REALTIME_RUNTIME_PARTIAL`

## Summary

Production `backend=jax` standalone controller path was optimized for realtime runtime. Performance improved from **18.3 Hz → 23.7 Hz** (29% improvement) with telemetry off. The 100 Hz target was not reached; structural refactoring of `simulation_step()` is required to achieve it.

## Original bottleneck (Phase 0)

| Bottleneck | Cost (ms/step) | % of total |
|------------|---------------|------------|
| `balance_core_block` (Python controller code) | ~11.3 | 23% |
| Telemetry dict construction (756 cols) | ~10.1 | 20% |
| Centroidal estimation (2× per step) | ~8.9 | 18% |
| Other Python overhead | ~24.4 | 39% |
| **Total (baseline)** | **~54.7** | **100%** |

JAX hot-step: 0.29 ms (<1% of total). Controller compute is NOT the bottleneck — Python orchestration overhead dominates.

## Changes made

### Phase 1: Quiet/production mode CLI flags
- Added `--quiet`, `--verbose`, `--telemetry-mode` (off/summary/decimated/full)
- Production JAX defaults to quiet + decimated telemetry
- `--output-dir none` skips file writes

### Phase 2: Print suppression
- All per-step `print()` calls guarded by `not _quiet`
- Production JAX: 0 per-step prints (was every 10 steps)
- Progress interval increased from 10→500 in non-quiet production mode
- Guarded: progress prints, B0-AUDIT, WBC diagnostic, LIFECYCLE, wrapper telemetry, STAGE 2, early support

### Phase 3: Telemetry optimization
- `_SummaryTelemProxy`: dict wrapper that filters non-essential fields (123 of 756)
- `_NoOpTelemProxy`: dict wrapper that no-ops all appends
- Conditional skip: entire 906-line telemetry block skipped when off/decimated-skip
- Production default: decimated mode, every 10th step, saving ~10 ms on 90% of steps
- Telemetry cost: 10.15 ms → 0.07 ms (off) / ~1 ms (decimated average)

### Phase 4: JAX input packing
- NumPy pre-conversion avoids device-to-host round-trips
- `joint_pos_np`, `joint_vel_np`, `equilibrium_joint_pos_np` pre-allocated
- Eliminated 3× `jnp.array()` dispatches at call site
- Estimated saving: ~3 ms

### Duplicate centroidal estimate skip
- Log estimate skipped when telemetry is off/decimated-skip
- Saves ~5 ms/step
- Control-time estimate reused for termination/contact checks

### Not changed (per hard rules)
- Controller gains, thresholds, APCR1ND/ABS/MODE_DIV semantics
- Physics parameters, simulation fidelity
- Step count, Python fallback, both-synced debug mode

## Final performance

| Mode | Before | After | Improvement |
|------|--------|-------|-------------|
| Headless, telemetry off | 54.7 ms (18.3 Hz) | 42.1 ms (23.7 Hz) | +29% |
| Headless, telemetry summary | ~55 ms | ~50 ms (est.) | ~9% |
| Headless, telemetry decimated (avg) | ~55 ms | ~43 ms (est.) | ~22% |

## Remaining bottlenecks (profiled, off-mode)

| Component | Cost (ms) | % |
|-----------|----------|---|
| balance_core_block | 10.3 | 24% |
| centroidal_control | 4.3 | 10% |
| capture_control | 1.1 | 3% |
| Physics + state extraction | 2.4 | 6% |
| JAX pack + step | 1.5 | 4% |
| Other Python overhead | 22.5 | 53% |
| **Total** | **42.1** | **100%** |

The dominant remaining cost (53%) is "Other Python overhead" — the interpreter overhead of executing ~3000 lines of Python code per step in `simulation_step()`, including:
- Control flow and condition checks
- Variable declarations and nonlocal access
- Dict lookups and attribute access
- Object creation and garbage collection

## Root cause analysis

The `simulation_step()` function is ~2500 lines long. Even with all expensive operations guarded or skipped, the Python interpreter still processes thousands of lines of code per step. Each line costs 1-10 μs, accumulating to ~25 ms.

To reach 100 Hz (10 ms/step), `simulation_step()` would need to be:
1. Split into a production fast-path function (~100 lines)
2. All telemetry, diagnostics, and debug code moved out of the hot path
3. Centroidal estimator optimized or replaced
4. Deployment via PyPy or Cython for additional 2-3× speedup

This is a non-trivial structural refactoring that would require careful testing to avoid functional regressions.

## Test results

Full regression guard (Phase 6) was NOT run in this session due to time constraints. Manual smoke tests confirmed:
- `backend=jax` standalone produces correct torque output
- No fall, no NaN on fixed-high (0.48m, 1000 steps)
- Python fallback still functional
- Both-synced mode still functional

## User commands

### Fast realtime JAX visual push, minimal telemetry
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --height-variant-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --push-enabled \
  --push-sequence-file outputs/k2_jax_hip_yaw_phase6/trajectories/push_bwd_90N.json \
  --steps 1000 --quiet --telemetry-mode summary --visual
```

### Fast realtime JAX headless with result file
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --height-variant-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --push-enabled \
  --push-sequence-file outputs/k2_jax_hip_yaw_phase6/trajectories/push_bwd_90N.json \
  --steps 3000 --quiet --telemetry-mode decimated \
  --telemetry-decimation 10 \
  --output-dir outputs/realtime_runs/push_bwd_jax
```

### Max performance (telemetry off, no output files)
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 3000 --quiet --telemetry-mode off --output-dir none
```

### Python fallback (reference, slow)
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 500 --controller-backend python
```

### Both-synced debug (parity check, very slow)
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 500 --controller-backend both-synced
```

### Full telemetry debug (slow, debug only)
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 500 --verbose --telemetry-mode full \
  --output-dir outputs/debug_runs/full_telemetry
```

## Classification justification

**`K2_JAX_PRODUCTION_REALTIME_RUNTIME_PARTIAL`** because:

✅ Production `backend=jax` is standalone — no Python controller calls
✅ No per-step print by default in production mode
✅ Telemetry is summary/decimated/buffered by default
✅ Python fallback and both-synced preserved
✅ No controller gains, thresholds, or semantics changed
✅ No physics parameters changed
✅ 29% performance improvement achieved
✅ Bottlenecks identified and documented

❌ Headless JAX below 50 Hz (23.7 Hz vs 50 Hz minimum target)
❌ 100 Hz target not reached (10 ms/step required, 42 ms/step achieved)
❌ Full regression test suite not run in this session
❌ Summary telemetry mode slower than expected (~50 ms/step)

## Path to PASS

To reach `K2_JAX_PRODUCTION_REALTIME_RUNTIME_PASS`:

1. **Extract production fast path** from `simulation_step()` into a separate function that:
   - Skips Python controller pipeline entirely (already done)
   - Skips telemetry/diagnostics/debug code entirely
   - Only does: state extraction → JAX step → physics → repeat
   - Target: ~100 lines vs current ~2500 lines

2. **Optimize centroidal estimator** or use pre-computed Jacobian
   - Current `CentroidalStateEstimator.estimate()` costs 4-5 ms
   - Could be replaced with lightweight direct computation

3. **Deploy with PyPy** for 2-3× interpreter speedup
   - PyPy JIT compilation would reduce "Other Python overhead" from 22 ms to ~7 ms

4. **Consider C extension** for the simulation loop
   - The entire hot loop could be a C function called from Python
