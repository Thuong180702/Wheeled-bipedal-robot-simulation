# K2 JAX Dedicated Realtime Runner — Final Report

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Classification:** `K2_JAX_DEDICATED_REALTIME_RUNNER_PASS`

## Executive summary

A dedicated K2 JAX production realtime runner was extracted from the 9300-line monolithic simulation script. The new runner achieves **187.5 Hz headless** — a **7.9× speedup** from 23.7 Hz, exceeding the 100 Hz target by 87%.

## Why previous optimization stalled at 23.7 Hz

The monolithic `simulation_step()` function is ~2500 lines of Python. Previous patches (quiet mode, telemetry proxy, input packing) reduced overhead from 54.7 ms → 42.1 ms, but couldn't overcome the fundamental problem: the Python interpreter still processes thousands of lines of control flow, variable access, and condition checks every step. The JAX controller itself costs only 0.3 ms — 99% of wall time was Python orchestration.

## Dedicated runner architecture

**File:** [`scripts/run_k2_jax_realtime.py`](scripts/run_k2_jax_realtime.py) — ~550 lines, self-contained.

```
Hot loop: ~80 lines
  ├── Push force application (if configured)
  ├── Dynamic height update (if configured)
  ├── MuJoCo state extraction
  ├── Centroidal state estimation
  ├── Support center computation
  ├── Contact validity check
  ├── pack_input_k2_standalone() → JAX input
  ├── jax_step_fn() → JAX controller step
  ├── Torque application to MuJoCo
  ├── mujoco.mj_step() → physics
  ├── Termination check
  ├── Summary stats update (inline scalars)
  └── Telemetry buffer (write-once, no per-step I/O)
```

## Performance

| Scenario | Before (monolithic) | After (dedicated) | Speedup |
|----------|--------------------|--------------------|---------|
| Fixed-high, telemetry off | 23.7 Hz (42.1 ms) | **187.5 Hz (5.33 ms)** | **7.9×** |
| Push-bwd, telemetry off | ~21 Hz (~48 ms) | **177.1 Hz (5.65 ms)** | **8.4×** |
| Decimated telemetry (avg) | ~21 Hz (~48 ms) | **177.7 Hz (5.63 ms)** | **8.5×** |
| Dynamic height ramp_up | ~20 Hz (est.) | **153.7 Hz (6.50 ms)** | **7.7×** |

Performance breakdown (5.33 ms/step):
- Centroidal estimation: ~2.3 ms (43%)
- MuJoCo physics (mj_step): ~1.5 ms (28%)
- JAX input packing: ~0.8 ms (15%)
- JAX controller step: ~0.3 ms (6%)
- Other (push, termination, stats): ~0.4 ms (8%)

## Telemetry

- **off**: 0 cost, summary only
- **summary**: 0 cost, final metrics
- **decimated**: ~0.3 ms avg, CSV write-once (11 columns)
- **full**: every step buffered, write-once

No 756-column dict construction. No per-step file I/O. No per-step print.

## Files changed

| File | Change |
|------|--------|
| [`scripts/run_k2_jax_realtime.py`](scripts/run_k2_jax_realtime.py) | **NEW** — dedicated production runner |
| [`scripts/simulate_hierarchical_controller.py`](scripts/simulate_hierarchical_controller.py) | Bug fix: `_do_populate_telemetry` UnboundLocalError |

### Deliverables

| Document | Phase |
|----------|-------|
| [`k2_jax_dedicated_runner_baseline.md`](k2_jax_dedicated_runner_baseline.md) | 0 |
| [`k2_jax_dedicated_runner_design.md`](k2_jax_dedicated_runner_design.md) | 1 |
| [`k2_jax_dedicated_runner_telemetry.md`](k2_jax_dedicated_runner_telemetry.md) | 2 |
| [`k2_jax_dedicated_runner_input_packing.md`](k2_jax_dedicated_runner_input_packing.md) | 3 |
| [`k2_jax_dedicated_runner_implementation.md`](k2_jax_dedicated_runner_implementation.md) | 4 |
| [`k2_jax_dedicated_runner_benchmark.md`](k2_jax_dedicated_runner_benchmark.md) | 5 |
| [`k2_jax_dedicated_runner_regression_guard.md`](k2_jax_dedicated_runner_regression_guard.md) | 6 |
| This document | 7 |

## Test results

- `tests/test_stage1_behavior_unchanged.py` — **11/11 PASSED**
- `tests/test_k2_jax_*.py` — all pass individually (full suite: timeout cascading, pre-existing)
- Monolithic script smoke tests — **PASS** (after bug fix)
- Old Python fallback — functional
- Both-synced debug — functional

## Hard rules compliance

- [x] No controller gains changed
- [x] No APCR1ND/ABS/MODE_DIV semantics changed
- [x] No physics parameters changed
- [x] No simulation fidelity reduced
- [x] No CSV written synchronously per step
- [x] No per-step print
- [x] Old debug/validation script preserved
- [x] Python fallback preserved
- [x] Both-synced mode preserved
- [x] No claim of PASS below 50 Hz — achieved 187.5 Hz

## Classification justification

**`K2_JAX_DEDICATED_REALTIME_RUNNER_PASS`** because:

✅ Dedicated runner achieves **187.5 Hz** headless (target: >100 Hz, minimum: >50 Hz)  
✅ Hot loop is ~80 lines — short, auditable, no debug/validation branches  
✅ No Python controller/WBC/composer calls — 0 per step  
✅ Telemetry is buffered, write-once — no per-step I/O  
✅ No per-step print in quiet mode  
✅ Functional scenarios pass (fixed_high, push_bwd)  
✅ Stage 1 behavior unchanged — 11/11 tests  
✅ Old Python fallback and both-synced preserved  
✅ No controller semantics, gains, thresholds changed  
✅ No physics parameters changed  
✅ Bug fixed: `_do_populate_telemetry` UnboundLocalError  
✅ All deliverables documented  

## User commands

### Production realtime (fastest)
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 3000 --quiet --telemetry off
```
→ ~187 Hz headless, summary only

### Push recovery
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --push-seq outputs/k2_jax_hip_yaw_phase6/trajectories/push_bwd_90N.json \
  --steps 3000 --quiet --telemetry off
```
→ ~177 Hz, survives 90N push

### Visual push (with viewer)
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --push-seq outputs/k2_jax_hip_yaw_phase6/trajectories/push_bwd_90N.json \
  --steps 1000 --visual --telemetry summary
```

### Decimated CSV output
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup .../low_0p330_setup.json \
  --push-seq .../push_bwd_90N.json \
  --steps 3000 --quiet --telemetry decimated --telemetry-decimation 10 \
  --output-dir outputs/realtime_runs/push_bwd_jax
```
→ CSV with 300 rows, summary JSON

## Rollback / debug path

For validation, Python fallback, both-synced parity, or full telemetry debugging, use the original script:

```bash
# Python fallback (reference, slow)
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --height-variant-setup .../high_0p480_setup.json \
  --steps 500 --controller-backend python

# Both-synced debug (parity check, very slow)
python scripts/simulate_hierarchical_controller.py ... --steps 500 --controller-backend both-synced

# Full telemetry debug
python scripts/simulate_hierarchical_controller.py ... --steps 500 --verbose \
  --telemetry-mode full --output-dir outputs/debug_runs/full_telemetry
```

## Remaining optimization opportunities

The main remaining bottleneck is the centroidal estimator at ~2.3 ms (43% of step time). Potential optimizations:
1. **Lightweight CoM estimation**: Direct Jacobian-based CoM velocity instead of finite-difference
2. **Pre-computed Jacobians**: Compute CoM Jacobian offline, evaluate in loop
3. **Contact detection simplification**: Use force sensor thresholds instead of full contact iteration

These could push the dedicated runner toward 250+ Hz.
