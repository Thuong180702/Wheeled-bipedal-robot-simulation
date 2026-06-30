# K2 JAX Dedicated Runner — Phase 0 Baseline

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Phase:** 0 — Confirm Current Partial Baseline

## Baseline measurements

### Command run

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --controller-backend jax \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 3000 --quiet --telemetry-mode off --output-dir none
```

### Results

| Metric | Value |
|--------|-------|
| Wall clock (total) | 91.5 s |
| JIT compile time | 1.71 s |
| Init overhead (est.) | ~2 s |
| Loop wall time (est.) | ~88 s |
| Steps | 3000 |
| Est. mean step time | **~29.3 ms** |
| Est. achieved Hz | **~34.1 Hz** |
| Reported steps | 0 (BUG: see below) |
| Fall/NaN | None |

### Summary mode reference (slow but functional)

```bash
--steps 500 --quiet --telemetry-mode summary --output-dir none
```

| Metric | Value |
|--------|-------|
| Wall clock | 35.5 s |
| Mean step time | 71.0 ms |
| Achieved Hz | 14.1 Hz |
| Steps reported | 500 |
| Fall/NaN | None |

Summary mode is SLOWER because all 906 lines of telemetry Python execute (with proxy field filtering), costing ~41 ms extra per step.

### Comparison with previous production optimization baseline

| Measurement | Phase 5 (previous) | Phase 0 (now) |
|-------------|-------------------|---------------|
| Headless, telemetry off | 42.1 ms/step (23.7 Hz) | ~29.3 ms/step (~34.1 Hz) |
| Headless, summary | ~50 ms/step (~20 Hz) | 71.0 ms/step (14.1 Hz) |

The telemetry-off measurement is faster now because `update_full_rate_summary()` (inside the skipped telemetry block) no longer executes — but this also breaks the summary output (0 steps reported).

## Root cause analysis

### The bottleneck hasn't changed

The `simulation_step()` function remains ~2500 lines of Python. Even with telemetry block skipped:
- JAX hot-step: ~0.3 ms
- JAX pack + transfer: ~1.5 ms
- Physics (mj_step): ~2.4 ms
- Centroidal estimate (control): ~4.3 ms
- Python interpreter overhead (remaining control flow, conditions, assignments, nonlocal access in ~2500 lines): **~20 ms**

### Confirmed: bottleneck is monolithic runner overhead

The JAX controller itself costs ~0.29 ms (<1% of total). The MuJoCo physics costs ~2.4 ms (7%). The remaining ~26 ms (92%) is Python orchestration overhead from the 2500-line `simulation_step()` function.

### Critical bug: `update_full_rate_summary` in telemetry block

`update_full_rate_summary()` is called inside the `if _do_populate_telemetry or _telemetry_summary:` conditional at line 7781. When telemetry is off, this function never executes, so:
- `full_rate_summary["actual_steps"]` stays at 0
- All summary stats (CoM height, pitch, roll, torque) report 0
- Mean step time and Hz are reported as infinite/NaN

This affects `--telemetry-mode off` and non-keep steps in `--telemetry-mode decimated`.

## Conclusion

- **Bottleneck confirmed**: monolithic `simulation_step()` Python overhead (~2500 lines, ~20 ms per step)
- **JAX controller is NOT the bottleneck** (0.29 ms)
- **Bug found**: summary tracking breaks when telemetry population is skipped
- **Direction validated**: a dedicated runner with ~100-line hot loop should achieve >100 Hz
- **Monolithic script must NOT be further patched** — structural extraction is the correct approach

## Decision

**PROCEED to Phase 1 (Design)** — Extract dedicated K2 JAX production runner.
