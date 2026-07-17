# K2 JAX Dedicated Runner — Phase 4 Implementation

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j

## New file

[`scripts/run_k2_jax_realtime.py`](scripts/run_k2_jax_realtime.py) — ~550 lines, self-contained production runner.

## Architecture

```
run_k2_jax_realtime.py
├── parse_args()              — CLI (argparse)
├── load_json()               — JSON loader
├── load_push_sequence()      — Push sequence parser
├── load_dynamic_height_trajectory() — Height trajectory
├── compute_velocity_damping_scale() — K2 profile lookup
├── check_termination()       — Height floor + orientation check
└── main()                    — Orchestration + hot loop
    ├── Model/data init       (~15 lines)
    ├── Height setup          (~20 lines)
    ├── Posture application   (~15 lines)
    ├── Calibration           (~25 lines)
    ├── Centroidal estimator  (~10 lines)
    ├── JAX controller init   (~30 lines)
    ├── Pre-loop setup        (~30 lines)
    ├── HOT LOOP              (~80 lines)
    ├── Summary output        (~60 lines)
    └── CSV/JSON write        (~30 lines)
```

## Hot loop (the ~80-line production path)

The hot loop contains ONLY:
1. Push force application (if configured)
2. Dynamic height update (if configured)
3. Joint state extraction from MuJoCo
4. Centroidal state estimation (CoM, orientation, contacts)
5. Support center computation
6. Contact validity check
7. `pack_input_k2_standalone()` — canonical JAX input packing
8. `jax_step_fn()` — JAX controller step
9. Torque application to MuJoCo
10. `mujoco.mj_step()` — physics step
11. Termination check
12. Summary stats update (inline, no dict overhead)
13. Optional telemetry buffer (append to list, no file I/O)

## What the hot loop does NOT contain

| Excluded | Where it lives |
|----------|---------------|
| Python sagittal controller compute | Removed — JAX does all control internally |
| WBC QP solver | Removed |
| Torque composer | Removed — JAX `k2_jax_torque_composer_step` |
| Both-synced comparison | Removed — old script only |
| 756-column telemetry dict | Removed — 11 columns max |
| `update_full_rate_summary()` | Replaced by inline scalar tracking |
| Per-step `print()` | None in `--quiet` mode |
| CSV file writes | Write-once after loop |
| `balance_core_controllers` dict lookups | Removed |
| `nonlocal` variable declarations (250+) | None needed |
| Duplicate centroidal estimate | Removed — single estimate per step |
| B0-AUDIT / LIFECYCLE / STAGE 2 diagnostics | Removed |
| Controller profile resolution at runtime | Hardcoded K2 profile constants |

## Verified commands

### A. Fixed-high headless
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 3000 --quiet --telemetry off
# Result: 187.5 Hz, [OK]
```

### B. Push backward
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --push-seq outputs/k2_jax_hip_yaw_phase6/trajectories/push_bwd_90N.json \
  --steps 3000 --quiet --telemetry off
# Result: 177.1 Hz, [OK] — survived
```

### C. Decimated CSV
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup .../low_0p330_setup.json \
  --push-seq .../push_bwd_90N.json \
  --steps 3000 --quiet --telemetry decimated --telemetry-decimation 10 \
  --output-dir outputs/realtime_test/push_bwd_decimated
# Result: 121.1 Hz, CSV with 301 lines
```

### D. Dynamic height (ramp_up)
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup .../low_0p330_setup.json \
  --dynamic-height-trajectory .../ramp_up.json \
  --steps 5000 --quiet --telemetry off
# Result: 153.7 Hz, [FALL] at step 2989 (known K2 limitation)
```

## Compatibility

- **Python fallback**: Available in `scripts/simulate_hierarchical_controller.py` (unchanged)
- **Both-synced debug**: Available in `scripts/simulate_hierarchical_controller.py` (unchanged)
- **Controller gains**: Unchanged — same `pack_params_stage2()` and `k2_jax_controller_step()`
- **Physics**: Unchanged — same model, same `mj_step`
- **JAX controller semantics**: Unchanged — same functions, same params

## Acceptance

- [x] Command A/B run without using monolithic `simulation_step()`
- [x] No Python controller call — 0 calls to sagittal/WBC/composer
- [x] No per-step print in quiet mode
- [x] No synchronous per-step CSV writes
- [x] Summary prints only at start/end
- [x] Old script unchanged and available for validation
