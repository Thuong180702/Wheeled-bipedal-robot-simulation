# K2 JAX Dedicated Runner — Phase 1 Design

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Phase:** 1 — Design

## Decision: New script `scripts/run_k2_jax_realtime.py`

A separate script is preferred over a new function in `simulate_hierarchical_controller.py` because:
1. The monolithic script is ~9300 lines; adding a function there risks accidental dependency on debug paths
2. A separate script is auditable in its entirety (~400-600 lines)
3. No risk of importing or triggering debug/validation branches
4. The old script remains unchanged for validation/debug/both-synced
5. Clean separation of concerns: production vs validation

## Architecture

```
run_k2_jax_realtime.py (~500 lines)
├── CLI parsing (argparse)                    ~80 lines
├── MuJoCo model/data initialization          ~60 lines
├── Height variant & config loading           ~30 lines
├── Equilibrium calibration                   ~50 lines
├── K2 JAX controller initialization          ~50 lines
├── Input buffer preallocation                ~20 lines
├── Hot loop: run_k2_jax_realtime_loop()      ~80 lines
├── Telemetry buffer & write-once CSV         ~60 lines
└── Summary print                             ~40 lines
```

## Hot loop design (the ~80-line production path)

```python
def run_k2_jax_realtime_loop(
    mj_model, mj_data, max_steps, control_dt,
    jax_step_fn, jax_params, jax_state,
    equilibrium_joint_pos, initial_yaw_z,
    height_variant_setup, contact_supervisor,
    push_sequence, telemetry_cfg, input_buf, ...
):
    for step in range(max_steps):
        # 1. Apply push disturbance (if configured)
        if push_sequence and step in push_sequence:
            apply_push(mj_data, push_sequence[step])

        # 2. Extract raw state from MuJoCo (~0.3 ms)
        joint_pos = mj_data.qpos[7:17]
        joint_vel = mj_data.qvel[6:16]
        l_wx = mj_data.xpos[l_wheel_body_id]
        r_wx = mj_data.xpos[r_wheel_body_id]

        # 3. Compute support center (~0.01 ms)
        support_x = 0.5 * (l_wx[0] + r_wx[0])
        support_y = 0.5 * (l_wx[1] + r_wx[1])

        # 4. Centroidal estimate (~4 ms — main cost)
        #    Provides: pitch, roll, yaw, com_pos, com_vel, contact
        centroidal = centroidal_estimator.estimate(mj_data)

        # 5. Contact validity (~0.01 ms)
        contact_valid = float(
            centroidal.left_wheel_contact
            and centroidal.right_wheel_contact
            and centroidal.contact_force_valid
        )

        # 6. Fill preallocated input buffer (~1 ms)
        pack_standalone_input_fast(
            input_buf, centroidal, joint_pos, joint_vel,
            equilibrium_joint_pos, initial_yaw_z,
            support_x, support_y, contact_valid,
            height_variant_setup
        )

        # 7. JAX controller step (~0.3 ms)
        jax_tau, jax_state, jax_diag = jax_step_fn(
            jax_state, input_buf, jax_params
        )

        # 8. Apply torque (~0.01 ms)
        mj_data.ctrl[:] = np.array(jax_tau)

        # 9. Physics step (~2.4 ms)
        mujoco.mj_step(mj_model, mj_data)

        # 10. Update summary stats in-place (~0.1 ms)
        update_summary_stats(step, centroidal, jax_tau)

        # 11. Buffer telemetry if decimated (~0.01 ms avg)
        if telemetry_keep_step:
            buffer_minimal_telemetry(step, centroidal, jax_tau)
```

**Estimated per-step cost:**
| Component | Est. ms | % |
|-----------|---------|---|
| State extraction | 0.3 | 4% |
| Centroidal estimate | 4.0 | 50% |
| Input packing (fast) | 1.0 | 12% |
| JAX step | 0.3 | 4% |
| Physics (mj_step) | 2.4 | 30% |
| Summary + telemetry | 0.1 | 1% |
| **Total** | **~8.1 ms** | **100%** |

Expected: **~123 Hz** with centroidal estimator, reaching the >100 Hz target.

If centroidal is optimized to ~1 ms: **~5.1 ms = ~196 Hz**.

## What is NOT in the hot loop

- ❌ Python sagittal controller compute
- ❌ WBC QP solver
- ❌ Torque composer (done inside JAX)
- ❌ Both-synced comparison
- ❌ 756-column telemetry dict construction
- ❌ `update_full_rate_summary()` (replaced by lightweight inline stats)
- ❌ Per-step print
- ❌ CSV file writes
- ❌ Debug diagnostics (B0-AUDIT, LIFECYCLE, STAGE 2, etc.)
- ❌ `balance_core_controllers` dict lookups
- ❌ `not _quiet` condition checks on every line
- ❌ Nonlocal variable declarations (250+)
- ❌ Centroidal log duplicate estimate

## CLI interface

```bash
# Fixed-high headless (fastest)
python scripts/run_k2_jax_realtime.py \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 3000 --telemetry-mode off --output-dir none

# Push backward
python scripts/run_k2_jax_realtime.py \
  --height-variant-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --push-enabled --push-sequence-file .../push_bwd_90N.json \
  --steps 3000 --telemetry-mode off

# Visual push
python scripts/run_k2_jax_realtime.py \
  --height-variant-setup .../low_0p330_setup.json \
  --push-enabled --push-sequence-file .../push_bwd_90N.json \
  --steps 1000 --telemetry-mode summary --visual

# Decimated CSV output
python scripts/run_k2_jax_realtime.py \
  --height-variant-setup .../low_0p330_setup.json \
  --push-enabled --push-sequence-file .../push_bwd_90N.json \
  --steps 3000 --telemetry-mode decimated --telemetry-decimation 10 \
  --output-dir outputs/realtime_runs/push_bwd_jax
```

## Telemetry design (Phase 2)

Modes: `off`, `summary`, `decimated`, `full`

- **off**: No per-step recording. Only final summary metrics.
- **summary**: Final metrics only (Hz, max pitch/roll, max torque, final height, fall status).
- **decimated**: Buffer rows every N steps in memory, write CSV once at end.
- **full**: Buffer every step (for debug), still write once at end.

Minimal CSV columns (decimated mode):
```python
MINIMAL_CSV_COLUMNS = [
    "step", "sim_time", "com_z", "pitch_deg", "roll_deg",
    "left_wheel_tau", "right_wheel_tau", "max_abs_tau",
    "height_ref", "contact_valid", "fall"
]
```

## Input packing optimization (Phase 3)

Preallocate a single `np.zeros(45, dtype=np.float64)` buffer and fill by direct index assignment:
```python
input_buf[0] = pitch_x_rad
input_buf[1] = pitch_rate_x_rad_s
# ... etc
```

This avoids:
- `np.array()` slice creation
- `jnp.array()` dispatch
- Dict/object allocation in the hot loop
- Python function call overhead for packing

Then: `jax_input = jnp.asarray(input_buf)` for the final JAX transfer.

## Regression guard

- All controller gains/thresholds/semantics unchanged — the SAME `pack_params_stage2()` and `k2_jax_controller_step()` functions are called
- Physics parameters unchanged — same model, same `mj_step`
- Python fallback still available in the OLD script
- Both-synced still available in the OLD script
- Step count unchanged — same 0.01s control_dt

## Acceptance criteria

- [x] Hot loop is short and auditable (~80 lines)
- [x] No debug/validation branches in hot path
- [x] Existing script remains available for validation/debug
- [x] No Python controller/WBC/composer calls
- [x] No per-step print
- [x] No synchronous per-step CSV writes
- [x] Controller semantics unchanged (same JAX functions)
- [ ] Runtime target: >50 Hz minimum, >100 Hz preferred
