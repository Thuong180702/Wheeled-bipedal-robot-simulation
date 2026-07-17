# K2 JAX Standalone — Phase 3+4 Implementation Report

**Date:** 2026-06-29

## Changes Made

### 1. `wheeled_biped/controllers/k2_jax_controller.py`

#### New input contract (standalone, 45 fields)
- Added `_I_COM_VX = 42`, `_I_SUPPORT_CENTER_X = 43`, `_I_SUPPORT_CENTER_Y = 44`
- Added `K2_JAX_INPUT_SIZE_STANDALONE = 45`
- Created `pack_input_k2_standalone()` — accepts ONLY raw state, no Python controller outputs

#### New params (standalone, 54 total)
- Added `_IDX_STANDALONE_MODE = 48`
- Added `_IDX_PITCH_X_EQ_RAD = 49`, `_IDX_SUPPORT_CENTER_EQ_X/Y = 50-51`
- Added `_IDX_SAGITTAL_AXIS_X/Y = 52-53`
- Updated `pack_params_stage2()` to accept `standalone_mode` + equilibrium constants

#### JAX-native preprocessing in `k2_jax_controller_step()`
When `_standalone_mode > 0.5`:
1. **sag_pos_err**: `project_sagittal_displacement(support_center, eq_center, sag_axis)` — matches Python's formula
2. **sag_vel**: `project_sagittal_velocity(com_vx, com_vy, sag_axis)` — matches Python's formula
3. **support_vel**: `(sag_pos_err - prev_support_error) / control_dt` — numerical derivative matching Python's svdbc internal
4. **effective_pitch_x**: `raw_pitch_x - pitch_x_eq - deg2rad(total_pitch_ref_offset_deg)` — matches Python's pitch_x_error with full outer loop + physics FF offset

All downstream references to `pitch_x` in ABS, APCR1ND, and sagittal torque now use `effective_pitch_x`.

### 2. `scripts/simulate_hierarchical_controller.py`

#### Standalone fast-path in sim loop
Added `if _jax_fast_path:` block at line ~6102 that:
1. Extracts support center from `mj_data.xpos[l/r_wheel_body_id]` (sensor read, not controller)
2. Packs JAX input via `pack_input_k2_standalone()` with raw state only
3. Runs JAX step → sets `tau_smooth` directly
4. Updates `prev_support_error` from JAX state

#### Python controller skip guards
- `shape_posture.compute()` — guarded with `if not _jax_fast_path:`
- Sagittal controller dispatch — sets `sagittal_ctrl_name = "standalone-skipped"` to bypass velocity-damped + baseline branches
- `lateral_roll_balance.compute()` — guarded with `if not _jax_fast_path:`
- Old JAX override section — guarded with `if _jax_enabled and not _jax_fast_path:` (prevents double execution)

## Verification

### Smoke test (50 steps, high_0p480)
- ✅ `[JAX BACKEND] Enabled [STANDALONE]` confirmed
- ✅ Params size: 54 (includes equilibrium constants)
- ✅ Input size: 45 (includes support_center, com_vx)
- ✅ Robot stays upright
- ✅ Torques reasonable (physics FF at wheels, posture at legs)

### Performance (1000 steps, high_0p480)
| Metric | Pre-Standalone | Post-Standalone | Change |
|--------|---------------|-----------------|--------|
| Total step mean | 61.82 ms | 55.00 ms | -11% |
| JAX hot-step | 0.247 ms | 0.288 ms | — |
| JAX pack input | 5.05 ms | 5.65 ms | — |
| Physics step | 0.23 ms | 0.26 ms | — |
| Python overhead | ~56 ms | ~49 ms | -12% |

**Performance note:** The 5-6 ms `jax_pack_input` time is unexpectedly high (should be ~0.01 ms for a 45-element numpy→JAX conversion). Investigation deferred to Phase 7. The ~49 ms remaining Python overhead is dominated by per-step terminal I/O (`print()` statements) and telemetry CSV writing, NOT controller compute.

## Acceptance

| Criterion | Status |
|-----------|--------|
| backend=jax no longer calls Python svdbc.compute() | ✅ Guarded via sagittal_ctrl_name override |
| backend=jax no longer calls shape_posture.compute() | ✅ Guarded via if not _jax_fast_path |
| backend=jax no longer calls lateral_roll_balance.compute() | ✅ Guarded via if not _jax_fast_path |
| JAX computes sag_pos_err from raw state | ✅ project_sagittal_displacement in JAX |
| JAX computes sag_vel from raw state | ✅ project_sagittal_velocity in JAX |
| JAX computes support_vel from state derivative | ✅ (sag_pos_err - prev_support_error) / dt |
| JAX computes effective_pitch_x from raw + offset | ✅ raw_pitch_x - pitch_eq - outer_loop_offset |
| No Python control output in production JAX input | ✅ pack_input_k2_standalone uses raw state only |
| both-synced path preserved | ✅ _jax_fast_path = False for both-synced |
| Python fallback preserved | ✅ Non-jax backends unchanged |
| No NaN/fall | ✅ 1000-step run completed cleanly |
| JIT compile count = 0 (no recompilation) | ✅ Single compile at startup |
