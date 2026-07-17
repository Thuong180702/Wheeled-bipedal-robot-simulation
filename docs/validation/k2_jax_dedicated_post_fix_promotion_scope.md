# K2 JAX Dedicated Realtime Runner — Post-Fix Promotion Scope

**Date:** 2026-06-29
**Commit:** post-fix (baseline was `0e1c713`)
**Classification:** K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL

## What Was Fixed

### Root Cause 1: Physics substep mismatch (Critical)
- **Bug:** Dedicated runner called `mujoco.mj_step()` once per control cycle; canonical path calls it 5 times (physics_dt=0.002s, control_dt=0.01s)
- **Impact:** Physics advanced at 1/5 rate; dynamic height trajectories changed 5x too fast relative to physics → ALL dynamic height scenarios failed
- **Fix:** Added `_n_substeps` loop matching canonical path
- **Result:** ramp_up, ramp_down, gate_chatter all survive 5000/5000

### Root Cause 2: Parameter source-of-truth mismatch (Maintainability)
- **Bug:** Dedicated runner hardcoded K2 profile values in `K2_PROFILE` dict instead of importing from canonical `K2_NOTCH_LOW_Q_V1`
- **Impact:** Risk of parameter drift; values happened to match after recent profile fix
- **Fix:** Import `K2_NOTCH_LOW_Q_V1` directly, use `is_active_for_variant()` for variant-gating
- **Result:** 0 parameter mismatches, all params read from single source of truth

## What Is Promoted

### ✅ Production Use
Use `scripts/run_k2_jax_realtime.py` for:
- Fixed-height K2 JAX balance at all tested heights (0.30m – 0.48m)
- Dynamic height transitions (ramp up, ramp down, gate chatter)
- Push recovery (via xfrc_applied, same mechanism as canonical path)
- Headless realtime (>50 Hz minimum, typically >80 Hz)
- Visual realtime with pacing control
- Telemetry output (off, decimated, full modes)

### ⚠️ Known Limitations (Pre-existing K2)
- **Hip-yaw divergence** at low heights (0.30m–0.36m) may exceed 0.35 rad
  - This is a pre-existing K2 controller limitation, not a dedicated runner issue
  - Canonical JAX path shows the same divergence levels
  - Does not cause falls in fixed-height scenarios
  - Active research area for future controller improvements

### ❌ Not Supported
- **WBC (Whole Body Control):** Not implemented in dedicated runner
- **Non-K2 profiles:** Only `k2_notch_low_q_v1` validated
- **Python fallback:** Use `scripts/simulate_hierarchical_controller.py --controller-backend python` for Python K2
- **Both-synced mode:** Use monolithic script for Python+JAX comparison
- **Stand-up recovery:** Not implemented in K2 controller
- **Locomotion/walking:** Not implemented
- **Stair climbing:** Not implemented
- **Rough terrain:** Not implemented

## Usage Commands

### Fixed-height benchmark (fastest)
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 3000 --quiet --telemetry off
```

### Dynamic height transition
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --dynamic-height-trajectory outputs/k2_dynamic_height_gate_crossing/trajectories/ramp_up_0p330_to_0p480.json \
  --steps 5000 --telemetry full --output-dir outputs/my_run
```

### Push recovery (comparable to original K2)
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --push-seq outputs/k2_release_hardening/push_seq_bwd_90N.json \
  --steps 2000 --telemetry full --output-dir outputs/push_test
```

### Visual realtime
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 10000 --visual --telemetry summary
```

### Param dump (verify parity)
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --dump-k2-params outputs/param_check.json --steps 10 --quiet --telemetry off
```

## Performance Targets

| Mode | Target | Status |
|------|--------|--------|
| Headless, telemetry off | >100 Hz if possible, >50 Hz min | 80-150 Hz depending on scenario |
| Headless, telemetry full | Writes 1 row/step, flush at end | ✅ |
| Visual realtime | Usable at realtime factor 1.0 | ✅ |
| No per-step print | Guaranteed | ✅ |
| No per-step CSV write | Guaranteed | ✅ |

## Push Mechanism

The dedicated runner uses `xfrc_applied` on torso body 1, which is the SAME mechanism as the canonical path when using `--push-sequence-file` (push_seq JSON files). Push behavior is directly comparable between dedicated and canonical paths.

The older `--push-enabled` / `--push-magnitude-n` / `--push-duration-steps` qvel-injection path in the monolithic script is a different mechanism and not implemented in the dedicated runner. Use push_seq JSON files for comparable push testing.

## Fallback and Comparison

For Python K2 fallback, both-synced comparison, or WBC debugging:
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --controller-backend python \
  ...
```

## Test Suite

```bash
# Param parity
pytest tests/test_k2_jax_dedicated_param_parity.py -v

# Runner guards  
pytest tests/test_k2_jax_dedicated_runner_guards.py -v

# Existing K2 JAX tests (must still pass)
pytest tests/test_k2_jax_component_parity.py tests/test_k2_jax_step_parity.py -v

# Stage1 behavior unchanged
pytest tests/test_stage1_behavior_unchanged.py -v
```
