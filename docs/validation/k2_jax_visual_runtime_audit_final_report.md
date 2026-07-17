# K2 JAX Visual Runtime Audit — Final Report

**Date:** 2026-06-29
**Classification:** K2_JAX_VISUAL_RUNTIME_BUG_FIXED

## Summary

The JAX backend was correctly selected, but the Python WBC+controller pipeline ran every step regardless, adding ~43 ms/step of wasted computation. The JAX controller itself runs at 0.3 ms/step (3500+ Hz capability).

Two root causes identified and fixed:
1. `pack_input_k2` JAX dispatch overhead (17.4 ms → 5.0 ms)
2. Python WBC QP solve running every step (~30 ms → skipped)

## Performance Comparison

| Metric | Before Fix | After Fix | Improvement |
|--------|-----------|-----------|-------------|
| jax_pack_input | 17.41 ms | 5.04 ms | 3.5x faster |
| jax_jit_step | 0.28 ms | 0.30 ms | unchanged |
| jax_total (JAX block) | 17.69 ms | 5.34 ms | 3.3x faster |
| total_step | 105.41 ms | 62.53 ms | 1.7x faster |
| headless sim rate | ~9.5 Hz | ~16 Hz | 1.7x faster |
| JIT compile (one-time) | 1.45s | 1.44s | unchanged |
| recompilation count | 0 | 0 | ✓ no recompile |

## Root Causes (Phase 0-5)

| Phase | Finding | Status |
|-------|---------|--------|
| 0 | Backend resolves to JAX correctly | ✓ PASS |
| 1 | Headless also slow (not visual-specific) | ✓ Confirmed |
| 2 | Visual pacing not the bottleneck | ✓ Confirmed |
| 3 | Python WBC runs every step in JAX mode | ✗ BUG FIXED |
| 4 | JAX compiles once, no recompile | ✓ PASS |
| 5 | Viewer overhead not the bottleneck | ✓ Confirmed |

## Changes Made (Phase 6)

### 1. `wheeled_biped/controllers/k2_jax_controller.py` — `pack_input_k2`

Replaced JAX `.at[idx].set()` pattern with NumPy direct indexing:
- Before: `jnp.zeros(42) + 20x .at[idx].set(val)` → 17.4 ms of JAX dispatch overhead
- After: `np.zeros(42) + direct assignment + jnp.asarray()` → ~5 ms

### 2. `scripts/simulate_hierarchical_controller.py` — JAX Fast Path

Added `_jax_fast_path` flag (`_jax_enabled and not _both_synced_enabled`):
- Skips WBC QP solve (~30 ms/step)
- Skips static_balance_wrapper  
- Skips posture regularizer
- Skips composer (~2 ms/step)
- Keeps sagittal controller (needed for JAX input packing)
- Provides dummy `qp_diagnostics` and `balance_core_result` for telemetry compatibility

## Remaining Bottleneck

The Python sagittal controller compute() still runs (~30+ ms/step) because JAX input packing depends on its outputs (`pitch_x_error`, `sagittal_diag`). The remaining balance_core_block is ~44 ms, dominated by the sagittal controller.

This could be eliminated by modifying the JAX controller to accept raw sensor values and compute pitch_reference internally, allowing the entire Python controller pipeline to be skipped.

## Corrected User Commands

### A. Fastest headless JAX check
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
Expected: ~62 ms/step, ~16 Hz, JIT compile 1.4s (one-time)

### B. Normal realtime visual JAX
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
  --visual
```
Expected: ~62 ms/step physics + ~15-30 ms viewer overhead → ~10-15 Hz visual

### C. Visual JAX with low viewer sync rate (less overhead)
```bash
# same as B but add:
  --visual-sync-hz 15
```
Expected: reduced viewer rendering overhead

### D. Visual JAX with explicit backend (redundant — K2 defaults to JAX)
```bash
# same as B — the --controller-backend jax is now the default for K2
```

### E. Python fallback visual (SLOW — for comparison only)
```bash
# same as B but:
  --controller-backend python
```
Expected: ~130 ms/step, ~8 Hz, WBC QP solve overhead

### F. Both-synced debug visual (VERY SLOW — parity diagnostics only)
```bash
# same as B but:
  --controller-backend both-synced
```
Expected: ~150+ ms/step, ~6 Hz, extensive per-step diagnostic printing

## Expected Performance Summary

| Mode | Backend | Step time | Sim Hz | Notes |
|------|---------|-----------|--------|-------|
| Headless | python | ~130 ms | ~8 Hz | WBC QP solve |
| Headless | jax (before fix) | ~105 ms | ~10 Hz | Python WBC still ran |
| Headless | jax (after fix) | ~62 ms | ~16 Hz | WBC skipped |
| Visual | jax (after fix) | ~75-95 ms | ~10-13 Hz | + viewer overhead |
| both-synced | both-synced | ~150+ ms | ~6 Hz | parity diagnostics |
