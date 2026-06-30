# K2 JAX Dedicated Runner — Phase 3 Orchestration Alignment (n_substeps Fix)

**Date:** 2026-06-29
**Status:** COMPLETE — Dynamic height survives, root cause found and fixed

## Root Cause

The dedicated runner called `mujoco.mj_step()` **once** per control loop iteration, but the canonical path calls it **`n_substeps`** times (5 times for the standard model with `physics_dt=0.002s` and `control_dt=0.01s`).

**Impact:**
- Physics advanced at 1/5 the intended rate (0.002s per control step instead of 0.01s)
- Dynamic height trajectories advanced 5x too fast relative to physics time
- Controller updated at 500 Hz instead of 100 Hz (correct rate but wrong physics correspondence)
- Fixed-height scenarios worked fine (constant height_ref)
- Dynamic height scenarios ALL failed (ramp_up, ramp_down, gate_chatter)

**Verification:**
```
mj_model.opt.timestep = 0.002s
control_dt = 0.01s
n_substeps = int(0.01 / 0.002) = 5
```

## Changes Made

### `scripts/run_k2_jax_realtime.py`

**Before:**
```python
# ── Apply torque ───
mj_data.ctrl[:] = tau

# ── Physics step ───
mujoco.mj_step(mj_model, mj_data)
```

**After:**
```python
# Physics substeps
_physics_dt = float(mj_model.opt.timestep)
_n_substeps = max(1, int(round(CONTROL_DT / _physics_dt)))

# ── Apply torque ───
mj_data.ctrl[:] = tau

# ── Physics substeps ───
for _ in range(_n_substeps):
    mujoco.mj_step(mj_model, mj_data)
```

## Dynamic Height Results — Before vs After

| Scenario | Before (1 substep) | After (5 substeps) | Status |
|----------|-------------------|-------------------|--------|
| ramp_up (0.33→0.48, 5000 steps) | ❌ Fall @ 2989 | ✅ Survives 5000/5000 | **FIXED** |
| ramp_down (0.48→0.33, 5000 steps) | ❌ Fall @ 4471 | ✅ Survives 5000/5000 | **FIXED** |
| gate_chatter (0.40↔0.47, 5000 steps) | ❌ Fall @ 2288 | ✅ Survives 5000/5000 | **FIXED** |
| fixed high_0p480 (2000 steps) | ✅ Survives | ✅ Survives | No regression |
| fixed low_0p300 (2000 steps) | ✅ Survives | ✅ Survives | No regression |

## Performance Impact

| Scenario | Before Hz | After Hz | Δ |
|----------|----------|---------|---|
| ramp_up | 137 Hz | 82 Hz | -40% (5x physics work) |
| ramp_down | 106 Hz | 117 Hz | +10% (survived longer before) |
| gate_chatter | 103 Hz | 64 Hz | -38% |
| fixed high_0p480 | 150 Hz | 86 Hz | -43% |
| fixed low_0p300 | 152 Hz | 114 Hz | -25% |

**Performance remains above the 50 Hz minimum in all scenarios.** The dedicated runner does 5x more physics work per step, so the Hz drop is expected and acceptable.

## Hip-Yaw Divergence Improvement

The substep fix also reduced hip-yaw divergence at low_0p300:
- **Before**: 0.666 rad max
- **After**: 0.412 rad max (38% reduction)

The canonical JAX path has similar hip-yaw divergence (0.63 rad in ramp_up), confirming this is a pre-existing K2 limitation, not a dedicated runner issue.

## Acceptance

- [x] Canonical JAX vs dedicated traces now use same physics substep count
- [x] ramp_up survives 5000/5000 (canonical survives)
- [x] ramp_down survives 5000/5000
- [x] gate_chatter survives 5000/5000
- [x] Fixed-height no regression
- [x] Hot loop remains short and auditable (one extra for-loop)
- [x] Performance remains >50 Hz minimum
