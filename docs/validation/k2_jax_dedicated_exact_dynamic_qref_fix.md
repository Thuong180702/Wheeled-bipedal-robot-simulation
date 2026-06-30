# K2 JAX Dedicated — Exact Dynamic q_ref Fix

**Date:** 2026-06-29
**Task:** Phase 2 — Remove approximate dynamic q_ref interpolation as promotion path
**Status:** IMPLEMENTED

---

## 1. Problem

The dedicated runner was using `build_height_qref_interpolator()` to linearly interpolate posture references from height setup files. This was:

1. **An approximation** not used by the canonical K2 JAX path
2. **Making results WORSE**: ramp_down hip-yaw = 0.3728 rad with interpolation vs 0.0977 rad with static q_ref in canonical path
3. **Added as a workaround** for a CoM drift issue that had a different root cause (physics substep mismatch)

## 2. Root Cause Analysis

The canonical K2 JAX path (`simulate_hierarchical_controller.py` with JAX backend) achieves excellent dynamic height hip-yaw control using STATIC q_ref:

```python
equilibrium_joint_pos = jnp.array(mj_data.qpos[7:17])  # captured ONCE at init
# ...
q_ref=equilibrium_joint_pos_np,  # NEVER updated during dynamic height
```

And yet achieves:
- ramp_up: pitch_rms=3.15°, hy_max=0.0534 rad
- ramp_down: pitch_rms=5.84°, hy_max=0.0977 rad
- gate_chatter: pitch_rms=2.98°, hy_max=0.0629 rad

The canonical path dynamically updates `height_variant_setup["target_com_z_m"]` (line 5705 of simulate_hierarchical_controller.py) so the JAX controller sees the correct `commanded_height_ref_m`, but q_ref remains static.

The dedicated runner's interpolation was:
1. Producing incorrect posture references that excited hip-yaw modes
2. Causing ramp_down hy=0.3728 (>0.35 rad SAFETY_FAIL)

## 3. Fix

### Changes to `scripts/run_k2_jax_realtime.py`

1. **Added `--dynamic-qref-mode` CLI flag:**
   - `original-k2-exact` (DEFAULT): Static q_ref from initial equilibrium, matching canonical K2 JAX path
   - `setup-interp-debug`: Approximate interpolation (debug/ablation only, NOT for promotion)

2. **Updated `build_height_qref_interpolator()` docstring** to warn it's debug-only/approximate

3. **Updated dynamic height initialization:**
   - Default mode is `original-k2-exact` → qref_interp = None → static q_ref
   - Debug mode `setup-interp-debug` → qref_interp = build_height_qref_interpolator()

4. **Updated summary JSON** to include `dynamic_qref_mode` field

5. **Updated terminal output** to show q_ref mode

## 4. Acceptance

| Criterion | Status |
|-----------|--------|
| `build_height_qref_interpolator()` documented as debug-only approximate mode | ✅ |
| New default dynamic q_ref mode is `original-k2-exact` | ✅ |
| CLI: `--dynamic-qref-mode original-k2-exact` | ✅ |
| CLI: `--dynamic-qref-mode setup-interp-debug` | ✅ |
| Promotion runs must use `original-k2-exact` | ✅ (default) |
| `ramp_down` hip-yaw expected to improve from 0.3728 toward 0.0977 | ⏳ (pending validation) |
| `ramp_up` and `gate_chatter` metrics expected to improve | ⏳ (pending validation) |

## 5. Files Changed

1. `scripts/run_k2_jax_realtime.py`:
   - Line ~175: Added `--dynamic-qref-mode` argument
   - Line ~245: Updated `build_height_qref_interpolator()` docstring
   - Lines ~502-518: Updated dynamic height q_ref initialization
   - Line ~911: Added `dynamic_qref_mode` to summary JSON
   - Line ~987: Added q_ref mode to terminal output
