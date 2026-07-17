# K2 JAX — Hip-Yaw Metric Regression Fix

**Date:** 2026-06-29
**Task:** Phase 4 — Fix remaining hip-yaw metric regressions
**Status:** ROOT CAUSE IDENTIFIED; FIX IMPLEMENTED (Phase 2); PENDING RE-VALIDATION

---

## 1. Regression Summary

After mode_div fix but before q_ref fix (interpolation active):

| Scenario | Original K2 hy | Candidate hy | Delta | Class |
|----------|---------------|-------------|-------|-------|
| ramp_down | 0.0977 | **0.3728** | +0.2751 | SAFETY_FAIL |
| ramp_up | 0.0534 | 0.1242 | +0.0708 | SAFE_BUT_WORSE |
| gate_chatter | 0.0629 | 0.2160 | +0.1531 | SAFE_BUT_WORSE |
| low_0p300 | 0.1314 | 0.2008 | +0.0694 | SAFE_BUT_WORSE |
| low_0p340 | 0.0445 | 0.1255 | +0.0810 | SAFE_BUT_WORSE |
| high_0p430 | 0.0236 | 0.0833 | +0.0597 | SAFE_BUT_WORSE |

## 2. Root Cause: Approximate q_ref Interpolation

The dedicated runner was using `build_height_qref_interpolator()` which linearly interpolated posture references from height setup files. This was:

1. **Not used by the canonical K2 JAX path** — the canonical path uses STATIC q_ref
2. **Producing incorrect posture references** — the interpolation produced hip_pitch/knee values that didn't match the physics of the height transitions
3. **Exciting hip-yaw modes** — especially during ramp_down through the notch gate region (0.42-0.48m)

### Why ramp_down was worst

During ramp_down (0.48→0.33m):
- The interpolation changed q_ref from 0.48m posture (hip_pitch=0.6261) to 0.33m posture (hip_pitch=1.0761)
- The posture controller tracked these changing references, introducing discontinuities
- As the robot descended through the notch gate region (0.48→0.42m), the WIP filter behavior changed, creating a nonlinear transition
- The interpolation's piecewise-linear nature amplified oscillations at setup file boundaries
- These oscillations excited the hip-yaw mode, which mode_div partially but incompletely suppressed

### Why static q_ref works (canonical path)

The canonical path proves that static q_ref works:
- ramp_down with static q_ref: hy=0.0977 (EXCELLENT)
- The posture controller maintains a consistent reference
- Other mechanisms (support FF, sagittal balance, adaptive bias trim) handle the height transition
- Mode-div has a stable equilibrium to work against

## 3. Fix Implemented (Phase 2)

The `--dynamic-qref-mode original-k2-exact` flag (now DEFAULT) uses static q_ref matching the canonical path:

```python
# In original-k2-exact mode:
qref_interp = None  # No interpolation
# eq_joint stays at initial equilibrium_joint_pos throughout the simulation
```

Expected improvements after fix:
- ramp_down hy should drop from 0.3728 → toward 0.0977 (matching canonical)
- ramp_up hy should drop from 0.1242 → toward 0.0534
- gate_chatter hy should drop from 0.2160 → toward 0.0629

## 4. Remaining Potential Regressions

Even with static q_ref, the dedicated runner may differ from canonical in these areas:

### 4.1 Mode-div Reference

| Aspect | Canonical | Dedicated | Match? |
|--------|-----------|-----------|--------|
| ref_source | "target" | "target" | ✅ |
| div_error formula | (q[1]-q[6]) - (qref[1]-qref[6]) | Same | ✅ |
| kp, kd, max_torque | 10.0, 0.50, 7.5 | Same (from K2_NOTCH_LOW_Q_V1) | ✅ |
| Height gate | 0.30-1.10 smoothstep | Same | ✅ |
| Torque sign convention | +left, -right | Same | ✅ |

### 4.2 Physics Application

| Aspect | Canonical | Dedicated | Match? |
|--------|-----------|-----------|--------|
| Control dt | 0.01s | 0.01s | ✅ |
| Physics substeps | control_dt / physics_dt | Same | ✅ |
| Torque application | mj_data.ctrl[:] | Same | ✅ |
| No PID wrapping in JAX fast path | ✅ | ✅ | ✅ |

### 4.3 JAX vs Python Formula Parity

The JAX mode_div is a direct port of the Python implementation. Both use:
- Same equation: `torque = clip(-(kp*err + kd*rate) * height_gate, -max, +max)`
- Same indices: [1, 6] for hip_yaw_left/right
- Same height gate: smoothstep(com_z, 0.30, 1.10)
- Same ref_source handling

## 5. Verification Plan

After Phase 2 fix (static q_ref), re-run and compare:

```bash
# ramp_down (was SAFETY_FAIL hy=0.3728)
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --dynamic-height-trajectory outputs/k2_dynamic_height_gate_crossing/trajectories/ramp_down_0p480_to_0p330.json \
  --dynamic-qref-mode original-k2-exact \
  --steps 5000 --telemetry full \
  --output-dir outputs/k2_jax_dedicated_phase4/ramp_down_exact

# ramp_up (was SAFE_BUT_WORSE hy=0.1242)
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --dynamic-height-trajectory outputs/k2_dynamic_height_gate_crossing/trajectories/ramp_up_0p330_to_0p480.json \
  --dynamic-qref-mode original-k2-exact \
  --steps 5000 --telemetry full \
  --output-dir outputs/k2_jax_dedicated_phase4/ramp_up_exact

# gate_chatter (was SAFE_BUT_WORSE hy=0.2160)
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --dynamic-height-trajectory outputs/k2_dynamic_height_gate_crossing/trajectories/gate_chatter_0p400_0p470.json \
  --dynamic-qref-mode original-k2-exact \
  --steps 5000 --telemetry full \
  --output-dir outputs/k2_jax_dedicated_phase4/gate_chatter_exact
```

Then classify with strict rules:
```python
from wheeled_biped.validation.strict_promotion_classifier import load_classifier
classifier = load_classifier()
# Classify each scenario against baseline
```

## 6. Acceptance

| Criterion | Status |
|-----------|--------|
| low-height Step E hip_yaw metrics identified | ✅ |
| dynamic hip_yaw metrics identified | ✅ |
| ramp_down SAFETY_FAIL root cause found (interpolation) | ✅ |
| Fix implemented (static q_ref default) | ✅ |
| ramp_up/gate_chatter regressions root cause same as ramp_down | ✅ |
| mode_div Python/JAX formula parity confirmed | ✅ |
| no case exceeds 0.35 absolute gate (expected after fix) | ⏳ pending re-run |
| no pitch/support/drift regression introduced | ⏳ pending re-run |
