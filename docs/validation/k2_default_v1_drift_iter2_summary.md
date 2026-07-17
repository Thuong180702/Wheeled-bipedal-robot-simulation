# K2 Default V1 Drift Iteration 2 — Final Summary

**Date:** 2026-06-30
**Best Profile:** `DRIFT_ITER2_VEL_ONLY_WIDE_GATE`
**Decision:** **REJECT — DO NOT PROMOTE**

---

## Critical Bug Found

The original `K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED_CANDIDATE` profile was **never active** during previous validation. The profile map key (`"k2_jax_dedicated_default_v1_drift_fixed"`) didn't match the CLI argument (`K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED_CANDIDATE`), causing fallback to DEFAULT_V1 (`enable_drift_controller=False`). All prior conclusions about the drift controller being "safe but ineffective" were actually testing DEFAULT_V1 twice.

**Fix:** Added CLI key to `_PROFILE_MAP` in `run_k2_jax_realtime.py`.

---

## Phase 1 Diagnostic Findings

With drift controller actually active (narrow gate: hgate 0.005→0.03 m/s):

| Metric | Value |
|---|---|
| height_gate mean (low_0p320) | 0.053 |
| height_gate active (>0.01) | 9.0% of steps |
| drift torque nonzero | 6.4% of steps |
| Root cause | `height_gate` suppressed 91% of drift control due to too-narrow smoothstep (0.005→0.03 m/s) |

Normal CoM oscillation during balance (~0.16 m/s) constantly exceeded the 0.005 m/s threshold.

---

## Phase 2-3: Profile Variants Created and Tested

| Variant | Description | low_0p320 | high_0p430 |
|---|---|---|---|
| DRIFT_FIXED (updated) | k_vel=6.0, k_head=3.0, wide gate | 0 falls ✅ | 0 falls ✅ |
| **A (VEL_ONLY)** | k_vel=10.0, no heading, wide gate | **0 falls ✅** | **0 falls ✅** |
| B (VEL_HEADING) | +k_heading=5.0 | **FALL** ❌ | 0 falls |
| C (LATE_POS) | +k_pos=1.5 | **FALL** ❌ | 0 falls |
| D (PUSH_DAMP) | push_damp=3.0 | **FALL** ❌ | 0 falls |
| E (DYN_YIELD) | late pos, wide gate | **FALL** ❌ | 0 falls |

**Heading-based variants (B-E) all fall at low height (0.32m).** The antisymmetric wheel torque from heading correction (k_head=5.0 × yaw_error=0.34 rad = 1.7 Nm differential) destabilizes the robot at low CoM.

---

## Phase 4: Variant A Validation Results

| Scope | Result | Details |
|---|---|---|
| step_e (10 fixed-height) | PARTIAL | 7 PASS, 3 SAFE_BUT_WORSE, 0 falls |
| step_d (12 push) | **PASS** | 12/12 WITHIN_OLD_TOLERANCE |
| dynamic_height (5) | PARTIAL | 1 PASS, 4 SAFE_BUT_WORSE, 0 falls |
| long_run (5) | PARTIAL | 2 PASS, 3 SAFE_BUT_WORSE, 0 falls |

---

## Phase 5: Full Matrix Evaluation

| Metric | Baseline (DEFAULT_V1) | Variant A | Delta |
|---|---|---|---|
| Aggregate Score | 0.6935 | 0.6936 | +0.0000 |
| Posture | 0.6776 | 0.6781 | +0.0004 |
| Support/Drift | 0.5019 | 0.4998 | **-0.0021** |
| Leg Health | 0.8326 | 0.8348 | +0.0022 |
| Dynamic Height | 0.5371 | 0.5371 | -0.0000 |
| Torque Quality | 0.9139 | 0.9139 | -0.0000 |
| Robustness | 0.9303 | 0.9302 | -0.0001 |
| **Falls** | **0** | **0** | ✅ |
| Performance Hz | 150.2 | 147.4 | -2.8 |

### Fixed-height displacement (60s runs, from telemetry)

| Scenario | DRIFT_FIXED (updated) | Variant A | Improvement |
|---|---|---|---|
| low_0p320 final_disp | 0.085m | 0.045m | **-47%** |
| high_0p430 final_disp | 0.185m | 0.046m | **-75%** |

---

## Why Variant A Doesn't Improve the Aggregate Score

1. **Velocity damping only slows drift, doesn't prevent it.** No position return means long-term displacement still accumulates.
2. **No heading correction.** Yaw drift is unaddressed (yaw drift got *worse* in some scenarios due to asymmetric torque from velocity damping).
3. **Dynamic height scores dominate the aggregate.** Drift controller is fully suppressed during height transitions (height_gate=0), so dynamic height scenarios show no benefit.
4. **Push scenarios show no difference.** Push torque (10+ Nm) overwhelms drift velocity damping (max 1.6 Nm).
5. **Support/Drift dimension actually *worsens* slightly** (-0.0021), suggesting the drift controller's wheel torque may slightly alter support center dynamics.

---

## Decision: REJECT — DO NOT PROMOTE

### Why rejection:

The wheel-only drift controller architecture with velocity damping can slow drift velocity (47-75% improvement at fixed height) but cannot:

1. **Prevent long-term displacement accumulation** (no position return that works without causing falls)
2. **Correct heading drift** (heading torque destabilizes at low heights)
3. **Operate during height transitions** (height_gate fully suppresses drift control)
4. **Improve aggregate quality score** (flat at 0.69)

### Root architectural limitation:

The drift controller uses wheel torques only. But:
- Velocity damping works against sagittal balance (wheels are the primary pitch stabilizer)
- Heading correction via wheel differential fights the sagittal balance controller
- Position return requires sustained wheel torque asymmetry, which slowly rotates the robot

### Recommended next direction:

The heading/yaw drift problem should be addressed at the **estimation and yaw coupling level**, not through wheel torque:
1. **Yaw estimation drift**: The `est_*` inputs come from centroidal state which accumulates yaw error over time. Investigate yaw estimation quality.
2. **Wheel asymmetry / yaw coupling**: The sagittal velocity damping may introduce slight left/right asymmetry that causes slow yaw rotation. This should be characterized.
3. **Hip yaw joints**: Heading correction through leg-joint yaw (indices [1, 6]) may be more appropriate than wheel differential.
4. **Split height gate**: Velocity damping should persist during slow height motion; only fast transitions should suppress it.

Increasing drift gains further is not recommended — heading variants at k_heading=5.0 already cause falls, and higher k_vel would fight the sagittal balance controller.
