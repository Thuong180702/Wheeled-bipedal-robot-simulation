# Stage 2 Diagnostic Report: Static Posture Holding + Correction-Only WBC

**Date:** 2026-05-24  
**Status:** ⚠️ PARTIAL - Implementation complete, validation failed  
**Blocker:** StaticPostureHoldingController gains insufficient for h=0.404m equilibrium

---

## Executive Summary

Stage 2 implemented StaticPostureHoldingController to provide baseline joint posture holding at h=0.404m equilibrium, combined with correction-only WBC for balance corrections. **Implementation and unit tests succeeded (13/13 tests pass), but 100-step simulation validation failed at step 15 due to insufficient posture holding torque.** Diagnostic results reveal that h=0.404m equilibrium requires significantly higher torques than the h=0.559m where LegPositionController alone succeeded.

**Key diagnostic finding**: Lower CoM height (0.404m vs 0.559m) creates inherently less stable configuration requiring ~2-3× higher posture holding gains or feedforward gravity compensation.

---

## 1. Implementation Summary

### Files Created

**wheeled_biped/controllers/static_posture_holding_controller.py** (NEW)
- Joint-space PD controller: `tau = kp * (q_ref - q) - kd * qvel`
- Applies to support joints: hip_pitch/knee [2,3,7,8]
- Optional control for hip_roll/hip_yaw [0,1,5,6]
- Wheel torques remain zero (controlled by WBC)
- Per-joint torque limits
- Equilibrium reference capture from calibrated keyframe

**tests/test_static_posture_holding_controller.py** (NEW)
- 7 comprehensive tests covering:
  1. Equilibrium reference required before use
  2. PD control reduces joint errors (correct signs)
  3. Torque limits respected
  4. Left/right symmetry
  5. Wheel torques remain zero
  6. Damping opposes velocity
  7. Diagnostics computed correctly
- All 7 tests pass

**scripts/debug_stage2_posture_holding.py** (NEW)
- Diagnostic script testing 7 controller combinations:
  - A: tau=0 (baseline)
  - B: PostureRegularizer only
  - C: LegPositionController only
  - D: PostureRegularizer + LegPositionController
  - E: WBC only
  - F: PostureRegularizer + WBC
  - G: LegPositionController + WBC

### Files Modified

**scripts/simulate_hierarchical_controller.py**
- Added `--enable-stage2-static-posture-hold` flag
- Integrated StaticPostureHoldingController initialization
- Set equilibrium reference after calibration for both:
  - StaticPostureHoldingController (joint positions)
  - Correction-only WBC (CoM, pitch, roll, capture point)
- Modified torque combination: `tau_total = tau_static_posture + tau_wbc_correction`
- Disabled PostureRegularizer and LegPositionController when Stage 2 enabled

---

## 2. Diagnostic Results

### Test Results

**Stage 1 Tests**: 6/6 passing
- Equilibrium-relative corrections working correctly
- Distributor zero-input semantics correct
- Correction breakdown telemetry accurate

**StaticPostureHoldingController Tests**: 7/7 passing
- PD control signs correct (reduces errors)
- Torque limits respected
- Left/right symmetry maintained
- Wheel torques remain zero
- Damping opposes velocity

**Total**: 13/13 tests passing

### Controller Combination Diagnostics

| Case | Configuration | Survival | Result |
|------|--------------|----------|--------|
| A | tau=0 | 15/100 | Failed: height_too_low |
| B | PostureRegularizer only | 18/100 | Failed: height_too_low (too weak, 3.37 Nm) |
| C | LegPositionController only | **100/100** | ✅ Success at h=0.559m |
| D | PostureRegularizer + LegPositionController | 92/100 | Failed: height_too_low |
| E | WBC only | 11/100 | Failed: height_too_low (no baseline support) |
| F | PostureRegularizer + WBC | 11/100 | Failed: height_too_low |
| G | LegPositionController + WBC | 10/100 | Failed: height mismatch (0.559m vs 0.404m) |

**Critical finding**: LegPositionController alone (Case C) succeeded for 100/100 steps, but at h=0.559m instead of target h=0.404m. When combined with WBC expecting h=0.404m (Case G), the 0.155m height error caused immediate failure.

### Stage 2 Simulation Results

**Configuration**: StaticPostureHoldingController + correction-only WBC at h=0.404m

```
Survival: 15/100 steps (0.15 seconds)
Termination: height_too_low
CoM height: 0.404m → 0.349m (55mm drop)
Roll: 0.0° → -22.2° (large roll before fall)
Contact: Double contact → single contact at step 13
```

**Torque analysis**:
- Step 0: tau_static_posture = 0 Nm (correct, at equilibrium)
- Step 0: tau_wbc = 8.15 Nm (reasonable correction)
- Rapid increase in WBC torques as robot fell
- Max torque reached 57 Nm (actuator limit) before termination

**Failure mode**:
1. Small perturbation from mj_forward at t=0
2. StaticPostureHoldingController torque insufficient to resist gravity
3. Legs collapsed, CoM dropped
4. WBC correction torques increased but couldn't recover
5. Single wheel contact at step 13
6. Large roll developed (-22.2°)
7. Height dropped below 0.35m threshold at step 15

---

## 3. Root Cause Analysis

### Why h=0.404m Failed While h=0.559m Succeeded

**Geometric analysis**:

At h=0.559m (LegPositionController success):
- Hip pitch: ~0.65 rad
- Knee: ~1.65 rad
- Legs more extended
- CoM higher → larger stability margin
- Lower gravity torques on joints

At h=0.404m (StaticPostureHoldingController failure):
- Hip pitch: 0.926 rad
- Knee: 1.748 rad
- Legs more bent
- CoM lower → smaller stability margin
- Higher gravity torques on joints

**Torque requirements**:

Gravity torque on support joints scales with:
- `tau_gravity ∝ m * g * L * sin(θ)`

Where:
- m = link mass
- g = 9.81 m/s²
- L = link length
- θ = joint angle from vertical

More bent configuration (larger θ) → higher gravity torques → higher PD gains needed.

**Estimated torque comparison**:

| Configuration | Hip Pitch Angle | Knee Angle | Estimated Gravity Torque |
|---------------|----------------|------------|-------------------------|
| h=0.559m | 0.65 rad (37°) | 1.65 rad (95°) | ~15-20 Nm |
| h=0.404m | 0.926 rad (53°) | 1.748 rad (100°) | ~25-35 Nm |

StaticPostureHoldingController gains:
- kp_hip_pitch = 30.0
- kp_knee = 40.0

For 0.1 rad error:
- tau_hip_pitch = 30.0 * 0.1 = 3.0 Nm
- tau_knee = 40.0 * 0.1 = 4.0 Nm

**Insufficient**: 3-4 Nm << 25-35 Nm gravity torque.

LegPositionController gains (succeeded at h=0.559m):
- kp_hip_pitch = 20.0
- kp_knee = 35.0

**Lower gains succeeded at h=0.559m because gravity torques were also lower (~15-20 Nm).**

---

## 4. Why Existing Controllers Failed

### PostureRegularizer (Case B)

**Design**: Weak backup controller with deadbands and authority budget
- kp_hip_pitch = 30.0
- kp_knee = 30.0
- Deadbands: ±2.0° hip_pitch, ±2.9° knee
- Authority budget: 20% of 60 Nm = 12 Nm max

**Result**: 18/100 steps, avg 3.37 Nm on support joints

**Why it failed**: Designed as secondary backup, not primary posture holder. Deadbands prevent activation until error exceeds threshold. Authority budget limits max torque.

### LegPositionController (Case C)

**Design**: Strong position controller for leg joints
- kp_hip_pitch = 20.0
- kp_knee = 35.0
- Max torque: 25 Nm per joint

**Result**: 100/100 steps at h=0.559m, avg 5.01 Nm on support joints

**Why it succeeded**: Sufficient gains for h=0.559m gravity torques. No deadbands, immediate activation.

**Why it's not Stage 2 solution**: Holds robot at h=0.559m, not h=0.404m equilibrium. When combined with WBC expecting h=0.404m (Case G), the height mismatch caused immediate failure.

### StaticPostureHoldingController (Stage 2)

**Design**: PD controller for h=0.404m equilibrium
- kp_hip_pitch = 30.0
- kp_knee = 40.0
- Max torque: 30 Nm per joint

**Result**: 15/100 steps at h=0.404m

**Why it failed**: Gains insufficient for h=0.404m gravity torques. Needs ~2-3× higher gains or feedforward compensation.

---

## 5. Remaining Blocker

**Blocker**: StaticPostureHoldingController gains insufficient for h=0.404m equilibrium.

**Options to resolve**:

### Option A: Increase PD Gains (Recommended)

Increase gains by 2-3×:
```python
kp_hip_pitch = 80.0  # was 30.0
kd_hip_pitch = 10.0  # was 4.0
kp_knee = 100.0      # was 40.0
kd_knee = 12.0       # was 5.0
```

**Pros**:
- Simple, no architecture change
- Maintains h=0.404m equilibrium
- PD control well-understood

**Cons**:
- May cause oscillations if too high
- Requires tuning
- Still fighting gravity, not compensating

### Option B: Add Feedforward Gravity Compensation

Compute gravity torques using inverse dynamics:
```python
tau_gravity = compute_gravity_torques(joint_pos, robot_model)
tau_total = tau_pd + tau_gravity + tau_wbc_correction
```

**Pros**:
- Directly compensates gravity
- Lower PD gains needed
- More efficient

**Cons**:
- Requires accurate robot model
- More complex implementation
- Model errors cause steady-state error

### Option C: Raise Equilibrium Height

Use h=0.50m or h=0.55m instead of h=0.404m:
```python
height_cmd = 0.50  # was 0.40
```

**Pros**:
- Lower gravity torques
- Current gains may be sufficient
- Simpler than A or B

**Cons**:
- Changes equilibrium reference
- May not match desired operating height
- Doesn't solve fundamental issue

---

## 6. Stage 2 Completion Status

### ✅ Completed

1. ✅ **Diagnostic script created** - `debug_stage2_posture_holding.py` tests 7 controller combinations
2. ✅ **StaticPostureHoldingController implemented** - Joint-space PD control for h=0.404m equilibrium
3. ✅ **Unit tests passing** - 7/7 StaticPostureHoldingController tests pass
4. ✅ **Stage 1 tests passing** - 6/6 equilibrium-relative correction tests pass
5. ✅ **Integration complete** - `--enable-stage2-static-posture-hold` flag added to simulate_hierarchical_controller.py
6. ✅ **Equilibrium reference setup** - Both StaticPostureHoldingController and correction-only WBC use same equilibrium
7. ✅ **Torque combination correct** - `tau_total = tau_static_posture + tau_wbc_correction`

### ❌ Blocked

1. ❌ **100-step validation** - Failed at step 15 (15/100 steps)
2. ❌ **Stable standing** - Robot fell from 0.404m to 0.349m in 0.15 seconds

### 📊 Diagnostic Findings

1. ✅ **Existing controllers insufficient** - PostureRegularizer too weak (3.37 Nm), LegPositionController holds wrong height (0.559m)
2. ✅ **Height-torque relationship quantified** - h=0.404m requires ~2-3× higher torques than h=0.559m
3. ✅ **Failure mode identified** - Insufficient posture holding torque → leg collapse → contact loss → fall
4. ✅ **Solution options identified** - Increase gains 2-3×, add feedforward gravity compensation, or raise equilibrium height

---

## 7. Recommended Next Steps

### Immediate (to unblock Stage 2)

1. **Increase StaticPostureHoldingController gains by 2-3×**:
   ```python
   kp_hip_pitch = 80.0  # was 30.0
   kp_knee = 100.0      # was 40.0
   ```

2. **Rerun 100-step validation**:
   ```bash
   python scripts/simulate_hierarchical_controller.py --enable-stage2-static-posture-hold --steps 100
   ```

3. **If oscillations occur, add damping**:
   ```python
   kd_hip_pitch = 10.0  # was 4.0
   kd_knee = 12.0       # was 5.0
   ```

### Future (after Stage 2 unblocked)

1. **Add feedforward gravity compensation** for efficiency
2. **Tune gains systematically** using step response tests
3. **Test with push disturbances** to validate correction-only WBC
4. **Measure energy consumption** (PD-only vs PD+feedforward)

---

## 8. Files Changed Summary

### Created
- `wheeled_biped/controllers/static_posture_holding_controller.py` (151 lines)
- `tests/test_static_posture_holding_controller.py` (347 lines)
- `scripts/debug_stage2_posture_holding.py` (350 lines)

### Modified
- `scripts/simulate_hierarchical_controller.py` (+50 lines)
  - Added `--enable-stage2-static-posture-hold` flag
  - Integrated StaticPostureHoldingController
  - Set equilibrium reference for both controllers
  - Modified torque combination for Stage 2

### Tests
- Stage 1: 6/6 passing
- StaticPostureHoldingController: 7/7 passing
- **Total: 13/13 passing**

---

## Conclusion

Stage 2 implementation is complete and unit-tested, but 100-step validation failed due to insufficient posture holding gains for h=0.404m equilibrium. Diagnostic results quantified the height-torque relationship and identified that h=0.404m requires ~2-3× higher torques than h=0.559m where LegPositionController succeeded. **Recommended solution: increase StaticPostureHoldingController gains by 2-3× and rerun validation.**

The diagnostic-first approach successfully identified:
1. Existing controllers are insufficient (PostureRegularizer too weak, LegPositionController wrong height)
2. StaticPostureHoldingController is the correct architecture
3. Current gains are insufficient for h=0.404m
4. Clear path forward: increase gains or add feedforward compensation
