# Step E Correct-Mode Validation Report

**Date:** 2026-05-31  
**Run:** telemetry_1780208317.csv (2000 steps, 20.0 seconds)  
**Mode:** balance-core (WBC disabled, verified)

---

## Executive Summary

**CRITICAL FINDING:** The velocity-damped sagittal balance controller exhibits significant forward drift (max 0.595m) in true balance-core mode, **failing all position gates**.

**Phase 1 hypothesis REJECTED:** The posture regression observed in the visual run was caused by wrong controller mode (legacy/upright), but the **position drift problem persists in correct balance-core mode**.

---

## Balance-Core Mode Verification

✅ **WBC Successfully Disabled:**
- `tau_wbc_scaled_per_joint` = 0.0 at ALL steps (verified via telemetry)
- `tau_legacy_wheel_balance_norm` = 0.0
- `tau_legacy_hip_roll_centering_norm` = 0.0
- `hidden_torque_norm` = 0.0
- `ownership_violation_count` = 0

✅ **Balance-Core Controllers Active:**
- Console output confirms: `[BALANCE-CORE] Functional four-source controller stack enabled`
- Console output confirms: `[BALANCE-CORE] Sagittal controller: velocity-damped`
- `[EARLY SUPPORT]` debug messages show `tau_wbc_scaled=[0. 0. 0. ...]`

✅ **Code Fix Applied:**
- Added check at line 2240: `if args.controller_mode == "balance-core": include_wbc = False`
- Fix verified working via telemetry and console output

---

## Position Control Results

### Support Position Error

| Metric | Value | Status |
|--------|-------|--------|
| Min | -0.007 m | - |
| **Max** | **+0.595 m** | ❌ **FAIL** |
| Final | +0.039 m | ✅ PASS (< 0.05m) |
| Max abs | 0.595 m | ❌ **FAIL** |

### Position Gate Compliance

| Gate | Criteria | Result |
|------|----------|--------|
| **Preferred** | max ±0.10m, final ≤0.05m | ❌ **FAIL** (max 0.595m) |
| **Fallback** | max ±0.15m, final ≤0.10m | ❌ **FAIL** (max 0.595m) |
| **Hard Minimum** | max ≤0.30m, final ≤0.10m | ❌ **FAIL** (max 0.595m) |

### Drift Analysis

- **Max error occurs at:** Step 1360 (13.6 seconds) - **NOT initialization**
- **First 100 steps:** Max abs 0.068m (acceptable)
- **After step 100:** Max abs 0.595m (drift develops mid-run)
- **Steady-state (last 1000 steps):**
  - Max abs: 0.595m
  - Mean: +0.223m (persistent forward bias)
  - Std: 0.179m (high variability)

**Conclusion:** The robot drifts forward significantly during the run. This is NOT an initialization transient but a **persistent control problem**.

---

## Posture/Stance Results

### CoM Height
- Min: 0.362 m
- Max: 0.409 m
- Final: 0.375 m
- Range: 0.046 m (4.6 cm) - **Acceptable**

### Orientation
- Pitch range: 7.19° - **Acceptable**
- Roll range: 2.92° - **Good**
- Yaw range: 16.03° - **Moderate**

### Hip Roll
- Common-mode drift max: 0.32° - **Excellent**
- Symmetric splay max: 5.26° - **Acceptable**

### Joint Tracking
- Support joint error norm max: 0.456 rad - **Acceptable**

**Conclusion:** Posture and stance are stable. No leg collapse or splay issues in balance-core mode.

---

## Capture Gate Behavior

- **Enabled:** Yes
- **Active steps:** 0 / 2000 (0.00%)
- **Conclusion:** Gate never activated during nominal standing - **correct behavior**

The capture gate is designed to activate during transients (push recovery, height transitions). During nominal standing, it should remain inactive, which it does.

---

## Comparison: WBC Active vs WBC Disabled

| Metric | V3 (WBC active) | Current (WBC disabled) | Difference |
|--------|-----------------|------------------------|------------|
| Max abs error | 0.595 m | 0.595 m | 0.000 m |
| Final error | +0.053 m | +0.039 m | -0.014 m (better) |

**Finding:** The 0.595m drift occurs in BOTH modes. WBC is NOT the cause of the drift.

---

## Root Cause Analysis

### What We Know

1. **Forward drift of 0.595m occurs in both legacy and balance-core modes**
2. **Drift develops mid-run (step 1360), not during initialization**
3. **Posture remains stable** (no leg collapse/splay in balance-core mode)
4. **Capture gate never activates** (no detected position-capture conflict)
5. **Final error is acceptable** (0.039m < 0.05m), but peak is not

### Possible Causes

1. **Velocity-damped controller gain tuning:**
   - `k_position = 20.0` may be too weak
   - `k_velocity = 15.0` may be insufficient damping
   - Controller may not generate enough corrective torque for large errors

2. **Position reference drift:**
   - Support center equilibrium may be drifting
   - Pitch reference may have bias

3. **Wheel slip or contact issues:**
   - Wheels may be slipping forward
   - Contact model may allow drift

4. **Missing integral term:**
   - Pure PD control (position + velocity) has no integral action
   - Cannot reject constant disturbances or biases

### What Phase 1 Got Wrong

**Phase 1 claimed:** "Previous visual posture regression came from wrong runtime mode (legacy/upright)"

**Phase 1 was partially correct:** The visual run WAS in wrong mode, and that DID cause WBC to be active when it shouldn't be.

**But Phase 1 missed:** The position drift problem (0.595m) exists in BOTH modes and is NOT caused by WBC. It's a fundamental issue with the velocity-damped controller or the robot dynamics.

---

## Recommendations

### Immediate Actions

1. **DO NOT proceed to Step C** - position control must pass at least hard minimum gate first
2. **DO NOT claim Step E solved** - all position gates failed
3. **Investigate velocity-damped controller gains:**
   - Try increasing `k_position` from 20.0 to 40.0 or 60.0
   - Try increasing `k_velocity` from 15.0 to 30.0
   - Check if `max_position_tau = 3.0` is saturating

4. **Check for systematic biases:**
   - Verify pitch reference is zero
   - Verify support center equilibrium is correct
   - Check wheel slip in telemetry

5. **Consider adding integral term:**
   - Implement PID instead of PD
   - Or add feed-forward compensation for known biases

### Next Steps

**Option A: Tune velocity-damped controller**
- Systematically sweep `k_position` and `k_velocity`
- Find gains that keep max error < 0.30m (hard minimum gate)
- Verify no instability or oscillation

**Option B: Investigate root cause**
- Analyze telemetry for pitch bias, wheel slip, or contact issues
- Check if support center equilibrium is drifting
- Verify capture point calculation is correct

**Option C: Return to baseline LQR**
- If velocity-damped controller cannot be fixed, revert to baseline LQR
- Baseline LQR may have better position hold performance

---

## Files Generated

- `critical_bug_wbc_not_disabled_in_balance_core.md` - Bug documentation
- `critical_bug_wbc_not_disabled_in_balance_core.json` - Bug data
- `correct_mode_visual_validation_report.md` - This file

---

## Status

**Step E Smart Position-Hold Capture Gate:** ❌ **FAILED**

- Position control: FAIL (max 0.595m exceeds all gates)
- Posture control: PASS (stable, no collapse/splay)
- WBC disabled: PASS (verified)
- Balance-core mode: ACTIVE (verified)
- Capture gate: FUNCTIONAL (never activated during nominal standing - correct)

**Cannot proceed to Step C until position control passes at least hard minimum gate (max ≤ 0.30m).**
