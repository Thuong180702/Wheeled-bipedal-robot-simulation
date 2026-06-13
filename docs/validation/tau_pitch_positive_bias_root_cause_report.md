# Tau_pitch Positive Bias Root Cause Report

**Date:** 2026-06-08
**Profiles:** D2, F1b, G1a, G1b at low_0p300
**Classification Task:** Phase 5

---

## 1. Executive Summary

**Root cause: TAU_PITCH_BIAS_FROM_INITIAL_CONDITION + TAU_PITCH_BIAS_FROM_GAIN_TOO_HIGH at low height**

The persistent positive tau_pitch bias at low_0p300 is caused by:
1. **Initial hip_pitch error of 0.45 rad** - steady bias from non-equilibrium start
2. **tau_position cap of 4.0 Nm insufficient** - cannot cancel tau_pitch up to 5.5 Nm
3. **Position authority too low for low-height dynamics** - net positive torque builds up

**tau_pitch computation, sign convention, and pitch reference are all correct.**

---

## 2. Evidence Summary

### Phase 1: Computation Audit ✅

| Check | Result | Evidence |
|-------|--------|----------|
| Formula | `tau_pitch = kp_pitch * pitch_x_error` | Correct |
| kp_pitch | 50.0 | Standard gain |
| Pitch reference | 0.0 | Correct per equilibrium search |
| Sign convention | positive pitch → positive torque | Correct (opposes lean) |

**Conclusion:** Computation is correct.

### Phase 2: Telemetry Decomposition ✅

| Observation | Evidence |
|-------------|----------|
| pitch_x positive% | 89.0% (D2), 82.6-84.0% (F1b-G1a) |
| tau_pitch positive% | 89.2% (D2), 82.8-84.2% (F1b-G1a) |
| tau_position clips at | ±4.0 Nm |
| Position saturation | 35.4% of steps (D2) |
| Position cannot cancel tau_pitch | tau_pitch up to 5.5 Nm, cap at 4.0 Nm |

**Conclusion:** Position authority insufficient.

### Phase 3: Pitch Reference Audit ⚠️

| Observation | Evidence |
|-------------|----------|
| equilibrium_pitch_x | 0.0 (from setup) |
| pitch_x_ref_rad | 0.0 throughout |
| hip_pitch_error_max | 0.45 rad (INITIAL BIAS) |
| hip_pitch_error_left_rad | -0.49 rad (steady negative) |

**Conclusion:** Initial hip_pitch error creates steady bias.

### Phase 4: Sign Sanity Check ✅

| Check | Result | Evidence |
|-------|--------|----------|
| tau_pitch sign when pitch > 0 | Correct (opposes forward lean) | 100% correct |
| tau_pitch_rate sign | Correct (damping) | 100% correct |
| tau_pitch direction | Responding, not causing | Correlation = 1.0 |

**Conclusion:** tau_pitch is a correct response, not the cause.

---

## 3. Root Cause Classification

### Primary Root Cause: TAU_PITCH_BIAS_FROM_INITIAL_CONDITION

**Classification:** TAU_PITCH_BIAS_FROM_INITIAL_CONDITION

**Evidence:**
1. hip_pitch_error_max = 0.45 rad at step 0
2. Error remains throughout run (never corrects below 0.45 rad)
3. hip_pitch_error_left_rad = -0.49 rad (joint too extended)
4. This creates backward moment → forward pitch → positive tau_pitch

**Mechanism:**
```
Initial state: hip_pitch too extended (error = 0.45 rad)
    ↓
Backward moment from extended hips
    ↓
Robot pitches forward (pitch_x > 0)
    ↓
tau_pitch = 50 * pitch_x > 0 (opposing)
    ↓
Forward wheel torque → forward drift
    ↓
pitch_x stays positive → tau_pitch stays positive
```

### Secondary Root Cause: TAU_PITCH_BIAS_FROM_GAIN_TOO_HIGH

**Classification:** TAU_PITCH_BIAS_FROM_UNMODELED_SUPPORT_DRIFT (at low height)

**Evidence:**
1. tau_position cap of 4.0 Nm
2. tau_pitch reaches 5.5 Nm peak
3. Position cannot fully cancel pitch
4. Net positive torque builds up
5. Forward lean persists

**Mechanism:**
```
pitch_x > 0 → tau_pitch = 50 * pitch_x
    ↓
tau_position tries to cancel (capped at 4.0 Nm)
    ↓
Net torque = tau_pitch - min(tau_pitch, 4.0) > 0
    ↓
Forward lean continues
    ↓
pitch_x stays positive
```

### NOT the Root Cause

| Suspected Cause | Ruled Out | Evidence |
|-----------------|-----------|----------|
| Wrong pitch reference | ✅ Ruled out | Reference = 0.0, correct per setup |
| Wrong tau_pitch sign | ✅ Ruled out | Sign correct (opposes lean) |
| Wrong pitch rate sign | ✅ Ruled out | Damping correct |
| tau_pitch drives forward fall | ✅ Ruled out | tau_pitch responds to pitch_x |
| Low-height geometry needs pitch offset | ⚠️ Possible | Extreme posture may need offset |

---

## 4. Why Downstream Bias Cancellation Fails

### G1b Analysis

**G1b configuration:**
- enable_bias_cancel = true
- bias_cancel_k = 15.0 (stronger than G1a)
- bias_cancel_max_tau = 1.5 Nm

**Result:**
- positive% reduced: 80.4% (G1b) vs 89.2% (D2) ✓
- outside ±0.15 increased: 26.8% (G1b) vs 19.2% (D2) ✗

**Why G1b fails:**
1. Bias cancel applies opposite torque when signed_error > 0
2. But tau_pitch is a CORRECT response to real pitch_x
3. Bias cancel fights the symptom, not the cause
4. Overcorrection causes oscillations outside ±0.15

### The Vicious Cycle

```
Initial hip_pitch error (0.45 rad)
    ↓
Forward lean (pitch_x > 0)
    ↓
tau_pitch > 0 (correct response)
    ↓
Position cannot cancel (cap 4.0 Nm < tau_pitch 5.5 Nm)
    ↓
Net forward torque
    ↓
Forward drift continues
    ↓
G1b applies bias cancel (wrong direction)
    ↓
Overcorrection → oscillation
```

---

## 5. Recommended Fixes (No Implementation Yet)

### Fix A: Reduce Initial Hip_pitch Error

**Approach:** Fix the initial state mismatch
- Robot should start AT equilibrium, not near it
- hip_pitch_error should be < 0.05 rad, not 0.45 rad

**Risk:** Low (fixes root cause)

### Fix B: Increase Position Authority at Low Height

**Approach:** Continuous height-scheduled position cap
- tau_position cap increases smoothly below 0.40 m
- At 0.30 m: cap = 6.0 Nm (vs current 4.0 Nm)
- Maintains D2 protection at nominal heights

**Risk:** Medium (may affect D2 stability)

### Fix C: Add Height-Dependent Pitch Offset

**Approach:** Equilibrium pitch_x varies with height
- At low_0p300: equilibrium_pitch_x = +0.05 rad (forward lean)
- This matches the natural posture at extreme squat
- Reduces pitch error from initial condition bias

**Risk:** High (requires equilibrium re-search)

---

## 6. Conclusion

**Final Classification:** TAU_PITCH_BIAS_SOURCE_INCONCLUSIVE? NO - **IDENTIFIED**

**Root causes (in order of impact):**
1. **Initial hip_pitch error** (0.45 rad) - creates steady forward moment
2. **Position authority insufficient** (cap 4.0 Nm < tau_pitch 5.5 Nm) - net positive torque
3. **Possible height-dependent equilibrium pitch** - low_0p300 may need non-zero reference

**tau_pitch computation and sign are correct.** tau_pitch is responding to real forward lean, not causing it.

**Downstream bias cancellation (G1b) is a symptom patch that worsens oscillation.**

**Recommended next step:** Investigate initial condition mismatch and position authority scheduling before attempting any controller changes.