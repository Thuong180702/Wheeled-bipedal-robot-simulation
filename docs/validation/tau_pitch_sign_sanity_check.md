# Tau_pitch Sign and Direction Sanity Check

**Date:** 2026-06-08
**Profile:** D2 at low_0p300

---

## 1. Sign Convention Verification

### From Controller Documentation

```
Control law:
    tau = k_pitch * pitch_x + k_pitch_rate * pitch_rate_x + ...
    
Signs verified by unit tests:
    - positive pitch → restoring torque (opposes tilt)
    - positive pitch_rate → damping torque (opposes angular velocity)
```

### Physics Interpretation

**pitch_x (body pitch angle):**
- `pitch_x > 0` = robot nose UP = falling FORWARD
- `pitch_x < 0` = robot nose DOWN = falling BACKWARD

**tau_pitch:**
- `tau_pitch > 0` = apply FORWARD wheel torque
- Forward wheel torque pushes wheels forward → body goes BACK
- **Result:** tau_pitch > 0 CORRECTLY opposes forward pitch

**tau_pitch_rate:**
- `tau_pitch_rate = kd_pitch * pitch_rate_x`
- `kd_pitch = 10.0`
- `pitch_rate_x > 0` = nose going UP faster (angular velocity positive)
- `tau_pitch_rate > 0` = opposing torque during upward pitch
- **Result:** tau_pitch_rate CORRECTLY damps pitch angular velocity

---

## 2. D2 Telemetry Evidence

### tau_pitch Sign During Forward Lean

| Condition | Steps | tau_pitch mean | tau_pitch correct% |
|-----------|-------|----------------|-------------------|
| pitch_x > 0 (forward lean) | 445 (89.0%) | +2.95 Nm | 100% |
| pitch_x < 0 (backward lean) | 54 (10.8%) | -0.24 Nm | 98% |

**Conclusion:** tau_pitch sign is CORRECT. When robot leans forward, tau_pitch is positive (correctly opposing).

### tau_pitch_rate Damping Sign

| Condition | Steps | tau_pitch_rate mean | tau_pitch_rate correct% |
|-----------|-------|--------------------|----------------------|
| pitch_rate > 0 (nose up) | 210 (42.0%) | +0.95 Nm | 100% |
| pitch_rate < 0 (nose down) | 289 (57.8%) | -0.53 Nm | 100% |

**Conclusion:** tau_pitch_rate sign is CORRECT. Damping opposes angular velocity.

### tau_wheel_velocity Sign

| Condition | Steps | tau_wheel_vel_left mean |
|-----------|-------|------------------------|
| wheel_vel_mean > 0 (forward) | 259 (51.8%) | -0.73 Nm |
| wheel_vel_mean < 0 (backward) | 240 (48.0%) | +0.53 Nm |

**Conclusion:** tau_wheel_velocity sign is CORRECT. Damping opposes wheel velocity.

---

## 3. Is tau_pitch Causing Forward Fall or Responding?

### Evidence Analysis

**Evidence 1: Event order**
```
Step 0: pitch_x = 0, tau_pitch = 0 (equilibrium)
Step 1-3: Small oscillations, tau_pitch near zero
Step 9+: pitch_x > 0, tau_pitch > 0 (both grow together)
```

**Evidence 2: Correlation**
- corr(pitch_x, tau_pitch) = 1.0000
- tau_pitch directly proportional to pitch_x
- tau_pitch = 50.0 * pitch_x (by formula)

**Evidence 3: tau_pitch is proportional feedback**
- tau_pitch is not a feedforward term
- It responds to pitch_x error
- **tau_pitch is NOT driving the forward lean; it is responding to it**

---

## 4. Classification

| Check | Result | Evidence |
|-------|--------|----------|
| TAU_PITCH_SIGN_CORRECT_RESPONDING_TO_FORWARD_FALL | ✅ CONFIRMED | pitch_x > 0 → tau_pitch > 0 correctly opposes lean |
| TAU_PITCH_SIGN_WRONG_DRIVING_FORWARD_FALL | ❌ REFUTED | tau_pitch sign is opposite to direction of fall |
| TAU_PITCH_DAMPING_SIGN_SUSPECT | ❌ CLEAR | tau_pitch_rate opposes pitch_rate correctly |
| TAU_PITCH_DIRECTION_INCONCLUSIVE | ❌ NOT INCONCLUSIVE | Clear evidence tau_pitch responds to real lean |

---

## 5. Why is tau_pitch Persistently Positive?

**tau_pitch is persistently positive because pitch_x is persistently positive.**

The question is: **Why is pitch_x persistently positive?**

### Potential Causes

1. **Initial hip_pitch error (0.45 rad)**
   - Hips start more extended than equilibrium
   - Creates backward moment → forward pitch response
   - tau_pitch responds correctly to this

2. **Position authority insufficient**
   - tau_position cap of 4.0 Nm cannot fully cancel tau_pitch (~5.5 Nm peak)
   - Net positive torque → forward lean continues
   - tau_pitch responds correctly to the resulting lean

3. **Low-height geometry instability**
   - low_0p300 has extreme posture (hip=78.85°, knee=134.56°)
   - May have natural forward instability tendency
   - tau_pitch responds correctly to the lean

---

## 6. Conclusion

### tau_pitch Sign: CORRECT
- tau_pitch > 0 when pitch_x > 0 (opposes forward lean)
- tau_pitch < 0 when pitch_x < 0 (opposes backward lean)
- **tau_pitch is NOT the source of the problem**

### tau_pitch Direction: RESPONDING, NOT CAUSING
- tau_pitch is proportional feedback to pitch error
- It does not drive the forward lean
- It responds to real pitch_x deviations

### tau_pitch Damping: CORRECT BUT WEAK
- tau_pitch_rate opposes pitch_rate correctly
- But kd_pitch = 10.0 is relatively small
- Damping is insufficient to prevent pitch buildup

### Final Classification

**TAU_PITCH_SIGN_CORRECT_RESPONDING_TO_FORWARD_FALL**

tau_pitch is a correct response to persistent forward lean. The root cause lies elsewhere:
1. Initial hip_pitch error (steady bias)
2. Insufficient position authority (cannot cancel tau_pitch)
3. Low-height geometry instability

**Do not reduce tau_pitch gain or change tau_pitch sign.** This would make balance WORSE.

The fix should target:
1. Reduce initial hip_pitch error
2. Increase position authority at low height
3. Investigate height-dependent equilibrium pitch