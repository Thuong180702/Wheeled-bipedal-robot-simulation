# low_0p300 Pitch Reference and Setup Audit

**Date:** 2026-06-08
**Profile:** D2 at low_0p300

---

## 1. Setup File Evidence

### low_0p300 Setup Parameters

```json
{
  "target_com_z_m": 0.30,
  "achieved_com_z_m": 0.295,
  "hip_pitch_ref": 1.376 rad (78.85 deg),
  "knee_ref": 2.348 rad (134.56 deg),
  "equilibrium_pitch_x": 0.0 rad,
  "equilibrium_roll_y": 0.0 rad,
  "equilibrium_yaw_z": 0.0 rad,
  "com_support_error_y": 0.0125 m,
  "com_y_m": -0.0220 m,
  "support_center_y": -0.0345 m
}
```

### Pitch Reference
- `equilibrium_pitch_x = 0.0` in setup file
- `pitch_x_ref_rad = 0.0` throughout telemetry
- **Confirmed: pitch reference is correctly set to 0.0**

---

## 2. CRITICAL FINDING: Initial Joint Position Error

### hip_pitch_error_max Statistics
```
initial: 0.4500 rad (25.79 deg)
final: 0.4993 rad (28.61 deg)
min: 0.4500 rad (25.79 deg)
max: 0.4993 rad (28.61 deg)
mean: 0.4946 rad (28.34 deg)
```

### hip_pitch_error_left_rad and hip_pitch_error_right_rad
```
hip_pitch_error_left_rad:
  mean=-0.4946 rad (-28.34 deg)
  negative throughout (joint too extended)
  
hip_pitch_error_right_rad:
  mean=-0.4786 rad (-27.43 deg)
  negative throughout (joint too extended)
```

### Interpretation
**hip_pitch_error = target - current**
- Negative error means **current > target** → hip_pitch is MORE extended than equilibrium
- This creates a forward pitch moment because extended hips push the body backward

### Event Order
1. **Step 0:** hip_pitch_error = -0.45 rad (hips too extended)
2. **Step 0-5:** Controller corrects hip_pitch (error decreases to ~-0.46)
3. **Step 5-20:** pitch_x grows from 0 to 0.34 deg
4. **Step 20-100:** pitch_x grows to 6.36 deg (peak)

**The initial hip_pitch error creates a restoring moment that causes forward pitch.**

---

## 3. Equilibrium vs Initial State

### Equilibrium State (from setup)
- hip_pitch_L = hip_pitch_R = 1.376 rad
- com_y = -0.022 m
- pitch = 0 deg

### Initial State (step 0)
- hip_pitch_L = hip_pitch_R = 0.926 rad (target - 0.45 rad error)
- **Hips are 0.45 rad MORE EXTENDED than equilibrium**
- This pushes the body backward relative to the wheels
- Creates forward pitch moment

### Physical Interpretation
When hips are too extended (knees straighter than equilibrium):
- CoM moves backward relative to wheel contact points
- Robot pitches forward to compensate
- Controller sees pitch > 0 and applies tau_pitch > 0 (forward wheel torque)
- This creates forward drift

---

## 4. Position Error vs Pitch Correlation

| Correlation | Value |
|-------------|-------|
| corr(pitch_x, tau_pitch) | 1.0000 |
| corr(pitch_x, sagittal_pos_error) | 0.9353 |
| corr(pitch_x, hip_yaw_comp_support_error) | 0.9309 |
| corr(hip_pitch_error_max, pitch_x) | 0.1874 |

**hip_pitch_error_max and pitch_x have LOW correlation (0.19).**

This suggests:
1. hip_pitch_error is a steady bias (always ~0.45 rad)
2. pitch_x varies dynamically based on balance control
3. Both contribute to forward lean tendency

---

## 5. Pitch Reference Classification

### Is pitch reference wrong?

**NO.** The pitch reference of 0.0 is correct per the equilibrium setup.

The problem is NOT the pitch reference itself, but:

1. **Initial condition bias:** hip_pitch_error of 0.45 rad at step 0
2. **Position authority insufficiency:** tau_position capped at 4.0 Nm cannot fully cancel tau_pitch
3. **Initial perturbation:** The robot starts in a non-equilibrium state

---

## 6. Height-Dependent Pitch Offset Analysis

### low_0p300 Geometry
- Hip pitch = 78.85 deg (very bent)
- Knee = 134.56 deg (near maximum bend)
- This is an extreme squat posture

### Physical Consideration
At such a low height with such bent joints, the robot may naturally want a slight forward pitch to maintain ground contact stability.

However, the equilibrium search explicitly set `equilibrium_pitch_x = 0.0`.

### Question
**Should low_0p300 have a non-zero equilibrium pitch?**

This would require re-running the equilibrium search with a pitch constraint.

---

## 7. Conclusion: Classification

### PITCH_REFERENCE_ZERO_BUT_LOW_HEIGHT_NEEDS_OFFSET
**Partially applies.**

Evidence:
1. Pitch reference IS zero (correct per equilibrium search)
2. low_0p300 has extreme geometry (hip=78.85°, knee=134.56°)
3. The robot persistently leans forward (pitch_x > 0 for 89% of steps)
4. Initial hip_pitch error of 0.45 rad creates persistent forward moment

### PITCH_INITIAL_CONDITION_BIAS
**CONFIRMED.**

Evidence:
1. hip_pitch_error_max = 0.45 rad at step 0
2. Error remains high throughout (never corrects below 0.45 rad)
3. This creates a steady forward pitch moment

### PITCH_REFERENCE_OK_BIAS_FROM_DYNAMICS
**CONFIRMED as secondary factor.**

Evidence:
1. Even with correct pitch reference, position authority is insufficient
2. tau_position cannot fully cancel tau_pitch
3. Net positive torque builds up → forward lean

---

## 8. Recommended Investigation

### Immediate: Re-run equilibrium search for low_0p300
The current equilibrium search found pitch_x = 0.0, but the robot immediately leans forward.

Possible reasons:
1. Equilibrium search converged to a local minimum
2. Contact model at low height creates different equilibrium
3. Initial state from keyframe doesn't match equilibrium

### Alternative: Increase position authority for low_0p300
If equilibrium is correct, the position authority (tau_position cap = 4.0 Nm) is too low.

### Test: Add small forward pitch offset for low_0p300
If physics requires forward pitch at low height, add it to equilibrium.

---

## 9. Summary

| Finding | Classification | Evidence |
|---------|----------------|----------|
| Pitch reference is 0 | ✅ Correct | Setup file, telemetry |
| low_0p300 needs pitch offset | ⚠️ Possible | Extreme geometry, persistent forward lean |
| Initial hip_pitch error causes bias | ✅ CONFIRMED | Error = 0.45 rad throughout |
| Position authority insufficient | ✅ CONFIRMED | tau_position clips at ±4.0 Nm |

**Root causes (in order of impact):**
1. **Initial hip_pitch error** - creates steady forward moment
2. **Position authority cap** - cannot fully cancel tau_pitch
3. **Possible height-dependent pitch offset** - low_0p300 may need non-zero equilibrium pitch