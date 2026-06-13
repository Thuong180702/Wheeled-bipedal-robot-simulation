# Tau_pitch Telemetry Decomposition Audit

**Date:** 2026-06-08
**Profiles:** D2, F1b, G1a, G1b at low_0p300

---

## 1. Summary Comparison

| Metric | D2 | F1b | G1a | G1b |
|--------|----|----|----|----|
| pitch_x mean (deg) | +2.98 | +3.09 | +3.30 | +3.53 |
| pitch_x positive% | 89.0% | 82.6% | 84.0% | 82.8% |
| tau_pitch mean (Nm) | +2.60 | +2.70 | +2.88 | +3.08 |
| tau_pitch positive% | 89.2% | 82.8% | 84.2% | 83.0% |
| tau_position mean (Nm) | -2.61 | -2.36 | -2.22 | -2.16 |
| tau_position positive% | 6.6% | 17.0% | 18.0% | 19.4% |
| sagittal_pos_error mean (m) | +0.082 | +0.076 | +0.072 | +0.078 |
| sagittal_pos_error positive% | 93.2% | 82.8% | 81.8% | 80.4% |

---

## 2. Tau_pitch Decomposition

### D2 Tau_pitch Components

| Component | Mean | Positive% | Note |
|-----------|------|-----------|------|
| tau_pitch | +2.60 Nm | 89.2% | kp_pitch × pitch_error |
| tau_pitch_rate | +0.09 Nm | 42.0% | kd_pitch × pitch_rate |
| tau_position | -2.61 Nm | 6.6% | bounded at ±4.0 Nm |
| tau_sagittal_velocity | -0.15 Nm | 50.4% | damping term |
| tau_wheel_velocity_left | +0.10 Nm | 48.0% | per-wheel damping |
| tau_wheel_velocity_right | +0.10 Nm | 48.2% | per-wheel damping |
| tau_support_velocity | ~0 Nm | 0% | disabled in D2 |

### D2 Final Wheel Torque

| Component | Mean | Positive% |
|-----------|------|-----------|
| tau_wheel_total_raw_left | -0.02 Nm | 59.2% |
| tau_wheel_total_raw_right | -0.02 Nm | 59.2% |
| tau_total_unclipped | -0.02 Nm | 59.2% |

**Key observation:** Final wheel torque is near zero, but 59% positive. This means the controller is trying to push slightly forward on average, which is a restoring action (counteracting backward lean tendency).

---

## 3. Pitch Error Analysis

### Pitch Reference
- pitch_x_ref_rad = 0.0 throughout all profiles
- equilibrium_pitch_x from setup = 0.0
- **No pitch reference offset exists**

### Pitch Error vs Raw Pitch
- pitch_x ≈ pitch_x_error (since ref=0)
- pitch_x positive% matches tau_pitch positive% (both ~89% for D2)
- **tau_pitch directly tracks pitch_x**

### Pitch Evolution (D2)

| Step Range | pitch_x mean (deg) | tau_pitch mean (Nm) |
|------------|-------------------|---------------------|
| 0-50 | +0.85 | +0.75 |
| 50-100 | +5.16 | +4.52 |
| 100-200 | +5.10 | +4.45 |
| 200-300 | +2.67 | +2.32 |
| 300-400 | +0.32 | +0.28 |
| 400-500 | +3.79 | +3.31 |

**Pattern:** Pitch grows from 0 to ~5 deg by step 50-100, then gradually recovers toward 0, then grows again.

---

## 4. Position Error Analysis

### Sagittal Position Error Origin
```
sagittal_position_error_m = project_sagittal_displacement(
    origin_xy = support_center_eq_xy,  # (-0.0345, ...) at equilibrium
    sagittal_axis_xy = sagittal_axis_xy_initial,
    current_xy = support_center_ctrl_xy,
)
```

### Position Error Correlation with Pitch
- corr(pitch_x, sagittal_pos_error) = 0.9353
- corr(tau_pitch, sagittal_pos_error) = 0.9345
- **Both pitch and position error are driven by the same underlying instability**

### Event Order (D2 first 30 steps)
```
step | pitch_x  | tau_pitch | sagittal_pos | event
-----|----------|-----------|--------------|-------
0    | -0.0000  | +0.00     | +0.0000      | equilibrium
1    | +0.0000  | +0.01     | +0.0000      | pitch becomes slightly positive
2    | +0.0002  | +0.00     | -0.0002      | small oscillation
3    | -0.0006  | -0.04     | -0.0005      | 
...
9    | +0.0002  | +0.01     | +0.0005      | drift begins
10   | +0.0004  | +0.02     | +0.0009      |
...
20   | +0.0059  | +0.30     | +0.0077      | clear forward drift
```

**Event order:** pitch → tau_pitch → sagittal_position_error (all become positive together)

---

## 5. Position Saturation Analysis

### D2 Position Authority

| Parameter | Value |
|-----------|-------|
| effective_max_position_tau | 4.0 Nm |
| tau_position_saturation_flag true% | 35.4% |
| Position saturation starts at step | 65 |
| Position saturation ends at step | 241 |

### When Position Saturates (step 65-241)
- tau_pitch grows to ~2.5-5.1 Nm
- tau_position is clamped at -4.0 Nm (opposing tau_pitch)
- tau_common = tau_pitch + tau_position ≈ small residual

### Key Insight
**Position authority is insufficient to cancel tau_pitch at low height.**
- At step 50-80: tau_pitch ≈ +2.5-5.1 Nm, tau_position ≈ -4.0 Nm (clipped)
- Net: tau_pitch + tau_position ≈ -0.5 to +1.1 Nm
- This allows persistent forward lean to continue

---

## 6. Tau_pitch vs tau_position Cancellation

### D2 Steps 50-80 Detailed

| step | pitch_x | tau_pitch | tau_position | net |
|------|---------|-----------|-------------|-----|
| 50 | +0.049 | +2.48 | -2.48 | ~0 |
| 60 | +0.070 | +3.50 | -3.47 | +0.03 |
| 65 | +0.080 | +4.02 | -3.97 | +0.05 |
| 66 | +0.082 | +4.13 | -4.00 | +0.13 | ← position saturates
| 70 | +0.090 | +4.51 | -4.00 | +0.51 |
| 75 | +0.097 | +4.88 | -4.00 | +0.88 |
| 80 | +0.102 | +5.11 | -4.00 | +1.11 |

**After step 65:** Position cap prevents full cancellation, net torque becomes positive (forward).

---

## 7. Correlation Matrix

| | pitch_x | tau_pitch | sagittal_pos_error | hip_yaw_comp |
|---|---------|-----------|-------------------|--------------|
| pitch_x | 1.0000 | 1.0000 | 0.9353 | 0.9309 |
| tau_pitch | 1.0000 | 1.0000 | 0.9345 | 0.9309 |
| sagittal_pos_error | 0.9353 | 0.9345 | 1.0000 | 0.9999 |
| hip_yaw_comp | 0.9309 | 0.9309 | 0.9999 | 1.0000 |

**All correlations > 0.93: pitch, position error, and hip-yaw compensation are all driven by the same underlying instability.**

---

## 8. Profile Evolution Analysis

### Trend Across Profiles

| Profile | tau_pitch mean | tau_position mean | Position Sat% | Outside ±0.15 |
|---------|---------------|-------------------|---------------|---------------|
| D2 | +2.60 | -2.61 | 35.4% | 19.2% |
| F1b | +2.70 | -2.36 | ? | 16.2% |
| G1a | +2.88 | -2.22 | ? | 13.4% |
| G1b | +3.08 | -2.16 | ? | 26.8% |

**Trend:** 
- tau_pitch increases from D2→G1b
- tau_position magnitude decreases (less cancellation)
- G1b shows worst outside-band behavior despite reduced positive%

---

## 9. Root Cause Hypothesis

**tau_pitch positive bias is a CORRECT response to real forward lean.**

The robot leans forward at low_0p300 because:
1. **tau_position cannot fully cancel tau_pitch** - position cap is insufficient
2. **Net positive torque builds up** → forward lean increases
3. **tau_pitch increases further** → forward lean increases more
4. **Vicious cycle** until position error grows enough to trigger recenter (F1/F2) or bias cancel (G1)

**Why does the robot lean forward at low height?**
- Unknown initial perturbation at step 0 (hip_pitch_error_max = 0.45 rad!)
- Low-height geometry may have natural forward instability tendency
- Control coupling from other controllers

---

## 10. Conclusion

### tau_pitch computation is NOT the problem
- Formula: `tau_pitch = kp_pitch * pitch_error` (correct)
- Reference: `pitch_x_ref = 0.0` (correct per setup)
- Sign: positive pitch → positive torque (correct)

### tau_pitch is responding to real forward lean
- pitch_x is persistently positive (89% of steps)
- tau_pitch is the correct response to this lean
- G1b stronger bias cancellation reduces positive% slightly but worsens outside-band

### The root cause is position authority insufficiency
- tau_position cap of 4.0 Nm is too low for tau_pitch up to 5.5 Nm
- Position cannot cancel pitch, leading to net forward torque
- This drives the forward lean-drift-pitch cycle

### Downstream bias cancellation is a patch, not a fix
- G1b reduces positive% slightly (80.4% vs 89.2%)
- But it increases outside-band behavior (26.8% vs 19.2%)
- The source (insufficient position authority) must be addressed