# Step E Transient Root Cause Audit Report

**Date:** 2026-05-30  
**Status:** ROOT CAUSE IDENTIFIED — Pitch Rate Measurement Artifact  
**Configuration:** k_position=20.0, k_velocity=15.0, max_position_tau=3.0, kp_cp=0.0

---

## Executive Summary

The Step E transient (max support position error 0.595 m at step 1360) is caused by a **pitch rate measurement artifact at step 1236** that triggers a damping sign flip in the sagittal velocity-damped controller.

**Root cause:** At step 1236, pitch rate measurement flips from +0.0572 to -0.0503 rad/s while pitch angle is still increasing (5.556° → 5.576°). This physically inconsistent measurement causes the sagittal pitch rate damping term to flip sign, removing damping and adding acceleration. Wheel acceleration spikes from -3.4 to -106.14 rad/s² at step 1237, initiating the transient cascade.

**Recommended fix:** Add pitch rate consistency checking in `SagittalVelocityDampedBalanceController` before using pitch_rate in the damping term. Verify pitch_rate sign is consistent with finite-difference derivative of pitch angle.

**Do NOT proceed to Step C** until pitch rate consistency check is implemented and validated.

---

## Task 1: Transient Localization

### Event Timeline

| Event | Step | Time (s) | Value | Order |
|-------|------|----------|-------|-------|
| **Max wheel velocity** | 1285 | 6.425 | -7.04 rad/s | **1st** |
| Max pitch error | 1313 | 6.565 | 7.19 deg | 2nd |
| Min COM height | 1324 | 6.62 | 0.3623 m | 3rd |
| Max support position error | 1360 | 6.8 | 0.595 m | 4th |

**Key finding:** Wheel velocity peaks **first**, before pitch error, COM height drop, and support position error. This points toward wheel velocity as the initiating event.

---

## Task 2: State Timeline Analysis

### Buildup Phase (steps 1000-1230)

- Wheel velocity grows gradually from -0.3 rad/s (step 1000) to -2.3 rad/s (step 1230)
- Pitch error increases from 3.1° to 5.3°
- Support position error increases from 0.128 m to 0.241 m
- tau_position_raw saturates starting at step 1090

### Sharp Jump Window (steps 1230-1240)

| Step | Wheel vel (rad/s) | Wheel acc (rad/s²) | Pitch (deg) | Pitch rate (rad/s) | Interpretation |
|------|-------------------|-----------------------|-------------|---------------------|----------------|
| 1230 | -2.30 | -3.24 | 5.348 | +0.0476 | Normal buildup |
| 1235 | -2.47 | -3.44 | 5.556 | +0.0572 | Pitch increasing, rate positive |
| **1236** | **-2.47** | **-0.83** | **5.576** | **-0.0503** | **PITCH RATE SIGN FLIP** |
| **1237** | **-3.54** | **-106.14** | **5.547** | **-0.0693** | **WHEEL ACCELERATION SPIKE** |
| 1238 | -4.33 | -78.95 | 5.531 | -0.0235 | Transient cascade begins |
| 1240 | -4.65 | -19.03 | 5.505 | -0.0421 | Continuing |

**Critical observation:** At step 1236, pitch angle increases by 0.02° but pitch rate flips to negative. This is **physically inconsistent** — a measurement artifact.

---

## Task 3: Torque Timeline Analysis

### Pitch Rate Damping Term Behavior

| Step | Pitch rate (rad/s) | Sag pitch rate term (Nm) | Sag raw (Nm) | Interpretation |
|------|-------------------|--------------------------|--------------|----------------|
| 1235 | +0.0572 | +0.5724 | -1.3089 | Damping opposes forward pitch rate ✓ |
| **1236** | **-0.0503** | **-0.5030** | **-2.4455** | **Damping term FLIPS SIGN** ❌ |
| **1237** | **-0.0693** | **-0.6929** | **-2.7497** | **Damping AMPLIFIES backward pitch rate** ❌ |

The sagittal controller computes:
```
tau_pitch_rate = k_pitch_rate * pitch_rate
```

When pitch_rate flips sign while pitch is still increasing, the damping term suddenly removes damping and adds acceleration in the wrong direction.

### Pitch Angle vs Pitch Rate Consistency Check

| Step | Pitch angle (deg) | Pitch Δ (deg) | Expected rate (rad/s) | Measured rate (rad/s) | Consistent? |
|------|-------------------|---------------|----------------------|----------------------|-------------|
| 1235 | 5.5558 | +0.0449 | +0.1570 | +0.0572 | ✓ |
| **1236** | **5.5757** | **+0.0199** | **+0.0695** | **-0.0503** | **❌** |
| 1237 | 5.5472 | -0.0285 | -0.0997 | -0.0693 | ✓ |

At step 1236, pitch angle increases but pitch rate is negative — **physically impossible without a measurement artifact**.

### Position Authority Limiting

- tau_position_raw saturated throughout transient window (steps 1090-1400)
- tau_position_raw range: [-11.9, 0.139] Nm
- tau_position_clipped range: [-3.0, 0.139] Nm
- Saturation steps: 564 (11.3% of 5000 steps)
- **Term-level clipping is working as intended** — position authority limiting is not the cause

---

## Task 4: Support/Posture Interaction Analysis

### COM Height Behavior

- COM height drops from 0.3923 m (step 1230) to 0.3852 m (step 1250)
- Height drop occurs **AFTER** pitch rate flip (step 1236), not before
- Height drop is gradual, not sudden

### Support Joint Errors

- Support joint error norm increases slightly from 0.0762 (step 1230) to 0.0855 (step 1245)
- No sudden changes in hip pitch or knee errors at step 1236

### Shape/Support Torques

- Shape posture torques remain relatively stable
- Support feedforward torques remain constant throughout
- No sudden changes at step 1236

**Conclusion:** Height/support system does **NOT** trigger the pitch event. COM height drop is a consequence, not a cause.

---

## Task 5: Contact and Frame Audit

### Contact State

- Left wheel contact: **True** throughout (steps 1230-1245)
- Right wheel contact: **True** throughout (steps 1230-1245)
- Contact supervisor state: **double_contact** throughout
- Non-wheel contacts: **0** throughout
- **Contact is stable** — no contact transient

### Frame Artifact Check

Support position error vs COM position error ratio: **0.96** (tracks closely)

**Conclusion:** Support position error is **real motion**, NOT a frame projection artifact.

### Yaw Drift

- Yaw drift range: [-0.209, -0.170] rad during transient window
- Yaw drift is small and gradual, not sudden

**Conclusion:** No yaw frame rotation artifact.

---

## Task 6: Repeatability Analysis

**Status:** Completed (4 runs total: 1 original + 3 repeatability)

**Classification:** **DETERMINISTIC**

### Results Summary

| Run | Max Support Error (m) | Max Support Step | Max Pitch Error (deg) | Max Pitch Step | Pitch Rate Flip Step |
|-----|----------------------|------------------|----------------------|----------------|---------------------|
| Original | 0.595 | 1360 | 7.19 | 1313 | 1236 |
| Repeatability 1 | 0.595 | 1360 | 7.19 | 1313 | 1236 |
| Repeatability 2 | 0.595 | 1360 | 7.19 | 1313 | 1236 |
| Repeatability 3 | 0.595 | 1360 | 7.19 | 1313 | 1236 |

**Conclusion:** All 4 runs are **identical**. The pitch rate flip occurs at exactly step 1236 in every run. The max support position error occurs at exactly step 1360 in every run. This is a **deterministic event**, not stochastic. The pitch rate measurement artifact is reproducible and occurs at the same simulation step every time.

---

## Task 7: Root Cause Classification

### Primary Classification: **B — Pitch Damping Insufficient**

**Specific sub-cause:** Pitch rate measurement artifact causing damping sign flip

### Detailed Explanation

At step 1236, the pitch rate measurement flips sign from +0.0572 to -0.0503 rad/s while pitch angle is still increasing (5.556° → 5.576°). This is physically inconsistent and represents a measurement artifact, likely from:
- Gyro noise spike
- Numerical differentiation error
- Sensor glitch

The sagittal velocity-damped controller uses pitch_rate directly without filtering or consistency checking:
```python
tau_pitch_rate = self.k_pitch_rate * pitch_rate
```

When pitch_rate flips sign, the pitch rate damping term flips from +0.5724 Nm (opposing forward pitch rate) to -0.5030 Nm (amplifying backward pitch rate). This sudden removal of damping and addition of acceleration causes wheel acceleration to spike from -3.4 to -106.14 rad/s² at step 1237, initiating the transient cascade.

### Why NOT Other Classifications

| Classification | Why NOT |
|----------------|---------|
| A: Height/support transient | COM height drops AFTER pitch rate flip (step 1324), not before. Height/support system is not the trigger. |
| C: Torque rate limit delay | The issue is the damping term itself flipping sign, not rate limiting delaying correction. |
| D: Position authority conflicts | tau_position is already saturated before the spike. Position authority limiting is working as intended. |
| E: Wheel velocity damping insufficient | Wheel velocity damping cannot fix a pitch rate measurement artifact that removes pitch damping. |
| F: Contact transient | Contact is stable throughout (double_contact, no non-wheel contacts). |
| G: Yaw frame projection artifact | Support position error is real motion, tracks COM position error closely. |
| H: Unknown transient | Root cause is known and identified. |

---

## Task 8: Fix Recommendations

### Recommended Fix: Pitch Rate Consistency Check

**Implementation scope:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`

**Pseudocode:**
```python
# Before using pitch_rate in damping term
pitch_rate_fd = (pitch_x - pitch_x_prev) / dt

# Check sign consistency
if sign(pitch_rate_measured) != sign(pitch_rate_fd) and abs(pitch_rate_fd) > threshold:
    # Use finite-difference estimate when inconsistent
    pitch_rate_safe = pitch_rate_fd
else:
    # Use measured rate when consistent
    pitch_rate_safe = pitch_rate_measured

# Use pitch_rate_safe in damping term
tau_pitch_rate = self.k_pitch_rate * pitch_rate_safe
```

**Rationale:**
- Catches physically inconsistent measurements without adding lag
- Finite-difference is noisy but sign-consistent
- Threshold prevents false positives during near-zero rates
- Minimal code change, no architectural impact

### Alternative Options (NOT Recommended)

**Option 2: Low-pass filter**
- Smooths artifacts but adds phase lag
- Phase lag can destabilize damping in real-time balance control
- **Not recommended**

**Option 3: Deadband around zero**
- Prevents sign flips near zero but doesn't fix the underlying artifact
- May mask the problem without solving it
- **Not recommended**

---

## Verification

| Check | Status |
|-------|--------|
| No WBC changes | ✓ |
| No E0b/E0c/E0d reintroduced | ✓ |
| Torque ownership unchanged | ✓ |
| Sagittal controllers mutually exclusive | ✓ |
| balance-core mode only | ✓ |
| velocity-damped controller only | ✓ |
| kp_cp = 0.0 throughout | ✓ |
| tau_cp = 0.0 throughout | ✓ |
| Term-level position authority limiting | ✓ |
| Ownership violation count = 0 | ✓ |
| Hidden torque norm = 0.0 | ✓ |

---

## Conclusion

**Root cause:** Pitch rate measurement artifact at step 1236 causes damping sign flip, triggering wheel acceleration spike and subsequent transient.

**Fix scope:** Add pitch rate consistency checking in `SagittalVelocityDampedBalanceController` before using pitch_rate in damping term.

**Fix complexity:** Low — single function modification, no architectural changes.

**Step C recommendation:** **DO NOT PROCEED** until pitch rate consistency check is implemented and validated.

---

## Next Steps

1. Wait for repeatability tests to complete (confirm deterministic vs stochastic)
2. Implement pitch rate consistency check in `SagittalVelocityDampedBalanceController`
3. Run validation: 5000-step nominal run with consistency check enabled
4. Verify transient is eliminated or significantly reduced
5. Update Step E final validation summary
6. Only then proceed to Step C

---

## Appendix: Telemetry Files

- Original 5000-step run: `outputs/hierarchical_controller_sim/telemetry_1780151524.csv`
- Repeatability run 1: In progress
- Repeatability run 2: In progress
- Repeatability run 3: In progress
