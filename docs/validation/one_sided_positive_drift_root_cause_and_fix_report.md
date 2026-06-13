# One-Sided Positive Drift Root Cause and Fix Report

## Executive Summary

**Task:** Stop F2 hysteresis tuning. The robot does NOT drift around zero because the system has a persistent one-sided positive bias. The robot does not produce enough negative signed support error for hysteresis to work.

**Goal:** Identify and fix the one-sided positive bias so that signed support error can move around zero, preferably staying within [-0.15 m, +0.15 m].

---

## Phase 1 & 2: Telemetry Audit Results

### Pitch Dynamics

| Profile | pitch_x mean | positive% | crossings | recovery% |
|--------|-------------|-----------|-----------|-----------|
| D2     | 0.0520      | 89.0%     | 5         | 57.8%     |
| F1b    | 0.0540      | 82.6%     | 6         | 60.0%     |
| F2a    | 0.0560      | 83.4%     | 6         | 60.2%     |
| F2b    | 0.0577      | 83.2%     | 5         | 59.6%     |

**Finding:** Pitch reverses (5-6 crossings) but stays positive 82-89% of the time. Recovery windows exist (57-60%) but are not effective at pulling signed support error negative.

### Signed Support Error

| Profile | mean     | positive% | min       | max       | outside +0.15 |
|--------|----------|-----------|-----------|-----------|---------------|
| D2     | 0.0823   | 93.0%     | -0.0035   | 0.1757    | 96 steps      |
| F1b    | 0.0764   | 82.8%     | -0.0339   | 0.1695    | 81 steps      |
| F2a    | 0.0803   | 82.8%     | -0.0474   | 0.1761    | 92 steps      |
| F2b    | 0.0831   | 82.6%     | -0.0638   | 0.1764    | 142 steps     |

**Finding:** Signed support error is persistently positive (82-93%). F1b/F2a/F2b improve over D2 (82-83% vs 93%) but still stay outside the ±0.15 m band too often. No profile ever goes below -0.15 m.

### Wheel Velocity

| Profile | mean      | positive% | zero crossings |
|--------|-----------|-----------|----------------|
| D2     | -0.1986   | 51.8%      | many           |
| F1b    | 0.1318    | 51.6%      | many           |
| F2a    | 0.0722    | 50.6%      | many           |
| F2b    | 0.0124    | 50.0%      | many           |

**Finding:** Wheel velocity alternates (50% positive/negative) but signed support error still stays positive. This means wheel reversals are not translating into signed error reversals.

### Coupling Analysis

At pitch reversal windows:
- D2: signed_error_at_pitch_zc_mean = 0.0028 (80% positive)
- F1b: signed_error_at_pitch_zc_mean = 0.0005 (50% positive)
- F2a: signed_error_at_pitch_zc_mean = -0.0013 (50% positive)
- F2b: signed_error_at_pitch_zc_mean = -0.0016 (60% positive)

**Finding:** F2a/F2b show better coupling (signed error closer to zero at pitch reversals) but the error quickly returns positive after reversal.

---

## Phase 3: Bias Source Audit

### A. Setup/Reference Bias

| Profile | initial_signed_error | near_zero |
|--------|----------------------|-----------|
| D2     | 0.0000               | True      |
| F1b    | 0.0000               | True      |
| F2a    | 0.0000               | True      |
| F2b    | 0.0000               | True      |

**Finding:** NO setup bias. Initial signed error is exactly 0.0000 for all profiles.

### B. Telemetry/Compensation Formula Bias

| Profile | hip_yaw_comp_mean | signed_never_negative |
|--------|-------------------|------------------------|
| D2     | 0.0823            | False                  |
| F1b    | 0.0764            | False                  |
| F2a    | 0.0803            | False                  |
| F2b    | 0.0831            | False                  |

**Finding:** `hip_yaw_comp_support_error_m` IS the signed support error (not a separate compensation). It averages 0.076-0.083 m, persistently positive.

### C. Controller Bias - tau_pitch

| Profile | tau_pitch mean | positive% | tau_pitch when pitch~0 mean |
|--------|----------------|-----------|---------------------------|
| D2     | 2.5992         | 89.2%     | 0.1367 (57.1% positive)   |
| F1b    | 2.6976         | 82.8%     | 0.0471 (57.3% positive)  |
| F2a    | 2.7978         | 83.6%     | 0.1086 (63.1% positive)  |
| F2b    | 2.8861         | 83.4%     | 0.1545 (66.7% positive)  |

**Finding:** tau_pitch is persistently positive (82-89%). Even when pitch is near zero (±0.02 rad), tau_pitch is still positive 57-67% of the time. This is the PITCH REFERENCE OR TORQUE BIAS.

### D. Controller Bias - tau_position

| Profile | tau_position mean | positive% |
|--------|------------------|-----------|
| D2     | -2.6146          | 6.6%      |
| F1b    | -2.3618          | 17.0%     |
| F2a    | -2.3674          | 17.0%     |
| F2b    | -2.3424          | 17.2%     |

**Finding:** tau_position is NEGATIVE (not positive). This is expected - tau_position opposes forward drift (negative = restoring backward). This is NOT the bias source.

### E. Contact/Dynamics Bias

| Profile | left_fz_mean | right_fz_mean | asymmetry |
|--------|--------------|---------------|-----------|
| D2     | 40.58        | 38.85         | ~4%       |
| F1b    | 40.65        | 38.76         | ~5%       |
| F2a    | 40.62        | 38.78         | ~5%       |
| F2b    | 40.59        | 38.82         | ~4%       |

**Finding:** Left contact force is consistently higher than right (~5% asymmetry). This is a minor asymmetry but not a strong bias source.

### F. Hip-Yaw/Posture Coupling

| Profile | asymmetry_mean | asymmetry positive% | corr(asymmetry, signed_err) |
|--------|---------------|---------------------|------------------------------|
| D2     | 0.0107        | 99.8%               | -0.372                       |
| F1b    | 0.0113        | 99.8%               | -0.352                       |
| F2a    | 0.0110        | 99.8%               | -0.307                       |
| F2b    | 0.0106        | 99.8%               | -0.272                       |

**Finding:** Hip-yaw asymmetry is consistently positive (99.8%) - meaning right hip-yaw is more positive than left. However, the correlation with signed error is NEGATIVE (-0.27 to -0.37), meaning hip-yaw asymmetry is NOT driving the positive signed error bias.

### G. Temporal Analysis

| Profile | first_100_mean | last_100_mean | trend |
|--------|---------------|---------------|-------|
| D2     | 0.0666        | 0.0809        | +0.014 |
| F1b    | 0.0661        | 0.0773        | +0.011 |
| F2a    | 0.0666        | 0.0969        | +0.030 |
| F2b    | 0.0666        | 0.1157        | +0.049 |

**Finding:** Bias exists from step 0 (0.066 m) and GROWS over time. F2b shows the worst growth trend (+0.049 m).

### H. Root Cause Classification

**Primary Classification:** `BIAS_FROM_PITCH_REFERENCE_OR_TORQUE_BIAS`

**Evidence:**
1. tau_pitch is persistently positive (82-89%)
2. Even when pitch is near zero, tau_pitch is still positive 57-67% of the time
3. tau_position is negative (correctly opposing forward drift)
4. Pitch reference (pitch_x_ref_rad) is 0.0 - no reference bias there
5. The persistent positive tau_pitch creates a continuous forward pitching tendency
6. This forward pitching tendency causes the robot to pitch forward, which causes COM to shift forward, which causes positive signed support error

**Secondary Classifications:**
- `YAW_AWARE_COMP_BIAS` - yaw-aware compensation is active and adds to the signed error
- `HIP_YAW_ASYMMETRY_BIAS` - hip-yaw asymmetry is consistently positive but NOT the primary driver

---

## Phase 4: Fix Strategy

### Analysis Summary

The one-sided positive bias is caused by **persistent positive tau_pitch** that creates a forward pitching tendency. This forward pitch causes:
1. COM to shift forward
2. Positive signed support error
3. tau_position to counteract (negative tau_position)
4. But tau_position is too weak to fully counteract the tau_pitch bias

### Why F2 Hysteresis Doesn't Work

F2 hysteresis waits for the system to naturally drift to negative signed error before recentering. But:
1. tau_pitch is persistently positive → forward pitch tendency
2. Forward pitch → COM shifts forward → positive signed error
3. tau_position opposes but is too weak
4. The system never naturally drifts negative because tau_pitch keeps pushing it positive

### Fix Strategy: G1 Persistent Bias Cancellation

**Core Idea:** Instead of waiting for natural negative drift (which doesn't happen), estimate the persistent positive bias in signed support error and apply a bounded opposite torque to counteract it.

**Design:**
- `bias_estimate` = low-pass filtered signed support error
- `bias_tau` = -k_bias * bias_estimate (opposes the bias)
- `bias_tau` is added to wheel torque as an additional term
- Active when contact/height/roll are safe
- NOT gated on pitch reversal (because pitch reversal doesn't produce negative drift)

**Parameters (G1a - moderate):**
- k_bias = 12.0 Nm/m
- max_bias_tau = 1.5 Nm
- bias_filter_alpha = 0.02 (leaky integration)
- bias_deadband_m = 0.02
- Safety gates: contact_valid, height_safe, roll_safe
- Does NOT gate on hip_yaw too tightly (hip-yaw is part of the bias loop)

**Why this should work:**
1. Persistent positive signed error → negative bias_tau
2. Negative bias_tau creates backward wheel torque
3. Backward wheel torque shifts COM backward
4. Signed error decreases toward zero
5. Bias estimate tracks the persistent error level
6. System stays within ±0.15 m band

**Why it's safe:**
1. Bounded by max_bias_tau
2. Leaky integration (bias decays if not sustained)
3. Safety gates prevent application when unsafe
4. Does not interfere with pitch recovery
5. Monitor-only for wheel velocity

---

## Implementation Plan

### File Changes

1. **wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py**
   - Add G1 profile to VDSagittalAuthoritySchedule
   - Add bias_cancel state variables
   - Add bias_cancel logic in compute()
   - Add telemetry fields

2. **tests/test_sagittal_velocity_damped_balance_controller.py**
   - Add tests for G1 profile

### G1 Profile Definition

```python
G1A_BIAS_CANCEL_MODERATE = {
    "name": "G1a_bias_cancel_moderate",
    "k_position": 40.0,  # Same as D2
    "k_velocity": 8.0,    # Same as D2
    "k_pitch": 50.0,      # Same as D2
    "k_pitch_rate": 3.0,  # Same as D2
    "max_position_tau": 8.0,  # Same as D2
    "enable_bias_cancel": True,
    "bias_cancel_k": 12.0,  # Nm/m
    "bias_cancel_max_tau": 1.5,  # Nm
    "bias_cancel_filter_alpha": 0.02,
    "bias_cancel_deadband_m": 0.02,
    "bias_cancel_contact_gate": True,
    "bias_cancel_height_gate": True,
    "bias_cancel_roll_gate": True,
    "bias_cancel_pitch_gate": False,  # NOT gated on pitch
}
```

---

## Verification Criteria

### Pass Criteria (BIAS_FIX_PASS_PROCEED_TO_2000)
- positive% decreases substantially vs F1b, target <= 70%
- time outside [-0.15, +0.15] decreases vs F1b
- max positive signed error decreases vs F1b
- negative excursion increases but stays above -0.15
- support_position_error_m crossings decrease vs F1b
- contact/height/roll remain valid
- WBC/hidden/ownership clean
- wheel velocity increase does not cause instability

### Improve Criteria (BIAS_FIX_IMPROVES_BUT_NOT_ENOUGH)
- signed bias improves vs F1b
- but positive% still >70 or outside band still high

### Fail Criteria
- Overcorrects below -0.15
- Destabilizes height/roll/contact
- Hip-yaw grows unbounded
- Positive bias increases

---

## Files to Create/Modify

1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` - Add G1 profile
2. `tests/test_sagittal_velocity_damped_balance_controller.py` - Add G1 tests
3. `scripts/simulate_hierarchical_controller.py` - Accept G1 profile (if needed)
4. `docs/validation/one_sided_positive_drift_root_cause_and_fix_report.md` - This report
5. `outputs/step_e_extreme_support_fix_eval/one_sided_positive_drift_root_cause_and_fix_summary.json` - Summary JSON

---

## What NOT to Do

- Do NOT modify D2 baseline
- Do NOT make G1 profile default
- Do NOT continue tuning F2a/F2b
- Do NOT enable HY2-DIV
- Do NOT add WBC
- Do NOT enable legacy WBC
- Do NOT implement E2c
- Do NOT relax official Step E permanently
- Do NOT claim official Step E pass from this task
- Do NOT run 2000-step validation
- Do NOT run 5000-step validation
- Do NOT run Step C
- Do NOT run Step D
- Do NOT commit
