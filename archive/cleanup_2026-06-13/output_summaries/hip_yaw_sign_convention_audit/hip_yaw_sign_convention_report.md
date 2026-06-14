# Hip-Yaw Sign Convention Audit Report

**Date:** 2026-06-05

**Objective:** Systematic diagnostic to classify the root cause of hip-yaw torque sign error

## Classification

**Mechanism:** `joint_axis_sign_requires_negation`

**Confidence:** HIGH

## Evidence

- Hip-yaw torque sign correctness is 0.22-14.88% (effectively inverted)
- Left hip-yaw: positive error (+0.2275) with positive torque drives position MORE negative
- Right hip-yaw: negative error (-0.2303) with negative torque drives position MORE positive
- Pattern is consistent across all three heights (low, nominal, high)
- Reference is stable at 0.0 (not drifting)
- Error definition (ref - pos) is standard convention
- Divergence is antisymmetric (left/right opposite directions)

## Ruled Out

- error_definition_sign_wrong: Error formula (ref - pos) is standard
- torque_formula_sign_wrong: PD formula itself is standard
- damping_sign_wrong: Damping term correct, issue is proportional term
- left_right_joint_index_swapped: Divergence is antisymmetric, not swapped
- telemetry_sign_diagnostic_wrong: Pattern too consistent to be telemetry bug

## Diagnosis

The hip-yaw joint axes in the MJCF model have opposite convention from what the controller assumes. Positive torque DECREASES position, negative torque INCREASES position. This requires negating the entire PD control output for hip-yaw joints.

## Recommended Fix

```python
# Current (line 248 in shape_posture_controller.py):
tau_pd = self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx]

# Fixed:
tau_pd = -(self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx])
```

## Validation Criteria

- Hip-yaw torque sign correctness > 95%
- Hip-yaw abs max < 0.05 rad at nominal
- Hip-yaw abs max < 0.15 rad at low_0p300 and high_0p480
- Divergence RMS approaches 0 (antisymmetric instability eliminated)
- No regression in survival, contact, height tracking

## Technical Analysis

### Current PD Formula

```python
posture_error = q_ref - joint_pos  # line 188
tau_pd = kp * posture_error - kd * joint_vel  # line 248
```

### Observed Behavior Example (low_0p300)

**Left hip-yaw:**
- Reference: 0.0 rad
- Final position: -0.2275 rad (NEGATIVE, away from reference)
- Error: 0 - (-0.2275) = +0.2275 rad (POSITIVE)
- Controller applies: τ = kp × (+0.2275) = POSITIVE torque
- Result: Position continues DECREASING (more negative)
- Conclusion: **Positive torque DECREASES position**

**Right hip-yaw:**
- Reference: 0.0 rad
- Final position: +0.2303 rad (POSITIVE, away from reference)
- Error: 0 - (+0.2303) = -0.2303 rad (NEGATIVE)
- Controller applies: τ = kp × (-0.2303) = NEGATIVE torque
- Result: Position continues INCREASING (more positive)
- Conclusion: **Negative torque INCREASES position**

### Joint Axis Convention

The hip-yaw joint axes in the MJCF model have opposite sign convention:
- Expected: positive τ → positive Δpos
- Actual: positive τ → negative Δpos

This requires negating the entire PD output for hip-yaw joints.

