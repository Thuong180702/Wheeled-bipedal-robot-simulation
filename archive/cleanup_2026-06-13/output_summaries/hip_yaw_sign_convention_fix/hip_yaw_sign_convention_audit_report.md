# Hip-Yaw Sign Convention Audit Report

**Date:** 2026-06-05
**Phase:** Phase 1
**Classification:** `torque_formula_sign_wrong`

## Executive Summary

The hip-yaw sign convention error is caused by an **incorrect negation** in the PD control law. The "inverted axis" comment is wrong - the joint axes are NOT inverted in the MJCF model.

## Audit Results

| Component | Status | Details |
|-----------|--------|---------|
| Error Definition | CORRECT | `posture_error = q_ref - joint_pos` (standard) |
| Torque Formula | WRONG | Has incorrect negation: `tau = -(kp*error - kd*vel)` |
| Joint Axis Convention | NOT INVERTED | Standard axis - no inversion needed |
| Left/Right Mapping | CORRECT | Indices [1, 6] correctly mapped |
| Telemetry Diagnostic | CORRECT | Properly detects the sign bug |

## Technical Analysis

### Current Code (WRONG)
```python
# Line 250 in shape_posture_controller.py
tau_pd = -(self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx])
```

With kp=10, error=+0.1, vel=0:
- tau = -(10.0 * 0.1 - 2.0 * 0.0) = -1.0

### Telemetry Expectation
```python
sign_correct = error * tau >= 0  # torque opposes error
```

With error=+0.1, tau=-1.0:
- sign_correct = 0.1 * (-1.0) = -0.1 < 0 => FALSE (0%)

### Correct Code (FIX)
```python
tau_pd = self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx]
```

With kp=10, error=+0.1, vel=0:
- tau = 10.0 * 0.1 - 2.0 * 0.0 = 1.0

With error=+0.1, tau=+1.0:
- sign_correct = 0.1 * 1.0 = 0.1 >= 0 => TRUE (100%)

## Why the "Inverted Axis" Comment is Wrong

The comment claims "Hip-yaw joint axes are inverted in MJCF model". This is incorrect because:

1. The telemetry shows 0% sign correctness across ALL heights
2. If axes were inverted consistently, the sign would still be consistent
3. The negation produces wrong-direction torque, not correct torque for inverted axes

The negation was likely added based on a misunderstanding of the axis convention.

## Fix Location

**File:** `wheeled_biped/controllers/shape_posture_controller.py`
**Line:** 250
**Change:** Remove the negation

```python
# BEFORE (WRONG):
tau_pd = -(self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx])

# AFTER (CORRECT):
tau_pd = self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx]
```

## Impact of Fix

After the fix:
- `hip_yaw_torque_sign_correct_left` should be > 95%
- `hip_yaw_torque_sign_correct_right` should be > 95%
- `hip_yaw_abs_max` should decrease at all heights
- `divergence_rms` should decrease at boundary heights

## Test Expectation

The existing test `test_shape_posture_hip_yaw_torque_sign_remains_correct` in `test_step_e_hip_yaw_authority_fix.py` expects:
- `tau[1] > 0.0` for positive error (ref=0, pos=-0.1, error=+0.1)

This test is correct - it tests the EXPECTED behavior after the fix.

The test currently FAILS because the negation is wrong. After the fix, it should PASS.

## Confidence

**HIGH** - The analysis is based on:
1. Direct code inspection
2. Telemetry data showing 0% correctness
3. Mathematical verification of the formula
4. Test expectations that match the corrected behavior