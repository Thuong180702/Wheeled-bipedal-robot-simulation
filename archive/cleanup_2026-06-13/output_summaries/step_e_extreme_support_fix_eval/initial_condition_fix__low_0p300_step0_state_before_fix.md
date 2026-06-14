# Low 0p300 Step-0 State (Before Fix)

## Summary

| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| hip_pitch_error_max | 0.4500 rad (25.79 deg) | 0.05 rad (2.86 deg) | FAIL |
| knee_error_max | 0.6000 rad (34.38 deg) | 0.05 rad (2.86 deg) | FAIL |

## Root Cause

The simulation initialization correctly sets actual joint positions from the setup file:
- hip_pitch_ref = 1.3761 rad
- knee_ref = 2.3484 rad

BUT the target_joint_pos comes from posture_regularizer.height_targets which has:
- hip_pitch at h=0.40 = 0.9261 rad (NOT 1.3761 rad)

This causes a ~0.45 rad error in hip_pitch.

## Joint Details

| Joint | Actual (rad) | Target (rad) | Error (rad) | Error (deg) |
|-------|--------------|--------------|-------------|-------------|
| l_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.00 | PASS
| l_hip_yaw | 0.0000 | -0.0007 | -0.0007 | 0.04 | PASS
| l_hip_pitch | 1.3761 | 0.9261 | -0.4500 | 25.79 | FAIL
| l_knee | 2.3484 | 1.7484 | -0.6000 | 34.38 | FAIL
| l_wheel | 0.0000 | 0.0000 | 0.0000 | 0.00 | PASS
| r_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.00 | PASS
| r_hip_yaw | 0.0000 | 0.0009 | 0.0009 | 0.05 | PASS
| r_hip_pitch | 1.3761 | 0.9261 | -0.4500 | 25.79 | FAIL
| r_knee | 2.3484 | 1.7484 | -0.6000 | 34.38 | FAIL
| r_wheel | 0.0000 | 0.0000 | 0.0000 | 0.00 | PASS

## Body Orientation at Step 0

- pitch_x: 0.000000 rad
- roll_y: 0.000000 rad
- yaw_z: 0.000000 rad

## COM Height at Step 0

- com_z: 0.295485 m
- target_com_z: 0.295485 m
- root_z: 0.397088 m
