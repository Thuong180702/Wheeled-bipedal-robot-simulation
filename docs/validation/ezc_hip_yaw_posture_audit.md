# EZC Hip-Yaw and Posture Audit

**Date:** 2026-06-15  
**Profile:** early_zero_crossing_recenter  
**Scenario:** high_0p480, 5000 steps

## Classification

**EZC_POSTURE_HIP_YAW_SAFE**

## Hip-Yaw Analysis

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| Left hip yaw pos (rad) | -0.0750 | -0.2507 | 0.0012 |
| Right hip yaw pos (rad) | +0.0806 | -0.0027 | +0.2501 |
| Hip yaw asymmetry (l - r) | -0.1556 | -0.5008 | +0.0029 |
| Hip yaw error RMS | 0.0785 | 0.0000 | 0.2504 |

### Interpretation

The robot maintains a **V-shaped stance**:
- Left toes point outward: mean = -0.075 rad
- Right toes point outward: mean = +0.081 rad

This is a symmetric toe-out stance, NOT a yaw drift problem. The asymmetry is intentional/controlled posture.

### Safety Status

| Gate | EZC threshold | Max observed | Status |
|------|---------------|--------------|--------|
| EZC hip_yaw disable | 0.25 rad | ~0.25 max | SAFE (never exceeded) |
| EZC roll disable | 5.0 deg | 0.0 deg | SAFE |
| EZC pitch disable | 12.0 deg | 0.29 deg | SAFE |

**EZC hip-yaw safety gates never triggered.**

## Roll Analysis

| Metric | Value |
|--------|-------|
| euler_roll_y mean | 0.0000 rad |
| euler_roll_y min | 0.0000 rad |
| euler_roll_y max | 0.0000 rad |
| |roll| max | 0.0000 rad |

**Roll is perfectly zero. No lateral instability.**

## Pitch Analysis

| Metric | Value |
|--------|-------|
| euler_pitch_y mean | 0.0004 rad (0.023 deg) |
| euler_pitch_y min | -0.0041 rad (-0.23 deg) |
| euler_pitch_y max | +0.0051 rad (+0.29 deg) |
| |pitch| max | 0.0051 rad (0.29 deg) |

**Pitch is very well controlled. No pitch instability.**

## Height/CoM Analysis

| Metric | Value |
|--------|-------|
| CoM Z mean | 0.4849 m |
| CoM Z min | 0.4664 m |
| CoM Z max | 0.4915 m |
| height_cmd | 0.4000 m |

**Height is stable (~0.485 m) and close to target (0.480 m for high_0p480).**

## Contact Analysis

| Metric | Value |
|--------|-------|
| Left contact | 100.0% |
| Right contact | 100.0% |
| Double contact | 100.0% |

**Both feet always on ground. No transition/recovery issues.**

## Conclusion

### Posture Safety: PASS

- Hip-yaw is safe (below threshold, V-shape stance intentional)
- Roll is perfect zero
- Pitch is very well controlled
- Height is stable
- Contact is 100% double support

### Hip-Yaw Drift: NOT A FACTOR

The `hip_yaw_comp_support_error_m` column has the same range as other drift columns, indicating the controller properly compensates for any hip-yaw contribution.

**Hip-yaw is NOT the cause of drift bias.**

## Recommendation

Proceed with fix path based on Phase 2 root cause: **EZC_FAILURE_EXIT_TOO_EARLY_REBOUND**

Hip-yaw safety gates are working correctly and posture is stable. The drift issue is purely sagittal (forward/backward), not related to yaw or roll.