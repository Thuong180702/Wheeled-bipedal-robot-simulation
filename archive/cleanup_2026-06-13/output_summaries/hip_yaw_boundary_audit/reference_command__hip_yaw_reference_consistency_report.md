# Hip-Yaw Reference and Command Audit Report

## Phase 2: Reference and Command Audit

Date: 2026-06-04

## Purpose

Verify that hip-yaw reference values are correctly set and torque commands
have the correct sign before investigating authority issues.

## Results

### low_0p300

**Classification:** `reference_correct`

**Reference:**
- l_hip_yaw_ref: 0.000000 rad (std: 0.000000)
- r_hip_yaw_ref: 0.000000 rad (std: 0.000000)
- l_hip_yaw_initial_pos: 0.000000 rad
- r_hip_yaw_initial_pos: 0.000000 rad
- l_hip_yaw_initial_error: 0.000000 rad
- r_hip_yaw_initial_error: 0.000000 rad
- shape_posture_reference_source: height_variant_equilibrium_joint_pos
- support_reference_captured: True

**Command:**
- l_hip_yaw_tau_shape_raw mean: 1.590099 Nm
- r_hip_yaw_tau_shape_raw mean: -1.705532 Nm
- l_hip_yaw_tau_shape_final mean: 1.590099 Nm
- r_hip_yaw_tau_shape_final mean: -1.705532 Nm
- l_hip_yaw_tau_shape_raw max_abs: 3.195270 Nm
- r_hip_yaw_tau_shape_raw max_abs: 3.299836 Nm
- sign_correct_rate_left: 1.000
- sign_correct_rate_right: 1.000

---

### high_0p480

**Classification:** `reference_correct`

**Reference:**
- l_hip_yaw_ref: 0.000000 rad (std: 0.000000)
- r_hip_yaw_ref: 0.000000 rad (std: 0.000000)
- l_hip_yaw_initial_pos: 0.000000 rad
- r_hip_yaw_initial_pos: 0.000000 rad
- l_hip_yaw_initial_error: 0.000000 rad
- r_hip_yaw_initial_error: 0.000000 rad
- shape_posture_reference_source: height_variant_equilibrium_joint_pos
- support_reference_captured: True

**Command:**
- l_hip_yaw_tau_shape_raw mean: 0.179407 Nm
- r_hip_yaw_tau_shape_raw mean: -0.297929 Nm
- l_hip_yaw_tau_shape_final mean: 0.179407 Nm
- r_hip_yaw_tau_shape_final mean: -0.297929 Nm
- l_hip_yaw_tau_shape_raw max_abs: 0.402781 Nm
- r_hip_yaw_tau_shape_raw max_abs: 0.708236 Nm
- sign_correct_rate_left: 0.964
- sign_correct_rate_right: 0.978

---

### nominal

**Classification:** `sign_error`

**Reference:**
- l_hip_yaw_ref: 0.000000 rad (std: 0.000000)
- r_hip_yaw_ref: 0.000000 rad (std: 0.000000)
- l_hip_yaw_initial_pos: 0.000000 rad
- r_hip_yaw_initial_pos: 0.000000 rad
- l_hip_yaw_initial_error: 0.000000 rad
- r_hip_yaw_initial_error: 0.000000 rad
- shape_posture_reference_source: nominal_equilibrium_joint_pos
- support_reference_captured: False

**Command:**
- l_hip_yaw_tau_shape_raw mean: 0.120826 Nm
- r_hip_yaw_tau_shape_raw mean: -0.268679 Nm
- l_hip_yaw_tau_shape_final mean: 0.120826 Nm
- r_hip_yaw_tau_shape_final mean: -0.268679 Nm
- l_hip_yaw_tau_shape_raw max_abs: 0.363921 Nm
- r_hip_yaw_tau_shape_raw max_abs: 0.621251 Nm
- sign_correct_rate_left: 0.885
- sign_correct_rate_right: 0.989

**Issues:** sign_error

---

## Summary

❌ **BLOCKER: Torque sign error detected**

Hip-yaw torque commands have incorrect sign.
Must fix sign error before proceeding to authority audit.