# Hip-Yaw Torque Authority Audit Report

## Phase 3: Torque Authority Audit

Date: 2026-06-04

## Purpose

Determine if hip-yaw torque is sufficient, saturated, rate-limited, or overwritten.

## Results

### low_0p300

**Classification:** `no_torque_issue_detected`

**Overall Torque:**
- l_tau_raw max_abs: 3.1953 Nm
- r_tau_raw max_abs: 3.2998 Nm
- l_sign_correct_rate: 1.000
- r_sign_correct_rate: 1.000
- l_saturated_rate: 0.000
- r_saturated_rate: 0.000
- torque_matches_shape: True
- ownership_violations: 0

**Hip-Yaw Onset Window:**
- center_step: 348
- l_tau_raw max_abs: 0.2938 Nm
- r_tau_raw max_abs: 0.6128 Nm
- l_error max_abs: 0.0139 rad
- r_error max_abs: 0.0334 rad

**Hip-Yaw Peak Window:**
- center_step: 562
- l_tau_raw max_abs: 3.1804 Nm
- r_tau_raw max_abs: 3.2757 Nm
- l_error max_abs: 0.2063 rad
- r_error max_abs: 0.2137 rad

---

### high_0p480

**Classification:** `no_torque_issue_detected`

**Overall Torque:**
- l_tau_raw max_abs: 0.4028 Nm
- r_tau_raw max_abs: 0.7082 Nm
- l_sign_correct_rate: 0.964
- r_sign_correct_rate: 0.978
- l_saturated_rate: 0.000
- r_saturated_rate: 0.000
- torque_matches_shape: True
- ownership_violations: 0

**Hip-Yaw Onset Window:**
- center_step: 716
- l_tau_raw max_abs: 0.2471 Nm
- r_tau_raw max_abs: 0.4758 Nm
- l_error max_abs: 0.0162 rad
- r_error max_abs: 0.0306 rad

**Hip-Yaw Peak Window:**
- center_step: 999
- l_tau_raw max_abs: 0.4028 Nm
- r_tau_raw max_abs: 0.7082 Nm
- l_error max_abs: 0.0256 rad
- r_error max_abs: 0.0462 rad

---

### nominal

**Classification:** `sign_error_detected`

**Overall Torque:**
- l_tau_raw max_abs: 0.3639 Nm
- r_tau_raw max_abs: 0.6213 Nm
- l_sign_correct_rate: 0.885
- r_sign_correct_rate: 0.989
- l_saturated_rate: 0.000
- r_saturated_rate: 0.000
- torque_matches_shape: True
- ownership_violations: 0

**Hip-Yaw Onset Window:**
- center_step: 464
- l_tau_raw max_abs: 0.2831 Nm
- r_tau_raw max_abs: 0.5714 Nm
- l_error max_abs: 0.0170 rad
- r_error max_abs: 0.0328 rad

**Hip-Yaw Peak Window:**
- center_step: 937
- l_tau_raw max_abs: 0.3274 Nm
- r_tau_raw max_abs: 0.6163 Nm
- l_error max_abs: 0.0225 rad
- r_error max_abs: 0.0392 rad

**Issues:** sign_error_detected

---

## Summary
