# Subsystem Audit Report

**Date:** 2026-06-05
**Phase:** Phase 5

## Summary by Variant

### low_0p300

**A. Hip-Yaw Subsystem:**
- Classification: one_sided_drift
- hip_yaw_abs_max: 0.2807 rad
- divergence_rms: 0.3575 rad
- sign_correct_l: 0.0%
- sign_correct_r: 0.0%

**B. Body Yaw Subsystem:**
- Classification: stable
- body_yaw_max: 0.0109 rad
- yaw_drift_max: 0.0149 rad

**C. Support/Sagittal Subsystem:**
- Classification: large_pitch
- support_error_max: 0.0000 m
- wheel_vel_max: 7.25 rad/s
- pitch_max: 0.1571 rad

**D. Roll/Lateral Subsystem:**
- Classification: hip_roll_saturation
- roll_max: 0.0140 rad
- hip_roll_abs_max: 0.2167 rad

**E. Height/Contact Subsystem:**
- Classification: moderate_height_error
- height_error_max: 0.0214 m
- contact_valid_pct: 100.0%

**F. Torque Composer:**
- ownership_violations_max: 0

---

### nominal

**A. Hip-Yaw Subsystem:**
- Classification: divergence_present
- hip_yaw_abs_max: 0.0576 rad
- divergence_rms: 0.0447 rad
- sign_correct_l: 0.0%
- sign_correct_r: 0.0%

**B. Body Yaw Subsystem:**
- Classification: stable
- body_yaw_max: 0.0505 rad
- yaw_drift_max: 0.0946 rad

**C. Support/Sagittal Subsystem:**
- Classification: controlled
- support_error_max: 0.0000 m
- wheel_vel_max: 3.84 rad/s
- pitch_max: 0.0708 rad

**D. Roll/Lateral Subsystem:**
- Classification: hip_roll_saturation
- roll_max: 0.0130 rad
- hip_roll_abs_max: 0.1749 rad

**E. Height/Contact Subsystem:**
- Classification: controlled
- height_error_max: 0.0085 m
- contact_valid_pct: 100.0%

**F. Torque Composer:**
- ownership_violations_max: 0

---

### high_0p480

**A. Hip-Yaw Subsystem:**
- Classification: one_sided_drift
- hip_yaw_abs_max: 0.2619 rad
- divergence_rms: 0.2825 rad
- sign_correct_l: 0.0%
- sign_correct_r: 0.0%

**B. Body Yaw Subsystem:**
- Classification: stable
- body_yaw_max: 0.0434 rad
- yaw_drift_max: 0.1036 rad

**C. Support/Sagittal Subsystem:**
- Classification: controlled
- support_error_max: 0.0000 m
- wheel_vel_max: 5.37 rad/s
- pitch_max: 0.0926 rad

**D. Roll/Lateral Subsystem:**
- Classification: stable
- roll_max: 0.0023 rad
- hip_roll_abs_max: 0.0773 rad

**E. Height/Contact Subsystem:**
- Classification: controlled
- height_error_max: 0.0154 m
- contact_valid_pct: 100.0%

**F. Torque Composer:**
- ownership_violations_max: 0

---

