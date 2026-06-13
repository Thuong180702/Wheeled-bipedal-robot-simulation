# Hip-Yaw Baseline Telemetry Report

## Phase 1: Fresh Baseline Telemetry Collection

Date: 2026-06-04

## Controller State

- WBC: disabled
- Experimental hip-yaw fix: disabled
- Sagittal hybrid fix: disabled
- Passive feedforward fix: disabled
- Global hip-yaw gain change: disabled

## Cases Evaluated

### low_0p300

Status: SUCCESS

**Hip-Yaw Metrics:**
- hip_yaw_abs_max: 0.2137 rad
- l_hip_yaw_error final: 0.1937 rad
- r_hip_yaw_error final: -0.1969 rad
- hip_yaw_divergence_max: 0.4200 rad

**Support Position:**
- support_position_error_max_abs: 0.2430 m

**Pitch:**
- pitch_x_max_abs: 0.0951 rad

**Event Order:**
- First hip_yaw > 0.07: step 418
- First support > 0.15: step 89
- First pitch > 0.10: step None
- Classification: **support_position_led**

---

### high_0p480

Status: SUCCESS

**Hip-Yaw Metrics:**
- hip_yaw_abs_max: 0.0462 rad
- l_hip_yaw_error final: 0.0256 rad
- r_hip_yaw_error final: -0.0462 rad
- hip_yaw_divergence_max: 0.0719 rad

**Support Position:**
- support_position_error_max_abs: 0.2336 m

**Pitch:**
- pitch_x_max_abs: 0.0926 rad

**Event Order:**
- First hip_yaw > 0.07: step None
- First support > 0.15: step 108
- First pitch > 0.10: step None
- Classification: **support_position_only**

---

### nominal

Status: SUCCESS

**Hip-Yaw Metrics:**
- hip_yaw_abs_max: 0.0392 rad
- l_hip_yaw_error final: 0.0016 rad
- r_hip_yaw_error final: -0.0131 rad
- hip_yaw_divergence_max: 0.0612 rad

**Support Position:**
- support_position_error_max_abs: 0.1026 m

**Pitch:**
- pitch_x_max_abs: 0.0706 rad

**Event Order:**
- First hip_yaw > 0.07: step None
- First support > 0.15: step None
- First pitch > 0.10: step None
- Classification: **none_exceeded**

---

## Summary

- Successful simulations: 3/3
- All baseline telemetry collected successfully
- Ready for Phase 2: Hip-yaw reference and command audit