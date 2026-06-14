# Telemetry Completeness Audit

**Date:** 2026-06-05
**Phase:** Phase 6

## Required vs Available Telemetry

### Hip-Yaw Subsystem

| Required Column | Available | Notes |
|-----------------|-----------|-------|
| l_hip_yaw_pos | ✓ | Present |
| r_hip_yaw_pos | ✓ | Present |
| l_hip_yaw_ref | ✓ | Present |
| r_hip_yaw_ref | ✓ | Present |
| l_hip_yaw_error | ✓ | Present |
| r_hip_yaw_error | ✓ | Present |
| l_hip_yaw_vel | ✓ | Present |
| r_hip_yaw_vel | ✓ | Present |
| l_hip_yaw_tau_shape_raw | ✓ | Present |
| r_hip_yaw_tau_shape_raw | ✓ | Present |
| l_hip_yaw_tau_shape_final | ✓ | Present |
| r_hip_yaw_tau_shape_final | ✓ | Present |
| hip_yaw_torque_sign_correct_left | ✓ | Present |
| hip_yaw_torque_sign_correct_right | ✓ | Present |
| hip_yaw_abs_max | ✓ | Present |

**Status:** COMPLETE - All required hip-yaw telemetry available

### Body Yaw Subsystem

| Required Column | Available | Notes |
|-----------------|-----------|-------|
| yaw_z_rad | ✓ | Present |
| yaw_rate_z_rad_s | ✓ | Present |
| yaw_error_from_equilibrium_rad | ✓ | Present |
| yaw_drift_from_initial_rad | ✓ | Present |
| root_yaw_z_rad | ✓ | Present |

**Status:** COMPLETE - All required body yaw telemetry available

### Support/Sagittal Subsystem

| Required Column | Available | Notes |
|-----------------|-----------|-------|
| support_position_error | ✓ | Present |
| support_center_x | ✓ | Present |
| support_center_y | ✓ | Present |
| com_x_m | ✓ | Present |
| com_y_m | ✓ | Present |
| com_z_m | ✓ | Present |
| com_vx_m_s | ✓ | Present |
| wheel_vel_mean_rad_s | ✓ | Present |
| height_error_m | ✓ | Present |
| pitch_x_rad | ✓ | Present |

**Status:** COMPLETE - All required support/sagittal telemetry available

### Roll/Lateral Subsystem

| Required Column | Available | Notes |
|-----------------|-----------|-------|
| roll_y_rad | ✓ | Present |
| roll_rate_y_rad_s | ✓ | Present |
| hip_roll_left_rad | ✓ | Present |
| hip_roll_right_rad | ✓ | Present |
| hip_roll_error_left_rad | ✓ | Present |
| hip_roll_error_right_rad | ✓ | Present |
| hip_roll_abs_max | ✓ | Present |

**Status:** COMPLETE - All required roll/lateral telemetry available

### Height/Contact Subsystem

| Required Column | Available | Notes |
|-----------------|-----------|-------|
| com_z_m | ✓ | Present |
| root_z_m | ✓ | Present |
| height_error_m | ✓ | Present |
| target_com_z_m | ✓ | Present |
| current_com_z_m | ✓ | Present |
| left_contact_active | ✓ | Present |
| right_contact_active | ✓ | Present |
| contact_force_valid | ✓ | Present |
| non_wheel_floor_contacts | ✓ | Present |

**Status:** COMPLETE - All required height/contact telemetry available

### Torque Composer Subsystem

| Required Column | Available | Notes |
|-----------------|-----------|-------|
| tau_shape_posture_per_joint | ✓ | Present |
| tau_support_feedforward_per_joint | ✓ | Present |
| tau_sagittal_wheel_balance_per_joint | ✓ | Present |
| tau_lateral_roll_balance_per_joint | ✓ | Present |
| tau_final_per_joint | ✓ | Present |
| torque_saturation_mask_per_joint | ✓ | Present |
| ownership_violation_count | ✓ | Present |

**Status:** COMPLETE - All required torque composer telemetry available

## Missing Telemetry Analysis

### Critical Missing Telemetry

**None identified.** All required telemetry for root-cause analysis is available.

### Telemetry Gaps (Non-Critical)

1. **WBC applied torque per joint:** `tau_wbc_scaled_per_joint` exists but is all zeros (correct for balance-core mode)

2. **Yaw controller torque:** No separate yaw controller torque column, but this is composed into shape posture

3. **Per-joint pitch error:** `pitch_x` is body-level, not per-joint

## Completeness Assessment

| Subsystem | Completeness | Root-Cause Identification |
|-----------|--------------|----------------------------|
| Hip-Yaw | COMPLETE | ✓ Fully identifiable |
| Body Yaw | COMPLETE | ✓ Fully identifiable |
| Support/Sagittal | COMPLETE | ✓ Fully identifiable |
| Roll/Lateral | COMPLETE | ✓ Fully identifiable |
| Height/Contact | COMPLETE | ✓ Fully identifiable |
| Torque Composer | COMPLETE | ✓ Fully identifiable |

**Overall Status:** TELEMETRY_COMPLETE

All telemetry required for root-cause identification is present and interpretable.

## Confidence Assessment

| Finding | Confidence | Basis |
|---------|------------|-------|
| Hip-yaw sign correctness = 0% | HIGH | Direct telemetry measurement |
| Divergence RMS high at boundary heights | HIGH | Direct telemetry measurement |
| Body yaw stable | HIGH | Direct telemetry measurement |
| Hip-roll saturation at low heights | MEDIUM | telemetry shows abs_max but cause unclear |
| Pitch large at low_0p300 | HIGH | Direct telemetry measurement |

## Root-Cause Identifiability

Based on telemetry completeness, the following root causes are **IDENTIFIABLE**:

1. **Hip-yaw torque sign convention error** - Confirmed by sign_correct_left/right = 0%
2. **Hip-yaw divergence at boundary heights** - Confirmed by divergence_rms measurements
3. **Body yaw stability** - Confirmed by yaw_drift measurements
4. **Height/contact validity** - Confirmed by contact telemetry

The following requires **FURTHER ANALYSIS** but telemetry is available:

1. **Hip-roll saturation cause** - Need to correlate with roll_y and controller outputs
2. **Pitch behavior at low heights** - Need to correlate with sagittal controller