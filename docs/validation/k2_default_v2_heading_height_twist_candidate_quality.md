# K2 JAX Dedicated Realtime — Behavior Quality Baseline

**Generated:** 2026-06-30T18:52:23.408016
**Input:** `outputs\validation\k2_default_v2_heading_height_twist_candidate`
**input_dir:** outputs\validation\k2_default_v2_heading_height_twist_candidate
**analyzer_version:** 1.0.0
**baseline_type:** K2_JAX_DEDICATED_REALTIME_IMPROVEMENT_BASELINE

## Executive Summary

- **Total scenarios:** 39
- **Falls:** 0 (none)
- **Scenarios with full telemetry:** 0/39
- **Performance:** 170.1 Hz avg (min 113.0, max 180.6)

## A. Safety — Hard Gates

#### Safety Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| contact_loss_steps | 2.2821 | 3.0797 | 1.0000 | 11.0000 | 1.0000 |
| fall_step | -1.0000 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| fell | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hip_yaw_joint_max_rad | 0.0834 | 0.0447 | 0.0155 | 0.2328 | 0.0833 |
| nan_inf_detected | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pitch_max_deg | 6.4677 | 3.4264 | 1.8240 | 13.8442 | 7.4686 |
| pitch_min_deg | -4.8504 | 3.2837 | -14.5497 | 0.0000 | -4.2521 |
| roll_max_deg | 0.7619 | 0.3969 | 0.1722 | 1.6524 | 0.8866 |
| roll_min_deg | -0.2240 | 0.1811 | -0.5849 | 0.0000 | -0.1792 |

## B. Posture Stability

#### Posture Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| angular_velocity_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| orientation_energy_integral | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pitch_peak_deg | 8.4411 | 2.1273 | 3.7462 | 14.5497 | 7.9876 |
| pitch_rate_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pitch_rms_deg | 3.8885 | 1.0321 | 1.6477 | 6.5031 | 3.8918 |
| pitch_settling_steps | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| roll_peak_deg | 0.8099 | 0.3438 | 0.2541 | 1.6524 | 0.8866 |
| roll_rate_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| roll_rms_deg | 0.3586 | 0.2104 | 0.1072 | 1.0172 | 0.3569 |
| yaw_drift_deg | 9.0300 | 10.0796 | 1.8620 | 64.0563 | 6.0580 |
| yaw_rate_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Pitch RMS by Height Region

| Region | Mean Pitch RMS (deg) | Std | Count |
|--------|----------------------|-----|-------|
| low | 3.82 | 0.81 | 12 |
| mid | 3.34 | 1.67 | 9 |
| high | 4.31 | 0.52 | 13 |

### Pitch RMS by Scenario Type

| Type | Mean Pitch RMS (deg) | Std | Count |
|------|----------------------|-----|-------|
| fixed_height | 3.74 | 0.96 | 17 |
| push | 3.80 | 1.22 | 12 |
| dynamic_height | 4.54 | 1.05 | 5 |

## C. Leg Symmetry / Twist

#### Leg Symmetry Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| hip_pitch_symmetry_error_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hip_roll_symmetry_error_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hip_yaw_div_max_rad | 0.1460 | 0.0868 | 0.0263 | 0.4637 | 0.1413 |
| hip_yaw_div_rms_rad | 0.0677 | 0.0456 | 0.0111 | 0.2358 | 0.0513 |
| hip_yaw_joint_max_rad | 0.0834 | 0.0447 | 0.0155 | 0.2328 | 0.0833 |
| hip_yaw_lr_divergence_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| knee_symmetry_error_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| leg_posture_error_rms | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## D. Support / Drift

#### Support & Drift Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| com_support_offset_rms_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| final_displacement_m | 0.0937 | 0.2043 | 0.0110 | 1.2865 | 0.0419 |
| lateral_drift_m | 0.0010 | 0.2099 | -0.1575 | 1.2458 | -0.0200 |
| max_displacement_m | 0.2117 | 0.2291 | 0.0790 | 1.2865 | 0.1206 |
| sagittal_drift_m | 0.0073 | 0.0803 | -0.3756 | 0.3213 | 0.0076 |
| support_peak_m | 0.2312 | 0.2196 | 0.1019 | 1.2963 | 0.1400 |
| support_rms_m | 0.0861 | 0.0747 | 0.0359 | 0.4569 | 0.0664 |
| support_velocity_rms_m_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wheel_travel_asymmetry_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## E. Dynamic Height Tracking

#### Dynamic Height Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| dynamic_transition_smoothness | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| height_final_m | 0.4013 | 0.0679 | 0.2931 | 0.4904 | 0.4075 |
| height_initial_m | 0.3949 | 0.0638 | 0.2955 | 0.4810 | 0.3993 |
| height_max_m | 0.4096 | 0.0692 | 0.2955 | 0.4978 | 0.4166 |
| height_min_m | 0.3927 | 0.0651 | 0.2917 | 0.4809 | 0.3993 |
| height_overshoot_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| height_rmse_m | 0.0127 | 0.0223 | 0.0008 | 0.1122 | 0.0055 |
| height_tracking_lag_steps | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| height_undershoot_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_ref_tracking_error_rms | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## F. Torque Quality

#### Torque Quality Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| controller_conflict_index | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_hip_roll_nm | 0.4989 | 0.3528 | 0.1959 | 1.3358 | 0.3946 |
| torque_peak_hip_yaw_nm | 2.1191 | 0.9876 | 0.5733 | 5.4888 | 1.7792 |
| torque_peak_l_hip_pitch_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_hip_roll_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_hip_yaw_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_knee_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_wheel_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_legs_nm | 9.3041 | 1.7052 | 8.0000 | 13.7135 | 8.6346 |
| torque_peak_r_hip_pitch_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_hip_roll_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_hip_yaw_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_knee_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_wheel_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_total_nm | 9.3823 | 1.6995 | 8.0000 | 13.7135 | 8.6346 |
| torque_peak_wheels_nm | 5.0992 | 3.1072 | 1.4612 | 11.5747 | 3.5343 |
| torque_rate_peak_nm_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_rate_rms_nm_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_rms_l_hip_pitch_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_rms_l_hip_roll_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_rms_l_hip_yaw_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_rms_l_knee_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_rms_l_wheel_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_rms_r_hip_pitch_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_rms_r_hip_roll_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_rms_r_hip_yaw_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_rms_r_knee_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_rms_r_wheel_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_saturation_count | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## G. Robustness

#### Robustness Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| contact_loss_frac | 0.0010 | 0.0014 | 0.0001 | 0.0055 | 0.0005 |
| contact_loss_steps | 2.2821 | 3.0797 | 1.0000 | 11.0000 | 1.0000 |
| long_run_drift_rate_m_per_kstep | 0.0292 | 0.0409 | 0.0040 | 0.2573 | 0.0184 |
| post_pitch_rms_500_deg | 1.3756 | 2.1671 | 0.0000 | 6.3352 | 0.0000 |
| post_push_active | 0.3077 | 0.4615 | 0.0000 | 1.0000 | 0.0000 |
| post_support_rms_500_m | 0.0453 | 0.0758 | 0.0000 | 0.3123 | 0.0000 |
| recovery_time_steps | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stability_score_0_to_1 | 0.7841 | 0.0482 | 0.6673 | 0.8847 | 0.7811 |

## Per-Scenario Detail

### Scope: dynamic_height

#### [SUM] gate_chatter_0p400_0p470 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 10.10 | pitch_min_deg | -2.72 |
| roll_max_deg | 0.30 | roll_min_deg | -0.58 |
| hip_yaw_joint_max_rad | 0.0914 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.76 | pitch_peak_deg | 10.10 |
| roll_rms_deg | 0.15 | roll_peak_deg | 0.58 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0914 | hip_yaw_div_rms_rad | 0.0796 |
| **Support/Drift** | | | |
| support_rms_m | 0.0784 | support_peak_m | 0.1812 |
| sagittal_drift_m | 0.040 | lateral_drift_m | -0.011 |
| final_displacement_m | 0.042 | max_displacement_m | 0.178 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0713 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.07 |
| **Robustness** | | | |
| stability_score | 0.753 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0084 | | |

#### [SUM] gate_dwell_0p420_0p450_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 10.77 | pitch_min_deg | -3.04 |
| roll_max_deg | 0.27 | roll_min_deg | -0.50 |
| hip_yaw_joint_max_rad | 0.2328 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 6.50 | pitch_peak_deg | 10.77 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.50 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.2328 | hip_yaw_div_rms_rad | 0.2358 |
| **Support/Drift** | | | |
| support_rms_m | 0.3027 | support_peak_m | 0.7650 |
| sagittal_drift_m | -0.376 | lateral_drift_m | -0.157 |
| final_displacement_m | 0.407 | max_displacement_m | 0.921 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0777 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.667 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0679 | | |

#### [SUM] ramp_down_0p480_to_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.78 | pitch_min_deg | -1.92 |
| roll_max_deg | 0.30 | roll_min_deg | -0.48 |
| hip_yaw_joint_max_rad | 0.1215 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.03 | pitch_peak_deg | 8.78 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.48 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1215 | hip_yaw_div_rms_rad | 0.1176 |
| **Support/Drift** | | | |
| support_rms_m | 0.4569 | support_peak_m | 1.2963 |
| sagittal_drift_m | 0.321 | lateral_drift_m | 1.246 |
| final_displacement_m | 1.287 | max_displacement_m | 1.287 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.1122 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.30 |
| **Robustness** | | | |
| stability_score | 0.790 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.2573 | | |

#### [SUM] ramp_up_0p330_to_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.50 | pitch_min_deg | -7.14 |
| roll_max_deg | 0.89 | roll_min_deg | -0.20 |
| hip_yaw_joint_max_rad | 0.1009 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.67 | pitch_peak_deg | 8.50 |
| roll_rms_deg | 0.37 | roll_peak_deg | 0.89 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1009 | hip_yaw_div_rms_rad | 0.0932 |
| **Support/Drift** | | | |
| support_rms_m | 0.0660 | support_peak_m | 0.1361 |
| sagittal_drift_m | 0.014 | lateral_drift_m | -0.031 |
| final_displacement_m | 0.034 | max_displacement_m | 0.120 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0055 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.61 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.794 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0068 | | |

#### [SUM] up_down_cycle_0p330_0p480_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.76 | pitch_min_deg | -7.14 |
| roll_max_deg | 0.80 | roll_min_deg | -0.25 |
| hip_yaw_joint_max_rad | 0.1324 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.75 | pitch_peak_deg | 8.76 |
| roll_rms_deg | 0.30 | roll_peak_deg | 0.80 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1324 | hip_yaw_div_rms_rad | 0.1336 |
| **Support/Drift** | | | |
| support_rms_m | 0.0688 | support_peak_m | 0.1581 |
| sagittal_drift_m | 0.015 | lateral_drift_m | -0.027 |
| final_displacement_m | 0.031 | max_displacement_m | 0.155 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0056 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.61 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.794 | contact_loss_frac | 0.0001 |
| drift_rate_m_per_kstep | 0.0044 | | |

### Scope: long_run

#### [SUM] high_0p430 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.84 | pitch_min_deg | -2.43 |
| roll_max_deg | 0.68 | roll_min_deg | -0.05 |
| hip_yaw_joint_max_rad | 0.1162 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.73 | pitch_peak_deg | 7.84 |
| roll_rms_deg | 0.32 | roll_peak_deg | 0.68 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1162 | hip_yaw_div_rms_rad | 0.0811 |
| **Support/Drift** | | | |
| support_rms_m | 0.0651 | support_peak_m | 0.1584 |
| sagittal_drift_m | 0.032 | lateral_drift_m | -0.055 |
| final_displacement_m | 0.064 | max_displacement_m | 0.164 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.71 | torque_peak_wheels_nm | 2.96 |
| **Robustness** | | | |
| stability_score | 0.794 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0106 | | |

#### [SUM] high_0p450 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.99 | pitch_min_deg | -0.09 |
| roll_max_deg | 0.54 | roll_min_deg | -0.09 |
| hip_yaw_joint_max_rad | 0.0984 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.53 | pitch_peak_deg | 7.99 |
| roll_rms_deg | 0.20 | roll_peak_deg | 0.54 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0984 | hip_yaw_div_rms_rad | 0.0880 |
| **Support/Drift** | | | |
| support_rms_m | 0.0687 | support_peak_m | 0.1458 |
| sagittal_drift_m | 0.006 | lateral_drift_m | -0.140 |
| final_displacement_m | 0.140 | max_displacement_m | 0.141 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.77 | torque_peak_wheels_nm | 4.00 |
| **Robustness** | | | |
| stability_score | 0.761 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0234 | | |

#### [SUM] high_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.85 | pitch_min_deg | -1.49 |
| roll_max_deg | 0.29 | roll_min_deg | -0.55 |
| hip_yaw_joint_max_rad | 0.0999 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.62 | pitch_peak_deg | 8.85 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.55 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0999 | hip_yaw_div_rms_rad | 0.0979 |
| **Support/Drift** | | | |
| support_rms_m | 0.0717 | support_peak_m | 0.1345 |
| sagittal_drift_m | 0.046 | lateral_drift_m | 0.005 |
| final_displacement_m | 0.046 | max_displacement_m | 0.144 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.30 |
| **Robustness** | | | |
| stability_score | 0.761 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0077 | | |

#### [SUM] low_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 1.65 | roll_min_deg | -0.11 |
| hip_yaw_joint_max_rad | 0.1101 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.83 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 1.02 | roll_peak_deg | 1.65 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1101 | hip_yaw_div_rms_rad | 0.1145 |
| **Support/Drift** | | | |
| support_rms_m | 0.0359 | support_peak_m | 0.1230 |
| sagittal_drift_m | 0.013 | lateral_drift_m | -0.020 |
| final_displacement_m | 0.024 | max_displacement_m | 0.117 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.63 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.697 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0040 | | |

#### [SUM] mid_0p400 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.75 | pitch_min_deg | -3.43 |
| roll_max_deg | 1.15 | roll_min_deg | -0.08 |
| hip_yaw_joint_max_rad | 0.1163 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 1.79 | pitch_peak_deg | 3.75 |
| roll_rms_deg | 0.45 | roll_peak_deg | 1.15 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1163 | hip_yaw_div_rms_rad | 0.1464 |
| **Support/Drift** | | | |
| support_rms_m | 0.0778 | support_peak_m | 0.1308 |
| sagittal_drift_m | 0.003 | lateral_drift_m | -0.121 |
| final_displacement_m | 0.121 | max_displacement_m | 0.121 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| **Robustness** | | | |
| stability_score | 0.883 | contact_loss_frac | 0.0017 |
| drift_rate_m_per_kstep | 0.0201 | | |

### Scope: step_c

#### [SUM] C1_slow_ladder_up_down — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 0.96 | roll_min_deg | -0.11 |
| hip_yaw_joint_max_rad | 0.0608 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.89 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.40 | roll_peak_deg | 0.96 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0608 | hip_yaw_div_rms_rad | 0.0443 |
| **Support/Drift** | | | |
| support_rms_m | 0.0491 | support_peak_m | 0.1230 |
| sagittal_drift_m | 0.008 | lateral_drift_m | 0.014 |
| final_displacement_m | 0.016 | max_displacement_m | 0.117 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.63 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.781 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0079 | | |

#### [SUM] C2_random_500dwell — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 0.96 | roll_min_deg | -0.11 |
| hip_yaw_joint_max_rad | 0.0608 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.89 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.40 | roll_peak_deg | 0.96 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0608 | hip_yaw_div_rms_rad | 0.0443 |
| **Support/Drift** | | | |
| support_rms_m | 0.0491 | support_peak_m | 0.1230 |
| sagittal_drift_m | 0.008 | lateral_drift_m | 0.014 |
| final_displacement_m | 0.016 | max_displacement_m | 0.117 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.63 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.781 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0079 | | |

#### [SUM] C3_random_200dwell — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 0.96 | roll_min_deg | -0.11 |
| hip_yaw_joint_max_rad | 0.0608 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.89 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.40 | roll_peak_deg | 0.96 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0608 | hip_yaw_div_rms_rad | 0.0443 |
| **Support/Drift** | | | |
| support_rms_m | 0.0491 | support_peak_m | 0.1230 |
| sagittal_drift_m | 0.008 | lateral_drift_m | 0.014 |
| final_displacement_m | 0.016 | max_displacement_m | 0.117 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.63 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.781 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0079 | | |

#### [SUM] C4_abrupt_stress — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 0.96 | roll_min_deg | -0.11 |
| hip_yaw_joint_max_rad | 0.0608 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.89 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.40 | roll_peak_deg | 0.96 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0608 | hip_yaw_div_rms_rad | 0.0443 |
| **Support/Drift** | | | |
| support_rms_m | 0.0491 | support_peak_m | 0.1230 |
| sagittal_drift_m | 0.008 | lateral_drift_m | 0.014 |
| final_displacement_m | 0.016 | max_displacement_m | 0.117 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.63 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.781 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0079 | | |

#### [SUM] C5_long_random — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 1.44 | roll_min_deg | -0.11 |
| hip_yaw_joint_max_rad | 0.0955 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.20 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.68 | roll_peak_deg | 1.44 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0955 | hip_yaw_div_rms_rad | 0.0732 |
| **Support/Drift** | | | |
| support_rms_m | 0.0420 | support_peak_m | 0.1230 |
| sagittal_drift_m | 0.015 | lateral_drift_m | -0.020 |
| final_displacement_m | 0.025 | max_displacement_m | 0.117 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.63 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.749 | contact_loss_frac | 0.0003 |
| drift_rate_m_per_kstep | 0.0083 | | |

#### [SUM] focused_high_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.73 | pitch_min_deg | -1.35 |
| roll_max_deg | 0.20 | roll_min_deg | -0.55 |
| hip_yaw_joint_max_rad | 0.0507 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.28 | pitch_peak_deg | 8.73 |
| roll_rms_deg | 0.14 | roll_peak_deg | 0.55 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0507 | hip_yaw_div_rms_rad | 0.0323 |
| **Support/Drift** | | | |
| support_rms_m | 0.0580 | support_peak_m | 0.1232 |
| sagittal_drift_m | 0.014 | lateral_drift_m | 0.037 |
| final_displacement_m | 0.040 | max_displacement_m | 0.093 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.30 |
| **Robustness** | | | |
| stability_score | 0.778 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0198 | | |

#### [SUM] focused_low_0p320 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.02 | pitch_min_deg | -7.15 |
| roll_max_deg | 0.81 | roll_min_deg | -0.10 |
| hip_yaw_joint_max_rad | 0.0693 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.44 | pitch_peak_deg | 7.15 |
| roll_rms_deg | 0.33 | roll_peak_deg | 0.81 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0693 | hip_yaw_div_rms_rad | 0.0356 |
| **Support/Drift** | | | |
| support_rms_m | 0.0606 | support_peak_m | 0.1116 |
| sagittal_drift_m | -0.001 | lateral_drift_m | -0.086 |
| final_displacement_m | 0.086 | max_displacement_m | 0.086 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.74 | torque_peak_wheels_nm | 2.04 |
| **Robustness** | | | |
| stability_score | 0.808 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0430 | | |

### Scope: step_d

#### [SUM] high_0p480_sagittal_backward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.83 | pitch_min_deg | -4.25 |
| roll_max_deg | 0.25 | roll_min_deg | -0.24 |
| hip_yaw_joint_max_rad | 0.0205 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.68 | pitch_peak_deg | 8.83 |
| roll_rms_deg | 0.11 | roll_peak_deg | 0.25 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0205 | hip_yaw_div_rms_rad | 0.0165 |
| **Support/Drift** | | | |
| support_rms_m | 0.0780 | support_peak_m | 0.2431 |
| sagittal_drift_m | -0.002 | lateral_drift_m | -0.076 |
| final_displacement_m | 0.076 | max_displacement_m | 0.185 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 6.23 |
| **Robustness** | | | |
| stability_score | 0.759 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0381 | | |

#### [SUM] high_0p480_sagittal_backward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.79 | pitch_min_deg | -6.85 |
| roll_max_deg | 0.33 | roll_min_deg | -0.26 |
| hip_yaw_joint_max_rad | 0.0155 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.54 | pitch_peak_deg | 8.79 |
| roll_rms_deg | 0.11 | roll_peak_deg | 0.33 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0155 | hip_yaw_div_rms_rad | 0.0111 |
| **Support/Drift** | | | |
| support_rms_m | 0.0924 | support_peak_m | 0.3672 |
| sagittal_drift_m | 0.001 | lateral_drift_m | 0.064 |
| final_displacement_m | 0.064 | max_displacement_m | 0.294 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 9.64 | torque_peak_wheels_nm | 9.64 |
| **Robustness** | | | |
| stability_score | 0.766 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0321 | | |

#### [SUM] high_0p480_sagittal_forward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 11.53 | pitch_min_deg | -1.48 |
| roll_max_deg | 0.17 | roll_min_deg | -0.26 |
| hip_yaw_joint_max_rad | 0.0206 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.90 | pitch_peak_deg | 11.53 |
| roll_rms_deg | 0.11 | roll_peak_deg | 0.26 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0206 | hip_yaw_div_rms_rad | 0.0160 |
| **Support/Drift** | | | |
| support_rms_m | 0.0787 | support_peak_m | 0.2334 |
| sagittal_drift_m | 0.008 | lateral_drift_m | 0.036 |
| final_displacement_m | 0.037 | max_displacement_m | 0.179 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.03 | torque_peak_wheels_nm | 6.30 |
| **Robustness** | | | |
| stability_score | 0.748 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0184 | | |

#### [SUM] high_0p480_sagittal_forward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 13.84 | pitch_min_deg | -1.30 |
| roll_max_deg | 0.17 | roll_min_deg | -0.31 |
| hip_yaw_joint_max_rad | 0.0208 | contact_loss_steps | 2 |
| **Posture** | | | |
| pitch_rms_deg | 4.94 | pitch_peak_deg | 13.84 |
| roll_rms_deg | 0.11 | roll_peak_deg | 0.31 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0208 | hip_yaw_div_rms_rad | 0.0153 |
| **Support/Drift** | | | |
| support_rms_m | 0.0908 | support_peak_m | 0.3542 |
| sagittal_drift_m | -0.001 | lateral_drift_m | -0.077 |
| final_displacement_m | 0.077 | max_displacement_m | 0.283 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 10.38 | torque_peak_wheels_nm | 9.88 |
| **Robustness** | | | |
| stability_score | 0.747 | contact_loss_frac | 0.0010 |
| drift_rate_m_per_kstep | 0.0387 | | |

#### [SUM] low_0p330_sagittal_backward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.78 | pitch_min_deg | -11.64 |
| roll_max_deg | 0.89 | roll_min_deg | -0.19 |
| hip_yaw_joint_max_rad | 0.0735 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.37 | pitch_peak_deg | 11.64 |
| roll_rms_deg | 0.37 | roll_peak_deg | 0.89 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0735 | hip_yaw_div_rms_rad | 0.0484 |
| **Support/Drift** | | | |
| support_rms_m | 0.0624 | support_peak_m | 0.1682 |
| sagittal_drift_m | 0.004 | lateral_drift_m | -0.028 |
| final_displacement_m | 0.028 | max_displacement_m | 0.151 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.81 | torque_peak_wheels_nm | 6.68 |
| **Robustness** | | | |
| stability_score | 0.759 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0142 | | |

#### [SUM] low_0p330_sagittal_backward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.75 | pitch_min_deg | -14.55 |
| roll_max_deg | 0.89 | roll_min_deg | -0.27 |
| hip_yaw_joint_max_rad | 0.0898 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.67 | pitch_peak_deg | 14.55 |
| roll_rms_deg | 0.40 | roll_peak_deg | 0.89 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0898 | hip_yaw_div_rms_rad | 0.0598 |
| **Support/Drift** | | | |
| support_rms_m | 0.0731 | support_peak_m | 0.2656 |
| sagittal_drift_m | 0.007 | lateral_drift_m | -0.016 |
| final_displacement_m | 0.018 | max_displacement_m | 0.200 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 10.39 | torque_peak_wheels_nm | 10.39 |
| **Robustness** | | | |
| stability_score | 0.743 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0090 | | |

#### [SUM] low_0p330_sagittal_forward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.47 | pitch_min_deg | -7.14 |
| roll_max_deg | 1.63 | roll_min_deg | -0.54 |
| hip_yaw_joint_max_rad | 0.1111 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.60 | pitch_peak_deg | 7.47 |
| roll_rms_deg | 0.83 | roll_peak_deg | 1.63 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1111 | hip_yaw_div_rms_rad | 0.0971 |
| **Support/Drift** | | | |
| support_rms_m | 0.0769 | support_peak_m | 0.3227 |
| sagittal_drift_m | -0.005 | lateral_drift_m | -0.010 |
| final_displacement_m | 0.011 | max_displacement_m | 0.296 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 10.67 | torque_peak_wheels_nm | 7.05 |
| **Robustness** | | | |
| stability_score | 0.720 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0055 | | |

#### [SUM] low_0p330_sagittal_forward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 9.90 | pitch_min_deg | -8.08 |
| roll_max_deg | 1.11 | roll_min_deg | -0.50 |
| hip_yaw_joint_max_rad | 0.1771 | contact_loss_steps | 4 |
| **Posture** | | | |
| pitch_rms_deg | 4.39 | pitch_peak_deg | 9.90 |
| roll_rms_deg | 0.55 | roll_peak_deg | 1.11 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1771 | hip_yaw_div_rms_rad | 0.0988 |
| **Support/Drift** | | | |
| support_rms_m | 0.1075 | support_peak_m | 0.4355 |
| sagittal_drift_m | -0.024 | lateral_drift_m | -0.068 |
| final_displacement_m | 0.072 | max_displacement_m | 0.400 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 13.71 | torque_peak_wheels_nm | 11.57 |
| **Robustness** | | | |
| stability_score | 0.747 | contact_loss_frac | 0.0020 |
| drift_rate_m_per_kstep | 0.0362 | | |

#### [SUM] mid_0p400_sagittal_backward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 2.81 | pitch_min_deg | -6.02 |
| roll_max_deg | 1.06 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.1183 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 1.65 | pitch_peak_deg | 6.02 |
| roll_rms_deg | 0.55 | roll_peak_deg | 1.06 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1183 | hip_yaw_div_rms_rad | 0.1000 |
| **Support/Drift** | | | |
| support_rms_m | 0.1241 | support_peak_m | 0.3691 |
| sagittal_drift_m | 0.025 | lateral_drift_m | -0.047 |
| final_displacement_m | 0.054 | max_displacement_m | 0.329 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| **Robustness** | | | |
| stability_score | 0.885 | contact_loss_frac | 0.0050 |
| drift_rate_m_per_kstep | 0.0269 | | |

#### [SUM] mid_0p400_sagittal_backward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.28 | pitch_min_deg | -8.28 |
| roll_max_deg | 1.04 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.1198 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 1.91 | pitch_peak_deg | 8.28 |
| roll_rms_deg | 0.50 | roll_peak_deg | 1.04 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1198 | hip_yaw_div_rms_rad | 0.1124 |
| **Support/Drift** | | | |
| support_rms_m | 0.1722 | support_peak_m | 0.5108 |
| sagittal_drift_m | 0.025 | lateral_drift_m | -0.123 |
| final_displacement_m | 0.125 | max_displacement_m | 0.465 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| **Robustness** | | | |
| stability_score | 0.874 | contact_loss_frac | 0.0050 |
| drift_rate_m_per_kstep | 0.0627 | | |

#### [SUM] mid_0p400_sagittal_forward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 9.31 | pitch_min_deg | -3.74 |
| roll_max_deg | 1.11 | roll_min_deg | -0.18 |
| hip_yaw_joint_max_rad | 0.1145 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 2.20 | pitch_peak_deg | 9.31 |
| roll_rms_deg | 0.56 | roll_peak_deg | 1.11 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1145 | hip_yaw_div_rms_rad | 0.0938 |
| **Support/Drift** | | | |
| support_rms_m | 0.0950 | support_peak_m | 0.2287 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.060 |
| final_displacement_m | 0.061 | max_displacement_m | 0.194 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| **Robustness** | | | |
| stability_score | 0.856 | contact_loss_frac | 0.0050 |
| drift_rate_m_per_kstep | 0.0305 | | |

#### [SUM] mid_0p400_sagittal_forward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 11.77 | pitch_min_deg | -3.43 |
| roll_max_deg | 0.97 | roll_min_deg | -0.46 |
| hip_yaw_joint_max_rad | 0.1116 | contact_loss_steps | 11 |
| **Posture** | | | |
| pitch_rms_deg | 2.72 | pitch_peak_deg | 11.77 |
| roll_rms_deg | 0.48 | roll_peak_deg | 0.97 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1116 | hip_yaw_div_rms_rad | 0.0867 |
| **Support/Drift** | | | |
| support_rms_m | 0.0945 | support_peak_m | 0.2531 |
| sagittal_drift_m | -0.007 | lateral_drift_m | -0.109 |
| final_displacement_m | 0.109 | max_displacement_m | 0.201 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.42 |
| **Robustness** | | | |
| stability_score | 0.835 | contact_loss_frac | 0.0055 |
| drift_rate_m_per_kstep | 0.0547 | | |

### Scope: step_e

#### [SUM] high_0p430 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.16 | pitch_min_deg | -1.95 |
| roll_max_deg | 0.68 | roll_min_deg | -0.05 |
| hip_yaw_joint_max_rad | 0.0516 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.14 | pitch_peak_deg | 7.16 |
| roll_rms_deg | 0.35 | roll_peak_deg | 0.68 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0516 | hip_yaw_div_rms_rad | 0.0380 |
| **Support/Drift** | | | |
| support_rms_m | 0.0499 | support_peak_m | 0.1400 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.083 |
| final_displacement_m | 0.083 | max_displacement_m | 0.103 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.71 | torque_peak_wheels_nm | 2.96 |
| **Robustness** | | | |
| stability_score | 0.822 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0417 | | |

#### [SUM] high_0p450 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.98 | pitch_min_deg | -0.05 |
| roll_max_deg | 0.37 | roll_min_deg | -0.08 |
| hip_yaw_joint_max_rad | 0.0193 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.68 | pitch_peak_deg | 7.98 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.37 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0193 | hip_yaw_div_rms_rad | 0.0134 |
| **Support/Drift** | | | |
| support_rms_m | 0.0664 | support_peak_m | 0.1458 |
| sagittal_drift_m | 0.006 | lateral_drift_m | -0.106 |
| final_displacement_m | 0.106 | max_displacement_m | 0.112 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.77 | torque_peak_wheels_nm | 4.00 |
| **Robustness** | | | |
| stability_score | 0.758 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0531 | | |

#### [SUM] high_0p465 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.25 | pitch_min_deg | -2.22 |
| roll_max_deg | 0.34 | roll_min_deg | -0.25 |
| hip_yaw_joint_max_rad | 0.0314 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.63 | pitch_peak_deg | 7.25 |
| roll_rms_deg | 0.14 | roll_peak_deg | 0.34 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0314 | hip_yaw_div_rms_rad | 0.0186 |
| **Support/Drift** | | | |
| support_rms_m | 0.0634 | support_peak_m | 0.1221 |
| sagittal_drift_m | 0.014 | lateral_drift_m | 0.077 |
| final_displacement_m | 0.078 | max_displacement_m | 0.093 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.04 | torque_peak_wheels_nm | 1.78 |
| **Robustness** | | | |
| stability_score | 0.810 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0391 | | |

#### [SUM] high_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.73 | pitch_min_deg | -1.35 |
| roll_max_deg | 0.20 | roll_min_deg | -0.55 |
| hip_yaw_joint_max_rad | 0.0507 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.28 | pitch_peak_deg | 8.73 |
| roll_rms_deg | 0.14 | roll_peak_deg | 0.55 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0507 | hip_yaw_div_rms_rad | 0.0323 |
| **Support/Drift** | | | |
| support_rms_m | 0.0580 | support_peak_m | 0.1232 |
| sagittal_drift_m | 0.014 | lateral_drift_m | 0.037 |
| final_displacement_m | 0.040 | max_displacement_m | 0.093 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.30 |
| **Robustness** | | | |
| stability_score | 0.778 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0198 | | |

#### [SUM] low_0p300 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.12 | pitch_min_deg | -1.71 |
| roll_max_deg | 0.95 | roll_min_deg | -0.06 |
| hip_yaw_joint_max_rad | 0.1107 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 2.92 | pitch_peak_deg | 7.12 |
| roll_rms_deg | 0.63 | roll_peak_deg | 0.95 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1107 | hip_yaw_div_rms_rad | 0.0880 |
| **Support/Drift** | | | |
| support_rms_m | 0.0411 | support_peak_m | 0.1038 |
| sagittal_drift_m | 0.001 | lateral_drift_m | 0.027 |
| final_displacement_m | 0.027 | max_displacement_m | 0.084 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 9.02 | torque_peak_wheels_nm | 1.46 |
| **Robustness** | | | |
| stability_score | 0.816 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0136 | | |

#### [SUM] low_0p320 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.02 | pitch_min_deg | -7.15 |
| roll_max_deg | 0.81 | roll_min_deg | -0.10 |
| hip_yaw_joint_max_rad | 0.0693 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.44 | pitch_peak_deg | 7.15 |
| roll_rms_deg | 0.33 | roll_peak_deg | 0.81 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0693 | hip_yaw_div_rms_rad | 0.0356 |
| **Support/Drift** | | | |
| support_rms_m | 0.0606 | support_peak_m | 0.1116 |
| sagittal_drift_m | -0.001 | lateral_drift_m | -0.086 |
| final_displacement_m | 0.086 | max_displacement_m | 0.086 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.74 | torque_peak_wheels_nm | 2.04 |
| **Robustness** | | | |
| stability_score | 0.808 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0430 | | |

#### [SUM] low_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 0.96 | roll_min_deg | -0.11 |
| hip_yaw_joint_max_rad | 0.0608 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.89 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.40 | roll_peak_deg | 0.96 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0608 | hip_yaw_div_rms_rad | 0.0443 |
| **Support/Drift** | | | |
| support_rms_m | 0.0491 | support_peak_m | 0.1230 |
| sagittal_drift_m | 0.008 | lateral_drift_m | 0.014 |
| final_displacement_m | 0.016 | max_displacement_m | 0.117 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.63 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.781 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0079 | | |

#### [SUM] low_0p340 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 4.55 | pitch_min_deg | -3.05 |
| roll_max_deg | 1.11 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.0798 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 1.86 | pitch_peak_deg | 4.55 |
| roll_rms_deg | 0.56 | roll_peak_deg | 1.11 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0798 | hip_yaw_div_rms_rad | 0.0513 |
| **Support/Drift** | | | |
| support_rms_m | 0.0501 | support_peak_m | 0.1084 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.013 |
| final_displacement_m | 0.016 | max_displacement_m | 0.110 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.62 | torque_peak_wheels_nm | 1.95 |
| **Robustness** | | | |
| stability_score | 0.874 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0081 | | |

#### [SUM] low_0p360 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 2.40 | pitch_min_deg | -7.05 |
| roll_max_deg | 0.94 | roll_min_deg | -0.18 |
| hip_yaw_joint_max_rad | 0.0833 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.21 | pitch_peak_deg | 7.05 |
| roll_rms_deg | 0.36 | roll_peak_deg | 0.94 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0833 | hip_yaw_div_rms_rad | 0.0439 |
| **Support/Drift** | | | |
| support_rms_m | 0.0541 | support_peak_m | 0.1019 |
| sagittal_drift_m | 0.008 | lateral_drift_m | 0.036 |
| final_displacement_m | 0.036 | max_displacement_m | 0.079 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.55 | torque_peak_wheels_nm | 1.84 |
| **Robustness** | | | |
| stability_score | 0.818 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0182 | | |

#### [SUM] low_0p380 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 9.11 | pitch_min_deg | 0.00 |
| roll_max_deg | 0.58 | roll_min_deg | -0.07 |
| hip_yaw_joint_max_rad | 0.0227 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 5.34 | pitch_peak_deg | 9.11 |
| roll_rms_deg | 0.26 | roll_peak_deg | 0.58 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0227 | hip_yaw_div_rms_rad | 0.0150 |
| **Support/Drift** | | | |
| support_rms_m | 0.0701 | support_peak_m | 0.1059 |
| sagittal_drift_m | -0.003 | lateral_drift_m | -0.024 |
| final_displacement_m | 0.024 | max_displacement_m | 0.092 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 9.16 | torque_peak_wheels_nm | 3.13 |
| **Robustness** | | | |
| stability_score | 0.717 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0122 | | |

## Missing Telemetry Fields

The following rich metrics were NOT available because full telemetry CSVs were not present:

- **pitch_rate_rms_deg_s**, **roll_rate_rms_deg_s**, **yaw_rate_rms_deg_s** — angular velocity RMS
- **pitch_settling_steps** — settling time analysis
- **angular_velocity_rms_deg_s** — combined angular velocity
- **orientation_energy_integral** — cumulative orientation energy
- **hip_yaw_lr_divergence_deg** — left-right hip yaw joint divergence
- **hip_pitch_symmetry_error_deg** — hip pitch symmetry
- **knee_symmetry_error_deg** — knee symmetry
- **leg_posture_error_rms** — total leg posture deviation
- **support_velocity_rms_m_s** — support center velocity
- **wheel_travel_asymmetry_m** — wheel travel difference
- **com_support_offset_rms_m** — COM-support offset
- **q_ref_tracking_error_rms** — posture reference tracking
- **dynamic_transition_smoothness** — height jerk smoothness
- **torque_rate_rms_nm_s**, **torque_rate_peak_nm_s** — torque rate metrics
- **torque_saturation_count** — saturation frequency
- **per-joint torque RMS** — detailed torque distribution
- **controller_conflict_index** — component conflict analysis (requires component-level telemetry)

**Recommendation:** Run Phase 0 baseline with `--telemetry full` to enable all rich metrics.
For Phase 3+ (controller conflict analysis), additional component-level telemetry
instrumentation will be needed in the JAX controller itself.

## JSON Data Export

Full metrics exported to: `docs\validation\k2_default_v2_heading_height_twist_candidate_quality.json`