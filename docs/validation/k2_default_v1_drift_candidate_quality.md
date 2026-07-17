# K2 JAX Dedicated Realtime — Behavior Quality Baseline

**Generated:** 2026-06-30T15:25:21.885996
**Input:** `outputs\k2_default_v1_drift_candidate`
**input_dir:** outputs\k2_default_v1_drift_candidate
**analyzer_version:** 1.0.0
**baseline_type:** K2_JAX_DEDICATED_REALTIME_IMPROVEMENT_BASELINE

## Executive Summary

- **Total scenarios:** 39
- **Falls:** 0 (none)
- **Scenarios with full telemetry:** 0/39
- **Performance:** 164.2 Hz avg (min 150.2, max 172.0)

## A. Safety — Hard Gates

#### Safety Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| contact_loss_steps | 2.2821 | 3.0797 | 1.0000 | 11.0000 | 1.0000 |
| fall_step | -1.0000 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| fell | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hip_yaw_joint_max_rad | 0.0858 | 0.0481 | 0.0155 | 0.2695 | 0.0799 |
| nan_inf_detected | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pitch_max_deg | 6.5046 | 3.4000 | 1.8240 | 13.8442 | 7.4669 |
| pitch_min_deg | -4.8426 | 3.2764 | -14.5494 | -0.0512 | -4.2521 |
| roll_max_deg | 0.7970 | 0.4245 | 0.1722 | 1.6971 | 0.8304 |
| roll_min_deg | -0.2177 | 0.1841 | -0.5823 | 0.0000 | -0.1342 |

## B. Posture Stability

#### Posture Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| angular_velocity_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| orientation_energy_integral | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pitch_peak_deg | 8.4352 | 2.1377 | 3.7432 | 14.5494 | 7.9876 |
| pitch_rate_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pitch_rms_deg | 3.9125 | 1.0174 | 1.6539 | 6.1875 | 3.9627 |
| pitch_settling_steps | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| roll_peak_deg | 0.8450 | 0.3707 | 0.2541 | 1.6971 | 0.8304 |
| roll_rate_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| roll_rms_deg | 0.3702 | 0.2288 | 0.1072 | 1.1086 | 0.3535 |
| yaw_drift_deg | 9.0570 | 8.5127 | 0.8679 | 52.8267 | 6.3835 |
| yaw_rate_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Pitch RMS by Height Region

| Region | Mean Pitch RMS (deg) | Std | Count |
|--------|----------------------|-----|-------|
| low | 3.87 | 0.84 | 12 |
| mid | 3.30 | 1.58 | 9 |
| high | 4.31 | 0.52 | 13 |

### Pitch RMS by Scenario Type

| Type | Mean Pitch RMS (deg) | Std | Count |
|------|----------------------|-----|-------|
| fixed_height | 3.77 | 0.97 | 17 |
| push | 3.81 | 1.22 | 12 |
| dynamic_height | 4.47 | 0.94 | 5 |

## C. Leg Symmetry / Twist

#### Leg Symmetry Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| hip_pitch_symmetry_error_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hip_roll_symmetry_error_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hip_yaw_div_max_rad | 0.1504 | 0.0951 | 0.0263 | 0.5375 | 0.1255 |
| hip_yaw_div_rms_rad | 0.0709 | 0.0494 | 0.0111 | 0.2603 | 0.0518 |
| hip_yaw_joint_max_rad | 0.0858 | 0.0481 | 0.0155 | 0.2695 | 0.0799 |
| hip_yaw_lr_divergence_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| knee_symmetry_error_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| leg_posture_error_rms | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## D. Support / Drift

#### Support & Drift Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| com_support_offset_rms_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| final_displacement_m | 0.0971 | 0.2180 | 0.0049 | 1.2909 | 0.0528 |
| lateral_drift_m | -0.0009 | 0.2148 | -0.3303 | 1.2530 | -0.0125 |
| max_displacement_m | 0.2081 | 0.2176 | 0.0810 | 1.2909 | 0.1199 |
| sagittal_drift_m | 0.0037 | 0.1037 | -0.5594 | 0.3105 | 0.0101 |
| support_peak_m | 0.2308 | 0.2205 | 0.0937 | 1.3012 | 0.1401 |
| support_rms_m | 0.0860 | 0.0748 | 0.0373 | 0.4590 | 0.0672 |
| support_velocity_rms_m_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wheel_travel_asymmetry_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## E. Dynamic Height Tracking

#### Dynamic Height Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| dynamic_transition_smoothness | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| height_final_m | 0.4010 | 0.0677 | 0.2928 | 0.4904 | 0.4084 |
| height_initial_m | 0.3949 | 0.0638 | 0.2955 | 0.4810 | 0.3993 |
| height_max_m | 0.4096 | 0.0691 | 0.2955 | 0.4977 | 0.4166 |
| height_min_m | 0.3926 | 0.0649 | 0.2914 | 0.4809 | 0.3993 |
| height_overshoot_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| height_rmse_m | 0.0127 | 0.0222 | 0.0008 | 0.1123 | 0.0052 |
| height_tracking_lag_steps | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| height_undershoot_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_ref_tracking_error_rms | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## F. Torque Quality

#### Torque Quality Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| controller_conflict_index | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_hip_roll_nm | 0.4980 | 0.3529 | 0.1960 | 1.3358 | 0.3710 |
| torque_peak_hip_yaw_nm | 2.1153 | 0.9447 | 0.6552 | 5.1269 | 1.8894 |
| torque_peak_l_hip_pitch_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_hip_roll_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_hip_yaw_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_knee_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_wheel_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_legs_nm | 9.3097 | 1.7031 | 8.0000 | 13.7141 | 8.6681 |
| torque_peak_r_hip_pitch_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_hip_roll_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_hip_yaw_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_knee_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_wheel_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_total_nm | 9.3879 | 1.6972 | 8.0000 | 13.7141 | 8.6681 |
| torque_peak_wheels_nm | 5.0993 | 3.1072 | 1.4612 | 11.5753 | 3.5343 |
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
| long_run_drift_rate_m_per_kstep | 0.0288 | 0.0425 | 0.0024 | 0.2582 | 0.0184 |
| post_pitch_rms_500_deg | 1.3756 | 2.1670 | 0.0000 | 6.3304 | 0.0000 |
| post_push_active | 0.3077 | 0.4615 | 0.0000 | 1.0000 | 0.0000 |
| post_support_rms_500_m | 0.0453 | 0.0758 | 0.0000 | 0.3123 | 0.0000 |
| recovery_time_steps | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stability_score_0_to_1 | 0.7822 | 0.0490 | 0.6802 | 0.8847 | 0.7735 |

## Per-Scenario Detail

### Scope: dynamic_height

#### [SUM] gate_chatter_0p400_0p470 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 10.09 | pitch_min_deg | -2.73 |
| roll_max_deg | 0.29 | roll_min_deg | -0.58 |
| hip_yaw_joint_max_rad | 0.0917 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.74 | pitch_peak_deg | 10.09 |
| roll_rms_deg | 0.14 | roll_peak_deg | 0.58 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0917 | hip_yaw_div_rms_rad | 0.0817 |
| **Support/Drift** | | | |
| support_rms_m | 0.0785 | support_peak_m | 0.1848 |
| sagittal_drift_m | 0.042 | lateral_drift_m | -0.007 |
| final_displacement_m | 0.043 | max_displacement_m | 0.180 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0712 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.07 |
| **Robustness** | | | |
| stability_score | 0.754 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0085 | | |

#### [SUM] gate_dwell_0p420_0p450_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 10.87 | pitch_min_deg | -3.04 |
| roll_max_deg | 0.27 | roll_min_deg | -0.51 |
| hip_yaw_joint_max_rad | 0.2695 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 6.19 | pitch_peak_deg | 10.87 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.51 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.2695 | hip_yaw_div_rms_rad | 0.2603 |
| **Support/Drift** | | | |
| support_rms_m | 0.2996 | support_peak_m | 0.7650 |
| sagittal_drift_m | -0.559 | lateral_drift_m | -0.330 |
| final_displacement_m | 0.650 | max_displacement_m | 0.759 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0773 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.683 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.1083 | | |

#### [SUM] ramp_down_0p480_to_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.78 | pitch_min_deg | -1.95 |
| roll_max_deg | 0.30 | roll_min_deg | -0.47 |
| hip_yaw_joint_max_rad | 0.1191 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.03 | pitch_peak_deg | 8.78 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.47 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1191 | hip_yaw_div_rms_rad | 0.1156 |
| **Support/Drift** | | | |
| support_rms_m | 0.4590 | support_peak_m | 1.3012 |
| sagittal_drift_m | 0.311 | lateral_drift_m | 1.253 |
| final_displacement_m | 1.291 | max_displacement_m | 1.291 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.1123 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.30 |
| **Robustness** | | | |
| stability_score | 0.791 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.2582 | | |

#### [SUM] ramp_up_0p330_to_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.46 | pitch_min_deg | -7.14 |
| roll_max_deg | 0.90 | roll_min_deg | -0.21 |
| hip_yaw_joint_max_rad | 0.1117 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.71 | pitch_peak_deg | 8.46 |
| roll_rms_deg | 0.36 | roll_peak_deg | 0.90 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1117 | hip_yaw_div_rms_rad | 0.1063 |
| **Support/Drift** | | | |
| support_rms_m | 0.0675 | support_peak_m | 0.1347 |
| sagittal_drift_m | 0.026 | lateral_drift_m | 0.036 |
| final_displacement_m | 0.044 | max_displacement_m | 0.120 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0052 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.61 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.793 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0089 | | |

#### [SUM] up_down_cycle_0p330_0p480_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.80 | pitch_min_deg | -7.14 |
| roll_max_deg | 0.83 | roll_min_deg | -0.21 |
| hip_yaw_joint_max_rad | 0.1428 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.71 | pitch_peak_deg | 8.80 |
| roll_rms_deg | 0.28 | roll_peak_deg | 0.83 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1428 | hip_yaw_div_rms_rad | 0.1431 |
| **Support/Drift** | | | |
| support_rms_m | 0.0681 | support_peak_m | 0.1552 |
| sagittal_drift_m | 0.016 | lateral_drift_m | -0.050 |
| final_displacement_m | 0.052 | max_displacement_m | 0.151 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0058 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.61 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.798 | contact_loss_frac | 0.0001 |
| drift_rate_m_per_kstep | 0.0075 | | |

### Scope: long_run

#### [SUM] high_0p430 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.94 | pitch_min_deg | -2.19 |
| roll_max_deg | 0.68 | roll_min_deg | -0.04 |
| hip_yaw_joint_max_rad | 0.1087 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.74 | pitch_peak_deg | 7.94 |
| roll_rms_deg | 0.32 | roll_peak_deg | 0.68 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1087 | hip_yaw_div_rms_rad | 0.0810 |
| **Support/Drift** | | | |
| support_rms_m | 0.0653 | support_peak_m | 0.1474 |
| sagittal_drift_m | 0.034 | lateral_drift_m | -0.045 |
| final_displacement_m | 0.057 | max_displacement_m | 0.151 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.71 | torque_peak_wheels_nm | 2.96 |
| **Robustness** | | | |
| stability_score | 0.794 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0095 | | |

#### [SUM] high_0p450 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.99 | pitch_min_deg | -0.11 |
| roll_max_deg | 0.55 | roll_min_deg | -0.08 |
| hip_yaw_joint_max_rad | 0.1107 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.55 | pitch_peak_deg | 7.99 |
| roll_rms_deg | 0.20 | roll_peak_deg | 0.55 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1107 | hip_yaw_div_rms_rad | 0.0970 |
| **Support/Drift** | | | |
| support_rms_m | 0.0686 | support_peak_m | 0.1460 |
| sagittal_drift_m | 0.006 | lateral_drift_m | -0.142 |
| final_displacement_m | 0.142 | max_displacement_m | 0.149 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.77 | torque_peak_wheels_nm | 4.00 |
| **Robustness** | | | |
| stability_score | 0.761 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0236 | | |

#### [SUM] high_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.87 | pitch_min_deg | -1.56 |
| roll_max_deg | 0.28 | roll_min_deg | -0.54 |
| hip_yaw_joint_max_rad | 0.0995 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.69 | pitch_peak_deg | 8.87 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.54 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0995 | hip_yaw_div_rms_rad | 0.0823 |
| **Support/Drift** | | | |
| support_rms_m | 0.0721 | support_peak_m | 0.1368 |
| sagittal_drift_m | 0.052 | lateral_drift_m | 0.011 |
| final_displacement_m | 0.053 | max_displacement_m | 0.145 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.30 |
| **Robustness** | | | |
| stability_score | 0.758 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0088 | | |

#### [SUM] low_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 1.70 | roll_min_deg | -0.10 |
| hip_yaw_joint_max_rad | 0.1153 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 5.07 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 1.11 | roll_peak_deg | 1.70 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1153 | hip_yaw_div_rms_rad | 0.1272 |
| **Support/Drift** | | | |
| support_rms_m | 0.0373 | support_peak_m | 0.1230 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.030 |
| final_displacement_m | 0.032 | max_displacement_m | 0.117 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.67 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.680 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0053 | | |

#### [SUM] mid_0p400 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.74 | pitch_min_deg | -3.43 |
| roll_max_deg | 1.17 | roll_min_deg | -0.09 |
| hip_yaw_joint_max_rad | 0.1076 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 1.82 | pitch_peak_deg | 3.74 |
| roll_rms_deg | 0.44 | roll_peak_deg | 1.17 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1076 | hip_yaw_div_rms_rad | 0.1556 |
| **Support/Drift** | | | |
| support_rms_m | 0.0776 | support_peak_m | 0.1306 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.065 |
| final_displacement_m | 0.066 | max_displacement_m | 0.120 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| **Robustness** | | | |
| stability_score | 0.883 | contact_loss_frac | 0.0017 |
| drift_rate_m_per_kstep | 0.0110 | | |

### Scope: step_c

#### [SUM] C1_slow_ladder_up_down — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 1.15 | roll_min_deg | -0.10 |
| hip_yaw_joint_max_rad | 0.0736 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.96 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.47 | roll_peak_deg | 1.15 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0736 | hip_yaw_div_rms_rad | 0.0518 |
| **Support/Drift** | | | |
| support_rms_m | 0.0468 | support_peak_m | 0.1230 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.007 |
| final_displacement_m | 0.012 | max_displacement_m | 0.117 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.67 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.774 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0062 | | |

#### [SUM] C2_random_500dwell — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 1.15 | roll_min_deg | -0.10 |
| hip_yaw_joint_max_rad | 0.0736 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.96 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.47 | roll_peak_deg | 1.15 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0736 | hip_yaw_div_rms_rad | 0.0518 |
| **Support/Drift** | | | |
| support_rms_m | 0.0468 | support_peak_m | 0.1230 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.007 |
| final_displacement_m | 0.012 | max_displacement_m | 0.117 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.67 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.774 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0062 | | |

#### [SUM] C3_random_200dwell — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 1.15 | roll_min_deg | -0.10 |
| hip_yaw_joint_max_rad | 0.0736 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.96 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.47 | roll_peak_deg | 1.15 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0736 | hip_yaw_div_rms_rad | 0.0518 |
| **Support/Drift** | | | |
| support_rms_m | 0.0468 | support_peak_m | 0.1230 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.007 |
| final_displacement_m | 0.012 | max_displacement_m | 0.117 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.67 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.774 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0062 | | |

#### [SUM] C4_abrupt_stress — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 1.15 | roll_min_deg | -0.10 |
| hip_yaw_joint_max_rad | 0.0736 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.96 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.47 | roll_peak_deg | 1.15 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0736 | hip_yaw_div_rms_rad | 0.0518 |
| **Support/Drift** | | | |
| support_rms_m | 0.0468 | support_peak_m | 0.1230 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.007 |
| final_displacement_m | 0.012 | max_displacement_m | 0.117 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.67 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.774 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0062 | | |

#### [SUM] C5_long_random — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 1.63 | roll_min_deg | -0.10 |
| hip_yaw_joint_max_rad | 0.1115 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.51 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.83 | roll_peak_deg | 1.63 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1115 | hip_yaw_div_rms_rad | 0.0931 |
| **Support/Drift** | | | |
| support_rms_m | 0.0418 | support_peak_m | 0.1230 |
| sagittal_drift_m | 0.011 | lateral_drift_m | -0.018 |
| final_displacement_m | 0.021 | max_displacement_m | 0.117 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.67 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.724 | contact_loss_frac | 0.0003 |
| drift_rate_m_per_kstep | 0.0070 | | |

#### [SUM] focused_high_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.74 | pitch_min_deg | -1.34 |
| roll_max_deg | 0.20 | roll_min_deg | -0.54 |
| hip_yaw_joint_max_rad | 0.0505 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.28 | pitch_peak_deg | 8.74 |
| roll_rms_deg | 0.14 | roll_peak_deg | 0.54 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0505 | hip_yaw_div_rms_rad | 0.0326 |
| **Support/Drift** | | | |
| support_rms_m | 0.0581 | support_peak_m | 0.1229 |
| sagittal_drift_m | 0.015 | lateral_drift_m | 0.036 |
| final_displacement_m | 0.039 | max_displacement_m | 0.093 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.30 |
| **Robustness** | | | |
| stability_score | 0.778 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0196 | | |

#### [SUM] focused_low_0p320 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.62 | pitch_min_deg | -7.18 |
| roll_max_deg | 0.80 | roll_min_deg | -0.08 |
| hip_yaw_joint_max_rad | 0.0489 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.57 | pitch_peak_deg | 7.18 |
| roll_rms_deg | 0.29 | roll_peak_deg | 0.80 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0489 | hip_yaw_div_rms_rad | 0.0287 |
| **Support/Drift** | | | |
| support_rms_m | 0.0650 | support_peak_m | 0.1085 |
| sagittal_drift_m | 0.001 | lateral_drift_m | -0.078 |
| final_displacement_m | 0.078 | max_displacement_m | 0.098 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.73 | torque_peak_wheels_nm | 2.04 |
| **Robustness** | | | |
| stability_score | 0.804 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0390 | | |

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
| roll_max_deg | 0.93 | roll_min_deg | -0.19 |
| hip_yaw_joint_max_rad | 0.0715 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.41 | pitch_peak_deg | 11.64 |
| roll_rms_deg | 0.39 | roll_peak_deg | 0.93 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0715 | hip_yaw_div_rms_rad | 0.0516 |
| **Support/Drift** | | | |
| support_rms_m | 0.0627 | support_peak_m | 0.1682 |
| sagittal_drift_m | 0.004 | lateral_drift_m | -0.024 |
| final_displacement_m | 0.025 | max_displacement_m | 0.151 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.81 | torque_peak_wheels_nm | 6.68 |
| **Robustness** | | | |
| stability_score | 0.756 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0124 | | |

#### [SUM] low_0p330_sagittal_backward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.74 | pitch_min_deg | -14.55 |
| roll_max_deg | 0.91 | roll_min_deg | -0.27 |
| hip_yaw_joint_max_rad | 0.0891 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.71 | pitch_peak_deg | 14.55 |
| roll_rms_deg | 0.41 | roll_peak_deg | 0.91 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0891 | hip_yaw_div_rms_rad | 0.0618 |
| **Support/Drift** | | | |
| support_rms_m | 0.0735 | support_peak_m | 0.2656 |
| sagittal_drift_m | 0.006 | lateral_drift_m | -0.006 |
| final_displacement_m | 0.009 | max_displacement_m | 0.201 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 10.39 | torque_peak_wheels_nm | 10.39 |
| **Robustness** | | | |
| stability_score | 0.740 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0045 | | |

#### [SUM] low_0p330_sagittal_forward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.47 | pitch_min_deg | -7.14 |
| roll_max_deg | 1.64 | roll_min_deg | -0.54 |
| hip_yaw_joint_max_rad | 0.1125 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.62 | pitch_peak_deg | 7.47 |
| roll_rms_deg | 0.84 | roll_peak_deg | 1.64 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1125 | hip_yaw_div_rms_rad | 0.0989 |
| **Support/Drift** | | | |
| support_rms_m | 0.0770 | support_peak_m | 0.3226 |
| sagittal_drift_m | -0.004 | lateral_drift_m | -0.012 |
| final_displacement_m | 0.013 | max_displacement_m | 0.296 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 10.67 | torque_peak_wheels_nm | 7.05 |
| **Robustness** | | | |
| stability_score | 0.719 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0064 | | |

#### [SUM] low_0p330_sagittal_forward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 9.90 | pitch_min_deg | -8.24 |
| roll_max_deg | 1.04 | roll_min_deg | -0.50 |
| hip_yaw_joint_max_rad | 0.1692 | contact_loss_steps | 4 |
| **Posture** | | | |
| pitch_rms_deg | 4.40 | pitch_peak_deg | 9.90 |
| roll_rms_deg | 0.55 | roll_peak_deg | 1.04 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1692 | hip_yaw_div_rms_rad | 0.1003 |
| **Support/Drift** | | | |
| support_rms_m | 0.1076 | support_peak_m | 0.4354 |
| sagittal_drift_m | -0.023 | lateral_drift_m | -0.061 |
| final_displacement_m | 0.066 | max_displacement_m | 0.400 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 13.71 | torque_peak_wheels_nm | 11.58 |
| **Robustness** | | | |
| stability_score | 0.747 | contact_loss_frac | 0.0020 |
| drift_rate_m_per_kstep | 0.0328 | | |

#### [SUM] mid_0p400_sagittal_backward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 2.81 | pitch_min_deg | -6.02 |
| roll_max_deg | 1.05 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.1175 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 1.65 | pitch_peak_deg | 6.02 |
| roll_rms_deg | 0.54 | roll_peak_deg | 1.05 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1175 | hip_yaw_div_rms_rad | 0.1003 |
| **Support/Drift** | | | |
| support_rms_m | 0.1240 | support_peak_m | 0.3691 |
| sagittal_drift_m | 0.025 | lateral_drift_m | -0.049 |
| final_displacement_m | 0.055 | max_displacement_m | 0.329 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| **Robustness** | | | |
| stability_score | 0.885 | contact_loss_frac | 0.0050 |
| drift_rate_m_per_kstep | 0.0274 | | |

#### [SUM] mid_0p400_sagittal_backward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.28 | pitch_min_deg | -8.28 |
| roll_max_deg | 1.08 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.1284 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 1.96 | pitch_peak_deg | 8.28 |
| roll_rms_deg | 0.52 | roll_peak_deg | 1.08 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1284 | hip_yaw_div_rms_rad | 0.1187 |
| **Support/Drift** | | | |
| support_rms_m | 0.1728 | support_peak_m | 0.5108 |
| sagittal_drift_m | 0.029 | lateral_drift_m | -0.094 |
| final_displacement_m | 0.098 | max_displacement_m | 0.465 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| **Robustness** | | | |
| stability_score | 0.871 | contact_loss_frac | 0.0050 |
| drift_rate_m_per_kstep | 0.0491 | | |

#### [SUM] mid_0p400_sagittal_forward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 9.31 | pitch_min_deg | -3.74 |
| roll_max_deg | 1.11 | roll_min_deg | -0.18 |
| hip_yaw_joint_max_rad | 0.1154 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 2.21 | pitch_peak_deg | 9.31 |
| roll_rms_deg | 0.55 | roll_peak_deg | 1.11 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1154 | hip_yaw_div_rms_rad | 0.0954 |
| **Support/Drift** | | | |
| support_rms_m | 0.0948 | support_peak_m | 0.2287 |
| sagittal_drift_m | 0.009 | lateral_drift_m | -0.066 |
| final_displacement_m | 0.066 | max_displacement_m | 0.194 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| **Robustness** | | | |
| stability_score | 0.856 | contact_loss_frac | 0.0050 |
| drift_rate_m_per_kstep | 0.0332 | | |

#### [SUM] mid_0p400_sagittal_forward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 11.77 | pitch_min_deg | -3.43 |
| roll_max_deg | 1.00 | roll_min_deg | -0.46 |
| hip_yaw_joint_max_rad | 0.1172 | contact_loss_steps | 11 |
| **Posture** | | | |
| pitch_rms_deg | 2.74 | pitch_peak_deg | 11.77 |
| roll_rms_deg | 0.48 | roll_peak_deg | 1.00 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1172 | hip_yaw_div_rms_rad | 0.0902 |
| **Support/Drift** | | | |
| support_rms_m | 0.0951 | support_peak_m | 0.2531 |
| sagittal_drift_m | -0.006 | lateral_drift_m | -0.097 |
| final_displacement_m | 0.098 | max_displacement_m | 0.201 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.42 |
| **Robustness** | | | |
| stability_score | 0.834 | contact_loss_frac | 0.0055 |
| drift_rate_m_per_kstep | 0.0488 | | |

### Scope: step_e

#### [SUM] high_0p430 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.19 | pitch_min_deg | -1.95 |
| roll_max_deg | 0.68 | roll_min_deg | -0.03 |
| hip_yaw_joint_max_rad | 0.0525 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.13 | pitch_peak_deg | 7.19 |
| roll_rms_deg | 0.35 | roll_peak_deg | 0.68 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0525 | hip_yaw_div_rms_rad | 0.0385 |
| **Support/Drift** | | | |
| support_rms_m | 0.0500 | support_peak_m | 0.1401 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.085 |
| final_displacement_m | 0.085 | max_displacement_m | 0.103 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.71 | torque_peak_wheels_nm | 2.96 |
| **Robustness** | | | |
| stability_score | 0.822 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0427 | | |

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
| pitch_max_deg | 7.24 | pitch_min_deg | -2.20 |
| roll_max_deg | 0.34 | roll_min_deg | -0.25 |
| hip_yaw_joint_max_rad | 0.0314 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.63 | pitch_peak_deg | 7.24 |
| roll_rms_deg | 0.14 | roll_peak_deg | 0.34 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0314 | hip_yaw_div_rms_rad | 0.0186 |
| **Support/Drift** | | | |
| support_rms_m | 0.0634 | support_peak_m | 0.1220 |
| sagittal_drift_m | 0.014 | lateral_drift_m | 0.076 |
| final_displacement_m | 0.078 | max_displacement_m | 0.093 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.04 | torque_peak_wheels_nm | 1.78 |
| **Robustness** | | | |
| stability_score | 0.810 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0389 | | |

#### [SUM] high_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.74 | pitch_min_deg | -1.34 |
| roll_max_deg | 0.20 | roll_min_deg | -0.54 |
| hip_yaw_joint_max_rad | 0.0505 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.28 | pitch_peak_deg | 8.74 |
| roll_rms_deg | 0.14 | roll_peak_deg | 0.54 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0505 | hip_yaw_div_rms_rad | 0.0326 |
| **Support/Drift** | | | |
| support_rms_m | 0.0581 | support_peak_m | 0.1229 |
| sagittal_drift_m | 0.015 | lateral_drift_m | 0.036 |
| final_displacement_m | 0.039 | max_displacement_m | 0.093 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.30 |
| **Robustness** | | | |
| stability_score | 0.778 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0196 | | |

#### [SUM] low_0p300 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.12 | pitch_min_deg | -1.38 |
| roll_max_deg | 0.95 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.1115 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 2.91 | pitch_peak_deg | 7.12 |
| roll_rms_deg | 0.63 | roll_peak_deg | 0.95 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1115 | hip_yaw_div_rms_rad | 0.0917 |
| **Support/Drift** | | | |
| support_rms_m | 0.0419 | support_peak_m | 0.0960 |
| sagittal_drift_m | 0.005 | lateral_drift_m | -0.001 |
| final_displacement_m | 0.005 | max_displacement_m | 0.084 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 9.02 | torque_peak_wheels_nm | 1.46 |
| **Robustness** | | | |
| stability_score | 0.817 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0024 | | |

#### [SUM] low_0p320 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.62 | pitch_min_deg | -7.18 |
| roll_max_deg | 0.80 | roll_min_deg | -0.08 |
| hip_yaw_joint_max_rad | 0.0489 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.57 | pitch_peak_deg | 7.18 |
| roll_rms_deg | 0.29 | roll_peak_deg | 0.80 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0489 | hip_yaw_div_rms_rad | 0.0287 |
| **Support/Drift** | | | |
| support_rms_m | 0.0650 | support_peak_m | 0.1085 |
| sagittal_drift_m | 0.001 | lateral_drift_m | -0.078 |
| final_displacement_m | 0.078 | max_displacement_m | 0.098 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.73 | torque_peak_wheels_nm | 2.04 |
| **Robustness** | | | |
| stability_score | 0.804 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0390 | | |

#### [SUM] low_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 1.15 | roll_min_deg | -0.10 |
| hip_yaw_joint_max_rad | 0.0736 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.96 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.47 | roll_peak_deg | 1.15 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0736 | hip_yaw_div_rms_rad | 0.0518 |
| **Support/Drift** | | | |
| support_rms_m | 0.0468 | support_peak_m | 0.1230 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.007 |
| final_displacement_m | 0.012 | max_displacement_m | 0.117 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.67 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.774 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0062 | | |

#### [SUM] low_0p340 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 4.55 | pitch_min_deg | -3.05 |
| roll_max_deg | 1.11 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.0799 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 1.86 | pitch_peak_deg | 4.55 |
| roll_rms_deg | 0.56 | roll_peak_deg | 1.11 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0799 | hip_yaw_div_rms_rad | 0.0516 |
| **Support/Drift** | | | |
| support_rms_m | 0.0501 | support_peak_m | 0.1084 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.012 |
| final_displacement_m | 0.016 | max_displacement_m | 0.110 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.62 | torque_peak_wheels_nm | 1.95 |
| **Robustness** | | | |
| stability_score | 0.873 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0080 | | |

#### [SUM] low_0p360 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 2.46 | pitch_min_deg | -6.57 |
| roll_max_deg | 0.77 | roll_min_deg | -0.13 |
| hip_yaw_joint_max_rad | 0.0542 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.12 | pitch_peak_deg | 6.57 |
| roll_rms_deg | 0.28 | roll_peak_deg | 0.77 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0542 | hip_yaw_div_rms_rad | 0.0302 |
| **Support/Drift** | | | |
| support_rms_m | 0.0531 | support_peak_m | 0.0937 |
| sagittal_drift_m | 0.006 | lateral_drift_m | 0.057 |
| final_displacement_m | 0.057 | max_displacement_m | 0.081 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.55 | torque_peak_wheels_nm | 1.84 |
| **Robustness** | | | |
| stability_score | 0.827 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0287 | | |

#### [SUM] low_0p380 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 9.03 | pitch_min_deg | -0.45 |
| roll_max_deg | 0.83 | roll_min_deg | -0.12 |
| hip_yaw_joint_max_rad | 0.0453 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 5.24 | pitch_peak_deg | 9.03 |
| roll_rms_deg | 0.32 | roll_peak_deg | 0.83 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0453 | hip_yaw_div_rms_rad | 0.0216 |
| **Support/Drift** | | | |
| support_rms_m | 0.0672 | support_peak_m | 0.1169 |
| sagittal_drift_m | 0.005 | lateral_drift_m | 0.005 |
| final_displacement_m | 0.008 | max_displacement_m | 0.089 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 9.16 | torque_peak_wheels_nm | 3.13 |
| **Robustness** | | | |
| stability_score | 0.718 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0038 | | |

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

Full metrics exported to: `docs\validation\k2_default_v1_drift_candidate_quality.json`