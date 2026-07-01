# K2 JAX Dedicated Realtime — Behavior Quality Baseline

**Generated:** 2026-06-30T10:49:17.408597
**Input:** `outputs\k2_improvement_baseline`
**input_dir:** outputs\k2_improvement_baseline
**analyzer_version:** 1.0.0
**baseline_type:** K2_JAX_DEDICATED_REALTIME_IMPROVEMENT_BASELINE

## Executive Summary

- **Total scenarios:** 39
- **Falls:** 0 (none)
- **Scenarios with full telemetry:** 9/39
- **Performance:** 147.4 Hz avg (min 59.3, max 199.2)

## A. Safety — Hard Gates

#### Safety Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| contact_loss_steps | 2.2821 | 3.0797 | 1.0000 | 11.0000 | 1.0000 |
| fall_step | -1.0000 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| fell | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hip_yaw_joint_max_rad | 0.0860 | 0.0479 | 0.0155 | 0.2692 | 0.0799 |
| nan_inf_detected | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pitch_max_deg | 6.5382 | 3.3816 | 1.8240 | 13.8442 | 7.4669 |
| pitch_min_deg | -4.8549 | 3.2645 | -14.5263 | -0.0512 | -4.2521 |
| roll_max_deg | 0.8041 | 0.4279 | 0.1722 | 1.6971 | 0.8845 |
| roll_min_deg | -0.2180 | 0.1857 | -0.5823 | 0.0000 | -0.1342 |

## B. Posture Stability

#### Posture Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| angular_velocity_rms_deg_s | 1.8449 | 3.4494 | 0.0000 | 10.5214 | 0.0000 |
| orientation_energy_integral | 416.2524 | 821.4177 | 0.0000 | 3256.9748 | 0.0000 |
| pitch_peak_deg | 8.4540 | 2.1290 | 3.7218 | 14.5263 | 7.9876 |
| pitch_rate_rms_deg_s | 1.8248 | 3.4126 | 0.0000 | 10.3994 | 0.0000 |
| pitch_rms_deg | 3.9158 | 1.0225 | 1.6634 | 6.1883 | 3.9627 |
| pitch_settling_steps | 503.5128 | 1278.2905 | 0.0000 | 6000.0000 | 0.0000 |
| roll_peak_deg | 0.8520 | 0.3736 | 0.2541 | 1.6971 | 0.8845 |
| roll_rate_rms_deg_s | 0.1092 | 0.2186 | 0.0000 | 0.8447 | 0.0000 |
| roll_rms_deg | 0.3742 | 0.2299 | 0.1072 | 1.1086 | 0.3535 |
| yaw_drift_deg | 6.1539 | 10.4826 | -14.5345 | 52.8563 | 5.4500 |
| yaw_rate_rms_deg_s | 0.2449 | 0.4548 | 0.0000 | 1.3557 | 0.0000 |

### Pitch RMS by Height Region

| Region | Mean Pitch RMS (deg) | Std | Count |
|--------|----------------------|-----|-------|
| low | 3.89 | 0.84 | 12 |
| mid | 3.28 | 1.59 | 9 |
| high | 4.31 | 0.52 | 13 |

### Pitch RMS by Scenario Type

| Type | Mean Pitch RMS (deg) | Std | Count |
|------|----------------------|-----|-------|
| fixed_height | 3.78 | 0.97 | 17 |
| push | 3.81 | 1.22 | 12 |
| dynamic_height | 4.46 | 0.95 | 5 |

## C. Leg Symmetry / Twist

#### Leg Symmetry Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| hip_pitch_symmetry_error_deg | 0.2650 | 0.4891 | 0.0000 | 1.4296 | 0.0000 |
| hip_roll_symmetry_error_deg | 2.0634 | 4.7294 | 0.0000 | 19.2183 | 0.0000 |
| hip_yaw_div_max_rad | 0.1495 | 0.0947 | 0.0263 | 0.5370 | 0.1269 |
| hip_yaw_div_rms_rad | 0.0699 | 0.0489 | 0.0111 | 0.2602 | 0.0518 |
| hip_yaw_joint_max_rad | 0.0860 | 0.0479 | 0.0155 | 0.2692 | 0.0799 |
| hip_yaw_lr_divergence_deg | 0.8788 | 1.9806 | 0.0000 | 8.2454 | 0.0000 |
| knee_symmetry_error_deg | 0.1664 | 0.3648 | 0.0000 | 1.5092 | 0.0000 |
| leg_posture_error_rms | 0.0574 | 0.1718 | 0.0000 | 1.0264 | 0.0000 |

## D. Support / Drift

#### Support & Drift Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| com_support_offset_rms_m | 0.0042 | 0.0081 | 0.0000 | 0.0342 | 0.0000 |
| final_displacement_m | 0.0982 | 0.2178 | 0.0049 | 1.2909 | 0.0535 |
| lateral_drift_m | -0.0016 | 0.2153 | -0.3304 | 1.2530 | -0.0175 |
| max_displacement_m | 0.2085 | 0.2175 | 0.0810 | 1.2909 | 0.1199 |
| sagittal_drift_m | 0.0027 | 0.1036 | -0.5597 | 0.3105 | 0.0086 |
| support_peak_m | 0.2316 | 0.2202 | 0.0937 | 1.3012 | 0.1399 |
| support_rms_m | 0.0863 | 0.0748 | 0.0373 | 0.4590 | 0.0686 |
| support_velocity_rms_m_s | 0.0388 | 0.0722 | 0.0000 | 0.2069 | 0.0000 |
| wheel_travel_asymmetry_m | 0.0140 | 0.0312 | 0.0000 | 0.1530 | 0.0000 |

## E. Dynamic Height Tracking

#### Dynamic Height Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| dynamic_transition_smoothness | 3.7665 | 8.8141 | 0.0000 | 44.5868 | 0.0000 |
| height_final_m | 0.4011 | 0.0677 | 0.2928 | 0.4904 | 0.4084 |
| height_initial_m | 0.3949 | 0.0638 | 0.2955 | 0.4810 | 0.3993 |
| height_max_m | 0.4096 | 0.0692 | 0.2955 | 0.4977 | 0.4166 |
| height_min_m | 0.3926 | 0.0649 | 0.2914 | 0.4809 | 0.3993 |
| height_overshoot_m | 0.0060 | 0.0249 | 0.0000 | 0.1575 | 0.0000 |
| height_rmse_m | 0.0127 | 0.0222 | 0.0008 | 0.1123 | 0.0055 |
| height_tracking_lag_steps | 5.9744 | 15.3664 | 0.0000 | 49.0000 | 0.0000 |
| height_undershoot_m | -0.0002 | 0.0012 | -0.0047 | 0.0022 | 0.0000 |
| q_ref_tracking_error_rms | 0.0428 | 0.1614 | 0.0000 | 1.0118 | 0.0000 |

## F. Torque Quality

#### Torque Quality Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| controller_conflict_index | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_hip_roll_nm | 0.4958 | 0.3547 | 0.1564 | 1.3358 | 0.3574 |
| torque_peak_hip_yaw_nm | 2.1158 | 0.9513 | 0.6552 | 5.2169 | 1.8894 |
| torque_peak_l_hip_pitch_nm | 0.6159 | 1.1977 | 0.0000 | 3.9270 | 0.0000 |
| torque_peak_l_hip_roll_nm | 0.1259 | 0.3087 | 0.0000 | 1.3358 | 0.0000 |
| torque_peak_l_hip_yaw_nm | 0.4224 | 0.8314 | 0.0000 | 2.8312 | 0.0000 |
| torque_peak_l_knee_nm | 2.1473 | 3.9894 | 0.0000 | 12.1224 | 0.0000 |
| torque_peak_l_wheel_nm | 1.2529 | 2.8370 | 0.0000 | 10.3912 | 0.0000 |
| torque_peak_legs_nm | 9.3093 | 1.7032 | 8.0000 | 13.7141 | 8.6681 |
| torque_peak_r_hip_pitch_nm | 0.5498 | 1.1079 | 0.0000 | 3.8202 | 0.0000 |
| torque_peak_r_hip_roll_nm | 0.1230 | 0.3033 | 0.0000 | 1.3146 | 0.0000 |
| torque_peak_r_hip_yaw_nm | 0.4618 | 0.9226 | 0.0000 | 3.0408 | 0.0000 |
| torque_peak_r_knee_nm | 2.1896 | 4.0979 | 0.0000 | 12.9541 | 0.0000 |
| torque_peak_r_wheel_nm | 1.2509 | 2.8358 | 0.0000 | 10.3802 | 0.0000 |
| torque_peak_total_nm | 9.3877 | 1.6973 | 8.0000 | 13.7141 | 8.6681 |
| torque_peak_wheels_nm | 5.0992 | 3.1072 | 1.4612 | 11.5753 | 3.5343 |
| torque_rate_peak_nm_s | 92.3077 | 168.5300 | 0.0000 | 400.0000 | 0.0000 |
| torque_rate_rms_nm_s | 2.9830 | 5.9548 | 0.0000 | 23.5272 | 0.0000 |
| torque_rms_l_hip_pitch_nm | 0.1784 | 0.3312 | 0.0000 | 0.9651 | 0.0000 |
| torque_rms_l_hip_roll_nm | 0.0260 | 0.0514 | 0.0000 | 0.1912 | 0.0000 |
| torque_rms_l_hip_yaw_nm | 0.1921 | 0.4183 | 0.0000 | 1.9632 | 0.0000 |
| torque_rms_l_knee_nm | 1.5823 | 2.9411 | 0.0000 | 8.2844 | 0.0000 |
| torque_rms_l_wheel_nm | 0.0685 | 0.1345 | 0.0000 | 0.5099 | 0.0000 |
| torque_rms_r_hip_pitch_nm | 0.2121 | 0.3893 | 0.0000 | 1.0338 | 0.0000 |
| torque_rms_r_hip_roll_nm | 0.0254 | 0.0504 | 0.0000 | 0.1928 | 0.0000 |
| torque_rms_r_hip_yaw_nm | 0.2150 | 0.4670 | 0.0000 | 2.1241 | 0.0000 |
| torque_rms_r_knee_nm | 1.5448 | 2.8642 | 0.0000 | 8.2216 | 0.0000 |
| torque_rms_r_wheel_nm | 0.0685 | 0.1346 | 0.0000 | 0.5120 | 0.0000 |
| torque_saturation_count | 238.5128 | 587.1243 | 0.0000 | 2000.0000 | 0.0000 |

## G. Robustness

#### Robustness Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| contact_loss_frac | 0.0010 | 0.0014 | 0.0001 | 0.0055 | 0.0005 |
| contact_loss_steps | 2.2821 | 3.0797 | 1.0000 | 11.0000 | 1.0000 |
| long_run_drift_rate_m_per_kstep | 0.0291 | 0.0424 | 0.0024 | 0.2582 | 0.0184 |
| post_pitch_rms_500_deg | 1.3723 | 2.1610 | 0.0000 | 6.3163 | 0.0000 |
| post_push_active | 0.3077 | 0.4615 | 0.0000 | 1.0000 | 0.0000 |
| post_support_rms_500_m | 0.0454 | 0.0759 | 0.0000 | 0.3129 | 0.0000 |
| recovery_time_steps | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stability_score_0_to_1 | 0.7818 | 0.0492 | 0.6802 | 0.8844 | 0.7735 |

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
| hip_yaw_joint_max_rad | 0.2692 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 6.19 | pitch_peak_deg | 10.87 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.51 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.2692 | hip_yaw_div_rms_rad | 0.2602 |
| **Support/Drift** | | | |
| support_rms_m | 0.2996 | support_peak_m | 0.7650 |
| sagittal_drift_m | -0.560 | lateral_drift_m | -0.330 |
| final_displacement_m | 0.650 | max_displacement_m | 0.759 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0773 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.683 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.1083 | | |

#### [TEL] ramp_down_0p480_to_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.78 | pitch_min_deg | -1.95 |
| roll_max_deg | 0.30 | roll_min_deg | -0.47 |
| hip_yaw_joint_max_rad | 0.1191 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.03 | pitch_peak_deg | 8.78 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.47 |
| pitch_rate_rms_deg_s | 6.58 | pitch_settling_steps | 5000 |
| angular_velocity_rms_deg_s | 6.64 | yaw_drift_deg | -14.53 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1191 | hip_yaw_div_rms_rad | 0.1156 |
| hip_yaw_lr_divergence_deg | 6.62 | hip_pitch_symmetry_error_deg | 1.31 |
| knee_symmetry_error_deg | 0.52 | leg_posture_error_rms | 0.1425 |
| **Support/Drift** | | | |
| support_rms_m | 0.4590 | support_peak_m | 1.3012 |
| sagittal_drift_m | 0.311 | lateral_drift_m | 1.253 |
| final_displacement_m | 1.291 | max_displacement_m | 1.291 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.1123 | height_overshoot_m | 0.1575 |
| height_undershoot_m | 0.0009 | tracking_lag_steps | 0 |
| q_ref_tracking_error_rms | 0.1107 | transition_smoothness | 8.9484 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.30 |
| torque_rate_rms_nm_s | 7.9 | torque_saturation_count | 395 |
| **Robustness** | | | |
| stability_score | 0.791 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.2582 | | |

#### [TEL] ramp_up_0p330_to_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.86 | pitch_min_deg | -7.14 |
| roll_max_deg | 0.89 | roll_min_deg | -0.34 |
| hip_yaw_joint_max_rad | 0.0946 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.61 | pitch_peak_deg | 8.86 |
| roll_rms_deg | 0.39 | roll_peak_deg | 0.89 |
| pitch_rate_rms_deg_s | 7.99 | pitch_settling_steps | 1283 |
| angular_velocity_rms_deg_s | 8.07 | yaw_drift_deg | -5.93 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0946 | hip_yaw_div_rms_rad | 0.0692 |
| hip_yaw_lr_divergence_deg | 3.96 | hip_pitch_symmetry_error_deg | 1.17 |
| knee_symmetry_error_deg | 1.03 | leg_posture_error_rms | 1.0264 |
| **Support/Drift** | | | |
| support_rms_m | 0.0647 | support_peak_m | 0.1364 |
| sagittal_drift_m | 0.007 | lateral_drift_m | 0.052 |
| final_displacement_m | 0.053 | max_displacement_m | 0.117 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0062 | height_overshoot_m | 0.0102 |
| height_undershoot_m | -0.0047 | tracking_lag_steps | 49 |
| q_ref_tracking_error_rms | 1.0118 | transition_smoothness | 6.3058 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.61 | torque_peak_wheels_nm | 3.53 |
| torque_rate_rms_nm_s | 9.1 | torque_saturation_count | 1526 |
| **Robustness** | | | |
| stability_score | 0.796 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0106 | | |

#### [SUM] up_down_cycle_0p330_0p480_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.76 | pitch_min_deg | -7.14 |
| roll_max_deg | 0.80 | roll_min_deg | -0.22 |
| hip_yaw_joint_max_rad | 0.1464 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.73 | pitch_peak_deg | 8.76 |
| roll_rms_deg | 0.29 | roll_peak_deg | 0.80 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1464 | hip_yaw_div_rms_rad | 0.1520 |
| **Support/Drift** | | | |
| support_rms_m | 0.0690 | support_peak_m | 0.1532 |
| sagittal_drift_m | 0.016 | lateral_drift_m | -0.056 |
| final_displacement_m | 0.058 | max_displacement_m | 0.145 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0055 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.61 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.796 | contact_loss_frac | 0.0001 |
| drift_rate_m_per_kstep | 0.0083 | | |

### Scope: long_run

#### [SUM] high_0p430 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.95 | pitch_min_deg | -2.21 |
| roll_max_deg | 0.69 | roll_min_deg | -0.05 |
| hip_yaw_joint_max_rad | 0.1096 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.77 | pitch_peak_deg | 7.95 |
| roll_rms_deg | 0.32 | roll_peak_deg | 0.69 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1096 | hip_yaw_div_rms_rad | 0.0835 |
| **Support/Drift** | | | |
| support_rms_m | 0.0660 | support_peak_m | 0.1503 |
| sagittal_drift_m | 0.025 | lateral_drift_m | -0.081 |
| final_displacement_m | 0.085 | max_displacement_m | 0.153 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.71 | torque_peak_wheels_nm | 2.96 |
| **Robustness** | | | |
| stability_score | 0.793 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0142 | | |

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

#### [TEL] mid_0p400 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.72 | pitch_min_deg | -3.43 |
| roll_max_deg | 1.15 | roll_min_deg | -0.07 |
| hip_yaw_joint_max_rad | 0.1152 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 1.75 | pitch_peak_deg | 3.72 |
| roll_rms_deg | 0.47 | roll_peak_deg | 1.15 |
| pitch_rate_rms_deg_s | 4.98 | pitch_settling_steps | 6000 |
| angular_velocity_rms_deg_s | 5.05 | yaw_drift_deg | -5.39 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1152 | hip_yaw_div_rms_rad | 0.1439 |
| hip_yaw_lr_divergence_deg | 8.25 | hip_pitch_symmetry_error_deg | 1.14 |
| knee_symmetry_error_deg | 1.25 | leg_posture_error_rms | 0.2630 |
| **Support/Drift** | | | |
| support_rms_m | 0.0772 | support_peak_m | 0.1302 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.059 |
| final_displacement_m | 0.060 | max_displacement_m | 0.120 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| torque_rate_rms_nm_s | 17.3 | torque_saturation_count | 1098 |
| **Robustness** | | | |
| stability_score | 0.884 | contact_loss_frac | 0.0017 |
| drift_rate_m_per_kstep | 0.0100 | | |

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
| pitch_max_deg | 4.08 | pitch_min_deg | -7.44 |
| roll_max_deg | 0.80 | roll_min_deg | -0.07 |
| hip_yaw_joint_max_rad | 0.0494 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.69 | pitch_peak_deg | 7.44 |
| roll_rms_deg | 0.29 | roll_peak_deg | 0.80 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0494 | hip_yaw_div_rms_rad | 0.0280 |
| **Support/Drift** | | | |
| support_rms_m | 0.0692 | support_peak_m | 0.1183 |
| sagittal_drift_m | -0.001 | lateral_drift_m | -0.072 |
| final_displacement_m | 0.072 | max_displacement_m | 0.105 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.78 | torque_peak_wheels_nm | 2.04 |
| **Robustness** | | | |
| stability_score | 0.798 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0362 | | |

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
| pitch_max_deg | 3.87 | pitch_min_deg | -11.63 |
| roll_max_deg | 0.95 | roll_min_deg | -0.16 |
| hip_yaw_joint_max_rad | 0.0731 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.38 | pitch_peak_deg | 11.63 |
| roll_rms_deg | 0.39 | roll_peak_deg | 0.95 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0731 | hip_yaw_div_rms_rad | 0.0502 |
| **Support/Drift** | | | |
| support_rms_m | 0.0624 | support_peak_m | 0.1719 |
| sagittal_drift_m | 0.004 | lateral_drift_m | -0.027 |
| final_displacement_m | 0.027 | max_displacement_m | 0.154 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.70 | torque_peak_wheels_nm | 6.68 |
| **Robustness** | | | |
| stability_score | 0.758 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0136 | | |

#### [TEL] low_0p330_sagittal_backward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.78 | pitch_min_deg | -14.53 |
| roll_max_deg | 0.99 | roll_min_deg | -0.28 |
| hip_yaw_joint_max_rad | 0.0909 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.80 | pitch_peak_deg | 14.53 |
| roll_rms_deg | 0.46 | roll_peak_deg | 0.99 |
| pitch_rate_rms_deg_s | 10.40 | pitch_settling_steps | 1061 |
| angular_velocity_rms_deg_s | 10.52 | yaw_drift_deg | -3.66 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0909 | hip_yaw_div_rms_rad | 0.0702 |
| hip_yaw_lr_divergence_deg | 4.02 | hip_pitch_symmetry_error_deg | 0.98 |
| knee_symmetry_error_deg | 0.75 | leg_posture_error_rms | 0.1486 |
| **Support/Drift** | | | |
| support_rms_m | 0.0736 | support_peak_m | 0.2657 |
| sagittal_drift_m | 0.007 | lateral_drift_m | -0.006 |
| final_displacement_m | 0.009 | max_displacement_m | 0.201 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 10.39 | torque_peak_wheels_nm | 10.39 |
| torque_rate_rms_nm_s | 17.4 | torque_saturation_count | 1984 |
| **Robustness** | | | |
| stability_score | 0.732 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0044 | | |

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
| pitch_max_deg | 9.86 | pitch_min_deg | -7.84 |
| roll_max_deg | 1.13 | roll_min_deg | -0.49 |
| hip_yaw_joint_max_rad | 0.1699 | contact_loss_steps | 4 |
| **Posture** | | | |
| pitch_rms_deg | 4.37 | pitch_peak_deg | 9.86 |
| roll_rms_deg | 0.56 | roll_peak_deg | 1.13 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1699 | hip_yaw_div_rms_rad | 0.0958 |
| **Support/Drift** | | | |
| support_rms_m | 0.1080 | support_peak_m | 0.4380 |
| sagittal_drift_m | -0.023 | lateral_drift_m | -0.062 |
| final_displacement_m | 0.066 | max_displacement_m | 0.403 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 13.71 | torque_peak_wheels_nm | 11.58 |
| **Robustness** | | | |
| stability_score | 0.748 | contact_loss_frac | 0.0020 |
| drift_rate_m_per_kstep | 0.0332 | | |

#### [SUM] mid_0p400_sagittal_backward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 2.84 | pitch_min_deg | -6.03 |
| roll_max_deg | 1.06 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.1201 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 1.66 | pitch_peak_deg | 6.03 |
| roll_rms_deg | 0.54 | roll_peak_deg | 1.06 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1201 | hip_yaw_div_rms_rad | 0.1029 |
| **Support/Drift** | | | |
| support_rms_m | 0.1244 | support_peak_m | 0.3701 |
| sagittal_drift_m | 0.025 | lateral_drift_m | -0.047 |
| final_displacement_m | 0.053 | max_displacement_m | 0.330 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| **Robustness** | | | |
| stability_score | 0.884 | contact_loss_frac | 0.0050 |
| drift_rate_m_per_kstep | 0.0267 | | |

#### [SUM] mid_0p400_sagittal_backward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.29 | pitch_min_deg | -8.29 |
| roll_max_deg | 1.07 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.1273 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 1.94 | pitch_peak_deg | 8.29 |
| roll_rms_deg | 0.50 | roll_peak_deg | 1.07 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1273 | hip_yaw_div_rms_rad | 0.1196 |
| **Support/Drift** | | | |
| support_rms_m | 0.1730 | support_peak_m | 0.5118 |
| sagittal_drift_m | 0.025 | lateral_drift_m | -0.119 |
| final_displacement_m | 0.121 | max_displacement_m | 0.466 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| **Robustness** | | | |
| stability_score | 0.873 | contact_loss_frac | 0.0050 |
| drift_rate_m_per_kstep | 0.0607 | | |

#### [TEL] mid_0p400_sagittal_forward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 9.29 | pitch_min_deg | -3.73 |
| roll_max_deg | 1.16 | roll_min_deg | -0.18 |
| hip_yaw_joint_max_rad | 0.1163 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 2.19 | pitch_peak_deg | 9.29 |
| roll_rms_deg | 0.56 | roll_peak_deg | 1.16 |
| pitch_rate_rms_deg_s | 6.98 | pitch_settling_steps | 903 |
| angular_velocity_rms_deg_s | 7.09 | yaw_drift_deg | -5.36 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1163 | hip_yaw_div_rms_rad | 0.0952 |
| hip_yaw_lr_divergence_deg | 5.45 | hip_pitch_symmetry_error_deg | 0.95 |
| knee_symmetry_error_deg | 1.51 | leg_posture_error_rms | 0.2769 |
| **Support/Drift** | | | |
| support_rms_m | 0.0953 | support_peak_m | 0.2287 |
| sagittal_drift_m | 0.009 | lateral_drift_m | -0.069 |
| final_displacement_m | 0.069 | max_displacement_m | 0.194 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| torque_rate_rms_nm_s | 23.5 | torque_saturation_count | 148 |
| **Robustness** | | | |
| stability_score | 0.857 | contact_loss_frac | 0.0050 |
| drift_rate_m_per_kstep | 0.0346 | | |

#### [SUM] mid_0p400_sagittal_forward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 11.75 | pitch_min_deg | -3.43 |
| roll_max_deg | 1.01 | roll_min_deg | -0.46 |
| hip_yaw_joint_max_rad | 0.1125 | contact_loss_steps | 11 |
| **Posture** | | | |
| pitch_rms_deg | 2.73 | pitch_peak_deg | 11.75 |
| roll_rms_deg | 0.52 | roll_peak_deg | 1.01 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1125 | hip_yaw_div_rms_rad | 0.0836 |
| **Support/Drift** | | | |
| support_rms_m | 0.0954 | support_peak_m | 0.2522 |
| sagittal_drift_m | -0.001 | lateral_drift_m | -0.071 |
| final_displacement_m | 0.071 | max_displacement_m | 0.200 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.42 |
| **Robustness** | | | |
| stability_score | 0.832 | contact_loss_frac | 0.0055 |
| drift_rate_m_per_kstep | 0.0356 | | |

### Scope: step_e

#### [SUM] high_0p430 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.18 | pitch_min_deg | -1.95 |
| roll_max_deg | 0.68 | roll_min_deg | -0.03 |
| hip_yaw_joint_max_rad | 0.0524 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.13 | pitch_peak_deg | 7.18 |
| roll_rms_deg | 0.35 | roll_peak_deg | 0.68 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0524 | hip_yaw_div_rms_rad | 0.0385 |
| **Support/Drift** | | | |
| support_rms_m | 0.0500 | support_peak_m | 0.1399 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.086 |
| final_displacement_m | 0.087 | max_displacement_m | 0.103 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.71 | torque_peak_wheels_nm | 2.96 |
| **Robustness** | | | |
| stability_score | 0.822 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0433 | | |

#### [TEL] high_0p450 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.98 | pitch_min_deg | -0.05 |
| roll_max_deg | 0.37 | roll_min_deg | -0.08 |
| hip_yaw_joint_max_rad | 0.0193 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.68 | pitch_peak_deg | 7.98 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.37 |
| pitch_rate_rms_deg_s | 8.05 | pitch_settling_steps | 520 |
| angular_velocity_rms_deg_s | 8.12 | yaw_drift_deg | -0.66 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0193 | hip_yaw_div_rms_rad | 0.0134 |
| hip_yaw_lr_divergence_deg | 0.77 | hip_pitch_symmetry_error_deg | 1.24 |
| knee_symmetry_error_deg | 0.26 | leg_posture_error_rms | 0.0961 |
| **Support/Drift** | | | |
| support_rms_m | 0.0664 | support_peak_m | 0.1458 |
| sagittal_drift_m | 0.006 | lateral_drift_m | -0.106 |
| final_displacement_m | 0.106 | max_displacement_m | 0.112 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.77 | torque_peak_wheels_nm | 4.00 |
| torque_rate_rms_nm_s | 9.7 | torque_saturation_count | 112 |
| **Robustness** | | | |
| stability_score | 0.758 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0531 | | |

#### [SUM] high_0p465 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.21 | pitch_min_deg | -2.35 |
| roll_max_deg | 0.34 | roll_min_deg | -0.17 |
| hip_yaw_joint_max_rad | 0.0319 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.62 | pitch_peak_deg | 7.21 |
| roll_rms_deg | 0.14 | roll_peak_deg | 0.34 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0319 | hip_yaw_div_rms_rad | 0.0181 |
| **Support/Drift** | | | |
| support_rms_m | 0.0636 | support_peak_m | 0.1208 |
| sagittal_drift_m | 0.008 | lateral_drift_m | 0.084 |
| final_displacement_m | 0.084 | max_displacement_m | 0.093 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.04 | torque_peak_wheels_nm | 1.78 |
| **Robustness** | | | |
| stability_score | 0.811 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0421 | | |

#### [TEL] high_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.74 | pitch_min_deg | -1.34 |
| roll_max_deg | 0.20 | roll_min_deg | -0.54 |
| hip_yaw_joint_max_rad | 0.0505 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.28 | pitch_peak_deg | 8.74 |
| roll_rms_deg | 0.14 | roll_peak_deg | 0.54 |
| pitch_rate_rms_deg_s | 8.51 | pitch_settling_steps | 870 |
| angular_velocity_rms_deg_s | 8.58 | yaw_drift_deg | -4.27 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0505 | hip_yaw_div_rms_rad | 0.0326 |
| hip_yaw_lr_divergence_deg | 1.87 | hip_pitch_symmetry_error_deg | 1.43 |
| knee_symmetry_error_deg | 0.33 | leg_posture_error_rms | 0.1187 |
| **Support/Drift** | | | |
| support_rms_m | 0.0581 | support_peak_m | 0.1229 |
| sagittal_drift_m | 0.015 | lateral_drift_m | 0.036 |
| final_displacement_m | 0.039 | max_displacement_m | 0.093 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.30 |
| torque_rate_rms_nm_s | 8.9 | torque_saturation_count | 39 |
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

#### [TEL] low_0p320 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 4.08 | pitch_min_deg | -7.44 |
| roll_max_deg | 0.80 | roll_min_deg | -0.07 |
| hip_yaw_joint_max_rad | 0.0494 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.69 | pitch_peak_deg | 7.44 |
| roll_rms_deg | 0.29 | roll_peak_deg | 0.80 |
| pitch_rate_rms_deg_s | 9.78 | pitch_settling_steps | 2000 |
| angular_velocity_rms_deg_s | 9.87 | yaw_drift_deg | -3.71 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0494 | hip_yaw_div_rms_rad | 0.0280 |
| hip_yaw_lr_divergence_deg | 1.60 | hip_pitch_symmetry_error_deg | 1.03 |
| knee_symmetry_error_deg | 0.39 | leg_posture_error_rms | 0.0809 |
| **Support/Drift** | | | |
| support_rms_m | 0.0692 | support_peak_m | 0.1183 |
| sagittal_drift_m | -0.001 | lateral_drift_m | -0.072 |
| final_displacement_m | 0.072 | max_displacement_m | 0.105 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.78 | torque_peak_wheels_nm | 2.04 |
| torque_rate_rms_nm_s | 12.5 | torque_saturation_count | 2000 |
| **Robustness** | | | |
| stability_score | 0.798 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0362 | | |

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

#### [TEL] low_0p360 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 2.46 | pitch_min_deg | -6.57 |
| roll_max_deg | 0.77 | roll_min_deg | -0.13 |
| hip_yaw_joint_max_rad | 0.0542 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.12 | pitch_peak_deg | 6.57 |
| roll_rms_deg | 0.28 | roll_peak_deg | 0.77 |
| pitch_rate_rms_deg_s | 7.89 | pitch_settling_steps | 2000 |
| angular_velocity_rms_deg_s | 8.01 | yaw_drift_deg | -2.70 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0542 | hip_yaw_div_rms_rad | 0.0302 |
| hip_yaw_lr_divergence_deg | 1.73 | hip_pitch_symmetry_error_deg | 1.07 |
| knee_symmetry_error_deg | 0.44 | leg_posture_error_rms | 0.0868 |
| **Support/Drift** | | | |
| support_rms_m | 0.0531 | support_peak_m | 0.0937 |
| sagittal_drift_m | 0.006 | lateral_drift_m | 0.057 |
| final_displacement_m | 0.057 | max_displacement_m | 0.081 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.55 | torque_peak_wheels_nm | 1.84 |
| torque_rate_rms_nm_s | 10.1 | torque_saturation_count | 2000 |
| **Robustness** | | | |
| stability_score | 0.827 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0287 | | |

#### [SUM] low_0p380 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 9.03 | pitch_min_deg | -0.65 |
| roll_max_deg | 0.88 | roll_min_deg | -0.11 |
| hip_yaw_joint_max_rad | 0.0527 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 5.24 | pitch_peak_deg | 9.03 |
| roll_rms_deg | 0.33 | roll_peak_deg | 0.88 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0527 | hip_yaw_div_rms_rad | 0.0227 |
| **Support/Drift** | | | |
| support_rms_m | 0.0679 | support_peak_m | 0.1213 |
| sagittal_drift_m | 0.002 | lateral_drift_m | -0.017 |
| final_displacement_m | 0.018 | max_displacement_m | 0.093 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 9.16 | torque_peak_wheels_nm | 3.13 |
| **Robustness** | | | |
| stability_score | 0.718 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0088 | | |

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

Full metrics exported to: `docs\validation\k2_improvement_baseline_quality.json`