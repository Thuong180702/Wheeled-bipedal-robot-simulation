# K2 JAX Dedicated Realtime — Behavior Quality Baseline

**Generated:** 2026-06-30T10:28:59.994381
**Input:** `outputs\k2_jax_dedicated_promotion_validation`
**input_dir:** outputs\k2_jax_dedicated_promotion_validation
**analyzer_version:** 1.0.0
**baseline_type:** K2_JAX_DEDICATED_REALTIME_IMPROVEMENT_BASELINE

## Executive Summary

- **Total scenarios:** 39
- **Falls:** 2 (ramp_up_0p330_to_0p480, up_down_cycle_0p330_0p480_0p330)
- **Scenarios with full telemetry:** 0/39
- **Performance:** 144.9 Hz avg (min 52.3, max 169.7)

## A. Safety — Hard Gates

#### Safety Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| contact_loss_steps | 2.2821 | 3.0797 | 1.0000 | 11.0000 | 1.0000 |
| fall_step | 68.1538 | 299.6819 | -1.0000 | 1509.0000 | -1.0000 |
| fell | 0.0513 | 0.2206 | 0.0000 | 1.0000 | 0.0000 |
| hip_yaw_joint_max_rad | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| nan_inf_detected | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pitch_max_deg | 6.1797 | 3.4900 | 1.8240 | 13.8442 | 7.1802 |
| pitch_min_deg | -4.8873 | 3.2923 | -14.5263 | -0.0512 | -4.2521 |
| roll_max_deg | 0.8056 | 0.4290 | 0.1722 | 1.6971 | 0.8845 |
| roll_min_deg | -0.2087 | 0.1864 | -0.5823 | 0.0000 | -0.0994 |

## B. Posture Stability

#### Posture Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| angular_velocity_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| orientation_energy_integral | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pitch_peak_deg | 8.4004 | 2.1359 | 3.7218 | 14.5263 | 7.9782 |
| pitch_rate_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pitch_rms_deg | 3.9275 | 1.0208 | 1.6634 | 6.1883 | 3.9627 |
| pitch_settling_steps | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| roll_peak_deg | 0.8536 | 0.3746 | 0.2541 | 1.6971 | 0.8845 |
| roll_rate_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| roll_rms_deg | 0.3794 | 0.2304 | 0.1072 | 1.1086 | 0.3762 |
| yaw_drift_deg | 8.5571 | 8.5766 | 0.8679 | 52.8563 | 6.1542 |
| yaw_rate_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Pitch RMS by Height Region

| Region | Mean Pitch RMS (deg) | Std | Count |
|--------|----------------------|-----|-------|
| low | 3.93 | 0.84 | 12 |
| mid | 3.28 | 1.59 | 9 |
| high | 4.31 | 0.52 | 13 |

### Pitch RMS by Scenario Type

| Type | Mean Pitch RMS (deg) | Std | Count |
|------|----------------------|-----|-------|
| fixed_height | 3.78 | 0.97 | 17 |
| push | 3.81 | 1.22 | 12 |
| dynamic_height | 4.55 | 0.88 | 5 |

## C. Leg Symmetry / Twist

#### Leg Symmetry Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| hip_pitch_symmetry_error_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hip_roll_symmetry_error_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hip_yaw_div_max_rad | 0.1538 | 0.0986 | 0.0263 | 0.5370 | 0.1269 |
| hip_yaw_div_rms_rad | 0.0686 | 0.0473 | 0.0111 | 0.2602 | 0.0518 |
| hip_yaw_joint_max_rad | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hip_yaw_lr_divergence_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| knee_symmetry_error_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| leg_posture_error_rms | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## D. Support / Drift

#### Support & Drift Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| com_support_offset_rms_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| final_displacement_m | 0.1105 | 0.2218 | 0.0049 | 1.2909 | 0.0574 |
| lateral_drift_m | -0.0166 | 0.2245 | -0.3304 | 1.2530 | -0.0177 |
| max_displacement_m | 0.2170 | 0.2175 | 0.0810 | 1.2909 | 0.1445 |
| sagittal_drift_m | 0.0019 | 0.1036 | -0.5597 | 0.3105 | 0.0082 |
| support_peak_m | 0.2398 | 0.2198 | 0.0937 | 1.3012 | 0.1458 |
| support_rms_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| support_velocity_rms_m_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wheel_travel_asymmetry_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## E. Dynamic Height Tracking

#### Dynamic Height Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| dynamic_transition_smoothness | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| height_final_m | 0.3971 | 0.0669 | 0.2928 | 0.4904 | 0.4067 |
| height_initial_m | 0.3949 | 0.0638 | 0.2955 | 0.4810 | 0.3993 |
| height_max_m | 0.4017 | 0.0684 | 0.2955 | 0.4977 | 0.4132 |
| height_min_m | 0.3927 | 0.0649 | 0.2914 | 0.4809 | 0.3993 |
| height_overshoot_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| height_rmse_m | 0.0135 | 0.0222 | 0.0008 | 0.1123 | 0.0071 |
| height_tracking_lag_steps | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| height_undershoot_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_ref_tracking_error_rms | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## F. Torque Quality

#### Torque Quality Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| controller_conflict_index | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_hip_roll_nm | 0.4970 | 0.3542 | 0.1564 | 1.3358 | 0.3749 |
| torque_peak_hip_yaw_nm | 2.2387 | 1.1860 | 0.6552 | 6.2107 | 1.8894 |
| torque_peak_l_hip_pitch_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_hip_roll_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_hip_yaw_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_knee_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_wheel_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_legs_nm | 9.3098 | 1.7030 | 8.0000 | 13.7141 | 8.6681 |
| torque_peak_r_hip_pitch_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_hip_roll_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_hip_yaw_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_knee_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_wheel_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_total_nm | 9.3882 | 1.6970 | 8.0000 | 13.7141 | 8.6681 |
| torque_peak_wheels_nm | 5.0992 | 3.1072 | 1.4612 | 11.5753 | 3.5343 |
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
| contact_loss_frac | 0.0010 | 0.0014 | 0.0002 | 0.0055 | 0.0005 |
| contact_loss_steps | 2.2821 | 3.0797 | 1.0000 | 11.0000 | 1.0000 |
| long_run_drift_rate_m_per_kstep | 0.0400 | 0.0599 | 0.0024 | 0.2582 | 0.0196 |
| post_pitch_rms_500_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| post_push_active | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| post_support_rms_500_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| recovery_time_steps | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stability_score_0_to_1 | 0.7409 | 0.1791 | 0.0000 | 0.8844 | 0.7735 |

## Per-Scenario Detail

### Scope: dynamic_height

#### [SUM] gate_chatter_0p400_0p470 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 10.09 | pitch_min_deg | -2.73 |
| roll_max_deg | 0.29 | roll_min_deg | -0.58 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.74 | pitch_peak_deg | 10.09 |
| roll_rms_deg | 0.14 | roll_peak_deg | 0.58 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0817 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1848 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 6.19 | pitch_peak_deg | 10.87 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.51 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.2602 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.7650 |
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

#### [SUM] ramp_down_0p480_to_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.78 | pitch_min_deg | -1.95 |
| roll_max_deg | 0.30 | roll_min_deg | -0.47 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.03 | pitch_peak_deg | 8.78 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.47 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.1156 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 1.3012 |
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

#### [SUM] ramp_up_0p330_to_0p480 — FALL

| Safety | | | |
|--------|-----|-----|-----|
| **fell** | True | fall_step=1509 | reason=height_too_low (0.330 < 0.330) |
| pitch_max_deg | 1.82 | pitch_min_deg | -7.23 |
| roll_max_deg | 1.00 | roll_min_deg | -0.10 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.87 | pitch_peak_deg | 7.23 |
| roll_rms_deg | 0.51 | roll_peak_deg | 1.00 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0936 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.3142 |
| sagittal_drift_m | -0.003 | lateral_drift_m | -0.303 |
| final_displacement_m | 0.303 | max_displacement_m | 0.303 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0215 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.62 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.000 | contact_loss_frac | 0.0007 |
| drift_rate_m_per_kstep | 0.2008 | | |

#### [SUM] up_down_cycle_0p330_0p480_0p330 — FALL

| Safety | | | |
|--------|-----|-----|-----|
| **fell** | True | fall_step=1186 | reason=height_too_low (0.331 < 0.331) |
| pitch_max_deg | 1.82 | pitch_min_deg | -8.31 |
| roll_max_deg | 0.74 | roll_min_deg | -0.10 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.92 | pitch_peak_deg | 8.31 |
| roll_rms_deg | 0.38 | roll_peak_deg | 0.74 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0752 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.2941 |
| sagittal_drift_m | -0.003 | lateral_drift_m | -0.287 |
| final_displacement_m | 0.287 | max_displacement_m | 0.287 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0206 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.62 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.000 | contact_loss_frac | 0.0008 |
| drift_rate_m_per_kstep | 0.2421 | | |

### Scope: long_run

#### [SUM] high_0p430 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.95 | pitch_min_deg | -2.21 |
| roll_max_deg | 0.69 | roll_min_deg | -0.05 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.77 | pitch_peak_deg | 7.95 |
| roll_rms_deg | 0.32 | roll_peak_deg | 0.69 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0835 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1503 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.55 | pitch_peak_deg | 7.99 |
| roll_rms_deg | 0.20 | roll_peak_deg | 0.55 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0970 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1460 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.69 | pitch_peak_deg | 8.87 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.54 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0823 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1368 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 5.07 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 1.11 | roll_peak_deg | 1.70 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.1272 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1230 |
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
| pitch_max_deg | 3.72 | pitch_min_deg | -3.43 |
| roll_max_deg | 1.15 | roll_min_deg | -0.07 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 1.75 | pitch_peak_deg | 3.72 |
| roll_rms_deg | 0.47 | roll_peak_deg | 1.15 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.1439 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1302 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.059 |
| final_displacement_m | 0.060 | max_displacement_m | 0.120 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| **Robustness** | | | |
| stability_score | 0.884 | contact_loss_frac | 0.0017 |
| drift_rate_m_per_kstep | 0.0100 | | |

### Scope: step_c

#### [SUM] C1_slow_ladder_up_down — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 1.15 | roll_min_deg | -0.10 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.96 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.47 | roll_peak_deg | 1.15 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0518 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1230 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.96 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.47 | roll_peak_deg | 1.15 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0518 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1230 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.96 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.47 | roll_peak_deg | 1.15 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0518 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1230 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.96 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.47 | roll_peak_deg | 1.15 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0518 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1230 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.51 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.83 | roll_peak_deg | 1.63 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0931 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1230 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.28 | pitch_peak_deg | 8.74 |
| roll_rms_deg | 0.14 | roll_peak_deg | 0.54 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0326 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1229 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.69 | pitch_peak_deg | 7.44 |
| roll_rms_deg | 0.29 | roll_peak_deg | 0.80 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0280 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1183 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.68 | pitch_peak_deg | 8.83 |
| roll_rms_deg | 0.11 | roll_peak_deg | 0.25 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0165 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.2431 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.54 | pitch_peak_deg | 8.79 |
| roll_rms_deg | 0.11 | roll_peak_deg | 0.33 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0111 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.3672 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.90 | pitch_peak_deg | 11.53 |
| roll_rms_deg | 0.11 | roll_peak_deg | 0.26 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0160 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.2334 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 2 |
| **Posture** | | | |
| pitch_rms_deg | 4.94 | pitch_peak_deg | 13.84 |
| roll_rms_deg | 0.11 | roll_peak_deg | 0.31 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0153 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.3542 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.38 | pitch_peak_deg | 11.63 |
| roll_rms_deg | 0.39 | roll_peak_deg | 0.95 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0502 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1719 |
| sagittal_drift_m | 0.004 | lateral_drift_m | -0.027 |
| final_displacement_m | 0.027 | max_displacement_m | 0.154 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.70 | torque_peak_wheels_nm | 6.68 |
| **Robustness** | | | |
| stability_score | 0.758 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0136 | | |

#### [SUM] low_0p330_sagittal_backward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.78 | pitch_min_deg | -14.53 |
| roll_max_deg | 0.99 | roll_min_deg | -0.28 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.80 | pitch_peak_deg | 14.53 |
| roll_rms_deg | 0.46 | roll_peak_deg | 0.99 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0702 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.2657 |
| sagittal_drift_m | 0.007 | lateral_drift_m | -0.006 |
| final_displacement_m | 0.009 | max_displacement_m | 0.201 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 10.39 | torque_peak_wheels_nm | 10.39 |
| **Robustness** | | | |
| stability_score | 0.732 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0044 | | |

#### [SUM] low_0p330_sagittal_forward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.47 | pitch_min_deg | -7.14 |
| roll_max_deg | 1.64 | roll_min_deg | -0.54 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.62 | pitch_peak_deg | 7.47 |
| roll_rms_deg | 0.84 | roll_peak_deg | 1.64 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0989 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.3226 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 4 |
| **Posture** | | | |
| pitch_rms_deg | 4.37 | pitch_peak_deg | 9.86 |
| roll_rms_deg | 0.56 | roll_peak_deg | 1.13 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0958 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.4380 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 1.66 | pitch_peak_deg | 6.03 |
| roll_rms_deg | 0.54 | roll_peak_deg | 1.06 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.1029 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.3701 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 1.94 | pitch_peak_deg | 8.29 |
| roll_rms_deg | 0.50 | roll_peak_deg | 1.07 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.1196 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.5118 |
| sagittal_drift_m | 0.025 | lateral_drift_m | -0.119 |
| final_displacement_m | 0.121 | max_displacement_m | 0.466 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| **Robustness** | | | |
| stability_score | 0.873 | contact_loss_frac | 0.0050 |
| drift_rate_m_per_kstep | 0.0607 | | |

#### [SUM] mid_0p400_sagittal_forward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 9.29 | pitch_min_deg | -3.73 |
| roll_max_deg | 1.16 | roll_min_deg | -0.18 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 10 |
| **Posture** | | | |
| pitch_rms_deg | 2.19 | pitch_peak_deg | 9.29 |
| roll_rms_deg | 0.56 | roll_peak_deg | 1.16 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0952 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.2287 |
| sagittal_drift_m | 0.009 | lateral_drift_m | -0.069 |
| final_displacement_m | 0.069 | max_displacement_m | 0.194 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 12.95 | torque_peak_wheels_nm | 10.23 |
| **Robustness** | | | |
| stability_score | 0.857 | contact_loss_frac | 0.0050 |
| drift_rate_m_per_kstep | 0.0346 | | |

#### [SUM] mid_0p400_sagittal_forward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 11.75 | pitch_min_deg | -3.43 |
| roll_max_deg | 1.01 | roll_min_deg | -0.46 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 11 |
| **Posture** | | | |
| pitch_rms_deg | 2.73 | pitch_peak_deg | 11.75 |
| roll_rms_deg | 0.52 | roll_peak_deg | 1.01 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0836 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.2522 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.13 | pitch_peak_deg | 7.18 |
| roll_rms_deg | 0.35 | roll_peak_deg | 0.68 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0385 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1399 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.086 |
| final_displacement_m | 0.087 | max_displacement_m | 0.103 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.71 | torque_peak_wheels_nm | 2.96 |
| **Robustness** | | | |
| stability_score | 0.822 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0433 | | |

#### [SUM] high_0p450 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.98 | pitch_min_deg | -0.05 |
| roll_max_deg | 0.37 | roll_min_deg | -0.08 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.68 | pitch_peak_deg | 7.98 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.37 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0134 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1458 |
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
| pitch_max_deg | 7.21 | pitch_min_deg | -2.35 |
| roll_max_deg | 0.34 | roll_min_deg | -0.17 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.62 | pitch_peak_deg | 7.21 |
| roll_rms_deg | 0.14 | roll_peak_deg | 0.34 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0181 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1208 |
| sagittal_drift_m | 0.008 | lateral_drift_m | 0.084 |
| final_displacement_m | 0.084 | max_displacement_m | 0.093 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.04 | torque_peak_wheels_nm | 1.78 |
| **Robustness** | | | |
| stability_score | 0.811 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0421 | | |

#### [SUM] high_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.74 | pitch_min_deg | -1.34 |
| roll_max_deg | 0.20 | roll_min_deg | -0.54 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.28 | pitch_peak_deg | 8.74 |
| roll_rms_deg | 0.14 | roll_peak_deg | 0.54 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0326 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1229 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 2.91 | pitch_peak_deg | 7.12 |
| roll_rms_deg | 0.63 | roll_peak_deg | 0.95 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0917 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.0960 |
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
| pitch_max_deg | 4.08 | pitch_min_deg | -7.44 |
| roll_max_deg | 0.80 | roll_min_deg | -0.07 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.69 | pitch_peak_deg | 7.44 |
| roll_rms_deg | 0.29 | roll_peak_deg | 0.80 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0280 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1183 |
| sagittal_drift_m | -0.001 | lateral_drift_m | -0.072 |
| final_displacement_m | 0.072 | max_displacement_m | 0.105 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.78 | torque_peak_wheels_nm | 2.04 |
| **Robustness** | | | |
| stability_score | 0.798 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0362 | | |

#### [SUM] low_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.82 | pitch_min_deg | -7.14 |
| roll_max_deg | 1.15 | roll_min_deg | -0.10 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.96 | pitch_peak_deg | 7.14 |
| roll_rms_deg | 0.47 | roll_peak_deg | 1.15 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0518 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1230 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 1.86 | pitch_peak_deg | 4.55 |
| roll_rms_deg | 0.56 | roll_peak_deg | 1.11 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0516 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1084 |
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
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.12 | pitch_peak_deg | 6.57 |
| roll_rms_deg | 0.28 | roll_peak_deg | 0.77 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0302 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.0937 |
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
| pitch_max_deg | 9.03 | pitch_min_deg | -0.65 |
| roll_max_deg | 0.88 | roll_min_deg | -0.11 |
| hip_yaw_joint_max_rad | 0.0000 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 5.24 | pitch_peak_deg | 9.03 |
| roll_rms_deg | 0.33 | roll_peak_deg | 0.88 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0000 | hip_yaw_div_rms_rad | 0.0227 |
| **Support/Drift** | | | |
| support_rms_m | 0.0000 | support_peak_m | 0.1213 |
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

Full metrics exported to: `docs\validation\k2_improvement_baseline_quality_smoke_test.json`