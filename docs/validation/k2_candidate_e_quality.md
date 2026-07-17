# K2 JAX Dedicated Realtime — Behavior Quality Baseline

**Generated:** 2026-06-30T12:01:01.576561
**Input:** `outputs\k2_candidate_e_full`
**input_dir:** outputs\k2_candidate_e_full
**analyzer_version:** 1.0.0
**baseline_type:** K2_JAX_DEDICATED_REALTIME_IMPROVEMENT_BASELINE

## Executive Summary

- **Total scenarios:** 39
- **Falls:** 0 (none)
- **Scenarios with full telemetry:** 0/39
- **Performance:** 191.8 Hz avg (min 178.8, max 199.1)

## A. Safety — Hard Gates

#### Safety Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| contact_loss_steps | 2.1282 | 2.6716 | 1.0000 | 9.0000 | 1.0000 |
| fall_step | -1.0000 | 0.0000 | -1.0000 | -1.0000 | -1.0000 |
| fell | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hip_yaw_joint_max_rad | 0.0919 | 0.0461 | 0.0191 | 0.2518 | 0.0857 |
| nan_inf_detected | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pitch_max_deg | 5.8392 | 3.7462 | 1.1145 | 15.3293 | 6.7352 |
| pitch_min_deg | -4.4566 | 3.3917 | -15.9553 | 0.0000 | -5.0128 |
| roll_max_deg | 0.9017 | 0.4836 | 0.1740 | 1.6789 | 1.0158 |
| roll_min_deg | -0.1462 | 0.2130 | -0.7359 | 0.0000 | -0.0083 |

## B. Posture Stability

#### Posture Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| angular_velocity_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| orientation_energy_integral | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pitch_peak_deg | 7.8922 | 2.7403 | 3.5966 | 15.9553 | 7.2577 |
| pitch_rate_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pitch_rms_deg | 3.8561 | 1.0509 | 1.7320 | 6.3953 | 4.2558 |
| pitch_settling_steps | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| roll_peak_deg | 0.9158 | 0.4657 | 0.2517 | 1.6789 | 1.0158 |
| roll_rate_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| roll_rms_deg | 0.4931 | 0.3096 | 0.0833 | 1.1350 | 0.4760 |
| yaw_drift_deg | 7.9058 | 9.1660 | 2.6215 | 54.3145 | 5.3908 |
| yaw_rate_rms_deg_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Pitch RMS by Height Region

| Region | Mean Pitch RMS (deg) | Std | Count |
|--------|----------------------|-----|-------|
| low | 3.86 | 1.01 | 12 |
| mid | 3.23 | 1.53 | 9 |
| high | 4.06 | 0.53 | 13 |

### Pitch RMS by Scenario Type

| Type | Mean Pitch RMS (deg) | Std | Count |
|------|----------------------|-----|-------|
| fixed_height | 3.53 | 0.92 | 17 |
| push | 3.83 | 1.22 | 12 |
| dynamic_height | 4.41 | 1.04 | 5 |

## C. Leg Symmetry / Twist

#### Leg Symmetry Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| hip_pitch_symmetry_error_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hip_roll_symmetry_error_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hip_yaw_div_max_rad | 0.1583 | 0.0937 | 0.0314 | 0.5036 | 0.1393 |
| hip_yaw_div_rms_rad | 0.0874 | 0.0504 | 0.0144 | 0.2492 | 0.0889 |
| hip_yaw_joint_max_rad | 0.0919 | 0.0461 | 0.0191 | 0.2518 | 0.0857 |
| hip_yaw_lr_divergence_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| knee_symmetry_error_deg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| leg_posture_error_rms | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## D. Support / Drift

#### Support & Drift Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| com_support_offset_rms_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| final_displacement_m | 0.0976 | 0.2449 | 0.0067 | 1.4862 | 0.0374 |
| lateral_drift_m | -0.0053 | 0.2193 | -0.2817 | 1.2988 | -0.0293 |
| max_displacement_m | 0.1915 | 0.2441 | 0.0598 | 1.4862 | 0.1040 |
| sagittal_drift_m | 0.0136 | 0.1456 | -0.5536 | 0.7223 | 0.0111 |
| support_peak_m | 0.2132 | 0.2363 | 0.0710 | 1.4248 | 0.1225 |
| support_rms_m | 0.0766 | 0.0899 | 0.0290 | 0.5538 | 0.0552 |
| support_velocity_rms_m_s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wheel_travel_asymmetry_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## E. Dynamic Height Tracking

#### Dynamic Height Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| dynamic_transition_smoothness | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| height_final_m | 0.4012 | 0.0669 | 0.2925 | 0.4901 | 0.4080 |
| height_initial_m | 0.3949 | 0.0638 | 0.2955 | 0.4810 | 0.3993 |
| height_max_m | 0.4096 | 0.0692 | 0.2955 | 0.4975 | 0.4165 |
| height_min_m | 0.3926 | 0.0646 | 0.2910 | 0.4809 | 0.3993 |
| height_overshoot_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| height_rmse_m | 0.0125 | 0.0218 | 0.0006 | 0.1080 | 0.0055 |
| height_tracking_lag_steps | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| height_undershoot_m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_ref_tracking_error_rms | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## F. Torque Quality

#### Torque Quality Metrics (aggregate)

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| controller_conflict_index | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_hip_roll_nm | 0.3721 | 0.2034 | 0.1290 | 1.2418 | 0.3718 |
| torque_peak_hip_yaw_nm | 2.2295 | 0.9220 | 0.5468 | 4.4454 | 2.2754 |
| torque_peak_l_hip_pitch_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_hip_roll_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_hip_yaw_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_knee_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_l_wheel_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_legs_nm | 9.3868 | 1.8453 | 8.0000 | 13.5576 | 8.5872 |
| torque_peak_r_hip_pitch_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_hip_roll_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_hip_yaw_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_knee_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_r_wheel_nm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| torque_peak_total_nm | 9.3868 | 1.8453 | 8.0000 | 13.5576 | 8.5872 |
| torque_peak_wheels_nm | 4.3487 | 2.1610 | 1.3404 | 9.3471 | 3.5343 |
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
| contact_loss_frac | 0.0009 | 0.0013 | 0.0001 | 0.0045 | 0.0005 |
| contact_loss_steps | 2.1282 | 2.6716 | 1.0000 | 9.0000 | 1.0000 |
| long_run_drift_rate_m_per_kstep | 0.0279 | 0.0480 | 0.0015 | 0.2972 | 0.0148 |
| post_pitch_rms_500_deg | 1.3758 | 2.1642 | 0.0000 | 6.3276 | 0.0000 |
| post_push_active | 0.3077 | 0.4615 | 0.0000 | 1.0000 | 0.0000 |
| post_support_rms_500_m | 0.0413 | 0.0716 | 0.0000 | 0.3157 | 0.0000 |
| recovery_time_steps | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stability_score_0_to_1 | 0.7776 | 0.0549 | 0.6728 | 0.8861 | 0.7767 |

## Per-Scenario Detail

### Scope: dynamic_height

#### [SUM] gate_chatter_0p400_0p470 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.30 | pitch_min_deg | -1.65 |
| roll_max_deg | 0.28 | roll_min_deg | -0.17 |
| hip_yaw_joint_max_rad | 0.0760 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.46 | pitch_peak_deg | 8.30 |
| roll_rms_deg | 0.12 | roll_peak_deg | 0.28 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0760 | hip_yaw_div_rms_rad | 0.0844 |
| **Support/Drift** | | | |
| support_rms_m | 0.0627 | support_peak_m | 0.1445 |
| sagittal_drift_m | 0.011 | lateral_drift_m | -0.100 |
| final_displacement_m | 0.101 | max_displacement_m | 0.124 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0711 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.07 |
| **Robustness** | | | |
| stability_score | 0.770 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0202 | | |

#### [SUM] gate_dwell_0p420_0p450_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 10.78 | pitch_min_deg | -3.35 |
| roll_max_deg | 0.22 | roll_min_deg | -0.51 |
| hip_yaw_joint_max_rad | 0.2518 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 6.40 | pitch_peak_deg | 10.78 |
| roll_rms_deg | 0.12 | roll_peak_deg | 0.51 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.2518 | hip_yaw_div_rms_rad | 0.2492 |
| **Support/Drift** | | | |
| support_rms_m | 0.2860 | support_peak_m | 0.7315 |
| sagittal_drift_m | -0.554 | lateral_drift_m | -0.282 |
| final_displacement_m | 0.621 | max_displacement_m | 0.725 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0775 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.00 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.673 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.1035 | | |

#### [SUM] ramp_down_0p480_to_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.95 | pitch_min_deg | -1.66 |
| roll_max_deg | 0.25 | roll_min_deg | -0.18 |
| hip_yaw_joint_max_rad | 0.2242 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.96 | pitch_peak_deg | 7.95 |
| roll_rms_deg | 0.10 | roll_peak_deg | 0.25 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.2242 | hip_yaw_div_rms_rad | 0.2196 |
| **Support/Drift** | | | |
| support_rms_m | 0.5538 | support_peak_m | 1.4248 |
| sagittal_drift_m | 0.722 | lateral_drift_m | 1.299 |
| final_displacement_m | 1.486 | max_displacement_m | 1.486 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.1080 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.02 | torque_peak_wheels_nm | 3.30 |
| **Robustness** | | | |
| stability_score | 0.796 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.2972 | | |

#### [SUM] ramp_up_0p330_to_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.18 | pitch_min_deg | -6.16 |
| roll_max_deg | 1.24 | roll_min_deg | -0.59 |
| hip_yaw_joint_max_rad | 0.1236 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.66 | pitch_peak_deg | 8.18 |
| roll_rms_deg | 0.48 | roll_peak_deg | 1.24 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1236 | hip_yaw_div_rms_rad | 0.1037 |
| **Support/Drift** | | | |
| support_rms_m | 0.0546 | support_peak_m | 0.1144 |
| sagittal_drift_m | 0.004 | lateral_drift_m | -0.045 |
| final_displacement_m | 0.045 | max_displacement_m | 0.103 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0058 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.53 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.788 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0090 | | |

#### [SUM] up_down_cycle_0p330_0p480_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.32 | pitch_min_deg | -6.16 |
| roll_max_deg | 1.14 | roll_min_deg | -0.74 |
| hip_yaw_joint_max_rad | 0.1228 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.59 | pitch_peak_deg | 8.32 |
| roll_rms_deg | 0.38 | roll_peak_deg | 1.14 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1228 | hip_yaw_div_rms_rad | 0.1349 |
| **Support/Drift** | | | |
| support_rms_m | 0.0571 | support_peak_m | 0.1341 |
| sagittal_drift_m | 0.024 | lateral_drift_m | 0.065 |
| final_displacement_m | 0.069 | max_displacement_m | 0.139 |
| **Dynamic Height** | | | |
| height_rmse_m | 0.0055 | height_overshoot_m | 0.0000 |
| height_undershoot_m | 0.0000 | tracking_lag_steps | 0 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.53 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.798 | contact_loss_frac | 0.0001 |
| drift_rate_m_per_kstep | 0.0099 | | |

### Scope: long_run

#### [SUM] high_0p430 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 6.66 | pitch_min_deg | -2.00 |
| roll_max_deg | 0.68 | roll_min_deg | -0.05 |
| hip_yaw_joint_max_rad | 0.1121 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.49 | pitch_peak_deg | 6.66 |
| roll_rms_deg | 0.29 | roll_peak_deg | 0.68 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1121 | hip_yaw_div_rms_rad | 0.1170 |
| **Support/Drift** | | | |
| support_rms_m | 0.0552 | support_peak_m | 0.1345 |
| sagittal_drift_m | 0.016 | lateral_drift_m | -0.038 |
| final_displacement_m | 0.041 | max_displacement_m | 0.103 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.79 | torque_peak_wheels_nm | 2.96 |
| **Robustness** | | | |
| stability_score | 0.808 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0069 | | |

#### [SUM] high_0p450 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 6.91 | pitch_min_deg | 0.00 |
| roll_max_deg | 0.55 | roll_min_deg | -0.09 |
| hip_yaw_joint_max_rad | 0.1206 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.26 | pitch_peak_deg | 6.91 |
| roll_rms_deg | 0.20 | roll_peak_deg | 0.55 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1206 | hip_yaw_div_rms_rad | 0.1441 |
| **Support/Drift** | | | |
| support_rms_m | 0.0580 | support_peak_m | 0.1318 |
| sagittal_drift_m | 0.020 | lateral_drift_m | -0.051 |
| final_displacement_m | 0.055 | max_displacement_m | 0.126 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.85 | torque_peak_wheels_nm | 4.00 |
| **Robustness** | | | |
| stability_score | 0.775 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0091 | | |

#### [SUM] high_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.62 | pitch_min_deg | -1.04 |
| roll_max_deg | 0.25 | roll_min_deg | -0.22 |
| hip_yaw_joint_max_rad | 0.0684 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.36 | pitch_peak_deg | 7.62 |
| roll_rms_deg | 0.12 | roll_peak_deg | 0.25 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0684 | hip_yaw_div_rms_rad | 0.0848 |
| **Support/Drift** | | | |
| support_rms_m | 0.0601 | support_peak_m | 0.1107 |
| sagittal_drift_m | 0.016 | lateral_drift_m | -0.084 |
| final_displacement_m | 0.086 | max_displacement_m | 0.094 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.02 | torque_peak_wheels_nm | 3.30 |
| **Robustness** | | | |
| stability_score | 0.774 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0143 | | |

#### [SUM] low_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.16 | pitch_min_deg | -6.16 |
| roll_max_deg | 1.65 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.1094 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 5.05 | pitch_peak_deg | 6.16 |
| roll_rms_deg | 1.13 | roll_peak_deg | 1.65 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1094 | hip_yaw_div_rms_rad | 0.1273 |
| **Support/Drift** | | | |
| support_rms_m | 0.0290 | support_peak_m | 0.1057 |
| sagittal_drift_m | 0.005 | lateral_drift_m | 0.007 |
| final_displacement_m | 0.009 | max_displacement_m | 0.103 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.61 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.679 | contact_loss_frac | 0.0002 |
| drift_rate_m_per_kstep | 0.0015 | | |

#### [SUM] mid_0p400 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.60 | pitch_min_deg | -3.48 |
| roll_max_deg | 1.07 | roll_min_deg | -0.08 |
| hip_yaw_joint_max_rad | 0.1069 | contact_loss_steps | 9 |
| **Posture** | | | |
| pitch_rms_deg | 1.82 | pitch_peak_deg | 3.60 |
| roll_rms_deg | 0.38 | roll_peak_deg | 1.07 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1069 | hip_yaw_div_rms_rad | 0.1546 |
| **Support/Drift** | | | |
| support_rms_m | 0.0740 | support_peak_m | 0.1320 |
| sagittal_drift_m | 0.004 | lateral_drift_m | -0.089 |
| final_displacement_m | 0.089 | max_displacement_m | 0.123 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 13.56 | torque_peak_wheels_nm | 7.07 |
| **Robustness** | | | |
| stability_score | 0.886 | contact_loss_frac | 0.0015 |
| drift_rate_m_per_kstep | 0.0148 | | |

### Scope: step_c

#### [SUM] C1_slow_ladder_up_down — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.16 | pitch_min_deg | -6.16 |
| roll_max_deg | 1.30 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.0857 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.39 | pitch_peak_deg | 6.16 |
| roll_rms_deg | 0.83 | roll_peak_deg | 1.30 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0857 | hip_yaw_div_rms_rad | 0.0889 |
| **Support/Drift** | | | |
| support_rms_m | 0.0306 | support_peak_m | 0.1057 |
| sagittal_drift_m | 0.012 | lateral_drift_m | -0.008 |
| final_displacement_m | 0.014 | max_displacement_m | 0.102 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.58 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.730 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0072 | | |

#### [SUM] C2_random_500dwell — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.16 | pitch_min_deg | -6.16 |
| roll_max_deg | 1.30 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.0857 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.39 | pitch_peak_deg | 6.16 |
| roll_rms_deg | 0.83 | roll_peak_deg | 1.30 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0857 | hip_yaw_div_rms_rad | 0.0889 |
| **Support/Drift** | | | |
| support_rms_m | 0.0306 | support_peak_m | 0.1057 |
| sagittal_drift_m | 0.012 | lateral_drift_m | -0.008 |
| final_displacement_m | 0.014 | max_displacement_m | 0.102 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.58 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.730 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0072 | | |

#### [SUM] C3_random_200dwell — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.16 | pitch_min_deg | -6.16 |
| roll_max_deg | 1.30 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.0857 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.39 | pitch_peak_deg | 6.16 |
| roll_rms_deg | 0.83 | roll_peak_deg | 1.30 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0857 | hip_yaw_div_rms_rad | 0.0889 |
| **Support/Drift** | | | |
| support_rms_m | 0.0306 | support_peak_m | 0.1057 |
| sagittal_drift_m | 0.012 | lateral_drift_m | -0.008 |
| final_displacement_m | 0.014 | max_displacement_m | 0.102 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.58 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.730 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0072 | | |

#### [SUM] C4_abrupt_stress — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.16 | pitch_min_deg | -6.16 |
| roll_max_deg | 1.30 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.0857 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.39 | pitch_peak_deg | 6.16 |
| roll_rms_deg | 0.83 | roll_peak_deg | 1.30 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0857 | hip_yaw_div_rms_rad | 0.0889 |
| **Support/Drift** | | | |
| support_rms_m | 0.0306 | support_peak_m | 0.1057 |
| sagittal_drift_m | 0.012 | lateral_drift_m | -0.008 |
| final_displacement_m | 0.014 | max_displacement_m | 0.102 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.58 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.730 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0072 | | |

#### [SUM] C5_long_random — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.16 | pitch_min_deg | -6.16 |
| roll_max_deg | 1.53 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.1015 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.71 | pitch_peak_deg | 6.16 |
| roll_rms_deg | 0.98 | roll_peak_deg | 1.53 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1015 | hip_yaw_div_rms_rad | 0.1080 |
| **Support/Drift** | | | |
| support_rms_m | 0.0292 | support_peak_m | 0.1057 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.001 |
| final_displacement_m | 0.010 | max_displacement_m | 0.103 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.58 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.706 | contact_loss_frac | 0.0003 |
| drift_rate_m_per_kstep | 0.0034 | | |

#### [SUM] focused_high_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.26 | pitch_min_deg | -1.03 |
| roll_max_deg | 0.25 | roll_min_deg | -0.03 |
| hip_yaw_joint_max_rad | 0.0571 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.08 | pitch_peak_deg | 7.26 |
| roll_rms_deg | 0.15 | roll_peak_deg | 0.25 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0571 | hip_yaw_div_rms_rad | 0.0362 |
| **Support/Drift** | | | |
| support_rms_m | 0.0542 | support_peak_m | 0.1107 |
| sagittal_drift_m | 0.011 | lateral_drift_m | -0.080 |
| final_displacement_m | 0.081 | max_displacement_m | 0.081 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.02 | torque_peak_wheels_nm | 3.30 |
| **Robustness** | | | |
| stability_score | 0.787 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0403 | | |

#### [SUM] focused_low_0p320 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.11 | pitch_min_deg | -5.45 |
| roll_max_deg | 1.13 | roll_min_deg | -0.01 |
| hip_yaw_joint_max_rad | 0.0773 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 2.96 | pitch_peak_deg | 5.45 |
| roll_rms_deg | 0.53 | roll_peak_deg | 1.13 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0773 | hip_yaw_div_rms_rad | 0.0585 |
| **Support/Drift** | | | |
| support_rms_m | 0.0351 | support_peak_m | 0.0710 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.035 |
| final_displacement_m | 0.037 | max_displacement_m | 0.063 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.77 | torque_peak_wheels_nm | 1.77 |
| **Robustness** | | | |
| stability_score | 0.820 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0183 | | |

### Scope: step_d

#### [SUM] high_0p480_sagittal_backward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.97 | pitch_min_deg | -5.26 |
| roll_max_deg | 0.28 | roll_min_deg | -0.21 |
| hip_yaw_joint_max_rad | 0.0191 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.37 | pitch_peak_deg | 7.97 |
| roll_rms_deg | 0.11 | roll_peak_deg | 0.28 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0191 | hip_yaw_div_rms_rad | 0.0151 |
| **Support/Drift** | | | |
| support_rms_m | 0.0665 | support_peak_m | 0.2206 |
| sagittal_drift_m | 0.011 | lateral_drift_m | 0.043 |
| final_displacement_m | 0.044 | max_displacement_m | 0.167 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.02 | torque_peak_wheels_nm | 4.65 |
| **Robustness** | | | |
| stability_score | 0.775 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0220 | | |

#### [SUM] high_0p480_sagittal_backward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 8.26 | pitch_min_deg | -8.26 |
| roll_max_deg | 0.38 | roll_min_deg | -0.24 |
| hip_yaw_joint_max_rad | 0.0329 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.27 | pitch_peak_deg | 8.26 |
| roll_rms_deg | 0.13 | roll_peak_deg | 0.38 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0329 | hip_yaw_div_rms_rad | 0.0194 |
| **Support/Drift** | | | |
| support_rms_m | 0.0806 | support_peak_m | 0.3350 |
| sagittal_drift_m | 0.009 | lateral_drift_m | -0.076 |
| final_displacement_m | 0.077 | max_displacement_m | 0.265 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.02 | torque_peak_wheels_nm | 7.93 |
| **Robustness** | | | |
| stability_score | 0.779 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0383 | | |

#### [SUM] high_0p480_sagittal_forward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 12.29 | pitch_min_deg | -0.90 |
| roll_max_deg | 0.18 | roll_min_deg | -0.32 |
| hip_yaw_joint_max_rad | 0.0192 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.76 | pitch_peak_deg | 12.29 |
| roll_rms_deg | 0.11 | roll_peak_deg | 0.32 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0192 | hip_yaw_div_rms_rad | 0.0144 |
| **Support/Drift** | | | |
| support_rms_m | 0.0677 | support_peak_m | 0.2269 |
| sagittal_drift_m | 0.000 | lateral_drift_m | -0.082 |
| final_displacement_m | 0.082 | max_displacement_m | 0.171 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.21 | torque_peak_wheels_nm | 4.89 |
| **Robustness** | | | |
| stability_score | 0.756 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0408 | | |

#### [SUM] high_0p480_sagittal_forward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 15.33 | pitch_min_deg | -1.19 |
| roll_max_deg | 0.17 | roll_min_deg | -0.29 |
| hip_yaw_joint_max_rad | 0.0243 | contact_loss_steps | 3 |
| **Posture** | | | |
| pitch_rms_deg | 4.84 | pitch_peak_deg | 15.33 |
| roll_rms_deg | 0.08 | roll_peak_deg | 0.29 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0243 | hip_yaw_div_rms_rad | 0.0231 |
| **Support/Drift** | | | |
| support_rms_m | 0.0803 | support_peak_m | 0.3350 |
| sagittal_drift_m | 0.010 | lateral_drift_m | 0.036 |
| final_displacement_m | 0.037 | max_displacement_m | 0.263 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 10.84 | torque_peak_wheels_nm | 8.08 |
| **Robustness** | | | |
| stability_score | 0.753 | contact_loss_frac | 0.0015 |
| drift_rate_m_per_kstep | 0.0185 | | |

#### [SUM] low_0p330_sagittal_backward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 2.82 | pitch_min_deg | -12.73 |
| roll_max_deg | 1.26 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.0859 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.71 | pitch_peak_deg | 12.73 |
| roll_rms_deg | 0.77 | roll_peak_deg | 1.26 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0859 | hip_yaw_div_rms_rad | 0.0855 |
| **Support/Drift** | | | |
| support_rms_m | 0.0451 | support_peak_m | 0.1652 |
| sagittal_drift_m | 0.014 | lateral_drift_m | -0.010 |
| final_displacement_m | 0.017 | max_displacement_m | 0.116 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.76 | torque_peak_wheels_nm | 4.98 |
| **Robustness** | | | |
| stability_score | 0.718 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0086 | | |

#### [SUM] low_0p330_sagittal_backward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 2.64 | pitch_min_deg | -15.96 |
| roll_max_deg | 1.41 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.1002 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.97 | pitch_peak_deg | 15.96 |
| roll_rms_deg | 0.76 | roll_peak_deg | 1.41 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1002 | hip_yaw_div_rms_rad | 0.0973 |
| **Support/Drift** | | | |
| support_rms_m | 0.0590 | support_peak_m | 0.2688 |
| sagittal_drift_m | 0.014 | lateral_drift_m | 0.006 |
| final_displacement_m | 0.015 | max_displacement_m | 0.200 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 9.09 | torque_peak_wheels_nm | 8.29 |
| **Robustness** | | | |
| stability_score | 0.706 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0076 | | |

#### [SUM] low_0p330_sagittal_forward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.66 | pitch_min_deg | -6.35 |
| roll_max_deg | 1.68 | roll_min_deg | -0.56 |
| hip_yaw_joint_max_rad | 0.1191 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.67 | pitch_peak_deg | 7.66 |
| roll_rms_deg | 0.91 | roll_peak_deg | 1.68 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1191 | hip_yaw_div_rms_rad | 0.1041 |
| **Support/Drift** | | | |
| support_rms_m | 0.0633 | support_peak_m | 0.2752 |
| sagittal_drift_m | 0.002 | lateral_drift_m | -0.040 |
| final_displacement_m | 0.040 | max_displacement_m | 0.247 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 10.61 | torque_peak_wheels_nm | 5.59 |
| **Robustness** | | | |
| stability_score | 0.712 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0198 | | |

#### [SUM] low_0p330_sagittal_forward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 10.52 | pitch_min_deg | -6.44 |
| roll_max_deg | 1.68 | roll_min_deg | -0.62 |
| hip_yaw_joint_max_rad | 0.1201 | contact_loss_steps | 3 |
| **Posture** | | | |
| pitch_rms_deg | 4.81 | pitch_peak_deg | 10.52 |
| roll_rms_deg | 0.90 | roll_peak_deg | 1.68 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1201 | hip_yaw_div_rms_rad | 0.1105 |
| **Support/Drift** | | | |
| support_rms_m | 0.0928 | support_peak_m | 0.3831 |
| sagittal_drift_m | -0.019 | lateral_drift_m | -0.020 |
| final_displacement_m | 0.028 | max_displacement_m | 0.352 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 13.14 | torque_peak_wheels_nm | 9.35 |
| **Robustness** | | | |
| stability_score | 0.706 | contact_loss_frac | 0.0015 |
| drift_rate_m_per_kstep | 0.0139 | | |

#### [SUM] mid_0p400_sagittal_backward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 2.99 | pitch_min_deg | -6.55 |
| roll_max_deg | 1.01 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.0975 | contact_loss_steps | 9 |
| **Posture** | | | |
| pitch_rms_deg | 1.73 | pitch_peak_deg | 6.55 |
| roll_rms_deg | 0.57 | roll_peak_deg | 1.01 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0975 | hip_yaw_div_rms_rad | 0.0691 |
| **Support/Drift** | | | |
| support_rms_m | 0.1224 | support_peak_m | 0.3470 |
| sagittal_drift_m | 0.007 | lateral_drift_m | -0.103 |
| final_displacement_m | 0.103 | max_displacement_m | 0.312 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 13.56 | torque_peak_wheels_nm | 7.07 |
| **Robustness** | | | |
| stability_score | 0.879 | contact_loss_frac | 0.0045 |
| drift_rate_m_per_kstep | 0.0516 | | |

#### [SUM] mid_0p400_sagittal_backward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 3.00 | pitch_min_deg | -9.05 |
| roll_max_deg | 1.10 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.1337 | contact_loss_steps | 9 |
| **Posture** | | | |
| pitch_rms_deg | 1.99 | pitch_peak_deg | 9.05 |
| roll_rms_deg | 0.52 | roll_peak_deg | 1.10 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1337 | hip_yaw_div_rms_rad | 0.1247 |
| **Support/Drift** | | | |
| support_rms_m | 0.1739 | support_peak_m | 0.4746 |
| sagittal_drift_m | 0.024 | lateral_drift_m | -0.095 |
| final_displacement_m | 0.098 | max_displacement_m | 0.438 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 13.56 | torque_peak_wheels_nm | 8.22 |
| **Robustness** | | | |
| stability_score | 0.869 | contact_loss_frac | 0.0045 |
| drift_rate_m_per_kstep | 0.0488 | | |

#### [SUM] mid_0p400_sagittal_forward_60N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 9.77 | pitch_min_deg | -3.53 |
| roll_max_deg | 1.00 | roll_min_deg | -0.22 |
| hip_yaw_joint_max_rad | 0.1043 | contact_loss_steps | 9 |
| **Posture** | | | |
| pitch_rms_deg | 2.14 | pitch_peak_deg | 9.77 |
| roll_rms_deg | 0.55 | roll_peak_deg | 1.00 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1043 | hip_yaw_div_rms_rad | 0.0911 |
| **Support/Drift** | | | |
| support_rms_m | 0.0878 | support_peak_m | 0.1882 |
| sagittal_drift_m | 0.004 | lateral_drift_m | -0.108 |
| final_displacement_m | 0.108 | max_displacement_m | 0.157 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 13.56 | torque_peak_wheels_nm | 7.07 |
| **Robustness** | | | |
| stability_score | 0.860 | contact_loss_frac | 0.0045 |
| drift_rate_m_per_kstep | 0.0539 | | |

#### [SUM] mid_0p400_sagittal_forward_90N — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 12.51 | pitch_min_deg | -3.48 |
| roll_max_deg | 1.02 | roll_min_deg | -0.53 |
| hip_yaw_joint_max_rad | 0.1196 | contact_loss_steps | 9 |
| **Posture** | | | |
| pitch_rms_deg | 2.71 | pitch_peak_deg | 12.51 |
| roll_rms_deg | 0.47 | roll_peak_deg | 1.02 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1196 | hip_yaw_div_rms_rad | 0.0977 |
| **Support/Drift** | | | |
| support_rms_m | 0.0906 | support_peak_m | 0.2394 |
| sagittal_drift_m | -0.009 | lateral_drift_m | -0.086 |
| final_displacement_m | 0.087 | max_displacement_m | 0.181 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 13.56 | torque_peak_wheels_nm | 7.91 |
| **Robustness** | | | |
| stability_score | 0.836 | contact_loss_frac | 0.0045 |
| drift_rate_m_per_kstep | 0.0433 | | |

### Scope: step_e

#### [SUM] high_0p430 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 5.51 | pitch_min_deg | -1.63 |
| roll_max_deg | 0.66 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.0701 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 2.91 | pitch_peak_deg | 5.51 |
| roll_rms_deg | 0.40 | roll_peak_deg | 0.66 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0701 | hip_yaw_div_rms_rad | 0.0472 |
| **Support/Drift** | | | |
| support_rms_m | 0.0437 | support_peak_m | 0.1208 |
| sagittal_drift_m | 0.014 | lateral_drift_m | 0.016 |
| final_displacement_m | 0.021 | max_displacement_m | 0.091 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.79 | torque_peak_wheels_nm | 2.96 |
| **Robustness** | | | |
| stability_score | 0.831 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0104 | | |

#### [SUM] high_0p450 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 6.85 | pitch_min_deg | 0.00 |
| roll_max_deg | 0.55 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.0677 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.11 | pitch_peak_deg | 6.85 |
| roll_rms_deg | 0.29 | roll_peak_deg | 0.55 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0677 | hip_yaw_div_rms_rad | 0.0375 |
| **Support/Drift** | | | |
| support_rms_m | 0.0554 | support_peak_m | 0.1225 |
| sagittal_drift_m | 0.024 | lateral_drift_m | -0.021 |
| final_displacement_m | 0.031 | max_displacement_m | 0.104 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.85 | torque_peak_wheels_nm | 4.00 |
| **Robustness** | | | |
| stability_score | 0.777 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0157 | | |

#### [SUM] high_0p465 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 6.34 | pitch_min_deg | -1.63 |
| roll_max_deg | 0.32 | roll_min_deg | -0.01 |
| hip_yaw_joint_max_rad | 0.0303 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.27 | pitch_peak_deg | 6.34 |
| roll_rms_deg | 0.17 | roll_peak_deg | 0.32 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0303 | hip_yaw_div_rms_rad | 0.0267 |
| **Support/Drift** | | | |
| support_rms_m | 0.0512 | support_peak_m | 0.0869 |
| sagittal_drift_m | 0.011 | lateral_drift_m | -0.036 |
| final_displacement_m | 0.037 | max_displacement_m | 0.064 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.09 | torque_peak_wheels_nm | 1.78 |
| **Robustness** | | | |
| stability_score | 0.826 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0187 | | |

#### [SUM] high_0p480 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.26 | pitch_min_deg | -1.03 |
| roll_max_deg | 0.25 | roll_min_deg | -0.03 |
| hip_yaw_joint_max_rad | 0.0571 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.08 | pitch_peak_deg | 7.26 |
| roll_rms_deg | 0.15 | roll_peak_deg | 0.25 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0571 | hip_yaw_div_rms_rad | 0.0362 |
| **Support/Drift** | | | |
| support_rms_m | 0.0542 | support_peak_m | 0.1107 |
| sagittal_drift_m | 0.011 | lateral_drift_m | -0.080 |
| final_displacement_m | 0.081 | max_displacement_m | 0.081 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.02 | torque_peak_wheels_nm | 3.30 |
| **Robustness** | | | |
| stability_score | 0.787 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0403 | | |

#### [SUM] low_0p300 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 6.74 | pitch_min_deg | -1.63 |
| roll_max_deg | 0.93 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.1205 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 2.66 | pitch_peak_deg | 6.74 |
| roll_rms_deg | 0.69 | roll_peak_deg | 0.93 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.1205 | hip_yaw_div_rms_rad | 0.1113 |
| **Support/Drift** | | | |
| support_rms_m | 0.0393 | support_peak_m | 0.1001 |
| sagittal_drift_m | 0.002 | lateral_drift_m | -0.006 |
| final_displacement_m | 0.007 | max_displacement_m | 0.077 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 9.10 | torque_peak_wheels_nm | 1.45 |
| **Robustness** | | | |
| stability_score | 0.825 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0033 | | |

#### [SUM] low_0p320 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.11 | pitch_min_deg | -5.45 |
| roll_max_deg | 1.13 | roll_min_deg | -0.01 |
| hip_yaw_joint_max_rad | 0.0773 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 2.96 | pitch_peak_deg | 5.45 |
| roll_rms_deg | 0.53 | roll_peak_deg | 1.13 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0773 | hip_yaw_div_rms_rad | 0.0585 |
| **Support/Drift** | | | |
| support_rms_m | 0.0351 | support_peak_m | 0.0710 |
| sagittal_drift_m | 0.010 | lateral_drift_m | -0.035 |
| final_displacement_m | 0.037 | max_displacement_m | 0.063 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.77 | torque_peak_wheels_nm | 1.77 |
| **Robustness** | | | |
| stability_score | 0.820 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0183 | | |

#### [SUM] low_0p330 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.16 | pitch_min_deg | -6.16 |
| roll_max_deg | 1.30 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.0857 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.39 | pitch_peak_deg | 6.16 |
| roll_rms_deg | 0.83 | roll_peak_deg | 1.30 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0857 | hip_yaw_div_rms_rad | 0.0889 |
| **Support/Drift** | | | |
| support_rms_m | 0.0306 | support_peak_m | 0.1057 |
| sagittal_drift_m | 0.012 | lateral_drift_m | -0.008 |
| final_displacement_m | 0.014 | max_displacement_m | 0.102 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.58 | torque_peak_wheels_nm | 3.53 |
| **Robustness** | | | |
| stability_score | 0.730 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0072 | | |

#### [SUM] low_0p340 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 4.39 | pitch_min_deg | -2.59 |
| roll_max_deg | 1.44 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.0903 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 1.89 | pitch_peak_deg | 4.39 |
| roll_rms_deg | 0.95 | roll_peak_deg | 1.44 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0903 | hip_yaw_div_rms_rad | 0.0849 |
| **Support/Drift** | | | |
| support_rms_m | 0.0356 | support_peak_m | 0.1035 |
| sagittal_drift_m | 0.013 | lateral_drift_m | -0.010 |
| final_displacement_m | 0.016 | max_displacement_m | 0.106 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.59 | torque_peak_wheels_nm | 1.34 |
| **Robustness** | | | |
| stability_score | 0.849 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0082 | | |

#### [SUM] low_0p360 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 1.49 | pitch_min_deg | -5.01 |
| roll_max_deg | 1.25 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.0825 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 3.01 | pitch_peak_deg | 5.01 |
| roll_rms_deg | 0.66 | roll_peak_deg | 1.25 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0825 | hip_yaw_div_rms_rad | 0.0643 |
| **Support/Drift** | | | |
| support_rms_m | 0.0340 | support_peak_m | 0.0720 |
| sagittal_drift_m | 0.014 | lateral_drift_m | -0.029 |
| final_displacement_m | 0.032 | max_displacement_m | 0.070 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 8.55 | torque_peak_wheels_nm | 1.70 |
| **Robustness** | | | |
| stability_score | 0.810 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0162 | | |

#### [SUM] low_0p380 — OK

| Safety | | | |
|--------|-----|-----|-----|
| pitch_max_deg | 7.47 | pitch_min_deg | 0.00 |
| roll_max_deg | 0.71 | roll_min_deg | 0.00 |
| hip_yaw_joint_max_rad | 0.0336 | contact_loss_steps | 1 |
| **Posture** | | | |
| pitch_rms_deg | 4.77 | pitch_peak_deg | 7.47 |
| roll_rms_deg | 0.34 | roll_peak_deg | 0.71 |
| **Leg Symmetry** | | | |
| hip_yaw_joint_max_rad | 0.0336 | hip_yaw_div_rms_rad | 0.0228 |
| **Support/Drift** | | | |
| support_rms_m | 0.0462 | support_peak_m | 0.0895 |
| sagittal_drift_m | 0.007 | lateral_drift_m | 0.006 |
| final_displacement_m | 0.009 | max_displacement_m | 0.060 |
| **Torque Quality** | | | |
| torque_peak_total_nm | 9.23 | torque_peak_wheels_nm | 3.13 |
| **Robustness** | | | |
| stability_score | 0.741 | contact_loss_frac | 0.0005 |
| drift_rate_m_per_kstep | 0.0044 | | |

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

Full metrics exported to: `docs\validation\k2_candidate_e_quality.json`