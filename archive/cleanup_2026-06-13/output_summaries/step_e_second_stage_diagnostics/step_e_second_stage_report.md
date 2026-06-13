# Step E Second-Stage Diagnostics Report

## Executive summary

Simple sagittal-axis flipping is rejected. The first-stage current 5000-step run had max drift about 0.543 m and final drift about -0.006 m, while the flipped run had max/final drift about -20.667 m. H1 therefore indicates sign-convention ambiguity, not a simple axis-flip fix.

Current-axis transient classification: **height_drop_coupled_transient**.
Hip-yaw posture classification: **shape-posture torque too weak**.
Final next recommended fix: **increase_or_redesign_hip_yaw_posture_authority**.

## Environment

- Commit: `9971615447f36e0127482db28c5b4139b742bc3b`
- Date/time UTC: `2026-06-01T03:27:10.713181+00:00`
- Python: `3.10.2 (tags/v3.10.2:a58ebcc, Jan 17 2022, 14:12:15) [MSC v.1929 64 bit (AMD64)]`
- MuJoCo: `3.6.0`
- Platform: `Windows-10-10.0.26200-SP0`
- Inputs: `outputs/step_e_root_cause_diagnostics/`
- Outputs: `outputs/step_e_second_stage_diagnostics/`

## Part A: Current-axis transient drift root-cause analysis

Peak support-position excursion:

- Peak step: `1520`
- Peak support_position_error_m: `0.543381258`
- Window start/end steps: `1320` / `1720`
- Support error before/after window: `0.126321159` / `0.234310785`
- Current 5000-step final support error: `-0.005957347`
- Position problem type: `transient`

Peak-state metrics:

- pitch_x_rad: `0.097467179`
- pitch_x_error_rad: `0.098484027`
- pitch_rate_x_rad_s: `-0.101961687`
- roll_y_rad: `-0.000177344`
- yaw_z_rad: `0.021762956`
- com_z_m: `0.365203112`
- wheel_vel_mean_rad_s: `0.076095354`
- tau_position: `-3.000000000`
- tau_position_raw: `-3.000000000`
- tau_position_saturation_flag: `True`
- tau_pitch: `4.924201349`
- tau_pitch_rate: `0.000000000`
- tau_sagittal_velocity: `-0.496745110`
- tau_support_velocity: `0.000000000`
- tau_total_before_final_clip: `0.213037580`
- tau_total_after_final_clip: `0.213037580`
- final_wheel_torque_margin: `29.785622570`
- wheel torque-rate saturated near peak: `False`
- contact valid at peak: `True`
- ownership_violation_count: `0`
- hidden_torque_norm: `0.000000000`
- tau_wbc_norm: `0.000000000`

Quantitative classification: **height_drop_coupled_transient**.

Tau_position saturation involved: `True`.
Pitch priority override evidence: `abs(tau_pitch)=4.924201` vs `abs(tau_position)=3.000000`.
Wheel velocity runaway evidence: `abs(wheel_vel_mean)=0.076095` rad/s.

## Part B: Hip-yaw confirmed posture error diagnosis

- Peak abs hip-yaw error: `0.109224766` rad
- RMS hip-yaw error: `0.041609009` rad
- Peak abs shape-posture hip-yaw torque: `0.584558840` Nm
- RMS shape-posture hip-yaw torque: `0.215325320` Nm
- Error/torque correlation: `0.999446154`
- Left positive torque gives positive joint delta: `True`
- Right positive torque gives positive joint delta: `True`
- Shape controller torque reduces left error: `True`
- Shape controller torque reduces right error: `True`

Hip-yaw sign/authority conclusion: **shape-posture torque too weak**.

## Missing artifacts

None

## Final next recommended fix

**increase_or_redesign_hip_yaw_posture_authority**

No production fix was made. Do not tune gains, add WBC, modify hip-roll, or flip sagittal axis based on this report.
