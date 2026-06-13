# Step E Hip-Yaw Authority Fix Report

## Executive summary

Controlled hip-yaw authority candidates were evaluated without changing sagittal axis, WBC, legacy torque paths, hip-roll logic, position gains, sagittal velocity damping, height recovery logic, or controller ownership rules.

Selected best candidate: **candidate_b (kp=15.0, kd=3.0)**.

## Files changed

- `wheeled_biped/controllers/shape_posture_controller.py`
- `tests/test_step_e_hip_yaw_authority_fix.py`
- `scripts/evaluate_step_e_hip_yaw_authority.py`

## Candidate profiles tested

- baseline/current: kp_hip_yaw=5.0, kd_hip_yaw=1.0
- candidate A: kp_hip_yaw=10.0, kd_hip_yaw=2.0
- candidate B: kp_hip_yaw=15.0, kd_hip_yaw=3.0
- candidate C: kp_hip_yaw=20.0, kd_hip_yaw=4.0

## Command lines run

- `python scripts/evaluate_step_e_hip_yaw_authority.py`

## Candidate comparison

| Candidate | Steps | kp | kd | max yaw err | RMS yaw err | support max abs | pitch max abs | roll max abs | com_z min | wheel vel max abs | Pass | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline_current | 1000 | 5.0 | 1.0 | 0.109225 | 0.041609 | 0.112218 | 0.073142 | 0.003469 | 0.403835 | 3.163417 | False | hip_yaw_error_above_minimum_threshold;hip_yaw_error_exceeds_0p10 |
| candidate_a | 1000 | 10.0 | 2.0 | 0.071745 | 0.024761 | 0.102117 | 0.070481 | 0.012998 | 0.403835 | 3.140476 | False | hip_yaw_error_above_minimum_threshold |
| candidate_b | 1000 | 15.0 | 3.0 | 0.039158 | 0.016488 | 0.102559 | 0.070554 | 0.011048 | 0.403835 | 3.188482 | True |  |
| candidate_b | 5000 | 15.0 | 3.0 | 0.057555 | 0.023482 | 0.104457 | 0.070771 | 0.012999 | 0.403835 | 3.839568 | True |  |
| candidate_c | 1000 | 20.0 | 4.0 | 0.032265 | 0.015121 | 0.102791 | 0.070590 | 0.010761 | 0.403835 | 3.570819 | True |  |
| candidate_c | 5000 | 20.0 | 4.0 | 0.148771 | 0.087415 | 0.143976 | 0.083449 | 0.010761 | 0.398613 | 5.700391 | False | hip_yaw_error_above_minimum_threshold;hip_yaw_error_exceeds_0p10 |

## Before/after metrics

Baseline reference from previous diagnostics:

- support_position_error max_abs: `0.543381258` m
- final support_position_error: `-0.005957347` m
- pitch_x max_abs: `0.125439967` rad
- roll_y max_abs: `0.044964724` rad
- com_z_min: `0.362271577` m
- wheel_vel_mean max_abs: `7.035612822` rad/s
- peak abs hip-yaw error: `0.109224766` rad
- RMS hip-yaw error: `0.041609009` rad

## Regression checks

Acceptance required survival, WBC off, hidden torque norm zero, ownership violations zero, legacy torque paths off, hip-yaw max error <= 0.07 rad, zero time above 0.10 rad, and no >10% regressions in pitch, roll, support-position max abs, or wheel velocity. It also required com_z_min not more than 0.01 m lower than baseline and non-persistent torque-rate saturation.

## Structural invariants

- Sagittal axis was not flipped.
- Hip-roll was not modified.
- No WBC or legacy torque path was introduced.
- Balance-core four-source architecture was preserved.
- Controller ownership rules were not modified.

## Missing artifacts

- hip_yaw_authority_fix_report.md
- hip_yaw_authority_fix_summary.json
