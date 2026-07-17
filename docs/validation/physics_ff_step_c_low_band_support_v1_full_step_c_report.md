# Physics FF Low-Band Support v1 Full Step C Report

Date: 2026-06-21

Classification: `PHYSICS_FF_LOW_BAND_V1_STEP_C_FAIL`

## Scope

This validation compares three opt-in/selected profiles without changing defaults:

- A Baseline: `calibrated_support_position_outer_loop_pitch_ref_v2`
- B Current PFF: `physics_equilibrium_feedforward_outer_loop`
- C Candidate: `physics_equilibrium_feedforward_outer_loop_low_band_support_v1`

The suite uses `outputs/physical_target_height_setups_centered` (`centered_posture_height_schedule`).
The project simulator currently validates the random/changing-height cases as fixed-height dwell segments, matching the existing Step C random-height artifacts.

Corrected hip-yaw policy was used: `hip_yaw_abs_max_tracking` is preferred, then `hip_yaw_abs_max`, then per-joint hip-yaw error/position fallbacks.
`tau_wbc_norm` is treated as diagnostic only; WBC applied rows come from ownership, per-actuator authority, or nonzero post-authority WBC torque rows.

## Artifacts

- Output directory: `outputs/physics_ff_step_c_low_band_support_v1_full_step_c/`
- Segment metrics: `outputs/physics_ff_step_c_low_band_support_v1_full_step_c/step_c_segment_metrics.csv`
- Step C summary: `outputs/physics_ff_step_c_low_band_support_v1_full_step_c/step_c_case_summary.csv`
- Fixed-height metrics: `outputs/physics_ff_step_c_low_band_support_v1_full_step_c/fixed_height_metrics.csv`
- Fixed-height summary: `outputs/physics_ff_step_c_low_band_support_v1_full_step_c/fixed_height_summary.csv`
- Decision JSON: `outputs/physics_ff_step_c_low_band_support_v1_full_step_c/decision_summary.json`

## Gate Summary

- Elapsed wall time: 37.6 s
- Fixed-height dwell: 2000 steps per height/profile
- Fixed-height classification: `FAIL`
- Hard failures: 1
- Monitoring items: 0
- Inconclusive items: 0

Hard failure details:
- fixed:fixed_height:1:low_0p320:vs_current_PFF:p2p

## Step C Case Summary

| Case | Profile | any_fell | any_unsafe | max_maxabs m | max_trans m | p2p m | out15% | pitch max deg | roll max deg | hip-yaw max | WBC rows | low-band scale max | Kp eff max | trim max deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1_slow_ladder_up_down | Baseline B2v2 | False | False | 0.1748 | 0.0315 | 0.1928 | 21.4 | 6.95 | 0.93 | 0.0602 | 0 | 0.0000 | 1.000 | 0.000 |
| C1_slow_ladder_up_down | Current PFF | False | False | 0.1412 | 0.0316 | 0.1765 | 0.0 | 6.73 | 0.84 | 0.0548 | 0 | 0.0000 | 1.000 | 0.000 |
| C1_slow_ladder_up_down | Low-band support v1 | False | False | 0.1305 | 0.0310 | 0.1659 | 0.0 | 6.72 | 0.85 | 0.0599 | 0 | 1.0000 | 1.500 | 1.000 |
| C2_random_500dwell | Baseline B2v2 | False | False | 0.1748 | 0.0238 | 0.1928 | 14.0 | 7.10 | 1.13 | 0.0762 | 0 | 0.0000 | 1.000 | 0.000 |
| C2_random_500dwell | Current PFF | False | False | 0.1311 | 0.0247 | 0.1648 | 0.0 | 6.73 | 0.85 | 0.0684 | 0 | 0.0000 | 1.000 | 0.000 |
| C2_random_500dwell | Low-band support v1 | False | False | 0.1302 | 0.0248 | 0.1450 | 0.0 | 6.72 | 0.89 | 0.0684 | 0 | 1.0000 | 1.500 | 1.000 |
| C3_random_200dwell | Baseline B2v2 | False | False | 0.1748 | 0.0315 | 0.1928 | 35.1 | 6.95 | 0.55 | 0.0380 | 0 | 0.0000 | 1.000 | 0.000 |
| C3_random_200dwell | Current PFF | False | False | 0.1412 | 0.0316 | 0.1765 | 0.0 | 5.77 | 0.55 | 0.0372 | 0 | 0.0000 | 1.000 | 0.000 |
| C3_random_200dwell | Low-band support v1 | False | False | 0.1305 | 0.0310 | 0.1659 | 0.0 | 5.77 | 0.55 | 0.0376 | 0 | 1.0000 | 1.500 | 1.000 |
| C4_abrupt_stress | Baseline B2v2 | False | False | 0.1375 | 0.0315 | 0.1729 | 0.0 | 5.96 | 0.82 | 0.0438 | 0 | 0.0000 | 1.000 | 0.000 |
| C4_abrupt_stress | Current PFF | False | False | 0.1412 | 0.0316 | 0.1765 | 0.0 | 6.73 | 0.81 | 0.0478 | 0 | 0.0000 | 1.000 | 0.000 |
| C4_abrupt_stress | Low-band support v1 | False | False | 0.1305 | 0.0310 | 0.1659 | 0.0 | 6.72 | 0.81 | 0.0478 | 0 | 0.2494 | 0.374 | 0.249 |
| C5_long_random | Baseline B2v2 | False | False | 0.1748 | 0.0315 | 0.1928 | 35.2 | 6.95 | 0.98 | 0.0602 | 0 | 0.0000 | 1.000 | 0.000 |
| C5_long_random | Current PFF | False | False | 0.1412 | 0.0316 | 0.1765 | 0.0 | 6.73 | 0.85 | 0.0548 | 0 | 0.0000 | 1.000 | 0.000 |
| C5_long_random | Low-band support v1 | False | False | 0.1305 | 0.0310 | 0.1659 | 0.0 | 6.72 | 0.89 | 0.0599 | 0 | 1.0000 | 1.500 | 1.000 |
| focused_low_0p320 | Baseline B2v2 | False | False | 0.0715 | 0.0209 | 0.1410 | 0.0 | 5.50 | 0.84 | 0.0602 | 0 | 0.0000 | 1.000 | 0.000 |
| focused_low_0p320 | Current PFF | False | False | 0.1158 | 0.0232 | 0.1648 | 0.0 | 5.39 | 0.84 | 0.0548 | 0 | 0.0000 | 1.000 | 0.000 |
| focused_low_0p320 | Low-band support v1 | False | False | 0.0726 | 0.0210 | 0.1414 | 0.0 | 5.47 | 0.85 | 0.0599 | 0 | 1.0000 | 1.500 | 1.000 |
| focused_high_0p480 | Baseline B2v2 | False | False | 0.0329 | 0.0030 | 0.0427 | 0.0 | 4.51 | 0.10 | 0.0107 | 0 | 0.0000 | 1.000 | 0.000 |
| focused_high_0p480 | Current PFF | False | False | 0.0140 | 0.0045 | 0.0227 | 0.0 | 4.17 | 0.10 | 0.0119 | 0 | 0.0000 | 1.000 | 0.000 |
| focused_high_0p480 | Low-band support v1 | False | False | 0.0140 | 0.0045 | 0.0227 | 0.0 | 4.17 | 0.10 | 0.0119 | 0 | 0.0000 | 0.000 | 0.000 |

## Candidate Comparisons

| Case | C maxabs vs A m | C maxabs vs B m | C p2p vs A % | C p2p vs B % | C out15 vs A pp | C out15 vs B pp |
|---|---:|---:|---:|---:|---:|---:|
| C1_slow_ladder_up_down | -0.0443 | -0.0107 | -13.94 | -5.97 | -21.4 | 0.0 |
| C2_random_500dwell | -0.0446 | -0.0008 | -24.77 | -11.99 | -14.0 | 0.0 |
| C3_random_200dwell | -0.0443 | -0.0107 | -13.94 | -5.97 | -35.1 | 0.0 |
| C4_abrupt_stress | -0.0069 | -0.0107 | -4.01 | -5.97 | 0.0 | 0.0 |
| C5_long_random | -0.0443 | -0.0107 | -13.94 | -5.97 | -35.2 | 0.0 |
| focused_low_0p320 | 0.0010 | -0.0433 | 0.35 | -14.17 | 0.0 | 0.0 |
| focused_high_0p480 | -0.0189 | 0.0000 | -46.85 | 0.00 | 0.0 | 0.0 |

## Focused Gates

| Case | Profile | maxabs m | p2p m | out15% | pitch max deg | hip-yaw max | hidden max | WBC rows | low-band scale max | Kp eff max | trim max deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| focused_low_0p320 | Baseline B2v2 | 0.0715256847 | 0.1409512889 | 0.0 | 5.50 | 0.0602 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| focused_low_0p320 | Current PFF | 0.1158272566 | 0.1648027009 | 0.0 | 5.39 | 0.0548 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| focused_low_0p320 | Low-band support v1 | 0.0725591931 | 0.1414486599 | 0.0 | 5.47 | 0.0599 | 0.0000 | 0 | 1.0000 | 1.500 | 1.000 |
| focused_high_0p480 | Baseline B2v2 | 0.0329109804 | 0.0427096017 | 0.0 | 4.51 | 0.0107 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| focused_high_0p480 | Current PFF | 0.0139655546 | 0.0227017949 | 0.0 | 4.17 | 0.0119 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| focused_high_0p480 | Low-band support v1 | 0.0139655546 | 0.0227017949 | 0.0 | 4.17 | 0.0119 | 0.0000 | 0 | 0.0000 | 0.000 | 0.000 |

## Fixed-Height 10-Height Summary

| Height | Profile | any_fell | any_unsafe | maxabs m | p2p m | out15% | pitch max deg | roll max deg | hip-yaw max | hidden max | WBC rows | low-band scale max | Kp eff max | trim max deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low_0p300 | Baseline B2v2 | False | False | 0.0874291862 | 0.1594604060 | 0.0 | 6.65 | 0.82 | 0.2125 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| low_0p300 | Current PFF | False | False | 0.1033401284 | 0.1407840175 | 0.0 | 6.73 | 0.81 | 0.2034 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| low_0p300 | Low-band support v1 | False | False | 0.1031951485 | 0.1406596312 | 0.0 | 6.72 | 0.81 | 0.2033 | 0.0000 | 0 | 0.0039 | 0.006 | 0.004 |
| low_0p320 | Baseline B2v2 | False | False | 0.1477870331 | 0.2789445243 | 0.0 | 8.28 | 0.90 | 0.1694 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| low_0p320 | Current PFF | False | False | 0.1548515529 | 0.2452056449 | 1.5 | 7.28 | 0.93 | 0.1616 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| low_0p320 | Low-band support v1 | False | False | 0.1488253699 | 0.2849946604 | 0.0 | 8.55 | 0.90 | 0.1735 | 0.0000 | 0 | 1.0000 | 1.500 | 1.000 |
| low_0p330 | Baseline B2v2 | False | False | 0.1374565746 | 0.2107868242 | 0.0 | 7.60 | 1.12 | 0.1654 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| low_0p330 | Current PFF | False | False | 0.1412146889 | 0.2174361304 | 0.0 | 7.70 | 1.21 | 0.1756 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| low_0p330 | Low-band support v1 | False | False | 0.1305278906 | 0.2191093989 | 0.0 | 8.05 | 1.29 | 0.1839 | 0.0000 | 0 | 0.2494 | 0.374 | 0.249 |
| low_0p340 | Baseline B2v2 | False | False | 0.1634753494 | 0.2428558733 | 2.6 | 6.92 | 1.23 | 0.1830 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| low_0p340 | Current PFF | False | False | 0.1515258720 | 0.2718639230 | 0.7 | 8.11 | 0.98 | 0.1860 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| low_0p340 | Low-band support v1 | False | False | 0.1452146738 | 0.2634766251 | 0.0 | 7.97 | 1.13 | 0.1872 | 0.0000 | 0 | 0.0039 | 0.006 | 0.004 |
| low_0p360 | Baseline B2v2 | False | False | 0.1748376595 | 0.2060580947 | 3.2 | 4.46 | 1.30 | 0.1789 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| low_0p360 | Current PFF | False | False | 0.1310727611 | 0.1863893965 | 0.0 | 4.56 | 1.16 | 0.1531 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| low_0p360 | Low-band support v1 | False | False | 0.1302269238 | 0.1801366885 | 0.0 | 4.28 | 1.14 | 0.1502 | 0.0000 | 0 | 0.0000 | 0.000 | 0.000 |
| low_0p380 | Baseline B2v2 | False | False | 0.1620237096 | 0.1859077107 | 3.8 | 5.99 | 1.17 | 0.1572 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| low_0p380 | Current PFF | False | False | 0.1392457300 | 0.2191931231 | 0.0 | 7.29 | 1.00 | 0.1561 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| low_0p380 | Low-band support v1 | False | False | 0.1269999718 | 0.1967641075 | 0.0 | 6.78 | 1.00 | 0.1365 | 0.0000 | 0 | 0.0000 | 0.000 | 0.000 |
| high_0p430 | Baseline B2v2 | False | False | 0.1173803937 | 0.1617947861 | 0.0 | 7.15 | 0.65 | 0.0579 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| high_0p430 | Current PFF | False | False | 0.1222769910 | 0.1976333598 | 0.0 | 8.60 | 0.49 | 0.0283 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| high_0p430 | Low-band support v1 | False | False | 0.1222769910 | 0.1976333598 | 0.0 | 8.60 | 0.49 | 0.0283 | 0.0000 | 0 | 0.0000 | 0.000 | 0.000 |
| high_0p450 | Baseline B2v2 | False | False | 0.1808083339 | 0.2864761054 | 8.9 | 10.23 | 0.42 | 0.0290 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| high_0p450 | Current PFF | False | False | 0.1185123391 | 0.1324036337 | 0.0 | 5.69 | 0.62 | 0.1067 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| high_0p450 | Low-band support v1 | False | False | 0.1185123391 | 0.1324036337 | 0.0 | 5.69 | 0.62 | 0.1067 | 0.0000 | 0 | 0.0000 | 0.000 | 0.000 |
| high_0p465 | Baseline B2v2 | False | False | 0.1529834623 | 0.3002280349 | 1.3 | 9.79 | 0.36 | 0.0345 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| high_0p465 | Current PFF | False | False | 0.1656217508 | 0.2820372138 | 4.9 | 9.58 | 0.34 | 0.0291 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| high_0p465 | Low-band support v1 | False | False | 0.1656217508 | 0.2820372138 | 4.9 | 9.58 | 0.34 | 0.0291 | 0.0000 | 0 | 0.0000 | 0.000 | 0.000 |
| high_0p480 | Baseline B2v2 | False | False | 0.1854218257 | 0.3206495316 | 9.9 | 11.31 | 0.28 | 0.0454 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| high_0p480 | Current PFF | False | False | 0.1519968199 | 0.2957432948 | 1.4 | 10.29 | 0.25 | 0.0725 | 0.0000 | 0 | 0.0000 | 1.000 | 0.000 |
| high_0p480 | Low-band support v1 | False | False | 0.1519968199 | 0.2957432948 | 1.4 | 10.29 | 0.25 | 0.0725 | 0.0000 | 0 | 0.0000 | 0.000 | 0.000 |

## Decision

Step C failed. Candidate worse than original/current PFF on focused low maxabs: no.
Failing protected metric: fixed-height `low_0p320` P2P 0.2849946604 m vs current PFF 0.2452056449 m; 15% threshold 0.2819864917 m; exceeded by 0.0030081687 m.
Worse than current/original PFF on the failing fixed-height P2P metric: yes.
Step D was not run.
The candidate remains opt-in; this report does not promote PFF or change defaults.
