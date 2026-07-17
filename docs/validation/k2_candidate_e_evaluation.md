# K2 Stability Improvement — Candidate Evaluation

**Classification:** `STABILITY_REGRESSED`
**baseline_source:** docs/validation/k2_improvement_baseline_quality.json
**candidate_source:** docs/validation/k2_candidate_e_quality.json
**evaluator_version:** 1.0.0

## Executive Summary

| Metric | Value |
|--------|-------|
| Classification | **STABILITY_REGRESSED** |
| Baseline aggregate score | 0.6834 |
| Candidate aggregate score | 0.6906 |
| Score delta | +0.0072 |
| Evaluated scenarios | 35/35 |
| Safety fails | 0 |
| Major regressions | 3 |
| Baseline performance | 147.4 Hz |
| Candidate performance | 191.8 Hz |

## Per-Scenario Results

| Scenario | Type | Base Score | Cand Score | Delta | Safety | Regressions |
|----------|------|------------|------------|-------|--------|-------------|
| C1_slow_ladder_up_down | other | 0.7437 | 0.7045 | -0.0392 | PASS | OK |
| C2_random_500dwell | other | 0.7437 | 0.7045 | -0.0392 | PASS | OK |
| C3_random_200dwell | other | 0.7437 | 0.7045 | -0.0392 | PASS | OK |
| C4_abrupt_stress | other | 0.7437 | 0.7045 | -0.0392 | PASS | OK |
| C5_long_random | other | 0.6842 | 0.6743 | -0.0098 | PASS | OK |
| focused_high_0p480 | fixed_height | 0.7166 | 0.6901 | -0.0265 | PASS | OK |
| focused_low_0p320 | fixed_height | 0.7132 | 0.7813 | +0.0680 | PASS | OK |
| gate_chatter_0p400_0p470 | dynamic_height | 0.6919 | 0.6575 | -0.0345 | PASS | OK |
| gate_dwell_0p420_0p450_0p480 | dynamic_height | 0.4455 | 0.4468 | +0.0013 | PASS | OK |
| high_0p430 | fixed_height | 0.6727 | 0.7157 | +0.0429 | PASS | OK |
| high_0p450 | fixed_height | 0.6052 | 0.6537 | +0.0485 | PASS | OK |
| high_0p465 | fixed_height | 0.7136 | 0.7763 | +0.0627 | PASS | OK |
| high_0p480 | fixed_height | 0.6756 | 0.6567 | -0.0189 | PASS | OK |
| high_0p480_sagittal_backward_60N | push | 0.6557 | 0.7059 | +0.0502 | PASS | OK |
| high_0p480_sagittal_backward_90N | push | 0.6673 | 0.6760 | +0.0087 | PASS | OK |
| high_0p480_sagittal_forward_60N | push | 0.6722 | 0.6337 | -0.0385 | PASS | OK |
| high_0p480_sagittal_forward_90N | push | 0.6195 | 0.6605 | +0.0410 | PASS | OK |
| low_0p300 | fixed_height | 0.7665 | 0.7657 | -0.0008 | PASS | OK |
| low_0p320 | fixed_height | 0.6962 | 0.7813 | +0.0851 | PASS | OK |
| low_0p330 | fixed_height | 0.6335 | 0.6472 | +0.0137 | PASS | OK |
| low_0p330_sagittal_backward_60N | push | 0.6938 | 0.6630 | -0.0308 | PASS | OK |
| low_0p330_sagittal_backward_90N | push | 0.6373 | 0.6365 | -0.0008 | PASS | OK |
| low_0p330_sagittal_forward_60N | push | 0.6685 | 0.6427 | -0.0258 | PASS | OK |
| low_0p330_sagittal_forward_90N | push | 0.5973 | 0.6289 | +0.0316 | PASS | OK |
| low_0p340 | fixed_height | 0.8335 | 0.8112 | -0.0224 | PASS | OK |
| low_0p360 | fixed_height | 0.7444 | 0.7814 | +0.0370 | PASS | OK |
| low_0p380 | fixed_height | 0.6883 | 0.7116 | +0.0233 | PASS | OK |
| mid_0p400 | fixed_height | 0.7247 | 0.7242 | -0.0005 | PASS | OK |
| mid_0p400_sagittal_backward_60N | push | 0.7578 | 0.7374 | -0.0204 | PASS | OK |
| mid_0p400_sagittal_backward_90N | push | 0.6889 | 0.6908 | +0.0018 | PASS | OK |
| mid_0p400_sagittal_forward_60N | push | 0.6997 | 0.7025 | +0.0028 | PASS | OK |
| mid_0p400_sagittal_forward_90N | push | 0.6943 | 0.6737 | -0.0206 | PASS | OK |
| ramp_down_0p480_to_0p330 | dynamic_height | 0.4535 | 0.5112 | +0.0577 | PASS | 3 regressions |
| ramp_up_0p330_to_0p480 | dynamic_height | 0.6996 | 0.7711 | +0.0715 | PASS | OK |
| up_down_cycle_0p330_0p480_0p330 | dynamic_height | 0.7322 | 0.7433 | +0.0111 | PASS | OK |

## Dimension Score Comparison

| Dimension | Baseline Mean | Candidate Mean | Delta |
|-----------|---------------|----------------|-------|
| Posture | 0.6749 | 0.6877 | +0.0128 |
| Support/Drift | 0.4895 | 0.5006 | +0.0111 |
| Leg Health | 0.8236 | 0.7950 | -0.0287 |
| Dynamic Height | 0.5128 | 0.5371 | +0.0243 |
| Torque Quality | 0.8954 | 0.9134 | +0.0180 |
| Robustness | 0.9298 | 0.9299 | +0.0001 |

## Regression Details

| Scenario | Metric | Baseline | Candidate | Delta | Limit |
|----------|--------|----------|-----------|-------|-------|
| ramp_down_0p480_to_0p330 | leg_symmetry.hip_yaw_joint_max_rad | 0.1191 | 0.2242 | +0.1051 | 0.0500 |
| ramp_down_0p480_to_0p330 | support_drift.support_rms_m | 0.4590 | 0.5538 | +0.0948 | 0.0150 |
| ramp_down_0p480_to_0p330 | support_drift.final_displacement_m | 1.2909 | 1.4862 | +0.1953 | 0.1000 |

## Classification

**Result:** `STABILITY_REGRESSED`

- 3 major regression(s) detected