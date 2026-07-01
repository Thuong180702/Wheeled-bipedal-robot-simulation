# K2 Stability Improvement — Candidate Evaluation

**Classification:** `STABILITY_PARTIAL`
**baseline_source:** docs/validation/k2_default_v2_quality.json
**candidate_source:** docs/validation/k2_default_v2_heading_height_twist_candidate_quality.json
**evaluator_version:** 1.0.0

## Executive Summary

| Metric | Value |
|--------|-------|
| Classification | **STABILITY_PARTIAL** |
| Baseline aggregate score | 0.6936 |
| Candidate aggregate score | 0.6951 |
| Score delta | +0.0015 |
| Evaluated scenarios | 35/35 |
| Safety fails | 0 |
| Major regressions | 0 |
| Baseline performance | 94.7 Hz |
| Candidate performance | 170.1 Hz |

## Per-Scenario Results

| Scenario | Type | Base Score | Cand Score | Delta | Safety | Regressions |
|----------|------|------------|------------|-------|--------|-------------|
| C1_slow_ladder_up_down | other | 0.7437 | 0.7511 | +0.0073 | PASS | OK |
| C2_random_500dwell | other | 0.7437 | 0.7511 | +0.0073 | PASS | OK |
| C3_random_200dwell | other | 0.7437 | 0.7511 | +0.0073 | PASS | OK |
| C4_abrupt_stress | other | 0.7437 | 0.7511 | +0.0073 | PASS | OK |
| C5_long_random | other | 0.6842 | 0.7146 | +0.0305 | PASS | OK |
| focused_high_0p480 | fixed_height | 0.7166 | 0.7162 | -0.0004 | PASS | OK |
| focused_low_0p320 | fixed_height | 0.7296 | 0.7173 | -0.0124 | PASS | OK |
| gate_chatter_0p400_0p470 | dynamic_height | 0.6919 | 0.6927 | +0.0007 | PASS | OK |
| gate_dwell_0p420_0p450_0p480 | dynamic_height | 0.4455 | 0.4560 | +0.0105 | PASS | OK |
| high_0p430 | fixed_height | 0.7185 | 0.6919 | -0.0266 | PASS | OK |
| high_0p450 | fixed_height | 0.6052 | 0.6149 | +0.0097 | PASS | OK |
| high_0p465 | fixed_height | 0.7162 | 0.7160 | -0.0002 | PASS | OK |
| high_0p480 | fixed_height | 0.6774 | 0.6754 | -0.0019 | PASS | OK |
| high_0p480_sagittal_backward_60N | push | 0.6557 | 0.6557 | -0.0000 | PASS | OK |
| high_0p480_sagittal_backward_90N | push | 0.6673 | 0.6673 | -0.0000 | PASS | OK |
| high_0p480_sagittal_forward_60N | push | 0.6722 | 0.6722 | +0.0000 | PASS | OK |
| high_0p480_sagittal_forward_90N | push | 0.6195 | 0.6195 | -0.0000 | PASS | OK |
| low_0p300 | fixed_height | 0.7665 | 0.7615 | -0.0050 | PASS | OK |
| low_0p320 | fixed_height | 0.7296 | 0.7173 | -0.0124 | PASS | OK |
| low_0p330 | fixed_height | 0.6335 | 0.6571 | +0.0236 | PASS | OK |
| low_0p330_sagittal_backward_60N | push | 0.6889 | 0.6937 | +0.0048 | PASS | OK |
| low_0p330_sagittal_backward_90N | push | 0.6664 | 0.6655 | -0.0009 | PASS | OK |
| low_0p330_sagittal_forward_60N | push | 0.6685 | 0.6712 | +0.0026 | PASS | OK |
| low_0p330_sagittal_forward_90N | push | 0.5951 | 0.5862 | -0.0089 | PASS | OK |
| low_0p340 | fixed_height | 0.8335 | 0.8336 | +0.0001 | PASS | OK |
| low_0p360 | fixed_height | 0.7589 | 0.7686 | +0.0097 | PASS | OK |
| low_0p380 | fixed_height | 0.6804 | 0.6824 | +0.0020 | PASS | OK |
| mid_0p400 | fixed_height | 0.7390 | 0.7098 | -0.0292 | PASS | OK |
| mid_0p400_sagittal_backward_60N | push | 0.7583 | 0.7594 | +0.0011 | PASS | OK |
| mid_0p400_sagittal_backward_90N | push | 0.6999 | 0.6932 | -0.0067 | PASS | OK |
| mid_0p400_sagittal_forward_60N | push | 0.7215 | 0.7281 | +0.0066 | PASS | OK |
| mid_0p400_sagittal_forward_90N | push | 0.6757 | 0.6743 | -0.0014 | PASS | OK |
| ramp_down_0p480_to_0p330 | dynamic_height | 0.5517 | 0.5500 | -0.0017 | PASS | OK |
| ramp_up_0p330_to_0p480 | dynamic_height | 0.7979 | 0.7952 | -0.0027 | PASS | OK |
| up_down_cycle_0p330_0p480_0p330 | dynamic_height | 0.7348 | 0.7674 | +0.0326 | PASS | OK |

## Dimension Score Comparison

| Dimension | Baseline Mean | Candidate Mean | Delta |
|-----------|---------------|----------------|-------|
| Posture | 0.6781 | 0.6830 | +0.0049 |
| Support/Drift | 0.4998 | 0.4936 | -0.0061 |
| Leg Health | 0.8348 | 0.8423 | +0.0075 |
| Dynamic Height | 0.5371 | 0.5371 | +0.0000 |
| Torque Quality | 0.9139 | 0.9141 | +0.0002 |
| Robustness | 0.9302 | 0.9315 | +0.0013 |

## Classification

**Result:** `STABILITY_PARTIAL`

- Some improvements, but aggregate score below 0.80
- Current score: 0.6951 (need >= 0.80)