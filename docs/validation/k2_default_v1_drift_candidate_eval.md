# K2 Stability Improvement — Candidate Evaluation

**Classification:** `STABILITY_PARTIAL`
**baseline_source:** docs/validation/k2_default_v1_quality.json
**candidate_source:** docs/validation/k2_default_v1_drift_candidate_quality.json
**evaluator_version:** 1.0.0

## Executive Summary

| Metric | Value |
|--------|-------|
| Classification | **STABILITY_PARTIAL** |
| Baseline aggregate score | 0.6935 |
| Candidate aggregate score | 0.6923 |
| Score delta | -0.0012 |
| Evaluated scenarios | 35/35 |
| Safety fails | 0 |
| Major regressions | 0 |
| Baseline performance | 150.2 Hz |
| Candidate performance | 164.2 Hz |

## Per-Scenario Results

| Scenario | Type | Base Score | Cand Score | Delta | Safety | Regressions |
|----------|------|------------|------------|-------|--------|-------------|
| C1_slow_ladder_up_down | other | 0.7437 | 0.7437 | +0.0000 | PASS | OK |
| C2_random_500dwell | other | 0.7437 | 0.7437 | +0.0000 | PASS | OK |
| C3_random_200dwell | other | 0.7437 | 0.7437 | +0.0000 | PASS | OK |
| C4_abrupt_stress | other | 0.7437 | 0.7437 | +0.0000 | PASS | OK |
| C5_long_random | other | 0.6842 | 0.6842 | +0.0000 | PASS | OK |
| focused_high_0p480 | fixed_height | 0.7166 | 0.7166 | +0.0000 | PASS | OK |
| focused_low_0p320 | fixed_height | 0.7313 | 0.7168 | -0.0145 | PASS | OK |
| gate_chatter_0p400_0p470 | dynamic_height | 0.6919 | 0.6919 | +0.0000 | PASS | OK |
| gate_dwell_0p420_0p450_0p480 | dynamic_height | 0.4456 | 0.4456 | -0.0000 | PASS | OK |
| high_0p430 | fixed_height | 0.7286 | 0.7035 | -0.0251 | PASS | OK |
| high_0p450 | fixed_height | 0.6052 | 0.6052 | +0.0000 | PASS | OK |
| high_0p465 | fixed_height | 0.7164 | 0.7163 | -0.0001 | PASS | OK |
| high_0p480 | fixed_height | 0.6756 | 0.6756 | +0.0000 | PASS | OK |
| high_0p480_sagittal_backward_60N | push | 0.6557 | 0.6557 | +0.0000 | PASS | OK |
| high_0p480_sagittal_backward_90N | push | 0.6673 | 0.6673 | +0.0000 | PASS | OK |
| high_0p480_sagittal_forward_60N | push | 0.6722 | 0.6722 | +0.0000 | PASS | OK |
| high_0p480_sagittal_forward_90N | push | 0.6195 | 0.6195 | +0.0000 | PASS | OK |
| low_0p300 | fixed_height | 0.7665 | 0.7665 | +0.0000 | PASS | OK |
| low_0p320 | fixed_height | 0.7313 | 0.7168 | -0.0145 | PASS | OK |
| low_0p330 | fixed_height | 0.6335 | 0.6335 | +0.0000 | PASS | OK |
| low_0p330_sagittal_backward_60N | push | 0.6913 | 0.6935 | +0.0022 | PASS | OK |
| low_0p330_sagittal_backward_90N | push | 0.6640 | 0.6640 | +0.0000 | PASS | OK |
| low_0p330_sagittal_forward_60N | push | 0.6685 | 0.6685 | +0.0000 | PASS | OK |
| low_0p330_sagittal_forward_90N | push | 0.5953 | 0.5952 | -0.0001 | PASS | OK |
| low_0p340 | fixed_height | 0.8335 | 0.8335 | +0.0000 | PASS | OK |
| low_0p360 | fixed_height | 0.7589 | 0.7589 | +0.0000 | PASS | OK |
| low_0p380 | fixed_height | 0.6897 | 0.6896 | -0.0001 | PASS | OK |
| mid_0p400 | fixed_height | 0.7390 | 0.7390 | +0.0000 | PASS | OK |
| mid_0p400_sagittal_backward_60N | push | 0.7583 | 0.7583 | +0.0000 | PASS | OK |
| mid_0p400_sagittal_backward_90N | push | 0.6999 | 0.6999 | +0.0000 | PASS | OK |
| mid_0p400_sagittal_forward_60N | push | 0.7215 | 0.7215 | +0.0000 | PASS | OK |
| mid_0p400_sagittal_forward_90N | push | 0.6757 | 0.6757 | +0.0000 | PASS | OK |
| ramp_down_0p480_to_0p330 | dynamic_height | 0.5517 | 0.5517 | +0.0000 | PASS | OK |
| ramp_up_0p330_to_0p480 | dynamic_height | 0.7787 | 0.7787 | -0.0000 | PASS | OK |
| up_down_cycle_0p330_0p480_0p330 | dynamic_height | 0.7320 | 0.7413 | +0.0093 | PASS | OK |

## Dimension Score Comparison

| Dimension | Baseline Mean | Candidate Mean | Delta |
|-----------|---------------|----------------|-------|
| Posture | 0.6776 | 0.6775 | -0.0001 |
| Support/Drift | 0.5019 | 0.4956 | -0.0063 |
| Leg Health | 0.8326 | 0.8333 | +0.0008 |
| Dynamic Height | 0.5371 | 0.5371 | -0.0000 |
| Torque Quality | 0.9139 | 0.9140 | +0.0001 |
| Robustness | 0.9303 | 0.9298 | -0.0005 |

## Classification

**Result:** `STABILITY_PARTIAL`

- Some improvements, but aggregate score below 0.80
- Current score: 0.6923 (need >= 0.80)