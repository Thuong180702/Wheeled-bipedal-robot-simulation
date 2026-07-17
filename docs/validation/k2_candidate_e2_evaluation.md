# K2 Stability Improvement — Candidate Evaluation

**Classification:** `STABILITY_PARTIAL`
**baseline_source:** docs/validation/k2_improvement_baseline_quality.json
**candidate_source:** docs/validation/k2_candidate_e2_quality.json
**evaluator_version:** 1.0.0

## Executive Summary

| Metric | Value |
|--------|-------|
| Classification | **STABILITY_PARTIAL** |
| Baseline aggregate score | 0.6834 |
| Candidate aggregate score | 0.6935 |
| Score delta | +0.0102 |
| Evaluated scenarios | 35/35 |
| Safety fails | 0 |
| Major regressions | 0 |
| Baseline performance | 147.4 Hz |
| Candidate performance | 189.2 Hz |

## Per-Scenario Results

| Scenario | Type | Base Score | Cand Score | Delta | Safety | Regressions |
|----------|------|------------|------------|-------|--------|-------------|
| C1_slow_ladder_up_down | other | 0.7437 | 0.7437 | +0.0000 | PASS | OK |
| C2_random_500dwell | other | 0.7437 | 0.7437 | +0.0000 | PASS | OK |
| C3_random_200dwell | other | 0.7437 | 0.7437 | +0.0000 | PASS | OK |
| C4_abrupt_stress | other | 0.7437 | 0.7437 | +0.0000 | PASS | OK |
| C5_long_random | other | 0.6842 | 0.6842 | +0.0000 | PASS | OK |
| focused_high_0p480 | fixed_height | 0.7166 | 0.7166 | +0.0000 | PASS | OK |
| focused_low_0p320 | fixed_height | 0.7132 | 0.7313 | +0.0180 | PASS | OK |
| gate_chatter_0p400_0p470 | dynamic_height | 0.6919 | 0.6919 | +0.0000 | PASS | OK |
| gate_dwell_0p420_0p450_0p480 | dynamic_height | 0.4455 | 0.4456 | +0.0001 | PASS | OK |
| high_0p430 | fixed_height | 0.6727 | 0.7286 | +0.0559 | PASS | OK |
| high_0p450 | fixed_height | 0.6052 | 0.6052 | +0.0000 | PASS | OK |
| high_0p465 | fixed_height | 0.7136 | 0.7164 | +0.0028 | PASS | OK |
| high_0p480 | fixed_height | 0.6756 | 0.6756 | +0.0000 | PASS | OK |
| high_0p480_sagittal_backward_60N | push | 0.6557 | 0.6557 | +0.0000 | PASS | OK |
| high_0p480_sagittal_backward_90N | push | 0.6673 | 0.6673 | +0.0000 | PASS | OK |
| high_0p480_sagittal_forward_60N | push | 0.6722 | 0.6722 | +0.0000 | PASS | OK |
| high_0p480_sagittal_forward_90N | push | 0.6195 | 0.6195 | +0.0000 | PASS | OK |
| low_0p300 | fixed_height | 0.7665 | 0.7665 | +0.0000 | PASS | OK |
| low_0p320 | fixed_height | 0.6962 | 0.7313 | +0.0351 | PASS | OK |
| low_0p330 | fixed_height | 0.6335 | 0.6335 | +0.0000 | PASS | OK |
| low_0p330_sagittal_backward_60N | push | 0.6938 | 0.6913 | -0.0025 | PASS | OK |
| low_0p330_sagittal_backward_90N | push | 0.6373 | 0.6640 | +0.0267 | PASS | OK |
| low_0p330_sagittal_forward_60N | push | 0.6685 | 0.6685 | +0.0000 | PASS | OK |
| low_0p330_sagittal_forward_90N | push | 0.5973 | 0.5953 | -0.0020 | PASS | OK |
| low_0p340 | fixed_height | 0.8335 | 0.8335 | +0.0000 | PASS | OK |
| low_0p360 | fixed_height | 0.7444 | 0.7589 | +0.0145 | PASS | OK |
| low_0p380 | fixed_height | 0.6883 | 0.6897 | +0.0014 | PASS | OK |
| mid_0p400 | fixed_height | 0.7247 | 0.7390 | +0.0143 | PASS | OK |
| mid_0p400_sagittal_backward_60N | push | 0.7578 | 0.7583 | +0.0005 | PASS | OK |
| mid_0p400_sagittal_backward_90N | push | 0.6889 | 0.6999 | +0.0109 | PASS | OK |
| mid_0p400_sagittal_forward_60N | push | 0.6997 | 0.7215 | +0.0218 | PASS | OK |
| mid_0p400_sagittal_forward_90N | push | 0.6943 | 0.6757 | -0.0186 | PASS | OK |
| ramp_down_0p480_to_0p330 | dynamic_height | 0.4535 | 0.5517 | +0.0982 | PASS | OK |
| ramp_up_0p330_to_0p480 | dynamic_height | 0.6996 | 0.7787 | +0.0791 | PASS | OK |
| up_down_cycle_0p330_0p480_0p330 | dynamic_height | 0.7322 | 0.7320 | -0.0002 | PASS | OK |

## Dimension Score Comparison

| Dimension | Baseline Mean | Candidate Mean | Delta |
|-----------|---------------|----------------|-------|
| Posture | 0.6749 | 0.6776 | +0.0027 |
| Support/Drift | 0.4895 | 0.5019 | +0.0123 |
| Leg Health | 0.8236 | 0.8326 | +0.0089 |
| Dynamic Height | 0.5128 | 0.5371 | +0.0243 |
| Torque Quality | 0.8954 | 0.9139 | +0.0186 |
| Robustness | 0.9298 | 0.9303 | +0.0005 |

## Classification

**Result:** `STABILITY_PARTIAL`

- Some improvements, but aggregate score below 0.80
- Current score: 0.6935 (need >= 0.80)