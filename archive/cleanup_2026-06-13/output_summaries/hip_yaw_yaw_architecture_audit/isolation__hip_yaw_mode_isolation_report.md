# Hip-Yaw Mode Isolation Experiment Report

**Date:** 2026-06-05
**Status:** PHASE 4 - Isolation experiments

## Objective

Systematically test kinematic coupling and mode authority:
1. Can hip-yaw common-mode control body yaw?
2. Can hip-yaw divergence-mode stabilize leg geometry?
3. Are modes independent or coupled through contact/roll?

## Experiment Results

### A_baseline_shape_posture_only

**Survived steps:** 212
**Termination:** unknown

**Body yaw:**
  - Max: 114.19°
  - Final: 113.35°
  - RMS: 57.65°

**Hip-yaw common-mode error:**
  - Max: 25.52°
  - Final: 23.03°
  - RMS: 19.48°

**Hip-yaw divergence-mode error:**
  - Max: 6.80°
  - Final: -0.08°
  - RMS: 2.69°

## Classification

**Status:** PENDING_PHASE_4_EXPERIMENTS

Additional experiments (B-F) required to complete classification.

## Next Steps

1. Implement controller modifications for experiments B-F
2. Run remaining isolation experiments
3. Analyze kinematic coupling from pulse tests
4. Classify body yaw authority and hip-yaw divergence authority
5. Design final architecture based on experimental evidence
