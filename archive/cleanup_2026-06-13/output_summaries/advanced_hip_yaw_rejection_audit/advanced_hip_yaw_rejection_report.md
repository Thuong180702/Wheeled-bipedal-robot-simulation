# Advanced Hip-Yaw Disturbance Rejection Audit

**Date:** 2026-06-04
**Phase:** 2
**Status:** COMPLETE

---

## Executive Summary

Analyzed baseline and best HY-FF candidate (C: sign=-1.0, k=2.0) to classify which advanced hip-yaw mechanisms are most promising for Phase 3 experiments.

**HY-FF Improvement:** 9.2% (0.2137 → 0.1941 rad)
**Still Over Threshold:** 177.3% (threshold: 0.070 rad)

**Mode Classification:** divergence_dominant

**Recommended Candidates for Phase 3:** HY2-DIV

---

## Mechanism Classification

- `divergence_dominant`: [YES]
- `common_mode_dominant`: [NO]
- `support_velocity_lead_needed`: [NO]
- `support_error_feedforward_too_late`: [NO]
- `hip_yaw_integral_needed`: [NO]
- `hip_yaw_pd_gains_too_low`: [YES]
- `hip_yaw_not_locally_rejectable_without_support_fix`: [YES]
- `coupled_sagittal_yaw_required`: [YES]

---

## Rationale

Hip-yaw error is divergence-dominant (divergence_mean=0.2119 > common_mode_mean=0.0077). Recommend HY2-DIV: divergence damping/authority. Hip-yaw velocity response is low relative to error (vel/error ratio=0.37 < 1.0). PD gains may be too low, but global increase violates restrictions. HY-FF provided < 15% improvement and error still > 150% over threshold. Hip-yaw cannot be fixed locally without addressing support drift first. COUPLED SAGITTAL-YAW FIX REQUIRED: Joint fix must address both support and hip-yaw together.

---

## Baseline Analysis

- **hip_yaw_abs_max:** 0.2137 rad
- **divergence_max:** 0.4200 rad
- **common_mode_max:** 0.0225 rad
- **divergence_mean:** 0.2119 rad
- **common_mode_mean:** 0.0077 rad
- **mode:** divergence_dominant

### Lag Correlations (baseline)

- **support_error_to_divergence:** lag=-50 steps (-500.0 ms), corr=-0.818
- **support_velocity_to_divergence:** lag=-50 steps (-500.0 ms), corr=0.373
- **body_yaw_to_common_mode:** lag=50 steps (500.0 ms), corr=-0.682
- **pitch_to_divergence:** lag=-50 steps (-500.0 ms), corr=-0.276

### PD Assessment (baseline)

- **vel_to_error_ratio:** 0.37
- **pd_gains_likely_too_low:** True

### Integral Assessment (baseline)

- **hip_yaw_error_mean:** 0.1173 rad
- **persistent_offset:** False

---

## Best HY-FF Analysis

- **hip_yaw_abs_max:** 0.1941 rad
- **divergence_max:** 0.3846 rad
- **common_mode_max:** 0.0216 rad
- **mode:** divergence_dominant

### HY-FF Assessment

- **hy_ff_too_late:** False
- **support_velocity_lead_useful:** False

---

## Artifacts Generated

- `advanced_hip_yaw_rejection_summary.json`
- `advanced_hip_yaw_mechanism_classification.json`
- `baseline_hip_yaw_error_phase_portrait.csv`
- `best_hy_ff_hip_yaw_error_phase_portrait.csv`
- `baseline_hip_yaw_divergence_vs_support.csv`
- `best_hy_ff_hip_yaw_divergence_vs_support.csv`
- `baseline_hip_yaw_body_yaw_coupling.csv`
- `best_hy_ff_hip_yaw_body_yaw_coupling.csv`
- `hip_yaw_disturbance_lag_correlation.csv`

---

## Next Steps

Proceed to **Phase 3: Advanced Hip-Yaw Candidate Experiments**.

Evaluate the following candidates: HY2-DIV
