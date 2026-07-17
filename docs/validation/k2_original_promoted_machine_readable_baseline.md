# K2 Original Promoted — Machine-Readable Baseline

**Date:** 2026-06-29
**Task:** Phase 1 — Lock original promoted K2 numbers and tolerances
**Baseline file:** `outputs/k2_original_promoted_baseline/k2_original_metrics.json`

---

## 1. Source of Truth

The baseline is extracted from original K2 Python promoted reports:

| Report | Classification | Key Data |
|--------|---------------|----------|
| `k2_notch_low_q_v1_create_and_validate_report.md` | K2_STRONG_PASS_READY_FOR_PROMOTION | K2 profile definition, paired K1/K2 runs |
| `k2_step_c_e_validation_and_best_current_promotion_report.md` | K2_STEP_C_E_STRONG_PASS_PROMOTE_NOW | Step C (7 cases) + Step E (10 heights) |
| `k2_step_d_push_matrix_validation_report.md` | K2_STEP_D_STRONG_PASS_PROMOTE_READY | Step D push matrix (12 conditions) |
| `k2_post_promotion_long_run_and_dynamic_height_regression_report.md` | K2_POST_PROMOTION_MIXED | Long-run (5 heights) + Dynamic (5 scenarios) |

---

## 2. K2 Profile Identity

K2 (`k2_notch_low_q_v1`) differs from K1 in EXACTLY ONE parameter:

| Parameter | K1 | K2 |
|-----------|----|----|
| `wip_notch_q` | 6.0 | **2.0** |

All other parameters identical. See baseline JSON for full profile.

---

## 3. Absolute Safety Gates

These are non-negotiable. ANY violation = BLOCKED.

| Gate | Threshold | Type |
|------|-----------|------|
| Falls | = 0 | Safety |
| hip_yaw_max | <= 0.35 rad | Safety |
| NaN/Inf | None allowed | Safety |
| Hidden torque | <= 0.5 Nm (magnitude) | Safety |
| WBC authority | Must be 0 | Safety |
| Real simulation | Must be YES | Safety |

---

## 4. Equivalence Tolerances

For metrics that are NOT safety gates, tolerance defines the WITHIN_OLD_TOLERANCE boundary:

| Metric | Absolute Tol | Relative Tol | Rule |
|--------|-------------|--------------|------|
| hip_yaw_max | 0.05 rad | 2.0x | min(abs, rel × orig) |
| pitch_rms | 1.0 deg | 30% | min(abs, rel × orig) |
| support_rms | 0.02 m | 50% | min(abs, rel × orig) |
| height_rmse | 0.02 m | 20% | min(abs, rel × orig) |
| LF_power | 0.005 | 5.0x | min(abs, rel × orig) |
| WIP_power | 0.005 | 5.0x | min(abs, rel × orig) |
| post_pitch500 | 0.05 rad | 30% | min(abs, rel × orig) |
| post_support500 | 0.05 m | 30% | min(abs, rel × orig) |

---

## 5. Required Scenario Inventory

### Step E (10 heights, 2000 steps each)
- [x] low_0p300 — hy=0.1314
- [x] low_0p320 — hy=0.0502
- [x] low_0p330 — hy=0.0851
- [x] low_0p340 — hy=0.0445
- [x] low_0p360 — hy=0.0959
- [x] low_0p380 — hy=0.0392
- [x] high_0p430 — hy=0.0236
- [x] high_0p450 — hy=0.0904
- [x] high_0p465 — hy=0.0296
- [x] high_0p480 — hy=0.0563

### Step C (7 cases, 2000 steps each)
- [x] C1_slow_ladder_up_down — hy=0.0851
- [x] C2_random_500dwell — hy=0.0851
- [x] C3_random_200dwell — hy=0.0851
- [x] C4_abrupt_stress — hy=0.0851
- [x] C5_long_random — hy=0.0851
- [x] focused_low_0p320 — hy=0.0502
- [x] focused_high_0p480 — hy=0.0563

### Step D (12 conditions, 2000 steps each)
- [x] high_0p480 fwd 60N/90N, bwd 60N/90N
- [x] mid_0p400 fwd 60N/90N, bwd 60N/90N
- [x] low_0p330 fwd 60N/90N, bwd 60N/90N

### Dynamic Height (5 scenarios)
- [x] ramp_up — hy=0.0534
- [x] ramp_down — hy=0.0977
- [x] up_down_cycle — hy=0.0534
- [x] gate_dwell — hy=0.0534
- [x] gate_chatter — hy=0.0629

### Long-Run Equilibrium (5 heights, 6000 steps each)
- [x] low_0p330 — hy=0.2048
- [x] mid_0p400 — hy=0.1071
- [x] high_0p430 — hy=0.0496
- [x] high_0p450 — hy=0.0882
- [x] high_0p480 — hy=0.0574

---

## 6. Usage

### Load baseline
```python
import json
with open("outputs/k2_original_promoted_baseline/k2_original_metrics.json") as f:
    baseline = json.load(f)

original_hy = baseline["step_e"]["scenarios"]["low_0p300"]["hip_yaw_max_rad"]
# 0.1314
```

### Classify candidate
```python
from wheeled_biped.validation.strict_promotion_classifier import Classifier

classifier = Classifier(baseline_path="outputs/k2_original_promoted_baseline/k2_original_metrics.json")
result = classifier.classify_step_e("low_0p300", candidate_metrics)
# Returns: StrictClass enum + delta + tolerance + class
```

### Check promotion readiness
```python
is_ready = classifier.is_promotion_pass(required_scopes=["step_e", "step_c", "step_d", "dynamic_height", "long_run"])
# Returns: True if ALL required scopes are EXACT_OR_BETTER or WITHIN_OLD_TOLERANCE
```

---

## 7. No Candidate May Be Marked FULL PASS Without Comparison To This File

Any promotion report must:
1. Load this baseline JSON
2. Compare EVERY metric for EVERY required scenario
3. Show Original Value, Candidate Value, Delta, Tolerance, and Strict Class
4. Only claim FULL PASS if all classes are EXACT_OR_BETTER or WITHIN_OLD_TOLERANCE
