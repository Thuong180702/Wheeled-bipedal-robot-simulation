# K2 Step D Hip-Yaw Baseline Verification Report

**Date:** 2026-06-29
**Phase:** 2 — VERIFY AND FIX STEP D HIP-YAW BASELINE
**Status:** ✅ COMPLETE

---

## 1. Root Cause

The original Step D baseline summary (`k2_original_metrics.json`) reported `hip_yaw_max_rad: 0.0` for all 12 push scenarios. This was a **column name mismatch bug** in the original validation script.

### Bug location

`scripts/validate_k2_step_d_push_matrix.py` lines 289-290:

```python
l_hy = clean(fcol(rows, "hip_yaw_left_rad"))   # ← DOESN'T EXIST
r_hy = clean(fcol(rows, "hip_yaw_right_rad"))  # ← DOESN'T EXIST
```

### Actual telemetry columns

```
l_hip_yaw_pos   ← correct column
r_hip_yaw_pos   ← correct column
```

The `fcol()` helper returns `float("nan")` for missing columns, and `clean()` filters out NaN values. Result: empty lists → `hip_yaw_max = 0.0`.

### Verdict

**Baseline artifact (Type F):** Original metric was not recorded (wrong column name) and defaulted to zero. Not real zero divergence.

---

## 2. Corrected Baseline Values

Recomputed directly from raw telemetry CSVs at:
`outputs/k2_step_d_push_matrix_validation/k2_notch_low_q_v1/*/telemetry_2000.csv`

| Scenario | Old hy_max (rad) | Corrected hy_max (rad) | Delta |
|---|---|---|---|
| high_0p480_sagittal_forward_60N | 0.0 | 0.017767 | +0.018 |
| high_0p480_sagittal_forward_90N | 0.0 | 0.019142 | +0.019 |
| high_0p480_sagittal_backward_60N | 0.0 | 0.019230 | +0.019 |
| high_0p480_sagittal_backward_90N | 0.0 | 0.017241 | +0.017 |
| mid_0p400_sagittal_forward_60N | 0.0 | 0.107930 | +0.108 |
| mid_0p400_sagittal_forward_90N | 0.0 | 0.129353 | +0.129 |
| mid_0p400_sagittal_backward_60N | 0.0 | 0.105359 | +0.105 |
| mid_0p400_sagittal_backward_90N | 0.0 | 0.107434 | +0.107 |
| low_0p330_sagittal_forward_60N | 0.0 | 0.118896 | +0.119 |
| low_0p330_sagittal_forward_90N | 0.0 | 0.192299 | +0.192 |
| low_0p330_sagittal_backward_60N | 0.0 | 0.094122 | +0.094 |
| low_0p330_sagittal_backward_90N | 0.0 | 0.128799 | +0.129 |

**Range:** 0.0172–0.1923 rad (all well under 0.35 rad safety gate)

---

## 3. Impact on Classification

### Before correction
- Candidate hy_max ~0.303 vs baseline 0.0 → artificial huge delta
- Every Step D scenario classified as worse than it actually was

### After correction
- Candidate hy_max ~0.303 vs baseline 0.192 → real but manageable delta
- Classification: SAFE_BUT_WORSE (within 0.35 rad safety gate)
- The regression is real but the comparison is now honest

### Example: low_0p330 sagittal_forward 90N

| Metric | Original (corrected) | Candidate | Class |
|---|---|---|---|
| fell | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.1923 | 0.303 | SAFE_BUT_WORSE |
| post_pitch_rms_500_deg | 0.2517 | 5.805 | WITHIN_OLD_TOLERANCE |
| post_support_rms_500_m | 0.2473 | 0.105 | EXACT_OR_BETTER |

---

## 4. Files Modified

| File | Change |
|---|---|
| `outputs/k2_original_promoted_baseline/k2_original_metrics.json` | Step D hip_yaw_max_rad corrected for all 12 scenarios + `_hip_yaw_correction` metadata |
| `outputs/k2_step_d_push_matrix_validation/corrected_hip_yaw_baseline.json` | NEW — full recomputed baseline with source metadata |

---

## 5. Acceptance Criteria

| Criterion | Status |
|---|---|
| Step D hip_yaw_max=0.0 proven artifact, not real zero | ✅ Column name mismatch confirmed |
| Corrected baseline from raw telemetry | ✅ All 12 cases recomputed |
| Source file and method documented | ✅ `_hip_yaw_correction` metadata in baseline JSON |
| No Step D classification uses suspicious zero baseline | ✅ Baseline updated |
| Corrected values include source file and recomputation method | ✅ |

---

## 6. Key Insight

The original Python K2 controller has very low hip-yaw divergence at high heights (~0.018 rad) and moderate values at low heights (~0.19 rad). The dedicated JAX runner shows elevated hip-yaw at low heights (~0.30 rad). The 0.11 rad gap is real and points to a controller behavior difference, not a measurement artifact.

This confirms the need for Phases 3-5 (scalar trace → identify divergence → patch).
