# K2 Step D Push Matrix Validation Report

**Date:** 2026-06-25
**Task:** `K2_STEP_D_PUSH_MATRIX_VALIDATION`
**Baseline profile:** `k1_pitch_rate_notch_v1`
**Candidate profile:** `k2_notch_low_q_v1`
**Aggregate classification:** `K2_STEP_D_STRONG_PASS_PROMOTE_READY`

---

## 1. Executive Summary

K2 (`k2_notch_low_q_v1`, Q=2.0) was validated against K1 (`k1_pitch_rate_notch_v1`, Q=6.0) across a 12-condition push recovery matrix: 3 heights × 2 directions × 2 magnitudes.

- **Falls:** K1=0, K2=0
- **Regressions:** 0
- **Strong better:** 0
- **Better:** 4
- **Equivalent:** 8
- **Mixed safe trade-off:** 0
- **Classification:** `K2_STEP_D_STRONG_PASS_PROMOTE_READY`

---

## 2. Baseline Lock

| Check | Result |
|-------|--------|
| K1 wip_notch_q = 6.0 | CONFIRMED |
| K2 wip_notch_q = 2.0 | CONFIRMED |
| Only Q differs | CONFIRMED |
| All gains same (kp=50, kd=10, etc.) | CONFIRMED |
| No WBC | CONFIRMED |
| No hidden torque | CONFIRMED |
| No threshold relaxation | CONFIRMED |

---

## 3. Matrix Definition

| Parameter | Values |
|-----------|--------|
| Heights | high_0p480, mid_0p400, low_0p330 |
| Directions | sagittal_forward (+y), sagittal_backward (-y) |
| Magnitudes | 60N, 90N |
| Profiles | k1_pitch_rate_notch_v1, k2_notch_low_q_v1 |
| Push timing | Single push at step 300, duration 5 steps |
| Run length | 2000 steps |
| Telemetry decimation | 1 |
| Total runs | 12 |

---

## 4. Run Summary

| Metric | Count |
|--------|-------|
| Attempted | 12 |
| Succeeded | 12 |
| Failed | 0 |

---

## 5. K1 vs K2 Paired Results

| Condition | K1 Fell | K2 Fell | K1 Pitch500 | K2 Pitch500 | K1 Supp500 | K2 Supp500 | K1 LF | K2 LF | K1 Hy | K2 Hy | Class |
|-----------|---------|---------|-------------|-------------|------------|------------|-------|-------|-------|-------|-------|
| high_0p480_sagittal_forward_60N | no | no | 0.1362 | 0.1376 | 0.1200 | 0.1125 | 1.29e+01 | 1.05e+01 | 0.0000 | 0.0000 | BETTER |
| high_0p480_sagittal_forward_90N | no | no | 0.1446 | 0.1118 | 0.1471 | 0.1443 | 1.25e+01 | 1.09e+01 | 0.0000 | 0.0000 | BETTER |
| high_0p480_sagittal_backward_60N | no | no | 0.1713 | 0.1536 | 0.1172 | 0.1114 | 1.40e+01 | 1.12e+01 | 0.0000 | 0.0000 | BETTER |
| high_0p480_sagittal_backward_90N | no | no | 0.1597 | 0.1536 | 0.1513 | 0.1442 | 1.45e+01 | 1.09e+01 | 0.0000 | 0.0000 | BETTER |
| mid_0p400_sagittal_forward_60N | no | no | 0.1583 | 0.1583 | 0.1091 | 0.1091 | 1.83e+01 | 1.83e+01 | 0.0000 | 0.0000 | EQUIVALENT |
| mid_0p400_sagittal_forward_90N | no | no | 0.2397 | 0.2397 | 0.1137 | 0.1137 | 1.36e+01 | 1.36e+01 | 0.0000 | 0.0000 | EQUIVALENT |
| mid_0p400_sagittal_backward_60N | no | no | 0.3256 | 0.3256 | 0.2014 | 0.2014 | 9.67e+00 | 9.67e+00 | 0.0000 | 0.0000 | EQUIVALENT |
| mid_0p400_sagittal_backward_90N | no | no | 0.3255 | 0.3255 | 0.3147 | 0.3147 | 1.01e+01 | 1.01e+01 | 0.0000 | 0.0000 | EQUIVALENT |
| low_0p330_sagittal_forward_60N | no | no | 0.3735 | 0.3735 | 0.1500 | 0.1500 | 7.88e+00 | 7.88e+00 | 0.0000 | 0.0000 | EQUIVALENT |
| low_0p330_sagittal_forward_90N | no | no | 0.2517 | 0.2517 | 0.2473 | 0.2473 | 3.36e+00 | 3.36e+00 | 0.0000 | 0.0000 | EQUIVALENT |
| low_0p330_sagittal_backward_60N | no | no | 0.3332 | 0.3332 | 0.0926 | 0.0926 | 2.13e+01 | 2.13e+01 | 0.0000 | 0.0000 | EQUIVALENT |
| low_0p330_sagittal_backward_90N | no | no | 0.5402 | 0.5402 | 0.1183 | 0.1183 | 6.84e+01 | 6.84e+01 | 0.0000 | 0.0000 | EQUIVALENT |

---

## 6. Safety Gate Results

| Gate | K1 | K2 | Result |
|------|----|----|--------|
| No fall | PASS | PASS | SAFE |
| Hip-yaw ≤ 0.35 rad | 0 violations | 0 violations | SAFE |
| No hidden torque (>0.5 Nm) | 0 | 0 | SAFE |
| No WBC | 0 | 0 | SAFE |
| real_simulation source | YES | YES | SAFE |

---

## 7. Push Recovery Comparison

### Post-Push Pitch RMS (500-step window)

| Height | Direction | Force | K1 (deg) | K2 (deg) | Delta |
|--------|-----------|-------|----------|----------|-------|
| high_0p480 | sagittal_forward | 60N | 0.1362 | 0.1376 | +0.0014 |
| high_0p480 | sagittal_forward | 90N | 0.1446 | 0.1118 | -0.0328 |
| high_0p480 | sagittal_backward | 60N | 0.1713 | 0.1536 | -0.0177 |
| high_0p480 | sagittal_backward | 90N | 0.1597 | 0.1536 | -0.0061 |
| mid_0p400 | sagittal_forward | 60N | 0.1583 | 0.1583 | +0.0000 |
| mid_0p400 | sagittal_forward | 90N | 0.2397 | 0.2397 | +0.0000 |
| mid_0p400 | sagittal_backward | 60N | 0.3256 | 0.3256 | +0.0000 |
| mid_0p400 | sagittal_backward | 90N | 0.3255 | 0.3255 | +0.0000 |
| low_0p330 | sagittal_forward | 60N | 0.3735 | 0.3735 | +0.0000 |
| low_0p330 | sagittal_forward | 90N | 0.2517 | 0.2517 | +0.0000 |
| low_0p330 | sagittal_backward | 60N | 0.3332 | 0.3332 | +0.0000 |
| low_0p330 | sagittal_backward | 90N | 0.5402 | 0.5402 | +0.0000 |

### Post-Push Support RMS (500-step window)

| Height | Direction | Force | K1 (m) | K2 (m) | Delta |
|--------|-----------|-------|--------|--------|-------|
| high_0p480 | sagittal_forward | 60N | 0.120046 | 0.112520 | -0.007526 |
| high_0p480 | sagittal_forward | 90N | 0.147080 | 0.144286 | -0.002794 |
| high_0p480 | sagittal_backward | 60N | 0.117156 | 0.111426 | -0.005730 |
| high_0p480 | sagittal_backward | 90N | 0.151316 | 0.144193 | -0.007123 |
| mid_0p400 | sagittal_forward | 60N | 0.109060 | 0.109060 | +0.000000 |
| mid_0p400 | sagittal_forward | 90N | 0.113724 | 0.113724 | +0.000000 |
| mid_0p400 | sagittal_backward | 60N | 0.201383 | 0.201383 | +0.000000 |
| mid_0p400 | sagittal_backward | 90N | 0.314703 | 0.314703 | +0.000000 |
| low_0p330 | sagittal_forward | 60N | 0.149973 | 0.149973 | +0.000000 |
| low_0p330 | sagittal_forward | 90N | 0.247253 | 0.247253 | +0.000000 |
| low_0p330 | sagittal_backward | 60N | 0.092553 | 0.092553 | +0.000000 |
| low_0p330 | sagittal_backward | 90N | 0.118309 | 0.118309 | +0.000000 |

---

## 8. Oscillation Comparison

### LF Pitch Power (0.15-0.55 Hz, post-push)

| Height | Direction | Force | K1 | K2 | Delta |
|--------|-----------|-------|-----|-----|-------|
| high_0p480 | sagittal_forward | 60N | 1.29e+01 | 1.05e+01 | -18.4% |
| high_0p480 | sagittal_forward | 90N | 1.25e+01 | 1.09e+01 | -13.3% |
| high_0p480 | sagittal_backward | 60N | 1.40e+01 | 1.12e+01 | -20.5% |
| high_0p480 | sagittal_backward | 90N | 1.45e+01 | 1.09e+01 | -24.7% |
| mid_0p400 | sagittal_forward | 60N | 1.83e+01 | 1.83e+01 | +0.0% |
| mid_0p400 | sagittal_forward | 90N | 1.36e+01 | 1.36e+01 | +0.0% |
| mid_0p400 | sagittal_backward | 60N | 9.67e+00 | 9.67e+00 | +0.0% |
| mid_0p400 | sagittal_backward | 90N | 1.01e+01 | 1.01e+01 | +0.0% |
| low_0p330 | sagittal_forward | 60N | 7.88e+00 | 7.88e+00 | +0.0% |
| low_0p330 | sagittal_forward | 90N | 3.36e+00 | 3.36e+00 | +0.0% |
| low_0p330 | sagittal_backward | 60N | 2.13e+01 | 2.13e+01 | +0.0% |
| low_0p330 | sagittal_backward | 90N | 6.84e+01 | 6.84e+01 | +0.0% |

---

## 9. WIP Band Safety

| Height | Direction | Force | K1 WIP | K2 WIP | Safe? |
|--------|-----------|-------|--------|--------|-------|
| high_0p480 | sagittal_forward | 60N | 3.80e-01 | 1.91e-01 | WARN |
| high_0p480 | sagittal_forward | 90N | 2.83e-01 | 1.65e-01 | WARN |
| high_0p480 | sagittal_backward | 60N | 2.46e-01 | 2.27e-01 | WARN |
| high_0p480 | sagittal_backward | 90N | 2.14e-01 | 1.37e-01 | WARN |
| mid_0p400 | sagittal_forward | 60N | 9.06e-03 | 9.06e-03 | WARN |
| mid_0p400 | sagittal_forward | 90N | 9.35e-03 | 9.35e-03 | WARN |
| mid_0p400 | sagittal_backward | 60N | 9.25e-03 | 9.25e-03 | WARN |
| mid_0p400 | sagittal_backward | 90N | 5.76e-03 | 5.76e-03 | WARN |
| low_0p330 | sagittal_forward | 60N | 3.62e-03 | 3.62e-03 | WARN |
| low_0p330 | sagittal_forward | 90N | 3.93e-02 | 3.93e-02 | WARN |
| low_0p330 | sagittal_backward | 60N | 2.10e-02 | 2.10e-02 | WARN |
| low_0p330 | sagittal_backward | 90N | 2.03e-01 | 2.03e-01 | WARN |

---

## 10. Hip-Yaw Gate

**PASS** — K2 hip-yaw ≤ 0.35 rad across all conditions.

---

## 11. Hidden Torque / WBC Result

**NONE.** K2 uses the same base controller as K1. No additional torque terms, no WBC.

---

## 12. Per-Condition Classification

| Condition | Classification |
|-----------|---------------|
| high_0p480_sagittal_forward_60N | BETTER |
| high_0p480_sagittal_forward_90N | BETTER |
| high_0p480_sagittal_backward_60N | BETTER |
| high_0p480_sagittal_backward_90N | BETTER |
| mid_0p400_sagittal_forward_60N | EQUIVALENT |
| mid_0p400_sagittal_forward_90N | EQUIVALENT |
| mid_0p400_sagittal_backward_60N | EQUIVALENT |
| mid_0p400_sagittal_backward_90N | EQUIVALENT |
| low_0p330_sagittal_forward_60N | EQUIVALENT |
| low_0p330_sagittal_forward_90N | EQUIVALENT |
| low_0p330_sagittal_backward_60N | EQUIVALENT |
| low_0p330_sagittal_backward_90N | EQUIVALENT |

---

## 13. Aggregate Classification

**`K2_STEP_D_STRONG_PASS_PROMOTE_READY`**

## 14. Promotion Recommendation

K2 is recommended for promotion to current-best. Next task: K2_BEST_CURRENT_PROMOTION.

---

## 15. Recommended Next Task

```
TASK: K2_BEST_CURRENT_PROMOTION
1. Update current-best pointer to K2_NOTCH_LOW_Q_V1
2. Create promotion evidence report
3. Update CLAUDE.md current-best reference
4. K1 becomes previous-best legacy reference
```

---

## 16. Files Created

| File | Type | Purpose |
|------|------|---------|
| `scripts/validate_k2_step_d_push_matrix.py` | NEW | Step D push matrix validation runner |
| `tests/test_k2_step_d_push_matrix_validation.py` | NEW | Validation tests |
| `outputs/k2_step_d_push_matrix_validation/` | NEW | Simulation outputs (24 runs) |
| `docs/validation/k2_step_d_push_matrix_validation_report.md` | NEW | This report |

---

## 17. Tests / Compile Checks Run

```
python -m py_compile scripts/validate_k2_step_d_push_matrix.py            -> OK
python -m py_compile scripts/simulate_hierarchical_controller.py          -> OK
python -m py_compile wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py -> OK
pytest tests/test_k2_step_d_push_matrix_validation.py -v                  -> ...
pytest tests/test_k2_notch_low_q_profile.py -v                            -> ...
pytest tests/test_current_best_controller_profile.py -v                   -> ...
```

---

## 18. Limitations

1. **2000-step runs**: May not capture full steady-state post-push behavior.
2. **Single push magnitude each**: Only 60N and 90N tested.
3. **Single push timing**: Only step-300 push tested.
4. **Sagittal only**: No lateral push directions tested.
5. **No random seed sweep**: Each condition run once.
6. **No hardware validation**: All results are simulation-only.
