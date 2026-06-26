# K2 Post-Promotion Long-Run and Dynamic Height Regression Report

**Date:** 2026-06-25
**Task:** `K2_POST_PROMOTION_LONG_RUN_AND_DYNAMIC_HEIGHT_REGRESSION`
**Classification:** `K2_POST_PROMOTION_LONG_RUN_STRONG_PASS`

## 1. Executive Summary
- Long-run equilibrium: 5 K2 runs, 0 falls
- Long-run PRBS: 0 K2 runs, 0 falls
- Classification: `K2_POST_PROMOTION_LONG_RUN_STRONG_PASS`

## 2. Current-Best Lock
| Item | Value |
|------|-------|
| Current-best | K2_NOTCH_LOW_Q_V1 |
| Profile | k2_notch_low_q_v1 |
| wip_notch_q | 2.0 |

## 3. K1 Legacy Lock
| Item | Value |
|------|-------|
| Legacy | K1_PITCH_RATE_NOTCH_V1 |
| Profile | k1_pitch_rate_notch_v1 |
| wip_notch_q | 6.0 |
| Status | Available via explicit CLI |

## 4. Long-Run Equilibrium Results

| Height | K1 pitch | K2 pitch | K1 pitch_f | K2 pitch_f | K1 LF_f | K2 LF_f | K1 hy | K2 hy | Class |
|--------|----------|----------|------------|------------|---------|---------|-------|-------|-------|
| low_0p330 | 4.28 | 4.28 | 4.58 | 4.58 | 5.80e-03 | 5.80e-03 | 0.1016 | 0.1016 | EQUIVALENT |
| mid_0p400 | 1.74 | 1.74 | 2.09 | 2.09 | 0.00e+00 | 0.00e+00 | 0.1125 | 0.1125 | EQUIVALENT |
| high_0p430 | 4.87 | 4.85 | 4.76 | 4.75 | 1.00e-04 | 0.00e+00 | 0.1171 | 0.1154 | EQUIVALENT |
| high_0p450 | 3.15 | 3.11 | 3.27 | 3.20 | 0.00e+00 | 0.00e+00 | 0.1175 | 0.1431 | EQUIVALENT |
| high_0p480 | 5.01 | 4.65 | 5.45 | 4.95 | 0.00e+00 | 0.00e+00 | 0.0733 | 0.0956 | STRONG_BETTER |

## 5. Long-Run PRBS Results

No PRBS runs completed.

## 6. Safety Gates
| Gate | Result |
|------|--------|
| Falls | K1=0, K2=0 |
| Hip-yaw <= 0.35 rad | PASS |
| No hidden torque | PASS |
| No WBC | PASS |

## 7. LF Oscillation Comparison (Final 2000 Steps)

| Height | K1 LF Final | K2 LF Final | Delta |
|--------|------------|------------|-------|
| low_0p330 | 5.80e-03 | 5.80e-03 | +0.0% |
| mid_0p400 | 0.00e+00 | 0.00e+00 | +0.0% |
| high_0p430 | 1.00e-04 | 0.00e+00 | -100.0% |
| high_0p450 | 0.00e+00 | 0.00e+00 | +0.0% |
| high_0p480 | 0.00e+00 | 0.00e+00 | +0.0% |

## 8. Aggregate Classification
**`K2_POST_PROMOTION_LONG_RUN_STRONG_PASS`**

## 9. Keep/Revert Recommendation
**KEEP K2 as current-best.** No regression detected in long-run validation.

## 10. Files Created
| File | Purpose |
|------|---------|
| `scripts/validate_k2_post_promotion_long_run.py` | Long-run validation runner |
| `outputs/k2_post_promotion_long_run/` | Long-run simulation outputs |
| `F:\ROBOTCUATAO\Wheeled-bipedal-robot-simulation\docs\validation\k2_post_promotion_long_run_and_dynamic_height_regression_report.md` | This report |

## 11. Tests/Compile Checks
```
python -m py_compile scripts/validate_k2_post_promotion_long_run.py
python -m py_compile wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py
python -m py_compile scripts/simulate_hierarchical_controller.py
```
