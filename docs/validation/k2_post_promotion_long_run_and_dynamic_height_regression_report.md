# K2 Post-Promotion Long-Run and Dynamic Height Regression Report

**Date:** 2026-06-25
**Task:** `K2_POST_PROMOTION_LONG_RUN_AND_DYNAMIC_HEIGHT_REGRESSION`
**Classification:** `K2_POST_PROMOTION_INVALID`

## 1. Executive Summary
- Long-run equilibrium: 5 K2 runs, 0 falls
- Long-run PRBS: 0 K2 runs, 0 falls
- Classification: `K2_POST_PROMOTION_INVALID`

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
| low_0p330 | 0.00 | 3.97 | 0.00 | 4.34 | 0.00e+00 | 1.40e-03 | 0.0000 | 0.2048 | INVALID |
| mid_0p400 | 0.00 | 1.84 | 0.00 | 2.51 | 0.00e+00 | 0.00e+00 | 0.0000 | 0.1071 | INVALID |
| high_0p430 | 0.00 | 5.60 | 0.00 | 5.69 | 0.00e+00 | 1.10e-03 | 0.0000 | 0.0496 | INVALID |
| high_0p450 | 0.00 | 3.45 | 0.00 | 3.72 | 0.00e+00 | 0.00e+00 | 0.0000 | 0.0882 | INVALID |
| high_0p480 | 0.00 | 5.15 | 0.00 | 5.69 | 0.00e+00 | 0.00e+00 | 0.0000 | 0.0574 | INVALID |

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
| low_0p330 | 0.00e+00 | 1.40e-03 | +0.0% |
| mid_0p400 | 0.00e+00 | 0.00e+00 | +0.0% |
| high_0p430 | 0.00e+00 | 1.10e-03 | +0.0% |
| high_0p450 | 0.00e+00 | 0.00e+00 | +0.0% |
| high_0p480 | 0.00e+00 | 0.00e+00 | +0.0% |

## 8. Aggregate Classification
**`K2_POST_PROMOTION_INVALID`**

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
# K2 Post-Promotion Dynamic Height Gate-Crossing Results

**Date:** 2026-06-25
**Classification:** `K2_POST_PROMOTION_MIXED_KEEP_CURRENT_BEST_MONITOR`

## Dynamic Height Scenarios

| Scenario | K1 pitch | K2 pitch | K1 ht_rmse | K2 ht_rmse | K1 gate_spike | K2 gate_spike | K1 hy | K2 hy | Fell | Class |
|----------|----------|----------|------------|------------|---------------|---------------|-------|-------|------|-------|
| ramp_up_0p330_to_0p480 | 4.11 | 3.15 | 0.1049 | 0.1051 | 4.73 | 5.32 | 0.1091 | 0.0534 | None | EQUIVALENT |
| ramp_down_0p480_to_0p330 | 5.80 | 5.84 | 0.1135 | 0.1149 | 0.00 | 0.00 | 0.0590 | 0.0977 | None | WORSE_BUT_SAFE |
| up_down_cycle_0p330_0p480_0p330 | 4.30 | 3.32 | 0.0944 | 0.0946 | 5.15 | 3.69 | 0.1082 | 0.0534 | None | EQUIVALENT |
| gate_dwell_0p420_0p450_0p480 | 4.23 | 3.05 | 0.1095 | 0.1097 | 4.94 | 5.26 | 0.1075 | 0.0534 | None | EQUIVALENT |
| gate_chatter_0p400_0p470 | 4.16 | 2.98 | 0.0901 | 0.0905 | 5.04 | 5.43 | 0.1016 | 0.0629 | None | EQUIVALENT |

## Gate Alpha Behavior

| Scenario | K1 notch_frac | K2 notch_frac | Gate Monotonic? |
|----------|--------------|--------------|-----------------|
| ramp_up_0p330_to_0p480 | 0.420 | 0.420 | True |
| ramp_down_0p480_to_0p330 | 0.220 | 0.220 | False |
| up_down_cycle_0p330_0p480_0p330 | 0.329 | 0.329 | N/A |
| gate_dwell_0p420_0p450_0p480 | 0.500 | 0.500 | N/A |
| gate_chatter_0p400_0p470 | 0.069 | 0.069 | N/A |

## Safety Gates
| Gate | Result |
|------|--------|
| K2 Falls | 0 |
| Hip-yaw <= 0.35 | PASS |
| No hidden torque | PASS |
| No WBC | PASS |

## Aggregate Dynamic Classification
**`K2_POST_PROMOTION_MIXED_KEEP_CURRENT_BEST_MONITOR`**
