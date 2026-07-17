# K2 Step C/E Validation and Best-Current Promotion Report

**Date:** 2026-06-25
**Task:** `K2_STEP_C_E_VALIDATION_AND_BEST_CURRENT_PROMOTION`
**Baseline profile:** `k1_pitch_rate_notch_v1`
**Candidate profile:** `k2_notch_low_q_v1`
**Final classification:** `K2_STEP_C_E_STRONG_PASS_PROMOTE_NOW`

---

## 1. Executive Summary

K2 (`k2_notch_low_q_v1`, Q=2.0) was validated against K1 (`k1_pitch_rate_notch_v1`, Q=6.0) across Step C (7 cases) and Step E (10 heights) fixed-height validation suites.

- **Step C:** 7/7 cases completed, 0 falls
- **Step E:** 10/10 heights completed, 0 falls
- **Classification:** `K2_STEP_C_E_STRONG_PASS_PROMOTE_NOW`

---

## 2. Pre-Promotion Current-Best

| Item | Value |
|------|-------|
| Current-best | `K1_PITCH_RATE_NOTCH_V1` |
| Profile | `k1_pitch_rate_notch_v1` |
| wip_notch_q | 6.0 |

## 3. K1/K2 Profile Diff

| Parameter | K1 | K2 |
|-----------|----|----|
| wip_notch_q | 6.0 | **2.0** |
| wip_notch_center_hz | 2.5 | 2.5 |
| wip_notch_target_signal | pitch_rate | pitch_rate |
| wip_notch_filter_blend | 1.0 | 1.0 |
| wip_notch_height_gate_start_m | 0.42 | 0.42 |
| wip_notch_height_gate_full_m | 0.48 | 0.48 |
| All other gains | Same | Same |
| WBC | None | None |
| Hidden torque | None | None |

## 4. Step D Evidence Verification

| Check | Result |
|-------|--------|
| Step D report exists | YES |
| Step D outputs exist | YES |
| K2 Step D classification | K2_STEP_D_STRONG_PASS_PROMOTE_READY |
| 24/24 runs succeeded | YES |
| K1 falls = 0 | YES |
| K2 falls = 0 | YES |
| Regressions = 0 | YES |

## 5. Step C Validation Matrix

### Cases

| Case ID | Height | Steps | Notch Active? |
|---------|--------|-------|--------------|
| C1_slow_ladder_up_down | low_0p330 | 2000 | No |
| C2_random_500dwell | low_0p330 | 2000 | No |
| C3_random_200dwell | low_0p330 | 2000 | No |
| C4_abrupt_stress | low_0p330 | 2000 | No |
| C5_long_random | low_0p330 | 2000 | No |
| focused_low_0p320 | low_0p320 | 2000 | No |
| focused_high_0p480 | high_0p480 | 2000 | Yes |

### K2 Step C Results

| Case | pitch_rms_deg | support_rms_m | hip_yaw_max | LF_power | WIP_power | fell |
|------|--------------|---------------|-------------|----------|-----------|------|
| C1_slow_ladder_up_down | 3.63 | 0.0386 | 0.0851 | 1.80e-03 | 0.00e+00 | False |
| C2_random_500dwell | 3.63 | 0.0386 | 0.0851 | 1.80e-03 | 0.00e+00 | False |
| C3_random_200dwell | 3.63 | 0.0386 | 0.0851 | 1.80e-03 | 0.00e+00 | False |
| C4_abrupt_stress | 3.63 | 0.0386 | 0.0851 | 1.80e-03 | 0.00e+00 | False |
| C5_long_random | 3.63 | 0.0386 | 0.0851 | 1.80e-03 | 0.00e+00 | False |
| focused_low_0p320 | 2.83 | 0.0525 | 0.0502 | 2.00e-03 | 1.00e-04 | False |
| focused_high_0p480 | 3.96 | 0.0471 | 0.0563 | 0.00e+00 | 0.00e+00 | False |

## 6. Step E Validation Matrix

### Heights

| Height | Notch Gate |
|--------|-----------|
| low_0p300 | Inactive |
| low_0p320 | Inactive |
| low_0p330 | Inactive |
| low_0p340 | Inactive |
| low_0p360 | Inactive |
| low_0p380 | Inactive |
| high_0p430 | Partial |
| high_0p450 | Partial |
| high_0p465 | Partial |
| high_0p480 | Active (100%) |

### K2 Step E Results

| Height | pitch_rms_deg | support_rms_m | hip_yaw_max | LF_power | WIP_power | fell |
|--------|--------------|---------------|-------------|----------|-----------|------|
| low_0p300 | 2.68 | 0.0421 | 0.1314 | 9.00e-04 | 0.00e+00 | False |
| low_0p320 | 2.83 | 0.0525 | 0.0502 | 2.00e-03 | 1.00e-04 | False |
| low_0p330 | 3.63 | 0.0386 | 0.0851 | 1.80e-03 | 0.00e+00 | False |
| low_0p340 | 2.97 | 0.0541 | 0.0445 | 1.00e-04 | 1.00e-04 | False |
| low_0p360 | 1.90 | 0.0371 | 0.0959 | 1.30e-03 | 0.00e+00 | False |
| low_0p380 | 3.33 | 0.0480 | 0.0392 | 1.00e-04 | 0.00e+00 | False |
| high_0p430 | 4.98 | 0.0637 | 0.0236 | 1.00e-04 | 3.00e-04 | False |
| high_0p450 | 2.75 | 0.0694 | 0.0904 | 2.00e-04 | 0.00e+00 | False |
| high_0p465 | 3.55 | 0.0617 | 0.0296 | 2.00e-04 | 3.00e-04 | False |
| high_0p480 | 3.96 | 0.0471 | 0.0563 | 0.00e+00 | 0.00e+00 | False |

## 7. K1 vs K2 Paired Tables

### Step C Comparison

| Case | K1 pitch | K2 pitch | K1 support | K2 support | K1 LF | K2 LF | K1 hy | K2 hy | Class |
|------|----------|----------|------------|------------|-------|-------|-------|-------|-------|
| C1_slow_ladder_up_down | 3.63 | 3.63 | 0.0386 | 0.0386 | 1.80e-03 | 1.80e-03 | 0.0851 | 0.0851 | EQUIVALENT |
| C2_random_500dwell | 3.63 | 3.63 | 0.0386 | 0.0386 | 1.80e-03 | 1.80e-03 | 0.0851 | 0.0851 | EQUIVALENT |
| C3_random_200dwell | 3.63 | 3.63 | 0.0386 | 0.0386 | 1.80e-03 | 1.80e-03 | 0.0851 | 0.0851 | EQUIVALENT |
| C4_abrupt_stress | 3.63 | 3.63 | 0.0386 | 0.0386 | 1.80e-03 | 1.80e-03 | 0.0851 | 0.0851 | EQUIVALENT |
| C5_long_random | 3.63 | 3.63 | 0.0386 | 0.0386 | 1.80e-03 | 1.80e-03 | 0.0851 | 0.0851 | EQUIVALENT |
| focused_low_0p320 | 2.83 | 2.83 | 0.0525 | 0.0525 | 2.00e-03 | 2.00e-03 | 0.0502 | 0.0502 | EQUIVALENT |
| focused_high_0p480 | 4.32 | 3.96 | 0.0622 | 0.0471 | 0.00e+00 | 0.00e+00 | 0.0613 | 0.0563 | STRONG_BETTER |

### Step E Comparison

| Height | K1 pitch | K2 pitch | K1 support | K2 support | K1 LF | K2 LF | K1 hy | K2 hy | Class |
|--------|----------|----------|------------|------------|-------|-------|-------|-------|-------|
| low_0p300 | 2.68 | 2.68 | 0.0421 | 0.0421 | 9.00e-04 | 9.00e-04 | 0.1314 | 0.1314 | EQUIVALENT |
| low_0p320 | 2.83 | 2.83 | 0.0525 | 0.0525 | 2.00e-03 | 2.00e-03 | 0.0502 | 0.0502 | EQUIVALENT |
| low_0p330 | 3.63 | 3.63 | 0.0386 | 0.0386 | 1.80e-03 | 1.80e-03 | 0.0851 | 0.0851 | EQUIVALENT |
| low_0p340 | 2.97 | 2.97 | 0.0541 | 0.0541 | 1.00e-04 | 1.00e-04 | 0.0445 | 0.0445 | EQUIVALENT |
| low_0p360 | 1.90 | 1.90 | 0.0371 | 0.0371 | 1.30e-03 | 1.30e-03 | 0.0959 | 0.0959 | EQUIVALENT |
| low_0p380 | 3.33 | 3.33 | 0.0480 | 0.0480 | 1.00e-04 | 1.00e-04 | 0.0392 | 0.0392 | EQUIVALENT |
| high_0p430 | 4.99 | 4.98 | 0.0643 | 0.0637 | 1.00e-04 | 1.00e-04 | 0.0231 | 0.0236 | EQUIVALENT |
| high_0p450 | 2.89 | 2.75 | 0.0697 | 0.0694 | 1.00e-04 | 2.00e-04 | 0.0881 | 0.0904 | EQUIVALENT |
| high_0p465 | 4.16 | 3.55 | 0.0747 | 0.0617 | 1.00e-04 | 2.00e-04 | 0.0295 | 0.0296 | STRONG_BETTER |
| high_0p480 | 4.32 | 3.96 | 0.0622 | 0.0471 | 0.00e+00 | 0.00e+00 | 0.0613 | 0.0563 | STRONG_BETTER |

## 8. Safety Gates

| Gate | K1 Step C | K2 Step C | K1 Step E | K2 Step E | Result |
|------|-----------|-----------|-----------|-----------|--------|
| Falls | 0 | 0 | 0 | 0 | SAFE |
| Hip-yaw <= 0.35 rad | PASS | PASS | PASS | PASS | SAFE |
| No hidden torque | PASS | PASS | PASS | PASS | SAFE |
| No WBC | PASS | PASS | PASS | PASS | SAFE |
| real_simulation source | YES | YES | YES | YES | SAFE |

## 9. Hip-Yaw Gates

| Suite | K2 max hip_yaw | Gate (0.35 rad) |
|-------|---------------|-----------------|
| Step C | 0.0851 | PASS |
| Step E | 0.1314 | PASS |

## 10. Hidden Torque/WBC Result

**NONE.** K2 uses the same base controller as K1. No additional torque terms, no WBC.

## 11. Low-Frequency Mode Comparison

### Step C

| Case/Height | K1 LF Power | K2 LF Power | Delta |
|-------------|-------------|-------------|-------|
| C1_slow_ladder_up_down | 1.80e-03 | 1.80e-03 | +0.0% |
| C2_random_500dwell | 1.80e-03 | 1.80e-03 | +0.0% |
| C3_random_200dwell | 1.80e-03 | 1.80e-03 | +0.0% |
| C4_abrupt_stress | 1.80e-03 | 1.80e-03 | +0.0% |
| C5_long_random | 1.80e-03 | 1.80e-03 | +0.0% |
| focused_low_0p320 | 2.00e-03 | 2.00e-03 | +0.0% |
| focused_high_0p480 | 0.00e+00 | 0.00e+00 | +0.0% |

### Step E

| Case/Height | K1 LF Power | K2 LF Power | Delta |
|-------------|-------------|-------------|-------|
| low_0p300 | 9.00e-04 | 9.00e-04 | +0.0% |
| low_0p320 | 2.00e-03 | 2.00e-03 | +0.0% |
| low_0p330 | 1.80e-03 | 1.80e-03 | +0.0% |
| low_0p340 | 1.00e-04 | 1.00e-04 | +0.0% |
| low_0p360 | 1.30e-03 | 1.30e-03 | +0.0% |
| low_0p380 | 1.00e-04 | 1.00e-04 | +0.0% |
| high_0p430 | 1.00e-04 | 1.00e-04 | +0.0% |
| high_0p450 | 1.00e-04 | 2.00e-04 | +100.0% |
| high_0p465 | 1.00e-04 | 2.00e-04 | +100.0% |
| high_0p480 | 0.00e+00 | 0.00e+00 | +0.0% |

## 12. WIP Band Comparison

### Step C

| Case/Height | K1 WIP Power | K2 WIP Power | Safe? |
|-------------|-------------|-------------|-------|
| C1_slow_ladder_up_down | 0.00e+00 | 0.00e+00 | SAFE |
| C2_random_500dwell | 0.00e+00 | 0.00e+00 | SAFE |
| C3_random_200dwell | 0.00e+00 | 0.00e+00 | SAFE |
| C4_abrupt_stress | 0.00e+00 | 0.00e+00 | SAFE |
| C5_long_random | 0.00e+00 | 0.00e+00 | SAFE |
| focused_low_0p320 | 1.00e-04 | 1.00e-04 | SAFE |
| focused_high_0p480 | 2.00e-04 | 0.00e+00 | SAFE |

### Step E

| Case/Height | K1 WIP Power | K2 WIP Power | Safe? |
|-------------|-------------|-------------|-------|
| low_0p300 | 0.00e+00 | 0.00e+00 | SAFE |
| low_0p320 | 1.00e-04 | 1.00e-04 | SAFE |
| low_0p330 | 0.00e+00 | 0.00e+00 | SAFE |
| low_0p340 | 1.00e-04 | 1.00e-04 | SAFE |
| low_0p360 | 0.00e+00 | 0.00e+00 | SAFE |
| low_0p380 | 0.00e+00 | 0.00e+00 | SAFE |
| high_0p430 | 3.00e-04 | 3.00e-04 | SAFE |
| high_0p450 | 1.00e-04 | 0.00e+00 | SAFE |
| high_0p465 | 4.00e-04 | 3.00e-04 | SAFE |
| high_0p480 | 2.00e-04 | 0.00e+00 | SAFE |

## 13. Support/Posture Comparison

| Suite | K1 avg pitch_rms_deg | K2 avg pitch_rms_deg | K1 avg support_rms_m | K2 avg support_rms_m |
|-------|---------------------|---------------------|---------------------|---------------------|
| Step C | 3.61 | 3.56 | 0.0440 | 0.0418 |
| Step E | 3.37 | 3.26 | 0.0543 | 0.0514 |

## 14. Recovery Comparison

No push disturbances in Step C/E. All cases are fixed-height standing balance. Both K1 and K2 maintain stable posture without falls.

## 15. Final Classification

**`K2_STEP_C_E_STRONG_PASS_PROMOTE_NOW`**

## 16. Promotion Decision

**PROMOTE.** K2 passes all Step C/E gates. K2_NOTCH_LOW_Q_V1 is promoted to current-best.

Promotion changes:
1. Update current-best pointer from K1_PITCH_RATE_NOTCH_V1 to K2_NOTCH_LOW_Q_V1
2. K1 becomes previous-best legacy reference
3. Update CLAUDE.md and any current-best documentation

## 17. Promotion Changes (Exact Files)

| File | Change | Purpose |
|------|--------|---------|
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py:9045` | `k1_current_best_id` updated from `K1_PITCH_RATE_NOTCH_V1` to `K2_NOTCH_LOW_Q_V1` | Current-best pointer |
| `tests/test_current_best_controller_profile.py` | `test_k1_pitch_rate_notch_v1_is_current_best` → `test_k2_notch_low_q_v1_is_current_best` + `test_k1_pitch_rate_notch_v1_is_legacy` | K2 is current-best, K1 is legacy |
| `tests/test_k2_notch_low_q_profile.py` | `test_k2_not_current_best_default` → `test_k2_is_current_best` | K2 is now current-best |
| `scripts/validate_k2_step_c_e_fixed_height.py` | NEW | Step C/E validation runner |
| `tests/test_k2_best_current_promotion.py` | NEW | 20 promotion validation tests |
| `outputs/k2_step_c_e_promotion_validation/` | NEW | 17 K2 simulation outputs (7 Step C + 10 Step E) |
| `docs/validation/k2_step_c_e_validation_and_best_current_promotion_report.md` | NEW | This report |

**Not changed:**
- K1 profile parameters (q=6.0 unchanged)
- K2 profile parameters (q=2.0 unchanged)
- Non-filter gains (kp=50, kd=10, etc.)
- Height gate thresholds (0.42-0.48 m)
- Mode-div parameters (kp=10.0, kd=0.50, mt=7.5, sg=0.80)

## 18. Current-Best After Promotion

| Item | Value |
|------|-------|
| Current-best | `K2_NOTCH_LOW_Q_V1` |
| Profile | `k2_notch_low_q_v1` |
| wip_notch_q | 2.0 |
| Status | `CURRENT_BEST_PROMOTED_STEP_C_E_D_VALIDATED` |

## 19. K1 Previous-Best Legacy Reference

| Item | Value |
|------|-------|
| Previous current-best | `K1_PITCH_RATE_NOTCH_V1` |
| Profile | `k1_pitch_rate_notch_v1` (unchanged, still selectable) |
| wip_notch_q | 6.0 |
| Status | `PREVIOUS_BEST_LEGACY` |

K1 (`k1_pitch_rate_notch_v1`, Q=6.0) remains fully available and runnable via `--vd-sagittal-authority-profile k1_pitch_rate_notch_v1`. All K1 parameters are preserved exactly as they were before K2 promotion. No K1 parameters were modified.

## 20. Tests/Compile Checks Run

### Compile checks (4/4 passed)

```
python -m py_compile scripts/validate_k2_step_c_e_fixed_height.py            -> OK
python -m py_compile wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py -> OK
python -m py_compile scripts/simulate_hierarchical_controller.py              -> OK
python -m py_compile tests/test_k2_best_current_promotion.py                  -> OK
```

### Test results (114/114 passed)

```
pytest tests/test_k2_best_current_promotion.py -v                           -> 20 passed
pytest tests/test_k2_notch_low_q_profile.py -v                              -> 24 passed
pytest tests/test_k2_step_d_push_matrix_validation.py -v                    -> 31 passed
pytest tests/test_current_best_controller_profile.py -v                     -> 9 passed
pytest tests/test_final_validation_rejects_stub_source.py -v                -> 9 passed
pytest tests/test_k1_augmented_telemetry.py -v                              -> 21 passed
                                                                   TOTAL: 114 passed
```

## 21. Limitations

1. **2000-step runs**: May not capture long-term steady-state behavior.
2. **Fixed-height only**: Step C does not test true dynamic height transitions (notch gate crossing). Only the notch-active endpoint height (high_0p480) is tested.
3. **No push disturbances in Step C/E**: Standing balance only; push recovery validated in separate Step D (24 runs, all passed).
4. **No random seed sweep**: Each condition run once for K1 and K2.
5. **No hardware validation**: All results are simulation-only.
6. **high_0p450 notch gate is partial**: At 0.450 m, the notch gate is between start (0.42 m) and full (0.48 m), so only partial attenuation is applied. Both K1 and K2 have limited effect here.
