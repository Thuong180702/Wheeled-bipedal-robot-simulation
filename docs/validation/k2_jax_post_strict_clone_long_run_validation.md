# K2 JAX Post-Strict-Clone Long-Run Validation

## Date: 2026-06-28

## Classification: **K2_JAX_POST_STRICT_CLONE_LONG_RUN_PASS**

---

## 1. Context

This long-run was executed after the final two hip-yaw parity fixes:

1. **Yaw error sign fix** (`scripts/simulate_hierarchical_controller.py:6544`):
   `yaw_error_rad = initial_yaw_z - body_yaw_z` (negated to match Python YawController's `0.0 - current_yaw`)

2. **Mode-div height source fix** (`wheeled_biped/controllers/k2_jax_controller.py:1456`):
   `com_z` instead of `schedule_h` for mode-div height gate (matches Python's use of `centroidal_state_control.com_pos[2]`)

The yaw sign fix changes JAX hip-yaw output (previously produced opposite-sign yaw correction). This long-run validates that the corrected JAX controller survives all heights.

---

## 2. Run Configuration

| Parameter | Value |
|-----------|-------|
| Backend | JAX |
| Profile | k2_notch_low_q_v1 |
| Controller mode | balance-core |
| Sagittal controller | velocity-damped |
| Steps per height | 6000 |
| Total JAX steps | 30,000 |
| Heights | 5 |
| Mode-div | enabled (kp=10.0, kd=0.50, max=7.5, soft_limit=0.30, soft_gain=0.80) |
| Runner script | `scripts/validate_k2_post_promotion_long_run.py --suite eq --profile k2 --controller-backend jax` |

---

## 3. Results Per Height

### low_0p330
| Metric | Value |
|--------|-------|
| Steps requested | 6000 |
| Steps completed | 5999 |
| Fall | False |
| Termination reason | (none) |
| hip_yaw_abs_max | 0.2048 rad |
| pitch_max_abs | 9.81 deg |
| pitch_rms | 3.97 deg |
| pitch_rms_final | 4.34 deg |
| roll_max_abs | 1.04 deg |
| roll_rms | 0.39 deg |
| support_error_rms | 0.0813 m |
| support_error_final | 0.083 m |
| hidden_torque_max | 0.0 |
| WBC rows | 0 |
| NaN count | 0 |
| LF power final | 0.0014 |
| Wall-clock | 710 s |

### mid_0p400
| Metric | Value |
|--------|-------|
| Steps requested | 6000 |
| Steps completed | 5999 |
| Fall | False |
| Termination reason | (none) |
| hip_yaw_abs_max | 0.1071 rad |
| pitch_max_abs | 4.56 deg |
| pitch_rms | 1.84 deg |
| pitch_rms_final | 2.51 deg |
| roll_max_abs | 1.15 deg |
| roll_rms | 0.49 deg |
| support_error_rms | 0.0799 m |
| support_error_final | 0.0774 m |
| hidden_torque_max | 0.0 |
| WBC rows | 0 |
| NaN count | 0 |
| LF power final | 0.0 |
| Wall-clock | 1055 s |

### high_0p430
| Metric | Value |
|--------|-------|
| Steps requested | 6000 |
| Steps completed | 5999 |
| Fall | False |
| Termination reason | (none) |
| hip_yaw_abs_max | 0.0496 rad |
| pitch_max_abs | 9.61 deg |
| pitch_rms | 5.60 deg |
| pitch_rms_final | 5.69 deg |
| roll_max_abs | 0.68 deg |
| roll_rms | 0.23 deg |
| support_error_rms | 0.0926 m |
| support_error_final | 0.0955 m |
| hidden_torque_max | 0.0 |
| WBC rows | 0 |
| NaN count | 0 |
| LF power final | 0.0011 |
| Wall-clock | 896 s |

### high_0p450
| Metric | Value |
|--------|-------|
| Steps requested | 6000 |
| Steps completed | 5999 |
| Fall | False |
| Termination reason | (none) |
| hip_yaw_abs_max | 0.0882 rad |
| pitch_max_abs | 6.54 deg |
| pitch_rms | 3.45 deg |
| pitch_rms_final | 3.72 deg |
| roll_max_abs | 0.61 deg |
| roll_rms | 0.23 deg |
| support_error_rms | 0.0782 m |
| support_error_final | 0.0809 m |
| hidden_torque_max | 0.0 |
| WBC rows | 0 |
| NaN count | 0 |
| LF power final | 0.0 |
| Wall-clock | 885 s |

### high_0p480
| Metric | Value |
|--------|-------|
| Steps requested | 6000 |
| Steps completed | 5999 |
| Fall | False |
| Termination reason | (none) |
| hip_yaw_abs_max | 0.0574 rad |
| pitch_max_abs | 9.38 deg |
| pitch_rms | 5.15 deg |
| pitch_rms_final | 5.69 deg |
| roll_max_abs | 0.31 deg |
| roll_rms | 0.13 deg |
| support_error_rms | 0.0896 m |
| support_error_final | 0.102 m |
| hidden_torque_max | 0.0 |
| WBC rows | 0 |
| NaN count | 0 |
| LF power final | 0.0 |
| Wall-clock | 679 s |

---

## 4. Summary

| Height | Fell | hy_max | pitch_rms | pitch_final | LF final |
|--------|------|--------|-----------|-------------|----------|
| low_0p330 | No | 0.2048 | 3.97 | 4.34 | 0.0014 |
| mid_0p400 | No | 0.1071 | 1.84 | 2.51 | 0.0 |
| high_0p430 | No | 0.0496 | 5.60 | 5.69 | 0.0011 |
| high_0p450 | No | 0.0882 | 3.45 | 3.72 | 0.0 |
| high_0p480 | No | 0.0574 | 5.15 | 5.69 | 0.0 |

---

## 5. Acceptance Gates

| Gate | Result |
|------|--------|
| 5/5 heights pass | ✓ |
| 6000/6000 steps completed | ✓ (5999 rows = 6000 control steps) |
| No falls | ✓ |
| No NaN | ✓ |
| No hidden torque | ✓ (max = 0.0 all heights) |
| No WBC leakage | ✓ (0 rows all heights) |
| No actuator safety violation | ✓ (hy_max <= 0.2048, all < 0.35 bound) |
| hip_yaw within K2 safety | ✓ |
| No long-run drift regression | ✓ (pitch/roll/support comparable to prior runs) |
| No yaw instability | ✓ (hy_max similar to pre-fix JAX runs) |

## 6. Classification

### K2_JAX_POST_STRICT_CLONE_LONG_RUN_PASS

The corrected JAX controller (with proper yaw sign and mode-div height source) survives all 5 heights for 6000 steps each. No falls, no NaN, no safety violations. Metrics are comparable to or better than pre-fix JAX runs. The yaw sign correction did not introduce instability — hip-yaw abs_max values are well within the 0.35 rad safety bound.
