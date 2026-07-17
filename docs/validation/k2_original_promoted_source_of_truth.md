# K2 Original Promoted Source of Truth

**Date:** 2026-06-29  
**Task:** Lock original promoted K2 metrics as behavioral source of truth  
**Source reports (4):**

1. `k2_notch_low_q_v1_create_and_validate_report.md` (2026-06-25)
2. `k2_step_d_push_matrix_validation_report.md` (2026-06-25)
3. `k2_step_c_e_validation_and_best_current_promotion_report.md` (2026-06-25)
4. `k2_post_promotion_long_run_and_dynamic_height_regression_report.md` (2026-06-25)

**Original output dirs (5):**

- `outputs/k2_step_c_e_promotion_validation/`
- `outputs/k2_step_d_push_matrix_validation/`
- `outputs/k2_post_promotion_long_run/`
- `outputs/k2_dynamic_height_gate_crossing/`
- `outputs/k2_notch_low_q_v1_validation/`

**Final classification in original reports:**
- `K2_STEP_C_E_STRONG_PASS_PROMOTE_NOW`
- `K2_STEP_D_STRONG_PASS_PROMOTE_READY`

---

## 1. K2 Profile Definition

K2 (`k2_notch_low_q_v1`) differs from K1 (`k1_pitch_rate_notch_v1`) in EXACTLY ONE parameter:

| Parameter | K1 | K2 |
|-----------|----|----|
| `wip_notch_q` | 6.0 | **2.0** |

All other gains and control params identical between K1 and K2:
- `kp_pitch=50.0`, `kd_pitch=10.0`
- `k_position=40.0`, `k_velocity=15.0`
- `k_wheel_velocity=0.5`, `k_support_velocity=0.0`
- `max_position_tau=3.0 Nm`, `max_tau_wheel=5.0 Nm`
- `wip_notch_center_hz=2.5`, blend=1.0
- height gate: start=0.42 m, full=0.48 m
- Mode-div: kp=10.0, kd=0.50, mt=7.5, sg=0.80
- `velocity_damping_scale=1.10`
- `apcr1nd_hold_outside_band=True`
- No WBC, no hidden torque

**K2 variant applicability:**
```
applies_to_variants = (
    "low_0p300", "low_0p330", "low_0p360", "extreme_height",
    "high_0p430", "high_0p450", "high_0p465", "high_0p480",
)
```
Note: low_0p320, low_0p340, low_0p380 are NOT in applies_to_variants, so
`is_active_for_variant()` returns False for those, setting `velocity_damping_scale=1.0`.

**Mode-div was ENABLED in ALL original K2 Python validation runs.**
Validation scripts use:
```
--enable-mode-hip-yaw-divergence
--mode-hip-yaw-div-kp 10.0
--mode-hip-yaw-div-kd 0.50
--mode-hip-yaw-div-max-torque 7.5
--mode-hip-yaw-div-soft-limit-rad 0.30
--mode-hip-yaw-div-soft-gain 0.80
--mode-hip-yaw-div-ref-source target
```

---

## 2. Step C Results (Original K2 Python — 7 cases, 2000 steps each)

All cases: mode_div ENABLED. Cases C1–C5 at low_0p330 (notch inactive).

| Case | pitch_rms_deg | support_rms_m | hip_yaw_max_rad | LF_power | WIP_power | fell |
|------|--------------|---------------|-----------------|----------|-----------|------|
| C1_slow_ladder_up_down | 3.63 | 0.0386 | 0.0851 | 1.80e-03 | 0.00e+00 | False |
| C2_random_500dwell | 3.63 | 0.0386 | 0.0851 | 1.80e-03 | 0.00e+00 | False |
| C3_random_200dwell | 3.63 | 0.0386 | 0.0851 | 1.80e-03 | 0.00e+00 | False |
| C4_abrupt_stress | 3.63 | 0.0386 | 0.0851 | 1.80e-03 | 0.00e+00 | False |
| C5_long_random | 3.63 | 0.0386 | 0.0851 | 1.80e-03 | 0.00e+00 | False |
| focused_low_0p320 | 2.83 | 0.0525 | 0.0502 | 2.00e-03 | 1.00e-04 | False |
| focused_high_0p480 | 3.96 | 0.0471 | 0.0563 | 0.00e+00 | 0.00e+00 | False |

Step C max hip_yaw: 0.0851 rad, gate ≤0.35 → PASS

---

## 3. Step E Results (Original K2 Python — 10 heights, 2000 steps each)

All heights: mode_div ENABLED. Notch activates progressively from high_0p430.

| Height | pitch_rms_deg | support_rms_m | hip_yaw_max_rad | LF_power | WIP_power | fell |
|--------|--------------|---------------|-----------------|----------|-----------|------|
| low_0p300 | 2.68 | 0.0421 | **0.1314** | 9.00e-04 | 0.00e+00 | False |
| low_0p320 | 2.83 | 0.0525 | **0.0502** | 2.00e-03 | 1.00e-04 | False |
| low_0p330 | 3.63 | 0.0386 | **0.0851** | 1.80e-03 | 0.00e+00 | False |
| low_0p340 | 2.97 | 0.0541 | **0.0445** | 1.00e-04 | 1.00e-04 | False |
| low_0p360 | 1.90 | 0.0371 | **0.0959** | 1.30e-03 | 0.00e+00 | False |
| low_0p380 | 3.33 | 0.0480 | **0.0392** | 1.00e-04 | 0.00e+00 | False |
| high_0p430 | 4.98 | 0.0637 | **0.0236** | 1.00e-04 | 3.00e-04 | False |
| high_0p450 | 2.75 | 0.0694 | **0.0904** | 2.00e-04 | 0.00e+00 | False |
| high_0p465 | 3.55 | 0.0617 | **0.0296** | 2.00e-04 | 3.00e-04 | False |
| high_0p480 | 3.96 | 0.0471 | **0.0563** | 0.00e+00 | 0.00e+00 | False |

Step E max hip_yaw: **0.1314 rad** (at low_0p300), gate ≤0.35 → PASS

---

## 4. Step D Push Matrix Results (Original K2 Python — 12 conditions, 2000 steps)

Push timing: single push at step 300, duration 5 steps, xfrc_applied to body 1 (torso).
Mode_div ENABLED in both K1 and K2 runs.

| Condition | K1 fell | K2 fell | K1 Pitch500 | K2 Pitch500 | K1 Supp500 | K2 Supp500 | K1 Hy | K2 Hy | Class |
|-----------|---------|---------|-------------|-------------|------------|------------|-------|-------|-------|
| high_0p480_sagittal_forward_60N | no | no | 0.1362 | 0.1376 | 0.1200 | 0.1125 | 0.0000 | 0.0000 | BETTER |
| high_0p480_sagittal_forward_90N | no | no | 0.1446 | 0.1118 | 0.1471 | 0.1443 | 0.0000 | 0.0000 | BETTER |
| high_0p480_sagittal_backward_60N | no | no | 0.1713 | 0.1536 | 0.1172 | 0.1114 | 0.0000 | 0.0000 | BETTER |
| high_0p480_sagittal_backward_90N | no | no | 0.1597 | 0.1536 | 0.1513 | 0.1442 | 0.0000 | 0.0000 | BETTER |
| mid_0p400_sagittal_forward_60N | no | no | 0.1583 | 0.1583 | 0.1091 | 0.1091 | 0.0000 | 0.0000 | EQUIVALENT |
| mid_0p400_sagittal_forward_90N | no | no | 0.2397 | 0.2397 | 0.1137 | 0.1137 | 0.0000 | 0.0000 | EQUIVALENT |
| mid_0p400_sagittal_backward_60N | no | no | 0.3256 | 0.3256 | 0.2014 | 0.2014 | 0.0000 | 0.0000 | EQUIVALENT |
| mid_0p400_sagittal_backward_90N | no | no | 0.3255 | 0.3255 | 0.3147 | 0.3147 | 0.0000 | 0.0000 | EQUIVALENT |
| low_0p330_sagittal_forward_60N | no | no | 0.3735 | 0.3735 | 0.1500 | 0.1500 | 0.0000 | 0.0000 | EQUIVALENT |
| low_0p330_sagittal_forward_90N | no | no | 0.2517 | 0.2517 | 0.2473 | 0.2473 | 0.0000 | 0.0000 | EQUIVALENT |
| low_0p330_sagittal_backward_60N | no | no | 0.3332 | 0.3332 | 0.0926 | 0.0926 | 0.0000 | 0.0000 | EQUIVALENT |
| low_0p330_sagittal_backward_90N | no | no | 0.5402 | 0.5402 | 0.1183 | 0.1183 | 0.0000 | 0.0000 | EQUIVALENT |

Step D falls: K1=0, K2=0. All hip_yaw = 0.0000 (mode_div keeping divergence at zero).

Push sequence JSON format: `[[start_step, fx_N, fy_N, duration_steps]]`
Example (bwd 90N): `[[300, 0.0, -90, 5]]`

---

## 5. Dynamic Height Results (Original K2 Python — 5000 steps)

Mode_div ENABLED for all dynamic height cases.

| Scenario | K2 pitch_rms | K2 ht_rmse | K2 hy_max | fell |
|----------|-------------|-----------|-----------|------|
| ramp_up_0p330_to_0p480 | 3.15 | 0.1051 | 0.0534 | None |
| ramp_down_0p480_to_0p330 | 5.84 | 0.1149 | 0.0977 | None |
| up_down_cycle | 3.32 | 0.0946 | 0.0534 | None |
| gate_dwell | 3.05 | 0.1097 | 0.0534 | None |
| gate_chatter | 2.98 | 0.0905 | 0.0629 | None |

Dynamic hy max: 0.0977 rad, gate ≤0.35 → PASS

---

## 6. Long-Run Equilibrium Results (Original K2 Python — 6000 steps)

Mode_div ENABLED.

| Height | K2 pitch_rms | K2 pitch_f | K2 hy_max | fell |
|--------|-------------|-----------|-----------|------|
| low_0p330 | 3.97 | 4.34 | 0.2048 | False |
| mid_0p400 | 1.84 | 2.51 | 0.1071 | False |
| high_0p430 | 5.60 | 5.69 | 0.0496 | False |
| high_0p450 | 3.45 | 3.72 | 0.0882 | False |
| high_0p480 | 5.15 | 5.69 | 0.0574 | False |

Long-run hy max: 0.2048 rad, gate ≤0.35 → PASS
Classification: K2_POST_PROMOTION_INVALID (K1 runs were not run, so comparison invalid).
But hy gate still passes.

---

## 7. Absolute Safety Gates

| Gate | Threshold | Original K2 | Current JAX/dedicated |
|------|-----------|-------------|----------------------|
| Falls | 0 | PASS (all) | Currently PASS (survival ok) |
| hip_yaw_max | ≤0.35 rad | Max 0.2048 | **~0.412 rad** — FAIL |
| NaN/Inf | none | PASS | Unknown |
| Hidden torque | none | PASS | Unknown |
| WBC | none | PASS | PASS |
| real_simulation | yes | YES | YES |

---

## 8. Current JAX/Dedicated vs Original K2 Divergence

### Critical: hip_yaw_divergence at low heights

| Height | Original K2 hy_max | Current JAX (~) | Ratio | Gate |
|--------|-------------------|-----------------|-------|------|
| low_0p300 | 0.1314 | ~0.412 | 3.1× | FAIL (0.412 > 0.35) |
| low_0p320 | 0.0502 | ~0.392 | 7.8× | FAIL (0.392 > 0.35) |
| low_0p330 | 0.0851 | ~0.361 | 4.2× | FAIL (0.361 > 0.35) |

### Root cause identified: mode_div disabled in dedicated runner

- **Original K2 Python:** `--enable-mode-hip-yaw-divergence` (mode_div ENABLED)
- **Dedicated runner:** `DEFAULT_MODE_DIV_REF_SOURCE = "disabled"`, no `--enable-mode-hip-yaw-divergence` flag

The mode_div controller provides antisymmetric ±7.5 Nm hip-yaw counter-torque
via `-(kp_div * div_error + kd_div * div_rate)`, height-gated. Without it, there
is nothing actively fighting left-right hip-yaw differential excitation.

### Push mechanism parity

- Both use `xfrc_applied` on body 1 (torso): ✅ match
- Push sequence format `[step, fx, fy, dur]`: ✅ match
- BUT: original K2 Step D had mode_div ENABLED → all hip_yaw=0.0000
- Current dedicated runner has mode_div DISABLED → push hip_yaw likely non-zero
- Current push 90N fails in dedicated path — mode_div absence is likely cause

---

## 9. Push Mechanism Specification

- Method: `mj_data.xfrc_applied[1, ...]` applied to body 1 (torso)
- Timing: single push at specified step, duration 5 control steps
- Direction conventions:
  - Forward: +fy (sagittal forward)
  - Backward: -fy (sagittal backward)
- Force magnitudes: 60N, 90N
- Push JSON format: `{"sequence": [[start_step, fx_N, fy_N, duration_steps]]}`

---

## 10. Mode-Div Controller Specification

```
raw = -(kp_div * div_error + kd_div * div_rate)
div_error = (q_hip_yaw_left - q_hip_yaw_right) - (q_ref_hip_yaw_left - q_ref_hip_yaw_right)
div_rate = qd_hip_yaw_left - qd_hip_yaw_right

Height gate (smoothstep down):
  low = soft_limit_rad = 0.30 m
  high = soft_limit_rad + soft_gain = 0.30 + 0.80 = 1.10 m
  gate = 1.0 when height ≤ 0.30, 0.0 when height ≥ 1.10

Torque = clip(raw * height_gate, -max_torque, +max_torque)
tau_left = +Torque (index 1)
tau_right = -Torque (index 6)
```

Parameters: kp=10.0, kd=0.50, max_torque=7.5 Nm, soft_limit_rad=0.30, soft_gain=0.80

---

## 11. File Inventory

### Required scripts (original K2 Python validation)
- `scripts/validate_k2_step_c_e_fixed_height.py` — Step C/E runner (MODE_DIV_FLAGS hardcoded)
- `scripts/validate_k2_step_d_push_matrix.py` — Step D push matrix runner (MODE_DIV_FLAGS hardcoded)
- `scripts/validate_k2_post_promotion_long_run.py` — Long run runner
- `scripts/validate_k2_dynamic_height_gate_crossing.py` — Dynamic height runner
- `scripts/simulate_hierarchical_controller.py` — General simulation (supports `--controller-backend`)

### Required controller files
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` — K2 profile definition
- `wheeled_biped/controllers/mode_based_hip_yaw_divergence_controller.py` — Python mode_div
- `wheeled_biped/controllers/k2_jax_controller.py` — JAX K2 controller (mode_div included)

### Current runner under test
- `scripts/run_k2_jax_realtime.py` — Dedicated realtime runner (mode_div DISABLED by default)

---

## 12. Required Candidate Behavior

For `K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PASS`:

| Metric | Required |
|--------|----------|
| hip_yaw_max @ low_0p300 | ≤0.35 abs, ≤0.1314 ideal (match original) |
| hip_yaw_max @ low_0p320 | ≤0.35 abs, ≤0.0502 ideal |
| hip_yaw_max @ low_0p330 | ≤0.35 abs, ≤0.0851 ideal |
| hip_yaw_max @ all Step E heights | ≤0.35 abs, no worse than original K2 |
| Step C all cases | 0 falls, hip_yaw ≤0.35 |
| Step E all heights | 0 falls, hip_yaw ≤0.35 |
| Step D all push conditions | 0 falls, hip_yaw ≤0.35 |
| Single push high_0p480 90N | 0 falls |
| Dynamic height (ramp_up/down/chatter) | 0 falls, metrics no worse than original |
| No NaN/Inf | PASS |
| No hidden torque | PASS |
| No WBC | PASS |
| Push mechanism | xfrc_applied, step 300, duration 5 |
| Mode-div | ENABLED, matching original K2 params |
| Realtime | ≥50 Hz min, preferably >100 Hz |
| Telemetry/metadata | Correct, no per-step print |
