# K2 JAX Post-Strict-Clone Functional Spot Check

## Date: 2026-06-28

## Classification: **K2_JAX_POST_STRICT_CLONE_FUNCTIONAL_SPOT_CHECK_PASS**

---

## 1. Context

Functional spot checks rerun after the final two hip-yaw parity fixes to confirm JAX backend survives all operational scenarios. The fixes change hip-yaw control output (yaw sign correction, mode-div height source correction) — these spot checks validate that the CORRECTED control does not introduce functional regressions.

---

## 2. Run Configuration

| Parameter | Value |
|-----------|-------|
| Backend | JAX |
| Profile | k2_notch_low_q_v1 |
| Controller mode | balance-core |
| Sagittal controller | velocity-damped |
| Mode-div | enabled (kp=10.0, kd=0.50, max=7.5, soft_limit=0.30, soft_gain=0.80) |

---

## 3. Fixed-Height Spot Checks

### high_0p480 (from long-run, 6000 steps)

| Metric | Value |
|--------|-------|
| Steps | 5999 |
| Fall | False |
| hip_yaw_abs_max | 0.0574 rad |
| pitch_rms | 5.15 deg |
| roll_max | 0.31 deg |
| hidden_torque | 0.0 |
| WBC | 0 |
| NaN | 0 |
| Status | **PASS** |

### low_0p330 (from long-run, 6000 steps)

| Metric | Value |
|--------|-------|
| Steps | 5999 |
| Fall | False |
| hip_yaw_abs_max | 0.2048 rad |
| pitch_rms | 3.97 deg |
| roll_max | 1.04 deg |
| hidden_torque | 0.0 |
| WBC | 0 |
| NaN | 0 |
| Status | **PASS** |

---

## 4. Push Recovery Spot Checks

### high_0p480 forward 90N
Command:
```
python scripts/simulate_hierarchical_controller.py
  --controller-mode balance-core --sagittal-controller velocity-damped
  --vd-sagittal-authority-profile k2_notch_low_q_v1
  --height-variant-setup .../high_0p480_setup.json
  --steps 1000 --controller-backend jax
  --push-sequence-file .../push_seq_fwd_90N.json
```
Push: force_y=+90.0 N, step 200, duration 20 steps.

| Metric | Value |
|--------|-------|
| Rows | 229 |
| Fall | False |
| hip_yaw_abs_max | 0.1767 rad |
| pitch_max | 29.99 deg |
| pitch_rms | 6.27 deg |
| roll_max | 0.36 deg |
| hidden_torque | 0.0 |
| WBC | 0 |
| Push active | 20 rows |
| Status | **PASS** (recovered from push) |

### high_0p480 backward 90N
Command: Same as forward, push force_y=-90.0 N.

| Metric | Value |
|--------|-------|
| Rows | 229 |
| Fall | False |
| hip_yaw_abs_max | 0.1270 rad |
| pitch_max | 23.84 deg |
| pitch_rms | 4.82 deg |
| roll_max | 1.93 deg |
| hidden_torque | 0.0 |
| WBC | 0 |
| Push active | 20 rows |
| Status | **PASS** (recovered from push) |

---

## 5. Dynamic Height Spot Checks

Run via `scripts/validate_k2_dynamic_height_gate_crossing.py --controller-backend jax`.

| Scenario | Steps | Fell | hy_max | pitch_rms | Status |
|----------|-------|------|--------|-----------|--------|
| ramp_up (0.33→0.48) | 5000 | False | 0.0534 | 3.15 | PASS |
| ramp_down (0.48→0.33) | 5000 | False | 0.0977 | 5.84 | PASS |
| up_down_cycle | 7000 | False | 0.0534 | 3.32 | PASS |
| gate_dwell (0.42→0.48) | 6000 | False | 0.0534 | 3.05 | PASS |
| gate_chatter (0.40⇌0.47) | 5000 | False | 0.0629 | 2.98 | PASS |

All 5 dynamic scenarios pass with JAX backend. No falls, hy_max well within safety bound.

---

## 6. Summary

| Scenario | Result |
|----------|--------|
| fixed_high_0p480 | PASS |
| fixed_low_0p330 | PASS |
| push_fwd_90N | PASS |
| push_bwd_90N | PASS |
| ramp_up | PASS |
| ramp_down | PASS |
| up_down_cycle | PASS |
| gate_dwell | PASS |
| gate_chatter | PASS |

---

## 7. Classification

### K2_JAX_POST_STRICT_CLONE_FUNCTIONAL_SPOT_CHECK_PASS

All 9 spot checks pass with JAX backend after the hip-yaw fixes. No falls, no NaN, no actuator violations, no hidden torque, no WBC leakage. The yaw sign correction and mode-div height source correction do not introduce functional regressions.
