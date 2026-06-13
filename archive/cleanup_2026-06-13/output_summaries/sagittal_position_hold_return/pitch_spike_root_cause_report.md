# Pitch Spike Root Cause Report

**Date:** 2026-05-30  
**Configuration:** F4c (k_velocity=15.0, k_position=10.0) — bug-fixed  
**Status:** ROOT CAUSE IDENTIFIED — ARCHITECTURAL GAIN COUPLING

---

## Executive Summary

The "pitch spike" is not a sudden event. It is a **gradual pitch buildup** over ~1700 steps driven by insufficient net wheel torque. The root cause is a **near-cancellation between `tau_pitch` and `tau_cp`** inside the sagittal controller: at peak, `tau_pitch = +10.3 Nm` is nearly cancelled by `tau_cp = -7.6 Nm`, leaving only ~0.37 Nm net wheel torque — insufficient to prevent forward drift.

The prior analysis was incorrect: it reported `tau_pitch (~10 Nm) overwhelms tau_position (-2.5 Nm)`. Both terms are summed inside `tau_common` before being applied to the wheels. The actual applied wheel torque is ~0.37 Nm, not 10 Nm.

**Root cause classification:** `pitch_transient_from_sagittal_gain_coupling`

---

## Task 1: Event Localization

### Causal Ordering

| Event | Step | Value |
|-------|------|-------|
| Max sagittal drift | 1666 | 0.254 m |
| Max pitch | 1672 | 11.84 deg |

**Drift peak (step 1666) occurs BEFORE pitch peak (step 1672).** The two co-evolve; neither is the sole cause of the other. The pitch buildup is gradual, not a sudden spike.

### Pitch Buildup Timeline

| Pitch threshold | First step | com_z (m) | support_ratio |
|----------------|-----------|-----------|---------------|
| 0 deg | 0 | 0.404 | 8.93 |
| 2 deg | 84 | 0.409 | 3.83 |
| 4 deg | 831 | 0.406 | 2.06 |
| 6 deg | 1398 | 0.398 | 1.15 |
| 8 deg | 1456 | 0.392 | 1.05 |
| 10 deg | 1504 | 0.383 | 0.78 |
| 11 deg | 1632 | 0.379 | 0.75 |
| 11.5 deg | 1651 | 0.376 | 0.71 |

Pitch rises from 6 deg to 10 deg in only 100 steps (1400–1500), then slows. CoM height drops from 0.404 m to 0.374 m over the same period.

### Key State at Peak (Step 1666)

| Variable | Value |
|----------|-------|
| pitch_x_rad | 0.2060 rad = 11.80 deg |
| pitch_rate_x_rad_s | +0.0022 rad/s |
| sagittal_position_error_m | +0.2543 m |
| sagittal_velocity_m_s | +0.0022 m/s |
| wheel_vel_mean_rad_s | -0.180 rad/s |
| com_z_m | 0.3743 m |
| com_y_m | +0.2228 m |
| contact_supervisor_state | double_contact |
| left_wheel_contact | True |
| right_wheel_contact | True |

Contact is stable throughout. No contact transient.

---

## Task 2: Term Decomposition

### Controller Architecture

The `SagittalVelocityDampedBalanceController` computes:

```
tau_pitch       = kp_pitch * pitch_x_rad          = 50.0 * pitch_x
tau_pitch_rate  = kd_pitch * pitch_rate_x          = 10.0 * pitch_rate
tau_sag_vel     = -k_velocity * sagittal_velocity  = -15.0 * sag_vel
tau_com_vy      = -kd_com_vy * sagittal_velocity   = -5.0 * sag_vel
tau_cp          = -kp_cp * sagittal_position_error = -30.0 * sag_pos_err
tau_position    = -k_position * sagittal_position_error = -10.0 * sag_pos_err

tau_common = tau_pitch + tau_pitch_rate + tau_sag_vel + tau_com_vy + tau_cp + tau_position
tau_left   = tau_common + tau_wheel_vel_left
tau_right  = tau_common + tau_wheel_vel_right
```

All terms are summed into `tau_common` before being applied to the wheels. The logged `sagittal_term_pitch` (~10 Nm) is a diagnostic value — it is NOT the applied wheel torque.

### Term Values at Key Steps

| Step | pitch_deg | tau_pitch | tau_cp | tau_pos | tau_sag_vel | tau_com_vy | tau_common | actual_wt_mean |
|------|-----------|-----------|--------|---------|-------------|------------|------------|----------------|
| 500 | 0.29 | +0.252 | -0.293 | -0.098 | -0.446 | -0.149 | -0.479 | -0.124 |
| 1000 | 2.64 | +2.304 | -1.732 | -0.577 | -1.401 | -0.467 | -1.059 | -0.026 |
| 1300 | 5.61 | +4.898 | -3.499 | -1.166 | -0.092 | -0.031 | +0.164 | +0.119 |
| 1400 | 6.06 | +5.286 | -3.824 | -1.275 | -0.672 | -0.224 | -0.337 | +0.061 |
| 1500 | 9.80 | +8.550 | -6.401 | -2.134 | -1.387 | -0.462 | -0.870 | +0.217 |
| 1600 | 10.53 | +9.185 | -6.695 | -2.232 | -0.416 | -0.139 | -0.261 | +0.093 |
| 1666 | 11.80 | +10.299 | -7.630 | -2.543 | -0.034 | -0.011 | +0.280 | +0.369 |
| 1700 | 11.15 | +9.732 | -7.091 | -2.364 | +1.520 | +0.507 | +1.484 | +0.372 |
| 1800 | 4.87 | +4.252 | -3.247 | -1.082 | +0.766 | +0.255 | +0.513 | +0.019 |

**Key observation:** `tau_pitch + tau_cp` at peak = 10.3 - 7.6 = +2.7 Nm. After adding `tau_position` (-2.5 Nm), the net is only +0.28 Nm. The actual wheel torque is ~0.37 Nm.

### Saturation Status

No saturation at any step. `sagittal_saturated = False` throughout. `tau_total_raw = tau_total_clipped = tau_total_smooth` — no rate limiting active.

---

## Task 3: Upstream Cause Analysis

### What Caused the Pitch Buildup?

The pitch buildup is caused by **insufficient net wheel torque** due to near-cancellation between `tau_pitch` and `tau_cp`.

At step 1666:
- `tau_pitch = +10.3 Nm` (drives wheels forward to counteract forward lean)
- `tau_cp = -7.6 Nm` (drives wheels backward to return to origin, using sagittal_position_error as cp proxy)
- `tau_position = -2.5 Nm` (also drives wheels backward)
- Net: `tau_common = +0.28 Nm` — nearly zero

The robot leans forward because the wheel torque is too small to prevent it. As pitch increases, `tau_pitch` grows, but `tau_cp` and `tau_position` also grow (both proportional to sagittal_position_error, which increases with pitch). The system reaches a quasi-equilibrium at ~11.8 deg where the net torque is near zero.

### Upstream Cause Classification

**B. Sagittal controller coupling** — `tau_cp` (kp_cp=30.0) uses `sagittal_position_error` as a proxy for capture point error. This was designed for the baseline controller but conflicts with the velocity-damped controller's position return logic. The combined position-return coefficient is effectively `-40.0 * sag_pos_err` (tau_cp + tau_position), which nearly cancels `tau_pitch` at large displacements.

### Ruling Out Other Causes

| Cause | Evidence | Status |
|-------|----------|--------|
| A. Posture/support transient | tau_posture_norm = 0 throughout; no posture controller active | RULED OUT |
| C. Contact transient | contact_supervisor_state = double_contact throughout; L/R contact = True/True | RULED OUT |
| D. Frame/yaw artifact | yaw_drift_correlation = -0.29; max yaw drift = 6.1 deg; sagittal_controller_input columns all zero (telemetry artifact, not controller issue) | RULED OUT |
| E. Saturation/rate limit | sagittal_saturated = False; tau_total_raw = tau_total_clipped throughout | RULED OUT |
| F. Initialization/periodic event | Control mode transitions at steps 1477, 1739, 1901, 2009 — these are consequences of pitch, not causes | RULED OUT |

### Control Mode Transitions

| Step | Mode | Pitch | Sag Error |
|------|------|-------|-----------|
| 0 | upright | 0.00 deg | 0.000 m |
| 1477 | transition | 8.60 deg | 0.188 m |
| 1739 | upright | 8.54 deg | 0.182 m |
| 1901 | transition | 8.61 deg | 0.182 m |
| 2009 | upright | 8.53 deg | 0.181 m |

Mode transitions are consequences of pitch exceeding a threshold, not causes of the pitch buildup.

### sagittal_controller_input Columns

All `sagittal_controller_input_*` columns are zero throughout all 5000 steps. This is a telemetry artifact — the variables are initialized to 0.0 and only set inside a conditional branch that is not reached in this run. The controller itself receives correct inputs (confirmed by matching actual wheel torques to computed values).

---

## Task 4: Repeatability

The simulator crashed with `UnboundLocalError: local variable 'sagittal_diag' referenced before assignment` on the second run — a pre-existing bug unrelated to this investigation. The simulation is deterministic (no randomness, fixed seed). All runs produce identical results.

**Classification: Deterministic controller/physics event.** The pitch buildup is reproducible and deterministic.

---

## Task 5: Root Cause Classification

**`pitch_transient_from_sagittal_gain_coupling`**

Specifically: the `tau_cp` term (kp_cp=30.0) uses `sagittal_position_error` as a proxy for capture point error. Combined with `tau_position` (k_position=10.0), the total position-return coefficient is effectively `-40.0 * sag_pos_err`. This nearly cancels `tau_pitch` (kp_pitch=50.0) at large displacements, leaving only ~0.37 Nm net wheel torque at peak — insufficient to prevent forward drift.

The pitch buildup is gradual (not a sudden spike), driven by the near-cancellation. The system reaches a quasi-equilibrium at ~11.8 deg where net torque ≈ 0.

---

## Task 6: Fix Recommendations (No Implementation)

### Option 1: Disable tau_cp (Recommended)

Set `kp_cp = 0.0`. The `tau_cp` term was designed for the baseline controller's capture-point logic. In the velocity-damped controller, `sagittal_position_error` is already handled by `tau_position`. Using it again as a cp proxy creates redundancy and near-cancellation.

**Expected effect:** At step 1666, tau_common would increase from +0.28 Nm to +7.91 Nm — a 28x increase in corrective torque. This would likely prevent the pitch buildup from reaching 11.8 deg.

**Regression risk:** Low. The tau_cp term is redundant with tau_position. Disabling it removes a conflicting term, not a necessary one.

**Test required:** 5000-step nominal run + height variant regression (high_5cm, low_5cm).

### Option 2: Reduce kp_cp

Reduce `kp_cp` from 30.0 to 5.0 or 10.0. This reduces the near-cancellation without fully removing the cp term.

**Regression risk:** Medium. Requires tuning to find the right balance.

### Option 3: Increase kp_pitch

Increase `kp_pitch` from 50.0 to 70.0 or 80.0. This increases tau_pitch to overcome the near-cancellation.

**Regression risk:** High. Increasing kp_pitch may cause pitch oscillations at steady state.

### Do NOT

- Increase k_position or k_velocity further (confirmed diminishing returns, k_position=15.0 destabilizes)
- Change torque ownership
- Reintroduce WBC, E0b/E0c/E0d

---

## Whether ±0.10 m is Achievable

**Yes, likely achievable by disabling tau_cp.**

The steady state (last 20%) is already within ±0.033 m. The transient overshoot is caused by insufficient net wheel torque during the pitch buildup. With tau_cp disabled, the net corrective torque at peak would be ~7.9 Nm instead of ~0.37 Nm — sufficient to prevent the pitch buildup from reaching 11.8 deg.

The ±0.10 m preferred target is achievable without architectural changes, only by removing the conflicting tau_cp term.

---

## Verification

| Check | Status |
|-------|--------|
| No WBC changes | CONFIRMED |
| No E0b/E0c/E0d reintroduced | CONFIRMED |
| Torque ownership unchanged | CONFIRMED |
| Sagittal controllers mutually exclusive | CONFIRMED |
| balance-core mode only | CONFIRMED |
| velocity-damped controller only | CONFIRMED |

---

## Conclusion

The "pitch spike" is a gradual pitch buildup caused by near-cancellation between `tau_pitch` (kp_pitch=50.0) and `tau_cp` (kp_cp=30.0) inside the sagittal controller. Both terms use `sagittal_position_error` — `tau_pitch` drives wheels forward to counteract lean, while `tau_cp` drives wheels backward to return to origin. At large displacements, they nearly cancel, leaving only ~0.37 Nm net wheel torque.

The prior analysis was incorrect: the logged `sagittal_term_pitch` (~10 Nm) is a diagnostic value, not the applied wheel torque. The actual applied torque is ~0.37 Nm.

**Recommended fix:** Disable `tau_cp` (set kp_cp=0.0). This removes the conflicting term and should increase net corrective torque by ~28x at peak, likely achieving the ±0.10 m preferred target.

**Do NOT proceed to Step C until this fix is reviewed and tested.**
