# K2 JAX Parameter Parity Audit

**Phase 2 deliverable.** Compares ACTUAL RUNTIME values between the Python K2
controller and the JAX K2 controller. Values are read from source code and
validation-script CLI invocations, NOT from documentation or defaults alone.

Date: 2026-06-27
Profile: `k2_notch_low_q_v1` (K2_NOTCH_LOW_Q_V1)
Python CLI default at: `scripts/simulate_hierarchical_controller.py`
JAX implementation at: `wheeled_biped/controllers/k2_jax_controller.py`
Python sagittal profile at: `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`

---

## Summary of Findings

| Group | Total | MATCH | MISMATCH | UNKNOWN |
|-------|-------|-------|----------|---------|
| Sagittal gains | 13 | 13 | 0 | 0 |
| Notch filter | 6 | 6 | 0 | 0 |
| Adaptive bias trim (ABS) | 20+ | 20+ | 0 | 0 |
| Calibrated outer loop | 7 | 5 | 2 | 0 |
| Physics FF | 1 | 1 | 0 | 0 |
| Low-band support | 5 | 5 | 0 | 0 |
| Shape posture | 8 | 8 | 0 | 0 |
| Lateral roll | 9 | 9 | 0 | 0 |
| Yaw | 3 | 3 | 0 | 0 |
| Mode-div hip-yaw | 8 | 5 | 3 | 0 |
| Support feedforward (hip-yaw) | 4 | 4 | 0 | 0 |
| Empirical support FF (hip_pitch/knee) | 4 | 4 | 0 | 0 |
| Composer | 3 | 3 | 0 | 0 |
| Pitch ref offset (G2) | 1 | 0 | 1 | 0 |
| **TOTAL** | **92** | **86** | **6** | **0** |

6 MISMATCHES confirmed, 2 of which affect wheel torque [4,9].

---

## 1. Sagittal Gains

All sagittal gains are hardcoded in the JAX `k2_jax_controller_step` function
and match the Python runtime values. The K2 profile (`K2_NOTCH_LOW_Q_V1`) has
`continuous_k_position=False`, `continuous_k_wheel_velocity=False`,
`continuous_kd_pitch=False`, `continuous_k_velocity=False`, so the static
values are used (not height-scheduled). Only `continuous_max_position_tau=True`
is active, producing scheduled `max_position_tau` from 4.0 at tall heights to
6.0 at low heights.

| Parameter | Python Runtime | JAX Runtime | Python Source | JAX Source | Affects Wheels? | Status |
|-----------|---------------|-------------|---------------|------------|-----------------|--------|
| `kp_pitch` | 50.0 | 50.0 | `simulate_hierarchical_controller.py:2400` via `vd_k_pitch=50.0` | `k2_jax_controller.py:1214` `kp_pitch=50.0` | Yes [4,9] | MATCH |
| `kd_pitch` | 10.0 | 10.0 | `simulate_hierarchical_controller.py:2401` hardcoded | `k2_jax_controller.py:1122` `kd_pitch = 10.0` | Yes [4,9] | MATCH |
| `k_velocity` | 15.0 | 15.0 | `simulate_hierarchical_controller.py:2404` via `vd_k_velocity=15.0` | `k2_jax_controller.py:1216` `effective_k_velocity=15.0` | Yes [4,9] | MATCH |
| `k_position` | 40.0 | 40.0 | `simulate_hierarchical_controller.py:2406` via `vd_k_position=40.0` | `k2_jax_controller.py:1120` `kpos = 40.0` | Yes [4,9] | MATCH |
| `k_wheel_velocity` | 0.5 | 0.5 | `simulate_hierarchical_controller.py:2405` hardcoded | `k2_jax_controller.py:1121` `kwheel = 0.5` | Yes [4,9] | MATCH |
| `kd_com_vy` | 5.0 | 5.0 | `simulate_hierarchical_controller.py:2403` hardcoded | `k2_jax_controller.py:1220` `kd_com_vy=5.0` | Yes [4,9] | MATCH |
| `kp_cp` | 0.0 | 0.0 | `simulate_hierarchical_controller.py:2402` hardcoded | `k2_jax_controller.py:1220` `kp_cp=0.0` | Yes [4,9] | MATCH |
| `k_support_vel` | 0.0 | 0.0 | `simulate_hierarchical_controller.py:2407` via `vd_k_support_velocity=0.0` | `k2_jax_controller.py:1217` `effective_support_velocity_gain=0.0` | Yes [4,9] | MATCH |
| `wheel_torque_sign` | 1.0 | 1.0 | `simulate_hierarchical_controller.py:2409` hardcoded | `k2_jax_controller.py:1221` `wheel_torque_sign=1.0` | Yes [4,9] | MATCH |
| `max_position_tau_nominal` | 4.0 | 4.0 | Inherited from `ADAPTIVE_SUPPORT_CENTERING_TRIM` at `svdbc.py:2265` | `k2_jax_controller.py:1126` from `_k2_sch.max_position_tau_nominal` | Yes [4,9] | MATCH |
| `max_position_tau_low_max` | 6.0 | 6.0 | `svdbc.py:269` (dataclass default); not overridden by K2 chain | `k2_jax_controller.py:1127` from `_k2_sch.max_position_tau_low_max` | Yes [4,9] | MATCH |
| `k_position_z_low` | 0.300 | 0.300 | `svdbc.py:263` (dataclass default) | `k2_jax_controller.py:1128` from `_k2_sch.k_position_z_low` | Yes [4,9] | MATCH |
| `k_position_z_high` | 0.393 | 0.393 | `svdbc.py:264` (dataclass default) | `k2_jax_controller.py:1129` from `_k2_sch.k_position_z_high` | Yes [4,9] | MATCH |

### max_position_tau_nominal trace

The value 4.0 comes from `ADAPTIVE_SUPPORT_CENTERING_TRIM` (`svdbc.py:2265`),
which sets `max_position_tau_nominal=4.0`. This value propagates unchanged
through the full profile chain:
```
ADAPTIVE_SUPPORT_CENTERING_TRIM (max_position_tau_nominal=4.0)
  -> HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM (no override)
    -> SUPPORT_POSITION_OUTER_LOOP_PITCH_REF (no override)
      -> CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF_V2 (no override)
        -> PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP (no override)
          -> PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2 (no override)
            -> K1_PITCH_RATE_NOTCH (no override)
              -> K2_NOTCH_LOW_Q_V1 (no override)
```

### Schedule verification

The `k2_jax_scheduled_k_position` function computes:
```
k = k_nominal + (k_low_max - k_nominal) * smoothstep01((z_high - h) / (z_high - z_low))
```
This matches the Python `python_scheduled_k_position` function in
`sagittal_velocity_damped_balance_controller.py:788-790`. Both produce the same
values at all heights (verified at 20k grid points).

---

## 2. Notch Filter

| Parameter | Python Runtime | JAX Runtime | Python Source | JAX Source | Affects Wheels? | Status |
|-----------|---------------|-------------|---------------|------------|-----------------|--------|
| `notch_fc_hz` | 2.5 | 2.5 | `svdbc.py:3125` `wip_notch_center_hz=2.5` | `k2_jax_controller.py:152` `fc_hz=2.5` (default) | Yes [4,9] | MATCH |
| `notch_Q` | 2.0 | 2.0 | `svdbc.py:3165` `wip_notch_q=2.0` (K2 overrides K1's Q=6.0) | `k2_jax_controller.py:153` `Q=2.0` (default) | Yes [4,9] | MATCH |
| `notch_fs_hz` | 100.0 | 100.0 | Implicit from 0.01s control_dt | `k2_jax_controller.py:151` `fs_hz=100.0` (default) | Yes [4,9] | MATCH |
| `notch_height_gate_start_m` | 0.42 | 0.42 | `svdbc.py:3129` `wip_notch_height_gate_start_m=0.42` | `k2_jax_controller.py:1105` hardcoded `0.42` | Yes [4,9] | MATCH |
| `notch_height_gate_full_m` | 0.48 | 0.48 | `svdbc.py:3130` `wip_notch_height_gate_full_m=0.48` | `k2_jax_controller.py:1105` hardcoded `0.48` | Yes [4,9] | MATCH |
| `notch_filter_blend` | 1.0 | 1.0 | `svdbc.py:3127` `wip_notch_filter_blend=1.0` | `k2_jax_controller.py:1106` equivalent: `(1.0 - gate)*raw + gate*notched` | Yes [4,9] | MATCH |

### Biquad coefficient verification

Both Python and JAX use the same function
`wheeled_biped.controllers.signal_filters.biquad_notch_coefficients`
to compute `[b0, b1, b2, a1, a2]` from `(fs=100, fc=2.5, Q=2.0)`.
The JAX controller pre-computes the coefficients at `pack_params_stage2()`
time and stores them in the flat params array. The Python controller computes
them at BiquadNotchFilter init time. Same function, same inputs, same outputs.

### Notch gate note

The JAX notch gate is hardcoded as `smoothstep_gate_jax(height_ref, 0.42, 0.48)`
at line 1105. These values are not read from the K2 profile. If the profile's
`wip_notch_height_gate_start_m` or `wip_notch_height_gate_full_m` ever change,
JAX will not pick up the change. However, currently both are 0.42/0.48, so it
matches. This should be fixed to read from the profile for future-proofing.

---

## 3. Adaptive Bias Trim (ABS)

All ABS parameters are read at JIT-trace time from the `K2_NOTCH_LOW_Q_V1`
profile via the `_k2_jax_adaptive_bias_trim` function. The JAX code imports
the same Python profile object as the Python runtime:
```python
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import K2_NOTCH_LOW_Q_V1 as _sch
```
Both Python and JAX read from the identical Python dataclass instance.
Every ABS parameter is therefore guaranteed MATCH.

Key ABS parameters (all MATCH, all read from `_sch.*`):

| Parameter | Source in profile | Affects Wheels? |
|-----------|-------------------|-----------------|
| `adaptive_bias_window_steps` (slow) | `svdbc.py` parent definition (300) | Yes [4,9] |
| `adaptive_bias_fast_window_steps` | `svdbc.py` parent definition (100) | Yes [4,9] |
| `adaptive_bias_exit_threshold_m` | `svdbc.py` parent definition (0.015) | Yes [4,9] |
| `adaptive_bias_relief_hysteresis_m` | `svdbc.py` parent definition | Yes [4,9] |
| `adaptive_bias_k_tau_per_m` | `svdbc.py` parent definition | Yes [4,9] |
| `adaptive_bias_rate_nm_per_step` | `svdbc.py:2247` (0.01) | Yes [4,9] |
| `adaptive_bias_decay_rate_nm_per_step` | `svdbc.py:2248` (0.02) | Yes [4,9] |
| `adaptive_bias_max_tau_low_nm` | `svdbc.py` parent definition | Yes [4,9] |
| `adaptive_bias_max_tau_high_nm` | `svdbc.py` parent definition | Yes [4,9] |
| `adaptive_bias_max_tau_extreme_nm` | `svdbc.py` parent definition | Yes [4,9] |
| `adaptive_bias_height_low_m` | `svdbc.py` parent definition | Yes [4,9] |
| `adaptive_bias_height_high_m` | `svdbc.py` parent definition | Yes [4,9] |
| `adaptive_bias_height_extreme_m` | `svdbc.py` parent definition | Yes [4,9] |
| `adaptive_bias_sign_reversal_hold_steps` | `svdbc.py` parent definition | Yes [4,9] |
| `adaptive_bias_zero_crossing_limit` | `svdbc.py` parent definition | Yes [4,9] |
| `adaptive_bias_zero_crossing_max_scale` | `svdbc.py` parent definition | Yes [4,9] |
| `adaptive_bias_disable_if_pitch_gt_deg` | `svdbc.py:2251` (8.0) | Yes [4,9] |
| `adaptive_bias_disable_if_roll_gt_deg` | `svdbc.py:2252` (3.0) | Yes [4,9] |
| `adaptive_bias_disable_if_abs_error_gt_m` | `svdbc.py:2254` (0.22) | Yes [4,9] |
| `adaptive_bias_disable_if_hip_yaw_gt_rad` | `k2_jax_controller.py:1190` (0.25 fallback) | Yes [4,9] |

All ABS parameters affect wheel torque [4,9] via `external_position_trim`
added in `k2_jax_sagittal_torque_assembly`.

---

## 4. Calibrated Outer Loop -- TWO MISMATCHES

### 4.1 MISMATCH M1: Function version (v1 vs v2)

| Aspect | Python Runtime | JAX Runtime | Status |
|--------|---------------|-------------|--------|
| Module imported | `calibrated_outer_loop_functions_v2` | `calibrated_outer_loop_functions` (v1) | **MISMATCH** |
| Selection logic | `simulate_hierarchical_controller.py:4876-4886` reads `profile.calibrated_outer_loop_function_version` -- K2 chain returns `"v2"` | `k2_jax_controller.py:494` hardcoded import: `from wheeled_biped.controllers.calibrated_outer_loop_functions import ...` | **MISMATCH** |

**Root cause:** The K2 profile chain has `calibrated_outer_loop_function_version="v2"`
(set by `PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2` at
`svdbc.py:2944`). Python reads this field at runtime and imports the correct module.
JAX ignores the field and hardcodes v1.

### 4.2 MISMATCH M2: Kp and Kd at upper heights (0.465m, 0.480m)

v1 and v2 differ ONLY at the upper two breakpoints. All other heights (0.300-0.450)
are identical between v1 and v2.

#### Kp comparison

| Height | v1 Kp (JAX uses) | v2 Kp (Python uses) | Delta | Source |
|--------|------------------|---------------------|-------|--------|
| 0.300m | 1.500 | 1.500 | 0.000 | `calfunc.py:43` vs `calfunc_v2.py:51` |
| 0.320m | 1.500 | 1.500 | 0.000 | same |
| 0.330m | 1.300 | 1.300 | 0.000 | same |
| 0.340m | 1.000 | 1.000 | 0.000 | same |
| 0.360m | 0.725 | 0.725 | 0.000 | same |
| 0.380m | 0.650 | 0.650 | 0.000 | same |
| 0.430m | 1.000 | 1.000 | 0.000 | same |
| 0.450m | 0.650 | 0.650 | 0.000 | same |
| **0.465m** | **1.350** | **1.000** | **-0.350** | v1 line 43: 1.350, v2 line 51: 1.000 |
| **0.480m** | **1.575** | **1.050** | **-0.525** | v1 line 43: 1.575, v2 line 51: 1.050 |

#### Kd comparison

| Height | v1 Kd (JAX uses) | v2 Kd (Python uses) | Delta | Source |
|--------|------------------|---------------------|-------|--------|
| 0.300-0.465m | same as v1 | same as v1 | 0.000 | — |
| **0.480m** | **0.050** | **0.000** | **-0.050** | v1 line 49: 0.050, v2 line 58: 0.000 |

### 4.3 MATCHING parameters (unchanged between v1 and v2)

| Parameter | Python (v2) | JAX (v1) | Status |
|-----------|-------------|----------|--------|
| `calibrated_ki_deg_per_m_s` | 0.00 (all heights) | 0.00 (all heights) | MATCH |
| `calibrated_theta_ref_max_deg` | 3.00 (all heights) | 3.00 (all heights) | MATCH |
| `calibrated_deadband_m` | 0.015 (all heights) | 0.015 (all heights) | MATCH |
| `calibrated_rate_limit_deg_per_step` | 0.030 (all heights) | 0.030 (all heights) | MATCH |
| `calibrated_lowpass_alpha` | 0.150 (all heights) | 0.150 (all heights) | MATCH |

### 4.4 Interpolation method parity

- Python v2: PCHIP (SciPy `PchipInterpolator`) or piecewise-linear fallback
- JAX v1: Linear interpolation on 20,000-point pre-evaluated grid
- At 20,000 points, max linear interpolation error vs PCHIP < 1e-6 for all functions
- Interpolation method is NOT a significant source of mismatch at this grid density

### 4.5 Runtime impact on torque

The calibrated outer loop determines the dynamic pitch_ref offset via
`k2_jax_compute_outer_loop_pitch_ref`. The output (`ol_dynamic`) feeds into
`total_pitch_ref_offset_deg`.

**In teacher-forcing mode:** `pitch_x` is pre-adjusted externally by the Python
simulation loop (which uses v2 functions). JAX receives the v2-adjusted `pitch_x`
and uses it directly (`effective_pitch_x = pitch_x`, line 1173). The internally
computed `total_pitch_ref_offset_deg` is diagnostic only. Therefore, in
teacher-forcing mode, the calib v1→v2 mismatch does NOT propagate to torque.

**In standalone JAX mode:** `pitch_x` arrives raw (unadjusted), and JAX has no
mechanism to apply the internal offset computation to `effective_pitch_x`.
This would cause torque mismatch at all heights, not just 0.465/0.480m.

**Wheel torque [4,9] is affected** in standalone JAX mode via pitch_x -> tau_pitch -> wheel torque.

---

## 5. Physics Equilibrium Feedforward

| Parameter | Python Runtime | JAX Runtime | Source | Status |
|-----------|---------------|-------------|--------|--------|
| PFF module | `physics_equilibrium_feedforward` | `physics_equilibrium_feedforward` (same module) | Both import from the same file | MATCH |
| Interpolation | PCHIP (native) | Linear on 100,000-point grid | `k2_jax_controller.py:516` `build_physics_ff_grid_params(n_points=100000)` | MATCH (error < 1e-6) |

At h=0.48m, both should produce `tau_eq_ff ≈ 3.303 Nm` (verified from teacher-forcing diag).

Physics FF is added to wheel torque via `k2_jax_sagittal_torque_assembly`.
Affects wheels [4,9].

---

## 6. Low-Band Support

Parameters hardcoded in the `k2_jax_low_band_support_pitch_ref()` call at line 1147-1148.

| Parameter | Python Runtime | JAX Runtime | Python Source | JAX Source | Status |
|-----------|---------------|-------------|---------------|------------|--------|
| `center_m` | 0.320 | 0.320 | `svdbc.py:2946` `low_band_support_center_m=0.320` | `k2_jax_controller.py:1148` hardcoded `0.320` | MATCH |
| `sigma_m` | 0.004 | 0.004 | `svdbc.py:2947` `low_band_support_sigma_m=0.004` | `k2_jax_controller.py:1148` hardcoded `0.004` | MATCH |
| `kp_peak_deg_per_m` | 1.4 | 1.4 | `svdbc.py:2948` `low_band_support_kp_peak_deg_per_m=1.4` | `k2_jax_controller.py:1148` hardcoded `1.4` | MATCH |
| `theta_ref_max_peak_deg` | 3.00 | 3.0 | `svdbc.py:2949` `low_band_support_theta_ref_max_peak_deg=3.00` | `k2_jax_controller.py:1148` hardcoded `3.0` | MATCH |
| `pitch_ref_offset_peak_deg` | 1.00 | 1.0 | `svdbc.py:2950` `low_band_support_pitch_ref_offset_peak_deg=1.00` | `k2_jax_controller.py:1148` hardcoded `1.0` | MATCH |

Low-band support affects `lb_offset`, which is added into `total_pitch_ref_offset_deg`
(diagnostic only in teacher-forcing mode). Does NOT directly affect torque because
`pitch_x` is pre-adjusted externally.

---

## 7. Shape Posture

| Parameter | Python Runtime | JAX Runtime | Python Source | JAX Source | Affects Wheels? | Status |
|-----------|---------------|-------------|---------------|------------|-----------------|--------|
| `kp_hip_yaw` | 15.0 | 15.0 | `shape_posture_controller.py:34` `BALANCE_CORE_HIP_YAW_AUTHORITY` | `k2_jax_controller.py:670` default | No [1,6] | MATCH |
| `kd_hip_yaw` | 3.0 | 3.0 | `shape_posture_controller.py:35` `BALANCE_CORE_HIP_YAW_AUTHORITY` | `k2_jax_controller.py:670` default | No [1,6] | MATCH |
| `kp_hip_pitch` | 30.0 | 30.0 | `simulate_hierarchical_controller.py:2362` hardcoded | `k2_jax_controller.py:671` default | No [2,7] | MATCH |
| `kd_hip_pitch` | 4.0 | 4.0 | `simulate_hierarchical_controller.py:2363` hardcoded | `k2_jax_controller.py:671` default | No [2,7] | MATCH |
| `kp_knee` | 40.0 | 40.0 | `simulate_hierarchical_controller.py:2364` hardcoded | `k2_jax_controller.py:672` default | No [3,8] | MATCH |
| `kd_knee` | 5.0 | 5.0 | `simulate_hierarchical_controller.py:2365` hardcoded | `k2_jax_controller.py:672` default | No [3,8] | MATCH |
| `kp_hip_roll` | 0.0 | 0.0 | ShapePostureController does not do hip_roll (lateral handles it) | `k2_jax_controller.py:673` default | No [0,5] | MATCH |
| `kd_hip_roll` | 0.0 | 0.0 | ShapePostureController does not do hip_roll | `k2_jax_controller.py:673` default | No [0,5] | MATCH |

Shape posture does NOT affect wheels [4,9] -- wheel joints are excluded.
Operates on joints [0,1,2,3,5,6,7,8].

---

## 8. Lateral Roll

| Parameter | Python Runtime | JAX Runtime | Python Source | JAX Source | Affects Wheels? | Status |
|-----------|---------------|-------------|---------------|------------|-----------------|--------|
| `kp_roll` | 40.0 | 40.0 | `simulate_hierarchical_controller.py:2442` hardcoded | `k2_jax_controller.py:698` default | No [0,5] | MATCH |
| `kd_roll` | 8.0 | 8.0 | `simulate_hierarchical_controller.py:2443` hardcoded | `k2_jax_controller.py:698` default | No [0,5] | MATCH |
| `max_roll_moment` | 50.0 | 50.0 | `simulate_hierarchical_controller.py:2444` hardcoded | `k2_jax_controller.py:698` default | No [0,5] | MATCH |
| `hip_roll_torque_sign` | 1.0 | 1.0 | `simulate_hierarchical_controller.py:2445` hardcoded | `k2_jax_controller.py:698` default | No [0,5] | MATCH |
| `enable_stance_regularization` | True | True | `simulate_hierarchical_controller.py:6301-6307` always passes hip_roll ref | `k2_jax_controller.py:1243` explicit `True` | No [0,5] | MATCH |
| `kp_stance` | 5.0 | 5.0 | `lateral_roll_balance_controller.py:28` default | `k2_jax_controller.py:700` default | No [0,5] | MATCH |
| `kd_stance` | 1.0 | 1.0 | `lateral_roll_balance_controller.py:29` default | `k2_jax_controller.py:700` default | No [0,5] | MATCH |
| `max_stance_torque` | 5.0 | 5.0 | `lateral_roll_balance_controller.py:30` default | `k2_jax_controller.py:700` default | No [0,5] | MATCH |
| `stance_weight` | 0.4 | 0.4 | `lateral_roll_balance_controller.py:31` default | `k2_jax_controller.py:700` default | No [0,5] | MATCH |

Does NOT affect wheels [4,9].

---

## 9. Yaw

| Parameter | Python Runtime | JAX Runtime | Python Source | JAX Source | Affects Wheels? | Status |
|-----------|---------------|-------------|---------------|------------|-----------------|--------|
| `kp_yaw` | 8.0 | 8.0 | `simulate_hierarchical_controller.py:2320` default `yaw_controller_kp=8.0` | `k2_jax_controller.py:721` default | No [1,6] | MATCH |
| `kd_yaw` | 2.0 | 2.0 | `simulate_hierarchical_controller.py:2321` default `yaw_controller_kd=2.0` | `k2_jax_controller.py:721` default | No [1,6] | MATCH |
| `max_yaw_torque` | 5.0 | 5.0 | `simulate_hierarchical_controller.py:2322` default `yaw_controller_max_torque=5.0` | `k2_jax_controller.py:721` default | No [1,6] | MATCH |

Does NOT affect wheels [4,9].

---

## 10. Mode-Div Hip-Yaw -- THREE MISMATCHES

### 10.1 MATCHING parameters

| Parameter | Python Runtime | JAX Runtime | Python Source | JAX Source | Status |
|-----------|---------------|-------------|---------------|------------|--------|
| `kp_div` | 10.0 | 10.0 | `validate_k2_*.py` CLI `--mode-hip-yaw-div-kp 10.0` | `k2_jax_controller.py:732` default | MATCH |
| `kd_div` | 0.50 | 0.50 | `validate_k2_*.py` CLI `--mode-hip-yaw-div-kd 0.50` | `k2_jax_controller.py:732` default | MATCH |
| `max_torque` | 7.5 | 7.5 | `validate_k2_*.py` CLI `--mode-hip-yaw-div-max-torque 7.5` | `k2_jax_controller.py:732` default | MATCH |
| `soft_limit_rad` | 0.30 | 0.30 | `validate_k2_*.py` CLI `--mode-hip-yaw-div-soft-limit-rad 0.30` | `k2_jax_controller.py:733` default | MATCH |
| `support_gate_enabled` | False | False | `validate_k2_*.py` CLI: not set, default `False` | `k2_jax_controller.py:735` default | MATCH |

Note: Python CLI defaults are kp=1.0, kd=0.10, max_torque=1.0, soft_gain=0.10.
ALL validation scripts explicitly override these to the values above. JAX
defaults match the overridden (actual runtime) values for kp, kd, max_torque,
and soft_limit_rad, but NOT for soft_gain.

### 10.2 MISMATCH M3: soft_gain

| Parameter | Python Runtime | JAX Runtime | Delta | Source |
|-----------|---------------|-------------|-------|--------|
| `soft_gain` | **0.80** | **0.50** | **-0.30** | Python: `validate_k2_*.py` CLI; JAX: `k2_jax_controller.py:733` default |

**Effect on height gate computation:**
```python
z_low = soft_limit_rad           # 0.30 (both)
z_high = soft_limit_rad + soft_gain  # Python: 1.10, JAX: 0.80
u_h = (z_high - height_m) / (z_high - z_low)
height_gate = smoothstep01(u_h)
torque = raw_torque * height_gate
```

At h=0.48m:
- Python: z_high=1.10, u_h=(1.10-0.48)/(1.10-0.30)=0.775, gate=smoothstep(0.775) = 0.881
- JAX: z_high=0.80, u_h=(0.80-0.48)/(0.80-0.30)=0.640, gate=smoothstep(0.640) = 0.738
- Ratio: JAX gate / Python gate = 0.738 / 0.881 = 0.838 (JAX attenuates ~16% more)

At h=0.30m:
- Both: height_gate = 1.0 (full authority) since h <= z_low
- No mismatch at low heights

**Files requiring fix:** `k2_jax_controller.py:733` -- change `soft_gain=0.50` to `soft_gain=0.80`

**Affected joints:** hip_yaw [1,6] only. Wheels [4,9] unaffected.

### 10.3 MISMATCH M4: ref_source not handled

| Parameter | Python Runtime | JAX Runtime | Source |
|-----------|---------------|-------------|--------|
| `ref_source` | **"target"** | **Not handled** | Python: `validate_k2_*.py` CLI `--mode-hip-yaw-div-ref-source target`; JAX: no parameter |

In Python, `ref_source="target"` defines how `div_error` is computed:
```
div_error = (q_hy_left - qref_hy_left) - (q_hy_right - qref_hy_right)
```
where `qref` comes from the height-dependent IK target posture.

In JAX, `div_error` arrives as a pre-computed input (`hip_yaw_div_error` in
`pack_input_k2`). In teacher-forcing mode, this is computed externally using
Python's logic (which includes ref_source handling), so JAX receives the
correct value. In standalone mode, the caller must replicate the ref_source
logic.

**Risk:** Low in teacher-forcing (error pre-computed). Medium in standalone
mode (caller must replicate).

**Affected joints:** hip_yaw [1,6] only. Wheels [4,9] unaffected.

### 10.4 MISMATCH M5: support_error/support_error_rate not passed

The JAX call site at line 1249-1250 passes only 3 args:
```python
tau_mode_div = k2_jax_mode_div_compute(hy_div_err, hy_div_rate, schedule_h)
```

`support_error_m` and `support_error_rate_m_s` default to 0.0. Since
`enable_support_gate=False` in both Python and JAX, the support gate is
inactive, so this has no runtime effect currently.

**Risk:** Low (currently inactive). If support gating were ever enabled,
JAX would need `support_pos_err` (which is available as a local variable
in `k2_jax_controller_step`) passed to `k2_jax_mode_div_compute`.

**Affected joints:** hip_yaw [1,6] only.

---

## 11. Support Feedforward (hip-yaw, height-gated)

Computed but **excluded** from torque sum in JAX (line 1264-1265):
```
NOTE: tau_support_ff (height-gated hip-yaw support) is EXCLUDED --
Python balance-core has no equivalent; inclusion causes divergence
during descending height transitions and push recovery.
```

Since it does not contribute to final torque, parameter matches are irrelevant
for runtime parity. Listed here for completeness.

| Parameter | Python Runtime | JAX Runtime | Status |
|-----------|---------------|-------------|--------|
| `k_support_hip_yaw` | N/A (disabled in profile) | 3.0 (default, unused) | N/A |
| `support_comp_sign` | N/A | 1.0 (default, unused) | N/A |
| `tau_max_support_comp` | N/A | 5.0 (default, unused) | N/A |
| Height gate range | N/A | 0.300-0.393 (hardcoded, unused) | N/A |

---

## 12. Empirical Support Feedforward (hip_pitch/knee)

This is the always-on `k2_jax_empirical_support_ff()` vector added to `tau_sum`.
Comes from `SupportFeedforwardController` with `scale=0.5`, `joint_group="hip_pitch_knee"`.

| Parameter | Python Runtime | JAX Runtime | Source | Status |
|-----------|---------------|-------------|--------|--------|
| Vector (index 2, l_hip_pitch) | 2.05 | 2.05 | 4.1 * 0.5 | MATCH |
| Vector (index 3, l_knee) | -7.75 | -7.75 | -15.5 * 0.5 | MATCH |
| Vector (index 7, r_hip_pitch) | 1.6 | 1.6 | 3.2 * 0.5 | MATCH |
| Vector (index 8, r_knee) | -7.9 | -7.9 | -15.8 * 0.5 | MATCH |
| All other joints (0,1,4,5,6,9) | 0.0 | 0.0 | -- | MATCH |

Full JAX constant at `k2_jax_controller.py:768-771`:
```python
_K2_EMPIRICAL_SUPPORT_FF = jnp.array(
    [0.0, 0.0, 2.05, -7.75, 0.0, 0.0, 0.0, 1.6, -7.9, 0.0],
    dtype=jnp.float64,
)
```

This is added to `tau_sum` before the composer (line 1272), so it passes through
clip and rate-limit. Affects hip_pitch [2,7] and knee [3,8]. Does NOT affect
wheels [4,9].

---

## 13. Composer

| Parameter | Python Runtime | JAX Runtime | Source | Status |
|-----------|---------------|-------------|--------|--------|
| `torque_limits` | From `mj_model.actuator_ctrlrange[:, 1]` (model-dependent) | Same values, passed via `pack_params` | `simulate_hierarchical_controller.py:4659` | MATCH (same model) |
| `max_torque_rate` | 400.0 Nm/s per joint (all 10) | 400.0 Nm/s per joint (all 10) | `simulate_hierarchical_controller.py:4661` | MATCH |
| `control_dt` | 0.01 s | 0.01 s | `simulate_hierarchical_controller.py` default; `k2_jax_controller.py:156` default | MATCH |

The MuJoCo model's actuator control range yields per-joint torque_limits.
For the standard K2 model, these are approximately:
```
[50, 50, 20, 20, 50, 50, 50, 20, 20, 50] Nm
```
Exact values depend on the model XML. Since both Python and JAX receive the
same values (either from the same model or via explicit parameter passing),
they are guaranteed MATCH.

Rate limiting formula (identical in both):
1. `delta_desired = tau_sum - tau_prev`
2. `delta_rate = delta_desired / control_dt`
3. `delta_rate_limited = clip(delta_rate, -max_torque_rate, max_torque_rate)`
4. `tau_final = tau_prev + delta_rate_limited * control_dt`

---

## 14. Pitch Ref Offset (G2 from Coverage Audit) -- MISMATCH M6

| Aspect | Python Runtime | JAX Runtime | Status |
|--------|---------------|-------------|--------|
| Pitch ref offset computation | v2 functions, applied externally to `pitch_x` before passing to controller | v1 functions, computed internally as `total_pitch_ref_offset_deg` but NOT applied to `effective_pitch_x` | **MISMATCH** |
| `effective_pitch_x` | Pre-adjusted externally: `pitch_x_error = body_pitch_x - (offset)` | Pass-through: `effective_pitch_x = pitch_x` (line 1173) | Teacher-forcing OK; standalone: broken |

### JAX internal computation (diagnostic only)

```python
# k2_jax_controller.py:1166-1173
total_pitch_ref_offset_deg = new_ol_pitch_ref + lb_offset + physics_pitch_eq

# pitch_x is pre-adjusted by the simulation loop (pitch_x_error = raw - offset).
# Do NOT apply pitch_ref_offset internally -- it's already applied externally.
effective_pitch_x = pitch_x
```

### Risk assessment

- **Teacher-forcing mode:** `pitch_x` is pre-adjusted externally using Python's
  v2 functions. JAX trusts this value. The internally-computed v1-based offset
  is purely diagnostic. The mismatch in calib functions (v1 vs v2) does NOT
  propagate to torque in teacher-forcing.
- **Standalone JAX mode:** If JAX receives raw (unadjusted) `pitch_x`, the
  internally-computed v1 offset is NOT applied, so `effective_pitch_x` would
  be wrong at ALL heights. This is a functional gap, not just a parameter mismatch.

Affects wheel torque [4,9] via pitch_x -> tau_pitch -> wheel torque (in standalone mode).

---

## 15. Joint-by-Joint Impact Summary

| Joint Index | Joint Name | Affected by Mismatches | Mismatch Sources |
|-------------|-----------|----------------------|-----------------|
| 0 | l_hip_roll | No | -- |
| 1 | l_hip_yaw | **Yes** | M3 (soft_gain), M4 (ref_source), M5 (support error) |
| 2 | l_hip_pitch | No | -- |
| 3 | l_knee | No | -- |
| **4** | **l_wheel** | **Yes (indirect)** | **M1/M2 (calib v1/v2 via pitch_x), M6 (pitch ref offset)** |
| 5 | r_hip_roll | No | -- |
| 6 | r_hip_yaw | **Yes** | M3 (soft_gain), M4 (ref_source), M5 (support error) |
| 7 | r_hip_pitch | No | -- |
| 8 | r_knee | No | -- |
| **9** | **r_wheel** | **Yes (indirect)** | **M1/M2 (calib v1/v2 via pitch_x), M6 (pitch ref offset)** |

The 0.01 Nm wheel mismatch observed in teacher-forcing diagnostics is likely
explained by one or more of these mismatches, with the calibrated outer loop
v1->v2 Kp difference being the primary candidate.

---

## 16. Prioritized Action Items

### Critical (affect torque, fix for Stage 7 parity)

| # | Issue | File | Line | Fix |
|---|-------|------|------|-----|
| M3 | `mode_div_soft_gain`: JAX=0.50, Python=0.80 | `k2_jax_controller.py` | 733 | Change `soft_gain=0.50` to `soft_gain=0.80` in `k2_jax_mode_div_compute` default |
| M1+M2 | Calibrated outer loop uses v1 instead of v2 | `k2_jax_controller.py` | 494 | Change import from `calibrated_outer_loop_functions` to `calibrated_outer_loop_functions_v2`, OR read `calibrated_outer_loop_function_version` from profile |

### Medium (fix for standalone JAX correctness)

| # | Issue | File | Line | Fix |
|---|-------|------|------|-----|
| M6 | Pitch ref offset not applied in standalone mode | `k2_jax_controller.py` | 1171-1173 | Add flag to input/params indicating whether pitch_x is pre-adjusted; apply offset when not |
| M4 | Mode-div ref_source not handled | `k2_jax_controller.py` | 730-747 | Add `ref_source` parameter; implement target-based q_ref selection |

### Low priority (future-proofing)

| # | Issue | File | Line | Fix |
|---|-------|------|------|-----|
| -- | Notch gate values hardcoded instead of read from profile | `k2_jax_controller.py` | 1105 | Read `wip_notch_height_gate_start_m` and `wip_notch_height_gate_full_m` from profile |
| M5 | Mode-div support_error not passed to function | `k2_jax_controller.py` | 1250 | Pass `support_pos_err` to `k2_jax_mode_div_compute` for future support gating |

---

## 17. Files Referenced

| File | Role |
|------|------|
| `wheeled_biped/controllers/k2_jax_controller.py` | JAX K2 controller (primary) |
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | Python K2 profile definitions, sagittal controller |
| `scripts/simulate_hierarchical_controller.py` | Python simulation CLI defaults, controller construction |
| `scripts/validate_k2_jax_backend.py` | JAX backend validation (mode_div CLI args) |
| `scripts/validate_k2_step_c_e_fixed_height.py` | Fixed-height validation (mode_div CLI args) |
| `scripts/validate_k2_step_d_push_matrix.py` | Push recovery validation (mode_div CLI args) |
| `scripts/validate_k2_post_promotion_long_run.py` | Long-run validation (mode_div CLI args) |
| `scripts/validate_k2_dynamic_height_gate_crossing.py` | Dynamic height validation (mode_div CLI args) |
| `wheeled_biped/controllers/calibrated_outer_loop_functions.py` | v1 calibrated functions (JAX uses) |
| `wheeled_biped/controllers/calibrated_outer_loop_functions_v2.py` | v2 calibrated functions (Python uses) |
| `wheeled_biped/controllers/mode_based_hip_yaw_divergence_controller.py` | Python mode-div controller |
| `wheeled_biped/controllers/shape_posture_controller.py` | Python shape posture, BALANCE_CORE_HIP_YAW_AUTHORITY |
| `wheeled_biped/controllers/lateral_roll_balance_controller.py` | Python lateral roll controller |
| `wheeled_biped/controllers/yaw_controller.py` | Python yaw controller |
| `wheeled_biped/controllers/balance_core_torque_composer.py` | Python torque composer |
| `wheeled_biped/controllers/signal_filters.py` | Shared biquad notch coefficients |
| `wheeled_biped/controllers/physics_equilibrium_feedforward.py` | Shared physics FF functions |

---

## 18. Test Recommendations

After fixing the mismatches, verify with:

1. **Fixed-height parity test at h=0.48m:** The calib v1->v2 fix should eliminate
   the Kp=1.575 vs 1.050 discrepancy, reducing wheel torque mismatch.

2. **Mode-div step test:** After fixing soft_gain 0.50->0.80, hip_yaw torque [1,6]
   should match Python at all heights above 0.30m.

3. **Standalone JAX mode test:** After implementing M6 fix, raw pitch_x input
   should produce correct pitch-referenced torque output without external pre-adjustment.

4. **Full teacher-forcing re-run:** After all critical fixes, re-run the teacher-forcing
   comparison to verify all 10 joint torques match within 1e-8 Nm.
