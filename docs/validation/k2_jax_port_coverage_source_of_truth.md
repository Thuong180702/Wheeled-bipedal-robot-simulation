# K2 JAX Port Coverage Audit — Phase 0: Python K2 Source of Truth

**Date:** 2026-06-27
**Status:** Complete
**Audit scope:** Port coverage only — NOT numerical parity

---

## 1. Profile Identity

### Active K2 Profile
```python
K2_NOTCH_LOW_Q_V1 = replace(
    K1_PITCH_RATE_NOTCH,
    profile_name="k2_notch_low_q_v1",
    wip_notch_q=2.0,
)
```

**File:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py:3162`

### Full Inheritance Chain

```
SagittalAuthoritySchedule (base defaults)
  → T6J_CENTERING_BIAS_TRIM
    → PITCH_EQUILIBRIUM_TRIM
      → HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM
        → SUPPORT_POSITION_OUTER_LOOP_PITCH_REF
          → CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF
            → CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF_V2
              → PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP
                → PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2
                  → K1_PITCH_RATE_NOTCH (adds notch filter, Q=6.0)
                    → K2_NOTCH_LOW_Q_V1 (changes Q=2.0)
```

### Key K2-Overridden Parameters (vs base defaults)

| Parameter | Base Default | K2 Value | Status |
|-----------|-------------|----------|--------|
| `wip_notch_q` | 6.0 (K1) | 2.0 | **K2 override** |
| `wip_notch_center_hz` | — | 2.5 | From K1 |
| `wip_notch_filter_blend` | — | 1.0 | From K1 |
| `enable_wip_notch_filter` | False | True | From K1 |
| `wip_notch_target_signal` | — | "pitch_rate" | From K1 |
| `wip_notch_gate_enabled` | — | True | From K1 |
| `wip_notch_height_gate_start_m` | — | 0.42 | From K1 |
| `wip_notch_height_gate_full_m` | — | 0.48 | From K1 |
| `adaptive_bias_trim_enabled` | False | True | From T6J |
| `calibrated_outer_loop_function_version` | — | "v2" | From PHYSICS_FF |
| `low_band_support_outer_loop_enabled` | — | True | From PHYSICS_FF_LOW_BAND_V2 |
| `low_band_support_center_m` | — | 0.320 | From PHYSICS_FF_LOW_BAND_V2 |
| `low_band_support_sigma_m` | — | 0.004 | From PHYSICS_FF_LOW_BAND_V2 |
| `low_band_support_kp_peak_deg_per_m` | — | 1.4 | From PHYSICS_FF_LOW_BAND_V2 |
| `continuous_max_position_tau` | False | True | From T6J |

### Controller Mode
```
balance-core
```

Using `SagittalVelocityDampedBalanceController` with authority schedule `K2_NOTCH_LOW_Q_V1`.

---

## 2. Source Files and Classes

### Primary Controller Files

| File | Key Classes/Functions | Role |
|------|----------------------|------|
| `scripts/simulate_hierarchical_controller.py` | `simulate()`, `build_balance_core_controllers()` | Simulation loop, controller wiring |
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | `SagittalAuthoritySchedule`, `SagittalVelocityDampedBalanceController` | Sagittal wheel balance (K2 active) |
| `wheeled_biped/controllers/balance_core_torque_composer.py` | `BalanceCoreTorqueComposer` | Torque summation, clipping, rate limiting |
| `wheeled_biped/controllers/shape_posture_controller.py` | `ShapePostureController` | Hip-yaw/knee/hip-pitch PD posture |
| `wheeled_biped/controllers/lateral_roll_balance_controller.py` | `LateralRollBalanceController` | Roll stabilization |
| `wheeled_biped/controllers/yaw_controller.py` | `YawController` | Body yaw control |
| `wheeled_biped/controllers/mode_hip_yaw_divergence_controller.py` | `ModeBasedHipYawDivergenceController` | Mode-div hip-yaw (opt-in, CLI) |
| `wheeled_biped/controllers/support_feedforward_controller.py` | `SupportFeedforwardController` | Empirical support FF (hip_pitch/knee) |
| `wheeled_biped/controllers/signal_filters.py` | `BiquadNotchFilter`, `FirstOrderLowPassFilter`, `smoothstep_gate` | Signal processing |
| `wheeled_biped/controllers/calibrated_outer_loop_functions_v2.py` | PCHIP interpolation functions | Calibrated outer-loop parameter lookup |
| `wheeled_biped/controllers/physics_equilibrium_feedforward.py` | PCHIP interpolation functions | Physics-based equilibrium FF |
| `wheeled_biped/controllers/balance_core_types.py` | `BalanceCoreTorqueResult`, action constants | Type definitions |
| `wheeled_biped/controllers/sagittal_balance_state.py` | `compute_support_center_xy()`, projection functions | Support-center geometry |
| `wheeled_biped/controllers/contact_supervisor.py` | `ContactSupervisor` | Contact detection |
| `wheeled_biped/controllers/k2_jax_controller.py` | All `k2_jax_*` functions | **JAX port** (coverage target) |

---

## 3. Simulation Loop Call Order (Per Step)

### Python K2 Balance-Core Path (controller_mode = "balance-core")

```
STEP START
  │
  ├─ 1. mj_step1() + mj_step2()                         # Physics step
  │
  ├─ 2. CentroidalStateEstimator.update()               # State estimation
  │
  ├─ 3. CapturePointEstimator.update()                  # Capture point
  │
  ├─ 4. ContactSupervisor.update()                      # Contact flags
  │
  ├─ 5. ShapePostureController.compute()               # tau_shape_posture [all 10 joints]
  │     └─ inputs: q_ref, joint_pos, joint_vel, support_pos_error, target_com_height
  │
  ├─ 6. SupportFeedforwardController.compute()          # tau_support_feedforward [hip_pitch/knee 2,3,7,8]
  │     └─ Fixed vector × 0.5 scale
  │
  ├─ 7. SagittalVelocityDampedBalanceController.compute()  # tau_sagittal_wheel_balance [wheels 4,9]
  │     ├─ inputs: pitch_x_error, pitch_rate, sagittal_velocity, wheel_vel_L/R,
  │     │          sagittal_position_error, com_z, roll_y, contact_valid, height_variant, height_ref
  │     │
  │     ├─ 7a. Height scheduling (filtered_com_z, k_position, kd_pitch, k_wheel_velocity, max_position_tau)
  │     ├─ 7b. Pitch reference offset (physics FF + outer loop + low-band)
  │     ├─ 7c. Notch filter update (pitch_rate → pitch_rate_eff)
  │     ├─ 7d. Adaptive bias trim update (sliding window on support position error)
  │     ├─ 7e. Torque computation:
  │     │      tau_pitch = Kp * pitch_x_error (clamped)
  │     │      tau_pitch_rate = Kd * pitch_rate_eff
  │     │      tau_sagittal_velocity = -K_vel * sagittal_velocity
  │     │      tau_support_velocity = -K_support_vel * support_velocity
  │     │      tau_position = -K_pos * sagittal_position_error (+ integral + ABS trim)
  │     │      tau_wheel_vel_L/R = -K_wheel_vel * wheel_vel_L/R
  │     │      tau_common = wheel_torque_sign * sum(above)
  │     │      tau_L = tau_common + tau_wheel_vel_L
  │     │      tau_R = tau_common + tau_wheel_vel_R
  │     └─ Output: tau on [4,9]; full diagnostics dict
  │
  ├─ 8. LateralRollBalanceController.compute()          # tau_lateral_roll [hip_roll 0,5]
  │     └─ inputs: roll_y, roll_rate, hip_roll_pos/vel/ref
  │
  ├─ 9. YawController.compute()                         # tau_yaw [hip_yaw 1,6]
  │     └─ Added to tau_shape_posture → tau_shape_posture_with_yaw
  │
  ├─ 10. ModeBasedHipYawDivergenceController.compute()  # tau_mode_div [hip_yaw 1,6]
  │      └─ Opt-in via --enable-mode-hip-yaw-divergence
  │      └─ Added to tau_shape_posture_with_yaw
  │
  ├─ 11. BalanceCoreTorqueComposer.compose()
  │      ├─ tau_total_raw = sum of 4 sources + yaw + mode_div
  │      ├─ tau_total_clipped = clip(tau_total_raw, ±torque_limit)
  │      └─ tau_final = rate_limit(tau_total_clipped, tau_prev)
  │
  ├─ 12. [if JAX backend] JAX controller step runs in parallel
  │      └─ Overrides tau_smooth with JAX output
  │
  ├─ 13. [if wheel_yaw_stabilizer] Post-composer wheel yaw addition
  │
  ├─ 14. mj_data.ctrl[:] = tau_smooth                   # Final torque assignment
  │
  └─ 15. Telemetry collection
```

---

## 4. Final Torque Path into mj_data.ctrl

```
mj_data.ctrl[:] = tau_smooth

where tau_smooth comes from:

PYTHON PATH:
  tau_smooth = composer.compose(tau_shape_posture_with_yaw,
                                 tau_support_feedforward,
                                 tau_sagittal_wheel_balance,
                                 tau_lateral_roll_balance,
                                 tau_prev).tau_final
  (+ optional wheel_yaw_stabilizer post-composer)

JAX PATH:
  tau_smooth = jax_step_fn(jax_state, jax_input, jax_params)[0]
```

---

## 5. Backend Behavior

### Backend Selection

Controlled by `--controller-backend` CLI flag:
- `python` (default): Python-only path. JAX not initialized.
- `jax`: Python path runs for telemetry, but torque is overridden by JAX output.
- `both`: Both paths run. Teacher-forcing comparison printed for first 20 steps. Python torque used for physics (NOT JAX).

### Key: In `both` mode
- Physics uses Python torque (not JAX).
- Teacher-forcing comparison prints `max_tau_diff` per step.
- This is the mode used for parity testing.

### Key: In `jax` mode
- Python controller still runs (for telemetry).
- But `tau_smooth`, `tau_total_clipped`, and `tau_prev` are replaced by JAX output.
- The robot is physically controlled by JAX.

---

## 6. K2 Profile — Complete Active Feature List

Based on `K2_NOTCH_LOW_Q_V1` = `K1_PITCH_RATE_NOTCH` with `wip_notch_q=2.0`:

### Active Features (inherited from chain)

1. **WIP Notch Filter** (`enable_wip_notch_filter=True`)
   - Target signal: `pitch_rate`
   - Center frequency: 2.5 Hz
   - Q factor: 2.0 (K2 override)
   - Filter type: `biquad_notch`
   - Blend: 1.0 (full notch)
   - Gate: height 0.42→0.48 m smoothstep

2. **Physics Equilibrium Feedforward** (from PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP)
   - PCHIP-interpolated per-height equilibrium torque
   - Per-wheel tau_eq_ff(h) used as pitch_ref equivalent
   - Replaces empirical pitch_ref_offset schedule

3. **Calibrated Outer Loop v2** (`calibrated_outer_loop_function_version="v2"`)
   - PD(+I) dynamic pitch_ref offset on support-position error
   - PCHIP-interpolated gains: Kp(h), Kd(h), theta_max(h), deadband(h)
   - Rate-limited + low-pass filtered

4. **Low-Band Support Outer Loop** (`low_band_support_outer_loop_enabled=True`)
   - Gaussian height gate centered at 0.320 m
   - Additional Kp=1.4 deg/m on support error
   - Pitch ref offset: 1.0 deg peak at center height

5. **Adaptive Bias Trim (ABS)** (`adaptive_bias_trim_enabled=True`)
   - Sliding window (300-step slow, 100-step fast)
   - Zero-crossing guard
   - Height-scheduled max trim
   - Asymmetric rate limiting
   - Safety gates: pitch, roll, contact, hip_yaw, abs_error

6. **Continuous Max Position Tau** (`continuous_max_position_tau=True`)
   - Smoothstep from 4.0 Nm (at 0.393 m) to 6.0 Nm (at 0.300 m)

7. **Sagittal Torque Terms (all active in K2):**
   - tau_pitch: Kp=50.0 * pitch_x_error, no cap (pitch_tau_cap=0.0)
   - tau_pitch_rate: Kd=10.0 * pitch_rate_eff (after notch)
   - tau_sagittal_velocity: K_vel=15.0 * sagittal_velocity
   - tau_support_velocity: 0.0 (K_support_vel=0.0 in K2)
   - tau_position: K_pos=40.0 * sagittal_position_error (+ ABS trim)
   - tau_wheel_vel: K_wheel_vel=0.5 * wheel_vel
   - tau_cp: 0.0 (Kp_cp=0.0 — disabled)
   - tau_com_vy: Kd=5.0 * sagittal_velocity

### Inactive Features (confirmed zero/diabled in K2)

1. **Hysteresis blending** — disabled (K2 has no hysteresis gate)
2. **APC/APCR transient modes** — disabled (not in K2 profile chain)
3. **Position integral** — disabled (`enable_position_integral=False`)
4. **Torque budget aware position** — disabled
5. **Pitch-aware position scaling** — disabled
6. **Recenter/centering bias** — disabled (`zc_replace_adaptive=False`)
7. **Continuous k_position scheduling** — disabled (`continuous_k_position=False`)
8. **Continuous k_wheel_velocity** — disabled (`continuous_k_wheel_velocity=False`)
9. **Continuous kd_pitch** — disabled (`continuous_kd_pitch=False`)
10. **Continuous k_velocity** — disabled (`continuous_k_velocity=False`)
11. **Capture gate** — disabled
12. **ZC/EZC replace adaptive** — disabled
13. **Bias cancel** — disabled
14. **L_feedback (coordinated low-freq)** — disabled
15. **Phase lead** — disabled
16. **Pitch rate consistency estimator** — diagnostic only, not in control path
17. **Position authority scaling** — disabled (position_authority_scale=1.0)
18. **Pitch rate boost** — disabled (pitch_rate_boost_factor=1.0)
19. **Transient detect/capture modes** — disabled (transient_mode not in T1-T4 for K2)
20. **Boundary yaw-position fix** — disabled (not active for K2 default variants)
21. **Wheel yaw stabilizer** — disabled (only active for M-family profiles)

---

## 7. Joint Index Mapping

```
0: l_hip_roll    5: r_hip_roll
1: l_hip_yaw     6: r_hip_yaw
2: l_hip_pitch   7: r_hip_pitch
3: l_knee        8: r_knee
4: l_wheel       9: r_wheel
```

### Torque Source Ownership

| Indices | Joint Group | Primary Controller | Torque Type |
|---------|-------------|-------------------|-------------|
| 0, 5 | Hip Roll | LateralRollBalanceController | Position PD + stance |
| 1, 6 | Hip Yaw | ShapePosture + Yaw + ModeDiv | Position PD |
| 2, 7 | Hip Pitch | ShapePosture + SupportFF | Position PD + fixed FF |
| 3, 8 | Knee | ShapePosture + SupportFF | Position PD + fixed FF |
| 4, 9 | Wheel | SagittalVelocityDampedBalance | Velocity PD + pitch balance |

---

## 8. External/Precomputed Mechanisms (Python-only by design)

These are computed in Python and passed as inputs to JAX. They are EXTERNAL_PRECOMPUTED by design:

1. **PCHIP grid evaluation** — `build_calibrated_grid_params()` and `build_physics_ff_grid_params()` precompute fine grids (20k/100k points) in Python once at module load. JAX reads from these constant grids via linear interpolation.

2. **Joint index mapping** — Joint positions/velocities/references are repacked from 10-element arrays into 8-element arrays (excluding wheels) by `pack_input_k2()` in Python.

3. **Pitch reference offset** — `pitch_x_error` is pre-adjusted in the Python simulation loop before being passed to JAX. JAX uses it directly without internal offset application.

4. **Support center computation** — `compute_support_center_xy()` and sagittal projection run in Python.

5. **Contact detection** — Done in Python via `ContactSupervisor`.

6. **State estimation** — Centroidal state estimator runs in Python. Results passed as JAX inputs.

7. **Support feedforward vector** — `_K2_EMPIRICAL_SUPPORT_FF` is a compile-time constant in JAX.

---

## 9. Key Architectural Notes

1. **The JAX controller is a monolithic step function** (`k2_jax_controller_step`) that internally composes all sub-controllers (sagittal, shape posture, lateral roll, yaw, mode-div, support FF) and the composer (clipping + rate limiting).

2. **The Python path is modular** — separate controller objects compose via the `BalanceCoreTorqueComposer`.

3. **In `both` mode**, the Python path computes torque that drives physics, while JAX runs in parallel for comparison. This is true teacher-forcing — same state, same parameters.

4. **The ABS state is full-sized** in JAX (300-element ring buffer = 328 total state fields), matching the Python sliding window exactly since Stage 6L.

5. **Key difference from CLAUDE.md**: This is NOT a PPO/RL residual controller. This is a pure analytic balance-core controller with K2 profile. The JAX port is for deployment, not training.
