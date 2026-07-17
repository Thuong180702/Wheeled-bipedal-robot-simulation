# K2 JAX Full Port Audit — Phase 0: Source of Truth

> Generated: 2026-06-27
> Audit scope: Complete K2 controller port from Python → JAX
> Profile: `k2_notch_low_q_v1` (current-best)

---

## 1. Profile Definition

### 1.1 Profile hierarchy

```
K2_NOTCH_LOW_Q_V1 (line 3162)
  └─ replace(K1_PITCH_RATE_NOTCH, profile_name="k2_notch_low_q_v1", wip_notch_q=2.0)
       └─ K1_PITCH_RATE_NOTCH
            └─ PHYSICS_EQ_FF_OUTER_LOOP_LOW_BAND_SUPPORT_V2 (line 2940)
                 └─ PHYSICS_EQ_FF_OUTER_LOOP
                      └─ CALIBRATED_OUTER_LOOP_V2
                           └─ SUPPORT_POSITION_OUTER_LOOP
                                └─ HEIGHT_SCHEDULED_PITCH_EQ_TRIM
                                     └─ ADAPTIVE_SUPPORT_CENTERING_TRIM
```

### 1.2 K2 key parameters (inherited from K1_PITCH_RATE_NOTCH)

| Parameter | Value | Source |
|-----------|-------|--------|
| profile_name | `k2_notch_low_q_v1` | K2 override |
| wip_notch_q | **2.0** (K2 diff from K1 Q=6.0) | K2 override |
| wip_notch_center_hz | 2.5 | K1 |
| wip_notch_filter_blend | 1.0 | K1 |
| wip_notch_target_signal | `pitch_rate` | K1 |
| wip_notch_filter_type | `biquad_notch` | K1 |
| wip_notch_gate_enabled | True | K1 |
| wip_notch_height_gate_start_m | 0.42 | K1 |
| wip_notch_height_gate_full_m | 0.48 | K1 |
| enable_wip_notch_filter | True | K1 |
| calibrated_outer_loop_enabled | True | PHYSICS_EQ_FF_OUTER_LOOP_LOW_BAND_SUPPORT_V2 |
| physics_equilibrium_feedforward_enabled | True | PHYSICS_EQ_FF_OUTER_LOOP_LOW_BAND_SUPPORT_V2 |
| low_band_support_enabled | True | PHYSICS_EQ_FF_OUTER_LOOP_LOW_BAND_SUPPORT_V2 |
| adaptive_bias_trim_enabled | True | ADAPTIVE_SUPPORT_CENTERING_TRIM |
| adaptive_bias_window_steps | 300 | profile default |
| adaptive_bias_fast_window_steps | 100 | profile default |
| continuous_max_position_tau | True | K1 |
| continuous_k_position | False | K1 |
| continuous_k_wheel_velocity | False | K1 |
| continuous_kd_pitch | False | K1 |
| continuous_k_velocity | False | K1 |

### 1.3 Profile location

- **File:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- **Definition line:** 3162–3166
- **Registry line:** 1444 in `scripts/simulate_hierarchical_controller.py`
- **Current-best marker line:** 9045

---

## 2. Python K2 Source Files

### 2.1 Primary controller compute path

| File | Function/Class | Role |
|------|---------------|------|
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | `SagittalVelocityDampedBalanceController` (line 4076) | Sagittal wheel-balance torque: pitch, pitch_rate, velocity, wheel_velocity, position, support_velocity terms + notch filter + continuous scheduling + adaptive bias trim |
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | `SagittalVelocityDampedBalanceController.compute()` (line 4366) | Entry point → returns `(tau_10d, diagnostics)` |
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | `K2_NOTCH_LOW_Q_V1` (line 3162) | Authority schedule dataclass with all K2 parameters |
| `wheeled_biped/controllers/balance_core_torque_composer.py` | `BalanceCoreTorqueComposer` (line 22) | Sums 4 approved torque sources, applies clip + rate-limit |
| `wheeled_biped/controllers/balance_core_torque_composer.py` | `BalanceCoreTorqueComposer.compose()` (line 50) | `tau_total_raw = sum(4 sources)` → clip → rate-limit → tau_final |
| `wheeled_biped/controllers/support_feedforward_controller.py` | `SupportFeedforwardController` (line 15) | Empirical support FF on hip_pitch/knee |
| `wheeled_biped/controllers/support_feedforward_controller.py` | `SupportFeedforwardController.compute()` (line 53) | Returns `scale * support_vector[active_indices]` |
| `wheeled_biped/controllers/shape_posture_controller.py` | `ShapePostureController` | PD shape/posture control |
| `wheeled_biped/controllers/lateral_roll_balance_controller.py` | `LateralRollBalanceController` | Lateral roll balance |
| `wheeled_biped/controllers/yaw_controller.py` | `YawController` | Yaw stabilization |
| `wheeled_biped/controllers/mode_hip_yaw_divergence_controller.py` | `ModeBasedHipYawDivergenceController` | Hip-yaw divergence damping (mode-based) |
| `wheeled_biped/controllers/signal_filters.py` | `BiquadNotchFilter` (line 38) | Causal IIR biquad notch (Direct Form II Transposed) |
| `wheeled_biped/controllers/signal_filters.py` | `smoothstep_gate_jax()` (line 366) | JAX-compatible smoothstep height gate |
| `wheeled_biped/controllers/calibrated_outer_loop_functions_v2.py` | `calibrated_outer_loop_params()` | Height-dependent calibrated PD gains for outer loop |
| `wheeled_biped/controllers/physics_equilibrium_feedforward.py` | `physics_equilibrium_feedforward_params()` | Physics-derived equilibrium FF (pitch_eq, tau_eq per wheel) |

### 2.2 Simulation orchestration

| File | Lines | Role |
|------|-------|------|
| `scripts/simulate_hierarchical_controller.py` | 2585–2598 | `--controller-mode balance-core`, `--controller-backend python/jax/both` |
| `scripts/simulate_hierarchical_controller.py` | 2270–2500 | `build_balance_core_controllers()` — instantiates all controller objects |
| `scripts/simulate_hierarchical_controller.py` | 4690–4990 | Pitch ref offset setup, calibrated outer loop init, low-band support init |
| `scripts/simulate_hierarchical_controller.py` | 5910–6470 | Main balance-core control step: shape posture → support FF → sagittal → lateral → yaw → mode-div → composer |
| `scripts/simulate_hierarchical_controller.py` | 6470–6481 | Composer call: `composer.compose(tau_shape_posture_with_yaw, tau_support_ff, tau_sagittal, tau_lateral, tau_prev)` → `BalanceCoreTorqueResult` |
| `scripts/simulate_hierarchical_controller.py` | 6484–6559 | JAX fast-path override when `backend=jax` |
| `scripts/simulate_hierarchical_controller.py` | 6777 | Final torque: `mj_data.ctrl[:] = np.array(tau_smooth)` |

### 2.3 Notch filter implementation

| File | Function/Class | Role |
|------|---------------|------|
| `wheeled_biped/controllers/signal_filters.py` | `BiquadNotchFilter.__init__()` (line 59) | Initialize with fs_hz, fc_hz, Q. Computes coefficients via RBJ cookbook. |
| `wheeled_biped/controllers/signal_filters.py` | `BiquadNotchFilter.update()` (line 167) | Step one sample: `y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2` |
| `wheeled_biped/controllers/signal_filters.py` | `biquad_notch_coefficients()` (line 302) | Pure function for JAX: compute b0,b1,b2,a1,a2 from fc, Q, fs |
| `wheeled_biped/controllers/signal_filters.py` | `biquad_notch_update()` (line 333) | Pure function for JAX: step one sample |

---

## 3. CLI Flags for K2 Current-Best

### 3.1 Fixed-height validation

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --controller-backend python \
  --steps 500 \
  --height-variant <variant_name>
```

Where `<variant_name>` ∈:
- `low_0p320`
- `low_0p330`
- `mid_0p400`
- `high_0p480`
- Plus any other height variant from the physical_target_height_setups

### 3.2 Dynamic-height validation

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --controller-backend python \
  --steps 500 \
  --dynamic-height-command ramp_up    # or ramp_down, up_down_cycle, gate_dwell, gate_chatter
```

### 3.3 Push validation

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --controller-backend python \
  --steps 500 \
  --height-variant high_0p480 \
  --push-force-N 90 \
  --push-direction forward    # or backward
```

### 3.4 Backend flag behavior

| Flag value | Python controller runs? | JAX controller runs? | Torque applied to physics |
|------------|------------------------|---------------------|--------------------------|
| `python` (default) | ✓ Full compute + torque | ✗ Not loaded | Python `tau_smooth` |
| `jax` | ✓ For telemetry only | ✓ JIT-compiled, hot-step | JAX `tau_final` overrides Python |
| `both` | ✓ Full compute + torque | ✓ JIT-compiled, compared | Python `tau_smooth` (JAX for telemetry only) |

In `jax` mode:
1. Python path runs first for telemetry
2. JAX `pack_input_k2()` packs the same physical state
3. JIT-compiled `k2_jax_controller_step()` computes torque
4. `tau_smooth` is **overwritten** with JAX output
5. `mj_data.ctrl[:] = tau_smooth` (JAX torque)

In `both` mode:
1. Same as `jax` but Python `tau_smooth` is kept for physics
2. JAX vs Python diff is printed each step (teacher-forcing comparison)

---

## 4. Exact Final Torque Vector Path

### 4.1 Python pipeline (balance-core mode, K2 profile)

```
Step 0: tau_prev = mj_data.ctrl (initialized from robot initial state)

Step N:
  1. shape_posture.compute(q_ref, joint_pos, joint_vel, support_error, target_com_height)
     → tau_shape_posture (10-d)
  
  2. [Boundary yaw fix modifies tau_shape_posture[1,6] if boundary variant active]

  3. support_feedforward.compute()
     → tau_support_feedforward (10-d)
     = scale(0.5) × STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD[hip_pitch+knee]
     = [0, 0, 2.05, -7.75, 0, 0, 0, 1.6, -7.9, 0]

  4. sagittal_wheel_balance.compute(pitch_x_error, pitch_rate_boosted, sag_vel, 
        wheel_vel_L/R, sag_pos_error, com_z, roll_y, contact, height_variant, height_ref)
     → tau_sagittal_wheel_balance (10-d, nonzero on wheels [4,9])
     Internal: notch filter → blend → pitch/pitch_rate/velocity/position/wheel_velocity/
               support_velocity terms → adaptive bias trim → clip → saturation

  5. lateral_roll_balance.compute(roll_y, roll_rate, hip_roll_pos, hip_roll_vel, hip_roll_ref)
     → tau_lateral_roll_balance (10-d, nonzero on hip_roll [0,5])

  6. yaw_controller.compute(yaw_error, yaw_rate)
     → tau_yaw (10-d, nonzero on hip_yaw [1,6])
     Added to tau_shape_posture[1,6]

  7. mode_div_ctrl.compute(HipYawState)
     → mode_div_tau_left, mode_div_tau_right
     Added to tau_shape_posture[1,6]
     → tau_shape_posture_with_yaw (10-d)

  8. composer.compose(tau_shape_posture_with_yaw, tau_support_ff, tau_sagittal, tau_lateral, tau_prev)
     → tau_total_raw = sum(4 sources)                                    [10-d]
     → tau_total_clipped = clip(tau_total_raw, -torque_limit, +torque_limit)  [10-d]
     → delta = clip((tau_clipped - tau_prev)/dt, -max_rate, +max_rate) * dt
     → tau_final = tau_prev + delta                                      [10-d]
     → tau_prev ← tau_final (state update)

  9. mj_data.ctrl[:] = tau_final
```

### 4.2 JAX pipeline (k2_jax_controller_step)

```
Step 0: state_flat initialized by pack_state_k2() (prev_tau = zeros, all else zeros)

Step N:
  1. Unpack state_flat (328 fields: notch x4, prev_tau x10, filtered_com_z, prev_support_error,
     ol_pitch_ref_smoothed, ol_prev_support_error, ol_support_error_rate,
     ABS core x9, ABS ring buffer x300)

  2. Unpack input_flat (42 fields: pitch_x, pitch_rate, roll, roll_rate, yaw_err, yaw_rate,
     com_z, com_vy, sag_vel, sag_pos_err, wheel_vel_L/R, support_vel, height_ref,
     hy_div_err, hy_div_rate, joint_pos[0-7], joint_vel[0-7], q_ref[0-7], support_pos_err)

  3. Unpack params_flat (notch coeffs b0,b1,b2,a1,a2, torque_limits x10, max_torque_rate x10,
     control_dt, plus grid data)

  4. Step 1: Notch filter → notch_out, notch_gate, pitch_rate_eff
     State: new_notch_x1, new_notch_x2, new_notch_y1, new_notch_y2

  5. Step 2: Height scheduling → schedule_h, new_filtered_com_z, max_pos_tau

  6. Step 3: Calibrated outer loop + physics FF
     Grid interpolation → cal_kp, cal_kd, cal_theta_max, cal_deadband, cal_rate_limit, cal_lowpass_alpha
     physics_ff_tau, physics_pitch_eq
     Low-band support → lb_offset
     Outer loop state update → support_error_rate, ol_pitch_ref
     State: new_ol_support_error_rate, new_ol_pitch_ref, new_ol_prev_support_error

  7. Step 4a: Adaptive bias trim (sliding window ring buffer)
     → _trim_to_apply
     State: ring buffer updated, slow_sum, fast_sum, trim_tau, hold_steps, etc.

  8. Step 4b: Sagittal torque assembly
     → tau_sag (10-d, nonzero on wheels [4,9])

  9. Step 5: Shape posture compute
     → tau_posture (10-d, nonzero on all leg joints [0,1,2,3,5,6,7,8])

  10. Step 6: Lateral roll compute
      → tau_lateral (10-d, nonzero on hip_roll [0,5])

  11. Step 7: Yaw compute
      → tau_yaw (10-d, nonzero on hip_yaw [1,6])

  12. Step 8: Mode-div compute
      → tau_mode_div (10-d, nonzero on hip_yaw [1,6])

  13. Step 9: Support feedforward compute (EXCLUDED from tau_sum)
      → tau_support_ff (NOT added to tau_sum — JAX-only, no Python equivalent)

  14. Step 10: tau_sum = tau_sag + tau_posture + tau_lateral + k2_jax_empirical_support_ff()
      k2_jax_empirical_support_ff = [0, 0, 2.05, -7.75, 0, 0, 0, 1.6, -7.9, 0]
      (This IS the Python SupportFeedforwardController equivalent — included in tau_sum)

  15. composer(tau_sum, prev_tau, params)
      → tau_final = clip + rate-limit (same formula as Python)

  16. Post-composer: tau_final[1] += tau_yaw[1] + tau_mode_div[1]
                     tau_final[6] += tau_yaw[6] + tau_mode_div[6]

  17. Pack new_state_flat (all 328 fields)
  18. Pack diag_flat (30 fields)
  19. Return (tau_final, new_state_flat, diag_flat)
```

### 4.3 Python vs JAX insertion order comparison

| Torque source | Python insertion point | JAX insertion point | Match? |
|--------------|----------------------|---------------------|--------|
| Shape posture + yaw + mode_div | composer (tau_shape_posture_with_yaw) | composer (tau_posture only; yaw/mode_div POST-composer) | **DIFFERENT** |
| Empirical support FF (hip_pitch/knee) | composer (tau_support_feedforward) | composer (k2_jax_empirical_support_ff in tau_sum) | ✓ Same |
| Sagittal wheel balance | composer (tau_sagittal_wheel_balance) | composer (tau_sag in tau_sum) | ✓ Same |
| Lateral roll balance | composer (tau_lateral_roll_balance) | composer (tau_lateral in tau_sum) | ✓ Same |
| Yaw torque | Pre-composer on tau_shape_posture[1,6] | Post-composer on tau_final[1,6] | **DIFFERENT** |
| Mode-div torque | Pre-composer on tau_shape_posture[1,6] | Post-composer on tau_final[1,6] | **DIFFERENT** |

**Critical difference:** In Python, yaw and mode_div torques go through the composer's clip and rate-limit. In JAX, yaw and mode_div are added AFTER the composer, bypassing clip and rate-limit on hip-yaw joints [1,6].

---

## 5. Validation Commands

### 5.1 Step C — Fixed height (7 heights)

```bash
python scripts/validate_k2_jax_backend.py --step step_c --backend jax
# or directly:
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --controller-backend jax --steps 500 \
  --height-variant <variant>
```

### 5.2 Step E — Height sweep (10 heights)

```bash
python scripts/validate_k2_jax_backend.py --step step_e --backend jax
```

### 5.3 Step D — Push matrix

```bash
python scripts/validate_k2_jax_backend.py --step step_d --backend jax
```

### 5.4 Dynamic height validation

```bash
python scripts/validate_k2_jax_backend.py --step dynamic --backend jax
# Scenarios: ramp_up, ramp_down, up_down_cycle, gate_dwell, gate_chatter
```

### 5.5 Stage 7 benchmarks

```bash
python scripts/stage7_run_benchmarks.py --backend jax
```

### 5.6 Python vs JAX parity comparison

```bash
python scripts/compare_k2_python_vs_jax_step.py \
  --scenario fixed_high_0p480 --steps 300
```

---

## 6. State Inventory

### 6.1 Python state (object attributes on SagittalVelocityDampedBalanceController)

| Field | Type | Init | Update |
|-------|------|------|--------|
| `_filtered_com_z` | float | 0.4 | `alpha * prev + (1-alpha) * com_z` when no height_ref |
| `prev_support_position_error_m` | float | 0.0 | `= sagittal_position_error_m` each step |
| `_wip_notch_pitch_rate` | BiquadNotchFilter | None (lazy) | `filter.update(pitch_rate)` each step |
| `_wip_notch_wheel_left` | BiquadNotchFilter | None (lazy) | `filter.update(wheel_vel_left)` each step |
| `_wip_notch_wheel_right` | BiquadNotchFilter | None (lazy) | `filter.update(wheel_vel_right)` each step |
| `_wip_notch_support_vel` | BiquadNotchFilter | None (lazy) | `filter.update(support_vel)` each step |
| `_adaptive_bias_ring_buffer` | deque(maxlen=300) | deque of zeros | append(sag_pos_error) each step |
| `_adaptive_bias_trim_tau` | float | 0.0 | Updated by ABS logic |
| `_adaptive_bias_hold_steps` | int | 0 | Decremented or set by ABS logic |
| `_adaptive_bias_prev_error_sign` | float | 0.0 | Updated by ABS logic |
| `_adaptive_bias_zero_crossing_count` | int | 0 | Updated by ABS logic |
| `_adaptive_bias_slow_count` | int | 0 | Updated by ABS logic |
| `_adaptive_bias_slow_ptr` | int | 0 | Incremented modulo 300 |
| `_adaptive_bias_guard_trigger` | int | 0 | Flag set by ABS safety |

### 6.2 Python simulation-loop state (nonlocal in main sim loop)

| Field | Type | Init | Update |
|-------|------|------|--------|
| `prev_support_error` | float | 0.0 | `= sagittal_diag["support_position_error_m"]` |
| `prev_wheel_vel_left` | float | 0.0 | `= joint_vel[4]` |
| `prev_wheel_vel_right` | float | 0.0 | `= joint_vel[9]` |
| `tau_prev` | jnp.array(10) | `= mj_data.ctrl` | `= tau_smooth` |
| `vd_pitch_ref_offset_deg` | float | 0.0 or profile | Set once at init, may change with dynamic height |
| `outer_loop_pitch_ref_smoothed_deg` | float | 0.0 | Updated each step via outer loop logic |
| `outer_loop_prev_support_error_m` | float | 0.0 | Updated each step |
| `outer_loop_support_error_rate_smoothed` | float | 0.0 | Updated each step via lowpass filter |

### 6.3 JAX state_flat (328 fields)

| Group | Fields | Indices |
|-------|--------|---------|
| Notch filter | notch_x1, notch_x2, notch_y1, notch_y2 | 0–3 |
| Composer | prev_tau[0..9] | 4–13 |
| Height scheduling | filtered_com_z | 14 |
| Support error | prev_support_error | 15 |
| Outer loop | ol_pitch_ref_smoothed, ol_prev_support_error, ol_support_error_rate | 16–18 |
| ABS core | abs_slow_sum, abs_fast_sum, abs_trim_tau, abs_hold_steps, abs_prev_err_sign, abs_zc_count, abs_slow_count, abs_slow_ptr, abs_guard_trigger | 19–27 |
| ABS ring buffer | abs_buf_[0..299] | 28–327 |

Total: 328 fields

### 6.4 JAX input_flat (42 fields)

See `K2_JAX_INPUT_FIELDS` tuple at `k2_jax_controller.py:904` for full field listing.

Key inputs:
- pitch_x_rad (pre-adjusted by sim loop — DO NOT re-apply pitch_ref_offset)
- pitch_rate_x_rad_s
- roll_y_rad, roll_rate_y_rad_s
- yaw_error_rad, yaw_rate_rad_s
- com_z_m, com_vy_m_s
- sagittal_velocity_m_s, sagittal_position_error_m
- wheel_vel_left/right_rad_s
- support_velocity_m_s (always 0.0)
- commanded_height_ref_m
- hip_yaw_div_error, hip_yaw_div_rate
- Joint positions (8 leg joints, no wheels)
- Joint velocities (8 leg joints, no wheels)
- Reference positions (8 leg joints, no wheels)
- support_position_error_m

### 6.5 JAX params_flat

Built dynamically by `pack_params_stage2()` at init time. Contains:
- Notch filter coefficients (b0, b1, b2, a1, a2) — 5 fields
- Notch meta (fc_hz, Q, fs_hz) — 3 fields
- Torque limits per actuator — 10 fields
- Max torque rate per actuator — 10 fields
- Control dt — 1 field

Total base: 29 fields (plus grid data appended dynamically)

---

## 7. Known Issues and Discrepancy Status

### 7.1 Fixed (Stage 7B)

1. **Knee torque bypassing composer clipping (FIXED):**
   - Empirical support FF was applied post-composer, bypassing clip/rate-limit on hip_pitch/knee
   - Fix: moved `k2_jax_empirical_support_ff()` into tau_sum before composer
   - Python equivalent: `SupportFeedforwardController` output goes into composer via `tau_support_feedforward`

2. **Extra JAX-only hip-yaw support FF (FIXED):**
   - `k2_jax_support_feedforward_compute()` was height-gated hip-yaw support FF with no Python equivalent
   - Fix: excluded from tau_sum, kept only as reference
   - Python: no height-gated support FF on hip-yaw exists

3. **ramp_up and gate_chatter now PASS**

### 7.2 Still failing (as of 2026-06-27)

1. **ramp_down** — JAX backend falls or diverges
2. **push_fwd_90N** — JAX backend diverges
3. **push_bwd_90N** — JAX backend diverges

### 7.3 Remaining suspicions (to audit in Phase 1+)

- Long-horizon internal state divergence in one or more of:
  - Notch filter state
  - Outer loop state (pitch_ref_smoothed, support_error_rate)
  - Adaptive bias trim ring buffer
  - Composer tau_prev
  - filtered_com_z
  - pitch_ref_offset state
  - Dynamic target-height tracking
  - Support/feedforward state or insertion order
  - Yaw/mode_div post-composer insertion vs Python pre-composer

### 7.4 Known insertion-order difference

In Python, yaw and mode_div torques go through the composer's clip and rate-limit (pre-composer on hip_yaw).
In JAX, yaw and mode_div are added POST-composer, bypassing clip and rate-limit on hip_yaw [1,6].

For fixed-height scenarios this difference may not matter (torques stay within limits), but for dynamic/push scenarios where hip-yaw torques saturate, this bypass changes the effective final torque on joints [1,6].

---

## 8. Gate Criteria for This Audit

This document satisfies Phase 0 when:

1. ✅ Exact Python files/functions identified
2. ✅ Exact profile name confirmed: `k2_notch_low_q_v1`
3. ✅ Exact CLI flags documented
4. ✅ Fixed-height validation commands documented
5. ✅ Dynamic-height validation commands documented
6. ✅ Push validation commands documented
7. ✅ Backend flag behavior documented
8. ✅ Final torque vector path traced for Python and JAX
9. ✅ State fields inventoried for both backends
10. ✅ Known discrepancies catalogued

**Phase 0 COMPLETE. Proceed to Phase 1.**
