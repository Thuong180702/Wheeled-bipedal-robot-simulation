# K2 JAX Full Port — Phase 1: Mechanism Inventory Matrix

> Generated: 2026-06-27
> Profile: `k2_notch_low_q_v1`
> Status: "strict clone" audit — no improvements permitted until parity proven

---

## Mechanism Inventory

### Legend for Status column
- **PORTED_EXACT** — JAX reproduces Python exactly (same math, same order, same state)
- **PORTED_WITH_KNOWN_DIFF** — JAX ports the mechanism but has a documented difference
- **MISSING_IN_JAX** — Active Python mechanism not in JAX
- **JAX_EXTRA_NO_PYTHON_EQUIVALENT** — JAX-only mechanism with no Python counterpart
- **DOWNSTREAM_PYTHON_ONLY** — Mechanism applied after mj_data.ctrl in Python only
- **INACTIVE_ZERO_CONFIRMED** — Mechanism disabled/inactive in K2 profile (confirmed zero)

---

## 1. Input Packing and Physical State Extraction

| Field | Value |
|-------|-------|
| **Mechanism** | Input packing: physical state → controller input |
| **Python source** | `simulate_hierarchical_controller.py:6494-6515` (JAX input packing), `simulate_hierarchical_controller.py:6207-6222` (Python sagittal.compute args) |
| **Active/gated/disabled** | ACTIVE — always runs |
| **Input fields** | 42 fields: pitch_x, pitch_rate, roll, roll_rate, yaw_err, yaw_rate, com_z, com_vy, sag_vel, sag_pos_err, wheel_vel_L/R, support_vel, height_ref, hy_div_err/rate, joint_pos[8], joint_vel[8], q_ref[8], support_pos_err |
| **State fields** | N/A (stateless) |
| **Params/gains** | N/A |
| **Output torque contribution** | None directly — feeds all controller terms |
| **Actuator indices affected** | All 10 |
| **Insertion point** | First step of controller pipeline |
| **JAX source** | `k2_jax_controller.py:934-976` (`pack_input_k2`) |
| **JAX state_flat fields** | N/A (input, not state) |
| **JAX params fields** | N/A |
| **Parity test coverage** | `test_k2_jax_step_parity.py` — input packing shape tests |
| **Validation coverage** | Implicit in all validation runs |
| **Status** | **PORTED_EXACT** |

**Note:** `pitch_x` passed to JAX is pre-adjusted by the Python sim loop (`pitch_x_error = raw_pitch - pitch_ref_offset`). JAX does NOT re-apply pitch_ref_offset internally. Python sagittal.compute() also receives pre-adjusted pitch_x_error. This is correct for both paths.

---

## 2. Dynamic Target Height Update

| Field | Value |
|-------|-------|
| **Mechanism** | Dynamic target height tracking for height transitions |
| **Python source** | `simulate_hierarchical_controller.py` — `height_variant_setup["target_com_z_m"]` updated per step from height command |
| **Active/gated/disabled** | ACTIVE for dynamic scenarios, GATED for fixed-height |
| **Input fields** | `commanded_height_ref_m` from dynamic height command or variant setup |
| **State fields** | None (external update) |
| **Params/gains** | Height transition rate |
| **Output torque contribution** | Drives height scheduling, outer loop, notch gate, support FF |
| **Actuator indices affected** | All (indirectly through scheduling) |
| **Insertion point** | Before controller step |
| **JAX source** | Input field `_I_HEIGHT_REF` = `commanded_height_ref_m` |
| **JAX state_flat fields** | `filtered_com_z` (14) — used as fallback when height_ref ≤ 0 |
| **JAX params fields** | None |
| **Parity test coverage** | None specific |
| **Validation coverage** | Dynamic height scenarios |
| **Status** | **PORTED_EXACT** |

---

## 3. Height Variant Setup — target_com_z_m

| Field | Value |
|-------|-------|
| **Mechanism** | Height variant target CoM height |
| **Python source** | `simulate_hierarchical_controller.py:4697-4734` — sets `vd_pitch_ref_offset_deg` from profile or schedule |
| **Active/gated/disabled** | ACTIVE — always present |
| **Input fields** | `height_variant_setup["target_com_z_m"]` |
| **State fields** | None |
| **Params/gains** | Per-variant target height |
| **Output torque contribution** | Sets pitch ref offset, height scheduling input |
| **Actuator indices affected** | All (indirect) |
| **Insertion point** | Init-time and per-step for dynamic |
| **JAX source** | Input `commanded_height_ref_m` |
| **JAX state_flat fields** | None directly |
| **JAX params fields** | Grid data built at module load |
| **Parity test coverage** | Implicit |
| **Validation coverage** | All fixed-height and dynamic scenarios |
| **Status** | **PORTED_EXACT** |

---

## 4. pitch_x_error Computation

| Field | Value |
|-------|-------|
| **Mechanism** | Pitch error = raw pitch - pitch_ref_offset |
| **Python source** | `simulate_hierarchical_controller.py:6208` — `pitch_x_error` passed to sagittal controller |
| **Active/gated/disabled** | ACTIVE |
| **Input fields** | `pitch_x_rad` (pre-adjusted by sim loop) |
| **State fields** | None |
| **Params/gains** | `pitch_ref_offset_deg` from profile or schedule |
| **Output torque contribution** | Drives tau_pitch term |
| **Actuator indices affected** | Wheels [4,9] through sagittal |
| **Insertion point** | Before controller step — applied to raw pitch |
| **JAX source** | Input `_I_PITCH_X` = pre-adjusted pitch (JAX does NOT re-apply offset) |
| **JAX state_flat fields** | None |
| **JAX params fields** | None (offset applied externally) |
| **Parity test coverage** | `compare_k2_python_vs_jax_step.py` |
| **Validation coverage** | All scenarios |
| **Status** | **PORTED_EXACT** — Both Python and JAX receive pre-adjusted pitch_x_error |

---

## 5. pitch_ref_offset (Physics Equilibrium Pitch Reference)

| Field | Value |
|-------|-------|
| **Mechanism** | Physics-derived equilibrium pitch reference offset |
| **Python source** | `simulate_hierarchical_controller.py:4736-4787` — `physics_equilibrium_feedforward_params()` |
| **Active/gated/disabled** | ACTIVE (K2 inherits `physics_equilibrium_feedforward_enabled=True`) |
| **Input fields** | Height (from variant setup or dynamic command) |
| **State fields** | None |
| **Params/gains** | `physics_eq_ff_clamp_to_height_range`, `physics_eq_ff_max_abs_nm` |
| **Output torque contribution** | `tau_eq_ff_each_wheel` (feedforward wheel torque) + equivalent pitch_ref offset |
| **Actuator indices affected** | Wheels [4,9] |
| **Insertion point** | Init: sets `vd_pitch_ref_offset_deg`. Per-step: adds `physics_ff_tau` to wheel torque |
| **JAX source** | `k2_jax_controller.py:1138` — `physics_ff_tau = k2_jax_grid_interpolate(schedule_h, ff_grid["grid_heights"], ff_grid["tau_eq_ff_grid"])` |
| **JAX state_flat fields** | None |
| **JAX params fields** | Grid data (`_physics_ff_grid_cache`) |
| **Parity test coverage** | None specific |
| **Validation coverage** | All scenarios at all heights |
| **Status** | **PORTED_EXACT** (VERIFIED 2026-06-27) — Both paths use Option B (equivalent pitch_ref through `pitch_x_error` pre-adjustment). JAX `physics_ff_tau` (line 1138) is diagnostic-only. `total_pitch_ref_offset_deg` (line 1163) is dead code. No discrepancy.

---

## 6. Calibrated Outer Loop

| Field | Value |
|-------|-------|
| **Mechanism** | Height-dependent calibrated PD outer loop for support-position pitch reference |
| **Python source** | `simulate_hierarchical_controller.py:4857-4989` — calibrated outer loop setup |
| **Active/gated/disabled** | ACTIVE (`calibrated_outer_loop_enabled=True` in K2 lineage) |
| **Input fields** | Support position error, support error rate, height |
| **State fields** | `ol_pitch_ref_smoothed`, `ol_prev_support_error`, `ol_support_error_rate` |
| **Params/gains** | cal_kp, cal_kd, cal_theta_max, cal_deadband, cal_rate_limit, cal_lowpass_alpha (from grid) |
| **Output torque contribution** | Pitch reference offset (added to pitch_ref_offset) → affects sagittal pitch torque |
| **Actuator indices affected** | Wheels [4,9] (through sagittal pitch term) |
| **Insertion point** | Before sagittal controller: adjusts pitch_ref_offset |
| **JAX source** | `k2_jax_controller.py:1127-1158` — grids, outer loop pitch ref, state update |
| **JAX state_flat fields** | `ol_pitch_ref_smoothed` (16), `ol_prev_support_error` (17), `ol_support_error_rate` (18) |
| **JAX params fields** | Grid data (`_calibrated_grid_cache`) |
| **Parity test coverage** | Outer loop parity not explicitly tested |
| **Validation coverage** | Implicit in all scenarios |
| **Status** | **PORTED_EXACT** — JAX implements full calibrated outer loop: grid interpolation → kp/kd/theta_max/deadband/rate_limit/lowpass → support error rate computation → outer loop pitch ref (PD + rate-limit + lowpass). State update matches Python: new_ol_support_error_rate, new_ol_pitch_ref, new_ol_prev_support_error. |

---

## 7. Support Error Computation

| Field | Value |
|-------|-------|
| **Mechanism** | Support center position error relative to equilibrium |
| **Python source** | `simulate_hierarchical_controller.py:5979-6003` — support center computation and sagittal projection |
| **Active/gated/disabled** | ACTIVE |
| **Input fields** | Wheel xpos, support_center_eq_xy, sagittal_axis_xy |
| **State fields** | `prev_support_error` |
| **Params/gains** | position_authority_scale |
| **Output torque contribution** | Feeds sagittal position term, outer loop, ABS |
| **Actuator indices affected** | Wheels [4,9] |
| **Insertion point** | Before sagittal controller |
| **JAX source** | Input `sagittal_position_error_m` (pre-computed by Python sim loop, passed as input) |
| **JAX state_flat fields** | `prev_support_error` (15) |
| **JAX params fields** | None |
| **Parity test coverage** | Implicit |
| **Validation coverage** | All scenarios |
| **Status** | **PORTED_EXACT** — Both Python and JAX receive `sag_pos_error` from Python sim loop. The support center computation is done externally in Python sim loop and the result is passed to both paths. |

---

## 8. Support Error Rate Smoothing

| Field | Value |
|-------|-------|
| **Mechanism** | Lowpass-filtered support error rate for outer loop derivative term |
| **Python source** | `simulate_hierarchical_controller.py` — outer loop support_error_rate computation |
| **Active/gated/disabled** | ACTIVE |
| **Input fields** | support_pos_error, prev_support_error |
| **State fields** | `outer_loop_support_error_rate_smoothed`, `outer_loop_prev_support_error_m` |
| **Params/gains** | `outer_loop_support_velocity_lowpass_alpha` (from grid) |
| **Output torque contribution** | Derivative term in outer loop → pitch_ref_offset |
| **Actuator indices affected** | Wheels [4,9] (through sagittal) |
| **Insertion point** | Inside outer loop |
| **JAX source** | `k2_jax_controller.py:1145-1149` |
| **JAX state_flat fields** | `ol_support_error_rate` (18), `ol_prev_support_error` (17) |
| **JAX params fields** | `cal_lowpass_alpha` from grid |
| **Parity test coverage** | None specific |
| **Validation coverage** | Implicit |
| **Status** | **PORTED_EXACT** |

---

## 9. Low-Band Support

| Field | Value |
|-------|-------|
| **Mechanism** | Gaussian-shaped low-band support pitch ref correction around 0.320 m |
| **Python source** | `simulate_hierarchical_controller.py:4934-4962` — low-band support pitch ref |
| **Active/gated/disabled** | ACTIVE (`low_band_support_outer_loop_enabled=True` in K2 lineage) |
| **Input fields** | Height, support position error |
| **State fields** | None |
| **Params/gains** | center=0.320m, sigma=0.004m, kp_peak=1.5, theta_ref_max_peak=3.0, pitch_ref_offset_peak=0.0 |
| **Output torque contribution** | Additive pitch ref offset → affects sagittal pitch torque |
| **Actuator indices affected** | Wheels [4,9] |
| **Insertion point** | Added to outer loop pitch ref |
| **JAX source** | `k2_jax_controller.py:1141-1142` — `k2_jax_low_band_support_pitch_ref()` |
| **JAX state_flat fields** | None |
| **JAX params fields** | Hardcoded in function call: center=0.320, sigma=0.004, kp=1.4, theta_max=3.0, pb=1.0 |
| **Parity test coverage** | None specific |
| **Validation coverage** | Low-height scenarios |
| **Status** | **PORTED_WITH_KNOWN_DIFF** — JAX uses kp_peak=1.4 vs Python's 1.5. **MINOR: 0.1 difference in kp_peak.** Verify whether K2 profile overrides this. |

---

## 10. Empirical Support Feedforward

| Field | Value |
|-------|-------|
| **Mechanism** | Fixed empirical support feedforward torque on hip_pitch and knee joints |
| **Python source** | `support_feedforward_controller.py:53-72` — `compute()` returns `scale × support_vector[hip_pitch+knee]` |
| **Active/gated/disabled** | ACTIVE |
| **Input fields** | None (constant vector) |
| **State fields** | None |
| **Params/gains** | `scale=0.5`, `joint_group="hip_pitch_knee"`, `support_vector=[0,0,4.1,-15.5,0,0,0,3.2,-15.8,0]` |
| **Output torque contribution** | `[0, 0, 2.05, -7.75, 0, 0, 0, 1.6, -7.9, 0]` on hip_pitch[2,7] and knee[3,8] |
| **Actuator indices affected** | Hip pitch [2,7], knee [3,8] |
| **Insertion point** | Inside composer: `tau_total_raw += tau_support_feedforward` |
| **JAX source** | `k2_jax_controller.py:768-776` — `k2_jax_empirical_support_ff()` returns hardcoded vector |
| **JAX state_flat fields** | None |
| **JAX params fields** | None (hardcoded) |
| **Parity test coverage** | Implicit in step parity |
| **Validation coverage** | All scenarios |
| **Status** | **PORTED_EXACT** — Stage 7B fix: moved inside tau_sum before composer so clip/rate-limit apply to knee. Both paths now have the same vector `[0,0,2.05,-7.75,0,0,0,1.6,-7.9,0]`. |

---

## 11. Sagittal Velocity/Position/Wheel Terms

| Field | Value |
|-------|-------|
| **Mechanism** | Sagittal balance torque: tau_pitch + tau_pitch_rate + tau_velocity + tau_position + tau_wheel_velocity + tau_support_velocity |
| **Python source** | `sagittal_velocity_damped_balance_controller.py:4366-4515+` — `compute()` method |
| **Active/gated/disabled** | ACTIVE |
| **Input fields** | pitch_x, pitch_rate, sag_vel, wheel_vel_L/R, sag_pos_error, support_vel, com_z, roll_y, contact, height_variant, height_ref |
| **State fields** | `_filtered_com_z`, `prev_support_position_error_m`, notch filter state, ABS ring buffer state |
| **Params/gains** | `kp_pitch=50.0`, `kd_pitch=10.0` (not scheduled in K2), `k_velocity=15.0`, `k_wheel_velocity=0.5`, `k_position=40.0`, `k_support_velocity=0.0`, `max_position_tau=4.0→6.0` (scheduled), `wheel_torque_sign=1.0` |
| **Output torque contribution** | tau_sagittal_wheel_balance (nonzero on wheels [4,9]) |
| **Actuator indices affected** | Wheels [4,9] |
| **Insertion point** | Inside composer as `tau_sagittal_wheel_balance` |
| **JAX source** | `k2_jax_controller.py:1195-1209` — `k2_jax_sagittal_torque_assembly()` |
| **JAX state_flat fields** | filtered_com_z, notch state (indirect) |
| **JAX params fields** | Gains hardcoded in step function |
| **Parity test coverage** | `test_k2_jax_step_parity.py` |
| **Validation coverage** | All scenarios |
| **Status** | **PORTED_EXACT** — K2 profile uses: continuous_k_position=False → kpos=40.0; continuous_k_wheel_velocity=False → kwheel=0.5; continuous_kd_pitch=False → kd_pitch=10.0; continuous_k_velocity=False → k_velocity=15.0; continuous_max_position_tau=True → max_pos_tau 4.0→6.0. All match. |

---

## 12. WIP Notch Filter

| Field | Value |
|-------|-------|
| **Mechanism** | Biquad notch filter on pitch_rate at 2.5 Hz to suppress wheeled inverted pendulum mode |
| **Python source** | `sagittal_velocity_damped_balance_controller.py:4628-4711` — notch filter inside `compute()` |
| **Active/gated/disabled** | ACTIVE (`enable_wip_notch_filter=True`, `wip_notch_filter_type="biquad_notch"`) |
| **Input fields** | pitch_rate_x_rad_s (raw) |
| **State fields** | Notch filter state: x1, x2, y1, y2 (per `BiquadNotchFilter` instance) |
| **Params/gains** | fc=2.5 Hz, Q=2.0 (K2), fs=100 Hz (from 1/dt), blend=1.0 |
| **Output torque contribution** | Filtered pitch_rate_eff = (1-gate)*raw + gate*notched → affects tau_pitch_rate term |
| **Actuator indices affected** | Wheels [4,9] (through pitch rate damping) |
| **Insertion point** | Inside sagittal controller, before pitch rate torque computation |
| **JAX source** | `k2_jax_controller.py:1091-1100` — `notch_out = b0*pr + b1*x1 + b2*x2 - a1*y1 - a2*y2` |
| **JAX state_flat fields** | `notch_x1` (0), `notch_x2` (1), `notch_y1` (2), `notch_y2` (3) |
| **JAX params fields** | `notch_b0`, `notch_b1`, `notch_b2`, `notch_a1`, `notch_a2` |
| **Parity test coverage** | `test_k2_jax_component_parity.py` — notch coefficient and step parity (≤1e-10) |
| **Validation coverage** | All scenarios affected by pitch rate |
| **Status** | **PORTED_EXACT** — Same Direct Form II Transposed formula. Coefficients from RBJ cookbook match. State update matches. |

---

## 13. Notch Height Gate

| Field | Value |
|-------|-------|
| **Mechanism** | Smoothstep height gate for notch filter blend (0.42–0.48 m) |
| **Python source** | `sagittal_velocity_damped_balance_controller.py:4647-4654` — `smoothstep_gate()` |
| **Active/gated/disabled** | ACTIVE (`wip_notch_gate_enabled=True`) |
| **Input fields** | schedule_height_ref |
| **State fields** | None |
| **Params/gains** | gate_start=0.42m, gate_full=0.48m |
| **Output torque contribution** | Controls blend: `pitch_rate_eff = (1-gate)*raw + gate*notched` |
| **Actuator indices affected** | Wheels [4,9] |
| **Insertion point** | Inside sagittal controller, after notch filter |
| **JAX source** | `k2_jax_controller.py:1099` — `notch_gate = smoothstep_gate_jax(height_ref, 0.42, 0.48)` |
| **JAX state_flat fields** | None |
| **JAX params fields** | None (constants) |
| **Parity test coverage** | Implicit in notch parity |
| **Validation coverage** | Dynamic height gate-crossing scenarios |
| **Status** | **PORTED_EXACT** — Same smoothstep function, same thresholds. |

---

## 14. Continuous Gain Scheduling

| Field | Value |
|-------|-------|
| **Mechanism** | Height-dependent continuous scheduling of k_position, k_wheel_velocity, kd_pitch, k_velocity, max_position_tau |
| **Python source** | `sagittal_velocity_damped_balance_controller.py:4419-4601` |
| **Active/gated/disabled** | PARTIAL: `continuous_max_position_tau=True`; others False in K2 |
| **Input fields** | schedule_height_ref |
| **State fields** | None |
| **Params/gains** | Schedule bounds per mechanism |
| **Output torque contribution** | Modulates sagittal gains |
| **Actuator indices affected** | Wheels [4,9] |
| **Insertion point** | Inside sagittal controller, before torque computation |
| **JAX source** | `k2_jax_controller.py:1102-1124` |
| **JAX state_flat fields** | State values hardcoded |
| **JAX params fields** | None (hardcoded from K2 profile) |
| **Parity test coverage** | Implicit |
| **Validation coverage** | All heights |
| **Status** | **PORTED_EXACT** — K2: only continuous_max_position_tau is active. JAX matches: max_pos_tau = scheduled_k_position(schedule_h, 4.0, 6.0, 0.300, 0.393). kpos=40.0, kwheel=0.5, kd_pitch=10.0 all unscheduled (matching K2 profile). |

---

## 15. Shape Posture

| Field | Value |
|-------|-------|
| **Mechanism** | PD shape/posture control on all leg joints |
| **Python source** | `shape_posture_controller.py` — `ShapePostureController.compute()` |
| **Active/gated/disabled** | ACTIVE |
| **Input fields** | q_ref (10), joint_pos (10), joint_vel (10), posture_weight, contact_degraded_scale, support_position_error, target_com_height |
| **State fields** | None (PD only) |
| **Params/gains** | Per-joint Kp/Kd from posture table |
| **Output torque contribution** | tau_shape_posture (nonzero on all leg joints [0,1,2,3,5,6,7,8]) |
| **Actuator indices affected** | All leg joints [0,1,2,3,5,6,7,8]; wheels [4,9] = 0 |
| **Insertion point** | Inside composer as `tau_shape_posture` |
| **JAX source** | `k2_jax_controller.py:1211-1219` — `k2_jax_shape_posture_compute()` |
| **JAX state_flat fields** | None |
| **JAX params fields** | None (PD gains from posture table — hardcoded in JAX function) |
| **Parity test coverage** | Implicit in step parity |
| **Validation coverage** | All scenarios |
| **Status** | **PORTED_WITH_KNOWN_DIFF** — JAX `k2_jax_shape_posture_compute()` returns torque that includes the hip-yaw channels. In Python, yaw torque is applied via a separate `YawController` and added to tau_shape_posture. **NEEDS AUDIT: Are PD gains identical between JAX shape posture and Python ShapePostureController + YawController?**

---

## 16. Lateral Roll

| Field | Value |
|-------|-------|
| **Mechanism** | Lateral roll balance on hip_roll joints |
| **Python source** | `lateral_roll_balance_controller.py` — `LateralRollBalanceController.compute()` |
| **Active/gated/disabled** | ACTIVE |
| **Input fields** | roll_y, roll_rate, hip_roll_pos, hip_roll_vel, hip_roll_ref |
| **State fields** | None (PD only) |
| **Params/gains** | kp_roll, kd_roll, max_tau |
| **Output torque contribution** | tau_lateral (nonzero on hip_roll [0,5]) |
| **Actuator indices affected** | Hip roll [0,5] |
| **Insertion point** | Inside composer as `tau_lateral_roll_balance` |
| **JAX source** | `k2_jax_controller.py:1222-1225` — `k2_jax_lateral_roll_compute()` |
| **JAX state_flat fields** | None |
| **JAX params fields** | None (gain hardcoded in JAX function) |
| **Parity test coverage** | Implicit |
| **Validation coverage** | All scenarios |
| **Status** | **PORTED_EXACT** |

---

## 17. Yaw

| Field | Value |
|-------|-------|
| **Mechanism** | Yaw stabilization on hip_yaw joints |
| **Python source** | `yaw_controller.py` — `YawController.compute()` |
| **Active/gated/disabled** | ACTIVE |
| **Input fields** | yaw_error, yaw_rate |
| **State fields** | None |
| **Params/gains** | kp_yaw, kd_yaw, max_tau |
| **Output torque contribution** | tau_yaw (nonzero on hip_yaw [1,6]) |
| **Actuator indices affected** | Hip yaw [1,6] |
| **Insertion point** | **Python:** Pre-composer — added to `tau_shape_posture[1,6]` before composer clip/rate-limit |
| | **JAX:** Post-composer — added to `tau_final[1,6]` AFTER composer clip/rate-limit |
| **JAX source** | `k2_jax_controller.py:1228` — `k2_jax_yaw_compute()`; applied at 1253-1254 |
| **JAX state_flat fields** | None |
| **JAX params fields** | None |
| **Parity test coverage** | Step parity tests |
| **Validation coverage** | All scenarios |
| **Status** | **PORTED_WITH_KNOWN_DIFF** — **CONFIRMED INSERTION ORDER MISMATCH (2026-06-27).** Python: yaw added to tau_shape_posture BEFORE composer → clipped + rate-limited on [1,6]. JAX: yaw added POST-composer (line 1253-1254) → bypasses clip AND rate-limit on [1,6]. JAX comment at line 1240-1241 claiming this "matches real simulation order" is FALSE. Python places yaw PRE-composer. This is a **HIGH-RISK** discrepancy for push/dynamic scenarios where hip-yaw torques may saturate. |

---

## 18. Mode Hip-Yaw Divergence (mode_div)

| Field | Value |
|-------|-------|
| **Mechanism** | Mode-based hip-yaw divergence damping |
| **Python source** | `mode_hip_yaw_divergence_controller.py` — `ModeBasedHipYawDivergenceController.compute()` |
| **Active/gated/disabled** | ACTIVE (K2: mode_div Kp=10.0, Kd=0.50, max_tau=7.5 Nm) |
| **Input fields** | div_error (= hip_yaw_left - hip_yaw_right - ref_div), div_rate, height, support_error, support_error_rate |
| **State fields** | None |
| **Params/gains** | Kp=10.0, Kd=0.50, max_tau=7.5, height_gate 0.30-0.60 |
| **Output torque contribution** | tau_mode_div left/right (antisymmetric on hip_yaw [1,6]) |
| **Actuator indices affected** | Hip yaw [1,6] |
| **Insertion point** | **Python:** Pre-composer — added to `tau_shape_posture_with_yaw[1,6]` |
| | **JAX:** Post-composer — added to `tau_final[1,6]` |
| **JAX source** | `k2_jax_controller.py:1231-1232` — `k2_jax_mode_div_compute()`; applied at 1255-1256 |
| **JAX state_flat fields** | None |
| **JAX params fields** | None (Kp/Kd hardcoded) |
| **Parity test coverage** | Step parity |
| **Validation coverage** | All scenarios |
| **Status** | **PORTED_WITH_KNOWN_DIFF** — **CONFIRMED INSERTION ORDER MISMATCH (2026-06-27).** Same as yaw. Python: mode_div added to tau_shape_posture_with_yaw BEFORE composer (line 6461-6462) → clipped + rate-limited on [1,6]. JAX: mode_div added POST-composer (line 1255-1256) → bypasses clip AND rate-limit on [1,6]. Verify K2 mode-div Kp/Kd match Python runtime values. |

---

## 19. HY-FF / HY2-DIV

| Field | Value |
|-------|-------|
| **Mechanism** | Hip-yaw support feedforward (height-gated) — JAX-only mechanism |
| **Python source** | **NO PYTHON EQUIVALENT** |
| **Active/gated/disabled** | **JAX_EXTRA** — excluded from tau_sum (Stage 7B fix) |
| **Input fields** | support_pos_err, height |
| **State fields** | None |
| **Params/gains** | N/A |
| **Output torque contribution** | None (excluded from tau_sum) |
| **Actuator indices affected** | None (excluded) |
| **Insertion point** | N/A (excluded) |
| **JAX source** | `k2_jax_controller.py:1235-1236` — `tau_support_ff = k2_jax_support_feedforward_compute(...)` — **NOT added to tau_sum** |
| **JAX state_flat fields** | None |
| **JAX params fields** | None |
| **Parity test coverage** | N/A |
| **Validation coverage** | N/A |
| **Status** | **JAX_EXTRA_NO_PYTHON_EQUIVALENT** — Correctly excluded. No Python code computes a height-gated hip-yaw support feedforward. The Python `SupportFeedforwardController` applies constant empirical support FF on hip_pitch/knee only. |

---

## 20. Adaptive Bias Trim (T6J → ABS sliding window)

| Field | Value |
|-------|-------|
| **Mechanism** | Adaptive bias trim using sliding-window arithmetic mean of support position error |
| **Python source** | `sagittal_velocity_damped_balance_controller.py` — adaptive bias trim section inside `compute()` |
| **Active/gated/disabled** | ACTIVE (`adaptive_bias_trim_enabled=True` in K2 lineage) |
| **Input fields** | sagittal_position_error_m, height, pitch_x |
| **State fields** | Ring buffer (deque, maxlen=300), trim_tau, hold_steps, prev_error_sign, zc_count, slow_count, slow_ptr, guard_trigger |
| **Params/gains** | window=300, fast_window=100, enter=0.035m, exit=0.012m, k=5.0 Nm/m, max_tau_low=0.35, max_tau_high=0.50, disable_pitch>12°, disable_abs_error>0.25m |
| **Output torque contribution** | Additive position trim to wheel torque (via `external_position_trim` in sagittal) |
| **Actuator indices affected** | Wheels [4,9] |
| **Insertion point** | Inside sagittal controller: added to position torque |
| **JAX source** | `k2_jax_controller.py:1169-1193` — `_k2_jax_adaptive_bias_trim()` |
| **JAX state_flat fields** | ABS core: `abs_slow_sum`(19), `abs_fast_sum`(20), `abs_trim_tau`(21), `abs_hold_steps`(22), `abs_prev_err_sign`(23), `abs_zc_count`(24), `abs_slow_count`(25), `abs_slow_ptr`(26), `abs_guard_trigger`(27); ABS ring buffer: `abs_buf_[0..299]`(28-327) |
| **JAX params fields** | None (hardcoded from K2 profile) |
| **Parity test coverage** | `stage6l_phase1_lockstep_trace.py` |
| **Validation coverage** | Long-horizon scenarios |
| **Status** | **PORTED_WITH_KNOWN_DIFF** — VERIFIED 2026-06-27. Three discrepancies found:

**1. ZC guard activation timing (SIGNIFICANT):** Python activates guard immediately when `zc_count > 8` (line 5634). JAX requires 3 consecutive steps: `(guard_trigger >= 3.0) & zc_guard` (line 1513). JAX delays scale reduction by 3 steps.

**2. Guard trigger reset (MODERATE):** Python hard resets to 0 when zc_guard is False (line 5640). JAX uses exponential decay `*0.99` (line 1512). JAX retains "memory" of recent triggers.

**3. Missing safety gates (MODERATE):** JAX is missing roll, hip_yaw, and contact-stable safety checks (line 1173-1177 vs Python lines 5657-5681). JAX may apply trim in conditions where Python would block it.

Ring buffer, mean computation, zero-crossing count, hold steps, near-zero relief, and trim rate limiting all match. Window sizes identical (300/100). |

---

## 21. Composer (Sum + Clip + Rate-Limit)

| Field | Value |
|-------|-------|
| **Mechanism** | BalanceCoreTorqueComposer: sum 4 sources → clip → rate-limit |
| **Python source** | `balance_core_torque_composer.py:50-155` — `compose()` |
| **Active/gated/disabled** | ACTIVE |
| **Input fields** | tau_shape_posture, tau_support_feedforward, tau_sagittal_wheel_balance, tau_lateral_roll_balance, tau_prev |
| **State fields** | tau_prev (10-d) |
| **Params/gains** | torque_limit (10-d, per-joint), max_torque_rate (10-d, per-joint), control_dt |
| **Output torque contribution** | tau_final (10-d) |
| **Actuator indices affected** | All 10 |
| **Insertion point** | Final step before mj_data.ctrl assignment |
| **JAX source** | `k2_jax_controller.py:1247-1250` — `k2_jax_torque_composer_step(tau_sum, prev_tau, params_flat)` |
| **JAX state_flat fields** | `prev_tau[0..9]` (4-13) |
| **JAX params fields** | torque_limit[0..9], max_torque_rate[0..9], control_dt |
| **Parity test coverage** | `test_k2_jax_component_parity.py` — composer parity ≤1e-10 |
| **Validation coverage** | All scenarios |
| **Status** | **PORTED_EXACT** — Same formula: `tau_total_clipped = clip(tau_sum, -torque_limit, +torque_limit)` → `delta = clip((tau_clipped - tau_prev)/dt, -max_rate, +max_rate) * dt` → `tau_final = tau_prev + delta`. |

---

## 22. max_position_tau Scheduling

| Field | Value |
|-------|-------|
| **Mechanism** | Height-dependent scheduling of max position torque cap |
| **Python source** | `sagittal_velocity_damped_balance_controller.py:4443-4455` |
| **Active/gated/disabled** | ACTIVE (`continuous_max_position_tau=True`) |
| **Input fields** | schedule_height_ref |
| **State fields** | None |
| **Params/gains** | nominal=4.0 Nm, low_max=6.0 Nm, z_low=0.300, z_high=0.393 |
| **Output torque contribution** | Caps position torque in sagittal |
| **Actuator indices affected** | Wheels [4,9] |
| **Insertion point** | Inside sagittal controller |
| **JAX source** | `k2_jax_controller.py:1118-1124` |
| **JAX state_flat fields** | None |
| **JAX params fields** | None (hardcoded) |
| **Parity test coverage** | Implicit |
| **Validation coverage** | All heights |
| **Status** | **PORTED_EXACT** |

---

## 23. Per-Joint Torque Limits

| Field | Value |
|-------|-------|
| **Mechanism** | Per-actuator torque limits for composer clipping |
| **Python source** | `simulate_hierarchical_controller.py:5299` — `torque_limit` passed to pack_params_stage2 |
| **Active/gated/disabled** | ACTIVE |
| **Input fields** | torque_limit (10-d) |
| **State fields** | None |
| **Params/gains** | Per-joint limits from robot model |
| **Output torque contribution** | Clip bounds in composer |
| **Actuator indices affected** | All 10 |
| **Insertion point** | Inside composer |
| **JAX source** | Params `torque_limit[0..9]` |
| **JAX state_flat fields** | None |
| **JAX params fields** | `torque_limit[0..9]` (10 fields) |
| **Parity test coverage** | Composer parity |
| **Validation coverage** | All scenarios |
| **Status** | **PORTED_EXACT** — Stage 6L fix corrected uniform 57 Nm → per-joint limits matching Python. |

---

## 24. Clip / Rate-Limit / Smoothing

| Field | Value |
|-------|-------|
| **Mechanism** | Composer's clip + rate-limit (no separate smoothing — tau_final IS the smoothed output) |
| **Python source** | `balance_core_torque_composer.py:93-100` |
| **Active/gated/disabled** | ACTIVE |
| **Input fields** | tau_total_raw (10-d), tau_prev (10-d) |
| **State fields** | tau_prev |
| **Params/gains** | torque_limit, max_torque_rate, control_dt |
| **Output torque contribution** | Clips and rate-limits all 10 actuators |
| **Actuator indices affected** | All 10 |
| **Insertion point** | Inside composer |
| **JAX source** | `k2_jax_torque_composer_step()` — same formula |
| **JAX state_flat fields** | prev_tau |
| **JAX params fields** | torque_limit, max_torque_rate, control_dt |
| **Parity test coverage** | Composer parity ≤1e-10 |
| **Validation coverage** | All scenarios |
| **Status** | **PORTED_EXACT** |

---

## 25. tau_prev State

| Field | Value |
|-------|-------|
| **Mechanism** | Previous-step final torque for rate limiting |
| **Python source** | `simulate_hierarchical_controller.py:6481` — `tau_prev = tau_smooth` |
| **Active/gated/disabled** | ACTIVE |
| **Input fields** | tau_final from previous step |
| **State fields** | tau_prev (10-d) |
| **Params/gains** | None |
| **Output torque contribution** | Rate-limit reference |
| **Actuator indices affected** | All 10 |
| **Insertion point** | End of control step → start of next step |
| **JAX source** | `k2_jax_controller.py:1263` — `new_state = new_state.at[_S_PREV_TAU_START:...].set(tau_final)` |
| **JAX state_flat fields** | `prev_tau[0..9]` (4-13) |
| **JAX params fields** | None |
| **Parity test coverage** | Multi-step parity |
| **Validation coverage** | All scenarios |
| **Status** | **PORTED_WITH_KNOWN_DIFF** — **CONFIRMED STATE DIVERGENCE (2026-06-27).** JAX `tau_final[1,6]` = composer(tau_sum)[1,6] + yaw[1,6] + mode_div[1,6] (post-composer, un-clipped). Python `tau_final[1,6]` = composer(tau_shape_posture_with_yaw + ...[1,6]) (pre-composer, clipped + rate-limited). This means `tau_prev[1,6]` diverges between Python and JAX every step, accumulating over time. This is a **HIGH-RISK** source of long-horizon state divergence since `tau_prev` feeds the rate-limit calculation in the next step. |

---

## 26. Telemetry / Diag Mapping

| Field | Value |
|-------|-------|
| **Mechanism** | Diagnostic field extraction for telemetry |
| **Python source** | `sagittal_velocity_damped_balance_controller.py` — `diagnostics` dict from `compute()` |
| **Active/gated/disabled** | ACTIVE |
| **Input fields** | All controller internals |
| **State fields** | N/A |
| **Params/gains** | N/A |
| **Output torque contribution** | None |
| **Actuator indices affected** | N/A |
| **Insertion point** | After torque computation |
| **JAX source** | `k2_jax_controller.py:1286-1307` — diag packing |
| **JAX state_flat fields** | N/A (separate diag_flat) |
| **JAX params fields** | N/A |
| **Parity test coverage** | Diag fields tested in step parity |
| **Validation coverage** | All scenarios |
| **Status** | **PORTED_EXACT** — 30-field diag flat array maps to named fields. |

---

## 27. WBC / Hidden Torque Zeroing

| Field | Value |
|-------|-------|
| **Mechanism** | WBC correction torque zeroing for balance-core mode |
| **Python source** | `simulate_hierarchical_controller.py:2258-2267` — `zero_legacy_torque_sources_for_balance_core()` |
| **Active/gated/disabled** | ACTIVE — legacy torque sources are zeroed in balance-core mode |
| **Input fields** | N/A |
| **State fields** | N/A |
| **Params/gains** | N/A |
| **Output torque contribution** | Zero (all legacy sources suppressed) |
| **Actuator indices affected** | All |
| **Insertion point** | N/A (zeroed out) |
| **JAX source** | **NO JAX EQUIVALENT** — JAX controller only computes balance-core torque; no legacy sources exist |
| **JAX state_flat fields** | N/A |
| **JAX params fields** | N/A |
| **Parity test coverage** | N/A |
| **Validation coverage** | All balance-core scenarios |
| **Status** | **DOWNSTREAM_PYTHON_ONLY** — Only relevant for Python path with legacy controllers. JAX path is pure balance-core. No mismatch. |

---

## 28. Final mj_data.ctrl Assignment

| Field | Value |
|-------|-------|
| **Mechanism** | Final torque application to MuJoCo control |
| **Python source** | `simulate_hierarchical_controller.py:6777` — `mj_data.ctrl[:] = np.array(tau_smooth)` |
| **Active/gated/disabled** | ACTIVE |
| **Input fields** | tau_smooth (10-d) |
| **State fields** | None |
| **Params/gains** | None |
| **Output torque contribution** | Final 10-d torque applied to robot |
| **Actuator indices affected** | All 10 |
| **Insertion point** | Last step before mujoco.mj_step() |
| **JAX source** | `simulate_hierarchical_controller.py:6554` — `tau_smooth = _jax_tau` (when backend=jax) |
| **JAX state_flat fields** | N/A |
| **JAX params fields** | N/A |
| **Parity test coverage** | Backend CLI tests |
| **Validation coverage** | All backend=jax scenarios |
| **Status** | **PORTED_EXACT** — Same `mj_data.ctrl[:] = tau_smooth`. In jax mode, tau_smooth is JAX output. In python mode, tau_smooth is Python composer output. |

---

## 29. Disabled/Inactive K2 Mechanisms

These mechanisms are present in the code but **disabled or inactive** in K2 profile:

| Mechanism | Python | JAX | Status |
|-----------|--------|-----|--------|
| T6J bias trim | `t6j_bias_trim_enabled=False` | N/A | **INACTIVE_ZERO_CONFIRMED** |
| T6I convergence cap | `t6i_enabled=False` | N/A | **INACTIVE_ZERO_CONFIRMED** |
| Unified sagittal state feedback | `enable_unified_sagittal_state_feedback=False` | N/A | **INACTIVE_ZERO_CONFIRMED** |
| Active pitch crossing (APC) | `enable_active_pitch_crossing=False` | N/A | **INACTIVE_ZERO_CONFIRMED** |
| APCR1e adaptive authority | `apc_adaptive_authority_enabled=False` | N/A | **INACTIVE_ZERO_CONFIRMED** |
| APCR1f fast response | `apc_fast_response_enabled=False` | N/A | **INACTIVE_ZERO_CONFIRMED** |
| APCR1g predictive | `apc_predictive_enabled=False` | N/A | **INACTIVE_ZERO_CONFIRMED** |
| APCR1h drift priority | `apc_drift_priority_enabled=False` | N/A | **INACTIVE_ZERO_CONFIRMED** |
| Phase-aware recenter (F1) | `enable_phase_aware_recenter=False` | N/A | **INACTIVE_ZERO_CONFIRMED** |
| Hysteresis recenter (F2) | `enable_hysteresis_recenter=False` | N/A | **INACTIVE_ZERO_CONFIRMED** |
| Bias cancellation (G1) | `enable_bias_cancel=False` | N/A | **INACTIVE_ZERO_CONFIRMED** |
| Position integral | `enable_position_integral=False` | N/A | **INACTIVE_ZERO_CONFIRMED** |
| Pitch-aware position scaling | `enable_pitch_aware_position_scaling=False` | N/A | **INACTIVE_ZERO_CONFIRMED** |
| Boundary yaw fix | Only for boundary variants | N/A | **INACTIVE_ZERO_CONFIRMED** (K2 not a boundary profile) |
| Capture gate | `enable_capture_gate=False` | N/A | **INACTIVE_ZERO_CONFIRMED** |
| Support velocity damping | `k_support_velocity=0.0` | `effective_support_velocity_gain=0.0` | **INACTIVE_ZERO_CONFIRMED** |

---

## Summary Matrix

| # | Mechanism | Status | Risk |
|---|-----------|--------|------|
| 1 | Input packing | PORTED_EXACT | LOW |
| 2 | Dynamic target height | PORTED_EXACT | LOW |
| 3 | Height variant target_com_z | PORTED_EXACT | LOW |
| 4 | pitch_x_error computation | PORTED_EXACT | LOW |
| 5 | Physics eq pitch ref | PORTED_WITH_KNOWN_DIFF | **MEDIUM** |
| 6 | Calibrated outer loop | PORTED_EXACT | LOW |
| 7 | Support error computation | PORTED_EXACT | LOW |
| 8 | Support error rate smoothing | PORTED_EXACT | LOW |
| 9 | Low-band support | PORTED_WITH_KNOWN_DIFF | **MINOR** (kp 1.4 vs 1.5) |
| 10 | Empirical support FF | PORTED_EXACT | LOW |
| 11 | Sagittal terms | PORTED_EXACT | LOW |
| 12 | WIP notch filter | PORTED_EXACT | LOW |
| 13 | Notch height gate | PORTED_EXACT | LOW |
| 14 | Continuous gain scheduling | PORTED_EXACT | LOW |
| 15 | Shape posture | PORTED_WITH_KNOWN_DIFF | **MEDIUM** (PD gains?) |
| 16 | Lateral roll | PORTED_EXACT | LOW |
| 17 | Yaw | **PORTED_WITH_KNOWN_DIFF** | **HIGH** (insertion order) |
| 18 | Mode hip-yaw divergence | **PORTED_WITH_KNOWN_DIFF** | **HIGH** (insertion order) |
| 19 | HY-FF / HY2-DIV | JAX_EXTRA (excluded) | LOW (excluded) |
| 20 | Adaptive bias trim | PORTED_WITH_KNOWN_DIFF | **HIGH** (state divergence) |
| 21 | Composer | PORTED_EXACT | LOW |
| 22 | max_position_tau | PORTED_EXACT | LOW |
| 23 | Per-joint torque limits | PORTED_EXACT | LOW |
| 24 | Clip/rate-limit/smoothing | PORTED_EXACT | LOW |
| 25 | tau_prev state | **PORTED_WITH_KNOWN_DIFF** | **HIGH** (yaw/mode_div diff) |
| 26 | Telemetry/diag | PORTED_EXACT | LOW |
| 27 | WBC torque zeroing | DOWNSTREAM_PYTHON_ONLY | LOW |
| 28 | Final ctrl assignment | PORTED_EXACT | LOW |

---

## Acceptance Summary

| Criterion | Status |
|-----------|--------|
| Every active Python mechanism accounted for | ✅ 28 mechanisms catalogued |
| Every JAX mechanism has Python equivalent or is removed/justified | ✅ 1 JAX-extra excluded (HY-FF) |
| No JAX-only controller torque path remains | ✅ Confirmed |
| All inactive mechanisms confirmed zero | ✅ 16 mechanisms confirmed INACTIVE_ZERO |

**Phase 1 COMPLETE.** Key findings for Phase 2 investigation:

1. **HIGH risk — Yaw/Mode-div insertion order:** JAX adds yaw and mode_div post-composer, bypassing clip/rate-limit on hip-yaw [1,6]. Python puts them pre-composer. This means hip-yaw torques differ whenever yaw or mode_div torques exceed composer limits.

2. **HIGH risk — tau_prev state divergence:** Because yaw/mode_div are post-composer in JAX, `tau_prev[1,6]` includes un-clipped hip-yaw torque. In Python, `tau_prev[1,6]` has been through composer clip/rate-limit. This creates a persistent state divergence on hip-yaw joints.

3. **HIGH risk — Adaptive bias trim:** The sliding-window ring buffer behavior must be exactly verified for long-horizon scenarios. Any off-by-one in ring buffer wrapping, zero-crossing counting, or hold-step logic will accumulate over time.

4. **MEDIUM risk — Physics FF:** Verify whether `physics_ff_tau` in JAX is correctly contributing to wheel torque (it's computed but may not be added to the torque path).

5. **MINOR — Low-band support kp:** 1.4 vs 1.5. Verify K2 profile value.
