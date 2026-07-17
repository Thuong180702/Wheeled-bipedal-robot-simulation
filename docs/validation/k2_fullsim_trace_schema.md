# K2 Full-Sim Trace Schema

**Date:** 2026-06-30
**Phase:** 2 — DEFINE FULL TRACE SCHEMA

This schema defines every field that must be captured per control step for both source and dedicated paths to identify the first physics state divergence.

---

## Key Finding (Phase 1)

Both-synced controller comparison proves: **when given identical physics state, Python and JAX controllers produce torque outputs matching to within 1e-7 Nm at every step**. The pitch RMS gap is a physics/orchestration phenomenon, not a controller bug.

The trace schema therefore emphasizes **physics state comparison** between source and dedicated paths.

---

## Trace Structure

Each trace row is a JSON object with keys grouped by category. Rows are aligned by `control_step` (0-indexed) and optionally by `physics_substep` (0-indexed within the control step).

### A. Timing and Scenario Metadata

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `scenario_id` | str | — | Scenario identifier (e.g., "step_e/low_0p380") |
| `backend` | str | — | "source-python", "source-jax-mono", "dedicated-jax", or "both-synced" |
| `control_step` | int | — | Control step index (0-based) |
| `physics_substep` | int | — | Physics substep index within control step (0-based) |
| `sim_time` | float | s | Simulation time |
| `control_dt` | float | s | Control timestep (0.01) |
| `physics_dt` | float | s | Physics timestep (control_dt / n_substeps) |
| `qref_mode` | str | — | "static", "dynamic", or "original-k2-exact" |
| `height_ref_cmd` | float | m | Commanded target CoM height |
| `schedule_h` | float | m | Height used for scheduling (filtered or raw CoM z) |

### B. Raw MuJoCo State (sampled after physics substeps, before control)

| Field | Type | Shape | Unit | Description |
|-------|------|-------|------|-------------|
| `qpos` | float[] | (19,) | rad/m | Full generalized positions |
| `qvel` | float[] | (18,) | rad/s, m/s | Full generalized velocities |
| `qpos_joints` | float[] | (10,) | rad | Joint positions (actuated joints only) |
| `qvel_joints` | float[] | (10,) | rad/s | Joint velocities (actuated joints only) |
| `ctrl` | float[] | (10,) | Nm | Applied control (actuator force target) |
| `actuator_force` | float[] | (10,) | Nm | Actual actuator force (from mj_data.actuator_force) |
| `qfrc_actuator` | float[] | (18,) | — | Generalized actuator forces |
| `contact_count` | int | — | — | Number of active contacts |
| `contact_geom1` | str[] | — | — | First geom in each contact pair |
| `contact_geom2` | str[] | — | — | Second geom in each contact pair |
| `contact_force_normal` | float[] | — | N | Normal force for each contact |
| `solver_iterations` | int | — | — | MuJoCo solver iterations |
| `solver_stat` | float | — | — | Solver convergence statistic if available |

### C. Body and Kinematics

| Field | Type | Shape | Unit | Description |
|-------|------|-------|------|-------------|
| `body_quat` | float[] | (4,) | — | Torso quaternion (w,x,y,z) |
| `body_xpos` | float[] | (3,) | m | Torso world position |
| `body_xvelp` | float[] | (3,) | m/s | Torso linear velocity |
| `body_xvelr` | float[] | (3,) | rad/s | Torso angular velocity |
| `com_pos` | float[] | (3,) | m | Center of mass world position |
| `com_vel` | float[] | (3,) | m/s | Center of mass world velocity |
| `wheel_pos_left` | float[] | (3,) | m | Left wheel world position |
| `wheel_pos_right` | float[] | (3,) | m | Right wheel world position |
| `wheel_vel_left` | float | rad/s | Left wheel angular velocity |
| `wheel_vel_right` | float | rad/s | Right wheel angular velocity |
| `support_center_x` | float | m | Support center x in world frame |
| `support_center_y` | float | m | Support center y in world frame |
| `support_error` | float | m | Support position error (sagittal) |
| `support_velocity` | float | m/s | Support center velocity |
| `contact_valid` | int | — | 1 if both wheels in contact with valid force, 0 otherwise |

### D. Orientation (derived from quaternion)

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `body_pitch_x` | float | rad | Pitch angle (rotation about body x-axis) |
| `body_roll_y` | float | rad | Roll angle (rotation about body y-axis) |
| `body_yaw_z` | float | rad | Yaw angle |
| `pitch_rate` | float | rad/s | Pitch angular velocity |
| `roll_rate` | float | rad/s | Roll angular velocity |
| `yaw_rate` | float | rad/s | Yaw angular velocity |
| `yaw_error` | float | rad | Yaw deviation from initial heading |
| `gravity_body` | float[] | (3,) | m/s² | Gravity vector in body frame |

### E. Controller Input (45-element JAX input vector, expanded)

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `input_pitch_x_rad` | float | rad | Pitch angle input |
| `input_pitch_rate_x_rad_s` | float | rad/s | Pitch rate input (post-notch, boosted) |
| `input_roll_y_rad` | float | rad | Roll angle input |
| `input_roll_rate_y_rad_s` | float | rad/s | Roll rate input |
| `input_yaw_error_rad` | float | rad | Yaw error from initial heading |
| `input_yaw_rate_rad_s` | float | rad/s | Yaw rate |
| `input_com_z_m` | float | m | CoM height |
| `input_com_vy_m_s` | float | m/s | CoM lateral velocity |
| `input_sagittal_velocity_m_s` | float | m/s | Sagittal velocity |
| `input_sagittal_position_error_m` | float | m | Sagittal position error |
| `input_wheel_vel_left_rad_s` | float | rad/s | Left wheel velocity |
| `input_wheel_vel_right_rad_s` | float | rad/s | Right wheel velocity |
| `input_support_velocity_m_s` | float | m/s | Support velocity |
| `input_commanded_height_ref_m` | float | m | Commanded height reference |
| `input_hip_yaw_div_error` | float | rad | Hip yaw divergence |
| `input_hip_yaw_div_rate` | float | rad/s | Hip yaw divergence rate |
| `input_q_ref` | float[] | (10,) | rad | Equilibrium joint positions |
| `input_support_position_error_m` | float | m | Support position error (redundant) |
| `input_contact_valid` | float | — | Contact validity (0.0 or 1.0) |

### F. Controller State (836-element JAX state vector, key fields)

| Field | Type | Index | Description |
|-------|------|-------|-------------|
| `state_notch_x1` | float | 0 | Notch filter x1 |
| `state_notch_x2` | float | 1 | Notch filter x2 |
| `state_notch_y1` | float | 2 | Notch filter y1 |
| `state_notch_y2` | float | 3 | Notch filter y2 |
| `state_prev_tau` | float[] | 4-13 | Previous torque vector (10,) |
| `state_filtered_com_z` | float | 14 | Filtered CoM height |
| `state_prev_support_error` | float | 15 | Previous support position error |
| `state_ol_pitch_ref_smoothed` | float | 16 | Calibrated outer loop pitch ref |
| `state_ol_prev_support_error` | float | 17 | OL previous support error |
| `state_ol_support_error_rate` | float | 18 | OL support error rate |
| `state_abs_slow_sum` | float | 19 | ABS slow window sum |
| `state_abs_fast_sum` | float | 20 | ABS fast window sum |
| `state_abs_trim_tau` | float | 21 | ABS trim torque |
| `state_abs_hold_steps` | int | 22 | ABS hold steps |
| `state_abs_prev_err_sign` | int | 23 | ABS previous error sign |
| `state_abs_zc_count` | int | 24 | ABS zero-crossing count |
| `state_abs_slow_count` | int | 25 | ABS slow count |
| `state_abs_slow_ptr` | int | 26 | ABS slow buffer pointer |
| `state_abs_guard_trigger` | int | 27 | ABS guard trigger count |
| `state_abs_ring_buffer` | float[] | 28-327 | ABS ring buffer (300 entries) |
| `state_abs_zc_buf_count` | int | 328 | ABS ZC buffer count |
| `state_abs_zc_buf_ptr` | int | 329 | ABS ZC buffer pointer |
| `state_abs_zc_buffer` | float[] | 330-829 | ABS ZC buffer (500 entries) |
| `state_apcr1nd_step` | float | 830 | APCR1ND step counter |
| `state_apcr1nd_prev_err` | float | 831 | APCR1ND previous error |
| `state_apcr1nd_conv_steps` | float | 832 | APCR1ND converging steps |
| `state_apcr1nd_held` | float | 833 | APCR1ND recenter held |
| `state_eff_max_pos_tau` | float | 834 | Effective max position tau |
| `state_wd_override` | float | 835 | Wheel damping override active |

### G. Controller Output

| Field | Type | Shape | Unit | Description |
|-------|------|-------|------|-------------|
| `tau_final` | float[] | (10,) | Nm | Final torque applied to actuators |
| `tau_prev` | float[] | (10,) | Nm | Previous torque (rate limiter init) |
| `tau_before_clip` | float[] | (10,) | Nm | Torque sum before clip |
| `tau_after_clip` | float[] | (10,) | Nm | Torque after clip |
| `tau_after_rate_limit` | float[] | (10,) | Nm | Torque after rate limiter |

### H. Torque Components (from controller diagnostics)

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `tau_pitch` | float | Nm | Pitch damping torque |
| `tau_pitch_rate` | float | Nm | Pitch rate damping torque (post-notch) |
| `tau_sagittal_velocity` | float | Nm | Sagittal velocity damping |
| `tau_support_velocity` | float | Nm | Support velocity damping |
| `tau_position` | float | Nm | Position error torque (post-ABS, post-APCR1ND) |
| `tau_wheel_vel_left` | float | Nm | Left wheel velocity damping |
| `tau_wheel_vel_right` | float | Nm | Right wheel velocity damping |
| `tau_shape_posture` | float[] | Nm | Shape/posture PD torque (10,) |
| `tau_support_ff` | float[] | Nm | Support feedforward torque (10,) |
| `tau_lateral_roll` | float[] | Nm | Lateral roll torque (10,) |
| `tau_yaw` | float[] | Nm | Yaw torque (10,) |
| `tau_mode_div` | float[] | Nm | Mode-div torque (10,) |
| `tau_empirical_ff` | float[] | Nm | Empirical support FF (10,) |

### I. Height-Dependent Schedule Values (runtime)

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `sched_k_position` | float | Nm/m | Scheduled position gain |
| `sched_k_velocity` | float | Nm/(m/s) | Scheduled velocity gain |
| `sched_k_wheel_velocity` | float | Nm/(rad/s) | Scheduled wheel velocity gain |
| `sched_kd_pitch` | float | Nm/(rad/s) | Pitch rate damping gain |
| `sched_max_position_tau` | float | Nm | Max position torque cap |
| `sched_velocity_damping_scale` | float | — | Velocity damping scale factor |
| `sched_notch_gate` | float | — | Notch filter height gate |
| `sched_pitch_eq` | float | rad | Pitch equilibrium offset |
| `sched_physics_ff_tau` | float | Nm | Physics feedforward torque |
| `sched_cal_kp` | float | deg/m | Calibrated outer loop Kp |
| `sched_cal_kd` | float | deg/(m/s) | Calibrated outer loop Kd |
| `sched_cal_theta_max` | float | deg | Calibrated outer loop theta max |
| `sched_cal_deadband` | float | m | Calibrated outer loop deadband |
| `sched_cal_rate_limit` | float | deg/s | Calibrated outer loop rate limit |
| `sched_cal_lowpass_alpha` | float | — | Calibrated outer loop lowpass |
| `sched_low_band_gate` | float | — | Low-band support gate |
| `sched_low_band_ref` | float | deg | Low-band support pitch ref |
| `sched_low_band_theta_max` | float | deg | Low-band support theta max |

### J. Pitch/Sagittal Internals

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `pitch_error` | float | rad | Effective pitch error (body_pitch - pitch_eq - pitch_ref_offset) |
| `pitch_rate_error` | float | rad/s | Notch-filtered pitch rate |
| `sagittal_position_error_raw` | float | m | Raw sagittal position error |
| `sagittal_position_error_compensated` | float | m | Yaw-compensated sagittal error |
| `sagittal_velocity` | float | m/s | Sagittal velocity |
| `tau_position_raw` | float | Nm | Raw position torque (before ABS) |
| `tau_position_after_abs` | float | Nm | Position torque after ABS trim |
| `tau_position_after_apcr` | float | Nm | Position torque after APCR1ND cap |
| `tau_position_after_cap` | float | Nm | Position torque after final cap |

### K. ABS Trim Internals

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `abs_signed_error` | float | m | Signed position error (input to ABS) |
| `abs_slow_mean` | float | m | Slow window mean error |
| `abs_fast_mean` | float | m | Fast window mean error |
| `abs_sign_err` | float | — | Sign of current error |
| `abs_raw_target` | float | Nm | Raw ABS target torque |
| `abs_clipped_target` | float | Nm | Clipped ABS target |
| `abs_is_decay` | int | — | 1 if ABS is in decay mode |
| `abs_rate` | float | Nm/step | ABS adaptation rate |
| `abs_trim_delta` | float | Nm | ABS trim change this step |
| `abs_new_trim` | float | Nm | New ABS trim value |
| `abs_safety_pass` | int | — | 1 if all safety gates pass |
| `abs_external_trim` | float | Nm | External position trim applied |
| `abs_hold_steps` | int | — | Hold steps remaining |
| `abs_zc_guard_active` | int | — | 1 if ZC guard active |
| `abs_near_zero` | int | — | 1 if error near zero |
| `abs_in_hysteresis` | int | — | 1 if in hysteresis zone |

### L. APCR1ND Internals

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `apcr1nd_active` | int | — | 1 if APCR1ND gate active |
| `apcr1nd_step_counter` | float | — | Steps since gate activated |
| `apcr1nd_prev_error` | float | m | Previous position error |
| `apcr1nd_converging` | float | — | Converging steps count |
| `apcr1nd_held` | float | — | Recenter held |
| `apcr1nd_safety` | float | — | Safety gate status |
| `apcr1nd_wd_apply` | float | — | Wheel damping override applied |
| `apcr1nd_wd_scale` | float | — | Wheel damping override scale |
| `apcr1nd_boosted_cap` | float | Nm | Boosted position cap |

### M. Metrics Accumulators (running)

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `running_pitch_rms` | float | deg | Running pitch RMS |
| `running_support_rms` | float | m | Running support error RMS |
| `running_height_rmse` | float | m | Running height RMSE |
| `running_pitch_max` | float | deg | Running pitch max |
| `running_pitch_min` | float | deg | Running pitch min |
| `running_hip_yaw_max` | float | rad | Running hip yaw joint max |

---

## Comparison Priority Order (for first-divergence analyzer)

When comparing source vs dedicated traces, check fields in this order:

1. **Physics state first:** qpos[0:19], qvel[0:18] — if these diverge at substep N, all downstream fields are consequences
2. **Kinematics:** body_pitch_x, body_roll_y, com_z, support_center — derived from physics
3. **Controller input:** input_* fields — derived from physics + kinematics
4. **Height schedules:** sched_* — derived from filtered_com_z / height_ref
5. **Controller state:** state_* — updated from previous step
6. **Torque components:** tau_* components — computed from input + state + schedules
7. **Final torque:** tau_final — composed from components
8. **Metrics:** running_* — accumulated from physics

If physics state diverges at substep 0 of step 0, the root cause is initialization.
If physics state matches at step 0 but diverges later, trace substep-by-substep.

---

## Tolerance Levels

| Category | Tolerance | Rationale |
|----------|-----------|-----------|
| Exact fields (integers, flags) | 0 | Must be identical |
| qpos/qvel (float64) | 1e-12 rad/m | MuJoCo double precision |
| Controller state (float64) | 1e-12 | JAX float64 |
| Torque | 1e-9 Nm | Controller parity proven at 1e-7 |
| Orientation (rad) | 1e-12 rad | Pre-divergence |
| Schedule values (float64) | 1e-12 | Grid interpolation |
| Pitch RMS (deg) | 1.0° absolute OR 0.3× relative | From k2_original_metrics.json |

---

## Acceptance

- [x] Trace schema covers every control-affecting scalar
- [x] Physics state is prioritized (major finding from Phase 1)
- [x] Controller state and internals still covered for completeness
- [x] All units documented
- [x] Comparison priority order defined
- [x] Tolerance levels specified
- [x] Schema accounts for the finding that controllers are source-equivalent
