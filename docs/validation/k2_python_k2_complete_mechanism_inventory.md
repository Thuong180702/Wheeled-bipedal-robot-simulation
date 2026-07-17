# K2 Python Complete Mechanism Inventory

**Date:** 2026-06-27
**Profile:** k2_notch_low_q_v1
**Controller Mode:** balance-core
**Sagittal Controller:** SagittalVelocityDampedBalanceController
**Status:** Complete mechanism enumeration

---

## 1. Input / State Extraction Mechanisms

### M1 — Physical State Extraction
- **Python file:** `scripts/simulate_hierarchical_controller.py:5770-5790`
- **Function:** `CentroidalStateEstimator.update()`
- **Status:** ACTIVE
- **Inputs:** `mj_data` (MuJoCo state)
- **Outputs:** `centroidal_state_control` object with body orientation, CoM position/velocity, joint positions/velocities
- **Affected indices:** All
- **Pipeline position:** Pre-controller, every step
- **Scenario:** All

### M2 — Joint Index Mapping (10-DOF)
- **Python file:** `wheeled_biped/controllers/balance_core_types.py:1-30`
- **Status:** ACTIVE
- **Constants:** `WHEEL_INDICES=[4,9]`, `HIP_ROLL_INDICES=[0,5]`, `HIP_YAW_INDICES=[1,6]`, `HIP_PITCH_KNEE_INDICES=[2,3,7,8]`, `SUPPORT_SHAPE_INDICES=[1,2,3,6,7,8]`
- **Pipeline position:** Definition-time
- **Scenario:** All

### M3 — Contact Detection
- **Python file:** `scripts/simulate_hierarchical_controller.py:5898-5904`
- **Function:** `ContactSupervisor.update()`
- **Status:** ACTIVE
- **Inputs:** left/right wheel contact, contact force valid, normal forces
- **Outputs:** `contact_output` with contact flags and forces
- **Pipeline position:** Pre-controller, every step
- **Scenario:** All

### M4 — Capture Point Estimation
- **Python file:** `scripts/simulate_hierarchical_controller.py:5800-5810`
- **Function:** `CapturePointEstimator.update()`
- **Status:** ACTIVE (diagnostic only — capture gate disabled in K2)
- **Inputs:** CoM position, velocity, gravity
- **Outputs:** capture_point, cp_error_y_m
- **Affected indices:** Diagnostic only (wheels 4,9 if capture gate was enabled)
- **Pipeline position:** Pre-controller, every step
- **Scenario:** All

### M5 — Support Center Computation
- **Python file:** `scripts/simulate_hierarchical_controller.py:5980-5982`
- **Function:** `compute_support_center_xy()`
- **Status:** ACTIVE
- **Inputs:** left/right wheel body world positions
- **Outputs:** `support_center_ctrl_xy` (x, y)
- **Pipeline position:** Pre-controller, every step
- **Scenario:** All

### M6 — Sagittal Projection
- **Python file:** `scripts/simulate_hierarchical_controller.py:5991-6014`
- **Functions:** `project_sagittal_displacement()`, `project_sagittal_velocity()`
- **Status:** ACTIVE
- **Inputs:** support_center, equilibrium_support_center, sagittal_axis, com_position, com_velocity
- **Outputs:** `sag_pos_error` (m), `sag_vel` (m/s)
- **Pipeline position:** Pre-controller, every step
- **Scenario:** All

### M7 — Target Height / Commanded Height
- **Python file:** `scripts/simulate_hierarchical_controller.py:5348-5354`, `height_variant_setup`
- **Status:** ACTIVE
- **Inputs:** `height_cmd` (default 0.40 m), or `height_variant_setup["target_com_z_m"]`
- **Outputs:** `commanded_height_ref_m`
- **Pipeline position:** Init-time / dynamic update
- **Scenario:** All (dynamic height: varies per step)

### M8 — Dynamic Height Command Update
- **Python file:** `scripts/simulate_hierarchical_controller.py:5351-5355`
- **Status:** ACTIVE (when `dynamic_height_traj` is provided)
- **Inputs:** `dynamic_height_traj` trajectory array
- **Outputs:** `dynamic_height_target_m`, `dynamic_height_actual_m`
- **Pipeline position:** Every step, pre-controller
- **Scenario:** Dynamic-height-specific

### M9 — Pitch Reference Offset Generation
- **Python file:** `scripts/simulate_hierarchical_controller.py:6117-6118`
- **Status:** ACTIVE
- **Inputs:** `pitch_x_eq` (equilibrium pitch), `outer_loop_pitch_ref_total_deg` (sum of physics FF + outer loop + low-band)
- **Outputs:** `pitch_x_ref` (rad), `pitch_x_error` (rad)
- **Formula:** `pitch_x_error = body_pitch_x - (pitch_eq + total_offset)`
- **Pipeline position:** Pre-sagittal controller, every step
- **Scenario:** All

### M10 — q_ref Generation
- **Python file:** `scripts/simulate_hierarchical_controller.py:5720-5750`
- **Status:** ACTIVE
- **Inputs:** Equilibrium keyframe joint positions
- **Outputs:** `equilibrium_joint_pos` (10-D array)
- **Pipeline position:** Init-time (equilibrium solve) + per-step
- **Scenario:** All

### M11 — Torque Limit and Rate-Limit Params
- **Python file:** `scripts/simulate_hierarchical_controller.py:4658-4661`
- **Status:** ACTIVE
- **Values:** torque_limit per-joint, max_torque_rate = 400 Nm/s × 10, control_dt = 0.01 s
- **Pipeline position:** Init-time
- **Scenario:** All

---

## 2. Sagittal Wheel Balance Mechanisms

### S1 — Notch Filter (Biquad, 2.5 Hz, Q=2.0)
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()` line ~8500+
- **Class:** `BiquadNotchFilter` (from `signal_filters.py`)
- **Status:** ACTIVE (`enable_wip_notch_filter=True`)
- **Inputs:** `pitch_rate_x_rad_s`
- **State:** `x1, x2, y1, y2` (Direct Form II Transposed)
- **Parameters:** fs=100 Hz, fc=2.5 Hz, Q=2.0, blend=1.0
- **Outputs:** `pitch_rate_eff` (filtered pitch rate)
- **Gate:** Height gate 0.42→0.48 m smoothstep; `pitch_rate_eff = (1-gate)*raw + gate*notch_output`
- **Pipeline position:** Inside `SagittalVelocityDampedBalanceController.compute()`
- **Scenario:** All (notch gate engages above 0.42 m)

### S2 — Notch Height Gate
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()`
- **Function:** `smoothstep_gate(height_ref, 0.42, 0.48)`
- **Status:** ACTIVE (`wip_notch_gate_enabled=True`)
- **Inputs:** `height_ref` (commanded height)
- **Outputs:** Gate value [0,1] — 0 at ≤0.42 m, 1 at ≥0.48 m
- **Pipeline position:** Inside sagittal controller, applied to pitch_rate_eff blend
- **Scenario:** All (affects behavior above 0.42 m)

### S3 — Pitch Torque (tau_pitch)
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()`
- **Status:** ACTIVE
- **Formula:** `tau_pitch_raw = kp_pitch * pitch_x_error`
- **Parameters:** `kp_pitch=50.0 Nm/rad`, `effective_pitch_scale=1.0`, `pitch_tau_cap=0.0` (uncapped)
- **Clamping:** If `pitch_tau_cap > 0`: clamp to ±cap. K2: cap=0.0 → no clamping.
- **Sign:** Positive pitch_x → positive wheel torque (forward)
- **Outputs:** tau on indices [4,9]
- **Pipeline position:** Inside sagittal controller

### S4 — Pitch-Rate Torque (tau_pitch_rate)
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()`
- **Status:** ACTIVE
- **Formula:** `tau_pitch_rate = kd_pitch * pitch_rate_eff`
- **Parameters:** `kd_pitch=10.0 Nm/(rad/s)` (K2: not scheduled, `continuous_kd_pitch=False`)
- **Sign:** Damping opposes pitch rate
- **Outputs:** tau on indices [4,9]
- **Pipeline position:** Inside sagittal controller

### S5 — Sagittal Velocity Torque (tau_sagittal_velocity)
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()`
- **Status:** ACTIVE
- **Formula:** `tau_sagittal_velocity = -k_velocity * sagittal_velocity_m_s`
- **Parameters:** `k_velocity=15.0 Nm/(m/s)` (K2: not scheduled, `continuous_k_velocity=False`)
- **Sign:** Negative (opposes forward velocity)
- **Outputs:** tau on indices [4,9]
- **Pipeline position:** Inside sagittal controller

### S6 — Position Torque (tau_position)
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()`
- **Status:** ACTIVE
- **Formula:** `tau_position = -k_position * sagittal_position_error + position_integral + external_position_trim`
- **Parameters:** `k_position=40.0 Nm/m` (K2: not scheduled, `continuous_k_position=False`)
- **Integral:** Disabled in K2 (`enable_position_integral=False`)
- **External trim:** `adaptive_bias_trim` contribution (see ABS mechanism)
- **Clamping:** Position tau clamped to `±effective_max_position_tau`, also budget-clipped if enabled
- **Sign:** Negative (pulls toward equilibrium position)
- **Outputs:** tau on indices [4,9]
- **Pipeline position:** Inside sagittal controller

### S7 — Wheel Velocity Damping (tau_wheel_vel)
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()`
- **Status:** ACTIVE
- **Formula:** `tau_wheel_vel_L = -k_wheel_velocity * wheel_vel_L; tau_wheel_vel_R = -k_wheel_velocity * wheel_vel_R`
- **Parameters:** `k_wheel_velocity=0.5 Nm/(rad/s)` (K2: not scheduled)
- **Sign:** Negative (damps wheel velocity)
- **Outputs:** Individual left/right on indices [4,9]; added to common torque component
- **Pipeline position:** Inside sagittal controller

### S8 — Support Velocity Torque (tau_support_velocity)
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()`
- **Status:** DISABLED in K2 (`k_support_velocity=0.0`)
- **Formula:** `tau_support_velocity = -k_support_velocity * support_velocity`
- **Parameters:** `k_support_velocity=0.0` → always zero
- **Outputs:** Zero on indices [4,9]

### S9 — COM Velocity Damping (tau_com_vy)
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()`
- **Status:** ACTIVE
- **Formula:** `tau_com_vy = -kd_com_vy * sagittal_velocity_m_s`
- **Parameters:** `kd_com_vy=5.0 Nm/(m/s)`
- **Sign:** Negative
- **Outputs:** tau on indices [4,9]
- **Note:** This is separate from `k_velocity` — provides additional CoM velocity damping

### S10 — Capture Point Torque (tau_cp)
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()`
- **Status:** DISABLED in K2 (`kp_cp=0.0`)
- **Formula:** `tau_cp = -kp_cp * sagittal_position_error`
- **Parameters:** `kp_cp=0.0` → always zero
- **Outputs:** Zero on indices [4,9]

### S11 — Common Torque Assembly
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()`
- **Status:** ACTIVE
- **Formula:** `tau_common = sign * (tau_pitch + tau_pitch_rate + tau_sagittal_velocity + tau_support_velocity + tau_position + tau_cp + tau_com_vy)`
- **Outputs:** Common component → left wheel = tau_common + tau_wheel_vel_L; right wheel = tau_common + tau_wheel_vel_R
- **Parameters:** `wheel_torque_sign=1.0`
- **Pipeline position:** Final sagittal assembly

### S12 — Max Position Tau Scheduling
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()`
- **Status:** ACTIVE (`continuous_max_position_tau=True`)
- **Formula:** Monotonic piecewise-linear: `max_pos_tau(z) = 4.0 + (6.0-4.0)*(z-0.393)/(0.300-0.393)` for z ∈ [0.300, 0.393]
- **Below 0.300 m:** 6.0 Nm; **Above 0.393 m:** 4.0 Nm
- **Pipeline position:** Applied as clamp on tau_position

### S13 — Height Schedule (filtered_com_z)
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()`
- **Status:** ACTIVE
- **Formula:** `schedule_h = commanded_height if >0 else 0.9*filtered_com_z + 0.1*com_z`
- **State:** `filtered_com_z` updated each step
- **Pipeline position:** First step inside sagittal compute

### S14 — Sagittal Torque Sign Convention
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()`
- **Status:** ACTIVE
- **Convention:** `wheel_torque_sign=+1.0` → positive pitch → positive wheel torque → forward wheel spin
- **Outputs:** Common torque multiplied by sign before assignment

---

## 3. Outer-Loop / Support Mechanisms

### O1 — Calibrated Outer Loop (v2)
- **Python file:** `calibrated_outer_loop_functions_v2.py`
- **Status:** ACTIVE (`calibrated_outer_loop_function_version="v2"`)
- **Function:** PCHIP interpolation of Kp, Kd, Ki, theta_max, deadband, rate_limit, lowpass_alpha across 10 height breakpoints
- **Pipeline position:** Python-only, called once per step to get height-dependent gains

### O2 — Support Error Rate Smoothing
- **Python file:** `scripts/simulate_hierarchical_controller.py:6035-6046`
- **Status:** ACTIVE
- **Formula:** Numerical derivative of support_error, low-pass filtered with calibrated alpha
- **State:** `outer_loop_support_error_rate_smoothed`, `outer_loop_prev_support_error_m`
- **Pipeline position:** Pre-outer-loop, every step

### O3 — Outer Loop PID Pitch Ref
- **Python file:** `scripts/simulate_hierarchical_controller.py:6078-6087`
- **Function:** `compute_outer_loop_pitch_ref()`
- **Status:** ACTIVE
- **Formula:** PD(+I) with deadband: `dynamic = Kp*error_p + Kd*error_rate + Ki*integral`, saturated to `±theta_ref_max`
- **Pipeline position:** Inside outer-loop block

### O4 — Outer Loop Rate Limiting + Low-Pass
- **Python file:** `scripts/simulate_hierarchical_controller.py:6095-6107`
- **Functions:** `apply_rate_limit()`, `apply_lowpass()`
- **Status:** ACTIVE
- **Outputs:** `outer_loop_pitch_ref_smoothed_deg`
- **State:** `outer_loop_pitch_ref_smoothed_deg`
- **Pipeline position:** Post-PID, every step

### O5 — Outer Loop Safety Gates
- **Python file:** `scripts/simulate_hierarchical_controller.py:6050-6068`
- **Status:** ACTIVE
- **Gates:** Contact valid, abs(support_error) ≤ disable_abs_error, pitch ≤ disable_pitch_deg, roll ≤ disable_roll_deg
- **Behavior:** If any gate fails, target_dynamic_deg = 0.0 (decay toward zero)

### O6 — Physics Equilibrium Feedforward
- **Python file:** `physics_equilibrium_feedforward.py`
- **Status:** ACTIVE
- **Function:** PCHIP interpolation of per-wheel equilibrium torque across 10 heights
- **Outputs:** `tau_eq_ff_each_wheel_nm`, `pitch_eq_no_off_deg`
- **Pipeline position:** Python-only, called once per step
- **Note:** Used as pitch_ref offset equivalent (Option B), not direct torque injection (Option A FAILED)

### O7 — Low-Band Support Outer Loop
- **Python file:** `scripts/simulate_hierarchical_controller.py:5020-5035`, `sagittal_velocity_damped_balance_controller.py`
- **Status:** ACTIVE (`low_band_support_outer_loop_enabled=True`)
- **Formula:** Gaussian gate centered at 0.320 m, sigma 0.004 m: `Kp_eff = 1.4 * exp(-0.5*((h-0.320)/0.004)^2)`
- **Adds:** Pitch ref offset peak of 1.0 deg at center height
- **Pipeline position:** Python-only, computed per step

### O8 — Support Reference / Support Center Logic
- **Python file:** `scripts/simulate_hierarchical_controller.py:5980-6001`
- **Status:** ACTIVE
- **Details:** Support position error = wheel midpoint position vs equilibrium midpoint, projected onto sagittal axis
- **Pipeline position:** Pre-controller, every step

---

## 4. Adaptive Bias Trim (ABS) Mechanisms

### A1 — Sliding Window Ring Buffer
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()` ~line 5500-5700
- **Status:** ACTIVE
- **Parameters:** Slow window=300 steps, fast window=100 steps
- **State:** Ring buffer (300 entries), running sum, count, write pointer
- **Updates:** Push signed_position_error each step

### A2 — Slow/Fast Mean Computation
- **Status:** ACTIVE
- **Slow mean:** Average over full 300-step window
- **Fast mean:** Average over most recent 100 entries of slow buffer
- **Outputs:** `slow_mean`, `fast_mean` (m)

### A3 — Zero-Crossing Detection
- **Status:** ACTIVE (`adaptive_bias_zero_crossing_guard_enabled=True`)
- **Parameters:** Window=500, limit=8 crossings, max_scale=0.5
- **Formula:** Count sign changes in sliding window; if > 8, max_tau reduced to 50%
- **State:** zc_count, guard_trigger counter

### A4 — Trim Tau Update (Proportional)
- **Status:** ACTIVE
- **Formula:** `raw_target = -k_tau * (mean_err - sign*exit_threshold)` with entry/exit hysteresis
- **Parameters:** k_tau=5.0 Nm/m, enter_threshold=0.035 m, exit_threshold=0.012 m, relief_hysteresis=0.005 m
- **State:** trim_tau (Nm)

### A5 — Height-Scheduled Max Trim
- **Status:** ACTIVE
- **Formula:** Piecewise-linear: max_tau(z) = 0.35 Nm (at 0.38 m) → 0.50 Nm (at 0.48 m) → 0.55 Nm (at 0.52 m)
- **Applied as:** Clamp on trim_tau before rate limiting

### A6 — Asymmetric Rate Limiting
- **Status:** ACTIVE
- **Parameters:** Rate=0.006 Nm/step (increase), decay_rate=0.018 Nm/step (decrease, 3x faster)
- **Applied to:** trim_tau update

### A7 — Sign-Reversal Hold
- **Status:** ACTIVE
- **Parameters:** hold_steps=100
- **Behavior:** When mean error sign changes, freeze trim update for 100 steps

### A8 — Safety Gates
- **Status:** ACTIVE
- **Gates:**
  - `pitch_ok`: abs(pitch) ≤ 12° (adaptive_bias_disable_if_pitch_gt_deg)
  - `roll_ok`: abs(roll) ≤ 5° (adaptive_bias_disable_if_roll_gt_deg)
  - `contact_ok`: always True from Python perspective (true from sim)
  - `abs_error_ok`: abs(sag_pos_error) ≤ 0.24 m
  - `hip_yaw_ok`: max hip-yaw deviation ≤ 0.25 rad
  - `upright_ok`: pitch_ok AND roll_ok (adaptive_bias_only_when_upright=True)
- **Output:** trim_to_apply = new_trim if safety_pass else 0.0

---

## 5. Leg/Body Controller Mechanisms

### L1 — Shape/Posture PD Control
- **Python file:** `shape_posture_controller.py`, called at sim line 5914
- **Status:** ACTIVE
- **Formula:** PD on hip-yaw [1,6], hip-pitch [2,7], knee [3,8]
- **Parameters:** kp_hip_yaw=15.0, kd_hip_yaw=3.0, kp_hip_pitch=30.0, kd_hip_pitch=4.0, kp_knee=40.0, kd_knee=5.0
- **Hip-roll [0,5]:** kp=0.0, kd=0.0 (zero — owned by lateral roll)
- **Wheels [4,9]:** Zero (never written)
- **Pipeline position:** Inside balance-core block, before composer

### L2 — HY-FF (Hip-Yaw Support Feedforward)
- **Python file:** `shape_posture_controller.py` (inside compute)
- **Status:** DISABLED in K2 (`enable_hip_yaw_support_feedforward=False`)
- **Note:** K2 balance-core is built without HY-FF. The ShapePostureController has the code but it's disabled.

### L3 — HY2-DIV (Hip-Yaw Divergence Damping)
- **Python file:** `shape_posture_controller.py` (inside compute)
- **Status:** DISABLED in K2 (`enable_hip_yaw_divergence_damping=False`)
- **Note:** Disabled. Additionally, the imported module `mode_based_hip_yaw_divergence_controller` may not exist.

### L4 — Lateral Roll Balance
- **Python file:** `lateral_roll_balance_controller.py`, called at sim line 6302
- **Status:** ACTIVE
- **Formula:** `m_roll = kp_roll * roll_y + kd_roll * roll_rate`, antisymmetric on [0,5]
- **Parameters:** kp_roll=40.0, kd_roll=8.0, max_roll_moment=50.0, hip_roll_torque_sign=1.0
- **Stance regularization:** ENABLED (hip_roll_pos/vel/ref always passed)
- **Pipeline position:** Inside balance-core block

### L5 — Yaw Control
- **Python file:** `yaw_controller.py`, called at sim line 6328 or 6349
- **Status:** ACTIVE
- **Formula:** Antisymmetric hip-yaw torque: `tau = ±clamp(kp*yaw_err - kd*yaw_rate, ±max)`
- **Parameters:** kp_yaw=8.0, kd_yaw=2.0, max_yaw_torque=5.0
- **Pipeline position:** Inside balance-core block, added to tau_shape_posture before composer

### L6 — Mode Hip-Yaw Divergence (CLI opt-in)
- **Python file:** `mode_hip_yaw_divergence_controller.py`, called at sim line 6385
- **Status:** OPT-IN via `--enable-mode-hip-yaw-divergence`
- **Note:** NOT active in default K2. Enabled via CLI flag for certain validation scenarios.

### L7 — Empirical Support Feedforward
- **Python file:** `support_feedforward_controller.py`, called at sim line 5954
- **Status:** ACTIVE
- **Vector:** `[0, 0, 4.1, -15.5, 0, 0, 0, 3.2, -15.8, 0] × 0.5` scale
- **Applied to:** hip_pitch [2,7] and knee [3,8]
- **Pipeline position:** Inside balance-core block, one of four torque sources

### L8 — Wheel Yaw Stabilizer
- **Python file:** `differential_wheel_yaw_stabilizer.py`, optionally activated
- **Status:** DISABLED in K2 (only for M-family profiles)
- **Note:** Post-composer additive torque on wheels

---

## 6. Composer / Final Torque Mechanisms

### C1 — Four-Source Summation
- **Python file:** `balance_core_torque_composer.py:compose()`
- **Status:** ACTIVE
- **Formula:** `tau_total_raw = tau_shape_posture + tau_support_feedforward + tau_sagittal_wheel_balance + tau_lateral_roll_balance`
- **Note:** Yaw and mode-div are pre-added to tau_shape_posture before composer

### C2 — Actuator Torque Clipping
- **Python file:** `balance_core_torque_composer.py:compose()`
- **Status:** ACTIVE
- **Formula:** `tau_total_clipped = clip(tau_total_raw, -torque_limit, torque_limit)`
- **Saturation detection:** `|tau_total_raw - tau_total_clipped| > 1e-9`

### C3 — Rate Limiting
- **Python file:** `balance_core_torque_composer.py:compose()`
- **Status:** ACTIVE
- **Formula:** `delta = clip((tau_clipped - tau_prev)/dt, -max_rate, max_rate); tau_final = tau_prev + delta*dt`
- **State:** `tau_prev` (stored as prev_tau in state)
- **Detection:** `|delta_rate - delta_rate_limited| > 1e-9`

### C4 — tau_prev Update
- **Python file:** `scripts/simulate_hierarchical_controller.py:6482`
- **Status:** ACTIVE
- **Formula:** `tau_prev = tau_smooth`
- **Pipeline position:** Post-composer, every step

### C5 — mj_data.ctrl Assignment
- **Python file:** `scripts/simulate_hierarchical_controller.py:6799`
- **Status:** ACTIVE
- **Formula:** `mj_data.ctrl[:] = np.array(tau_smooth)`
- **Pipeline position:** Final step of control loop

### C6 — Legacy Torque Zeroing
- **Python file:** `scripts/simulate_hierarchical_controller.py:6632`
- **Status:** ACTIVE
- **Details:** `zero_legacy_torque_sources_for_balance_core()` zeros WBC, momentum, old posture, etc.
- **Pipeline position:** Post-composer, telemetry only

### C7 — Torque Ownership Validation
- **Python file:** `torque_ownership_validator.py`
- **Status:** ACTIVE (validate_ownership=True by default)
- **Purpose:** Ensures no double-write to the same joint index by different torque sources
- **Allowed sharing:** [2,3,7,8] between shape_posture and support_feedforward

---

## 7. Diagnostics Mechanisms

### D1 — Balance-Core Telemetry Columns
- **Python file:** `balance_core_types.py`
- **Status:** ACTIVE
- **40 fields:** state telemetry (29) + torque telemetry (11)
- **Pipeline position:** Post-composer, every step

### D2 — Sagittal Diagnostics
- **Python file:** `sagittal_velocity_damped_balance_controller.py:compute()` return
- **Status:** ACTIVE
- **Fields:** ~60+ diagnostic fields covering all torque components, scheduling, outer loop, ABS, contact, etc.

### D3 — Validation Telemetry Fields
- **Python file:** `balance_core_types.py` + `telemetry_adapter.py`
- **Status:** ACTIVE
- **Fields:** `add_validation_telemetry_fields()`, `normalize_balance_core_owner_names()`

---

## 8. Disabled / Inactive Mechanisms (confirmed zero)

| Mechanism | K2 Status | Confirmation |
|-----------|-----------|-------------|
| Position integral | DISABLED | `enable_position_integral=False` |
| Torque budget aware position | DISABLED | `enable_torque_budget_aware_position=False` |
| Pitch-aware position scaling | DISABLED | `enable_pitch_aware_position_scaling=False` |
| Capture gate | DISABLED | `vd_enable_capture_gate=False` |
| Continuous k_position | DISABLED | `continuous_k_position=False` |
| Continuous k_wheel_velocity | DISABLED | `continuous_k_wheel_velocity=False` |
| Continuous kd_pitch | DISABLED | `continuous_kd_pitch=False` |
| Continuous k_velocity | DISABLED | `continuous_k_velocity=False` |
| Hysteresis blending | DISABLED | Not in K2 profile chain |
| APC/APCR transient modes | DISABLED | Not in K2 profile chain |
| Recenter/centering bias | DISABLED | `zc_replace_adaptive=False` |
| ZC replace adaptive | DISABLED | `zc_replace_adaptive=False` |
| EZC replace adaptive | DISABLED | `ezc_replace_adaptive=False` |
| Bias cancel | DISABLED | Not in K2 profile chain |
| L_feedback (coordinated low-freq) | DISABLED | Not in K2 profile chain |
| Phase lead | DISABLED | Not in K2 profile chain |
| Pitch rate consistency (control) | DISABLED | Diagnostic only |
| Transient detect/capture | DISABLED | `transient_mode` not in T1-T4 |
| Boundary yaw fix | DISABLED | Not active for default variants |
| Wheel yaw stabilizer | DISABLED | Only M-family profiles |
| HY-FF in shape posture | DISABLED | `enable_hip_yaw_support_feedforward=False` |
| HY2-DIV in shape posture | DISABLED | `enable_hip_yaw_divergence_damping=False` |
| Position authority scaling | DISABLED | 1.0 (pass-through) |
| Pitch rate boost | DISABLED | 1.0 (pass-through) |
| Support velocity torque | DISABLED | `k_support_velocity=0.0` |
| Capture point torque | DISABLED | `kp_cp=0.0` |

---

## 9. Mechanism Summary Counts

| Category | Count |
|----------|-------|
| **Input/State extraction** | 11 |
| **Sagittal wheel balance (active)** | 9 |
| **Sagittal wheel balance (disabled)** | 5 |
| **Outer-loop/support (active)** | 7 |
| **Adaptive bias trim (active)** | 8 |
| **Leg/body controllers (active)** | 5 |
| **Leg/body controllers (disabled)** | 3 |
| **Composer/torque (active)** | 7 |
| **Diagnostics** | 3 |
| **Total active mechanisms** | **50** |
| **Total disabled (confirmed zero)** | **26** |
| **Total opt-in (CLI flag)** | **1** |
