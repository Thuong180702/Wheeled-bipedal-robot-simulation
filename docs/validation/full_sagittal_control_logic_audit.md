# Full Sagittal Control Logic Audit

**Date:** 2026-06-15  
**Profile scope:** SagittalVelocityDampedBalanceController + simulate_hierarchical_controller  
**Scenario:** high_0p480 equilibrium analysis

## Classification

**SAGITTAL_CONTROL_MAP_COMPLETE**

---

## 1. Input Variables and References

### 1.1 Orientation Inputs

| Variable | Source | Unit | Reference |
|----------|--------|------|-----------|
| `pitch_x_rad` | `centroidal_state_control.body_pitch_x` | rad | `pitch_x_ref = 0.0` (exact) |
| `pitch_rate_x_rad_s` | `centroidal_state_control.body_pitch_rate_x` (optionally corrected) | rad/s | N/A |
| `roll_y_rad` | `centroidal_state_control.body_roll_y` | rad | N/A |

**Critical finding:** `pitch_x_ref_rad = 0.0` (exact, never changes). The forward pitch equilibrium comes from robot dynamics, NOT from a biased reference.

### 1.2 Position/Velocity Inputs

| Variable | Source | Unit | Reference |
|----------|--------|------|-----------|
| `sagittal_position_error_m` | Support center projected to initial-heading axis | m | `support_center_eq_xy` (wheel midpoint at equilibrium) |
| `sagittal_velocity_m_s` | `centroidal_state_control.com_vel[1]` (CoM sagittal velocity) | m/s | 0 |
| `support_position_velocity_m_s` | Numerical derivative of `sagittal_position_error_m` | m/s | 0 |
| `wheel_vel_left_rad_s` | MuJoCo sensor | rad/s | 0 |
| `wheel_vel_right_rad_s` | MuJoCo sensor | rad/s | 0 |

**Design note:** Position tracks support center (wheel midpoint), NOT COM. This decouples position correction from pitch-induced COM motion.

### 1.3 Safety/Context Inputs

| Variable | Source | Unit | Notes |
|----------|--------|------|-------|
| `com_z_m` | `centroidal_state_control.com_pos[2]` | m | For scheduling and safety gates |
| `contact_valid` | Both wheels in contact with valid force | bool | Safety gate for many mechanisms |
| `height_variant_name` | From height setup | string | e.g., "high_0p480" |
| `commanded_height_ref_m` | From height setup | m | e.g., 0.480 |

### 1.4 Equilibrium Posture (from setup file)

From `high_0p480_setup.json`:

```
hip_pitch_ref: 0.626052 rad (35.9 deg)
knee_ref: 1.223364 rad (70.1 deg)
equilibrium_pitch_x: 0.0 (prescribed, NOT achieved)
equilibrium_com_pos: [-2.6e-7, -0.0057, 0.481] m
support_center_x: -1.6e-6 m
com_x relative to support: +1.35e-6 m (essentially centered)
```

The setup prescribes zero pitch equilibrium but the robot settles at +3 to +5 deg forward pitch.

---

## 2. All Torque Terms (in composition order)

### 2.1 tau_pitch (Proportional pitch correction)

```
tau_pitch_raw_orig = kp_pitch * pitch_x_rad
tau_pitch_scheduled = tau_pitch_raw_orig * effective_pitch_scale
tau_pitch = clip(tau_pitch_scheduled, -pitch_tau_cap, pitch_tau_cap)
```

| Parameter | Value | Source |
|-----------|-------|--------|
| kp_pitch | 50.0 | Default |
| effective_pitch_scale | 0.85–1.0 | Height-scheduled |
| pitch_tau_cap | 8.0 Nm | Height-scheduled at high heights |

**Transformations applied in order:**
1. APC recenter pitch suppression (APCR1l): sets tau_pitch=0 during RECENTER state if `apc_hysteresis_pitch_suppress_in_recenter=True`
2. Pitch bias DC compensation (Phase 7): subtracts slow DC estimate from tau_pitch
3. APCR1m conditional pitch blend: multiplies tau_pitch by `apc_pitch_blend_scale` (0.0–1.0) based on error magnitude
4. T6H soft pitch blend: multiplies by `t6h_pitch_blend_factor` (0.5–1.0) during arch_fix
5. Final clipping at motor cap (5.0 Nm per wheel)

**Sign convention:** Positive pitch → positive tau_pitch → forward wheel acceleration → robot falls more forward.
**Convention check:** Positive = correct restoring direction. Consistent with baseline.

**Contributions to positive support drift:**
- Directly produces forward wheel torque proportional to forward pitch
- At equilibrium pitch (+3.6 deg), tau_pitch ≈ +3.2 to +3.4 Nm
- This is the LARGEST single torque term at equilibrium

---

### 2.2 tau_pitch_rate (Derivative pitch damping)

```
tau_pitch_rate = kd_pitch * pitch_rate_x_rad_s
kd_pitch = 1.0 (default)
```

**Sign convention:** Positive pitch rate → negative tau_pitch → backward wheel acceleration → opposes forward pitch rate.
**Convention check:** Consistent. Positive pitch rate means falling forward; needs backward damping.

**Contributions:** Small RMS (~0.1–0.3 Nm). Primarily transient damping, not DC drift.

---

### 2.3 tau_sagittal_velocity (CoM velocity damping)

```
tau_sagittal_velocity = -k_velocity * velocity_damping_scale * sagittal_velocity_m_s
k_velocity: 1.0–1.5 (height-scheduled)
velocity_damping_scale: 0.9–1.0 (height-scheduled)
```

**Sign convention:** Positive sagittal velocity (forward COM motion) → negative tau → backward wheel torque → opposes forward motion.
**Convention check:** Consistent.

**Contributions to drift:** Small RMS (~0.3–0.5 Nm). Fights wheel velocity, not support position drift.

---

### 2.4 tau_support_velocity (Support velocity damping)

```
tau_support_velocity = -k_support_velocity * support_velocity_scale * support_position_velocity_m_s
k_support_velocity: 0.5–1.0 (height-scheduled)
support_velocity_scale: 0.5–1.0 (height-scheduled)
```

**Sign convention:** Positive support velocity (drift moving forward) → negative tau → backward wheel torque.
**Convention check:** Consistent.

**Contributions:** Small RMS. Directly opposes support drift velocity.

---

### 2.5 tau_cp (Capture point error proxy)

```
tau_cp = -kp_cp * sagittal_position_error_m
kp_cp: 0.0 (disabled by default)
```

**Status:** Disabled (0.0). Intended as capture-point-like term but replaced by explicit k_position.

---

### 2.6 tau_com_vy (COM velocity proxy)

```
tau_com_vy = -kd_com_vy * sagittal_velocity_m_s
kd_com_vy: 0.0 (disabled by default)
```

**Status:** Disabled (0.0). Redundant with tau_sagittal_velocity.

---

### 2.7 tau_position (Position hold + integral)

```
tau_position_p = -k_position * sagittal_position_error_m
tau_position_i = -ki_position_integral * integral_error
tau_position_raw = tau_position_p + tau_position_i
```

| Parameter | Nominal | Height-scheduled |
|-----------|---------|-----------------|
| k_position | 100.0 | 60.0–120.0 (decreases at low height) |
| ki_position_integral | 5.0 | Fixed |
| integral_max_abs | 0.5 | Fixed |

**Transformations:**
1. Capture gate (optional): modifies tau_position_raw based on capture point test
2. Pitch-aware position scaling: reduces authority when pitch is large
3. Torque budget aware (optional): bounds by total-torque budget
4. Architecture fix cap raise: increases max_position_tau during hard/emergency bands
5. T6I phase-aware cap release: decays cap when error converges

**Sign convention:** Positive position error (forward drift) → negative tau_position → backward wheel torque.
**Convention check:** Consistent. Positive error means drifted forward; need backward correction.

**Contributions to positive drift:** At equilibrium, tau_position ≈ -3.5 to -3.7 Nm (always pulling backward).
Position controller saturates at lower bound 27–31% of steps.

---

### 2.8 tau_position_lower_bound saturation

```
tau_position_lower_bound = -max_position_tau - tau_balance_before_position
tau_position_upper_bound = +max_position_tau - tau_balance_before_position
max_position_tau: 4.0–7.0 Nm (height-scheduled)
```

**Critical finding:** Position controller is **always clipped on the negative side**, never on positive side.
This means tau_balance_before_position is consistently positive, consuming headroom for backward torque.

---

### 2.9 Zero-Crossing Support Recenter (ZC)

```
State machine: CENTER_IDLE → RECENTER_FROM_POSITIVE → HOLD_THROUGH_ZERO → CENTER_IDLE
              ↘ RECENTER_FROM_NEGATIVE ↗
target_tau = zc_base_tau + zc_error_gain * abs_error
zc_base_tau: 0.35 Nm
zc_max_tau: 0.80 Nm
zc_error_gain: 5.0 Nm/m
zc_enter_m: 0.05 m
zc_cross_target_m: 0.01 m
```

**Status:** Enabled in zero_crossing_support_recenter profile. Replace mode.

**Sign convention:** From positive drift → applies negative correction torque.

**Effectiveness:** 86% of episodes cross zero. BUT: exits at zero and positive bias overwhelms after ~28 steps.

---

### 2.10 Early Zero-Crossing (EZC)

```
State machine: IDLE → APPLY_CORRECTION → ANTI_REBOUND → IDLE
entry_threshold: 0.05 m
ezc_base_tau_nm: 0.50 Nm
ezc_max_tau_nm: 0.70 Nm
ezc_rate_nm_per_step: 0.020 Nm/step
ezc_decay_rate_nm_per_step: 0.025 Nm/step
ezc_min_hold_steps: 0
ezc_max_hold_steps: 500
```

**Status:** Enabled in early_zero_crossing_recenter and early_zero_crossing_recenter_v2 profiles.
Replace mode.

**Key finding from audit:** Strong torque (reaches max 100% of episodes) but decays immediately after exit.
Net tau during episodes: -5.37 Nm. BUT: positive bias overwhelms after ~28 steps.

---

### 2.11 EZC V2 (Enhanced)

```
ezc_v2_base_tau_nm: 0.55 Nm
ezc_v2_max_tau_nm: 0.75 Nm
ezc_v2_rate_nm_per_step: 0.025 Nm/step
ezc_v2_decay_rate_nm_per_step: 0.020 Nm/step
ezc_v2_longer_dwell: 100 steps
ezc_v2_error_gain: 10.0 Nm/m
ezc_v2_antirebound_enabled: True
```

**Status:** Enabled in early_zero_crossing_recenter_v2 profile.

**Finding:** Improved short-horizon (500-step) but regressed at 5000 steps.

---

### 2.12 Adaptive Bias Trim

```
enable_adaptive_bias_trim: True
adaptive_proportional_target: k_tau_per_m * mean_error_m
k_tau_per_m: 5.0 Nm/m
adaptive_max_tau_nm: 0.35–0.55 Nm (height-scheduled)
window_steps: 200
trim_rate_nm_per_step: 0.002 Nm/step
decay_rate_nm_per_step: 0.010 Nm/step
zero_crossing_guard: True
near_zero_relief: True
```

**Status:** Enabled in adaptive_support_centering_trim profile.

**Effectiveness:** Improved saturation but did not center drift (positive % remained 84–92%).

---

### 2.13 T6J Bias Trim

```
t6j_bias_trim_enabled: True
t6j_proportional_target: k_tau_per_m * mean_error_m
k_tau_per_m: 3.0 Nm/m
t6j_max_tau_nm: 0.35 Nm
window_steps: 300
trim_rate_nm_per_step: 0.001 Nm/step
decay_rate_nm_per_step: 0.003 Nm/step
```

**Status:** Enabled in t6j profile.

**Effectiveness:** Modest improvement. Did not center drift.

---

### 2.14 Pitch Bias Compensation (Phase 7 mechanism)

```
pitch_bias_comp_enabled: True
pitch_bias_window_steps: 300
pitch_bias_max_comp_nm: 0.6 Nm
pitch_bias_only_when_abs_pitch_lt_deg: 2.0
pitch_bias_only_when_abs_error_lt_m: 0.03
pitch_bias_gate_abs_error_soft_m: 0.05
pitch_bias_gate_abs_error_hard_m: 0.10
pitch_bias_comp_rate_nm_per_step: 0.002 Nm/step
pitch_bias_decay_rate_nm_per_step: 0.003 Nm/step
```

**Status:** Enabled in pitch_bias_compensated_zero_crossing_recenter profile.

**Effectiveness:** Improved modestly but not enough. Key finding: removes only the +0.20–0.28 Nm residual during |pitch|<1° windows. Does NOT address the main +3.2 Nm at equilibrium pitch.

---

### 2.15 Active Pitch Crossing (APC/APCR)

```
enable_active_pitch_crossing: True
apc_pitch_enter_rad: 0.03 rad
apc_pitch_safe_limit_rad: 0.08 rad
apc_pitch_safe_threshold_rad: 0.05 rad
apc_pitch_danger_threshold_rad: 0.10 rad
```

**Status:** Enabled in APCR1, APCR1b, APCR1c, APCR1nD profiles.

**Effectiveness:** Moderate. Hard safety gates at 0.30 rad limit activation window.

---

### 2.16 Per-Wheel Damping

```
tau_wheel_vel_left = -k_wheel_velocity * wheel_vel_left_rad_s
tau_wheel_vel_right = -k_wheel_velocity * wheel_vel_right_rad_s
k_wheel_velocity: 0.1–0.3 (height-scheduled)
```

**Contributions:** Small, opposes wheel velocity. Mean ~+0.35–0.40 Nm (fighting backward drift).

---

### 2.17 Architecture Fix (T6F)

```
arch_fix_enabled: True
arch_fix_height_threshold_m: 0.45
arch_fix_hard_max_position_tau: 6.0 Nm
arch_fix_emergency_max_position_tau: 8.0 Nm
recenter_priority_direct_enabled: True
```

**Effect:** Raises position cap during hard/emergency bands. Does NOT change torque sign or direction.

---

### 2.18 T6H Soft Pitch Blend

```
t6h_enabled: True
t6h_soft_pitch_blend_factor: 0.5
t6h_pitch_error_threshold_m: 0.08
t6h_pitch_safety_threshold_deg: 8.0
```

**Effect:** Reduces pitch authority by 50% during arch_fix. Preserves partial stabilization.

---

### 2.19 T6I Phase-Aware Cap Release

```
t6i_enabled: True
t6i_convergence_window_steps: 10
t6i_convergence_threshold_m: 0.03
t6i_cap_decay_rate_nm_per_step: 0.05
t6i_cap_min_nm: 2.5
```

**Effect:** Decays cap when error converges. Preserves full pitch/damping authority.

---

## 3. Final Composition

```
tau_common_unclipped = (
    tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
    tau_support_velocity + tau_position + tau_cp + tau_com_vy
)
# Then add recenter terms:
tau_common_unclipped += recenter_tau_clipped  # Phase-aware recenter
tau_common_unclipped += hyst_tau_clipped      # Hysteresis recenter
tau_common_unclipped += bias_tau_clipped      # Bias cancellation
tau_common_unclipped += apc_tau_clipped       # Active pitch crossing

# Apply wheel sign
tau_common = wheel_torque_sign * tau_common_unclipped

# Per-wheel:
tau_left = tau_common + tau_wheel_vel_left
tau_right = tau_common + tau_wheel_vel_right

# Final clip at motor_cap (5.0 Nm)
tau_left = clip(tau_left, -5.0, 5.0)
tau_right = clip(tau_right, -5.0, 5.0)
```

---

## 4. Equilibrium Analysis (at high_0p480, 5000-step)

### 4.1 Steady-state torque budget

At equilibrium (drift near zero, pitch at +3.6 deg):

| Term | Mean (Nm) | Direction |
|------|-----------|-----------|
| tau_pitch | +3.38 | Forward (fighting fall forward) |
| tau_position | -3.74 | Backward (correcting drift) |
| tau_wheel_velocity | +0.39 | Forward (fighting backward drift) |
| Net common | ≈ -0.01 | Near zero (stalemate) |
| Final wheel tau | +0.01 | Near zero |

**Key insight:** tau_pitch and tau_position cancel to near-zero, leaving the robot in a forward-pitch equilibrium stalemate.

### 4.2 Pitch equilibrium

- Prescribed pitch_ref: 0.0 rad
- Actual equilibrium pitch: +3.6 deg (+3.3 to +3.9 deg across profiles)
- This is NOT a controller bias — it's a physical equilibrium from leg geometry

### 4.3 Position equilibrium

- tau_position always negative (backward) at equilibrium
- Position controller saturates at lower bound 27–31% of steps
- This means tau_balance_before_position consumes headroom for backward torque

### 4.4 Torque composition conflict

The core architectural issue:

```
tau_balance_before_position = tau_pitch + tau_pitch_rate + tau_sagittal_velocity + tau_support_velocity + tau_cp + tau_com_vy + 0.5*(tau_wheel_vel_left + tau_wheel_vel_right)
```

At equilibrium, tau_balance ≈ +3.5 to +4.0 Nm (positive from tau_pitch).
This POSITIVE headroom consumption means the negative bound for tau_position becomes:
`tau_position_lower_bound = -max_position_tau - (+3.5 to +4.0) = -7.5 to -8.0 Nm`

But max_position_tau is only 4.0–8.0 Nm, so the backward torque from tau_position is limited.

---

## 5. Fighting vs. Helping Terms for Support Recenter

| Term | At Positive Drift | At Negative Drift | Fights Recentering? |
|------|-------------------|-------------------|---------------------|
| tau_pitch | + (forward) | - (backward) | **YES** — always opposite to recentering direction |
| tau_position | - (backward) | + (forward) | No — correct direction |
| tau_pitch_rate | Opposes pitch rate | Opposes pitch rate | No — transient only |
| tau_sagittal_velocity | - (backward) | + (forward) | No — correct direction |
| tau_support_velocity | - (backward) | + (forward) | No — correct direction |
| tau_wheel_velocity | + (forward) | - (backward) | **YES** — always opposite to wheel motion |
| ZC/EZC recenter | Correct direction | Correct direction | No — correct direction |
| Adaptive bias trim | Correct direction | Correct direction | No — correct direction |
| Pitch bias comp | Reduces tau_pitch | N/A (small) | **Indirectly helps** |

**Primary fighting term: tau_pitch** — always produces torque opposite to support recenter direction.

---

## 6. Telemetry Columns (Complete List)

```
pitch_x_rad, pitch_x_error_rad, pitch_x_ref_rad
pitch_rate_x_rad_s
tau_pitch, tau_pitch_raw, tau_pitch_scheduled, tau_pitch_clipped
tau_pitch_rate
tau_pitch_before_bias_comp, tau_pitch_after_bias_comp
tau_pitch_to_position_ratio
tau_pitch_original, tau_pitch_outer_loop
tau_position, tau_position_raw, tau_position_p, tau_position_i
tau_position_integral, tau_position_before_clip, tau_position_clipped
tau_position_lower_bound, tau_position_upper_bound
tau_position_total_bound_clipped
tau_position_saturation_flag, tau_position_saturation_reason
tau_sagittal_velocity, tau_cp, tau_com_vy
tau_support_velocity
tau_wheel_velocity_left, tau_wheel_velocity_right
tau_common_unclipped, tau_common_clipped
tau_left, tau_right
tau_total_unclipped, tau_total_clipped
sagittal_position_error_m, sagittal_velocity_m_s
support_position_error_m, support_position_error_scaled_m
support_position_velocity_m_s
wheel_velocity_mean_rad_s
com_z_m, com_y_m
sagittal_controller_input_pitch_x, sagittal_controller_input_pitch_rate_x
apcr1m_pitch_blend_active, apcr1m_pitch_blend_scale
apc_pitch_tau, apc_pitch_active, apc_recenter_active
ezc_tau_nm, ezc_active, ezc_state, ezc_direction
zc_tau_nm, zc_active, zc_state, zc_direction
adaptive_bias_tau_nm, t6j_bias_trim_tau_nm
outer_loop_pitch_ref_rad, outer_loop_support_error_m
outer_loop_support_velocity_mps, outer_loop_integral_m_s
pitch_bias_comp_tau, pitch_bias_estimate_nm
pitch_bias_gate_pass, pitch_bias_estimation_active
apcr1m_pitch_blend_active, apcr1m_pitch_blend_scale
arch_fix_active, arch_fix_reason
t6i_converging, t6i_target_cap, t6i_current_cap
```

---

## 7. Key Findings Summary

1. **tau_pitch is not a bug** — it's the correct response to forward pitch equilibrium
2. **Forward pitch equilibrium comes from physics**, not controller reference bias
3. **tau_pitch is the PRIMARY fighting term** against support recenter
4. **tau_position is always corrective** but saturates at lower bound
5. **Pitch bias compensation removes only the residual** (+0.20–0.28 Nm), not the main bias (+3.2 Nm)
6. **ZC/EZC are effective but incomplete** — they work but don't solve the structural imbalance
7. **The architecture is sound but the equilibrium is wrong** — adjusting tau_pitch gains won't fix a physics problem

---

## 8. Root Cause Classification for Phase 2

Based on this control map:

**Root cause is not in the controller logic** — it's in the equilibrium posture.
The robot needs forward pitch torque to stay balanced at high_0p480 because:
- The COM is slightly forward of the wheel contact line at this height
- OR the hip_pitch/knee references produce a forward-leaning pose
- OR the low-level PD controllers produce a lean

**The controller correctly responds to this equilibrium, but the equilibrium itself is forward-pitched.**

Fix paths:
- **Path A:** Adjust equilibrium posture (hip_pitch/knee) to center the COM vertically above the wheels
- **Path B:** Use pitch reference offset to slightly bias the controller toward more upright posture
- **Path C:** Redesign sagittal controller as unified state-feedback (LQR) over multiple states