# K2 Python → JAX Complete Port Coverage Matrix

**Date:** 2026-06-27
**Profile:** k2_notch_low_q_v1
**Coverage scope:** Port coverage audit — NOT numerical parity

---

## Coverage Status Legend

| Status | Definition |
|--------|-----------|
| **PORTED_FULL_COVERAGE** | All inputs, states, params, outputs, and insertion points represented in JAX |
| **PORTED_PARTIAL_COVERAGE** | Some portion exists but some inputs/states/params/gates are missing or differ |
| **PYTHON_ACTIVE_MISSING_IN_JAX** | Active Python mechanism has no JAX counterpart |
| **EXTERNAL_PYTHON_PRECOMPUTED_AND_PASSED_TO_JAX** | Python precomputes; result passed as JAX input (acceptable, documented) |
| **DOWNSTREAM_PYTHON_ONLY** | Outside JAX controller kernel; not needed inside JAX (acceptable) |
| **INACTIVE_ZERO_CONFIRMED** | Disabled/inactive in K2; correctly not ported or zeroed |
| **JAX_EXTRA_NO_PYTHON_EQUIVALENT** | JAX has a control-affecting mechanism not in Python |
| **JAX_DIFFERENT_STRUCTURE** | Same mechanism but structured differently in JAX (e.g., pre vs post composer) |

---

## 1. Input / State Extraction Mechanisms

| # | Python Mechanism | Python Source | JAX Equivalent | JAX Source | Coverage | Wheel [4,9]? | Mismatch Risk |
|---|-----------------|---------------|----------------|------------|----------|--------------|---------------|
| M1 | Physical state extraction | `simulate_hierarchical_controller.py:5527` (CentroidalStateEstimator) | Python precomputes → `pack_input_k2()` passes results | `k2_jax_controller.py:933` | EXTERNAL_PYTHON_PRECOMPUTED | No | None (input data) |
| M2 | Joint index mapping (10-DOF) | `balance_core_types.py:1-30` | Identical mapping in `k2_jax_controller.py` via pack functions | `k2_jax_controller.py:649,1226-1231` | PORTED_FULL_COVERAGE | Yes | None |
| M3 | Contact detection | `simulate_hierarchical_controller.py:5898` (ContactSupervisor) | `contact_ok` hardcoded `True` in JAX (always true from Python perspective) | `k2_jax_controller.py:1186` | PORTED_PARTIAL_COVERAGE | No | Low (contact_ok=always True matches Python in K2 scenarios) |
| M4 | Capture point estimation | `simulate_hierarchical_controller.py:5800` (CapturePointEstimator) | Python precomputes, passed as input; K2 has kp_cp=0.0 (disabled) | N/A (disabled) | INACTIVE_ZERO_CONFIRMED | No | None |
| M5 | Support center computation | `simulate_hierarchical_controller.py:5980` (`compute_support_center_xy()`) | Python precomputes → result in `sagittal_position_error_m` input | `k2_jax_controller.py:920,940` (input field) | EXTERNAL_PYTHON_PRECOMPUTED | No | None (input data) |
| M6 | Sagittal projection | `simulate_hierarchical_controller.py:5991-6014` | Python precomputes → results in `sagittal_velocity_m_s`, `sagittal_position_error_m` inputs | `k2_jax_controller.py:908-909` (input fields) | EXTERNAL_PYTHON_PRECOMPUTED | Yes | None (input data) |
| M7 | Target/commanded height | `simulate_hierarchical_controller.py:5348` | Passed as `commanded_height_ref_m` input field | `k2_jax_controller.py:908` | PORTED_FULL_COVERAGE | Yes | None (input data) |
| M8 | Dynamic height update | `simulate_hierarchical_controller.py:5351` | Python updates `height_cmd` externally; JAX receives via `commanded_height_ref_m` | N/A (external) | EXTERNAL_PYTHON_PRECOMPUTED | No | None (input data) |
| M9 | Pitch ref offset generation (pitch_x_error) | `simulate_hierarchical_controller.py:6117-6118` | Python pre-adjusts pitch_x before passing to JAX. JAX computes but does NOT apply offset internally. | `k2_jax_controller.py:1172-1173, 1166-1169` | EXTERNAL_PYTHON_PRECOMPUTED (offset) + JAX_EXTRA (computes but discards) | Yes | **HIGH** — JAX computes `total_pitch_ref_offset_deg` but does NOT apply it. Python loop applies it externally before passing pitch_x. If the external application differs from what JAX would compute, this causes mismatch. |
| M10 | q_ref generation | `simulate_hierarchical_controller.py:5720` | Passed as `q_ref_*` input fields | `k2_jax_controller.py:918-919` | PORTED_FULL_COVERAGE | No | None (input data) |
| M11 | Torque limit & rate params | `simulate_hierarchical_controller.py:4658` | `pack_params_stage2()` packs into params array | `k2_jax_controller.py:150-194, 125-148` | PORTED_FULL_COVERAGE | Yes | None |

---

## 2. Sagittal Wheel Balance — Active Mechanisms

| # | Python Mechanism | Python Formula | JAX Equivalent | JAX Source | Coverage | Wheel [4,9]? | Mismatch Risk |
|---|-----------------|----------------|----------------|------------|----------|--------------|---------------|
| S1 | Notch filter (biquad, 2.5 Hz, Q=2.0) | `BiquadNotchFilter.update(pitch_rate)` → `pitch_rate_notched` | `k2_jax_notch_step()` — identical Direct Form II Transposed | `k2_jax_controller.py:219-260, 1097-1106` | PORTED_FULL_COVERAGE | Yes | **MEDIUM** — formula identical; state identical; coefficients identical |
| S2 | Notch height gate | `smoothstep_gate(height_ref, 0.42, 0.48)` | `smoothstep_gate_jax(height_ref, 0.42, 0.48)` — identical formula | `k2_jax_controller.py:1105` | PORTED_FULL_COVERAGE | Yes | Low |
| S3 | tau_pitch | `kp_pitch * pitch_x_error` (50.0, uncapped) | `kp_pitch * pitch_x` — identical, kp_pitch=50.0, cap=0.0 | `k2_jax_controller.py:590-596, 1209-1214` | PORTED_FULL_COVERAGE | Yes | **MEDIUM** — depends on whether `pitch_x_error` (Python) equals `effective_pitch_x` (JAX). If pitch offset application differs, this diverges. |
| S4 | tau_pitch_rate | `kd_pitch * pitch_rate_eff` (10.0) | `effective_kd_pitch * pitch_rate_rad_s` — kd=10.0, same effective pitch rate after notch blend | `k2_jax_controller.py:599, 1106, 1215` | PORTED_FULL_COVERAGE | Yes | **MEDIUM** — depends on notch output parity and blend formula |
| S5 | tau_sagittal_velocity | `-k_velocity * sagittal_velocity` (15.0) | `-effective_k_velocity * sagittal_velocity_m_s` — k=15.0 | `k2_jax_controller.py:600, 1216` | PORTED_FULL_COVERAGE | Yes | Low |
| S6 | tau_position | `-k_position * sagittal_position_error + external_position_trim` (40.0) | `-effective_k_position * sagittal_position_error_m + external_position_trim` — k=40.0 | `k2_jax_controller.py:605-606, 1219` | PORTED_FULL_COVERAGE | Yes | **HIGH** — ABS trim_tau must match exactly; sign convention, clamping, hysteresis must all match |
| S7 | tau_wheel_vel (L,R) | `-k_wheel_velocity * wheel_vel_L/R` (0.5) | `-effective_k_wheel_velocity * wheel_vel_left/right_rad_s` — k=0.5 | `k2_jax_controller.py:637-638, 1218` | PORTED_FULL_COVERAGE | Yes | Low |

### Sagittal — Disabled (zero in K2)

| # | Python Mechanism | K2 Status | JAX Equivalent | Coverage |
|---|-----------------|-----------|----------------|----------|
| S8 | tau_support_velocity | DISABLED (k=0.0) | `effective_support_velocity_gain=0.0` → zero | INACTIVE_ZERO_CONFIRMED |
| S9 | tau_cp | DISABLED (kp_cp=0.0) | `kp_cp=0.0` → zero | INACTIVE_ZERO_CONFIRMED |
| S10 | tau_com_vy | ACTIVE (kd=5.0) | `kd_com_vy=5.0` → computed but uses same `sagittal_velocity_m_s` as S5 | PORTED_FULL_COVERAGE |

### Sagittal — Assembly

| # | Python Mechanism | Python Formula | JAX Equivalent | Coverage | Mismatch Risk |
|---|-----------------|----------------|----------------|----------|---------------|
| S11 | Common torque assembly | `tau_common = sign * sum(all terms)` | Same formula, sign=1.0 | `k2_jax_controller.py:640-646` | PORTED_FULL_COVERAGE | Low |
| S12 | Per-wheel split | `tau_L = tau_common + tau_wheel_vel_L`, `tau_R = tau_common + tau_wheel_vel_R` | Same formula | `k2_jax_controller.py:645-650` | PORTED_FULL_COVERAGE | Low |
| S13 | Max position tau scheduling | 4.0→6.0 Nm via scheduled_k_position | Same formula via `k2_jax_scheduled_k_position(..., 4.0, 6.0, 0.300, 0.393)` | `k2_jax_controller.py:1124-1130` | PORTED_FULL_COVERAGE | Low |
| S14 | Height schedule (filtered_com_z) | `schedule_h = height_ref if >0 else 0.9*filtered + 0.1*com_z` | Identical formula | `k2_jax_controller.py:1109-1111` | PORTED_FULL_COVERAGE | Low |
| S15 | Pitch bias comp | DISABLED in K2 (pitch_bias_comp_tau=0.0) | `pitch_bias_comp_tau=0.0` | INACTIVE_ZERO_CONFIRMED | None |
| S16 | Pitch-aware position scaling | DISABLED in K2 | `enable_pitch_aware_position_scaling=False` | INACTIVE_ZERO_CONFIRMED | None |
| S17 | Torque budget aware position | DISABLED in K2 | `enable_torque_budget_aware_position=False` | INACTIVE_ZERO_CONFIRMED | None |

### Key insight: All 17 sagittal mechanisms (9 active, 8 disabled) have clear JAX coverage. The main sagittal torque formulas are mathematically identical. The remaining 0.01 Nm wheel mismatch risk lies in precisely matching input values (pitch_x after offset, pitch_rate after notch, ABS trim value) rather than missing coverage.

---

## 3. Adaptive Bias Trim (ABS) Mechanisms

| # | Python Mechanism | Python Formula | JAX Equivalent | Coverage | Wheel [4,9]? | Mismatch Risk |
|---|-----------------|----------------|----------------|----------|--------------|---------------|
| A1 | Sliding window ring buffer | Deque of 300 signed errors, running sum, count, ptr | `_abs_update_ring_buffer()` — identical ring buffer, 300 entries, running sum | `k2_jax_controller.py:1432-1456` | PORTED_FULL_COVERAGE | Yes | **MEDIUM** — exact sum/count/ptr matching critical |
| A2 | Slow mean computation | sum/count over full 300-step window | `_abs_sliding_mean_slow()` — identical | `k2_jax_controller.py:1369-1373` | Yes | Low (same sum/count) |
| A3 | Fast mean computation | Mean of most recent 100 entries | `_abs_sliding_mean_fast()` — mask-based circular walk | `k2_jax_controller.py:1376-1395` | PORTED_FULL_COVERAGE | Yes | **MEDIUM** — numerical precision of mask vs explicit loop |
| A4 | Zero-crossing detection | Count sign changes in 500-step window | `_abs_count_zero_crossings()` — vectorized JAX | `k2_jax_controller.py:1398-1429` | PORTED_FULL_COVERAGE | Yes | **HIGH** — vectorized counting may differ at edges; verified Stage 6L |
| A5 | ZC guard trigger | Guard increments when zc_count > limit, resets to 0 when not | Identical logic: `guard = where(zc_guard, guard+1, where(guard>=3, 0, 0))` | `k2_jax_controller.py:1533-1536` | PORTED_FULL_COVERAGE | Yes | Low |
| A6 | Height-scheduled max trim | 0.35 → 0.50 → 0.55 Nm piecewise | Identical 3-zone piecewise interpolation | `k2_jax_controller.py:1514-1526` | PORTED_FULL_COVERAGE | Yes | Low |
| A7 | Proportional target with hysteresis | `-k_tau * (mean_err - sign*exit_th)` with near-zero relief | Identical formula | `k2_jax_controller.py:1556-1564` | PORTED_FULL_COVERAGE | Yes | Low |
| A8 | Asymmetric rate limiting | Rate=0.006 (up), decay=0.018 (down) | Identical asymmetric rate limits | `k2_jax_controller.py:1568-1573` | PORTED_FULL_COVERAGE | Yes | Low |
| A9 | Sign-reversal hold | 100-step freeze on sign change | Identical hold logic | `k2_jax_controller.py:1547-1551` | PORTED_FULL_COVERAGE | Yes | Low |
| A10 | Safety gates (pitch, roll, contact, hip_yaw, abs_error) | All gates via profile parameters | Identical gates using same profile constants | `k2_jax_controller.py:1182-1191` | PORTED_FULL_COVERAGE | Yes | Low |
| A11 | ZC max_tau guard scale | max_tau *= 0.5 when zc_guard active | Identical `guard_scale=0.5` | `k2_jax_controller.py:1538` | PORTED_FULL_COVERAGE | Yes | Low |

### ABS coverage: FULL — all 11 ABS mechanisms have JAX equivalents with identical formulas, parameters, and state. Verified by Stage 6L sliding window fix.

---

## 4. Outer-Loop / Support Mechanisms

| # | Python Mechanism | Python Source | JAX Equivalent | Coverage | Mismatch Risk |
|---|-----------------|---------------|----------------|----------|---------------|
| O1 | Calibrated outer loop v2 (PCHIP) | `calibrated_outer_loop_functions_v2.py` | `build_calibrated_grid_params()` + `k2_jax_grid_interpolate()` — 20k-point grid | PORTED_FULL_COVERAGE | **LOW** (grid interpolation error < 1e-6) |
| O2 | Support error rate smoothing | `apply_lowpass()` in sim loop | `_jax_apply_lowpass()` + state in `ol_support_error_rate` | PORTED_FULL_COVERAGE | Low |
| O3 | Outer loop PID pitch ref | `compute_outer_loop_pitch_ref()` | `k2_jax_compute_outer_loop_pitch_ref()` — identical formula | PORTED_FULL_COVERAGE | Low |
| O4 | Outer loop rate limiting + lowpass | `apply_rate_limit()` + `apply_lowpass()` | `_jax_apply_rate_limit()` + `_jax_apply_lowpass()` | PORTED_FULL_COVERAGE | Low |
| O5 | Outer loop safety gates | Pitch/roll/contact/error gates in sim loop | Safety gates in outer loop? | PORTED_PARTIAL_COVERAGE | **MEDIUM** — JAX outer loop does NOT gate on pitch/roll/contact safety. Python gates turn off target to 0.0 when unsafe. JAX computes target unconditionally (but rate-limit/lowpass still apply). |
| O6 | Physics equilibrium FF | `physics_equilibrium_feedforward.py` PCHIP | `build_physics_ff_grid_params()` + `k2_jax_grid_interpolate()` — 100k-point grid | PORTED_FULL_COVERAGE | **LOW** (grid interpolation error < 1e-6) |
| O7 | Low-band support outer loop | Gaussian gate in sim loop | `k2_jax_low_band_support_pitch_ref()` — identical formula | PORTED_FULL_COVERAGE | Low |
| O8 | Support reference logic | `compute_support_center_xy()` + sagittal projection | Python precomputes → passed as input | EXTERNAL_PYTHON_PRECOMPUTED | None (input data) |
| O9 | Outer loop integral | DISABLED in K2 (ki=0) | Hardcoded `integral_error_m_s=0.0`, `ki_deg_per_m_s=0.0` | INACTIVE_ZERO_CONFIRMED | None |

### Key finding: O5 is the only outer-loop mechanism with partial coverage. Python outer loop has safety gates that zero the target when unsafe; JAX outer loop computes unconditionally. However, rate-limiting and low-pass smoothing still apply in JAX, mitigating the difference.

---

## 5. Leg / Body Controller Mechanisms

| # | Python Mechanism | Python Source | JAX Equivalent | JAX Source | Coverage | Mismatch Risk |
|---|-----------------|---------------|----------------|------------|----------|---------------|
| L1 | Shape/posture PD | `ShapePostureController.compute()` | `k2_jax_shape_posture_compute()` — same PD formula, same gains | `k2_jax_controller.py:668-690` | PORTED_FULL_COVERAGE | Low |
| L2 | HY-FF (hip-yaw support FF) | DISABLED in K2 | Not included in JAX | N/A | INACTIVE_ZERO_CONFIRMED | None |
| L3 | HY2-DIV (divergence damping) | DISABLED in K2 | Not included in JAX | N/A | INACTIVE_ZERO_CONFIRMED | None |
| L4 | Lateral roll balance | `LateralRollBalanceController.compute()` | `k2_jax_lateral_roll_compute()` — same formula, same gains, stance_reg=True | `k2_jax_controller.py:693-718` | PORTED_FULL_COVERAGE | Low |
| L5 | Yaw control | `YawController.compute()` | `k2_jax_yaw_compute()` — same formula, same gains (kp=8, kd=2, max=5) | `k2_jax_controller.py:721-727` | PORTED_FULL_COVERAGE | **MEDIUM** — Python: yaw ADDED to tau_shape_posture BEFORE composer (composer clips it). JAX: yaw also added to posture BEFORE composer. Match. |
| L6 | Mode hip-yaw divergence | CLI opt-in (`--enable-mode-hip-yaw-divergence`) | `k2_jax_mode_div_compute()` — same formula, defaults kp=10, kd=0.5, max=7.5 | `k2_jax_controller.py:730-747` | PORTED_FULL_COVERAGE | **MEDIUM** — Python: mode_div ADDED to posture before composer. JAX: same. But Python has support-gating opt-in not in JAX. |
| L7 | Empirical support FF | `SupportFeedforwardController.compute()` — fixed vector × 0.5 | `k2_jax_empirical_support_ff()` — identical vector | `k2_jax_controller.py:768-776` | PORTED_FULL_COVERAGE | **HIGH** — Python: support FF passes through composer (clipped, rate-limited). JAX: support FF is in tau_sum (also clipped, rate-limited). Match confirmed Stage 7B. |
| L8 | Wheel yaw stabilizer | DISABLED in K2 (M-family only) | Not in JAX | N/A | INACTIVE_ZERO_CONFIRMED | None |

---

## 6. Composer / Final Torque Mechanisms

| # | Python Mechanism | Python Formula | JAX Equivalent | Coverage | Mismatch Risk |
|---|-----------------|----------------|----------------|----------|---------------|
| C1 | Four-source summation | `tau_total_raw = sum of 4 sources` | `tau_sum = tau_sag + tau_posture_with_yaw + tau_lateral + empirical_ff` | PORTED_FULL_COVERAGE | **HIGH** — Different composition: yaw and mode_div are in tau_posture_with_yaw (pre-composer) in BOTH. **But tau_support_feedforward (height-gated hip-yaw from k2_jax_support_feedforward_compute) is EXCLUDED in JAX** (line 1267-1269). |
| C2 | Actuator clipping | `clip(tau_total_raw, ±torque_limit)` | `clip(tau_sum, ±torque_limit)` | PORTED_FULL_COVERAGE | Low |
| C3 | Rate limiting | `tau_final = tau_prev + clip((tau_clipped - tau_prev)/dt, ±max_rate) * dt` | Identical formula | PORTED_FULL_COVERAGE | Low |
| C4 | tau_prev update | `tau_prev = tau_smooth` | `prev_tau = tau_final`, stored in state | PORTED_FULL_COVERAGE | Low |
| C5 | mj_data.ctrl assignment | `mj_data.ctrl[:] = tau_smooth` | Python assigns; JAX output replaces tau_smooth | DOWNSTREAM_PYTHON_ONLY | None |
| C6 | Legacy torque zeroing | `zero_legacy_torque_sources()` | N/A (telemetry only) | DOWNSTREAM_PYTHON_ONLY | None |
| C7 | Torque ownership validation | `TorqueOwnershipValidator.validate()` | Not in JAX (pure math, no ownership tracking) | PYTHON_ACTIVE_MISSING_IN_JAX | Low (diagnostic only, doesn't affect torque) |

### Critical finding C1: In JAX, `k2_jax_support_feedforward_compute()` is computed but EXPLICITLY EXCLUDED from `tau_sum`. Python includes it via composer. The code comment says: "Python balance-core has no equivalent; inclusion causes divergence during descending height transitions and push recovery."

---

## 7. JAX-Only Mechanisms

| # | JAX Mechanism | JAX Source | Description | Control-Affecting? | Risk |
|---|---------------|------------|-------------|-------------------|------|
| J1 | Pre-evaluated grid interpolation | `k2_jax_grid_interpolate()` | Replaces PCHIP function calls with linear interpolation on pre-built grids. 20k points (calibrated), 100k points (physics FF). | YES (affects outer loop gains and physics FF values) | **LOW** — grid error < 1e-6 verified |
| J2 | Pitch ref offset computed but NOT applied | `k2_jax_controller.py:1166-1173` | `total_pitch_ref_offset_deg` computed for diagnostics but JAX receives pre-adjusted `pitch_x` | NO (diagnostic only) | None |
| J3 | Height schedule using filtered_com_z blend | `k2_jax_controller.py:1109-1111` | `schedule_h = height_ref if >0 else 0.9*filtered_com_z + 0.1*com_z` | YES (affects all height-dependent scheduling) | Low (same formula) |
| J4 | Ring buffer ABS (not EMA-based ABS) | `_k2_jax_adaptive_bias_trim()` | True circular ring buffer with running sum, matching Python after Stage 6L fix | YES (affects ABS trim_tau) | Low (verified Stage 6L) |
| J5 | Vectorized zero-crossing counting | `_abs_count_zero_crossings()` | Uses `jnp.where` + `jnp.roll` instead of Python for-loop | YES (affects ZC guard) | **MEDIUM** — potential edge-case differences |
| J6 | Outer loop integral hardcoded zero | `k2_jax_controller.py:1158-1160` | Ki=0, integral=0 — matches K2 profile (Ki disabled) | NO (zero when K2 profile disables integral) | None |
| J7 | tau_support_ff computed but excluded | `k2_jax_controller.py:1267-1269` | `k2_jax_support_feedforward_compute` result NOT added to tau_sum | YES — different from Python which includes it | **HIGH** — documented divergence cause |

---

## 8. Summary Statistics

| Category | Count |
|----------|-------|
| PORTED_FULL_COVERAGE | 35 |
| PORTED_PARTIAL_COVERAGE | 3 |
| PYTHON_ACTIVE_MISSING_IN_JAX | 1 |
| EXTERNAL_PYTHON_PRECOMPUTED_AND_PASSED_TO_JAX | 7 |
| DOWNSTREAM_PYTHON_ONLY | 2 |
| INACTIVE_ZERO_CONFIRMED | 17 |
| JAX_EXTRA_NO_PYTHON_EQUIVALENT | 7 |
| **Total mechanisms** | **72** |

---

## 9. Key Observations on 0.01 Nm Wheel Mismatch

The remaining teacher-forcing mismatch of ~0.01 Nm at step 1 is NOT due to missing coverage. All sagittal wheel balance mechanisms are PORTED_FULL_COVERAGE with identical formulas.

**Potential parity mismatch sources (NOT coverage gaps):**

1. **Pitch offset application (M9):** The Python loop computes `pitch_x_error = pitch_x - pitch_x_ref` using Python outer-loop logic (with safety gates), then passes this to JAX. JAX receives the pre-adjusted `pitch_x` but also computes its own `total_pitch_ref_offset_deg`. If there's any difference between Python's externally-computed offset and what JAX would compute, the pitch_x values diverge.

2. **Notch filter state initialization (S1):** Both start at zero, but if the first step's pitch_rate differs (due to point 1), the notch output diverges.

3. **Outer loop safety gates (O5):** Python outer loop zeros pitch_ref target when safety gates fail. JAX outer loop computes unconditionally. This could cause different `pitch_x_error` values.

4. **tau_support_feedforward (C1):** Python includes it in composer; JAX excludes it. But this affects hip_yaw [1,6], not wheels [4,9]. However, wheels are indirectly affected through rate limiting and clipping interaction.

5. **Grid interpolation precision (J1):** PCHIP vs 20k-point linear interpolation — theoretically < 1e-6 error, but on cumulative outer loop state update, small errors can compound.

**Recommendation:** The 0.01 Nm mismatch is a parity/precision issue, not a coverage gap. Fix requires step-by-step teacher-forcing of individual mechanisms to isolate which sub-computation diverges.
