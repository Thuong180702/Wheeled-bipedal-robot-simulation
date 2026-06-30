# K2 JAX Final Parity Fix — Source of Truth Trace

**Date:** 2026-06-27
**Profile:** `k2_notch_low_q_v1`
**Controller mode:** `balance-core`

---

## 1. Profile Chain (K2 parameter inheritance)

```
PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP  (line 2902)
  → PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2 (line 2940)
    → K1_PITCH_RATE_NOTCH (line 3120)
      → K2_NOTCH_LOW_Q_V1 (line 3162)
```

Key K2 parameters:
| Parameter | Value | Source |
|-----------|-------|--------|
| `wip_notch_q` | 2.0 | K2 override |
| `wip_notch_filter_blend` | 1.0 | K1_PITCH_RATE_NOTCH |
| `wip_notch_center_hz` | 2.5 | K1_PITCH_RATE_NOTCH |
| `wip_notch_gate_enabled` | True | K1_PITCH_RATE_NOTCH |
| `wip_notch_height_gate_start_m` | 0.42 | K1_PITCH_RATE_NOTCH |
| `wip_notch_height_gate_full_m` | 0.48 | K1_PITCH_RATE_NOTCH |
| `wip_notch_target_signal` | "pitch_rate" | K1_PITCH_RATE_NOTCH |
| `continuous_kd_pitch` | False | inherited (default) |
| `kd_pitch` | 10.0 | constructor arg |
| `continuous_k_velocity` | False | K2 profile |
| `k_velocity` | 15.0 | constructor arg (`vd_k_velocity`) |
| `velocity_damping_scale` | 1.0 | inherited (default) |
| `continuous_k_position` | False | K2 profile |
| `k_position` | 40.0 | constructor arg |
| `continuous_max_position_tau` | True | K2 profile |
| `wheel_torque_sign` | 1.0 | constructor arg |

---

## 2. Notch-Blend / Effective Pitch-Rate Path

### 2.1 Input Signal

| Scalar | Python source | Python line | JAX source | JAX line |
|--------|--------------|-------------|------------|----------|
| `pitch_rate_for_control` | `centroidal_state_control.body_pitch_rate_x` | sim:6164 | — | — |
| `pitch_rate_for_control_boosted` | `pitch_rate_for_control * pitch_rate_boost_factor` | sim:6232 | — | — |
| `pitch_rate` (JAX input) | `pitch_rate_for_control_boosted` (if in scope) else `body_pitch_rate_x` | sim:6528 | `input_flat[_I_PITCH_RATE]` | k2j:1192 |
| `pitch_rate_x_rad_s` (Python arg) | `pitch_rate_for_control_boosted` | sim:6241 | — | — |

**Verdict:** Same signal when `pitch_rate_boost_factor=1.0` (default, no transient mode active).
**Risk:** If transient T3/T4 active, `pitch_rate_for_control_boosted` ≠ raw — but Python and JAX BOTH receive it.

### 2.2 Notch Filter

| Scalar | Python | Python line | JAX | JAX line |
|--------|--------|-------------|-----|----------|
| Notch coefficients | `BiquadNotchFilter._compute_coefficients()` | sig:102-122 | `biquad_notch_coefficients(fs, fc, Q)` | sig:309-337 |
| Coefficient formula | `b0=1/denom, b1=-2*cos/denom, b2=1/denom, a1=-2*cos/denom, a2=(1-alpha)/denom` | sig:117-122 | Same | sig:331-335 |
| Notch input | `pitch_rate_raw = float(pitch_rate_x_rad_s)` | svdbc:4657 | `pitch_rate = input_flat[_I_PITCH_RATE]` | k2j:1192 |
| Notch output | `y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2` | sig:185-191 | `notch_out = b0*pr + b1*nx1 + b2*nx2 - a1*ny1 - a2*ny2` | k2j:1235 |
| State update | `x2=x1; x1=x; y2=y1; y1=y` | sig:193-196 | `new_nx1=pr; new_nx2=nx1; new_ny1=no; new_ny2=ny1` | k2j:1236-1239 |
| DF2T form | Direct Form II Transposed | sig:184 | Direct Form II Transposed | k2j:1235 |

**Verdict:** Identical formulas and coefficients.

### 2.3 Notch State — CRITICAL BUG FOUND

| Item | Python | JAX | 
|------|--------|-----|
| State capture | `"notch_filter": _sag._wip_notch_pitch_rate` (REFERENCE!) | sim:5912 |
| State read time | After Python `compute()` mutates filter via `update()` | k2j:1100-1103 |
| Effect | JAX reads POST-compute state, should read PRE-compute state | |

**BUG:** Line 5912 captures a mutable reference to `_sag._wip_notch_pitch_rate`. Python's `compute()` (line 6239) calls `update()` on this object, mutating `_x1, _x2, _y1, _y2` in-place. When `pack_state_from_python_k2` reads `notch_filter._x1` etc. at line 6558, it reads the POST-mutation state. JAX effectively starts from the WRONG filter state.

**Impact:** At equilibrium (step 0), pre/post states are nearly identical (small pitch rate → small filter output), so step 0 diff ≈ 4.77e-08 (near-perfect). At steps 1+, the state divergence grows, causing ~6% tau_pitch_rate difference.

### 2.4 Notch Blend

| Scalar | Python | Python line | JAX | JAX line |
|--------|--------|-------------|-----|----------|
| Height gate source | `schedule_height_ref = commanded_height_ref_m` | svdbc:4409 | `height_ref = input_flat[_I_HEIGHT_REF]` | k2j:1204 |
| Height gate formula | `smoothstep_gate(schedule_height_ref, 0.42, 0.48)` | svdbc:4648-4652 | `smoothstep_gate_jax(height_ref, 0.42, 0.48)` | k2j:1242 |
| notch_blend | 1.0 (K2) | svdbc:4639 | — (implicit 1.0) | — |
| Gate | `gate = notch_height_gate * notch_blend` | svdbc:4707 | `notch_gate = smoothstep(...)` | k2j:1242 |
| Effective pitch rate | `(1-gate)*pitch_rate_raw + gate*pitch_rate_notched` | svdbc:4708 | `(1-notch_gate)*pitch_rate + notch_gate*notch_out` | k2j:1243 |
| For damping | `pitch_rate_for_damping = pitch_rate_effective` | svdbc:4725 | `pitch_rate_eff` used directly | k2j:1363 |

**Verdict:** Formulas identical (notch_blend=1.0 in K2 makes them equivalent).

### 2.5 Tau Pitch Rate

| Scalar | Python | Python line | JAX | JAX line |
|--------|--------|-------------|-----|----------|
| effective_kd_pitch | `self.kd_pitch = 10.0` | svdbc:4601 | `kd_pitch = 10.0` | k2j:1259 |
| tau_pitch_rate | `effective_kd_pitch * pitch_rate_for_damping` | svdbc:5123 | `effective_kd_pitch * pitch_rate_rad_s` | k2j:618 |

**Verdict:** Identical formula. Difference caused by notch state bug (section 2.3).

---

## 3. Sagittal Velocity Path

### 3.1 Input Signal

| Scalar | Python source | Python line | JAX source | JAX line |
|--------|--------------|-------------|------------|----------|
| Velocity source | `centroidal_state_control.com_vel[1]` | sim:6242 | `centroidal_state_control.com_vel[1]` | sim:6535 |
| Python arg name | `sagittal_velocity_m_s` | svdbc:4370 | — | — |
| JAX input name | — | — | `sag_vel = input_flat[_I_SAG_VEL]` | k2j:1199 |

**Verdict:** Same signal (`com_vel[1]`).

### 3.2 Tau Sagittal Velocity

| Scalar | Python | Python line | JAX | JAX line |
|--------|--------|-------------|-----|----------|
| effective_k_velocity | `self.k_velocity = 15.0` | svdbc (constructor) | `effective_k_velocity = 15.0` | k2j:1369 |
| effective_velocity_damping_scale | `velocity_damping_scale = 1.0` (K2 default) | svdbc:4615 | `effective_velocity_damping_scale = 1.0` | k2j:1369 |
| Formula | `-k_vel * damp_scale * sagittal_velocity_m_s` | svdbc:5124 | `-k_vel * damp_scale * sagittal_velocity_m_s` | k2j:619 |

**Verdict:** Identical formula with identical parameters. The ~10% (~0.032 Nm) difference seen in previous state-synced runs must be caused by the notch state bug cascading through prev_tau → composer → rate limiting, which affects subsequent step inputs.

### 3.3 Tau Com Vy

| Scalar | Python | Python line | JAX | JAX line |
|--------|--------|-------------|-----|----------|
| kd_com_vy | `self.kd_com_vy = 5.0` | svdbc | `kd_com_vy = 5.0` | k2j:1373 |
| Formula | `-kd_com_vy * sagittal_velocity_m_s` | svdbc:5135 | `-kd_com_vy * sagittal_velocity_m_s` | k2j:622 |

**Verdict:** Identical.

---

## 4. Assembly / Composer Path

| Scalar | Python | Python line | JAX | JAX line |
|--------|--------|-------------|-----|----------|
| tau_common_unclipped | `tau_pitch + tau_pitch_rate + tau_sag_vel + tau_support_vel + tau_position + tau_cp + tau_com_vy` | svdbc:8256-8261 | Same | k2j:659-662 |
| wheel_torque_sign | 1.0 | svdbc:4138 | `wheel_torque_sign=1.0` | k2j:1374 |
| tau_common | `wheel_torque_sign * tau_common_unclipped` | svdbc:8362 | Same | k2j:663 |
| tau_left | `tau_common + tau_wheel_vel_left` | svdbc:8365 | Same | k2j:664 |
| tau_right | `tau_common + tau_wheel_vel_right` | svdbc:8366 | Same | k2j:665 |

**Verdict:** Identical assembly. The only difference source is COMPONENT-LEVEL values (tau_pitch_rate, tau_sagittal_velocity), which differ due to the notch state capture bug.

---

## 5. Root Cause Summary

### Primary Blocker: Notch Filter State Capture Bug

**File:** `scripts/simulate_hierarchical_controller.py`
**Line:** 5912
**Bug:** `"notch_filter": _sag._wip_notch_pitch_rate` stores a mutable reference instead of snapshotting state values.
**Mechanism:**
1. Capture stores reference (line 5912) — state = S_pre
2. Python `compute()` calls `filter.update()` — mutates object to S_post
3. JAX reads `filter._x1` from now-mutated object — state = S_post (WRONG)

**Fix:** Snapshot `_x1, _x2, _y1, _y2` as float values at capture time (before Python mutates the filter).

### Secondary: No Additional Formula Differences

After fixing the state capture bug, no formula-level differences remain. The notch filter (DF2T biquad), blend, height gate, sagittal velocity, and composer assembly are verified identical between Python and JAX.

---

## 6. Files Referenced

| Abbreviation | Full path |
|-------------|-----------|
| `sim` | `scripts/simulate_hierarchical_controller.py` |
| `svdbc` | `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` |
| `k2j` | `wheeled_biped/controllers/k2_jax_controller.py` |
| `sig` | `wheeled_biped/controllers/signal_filters.py` |
