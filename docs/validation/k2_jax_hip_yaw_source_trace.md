# K2 JAX Hip-Yaw Source Trace — Phase 0

## Purpose

Exact source-of-truth lock for hip-yaw actuator indices [1,6] in both
Python K2 (`simulate_hierarchical_controller.py`) and JAX K2
(`k2_jax_controller.py`).

## Python K2 Source of Truth

### 1. ShapePostureController.compute() — PD torque on [1,6]

**File:** `wheeled_biped/controllers/shape_posture_controller.py:367-372`

```python
for idx in [1, 6]:
    tau_pd = self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx]
    tau_comp = tau_comp_left_final if idx == 1 else tau_comp_right_final
    tau_div = tau_div_left_raw if idx == 1 else tau_div_right_raw
    tau_total = authority_scale * tau_pd + tau_comp + tau_div
    tau = tau.at[idx].set(tau_total)
```

**Active components for K2 validation (stage6l_phase1_lockstep_trace.py:77-84):**
- `enable_hip_yaw_support_feedforward=False` (default) → `tau_comp = 0`
- `enable_hip_yaw_divergence_damping=False` (default) → `tau_div = 0`
- `authority_scale = posture_weight * contact_degraded_scale = 1.0 * 1.0 = 1.0`

**Effective formula (K2):**
```
tau_shape_posture[1] = kp_hip_yaw * (q_ref[1] - q[1]) - kd_hip_yaw * qd[1]
tau_shape_posture[6] = kp_hip_yaw * (q_ref[6] - q[6]) - kd_hip_yaw * qd[6]
```

**Gains:** kp_hip_yaw=15.0, kd_hip_yaw=3.0 (BALANCE_CORE_HIP_YAW_AUTHORITY, default when args.shape_kp_hip_yaw is None)

**Call site:** `simulate_hierarchical_controller.py:5958-5966`
```python
tau_shape_posture, shape_diag = balance_core_controllers["shape_posture"].compute(
    q_ref=equilibrium_joint_pos,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    posture_weight=1.0,
    contact_degraded_scale=1.0,
    ...
)
```

### 2. YawController.compute() — antisymmetric yaw torque on [1,6]

**File:** `wheeled_biped/controllers/yaw_controller.py:34-74`

```python
tau_antisym_raw = self.kp_yaw * yaw_error - self.kd_yaw * yaw_rate
tau_antisym = jnp.clip(tau_antisym_raw, -self.max_yaw_torque, self.max_yaw_torque)
tau = zeros_action()
tau = tau.at[1].set(-tau_antisym)  # left hip-yaw
tau = tau.at[6].set(tau_antisym)   # right hip-yaw
```

**Gains:** kp_yaw=8.0, kd_yaw=2.0, max_yaw_torque=5.0 (defaults)

**Call site:** `simulate_hierarchical_controller.py:6372-6378` (legacy path, wheel_yaw_enabled=False for K2 profile)
```python
tau_yaw, yaw_diag = balance_core_controllers["yaw_controller"].compute(
    yaw_error=yaw_error,
    yaw_rate=yaw_rate,
)
tau_shape_posture_with_yaw = tau_shape_posture.at[1].add(tau_yaw[1])
tau_shape_posture_with_yaw = tau_shape_posture_with_yaw.at[6].add(tau_yaw[6])
```

**Sign convention:** `tau_yaw[1] = -tau_antisym`, `tau_yaw[6] = +tau_antisym`

### 3. ModeBasedHipYawDivergenceController.compute() — mode-div torque on [1,6]

**File:** `wheeled_biped/controllers/mode_based_hip_yaw_divergence_controller.py:119-169`

```python
raw = -(self.kp_div * state.div_error + self.kd_div * state.div_rate)
gate = self._height_gate(state.height)  # smoothstep_down at [0.30, 1.10]
torque = raw * gate
torque_clipped = clip(torque, -self.max_torque, self.max_torque)
tau_left = torque_clipped
tau_right = -torque_clipped
```

**Gains (K2 validation):** kp_div=10.0, kd_div=0.50, max_torque=7.5, soft_limit_rad=0.30, soft_gain=0.80, support_gate_enabled=False

**Input computation:** `simulate_hierarchical_controller.py:6457-6478`
```python
l_ref = equilibrium_joint_pos[1], r_ref = equilibrium_joint_pos[6]
ref_common, ref_div = decompose(l_ref, r_ref)        # ref_div = l_ref - r_ref
_act_common, actual_div = decompose(l_pos, r_pos)    # actual_div = l_pos - r_pos
div_rate = l_vel - r_vel
div_error = actual_div - ref_div = (l_pos - r_pos) - (l_ref - r_ref)
```

**Call site:** `simulate_hierarchical_controller.py:6506-6507`
```python
tau_shape_posture_with_yaw = tau_shape_posture_with_yaw.at[1].add(mode_div_tau_left)
tau_shape_posture_with_yaw = tau_shape_posture_with_yaw.at[6].add(mode_div_tau_right)
```

### 4. BalanceCoreTorqueComposer.compose() — clipping + rate limiting

**File:** `wheeled_biped/controllers/balance_core_torque_composer.py:50-155`

```python
tau_total_raw = tau_shape_posture + tau_support_feedforward + tau_sagittal_wheel_balance + tau_lateral_roll_balance
tau_total_clipped = jnp.clip(tau_total_raw, -self.torque_limit, self.torque_limit)
delta_desired = tau_total_clipped - tau_prev
delta_rate = delta_desired / self.control_dt
delta_rate_limited = jnp.clip(delta_rate, -self.max_torque_rate, self.max_torque_rate)
tau_final = tau_prev + delta_rate_limited * self.control_dt
```

**Note:** tau_support_feedforward (from SupportFeedforwardController) is ZERO at indices [1,6] (hip_pitch/knee only).
tau_sagittal_wheel_balance is ZERO at [1,6] (wheels only).
tau_lateral_roll_balance is ZERO at [1,6] (hip_roll only).

**Therefore at [1,6]:**
```
tau_total_raw[1] = tau_shape_posture_with_yaw[1] + 0 + 0 + 0
tau_total_raw[6] = tau_shape_posture_with_yaw[6] + 0 + 0 + 0
```

### 5. Simulation loop summary — Python hip-yaw assembly order

```
tau_shape_posture[1,6]  ←  PD only (kp=15.0, kd=3.0, authority=1.0)
    + tau_yaw[1,6]      ←  antisymmetric yaw (kp=8.0, kd=2.0, clamp=5.0)
    + mode_div[1,6]     ←  antisymmetric mode-div (kp=10.0, kd=0.50, clamp=7.5)
    → tau_shape_posture_with_yaw
    → composer clip + rate-limit
    → tau_final[1,6]
```

## JAX K2 Source

### 1. k2_jax_shape_posture_compute()

**File:** `wheeled_biped/controllers/k2_jax_controller.py:704-726`

```python
error = q_ref - joint_pos
authority = posture_weight * contact_degraded_scale  # = 1.0
tau = tau.at[1].set(authority * (kp_hip_yaw * error[1] - kd_hip_yaw * joint_vel[1]))
tau = tau.at[6].set(authority * (kp_hip_yaw * error[6] - kd_hip_yaw * joint_vel[6]))
```

**Defaults:** kp_hip_yaw=15.0, kd_hip_yaw=3.0, posture_weight=1.0, contact_degraded_scale=1.0

**No HY-FF. No HY2-DIV.**

### 2. k2_jax_yaw_compute()

**File:** `wheeled_biped/controllers/k2_jax_controller.py:757-763`

```python
tau_antisym = jnp.clip(kp_yaw * yaw_error_rad - kd_yaw * yaw_rate_rad_s, -max_yaw_torque, max_yaw_torque)
tau = tau.at[1].set(-tau_antisym)
tau = tau.at[6].set(tau_antisym)
```

**Defaults:** kp_yaw=8.0, kd_yaw=2.0, max_yaw_torque=5.0

### 3. k2_jax_mode_div_compute()

**File:** `wheeled_biped/controllers/k2_jax_controller.py:766-783`

```python
raw = -(kp_div * div_error + kd_div * div_rate)
z_low, z_high = soft_limit_rad, soft_limit_rad + soft_gain  # 0.30, 1.10
u_h = (z_high - height_m) / (z_high - z_low)
height_gate = _jax_smoothstep01(u_h)
torque = raw * height_gate
torque_clipped = jnp.clip(torque, -max_torque, max_torque)
tau = tau.at[1].set(torque_clipped)
tau = tau.at[6].set(-torque_clipped)
```

**Defaults:** kp_div=10.0, kd_div=0.50, max_torque=7.5, soft_limit_rad=0.30, soft_gain=0.50
**Runtime:** soft_gain overridden from params (0.80 in K2 validation)

**No support gate** (unlike Python ModeBasedHipYawDivergenceController which has `_support_error_gate` and `_support_rate_gate` — but these are disabled by default with `support_gate_enabled=False`).

### 4. k2_jax_torque_composer_step()

**File:** `wheeled_biped/controllers/k2_jax_controller.py:301-347`

```python
tau_clipped = jnp.clip(tau_sum, -torque_limit, torque_limit)
delta_desired = tau_clipped - tau_prev
delta_rate = delta_desired / control_dt
delta_rate_limited = jnp.clip(delta_rate, -max_torque_rate, max_torque_rate)
tau_final = tau_prev + delta_rate_limited * control_dt
```

**Matches Python composer exactly.**

### 5. k2_jax_controller_step() assembly order

**File:** `wheeled_biped/controllers/k2_jax_controller.py:1422-1483`

```python
tau_posture, _ = k2_jax_shape_posture_compute(q_ref_full, joint_pos_full, joint_vel_full)
tau_yaw = k2_jax_yaw_compute(yaw_err, yaw_rate)
tau_mode_div = k2_jax_mode_div_compute(hy_div_err, hy_div_rate, schedule_h, soft_gain=_mode_div_soft_gain)

tau_posture_with_yaw = tau_posture.at[1].add(tau_yaw[1])
tau_posture_with_yaw = tau_posture_with_yaw.at[6].add(tau_yaw[6])
tau_posture_with_yaw = tau_posture_with_yaw.at[1].add(tau_mode_div[1])
tau_posture_with_yaw = tau_posture_with_yaw.at[6].add(tau_mode_div[6])

tau_sum = tau_sag + tau_posture_with_yaw + tau_lateral + k2_jax_empirical_support_ff()
tau_final, tau_clipped, sat_mask, rate_mask = k2_jax_torque_composer_step(tau_sum, prev_tau, params_flat)
```

**k2_jax_empirical_support_ff()** = `[0, 0, 2.05, -7.75, 0, 0, 0, 1.6, -7.9, 0]` — ZERO at [1,6].

## State-Synced Teacher-Forcing

**Python state capture** (`simulate_hierarchical_controller.py:5912-5939`):
- Captured BEFORE Python computes each step
- Includes: notch state (x1,x2,y1,y2), prev_tau (10), filtered_com_z, prev_support_error, outer_loop state, ABS state

**JAX state packing** (`simulate_hierarchical_controller.py:6571` via `pack_state_from_python_k2`):
- Uses pre-snapshot notch values to avoid reference-mutation bug
- Copies prev_tau from captured Python state

**JAX input packing** (`simulate_hierarchical_controller.py:6539-6560`):
- `yaw_error_rad`: `body_yaw_z - initial_yaw_z` (same as Python)
- `yaw_rate_rad_s`: `body_yaw_rate_z` (same as Python)
- `hip_yaw_div_error`: `(joint_pos[1] - joint_pos[6]) - (equilibrium_joint_pos[1] - equilibrium_joint_pos[6])` — matches Python `actual_div - ref_div`
- `hip_yaw_div_rate`: `joint_vel[1] - joint_vel[6]` — matches Python `l_vel - r_vel`
- `joint_pos, joint_vel, q_ref`: from simulation (same as Python)

## Gain Comparison Table

| Parameter | Python K2 | JAX K2 | Match? |
|-----------|-----------|--------|--------|
| Shape kp_hip_yaw | 15.0 | 15.0 (default) | ✓ |
| Shape kd_hip_yaw | 3.0 | 3.0 (default) | ✓ |
| Shape posture_weight | 1.0 | 1.0 (default) | ✓ |
| Shape contact_degraded_scale | 1.0 | 1.0 (default) | ✓ |
| Shape HY-FF | Disabled | N/A | ✓ |
| Shape HY2-DIV | Disabled | N/A | ✓ |
| Yaw kp | 8.0 | 8.0 (default) | ✓ |
| Yaw kd | 2.0 | 2.0 (default) | ✓ |
| Yaw max_torque | 5.0 | 5.0 (default) | ✓ |
| Yaw sign (left) | -tau_antisym | -tau_antisym | ✓ |
| Yaw sign (right) | +tau_antisym | +tau_antisym | ✓ |
| Mode-div kp | 10.0 | 10.0 (default) | ✓ |
| Mode-div kd | 0.50 | 0.50 (default) | ✓ |
| Mode-div max_torque | 7.5 | 7.5 (default) | ✓ |
| Mode-div soft_limit_rad | 0.30 | 0.30 (default) | ✓ |
| Mode-div soft_gain | 0.80 | 0.80 (from params) | ✓ |
| Mode-div sign (raw) | -(kp*err + kd*rate) | -(kp*err + kd*rate) | ✓ |
| Mode-div sign (left) | +torque_clipped | +torque_clipped | ✓ |
| Mode-div sign (right) | -torque_clipped | -torque_clipped | ✓ |
| Mode-div support_gate | Disabled | N/A | ✓ |
| Div error formula | (l_pos-r_pos)-(l_ref-r_ref) | (l_pos-r_pos)-(l_ref-r_ref) | ✓ |
| Div rate formula | l_vel - r_vel | l_vel - r_vel | ✓ |
| Height gate (low→high) | 0.30→1.10 smoothstep | 0.30→1.10 smoothstep | ✓ |
| Composer clip | torque_limit | torque_limit (from params) | ✓ |
| Composer rate-limit | max_torque_rate | max_torque_rate (from params) | ✓ |
| Composer dt | control_dt | control_dt (from params) | ✓ |
| Support FF at [1,6] | 0.0 | 0.0 | ✓ |
| Sagittal at [1,6] | 0.0 | 0.0 | ✓ |
| Lateral at [1,6] | 0.0 | 0.0 | ✓ |

## Summation Order

Both Python and JAX follow the same effective summation at [1,6]:
```
tau_raw[1] = shape_posture[1] + yaw[1] + mode_div[1]
tau_raw[6] = shape_posture[6] + yaw[6] + mode_div[6]
```
Then: clip to torque_limit → rate-limit vs prev_tau → tau_final

Non-hip-yaw sources (support FF, sagittal, lateral) are all zero at [1,6], verified.

## Potential Divergence Sources (Identified)

Based on the trace, ALL formula-level elements match between Python and JAX:
gains, sign conventions, input formulas, gate formulas, summation order, and composer logic.

**This suggests the divergence must come from one of:**

1. **State leak in state-synced mode**: If prev_tau or other state elements are not perfectly captured/re-packed, JAX starts from a slightly different state than Python, and the composer rate-limit compounds the difference.

2. **Floating-point ordering**: If the same mathematical expression is evaluated in different order (e.g., `a + b + c` vs `a + c + b`), floating-point non-associativity can produce tiny differences that grow through rate-limiting. This is especially relevant for the `tau_posture_with_yaw` composition.

3. **Mode-div ALWAYS active in JAX**: JAX always calls `k2_jax_mode_div_compute()` even when Python's `mode_hip_yaw_div_enabled` would be False. When inputs are zero, output is zero — but the function still runs. With nonzero inputs (always true in practice), both sides produce output.

4. **Schedule height (`schedule_h`)**: JAX uses `jnp.where(height_ref > 0, height_ref, 0.9*filtered_com_z + 0.1*com_z)`. Python uses `height_cmd` directly. For mode-div gate, JAX uses `schedule_h` while Python uses `centroidal_state_control.com_pos[2]`. If these differ, the height gate would differ, causing different mode-div torque.

## Next Step: Phase 1 — Instrumentation

The Phase 1 decomposition MUST log every scalar input to hip-yaw computation in both Python and JAX to identify the first scalar that differs. This will immediately reveal whether the divergence is in:
- Input values (q/q_ref/qd/yaw_error/yaw_rate/div_error/div_rate/height)
- Shape posture PD output
- Yaw output  
- Mode-div output
- Summation before composer
- Composer output
