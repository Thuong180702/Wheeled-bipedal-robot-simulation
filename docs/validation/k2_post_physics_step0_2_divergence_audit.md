# K2 JAX Dedicated — Post-Physics Divergence Audit (Steps 0-2)

**Date:** 2026-06-30
**Phase:** 2 — TRACE POST-PHYSICS STATE BETWEEN STEPS 0-2
**Status:** Analysis in progress — structural audit complete, state-parity stepper next

---

## 1. Known Facts

From previous investigation (user-provided):
1. Steps 0-1 torques are bit-identical across all 10 actuators in dedicated vs Python source
2. Torque divergence begins at step 2
3. Pitch RMS metric/window/frame parity confirmed (same body_pitch_x, same formula, same window)
4. Divergence is NOT a metric artifact

---

## 2. Structural Equivalence Verified

### 2.1 Torque composer — EQUIVALENT

| Layer | Python (BalanceCoreTorqueComposer) | JAX (k2_jax_torque_composer_step) |
|-------|-------------------------------------|-------------------------------------|
| Summation | Sum 4 sources | Sum all sources |
| Clip | `jnp.clip(tau_sum, -torque_limit, torque_limit)` | Same |
| Rate limit | `tau_prev + clip(delta/dt, -max_rate, max_rate)*dt` | Same |
| Saturation detection | `abs(raw - clipped) > 1e-9` | Same |

### 2.2 Shape posture gains — EQUIVALENT

| Joint | Python ShapePostureController | JAX k2_jax_shape_posture_compute |
|-------|------------------------------|----------------------------------|
| hip_yaw | kp=15.0, kd=3.0 (configurable) | Same |
| hip_pitch | kp=30.0, kd=4.0 | Same |
| knee | kp=40.0, kd=5.0 | Same |
| hip_roll | Not in ShapePosture | kp=0.0, kd=0.0 |

**Note:** The comment in JAX code (lines 1041-1045) says "These gains differ from Python's LegPositionController" but the LegPositionController is ZEROED in balance-core mode (`zero_legacy_torque_sources_for_balance_core()` sets `tau_leg_position = zeros`). So the ShapePostureController gains ARE the correct comparison.

### 2.3 K2 profile scheduling — EQUIVALENT

| Parameter | K2_NOTCH_LOW_Q_V1 profile | JAX standalone | Match |
|-----------|--------------------------|----------------|-------|
| continuous_k_position | False → kpos=40.0 | kpos=40.0 | ✅ |
| continuous_k_velocity | False → k_velocity=15.0 | _k_velocity from params | ✅ |
| continuous_k_wheel_velocity | False → kwheel=0.5 | kwheel=0.5 | ✅ |
| continuous_kd_pitch | False → kd_pitch=10.0 | kd_pitch=10.0 | ✅ |
| continuous_max_position_tau | True → 4.0→6.0 | Scheduled identically | ✅ |
| velocity_damping_scale | 1.10 for active variants | compute_velocity_damping_scale() | ✅ |

### 2.4 WBC/LegPositionController — CORRECTLY ABSENT

In balance-core mode:
- `tau_wbc_correction = zeros`
- `tau_wbc_scaled = zeros`
- `tau_leg_position = zeros`
- `tau_posture = zeros` (legacy, replaced by tau_shape_posture)

The JAX standalone correctly omits these.

### 2.5 Active torque sources in balance-core — SAME

| Layer | Python | JAX | Equivalent? |
|-------|--------|-----|-------------|
| Sagittal balance | SagittalVelocityDampedBalanceController.compute() | k2_jax_sagittal_torque_assembly() | ✅ Same parameters |
| Shape posture | ShapePostureController.compute() | k2_jax_shape_posture_compute() | ✅ Same gains |
| Support FF | SupportFeedforwardController.compute() | k2_jax_empirical_support_ff() | ⚠️ Needs verification |
| Lateral roll | LateralRollBalanceController.compute() | k2_jax_lateral_roll_compute() | ✅ Same structure |
| Yaw | YawController.compute() | k2_jax_yaw_compute() | ✅ Same gains |
| Mode-div | ModeDivController.compute() | k2_jax_mode_div_compute() | ✅ Same parameters |

---

## 3. Potential Divergence Sources (NOT YET RULED OUT)

### 3.1 Centroidal estimator initialization difference

| | Python source | JAX dedicated |
|---|---|---|
| torso_inertia | `[0.1, 0.1, 0.05]` (hardcoded) | `[0.015241, 0.017751, 0.005598]` (from model) |
| Impact on pitch | None — pitch is from body quaternion, geometric computation | Same |
| Impact on CoM | Different inertia could shift CoM estimate | Needs verification |

**Verdict:** Unlikely to affect pitch, but could affect CoM-based terms (height tracking, support position). Needs verification via state-parity stepper.

### 3.2 Initialization sequence

Python source path:
1. Set qpos from height setup
2. `mj_forward` (with old root_z)
3. Set `calibrated_root_z_m`
4. `mj_forward` (with correct root_z)
5. Read equilibrium_joint_pos, orientation

Dedicated runner:
1. Set qpos from height setup + calibrated_root_z_m
2. `mj_forward` (ONCE with correct root_z)
3. Read equilibrium_joint_pos, orientation

**Potential issue:** The first `mj_forward` in Python path uses different root_z. MuJoCo's constraint solver uses warm-starting — the Lagrange multipliers from the first solve (at old root_z) are used as initial guess for the second solve (at correct root_z). This could produce slightly different constraint forces and thus slightly different eq_joint_pos or orientation.

**Verdict:** Needs verification via state-parity stepper.

### 3.3 Stateful terms (initial state + updates)

| Term | Python initial | JAX initial | Potential mismatch? |
|------|---------------|-------------|---------------------|
| Notch filter (x1, x2, y1, y2) | 0.0 | 0.0 (pack_state_k2) | ✅ |
| Prev torque (rate limiter) | zeros(10) | zeros(10) | ✅ |
| Filtered CoM height | 0.0 → first measurement | 0.0 → first measurement | ⚠️ |
| Prev support error | None → 0.0 on first step | 0.0 | ⚠️ |
| Outer loop pitch ref smoothed | 0.0 | 0.0 | ✅ |
| Outer loop prev supp err | None → set on first step | 0.0 | ⚠️ |
| ABS ring buffer | Empty → fills over N steps | Empty → fills over N steps | ✅ |
| APCR1ND state | All zeros | All zeros | ✅ |

**Potential issue:** The `filtered_com_z` first-measurement handling. In Python, `self._filtered_com_z` is initialized as 0.0 and updated via `alpha * prev + (1-alpha) * current`. In JAX, `new_filtered_com_z` uses this same formula. Both start at 0.0, so the first update should match.

### 3.4 Physics stepping subtle differences

Both paths use `n_substeps = control_dt / physics_dt = 5` `mj_step()` calls per control step. But:

| Aspect | Python source | JAX dedicated |
|--------|--------------|---------------|
| Push application | `_apply_pending_push()` between ctrl set and mj_step | Push applied BEFORE ctrl set in dedicated? |
| First step special handling | Debug prints + contact measurement between 1st and remaining substeps | No special handling |
| mj_forward before first ctrl | Yes (in calibration) | Yes |
| Post-physics estimation | centroidal_estimator.estimate(obs_42, mj_data, ...) | centroidal_estimator.estimate(np.zeros(42), mj_data, ...) |

**Potential issue:** The dedicated runner passes `np.zeros(42)` as the observation to the centroidal estimator, while the Python path passes the actual 42-dim observation. This affects the velocity estimation (filtering). The `np.zeros(42)` on the first call might produce different filtered velocity estimates than passing actual observation data.

**Verdict:** This is a likely candidate for subtle divergence. The observation contains IMU and joint data that affects the estimator's internal filtering. Starting with zeros vs actual data would produce different filtered values for at least the first few steps.

---

## 4. Next Steps

### Phase 3: State-parity stepper (DEFINITIVE TEST)

Implement `scripts/experiment_k2_state_parity_stepper.py` to run experiments A-F:

A. **Same state, two controllers:** Copy source qpos/qvel into dedicated path, compare torques
B. **Same torque, two physics:** Apply identical torque to both paths, compare post-physics state
C. **Cloned mj_model/mj_data:** Verify MuJoCo determinism
D. **Dedicated with source state reset:** Reset dedicated state to source state each step
E. **Source physics with dedicated torque:** Isolate physics from controller
F. **Dedicated physics with source torque:** Isolate controller from physics

### Priority investigation order:
1. **Experiment A first** — if torques match with same state, controller semantics are correct
2. **Experiment D second** — if resetting state each step fixes divergence, physics drift is root cause
3. **Experiment E/F third** — cross-apply torques to isolate physics vs controller
