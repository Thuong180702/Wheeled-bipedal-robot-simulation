# K2 JAX Dedicated Runner — Parameter and Initialization Parity Audit

**Date:** 2026-06-29
**Phase:** 1 — Parameter and Initialization Parity
**Status:** BLOCKED — 2 control-affecting parameter mismatches found

## Comparison Methodology

- **Canonical source of truth:** `scripts/simulate_hierarchical_controller.py` with `--controller-backend jax`, `--controller-mode balance-core`, `--sagittal-controller velocity-damped`, `--vd-sagittal-authority-profile k2_notch_low_q_v1`
- **Dedicated runner:** `scripts/run_k2_jax_realtime.py` with `--profile k2_notch_low_q_v1`
- **Parameter source:** `SagittalAuthoritySchedule` frozen dataclass fields inherited through the `K2_NOTCH_LOW_Q_V1` profile chain
- **Verification method:** Static code audit comparing `pack_params_stage2()` calls in both scripts

## Profile Inheritance Chain

```
K2_NOTCH_LOW_Q_V1 → K1_PITCH_RATE_NOTCH → PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2 → PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP → CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF_V2 → SUPPORT_POSITION_OUTER_LOOP_PITCH_REF → HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM → ... → SagittalAuthoritySchedule()
```

K2 overrides only: `profile_name="k2_notch_low_q_v1"`, `wip_notch_q=2.0`.
All other fields inherit from the chain; fields not overridden use `SagittalAuthoritySchedule` class defaults.

## JAX Parameter Comparison Table

Parameters passed to `pack_params_stage2()`:

| # | Field | Canonical Value | Dedicated Value | Source (Canonical) | Match? |
|---|---|---|---|---|---|
| 1 | `fs_hz` | 100.0 | 100.0 | hardcoded | ✓ PASS |
| 2 | `fc_hz` | 2.5 | 2.5 | hardcoded | ✓ PASS |
| 3 | `Q` | 2.0 | 2.0 | hardcoded | ✓ PASS |
| 4 | `torque_limit` | `mj_model.actuator_ctrlrange[:, 1]` | `mj_model.actuator_ctrlrange[:, 1]` | MuJoCo model | ✓ PASS |
| 5 | `max_torque_rate` | `np.ones(10) * 400.0` | `np.ones(10) * 400.0` | hardcoded | ✓ PASS |
| 6 | `control_dt` | `float(control_dt)` = 0.01 | 0.01 | hardcoded | ✓ PASS |
| 7 | `mode_div_soft_gain` | 0.80 | 0.80 | CLI default / hardcoded | ✓ PASS |
| 8 | `mode_div_ref_source` | `"disabled"` | `"disabled"` | `enable_mode_hip_yaw_divergence` defaults False | ✓ PASS |
| 9 | `k_velocity` | 15.0 | 15.0 | `SagittalVelocityDampedBalanceController.k_velocity` | ✓ PASS |
| 10 | **`velocity_damping_scale`** | **1.0** | **1.1** | `_auth_sched.velocity_damping_scale` via `is_active_for_variant()` | **✗ FAIL** |
| 11 | `apcr1nd_startup_guard_steps` | 100 | 100 | class default `recenter_priority_startup_guard_steps` | ✓ PASS |
| 12 | `apcr1nd_safe_min_com_z` | 0.27 | 0.27 | class default | ✓ PASS |
| 13 | `apcr1nd_safe_roll_rad` | 0.15 | 0.15 | class default | ✓ PASS |
| 14 | `apcr1nd_safe_pitch_rad` | 0.15 | 0.15 | class default | ✓ PASS |
| 15 | `apcr1nd_direct_enter_m` | 0.06 | 0.06 | class default `apcr1nd_direct_enter_m` | ✓ PASS |
| 16 | `apcr1nd_release_inner_m` | 0.03 | 0.03 | class default | ✓ PASS |
| 17 | **`apcr1nd_hold_outside_band`** | **False** | **True** | class default `apcr1nd_hold_outside_band: bool = False` | **✗ FAIL** |
| 18 | `apcr1nd_converging_release_steps` | 15 | 15 | class default | ✓ PASS |
| 19 | `standalone_mode` | True | True | `_jax_fast_path` / hardcoded | ✓ PASS |
| 20 | `pitch_x_eq_rad` | from `centroidal_state_eq` | from `compute_robot_frame_orientation_from_quaternion()` | Computed at init | ✓ PASS* |
| 21 | `support_center_eq_x_m` | from `support_center_eq_xy[0]` | `support_center_eq[0]` | `compute_support_center_xy()` at init | ✓ PASS* |
| 22 | `support_center_eq_y_m` | from `support_center_eq_xy[1]` | `support_center_eq[1]` | `compute_support_center_xy()` at init | ✓ PASS* |
| 23 | `sagittal_axis_x` | `sin(initial_yaw_z)` | `sin(yaw_z_eq)` | Computed at init | ✓ PASS* |
| 24 | `sagittal_axis_y` | `cos(initial_yaw_z)` | `cos(yaw_z_eq)` | Computed at init | ✓ PASS* |

*For equilibrium constants: identical computation method but actual values depend on initial model posture from the height setup JSON — will match when the same setup is loaded.

## JAX State Initialization

| Field | Canonical | Dedicated | Match? |
|---|---|---|---|
| State pack function | `pack_state_k2()` | `pack_state_k2()` | ✓ PASS |
| State size | `K2_JAX_STATE_SIZE` (834) | `K2_JAX_STATE_SIZE` (834) | ✓ PASS |
| Initial notch state | all zeros | all zeros | ✓ PASS |
| Initial prev_tau | all zeros | all zeros | ✓ PASS |
| Initial prev_support_error | 0.0 | 0.0 | ✓ PASS |
| Input size | `K2_JAX_INPUT_SIZE` = 45 | `K2_JAX_INPUT_SIZE` = 45 | ✓ PASS |
| JIT compile | `jax.jit(k2_jax_controller_step)` | `jax.jit(k2_jax_controller_step)` | ✓ PASS |

## Control-Affecting Mismatches — Detailed Analysis

### Mismatch 1: `velocity_damping_scale` (1.0 vs 1.1)

**Root cause:** The dedicated runner hardcodes `velocity_damping_scale=1.1` based on the assumption that K2 applies velocity_damping_scale=1.10 for supported height variants. However, the canonical `K2_NOTCH_LOW_Q_V1` profile inherits the class default `velocity_damping_scale=1.0` because:

1. `K2_NOTCH_LOW_Q_V1` overrides only `profile_name` and `wip_notch_q`
2. Its parent chain never overrides `velocity_damping_scale` (defaults to 1.0 in `SagittalAuthoritySchedule`)
3. `applies_to_variants` is empty (class default `= ()`), so `is_active_for_variant()` always returns False
4. The canonical script's `_eff_velocity_damping_scale` stays at 1.0

**Impact:** `velocity_damping_scale` is multiplied into the effective sagittal velocity damping gain inside JAX (`effective_k_velocity = k_velocity * velocity_damping_scale`). A value of 1.1 produces 10% stronger sagittal velocity damping than 1.0. This affects wheel torque computation and thus center-of-mass drift behavior.

**Severity:** CONTROL-AFFECTING — will produce different JAX torque outputs.

### Mismatch 2: `apcr1nd_hold_outside_band` (False vs True)

**Root cause:** The dedicated runner hardcodes `apcr1nd_hold_outside_band=True`. The `SagittalAuthoritySchedule` class default is `False` (line 772). K2_NOTCH_LOW_Q_V1 does not override this field.

**Impact:** When `hold_outside_band=False`, the APCR1ND recentering mechanism releases immediately once the support error falls below the release threshold (hysteresis is disabled). When `True`, the mechanism holds the boosted state until the error drops below the inner release band. This affects support position recentering behavior, particularly during push recovery where the support center drifts significantly.

**Severity:** CONTROL-AFFECTING — different APCR1ND gate behavior during transient support drift.

## Summary

| Category | Count |
|---|---|
| Total params compared | 24 |
| ✓ PASS | 22 |
| ✗ FAIL (control-affecting) | 2 |
| ✗ FAIL (diagnostic-only) | 0 |
| ⚠ PASS* (runtime-equivalent) | 4 |

## Recommendation

**BLOCKED.** Two control-affecting parameter mismatches must be resolved before promotion.

Options:

**Option A — Fix the K2 profile definition:**
- Add `velocity_damping_scale=1.10` and `apcr1nd_hold_outside_band=True` to `K2_NOTCH_LOW_Q_V1` via the `replace()` call
- Update `applies_to_variants` to include the supported height variants
- This aligns the source-of-truth profile with the intended K2 behavior

**Option B — Fix the dedicated runner:**
- Set `velocity_damping_scale` to 1.0
- Set `apcr1nd_hold_outside_band` to False
- This aligns the dedicated runner with current profile definition

**Recommendation:** Option A (fix the profile definition) is preferred because:
1. The K2 profile was designed with `velocity_damping_scale=1.10` in its documentation
2. `apcr1nd_hold_outside_band=True` is the intended behavior for push recovery
3. Other K2-variant profiles (APCR1n family) set these values
4. The current K2 profile appears to have an incomplete inheritance — it should have been based on a profile that includes these settings
5. Fixing the profile definition also fixes the canonical path, which is the actual source of truth
