# K2 JAX JIT Controller Stack Port — Design Document

**Date:** 2026-06-26
**Status:** Reviewed — approved for implementation planning after corrections
**Target:** PORT_K2_CONTROLLER_STACK_TO_JAX_JIT_WITH_STRICT_PARITY

---

## 1. Executive Summary

Port the K2_NOTCH_LOW_Q_V1 balance-core controller computation path from eager Python to
JAX JIT, achieving realtime (≤10 ms/step for 100 Hz target) while preserving **exact K2
behavior**. The Python reference implementation is preserved as oracle and fallback.

**Non-goals:** change K2 gains, notch topology, torque limits, control principle, or
any behavior. No WBC, no hidden torque, no new stabilizing terms.

---

## 2. Current Baseline

| Property | Value |
|----------|-------|
| Current-best ID | `K2_NOTCH_LOW_Q_V1` |
| Profile name | `k2_notch_low_q_v1` |
| Profile chain | K2 → K1 → PHYSICS_EQ_FF_OUTER_LOOP_LOW_BAND_SUPPORT_V2 → PHYSICS_EQ_FF_OUTER_LOOP → CALIBRATED_OUTER_LOOP_V2 → SUPPORT_POSITION_OUTER_LOOP → HEIGHT_SCHEDULED_PITCH_EQ_TRIM → ADAPTIVE_SUPPORT_CENTERING_TRIM |
| Notch fc | 2.5 Hz |
| Notch Q | 2.0 |
| Notch blend | 1.0 |
| Height gate | 0.42–0.48 m |
| Mode-div Kp/Kd | 10.0 / 0.50 |
| Mode-div max torque | 7.5 Nm |
| Measured controller cost | ~106 ms/step |
| Target controller cost | ≤10 ms/step (100 Hz) |

### 2.1 K2-Active Strategies (from profile chain inheritance)

Only these strategies are **enabled** for K2. Each must be classified during the branch
activity audit (§7) using the 4-level system:

- **ENABLED_ACTIVE:** branch enabled and contributes nonzero output
- **ENABLED_GATED_ZERO:** branch enabled but runtime output is zero due to gate/deadband/height/contact
- **DISABLED_INACTIVE:** branch disabled and must never execute
- **UNEXPECTED_ACTIVE:** branch executed unexpectedly and should fail audit

| Strategy | Enabled by | Expected Classification | State required |
|----------|-----------|------------------------|----------------|
| WIP notch filter (pitch_rate) | K1_PITCH_RATE_NOTCH | ENABLED_GATED_ZERO below 0.42 m, ENABLED_ACTIVE above 0.48 m | notch_x1,x2,y1,y2 (4) |
| Continuous height scheduling (k_position, etc.) | Always active | ENABLED_ACTIVE | filtered_com_z (1) |
| Physics equilibrium feedforward | PHYSICS_EQ_FF_OUTER_LOOP | ENABLED_ACTIVE (gated to zero outside calibration height range) | — (stateless, height lookup) |
| Low-band support outer loop (v2) | LOW_BAND_SUPPORT_V2 | ENABLED_GATED_ZERO except near 0.320 m | — (stateless, height-gated) |
| Calibrated outer loop (v2) | CALIBRATED_OUTER_LOOP_V2 | ENABLED_ACTIVE | prev_support_error (1) |
| Support-position outer loop pitch ref | SUPPORT_POSITION_OUTER_LOOP | ENABLED_ACTIVE (deadband-gated) | (shared with calibrated) |
| Height-scheduled pitch equilibrium trim | HEIGHT_SCHEDULED_PITCH_EQ_TRIM | ENABLED_ACTIVE | — (stateless) |
| Torque rate limiting / smoothing | Always active | ENABLED_ACTIVE | prev_tau (10) |
| Sagittal legacy torque assembly | Always active | ENABLED_ACTIVE | (various internal) |

### 2.2 Strategies Proven Inactive for K2

| Strategy | Default | K2 Value | Active? |
|----------|---------|----------|---------|
| `enable_unified_sagittal_state_feedback` | False | False | No |
| `enable_active_pitch_crossing` | False | False | No |
| `enable_phase_aware_recenter` | False | False | No |
| `enable_hysteresis_recenter` | False | False | No |
| `enable_bias_cancel` | False | False | No |
| `enable_position_integral` | False | False | No |
| `enable_capture_gate` | False | False | No |
| `t6j_bias_trim_enabled` | False | False | No |
| `adaptive_bias_trim_enabled` | False | False | No |
| `pitch_bias_comp_enabled` | False | False | No |
| `enable_body_yaw_wheel_stabilization` | False | False | No |
| `apcr1nd_*` variants | False | False | No |
| ZC/EZC recenter | False | False | No |
| T6H/I/J soft blend/release | False | False | No |
| L/LR/LP family | False | False | No |

**All disabled strategy branches must be proven inactive via tests across Step C, D, E,
single-push, and dynamic-height cases.** (§7)

---

## 3. Architecture

### 3.1 JIT Boundary

```
┌──────────────────────────────────────────────────────────────┐
│  Python (non-JIT) — per MuJoCo simulation step               │
│                                                              │
│  1. Extract numeric arrays from mj_data (qpos, qvel, etc.)   │
│  2. [Stage 1] Profile centroidal_estimator — remove dup calls│
│  3. [Stage 1] Profile capture_estimator — remove dup calls   │
│  4. Pack Python state → jax_state_flat (S,)                  │
│  5. Pack numeric inputs → jax_input_flat (I,)                │
│                                                              │
│     ┌────── JAX JIT boundary ──────────────────────┐         │
│     │  k2_jax_controller_step(                     │         │
│     │      state_flat: (S,),                       │         │
│     │      input_flat: (I,),                       │         │
│     │      params_flat: (P,),                      │         │
│     │  ) -> (tau: (10,), state_flat': (S,),        │         │
│     │         diag_flat: (D,))                     │         │
│     └──────────────────────────────────────────────┘         │
│                                                              │
│  6. Unpack state_flat' → update Python reference state       │
│  7. Map diag_flat → telemetry dict (non-realtime, Python)    │
│  8. Apply tau to mj_data.ctrl[:]                             │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Data Representation Rules

| Concern | Approach |
|---------|----------|
| **State** | Flat `jnp.ndarray` shape (S,), dtype float64. `K2_JAX_STATE_FIELDS` constant defines order. Python pack/unpack helpers at boundary. |
| **Inputs** | Flat `jnp.ndarray` shape (I,), dtype float64. `K2_JAX_INPUT_FIELDS` constant defines order. Packed from MuJoCo extracts. |
| **Params** | Flat `jnp.ndarray` shape (P,), dtype float64. `K2_JAX_PARAMS_FIELDS` constant defines order. Compiled from `SagittalAuthoritySchedule` + constructor args at init. Includes pre-evaluated interpolation grids as fixed-shape sub-arrays. |
| **Diagnostics** | Flat `jnp.ndarray` shape (D,), dtype float64. `K2_JAX_DIAG_FIELDS` constant defines order. Python `k2_jax_diag_flat_to_dict()` mapper outside JIT. Full mapping for validation; optional decimated mapping for realtime visual. |
| **Inside JIT** | No Python dicts, no string keys, no dynamic shapes, no mutable state, no object construction. Fixed integer indices for state/input/param/diag access. |
| **Precision** | **float64 for parity** (JAX x64 enabled). float32 is a later performance variant only after multi-step parity passes. |

### 3.3 File Plan

#### New Files

| File | Purpose |
|------|---------|
| `wheeled_biped/controllers/k2_jax_controller.py` | JAX JIT controller kernel, field constants, pack/unpack helpers, diag mapper, params compiler |
| `scripts/compare_k2_python_vs_jax_step.py` | Step-by-step Python vs JAX parity harness |
| `scripts/benchmark_k2_jax_controller.py` | Performance benchmark (cold/warm, headless/visual) |
| `tests/test_k2_jax_component_parity.py` | Per-component parity tests |
| `tests/test_k2_jax_step_parity.py` | Full-step and multi-step parity tests |
| `tests/test_k2_jax_backend_cli.py` | CLI/backend flag tests |
| `tests/test_k2_jax_branch_activity_audit.py` | Branch activity audit (§7) |

#### Modified Files

| File | Change |
|------|--------|
| `scripts/simulate_hierarchical_controller.py` | Add `--controller-backend {python,jax}` flag; JAX fast-path in `simulation_step()`; remove duplicate estimator calls in both paths |
| `wheeled_biped/controllers/signal_filters.py` | Add `biquad_notch_coefficients()` and `biquad_notch_update()` pure functions alongside existing class |

### 3.4 JIT Controller Step Internal Composition

```
k2_jax_controller_step(state, input, params):
  1. unpack_input(input)       → named scalars (via static indices)
  2. unpack_state(state)       → named substates
  3. notch_update(x_pr, notch_state, notch_coeffs) → y_notch, notch_state'
  4. height_schedule(z_ref, z_actual, params)      → scheduled_gains
  5. sagittal_torque_legacy(inputs, gains, params) → tau_sag, diag_sag
  6. shape_posture_compute(q_ref, q, qd, params)   → tau_posture, diag_posture
  7. support_feedforward_compute(...)              → tau_support
  8. lateral_roll_compute(...)                     → tau_lateral
  9. yaw_compute(...)                              → tau_yaw
  10. torque_composer(sum_tau, prev_tau, params)   → tau_final, clip_flags
  11. mode_div_compute(hip_yaw_pos, ...)           → tau_hy_div
  12. assemble_final(tau_final, tau_hy_div)        → tau (10,)
  13. pack_diagnostics(all_diag_parts)             → diag_flat (D,)
  14. pack_state(all_state_parts)                  → state_flat' (S,)
  15. return tau, state_flat', diag_flat
```

Each sub-function is pure JAX, independently testable, and composed in sequence inside
the top-level `@jax.jit` function. No sub-function is independently JIT-compiled
(avoids compile fragmentation).

---

## 4. State Layout (PROVISIONAL)

**IMPORTANT:** The state layout below is **provisional**. Every state field must be backed
by an actual Python reference state variable or proven required by multi-step parity.
Do not create fake state only for symmetry. Reserved fields are allowed only for layout
stability and must remain zero/preserved through all steps.

During implementation, audit each proposed state field against the current Python K2 path
before adding it to `K2_JAX_STATE_FIELDS`. Fields that lack a corresponding Python state
variable must be removed or proven necessary via multi-step parity divergence testing.

Fields marked `[AUDIT]` below may not exist in the Python reference path and must be
verified or removed during Stage 4:
- `hy_div_integral` — may not exist if mode-div uses PD-only (no integral)
- `hy_div_prev_error` — may not exist if mode-div is stateless per-step
- `hy_div_prev_torque` — may not exist if mode-div has no smoothing
- `hy_div_soft_clamp_active` — may be recomputed from position each step
- `lateral_roll_prev_tau` — verify if lateral roll uses smoothing state
- `physics_ff_smoothed_tau` — verify if physics FF applies low-pass smoothing

### 4.1 K2_JAX_STATE_FIELDS (Version 1 — PROVISIONAL)

```
K2_JAX_STATE_FIELDS = (
    # === Notch filter (4) ===
    "notch_x1",           # float: x[n-1]
    "notch_x2",           # float: x[n-2]
    "notch_y1",           # float: y[n-1]
    "notch_y2",           # float: y[n-2]

    # === Height scheduling (1) ===
    "filtered_com_z",     # float: first-order filtered CoM z

    # === Support error (1) ===
    "prev_support_error", # float: previous support_position_error_m

    # === Previous torque for rate limiting (10) ===
    "prev_tau_0",         # l_hip_roll
    "prev_tau_1",         # l_hip_yaw
    "prev_tau_2",         # l_hip_pitch
    "prev_tau_3",         # l_knee
    "prev_tau_4",         # l_wheel
    "prev_tau_5",         # r_hip_roll
    "prev_tau_6",         # r_hip_yaw
    "prev_tau_7",         # r_hip_pitch
    "prev_tau_8",         # r_knee
    "prev_tau_9",         # r_wheel

    # === Calibrated outer loop state (2) ===
    "calib_outer_pitch_offset_deg",   # current pitch_ref_offset (low-passed)
    "calib_outer_prev_error_m",       # previous deadbanded support error

    # === Physics equilibrium FF state (1) [AUDIT] ===
    "physics_ff_smoothed_tau",        # [AUDIT] low-passed feedforward torque — verify smoothing exists in Python path

    # === Mode-hip-yaw-div state (4) [AUDIT] ===
    "hy_div_integral",                # [AUDIT] divergence integral — verify integral term exists in Python mode-div
    "hy_div_prev_error",              # [AUDIT] previous hip_yaw error — verify stateful error in Python path
    "hy_div_prev_torque",             # [AUDIT] previous applied divergence torque — verify smoothing state
    "hy_div_soft_clamp_active",       # [AUDIT] 0.0 or 1.0 flag — verify stateful or recomputed each step

    # === Lateral roll state (1) [AUDIT] ===
    "lateral_roll_prev_tau",          # [AUDIT] previous lateral roll torque — verify smoothing state exists

    # === Reserved for future layout versions (8) ===
    "reserved_0", "reserved_1", "reserved_2", "reserved_3",
    "reserved_4", "reserved_5", "reserved_6", "reserved_7",
)
# STATE_SIZE = 32
# STATE_LAYOUT_VERSION = 1
```

### 4.2 State Tests

1. Field names are unique
2. `len(K2_JAX_STATE_FIELDS)` matches state_flat shape
3. `pack → unpack` roundtrip preserves all active K2 state
4. Python K2 state vs JAX state evolve equivalently over multi-step parity tests
5. No JIT recompilation due to state shape changes
6. Reserved fields are zero-initialized and preserved through steps
7. **State audit (Stage 4):** Every non-reserved field in `K2_JAX_STATE_FIELDS` maps to a
   confirmed Python reference state variable. Fields without a Python counterpart are
   removed. The audit produces `docs/validation/k2_jax_state_field_audit.json` mapping
   each JAX state field to its Python source (file:line:attribute).

---

## 5. Input Layout

### 5.1 K2_JAX_INPUT_FIELDS

```
K2_JAX_INPUT_FIELDS = (
    # === Body state (12) ===
    "pitch_x_rad",
    "pitch_rate_x_rad_s",
    "roll_y_rad",
    "roll_rate_y_rad_s",
    "yaw_z_rad",
    "yaw_rate_z_rad_s",
    "com_x_m", "com_y_m", "com_z_m",
    "com_vx_m_s", "com_vy_m_s", "com_vz_m_s",

    # === Joint positions (10) ===
    "q_l_hip_roll", "q_l_hip_yaw", "q_l_hip_pitch", "q_l_knee", "q_l_wheel",
    "q_r_hip_roll", "q_r_hip_yaw", "q_r_hip_pitch", "q_r_knee", "q_r_wheel",

    # === Joint velocities (10) ===
    "qd_l_hip_roll", "qd_l_hip_yaw", "qd_l_hip_pitch", "qd_l_knee", "qd_l_wheel",
    "qd_r_hip_roll", "qd_r_hip_yaw", "qd_r_hip_pitch", "qd_r_knee", "qd_r_wheel",

    # === Reference/task (9) ===
    "commanded_height_ref_m",
    "q_ref_l_hip_roll", "q_ref_l_hip_yaw",
    "q_ref_l_hip_pitch", "q_ref_l_knee",
    "q_ref_r_hip_roll", "q_ref_r_hip_yaw",
    "q_ref_r_hip_pitch", "q_ref_r_knee",

    # === Contact (4) ===
    "left_wheel_contact",
    "right_wheel_contact",
    "left_contact_force_n",
    "right_contact_force_n",

    # === Derived/estimator (9) ===
    "support_center_y_m",
    "capture_point_x_m", "capture_point_y_m",
    "divergence_x_m", "divergence_y_m",
    "sagittal_position_error_m",
    "sagittal_velocity_m_s",
    "wheel_vel_left_rad_s", "wheel_vel_right_rad_s",

    # === Mode flags (2) ===
    "height_variant_id",     # int → low=0, mid=1, high=2, etc.
    "contact_state_id",      # int: DOUBLE=0, LEFT_ONLY=1, RIGHT_ONLY=2, FLIGHT=3
)
# INPUT_SIZE = 56 (44 body+joint+ref+contact + 9 derived + 2 flags + 1 reserved)
# Exact count finalized during input audit in Stage 1.
```

---

## 6. Diagnostics Layout

### 6.1 K2_JAX_DIAG_FIELDS

Flat array of all K2 telemetry scalars needed for validation. Field order is fixed and
tested.

```
K2_JAX_DIAG_FIELDS = (
    # === Notch diagnostics (6) ===
    "notch_input_pitch_rate",
    "notch_output_pitch_rate",
    "notch_height_gate_alpha",
    "notch_filtered_pitch_rate_effective",
    "notch_b0", "notch_Q",

    # === Sagittal torque decomposition (12) ===
    "tau_pitch",
    "tau_pitch_rate",
    "tau_position",
    "tau_sagittal_velocity",
    "tau_support_velocity",
    "tau_wheel_vel_left",
    "tau_wheel_vel_right",
    "tau_common_unclipped",
    "tau_common_clipped",
    "tau_physics_ff",
    "tau_low_band_support",
    "tau_outer_loop",

    # === Scheduled gains (8) ===
    "scheduled_k_position",
    "scheduled_k_wheel_velocity",
    "scheduled_kd_pitch",
    "scheduled_max_position_tau",
    "scheduled_k_velocity",
    "scheduled_k_support_velocity",
    "scheduled_kp_pitch",
    "scheduled_max_tau_wheel",

    # === Calibrated outer loop (6) ===
    "calib_kp_deg_per_m",
    "calib_kd_deg_per_mps",
    "calib_deadband_m",
    "calib_theta_max_deg",
    "calib_pitch_ref_offset_deg",
    "calib_support_error_deadbanded_m",

    # === Torque composer (15) ===
    "tau_sum_max_abs",
    "tau_clipped_max_abs",
    "tau_rate_limited_max_abs",
    "clip_saturation_count",
    "rate_limit_active_count",
    "composer_tau_final_0", "composer_tau_final_1",
    "composer_tau_final_2", "composer_tau_final_3",
    "composer_tau_final_4", "composer_tau_final_5",
    "composer_tau_final_6", "composer_tau_final_7",
    "composer_tau_final_8", "composer_tau_final_9",

    # === Mode-div (6) ===
    "hy_div_torque_left",
    "hy_div_torque_right",
    "hy_div_error_left_rad",
    "hy_div_error_right_rad",
    "hy_div_soft_clamp_active_left",
    "hy_div_soft_clamp_active_right",

    # === Final output (10) ===
    "tau_final_0", "tau_final_1", "tau_final_2", "tau_final_3", "tau_final_4",
    "tau_final_5", "tau_final_6", "tau_final_7", "tau_final_8", "tau_final_9",

    # === Flags (4) ===
    "any_nan_detected",
    "torque_clip_occurred",
    "rate_limit_occurred",
    "height_gate_active",

    # === Posture diagnostics (10) ===
    "posture_tau_max_abs",
    "posture_hip_yaw_error_left_rad",
    "posture_hip_yaw_error_right_rad",
    "posture_hip_pitch_error_left_rad",
    "posture_hip_pitch_error_right_rad",
    "posture_knee_error_left_rad",
    "posture_knee_error_right_rad",
    "posture_hip_roll_error_left_rad",
    "posture_hip_roll_error_right_rad",
    "posture_active_joint_count",

    # === Lateral roll (3) ===
    "lateral_roll_tau",
    "lateral_roll_error_rad",
    "lateral_roll_rate_rad_s",

    # === Yaw (4) ===
    "yaw_torque",
    "yaw_error_rad",
    "yaw_rate_rad_s",
    "yaw_differential_torque",

    # === Support feedforward (4) ===
    "support_ff_tau_left",
    "support_ff_tau_right",
    "support_ff_com_accel_x",
    "support_ff_com_accel_y",

    # === K2 identity (4) ===
    "k2_profile_id_int",
    "k2_notch_q",
    "k2_notch_center_hz",
    "k2_notch_blend",
)
# DIAG_SIZE ≈ 130 (exact count finalized during telemetry audit in Stage 4)
# All field names and order are frozen once K2_JAX_DIAG_FIELDS_V1 is committed.
)
```

### 6.2 Python Mapper

```python
def k2_jax_diag_flat_to_dict(
    diag_flat: np.ndarray,
    mode: str = "full",  # "full" | "realtime_minimal"
) -> dict[str, float]:
    """Map flat JAX diagnostics array to named telemetry dict.
    
    Args:
        diag_flat: (D,) float64 array from JAX controller step
        mode: "full" for validation/parity, "realtime_minimal" for visual
        
    Returns:
        Dict mapping K2_JAX_DIAG_FIELDS names to scalar values
    """
```

### 6.3 Telemetry Rules

- JIT function never creates Python dicts or uses string keys
- Full mapping for `--controller-backend jax` with telemetry enabled
- Decimated/minimal mapping for realtime visual (skip non-critical fields)
- Missing fields in mapping → test failure (unless whitelisted with justification)
- Whitelist: fields that are Python-only by nature (e.g., `wall_clock_time`, `step_index`)
  must be explicitly marked and tested

---

## 7. Branch Activity Audit (K2)

### 7.1 Requirement

Before claiming the legacy torque assembly port is complete, every strategy branch must be
**classified** under K2 across all validation scenarios using the 4-level system:

- **ENABLED_ACTIVE:** branch enabled and contributes nonzero output at least once
- **ENABLED_GATED_ZERO:** branch enabled but runtime output is zero across all scenarios due to gate/deadband/height/contact
- **DISABLED_INACTIVE:** branch disabled and must never execute
- **UNEXPECTED_ACTIVE:** branch executed unexpectedly — audit failure

The audit must prove not only that disabled branches are not entered, but also that
enabled branches have their expected ENABLED_ACTIVE or ENABLED_GATED_ZERO behavior
across Step C, Step D, Step E, single-push, and dynamic-height cases.

### 7.2 Audit Method

Add `tests/test_k2_jax_branch_activity_audit.py` that:

1. Instruments the Python K2 controller to log which code branches execute AND their
   output contribution (zero vs nonzero)
2. Runs all 5 scenario categories (Step C × 7 cases, Step D × 24 runs, Step E × 10 heights,
   single-push × 2, dynamic-height × 5)
3. Classifies each branch into one of the 4 categories
4. Asserts no UNEXPECTED_ACTIVE branches
5. Asserts DISABLED_INACTIVE branches are NEVER entered
6. Asserts ENABLED_ACTIVE branches produce nonzero output in at least one scenario
7. Asserts ENABLED_GATED_ZERO branches have zero output in all scenarios but nonzero in at least one ungated scenario (proves the gate works)
8. Produces a branch coverage report: `docs/validation/k2_branch_activity_audit.json`

### 7.3 Branch Coverage Categories

| Category | Branches | Expected Classification |
|----------|----------|------------------------|
| Notch filter | update, height-gate blend | **ENABLED_ACTIVE** (above gate) / **ENABLED_GATED_ZERO** (below gate) |
| Height scheduling | smoothstep gains | **ENABLED_ACTIVE** |
| Legacy torque assembly | tau_pitch, tau_pitch_rate, tau_position, tau_velocity, tau_support_vel, wheel damping | **ENABLED_ACTIVE** |
| Physics FF | feedforward lookup, clamp, smooth | **ENABLED_ACTIVE** (in calibration range) / **ENABLED_GATED_ZERO** (outside range) |
| Low-band support | gate, kp compute | **ENABLED_GATED_ZERO** (except near 0.320 m where ENABLED_ACTIVE) |
| Calibrated outer loop | PCHIP calls, deadband, rate limit | **ENABLED_ACTIVE** |
| Support-position outer loop | pitch ref offset, deadband | **ENABLED_ACTIVE** (deadband-gated) |
| Unified state feedback | mode classify, priority weight | **DISABLED_INACTIVE** |
| APC (all variants) | state machine transitions | **DISABLED_INACTIVE** |
| Hysteresis recenter | state machine | **DISABLED_INACTIVE** |
| Bias cancel | estimate update | **DISABLED_INACTIVE** |
| ZC / EZC | state machine transitions | **DISABLED_INACTIVE** |
| Adaptive bias trim | error history, trim compute | **DISABLED_INACTIVE** |
| Pitch bias comp | EMA accumulate | **DISABLED_INACTIVE** |
| T6H/I/J | blend/release logic | **DISABLED_INACTIVE** |
| Position integral | accumulate, clamp | **DISABLED_INACTIVE** |
| Capture gate | gate update | **DISABLED_INACTIVE** |
| L/LR/LP family | state-feedback compute | **DISABLED_INACTIVE** |

---

## 8. Interpolation Strategy

### 8.1 Problem

`calibrated_outer_loop_functions.py` and `physics_equilibrium_feedforward.py` use SciPy
PCHIP (monotone cubic Hermite) interpolation. PCHIP is not available in JAX.

### 8.2 Solution: Pre-evaluated Fine Grid (Empirical Acceptance)

For each PCHIP function:

1. At Python init time, evaluate the SciPy PCHIP on a fine height grid over 0.30–0.48 m
2. Pack the pre-evaluated grid values into `params_flat` as fixed-shape JAX arrays
3. Inside JIT, use piecewise-linear interpolation on the fine grid

**Grid resolution selection (empirical, not assumed):**

1. **Start with 1000 points** over 0.30–0.48 m (0.18 mm spacing)
2. Evaluate Python PCHIP vs JAX grid-linear at 10,000 random heights across the full range
3. Compute max absolute error for each calibrated function:
   - `calibrated_kp_deg_per_m(h)`
   - `calibrated_kd_deg_per_mps(h)`
   - `calibrated_ki_deg_per_m_s(h)`
   - `calibrated_theta_ref_max_deg(h)`
   - `calibrated_deadband_m(h)`
   - `calibrated_rate_limit_deg_per_step(h)`
   - `calibrated_lowpass_alpha(h)`
   - `physics_eq_ff_tau_each_wheel_nm(h)`
   - `physics_eq_pitch_no_off_deg(h)`
4. **Acceptance criterion:** max_abs_error ≤ 1e-6 for every function
5. **If any function fails:** increase grid to 2000 points and re-measure
6. **If 2000 points fails:** increase to 5000 points
7. **If 5000 points fails:** implement exact JAX-compatible Hermite evaluation from
   PCHIP coefficients (port the SciPy PCHIP algorithm to JAX)
8. **Record final grid resolution and max errors** in
   `docs/validation/k2_jax_pchip_grid_verification.json`

Acceptance is **empirical**, not theoretical. The error bound must be measured from the
actual calibrated functions, not assumed from smoothness properties.

### 8.3 Verification

- Compare Python PCHIP output vs JAX grid-interpolated output at 10,000 random heights
- Assert max absolute difference ≤ 1e-6 for all calibrated functions
- Record empirical max error per function and grid resolution used
- Test included in `test_k2_jax_component_parity.py`
- If Hermite port was required, test PCHIP coefficient parity separately

---

## 9. Staged Milestone Plan

### Stage 1: Profile + Remove Duplicate Estimator Calls
- Profile centroidal_estimator.estimate() and capture_estimator.update()
- Remove duplicate calls in both Python and JAX paths
- Benchmark per-step savings
- Verify no behavioral change
- **Gate:** duplicate calls removed, behavior unchanged, savings quantified

### Stage 2: JAX Notch + Torque Limiter Parity
- Implement `biquad_notch_coefficients()` and `biquad_notch_update()` in JAX
- Implement JAX torque composer (clip + rate limit + smooth)
- Component parity tests for notch and composer
- **Gate:** notch output diff < 1e-10, composer torque diff < 1e-10 per joint

### Stage 3: JAX Sagittal/Support/Posture Active Path Parity
- Port K2-active sagittal torque assembly (legacy path only)
- Port shape_posture compute, support feedforward, lateral roll, yaw
- Port height scheduling (smoothstep gains)
- Port calibrated outer loop and physics FF (pre-evaluated grids)
- Component parity tests for each
- **Gate:** each sub-component torque diff < 1e-8

### Stage 4: Full Legacy K2 Torque Assembly Parity
- Integrate all sub-components into `k2_jax_controller_step()`
- Full-step parity: Python vs JAX on captured inputs
- Multi-step parity: Python and JAX evolve in parallel from same initial state
- Branch activity audit: prove disabled strategies inactive
- **Gate:** full-step tau diff < 1e-5 per joint, multi-step state drift < 1e-6 per field per 1000 steps

### Stage 5: Integrate `--controller-backend jax`
- Add `--controller-backend {python,jax}` CLI flag
- JAX fast-path branch in `simulation_step()`
- Default: `python`
- Warmup compile at startup, print compile time
- **Gate:** Python backend unchanged, JAX backend produces valid rollouts

### Stage 6: C/D/E + Push + Dynamic Validation
- Step C: 7 cases, 0 falls on JAX
- Step D: push matrix, 24 paired runs, 0 falls on JAX
- Step E: 10 heights, 0 falls on JAX
- Single-push: high_0p480 forward/backward 90N on JAX
- Dynamic height: ramp_up, ramp_down, cycle, gate_dwell, gate_chatter on JAX
- Compare JAX vs Python classifications (must be EQUIVALENT or BETTER)
- **Gate:** all validation suites pass, no regressions

### Stage 7: Realtime Visual Benchmark
- Benchmark Python backend (headless + visual)
- Benchmark JAX cold compile
- Benchmark JAX warm steady-state (headless + visual)
- Metrics: controller ms/step, end-to-end ms/step, realtime factor
- **Gate:** JAX warm controller ≤ 3 ms/step (target), no recompilation, no memory leak
- If target not met: classify bottleneck, report, do not claim realtime

---

## 10. Backend Default Decision

| Phase | Default | JAX Available? |
|-------|---------|---------------|
| During development (Stages 1–5) | `python` | Via `--controller-backend jax` opt-in |
| After Stage 6 passes | `python` | Via `--controller-backend jax` opt-in |
| After Stage 7 passes + user approval | `python` (Option A conservative) | `--controller-backend jax` documented as validated realtime |

**Option A (conservative)** is the default: keep backend=python, document jax as validated
realtime opt-in. Full promotion to jax default requires explicit user request.

Python implementation is **never removed**.

---

## 11. Performance Targets and Classification

### 11.1 Controller-Kernel Targets

| Metric | Target |
|--------|--------|
| JAX warm controller compute (JIT kernel only) | ≤ 3 ms/step |
| JIT compile time | Measured once, printed separately |
| Repeated recompilation | 0 occurrences |
| Memory growth over 10k steps | < 1 MB |
| Python backend performance | Unchanged (no regression) |

### 11.2 Python-Boundary Budget (measured, not assumed)

For `--controller-backend jax`, the implementation must measure each boundary cost
separately. Do not claim realtime based only on JAX kernel timing.

| Boundary Step | Budget Target | Measurement Method |
|---------------|--------------|-------------------|
| MuJoCo state extraction (`mj_data` → numpy) | ≤ 1 ms | `time.perf_counter()` around extraction block |
| `input_flat` packing (numpy → jnp) | ≤ 0.5 ms | Timer around pack function |
| **JAX controller step** (JIT kernel) | ≤ 3 ms | `block_until_ready()` + timer |
| `tau` conversion (jnp → numpy for `mj_data.ctrl`) | ≤ 0.2 ms | `np.array()` + timer |
| Telemetry `diag_flat` → dict mapping | ≤ 0.5 ms (decimated) / ≤ 3 ms (full) | Timer around mapper |
| `viewer.sync()` (MuJoCo render) | measured, not budgeted | External to controller |
| Telemetry write (CSV/JSON) | measured, not budgeted | External to controller |
| **Total controller-side** (extract + pack + JAX + unpack) | ≤ 5 ms | Sum of measured components |
| **End-to-end visual step** | ≤ 10 ms (100 Hz target) | Sum of all components |

If any boundary cost exceeds budget by >2×, classify the bottleneck and document it.
Do not silently absorb boundary overhead into "JAX controller time."

### 11.3 Performance Outcome Classifications

The report must distinguish controller-kernel realtime from full visual-loop realtime.

#### Controller-Kernel Classifications

| Classification | Criteria |
|---------------|----------|
| `JAX_CONTROLLER_REALTIME_PASS` | JAX warm kernel ≤ 3 ms/step, no recompilation, no memory leak |
| `JAX_CONTROLLER_REALTIME_FAIL` | JAX warm kernel > 3 ms/step, or recompilation detected, or memory leak |

#### End-to-End Classifications

| Classification | Criteria |
|---------------|----------|
| `END_TO_END_REALTIME_PASS` | End-to-end visual step ≤ 10 ms/step, ≥100 Hz achieved |
| `END_TO_END_REALTIME_BLOCKED_BY_VIEWER_OR_MUJOCO` | Controller ≤ 5 ms but viewer/render > 5 ms |
| `END_TO_END_REALTIME_BLOCKED_BY_TELEMETRY_IO` | Controller ≤ 5 ms but telemetry write > 5 ms |
| `END_TO_END_REALTIME_BLOCKED_BY_PYTHON_BOUNDARY` | Boundary costs (extract + pack + unpack + map) > 5 ms |

**Rule:** Do not claim `END_TO_END_REALTIME_PASS` unless all components together meet
the 10 ms target. If only the JAX kernel meets its target, report `JAX_CONTROLLER_REALTIME_PASS`
plus the appropriate `END_TO_END_REALTIME_BLOCKED_BY_*` classification.

---

## 12. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| PCHIP→grid interpolation mismatch | Low | Medium | Start 1000-pt, empirical measurement at 10k random points; escalate to 2000/5000 pts or Hermite port if needed |
| IIR state drift over many steps | Medium | High | float64, multi-step parity test to 1000+ steps |
| JIT compile time > 30s | Medium | Low | Print separately, warmup before timing |
| Shape mismatch triggers recompile | Medium | High | Fixed static shapes, test for recompilation count |
| Python path accidentally changed | Low | High | Feature flag, Python path guarded by `if backend=="python"` |
| Duplicate estimator calls have semantic side effects | Medium | Medium | Profile first, check for state mutation before removing |
| Telemetry field count mismatch | Medium | Medium | Automated field count test, explicit whitelist |
| MuJoCo data extraction overhead dominates | Low | Medium | Measure extraction separately from JAX compute |

---

## 13. How to Run

### Python backend (current, default)
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json \
  --visual
```

### JAX backend (opt-in, after Stage 5)
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json \
  --controller-backend jax \
  --visual
```

### Recommended realtime visual command (after Stage 7)
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json \
  --controller-backend jax \
  --visual-realtime-factor 1.0 \
  --visual-sync-hz 30 \
  --visual
```

### Profile controller cost (headless)
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json \
  --controller-backend jax \
  --no-visual --steps 1000
```

### Parity comparison
```bash
python scripts/compare_k2_python_vs_jax_step.py \
  --profile k2_notch_low_q_v1 \
  --scenarios fixed_high_0p480 fixed_low_0p330 push_90N ramp_up gate_chatter \
  --output-dir outputs/k2_jax_parity
```

### Benchmark
```bash
python scripts/benchmark_k2_jax_controller.py \
  --profile k2_notch_low_q_v1 \
  --scenarios fixed_high_0p480 fixed_low_0p330 push_90N ramp_up \
  --no-visual --steps 5000
```

### Tests
```bash
pytest tests/test_k2_jax_component_parity.py -v
pytest tests/test_k2_jax_step_parity.py -v
pytest tests/test_k2_jax_backend_cli.py -v
pytest tests/test_k2_jax_branch_activity_audit.py -v
```

---

## 14. Files Changed (Summary)

### New (7 files)
1. `wheeled_biped/controllers/k2_jax_controller.py`
2. `scripts/compare_k2_python_vs_jax_step.py`
3. `scripts/benchmark_k2_jax_controller.py`
4. `tests/test_k2_jax_component_parity.py`
5. `tests/test_k2_jax_step_parity.py`
6. `tests/test_k2_jax_backend_cli.py`
7. `tests/test_k2_jax_branch_activity_audit.py`

### Modified (2 files)
1. `scripts/simulate_hierarchical_controller.py` — backend flag, JAX path, remove dup estimator calls
2. `wheeled_biped/controllers/signal_filters.py` — add pure JAX-compatible functions

### Documentation (2 files)
1. `docs/validation/k2_jax_jit_controller_parity_and_realtime_report.md` (Phase 9)
2. `docs/validation/k2_verified_visual_commands.md` — add JAX backend commands

---

*Design reviewed 2026-06-26. Proceed to staged implementation planning after applying corrections.*
