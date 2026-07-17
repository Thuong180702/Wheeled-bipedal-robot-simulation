# K2 JAX Standalone — Phase 1 Dependency Map

**Date:** 2026-06-29

## Current JAX Input Contract (42 fields)

Each field is traced to its current source in `simulate_hierarchical_controller.py` lines 6698-6720 (`pack_input_k2` call), then classified for standalone migration.

| # | Index | Field | Current Source | Python Source Line | Control Effect | Replacement Plan | Classification |
|---|-------|-------|---------------|-------------------|----------------|-----------------|----------------|
| 0 | 0 | `pitch_x_rad` | `pitch_x_error` | L6303: `raw_pitch_x - pitch_x_ref` where `pitch_x_ref = pitch_x_eq + outer_loop_pitch_ref_total_deg` | Primary sagittal control: tau_pitch, tau_position via outer loop | Pass raw pitch_x + let JAX compute offset internally | **MUST_PORT_FORMULA_TO_JAX** |
| 1 | 1 | `pitch_rate_x_rad_s` | `pitch_rate_for_control_boosted` | L6318/6386: raw pitch rate × boost factor if transient detected | Sagittal damping via notch filter | Pass raw pitch_rate_x; JAX already has notch filter | **RAW_SENSOR_AVAILABLE** |
| 2 | 2 | `roll_y_rad` | `centroidal_state_control.body_roll_y` | Direct centroidal state read | Lateral roll control, safety gates | Keep as raw state | **RAW_SENSOR_AVAILABLE** |
| 3 | 3 | `roll_rate_y_rad_s` | `centroidal_state_control.body_roll_rate_y` | Direct centroidal state read | Lateral roll damping | Keep as raw state | **RAW_SENSOR_AVAILABLE** |
| 4 | 4 | `yaw_error_rad` | `initial_yaw_z - centroidal_state_control.body_yaw_z` | Equilibrium yaw minus current yaw | Yaw control | Compute from raw state in packing | **RAW_SENSOR_AVAILABLE** |
| 5 | 5 | `yaw_rate_rad_s` | `centroidal_state_control.body_yaw_rate_z` | Direct centroidal state read | Yaw damping | Keep as raw state | **RAW_SENSOR_AVAILABLE** |
| 6 | 6 | `com_z_m` | `centroidal_state_control.com_pos[2]` | Centroidal estimator output | Height schedule, safety gates, ABS trim gate | Keep as raw state | **SIM_STATE_AVAILABLE** |
| 7 | 7 | `com_vy_m_s` | `centroidal_state_control.com_vel[1]` | Centroidal estimator output | COM vertical velocity damping (tau_com_vy) | Keep as raw state | **SIM_STATE_AVAILABLE** |
| 8 | 8 | `sagittal_velocity_m_s` | `centroidal_state_control.com_vel[1]` | Same as com_vy (line 6196-6198: projected sagittal velocity) | Sagittal velocity damping | Currently same as com_vy; could use projected COM velocity | **SIM_STATE_AVAILABLE** |
| 9 | 9 | `sagittal_position_error_m` | `prev_support_error` | L6654: `sagittal_diag.get("support_position_error_m", 0.0)` from previous step's Python compute | Sagittal position hold, tau_position, ABS trim, APCR1ND, outer loop | **Compute from wheel xpos in JAX or pass raw support center XY** | **MUST_PORT_FORMULA_TO_JAX** |
| 10 | 10 | `wheel_vel_left_rad_s` | `joint_vel[4]` | Direct MuJoCo qvel | Wheel velocity damping | Keep as raw state | **RAW_SENSOR_AVAILABLE** |
| 11 | 11 | `wheel_vel_right_rad_s` | `joint_vel[9]` | Direct MuJoCo qvel | Wheel velocity damping | Keep as raw state | **RAW_SENSOR_AVAILABLE** |
| 12 | 12 | `support_velocity_m_s` | `sagittal_diag.get("support_position_velocity_m_s", 0.0)` | L4626 in svdbc.py: numerical derivative inside Python controller | Support velocity damping | **Compute in JAX as derivative of sag_pos_error (needs state.prev_support_error)** | **MUST_PORT_FORMULA_TO_JAX** |
| 13 | 13 | `commanded_height_ref_m` | `height_variant_setup["target_com_z_m"]` | Config, or dynamic height trajectory target | Height schedule index, all height-dependent gains | Keep as config/raw input | **HEIGHT_TRAJECTORY_AVAILABLE** |
| 14 | 14 | `hip_yaw_div_error` | `(joint_pos[1]-joint_pos[6]) - (eq_pos[1]-eq_pos[6])` | Raw joint positions minus equilibrium | Mode-div hip yaw control | Compute from raw state in packing | **RAW_SENSOR_AVAILABLE** |
| 15 | 15 | `hip_yaw_div_rate` | `joint_vel[1] - joint_vel[6]` | Raw joint velocities | Mode-div hip yaw damping | Compute from raw state in packing | **RAW_SENSOR_AVAILABLE** |
| 16-23 | 16-23 | `q_*` (8 joint positions) | `joint_pos` (reordered) | Direct MuJoCo qpos | Shape posture, lateral roll | Keep as raw state | **RAW_SENSOR_AVAILABLE** |
| 24-31 | 24-31 | `qd_*` (8 joint velocities) | `joint_vel` (reordered) | Direct MuJoCo qvel | Shape posture, lateral roll | Keep as raw state | **RAW_SENSOR_AVAILABLE** |
| 32-39 | 32-39 | `q_ref_*` (8 joint references) | `equilibrium_joint_pos` | Initial equilibrium keyframe | Shape posture reference | Keep as config | **PARAM_ONLY** |
| 40 | 40 | `support_position_error_m` | `prev_support_error` | Same as index 9, duplicate field | Support feedforward, outer loop state update | Same as index 9 | **MUST_PORT_FORMULA_TO_JAX** |
| 41 | 41 | `contact_valid` | `contact_output.*` | Contact detection from sim state | Safety gates (ABS, APCR1ND) | Keep as raw state | **RAW_SENSOR_AVAILABLE** |

## Fields Requiring Porting to JAX (5 of 42)

### 1. `pitch_x_rad` — MUST_PORT_FORMULA_TO_JAX

**Current Python source** (L6302-6303):
```python
pitch_x_ref = float(pitch_x_eq) + math.radians(outer_loop_pitch_ref_total_deg)
pitch_x_error = float(centroidal_state_control.body_pitch_x) - pitch_x_ref
```

**`outer_loop_pitch_ref_total_deg`** is (L6294-6298):
```python
outer_loop_pitch_ref_total_deg = (
    float(vd_pitch_ref_offset_deg)              # Height-scheduled or profile offset
    + float(support_outer_loop_pitch_ref_offset_deg)  # Low-band support shaping static offset
    + outer_loop_pitch_ref_dynamic_deg          # Dynamic outer loop PID output
)
```

**`pitch_x_eq`** is from equilibrium gravity vector (L3970):
```python
pitch_x_eq, roll_y_eq = compute_orientation_from_gravity(jnp.array(gravity_body_eq))
```

**Replacement plan:**
- JAX already computes `total_pitch_ref_offset_deg` = `new_ol_pitch_ref + lb_offset + physics_pitch_eq` (line 1808)
- JAX already computes outer loop, low-band, physics FF
- JAX currently receives pre-adjusted `pitch_x` and skips internal offset application
- **Change:** Pass raw `body_pitch_x` instead of `pitch_x_error`. Have JAX compute `effective_pitch_x = raw_pitch_x - deg2rad(total_pitch_ref_offset_deg)`
- Need to also include `pitch_x_eq` (equilibrium pitch) in the total offset for internal computation
- `pitch_x_eq` is a constant from initialization — can be added as a JAX param

### 2. `sagittal_position_error_m` / `support_position_error_m` — MUST_PORT_FORMULA_TO_JAX

**Current Python source** (L6165-6189):
```python
l_wheel_xpos = mj_data.xpos[l_wheel_body_id]  # World position from MuJoCo
r_wheel_xpos = mj_data.xpos[r_wheel_body_id]
support_center_xy = compute_support_center_xy(l_wheel_xpos, r_wheel_xpos)
sag_pos_error = project_sagittal_displacement(
    origin_xy=support_center_eq_xy,       # From initialization
    sagittal_axis_xy=sagittal_axis_xy_initial,  # From yaw equilibrium
    current_xy=support_center_xy,
)
# plus yaw compensation (L6181-6187)
```

**Replacement plan:**
- Option A: Pass wheel support center XY (2 floats) as raw input → JAX computes sag_pos_error from projection formula
- Option B: Compute sag_pos_error in Python from mj_data.xpos (fast, ~10 μs), pass as raw input
- **Choice: Option A** — compute in JAX for true standalone. Add 2 new input fields: `support_center_x_m`, `support_center_y_m`. Add equilibrium constants (`support_center_eq_x_m`, `support_center_eq_y_m`, `sagittal_axis_x`, `sagittal_axis_y`) as JAX params.
- The Python side only extracts wheel xpos from mj_data (sensor reading, not controller compute)

### 3. `support_velocity_m_s` — MUST_PORT_FORMULA_TO_JAX

**Current Python source** (svdbc.py L4626):
```python
support_position_velocity_m_s = (
    sagittal_position_error_m - self.prev_support_position_error_m
) / self.dt
```

**Replacement plan:**
- JAX state already has `prev_support_error` (index `_S_PREV_SUPPORT_ERROR`)
- Compute in JAX: `support_vel = (sag_pos_error - prev_support_error) / control_dt`
- Then update state: `new_prev_support_error = sag_pos_error`
- This is already partially implemented — the state update at line 1779: `new_ol_prev_support_error = support_pos_err`

### 4. `pitch_rate_x_rad_s` — RAW_SENSOR_AVAILABLE (with caveat)

**Current Python source** (L6314-6318):
```python
if args.vd_enable_pitch_rate_correction:
    pitch_rate_for_control = pitch_rate_estimate.pitch_rate_corrected
else:
    pitch_rate_for_control = float(centroidal_state_control.body_pitch_rate_x)
```

And boost factor (L6386):
```python
pitch_rate_for_control_boosted = pitch_rate_for_control * pitch_rate_boost_factor
```

**Replacement plan:**
- The JAX controller has its own notch filter for pitch rate processing
- The pitch rate boost is for transient capture — JAX currently receives boosted rate
- **Keep passing raw pitch_rate_x**; JAX handles notch filtering internally
- Pitch rate correction is disabled by default; if needed, port correction formula to JAX
- The boost factor would need to be ported or the transient detection logic moved to JAX

### 5. Implicit dependency: `prev_support_error` for JAX state initialization

The JAX state is initialized at startup (line 5424: `_jax_state = pack_state_k2()`), which zeros all state. The `prev_support_error` in state is used for:
- Support velocity computation (numerical derivative)
- Outer loop support error rate computation

For the first step, these should be 0 (cold start). After first step, JAX maintains them internally.

## Python sagittal_diag Fields Used for JAX Packing

| sagittal_diag Field | JAX Input Field | Control-Affecting? |
|---------------------|-----------------|---------------------|
| `support_position_velocity_m_s` | `support_velocity_m_s` | **YES** — Support velocity damping |
| `support_position_error_m` | `sagittal_position_error_m`, `support_position_error_m` | **YES** — Position hold, ABS, APCR1ND |
| (others via indirect path) | (none in JAX packing) | No — telemetry only |

## Python Sagittal Controller State Needed for Both-Synced (not production JAX)

These fields are captured for both-synced teacher-forcing parity checks ONLY (lines 6048-6079). They are NOT required for standalone production JAX:

| State Variable | Used In |
|----------------|---------|
| `notch_x1, notch_x2, notch_y1, notch_y2` | Both-synced: pre-snapshot notch state |
| `tau_prev` | Both-synced: previous torque for comparison |
| `filtered_com_z` | Both-synced: state sync |
| `ol_pitch_ref_smoothed`, `ol_prev_support_error`, `ol_support_error_rate` | Both-synced: outer loop state sync |
| `abs_trim_tau`, `abs_hold_steps`, `abs_prev_err_sign`, etc. | Both-synced: ABS state sync |
| `apcr1nd_step_counter`, `apcr1nd_prev_error`, etc. | Both-synced: APCR1ND state sync |

## New Raw Input Fields Needed for Standalone JAX

To compute `sag_pos_error` in JAX instead of Python, we need:

| New Field | Source | Type |
|-----------|--------|------|
| `support_center_x_m` | `compute_support_center_xy(l_wheel_xpos, r_wheel_xpos)[0]` | Raw state (from MuJoCo xpos, not controller) |
| `support_center_y_m` | `compute_support_center_xy(l_wheel_xpos, r_wheel_xpos)[1]` | Raw state (from MuJoCo xpos, not controller) |

New JAX params needed:
| Param | Source | Type |
|-------|--------|------|
| `support_center_eq_x_m` | Initialization constant | PARAM_ONLY |
| `support_center_eq_y_m` | Initialization constant | PARAM_ONLY |
| `sagittal_axis_x` | `sin(yaw_eq)` from initialization | PARAM_ONLY |
| `sagittal_axis_y` | `cos(yaw_eq)` from initialization | PARAM_ONLY |
| `pitch_x_eq_rad` | Equilibrium pitch from gravity | PARAM_ONLY |

## Summary Classification

| Classification | Count | Fields |
|----------------|-------|--------|
| RAW_SENSOR_AVAILABLE | 12 | pitch_rate, roll_y, roll_rate, yaw_err, yaw_rate, wheel_vel_l, wheel_vel_r, hy_div_err, hy_div_rate, q[8], qd[8], contact_valid |
| SIM_STATE_AVAILABLE | 2 | com_z_m, com_vy_m_s, sagittal_velocity (via com_vel) |
| HEIGHT_TRAJECTORY_AVAILABLE | 1 | commanded_height_ref_m |
| PARAM_ONLY | 1 | q_ref[8] |
| MUST_PORT_FORMULA_TO_JAX | 3 | pitch_x_rad, sagittal_position_error_m, support_velocity_m_s |
| NEW_RAW_INPUT_NEEDED | 2 | support_center_x_m, support_center_y_m |

**Total Python sagittal compute dependencies to eliminate:** 3 formulas + 2 new raw inputs

## Acceptance

| Criterion | Status |
|-----------|--------|
| No unknown JAX input source remains | ✅ All 42 fields traced to source |
| Every Python sagittal-derived control input has a replacement plan | ✅ 3 formulas + 2 new raw inputs identified |
| Diagnostic-only fields separated from control fields | ✅ Both-synced state captured separately from production input |
