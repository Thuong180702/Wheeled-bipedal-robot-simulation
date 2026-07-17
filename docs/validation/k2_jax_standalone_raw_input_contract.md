# K2 JAX Standalone — Phase 2 Raw Input Contract Design

**Date:** 2026-06-29

## Design Decision

We choose a hybrid approach:
- **Keep centroidal state estimation outside JAX** — it's already fast (~0.3 ms) and not the bottleneck
- **Port sagittal preprocessing formulas into JAX** — eliminate the ~55-75 ms Python sagittal compute bottleneck
- **Add minimal new raw inputs** — support center XY from MuJoCo wheel xpos (sensor read, not controller compute)

## Old Input Contract (Current — 42 fields)

```python
K2_JAX_INPUT_FIELDS = (
    "pitch_x_rad", "pitch_rate_x_rad_s",           # 0-1: PRE-ADJUSTED by Python
    "roll_y_rad", "roll_rate_y_rad_s",             # 2-3: raw state
    "yaw_error_rad", "yaw_rate_rad_s",             # 4-5: raw state
    "com_z_m", "com_vy_m_s",                       # 6-7: centroidal estimate
    "sagittal_velocity_m_s", "sagittal_position_error_m",  # 8-9: Python-computed
    "wheel_vel_left_rad_s", "wheel_vel_right_rad_s",       # 10-11: raw state
    "support_velocity_m_s",                        # 12: Python-computed (svdbc internal)
    "commanded_height_ref_m",                      # 13: config
    "hip_yaw_div_error", "hip_yaw_div_rate",       # 14-15: raw state
    "q_*" (8 joint positions),                     # 16-23: raw state
    "qd_*" (8 joint velocities),                   # 24-31: raw state
    "q_ref_*" (8 joint references),                # 32-39: config
    "support_position_error_m",                    # 40: Python-computed (dup of idx 9)
    "contact_valid",                               # 41: raw state
)
```

**Problems:**
1. `pitch_x_rad` (idx 0) is pre-adjusted by Python outer loop — JAX can't compute it independently
2. `sagittal_position_error_m` (idx 9) and `support_position_error_m` (idx 40) are Python-computed from wheel xpos
3. `support_velocity_m_s` (idx 12) is Python-computed numerical derivative

## New Input Contract (Standalone — 44 fields)

### Field Additions

| New Index | Field Name | Source | dtype |
|-----------|------------|--------|-------|
| 42 | `support_center_x_m` | `compute_support_center_xy(l_wheel_xpos, r_wheel_xpos)[0]` | float64 |
| 43 | `support_center_y_m` | `compute_support_center_xy(l_wheel_xpos, r_wheel_xpos)[1]` | float64 |

### Field Semantic Changes

| Index | Field Name | Old Source | New Source |
|-------|------------|------------|------------|
| 0 | `pitch_x_rad` | `pitch_x_error` (Python-adjusted) | `raw_pitch_x` (from centroidal_state.body_pitch_x) |
| 1 | `pitch_rate_x_rad_s` | `pitch_rate_for_control_boosted` | `raw_pitch_rate_x` (from centroidal_state.body_pitch_rate_x) |
| 9 | `sagittal_position_error_m` | `prev_support_error` (Python) | **Computed in JAX** from support_center + equilibrium params |
| 12 | `support_velocity_m_s` | `sagittal_diag` (Python) | **Computed in JAX** as derivative of sag_pos_error |
| 40 | `support_position_error_m` | `prev_support_error` (Python) | **Computed in JAX** (same as sagittal_position_error_m) |

### New JAX Params

| Param | Source | Purpose |
|-------|--------|---------|
| `pitch_x_eq_rad` | `compute_orientation_from_gravity(gravity_body_eq)[0]` | Equilibrium pitch (constant from init) |
| `support_center_eq_x_m` | `compute_support_center_xy(l_wheel_xpos_eq, r_wheel_xpos_eq)[0]` | Equilibrium support center |
| `support_center_eq_y_m` | `compute_support_center_xy(l_wheel_xpos_eq, r_wheel_xpos_eq)[1]` | Equilibrium support center |
| `sagittal_axis_x` | `sin(yaw_eq)` | Sagittal heading unit vector |
| `sagittal_axis_y` | `cos(yaw_eq)` | Sagittal heading unit vector |

### Full New Input Layout

```python
K2_JAX_INPUT_FIELDS_V2 = (
    # Orientation (from raw state, NOT pre-adjusted)
    "pitch_x_rad", "pitch_rate_x_rad_s",           # 0-1: RAW body pitch + rate
    "roll_y_rad", "roll_rate_y_rad_s",             # 2-3: raw state (unchanged)
    "yaw_error_rad", "yaw_rate_rad_s",             # 4-5: raw state (unchanged)
    # COM (from centroidal estimator)
    "com_z_m", "com_vy_m_s",                       # 6-7: centroidal estimate (unchanged)
    "com_vx_m_s",                                  # 8: NEW — for sagittal velocity projection
    # Wheel velocity
    "wheel_vel_left_rad_s", "wheel_vel_right_rad_s",  # 9-10: raw state (unchanged, shifted)
    # Height command
    "commanded_height_ref_m",                      # 11: config (unchanged, shifted)
    # Mode-div
    "hip_yaw_div_error", "hip_yaw_div_rate",       # 12-13: raw state (unchanged, shifted)
    # Joint state
    "q_*" (8), "qd_*" (8),                         # 14-29: raw state (unchanged, shifted)
    # Joint references
    "q_ref_*" (8),                                 # 30-37: config (unchanged, shifted)
    # Support center (NEW — raw sensor read from mj_data.xpos)
    "support_center_x_m", "support_center_y_m",    # 38-39: raw state (wheel xpos midpoint)
    # Contact
    "contact_valid",                               # 40: raw state (unchanged, shifted)
    # Total: 41 fields (compact; removed duplicates)
)
```

**Size: 41 fields** (down from 42 despite adding 2+1 new fields, by removing 3 duplicates/obsolete)

### Field Mapping: Old → New

| Old Index | Old Field | New Index | New Field | Change |
|-----------|-----------|-----------|-----------|--------|
| 0 | pitch_x_rad (adjusted) | 0 | pitch_x_rad (raw) | Semantic change |
| 1 | pitch_rate_x_rad_s (boosted) | 1 | pitch_rate_x_rad_s (raw) | Semantic change |
| 2 | roll_y_rad | 2 | roll_y_rad | Unchanged, shifted |
| 3 | roll_rate_y_rad_s | 3 | roll_rate_y_rad_s | Unchanged |
| 4 | yaw_error_rad | 4 | yaw_error_rad | Unchanged |
| 5 | yaw_rate_rad_s | 5 | yaw_rate_rad_s | Unchanged |
| 6 | com_z_m | 6 | com_z_m | Unchanged |
| 7 | com_vy_m_s | 7 | com_vy_m_s | Unchanged |
| — | — | 8 | com_vx_m_s | **NEW** — for sagittal vel projection |
| 8 | sagittal_velocity_m_s (dup) | — | — | Removed (compute from com_vx+com_vy+sag_axis) |
| 9 | sagittal_position_error_m (Python) | — | — | Removed (compute from support_center) |
| 10 | wheel_vel_left_rad_s | 9 | wheel_vel_left_rad_s | Shifted |
| 11 | wheel_vel_right_rad_s | 10 | wheel_vel_right_rad_s | Shifted |
| 12 | support_velocity_m_s (Python) | — | — | Removed (compute as derivative in JAX) |
| 13 | commanded_height_ref_m | 11 | commanded_height_ref_m | Shifted |
| 14 | hip_yaw_div_error | 12 | hip_yaw_div_error | Shifted |
| 15 | hip_yaw_div_rate | 13 | hip_yaw_div_rate | Shifted |
| 16-23 | q_* | 14-21 | q_* | Shifted |
| 24-31 | qd_* | 22-29 | qd_* | Shifted |
| 32-39 | q_ref_* | 30-37 | q_ref_* | Shifted |
| 40 | support_position_error_m (Python) | — | — | Removed (compute from support_center) |
| 41 | contact_valid | 40 | contact_valid | Shifted |
| — | — | 38-39 | support_center_x_m, support_center_y_m | **NEW** |

## JAX Internal Computation Flow (New)

```
INPUT (raw):
  raw_pitch_x, raw_pitch_rate_x
  roll_y, roll_rate_y, yaw_err, yaw_rate
  com_z, com_vx, com_vy
  wheel_vel_l, wheel_vel_r
  height_ref
  hy_div_err, hy_div_rate
  q[8], qd[8], q_ref[8]
  support_center_x, support_center_y  ← NEW raw inputs
  contact_valid

PARAMS (new additions):
  pitch_x_eq_rad                        ← equilibrium pitch constant
  support_center_eq_x, support_center_eq_y  ← equilibrium support center
  sagittal_axis_x, sagittal_axis_y      ← sagittal heading unit vectors

JAX COMPUTES:
  1. sag_pos_error = project_sagittal_displacement(
       origin=(support_center_eq_x, support_center_eq_y),
       axis=(sagittal_axis_x, sagittal_axis_y),
       current=(support_center_x, support_center_y))
  2. sag_vel = project_sagittal_velocity(
       axis=(sagittal_axis_x, sagittal_axis_y),
       velocity=(com_vx, com_vy))
  3. support_vel = (sag_pos_error - state.prev_support_error) / dt
  4. pitch_x_error = raw_pitch_x - pitch_x_eq - total_pitch_ref_offset
     (total_pitch_ref_offset from outer loop + physics FF + low-band)
  5. All existing JAX control logic (notch, schedule, ABS, APCR1ND, torque assembly)
```

## Python-Side Changes (pack_input_k2)

**Old** (depends on Python sagittal compute):
```python
_jax_input = pack_input_k2(
    pitch_x_rad=float(pitch_x_error),                    # Python
    pitch_rate_x_rad_s=float(pitch_rate_for_control_boosted),  # Python
    ...
    support_velocity_m_s=float(sagittal_diag.get(...)),  # Python
    sagittal_position_error_m=float(prev_support_error), # Python
    support_position_error_m=float(prev_support_error),  # Python
)
```

**New** (raw state only):
```python
_jax_input = pack_input_k2_v2(
    pitch_x_rad=float(centroidal_state_control.body_pitch_x),          # RAW
    pitch_rate_x_rad_s=float(centroidal_state_control.body_pitch_rate_x),  # RAW
    roll_y_rad=float(centroidal_state_control.body_roll_y),            # RAW
    roll_rate_y_rad_s=float(centroidal_state_control.body_roll_rate_y),# RAW
    yaw_error_rad=float(initial_yaw_z - centroidal_state_control.body_yaw_z),  # RAW
    yaw_rate_rad_s=float(centroidal_state_control.body_yaw_rate_z),    # RAW
    com_z_m=float(centroidal_state_control.com_pos[2]),               # Centroidal
    com_vx_m_s=float(centroidal_state_control.com_vel[0]),            # Centroidal
    com_vy_m_s=float(centroidal_state_control.com_vel[1]),            # Centroidal
    wheel_vel_left_rad_s=float(joint_vel[4]),                          # RAW
    wheel_vel_right_rad_s=float(joint_vel[9]),                         # RAW
    commanded_height_ref_m=float(height_target_m),                     # Config
    hip_yaw_div_error=float(...), hip_yaw_div_rate=float(...),        # RAW
    joint_pos=jnp.array(joint_pos), joint_vel=jnp.array(joint_vel),  # RAW
    q_ref=jnp.array(equilibrium_joint_pos),                           # Config
    support_center_x_m=float(support_center_xy[0]),                  # RAW (from xpos)
    support_center_y_m=float(support_center_xy[1]),                  # RAW (from xpos)
    contact_valid=float(contact_output.left_wheel_contact and ...),   # RAW
)
```

## Backward Compatibility: Both-Synced

Both-synced mode currently:
1. Captures Python controller state BEFORE compute
2. Packs JAX input using Python-computed values
3. Runs JAX from synced state with same inputs
4. Compares JAX torque vs Python torque

For Phase 3+ compatibility:
- Both-synced will use the NEW input contract (raw state) for JAX
- Python controller still runs (provides reference torque for comparison)
- The JAX controller computes everything from raw state internally
- Comparison is: Python torque vs JAX torque from same raw state
- This is actually a BETTER test because it tests that JAX replicates Python semantics from identical raw inputs

## Debug-Only Python Trace Fields

The following Python sagittal_diag fields are used ONLY for both-synced debug printing and telemetry — NOT for production JAX control:

- All `tau_*` per-component torques (for term-by-term diff)
- `support_position_velocity_m_s` (Python's internal derivative)
- `schedule_height_ref` (Python's filtered height)
- `effective_max_position_tau` (T6F/T6I raise value, used only in both-synced for state sync)
- `apcr1nd_*` fields (Python's gate state, used for state sync)
- `abs_*` fields (Python's ABS state, used for state sync)

These remain computed by Python in both-synced mode only. In standalone JAX mode, they are not computed at all.

## Acceptance

| Criterion | Status |
|-----------|--------|
| New contract removes Python sagittal compute dependency | ✅ Raw state only; 3 Python-derived fields eliminated |
| No Python control output required for backend=jax | ✅ All control-affecting values computed in JAX |
| both-synced can still compare Python vs JAX | ✅ Uses new raw-input contract; JAX independently computes from same raw state |
| Support center XY is sensor read, not controller compute | ✅ From mj_data.xpos (MuJoCo state), 2 float ops |
| No hidden Python dependency | ✅ All 5 old Python-derived fields have JAX-native replacements |
| Input size change documented | ✅ 42→41 fields (removed 3, added 2, net -1) |
