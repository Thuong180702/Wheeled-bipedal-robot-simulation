# K2 JAX Dedicated Realtime — Telemetry Schema

**Date:** 2026-06-29
**Phase:** 3 — Trace/CSV Support

---

## 1. Telemetry Modes

| Mode | CSV Written | Columns | Performance Impact | Use Case |
|------|------------|---------|-------------------|----------|
| `off` | No | 0 | 0% | Production realtime benchmark |
| `summary` | No (JSON only) | 0 | 0% | Final metrics only |
| `decimated` | Yes (every N steps) | 11 | ~0.3 ms/step avg | Lightweight monitoring |
| `full` | Yes (every step) | 60 | ~1.0 ms/step | Behavioral comparison, debugging |

All modes: data buffered in memory, CSV written once at end. No per-step file I/O. No per-step print in quiet mode.

---

## 2. Full Mode CSV Columns (60 columns)

### Metadata (2)
| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `step` | int | Simulation step index |
| 2 | `sim_time` | float | Simulation time in seconds |

### Orientation (6)
| # | Column | Type | Description |
|---|--------|------|-------------|
| 3 | `pitch_deg` | float | Robot pitch X in degrees |
| 4 | `roll_deg` | float | Robot roll Y in degrees |
| 5 | `yaw_deg` | float | Robot body yaw Z in degrees |
| 6 | `yaw_error_deg` | float | Yaw error (initial - current) in degrees |
| 7 | `pitch_rate_deg_s` | float | Pitch rate X in deg/s |
| 8 | `roll_rate_deg_s` | float | Roll rate Y in deg/s |
| 9 | `yaw_rate_deg_s` | float | Yaw rate Z in deg/s |

### CoM and Height (7)
| # | Column | Type | Description |
|---|--------|------|-------------|
| 10 | `com_x` | float | CoM X position (world) in meters |
| 11 | `com_y` | float | CoM Y position (world) in meters |
| 12 | `com_z` | float | CoM Z height in meters |
| 13 | `com_vx` | float | CoM X velocity in m/s |
| 14 | `com_vy` | float | CoM Y velocity in m/s |
| 15 | `height_ref` | float | Commanded height reference in meters |
| 16 | `height_error` | float | CoM Z - height_ref in meters |

### Support Center (2)
| # | Column | Type | Description |
|---|--------|------|-------------|
| 17 | `support_center_x` | float | Support center X (world) in meters |
| 18 | `support_center_y` | float | Support center Y (world) in meters |

### Joint Positions (10)
| # | Column | Type | Description |
|---|--------|------|-------------|
| 19 | `q_l_hip_roll` | float | Left hip roll position (rad) |
| 20 | `q_l_hip_yaw` | float | Left hip yaw position (rad) |
| 21 | `q_l_hip_pitch` | float | Left hip pitch position (rad) |
| 22 | `q_l_knee` | float | Left knee position (rad) |
| 23 | `q_l_wheel` | float | Left wheel position (rad) |
| 24 | `q_r_hip_roll` | float | Right hip roll position (rad) |
| 25 | `q_r_hip_yaw` | float | Right hip yaw position (rad) |
| 26 | `q_r_hip_pitch` | float | Right hip pitch position (rad) |
| 27 | `q_r_knee` | float | Right knee position (rad) |
| 28 | `q_r_wheel` | float | Right wheel position (rad) |

### Joint Velocities (10)
| # | Column | Type | Description |
|---|--------|------|-------------|
| 29-38 | `qd_l_hip_roll` ... `qd_r_wheel` | float | Joint velocities (rad/s) |

### Torques (10)
| # | Column | Type | Description |
|---|--------|------|-------------|
| 39-48 | `tau_l_hip_roll` ... `tau_r_wheel` | float | Per-joint torques (Nm) |

### Summary Torques (3)
| # | Column | Type | Description |
|---|--------|------|-------------|
| 49 | `max_abs_tau` | float | Maximum absolute torque across all joints |
| 50 | `max_wheel_tau` | float | Maximum absolute wheel torque |
| 51 | `max_leg_tau` | float | Maximum absolute leg joint torque |

### Hip Yaw Divergence (2)
| # | Column | Type | Description |
|---|--------|------|-------------|
| 52 | `hip_yaw_div_error` | float | (l_hip_yaw - r_hip_yaw) - eq_diff (rad) |
| 53 | `hip_yaw_div_rate` | float | l_hip_yaw_vel - r_hip_yaw_vel (rad/s) |

### Push Forces (2)
| # | Column | Type | Description |
|---|--------|------|-------------|
| 54 | `push_fx` | float | Push force X component (N) |
| 55 | `push_fy` | float | Push force Y component (N) |

### Contact (3)
| # | Column | Type | Description |
|---|--------|------|-------------|
| 56 | `contact_valid` | float | Overall contact validity (1.0 = valid) |
| 57 | `contact_left` | float | Left wheel contact (1.0 = in contact) |
| 58 | `contact_right` | float | Right wheel contact (1.0 = in contact) |

### Termination (2)
| # | Column | Type | Description |
|---|--------|------|-------------|
| 59 | `fall` | int | 1 if terminated due to fall, 0 otherwise |
| 60 | `terminated` | int | 1 if simulation terminated, 0 otherwise |

---

## 3. Decimated Mode CSV Columns (11 columns)

Same as before: `step, sim_time, com_z, pitch_deg, roll_deg, left_wheel_tau, right_wheel_tau, max_abs_tau, height_ref, contact_valid, fall`

---

## 4. Summary JSON Schema

```json
{
  "backend": "jax",
  "profile": "k2_notch_low_q_v1",
  "variant": "high_0p480",
  "steps": 2000,
  "max_steps": 2000,
  "sim_time_s": 20.0,
  "wall_time_s": 10.96,
  "achieved_hz": 182.4,
  "mean_step_ms": 5.48,
  "jax_compile_time_s": 1.65,
  "terminated": false,
  "termination_reason": "",
  "fall": false,
  "fall_step": -1,
  "com_z": {"initial": 0.488, "min": 0.481, "max": 0.492, "final": 0.488},
  "height_ref_m": 0.48,
  "height_floor_m": 0.43,
  "height_rms_error_m": 0.010,
  "pitch_x_deg": {"min": -0.0, "max": 8.5, "rms": 5.1},
  "roll_y_deg": {"min": -0.2, "max": 0.1, "rms": 0.1},
  "yaw_deg": {"min": ..., "max": ...},
  "yaw_error_deg": {"min": ..., "max": ...},
  "com_drift_m": {"x": 0.001, "y": -0.191, "final_displacement": 0.191, "max_displacement": 0.191},
  "support_center_range_m": {"x_min": ..., "x_max": ..., "y_min": ..., "y_max": ...},
  "max_torque_nm": {"total": 9.56, "wheels": 3.31, "hip_roll": ..., "hip_yaw": 0.36, "legs": 9.56},
  "hip_yaw_div": {"max_rad": 0.0155, "rms_rad": 0.0093},
  "contact_loss_steps": 1
}
```

---

## 5. Original K2 Python Telemetry Mapping

The monolithic script produces 1131-column CSV. Key fields for comparison:

| Original K2 Python Field | Dedicated JAX Field | Notes |
|--------------------------|---------------------|-------|
| `robot_pitch_x_deg` | `pitch_deg` | Direct mapping |
| `robot_roll_y_deg` | `roll_deg` | Direct mapping |
| `com_z_m` | `com_z` | Direct mapping |
| `com_x_m` / `com_y_m` | `com_x` / `com_y` | Direct mapping |
| `tau_*` (various naming) | `tau_l_hip_roll` etc. | Per-joint torques |
| `hip_yaw_divergence_error_rad` | `hip_yaw_div_error` | Similar computation |
| `contact_valid` | `contact_valid` | Direct mapping |
| `terminated` | `terminated` | Direct mapping |
| `height_cmd_m` | `height_ref` | Direct mapping |

The monolithic script has additional fields not in dedicated runner: WBC diagnostics, pitch rate sources, estimator internal state, etc. These are not needed for behavioral comparison.
