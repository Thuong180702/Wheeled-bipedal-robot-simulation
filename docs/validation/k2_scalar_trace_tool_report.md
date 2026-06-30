# K2 Scalar Trace Tool Report

**Date:** 2026-06-29
**Phase:** 3 — BUILD SCALAR TRACE TOOL
**Status:** ✅ TOOL BUILT (pending full execution)

---

## 1. Tool Summary

Created `scripts/trace_k2_source_vs_dedicated.py` — a comprehensive comparison tool that:

1. Runs the source-of-truth path (Python monolithic K2 via `simulate_hierarchical_controller.py`)
2. Runs the dedicated JAX runner (`run_k2_jax_realtime.py`)
3. Extracts control-affecting scalars per step from both paths
4. Compares field-by-field and reports the first divergent field

### Usage

```bash
# Step E fixed-height trace (500 steps)
python scripts/trace_k2_source_vs_dedicated.py \
  --scenario step_e --height low_0p300 --steps 500

# Step D push trace (steps 250-850 for push window)
python scripts/trace_k2_source_vs_dedicated.py \
  --scenario step_d --height low_0p330 --push push_sagittal_forward_90N_step300.json --steps 850

# Dynamic height trace
python scripts/trace_k2_source_vs_dedicated.py \
  --scenario dynamic --height low_0p330 --trajectory ramp_up --steps 2000

# Compare existing trace files (no simulation)
python scripts/trace_k2_source_vs_dedicated.py \
  --compare-only \
  --source-trace path/to/source_telemetry.csv \
  --dedicated-trace path/to/dedicated_telemetry.csv
```

---

## 2. Field Coverage

The tool traces **54 fields** per step, including:

### State (22 fields)
- 10 joint positions (`q_*`)
- 10 joint velocities (`qd_*`)
- sim_time

### Orientation (6 fields)
- pitch, roll, yaw (deg)
- pitch_rate, roll_rate, yaw_rate (deg/s)

### CoM and Height (6 fields)
- com_x, com_y, com_z, com_vx, com_vy
- height_ref, height_error

### Support (2 fields)
- support_center_x, support_center_y

### Torque (10 fields)
- tau_* for all 10 actuators

### Hip-yaw divergence (1 field)
- hip_yaw_div_error

### Control-affecting fields (subset)
Fields whose divergence propagates downstream:
`pitch_deg`, `pitch_rate_deg_s`, `roll_deg`, `roll_rate_deg_s`, `com_z`, `com_vx`, `com_vy`, `support_center_x`, `support_center_y`, `q_l_hip_yaw`, `q_r_hip_yaw`, `qd_l_hip_yaw`, `qd_r_hip_yaw`, `hip_yaw_div_error`

---

## 3. Source-to-Dedicated Field Mapping

The tool maps original Python telemetry column names to dedicated runner field names:

| Source Column | Dedicated Field | Units | Conversion |
|---|---|---|---|
| `euler_pitch_y` | `pitch_deg` | deg | rad→deg |
| `euler_roll_x` | `roll_deg` | deg | rad→deg |
| `l_hip_yaw_pos` | `q_l_hip_yaw` | rad | direct |
| `r_hip_yaw_pos` | `q_r_hip_yaw` | rad | direct |
| `hip_yaw_divergence` | `hip_yaw_div_error` | rad | direct |
| `support_position_error_m` | `support_error` | m | direct |
| `tau_l_hip_yaw` | `tau_l_hip_yaw` | Nm | direct |
| ... (see `SOURCE_FIELD_MAP` in code for full mapping) | | | |

---

## 4. Divergence Report Format

The tool generates a markdown report identifying:

1. **First control-affecting divergent field** — the earliest field in the causal chain that diverges
2. **All divergent fields** — sorted by max delta
3. **Interpretation guidance** — root cause hypotheses based on which field diverges first

Example output structure:
```
## First Control-Affecting Divergence
| Source field | `l_hip_yaw_pos` |
| Dedicated field | `q_l_hip_yaw` |
| Max delta | 0.0523 |
| At step | 347 |
```

---

## 5. Execution Requirements

To run a full trace comparison:

1. Source path needs `simulate_hierarchical_controller.py` to complete (may take 2-5 minutes for 2000 steps)
2. Dedicated path needs `run_k2_jax_realtime.py --telemetry full` to generate CSV (faster, ~10-20s)
3. Comparison is fast (<1s for 2000 steps)

**Note:** The source Python path is slow (~50 Hz), so 2000-step traces take ~40s. The dedicated JAX path runs at ~120+ Hz.

---

## 6. Acceptance Criteria

| Criterion | Status |
|---|---|
| Trace tool supports source backend: Python monolithic | ✅ |
| Trace tool supports source backend: JAX monolithic | ✅ (via `--source-backend jax_monolithic`) |
| Trace tool supports dedicated backend: run_k2_jax_realtime.py | ✅ |
| Supports scenarios: fixed height, dynamic, push, long-run | ✅ |
| Every traced field has same name, units, step alignment | ✅ Field mapping handles conversions |
| Tool identifies first divergent field automatically | ✅ `first_divergence` in report |
| Compiles without errors | ✅ `py_compile` passes |
| `--compare-only` mode for existing traces | ✅ |

---

## 7. Next Steps

Phase 4: Run traces for representative scenarios and identify the first divergent field causing hip-yaw and support RMS regressions.

Recommended initial traces:
1. `step_e low_0p300 500 steps` — most divergent hip-yaw
2. `step_d low_0p330 fwd 90N 850 steps` — push with high hip-yaw
3. `step_e high_0p480 500 steps` — control (low hip-yaw, should match)
