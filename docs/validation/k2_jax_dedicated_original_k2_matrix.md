# K2 JAX Dedicated Realtime — Original K2 Baseline Matrix & Behavioral Metrics

**Date:** 2026-06-29
**Phase:** 1-2 — Baseline Matrix Reconstruction + Behavioral Metric Specification

---

## Part A: Original K2 Promotion Scenario Matrix

### A.1 Step C — Fixed-Height Standing Balance (7 cases)

Original K2 Python baseline: `scripts/validate_k2_step_c_e_fixed_height.py --suite step_c`

| # | Case ID | Height | Steps | Notch Active? | Height Setup File |
|---|---------|--------|-------|---------------|-------------------|
| C1 | C1_slow_ladder_up_down | low_0p330 | 2000 | No | `low_0p330_setup.json` |
| C2 | C2_random_500dwell | low_0p330 | 2000 | No | `low_0p330_setup.json` |
| C3 | C3_random_200dwell | low_0p330 | 2000 | No | `low_0p330_setup.json` |
| C4 | C4_abrupt_stress | low_0p330 | 2000 | No | `low_0p330_setup.json` |
| C5 | C5_long_random | low_0p330 | 2000 | No | `low_0p330_setup.json` |
| C6 | focused_low_0p320 | low_0p320 | 2000 | No | `low_0p320_setup.json` |
| C7 | focused_high_0p480 | high_0p480 | 2000 | Yes | `high_0p480_setup.json` |

**Note:** Step C uses dynamic height trajectories for C1-C5. The dedicated runner uses `--dynamic-height-trajectory`.

### A.2 Step E — Fixed-Height Sweep (10 heights)

Original K2 Python baseline: `scripts/validate_k2_step_c_e_fixed_height.py --suite step_e`

| # | Height | Notch Gate | Height Setup File |
|---|--------|-----------|-------------------|
| E1 | low_0p300 | Inactive | `low_0p300_setup.json` |
| E2 | low_0p320 | Inactive | `low_0p320_setup.json` |
| E3 | low_0p330 | Inactive | `low_0p330_setup.json` |
| E4 | low_0p340 | Inactive | `low_0p340_setup.json` |
| E5 | low_0p360 | Inactive | `low_0p360_setup.json` |
| E6 | low_0p380 | Inactive | `low_0p380_setup.json` |
| E7 | high_0p430 | Partial | `high_0p430_setup.json` |
| E8 | high_0p450 | Partial | `high_0p450_setup.json` |
| E9 | high_0p465 | Partial | `high_0p465_setup.json` |
| E10 | high_0p480 | Active (100%) | `high_0p480_setup.json` |

All Step E cases: 2000 steps, fixed height.

### A.3 Step D — Push Recovery Matrix (12 conditions)

Original K2 Python baseline: `scripts/validate_k2_step_d_push_matrix.py`

| # | Condition | Height | Direction | Force | Height Setup | Push Seq |
|---|-----------|--------|-----------|-------|-------------|----------|
| D1 | high_0p480_fwd_60N | high_0p480 | forward | 60N | `high_0p480_setup.json` | ⚠️ MISSING |
| D2 | high_0p480_fwd_90N | high_0p480 | forward | 90N | `high_0p480_setup.json` | `push_fwd_90N.json` |
| D3 | high_0p480_bwd_60N | high_0p480 | backward | 60N | `high_0p480_setup.json` | ⚠️ MISSING |
| D4 | high_0p480_bwd_90N | high_0p480 | backward | 90N | `high_0p480_setup.json` | `push_bwd_90N.json` |
| D5 | mid_0p400_fwd_60N | mid_0p400 | forward | 60N | ⚠️ MISSING | ⚠️ MISSING |
| D6 | mid_0p400_fwd_90N | mid_0p400 | forward | 90N | ⚠️ MISSING | ⚠️ MISSING |
| D7 | mid_0p400_bwd_60N | mid_0p400 | backward | 60N | ⚠️ MISSING | ⚠️ MISSING |
| D8 | mid_0p400_bwd_90N | mid_0p400 | backward | 90N | ⚠️ MISSING | ⚠️ MISSING |
| D9 | low_0p330_fwd_60N | low_0p330 | forward | 60N | `low_0p330_setup.json` | ⚠️ MISSING |
| D10 | low_0p330_fwd_90N | low_0p330 | forward | 90N | `low_0p330_setup.json` | ⚠️ MISSING |
| D11 | low_0p330_bwd_60N | low_0p330 | backward | 60N | `low_0p330_setup.json` | ⚠️ MISSING |
| D12 | low_0p330_bwd_90N | low_0p330 | backward | 90N | `low_0p330_setup.json` | ⚠️ MISSING |

**Gap:** Only 2 of 12 Step D conditions have complete files. 60N push sequences and mid_0p400 setup are missing.

### A.4 Dynamic Height Gate-Crossing (5 scenarios)

Original K2 Python baseline: `scripts/validate_k2_dynamic_height_gate_crossing.py`

| # | Scenario | Steps | Trajectory File |
|---|----------|-------|-----------------|
| H1 | ramp_up_0p330_to_0p480 | 5000 | `k2_dynamic_height_gate_crossing/trajectories/ramp_up_0p330_to_0p480.json` |
| H2 | ramp_down_0p480_to_0p330 | 5000 | `k2_dynamic_height_gate_crossing/trajectories/ramp_down_0p480_to_0p330.json` |
| H3 | up_down_cycle_0p330_0p480_0p330 | 7000 | `k2_dynamic_height_gate_crossing/trajectories/up_down_cycle_0p330_0p480_0p330.json` |
| H4 | gate_dwell_0p420_0p450_0p480 | 6000 | `k2_dynamic_height_gate_crossing/trajectories/gate_dwell_0p420_0p450_0p480.json` |
| H5 | gate_chatter_0p400_0p470 | 5000 | `k2_dynamic_height_gate_crossing/trajectories/gate_chatter_0p400_0p470.json` |

### A.5 Long-Run Equilibrium (5 heights)

Original K2 Python baseline: `scripts/validate_k2_post_promotion_long_run.py`

| # | Height | Steps | Height Setup |
|---|--------|-------|-------------|
| L1 | low_0p330 | 6000 | `low_0p330_setup.json` |
| L2 | mid_0p400 | 6000 | ⚠️ MISSING |
| L3 | high_0p430 | 6000 | `high_0p430_setup.json` |
| L4 | high_0p450 | 6000 | `high_0p450_setup.json` |
| L5 | high_0p480 | 6000 | `high_0p480_setup.json` |

### A.6 Single Push (original K2 creation validation)

| # | Scenario | Height | Force | Steps | Height Setup | Push Seq |
|---|----------|--------|-------|-------|-------------|----------|
| P1 | 90N sagittal push | high_0p480 | 90N | 2000 | `high_0p480_setup.json` | `push_bwd_90N.json` |

### A.7 Available Scenario Summary

| Scenario Group | Total Cases | Files Complete | Can Run Now |
|---------------|-------------|---------------|-------------|
| Step C (fixed-height cases) | 7 | 7 height setups | 7 (all) |
| Step E (height sweep) | 10 | 10 height setups | 10 (all) |
| Step D (push matrix) | 12 | 2 push seqs, 2 heights | 2 (D2, D4) |
| Dynamic height | 5 | 5 trajectories | 5 (all) |
| Long-run equilibrium | 5 | 4 height setups | 4 of 5 |
| Single push | 1 | 1 push seq | 1 |
| **Total** | **40** | — | **29** |

---

## Part B: Behavioral Metrics Specification

### B.1 Stability and Termination

| # | Metric | Source | Threshold |
|---|--------|--------|-----------|
| B1.1 | fall_flag | termination check | Must match original K2 (0) |
| B1.2 | NaN_flag | any NaN in telemetry | Must be 0 |
| B1.3 | termination_reason | termination check string | Must match or be better |
| B1.4 | termination_step | last completed step | Must not be earlier than original K2 |
| B1.5 | min_com_z_m | com_z trace min | Not worse by >10% or 0.02 m |
| B1.6 | max_pitch_deg | pitch trace max | Not worse by >10% |
| B1.7 | max_roll_deg | roll trace max | Not worse by >10% |
| B1.8 | max_yaw_error_deg | yaw_error trace max | Not worse by >10% |

### B.2 Posture Quality

| # | Metric | Source | Threshold |
|---|--------|--------|-----------|
| B2.1 | mean_pitch_deg | pitch trace mean | Not worse by >10% |
| B2.2 | rms_pitch_deg | pitch trace RMS | Not worse by >10% |
| B2.3 | max_abs_pitch_deg | pitch trace max abs | Not worse by >10% |
| B2.4 | mean_roll_deg | roll trace mean | Not worse by >10% |
| B2.5 | rms_roll_deg | roll trace RMS | Not worse by >10% |
| B2.6 | max_abs_roll_deg | roll trace max abs | Not worse by >10% |
| B2.7 | final_pitch_deg | pitch at final step | Not worse by >10% |
| B2.8 | final_roll_deg | roll at final step | Not worse by >10% |
| B2.9 | final_yaw_error_deg | yaw_error at final step | Not worse by >10% |
| B2.10 | height_final_error_m | com_z - height_ref at final step | Not worse by >10% or 0.01 m |
| B2.11 | height_rms_error_m | height error RMS | Not worse by >10% |
| B2.12 | height_overshoot_m | max(com_z - height_ref) | Not worse by >10% |
| B2.13 | height_undershoot_m | max(height_ref - com_z) | Not worse by >10% |

### B.3 Drift and Position Holding

| # | Metric | Source | Threshold |
|---|--------|--------|-----------|
| B3.1 | support_center_drift_x_m | support_center_x - support_center_x_eq | Not worse by >10% or 0.02 m |
| B3.2 | support_center_drift_y_m | support_center_y - support_center_y_eq | Not worse by >10% or 0.02 m |
| B3.3 | com_drift_x_m | com_x - com_x_initial | Not worse by >10% or 0.02 m |
| B3.4 | com_drift_y_m | com_y - com_y_initial | Not worse by >10% or 0.02 m |
| B3.5 | final_displacement_m | sqrt(dx² + dy²) at final step | Not worse by >10% or 0.02 m |
| B3.6 | max_displacement_m | max sqrt(dx² + dy²) | Not worse by >10% or 0.02 m |
| B3.7 | drift_rate_m_per_s | displacement / sim_time | Not worse by >10% |

### B.4 Yaw and Leg/Foot Twist

| # | Metric | Source | Threshold |
|---|--------|--------|-----------|
| B4.1 | yaw_drift_deg | final_yaw - initial_yaw | Not worse by >10% |
| B4.2 | hip_yaw_div_max_rad | max abs(l_hip_yaw_pos - r_hip_yaw_pos - eq_diff) | Not worse by >10%; ≤0.35 rad absolute |
| B4.3 | hip_yaw_div_rms_rad | RMS of hip_yaw_div_error | Not worse by >10% |
| B4.4 | hip_yaw_div_rate_max_rad_s | max abs(l_hip_yaw_vel - r_hip_yaw_vel) | Not worse by >10% |
| B4.5 | left_right_hip_yaw_diff_rad | l_hip_yaw - r_hip_yaw at final step | Not worse by >10% |
| B4.6 | leg_symmetry_error | RMS(l_leg - r_leg) for hip_pitch, knee | Not worse by >10% |

### B.5 Push Recovery

| # | Metric | Source | Threshold |
|---|--------|--------|-----------|
| B5.1 | recovery_time_s | time from push end to pitch < threshold | Not worse by >10% |
| B5.2 | max_pitch_after_push_deg | max pitch after push start | Not worse by >10% |
| B5.3 | max_wheel_torque_after_push_Nm | max wheel torque after push | Not worse by >10% |
| B5.4 | max_displacement_after_push_m | max displacement after push | Not worse by >10% or 0.02 m |
| B5.5 | final_pitch_after_recovery_deg | pitch at final step | Not worse by >10% |
| B5.6 | final_drift_after_recovery_m | displacement at final step | Not worse by >10% or 0.02 m |
| B5.7 | contact_restored | contact_valid after recovery window | Must be 1.0 |
| B5.8 | final_upright | com_z > height_floor at final step | Must be True |

### B.6 Torque and Actuator Behavior

| # | Metric | Source | Threshold |
|---|--------|--------|-----------|
| B6.1 | max_torque_total_Nm | max abs(tau) across all joints | Not worse by >10% |
| B6.2 | max_wheel_torque_Nm | max abs(tau[4], tau[9]) | Not worse by >10% |
| B6.3 | max_hip_yaw_torque_Nm | max abs(tau[1], tau[6]) | Not worse by >10% |
| B6.4 | max_leg_torque_Nm | max abs(tau[0,2,3,5,7,8]) | Not worse by >10% |
| B6.5 | rms_torque_Nm | RMS of all tau | Not worse by >10% |
| B6.6 | torque_rate_max_Nm_s | max abs(delta_tau / dt) | Not worse by >10% |
| B6.7 | torque_saturation_count | count where abs(tau) >= torque_limit * 0.99 | Not higher |
| B6.8 | hidden_torque_flag | any non-K2 torque contribution | Must be 0 |
| B6.9 | wbc_torque_flag | any WBC torque contribution | Must be 0 |

### B.7 Contact and Safety

| # | Metric | Source | Threshold |
|---|--------|--------|-----------|
| B7.1 | contact_valid_fraction | fraction of steps with contact_valid=1 | Not worse by >5% |
| B7.2 | contact_loss_max_duration | max consecutive steps with contact_valid=0 | Not worse |
| B7.3 | wheel_lift_events | count of wheel contact loss | Not higher |
| B7.4 | safety_gate_active | any safety gate triggered | Must match or be better |

### B.8 Performance

| # | Metric | Source | Threshold |
|---|--------|--------|-----------|
| B8.1 | achieved_hz | step / wall_time | >50 Hz minimum, >100 Hz target |
| B8.2 | mean_step_ms | wall_time / step * 1000 | <20 ms target |
| B8.3 | jax_compile_count | number of JIT compilations | Must be 1 (warmup only) |
| B8.4 | telemetry_overhead_ms | time with telemetry - time without | Acceptable overhead |

### B.9 Original K2 Baseline Metrics (from old reports)

The original K2 reports used these specific metrics for classification:

**Step C/E metrics:**
- `pitch_rms_deg` — RMS pitch error in degrees
- `support_rms_m` — RMS support center error in meters  
- `hip_yaw_max` — Maximum hip yaw divergence in radians (gate: ≤0.35)
- `LF_power` — Low-frequency (0.15-0.55 Hz) pitch power
- `WIP_power` — WIP band (2.0-3.0 Hz) pitch power
- `fell` — Boolean fall flag

**Step D metrics (500-step post-push window):**
- Post-push pitch RMS (deg)
- Post-push support RMS (m)
- LF power (post-push)
- WIP power (post-push)
- Hip-yaw max

**Safety gates (absolute):**
- Falls = 0
- Hip-yaw ≤ 0.35 rad
- No hidden torque (>0.5 Nm)
- No WBC
- NaN/Inf = 0

**Classification thresholds:**
- STRONG_BETTER: K2 better on ≥2 of {pitch, support, LF} AND no regression
- BETTER: K2 better on ≥1 metric AND no regression
- EQUIVALENT: All metrics within noise
- WORSE_BUT_SAFE: K2 worse on 1 metric but still within safety gates
- REGRESSION: K2 falls where K1 does not, OR hip-yaw > 0.35, OR WIP K2 > 10× K1

---

## Part C: Telemetry Requirements for Behavioral Comparison

The current dedicated runner telemetry (11 columns) is INSUFFICIENT. Required additions:

### C.1 Current Telemetry (11 columns)
```
step, sim_time, com_z, pitch_deg, roll_deg, left_wheel_tau, right_wheel_tau,
max_abs_tau, height_ref, contact_valid, fall
```

### C.2 Required Additions for Full Behavioral Comparison
```
yaw_deg, yaw_error_deg, pitch_rate_deg_s, roll_rate_deg_s, yaw_rate_deg_s,
com_x, com_y, com_vx, com_vy,
support_center_x, support_center_y,
height_error,
joint_pos[0..9], joint_vel[0..9],
tau[0..9] (all 10 joints, not just wheels),
hip_yaw_div_error, hip_yaw_div_rate,
push_fx, push_fy,
contact_left, contact_right,
terminated, termination_reason,
```

Total columns for full mode: ~45 (vs current 11).

### C.3 Telemetry Mode Behavior

| Mode | Behavior | Columns |
|------|----------|---------|
| `off` | No telemetry, summary only | 0 |
| `summary` | Final stats only (no per-step) | N/A |
| `decimated` | Current 11 columns, every N steps | 11 |
| `full` | All 45 columns, every step, buffered, write-once | ~45 |

### C.4 Implementation Strategy

1. Add a `FULL_CSV_COLUMNS` list with all 45 fields
2. In `full` mode, buffer all fields every step
3. In `decimated` mode, keep current 11 columns for performance
4. All modes: buffer in memory, write CSV once at end
5. No per-step I/O, no per-step print
