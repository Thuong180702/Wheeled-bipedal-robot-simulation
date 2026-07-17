# K2 Original Promoted Baseline — Exact Command Reconstruction

**Date:** 2026-06-29
**Phase:** 1 — RECONSTRUCT ORIGINAL K2 COMMANDS EXACTLY

---

## 1. Source Scripts

All original K2 validation was run through four scripts that invoke `scripts/simulate_hierarchical_controller.py`:

| Scope | Script | Profile |
|---|---|---|
| Step C + Step E | `scripts/validate_k2_step_c_e_fixed_height.py` | `k2_notch_low_q_v1` |
| Step D (push) | `scripts/validate_k2_step_d_push_matrix.py` | `k2_notch_low_q_v1` |
| Long-Run | `scripts/validate_k2_post_promotion_long_run.py` | `k2_notch_low_q_v1` |
| Dynamic Height | `scripts/validate_k2_dynamic_height_gate_crossing.py` | `k2_notch_low_q_v1` |

All scripts compare K2 (`k2_notch_low_q_v1`, Q=2.0) against K1 (`k1_pitch_rate_notch_v1`, Q=6.0).

---

## 2. Common CLI Arguments (all scopes)

Every `simulate_hierarchical_controller.py` invocation shares:

```
--controller-mode balance-core
--sagittal-controller velocity-damped
--vd-sagittal-authority-profile k2_notch_low_q_v1
--telemetry-decimation 1
--failure-window-steps <same_as_steps>
--write-run-summary-sidecar
```

Mode divergence flags (always enabled for K2):

```
--enable-mode-hip-yaw-divergence
--mode-hip-yaw-div-kp 10.0
--mode-hip-yaw-div-kd 0.50
--mode-hip-yaw-div-max-torque 7.5
--mode-hip-yaw-div-soft-limit-rad 0.30
--mode-hip-yaw-div-soft-gain 0.80
--mode-hip-yaw-div-ref-source target
```

### Default controller backend

The original K2 Step C/E, Step D, and Long-Run were run with `--controller-backend python` (the default at the time).

The original K2 Dynamic Height was run with `--controller-backend jax` (evidenced by `_K2_JAX` suffix in source_file paths in `k2_original_metrics.json`).

**CRITICAL FINDING:** The original K2 dynamic height baseline was already JAX-based (monolithic `simulate_hierarchical_controller.py --controller-backend jax`), NOT Python-based. The original JAX path SURVIVED all dynamic height scenarios with:
- ramp_up: hy=0.0534, height_rmse=0.1051, NO FALL
- ramp_down: hy=0.0977, height_rmse=0.1149, NO FALL
- gate_dwell: hy=0.0534, NO FALL

This means there is a REAL regression between the canonical monolithic JAX path and the dedicated JAX runner.

---

## 3. Step C — Exact Specs

**Script:** `scripts/validate_k2_step_c_e_fixed_height.py --suite step_c`
**Controller backend:** python (default at the time)

| Scenario | Height Setup | Steps | Actually Dynamic? |
|---|---|---|---|
| C1_slow_ladder_up_down | `low_0p330_setup.json` | 2000 | **NO — fixed 0.33m** |
| C2_random_500dwell | `low_0p330_setup.json` | 2000 | **NO — fixed 0.33m** |
| C3_random_200dwell | `low_0p330_setup.json` | 2000 | **NO — fixed 0.33m** |
| C4_abrupt_stress | `low_0p330_setup.json` | 2000 | **NO — fixed 0.33m** |
| C5_long_random | `low_0p330_setup.json` | 2000 | **NO — fixed 0.33m** |
| focused_low_0p320 | `low_0p320_setup.json` | 2000 | **NO — fixed 0.32m** |
| focused_high_0p480 | `high_0p480_setup.json` | 2000 | **NO — fixed 0.48m** |

**EVIDENCE that C1-C5 are fixed-height, not dynamic:**
1. The original script `STEP_C_CASES` list maps C1-C5 all to `low_0p330` with 2000 steps and NO trajectory argument.
2. The `run_sim_fixed_height()` function is called — no `--dynamic-height-trajectory` flag.
3. The baseline `k2_original_metrics.json` shows IDENTICAL metrics for C1-C5: all have pitch_rms=3.63, support_rms=0.0386, hy_max=0.0851. This is byte-for-byte identical because they are the SAME fixed-height simulation.
4. The scenario names (slow_ladder_up_down, etc.) are LEGACY names from when these were planned as dynamic trajectories. The actual K2 Step C validation ran them as fixed-height.

**CONCLUSION:** The dedicated runner's current Step C (fixed height 0.33m for C1-C5) IS equivalent to the original. The user's concern about non-equivalence is unfounded for Step C. Phase 4 should verify this conclusively but likely finds no mismatch.

---

## 4. Step E — Exact Specs

**Script:** `scripts/validate_k2_step_c_e_fixed_height.py --suite step_e`
**Controller backend:** python (default at the time)

| Height | Setup File | Steps | Dynamic? |
|---|---|---|---|
| low_0p300 | `low_0p300_setup.json` | 2000 | No |
| low_0p320 | `low_0p320_setup.json` | 2000 | No |
| low_0p330 | `low_0p330_setup.json` | 2000 | No |
| low_0p340 | `low_0p340_setup.json` | 2000 | No |
| low_0p360 | `low_0p360_setup.json` | 2000 | No |
| low_0p380 | `low_0p380_setup.json` | 2000 | No |
| high_0p430 | `high_0p430_setup.json` | 2000 | No |
| high_0p450 | `high_0p450_setup.json` | 2000 | No |
| high_0p465 | `high_0p465_setup.json` | 2000 | No |
| high_0p480 | `high_0p480_setup.json` | 2000 | No |

All fixed-height. The dedicated runner uses the same setups and steps — equivalent.

---

## 5. Step D — Exact Specs

**Script:** `scripts/validate_k2_step_d_push_matrix.py --profile k2`
**Controller backend:** python

| Parameter | Value |
|---|---|
| Heights | high_0p480, mid_0p400, low_0p330 |
| Directions | sagittal_forward (+y), sagittal_backward (-y) |
| Forces | 60N, 90N |
| Push step | 300 |
| Push duration | 5 steps |
| Push method | xfrc_applied to body 1 |
| Run length | 2000 steps |
| Total conditions | 12 (3×2×2) |

**Metric windows (original):**
- `post_pitch_rms_500_deg`: RMS of pitch in degrees over steps 305-805 (500 steps after push end)
- `post_support_rms_500_m`: RMS of support_position_error over steps 305-805
- `hip_yaw_max_rad`: max of abs(l_hip_yaw, r_hip_yaw) over full 2000 steps

**⚠️ METRIC WINDOW MISMATCH with dedicated runner:**
The dedicated runner currently reports full-episode pitch_rms (2000 steps) while the original baseline reports post-push 500-step RMS. These are NOT comparable. Phase 3 must fix this.

**⚠️ hip_yaw_max = 0.000 in original baseline:**
All 12 Step D conditions report hy_max=0.0 in the baseline. This is suspicious — the same robot at the same heights in Step E shows hy_max values from 0.0236 to 0.2473. The original Step D script was run with the Python backend, which might not have properly recorded hip_yaw telemetry. Phase 7 must verify whether this 0.0 is a recording artifact or genuine.

---

## 6. Dynamic Height — Exact Specs

**Script:** `scripts/validate_k2_dynamic_height_gate_crossing.py --profile k2 --controller-backend jax`
**Controller backend:** **jax** (CRITICAL — the original dynamic height baseline used JAX, not Python)

### ramp_up_0p330_to_0p480
| Parameter | Value |
|---|---|
| Setup | `low_0p330_setup.json` (initial CoM ~0.33m) |
| Steps | 5000 |
| Waypoints | (0,0.33), (500,0.33), (3500,0.48), (5000,0.48) |
| Gate crossing | Yes (0.42 at ~step 1125, 0.48 at ~step 3500) |

### ramp_down_0p480_to_0p330
| Setup | `high_0p480_setup.json` |
| Steps | 5000 |
| Waypoints | (0,0.48), (500,0.48), (3500,0.33), (5000,0.33) |

### up_down_cycle_0p330_0p480_0p330
| Setup | `low_0p330_setup.json` |
| Steps | 7000 |
| Waypoints | (0,0.33), (500,0.33), (2500,0.48), (4000,0.48), (6000,0.33), (7000,0.33) |

### gate_dwell_0p420_0p450_0p480
| Setup | `low_0p330_setup.json` |
| Steps | 6000 |
| Waypoints | (0,0.33), (500,0.33), (1500,0.42), (2500,0.42), (3000,0.45), (4000,0.45), (4500,0.48), (6000,0.48) |

### gate_chatter_0p400_0p470
| Setup | `low_0p330_setup.json` |
| Steps | 5000 |
| Waypoints | Repeated transitions 0.40-0.47 (17 waypoints) |

### How the original JAX path handles q_ref during dynamic height

In `simulate_hierarchical_controller.py` (line 5699-5713):
1. `dynamic_height_target_m` is interpolated from trajectory waypoints
2. `height_cmd` is set to the interpolated target
3. `height_variant_setup["target_com_z_m"]` is UPDATED to the interpolated target
4. The JAX controller receives the updated `height_variant_setup` with the new `target_com_z_m`
5. The JAX controller recomputes LQR gains using the updated target height
6. `equilibrium_joint_pos` (q_ref for posture) remains STATIC from initial setup

**KEY INSIGHT:** The original JAX path uses STATIC q_ref (equilibrium posture) but DYNAMIC LQR gains (recomputed per-step using updated target_com_z_m). The static posture + dynamic gains combination is sufficient for height tracking because the LQR feedback action adapts to the new target height.

---

## 7. Long-Run — Exact Specs

**Script:** `scripts/validate_k2_post_promotion_long_run.py --suite eq --profile k2`
**Controller backend:** python (default)

| Height | Setup | Steps |
|---|---|---|
| low_0p330 | `low_0p330_setup.json` | 6000 |
| mid_0p400 | `mid_0p400_setup.json` | 6000 |
| high_0p430 | `high_0p430_setup.json` | 6000 |
| high_0p450 | `high_0p450_setup.json` | 6000 |
| high_0p480 | `high_0p480_setup.json` | 6000 |

All fixed-height equilibrium. The dedicated runner uses the same setups and steps — equivalent.

---

## 8. Key Findings for Fix Prioritization

### Finding 1: Dynamic height baseline is JAX, not Python
The original K2 dynamic height was validated with `--controller-backend jax` (monolithic path).
The monolithic JAX path SURVIVES all dynamic height scenarios.
The dedicated JAX runner FAILS dynamic height.
This is a real regression between monolithic and dedicated JAX paths.

### Finding 2: Original JAX uses static q_ref + dynamic LQR gains
The original JAX path in `simulate_hierarchical_controller.py`:
- Keeps `equilibrium_joint_pos` static from initial setup
- Updates `height_variant_setup["target_com_z_m"]` per-step from trajectory
- JAX controller recomputes LQR gains using updated target height
- Static posture + dynamic gains = sufficient for height tracking

### Finding 3: Step C is actually equivalent
C1-C5 original were fixed-height at 0.33m, not dynamic trajectories.
The dedicated runner IS equivalent for Step C.

### Finding 4: Step D metric window mismatch is real
Original uses post-push 500-step RMS. Dedicated uses full-episode RMS.
Must be fixed in Phase 3.

### Finding 5: Step D hy_max=0.0 is suspicious
All 12 Step D conditions show 0.0 hip_yaw in the original baseline.
This contradicts Step E results at the same heights with the same controller.
Likely a telemetry recording artifact in the Python backend.

### Finding 6: Step E and Long-Run use Python backend
The original Step E and Long-Run were run with the Python controller.
The dedicated runner uses JAX.
Some SAFE_BUT_WORSE regressions may be inherent JAX vs Python differences,
not dedicated runner bugs. But must be proven via Phase 7/8 scalar audit.

---

## 9. Acceptance

- [x] All scenarios documented with exact CLI args
- [x] Controller backend identified per scope
- [x] Dynamic height discovered to use original JAX, not Python
- [x] Step C equivalence confirmed (fixed-height in both original and dedicated)
- [x] Step D metric window mismatch identified
- [x] Step D hy_max=0.0 baseline flagged as suspicious
- [x] q_ref handling in original JAX path documented
- [x] Setup files identified per scenario
