# K1 Controller Completion — Sustained Recovery and D4/D5 Yaw Fix

**Date:** 2026-06-24
**Task:** `k1_controller_completion_sustained_recovery_and_d4d5_yaw_fix`
**Branch:** `repo-cleanup-t6j`
**Report path:** `docs/validation/k1_controller_completion_sustained_recovery_and_d4d5_fix_report.md`

---

## 1. Executive Summary

This task created the **L**, **M**, and **N** candidate families built on the K1 current-best baseline, targeting the two remaining known blockers:

1. **Sustained posture recovery** — K1's 2.4–2.5 Hz WIP mode is reduced by the notch filter but not eliminated; K1 never achieves a sustained ≥2 s posture hold after a single push (L family target).
2. **D4/D5 hip_yaw > 0.35 rad gate** — Body-yaw drift couples into hip-yaw joint angles; hip-yaw torque alone cannot correct body yaw (M family target).

### Key Findings

1. **L family implemented** — Three coordinated sagittal state-feedback candidates (L1 low-frequency feedback, L2 phase-lead compensation, L3 pitch-reference stabilization) created as opt-in profiles built on K1. Each adds a coordinated state-feedback term to the common wheel torque, replacing the independent term summation with a synchronized command.

2. **M family implemented** — Two body-yaw/wheel-yaw correct-actuator candidates (M1 low-band differential wheel yaw, M2 support-aware wheel yaw) created as opt-in profiles built on K1. These use the existing `DifferentialWheelYawStabilizer` infrastructure.

3. **N family implemented** — One mild phase-lead damping diagnostic (N1) using compensated damping without the full L2 architecture.

4. **True dynamic-height Step C harness created** — `scripts/run_true_dynamic_height_step_c_validation.py` generates 7 height trajectories that all cross the notch gate (0.42–0.48 m) in various patterns. This fixes the known validation gap where existing Step C cases only test fixed heights.

5. **Sustained recovery audit script created** — `scripts/audit_k1_sustained_recovery_failure.py` decomposes sagittal torque, performs frequency/phase analysis, and searches for recovery events.

6. **D4/D5 body-yaw audit script created** — `scripts/audit_k1_d4_d5_body_yaw_to_hip_yaw_coupling.py` analyzes body-yaw-to-hip-yaw coupling to confirm wheel-yaw correction is justified.

### Decision

**K1 remains current-best.** No L, M, or N candidate is promoted at this time. The candidates are opt-in infrastructure ready for evaluation. Promotion requires beating K1 across Step E + true dynamic Step C + full Step D, which has not yet been run for the new candidates.

---

## 2. K1 Current-Best Baseline Verification

| Item | Value |
|------|-------|
| Current-best | `K1_PITCH_RATE_NOTCH_V1` |
| Profile | `k1_pitch_rate_notch_v1` |
| Status | `CURRENT_BEST_PROMOTED_WITH_EXPANDED_KNOWN_LIMITATIONS` |
| Sagittal base | `PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2` |
| Notch enabled | True (pitch_rate, fc=2.5 Hz, Q=6.0, blend=1.0, gate 0.42–0.48 m) |
| Mode-div | kp=10.0, kd=0.50, max_torque=7.5, soft_limit=0.30, soft_gain=0.80 |
| Known limitation | D4/D5 hip_yaw_abs_max > 0.35 rad; no sustained posture recovery |

### Verified NOT in K1
- No WBC
- No hidden torque
- No K3 combined notch
- No J3a damping increase
- No global Kp_pitch reduction
- No wheel_velocity notch

### Previous Validation Results
- **Step E**: 10/10 completed, 0 falls, 0 WBC, 0 hidden torque
- **Step C**: 7/7 completed, 0 falls, 0 WBC, 0 hidden torque (fixed-height only)
- **Full Step D**: 6/6 completed, 0 falls, 0 WBC, 0 hidden torque
- **D4/D5 hip_yaw**: D4=0.3595 rad, D5=0.3529 rad (still > 0.35 gate)

---

## 3. True Dynamic Step C Harness

Created `scripts/run_true_dynamic_height_step_c_validation.py` with 7 height profiles that all cross the notch gate:

| Profile | Steps | Setup | Description |
|---------|-------|-------|-------------|
| slow_ladder_0p330_to_0p480_to_0p330 | 5000 | low_0p330 | Stepwise height changes crossing gate up and down |
| medium_ramp_0p330_to_0p480 | 6000 | low_0p330 | Smooth ramp crossing gate both ways |
| abrupt_0p330_to_0p480 | 5000 | low_0p330 | Abrupt transitions across gate |
| random_dwell_cross_gate | 5000 | low_0p330 | Random dwell periods crossing gate multiple times |
| high_to_low_0p480_to_0p330 | 4000 | high_0p480 | Starting high, descending through gate |
| repeated_gate_crossing_0p400_0p460 | 5000 | low_0p330 | Oscillating around gate edges |
| stress_gate_crossing_0p410_0p490 | 5000 | low_0p330 | Aggressive crossing at gate margins |

The harness dynamically updates `height_cmd` and `target_joint_pos` during simulation so the robot actively tracks the height trajectory, crossing the 0.42–0.48 m notch gate in both directions.

**Required telemetry columns added:**
- `dynamic_height_active`, `dynamic_height_target_m`, `notch_height_gate_from_traj`
- `pitch_rate_raw_rad_s`, `pitch_rate_notched_rad_s`, `pitch_rate_effective_rad_s`
- `wip_notch_height_gate`, `wip_notch_filter_valid`
- `tau_pitch_rate_raw_signal`, `tau_pitch_rate_filtered_signal`

**Pass condition:** No falls, no WBC/hidden/ownership violation, no NaN/Inf, notch gate changes smoothly, no discontinuity at 0.42–0.48 crossing.

**Status:** Harness created and compilable. Full simulation runs pending.

---

## 4. Sustained Recovery Root-Cause Audit

Created `scripts/audit_k1_sustained_recovery_failure.py` which decomposes K1 post-push telemetry across 5 audit dimensions:

1. **Oscillation classification** — FFT analysis to determine dominant frequency
2. **Torque decomposition** — RMS/peak/mean for pitch P, pitch rate D, support position, support velocity, wheel velocity, and notch effect
3. **Frequency/phase analysis** — Cross-correlation between pitch, pitch_rate, support_error, wheel_velocity
4. **Recovery event search** — First <5°, first <3°, sustained 2s hold, sustained 5s hold, recovery-later-lost detection
5. **Notch effect** — Attenuation ratio in active region

**Expected conclusions (pending data):**
- Whether the 2.4–2.5 Hz mode is a WIP natural mode or torque-term coupling
- Whether coordinated state feedback is justified vs further notch/tuning
- Whether the sum-of-independent-torques architecture is fighting itself

**Status:** Script created and compilable. Requires K1 focused recovery telemetry to produce conclusions.

---

## 5. L Family Architecture and Results

### L1 — Coordinated Low-Frequency State Feedback

**Profile:** `l1_k1_coordinated_low_freq_feedback_v1`
**Base:** K1 (`k1_pitch_rate_notch_v1`)
**Gains:** Height-scheduled LQR-style gains for pitch, pitch_rate, support_error, support_vel
**Architecture:** Adds `coordinated_feedback(x)` to common wheel torque AFTER normal sagittal torque computation
**Telemetry:** `L_enabled`, `L_candidate_kind`, `L_state_*`, `L_feedback_torque_nm`, `L_gains_kind`

**Control law:**
```
tau_feedback = k_pitch * pitch + k_pitch_rate * pitch_rate
             + k_support * support_error + k_support_vel * support_vel
```

### L2 — Coordinated Phase-Lead Compensation

**Profile:** `l2_k1_coordinated_phase_lead_v1`
**Base:** K1
**Addition:** Phase-lead term using pitch acceleration proxy (`d(pitch_rate)/dt`) to compensate the ~90° phase lag that causes damping to feed the 2.5 Hz mode

### L3 — Coordinated Pitch Reference Stabilization

**Profile:** `l3_k1_coordinated_pitch_ref_stabilization_v1`
**Base:** K1
**Addition:** Small pitch reference correction (`pitch_ref_gain * support_error`, amplitude-limited) to reduce pitch-vs-support conflict without suppressing torque

**Status:** All three profiles implemented and registered. No simulation results yet.

---

## 6. M Family Architecture and Results

### M1 — Low-Band Body-Yaw Damping

**Profile:** `m1_k1_body_yaw_diff_wheel_v1`
**Base:** K1
**Parameters:** kp=0.5, kd=0.1, max_torque=1.5 Nm, height gate 0.34–0.42 m, activation threshold 0.05 rad
**Architecture:** Uses `DifferentialWheelYawStabilizer` for body-yaw correction through differential wheel velocity

### M2 — Support-Aware Body-Yaw Damping

**Profile:** `m2_k1_body_yaw_support_aware_v1`
**Base:** K1
**Parameters:** kp=0.8, kd=0.15, max_torque=2.0 Nm, support gate threshold 0.15 m, support rate threshold 0.05 m/s
**Architecture:** Modulates wheel-yaw correction based on support/contact confidence

**Status:** Profiles implemented and registered. Wheel-yaw correction handled through existing `DifferentialWheelYawStabilizer` infrastructure.

---

## 7. N Diagnostic Result

### N1 — Mild Phase-Lead Damping

**Profile:** `n1_k1_mild_phase_lead_damping_v1`
**Base:** K1
**Parameters:** Very mild phase-lead-compensated pitch rate damping (k_rate=0.3–0.5, k_lead=0.02–0.04)
**Purpose:** Diagnostic only — checks whether mild compensated damping can recover the transient J3a benefit without J3a's growing oscillation

**Status:** Profile implemented and registered. Diagnostic only.

---

## 8. K1 vs L/M/N Comparisons

No comparison data available yet — simulation runs for L/M/N candidates are pending. The infrastructure is ready for evaluation.

---

## 9–16. Analysis Sections (Pending Data)

The following sections require simulation data from L/M/N candidates:

- Section 9: Sustained recovery event analysis
- Section 10: D4/D5 hip-yaw/body-yaw analysis
- Section 11: Notch telemetry analysis
- Section 12: Direct hip-yaw telemetry analysis
- Section 13: Support/pitch quality
- Section 14: Roll/yaw/COM safety
- Section 15: Source integrity audit
- Section 16: WBC/hidden/ownership audit

---

## 17. Promotion Decision

**K1 remains current-best/default controller.**

No L, M, or N candidate is promoted. All new candidates are opt-in infrastructure ready for evaluation. Promotion requires:

1. Beating K1 on sustained recovery (≥2 s hold preferred, ≥5 s hold ideal)
2. Beating K1 on D4/D5 hip-yaw gate (hip_yaw_abs_max < 0.35 rad)
3. No regression on Step C/E/D metrics
4. Direct telemetry, no WBC, no hidden torque, no ownership violation

---

## 18. Current-Best After Task

| Item | Value |
|------|-------|
| Current-best | `K1_PITCH_RATE_NOTCH_V1` |
| Profile | `k1_pitch_rate_notch_v1` |
| Status | `CURRENT_BEST_PROMOTED_WITH_EXPANDED_KNOWN_LIMITATIONS` |

---

## 19. Known Limitations

1. **D4/D5 hip_yaw_abs_max > 0.35 rad** — Universal across all A/B/C/D/K1 profiles. M family candidates provide correct-actuator path (wheel-yaw) but not yet evaluated.
2. **No sustained posture recovery** — K1 notch reduces 2.5 Hz amplitude but does not eliminate it. L family candidates target this with coordinated state feedback but not yet evaluated.
3. **True dynamic-height Step C not yet run** — The harness exists but simulation results are pending. This means K1 notch gate crossing (0.42–0.48 m) has not been dynamically validated.
4. **L/M/N candidates not yet simulated** — Infrastructure complete; evaluation pending.

---

## 20. Files Changed

### New files:
- `scripts/run_true_dynamic_height_step_c_validation.py` — True dynamic-height Step C harness
- `scripts/audit_k1_sustained_recovery_failure.py` — Sustained recovery root-cause audit
- `scripts/audit_k1_d4_d5_body_yaw_to_hip_yaw_coupling.py` — D4/D5 body-yaw coupling audit
- `scripts/analyze_k1_controller_completion_results.py` — Candidate ranking and analysis

### Modified files:
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`:
  - Added L family coordinated state-feedback profiles and gain functions
  - Added M family body-yaw/wheel-yaw profiles with support-aware gating
  - Added N family mild phase-lead damping diagnostic profile
  - Added `SagittalAuthoritySchedule` fields: `enable_coordinated_sagittal_feedback`, `coordinated_feedback_kind`, `enable_body_yaw_wheel_stabilization`, and wheel-yaw parameters
  - Added L/M/N telemetry to diagnostics dict
  - Added `_prev_pitch_rate_for_L` and `_prev_pitch_rate_for_N` state variables

- `scripts/simulate_hierarchical_controller.py`:
  - Imported L/M/N profile constants
  - Registered L/M/N profiles in `SAGITTAL_AUTHORITY_PROFILES`
  - Added `--dynamic-height-trajectory` CLI argument for dynamic height simulation
  - Added dynamic height trajectory loading, interpolation, and runtime updates
  - Added dynamic height telemetry columns to telemetry dict
  - Added notch pitch-rate telemetry columns
  - Modified `target_joint_pos` computation to use posture regularizer when dynamic height active

---

## 21. Tests/Compile Checks Run

See Phase 8 section below.

---

## 22. Next Recommended Task

**Run simulation evaluations for L/M/N candidates:**

1. Run sustained recovery audit on existing K1 telemetry:
   ```
   python scripts/audit_k1_sustained_recovery_failure.py
   ```

2. Run L1 focused recovery evaluation:
   ```
   python scripts/simulate_hierarchical_controller.py --controller-mode balance-core \
     --sagittal-controller velocity-damped --vd-sagittal-authority-profile l1_k1_coordinated_low_freq_feedback_v1 \
     --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json \
     --steps 3000 --telemetry-decimation 1 --failure-window-steps 3000 \
     --push-enabled --push-magnitude-n 90 --push-duration-steps 10 --push-count 1 --push-start-step 300 \
     --sagittal-push-only --enable-mode-hip-yaw-divergence --mode-hip-yaw-div-kp 10.0 \
     --mode-hip-yaw-div-kd 0.50 --mode-hip-yaw-div-max-torque 7.5 \
     --mode-hip-yaw-div-soft-limit-rad 0.30 --mode-hip-yaw-div-soft-gain 0.80 \
     --mode-hip-yaw-div-ref-source target \
     --output-dir outputs/k1_controller_completion/L1_focused_recovery
   ```

3. Repeat for L2, L3, M1, M2, N1

4. If any candidate beats K1 in focused recovery:
   - Run true dynamic Step C
   - Run Step E
   - Run full Step D
   - Compare comprehensive metrics

5. Run true dynamic Step C for K1 baseline:
   ```
   python scripts/run_true_dynamic_height_step_c_validation.py
   ```
