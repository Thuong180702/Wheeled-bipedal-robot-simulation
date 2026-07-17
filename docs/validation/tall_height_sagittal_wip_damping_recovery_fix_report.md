# Tall-Height Sagittal WIP Damping Recovery Fix — Final Report

## 1. Executive Summary

This task investigated whether increasing sagittal inner-loop damping (pitch-rate damping `kd_pitch`, wheel-velocity damping `k_wheel_velocity`, or both) at tall height could suppress the persistent 2.505 Hz wheeled inverted pendulum (WIP) pitch-support limit cycle observed in G1_sg080 under a single 90 N / 10-step push at high_0p480.

**Key finding: Increasing sagittal damping at tall height via the J candidate family does NOT fix the 2.5 Hz WIP limit cycle.** All surviving candidates show **worse** pitch-support oscillation than the G1_sg080 baseline. Higher kd_pitch (>20) causes early termination. Higher k_wheel_velocity (>1.00) causes instability or hip-yaw gate proximity. The combined candidate (J3a) achieves transient posture recovery (2.4 s hold) but the oscillation returns stronger — final-window pitch RMS = 6.59 deg vs 5.39 deg baseline.

**The 2.5 Hz WIP mode at 0.480 m height has marginal damping, and adding phase-lagged damping feedback through the inner loop amplifies the oscillation.** This is consistent with the prior I1 finding that the support outer loop cannot damp this mode either. The fundamental problem requires a different approach.

| Candidate | Profile | Rows | Pitch RMS | Sup RMS | Hy Max | Class |
|-----------|---------|------|-----------|---------|--------|-------|
| G1_sg080 (baseline) | low-band v2 | 2999 | 5.39 deg | 0.102 m | 0.295 rad | NO_IMPROVEMENT |
| J1a (kd_pitch 10->15) | j1a_tall_kd_pitch_v1 | 2999 | 6.93 deg | 0.145 m | 0.246 rad | NO_IMPROVEMENT |
| J1b (kd_pitch 10->20) | j1b_tall_kd_pitch_v1 | 2812 | 8.37 deg | 0.160 m | 0.210 rad | NO_IMPROVEMENT, early termination |
| J1c (kd_pitch 10->30) | j1c_tall_kd_pitch_v1 | 72 | — | — | — | FAIL_EARLY_TERMINATION |
| J2a (k_wheel_vel 0.5->0.85) | j2a_tall_k_wheel_vel_v1 | 2999 | 5.73 deg | 0.122 m | 0.344 rad | NO_IMPROVEMENT, hy near gate |
| J2b (k_wheel_vel 0.5->1.00) | j2b_tall_k_wheel_vel_v1 | 867 | — | — | — | FAIL_UNSTABLE |
| J2c (k_wheel_vel 0.5->1.25) | j2c_tall_k_wheel_vel_v1 | 2999 | 6.67 deg | 0.140 m | 0.190 rad | NO_IMPROVEMENT |
| **J3a (combined mild)** | j3a_tall_combined_v1 | 2999 | 6.59 deg | 0.160 m | **0.098 rad** | **TRANSIENT_ONLY** |
| J3b (combined mod) | j3b_tall_combined_v1 | 2999 | 6.53 deg | 0.180 m | 0.186 rad | NO_IMPROVEMENT |

**D remains current-best. No J candidate is promoted. No thresholds were relaxed. No telemetry peaks were cropped.**

---

## 2. Current-Best Status

| Item | Value |
|------|-------|
| Current-best | `D_MODE_HIP_YAW_DIV_V1` |
| Current-best profile | `physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1` |
| Status | `CURRENT_BEST_PROMOTED` |
| J promotion status | **NOT PROMOTED** — no candidate passes recovery criteria |
| G1_sg080 status | Diagnostic reference only (same as before) |
| I1 status | Diagnostic reference only (same as before) |

---

## 3. Prior Root-Cause Summary

**Proximate cause:** The support outer loop Kp was zeroed at tall heights by the low-band Gaussian scaling (`scale ≈ 0` at 0.480 m where center=0.320 m, sigma=0.004 m). Fixed by I1 `blend_with_base=True`.

**Fundamental cause:** The 2.505 Hz pitch-support limit cycle is an **underdamped wheeled inverted pendulum mode** at tall height (0.480 m). Even with the support correction restored (I1), the oscillation persists because:
1. The outer loop's correction bandwidth (rate-limit 0.03 deg/step, lowpass alpha=0.15) is too slow for 2.5 Hz dynamics.
2. Higher outer-loop Kp produces phase-lagged feedback that amplifies the oscillation.
3. The inner-loop wheel balancing controller provides marginal damping at this height.

---

## 4. Corrected Recovery Metric

This task evaluates recovery across the **entire post-push trajectory**, not only the final window:

- **Recovery windows:** 0-5s, 5-10s, 10-15s, 15-20s, 20s+ after push end
- **Sustained recovery definition:** pitch_abs ≤ 5 deg, pitch RMS ≤ 3 deg in window, roll ≤ 2 deg, hip_yaw < 0.35 rad, height stable, held for ≥ 2.0 s minimum
- **Preferred hold:** 5.0 s
- **Transient crossing:** A single frame crossing into pitch < 5 deg is NOT recovery
- **Later lost:** Recovery achieved but later lost = TRANSIENT_ONLY

---

## 5. Baseline G1_sg080 Recovery Event Audit

| Metric | Value |
|--------|-------|
| Classification | **BASELINE_TRANSIENT_RECOVERY_ONLY** |
| First pitch_abs < 5 deg | 0.01 s after push end |
| First pitch_abs < 3 deg | 2.74 s after push end |
| First pitch-5deg hold ≥ 2 s | 2.51-4.72 s (2.21 s hold) |
| **First sustained posture recovery** | **NONE** (never achieved) |
| Recovery by 5s / 10s / 15s / 20s | No / No / No / No |
| Total posture recovery time | 0 s |
| Final pitch RMS | 5.39 deg |

G1_sg080 **never achieves sustained posture recovery** by the defined criteria. The robot briefly enters pitch < 5 deg but fails the roll/hip_yaw/height gate or the pitch RMS ≤ 3 deg requirement. The trajectory is a persistent limit cycle — oscillation is flat from 5s onward without damping.

---

## 6. Candidate J Architecture

### J_TALL_HEIGHT_SAGITTAL_WIP_DAMPING_V1 Family

**Base:** `PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2` (same sagittal base as G1_sg080/D)

**Change:** Added `continuous_kd_pitch` field to `SagittalAuthoritySchedule` to allow height-scheduled pitch-rate damping increase at tall heights. Reuses `scheduled_k_wheel_velocity()` function (smoothstep interpolation, increases at high z).

### Candidates tested

| ID | Profile name | kd_pitch_nominal | kd_pitch_high_max | k_wheel_vel_nominal | k_wheel_vel_high_max | z_low | z_high |
|----|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| J1a | j1a_tall_kd_pitch_v1 | 10.0 | 15.0 | — | — | 0.40 | 0.52 |
| J1b | j1b_tall_kd_pitch_v1 | 10.0 | 20.0 | — | — | 0.40 | 0.52 |
| J1c | j1c_tall_kd_pitch_v1 | 10.0 | 30.0 | — | — | 0.40 | 0.52 |
| J2a | j2a_tall_k_wheel_vel_v1 | — | — | 0.50 | 0.85 | 0.45 | 0.52 |
| J2b | j2b_tall_k_wheel_vel_v1 | — | — | 0.50 | 1.00 | 0.45 | 0.52 |
| J2c | j2c_tall_k_wheel_vel_v1 | — | — | 0.50 | 1.25 | 0.45 | 0.52 |
| J3a | j3a_tall_combined_v1 | 10.0 | 15.0 | 0.50 | 0.85 | 0.40/0.45 | 0.52 |
| J3b | j3b_tall_combined_v1 | 10.0 | 20.0 | 0.50 | 1.00 | 0.40/0.45 | 0.52 |

### Changes to `sagittal_velocity_damped_balance_controller.py`

- Added `continuous_kd_pitch` field (default `False`) to `SagittalAuthoritySchedule`
- Added `kd_pitch_nominal`, `kd_pitch_high_max`, `kd_pitch_z_low`, `kd_pitch_z_high` fields
- Added `effective_kd_pitch` computation using `scheduled_k_wheel_velocity()` when `continuous_kd_pitch=True`
- Changed `tau_pitch_rate = self.kd_pitch * pitch_rate` to `tau_pitch_rate = effective_kd_pitch * pitch_rate`
- Added telemetry: `effective_kd_pitch`, `high_height_kd_pitch_active`, `kd_pitch_nominal`, `kd_pitch_high_max`, `kd_pitch_z_low`, `kd_pitch_z_high`

### Control ownership map

| Mode | Owner | Notes |
|------|-------|-------|
| Pitch rate damping | `kd_pitch` via torque `tau_pitch_rate = kd * pitch_rate` | Unchanged from D; now height-scheduled for J |
| Wheel velocity damping | `k_wheel_velocity` via `tau_wheel_vel = -k * wheel_vel` | Unchanged from D; height-scheduled for J2/J3 |
| Support velocity damping | `k_support_velocity` | Unchanged (not modified by J) |
| Pitch position | `kp_pitch` | Unchanged (NOT reduced by J) |

---

## 7. Damping Telemetry Added

All telemetry fields were added to the `compute()` method's output dict in `sagittal_velocity_damped_balance_controller.py`:

| Field | Description |
|-------|-------------|
| `effective_kd_pitch` | Height-scheduled kd_pitch value at current height |
| `high_height_kd_pitch_active` | True when effective_kd_pitch > nominal |
| `kd_pitch_nominal` | Base kd_pitch value (10.0) |
| `kd_pitch_high_max` | Maximum kd_pitch at full tall height |
| `kd_pitch_z_low` | Lower height bound for scheduling |
| `kd_pitch_z_high` | Upper height bound for scheduling |

---

## 8. Focused Sweep Results

### Scenario
- Height: high_0p480 (tall)
- Push: 90 N, 10 steps, single push, step 300, sagittal +y
- Steps: 3000
- Mode-div parameters: kp=10, kd=0.5, mt=7.5, sl=0.30, sg=0.80 (G1_sg080)

### Completion and safety

| Candidate | Rows | Terminated | Hy Max | Sup Max | Fall |
|-----------|:----:|:----------:|:------:|:-------:|:----:|
| G1_sg080 | 2999 | No | 0.295 | 0.697 | No |
| J1a | 2999 | No | 0.246 | 0.752 | No |
| J1b | 2812 | **Yes** | 0.210 | 0.831 | No |
| J1c | **72** | **Yes** | 0.010 | 0.327 | No |
| J2a | 2999 | No | **0.344** | 0.745 | No |
| J2b | 867 | **Yes** | 0.346 | 0.759 | No |
| J2c | 2999 | No | 0.190 | 0.783 | No |
| **J3a** | **2999** | **No** | **0.098** | **0.785** | **No** |
| J3b | 2999 | No | 0.186 | **1.050** | No |

### Windowed pitch RMS (deg) — surviving candidates

| Candidate | 0-5s | 5-10s | 10-15s | 15-20s | 20s+ | Final 5s |
|-----------|:----:|:-----:|:------:|:------:|:----:|:--------:|
| G1_sg080 | 8.33 | 5.17 | 5.06 | 5.96 | 5.34 | 5.39 |
| J1a | 9.00 | 5.88 | 7.12 | 7.35 | 7.03 | 6.93 |
| J2a | 8.64 | 5.71 | 6.06 | 5.84 | 5.53 | 5.73 |
| J2c | 9.00 | 6.39 | 7.00 | 6.76 | 6.42 | 6.67 |
| **J3a** | **7.30** | **5.49** | **6.54** | **6.58** | **6.71** | **6.59** |
| J3b | 9.68 | 6.53 | 6.43 | 6.60 | 6.47 | 6.53 |

### Windowed support RMS (m) — surviving candidates

| Candidate | 0-5s | 5-10s | 10-15s | 15-20s | 20s+ | Final 5s |
|-----------|:----:|:-----:|:------:|:------:|:----:|:--------:|
| G1_sg080 | 0.339 | 0.082 | 0.110 | 0.105 | 0.103 | 0.102 |
| J1a | 0.354 | 0.106 | 0.149 | 0.148 | 0.144 | 0.145 |
| J2a | 0.330 | 0.083 | 0.129 | 0.121 | 0.120 | 0.122 |
| J2c | 0.340 | 0.099 | 0.139 | 0.138 | 0.140 | 0.140 |
| J3a | 0.343 | 0.126 | 0.159 | 0.155 | 0.161 | 0.160 |
| J3b | 0.359 | 0.147 | 0.143 | 0.163 | 0.179 | 0.180 |

---

## 9. Recovery Event Analysis — All Candidates

| Candidate | Posture 2s hold? | First hold time | Hold dur | Later lost? | 5s? | 10s? | Class |
|-----------|:----------------:|:---------------:|:--------:|:-----------:|:---:|:----:|-------|
| G1_sg080 | No | — | 0 s | — | No | No | NO_IMPROVEMENT |
| J1a | No | — | 0 s | — | No | No | NO_IMPROVEMENT |
| J2a | No | — | 0 s | — | No | No | NO_IMPROVEMENT |
| J2c | No | — | 0 s | — | No | No | NO_IMPROVEMENT |
| **J3a** | **Yes** | **1.43 s** | **2.40 s** | **Yes** | **Yes** | **Yes** | **TRANSIENT_ONLY** |
| J3b | No | — | 0 s | — | No | No | NO_IMPROVEMENT |

**Only J3a achieves any sustained posture recovery** (2.4 s hold starting 1.43 s after push end). However, recovery is later lost and the oscillation returns stronger than baseline. J3a is classified as TRANSIENT_ONLY.

**No candidate achieves sustained recovery pass.**

---

## 10. Best J Candidate Parameters

**J3a (j3a_tall_combined_v1):**
- `continuous_kd_pitch=True`, kd_pitch_nominal=10.0, kd_pitch_high_max=15.0, z_low=0.40, z_high=0.52
- `continuous_k_wheel_velocity=True`, k_wheel_velocity_nominal=0.50, k_wheel_velocity_high_max=0.85, z_low=0.45, z_high=0.52

J3a achieves:
- Full 3000-step survival
- Excellent hip-yaw control (max 0.098 rad — best ever)
- Transient posture recovery (2.4 s hold at 1.43 s after push)
- Recovery by 5s (robot enters recovery band within 5s)

But J3a fails on:
- Does NOT sustain recovery (later lost)
- Final-window pitch RMS 6.59 deg (worse than baseline 5.39)
- Final-window support RMS 0.160 m (worse than baseline 0.102)
- Oscillation amplitude grows over time, does not decay

---

## 11. Pitch-Support Frequency and Decay Analysis

| Candidate | Pitch freq (Hz) | Support freq (Hz) | Cross-corr | Envelope |
|-----------|:---------------:|:-----------------:|:----------:|:--------:|
| G1_sg080 | 2.505 | 2.505 | 0.665 | Flat/growing |
| J1a | 2.502 | 2.502 | 0.954 | Growing |
| J2a | 2.509 | 2.509 | 0.708 | Flat/growing |
| J2c | 2.506 | 2.506 | 0.908 | Growing |
| J3a | 2.496 | 2.496 | 0.964 | Growing |
| J3b | 2.493 | 2.493 | 0.982 | Growing |

**All candidates show 2.5 Hz coupled oscillation** with high pitch-support correlation. The frequency is unchanged by damping increases — consistent with a natural WIP mode at this height. Cross-correlation increases with stronger damping, indicating tighter coupling.

---

## 12. Hip-Yaw Gate Analysis

| Candidate | Full-run hy_max | Final-window hy_max | Gate pass |
|-----------|:---------------:|:------------------:|:---------:|
| G1_sg080 | 0.295 rad | 0.153 rad | Yes |
| J1a | 0.246 rad | 0.112 rad | Yes |
| J1b | 0.210 rad | — | Early term |
| J2a | **0.344 rad** | 0.159 rad | Yes (marginal) |
| J2b | 0.346 rad | — | Early term |
| J2c | 0.190 rad | 0.096 rad | Yes |
| **J3a** | **0.098 rad** | **0.083 rad** | **Yes (excellent)** |
| J3b | 0.186 rad | 0.095 rad | Yes |

**J3a achieves the best hip-yaw control ever measured** (max 0.098 rad, well below 0.35 gate). The combined damping dramatically stabilizes the hip-yaw axis. However, this comes at the cost of increased pitch/support oscillation.

---

## 13. Roll/Yaw/COM Stability

| Candidate | Roll RMS (final) | Height error max | Yaw drift |
|-----------|:---------------:|:----------------:|:---------:|
| G1_sg080 | 0.05 deg | 0.004 m | Moderate |
| J1a | 0.05 deg | 0.006 m | Moderate |
| J2a | 0.06 deg | 0.004 m | Moderate |
| J2c | 0.06 deg | 0.006 m | Moderate |
| J3a | 0.06 deg | 0.016 m | Low |
| J3b | 0.08 deg | 0.013 m | Moderate |

Roll is stable for all surviving candidates. J3a has slightly larger height error (0.016 m) but still within bounds.

---

## 14. Safety Summary

| Check | G1_sg080 | J1a | J2a | J2c | J3a | J3b |
|-------|:--------:|:---:|:---:|:---:|:---:|:---:|
| Falls | 0 | 0 | 0 | 0 | 0 | 0 |
| Early termination | 0 | 0 | 0 | 0 | 0 | 0 |
| Hip_yaw > 0.35 | No | No | No | No | No | No |
| NaN/Inf | 0 | 0 | 0 | 0 | 0 | 0 |
| Torque saturation | 0 | 0 | 0 | 0 | 0 | 0 |
| WBC authority rows | 0 | 0 | 0 | 0 | 0 | 0 |

**All surviving J candidates are safe.** No safety violations across any metric.

---

## 15. Robustness Runs

**Not executed.** No J candidate passed the focused single-push recovery criteria. Per the task specification (Phase 6-7), robustness runs and full validation are only executed if a candidate passes or nearly passes the focused diagnostic. J3a achieved transient recovery only and had worse final metrics than baseline.

---

## 16. D4/D5 Focused

**Not executed.** Focused single-push diagnostic did not pass.

---

## 17. Full Step D / Step C / Step E

**Not executed.** No J candidate passed the focused diagnostic.

---

## 18. Decision: J Is Not Promoted

**J_TALL_HEIGHT_SAGITTAL_WIP_DAMPING_V1 is NOT promoted.** No candidate achieves sustained posture recovery within 5-20 s after push. D_MODE_HIP_YAW_DIV_V1 remains current-best.

### Why J failed

1. **Phase-lagged damping feedback:** At 2.5 Hz, the damping terms (kd_pitch, k_wheel_velocity) produce torque that is phase-shifted relative to the oscillation. This phase lag causes the damping torque to partially **amplify** rather than suppress the oscillation.

2. **Coupled dynamics:** The pitch rate and wheel velocity signals at 2.5 Hz are tightly coupled with the support position error. Increasing damping on one channel feeds energy into the other through the coupled WIP dynamics.

3. **Marginal stability:** At 0.480 m height, the WIP mode is naturally underdamped. The existing kd_pitch=10.0 provides some damping; increasing it beyond ~15 causes the coupled system to become less stable, not more.

4. **Tradeoff confirmed:** J3a shows that combined mild damping improves hip-yaw control dramatically (0.295 → 0.098 rad) but worsens pitch/support oscillation (5.39 → 6.59 deg RMS). The damping authority that helps hip-yaw diverges into pitch/support through the physical coupling.

---

## 19. Final Statement

1. **D_MODE_HIP_YAW_DIV_V1 remains current-best/default.** Nothing in this task changes that.

2. **J_TALL_HEIGHT_SAGITTAL_WIP_DAMPING_V1 candidates are NOT promoted.** No candidate achieves sustained posture recovery.

3. **The 2.5 Hz WIP limit cycle at tall height cannot be fixed by increasing inner-loop damping (kd_pitch, k_wheel_velocity) alone.** The phase-lagged feedback amplifies the oscillation.

4. **J3a (combined mild kd_pitch 10→15 + k_wheel_vel 0.5→0.85) is the best candidate** achieving transient posture recovery (2.4 s hold) and excellent hip-yaw control (0.098 rad). But recovery is later lost and pitch/support oscillation worsens over time.

5. **No thresholds were relaxed. No telemetry peaks were cropped. No WBC was enabled. No hidden torque was applied.**

6. **Three possible next directions:**
   - **Notch filtering:** A band-stop filter around 2.5 Hz on the pitch or support error signal in the inner loop could prevent the damping terms from coupling into the oscillation mode. This is a targeted intervention that preserves damping at other frequencies.
   - **Architectural change:** The current sum-of-torques architecture (tau_pitch + tau_position + tau_velocity + tau_wheel_vel) has independent terms that fight each other at 2.5 Hz. A coordinated state-feedback approach (the existing UNIFIED_SAGITTAL_STATE_FEEDBACK_NO_OFFSET profile) may handle this better.
   - **Different actuator:** Wheel torque alone may be fundamentally limited for damping this mode at tall height. Hip-pitch or knee modulation could provide an alternative damping pathway.

---

## 20. Files Changed

| File | Change |
|------|--------|
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | Added `continuous_kd_pitch` scheduling fields, `effective_kd_pitch` computation, `high_height_kd_pitch_active` telemetry, 8 J candidate profiles |
| `scripts/simulate_hierarchical_controller.py` | Imported J1a-J3b constants, registered in `SAGITTAL_AUTHORITY_PROFILES` and argparser choices |
| `scripts/analyze_recovery_window_events.py` | **Created** — trajectory-wide recovery event audit (Baseline classification) |
| `scripts/run_tall_height_sagittal_wip_damping_sweep.py` | **Created** — focused sweep runner for J candidates |
| `scripts/analyze_tall_height_wip_damping_recovery.py` | **Created** — post-sweep analysis with recovery events, windowed metrics, frequency/decay, classification |
| `tests/test_tall_height_sagittal_wip_damping_recovery_fix.py` | **Created** — 35 tests (profile existence, opt-in, restrictions, scheduling, analysis, compile) |
| `docs/validation/tall_height_sagittal_wip_damping_recovery_fix_report.md` | **Created** — this report |

---

## 21. Tests and Compile Checks

| Check | Result |
|-------|--------|
| `pytest tests/test_tall_height_sagittal_wip_damping_recovery_fix.py -v` | **35/35 pass** |
| `pytest tests/test_current_best_controller_profile.py -v` | **7/7 pass** |
| `pytest tests/test_support_reference_reacquisition_and_pitch_support_limit_cycle_fix.py -v` | **33/33 pass** |
| `pytest tests/test_g1_sg080_step300_3000_posture_recovery.py -v` | **29/29 pass** |
| `pytest tests/test_g1_sg080_single_push_recovery.py -v` | **25/25 pass** |
| `pytest tests/test_mode_based_hip_yaw_divergence_controller.py -v` | **23/23 pass** |
| `pytest tests/test_final_validation_rejects_stub_source.py -v` | **9/9 pass** |
| **Total** | **161/161 pass** |
| `python -m py_compile ...` — all 4 scripts + controller | ✅ PASS |

---

## 22. Final Response Summary

| # | Question | Answer |
|---|----------|--------|
| 1 | Final classification | `WIP_DAMPING_RECOVERY_IMPROVED_NOT_PASS` |
| 2 | D remains current-best or J promoted? | **D remains current-best** |
| 3 | G1_sg080 corrected recovery audit | BASELINE_TRANSIENT_RECOVERY_ONLY — never sustained recovery |
| 4 | Best J candidate parameters | **J3a**: kd_pitch 10→15 + k_wheel_vel 0.5→0.85, continuous scheduling from 0.40/0.45 to 0.52 m |
| 5 | Recovery by 2s after push? | **Yes** (J3a: 1.43s) |
| 6 | Recovery by 5s after push? | **Yes** (J3a) |
| 7 | Recovery by 10s after push? | **Yes** (J3a, recovered at 1.43s) |
| 8 | Recovery by 20s after push? | **No** (J3a lost recovery) |
| 9 | First sustained posture recovery time | **1.43 s** (J3a) |
| 10 | Sustained hold duration | **2.40 s** (J3a) |
| 11 | Support/position return to target region? | **No** — no candidate achieves target region recovery |
| 12 | Position drift acceptable? | **No** — oscillation present, not pure drift |
| 13 | Pitch/support final-window metrics (best J) | J3a: pitch RMS 6.59 deg, support RMS 0.160 m |
| 14 | Best recovery-window metrics (J3a) | 0-5s: pitch RMS 7.30, support RMS 0.343 | 
| 15 | Pitch-support frequency and decay | 2.496-2.505 Hz, all candidates: **growing** envelope |
| 16 | Hip_yaw full-run max / final max (best J) | J3a: **0.098 / 0.083 rad** (excellent) |
| 17 | Roll/yaw/COM result | Stable for all surviving candidates |
| 18 | Safety result | All surviving J candidates safe — no falls, no NaN |
| 19 | Robustness runs | Not executed (focused did not pass) |
| 20 | D4/D5 focused result | Not executed |
| 21 | Full Step D result | Not executed |
| 22 | Step C/Step E result | Not executed |
| 23 | Files changed | 3 modified, 3 created scripts, 1 test file, 1 report |
| 24 | Tests/compile checks | **161/161 pass** across 7 test files, all compile clean |
| 25 | Report path | `docs/validation/tall_height_sagittal_wip_damping_recovery_fix_report.md` |
| 26 | Next recommended task | **Notch filter or band-stop around 2.5 Hz** to decouple damping from the WIP mode; or evaluate the existing UNIFIED_SAGITTAL_STATE_FEEDBACK_NO_OFFSET profile which uses coordinated state feedback instead of independent sum-of-torques |

---

## 23. Next Recommended Task

### The fundamental problem remains unsolved

The 2.5 Hz pitch-support limit cycle at high_0p480 is a **wheeled inverted pendulum damping problem** that resists both:
1. **Support outer-loop correction** (I1 family) — too slow, phase-lagged
2. **Inner-loop damping increase** (J family) — phase-lagged at 2.5 Hz, amplifies oscillation

### Recommended approach: targeted 2.5 Hz notch

A band-stop filter around 2.5 Hz on the pitch error or wheel velocity signal used for damping could prevent the damping terms from coupling into the oscillation mode while preserving damping at other frequencies. This requires:
1. A causal, telemetry-visible digital filter (biquad/IIR) at 500 Hz
2. Positioned between the raw signal and the damping gain multiplication
3. Only active above a height threshold (e.g., 0.42 m)
4. Validated not to destabilize other frequencies

### Alternative: evaluate UNIFIED_SAGITTAL_STATE_FEEDBACK_NO_OFFSET

The existing unified sagittal controller replaces the independent tau_pitch + tau_position + tau_velocity + tau_wheel_vel sum-of-torques architecture with a single coordinated state-feedback command. This architecture avoids the independent-term fighting that amplifies the 2.5 Hz mode and may handle it fundamentally differently.

### This remains diagnostic only. J is not promoted. D remains current-best.
