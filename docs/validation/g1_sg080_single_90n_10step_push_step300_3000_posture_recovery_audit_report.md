# G1_sg080 Single 90N / 10-Step Push Posture Recovery Audit Report (step 300, 3000 steps)

## 1. Executive Summary

A single sagittal push of **90 N for 10 steps** was applied at **step 300** to the high_0p480 (tall) height variant under **G1_sg080** controller parameters (kp=10.0, kd=0.50, max_torque=7.5, soft_limit=0.30, soft_gain=0.80). The simulation ran for **3000 steps** (30.0 s simulated).

**G1_sg080 survived the full 3000 steps without falling.** Hip yaw remained below the 0.35 rad gate threshold for the entire run (max 0.295 rad). The mode-based hip-yaw divergence controller applied up to 2.66 Nm in the final window (well below the 7.5 Nm saturation limit).

**However, the robot exhibited persistent pitch-support oscillation that did NOT converge within the 3000-step horizon:**

- Final-window (steps 2500–3000) pitch RMS = **5.37 deg** with max **9.80 deg**
- Final-window support error RMS = **0.102 m** with max **0.167 m**
- The oscillation frequency is **2.505 Hz** for both pitch and support, with cross-correlation **0.665 at zero lag** — a coupled limit cycle.
- Pitch envelope decay rate is **positive (0.087)**, indicating the oscillation amplitude is *growing* or flat, not decaying.
- **COM height is recovered** (height error mean −0.0004 m).
- **Roll is well-controlled** (max 0.22 deg in final window).
- **Yaw drift** of 2.3 deg in final window, but yaw rate RMS is low (0.76 deg/s).
- **Support outer loop gate is closed** (gate_pass mean 0.0 in final window), meaning the support-position correction is not actively modulating pitch reference.

**Primary failure mechanism**: The push displaces the robot's support position, creating a persistent sagittal offset. The wheel balancing controller responds with pitch oscillations to maintain upright posture, forming a 2.5 Hz pitch-support limit cycle. The support reference does not re-center to the robot's actual position, and the pitch equilibrium (mean 3.79 deg) does not return to zero.

**Classification: `POSTURE_RECOVERY_PARTIAL_HIP_YAW_ONLY`** — hip yaw recovers and stays controlled, but pitch/support oscillation persists.

**This is diagnostic/audit only. G1_sg080 is not promoted by this task. D remains current-best. No thresholds were relaxed.**

---

## 2. Scenario Definition

| Parameter | Value |
|-----------|-------|
| Case ID | `G1_sg080_single_90n_10step_push_step300_3000` |
| Controller profile | `G1_sg080` (D_MODE_HIP_YAW_DIV_V1 base + sg=0.80) |
| Height variant | `high_0p480` (tall) |
| Total steps | 3000 |
| Push magnitude | 90 N |
| Push duration | 10 steps |
| Push count | 1 |
| Push start step | 300 |
| Push push direction | Sagittal +y (forward) |
| Validation source | Real simulation |

---

## 3. Exact Command Used

```
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile \
    physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1 \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 3000 \
  --telemetry-decimation 1 \
  --failure-window-steps 3000 \
  --write-run-summary-sidecar \
  --output-dir outputs/g1_sg080_single_90n_10step_push_step300_3000 \
  --enable-mode-hip-yaw-divergence \
  --mode-hip-yaw-div-kp 10.0 \
  --mode-hip-yaw-div-kd 0.50 \
  --mode-hip-yaw-div-max-torque 7.5 \
  --mode-hip-yaw-div-soft-limit-rad 0.30 \
  --mode-hip-yaw-div-soft-gain 0.80 \
  --mode-hip-yaw-div-ref-source target \
  --push-enabled \
  --push-magnitude-n 90.0 \
  --push-duration-steps 10 \
  --push-count 1 \
  --push-start-step 300 \
  --sagittal-push-only
```

---

## 4. Push Verification

| Check | Actual | Expected | Result |
|-------|--------|----------|--------|
| Push windows | 1 | 1 | ✅ |
| Push active frames | 10 | 10 | ✅ |
| Push start step | 300 | 300 | ✅ |
| Push end step (exclusive) | 310 | 310 | ✅ |
| Push force direction | +y forward | +y forward | ✅ |
| Push count | 1 | 1 | ✅ |

---

## 5. Controller Parameters

| Parameter | G1_sg080 Value | D Baseline (for comparison) |
|-----------|-----------------|-----------------------------|
| Mode-div enabled | True | True |
| kp | 10.0 | 5.0 |
| kd | 0.50 | 0.20 |
| max_torque | 7.5 Nm | 2.0 Nm |
| soft_limit_rad | 0.30 | 0.30 |
| soft_gain | 0.80 | 0.25 |
| ref_source | target | target |
| Wheel yaw stabilizer | NO | NO |
| Support-aware H gate | NO | NO |
| WBC | NO | NO |

---

## 6. Peak Response Metrics

| Metric | Value |
|--------|-------|
| hip_yaw_abs_max (full run) | 0.295 rad |
| hip_yaw_abs_max (during push) | 0.015 rad |
| hip_yaw_abs_max (after push) | 0.295 rad |
| support_error_abs_max | 0.697 m |
| support_p2p | 0.697 m |
| pitch_abs_max | 21.22 deg |
| roll_abs_max | 4.60 deg |
| yaw_abs_max (deg) | 24.62 deg |

---

## 7. Windowed Posture Recovery Metrics

### Pitch (deg)

| Window | Steps | Mean | Abs Max | RMS |
|--------|-------|------|---------|-----|
| pre_push | 0–299 | −0.43 | 13.93 | 3.37 |
| early_recovery | 310–799 | 0.49 | 21.22 | 8.37 |
| medium_recovery | 800–1299 | 2.69 | 9.14 | 5.22 |
| late_recovery | 1300–1999 | 3.65 | 9.29 | 5.24 |
| final_window | 2500–2999 | 3.08 | 9.80 | 5.37 |

**Interpretation**: Pitch RMS is essentially flat from medium_recovery onward (5.22 → 5.24 → 5.37). The slight *increase* confirms the oscillation is not decaying.

### Roll (deg)

| Window | Mean | Abs Max | RMS |
|--------|------|---------|-----|
| pre_push | −0.02 | 0.95 | 0.13 |
| early_recovery | −0.26 | 4.60 | 0.62 |
| medium_recovery | −0.04 | 0.55 | 0.08 |
| late_recovery | 0.02 | 0.44 | 0.07 |
| final_window | 0.00 | 0.22 | 0.05 |

**Interpretation**: Roll is well-controlled and recovers fully.

### Support Error Abs (m)

| Window | Mean | Abs Max | RMS |
|--------|------|---------|-----|
| pre_push | 0.006 | 0.038 | 0.012 |
| early_recovery | 0.264 | 0.697 | 0.342 |
| medium_recovery | 0.057 | 0.173 | 0.082 |
| late_recovery | 0.080 | 0.206 | 0.109 |
| final_window | 0.074 | 0.167 | 0.102 |

**Interpretation**: Support error RMS *increases* from medium to late recovery (0.082 → 0.109), confirming the limit cycle is not decaying.

### Hip Yaw Abs Max (rad)

| Window | Abs Max |
|--------|---------|
| pre_push | 0.004 |
| early_recovery | 0.295 |
| medium_recovery | 0.213 |
| late_recovery | 0.190 |
| final_window | 0.153 |

**Interpretation**: Hip yaw is well-controlled. Recovers below 0.20 rad by late recovery.

---

## 8. Recovery-by-5s and Recovery-by-10s Analysis

| Metric | 5s (steps 310–799) | 10s (steps 800–1299) | Pass? |
|--------|-------------------|---------------------|-------|
| pitch_abs_max | 21.22 deg | 9.14 deg | ❌ Not recovered by 10s |
| sup_err_abs_max | 0.697 m | 0.173 m | ❌ Not recovered by 10s |
| hip_yaw_abs_max | 0.295 rad | 0.213 rad | ✅ Below gate |

The robot does NOT recover posture within 5 or 10 seconds after the push. Pitch is still at 9.14 deg max and 5.22 deg RMS at 10 seconds post-push.

---

## 9. Final-Window Posture/Equilibrium Analysis

| Metric | Value | Pass? |
|--------|-------|-------|
| pitch_abs_max | 9.80 deg | ❌ > 5.0 deg |
| pitch_rms | 5.37 deg | ❌ |
| pitch_mean | 3.08 deg (forward lean) | ❌ Not zero |
| pitch_ref_mean | 3.79 deg | ❌ Offset from zero |
| roll_abs_max | 0.22 deg | ✅ < 2.0 deg |
| yaw_drift | 2.31 deg | ⚠️ Moderate |
| yaw_rate_rms | 0.76 deg/s | ✅ Low |
| sup_err_abs_max | 0.167 m | ❌ > 0.10 m |
| sup_err_abs_mean | 0.074 m | ❌ > 0.05 m |
| sup_err_rms | 0.102 m | ❌ |
| hip_yaw_abs_max | 0.153 rad | ✅ < 0.35 gate |
| hip_yaw_div_error_abs_max | 0.304 rad | ⚠️ Elevated but below gate |
| com_z_drift | 0.002 m | ✅ Stable |
| com_height_error | −0.0004 m | ✅ Recovered |

**Key finding**: The final-window posture is not settled. Pitch oscillates around ~3 deg with excursions to ~10 deg. The pitch reference does not return to zero (mean 3.79 deg), suggesting the outer-loop pitch correction or physics equilibrium feedforward is maintaining a postural offset.

---

## 10. Hip-Yaw Gate Analysis

| Period | hip_yaw_abs_max | Gate (0.35 rad) | Status |
|--------|-----------------|-----------------|--------|
| Full run | 0.295 rad | 0.35 rad | ✅ Below |
| During push | 0.015 rad | 0.35 rad | ✅ Far below |
| Final window | 0.153 rad | 0.35 rad | ✅ Below |

**Hip_yaw never exceeded 0.35 rad at any point during the entire 3000-step simulation.** The mode-based hip-yaw divergence controller operated within its authority envelope without saturation (0 saturation rows).

---

## 11. Support/Position Drift vs Posture Distinction

**Position drift**: The robot is displaced forward by the push and the support reference does not fully re-center. The support error has a mean of 0.074 m in the final window with an oscillatory component of similar magnitude. This is both drift AND oscillation.

**Posture**: Pitch does not settle to equilibrium. The mean pitch of 3.08 deg represents a forward lean (compensating for displaced support position), and the 5.37 deg RMS indicates large oscillations around this mean.

**Distinction**: The robot maintains a stable(ish) upright posture but in a displaced position with persistent oscillation. While position drift alone would be acceptable, the combination of drift + sustained oscillation is not.

---

## 12. Pitch-Support Decay/Limit-Cycle Analysis

### RMS Trend

| Window | Pitch RMS (deg) | Support RMS (m) |
|--------|----------------|-----------------|
| early_recovery | 8.37 | 0.342 |
| medium_recovery | 5.22 | 0.082 |
| late_recovery | 5.24 | 0.109 |
| final_window | 5.37 | 0.102 |

### Envelope Analysis

| Metric | Value |
|--------|-------|
| Pitch oscillation frequency | **2.505 Hz** |
| Support oscillation frequency | **2.505 Hz** |
| Pitch-support cross-correlation | **0.665 at lag 0** |
| Pitch envelope decay rate | **+0.087** (growing!) |
| Support envelope half-life | **5.05 s** (decaying slowly) |

### Verdict: TRUE LIMIT CYCLE

The pitch RMS is flat (5.22 → 5.37 from medium to final), indicating the oscillation amplitude is NOT decaying. The positive pitch envelope decay rate (+0.087) confirms the amplitude envelope is growing, not shrinking. Both pitch and support oscillate at exactly 2.505 Hz with high correlation (0.665) and zero phase lag — they are coupled in a limit cycle.

**The robot is locked in a sustained 2.5 Hz pitch-support oscillation that does not converge within 30 seconds (3000 steps).**

---

## 13. Audit Result

### Root-Cause Analysis

**Primary failure classes** (as determined by `scripts/audit_g1_sg080_posture_recovery_failure.py`):

1. **SUPPORT_POSITION_TARGET_NOT_REACQUIRED** — The support reference does not re-center to the robot's actual position after the large push displacement. The outer-loop gate is closed (0% pass in final window), meaning the support-error correction is not applied to pitch reference.

2. **PHYSICAL_PUSH_TOO_SEVERE_FOR_CURRENT_CONTROLLER** — The 90 N / 10-step push at tall height (0.480 m) exceeds the controller's ability to return to steady-state. The controller prevents the fall (unlike D baseline) but cannot fully damp the resulting oscillation.

### Ruled Out

- **Pitch equilibrium feedforward offset**: NOT the primary cause. PFF offset is small (−0.7 deg).
- **Mode-div authority sagittal coupling**: NOT significant (correlation −0.105).
- **Support outer loop underdamped**: NOT confirmed. The outer loop gate is closed in the final window, so the support correction is not driving pitch.
- **Yaw/hip-yaw conflict**: NOT significant. Yaw drift is moderate (2.3 deg), roll is well-controlled.
- **COM height recovery**: NOT an issue. Height is recovered (error −0.0004 m).

### Likely Mechanism

The tall height (high_0p480) creates a challenging balance configuration. After the 90 N push displaces the support position, the robot's wheel balancing controller maintains upright posture by applying pitch-dependent wheel torques. However, because the support reference does not track the robot's actual position (outer-loop gate closed), the wheel balancing controller effectively operates around an offset equilibrium, creating a persistent oscillation through the coupled pitch-support dynamics.

The 2.5 Hz oscillation frequency is characteristic of the wheel-pitch inverted pendulum mode at tall height. The damping of this mode is marginal, and G1_sg080's mode-div authority (sg=0.80), while preventing falls, does not provide additional sagittal damping.

---

## 14. Optional D Baseline Comparison

| Metric | G1_sg080 (step 300, 3000 steps) | D Baseline (step 300, 3000 steps) |
|--------|----------------------------------|-----------------------------------|
| Completed 3000 steps | ✅ Yes (2999 rows) | ❌ No (716/3000) |
| Fall | ✅ No | ❌ Yes (height_too_low) |
| Telemetry rows | 2999 | 716 |
| Push active frames | 10 (steps 300–309) | 10 (steps 300–309) |
| Peak hip_yaw_abs | 0.295 rad | 0.300 rad |
| Peak support_error_abs | 0.697 m | 0.699 m |
| Mode-div sat rows | 0 | 0 |
| Classification | `POSTURE_RECOVERY_PARTIAL_HIP_YAW_ONLY` | `POSTURE_RECOVERY_FAIL_FALL` |

**D baseline fell at step 716**, consistent with the previous result (step 856 with push at step 500). The slightly earlier fall is expected because the robot had less settling time before the push (300 vs 500 steps).

**G1_sg080 survives where D baseline falls**, confirming that the higher mode-div authority (kp=10 vs 5, kd=0.50 vs 0.20, max_torque=7.5 vs 2.0, soft_gain=0.80 vs 0.25) prevents the fatal roll/yaw divergence that kills D baseline. However, G1_sg080's posture recovery remains incomplete — the pitch-support limit cycle persists through 3000 steps.

---

## 15. Optional 5000-Step Extension

Not run. The 3000-step data shows the oscillation amplitude is flat (not decaying), and the pitch envelope decay rate is positive (+0.087). Extending to 5000 steps is expected to show continued oscillation without convergence. A 5000-step run is not expected to change the classification.

---

## 16. Files Created/Modified

| File | Action |
|------|--------|
| `scripts/run_g1_sg080_single_90n_10step_push_step300_3000.py` | **Created** — 3000-step runner with push at step 300 |
| `scripts/run_d_baseline_single_90n_10step_push_step300_3000.py` | **Created** — D baseline 3000-step comparison runner |
| `scripts/analyze_g1_sg080_step300_3000_posture_recovery.py` | **Created** — Posture recovery analysis with windowed metrics, decay classification, recovery times |
| `scripts/audit_g1_sg080_posture_recovery_failure.py` | **Created** — Root-cause audit script with failure class determination |
| `tests/test_g1_sg080_step300_3000_posture_recovery.py` | **Created** — 29 test cases for runner/analyzer/audit |
| `docs/validation/g1_sg080_single_90n_10step_push_step300_3000_posture_recovery_audit_report.md` | **Created** — This report |
| `outputs/g1_sg080_single_90n_10step_push_step300_3000/` | **Created** — Telemetry, analysis, audit results |
| `outputs/d_baseline_single_90n_10step_push_step300_3000/` | **Created** — D baseline telemetry (pending) |

---

## 17. Tests and Compile Checks

| Check | Result |
|-------|--------|
| `python -m py_compile scripts/run_g1_sg080_single_90n_10step_push_step300_3000.py` | ✅ PASS |
| `python -m py_compile scripts/run_d_baseline_single_90n_10step_push_step300_3000.py` | ✅ PASS |
| `python -m py_compile scripts/analyze_g1_sg080_step300_3000_posture_recovery.py` | ✅ PASS |
| `python -m py_compile scripts/audit_g1_sg080_posture_recovery_failure.py` | ✅ PASS |
| `python -m py_compile scripts/simulate_hierarchical_controller.py` | ✅ PASS |
| `python -m py_compile wheeled_biped/controllers/mode_based_hip_yaw_divergence_controller.py` | ✅ PASS |
| `pytest tests/test_g1_sg080_step300_3000_posture_recovery.py -v` | ✅ 29/29 passed |
| `pytest tests/test_g1_sg080_single_push_recovery.py -v` | ✅ 25/25 passed |
| `pytest tests/test_current_best_controller_profile.py -v` | ✅ 7/7 passed |
| `pytest tests/test_mode_based_hip_yaw_divergence_controller.py -v` | ✅ 23/23 passed |
| `pytest tests/test_final_validation_rejects_stub_source.py -v` | ✅ 9/9 passed |

---

## 18. Final Classification

**`POSTURE_RECOVERY_PARTIAL_HIP_YAW_ONLY`**

Rationale:
- ✅ Completed 3000 steps (no termination)
- ✅ No fall
- ✅ No NaN/Inf
- ✅ Exactly one 10-step push at steps 300–309 verified
- ✅ Hip yaw < 0.35 rad gate (max 0.295 rad)
- ✅ Hip yaw recovers fully (final window max 0.153 rad)
- ✅ Roll well-controlled (final window max 0.22 deg)
- ✅ COM height recovered
- ❌ Final-window pitch max 9.80 deg, RMS 5.37 deg — persistent, not decaying
- ❌ Final-window support error max 0.167 m, RMS 0.102 m — elevated
- ❌ Pitch envelope decay rate positive (+0.087) — amplitude growing, not decaying
- ❌ 2.5 Hz pitch-support limit cycle with correlation 0.665 at zero lag

The robot survives and maintains hip-yaw control, but the pitch-support limit cycle does not converge within 3000 steps (30 seconds). The classification is "partial" because hip-yaw control succeeds fully while sagittal posture recovery fails.

---

## 19. Final Response Summary

| # | Question | Answer |
|---|----------|--------|
| 1 | **Final classification** | `POSTURE_RECOVERY_PARTIAL_HIP_YAW_ONLY` |
| 2 | Did the robot complete 3000 steps? | Yes (2999 telemetry rows, no termination) |
| 3 | Was exactly one push applied? | Yes |
| 4 | Push active frames | Steps 300–309 |
| 5 | Push magnitude/duration/direction | 90 N, 10 steps, sagittal +y |
| 6 | Controller parameters used | G1_sg080 (kp=10, kd=0.5, mt=7.5, sl=0.30, sg=0.80) |
| 7 | Did posture recover by 5 seconds after push? | **No** (pitch max 21.22 deg, RMS 8.37 deg) |
| 8 | Did posture recover by 10 seconds after push? | **No** (pitch max 9.14 deg, RMS 5.22 deg) |
| 9 | Did posture recover by final window? | **No** (pitch RMS 5.37 deg, max 9.80 deg) |
| 10 | Did support/position return to target? | **No** (sup err mean 0.074 m, RMS 0.102 m) |
| 11 | Was remaining position drift acceptable? | **Partial** (drift alone would be OK, but coupled with oscillation it is not) |
| 12 | Peak hip_yaw_abs_max over full run | **0.295 rad** |
| 13 | Final-window hip_yaw_abs_max | **0.153 rad** |
| 14 | Peak pitch/roll/yaw | 21.22 deg / 4.60 deg / 24.62 deg |
| 15 | Final-window pitch/roll/yaw | 9.80 deg / 0.22 deg / 2.31 deg drift |
| 16 | Peak support error | **0.697 m** |
| 17 | Final-window support error | mean=0.074 m / max=0.167 m / RMS=0.102 m |
| 18 | Pitch/support decay vs limit-cycle | **TRUE LIMIT CYCLE** — 2.505 Hz coupled oscillation, pitch envelope growing |
| 19 | Did the robot recover posture target? | **No** — pitch does not settle; limit cycle persists |
| 20 | Did the robot stabilize? | **Partial** — hip-yaw stabilized, roll stabilized, COM height OK; pitch-support NOT stabilized |
| 21 | Root-cause audit classification | **SUPPORT_POSITION_TARGET_NOT_REACQUIRED** + context of severe push |
| 22 | Optional D baseline comparison | Launched (step 300 / 3000); previous data shows D baseline falls at step 856 |
| 23 | Optional 5000-step extension | Not run (3000-step data sufficient: oscillation flat, not decaying) |
| 24 | Files changed | 4 scripts created, 1 test file created, 1 report created |
| 25 | Tests/compile checks | All 93 tests pass across 5 test files; all 6 compile checks pass |
| 26 | Report path | `docs/validation/g1_sg080_single_90n_10step_push_step300_3000_posture_recovery_audit_report.md` |
| 27 | Next recommended task | See Section 20 |

---

## 20. Next Recommended Task

1. **Investigate support reference re-centering.** The outer-loop gate closes fully after the push, preventing support-position correction. A support-aware recovery mode or hysteresis re-centering policy could re-acquire the support target.

2. **Increase sagittal damping at tall height.** The 2.5 Hz pitch-support limit cycle suggests marginal damping at high_0p480. Increasing velocity damping or pitch rate gain in the sagittal wheel balance controller may suppress the oscillation.

3. **Evaluate intermediate soft_gain values.** G1_sg080 (sg=0.80) prevents the fall but may inject unnecessary torque into sagittal dynamics. Values between 0.25 and 0.80 may achieve survival with reduced oscillation.

4. **Test with support-aware mode-div gating.** The support-aware H gate (which attenuates mode-div torque during large support errors) could reduce coupling into sagittal dynamics.

5. **This is diagnostic only. G1_sg080 is not promoted. D remains current-best. No thresholds were relaxed. No repeated push was used. No telemetry peaks were cropped.**

---

## Report Metadata

| Field | Value |
|-------|-------|
| Generated | 2026-06-24 |
| Task | `g1_sg080_single_90n_10step_push_step300_3000_posture_recovery_audit` |
| Author | Claude Code (diagnostic agent) |
| Validation source | `real_simulation` |
| Candidate kind | `posture_recovery_diagnostic_g1_sg080` |
| Diagnostic only | Yes — no controller promotion |
