# G1_sg080 Single 90N / 10-Step Push Recovery Diagnostic Report

## 1. Executive Summary

A single sagittal push of **90 N for 10 steps** was applied at **step 500** to the high_0p480 (tall) height variant under **G1_sg080** controller parameters (kp=10.0, kd=0.50, max_torque=7.5, soft_limit=0.30, soft_gain=0.80). The simulation ran for **2000 steps** (20.0 s simulated).

**G1_sg080 survived the full 2000 steps without falling.** Hip yaw remained below the 0.35 rad gate threshold for the entire run (max 0.291 rad). The mode-based hip-yaw divergence controller applied up to 4.02 Nm without saturating.

**However, the robot exhibited persistent pitch oscillation and elevated support-position error in the final 500 steps:** pitch max 9.70° (mean 4.29°), support error max 0.176 m (mean 0.096 m). While the robot did not fall, it did not achieve steady-state recovery within the 2000-step horizon.

**D baseline comparison:** Under identical push conditions, **D baseline fell at step 856** (height_too_low), terminating early with pitch/roll divergence. G1_sg080's higher mode-div authority (kp=10 vs 5, kd=0.50 vs 0.20, max_torque=7.5 vs 2.0, soft_gain=0.80 vs 0.25) prevents the fall but does not eliminate residual pitch-support oscillation.

---

## 2. Scenario Definition

| Parameter | Value |
|-----------|-------|
| Case ID | `G1_sg080_single_90n_10step_push_high_2000` |
| Controller profile | `G1_sg080` (D_MODE_HIP_YAW_DIV_V1 base + sg=0.80) |
| Height variant | `high_0p480` (tall) |
| Total steps | 2000 |
| Push magnitude | 90 N |
| Push duration | 10 steps |
| Push count | 1 |
| Push start step | 500 |
| Push direction | Sagittal +y (forward) |
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
  --steps 2000 \
  --telemetry-decimation 1 \
  --failure-window-steps 2000 \
  --write-run-summary-sidecar \
  --output-dir outputs/g1_sg080_single_90n_10step_push_recovery_2000 \
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
  --push-start-step 500 \
  --sagittal-push-only
```

---

## 4. Push Verification

| Check | Actual | Expected | Result |
|-------|--------|----------|--------|
| Push windows | 1 | 1 | ✅ |
| Push active frames | 10 | 10 | ✅ |
| Push start step | 500 | 500 | ✅ |
| Push end step | 510 | 510 | ✅ |
| Push force direction | +y forward | +y forward | ✅ |

---

## 5. Controller Parameters

| Parameter | G1_sg080 Value | D Baseline Value |
|-----------|-----------------|------------------|
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

| Metric | G1_sg080 | D Baseline |
|--------|----------|------------|
| hip_yaw_abs_max (full run) | 0.291 rad | 0.313 rad |
| hip_yaw_abs_max (during push) | 0.016 rad | — |
| hip_yaw_abs_max (after push) | 0.291 rad | — |
| support_error_abs_max | 0.699 m | 0.698 m |
| support_p2p | 0.699 m | — |
| pitch_abs_max | 21.2° | 21.2° |
| roll_abs_max | 4.47° | 19.5° |
| yaw_abs_max | 0.419 rad | 0.362 rad |

**Key:** G1_sg080 maintains hip yaw below 0.35 rad gate (0.291 rad peak). D baseline hip yaw (0.313 rad) also stays nominally below gate but the robot falls due to roll divergence (19.5° vs G1's 4.5°). G1_sg080's higher mode-div authority helps control roll but pitch-support oscillation persists.

---

## 7. Recovery Metrics

| Metric | Steps to Recover | Notes |
|--------|-----------------|-------|
| Pitch < 5° | 253 steps after push end | ~2.5 s after push end |
| Support < 0.05 m | **Never** | Remains elevated |
| Support < 0.10 m | **Never** | Max final500 = 0.176 m |
| Roll < 5° | Immediately | Well-controlled |
| Hip yaw < 0.35 rad | N/A (never exceeded) | Max 0.291 rad |

---

## 8. Final 500-Step Stability Metrics

| Metric | Value |
|--------|-------|
| Window | Steps 1499–2000 |
| hip_yaw_abs_max | 0.198 rad |
| hip_yaw_abs_mean | 0.175 rad |
| sup_err_abs_max | 0.176 m |
| sup_err_abs_mean | 0.096 m |
| sup_err_abs_rms | 0.109 m |
| pitch_abs_max | 9.70° |
| pitch_abs_mean | 4.29° |
| pitch_rms | 5.09° |
| roll_abs_max | 0.23° |
| roll_abs_mean | 0.11° |
| yaw_drift | 0.011 rad |
| hip_yaw_divergence_error_abs_max | 0.392 rad |
| COM z drift | 0.004 m |

**Interpretation:**
- **Hip yaw is stable** (max 0.198 rad, well below 0.35 gate).
- **Support error is elevated** (mean 0.096 m, max 0.176 m). The robot oscillates due to pitch-support coupling.
- **Pitch oscillates** (mean 4.3°, max 9.7°) — never fully settles.
- **Roll is excellent** (max 0.23°).
- The robot is **alive but not fully stabilized** within 2000 steps.

---

## 9. Hip-Yaw Gate Analysis

| Period | hip_yaw_abs_max | Gate (0.35 rad) | Status |
|--------|-----------------|-----------------|--------|
| Full run | 0.291 rad | <!-- 0.35 rad --> | ✅ Below |
| During push | 0.016 rad | <!-- 0.35 rad --> | ✅ Far below |
| After push | 0.291 rad | <!-- 0.35 rad --> | ✅ Below |
| Final 500 steps | 0.198 rad | <!-- 0.35 rad --> | ✅ Below |

**Hip_yaw never exceeded 0.35 rad at any point during the entire 2000-step simulation.** The mode-based hip-yaw divergence controller operated within its authority envelope without saturation.

---

## 10. Support/Pitch/Roll/Yaw Analysis

### Support error
- Peak: 0.699 m (instantaneous at push onset, as robot is displaced forward)
- Drops to ~0.10 m range but **oscillates** rather than converging
- Final 500 mean: 0.096 m (close to the 0.10 m threshold but max of 0.176 m intermittently exceeds it)

### Pitch
- Peak: 21.2° (push response)
- Recovers to <5° after ~253 steps (~2.5 s)
- However, pitch continues oscillating around ~4° in the final window
- The oscillation amplitude grows and shrinks, suggesting marginal damping

### Roll
- Peak: 4.47° (brief)
- Recovers immediately (residual <0.3°)
- Well-controlled: D baseline reached 19.5° roll and fell

### Yaw
- Peak: 0.419 rad (body yaw drift from equilibrium)
- Final drift: 0.011 rad over final 500 steps

---

## 11. Mode-Div Torque / Saturation Analysis

| Metric | Value |
|--------|-------|
| tau_left_raw_max | 4.02 Nm |
| tau_right_raw_max | 2.70 Nm |
| tau_left_clipped_max | 4.02 Nm |
| tau_right_clipped_max | 2.70 Nm |
| Saturation rows | 0 |
| Max torque limit | 7.5 Nm per side |

No saturation occurred. The mode-div controller operated at 54% of its max torque (4.02/7.5 Nm). The soft gain of 0.80 provides strong attenuation when height is within the soft-limit band, but the divergence damping at kp=10/kd=0.50 was sufficient to keep hip_yaw bounded.

---

## 12. D Baseline Comparison

| Metric | G1_sg080 | D Baseline |
|--------|----------|------------|
| Completed 2000 steps | ✅ Yes | ❌ No (856/2000) |
| Fall | ✅ No | ❌ Yes (height_too_low) |
| Peak hip_yaw | 0.291 rad | 0.313 rad |
| Peak roll | 4.5° | 19.5° |
| Support error after push | Oscillating 0.10 m | Diverged → fall |
| Mode-div sat rows | 0 | 0 |
| Classification | FAIL_SUPPORT | FAIL_FALL |

**D baseline cannot survive a single 90 N / 10-step push at high_0p480.** It falls at step 856 due to height_too_low termination. G1_sg080's higher mode-div authority (kp=10, kd=0.50, max_torque=7.5, soft_gain=0.80) provides sufficient corrective torque to keep the robot alive but does not fully eliminate residual pitch-support oscillation.

---

## 13. Did the robot recover?

**Partially.** The robot survived 2000 steps without falling, which is a necessary condition for recovery. Hip yaw stayed well below the gate threshold. However, the pitch-support oscillation persists, with support error reaching 0.176 m and pitch reaching 9.7° even in the final 500 steps. The robot is alive but not stabilized to steady-state within the 2000-step window.

Under a looser definition of recovery ("alive at end of simulation"), the answer would be **yes, mostly**. Under a strict definition ("converged to baseline steady state"), the answer is **no** — the oscillation has not damped out.

---

## 14. Did the robot stabilize?

**No.** The final 500 steps show persistent oscillation:
- Support error mean 0.096 m, max 0.176 m (above 0.10 m threshold)
- Pitch mean 4.3°, max 9.7° (above 5° threshold for significant portion)
- Support error standard deviation in last 100 steps: 0.058 m (oscillating, not converged)

This suggests the low-band support position controller and the pitch equilibrium loop do not have sufficient damping to fully settle the robot after a large push at the tall height.

---

## 15. Did hip yaw stay below 0.35 rad or only recover later?

Hip yaw **never exceeded 0.35 rad at any point** during the entire 2000-step simulation. The max value was 0.291 rad, which is 83% of the gate threshold. This is a positive result.

---

## 16. D Baseline Comparison

See Section 12. D baseline cannot survive the 90 N / 10-step push.

---

## 17. Files Created/Modified

| File | Action |
|------|--------|
| `scripts/simulate_hierarchical_controller.py` | **Modified** — added `--push-count`, `--push-start-step`, `--sagittal-push-only` CLI flags |
| `scripts/run_g1_sg080_single_90n_10step_push_recovery.py` | **Created** — G1_sg080 single-push diagnostic runner |
| `scripts/run_d_baseline_single_90n_10step_push_recovery.py` | **Created** — D baseline single-push diagnostic runner |
| `scripts/analyze_g1_sg080_single_push_recovery.py` | **Created** — analysis script with full metrics pipeline |
| `tests/test_g1_sg080_single_push_recovery.py` | **Created** — 25 test cases for runner/analyzer |
| `docs/validation/g1_sg080_single_90n_10step_push_recovery_2000_report.md` | **Created** — this report |
| `outputs/g1_sg080_single_90n_10step_push_recovery_2000/` | **Created** — telemetry, summary, analysis results |
| `outputs/d_baseline_single_90n_10step_push_recovery_2000/` | **Created** — D baseline telemetry, summary |

---

## 18. Tests and Compile Checks

| Check | Result |
|-------|--------|
| `python -m py_compile scripts/run_g1_sg080_single_90n_10step_push_recovery.py` | ✅ PASS |
| `python -m py_compile scripts/run_d_baseline_single_90n_10step_push_recovery.py` | ✅ PASS |
| `python -m py_compile scripts/analyze_g1_sg080_single_push_recovery.py` | ✅ PASS |
| `python -m py_compile scripts/simulate_hierarchical_controller.py` | ✅ PASS |
| `python -m py_compile wheeled_biped/controllers/mode_based_hip_yaw_divergence_controller.py` | ✅ PASS |
| `pytest tests/test_g1_sg080_single_push_recovery.py -v` | ✅ 25/25 passed |
| `pytest tests/test_current_best_controller_profile.py -v` | ✅ 7/7 passed |
| `pytest tests/test_mode_based_hip_yaw_divergence_controller.py -v` | ✅ 23/23 passed |
| `pytest tests/test_final_validation_rejects_stub_source.py -v` | ✅ 9/9 passed |

---

## 19. Final Classification

**`SINGLE_PUSH_RECOVERY_FAIL_SUPPORT`**

Rationale:
- ✅ Completed 2000 steps
- ✅ No fall
- ✅ No NaN/Inf
- ✅ Exactly one 10-step push verified
- ✅ Hip yaw < 0.35 rad gate (max 0.291 rad)
- ❌ Final 500 support error max 0.176 m > 0.10 m threshold
- ❌ Persistent pitch oscillation (final 500 pitch max 9.7°, mean 4.3°)

The robot survives and does not fall, but the support position error and pitch oscillation do not converge to steady-state within the 2000-step horizon. The support error mean (0.096 m) is borderline — close to the 0.10 m threshold but the maximum excursions (0.176 m) and persistent oscillation prevent a PASS classification.

---

## 20. Next Recommendations

1. **Investigate pitch-support coupling at high height.** The persistent oscillation (pitch ~4° mean, support ~0.10 m mean) suggests the low-band support correction interacts with the physics-equilibrium feedforward pitch reference. A damping increase in the support outer loop (kd) may help.

2. **Extend simulation to 5000+ steps.** The oscillation amplitude at 2000 steps has not shown clear convergence. A longer horizon would determine whether it eventually damps or is a limit cycle.

3. **Test intermediate soft_gain values.** G1_sg080 (=0.80) prevents the fall that occurs at D baseline (=0.25). Values between 0.25 and 0.80 may achieve survival with less oscillation.

4. **Evaluate support-aware mode-div gating.** If the pitch-support oscillation continues, enabling the support-aware H gate (which attenuates mode-div torque during large support errors) could interact differently.

5. **This is diagnostic only. G1_sg080 is not promoted. D remains current-best. No thresholds were relaxed. No telemetry peaks were cropped.**

---

## Report Metadata

| Field | Value |
|-------|-------|
| Generated | 2026-06-23 |
| Task | `g1_sg080_single_90n_10step_push_recovery_2000` |
| Author | Claude Code (diagnostic agent) |
| Validation source | `real_simulation` |
| Candidate kind | `single_push_diagnostic_g1_sg080` |
| Diagnostic only | Yes — no controller promotion |
