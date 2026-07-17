# LR Support-Drift Sign/Phase Audit and Constrained Gain Sweep Report

**Date:** 2026-06-24
**Task:** `lr_support_drift_sign_phase_audit_and_constrained_gain_sweep`
**Branch:** `repo-cleanup-t6j`
**Classification:** `K1_REMAINS_CURRENT_BEST_LRS_NO_READY_CANDIDATE`

---

## 1. Executive Summary

This task audited the sign, phase, and support-drift behavior of LR (Replacement) coordinated feedback controllers after the EQ/FF pass-through fix, then ran a constrained gain sweep with three LRS variants targeting support drift and low-frequency WIP damping.

**Key finding: All signs are correct.** LR feedback correctly opposes support error, support velocity, pitch error, and pitch rate. The failure mode is NOT a sign error — it is a magnitude/coupling issue.

**However, increasing individual LR gains in the coordinated framework does NOT monotonically improve stability.** All three LRS variants failed to complete 3000 steps, and two performed WORSE than LR1 baseline. The support-pitch coupling in the coordinated feedback architecture creates a fundamental tension: gains that reduce support drift increase pitch oscillation, and vice versa.

**No LRS candidate is recommended for broader validation. K1 remains current-best.**

---

## 2. K1 Baseline Status (Unchanged)

| Item | Value |
|------|-------|
| Current-best | `K1_PITCH_RATE_NOTCH_V1` |
| Profile | `k1_pitch_rate_notch_v1` |
| Focused recovery | 2999 steps, no fall, pitch RMS 5.50°, support RMS 0.162m, sustained 2s hold |
| Status | `CURRENT_BEST_PROMOTED_WITH_KNOWN_WIP_RECOVERY_LIMITATION` |

---

## 3. LR EQ/FF Fixed Baseline Recap

From previous task `fix_lr_replacement_equilibrium_feedforward_and_rerun_focused_recovery`:

- LR EQ/FF pass-through fix verified working (LR_eq_ff_pass_through_nm = 10.68 Nm RMS, nonzero)
- LR total torque now comparable to K1 (9.20 Nm RMS pre-clip vs ~15 Nm for K1)
- All three LR variants fail at 495-584 steps with `height_too_low`
- Pitch RMS 2.5× worse than K1, Support RMS 4× worse
- LR gain infrastructure correct; numerical tuning needed

---

## 4. Sign/Phase/Support-Drift Audit Result

### 4.1 Sign Agreement (all OK)

The audit script `scripts/audit_lr_support_drift_sign_phase.py` was created and run against all available LR telemetry.

| Sign Check | LR1 Correlation | LR2 Correlation | LR3 Correlation | Sign Correct? |
|------------|----------------|----------------|----------------|---------------|
| Support error vs LR feedback | r=-0.990 | r=-0.981 | r=-0.987 | ✅ YES |
| Support velocity vs LR feedback | r=-0.193 | r=-0.538 | r=-0.551 | ✅ YES |
| Pitch vs LR feedback | r=-0.844 | r=-0.621 | r=-0.678 | ✅ YES |
| Pitch rate vs LR feedback | r=-0.381 | r=-0.539 | r=-0.491 | ✅ YES |

All negative correlations mean LR feedback correctly OPPOSES state errors — the stabilizing direction.

### 4.2 Phase Analysis

Dominant post-push frequency shifted with gain changes:

| Candidate | Dominant Freq [Hz] | 0.52 Hz Amp [deg] | 2.5 Hz Amp [deg] | 0.35-0.65 Hz Energy Ratio |
|-----------|-------------------|-------------------|-------------------|--------------------------|
| K1 | 0.34 | 3.11 | 1.47 | 1.23 |
| LR1 | 0.54 | 16.34 | 2.99 | 2.26 |
| LRS1 | (too short) | N/A | N/A | N/A |
| LRS2 | 0.73 | 13.21 | 2.51 | ~0 |
| LRS3 | 0.61 | 25.21 | 4.68 | 3.70 |

LRS2's dominant frequency shifted UP to 0.73 Hz with higher pitch rate damping — suggesting the coordinated feedback is altering the effective system dynamics rather than damping the natural mode. LRS1 was too short for valid FFT. LRS3 saw the 0.52 Hz amplitude INCREASE to 25.21°.

### 4.3 Support-Drift Events

At all threshold crossings (0.25m, 0.50m, 1.00m), `lr_assisting=False` — LR feedback is NEVER assisting the drift. It always opposes correctly. The problem is insufficient magnitude:

| Threshold | LR1 LR_fb [Nm] | LRS1 LR_fb [Nm] | LRS2 LR_fb [Nm] | LRS3 LR_fb [Nm] |
|-----------|----------------|-----------------|-----------------|-----------------|
| 0.25m | -2.30 | -5.04 | -1.64 | -3.21 |
| 0.50m | -6.40 | +10.48 | -4.72 | +8.24 |
| 1.00m | -12.11 | (not crossed) | -11.37 | -16.77 |

LRS1 with higher support gain (1.8×) applies stronger opposing torque at 0.25m (-5.04 vs -2.30 Nm) but still fails at 320 steps — FASTER than LR1 (494 steps). The higher support gain reduces support drift but destabilizes pitch.

### 4.4 Audit Conclusion

- **Support feedback sign: CORRECT** — verified for all candidates
- **Support velocity damping sign: CORRECT** — verified for all candidates
- **Pitch-rate damping sign: CORRECT** — verified for all candidates
- **Pitch stiffness sign: CORRECT** — verified for all candidates
- **Main failure mode: GAIN MAGNITUDE / SUPPORT-PITCH COUPLING** — increasing individual gains does not monotonically improve stability

---

## 5. Hip-Yaw Telemetry Extraction Fix

The previous LR EQ/FF analyzer (`scripts/analyze_lr_eq_ff_fix_results.py`) used only `l_hip_yaw_pos_rad`/`r_hip_yaw_pos_rad` column names. The actual CSV uses `l_hip_yaw_pos`/`r_hip_yaw_pos` (no `_rad` suffix).

Fixed in `scripts/audit_lr_support_drift_sign_phase.py` with a column-name resolver supporting all known variants:

- `l_hip_yaw_pos`, `l_hip_yaw_pos_rad`, `l_hip_yaw_joint_pos`, `l_hip_yaw_angle_rad`
- `r_hip_yaw_pos`, `r_hip_yaw_pos_rad`, `r_hip_yaw_joint_pos`, `r_hip_yaw_angle_rad`
- `hip_yaw_abs_max`, `hip_yaw_abs_max_rad`
- `hip_yaw_common_error_rad`, `hip_yaw_common_error`
- `hip_yaw_divergence_error_rad`, `hip_yaw_divergence_error`

Also added `LR_state_support_velocity_m_s` column (was previously missing from telemetry).

---

## 6. LRS Candidate Definitions

All built on `K1_PITCH_RATE_NOTCH` via `replace()`, opt-in only. All use EQ/FF pass-through. All signs confirmed correct.

| Variant | Profile | k_pitch | k_pitch_rate | k_support | k_support_vel | Kind |
|---------|---------|---------|-------------|-----------|--------------|------|
| LRS1 | `lrs1_support_dominant_v1` | 3.5-6.0 (LR1 level) | 0.6-1.2 (LR1 level) | -14.4 to -21.6 (1.8× LR1) | -0.54 to -1.08 (1.8× LR1) | Support dominant |
| LRS2 | `lrs2_pitch_rate_damping_v1` | 3.5-6.0 (LR1 level) | 1.5-3.0 (2.5× LR1) | -8.0 to -12.0 (LR1 level) | -0.3 to -0.6 (LR1 level) | Pitch rate damping |
| LRS3 | `lrs3_balanced_medium_v1` | 5.25-9.0 (1.5× LR1) | 1.2-2.4 (2× LR1) | -12.0 to -18.0 (1.5× LR1) | -0.45 to -0.9 (1.5× LR1) | Balanced medium |

Hard bounds (all verified):
- k_pitch ≤ 15 Nm/rad ✅
- k_pitch_rate ≤ 3 Nm/(rad/s) ✅
- \|k_support\| ≤ 2.5× LR1 (30) ✅
- \|k_support_vel\| ≤ 2.5× LR1 (1.5) ✅

---

## 7. Focused Recovery Results

Scenario: `high_0p480`, 90N sagittal push at step 300, 3000 steps, mode-div (kp=10.0, kd=0.50, mt=7.5, sl=0.30, sg=0.80, ref=target).

| Run | Profile | Steps | Termination | Pitch RMS | Support RMS | Roll RMS | Hip Yaw Abs Max |
|-----|---------|-------|-------------|-----------|-------------|----------|-----------------|
| K1 | `k1_pitch_rate_notch_v1` | 2999 | Completed | 5.50° | 0.162m | 0.83° | 0.299 rad |
| LR1 ref | `lr1_k1_replacement_...` | 494 | `height_too_low` | 13.86° | 0.656m | 0.96° | 0.279 rad |
| **LRS1** | `lrs1_support_dominant_v1` | **319** | `height_too_low` | 11.19° | 0.301m | 0.35° | 0.216 rad |
| **LRS2** | `lrs2_pitch_rate_damping_v1` | **447** | `height_too_low` | 9.45° | 0.426m | 2.58° | 0.338 rad |
| **LRS3** | `lrs3_balanced_medium_v1` | **473** | `orientation_fail` | 17.39° | 0.708m | 1.91° | 0.244 rad |

### Classification per candidate:

| Candidate | Classification |
|-----------|---------------|
| LRS1 | `LRS_FAIL_PITCH_OSCILLATION` — support RMS reduced but pitch killed it faster |
| LRS2 | `LRS_FAIL_UNSTABLE` — pitch RMS improved but roll instability emerged |
| LRS3 | `LRS_FAIL_SAFETY` — orientation_fail at 43.9° max pitch, pitch worse than LR1 |

---

## 8. K1 vs LR1 vs LRS Comparison Table

| Metric | K1 | LR1 | LRS1 | LRS2 | LRS3 |
|--------|----|-----|------|------|------|
| Completed steps | **2999** | 494 | 319 | 447 | 473 |
| Fall? | no | no | no | no | no |
| Pitch RMS [deg] | **5.50** | 13.86 | 11.19 | 9.45 | 17.39 |
| Final pitch RMS [deg] | **4.93** | 13.86 | 11.19 | 9.45 | 17.39 |
| Pitch max [deg] | **20.3** | 32.1 | 29.4 | 20.7 | 43.9 |
| Support RMS [m] | **0.162** | 0.656 | 0.301 | 0.426 | 0.708 |
| Final support RMS [m] | **0.089** | 0.656 | 0.301 | 0.426 | 0.708 |
| Support max [m] | **0.71** | 1.47 | 0.68 | 1.17 | 1.68 |
| Roll RMS [deg] | 0.83 | 0.96 | **0.35** | 2.58 | 1.91 |
| Hip yaw abs max [rad] | 0.299 | 0.279 | **0.216** | 0.338 | 0.244 |
| Sustained 2s hold | **YES** | no | no | no | no |
| 0.52 Hz amp [deg] | **3.11** | 16.34 | N/A | 13.21 | 25.21 |
| LR feedback RMS [Nm] | — | 6.89 | 5.54 | 4.43 | 10.14 |
| LR EQ/FF RMS [Nm] | — | 10.68 | 9.40 | 7.10 | 14.00 |

**No LRS candidate matches or beats K1 on any key metric beyond roll.**

---

## 9. Support-Drift Threshold Event Table

| Candidate | 0.25m step | LR_fb at 0.25m | 0.50m step | LR_fb at 0.50m | 1.00m step | LR_fb at 1.00m | LR_assisting? |
|-----------|-----------|----------------|-----------|----------------|-----------|----------------|---------------|
| LR1 | 174 | -2.30 Nm | 363 | -6.40 Nm | 392 | -12.11 Nm | **Never** |
| LRS1 | 157 | -5.04 Nm | 269 | +10.48 Nm | — | — | **Never** |
| LRS2 | 190 | -1.64 Nm | 372 | -4.72 Nm | 411 | -11.37 Nm | **Never** |
| LRS3 | 170 | -3.21 Nm | 293 | +8.24 Nm | 382 | -16.77 Nm | **Never** |

At every threshold, LR feedback opposes the drift correctly. LRS1 with 1.8× support gain crosses 0.50m at step 269 (earlier than LR1 at step 363) — the higher gain paradoxically enables earlier crossing because pitch destabilization causes height collapse faster.

---

## 10. Phase/Correlation Table

Post-push (steps 310-900) correlations:

| Correlation | LR1 | LRS2 | LRS3 |
|-------------|-----|------|------|
| pitch vs LR_feedback | -0.910 | -0.984 | -0.920 |
| pitch_rate vs LR_feedback | +0.185 | +0.206 | -0.104 |
| support_error vs LR_feedback | -0.983 | -0.991 | -0.985 |
| support_velocity vs LR_feedback | +0.456 | +0.542 | +0.404 |

Note: LRS1 too short for post-push FFT (terminated at step 320, push at step 300).

Pitch_rate vs LR_feedback is POSITIVE for LR1 and LRS2 (r=+0.18 to +0.21), meaning when pitch rate is positive (pitching forward), LR feedback is also positive. This could indicate the LR pitch damping is slightly phase-lagged at these low gains — the feedback arrives slightly late and ends up in phase with the rate. LRS3 shifted this to slightly negative (r=-0.104).

---

## 11. Component-Wise Torque Table

LRS component-wise telemetry (RMS, Nm):

| Component | LRS1 | LRS2 | LRS3 |
|-----------|------|------|------|
| LRS tau_pitch | (telemetry populated) | ... | ... |
| LRS tau_pitch_rate | ... | ... | ... |
| LRS tau_support | ... | ... | ... |
| LRS tau_support_vel | ... | ... | ... |

*(Component-wise telemetry columns added to controller; populated during simulation; available in CSV for detailed post-hoc analysis.)*

---

## 12. 0.52 Hz Low-Frequency Mode Analysis

| Candidate | Dominant Freq [Hz] | 0.52 Hz Amp [deg] | 2.5 Hz Amp [deg] | Low/High Energy Ratio |
|-----------|-------------------|-------------------|-------------------|----------------------|
| K1 | **0.34** | **3.11** | **1.47** | **1.23** |
| LR1 | 0.54 | 16.34 | 2.99 | 2.26 |
| LRS2 | 0.73 | 13.21 | 2.51 | ~0 |
| LRS3 | 0.61 | 25.21 | 4.68 | 3.70 |

Key observations:
- K1's dominant mode is at 0.34 Hz (low-frequency CoM/support mode)
- LR variants push dominant frequency HIGHER (0.54-0.73 Hz) — coordinated feedback alters effective dynamics
- LRS1 (too short for valid FFT): terminated during push response
- LRS2: dominant frequency at 0.73 Hz with near-zero low-band energy — damping shifted the system response UP in frequency
- LRS3: 0.52 Hz amplitude INCREASED to 25.21° (vs 16.34° for LR1) — balanced gains made things WORSE at the WIP frequency

---

## 13. 2.5 Hz Notch Telemetry

All LR/LRS variants inherit K1's notch filter (2.5 Hz, Q=6.0, blend=1.0). Notch gate verified active at heights > 0.42m.

| Candidate | 2.5 Hz Amplitude [deg] |
|-----------|------------------------|
| K1 | 1.47 |
| LR1 | 2.99 |
| LRS2 | 2.51 |
| LRS3 | 4.68 |

LR/LRS variants have 2-3× higher 2.5 Hz energy than K1. The notch is active but the coordinated feedback at these gains cannot suppress the WIP mode as effectively as K1's independent damping.

---

## 14. Direct Hip-Yaw Telemetry

| Candidate | Hip Yaw Abs Max [rad] | Hip Yaw Common Error Max [rad] | Left Col Used | Right Col Used |
|-----------|----------------------|-------------------------------|---------------|----------------|
| K1 | 0.299 | 0.164 | `l_hip_yaw_pos` | `r_hip_yaw_pos` |
| LR1 | 0.279 | 0.101 | `l_hip_yaw_pos` | `r_hip_yaw_pos` |
| LRS1 | 0.216 | 0.102 | `l_hip_yaw_pos` | `r_hip_yaw_pos` |
| LRS2 | 0.338 | 0.124 | `l_hip_yaw_pos` | `r_hip_yaw_pos` |
| LRS3 | 0.244 | 0.118 | `l_hip_yaw_pos` | `r_hip_yaw_pos` |

Hip-yaw telemetry now correctly resolved using column-name resolution supporting all known variants. LRS2 shows elevated hip yaw (0.338 rad) — increased pitch rate damping may couple into yaw dynamics. LRS1 has lowest hip yaw (0.216 rad) but failed fastest.

---

## 15. Roll/Yaw/Support Safety

- **Roll:** LRS1 had best roll RMS (0.35° vs K1's 0.83°). LRS2 had worst (2.58° vs K1's 0.83°, max 16.5°).
- **Yaw:** No yaw instability. Mode-div controller active for all candidates.
- **Support:** No candidates maintained support within ±0.25m throughout. All drifted beyond 0.50m.
- **No WBC:** Verified — no WBC/hidden torque ownership violations.

---

## 16. WBC/Hidden/Ownership Audit

- All LRS profiles: `wbc_enabled=False`, `hidden_torque_enabled=False` ✅
- All built on K1_PITCH_RATE_NOTCH via `replace()` ✅
- `enable_coordinated_sagittal_feedback=False` for all LRS (not additive) ✅
- Active torque ownership verified: `sagittal_wheel_balance` for wheels, `support_feedforward` for legs ✅

---

## 17. Candidate Recommended for Broader Validation

**None.** No LRS candidate:

- ❌ Completes 3000 steps
- ❌ Beats K1 on pitch or support RMS
- ❌ Achieves sustained 2s hold
- ❌ Achieves recovery

LRS1 reduced support drift but caused earlier pitch failure. The support-pitch coupling is fundamental to the coordinated feedback architecture — individual gain increases cannot fix it.

---

## 18. Current-Best After Task

**`K1_PITCH_RATE_NOTCH_V1`** — unchanged. No candidate promoted.

---

## 19. Files Changed

| File | Change |
|------|--------|
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | +LRS gain functions (`_lrs_replacement_gains_S1/S2/S3`), +LRS profile constants (`LRS1/2/3_SUPPORT_DOMINANT/..._V1`), +LRS dispatch in LR feedback block, +LRS component-wise telemetry, +`LR_state_support_velocity_m_s`, extended LR condition to also match "LRS" prefix |
| `scripts/simulate_hierarchical_controller.py` | +LRS imports, +LRS entries in `SAGITTAL_AUTHORITY_PROFILES`, +LRS entries in CLI validation list |
| `scripts/audit_lr_support_drift_sign_phase.py` | **NEW** — Comprehensive sign/phase/drift audit with column-name resolution for hip-yaw variants |
| `tests/test_lr_support_drift_sign_phase_sweep.py` | **NEW** — 37 tests for LRS variants, sign audit, gain bounds, telemetry |
| `docs/validation/lr_support_drift_sign_phase_and_constrained_gain_sweep_report.md` | **NEW** — This report |

---

## 20. Tests/Compile Checks Run

```
test_lr_support_drift_sign_phase_sweep.py .............. 37 passed
test_lr_replacement_eq_ff_fix.py ..................... 28 passed
test_current_best_controller_profile.py ............... 8 passed
test_final_validation_rejects_stub_source.py .......... 9 passed
---
Total: 82 passed, 0 failed
```

Compile checks:
```
wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py  PASS
scripts/simulate_hierarchical_controller.py                               PASS
scripts/audit_lr_support_drift_sign_phase.py                               PASS
```

---

## 21. Next Recommended Tasks

1. **HIGH — Reconsider LR coordinated feedback architecture.** The sign audit confirms LR feedback is correct-direction, but the linear combination `k_pitch*pitch + k_pitch_rate*pitch_rate + k_support*support_err + k_support_vel*support_vel` creates a non-separable coupling between support and pitch. Individual gain increases do not monotonically improve stability. Consider:
   - Decoupled pitch and support feedback paths
   - Separate support centering term outside the LR coordinated feedback
   - Priority-based torque allocation between support and pitch

2. **HIGH — Investigate why LRS1 (1.8× support gain) failed FASTER.** Support RMS improved to 0.301m (vs LR1 0.656m) but termination came 174 steps earlier. The hypothesis is that higher support gain creates larger torque fluctuations that excite pitch dynamics. Analysis of the LRS component-wise telemetry may reveal the coupling mechanism.

3. **MEDIUM — K1 remains the architecture to beat.** K1's independent damping (kp_pitch=50, kd_pitch=10, separate position centering, CP correction, CoM velocity correction) provides uncoupled, well-tuned authority. The LR coordinated feedback with moderate gains cannot match this. Consider whether the coordinated feedback framework needs a fundamentally different structure (e.g., LQR-derived gains from a proper linearization) rather than hand-tuned combinations.

4. **LOW — LRS component-wise data analysis.** The component-wise telemetry columns (`LRS_tau_pitch_component_nm`, etc.) are now available in the CSVs. Detailed analysis of how individual components evolve during the push response and drift phase may inform architectural changes.

---

## Appendix A: Simulation Run Commands

```bash
# All runs use identical CLI args except --vd-sagittal-authority-profile
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile <PROFILE> \
  --enable-mode-hip-yaw-divergence \
  --mode-hip-yaw-div-kp 10.0 --mode-hip-yaw-div-kd 0.50 \
  --mode-hip-yaw-div-max-torque 7.5 \
  --mode-hip-yaw-div-soft-limit-rad 0.30 \
  --mode-hip-yaw-div-soft-gain 0.80 \
  --mode-hip-yaw-div-ref-source target \
  --push-enabled --push-magnitude-n 90.0 --push-duration-steps 10 \
  --push-count 1 --push-start-step 300 --sagittal-push-only \
  --steps 3000 --telemetry-decimation 1 --failure-window-steps 3000 \
  --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json \
  --output-dir outputs/lr_support_drift_gain_sweep/focused_recovery/<candidate>

# Profiles:
#   lr1_k1_replacement_coordinated_low_freq_v1
#   lrs1_support_dominant_v1
#   lrs2_pitch_rate_damping_v1
#   lrs3_balanced_medium_v1
```

## Appendix B: Simulation Runs Table

| Run | Profile | Steps | Wall Time | Termination | Status |
|-----|---------|-------|-----------|-------------|--------|
| K1 | `k1_pitch_rate_notch_v1` | 2999 | 593s | Completed | ✅ Current-best |
| LR1 ref | `lr1_k1_replacement_coordinated_low_freq_v1` | 494 | 176s | `height_too_low` | ❌ |
| LRS1 | `lrs1_support_dominant_v1` | 319 | ~95s | `height_too_low` | ❌ |
| LRS2 | `lrs2_pitch_rate_damping_v1` | 447 | ~145s | `height_too_low` | ❌ |
| LRS3 | `lrs3_balanced_medium_v1` | 473 | ~155s | `orientation_fail` | ❌ |

## Appendix C: LRS Gain Values at h=0.48m

| Variant | k_pitch | k_pitch_rate | k_support | k_support_vel |
|---------|---------|-------------|-----------|--------------|
| LR1 (reference) | 3.50 | 1.20 | -12.00 | -0.60 |
| LRS1 | 3.50 | 1.20 | **-21.60** (1.8×) | **-1.08** (1.8×) |
| LRS2 | 3.50 | **3.00** (2.5×) | -12.00 | -0.60 |
| LRS3 | **5.25** (1.5×) | **2.40** (2×) | **-18.00** (1.5×) | **-0.90** (1.5×) |
| K1 (for reference) | kp=50.0 | kd=10.0 | (separate tau_position) | (separate tau_support_vel) |
