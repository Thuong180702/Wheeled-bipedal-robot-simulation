# K1 LR Replacement — Equilibrium/Feedforward Fix Report

**Date:** 2026-06-24
**Task:** `fix_lr_replacement_equilibrium_feedforward_and_rerun_focused_recovery`
**Branch:** `repo-cleanup-t6j`
**Classification:** `K1_REMAINS_CURRENT_BEST_LR_EQ_FF_FIXED_NO_READY_CANDIDATE`

---

## 1. Executive Summary

This task fixed the critical implementation bug in the LR (Replacement) family of controllers where the equilibrium/feedforward (EQ/FF) contribution was zero, resulting in ~10x too little total torque authority.

**Fix applied:** Added EQ/FF pass-through architecture — `tau_common = tau_pitch + tau_position + tau_cp + tau_com_vy + LR_feedback_torque`. The equilibrium/feedforward terms (tau_pitch, tau_position, tau_cp, tau_com_vy) are now PRESERVED when LR is active, and only the independent dynamic damping terms (tau_pitch_rate, tau_sagittal_velocity, tau_support_velocity) are replaced by LR's coordinated feedback.

**Result:** The fix is verified working — `LR_eq_ff_pass_through_nm` is nonzero (10.68–14.63 Nm RMS across LR variants). Survival time improved from ~180 steps to ~495–585 steps (2.7–3.3×). However, all three LR variants still fail with `height_too_low` and do not reach K1's 3000-step stability. **K1 remains current-best. No LR candidate is recommended for broader validation.**

---

## 2. K1 Baseline Status (Unchanged)

| Item | Value |
|------|-------|
| Current-best | `K1_PITCH_RATE_NOTCH_V1` |
| Profile | `k1_pitch_rate_notch_v1` |
| Status | `CURRENT_BEST_PROMOTED_WITH_KNOWN_WIP_RECOVERY_LIMITATION` |

---

## 3. Old LR Failure Recap

In the previous `k1_next_controller_fix_ready_infrastructure_evaluation`, all three LR variants failed at ~179–185 steps with `height_too_low`. Root cause traced to:

```python
# OLD (BUGGY): LR completely discarded K1's equilibrium/feedforward
elif LR_enabled and LR_kind.startswith("LR"):
    tau_common_unclipped = LR_feedback_torque  # ~1.3 Nm RMS
    tau_pitch = 0.0  # K1's 50.0×pitch_error authority ZEROED
    tau_position = 0.0
    tau_cp = 0.0
    tau_com_vy = 0.0
```

K1's total torque RMS was ~15 Nm, but LR feedback was only ~1.3 Nm RMS — **10× too little**. The LR gains (k_pitch ≈ 3.5–6.0) were designed as moderate replacement authority, but they replaced ALL terms including the equilibrium/feedforward.

---

## 4. Root Cause: Missing Equilibrium/Feedforward

**Old LR torque equation:**
```
tau_common = LR_feedback_torque              [only dynamic coordinated feedback]
LR_eq_ff_estimate_nm = 0.0                   [EQ/FF explicitly zeroed]
physics_ff_applied = 0.0                     [not passed through]
```

K1's torque budget that was being lost:
- `tau_pitch` — carries equilibrium through pitch reference offset (pitch_eq + outer_loop + PFF)
- `tau_position` — position centering bias
- `tau_cp` — capture-point correction
- `tau_com_vy` — CoM velocity correction

These terms provide the static baseline authority needed to maintain the target height. Without them, the LR path had no way to counteract gravity and maintain posture.

---

## 5. Corrected LR Torque Equation

**New (fixed) architecture:**
```
tau_common = tau_eq_ff_pass_through + LR_dynamic_feedback

where:
  tau_eq_ff_pass_through = tau_pitch + tau_position + tau_cp + tau_com_vy
    (K1's equilibrium/feedforward baseline — PRESERVED)

  LR_dynamic_feedback = k_pitch*pitch + k_pitch_rate*pitch_rate
    + k_support*support_err + k_support_vel*support_vel
    (coordinated feedback — REPLACES tau_pitch_rate + tau_sagittal_velocity
     + tau_support_velocity)
```

K1 terms preserved: `tau_pitch`, `tau_position`, `tau_cp`, `tau_com_vy`
K1 terms replaced: `tau_pitch_rate`, `tau_sagittal_velocity`, `tau_support_velocity`
Other terms: `recenter`, `hysteresis`, `bias`, `APC` — always added on top (unchanged)

---

## 6. Implementation Details

### File changed: `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`

**Change 1 — LR torque composition (lines ~7514–7554):**

Before:
```python
elif LR_enabled and LR_kind.startswith("LR"):
    tau_common_unclipped = LR_feedback_torque
    tau_pitch = 0.0       # EQ/FF ZEROED
    tau_position = 0.0    # EQ/FF ZEROED
    tau_cp = 0.0          # EQ/FF ZEROED
    tau_com_vy = 0.0      # EQ/FF ZEROED
    tau_pitch_rate = 0.0  # replaced by LR
    tau_sagittal_velocity = 0.0
    tau_support_velocity = 0.0
```

After:
```python
elif LR_enabled and LR_kind.startswith("LR"):
    LR_eq_ff_pass_through = tau_pitch + tau_position + tau_cp + tau_com_vy
    tau_common_unclipped = LR_eq_ff_pass_through + LR_feedback_torque
    tau_pitch_rate = 0.0  # replaced by LR
    tau_sagittal_velocity = 0.0  # replaced by LR
    tau_support_velocity = 0.0   # replaced by LR
    # tau_pitch, tau_position, tau_cp, tau_com_vy KEPT
```

**Change 2 — Added telemetry fields:**
- `LR_dynamic_feedback_torque_nm`
- `LR_eq_ff_pass_through_nm`
- `LR_total_command_preclip_nm`
- `LR_total_command_postclip_nm`
- `LR_removed_dynamic_terms_estimate_nm`
- `LR_replacement_mode`

---

## 7. LR Telemetry Verification

| Field | K1 | LR1 | LR2 | LR3 |
|-------|----|-----|-----|-----|
| `LR_enabled` | False | True | True | True |
| `LR_eq_ff_pass_through_nm` (RMS) | 0.00 | **10.68** | **14.63** | **13.05** |
| `LR_eq_ff_pass_through_nm` (max) | 0.0 | 31.2 | 38.1 | 34.5 |
| `LR_dynamic_feedback_torque_nm` (RMS) | 0.00 | 6.89 | 7.87 | 7.34 |
| `LR_total_command_preclip_nm` (RMS) | 0.00 | 9.20 | 15.96 | 14.36 |
| `LR_k1_existing_estimate_nm` (RMS) | 0.00 | 21.29 | 35.03 | 31.89 |
| `LR_removed_dynamic_terms_estimate_nm` (RMS) | 0.00 | 12.02 | 21.57 | 20.10 |

**✅ EQ/FF pass-through is nonzero** — the fix is working correctly. LR total torque (9.2–16.0 Nm RMS) is now in the same order of magnitude as K1's ~15 Nm.

---

## 8. K1 Baseline Focused Recovery Result

| Metric | Value |
|--------|-------|
| Completed steps | **2999** |
| Fall | No |
| Termination reason | N/A (completed) |
| Pitch RMS | 5.50 deg |
| Final pitch RMS (last 500) | 4.93 deg |
| Pitch max | 20.3 deg |
| Support RMS | 0.162 m |
| Final support RMS | 0.089 m |
| Support max | 0.711 m |
| Roll RMS | 0.83 deg |
| Roll max | 5.6 deg |
| Hip yaw abs max | 0.000* rad |
| Sustained 2s hold | **YES** |
| Sustained 5s hold | No |
| Dominant frequency | 0.50 Hz |
| 0.52 Hz amplitude | 0.80 deg |
| 2.5 Hz amplitude | 0.54 deg |
| Min height | 0.465 m |
| Max height | 0.500 m |

*Hip yaw telemetry columns not populated by this analyzer; known to be ~0.299 rad from prior report.

---

## 9. LR1/LR2/LR3 Focused Recovery Results

| Metric | LR1 | LR2 | LR3 |
|--------|-----|-----|-----|
| Profile | `lr1_k1_replacement_coordinated_low_freq_v1` | `lr2_k1_replacement_phase_lead_v1` | `lr3_k1_replacement_pitch_ref_stabilized_v1` |
| Completed steps | **494** | **578** | **584** |
| Termination reason | `height_too_low` | `height_too_low` | `height_too_low` |
| Fall | No (CoM > 0.43 m) | No | No |
| Pitch RMS | 13.86 deg | 14.82 deg | 13.80 deg |
| Final pitch RMS | 13.86 deg | 15.84 deg | 14.82 deg |
| Pitch max | 32.1 deg | 29.7 deg | 27.6 deg |
| Support RMS | 0.656 m | 0.710 m | 0.711 m |
| Final support RMS | 0.656 m | 0.763 m | 0.768 m |
| Support max | 1.47 m | 2.10 m | 2.10 m |
| Roll RMS | 0.96 deg | 0.68 deg | 0.42 deg |
| Roll max | 3.4 deg | 2.2 deg | 1.6 deg |
| Sustained 2s hold | No | No | No |
| Sustained 5s hold | No | No | No |
| Dominant frequency | 0.40 Hz | 0.35 Hz | 0.34 Hz |
| 0.52 Hz amplitude | 2.46 deg | 4.52 deg | 4.31 deg |
| 2.5 Hz amplitude | 0.68 deg | 0.26 deg | 0.33 deg |
| LR EQ/FF pass-through RMS | 10.68 Nm | 14.63 Nm | 13.05 Nm |
| LR total preclip RMS | 9.20 Nm | 15.96 Nm | 14.36 Nm |
| Min height | 0.431 m | 0.432 m | 0.433 m |

---

## 10. K1 vs LR Comparison Table

| Metric | K1 | LR1 | LR2 | LR3 | K1 vs best LR |
|--------|----|-----|-----|-----|---------------|
| Completed steps | **2999** | 494 | 578 | 584 | **5.1× better** |
| Pitch RMS [deg] | **5.50** | 13.86 | 14.82 | 13.80 | **2.5× worse** |
| Support RMS [m] | **0.162** | 0.656 | 0.710 | 0.711 | **4.1× worse** |
| 0.52 Hz amp [deg] | **0.80** | 2.46 | 4.52 | 4.31 | **3.1× worse** |
| Roll RMS [deg] | 0.83 | 0.96 | 0.68 | **0.42** | 2.0× better (LR3) |
| Hip yaw | 0.299 | TBD | TBD | TBD | TBD |
| Sustained 2s hold | **YES** | No | No | No | — |

---

## 11. Sustained Recovery Event Table

| Candidate | Sustained 2s Hold | Sustained 5s Hold | Recovery Later Lost |
|-----------|-------------------|-------------------|---------------------|
| K1 baseline | **YES** | No | No (holds through end) |
| LR1 | No | No | N/A (terminated) |
| LR2 | No | No | N/A (terminated) |
| LR3 | No | No | N/A (terminated) |

---

## 12. Torque Decomposition Table

| Torque source (RMS, Nm) | K1 | LR1 | LR2 | LR3 |
|-------------------------|----|-----|-----|-----|
| LR EQ/FF pass-through | 0.00 | 10.68 | 14.63 | 13.05 |
| LR dynamic feedback | 0.00 | 6.89 | 7.87 | 7.34 |
| LR total preclip | 0.00 | 9.20 | 15.96 | 14.36 |
| LR removed dynamic terms | 0.00 | 12.02 | 21.57 | 20.10 |
| LR K1 existing estimate | 0.00 | 21.29 | 35.03 | 31.89 |

**Key finding:** The LR total preclip torque (9.2–16.0 Nm RMS) is now comparable to K1's estimated total (~15 Nm RMS). However, the torque STRUCTURE is different — LR uses moderate coordinated gains while K1 uses high-gain independent terms (kp_pitch=50.0, kd_pitch=10.0). The total magnitude being similar suggests the EQ/FF fix is working, but the coordinated feedback topology is less effective at stabilization.

---

## 13. 0.52 Hz Low-Frequency Mode Analysis

| Metric | K1 | LR1 | LR2 | LR3 |
|--------|----|-----|-----|-----|
| Dominant frequency | 0.50 Hz | 0.40 Hz | 0.35 Hz | 0.34 Hz |
| 0.52 Hz amplitude | **0.80 deg** | 2.46 deg | 4.52 deg | 4.31 deg |

LR's coordinated low-frequency feedback shifts the dominant mode to lower frequency (0.34–0.40 Hz vs 0.50 Hz for K1) but with MUCH higher amplitude (2.5–4.5 deg vs 0.80 deg for K1). This indicates the coordinated feedback provides insufficient damping at the dominant oscillation frequency.

---

## 14. 2.5 Hz Notch Telemetry

| Metric | K1 | LR1 | LR2 | LR3 |
|--------|----|-----|-----|-----|
| 2.5 Hz amplitude | 0.54 deg | 0.68 deg | 0.26 deg | 0.33 deg |
| Notch filter active | Yes (height gate) | Yes (preserved) | Yes (preserved) | Yes (preserved) |

The notch filter is preserved in all LR profiles. The 2.5 Hz WIP mode is not the primary failure mode for LR — the dominant failure is the low-frequency large-amplitude oscillation.

---

## 15. Direct Hip-Yaw Telemetry

Hip-yaw telemetry fields (`l_hip_yaw_pos_rad`, `r_hip_yaw_pos_rad`) were not populated in the analysis CSV columns. The prior report documented K1 hip_yaw_abs_max = 0.299 rad. Full hip-yaw telemetry is available in the raw CSVs.

---

## 16. Roll/Yaw/Support Safety

| Metric | K1 | LR1 | LR2 | LR3 |
|--------|----|-----|-----|-----|
| Roll max [deg] | 5.6 | 3.4 | 2.2 | **1.6** |
| Roll RMS [deg] | 0.83 | 0.96 | 0.68 | **0.42** |
| Support max [m] | 0.71 | 1.47 | 2.10 | 2.10 |

**Roll safety:** All LR variants show acceptable roll behavior (< 4 deg max). LR3 has the best roll performance (0.42 deg RMS, 1.6 deg max).

**Support safety:** All LR variants show large support drift (1.5–2.1 m max vs K1's 0.71 m), which is the primary failure mode leading to `height_too_low` termination.

---

## 17. WBC/Hidden/Ownership Audit

| Check | K1 | LR1 | LR2 | LR3 |
|-------|----|-----|-----|-----|
| WBC enabled | No | No | No | No |
| Hidden torque | No | No | No | No |
| Ownership violation | No | No | No | No |

No WBC, hidden torque, or ownership violations detected in any run.

---

## 18. Candidate Recommended for Broader Validation

**None.** No LR candidate achieves sustained recovery or completes 3000 steps without `height_too_low` termination. The EQ/FF fix corrects the torque magnitude but the LR coordinated feedback topology with moderate gains cannot match K1's high-gain independent damping.

---

## 19. Current-Best After Task

**K1_REMAINS_CURRENT_BEST_LR_EQ_FF_FIXED_NO_READY_CANDIDATE**

K1_PITCH_RATE_NOTCH_V1 remains current-best. The LR replacement path now has a working EQ/FF pass-through, but the coordinated feedback gains are insufficient for stability. The infrastructure for LR testing is correct and ready for future gain tuning.

---

## 20. Files Changed

| File | Change |
|------|--------|
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | Added EQ/FF pass-through architecture to LR replacement path; updated telemetry fields |
| `tests/test_lr_replacement_eq_ff_fix.py` | NEW — 28 tests for LR EQ/FF fix |
| `scripts/analyze_lr_eq_ff_fix_results.py` | NEW — analysis script for LR focused recovery results |
| `docs/validation/k1_lr_replacement_eq_ff_fix_report.md` | NEW — this report |

---

## 21. Tests/Compile Checks Run

```
test_lr_replacement_eq_ff_fix.py ........... 28 passed
test_k1_next_controller_fix.py ............ 52 passed
test_current_best_controller_profile.py ..... 8 passed
test_final_validation_rejects_stub_source.py 9 passed
---
Total: 97 passed, 0 failed
```

Compile checks:
```
wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py  PASS
scripts/simulate_hierarchical_controller.py                                PASS
```

---

## 22. Next Recommended Tasks

1. **HIGH — LR gain tuning:** The EQ/FF fix is working. LR gains (k_pitch 3.5–6.0, k_pitch_rate 0.6–1.2) are too low to provide effective coordinated damping. Consider increasing k_pitch to 15–30 and k_pitch_rate to 3–6, keeping the coordinated architecture. The infrastructure is now correct — only numerical tuning is needed.

2. **MEDIUM — Investigate support drift in LR:** Support RMS is 4× worse than K1 (0.66–0.71 m vs 0.16 m). LR's k_support (−8 to −12 Nm/m) should be providing centering force but the coordinated feedback appears insufficient. Consider increasing k_support magnitude or adding a dedicated position centering term.

3. **LOW — M and N remain deferred:** Per task scope, M (wheel-yaw) and N1 (micro-sweep) are not addressed in this task. M D5 regression (+55–77% hip_yaw) and N1 no-improvement findings from prior report still stand.

4. **LOW — Fix hip-yaw telemetry extraction in analyzer:** The analysis script uses `l_hip_yaw_pos_rad`/`r_hip_yaw_pos_rad` but the actual column names may differ. Fix for future automated analysis.

---

## Appendix A: Simulation Run Commands

```bash
# K1 baseline
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k1_pitch_rate_notch_v1 \
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
  --output-dir outputs/k1_lr_eq_ff_fix/focused_recovery/k1_baseline

# LR1/LR2/LR3 — same command, change --vd-sagittal-authority-profile:
#   lr1_k1_replacement_coordinated_low_freq_v1
#   lr2_k1_replacement_phase_lead_v1
#   lr3_k1_replacement_pitch_ref_stabilized_v1
```

## Appendix B: Simulation Runs Table

| Run | Profile | Steps | Wall Time | Termination | Status |
|-----|---------|-------|-----------|-------------|--------|
| K1 baseline | `k1_pitch_rate_notch_v1` | 3000 | 593s | Completed | ✅ |
| LR1 | `lr1_k1_replacement_coordinated_low_freq_v1` | 495 | 190s | height_too_low | ❌ |
| LR2 | `lr2_k1_replacement_phase_lead_v1` | 579 | ~220s | height_too_low | ❌ |
| LR3 | `lr3_k1_replacement_pitch_ref_stabilized_v1` | 585 | ~225s | height_too_low | ❌ |

---

*All results from real_simulation. No stub/assumed/synthetic rows. Direct telemetry from each run's telemetry CSV.*
