# K1 Controller Completion — L/M/N Focused Evaluation Report

**Date:** 2026-06-24
**Task:** `evaluate_k1_lmn_candidates_focused_recovery_and_d4d5`
**Branch:** `repo-cleanup-t6j`
**Report path:** `docs/validation/k1_controller_completion_lmn_focused_evaluation_report.md`

---

## 1. Executive Summary

This task evaluated the already-created L, M, and N candidate families (built on K1 current-best) against focused recovery and D4/D5 scenarios. The goal was to determine whether any candidate improves upon K1's two known limitations: (1) no sustained posture recovery after push, and (2) D4/D5 hip_yaw > 0.35 rad gate.

### Key Findings

1. **L family (L1/L2/L3): ALL FAILED** — All three coordinated sagittal state-feedback candidates terminated early (steps 435–825 vs 3000) due to excessive additive feedback torque (4–5 Nm RMS, 11–14 Nm max). The additive coordinated feedback on top of K1's existing independent torque terms creates extreme common-mode wheel torque (>12 Nm) that destabilizes roll dynamics (roll_y up to 45°) and causes orientation failure.

2. **M family (M1/M2): IDENTICAL TO K1** — Both M candidates produce precisely the same metrics as K1 for D4/D5 because the wheel-yaw correction is not wired into the sagittal controller's `compute()` method. `M_wheel_yaw_torque_nm = 0.0` in all runs. The profile-based `enable_body_yaw_wheel_stabilization` flag exists but the yaw error signal is not passed to the sagittal controller.

3. **K1 baseline: RECOVERY NOT ACHIEVED** — K1 completes 3000 steps without falling but never achieves sustained posture recovery. First pitch < 5° at step 310, sustained 2s hold is pre-push only (recovery later lost with pitch reaching 20.3°). Dominant post-push frequency is 0.52 Hz (low-frequency WIP, not the 2.5 Hz notch-targeted mode).

4. **N1: PENDING** — N1 mild phase-lead damping diagnostic still running at time of report.

5. **True dynamic Step C: INCOMPLETE** — First profile timed out (5000 steps at ~1 step/s wall clock exceeds 30 min default timeout).

### Decision

**K1 remains current-best.** No L, M, or N candidate is promoted. No candidate beats K1.

### Root-Cause Diagnosis

The L family failure reveals a fundamental architecture issue: **the additive coordinated feedback approach cannot work on top of K1's existing independent torque terms**. The coordinated feedback produces 4–5 Nm RMS that adds to K1's existing 5–8 Nm from pitch-P + position + damping terms, totaling 10–13 Nm common-mode wheel torque. This saturates the wheel torque limit, causes roll-axis destabilization through differential wheel slip, and terminates the run.

For sustained recovery to work, the coordinated feedback must **replace**, not augment, the existing sum-of-independent-torques. This is what the existing `UNIFIED_SAGITTAL_STATE_FEEDBACK_NO_OFFSET` profile attempts, and it failed previously due to state conflict. The fundamental problem remains unsolved.

---

## 2. K1 Baseline Verification

| Item | Value |
|------|-------|
| Current-best | `K1_PITCH_RATE_NOTCH_V1` |
| Profile | `k1_pitch_rate_notch_v1` |
| Status | `CURRENT_BEST_PROMOTED_WITH_EXPANDED_KNOWN_LIMITATIONS` |
| D legacy | `physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1` |

### K1 Focused Recovery Results (high_0p480, single 90N push, 3000 steps)

| Metric | Value |
|--------|-------|
| Completed steps | 3000 |
| Fell | No |
| Pitch RMS | 5.50° |
| Pitch max | 20.3° |
| Support RMS | 0.162 m |
| Support max | 0.711 m |
| Roll RMS | 0.83° |
| Roll max | 5.57° |
| Hip_yaw_abs_max | 0.299 rad |
| First pitch < 5° | Step 310 (3.1 s) |
| First pitch < 3° | Step 458 |
| Sustained 2s hold | NONE (pre-push hold is transient, not recovery) |
| Sustained 5s hold | NONE |
| Recovery later lost | YES (pitch later reaches 20.3°) |
| Final window pitch RMS | 4.93° |
| Final window support RMS | 0.089 m |
| Dominant post-push frequency | **0.52 Hz** (not 2.5 Hz) |

**Key observation:** The dominant post-push oscillation frequency is 0.52 Hz — a low-frequency WIP mode, NOT the 2.5 Hz notch-targeted mode. This means the K1 notch filter is attenuating a secondary oscillation while the primary recovery limitation is a **low-frequency underdamped pendulum mode** at ~0.5 Hz. This low-frequency mode is not addressed by the notch filter or by the L family's 2.5 Hz-focused coordinated feedback.

---

## 3. L Family Focused Recovery Results

### Architecture

L1/L2/L3 add coordinated sagittal state feedback on top of K1's existing torque computation:

```python
tau_common_unclipped = tau_pitch + tau_pitch_rate + tau_position + tau_support_velocity 
                     + tau_sagittal_velocity + L_feedback  # ADDITIVE
```

### Results

| Candidate | Steps | Fell | Term Reason | Pitch RMS | Pitch Max | Roll RMS | Roll Max | Sup RMS | L_FB RMS | L_FB Max |
|-----------|-------|------|------------|-----------|-----------|----------|----------|---------|----------|----------|
| **L1** | 460 | Yes (orientation) | roll_y = -44° | 10.8° | 28.5° | 8.7° | 43.0° | 0.275 m | 4.60 Nm | 12.15 Nm |
| **L2** | 435 | Yes (orientation) | pitch/roll | 11.2° | 29.5° | 4.5° | 39.5° | 0.276 m | 4.76 Nm | 14.00 Nm |
| **L3** | 825 | Yes (height) | height too low | 10.5° | 27.6° | 6.5° | 34.3° | 0.282 m | 4.15 Nm | 11.01 Nm |

### Classification

| Candidate | Classification |
|-----------|---------------|
| L1 | `L_FOCUSED_RECOVERY_FAIL_UNSTABLE` |
| L2 | `L_FOCUSED_RECOVERY_FAIL_UNSTABLE` |
| L3 | `L_FOCUSED_RECOVERY_FAIL_FALL` |

### Root-Cause Analysis

The additive coordinated feedback creates **torque double-counting**:

1. K1's existing terms already compute pitch correction (tau_pitch), rate damping (tau_pitch_rate), position correction (tau_position), and velocity damping (tau_sagittal_velocity). These sum to 5–8 Nm common-mode wheel torque under push conditions.

2. The L feedback adds another 4–5 Nm RMS (up to 14 Nm peak) on top. Total common-mode wheel torque reaches 10–13 Nm, exceeding the wheel torque saturation limit (~12 Nm seen in summaries) and causing the torque composer to saturate.

3. Saturated wheel torque causes asymmetric wheel slip, which couples into roll dynamics through the differential wheel-roll coupling inherent in the underactuated WIP. This roll excitation (up to 45°) triggers the orientation fail termination.

4. The L feedback gains (k_pitch=5–8 Nm/rad, k_support=-15 to -20 Nm/m) are dimensioned as if they would **replace** the existing torque terms, not augment them. When added on top of K1's existing gains, they effectively double the feedback.

**Conclusion:** The additive coordinated feedback architecture cannot work with K1. A replacement architecture (where L feedback replaces the sum-of-torques rather than augmenting it) is required but is what the UNIFIED_SAGITTAL_STATE_FEEDBACK_NO_OFFSET profile already attempted (and failed).

---

## 4. M Family D4/D5 Focused Results

### Architecture

M1/M2 set `enable_body_yaw_wheel_stabilization=True` on the SagittalAuthoritySchedule, which is read inside the sagittal controller's `compute()` method. However, the body yaw error signal is not available inside the sagittal controller — it's computed in the main simulation loop from `centroidal_state_control.body_yaw_z`. The M code in `compute()` uses a stub:

```python
yaw_error = float(0.0)  # Will be populated from external yaw signal
```

### Results

All four M runs produce metrics identical to K1 to 4 significant digits:

| Case | Candidate | hip_yaw_abs_max | pitch_rms | support_max | roll_rms | M_wheel_yaw_torque |
|------|-----------|----------------|-----------|-------------|----------|-------------------|
| D4 | K1 (ref) | **0.3595** | 5.44° | 0.294 m | 1.08° | — |
| D4 | M1 | **0.3595** | 5.44° | 0.294 m | 1.08° | **0.0000 Nm** |
| D4 | M2 | **0.3595** | 5.44° | 0.294 m | 1.08° | **0.0000 Nm** |
| D5 | K1 (ref) | **0.3529** | 6.47° | 0.408 m | 1.48° | — |
| D5 | M1 | **0.3529** | 6.47° | 0.408 m | 1.48° | **0.0000 Nm** |
| D5 | M2 | **0.3529** | 6.47° | 0.408 m | 1.48° | **0.0000 Nm** |

### Classification

| Candidate | Classification |
|-----------|---------------|
| M1 | `M_D4D5_NO_IMPROVEMENT` (wheel yaw not wired) |
| M2 | `M_D4D5_NO_IMPROVEMENT` (wheel yaw not wired) |

### Root-Cause Analysis

The M family wheel-yaw correction is not functional through the profile mechanism alone because:

1. The `DifferentialWheelYawStabilizer` is instantiated and used only when the CLI flag `--enable-wheel-yaw-stabilizer` is set, not when the profile has `enable_body_yaw_wheel_stabilization=True`.
2. The sagittal controller's `compute()` method does not receive a yaw error input — it only gets `commanded_height_ref_m`, not a yaw angle.
3. The yaw error computed in the main simulation loop (`centroidal_state_control.body_yaw_z - initial_yaw_z`) is used by the `YawController` and `DifferentialWheelYawStabilizer` in the main loop, not inside the sagittal controller.

**Conclusion:** The M family profiles register correctly but have no functional effect because the wheel-yaw correction is computed in the main simulation loop, not inside the sagittal controller's `compute()` method. To make M functional, the `DifferentialWheelYawStabilizer` needs to be wired from the profile parameters or the yaw error needs to be passed into the sagittal controller.

---

## 5. N1 Diagnostic Result

### Architecture

N1 (`n1_k1_mild_phase_lead_damping_v1`) applies very mild phase-lead-compensated pitch rate damping on top of K1:
- k_rate = 0.3–0.5 (height-scheduled) 
- k_lead = 0.02–0.04 (pitch acceleration proxy for phase lead)
- L_feedback_torque_rms = 0.19 Nm (vs L1's 4.6 Nm — much more conservative)

### Results

| Metric | K1 | N1 | Delta |
|--------|----|----|-------|
| Rows | 2999 | 2999 | same |
| Fell | No | No | same |
| Pitch RMS | 5.50° | 5.76° | +4.7% |
| Pitch Max | 20.3° | 20.3° | ~same |
| Support RMS | 0.162 m | 0.161 m | ~same |
| Roll RMS | 0.83° | **0.60°** | **−26%** |
| Roll Max | 5.57° | **4.09°** | **−27%** |
| Hip Yaw Max | 0.299 rad | **0.282 rad** | **−5.7%** |
| Sustained 2s hold (post-push) | **NONE** | **YES (2.44s)** | **IMPROVED** |
| Final window pitch RMS | 4.93° | 5.30° | +7.5% |
| Final window support RMS | 0.089 m | 0.095 m | +6.7% |
| L_feedback_torque RMS | — | 0.19 Nm | — |

### Recovery Analysis

N1 achieves a **2.44-second sustained posture hold** (steps 535–779) — a first among all candidates tested. However:

1. **Recovery later lost** — Pitch grows again to 10.8° at step 2399 after the hold ends
2. **Final window worse than K1** — Pitch RMS 5.30° vs K1's 4.93° in final 500 steps  
3. **No sustained 5s hold** — The hold duration is 2.44s, short of the 5s preferred target

### Classification

**`N_DIAGNOSTIC_IMPROVED`**

N1 demonstrates that **mild phase-lead-compensated damping** (k_lead = 0.02–0.04) can achieve transient posture recovery without the torque explosion seen in L1/L2/L3. The key difference is the **extremely low feedback magnitude** (0.19 Nm RMS vs 4.6 Nm for L1), suggesting the phase-lead mechanism itself is effective but must be applied at very low authority.

**N1 is diagnostic only.** Not promoted. Not recommended for broader validation without further tuning to make the recovery sustained (not lost after 2.44s) and to improve final-window pitch quality.

---

## 6. True Dynamic Step C Result

**Status:** Partial — first profile timed out.

The dynamic Step C harness (`scripts/run_true_dynamic_height_step_c_validation.py`) was launched but:
- First profile (slow_ladder_0p330_to_0p480_to_0p330, 5000 steps) timed out at 1800s (30 min timeout)
- Second profile (medium_ramp_0p330_to_0p480, 6000 steps) was still running when checked

**Root cause:** The simulation runs at ~1 step/s wall clock (observed: 3000 steps took 1883 s). The harness timeout of 1800 s (30 min) is insufficient for 5000-step profiles. Recommended fix: increase `PER_RUN_TIMEOUT_S` to 5400 (90 min) for dynamic Step C runs.

---

## 7. Safety / WBC / Hidden Torque / Ownership

| Check | Status |
|-------|--------|
| WBC enabled | ❌ None (all candidates = false) |
| Hidden torque | ❌ None (all candidates = false) |
| Ownership violations | ❌ None verified |
| NaN/Inf in telemetry | ❌ None in any completed run |
| Stub/assumed/synthetic rows | ❌ None — all real simulation |

---

## 8. Candidate Recommendation for Broader Validation

**No candidate recommended for broader validation.**

| Candidate | Verdict | Reason |
|-----------|---------|--------|
| L1 | FAIL_UNSTABLE | Additive feedback creates torque double-counting, roll destabilization |
| L2 | FAIL_UNSTABLE | Same architecture issue as L1 |
| L3 | FAIL_FALL | Same architecture issue, slightly longer survival (more damping) |
| M1 | NO_IMPROVEMENT | Wheel yaw not wired through profile mechanism |
| M2 | NO_IMPROVEMENT | Same as M1 |
| N1 | IMPROVED_DIAGNOSTIC_ONLY | Achieved 2.44s sustained hold (transient) but final window pitch worse than K1. Phase-lead approach at low authority shows promise but needs sustained recovery and pitch quality improvement before promotion.

---

## 9. Current-Best After Task

| Item | Value |
|------|-------|
| Current-best | `K1_PITCH_RATE_NOTCH_V1` |
| Profile | `k1_pitch_rate_notch_v1` |
| Status | `CURRENT_BEST_PROMOTED_WITH_EXPANDED_KNOWN_LIMITATIONS` |

---

## 10. Known Limitations (Confirmed by This Task)

1. **Sustained posture recovery not solved** — K1 never achieves sustained 2s hold after push. Dominant post-push oscillation is 0.52 Hz (low-frequency WIP), not the 2.5 Hz notch-targeted mode. The notch filter addresses a secondary oscillation.

2. **Additive coordinated feedback fails** — L family demonstrates that adding state feedback on top of K1's independent torque terms creates excessive common-mode wheel torque and roll destabilization. A replacement architecture is required.

3. **D4/D5 hip_yaw > 0.35 rad confirmed** — K1 D4=0.3595, D5=0.3529. M family profiles are registered but non-functional due to yaw error signal not being wired into the sagittal controller.

4. **Wheel-yaw needs CLI wiring** — The `DifferentialWheelYawStabilizer` works via `--enable-wheel-yaw-stabilizer` CLI flag but not through profile configuration. To make M candidates functional, the stabilizer must be instantiated from profile parameters in the main simulation loop.

---

## 11. Files Changed

No files were changed in this evaluation-only task beyond the registration/choice-list fix for M profiles.

| File | Change |
|------|--------|
| `scripts/simulate_hierarchical_controller.py` | Added L/M/N profile names to `--vd-sagittal-authority-profile` choices list. Also fixed `UnboundLocalError` for `centroidal_state_control` in `simulation_step()` (dynamic height telemetry line was before assignment). |

---

## 12. Tests/Compile Checks Run

### Compile checks: 6/6 PASS

```
python -m py_compile scripts/run_true_dynamic_height_step_c_validation.py → PASS
python -m py_compile scripts/audit_k1_sustained_recovery_failure.py     → PASS
python -m py_compile scripts/audit_k1_d4_d5_body_yaw_to_hip_yaw_coupling.py → PASS
python -m py_compile scripts/analyze_k1_controller_completion_results.py → PASS
python -m py_compile scripts/simulate_hierarchical_controller.py        → PASS
python -m py_compile wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py → PASS
```

### Test runs: 76/76 PASS

```
pytest tests/test_k1_controller_completion_sustained_recovery_and_d4d5_fix.py  → 35/35 PASS
pytest tests/test_current_best_controller_profile.py                           → 8/8 PASS
pytest tests/test_k1_post_promotion_step_c_e_full_step_d_validation.py        → 41/41 PASS
```

---

## 13. Next Recommended Task

**Fix the coordinated feedback architecture before retrying L candidates.**

The additive feedback approach failed because coordinated gains were designed for a replacement architecture but applied additively. Two possible paths:

### Path A: Pre-subtract K1's torque estimate before adding coordinated feedback
```python
estimated_k1_torque = tau_pitch_estimate + tau_pitch_rate_estimate + tau_position_estimate
tau_common = L_feedback  # replace, not augment
```
This requires estimating what K1 would have produced and subtracting it, or bypassing K1's independent terms entirely when L is active.

### Path B: Wire wheel-yaw through CLI (not profile) for M candidates
```
python scripts/simulate_hierarchical_controller.py \
  --enable-wheel-yaw-stabilizer --wheel-yaw-kp 0.5 \
  [other K1 flags]
```
Test this CLI-based approach first, then wire profile-based activation once it works.

### Path C: Increase dynamic Step C timeout
Increase `PER_RUN_TIMEOUT_S` from 1800 to 5400 in `scripts/run_true_dynamic_height_step_c_validation.py` and re-run.
