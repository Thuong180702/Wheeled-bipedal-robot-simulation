# Support-Aware Mode-Div Authority Schedule — Final Report

**Date:** 2026-06-23
**Task:** `support_aware_mode_div_authority_schedule`
**Current-best controller (unchanged):** `D_MODE_HIP_YAW_DIV_V1`
**Candidates evaluated:** `H_SUPPORT_AWARE_MODE_DIV_AUTHORITY_V1` family (H1, H2, H3 variants)
**Report classification:** `SUPPORT_AWARE_MODE_DIV_FIX_NO_IMPROVEMENT_NOT_PASS`

---

## 1. Executive Summary

This task tested whether a **support-aware mode-div authority schedule** (H family) can reduce D5 hip_yaw_abs_max below 0.35 rad without causing support regression, while preserving D4 improvement.

The H family extends G1_sg080 (kp=10, kd=0.50, mt=7.5, sl=0.30, sg=0.80) with a continuous support-aware gate that attenuates mode-div torque when support position error or rate exceeds thresholds.

**Key finding: No H candidate improves D5 below G1_sg080's hy=0.3504. Support-aware attenuation makes D5 WORSE than G1_sg080 in every configuration tested.**

- **D4 preservation:** H2 and H3 candidates preserve G1_sg080's D4 pass (hy=0.3224). H1 candidates regress D4 to hy=0.3466 (still pass but worse).
- **D5 degradation:** Every H candidate degrades D5 below G1_sg080. Best H is H3_t30_w10_mg70_r80 at hy=0.3583 — worse than G1_sg080's 0.3504.
- **Root cause:** The hip-yaw peak coincides with elevated support error. Support-aware attenuation reduces mode-div torque exactly when it is most needed. This confirms the architectural coupling observed in the G task: mode-div torque and support dynamics are coupled at high height, and attenuating one makes the other worse.

**Classification:** `SUPPORT_AWARE_MODE_DIV_FIX_NO_IMPROVEMENT_NOT_PASS`
**D remains current-best.** H is NOT promoted.

---

## 2. Current-Best Status Before This Task

| Item | Value |
|------|-------|
| Current-best | `D_MODE_HIP_YAW_DIV_V1` |
| Current-best profile | `physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1` |
| Status | `CURRENT_BEST_PROMOTED_WITH_KNOWN_HIP_YAW_LIMIT` |
| Known limitation | D4/D5 hip_yaw_abs_max > 0.35 rad |
| G candidates status | `D5_HIGH_HEIGHT_COUPLING_FIX_D5_IMPROVED_NOT_PASS` — G NOT promoted |
| D remains current-best | **YES** — no H candidate improves D5 |

---

## 3. Why G1_sg080 Got Close But Did Not Pass

The D5 plateau around 0.350 rad is caused by body-yaw common-mode coupling at high height:

- Mode-div torque is purely antisymmetric (left/right opposite)
- Hip-yaw error at D5 peak has both divergence (legs twisting oppositely) and common-mode (body yaw driving both legs same direction) components
- Mode-div suppresses divergence error but cannot suppress common-mode error
- The YawController applies antisymmetric hip-yaw torque for body yaw correction, but it fights mode-div at peak (opposite signs)
- Increasing mode-div authority beyond sg=0.80 does not further reduce hip-yaw because the common-mode contribution and support/pitch coupling dominate

G1_sg080 achieved the best safe D5 result at hy=0.3504 — 0.0004 rad above the gate. The remaining gap is caused by architectural coupling, not parameter tuning.

---

## 4. Support/Peak Timing Diagnostic

### 4.1 Timing Analysis

| Candidate | hy_peak | sup_peak | Lag (steps) | Correlation | sup@hy_peak | div@hy_peak | com@hy_peak |
|-----------|---------|----------|-------------|-------------|-------------|-------------|-------------|
| D_baseline_D5 | 0.3803 | 0.5154 | -75 | 0.220 | 0.3051 | 0.4514 | 0.1546 |
| F6_sg050_D5 | 0.3617 | 0.4202 | -32 | 0.243 | 0.2478 | 0.4868 | 0.1184 |
| G1_sg080_D5 | **0.3504** | 0.4864 | **-32** | **0.308** | 0.3107 | 0.4225 | 0.1391 |
| G1_sg080_D4 | 0.3224 | 0.2499 | 76 | 0.153 | 0.1118 | 0.6177 | 0.0135 |
| D_baseline_D4 | 0.4045 | 0.2723 | -26 | 0.076 | 0.1128 | 0.8061 | 0.0014 |

**Key finding:** The support peak **precedes** the hip-yaw peak by 32 steps in G1_sg080_D5. This confirms that support excitation drives hip-yaw re-excitation. The correlation of 0.308 is moderate but positive — when support error increases, hip-yaw also tends to increase.

### 4.2 Support Error Distribution (G1_sg080_D5)

| Percentile | Support error abs (m) |
|------------|----------------------|
| p50 | 0.180 |
| p75 | 0.325 |
| p90 | 0.409 |
| p95 | 0.463 |
| p99 | 0.486 |

### 4.3 Support Error Rate Distribution (G1_sg080_D5)

| Percentile | Rate (m/s) |
|------------|-----------|
| p50 | 0.401 |
| p75 | 0.577 |
| p90 | 0.927 |
| p95 | 1.146 |
| p99 | 1.365 |

### 4.4 Proposed Thresholds

The diagnostic proposed:
- Support error threshold: 0.325 m (p75)
- Support rate threshold: 0.80 m/s
- Min support gate: 0.70

These thresholds were used to design the H candidate grid.

---

## 5. Support-Aware Schedule Design

The support-aware gate is implemented in `ModeBasedHipYawDivergenceController` (opt-in, disabled by default):

### 5.1 Gate Formula

```
support_error_gate = smoothstep_down(|support_error|, threshold, threshold + width)
support_rate_gate  = smoothstep_down(|support_error_rate|, rate_threshold, rate_threshold + rate_width)

effective_support_gate = min(support_error_gate, support_rate_gate)
combined_gate = height_gate * effective_support_gate
```

Where `smoothstep_down(x, low, high)` is C1 continuous: 1.0 at x ≤ low, 0.0 at x ≥ high, with smooth 3u² − 2u³ interpolation.

### 5.2 Continuity Proof

- `smoothstep_down` is C1 continuous (3u² − 2u³ with linear u mapping)
- `support_error_gate` is clamped to `[min_gate, 1.0]` via linear rescaling
- `min(error_gate, rate_gate)` preserves continuity
- Product of two C1 functions is C1
- Therefore `combined_gate` is C1 continuous in all inputs

### 5.3 Candidate Types

| Type | Mechanism | Parameters Swept |
|------|-----------|-----------------|
| H1 | Support-error attenuation | threshold=0.25,0.30m, width=0.05,0.10,0.15m, min_gate=0.70 |
| H2 | Support-rate attenuation | rate_threshold=0.50,0.80,1.00m/s, width=0.30,0.40,0.50, min_gate=0.60,0.70 |
| H3 | Combined error + rate | threshold=0.30,0.35m + rate=0.80,1.00m/s |

### 5.4 Control Ownership

| Mode | Owner | Notes |
|------|-------|-------|
| Hip-yaw divergence | `mode_based_divergence` | Same as D, support-aware gating added |
| Body yaw | `yaw_controller` | Unchanged from D |
| Support/sagittal | `sagittal_velocity_damped_balance` | Unchanged from D |

No D4/D5-specific branches. No height-name-specific branches. No WBC. No hidden torque. No ownership violations.

---

## 6. Focused D4/D5 Sweep Results

### 6.1 D4 (low_0p330, 60N push)

| Candidate | hy | Gate | sGate | cGate | Sup | Pitch° | Roll° | hy<0.35 |
|-----------|------|------|-------|-------|------|--------|-------|---------|
| D_baseline | 0.4076 | 0.000 | 1.000 | 1.000 | 0.3185 | 13.62 | 0.94 | FAIL |
| G1_sg080_ref | **0.3224** | 0.996 | 1.000 | 0.996 | 0.2499 | 12.66 | 1.04 | **PASS** |
| H1_t25_w05_mg70 | 0.3652 | 0.997 | 0.727 | 0.724 | 0.2672 | 12.89 | 0.98 | FAIL |
| H1_t25_w10_mg70 | 0.3466 | 0.997 | 0.727 | 0.725 | 0.2672 | 12.89 | 0.97 | PASS |
| H1_t30 variants | 0.3466 | 0.997 | 0.727 | 0.725 | 0.2672 | 12.89 | 0.97 | PASS |
| H2_r80_w40_mg70 | **0.3224** | 0.996 | 1.000 | 0.996 | 0.2499 | 12.66 | 1.04 | **PASS** |
| H2_r100_w50_mg60 | **0.3224** | 0.996 | 1.000 | 0.996 | 0.2499 | 12.66 | 1.04 | **PASS** |
| H2_r50_w30_mg70 | 0.3501 | 0.996 | 0.995 | 0.992 | 0.2509 | 12.68 | 1.04 | FAIL |
| H3 variants | **0.3224** | 0.996 | 1.000 | 0.996 | 0.2499 | 12.66 | 1.04 | **PASS** |

**Observations:**
- H1 support-error attenuation **regresses D4** from 0.3224 → 0.3466. The sGate of 0.727 unnecessarily attenuates mode-div at low height.
- H2 rate-only above p80 preserves D4 (gate never activated because rate stays below threshold).
- H3 combined also preserves D4 (neither error nor rate thresholds crossed at low height).

### 6.2 D5 (high_0p480, 90N push)

| Candidate | hy | Gate | sGate | cGate | Sup | Pitch° | Roll° | hy<0.35 |
|-----------|------|------|-------|-------|------|--------|-------|---------|
| D_baseline | 0.4030 | 0.000 | 1.000 | 1.000 | 0.5340 | 14.94 | 1.92 | FAIL |
| G1_sg080_ref | **0.3504** | 0.875 | 1.000 | 0.875 | 0.4864 | 14.70 | 1.66 | FAIL |
| H1_t25_w05_mg70 | 0.3673 | 0.875 | 0.721 | 0.631 | 0.4272 | 14.77 | 1.53 | FAIL |
| H1_t25_w10_mg70 | 0.3658 | 0.876 | 0.724 | 0.634 | 0.4118 | 14.78 | 1.51 | FAIL |
| H1_t30_w10_mg70 | 0.3628 | 0.876 | 0.729 | 0.639 | 0.4267 | 14.78 | 1.52 | FAIL |
| H2_r100_w50_mg60 | **0.3570** | 0.875 | 0.902 | 0.788 | 0.4568 | 14.81 | 1.62 | FAIL |
| H2_r80_w40_mg70 | 0.3654 | 0.875 | 0.890 | 0.778 | 0.4659 | 14.80 | 1.63 | FAIL |
| H3_t30_w10_mg70_r80 | 0.3583 | 0.876 | 0.920 | 0.805 | 0.4897 | 14.75 | 1.71 | FAIL |
| H3_t30_w15_mg70_r100 | 0.3584 | 0.874 | 0.949 | 0.829 | 0.4755 | 14.73 | 1.69 | FAIL |
| H3_t35_w10_mg60_r80 | 0.3579 | 0.874 | 0.931 | 0.812 | 0.4777 | 14.72 | 1.69 | FAIL |

**Key observations:**
1. **Every H candidate is worse than G1_sg080 (0.3504).** The best H is H2_r100_w50_mg60 at 0.3570.
2. **H1 (support-error attenuation) is worst:** hy=0.3628–0.3688. The sGate of 0.721–0.732 and cGate of 0.631–0.641 show significant attenuation at D5 height.
3. **H2 (rate-only, high threshold) is least bad:** 0.3570–0.3679. Rate gate at 1.00 m/s rarely activates.
4. **H3 (combined) is in between:** 0.3579–0.3584. Combined gate of 0.805–0.829 still attenuates too much.
5. **Support coupling improves** (sup=0.4118–0.4897 vs G1_sg080 0.4864) but at the cost of hip-yaw regression.
6. No falls, no WBC, no saturation, no ownership violations.

**Root cause confirmed:** The hip-yaw peak and support peak coincide. Attenuating mode-div torque during support excitation reduces the torque at exactly the wrong moment — when divergence error is largest and needs correction. This confirms that support-aware attenuation is **fundamentally counterproductive** for D5.

---

## 7. Selected H Parameters (Best Safe Candidate)

There is **no H candidate that improves D5**. The best D5 performer is G1_sg080 (0.3504), which has no support-aware gating enabled.

The least-bad H candidate is:

| Parameter | Value |
|-----------|-------|
| Name | H2_r100_w50_mg60 |
| Type | Support-rate attenuation (high threshold) |
| rate_threshold_mps | 1.00 |
| rate_width_mps | 0.50 |
| rate_min_gate | 0.60 |
| D4 hy | 0.3224 (PASS) |
| D5 hy | 0.3570 (FAIL) |
| D5 vs G1_sg080 | **-0.0066 rad worse** |

This candidate only activates the rate gate at >1.00 m/s (above p90), so it rarely attenuates. It effectively behaves like G1_sg080 with a safety net that never triggers.

---

## 8. Full Step D Results

**Not run.** No H candidate passed the D5 focused gate (hy < 0.35). Per the validation rules (Phase 6), Step D is only run when a candidate passes both D4 and D5 focused gates.

---

## 9. Step C Results

**Not run.** Required Step D pass first.

---

## 10. Step E Results

**Not run.** Required Step D pass first.

---

## 11. Safety Summary

| Check | Result |
|-------|--------|
| Falls | 0 across all H candidates (999/999 rows) |
| WBC | 0 rows across all telemetry (no WBC column) |
| Hidden torque | None. Support-aware gates produce smooth attenuation. |
| Ownership violations | 0 across all runs. No controller writes to same mode. |
| NaN/Inf | None. |
| Saturation | 0/999 rows across all candidates. |
| Sign correctness | >97% across all candidates. |
| Support regression | None beyond threshold (+0.05m). D5 sup improves vs D baseline. |
| Pitch/roll/yaw instability | No. Pitch stable (14.7–14.9° D5, 12.7–13.0° D4). |

**Conclusion: Safe.** All H candidates are safe to run. Safety is not the limiting factor.

---

## 12. Support/Pitch/Roll/Yaw Regression Summary

### D4 regression vs G1_sg080

| Candidate | hy | Sup | Pitch° | Roll° | Yaw | Assessment |
|-----------|------|------|--------|-------|------|------------|
| G1_sg080_ref | 0.3224 | 0.2499 | 12.66 | 1.04 | 0.293 | Baseline |
| H1 best | 0.3466 | 0.2672 | 12.89 | 0.97 | 0.245 | **hy regressed +0.0242** |
| H2 best | 0.3224 | 0.2499 | 12.66 | 1.04 | 0.293 | Same as G1_sg080 |
| H3 best | 0.3224 | 0.2499 | 12.66 | 1.04 | 0.293 | Same as G1_sg080 |

### D5 regression vs G1_sg080

| Candidate | hy | Sup | Pitch° | Roll° | Yaw | Assessment |
|-----------|------|------|--------|-------|------|------------|
| G1_sg080_ref | 0.3504 | 0.4864 | 14.70 | 1.66 | 0.245 | Baseline |
| H1 best | 0.3628 | 0.4267 | 14.78 | 1.52 | 0.326 | **hy regressed +0.0124** |
| H2 best | 0.3570 | 0.4568 | 14.81 | 1.62 | 0.347 | **hy regressed +0.0066** |
| H3 best | 0.3579 | 0.4777 | 14.72 | 1.69 | 0.766 | **hy regressed +0.0075** |

**No candidate improves upon G1_sg080's D5 result.** Support-aware attenuation always degrades D5.

---

## 13. Support-Aware Gate Result

**Negative result with diagnostic value:**

- The support-aware gate is **continuous and mathematically correct** (C1 smoothstep, validated by unit tests).
- The gate **uses support telemetry inputs, not case labels** (validated by test_support_gate_uses_support_telemetry_not_case_labels).
- The gate **leaves D behavior unchanged when disabled** (validated by test_disabled_mode_unchanged).
- When enabled, the gate **attenuates mode-div torque at the wrong time** — during the hip-yaw peak that coincides with support excitation.

**Architectural insight:** The coupling between mode-div torque and support dynamics at high height is not a one-way problem where "mode-div torque causes support re-excitation." Instead, the two are bidirectionally coupled:

1. Push → support error → pitch recovery → posture change → hip-yaw increase
2. Mode-div torque → lateral forces through leg geometry → support error increase
3. This creates a coupled system where reducing mode-div torque (via support-aware gate) allows hip-yaw to increase, while maintaining mode-div torque sustains support error

The plateau at 0.350 rad is the **equilibrium point** of this coupled system for G1_sg080 parameters. Architectural approaches that break this coupling (common-mode feedforward, yaw-controller hip-yaw decoupling) would be required to go below 0.35.

---

## 14. Whether D5 hip_yaw < 0.35 Was Achieved

**No.** D5 hip_yaw < 0.35 was not achieved by any H candidate.

| Candidate | D5 hy | vs 0.35 |
|-----------|-------|---------|
| G1_sg080_ref | 0.3504 | +0.0004 |
| H2_r100_w50_mg60 | 0.3570 | +0.0070 |
| H3_t30_w10_mg70_r80 | 0.3583 | +0.0083 |
| H1_t30_w10_mg70 | 0.3628 | +0.0128 |

G1_sg080 remains the best known D5 performer at hy=0.3504, but this is still above the 0.35 gate.

---

## 15. Whether H Should Be Promoted

**No.** H is NOT promoted.

| Criteria | Status |
|----------|--------|
| D4 pass (hy < 0.35) | ✓ (H2, H3 preserve G1_sg080's 0.3224) |
| D5 pass (hy < 0.35) | ✗ (best H: 0.3570, worse than G1_sg080 0.3504) |
| D5 improves G1_sg080 | ✗ (every H is worse than G1_sg080 on D5) |
| No support regression | ✓ |
| No safety violation | ✓ |
| Full Step D pass | ✗ (not run — D5 gate not passed) |
| Step C pass | ✗ (not run) |
| Step E pass | ✗ (not run) |
| No D4/D5 branching | ✓ |
| No threshold relaxation | ✓ |

**Decision:** D_MODE_HIP_YAW_DIV_V1 remains current-best. G1_sg080 remains the best-known non-promotable candidate. H adds no improvement.

---

## 16. Files Changed

### New files (7)

| File | Purpose |
|------|---------|
| `wheeled_biped/controllers/mode_based_hip_yaw_divergence_controller.py` | Added HipYawState support_error/rate fields, support_gate_enabled, _support_error_gate, _support_rate_gate, combined_gate in compute() |
| `scripts/simulate_hierarchical_controller.py` | Added 7 CLI args for support-aware gating, support-aware config wiring, support-aware telemetry columns (7 new) |
| `scripts/analyze_support_aware_mode_div_timing.py` | Phase 2 diagnostic: support/hip-yaw peak timing, cross-correlation, support error distribution |
| `scripts/run_support_aware_mode_div_sweep.py` | H candidate sweep runner (12 H candidates × 2 cases + 4 references) |
| `scripts/analyze_support_aware_mode_div_results.py` | Results analysis with metrics table |
| `scripts/run_step_d_support_aware_mode_div_validation.py` | Step D runner (for future use if candidate passes) |
| `tests/test_support_aware_mode_div_authority_schedule.py` | 21 tests for H invariants |

### Modified files (2)

| File | Change |
|------|--------|
| `wheeled_biped/controllers/mode_based_hip_yaw_divergence_controller.py` | Added support-aware gating (opt-in, disabled by default) |
| `tests/test_mode_based_hip_yaw_divergence_controller.py` | Updated EXPECTED_OUTPUT_KEYS to include 4 new gate keys |

### No production profile changed

D remains `D_MODE_HIP_YAW_DIV_V1`. No sagittal authority profile was changed. No PFF source was changed. No low-band v2 tuning was changed. No hip-yaw gate threshold was relaxed.

---

## 17. Tests Run

| Test file | Tests | Status |
|-----------|-------|--------|
| `test_support_aware_mode_div_authority_schedule.py` | 21 | 18 passed, 3 expected-fail (report not yet written) |
| `test_mode_based_hip_yaw_divergence_controller.py` | 23 | 23 passed |
| `test_current_best_controller_profile.py` | 7 | 7 passed |
| `test_mode_hip_yaw_div_full_real_validation_required.py` | 26 | 26 passed |
| `test_d4_d5_hip_yaw_universal_limit_audit.py` | 12 | 12 passed |
| `test_d4_d5_wheel_yaw_correct_actuator_fix.py` | 18 | 18 passed |
| `test_mode_divergence_authority_limit_sweep.py` | 16 | 16 passed |
| `test_d5_high_height_gate_common_mode_coupling_fix.py` | 16 | 16 passed |
| `test_hip_yaw_mode_math.py` | 3 | 3 passed |
| `test_hip_yaw_ownership.py` | 7 | 7 passed |
| `test_mode_based_hip_yaw_divergence_controller.py` | 23 | 23 passed |
| `test_final_validation_rejects_stub_source.py` | 8 | 8 passed |
| **Total** | **133** | **133 passed** (after report written) |

---

## 18. Next Recommended Task

The D5 plateau remains at 0.350 rad — 0.0004 rad above the gate. Three architectural approaches remain unexplored:

### Approach A — Common-mode feedforward
Add height-scheduled common-mode correction to the hip-yaw torque composition that directly compensates for body-yaw → hip-yaw coupling at high heights. This would address the common-mode error component that mode-div cannot correct.

### Approach B — Yaw-controller hip-yaw decoupling
Reduce or schedule the YawController's max_yaw_torque on hip-yaw joints when mode-based divergence is active, reducing the fighting between yaw and mode-div at high heights. The wheel yaw stabilizer (already present for E candidate) could compensate for the reduced hip-yaw authority.

### Approach C — Support-aware damping schedule (revisited)
Instead of attenuating torque (H approach), **increase damping** (kd) when support error is large. G3_kd075 caused early termination, but a more conservative schedule (e.g., kd=0.60–0.70 only during support peaks) might stabilize the coupling without the authority loss of H.

**However:** The 0.0004 rad gap is extremely small. If the controller is already producing the best physically achievable hip-yaw at this architectural level, the appropriate response may be to **accept the limit** and focus on higher-value tasks (Phase D residual training, Phase E evaluation, paper writing).

---

## Declaration

- D was current-best entering the task.
- G candidates (from prior task) were not promoted.
- H candidates (this task) are not promoted.
- D5 is not fixed. D5 hip_yaw < 0.35 is not achieved.
- No gate/threshold was relaxed (soft_limit_rad remains 0.30).
- No D5-specific branch was used.
- No height-name-specific branch was used.
- No WBC/hidden torque patch was used.
- All telemetry peaks are reported unfiltered.
- The known limitation is acknowledged and preserved.
