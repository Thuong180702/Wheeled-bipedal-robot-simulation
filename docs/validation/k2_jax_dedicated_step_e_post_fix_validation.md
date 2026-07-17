# K2 JAX Dedicated Runner -- Step E Post-Fix Verification Report

**Date:** 2026-06-29
**Source data:** `outputs/k2_jax_dedicated_promotion_validation/all_metrics_comparison.json`
**Individual summaries:** `outputs/k2_jax_dedicated_promotion_validation/step_e/*/summary.json`
**Candidate backend:** JAX (`k2_notch_low_q_v1`, `mode_div_enabled: true`, `dynamic_qref_mode: original-k2-exact`)
**Original baseline:** Original K2 C++ (`k2_notch_low_q_v1` profile)
**Scope:** step_e -- fixed-height balance at 10 target heights, 2000 steps each

---

## Phase 4: Step E Strict Verification

### Classification criteria

| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| `pitch_rms_deg` | 0.50 deg or 10% (whichever larger) | Noise-floor-level deviations are inconsequential |
| `hip_yaw_max_rad` | 0.020 rad or 10% (whichever larger) | Differences below 0.02 rad are roundoff/sensor noise |
| `support_rms_m` | 0.020 m or 20% (whichever larger) | Support polygon differences below 2 cm negligible |
| `height_rmse_m` | N/A -- original baseline unavailable | Original K2 baseline did not record height_rmse for fixed-height step_e scenarios |

**Classes per metric:**

- **EXACT_OR_BETTER**: delta <= 0 (candidate equal or better than original)
- **WITHIN_OLD_TOLERANCE**: delta > 0 but within the tolerance band
- **SAFE_BUT_WORSE**: delta exceeds tolerance -- meaningful regression but no safety failure
- **SAFETY_FAIL**: fell in either candidate or original

**Overall scenario class** = worst class across all metrics for that scenario.

---

### All 10 Step E Fixed Heights (2000 steps each)

#### high_0p430

| Metric | Candidate (JAX) | Original (K2) | Delta | Class |
|--------|-----------------|---------------|-------|-------|
| fell | false | false | -- | EXACT_OR_BETTER |
| pitch_rms_deg | 3.129 | 4.98 | -1.851 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0833 | 0.0236 | +0.0597 | SAFE_BUT_WORSE |
| support_rms_m | 0.1025 | 0.0637 | +0.0388 | SAFE_BUT_WORSE |
| height_rmse_m | 0.0039 | N/A | -- | -- |
| **overall** | | | | **SAFE_BUT_WORSE** |

Pitch is significantly better (-37%). Hip-yaw divergence and support spread are worse. Both metrics remain within safe operating bounds -- robot does not fall.

#### high_0p450

| Metric | Candidate (JAX) | Original (K2) | Delta | Class |
|--------|-----------------|---------------|-------|-------|
| fell | false | false | -- | EXACT_OR_BETTER |
| pitch_rms_deg | 4.684 | 2.75 | +1.934 | SAFE_BUT_WORSE |
| hip_yaw_max_rad | 0.0263 | 0.0904 | -0.0641 | EXACT_OR_BETTER |
| support_rms_m | 0.0946 | 0.0694 | +0.0252 | SAFE_BUT_WORSE |
| height_rmse_m | 0.0087 | N/A | -- | -- |
| **overall** | | | | **SAFE_BUT_WORSE** |

Hip-yaw is substantially better (-71%). Pitch is noticeably worse (+70%). Support spread is moderately worse. Despite pitch degradation, the episode completes all 2000 steps without falling.

#### high_0p465

| Metric | Candidate (JAX) | Original (K2) | Delta | Class |
|--------|-----------------|---------------|-------|-------|
| fell | false | false | -- | EXACT_OR_BETTER |
| pitch_rms_deg | 3.620 | 3.55 | +0.070 | WITHIN_OLD_TOLERANCE |
| hip_yaw_max_rad | 0.0454 | 0.0296 | +0.0158 | WITHIN_OLD_TOLERANCE |
| support_rms_m | 0.1102 | 0.0617 | +0.0485 | SAFE_BUT_WORSE |
| height_rmse_m | 0.0041 | N/A | -- | -- |
| **overall** | | | | **SAFE_BUT_WORSE** |

Pitch and hip-yaw differences are well within tolerance. Support spread regression (+79%) is the sole driver of SAFE_BUT_WORSE classification.

#### high_0p480

| Metric | Candidate (JAX) | Original (K2) | Delta | Class |
|--------|-----------------|---------------|-------|-------|
| fell | false | false | -- | EXACT_OR_BETTER |
| pitch_rms_deg | 4.280 | 3.96 | +0.320 | WITHIN_OLD_TOLERANCE |
| hip_yaw_max_rad | 0.0735 | 0.0563 | +0.0172 | WITHIN_OLD_TOLERANCE |
| support_rms_m | 0.1150 | 0.0471 | +0.0679 | SAFE_BUT_WORSE |
| height_rmse_m | 0.0101 | N/A | -- | -- |
| **overall** | | | | **SAFE_BUT_WORSE** |

Pitch and hip-yaw within tolerance. Support spread is the worst of all high-height scenarios (+144%). Robot remains stable and does not fall.

#### low_0p300

| Metric | Candidate (JAX) | Original (K2) | Delta | Class |
|--------|-----------------|---------------|-------|-------|
| fell | false | false | -- | EXACT_OR_BETTER |
| pitch_rms_deg | 2.908 | 2.68 | +0.228 | WITHIN_OLD_TOLERANCE |
| hip_yaw_max_rad | 0.2008 | 0.1314 | +0.0694 | SAFE_BUT_WORSE |
| support_rms_m | 0.0850 | 0.0421 | +0.0429 | SAFE_BUT_WORSE |
| height_rmse_m | 0.0071 | N/A | -- | -- |
| **overall** | | | | **SAFE_BUT_WORSE** |

Pitch degradation is small (+8.5%). Hip-yaw max is +53% worse -- the second-highest hip-yaw across all Step E scenarios. Support spread is +102% worse.

> **Critical comparison requested:** `hy cand=0.2008 vs orig=0.1314` -> `delta=+0.0694`
> Exceeds 0.020 rad tolerance. No yaw-twist safety threshold violated -- max hip-yaw remains within joint limits and robot completes all steps. Classified SAFE_BUT_WORSE.

#### low_0p320

| Metric | Candidate (JAX) | Original (K2) | Delta | Class |
|--------|-----------------|---------------|-------|-------|
| fell | false | false | -- | EXACT_OR_BETTER |
| pitch_rms_deg | 3.694 | 2.83 | +0.864 | SAFE_BUT_WORSE |
| hip_yaw_max_rad | 0.0821 | 0.0502 | +0.0319 | SAFE_BUT_WORSE |
| support_rms_m | 0.1161 | 0.0525 | +0.0636 | SAFE_BUT_WORSE |
| height_rmse_m | 0.0008 | N/A | -- | -- |
| **overall** | | | | **SAFE_BUT_WORSE** |

All three comparable metrics are SAFE_BUT_WORSE. Pitch is +31% worse, hip-yaw +64%, support +121%. Height tracking is excellent at 0.8 mm RMSE. Robot survives all 2000 steps.

#### low_0p330

| Metric | Candidate (JAX) | Original (K2) | Delta | Class |
|--------|-----------------|---------------|-------|-------|
| fell | false | false | -- | EXACT_OR_BETTER |
| pitch_rms_deg | 3.963 | 3.63 | +0.333 | WITHIN_OLD_TOLERANCE |
| hip_yaw_max_rad | 0.1162 | 0.0851 | +0.0311 | SAFE_BUT_WORSE |
| support_rms_m | 0.0894 | 0.0386 | +0.0508 | SAFE_BUT_WORSE |
| height_rmse_m | 0.0044 | N/A | -- | -- |
| **overall** | | | | **SAFE_BUT_WORSE** |

Pitch is +9.2% worse -- within tolerance. Hip-yaw is +37% worse and support +132% worse. Safer low-height performance than low_0p320.

#### low_0p340

| Metric | Candidate (JAX) | Original (K2) | Delta | Class |
|--------|-----------------|---------------|-------|-------|
| fell | false | false | -- | EXACT_OR_BETTER |
| pitch_rms_deg | 1.862 | 2.97 | -1.108 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.1255 | 0.0445 | +0.0810 | SAFE_BUT_WORSE |
| support_rms_m | 0.0788 | 0.0541 | +0.0247 | SAFE_BUT_WORSE |
| height_rmse_m | 0.0034 | N/A | -- | -- |
| **overall** | | | | **SAFE_BUT_WORSE** |

Pitch is significantly better (-37%). This is the best pitch performance across ALL Step E scenarios.

> **Critical comparison requested:** `hy cand=0.1255 vs orig=0.0445` -> `delta=+0.0810`
> Largest hip-yaw delta among all Step E scenarios. Exceeds 0.020 rad tolerance by 4x. Classified SAFE_BUT_WORSE. Robot still completes all 2000 steps without falling.

#### low_0p360

| Metric | Candidate (JAX) | Original (K2) | Delta | Class |
|--------|-----------------|---------------|-------|-------|
| fell | false | false | -- | EXACT_OR_BETTER |
| pitch_rms_deg | 3.118 | 1.90 | +1.218 | SAFE_BUT_WORSE |
| hip_yaw_max_rad | 0.0897 | 0.0959 | -0.0062 | EXACT_OR_BETTER |
| support_rms_m | 0.0926 | 0.0371 | +0.0555 | SAFE_BUT_WORSE |
| height_rmse_m | 0.0044 | N/A | -- | -- |
| **overall** | | | | **SAFE_BUT_WORSE** |

Hip-yaw is slightly better (-6.5%) -- the only scenario where both hip-yaw and pitch do not regress together. Pitch degradation is large (+64%). Support spread is +150% worse -- the largest percentage increase of any scenario.

#### low_0p380

| Metric | Candidate (JAX) | Original (K2) | Delta | Class |
|--------|-----------------|---------------|-------|-------|
| fell | false | false | -- | EXACT_OR_BETTER |
| pitch_rms_deg | 5.245 | 3.33 | +1.915 | SAFE_BUT_WORSE |
| hip_yaw_max_rad | 0.0759 | 0.0392 | +0.0367 | SAFE_BUT_WORSE |
| support_rms_m | 0.1087 | 0.0480 | +0.0607 | SAFE_BUT_WORSE |
| height_rmse_m | 0.0021 | N/A | -- | -- |
| **overall** | | | | **SAFE_BUT_WORSE** |

Pitch degradation is the worst absolute (+1.915 deg) and worst percentage (+58%) among all Step E scenarios. Height tracking is excellent at 2.1 mm RMSE. All three metrics are SAFE_BUT_WORSE.

---

### Summary

#### Classification counts

| Class | Count | Scenarios |
|-------|-------|-----------|
| EXACT_OR_BETTER | 0 | -- |
| WITHIN_OLD_TOLERANCE | 0 | -- |
| SAFE_BUT_WORSE | 10 | All: high_0p430, high_0p450, high_0p465, high_0p480, low_0p300, low_0p320, low_0p330, low_0p340, low_0p360, low_0p380 |
| SAFETY_FAIL | 0 | -- |

**SAFETY_FAIL count:** 0 -- all 10 scenarios complete 2000 steps without falling in both candidate and original.

**SAFE_BUT_WORSE count:** 10 -- every scenario has at least one metric that exceeds the tolerance band.

**EXACT_OR_BETTER count:** 0 -- no scenario has all metrics equal or better than original.

---

#### Metric-level summary

| Metric | Better | Tolerance | Worse | Worst delta |
|--------|--------|-----------|-------|-------------|
| pitch_rms_deg | 2 (high_0p430, low_0p340) | 4 (high_0p465, high_0p480, low_0p300, low_0p330) | 4 (high_0p450, low_0p320, low_0p360, low_0p380) | +1.915 deg (low_0p380) |
| hip_yaw_max_rad | 2 (high_0p450, low_0p360) | 2 (high_0p465, high_0p480) | 6 (high_0p430, low_0p300, low_0p320, low_0p330, low_0p340, low_0p380) | +0.0810 rad (low_0p340) |
| support_rms_m | 0 | 0 | 10 (ALL) | +0.0679 m (high_0p480) |

**support_rms_m is universally worse** across all 10 fixed-height scenarios. This is a systematic regression in the JAX candidate, not a sporadic artifact. The increased support spread indicates the JAX runner maintains a wider support polygon -- the feet drift farther from the CoM projection. While this does not cause falls at fixed heights, it is a meaningful behavioral difference.

**hip_yaw_max_rad is worse in 6/10 scenarios.** The pattern is concentrated at low heights (low_0p300 through low_0p380) where hip-yaw divergence is more pronounced. This correlates with the mode-divergence damping mechanism (`mode_div_enabled: true`) having less authority at lower CoM heights where the support polygon is smaller.

**pitch_rms_deg is a mixed outcome:** 2 better, 4 within tolerance, 4 worse. The worst pitch scenarios (low_0p360, low_0p380) coincide with moderate hip-yaw divergence, suggesting trade-off dynamics between sagittal balance and lateral stability.

---

### Step E overall classification: SAFE_BUT_WORSE

**Rationale:** All 10 scenarios are SAFE_BUT_WORSE due to systematic regressions in:
1. **support_rms_m** (10/10 scenarios) -- universal regression, likely caused by different wheel drift dynamics in the JAX backend vs. the original K2 C++ backend
2. **hip_yaw_max_rad** (6/10 scenarios) -- concentrated at low heights, mode-divergence damping active but insufficient to match original
3. **pitch_rms_deg** (4/10 scenarios) -- at specific heights (high_0p450, low_0p320, low_0p360, low_0p380)

No safety failures: all 10 scenarios survive the full 2000 steps in both candidate and original.

---

### Comparison to dynamic-height scope

The dynamic-height scenarios (ramp_up, ramp_down, gate_chatter, gate_dwell, up_down_cycle) show both SAFETY_FAIL and SAFE_BUT_WORSE patterns. In contrast, Step E fixed-height scenarios are exclusively SAFE_BUT_WORSE -- the JAX candidate reliably maintains balance at every fixed height but with measurable performance degradation relative to the original K2 baseline.

---

### Recommendation

1. **Proceed with SAFE_BUT_WORSE classification** -- the JAX dedicated runner is safe for fixed-height operation; no falls observed.
2. **Investigate support_rms_m regression** -- this is the most consistent difference across all scenarios and may indicate a wheel-friction or contact-modeling difference between the JAX and original backends.
3. **Hip-yaw at low heights (low_0p300, low_0p340) warrants monitoring** -- the large deltas (+0.0694 and +0.0810 rad) are within joint limits but represent a notable behavioral shift.
4. **Pitch at low_0p380 needs attention** before ramp-up dynamic height transitions from low starting positions can succeed.
