# T6F_sign_corrected 500-Step Diagnostic After Bug Fixes - Phase 5

**Date**: 2026-06-12  
**Task**: Phase 5 - 500-step diagnostic rerun after bug fixes  
**Classification**: T6F_SIGN_FIX_BUGFIX_500_FAIL_STABILITY

---

## Executive Summary

After fixing all three bugs identified in Phase 6 root cause investigation:
- ✅ **Bug 1 (Pitch suppression placement)**: FIXED - pitch suppression now activates correctly (126/126 eligible steps)
- ✅ **Bug 2 (Band state audit script)**: FIXED - band state transitions working correctly (289 hard/emergency steps)
- ✅ **Bug 3 (Profile identity telemetry)**: FIXED - all identity fields present and correct

**However, T6F_sign_corrected FAILED stability validation:**
- ❌ Sign correctness: 43.5% (target >80%, **worse than T6F baseline 48.9%**)
- ❌ Drift: max error 0.383m, final error 0.320m (**88% worse than T6F baseline**)
- ❌ Controller mode instability: 140 transition steps, 12 recovery steps (T5/T6F had 0)
- ❌ Pitch excursion: -12.5° to +19.7° (T6F: -0.5° to +8.4°)

**Verdict**: The sign fix implementation causes severe stability degradation. Bugs are fixed, but the **design hypothesis is invalidated**.

---

## Test Setup

**Profiles tested**:
1. **T5** (APCR1nD_T5_band_limited_balanced) - baseline without arch fix
2. **T6F** (T6F_budget_cap_raise) - baseline with arch fix but no sign fix
3. **T6F_sign_corrected** - arch fix with sign fix (damping override + pitch suppression)

**Configuration**:
- Height variant: high_0p480_setup (0.480m target CoM Z)
- Steps: 500 (5.0 seconds)
- Telemetry decimation: 1 (full resolution)
- Failure window: 500 steps

**All three runs completed without falling (terminated=False).**

---

## Phase 5C: Profile Identity Verification

### T5 (APCR1nD_T5_band_limited_balanced)

✅ **Profile identity: PASS**

| Field | Value |
|-------|-------|
| vd_sagittal_authority_profile | APCR1nD_T5_band_limited_balanced |
| controller_mode | upright (499/499 steps) |
| sagittal_controller | velocity-damped |
| height_variant_setup_name | high_0p480_setup |

### T6F (T6F_budget_cap_raise)

✅ **Profile identity: PASS**

| Field | Value |
|-------|-------|
| vd_sagittal_authority_profile | T6F_budget_cap_raise |
| controller_mode | upright (499/499 steps) |
| sagittal_controller | velocity-damped |
| height_variant_setup_name | high_0p480_setup |

### T6F_sign_corrected

✅ **Profile identity: PASS**

| Field | Value |
|-------|-------|
| vd_sagittal_authority_profile | T6F_sign_corrected |
| controller_mode | upright (347), transition (140), recovery (12) |
| sagittal_controller | velocity-damped |
| height_variant_setup_name | high_0p480_setup |

⚠️ **WARNING**: T6F_sign_corrected had **152 steps in transition/recovery modes** (30.5%), while T5 and T6F remained in upright mode for all 499 steps. This indicates controller instability.

---

## Phase 5D: Activation Verification

### 1. Pitch Suppression Activation (T6F_sign_corrected only)

✅ **Pitch suppression placement fix: WORKING**

| Metric | Count | Percentage |
|--------|-------|------------|
| arch_fix_active steps | 131 | 26.3% |
| abs(error) > 0.10m steps | 301 | 60.3% |
| **Eligible steps** (both conditions) | **126** | **25.3%** |
| **Pitch suppressed steps** | **131** | **26.3%** |
| Both eligible AND suppressed | 126 | 100.0% match |
| Eligible but NOT suppressed | 0 | 0.0% |

**Verification**: Pitch suppression now activates **exactly when eligible** (126/126 = 100%). Bug 1 fix confirmed.

**tau_pitch during eligible steps**:
- Mean: 0.50 Nm
- Std: 2.82 Nm
- Min: -8.29 Nm, Max: 7.97 Nm
- **Zero count: 105/126 (83.3%)**

The pitch suppression is working: 83.3% of eligible steps have tau_pitch ≈ 0, confirming suppression is active.

### 2. Band State Transitions

✅ **Band state transitions: WORKING**

**T5 (baseline without arch fix)**:

| Band State | Count | Percentage |
|------------|-------|------------|
| Normal (0) | 99 | 19.8% |
| Soft (1) | 118 | 23.6% |
| Desired (2) | 69 | 13.8% |
| Hard (3) | 87 | 17.4% |
| Emergency (4) | 126 | 25.3% |
| **Hard + Emergency** | **213** | **42.7%** |

Arch fix activation during hard/emergency: 0/213 (0.0%) - expected, T5 does not have arch fix.

**T6F (baseline with arch fix)**:

| Band State | Count | Percentage |
|------------|-------|------------|
| Normal (0) | 250 | 50.1% |
| Soft (1) | 42 | 8.4% |
| Desired (2) | 24 | 4.8% |
| Hard (3) | 22 | 4.4% |
| Emergency (4) | 161 | 32.3% |
| **Hard + Emergency** | **183** | **36.7%** |

Arch fix activation during hard/emergency: 183/183 (100.0%) - working correctly.

**T6F_sign_corrected**:

| Band State | Count | Percentage |
|------------|-------|------------|
| Normal (0) | 156 | 31.3% |
| Soft (1) | 34 | 6.8% |
| Desired (2) | 20 | 4.0% |
| Hard (3) | 36 | 7.2% |
| Emergency (4) | 253 | 50.7% |
| **Hard + Emergency** | **289** | **57.9%** |

Arch fix activation during hard/emergency: 118/289 (40.8%)

⚠️ **CRITICAL**: T6F_sign_corrected entered hard/emergency band **57.9%** of the time (vs T6F 36.7%, T5 42.7%). Despite this, arch_fix only activated 40.8% of the time during hard/emergency, indicating **safety gate failures** or other blocking conditions.

**Verification**: Band state transitions are working correctly (Phase 3 audit script fix confirmed). The controller logic correctly identifies emergency conditions based on error thresholds.

### 3. High Authority Transmission

❌ **High authority transmission analysis incomplete**: telemetry field `tau_position_after_upstream_clip` not found in analysis. Manual inspection needed.

However, from run logs:
- **T5**: Max final torque: 8.00 Nm
- **T6F**: Max final torque: 14.26 Nm
- **T6F_sign_corrected**: Max final torque: 27.08 Nm (knee torque, not wheel)

---

## Phase 5E: Sign Correctness and Drift Analysis

### Sign Correctness Comparison

| Profile | Overall Sign Correctness | During arch_fix | High Torque (>4.0 Nm) |
|---------|--------------------------|-----------------|------------------------|
| **T5 (baseline)** | **35.5%** | N/A | N/A |
| **T6F (baseline)** | **48.9%** | 18.4% | N/A |
| **T6F_sign_corrected** | **43.5%** | 46.6% | 47.1% |
| **Target** | **>80%** | **>80%** | **>80%** |

**Key Findings**:

1. ❌ **T6F_sign_corrected overall sign correctness (43.5%) is WORSE than T6F baseline (48.9%)** by 5.4 percentage points
2. ✅ Sign correctness during arch_fix improved: 46.6% vs T6F's 18.4% (+28.2pp)
3. ❌ Overall sign correctness still **36.5pp below target** (80%)
4. ❌ T6F_sign_corrected is only 7.9pp better than T5 baseline (35.5%), despite adding sign fix features

**Conclusion**: The sign fix improves behavior **during arch_fix** but **degrades overall performance**. The design hypothesis that fixing pitch/damping signs would improve global sign correctness is **invalidated**.

### Drift Comparison

| Profile | Max Abs Error | Final Error | Peak-to-Peak | Mean Abs Error |
|---------|---------------|-------------|--------------|----------------|
| **T5** | 0.187 m | 0.060 m | 0.203 m | 0.095 m |
| **T6F** | 0.203 m | 0.141 m | 0.219 m | 0.082 m |
| **T6F_sign_corrected** | **0.383 m** | **0.320 m** | **0.598 m** | **0.137 m** |
| **Degradation vs T6F** | **+88%** | **+127%** | **+173%** | **+67%** |

**Key Findings**:

1. ❌ **Max abs error doubled** from T6F (0.203m) to T6F_sign_corrected (0.383m)
2. ❌ **Final error more than doubled** from T6F (0.141m) to T6F_sign_corrected (0.320m)
3. ❌ **Peak-to-peak nearly tripled** from T6F (0.219m) to T6F_sign_corrected (0.598m)
4. ❌ All drift metrics are **worse than both baselines** (T5 and T6F)

**Excursion counts**:

| Threshold | T5 | T6F | T6F_sign_corrected |
|-----------|-------|-------|---------------------|
| Outside ±0.08m | 301 (60.3%) | 225 (45.1%) | **327 (65.5%)** |
| Outside ±0.10m | 226 (45.3%) | 195 (39.1%) | **301 (60.3%)** |
| Outside ±0.15m | 89 (17.8%) | 121 (24.2%) | **216 (43.3%)** |

T6F_sign_corrected spent **43.3%** of time with error >0.15m, compared to T6F's 24.2%.

**Conclusion**: The sign fix causes **severe drift degradation**, making the robot significantly less stable than both baselines.

### Stability Metrics

**Pitch excursion (degrees)**:

| Profile | Max Abs Pitch | RMS Pitch |
|---------|---------------|-----------|
| T5 | 6.1° | N/A |
| T6F | 8.4° | N/A |
| T6F_sign_corrected | **19.7°** | N/A |

T6F_sign_corrected had pitch excursions from **-12.5° to +19.7°**, far worse than T6F's -0.5° to +8.4°.

**CoM height**:

| Profile | Min CoM Z | Mean CoM Z | Max CoM Z |
|---------|-----------|------------|-----------|
| T5 | 0.481 m | 0.490 m | 0.491 m |
| T6F | 0.481 m | 0.490 m | 0.492 m |
| T6F_sign_corrected | **0.474 m** | 0.490 m | **0.496 m** |

T6F_sign_corrected had larger CoM height variation: 0.022m vs T6F's 0.011m.

**Wheel velocity**: Data incomplete in analysis.

---

## Phase 5F: Classification

**Classification**: `T6F_SIGN_FIX_BUGFIX_500_FAIL_STABILITY`

### Pass/Fail Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Profile identity telemetry valid | ✅ PASS | vd_sagittal_authority_profile correct for all 3 profiles |
| Pitch suppression activates when eligible | ✅ PASS | 126/126 eligible steps suppressed (100%) |
| Band state enters hard/emergency correctly | ✅ PASS | 289 hard/emergency steps (57.9%) |
| Arch fix activates when gates pass | ⚠️ PARTIAL | 118/289 hard/emergency (40.8%), safety gate blocking some |
| High authority >4.0 Nm transmitted | ⚠️ UNKNOWN | Analysis incomplete, manual check needed |
| **Sign correctness >80%** | ❌ **FAIL** | **43.5% (36.5pp below target)** |
| **Sign correctness improves vs T6F** | ❌ **FAIL** | **43.5% vs T6F 48.9% (5.4pp worse)** |
| **No fall** | ✅ PASS | All 3 profiles completed 500 steps |
| **No severe drift worse than T6F** | ❌ **FAIL** | **Max error 0.383m vs T6F 0.203m (+88%)** |
| No WBC/hidden/ownership violation | ⚠️ UNKNOWN | Not checked in analysis |

### Failure Mode

**Primary failure**: STABILITY DEGRADATION

The sign fix implementation causes:
1. **Worse overall sign correctness** than T6F baseline (43.5% vs 48.9%)
2. **Severe drift amplification** (2-3x worse across all metrics)
3. **Controller mode instability** (152 transition/recovery steps vs 0 for baselines)
4. **Pitch excursion amplification** (19.7° vs T6F 8.4°)

### Root Cause Hypothesis

The sign fix features (damping override + pitch suppression) **improve local behavior during arch_fix** (46.6% vs 18.4% sign correctness) but **destabilize global controller behavior**:

1. **Pitch suppression removes stabilization torque** during high-error conditions, allowing larger pitch excursions
2. **Damping override removes velocity damping** when wheel velocity opposes error correction, reducing energy dissipation
3. **Both features active during arch_fix** create a **narrow operating envelope** where the robot lacks stabilization authority
4. **Controller enters transition/recovery modes** more frequently, indicating state-machine transitions triggered by instability

The design hypothesis—that correcting component signs would improve overall stability—is **invalidated**. The sign corrections create **new instabilities** that outweigh local improvements.

---

## Comparison vs Phase 6 Original Diagnostic

**Phase 6 (before fixes)** - T6F_sign_corrected at high_0p480 (500 steps):
- Overall sign correctness: 49.3%
- Max abs error: 0.192m
- Pitch suppression activated: 0.0%
- Band state: stuck at normal (audit script bug)

**Phase 5 (after fixes)** - T6F_sign_corrected at high_0p480 (500 steps):
- Overall sign correctness: 43.5% (**5.8pp worse**)
- Max abs error: 0.383m (**99% worse**)
- Pitch suppression activated: 100.0% (**fixed**)
- Band state: transitions correctly (**fixed**)

**Finding**: Fixing the bugs revealed that **the sign fix implementation degrades stability**. The Phase 6 diagnostic showed better drift/sign metrics because **pitch suppression was not activating** (bug 1). Now that it activates correctly, the **destabilizing effects are visible**.

---

## Next Steps

### DO NOT PROCEED WITH:
- ❌ 1200-step evaluation
- ❌ 2000-step evaluation
- ❌ 5000-step evaluation
- ❌ Step C validation
- ❌ Step D validation
- ❌ Commit T6F_sign_corrected
- ❌ Paper claims about sign fix improving stability

### REQUIRED ACTIONS:

1. **Abandon T6F_sign_corrected as-is**: The implementation is fundamentally unstable
2. **Re-evaluate design hypothesis**: Sign corrections during arch_fix may not be the right approach
3. **Consider alternative hypotheses**:
   - a. Sign incorrectness is a **symptom, not a cause** of instability
   - b. Removing pitch/damping authority during recovery **removes necessary stabilization**
   - c. The correct fix may be **different position/velocity gains**, not sign overrides
4. **If pursuing sign fix further**:
   - Add **gradual fade-in/fade-out** for pitch suppression (avoid step discontinuities)
   - Add **energy-aware damping override** (only disable when wheel velocity exceeds threshold)
   - Add **pitch excursion safety limit** (re-enable pitch control if |pitch| > threshold)
   - Test with **shorter activation windows** or **weaker suppression**
5. **Document design invalidation**: Update design document to record that sign override approach failed stability validation

---

## Files Generated

- `docs/validation/t6f_sign_corrected_500_diagnostic_after_bugfix_report.md` (this file)
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_sign_corrected_500_diagnostic_after_bugfix.json`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_bugfix_500_T5/telemetry.csv`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_bugfix_500_T6F/telemetry.csv`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_bugfix_500_T6F_sign_corrected/telemetry.csv`

---

## Conclusion

**All three implementation bugs have been fixed:**
1. ✅ Profile identity telemetry present
2. ✅ Pitch suppression activates correctly
3. ✅ Band state transitions correctly

**However, T6F_sign_corrected FAILS stability validation:**
- Sign correctness: 43.5% (worse than T6F 48.9%, target >80%)
- Drift: 0.383m max error (88% worse than T6F 0.203m)
- Stability: 152 transition/recovery steps (T5/T6F had 0)

**The sign fix design hypothesis is invalidated.** The implementation improves local behavior during arch_fix but causes **severe global instability**. T6F_sign_corrected should **not proceed to longer evaluation** without fundamental redesign.

**Final Classification**: `T6F_SIGN_FIX_BUGFIX_500_FAIL_STABILITY`
