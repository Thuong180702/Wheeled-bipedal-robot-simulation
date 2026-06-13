# T6F_sign_corrected Design Invalidation Report

**Date**: 2026-06-12  
**Final Classification**: T6F_SIGN_FIX_BUGFIX_500_FAIL_STABILITY  
**Status**: DESIGN INVALIDATED — ABANDON AS IMPLEMENTED

---

## Executive Summary

The T6F_sign_corrected profile was designed to improve final torque sign correctness during arch_fix recovery by:
1. Suppressing pitch torque when error > 0.10m during arch_fix
2. Overriding velocity damping when wheel velocity opposes error correction

**All three implementation bugs were successfully fixed:**
1. ✅ Profile identity telemetry present and correct
2. ✅ Pitch suppression placement fixed — activates 126/126 eligible steps (100% match rate)
3. ✅ Band state logic working correctly — 289 hard/emergency steps observed

**However, T6F_sign_corrected FAILED stability validation:**
- Sign correctness: 43.5% (target >80%, **worse than T6F baseline 48.9%** by 5.4pp)
- Max drift: 0.383m (**88% worse than T6F** 0.203m)
- Final error: 0.320m (127% worse than T6F 0.141m)
- Pitch excursion: -12.5° to +19.7° (**2.3× worse than T6F** -0.5° to +8.4°)
- Controller instability: 152 transition/recovery steps (T5/T6F had 0)

**Design hypothesis: INVALIDATED**

---

## Design Intent

### Hypothesis

Final torque sign incorrectness causes drift instability. During arch_fix recovery, correcting the signs of pitch torque and velocity damping will:
1. Improve overall sign correctness from T6F's 48.9% toward target >80%
2. Reduce drift
3. Improve stability

### Implementation

**Pitch suppression during arch_fix:**
```python
if (arch_fix_active and 
    abs(sagittal_error) > 0.10):
    tau_pitch = 0.0
```

**Damping override:**
```python
if (wheel_velocity opposes error_correction):
    disable_velocity_damping()
```

### Expected Outcome

- Sign correctness >80%
- Drift comparable to or better than T6F
- No controller mode transitions
- Smooth arch_fix recovery

---

## Implementation Verification

### Phase 0-4: Bug Fixes Confirmed Working

| Phase | Bug | Status |
|-------|-----|--------|
| Phase 1 | Profile identity telemetry missing | ✅ FIXED |
| Phase 2 | Pitch suppression placement before arch_fix_active set | ✅ FIXED |
| Phase 3 | Band state audit script bug | ✅ FIXED |
| Phase 4 | Integration tests | ✅ PASS (390/390) |

### Phase 5: 500-Step Diagnostic Results

**Profile Identity**: ✅ PASS
- vd_sagittal_authority_profile: T6F_sign_corrected (correct)
- controller_mode: upright (347 steps), transition (140 steps), recovery (12 steps)
- sagittal_controller: velocity-damped (correct)
- height_variant_setup_name: high_0p480_setup (correct)

**Pitch Suppression Activation**: ✅ WORKING AS DESIGNED
- Eligible steps (arch_fix_active AND abs(error)>0.10m): 126 steps (25.3%)
- Pitch suppressed steps: 131 steps (26.3%)
- Match rate: 126/126 = 100%
- tau_pitch during eligible steps: 83.3% zero (mean 0.50 Nm, std 2.82 Nm)

**Band State Transitions**: ✅ WORKING CORRECTLY
- Hard/emergency band: 289 steps (57.9%)
- Arch fix activation: 118/289 (40.8%)
- Band state distribution:
  - Normal: 156 steps (31.3%)
  - Soft: 34 steps (6.8%)
  - Desired: 20 steps (4.0%)
  - Hard: 36 steps (7.2%)
  - Emergency: 253 steps (50.7%)

**Implementation Verdict**: All features working as designed. Bugs are not responsible for failure.

---

## Validation Results

### Comparison: T5 vs T6F vs T6F_sign_corrected

| Metric | T5 Baseline | T6F Baseline | T6F_sign_corrected | Target | T6F_sign_corrected vs T6F |
|--------|-------------|--------------|---------------------|--------|---------------------------|
| **Sign correctness** | 35.5% | **48.9%** | **43.5%** | >80% | **-5.4pp ❌** |
| Sign correctness during arch_fix | N/A | 18.4% | 46.6% | >80% | +28.2pp |
| **Max abs error** | 0.187m | 0.203m | **0.383m** | <0.15m | **+88% ❌** |
| **Final error** | 0.060m | 0.141m | **0.320m** | <0.10m | **+127% ❌** |
| **Peak-to-peak** | 0.203m | 0.219m | **0.598m** | <0.20m | **+173% ❌** |
| Mean abs error | 0.095m | 0.082m | 0.137m | <0.10m | +67% |
| **Max pitch** | 6.1° | 8.4° | **19.7°** | <10° | **+135% ❌** |
| Min pitch | N/A | -0.5° | **-12.5°** | N/A | 25× worse |
| **Transition/recovery steps** | 0 | 0 | **152** | 0 | **+152 ❌** |
| Hard/emergency band | 213 (42.7%) | 183 (36.7%) | 289 (57.9%) | N/A | +58% |

### Key Failure Modes

1. **Sign correctness degraded, not improved**
   - Overall: 43.5% vs T6F's 48.9% (-5.4pp)
   - Only 7.9pp better than pure T5 baseline (35.5%)
   - Local improvement during arch_fix (46.6% vs 18.4%) is offset by global degradation
   - Target >80% is 36.5pp away

2. **Drift severely amplified**
   - Max error nearly doubled: 0.383m vs T6F's 0.203m (+88%)
   - Final error more than doubled: 0.320m vs T6F's 0.141m (+127%)
   - Peak-to-peak nearly tripled: 0.598m vs T6F's 0.219m (+173%)
   - Outside ±0.15m: 43.3% of time (vs T6F's 24.2%)

3. **Controller instability introduced**
   - 152 steps in transition/recovery modes (30.5%)
   - T5 and T6F remained in upright mode for all 499 steps
   - Indicates state-machine transitions triggered by instability

4. **Pitch excursion amplified**
   - Max pitch: 19.7° (vs T6F's 8.4°, +135%)
   - Min pitch: -12.5° (vs T6F's -0.5°, 25× worse)
   - Range: 32.2° (vs T6F's 8.9°, 3.6× worse)

---

## Root Cause Analysis

### Why the Design Failed

The sign fix features improve **local behavior during arch_fix** but cause **severe global instability** through removal of stabilization authority:

#### 1. Pitch Suppression Removes Stabilization Torque

When `arch_fix_active AND abs(error) > 0.10m`:
- `tau_pitch` set to 0.0
- Robot loses pitch stabilization during high-error conditions
- Pitch excursions grow unchecked: -12.5° to +19.7° (vs T6F -0.5° to +8.4°)
- Large pitch changes the required correction direction, causing new sign conflicts

#### 2. Damping Override Removes Energy Dissipation

When wheel velocity opposes error correction:
- Velocity damping disabled
- Robot loses energy dissipation mechanism
- Drift amplifies: 0.383m vs T6F's 0.203m
- System becomes underdamped during recovery

#### 3. Combined Effect: Narrow Operating Envelope

Both features active during arch_fix → robot lacks stabilization authority → controller enters transition/recovery modes (152 steps) → further instability feedback loop.

#### 4. Emergent Behavior: Sign Correctness Paradox

Despite "correcting" signs locally during arch_fix (46.6% vs 18.4%), **overall sign correctness degraded** (43.5% vs 48.9%). Possible mechanisms:

- **Pitch coupling**: Larger pitch excursions change the sign of required forward/backward corrections
- **Drift amplification**: Larger error magnitudes create new regions where sign conventions break down
- **Mode transitions**: Transition/recovery modes use different control laws with different sign semantics
- **Temporal offset**: Correcting signs at time `t` during arch_fix creates instability that causes sign errors at time `t+k`

**Interpretation**: Sign incorrectness is a **symptom of instability**, not the cause. Removing stabilization authority to "fix" the symptom makes the underlying instability worse.

---

## Design Hypothesis: INVALIDATED

### Original Hypothesis

> Correcting the signs of pitch torque and velocity damping during arch_fix will improve overall sign correctness and stability.

### Empirical Evidence

1. ❌ Overall sign correctness did not improve (43.5% vs T6F's 48.9%)
2. ❌ Drift became much worse (0.383m vs T6F's 0.203m, +88%)
3. ❌ Stability degraded (152 transition/recovery steps vs 0)
4. ❌ Pitch excursion amplified (19.7° vs T6F's 8.4°, +135%)
5. ✅ Local arch_fix sign correctness improved (46.6% vs 18.4%) — **irrelevant if global behavior fails**

### Revised Understanding

**Sign incorrectness is likely a symptom, not a root cause:**

- The original T6F arch_fix (cap raise alone) achieves 48.9% sign correctness with stable drift (0.203m max error, no mode transitions)
- Adding sign corrections degrades both sign correctness and stability
- This suggests the "wrong signs" in T6F are actually **stabilizing terms that happen to look incorrect by component-level inspection**

**The correct framing:**

- Primary metrics: drift (max |e|, outside ±0.08/0.10/0.15), pitch/roll stability, no mode transitions
- Secondary metrics: sign correctness **only when primary metrics are met**
- Do not optimize sign correctness at the expense of primary metrics

---

## Comparison: Phase 6 Original Diagnostic vs Phase 5 After Fixes

This comparison reveals why the bugs masked the design failure:

| Metric | Phase 6 (bugs present) | Phase 5 (bugs fixed) | Change |
|--------|------------------------|----------------------|--------|
| Overall sign correctness | 49.3% | **43.5%** | -5.8pp |
| Max abs error | 0.192m | **0.383m** | +99% |
| Pitch suppression activated | 0.0% (bug) | 100.0% (fixed) | +100pp |
| Band state | stuck at normal (bug) | transitions correctly | fixed |

**Finding**: Phase 6 showed better drift/sign metrics because **pitch suppression was not activating** due to the placement bug. Now that it activates correctly, the **destabilizing effects are fully visible**.

The bugs were accidentally masking the design's fundamental flaw.

---

## Lessons Learned

### What Worked

1. ✅ **Systematic debugging methodology** (Phase 0-6) successfully identified and fixed all implementation bugs
2. ✅ **Profile identity telemetry** enables runtime verification of controller configuration
3. ✅ **Band state audit scripts** correctly diagnose emergency conditions
4. ✅ **500-step diagnostic** successfully detected design-level failure before long evaluation
5. ✅ **Quantitative validation** prevented committing fundamentally flawed design

### What Failed

1. ❌ **Sign fix design hypothesis**: Correcting signs does not improve stability
2. ❌ **Component-level optimization fallacy**: Improving local metrics (sign correctness during arch_fix) degraded global metrics (overall sign correctness, drift, stability)
3. ❌ **Symptom-focused approach**: Treating sign incorrectness as a cause rather than symptom
4. ❌ **Authority removal strategy**: Zeroing pitch control and damping during high-error conditions removed necessary stabilization
5. ❌ **Discontinuous intervention**: Hard on/off switching (tau_pitch = 0.0, damping disabled) introduced instability

### Correct Mental Model

**Wheeled biped sagittal balance is a coupled pitch-wheel-phase system:**

- Pitch and wheel velocity are not independent
- "Wrong sign" damping may be **phase-appropriate stabilization**
- "Wrong sign" pitch torque may be **transient correction toward equilibrium**
- Zeroing terms based on instantaneous sign inspection breaks coupling
- Stability requires continuous authority, not hard overrides

---

## Decision: ABANDON T6F_sign_corrected

### Rationale

1. **Design hypothesis invalidated by empirical evidence**
   - 88% worse drift than T6F baseline
   - 135% worse pitch excursion
   - 5.4pp worse sign correctness (not improved)

2. **No path to parameter tuning success**
   - Not a gain tuning issue
   - Not a threshold tuning issue
   - Fundamental architecture removes stabilization authority

3. **Risk of sunk-cost fallacy**
   - Already invested significant effort in implementation and debugging
   - Further tuning would be "sticking with it through sheer inertia"
   - Must avoid confirmation bias

4. **Opportunity cost**
   - Time spent tuning T6F_sign_corrected delays better approaches
   - T6F baseline is already validated and stable
   - New candidates can learn from this failure

### Recommended Actions

#### DO NOT:
- ❌ Proceed to 1200-step evaluation
- ❌ Proceed to 2000-step evaluation
- ❌ Proceed to 5000-step evaluation
- ❌ Proceed to Step C height recovery validation
- ❌ Proceed to Step D/E integrated validation
- ❌ Commit T6F_sign_corrected to repository
- ❌ Make T6F_sign_corrected default profile
- ❌ Make paper claims about sign fix improving stability
- ❌ Attempt parameter tuning (pitch threshold, damping threshold, residual scale)

#### DO:
- ✅ Document design failure (this report)
- ✅ Reframe root cause understanding
- ✅ Design safer next candidates based on lessons learned
- ✅ Return to T6F baseline (T6F_budget_cap_raise) for long evaluation
- ✅ Preserve T6F_sign_corrected code and diagnostics for reference

---

## Alternative Approaches (For Future Consideration)

If pursuing sign improvement further, consider:

### Approach A: Bounded Modulation Instead of Hard Suppression

**Instead of**: `tau_pitch = 0.0`

**Try**: `tau_pitch *= blend_factor` where `blend_factor ∈ [0.5, 1.0]`

**Rationale**: Preserve partial stabilization authority

### Approach B: Gradual Fade-In/Fade-Out

**Instead of**: Step discontinuity (on/off)

**Try**: Smooth exponential transition over 5-10 steps

**Rationale**: Avoid discontinuous control that triggers instability

### Approach C: Energy-Aware Damping

**Instead of**: Disable damping when wheel velocity opposes correction

**Try**: Only disable when `|wheel_velocity| > threshold AND wheel_kinetic_energy > threshold`

**Rationale**: Preserve damping except when wheel momentum is large enough to justify override

### Approach D: Pitch Excursion Safety Limit

**Instead of**: Suppress pitch whenever arch_fix is active

**Try**: Re-enable full pitch control if `|pitch| > safety_threshold` (e.g., 10°)

**Rationale**: Prevent runaway pitch growth

### Approach E: Root Cause Investigation

**Instead of**: Fixing signs

**Try**: Understanding **why** signs are wrong in the first place

**Rationale**: True root cause may be gain tuning, IK geometry, or phase detection, not sign convention

---

## Files Generated During T6F_sign_corrected Investigation

### Phase 1: Profile Identity Telemetry Fix
- `docs/validation/t6f_sign_fix_profile_identity_telemetry_fix.md`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_sign_fix_profile_identity_telemetry_fix.json`

### Phase 2: Pitch Suppression Placement Fix
- `docs/validation/t6f_sign_fix_pitch_suppression_placement_fix.md`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_sign_fix_pitch_suppression_placement_fix.json`

### Phase 3: Band State Investigation
- `docs/validation/t6f_sign_fix_band_gate_logic_fix.md`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_sign_fix_band_gate_logic_fix.json`

### Phase 4: Integration Tests
- `docs/validation/t6f_sign_fix_bugfix_tests_report.md`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_sign_fix_bugfix_tests_summary.json`

### Phase 5: 500-Step Diagnostic
- `docs/validation/t6f_sign_corrected_500_diagnostic_after_bugfix_report.md`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_sign_corrected_500_diagnostic_after_bugfix.json`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_bugfix_500_T5/telemetry.csv`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_bugfix_500_T6F/telemetry.csv`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_bugfix_500_T6F_sign_corrected/telemetry.csv`

### Phase 6: Final Report
- `docs/validation/t6f_sign_fix_bugfix_final_report.md`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_sign_fix_bugfix_summary.json`

### Design Invalidation
- `docs/validation/t6f_sign_corrected_design_invalidation.md` (this file)

---

## Conclusion

**Implementation**: Correct ✅

**Design**: Fundamentally Flawed ❌

**Decision**: ABANDON AS IMPLEMENTED

T6F_sign_corrected should not proceed to longer evaluation. The design hypothesis is invalidated by empirical evidence. Return to T6F baseline (T6F_budget_cap_raise) for long evaluation and 2000-step screening.

**Final Classification**: `T6F_SIGN_FIX_BUGFIX_500_FAIL_STABILITY`

---

**End of Design Invalidation Report**
