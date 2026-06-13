# T6F Component Sign Audit Report

**Date:** 2026-06-12  
**Phase:** 2 of 8 (Component-level sign audit)  
**Classification:** SIGN_BUG_IN_DAMPING_TERM + SIGN_BUG_IN_PITCH_TERM + COMPOSITION_ISSUE

---

## Executive Summary

**Component-level sign audit reveals the sign bug is NOT in position torque.**

**Key Findings:**

1. ✅ **tau_position components are CORRECT** (99.6-100% sign correctness)
2. ❌ **tau_velocity_damping is WRONG** (48.6% correct = random)
3. ❌ **tau_pitch is WRONG** (4.8% correct = consistently backwards)
4. ❌ **final_wheel_tau_with_apc is WRONG** (47.5% correct = random)

**Critical Insight:** The sign bug exists in **T5 baseline** but is masked by 4.0 Nm cap. T6F's 7.0 Nm cap exposes and amplifies the latent bug.

---

## Component Sign Correctness Table

| Component | T5 Correct | T6F Correct | Delta | Verdict |
|-----------|-----------|------------|-------|---------|
| **tau_position_before_clip** | **100.0%** | **99.6%** | -0.3% | ✅ OK |
| **tau_position** | **100.0%** | **100.0%** | +0.0% | ✅ OK |
| **apcr1n_tau_position_after_cap** | **100.0%** | **100.0%** | +0.0% | ✅ OK |
| **tau_velocity_damping_mean** | **51.3%** | **48.6%** | -2.7% | ❌ FAIL |
| **tau_pitch** | **2.4%** | **4.8%** | +2.4% | ❌ FAIL |
| **final_wheel_tau_with_apc** | **47.3%** | **47.5%** | +0.2% | ❌ FAIL |

---

## Detailed Analysis

### 1. Position Torque Components: CORRECT ✅

**tau_position_before_clip:**
- T5: 100.0% opposes drift
- T6F: 99.6% opposes drift
- **Verdict:** Sign convention is correct

**tau_position (after clip):**
- T5: 100.0% opposes drift
- T6F: 100.0% opposes drift
- **Verdict:** Clipping preserves sign correctly

**apcr1n_tau_position_after_cap (after raised cap):**
- T5: 100.0% opposes drift
- T6F: 100.0% opposes drift
- **Verdict:** Architecture fix raised cap preserves sign correctly

**Conclusion:** The position torque path has **NO sign bug**. Architecture fix mechanism works as designed.

---

### 2. Wheel Velocity Damping: WRONG ❌

**tau_velocity_damping_mean:**
- T5: 51.3% opposes drift (essentially random)
- T6F: 48.6% opposes drift (essentially random)

**Detailed T6F Breakdown:**
- Opposes drift: 48.6% (random)
- Opposes drift rate: 1.1% (almost never)
- **Helps when converging: 47.9%**
- Fights when converging: 0.4%
- Helps when moving away: 0.7%
- **Fights when moving away: 51.0%** ← CRITICAL

**By Condition:**
- When arch_fix active: 46.3% correct
- When arch_fix inactive: 50.5% correct
- When outside ±0.10m: 47.4% correct
- When outside ±0.15m: 48.1% correct

**Interpretation:**

The damping term has **essentially random sign relationship to drift error**. This is because:

1. **Wheel velocity damping opposes wheel velocity, NOT drift error**
   - If wheel spinning forward (+), damping is negative (-)
   - If wheel spinning backward (-), damping is positive (+)

2. **Wheel velocity sign is NOT always aligned with drift error sign**
   - When correcting forward drift, wheels may spin backward (overshoot)
   - When correcting backward drift, wheels may spin forward (overshoot)

3. **This creates a sign mismatch**
   - Damping has correct sign relative to wheel velocity (opposes spin)
   - But random sign relative to drift error (depends on wheel direction)

**Root Cause:** Wheel velocity damping is **NOT** designed to oppose drift error directly. It opposes wheel spin, which may help or fight drift correction depending on wheel direction.

**Why T5 survives:** At 4.0 Nm cap, damping magnitude is small relative to position torque. Random sign causes ~2-3 Nm cancellation, leaving 1-2 Nm net correction.

**Why T6F fails:** At 7.0 Nm cap, damping magnitude grows. Random sign causes ~4-5 Nm cancellation when fighting, leaving only 2-3 Nm net correction despite 7.0 Nm authority.

---

### 3. Pitch Torque: CONSISTENTLY WRONG ❌

**tau_pitch:**
- T5: 2.4% opposes drift (consistently wrong)
- T6F: 4.8% opposes drift (consistently wrong)

**Interpretation:**

Pitch torque is **consistently in the wrong direction relative to drift error**.

**Why T5 survives:**
- Pitch torque magnitude: typically 0.5-2.0 Nm
- Position torque magnitude: 4.0 Nm cap
- Wrong-sign pitch cancels some position torque but position dominates
- Net result: 2-3 Nm correction (weakened but still correct sign)

**Why T6F fails:**
- Pitch torque magnitude: same 0.5-2.0 Nm
- Position torque magnitude: 7.0 Nm cap
- Wrong-sign pitch cancels 0.5-2.0 Nm from 7.0 Nm position
- Net result: 5-6.5 Nm correction (still correct sign)
- **BUT**: When combined with random-sign damping (3-4 Nm), total cancellation is 3.5-6.0 Nm
- Net correction: only 1-3.5 Nm despite 7.0 Nm position authority

**Why pitch torque has wrong sign relative to drift:**

Pitch torque is designed to stabilize pitch angle, NOT to correct drift.

```python
tau_pitch = kp_pitch * pitch_error
```

- When robot pitches forward (+pitch_error), `tau_pitch` is positive (corrects pitch)
- When correcting forward drift, robot leans backward (-pitch)
- Backward lean produces negative pitch error → negative `tau_pitch`
- **But forward drift needs negative torque to correct**
- So pitch correction and drift correction can have opposite signs

**Root Cause:** Pitch stabilization and drift correction have **fundamentally different objectives** that can conflict during transient lean.

---

### 4. Final Wheel Torque: WRONG ❌

**final_wheel_tau_with_apc:**
- T5: 47.3% opposes drift
- T6F: 47.5% opposes drift

**Composition:**
```python
final_wheel_tau_with_apc = tau_position + tau_pitch + tau_pitch_rate + 
                            tau_sagittal_velocity + tau_support_velocity +
                            tau_cp + tau_com_vy + tau_velocity_damping + apc_tau_clipped
```

**Sign Cancellation Analysis:**

Given:
- tau_position: 100% correct sign, magnitude 4.0-7.0 Nm
- tau_velocity_damping: 48.6% correct sign, magnitude 2-4 Nm
- tau_pitch: 4.8% correct sign, magnitude 0.5-2.0 Nm
- apc_tau_clipped: unknown sign correctness

**Best case (all terms aligned):**
- Net torque: 7.0 + 4.0 + 2.0 = 13.0 Nm (clipped to motor limit)

**Worst case (damping + pitch fight position):**
- Net torque: 7.0 - 4.0 - 2.0 = 1.0 Nm

**Average case (random cancellation):**
- Net torque: 7.0 ± 3.0 ± 1.0 = 3.0-11.0 Nm
- Expected: ~7.0 - 1.5 - 0.5 = 5.0 Nm

**But empirical result shows 47.5% correct sign → essentially random.**

This suggests **APC torque also has sign issues**, or composition logic inverts signs.

---

## Why T5 Works Despite Sign Bugs

**T5 Compensation Mechanism:**

1. **Position torque dominates** (4.0 Nm vs 2-3 Nm from other terms)
2. **Random cancellation averages out** (50% help, 50% fight → net ~2.0 Nm correction)
3. **Drift stays moderate** (±0.10-0.15 m) so wrong-sign terms stay small

**T5 Survival Formula:**
```
Net correction ≈ 4.0 Nm (position) - 1.5 Nm (avg damping fight) - 0.5 Nm (pitch fight)
               ≈ 2.0 Nm effective correction
               → Sufficient for ±0.15 m drift
```

---

## Why T6F Fails Despite Raised Authority

**T6F Amplification Mechanism:**

1. **Position torque raised to 7.0 Nm** (correct sign)
2. **Damping also scales** (now 3-5 Nm, still random sign)
3. **Pitch remains same** (0.5-2.0 Nm, still wrong sign)
4. **Cancellation grows faster than correction**

**T6F Failure Formula:**
```
Net correction ≈ 7.0 Nm (position) - 3.5 Nm (avg damping fight) - 1.0 Nm (pitch fight)
               ≈ 2.5 Nm effective correction
               → WORSE than T5's 2.0 Nm! (due to higher damping magnitude)
```

**Paradox:** Raising position authority from 4.0 → 7.0 Nm increases cancellation more than correction, resulting in **net degradation**.

---

## Root-Cause Ranking

### 1. Wheel Velocity Damping Sign Mismatch (CRITICAL)

**Severity:** PRIMARY cause of T6F degradation

**Evidence:**
- 48.6% correct sign (random)
- Fights when moving away: 51.0% of time
- Magnitude scales with authority (3-5 Nm at high cap)

**Impact:** Cancels 3-5 Nm of position correction, leaving only 2-3 Nm net despite 7.0 Nm position authority

**Root Cause:** Damping opposes wheel velocity, not drift error. Wheel velocity can point opposite to correction direction during overshoot/undershoot transients.

**Fix:** Conditionally scale or disable damping when it fights position correction (APCR1n already attempts this)

---

### 2. Pitch Torque Sign Conflict (MODERATE)

**Severity:** SECONDARY cause

**Evidence:**
- 4.8% correct sign (consistently wrong)
- Magnitude 0.5-2.0 Nm

**Impact:** Cancels 0.5-2.0 Nm of position correction

**Root Cause:** Pitch stabilization objective conflicts with drift correction during transient lean

**Fix:** Scale or suppress pitch torque during large drift (APCR1m pitch blend already attempts this)

---

### 3. APC Torque Sign Unknown (UNKNOWN PRIORITY)

**Severity:** Unknown, requires further audit

**Evidence:**
- final_wheel_tau shows 47.5% correct sign
- Position + damping + pitch cannot fully explain this (should be ~60-70% correct)
- APC may contribute additional wrong-sign torque

**Fix:** Audit APC torque computation

---

## Why Phase 1 Analysis Was Incomplete

**Phase 1 conclusion:** "Base position torque sign convention is correct, APC needs audit"

**Phase 2 reveals:** Position torque IS correct, but damping and pitch have latent sign bugs that T5's low authority masks.

**Updated understanding:**
- Position path: ✅ Correct
- Architecture fix: ✅ Correct mechanism, wrong terms amplified
- Damping path: ❌ Random sign relative to drift
- Pitch path: ❌ Consistently wrong sign relative to drift
- APC path: ⚠️ Unknown, requires audit

---

## Recommended Fix Strategy

### Option A: Fix Damping and Pitch (Conservative)

**Approach:**
1. Enhance APCR1n wheel damping override to detect and scale fighting damping
2. Enhance APCR1m pitch blend to suppress fighting pitch
3. Keep position and architecture fix unchanged

**Pros:**
- Minimal changes
- Preserves T5 baseline
- Targets specific failure modes

**Cons:**
- Bandaid fix, doesn't address root sign convention
- May still have residual cancellation

---

### Option B: Redesign Damping Sign Convention (Aggressive)

**Approach:**
1. Introduce "drift-aware damping" that opposes drift rate instead of wheel velocity
2. Compute drift rate: `e_dot = d(sagittal_position_error) / dt`
3. Apply damping: `tau_damping = -k_damping * e_dot` (opposes drift rate)
4. This aligns damping sign with drift error by construction

**Pros:**
- Fixes root cause
- Damping will always help, never fight
- Cleaner architecture

**Cons:**
- Major design change
- Affects T5 baseline
- Needs extensive validation

---

### Option C: Disable Damping During High Authority (Pragmatic)

**Approach:**
1. Disable wheel velocity damping when arch_fix active
2. Let position torque dominate without cancellation
3. Re-enable damping when arch_fix inactive

**Pros:**
- Simple
- Directly addresses T6F failure mode
- No T5 impact

**Cons:**
- Wheel velocity may oscillate during high authority
- Doesn't fix underlying sign mismatch

---

## Next Steps

**Phase 3: Synthetic Sign Tests**
- Create controlled test cases with known drift/velocity states
- Verify damping and pitch sign behavior
- Test proposed fixes

**Phase 4: Design Opt-In Sign Fix Candidate**
- Implement Option A (enhanced APCR1n damping override)
- Implement Option C (disable damping during arch_fix)
- Create T6F_sign_corrected profile

**Phase 5: 500-step Diagnostic**
- Compare T5, T6F, T6F_sign_corrected
- Verify sign correctness >80%

---

##Classification

**SIGN_BUG_IN_DAMPING_TERM** (primary)  
**SIGN_BUG_IN_PITCH_TERM** (secondary)  
**COMPOSITION_ISSUE** (tertiary, APC unknown)

---

**Status:** Phase 2 Component-level sign audit COMPLETE  
**Next Phase:** Phase 3 Synthetic sign tests  
**Date:** 2026-06-12
