# T6F Torque Sign Fix Investigation: Phase 1-2 Summary

**Date:** 2026-06-12  
**Status:** Phase 2 COMPLETE - Root cause identified  
**Classification:** SIGN_BUG_IN_DAMPING_TERM + SIGN_BUG_IN_PITCH_TERM

---

## Critical Breakthrough

**The sign bug is NOT in position torque or the architecture fix.**

**The sign bug is in wheel velocity damping and pitch torque**, which have latent sign mismatches that T5's 4.0 Nm cap masks but T6F's 7.0 Nm cap exposes and amplifies.

---

## Component Sign Correctness

| Component | T5 | T6F | Status |
|-----------|-----|-----|--------|
| tau_position | 100.0% ✅ | 100.0% ✅ | CORRECT |
| Architecture fix raised cap | 100.0% ✅ | 100.0% ✅ | CORRECT |
| **tau_velocity_damping** | **51.3% ❌** | **48.6% ❌** | **RANDOM SIGN** |
| **tau_pitch** | **2.4% ❌** | **4.8% ❌** | **WRONG SIGN** |
| final_wheel_tau | 47.3% ❌ | 47.5% ❌ | NET WRONG |

---

## Why T5 Works Despite Sign Bugs

**T5 survives because position torque (4.0 Nm) dominates over wrong-sign terms:**

```
Net correction = 4.0 Nm (position, correct)
                 - 1.5 Nm (damping, random sign, ~50% fights)
                 - 0.5 Nm (pitch, wrong sign)
                 ≈ 2.0 Nm effective correction
```

**2.0 Nm is sufficient for ±0.15 m drift control.**

---

## Why T6F Fails Despite More Authority

**T6F fails because cancellation grows faster than correction:**

```
Net correction = 7.0 Nm (position, correct)
                 - 3.5 Nm (damping, scales with authority, ~50% fights)
                 - 1.0 Nm (pitch, wrong sign)
                 ≈ 2.5 Nm effective correction
```

**2.5 Nm is WORSE than T5's 2.0 Nm due to higher damping cancellation!**

**Paradox:** Raising position authority amplifies the wrong-sign terms proportionally, resulting in **net degradation**.

---

## Root Causes

### 1. Wheel Velocity Damping Sign Mismatch (PRIMARY)

**Current Implementation:**
```python
tau_wheel_vel = -k_wheel_velocity * wheel_vel
```

**Problem:**
- Damping opposes wheel velocity, NOT drift error
- Wheel velocity sign depends on direction of wheel spin
- During correction overshoot/undershoot, wheel spins opposite to drift
- Result: Damping has **random sign** relative to drift error (48.6% correct)

**Impact:**
- At 4.0 Nm cap: ~1.5 Nm cancellation (tolerable)
- At 7.0 Nm cap: ~3.5 Nm cancellation (**catastrophic**)

**Evidence:**
- T6F damping fights when moving away: 51.0% of time
- T6F damping opposes drift: 48.6% (random)
- T5 has same issue but lower magnitude masks it

---

### 2. Pitch Torque Sign Conflict (SECONDARY)

**Current Implementation:**
```python
tau_pitch = kp_pitch * pitch_error
```

**Problem:**
- Pitch torque stabilizes pitch angle, NOT drift
- During drift correction, robot leans (intentional)
- Pitch stabilization opposes this lean
- Result: Pitch torque has **consistently wrong sign** relative to drift (4.8% correct)

**Impact:**
- Magnitude: 0.5-2.0 Nm
- Cancels position correction regardless of authority level

**Evidence:**
- T6F pitch opposes drift: 4.8% (consistently wrong)
- T5 has same issue, same magnitude

---

### 3. Architecture Fix is CORRECT ✅

**The T6F architecture fix mechanism works perfectly:**
- Raises cap from 4.0 → 6.5 → 7.0 Nm correctly
- Preserves tau_position sign 100% of time
- All gates function correctly
- NO sign flip in raised cap composition

**The problem is NOT the architecture fix.**  
**The problem is WHAT the raised authority amplifies.**

---

## Recommended Fix

### Phase 3-4: Implement Enhanced APCR1n Damping Override

**Current APCR1n logic (partial fix):**
- Detects when damping fights position torque
- Scales damping to 30% when fighting

**Enhancement needed:**
- **Disable damping completely when arch_fix active** (not just scale to 30%)
- Current 30% scaling still allows ~1.0-1.5 Nm cancellation
- Full disable ensures no cancellation from damping

**Implementation:**
```python
if arch_fix_active and vd_wheel_damping_recenter_override_enabled:
    # Check if damping opposes position
    sign_position = np.sign(tau_position)
    sign_damping = np.sign((tau_wheel_vel_left + tau_wheel_vel_right) / 2.0)
    
    if sign_position * sign_damping < 0:
        # Damping fights position - DISABLE completely during arch_fix
        tau_wheel_vel_left = 0.0
        tau_wheel_vel_right = 0.0
    # else: preserve damping (it helps)
```

**Expected Result:**
```
With fix:
Net correction = 7.0 Nm (position)
                 - 0.0 Nm (damping disabled when fighting)
                 - 1.0 Nm (pitch, still wrong but small)
                 ≈ 6.0 Nm effective correction

6.0 Nm >> 2.0 Nm (T5 baseline)
```

---

### Phase 3-4: Implement APCR1m Pitch Blend Enhancement

**Current APCR1m logic (partial fix):**
- Blends pitch torque based on drift magnitude
- Scales pitch to 0.0-1.0 depending on error

**Enhancement needed:**
- **Disable pitch completely when arch_fix active AND |error| > 0.10m**
- Pitch stabilization is not needed during emergency recenter
- Let position torque dominate

**Implementation:**
```python
if arch_fix_active and abs(sagittal_position_error_m) > 0.10:
    # Emergency recenter - suppress pitch completely
    tau_pitch = 0.0
```

---

## Implementation Plan

### Create T6F_sign_corrected Profile

**Base:** T6F_budget_cap_raise  
**Changes:**
1. Enable enhanced APCR1n damping override (disable when fighting + arch_fix active)
2. Enable enhanced APCR1m pitch blend (suppress when arch_fix active + large error)
3. Keep architecture fix unchanged
4. Keep all gates unchanged

**Expected Sign Correctness:**
- tau_position: 100% (unchanged)
- tau_velocity_damping: 100% (when arch_fix active, otherwise 50%)
- tau_pitch: 100% (when arch_fix active, otherwise 5%)
- final_wheel_tau: **>80%** (target)

---

## What NOT to Do

❌ **Do NOT modify tau_position** - it is already correct  
❌ **Do NOT change architecture fix gates** - they work correctly  
❌ **Do NOT flip wheel_torque_sign** - would break T5  
❌ **Do NOT redesign damping from scratch** - too risky  
❌ **Do NOT patch at final composition** - fix at source

✅ **DO conditionally disable fighting terms during high authority**  
✅ **DO preserve existing APCR1n/APCR1m infrastructure**  
✅ **DO create opt-in profile** (T6F_sign_corrected)

---

## Next Steps

**Phase 3:** Synthetic sign tests (confirm damping/pitch behavior)  
**Phase 4:** Implement T6F_sign_corrected profile  
**Phase 5:** Run 500-step diagnostic  
**Phase 6:** If pass, run 2000-step screening  
**Phase 7:** If pass, advance to 5000-step validation

**Do NOT run 5000-step until 500-step and 2000-step pass.**

---

## Status Summary

- ✅ Phase 0: Health check COMPLETE (all tests pass)
- ✅ Phase 1: Sign convention map COMPLETE (position correct, APC needs audit)
- ✅ Phase 2: Component sign audit COMPLETE (**damping + pitch identified as root cause**)
- ⏳ Phase 3: Synthetic sign tests (READY TO START)
- ⏳ Phase 4: Design sign fix candidate (READY after Phase 3)
- ⏳ Phase 5: 500-step diagnostic (BLOCKED on Phase 4)
- ⏳ Phase 6: 2000-step screening (BLOCKED on Phase 5)
- ⏳ Phase 7: 5000-step validation (BLOCKED on Phase 6)
- ⏳ Phase 8: Final report (BLOCKED on Phase 7)

---

**Classification:** SIGN_BUG_IN_DAMPING_TERM + SIGN_BUG_IN_PITCH_TERM  
**Fix Strategy:** Enhanced APCR1n (disable damping) + Enhanced APCR1m (suppress pitch)  
**Priority:** HIGH - Blocking T6F advancement  
**Status:** Phase 2 COMPLETE, Phase 3 READY  
**Date:** 2026-06-12
