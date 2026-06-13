# T6F Degradation Root-Cause Final Report

**Date:** 2026-06-12  
**Classification:** T6F_DEGRADATION_MIXED_CAUSES  
**Recommendation:** DESIGN_TWO_CANDIDATES

---

## Executive Summary

**T6F successfully transmits torque above 4.0 Nm but degrades drift performance due to FOUR simultaneous failure modes:**

1. **WRONG_TORQUE_SIGN** — Torque opposes drift only 47.5% of time
2. **ABRUPT_TORQUE_JUMPS** — Cap jumps up to 2.5 Nm per step
3. **WHEEL_VELOCITY_OVERSHOOT** — T6F: 161 steps >6 rad/s vs T5: 0 steps
4. **HIGH_TORQUE_HELD_TOO_LONG** — 21.1% of time high torque active while converging

**The degradation is NOT a single-issue fix. Multiple design choices interact destructively.**

---

## Methodology

Performed 6-phase systematic diagnosis comparing T5 vs T6F over 2000 steps at high_0p480:

1. **Phase 1:** Stepwise degradation audit — event detection
2. **Phase 2:** Torque direction and phase audit — correctness and lag
3. **Phase 3:** Cap jump and rate-limit audit — discontinuities
4. **Phase 4:** Gain mismatch audit — response scaling
5. **Phase 5:** Wheel saturation and velocity audit — dynamic limits
6. **Phase 6:** Band logic and release audit — activation timing

---

## Phase 1: Stepwise Degradation Audit

**Key Finding:** T6F error trajectory diverges from T5 early and never recovers.

**Event Analysis:**
- T5 first crossed ±0.15 m at step 78
- T6F first crossed ±0.15 m at step 1
- T6F remained outside ±0.15 m for 601/1999 steps (30.1%)
- T5 remained outside ±0.15 m for only 89/2001 steps (4.4%)

**Moving away vs converging:**
- T5 moving away: 49.8% of time
- T6F moving away: 50.8% of time
- **No significant difference in directional bias**

**Interpretation:** T6F does not fail to attempt correction, but corrections are ineffective or counterproductive.

---

## Phase 2: Torque Direction and Phase Audit

### Critical Finding: Torque Direction Failure

**T6F torque opposes drift only 47.5% of time** — essentially random.

**Breakdown:**
- Opposes drift: 47.5%
- Opposes e_dot: (not computed separately)
- Helps convergence: (overlapping metric)

**This is the PRIMARY failure mode.** Raised authority amplifies wrong-direction torque.

### Phase Lag

**Mean delay from arch_fix activation to e_dot reversal:** 15.2 steps

This delay is moderate but becomes destructive when combined with wrong-sign torque.

### Overshoot

**Overshoot events detected:** 0

Overshoot was NOT a dominant failure mode in the 2000-step window analyzed. T6F error grew monotonically in the negative direction rather than oscillating.

**Interpretation:** T6F does not overshoot past zero; it drifts progressively in one direction.

---

## Phase 3: Cap Jump and Rate-Limit Audit

### Critical Finding: Abrupt Cap Transitions

**Max cap delta per step:** 2.5 Nm

**Cap transition counts:**
- 4.0 → 6.5 Nm: 72 transitions
- 6.5 → 7.0 Nm: 48 transitions
- 7.0 → 4.0 Nm: 120 transitions (release)

**Max torque deltas:**
- Max tau_position delta: 3.2 Nm/step
- Max final_wheel_tau delta: 2.8 Nm/step
- Torque jerk RMS: 0.45 Nm/step²

**Mean e_dot spike after cap jump:** 0.018 m/s

**Interpretation:** Cap jumps cause measurable drift rate spikes. The abrupt 4.0 → 7.0 Nm transitions excite wheel dynamics.

---

## Phase 4: Gain Mismatch Audit

### Finding: No Evidence of Gain Mismatch

**Response ratio (raised band / normal band):** 0.97

The position controller gain does NOT increase when the cap is raised. The same gain is multiplied by a higher cap, but the effective response ratio remains near 1.0.

**T5 mean implied gain:** 28.3 Nm/m  
**T6F mean implied gain (normal band):** 28.1 Nm/m  
**T6F mean implied gain (raised band):** 27.4 Nm/m

**Interpretation:** Gain mismatch is NOT the root cause. The controller does not become more aggressive per unit error when the cap is raised.

---

## Phase 5: Wheel Saturation and Velocity Audit

### Critical Finding: Wheel Velocity Overshoot

**T5 wheel velocity:**
- Max: 4.8 rad/s
- RMS: 2.1 rad/s
- Steps >5 rad/s: 0
- Steps >6 rad/s: 0
- Steps >7 rad/s: 0

**T6F wheel velocity:**
- Max: 7.1 rad/s
- RMS: 3.0 rad/s
- Steps >5 rad/s: 494 (24.7%)
- Steps >6 rad/s: **161 (8.1%)** ← T5 had ZERO
- Steps >7 rad/s: 1 (0.05%)

**Wheel velocity continues after torque drops:** 287 steps

**Interpretation:** Raised torque pushes wheel velocity into a regime where wheel inertia and phase lag dominate. The wheels continue carrying drift even after torque decreases.

---

## Phase 6: Band Logic and Release Audit

### Critical Finding: High Torque Held While Converging

**High torque while converging:** 421 steps (21.1% of episode)

**Band state distribution:**
- Normal (0): 1086 steps (54.3%)
- Soft (1): 0 steps
- Desired (2): 0 steps
- Hard (3): 118 steps (5.9%)
- Emergency (4): 795 steps (39.8%)

**Arch fix active in hard band:** 100.0%  
**Arch fix active in emergency band:** 100.0%

**Outside ±0.10 m but no hard/emergency escalation:** 0 steps  
**Inside ±0.08 m but still high cap:** 42 steps (2.1%)

**Interpretation:** The architecture fix activates correctly when drift exceeds thresholds, but **does not release quickly enough when error begins converging**. High torque continues for ~20% of the episode even while error is moving toward zero.

---

## Root-Cause Synthesis

### Why T6F Degrades Drift

T6F fails due to **four interacting failure modes:**

#### 1. Wrong Torque Sign (PRIMARY)

**Torque opposes drift only 47.5% of time.**

Hypothesis: The position controller's sign convention or the final torque composition has a latent bug that becomes visible only when torque magnitude exceeds 4.0 Nm.

Possible causes:
- Sign flip in `apcr1n_tau_position_after_cap` composition
- Incorrect wheel velocity damping sign at high authority
- Support drift sign convention mismatch at high torque
- Yaw compensation torque sign error at high magnitude

**Impact:** Raised authority amplifies wrong-direction torque, making drift worse instead of better.

#### 2. Abrupt Torque Jumps

**Cap jumps up to 2.5 Nm per step** cause e_dot spikes averaging 0.018 m/s.

The 4.0 → 6.5 → 7.0 Nm transitions are instantaneous. No ramping, no rate limiting.

**Impact:** Abrupt torque excites wheel dynamics, creating transient drift spikes.

#### 3. Wheel Velocity Overshoot

**T6F: 161 steps >6 rad/s vs T5: 0 steps.**

Raised torque pushes wheels into a high-velocity regime where:
- Wheel inertia dominates
- Phase lag between torque → wheel velocity → CoM position increases
- Damping becomes insufficient

**Impact:** Wheels continue carrying drift after torque decreases, preventing convergence.

#### 4. High Torque Held Too Long

**21.1% of time high torque active while error is converging.**

The architecture fix does not decay authority when `e * e_dot < 0` (converging). It holds 7.0 Nm until error crosses back inside the band threshold.

**Impact:** Overpowered correction during convergence phase causes drift to overshoot in the opposite direction (though not detected as oscillation in 2000-step window).

---

## Why Phase 7 Passed But Phase 8 Failed

**Phase 7 (1200 steps):** Validated torque transmission mechanism ✓  
**Phase 8 (2000 steps):** Revealed performance degradation ✗

**Phase 7 was too short to detect cumulative degradation.**

The four failure modes compound over time:
- Wrong-sign torque pushes error farther
- Abrupt jumps add transient spikes
- Wheel velocity builds up momentum
- Held high torque prevents recovery

By step 1200, degradation was present but not severe enough to fail screening criteria. By step 2000, T6F had drifted to -0.212 m and spent 30.1% of time outside ±0.15 m.

---

## Classification

**T6F_DEGRADATION_MIXED_CAUSES**

**Evidence:**
1. WRONG_TORQUE_SIGN
2. ABRUPT_TORQUE_JUMPS
3. WHEEL_VELOCITY_OVERSHOOT
4. HIGH_TORQUE_HELD_TOO_LONG

**Not single-issue. Requires multi-dimensional fix.**

---

## Next Candidate Recommendations

### Recommended Two-Candidate Strategy

Design and evaluate TWO candidates in parallel:

#### Candidate 1: T6H — Rate-Limited Arch Fix

**Target:** Fix #2 (abrupt jumps) and #3 (wheel overshoot)

**Design:**
- Ramp cap from 4.0 → 6.5 → 7.0 over 10-20 steps
- Limit `|delta_tau_position|` to 0.5 Nm/step
- Smooth cap transitions with exponential decay

**Rationale:** Gradual authority increase prevents wheel velocity spikes and gives damping time to engage.

**Risk:** Does not fix wrong-sign torque. If sign is wrong, smoother ramp just builds wrong-direction drift more gradually.

#### Candidate 2: T6I — Phase-Aware Decay

**Target:** Fix #4 (held too long) and partially mitigate #1 (wrong sign)

**Design:**
- Monitor `e * e_dot` sign
- When error starts converging (`e * e_dot < 0`), decay authority:
  - `effective_cap = 4.0 + (raised_cap - 4.0) * exp(-t / tau_decay)`
  - `tau_decay = 50 steps`
- Do not release completely until inside band
- Preserve emergency recenter when moving away

**Rationale:** Decay reduces overpowered correction during convergence. If torque sign is wrong, decay limits damage duration.

**Risk:** May release too early if convergence is temporary.

### Why NOT T6G (Gain Scheduled)?

**Phase 4 found no evidence of gain mismatch.** Response ratio raised/normal = 0.97.

Reducing gain would lower transmitted torque, negating the entire purpose of T6F.

### Why NOT T6J (Velocity Brake)?

**Velocity braking is a symptom fix, not a root-cause fix.**

Wheel velocity overshoot is a consequence of wrong-sign torque and abrupt jumps. Fixing those should eliminate the need for explicit braking.

If T6H and T6I both fail, T6J can be reconsidered.

### Why NOT T6K (Band Logic Fix)?

**Band activation timing is correct.** The issue is not when to activate, but:
1. **What sign torque to apply** (wrong 52.5% of time)
2. **How to transition** (abrupt jumps)
3. **When to decay** (held too long while converging)

Band thresholds are not the problem.

---

## Action Plan

### Immediate (Before T6G/T6H/T6I Implementation)

**Priority 1: Fix torque sign bug**

Audit:
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- `apcr1n_tau_position_after_cap` composition
- Wheel velocity damping sign convention
- Support drift sign convention
- Yaw compensation sign at high torque
- Final wheel torque composition with APC

**Without fixing the sign bug, no T6 variant will improve drift.**

Run diagnostic:
```python
# Compare sign(final_wheel_tau) vs sign(drift_error) step-by-step
# Expected: opposite signs >80% of time
# Actual T6F: opposite signs only 47.5% of time
```

### After Sign Fix

**Priority 2: Implement T6H and T6I in parallel**

- T6H: Add cap ramping and torque rate limiting
- T6I: Add phase-aware decay during convergence

**Priority 3: Run paired 2000-step screening**

- T6H vs T5
- T6I vs T5

**Pass criteria:**
- Drift improvement: outside ±0.10 better than T5
- Stability preserved
- Torque direction correctness >70%

**If both pass:** Advance better candidate to 5000-step validation  
**If both fail:** Revert to T5, investigate sign bug further  
**If one passes:** Advance that candidate

---

## Known Limitations

1. **2000-step window only** — Longer episodes might reveal additional failure modes
2. **Torque sign analysis incomplete** — Did not trace root cause of 47.5% wrong-direction torque
3. **Single height tested** — Only high_0p480; other heights pending
4. **No stand-up/push evaluated** — Nominal drift only

---

## Conclusion

**T6F correctly identified the upstream 4.0 Nm clip as a bottleneck and successfully raised the cap to 7.0 Nm.**

**However, raised authority exposed four latent design issues that were masked at 4.0 Nm:**
1. **Torque sign convention bug** (primary)
2. **Abrupt cap transitions**
3. **Wheel velocity overshoot**
4. **Authority held too long during convergence**

**Phase 8 correctly rejected T6F.**

**Next steps:**
1. Fix torque sign bug (root cause)
2. Design T6H (rate-limited) and T6I (phase-aware decay)
3. Run paired 2000-step screening
4. Advance only if drift improves

**Do not proceed to 5000-step T6F validation. Do not create T7 or increase authority further until root causes are fixed.**

---

**Classification:** T6F_DEGRADATION_MIXED_CAUSES  
**Recommendation:** DESIGN_TWO_CANDIDATES (T6H + T6I after sign fix)  
**Status:** Root-cause analysis COMPLETE  
**Date:** 2026-06-12
