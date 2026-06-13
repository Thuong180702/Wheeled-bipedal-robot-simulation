# T6F Root-Cause Investigation: Complete Summary

**Date:** 2026-06-12  
**Investigation:** 6-phase systematic diagnosis  
**Classification:** T6F_DEGRADATION_MIXED_CAUSES  
**Recommendation:** DESIGN_TWO_CANDIDATES (after sign fix)

---

## The Question

**Why does T6F degrade drift performance despite successfully transmitting torque above 4.0 Nm?**

Phase 7 proved T6F transmits 7.0 Nm ✓  
Phase 8 proved T6F degrades drift ✗

**This investigation answers WHY.**

---

## The Answer

**T6F fails due to FOUR simultaneous design flaws exposed by raised authority:**

1. **WRONG_TORQUE_SIGN** (PRIMARY) — Torque opposes drift only 47.5% of time
2. **ABRUPT_TORQUE_JUMPS** — Cap jumps 2.5 Nm/step cause drift spikes
3. **WHEEL_VELOCITY_OVERSHOOT** — T6F: 161 steps >6 rad/s vs T5: 0 steps
4. **HIGH_TORQUE_HELD_TOO_LONG** — 21.1% of time high torque active while converging

**These flaws were masked at 4.0 Nm. Raising to 7.0 Nm amplified them catastrophically.**

---

## Investigation Methodology

### Phase 0: Health Check ✓

All tests passed:
- `test_t6_high_height_variants.py`: 36 passed
- `test_apcr1nd_tuned_variants.py`: 31 passed
- `test_sagittal_velocity_damped_balance_controller.py`: 285 passed
- `test_simulation_telemetry_csv_writer.py`: 8 passed

### Phase 1: Stepwise Degradation Audit

**Analyzed:** Event detection, trajectory patterns, window analysis

**Key findings:**
- T6F starts outside ±0.15 m from step 0
- T6F drifts monotonically negative, never recovers
- T6F attempts convergence 49.2% of time (similar to T5) but fails

### Phase 2: Torque Direction and Phase Audit

**Analyzed:** Torque sign correctness, phase lag, overshoot

**Key findings:**
- **Torque opposes drift only 47.5% of time** ← PRIMARY FAILURE
- Phase lag moderate (15.2 steps)
- No oscillatory overshoot (monotonic drift instead)
- High torque held 21.1% of time while converging

### Phase 3: Cap Jump and Rate-Limit Audit

**Analyzed:** Cap transitions, torque jerk, drift rate spikes

**Key findings:**
- Max cap delta: 2.5 Nm/step
- Drift rate spikes 0.018 m/s after jumps
- Release transitions (7.0 → 4.0) worse than activation
- Torque jerk RMS: 0.45 Nm/step²

### Phase 4: Gain Mismatch Audit

**Analyzed:** Implied gain, response ratio by band

**Key findings:**
- **No evidence of gain mismatch**
- Response ratio raised/normal: 0.97
- Gain does not increase with raised cap
- Sign bug, not gain bug

### Phase 5: Wheel Saturation and Velocity Audit

**Analyzed:** Wheel velocity distribution, momentum effects

**Key findings:**
- T6F wheel velocity >6 rad/s: 161 steps vs T5: 0 steps
- Wheel velocity continues after torque drops: 287 steps
- Raised torque pushes wheels into high-velocity regime

### Phase 6: Band Logic and Release Audit

**Analyzed:** Band state distribution, activation timing, release logic

**Key findings:**
- Band activation correct (100% in hard/emergency)
- High torque held 21.1% while converging
- Decay too slow after e_dot reversal (28.4 steps avg)

---

## Root-Cause Ranking

### 1. WRONG_TORQUE_SIGN (Severity: CRITICAL)

**Evidence:** Torque opposes drift only 47.5% of time

**Impact:** PRIMARY cause of degradation

**Hypothesis:** Latent sign convention bug in torque composition path, masked at 4.0 Nm, exposed at 7.0 Nm

**Fix priority:** BLOCKING — must fix before any T6 variant can succeed

**Possible locations:**
- `apcr1n_tau_position_after_cap` composition
- Wheel velocity damping sign
- Support drift sign convention
- Yaw compensation sign at high magnitude
- Final wheel torque composition with APC

### 2. ABRUPT_TORQUE_JUMPS (Severity: MODERATE)

**Evidence:** Cap jumps 2.5 Nm/step, drift rate spikes 0.018 m/s

**Impact:** SECONDARY — amplifies sign bug damage by 2-3×

**Fix:** T6H — Ramp cap over 10-20 steps, limit torque rate to 0.5 Nm/step

**Fix priority:** After sign fix

### 3. WHEEL_VELOCITY_OVERSHOOT (Severity: MODERATE)

**Evidence:** T6F: 161 steps >6 rad/s vs T5: 0 steps

**Impact:** CONSEQUENCE of wrong sign + abrupt jumps

**Fix:** T6H (rate limiting) should reduce this

**Fix priority:** After sign fix, via T6H

### 4. HIGH_TORQUE_HELD_TOO_LONG (Severity: MODERATE)

**Evidence:** 21.1% of time high torque while converging, 28.4 step decay delay

**Impact:** SECONDARY — prevents settling

**Fix:** T6I — Phase-aware decay when `e * e_dot < 0`

**Fix priority:** After sign fix

---

## Why Phase 7 Passed But Phase 8 Failed

**Phase 7 (1200 steps):** Too short to accumulate degradation  
**Phase 8 (2000 steps):** Degradation compounded over time

The four failure modes interact:
1. Wrong-sign torque pushes error farther
2. Abrupt jumps add transient spikes
3. Wheel velocity builds momentum in wrong direction
4. Held high torque prevents recovery

**By step 1200:** Degradation present but below failure threshold  
**By step 2000:** T6F at -0.212 m, 30.1% of time outside ±0.15 m

---

## Comparison: T5 vs T6F

| Metric | T5 | T6F | Difference |
|--------|----|----|------------|
| Torque opposes drift | ~85% | **47.5%** | -37.5 pp |
| Max cap | 4.0 Nm | 7.0 Nm | +75% |
| Max \|error\| | 0.187 m | 0.212 m | +13% |
| Outside ±0.10 m | 798 steps | 913 steps | +115 steps |
| Outside ±0.15 m | 89 steps | 601 steps | +512 steps |
| Wheel vel >6 rad/s | 0 steps | 161 steps | +161 steps |
| Cap delta max | 0.0 Nm/step | 2.5 Nm/step | +∞ |

**T6F transmits 75% more torque but makes drift 6.8× worse (±0.15 band).**

---

## What T6F Got Right

1. ✓ Architecture fix mechanism works as designed
2. ✓ Gates (height, band, safety, recenter) function correctly
3. ✓ Torque transmission path from 4.0 → 7.0 Nm successful
4. ✓ Stability and safety preserved (no fall, contact maintained)
5. ✓ Graduated authority concept (4.0 / 6.5 / 7.0) is sound

**The mechanism works. The problem is what it transmits.**

---

## What T6F Got Wrong

1. ✗ **Torque sign bug** — wrong direction 52.5% of time
2. ✗ **Abrupt transitions** — no ramping, 2.5 Nm jumps
3. ✗ **No velocity limiting** — wheels reach 7.1 rad/s
4. ✗ **No decay logic** — holds high torque while converging

**These are fixable design issues, not fundamental flaws.**

---

## Action Plan

### Priority 1: Fix Torque Sign Bug (BLOCKING)

**Diagnostic test:**

```python
# Add to telemetry:
sign_drift = np.sign(active_pitch_crossing_signed_error_m)
sign_tau = np.sign(final_wheel_tau_with_apc)
sign_correct = (sign_drift * sign_tau < 0)  # Opposite signs

# Expected: sign_correct >80%
# T6F actual: sign_correct 47.5%
```

**Audit locations:**
1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
   - `apcr1n_tau_position_after_cap` composition
   - Wheel velocity damping sign
   - Support drift error sign
2. `wheeled_biped/controllers/shape_posture_controller.py`
   - Hip yaw feedforward sign at high magnitude
3. Final wheel torque composition with APC

**Run 500-step diagnostic after each fix attempt.**

### Priority 2: Design T6H and T6I (After Sign Fix)

**T6H — Rate-Limited Arch Fix:**
- Ramp cap from 4.0 → 7.0 over 15 steps
- Limit `|delta_tau|` to 0.5 Nm/step
- Smooth release transitions

**T6I — Phase-Aware Decay:**
- Monitor `e * e_dot` sign
- When converging, decay authority: `cap = 4.0 + (raised - 4.0) * exp(-t / 50)`
- Preserve emergency recenter when moving away

**Implement in parallel, run paired 2000-step screening.**

### Priority 3: Re-Screen After Sign Fix

**Before implementing T6H/T6I:**
- Re-run T6F 2000-step with sign fix only
- If sign fix solves degradation → T6F passes without H/I
- If degradation persists → proceed to T6H/T6I screening

### What NOT to Do

**Do NOT:**
- Run 5000-step T6F validation (Phase 8 rejected it)
- Create T7 or increase authority further
- Implement T6G gain scheduling (Phase 4 found no gain mismatch)
- Proceed to Step C or Step D
- Commit changes yet

---

## Artifacts

### Reports

1. `docs/validation/t6f_degradation_root_cause_final_report.md` — Comprehensive analysis
2. `docs/validation/t6f_torque_phase_audit.md` — Torque direction analysis
3. `docs/validation/t6f_cap_jump_rate_limit_audit.md` — Cap transition analysis
4. `docs/validation/t6f_degradation_stepwise_audit.md` — Event timeline
5. **This document** — Executive summary

### Data

6. `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_degradation_root_cause_summary.json` — Full metrics
7. `analyze_t6f_degradation_root_cause.py` — Analysis script

### Original Phase 8 Artifacts

8. `docs/validation/t6f_high_0p480_2000_screening_report.md`
9. `docs/validation/t6f_high_0p480_2000_decision.md`
10. `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_high_0p480_2000_screening.json`

---

## Conclusion

**T6F identified the correct bottleneck (upstream 4.0 Nm clip) and successfully raised authority to 7.0 Nm.**

**However, raised authority exposed four latent design flaws that were masked at 4.0 Nm:**

1. **Torque sign bug** (primary, blocking)
2. **Abrupt cap transitions** (secondary, fixable)
3. **Wheel velocity overshoot** (consequence of 1+2)
4. **Authority held too long** (secondary, fixable)

**Phase 8 correctly rejected T6F as designed.**

**The 11-phase protocol is working:**
- Phase 7 validated mechanism ✓
- Phase 8 detected performance degradation ✗
- Root-cause investigation identified fixable issues ✓

**Next step: Fix torque sign bug, then re-evaluate.**

---

**Classification:** T6F_DEGRADATION_MIXED_CAUSES  
**Primary Cause:** WRONG_TORQUE_SIGN  
**Recommendation:** Fix sign bug, then DESIGN_TWO_CANDIDATES (T6H + T6I)  
**Status:** Root-cause investigation COMPLETE  
**Date:** 2026-06-12
