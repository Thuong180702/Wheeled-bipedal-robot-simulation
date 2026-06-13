# APCR Amplitude-Constrained Candidate Design

**Date:** 2026-06-09
**Based on:** `apcr_drift_amplitude_audit.md` findings
**Classification:** `APCR1C_BETTER_CENTERING_BUT_LARGER_AMPLITUDE`

---

## Problem Statement

APCR1c achieves:
- **Better** positive bias reduction (77.8% vs D2's 93.2%)
- **Better** band violations at 500 steps (12.6% vs 19.2%)
- **Worse** peak-to-peak amplitude (+48.5% larger at 500 steps)

The APCR oscillation envelope is significantly wider than D2, even after APCR1c stabilizes in later windows.

---

## Evidence Summary

| Metric | D2 | APCR1c | Assessment |
|--------|-----|--------|------------|
| Max positive drift | 0.1757m | 0.1682m | APCR1c better |
| Min drift | +0.0142m | -0.0716m | D2 better (stays positive) |
| Peak-to-peak (500) | 0.1615m | 0.2398m | D2 better (-33%) |
| Peak-to-peak (2000) | 0.1792m | 0.2398m | D2 better (-25%) |
| Late window P2P | ~0.015m | ~0.180m | D2 12× tighter |
| Max abs drift | 0.1757m | 0.1682m | Similar |

**Root Cause:** APCR1c activates at outer_enter_m=0.08 and applies max_cross_tau=1.0 Nm until inner_exit_m=0.07 is reached. This creates an aggressive corrective torque that overshoots into negative territory.

---

## Candidate Design Options

### Option A: Narrower APCR Band (Softer Intervention)

**Idea:** Activate earlier but with softer torque to avoid momentum-driven overshoot.

| Parameter | APCR1c | APCR1d-A |
|-----------|--------|----------|
| `outer_enter_m` | 0.08 | 0.07 |
| `inner_exit_m` | 0.07 | 0.065 |
| `max_cross_tau` | 1.0 Nm | 0.75 Nm |
| `opposite_overshoot_m` | 0.00 | 0.00 |

**Rationale:**
- Activate at 0.07m (softer, earlier)
- Exit at 0.065m (tight hold before release)
- Reduce torque from 1.0 to 0.75 Nm
- Narrower band reduces opportunity for overshoot

**Pros:**
- Softer correction reduces overshoot
- Earlier entry catches drift before it grows
- Tighter exit prevents prolonged oscillation

**Cons:**
- More frequent activation (may increase wear)
- May not apply enough correction for large drifts

---

### Option B: Velocity-Aware Release

**Idea:** Release APCR when error is approaching zero AND moving in the right direction, not just when crossing threshold.

| Parameter | APCR1c | APCR1d-B |
|-----------|--------|----------|
| `outer_enter_m` | 0.08 | 0.08 |
| `inner_exit_m` | 0.07 | 0.07 |
| `release_condition` | error ≤ inner_exit | error ≤ 0.08 AND error_rate < 0 |
| `max_cross_tau` | 1.0 Nm | 1.0 Nm |

**Rationale:**
- Current APCR releases when error ≤ 0.07, regardless of momentum
- Adding velocity awareness prevents releasing during fast negative swing
- Release only when error is dropping AND below a safe threshold

**Pros:**
- Prevents releasing during negative overshoot
- APCR holds longer during dangerous swing phase
- No torque change needed

**Cons:**
- Requires telemetry field for error rate (signed_error change per step)
- May hold too long, causing more positive-side oscillation

---

### Option C: Proportional Torque Shaping

**Idea:** Scale torque inversely with distance from zero, so torque weakens as error approaches center.

| Parameter | APCR1c | APCR1d-C |
|-----------|--------|----------|
| `outer_enter_m` | 0.08 | 0.08 |
| `inner_exit_m` | 0.07 | 0.07 |
| `torque_mode` | constant | proportional |
| `max_cross_tau` | 1.0 Nm | 1.0 Nm (at outer) → 0.3 Nm (at inner) |
| `proportional_band` | N/A | 0.07-0.08m |

**Torque formula:**
```
torque = max_cross_tau * (error - inner_exit) / (outer_enter - inner_exit)
```

**Rationale:**
- At error = 0.08m (outer): torque = 1.0 Nm (full correction)
- At error = 0.07m (inner): torque = 0 Nm (no correction)
- Smooth ramp-down prevents hard cutoff that causes overshoot

**Pros:**
- Smooth torque reduction prevents overshoot
- Natural easing as error approaches center
- Maintains correction authority when error is large

**Cons:**
- More complex implementation
- Requires careful tuning of proportional band
- May be too weak for large errors

---

### Option D: Amplitude-Constrained APCR

**Idea:** Add hard guard that forces release/decay if error swings too far in the opposite direction.

| Parameter | APCR1c | APCR1d-D |
|-----------|--------|----------|
| `outer_enter_m` | 0.08 | 0.08 |
| `inner_exit_m` | 0.07 | 0.07 |
| `opposite_overshoot_m` | 0.00 | -0.05 |
| `max_cross_tau` | 1.0 Nm | 1.0 Nm |
| `anti_overshoot_guard` | false | true |

**Guard Logic:**
```
if state == CROSS_FROM_POSITIVE and signed_error < -0.05:
    force_exit_correction()  # Apply small opposite torque to slow swing
if state == CROSS_FROM_NEGATIVE and signed_error > +0.05:
    force_exit_correction()
```

**Rationale:**
- If APCR causes negative drift below -0.05m, apply counter-torque
- Prevents deep negative excursions
- Early counter-action constrains oscillation envelope

**Pros:**
- Directly addresses overshoot problem
- Keeps error within reasonable bounds
- Can be combined with other options

**Cons:**
- Additional complexity
- May interfere with normal APCR operation
- Requires careful guard threshold tuning

---

## Recommended Next Candidate

**Recommendation: Option C (Proportional Torque Shaping)**

**Rationale:**
1. Addresses the root cause (constant torque → overshoot)
2. Smooth transition reduces mechanical stress
3. Maintains correction authority when needed
4. Simpler than Option D's guard logic
5. Can be combined with Option A's narrower band

**Proposed APCR1d Parameters:**

| Parameter | APCR1c | APCR1d (Recommended) |
|-----------|--------|------------------------|
| `outer_enter_m` | 0.08 | 0.08 (keep) |
| `inner_exit_m` | 0.07 | 0.07 (keep) |
| `max_cross_tau` | 1.0 Nm | 1.0 Nm |
| `torque_mode` | constant | proportional |
| `proportional_decay` | N/A | linear from outer to inner |
| `opposite_overshoot_m` | 0.00 | 0.00 |
| `recovery_gate_mode` | true | true |

**Expected Outcome:**

| Metric | APCR1c | APCR1d Target |
|--------|--------|----------------|
| Peak-to-peak | 0.2398m | < 0.20m |
| Max abs drift | 0.1682m | < 0.170m |
| Positive% | 77.8% | < 80% |
| Outside ±0.15 | 12.6% | < 10% |

---

## Validation Plan

1. **500-step smoke test** with APCR1d at low_0p300
2. **Compute amplitude metrics** (P2P, MaxAbs, min/max)
3. **Compare vs APCR1c:**
   - P2P should decrease (target: < 0.20m)
   - MaxAbs should stay similar or improve
   - Positive% should not increase significantly
   - Outside band should decrease or stay same
4. **If 500-step shows improvement**, proceed to 2000-step validation
5. **If 2000-step shows improvement**, proceed to 5000-step validation

---

## Files

- `docs/validation/apcr_amplitude_constrained_candidate_design.md` (this file)
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr_amplitude_constrained_candidate_design.json`
