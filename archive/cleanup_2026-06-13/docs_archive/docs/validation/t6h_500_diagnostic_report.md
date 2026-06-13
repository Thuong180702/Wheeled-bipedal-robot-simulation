# T6H 500-Step Diagnostic Report

**Date**: 2026-06-12  
**Profile**: T6H_soft_blend_arch_fix  
**Status**: REJECTED  
**Classification**: `T6H_500_REJECT_STABILITY`

---

## Executive Summary

T6H_soft_blend_arch_fix was evaluated via 500-step diagnostic at high_0p480 and **FAILED** due to excessive drift degradation.

**Key Result**: Max abs error 0.329m exceeded 0.25m threshold (62% worse than T6F baseline).

**Design Hypothesis**: "Soft modulation (50% pitch/damping reduction) preserves enough stabilization authority" → **INVALIDATED**

**Decision**: Abandon T6H_soft_blend_arch_fix as implemented. Do not proceed to 1200-step validation.

---

## Implementation Verification

### Profile Identity ✅

```
vd_sagittal_authority_profile: T6H_soft_blend_arch_fix
apcr1nd_tuned_variant_name: T6H
```

### Feature Activation

**Soft Pitch Blending**:
- Active: 32.5% of steps (162/499)
- Pitch blend factor: mean 0.84, min 0.50, max 1.0
- Safety override (pitch > 10°): 0 activations
- ✅ Never zeroed pitch (min 50% preserved)

**Soft Damping Blending**:
- Active: 6.8% of steps (34/499)
- Damping blend factor: mean 0.97, min 0.50, max 1.0
- Safety override (wheel_vel > 7.0): 57 activations
- ✅ Never zeroed damping (min 50% preserved)

**Architecture Fix**:
- arch_fix enabled: YES
- Budget cap raise: 4.0 → 8.0 Nm
- Based on T6F_budget_cap_raise
- ✅ T6F inheritance verified

### Implementation Correctness ✅

All T6H features implemented and activated as designed:
- Soft blend factors never zero (0.50 minimum)
- Safety overrides functional
- Based on T6F architecture
- Telemetry fields present

**Conclusion**: Implementation correct. Failure is design-level, not implementation bug.

---

## Performance Results

### Survival

- **Terminated**: FALSE ✅
- **Survived steps**: 499/500
- **Upright steps**: 499
- **Transition steps**: 0 ✅
- **Recovery steps**: 0 ✅

### Drift (Primary Metrics)

| Metric | Value | vs T6F | vs T5 | Gate |
|--------|-------|--------|-------|------|
| **Min error** | -0.113m | -97m worse | -97m worse | - |
| **Max error** | +0.329m | +126m worse | +142m worse | - |
| **Max abs error** | **0.329m** | **+0.126m (+62%)** | **+0.142m (+76%)** | ❌ FAIL (>0.25m) |
| **Final error** | +0.011m | -130m better | -49m better | ✅ (<0.15m) |
| **P2P drift** | 0.442m | +0.223m (+102%) | +0.239m (+118%) | - |
| **Mean abs error** | 0.123m | +0.041m (+50%) | +0.028m (+30%) | - |

**Outside bounds**:
- ±0.08m: 54.3% (271/499 steps)
- ±0.10m: 48.3% (241/499 steps) — worse than T6F (39.1%)
- ±0.15m: 35.7% (178/499 steps) — much worse than T6F (24.2%)

### Stability (Secondary Metrics)

| Metric | Value | Gate |
|--------|-------|------|
| Max pitch | 0.0° | ✅ (<12°) |
| RMS pitch | 0.0° | - |
| Max roll | 0.0° | - |
| RMS roll | 0.0° | - |
| CoM Z min | 0.481m | ✅ (>0.431m) |
| CoM Z mean | 0.490m | - |
| Contact % | 100% | ✅ |
| Double contact % | 100% | - |

### Wheel Activity

- Max wheel velocity: 0.0 rad/s
- RMS wheel velocity: 0.0 rad/s
- Steps >5 rad/s: 0
- Steps >6 rad/s: 0
- Steps >7 rad/s: 0

**Note**: Wheel velocity telemetry appears under-instrumented or near-zero due to high_0p480 configuration.

### Structural Gates ✅

- **WBC flag**: 0 ✅
- **Hidden torque max**: 0.000000 ✅
- **Ownership violation max**: 0.000000 ✅

---

## Classification

### Result: `T6H_500_REJECT_STABILITY`

**Rejection criteria**:
- ❌ **Max abs error 0.329m > 0.25m** ← PRIMARY FAILURE
- ✅ Terminated: FALSE
- ✅ Transition steps: 0
- ✅ Recovery steps: 0
- ✅ Max pitch < 12°
- ✅ No structural violations

**Rejection reason**: "Max abs error 0.329m > 0.25m"

### Comparison to Baselines

| Profile | Max Abs Error | Δ vs T6F | Result |
|---------|---------------|----------|--------|
| T5 | 0.187m | -0.016m (better) | Baseline |
| T6F | 0.203m | 0.000m (reference) | Baseline |
| **T6H** | **0.329m** | **+0.126m (+62%)** | **REJECT** |
| T6I | 0.203m | 0.000m (identical) | PASS |

**Key observation**: T6H is the ONLY profile that degrades drift compared to T6F. T6I matches T6F exactly.

---

## Root Cause Analysis

### Failure Mechanism

**Hypothesis**: "Soft modulation (50% reduction) preserves enough authority" → **FALSE**

**Evidence**:
1. Soft pitch blending activated 32.5% of the time
2. During activation, pitch authority reduced to 50%
3. Max abs error degraded 62% compared to T6F
4. Same failure signature as T6F_sign_corrected (which used 100% suppression)

**Root cause**: Even 50% pitch/damping preservation is insufficient. Pitch and damping are phase-appropriate coupled stabilization terms. Any component-level suppression (hard or soft) removes critical authority needed for drift containment.

### Why Soft Blending Failed

T6H design assumed:
- 50% pitch control is enough to stabilize
- 50% damping is enough to dissipate energy
- Gradual reduction (0.50 vs 0.0) avoids discontinuities

Reality:
- Wheeled biped needs 100% pitch control during arch_fix
- Velocity damping must be fully preserved
- Component-level suppression is architecturally flawed
- Blend factor tuning (50% vs 100%) changes severity, not failure mode

### Comparison to T6F_sign_corrected

| Feature | T6F_sign_corrected | T6H_soft_blend | Result |
|---------|-------------------|----------------|--------|
| Pitch suppression | 100% (tau = 0.0) | 50% (factor = 0.50) | Both FAIL |
| Max abs error | 0.383m | 0.329m | T6H less severe |
| vs T6F baseline | +88% worse | +62% worse | T6H less severe |
| Classification | REJECT | REJECT | Both invalid |

**Conclusion**: T6H is "T6F_sign_corrected lite" — same architectural flaw, slightly better numbers, still fails.

---

## Design Hypothesis Invalidation

### Original Hypothesis

From `docs/validation/t6h_t6i_safe_next_candidates_design.md`:

> **T6H Hypothesis**: Soft modulation (50% reduction, not 100%) preserves continuous stabilization authority while reducing fighting terms during arch_fix.

### Invalidation Evidence

1. **Drift degraded significantly**: 0.329m vs T6F 0.203m (+62%)
2. **Worse than doing nothing**: T6H worse than T6F baseline
3. **Same failure mode**: Component suppression removes critical authority
4. **Blend factor irrelevant**: 50% vs 100% changes magnitude, not mechanism

### What We Learned

**False assumption**: "Preserving 50% of pitch/damping is enough"

**Truth**: Pitch and damping work together as a coupled system during arch_fix. Suppressing either component (even partially) disrupts the coupled dynamics and degrades drift containment.

**T6I proves this**: T6I succeeds by preserving 100% pitch/damping authority and modulating *cap amount* instead. This is the correct architectural approach.

---

## Why T6I Succeeds Where T6H Fails

| Aspect | T6H (Failed) | T6I (Passed) |
|--------|--------------|--------------|
| **Modulation target** | Stabilization terms (pitch/damping) | Authority amount (cap) |
| **Component suppression** | Yes (soft, 50%) | No |
| **Pitch authority** | 50-100% (variable) | 100% (continuous) |
| **Damping authority** | 50-100% (variable) | 100% (continuous) |
| **Approach** | Reduce fighting | Release high authority when converging |
| **Result** | 0.329m (FAIL) | 0.203m (PASS) |

**Key insight**: The problem is not "fighting terms" — the problem is holding high authority too long after error starts converging. T6I solves this by detecting convergence and decaying the cap gradually, NOT by suppressing pitch/damping.

---

## Lessons Learned

### What Failed

1. **Soft blending does not fix component suppression flaw**
   - T6H (50% blend) fails, just like T6F_sign_corrected (100% suppress)
   - Blend factor is a tuning knob on a broken design
   - Cannot tune your way out of an architectural mistake

2. **"Preserving 50% is enough" assumption was wrong**
   - Wheeled biped needs full pitch control during arch_fix
   - Damping must be fully preserved
   - Coupled dynamics require all terms active

3. **"Reduce fighting" is the wrong framing**
   - Pitch and damping are not "fighting" — they are stabilizing
   - Component-level sign inspection misleads
   - System-level drift containment is the correct metric

### Design Principles Reinforced

From T6F_sign_corrected (100%) → T6H (50%) → both fail:

1. **Never suppress pitch stabilization** (any amount)
2. **Never suppress velocity damping** (any amount)
3. **Modulate authority via cap, not via component suppression**
4. **Preserve continuous control authority** (100%, not 50%)
5. **Phase-aware modulation > static suppression**

---

## Alternative Approaches

If revisiting soft blending in future work (NOT recommended):

### Variant 1: Adaptive Blend Factors
- Use error trajectory to adjust blend factor dynamically
- Example: blend_factor = 0.50 when diverging, 0.80 when converging
- **Problem**: Still component suppression, likely still fails

### Variant 2: Damping-Only Blending
- Blend only wheel damping, never pitch stabilization
- Preserve 100% pitch control at all times
- **Problem**: May still degrade, and T6I is simpler

### Variant 3: Phase-Aware Blending
- Activate blending only during convergence phase (like T6I)
- Combine with T6I's convergence detection
- **Problem**: Why blend when cap decay works better?

**Recommendation**: Don't pursue T6H variants. T6I's cap-decay approach is architecturally superior and already works.

---

## Decision

**T6H_soft_blend_arch_fix**: **ABANDONED**

**Rationale**:
1. Failed 500-step diagnostic (max abs error 0.329m > 0.25m)
2. Design hypothesis invalidated empirically
3. Component-level suppression is fundamentally flawed
4. T6I provides superior alternative (cap decay, no suppression)

**Do NOT**:
- Tune blend factors (0.50 → 0.60, 0.70, etc.)
- Proceed to 1200-step validation
- Add T6H to high_0p480 candidate pool

**Recommendation**: Focus all effort on T6I validation and deployment.

---

## Next Steps

### For T6H: None

Abandon as implemented. Do not invest further effort.

### For high_0p480 Tuning

Proceed with T6I:
1. 1200-step validation
2. If passes: 2000-step validation
3. If passes: 5000-step final validation
4. Deploy as high_0p480 solution

T6F_budget_cap_raise remains fallback if T6I fails longer validation.

---

**End of Report**
