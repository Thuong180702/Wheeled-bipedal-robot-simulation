# Post-T6F_sign_corrected Next Step Recommendation

**Date**: 2026-06-12  
**Context**: T6F_sign_corrected design invalidated after 500-step diagnostic  
**Decision**: DESIGN_T6H_AND_T6I_NEXT

---

## Executive Summary

After systematic investigation of T6F_sign_corrected (Phases 0-6), all implementation bugs were fixed but the design fundamentally failed stability validation. Sign correctness degraded (43.5% vs T6F's 48.9%), drift amplified (+88%), and pitch instability emerged (19.7° vs 8.4°).

**Root cause**: Sign incorrectness is a symptom of coupled pitch-wheel-phase dynamics, not a cause of drift. Removing pitch and damping authority to "fix" signs destroyed stabilization.

**Recommended path forward**: DESIGN_T6H_AND_T6I_NEXT

Two safer candidates have been designed that preserve stabilization authority while exploring alternative approaches to drift reduction. Implementation requires explicit approval and mandatory 500-step validation before any long evaluation.

---

## Decision Framework

### Option 1: RETURN_TO_T5_CURRENT_BEST

**Description**: Abandon all high_0p480 tuning efforts and return to T5 (APCR1nD_T5_band_limited_balanced) as current best profile.

**Rationale for this option**:
- T5 is validated and stable
- T5 max drift: 0.187m (better than T6F's 0.203m)
- T5 has no arch_fix complexity
- Avoids further time investment in high_0p480 tuning

**Rationale against this option**:
- T5 does not have arch_fix → cannot recover from large disturbances
- T5 enters hard/emergency band 42.7% of time vs T6F's 36.7%
- T5 sign correctness 35.5% vs T6F's 48.9% (T6F is better)
- T6F provides better high-authority recovery capability

**Verdict**: ❌ REJECT — T6F is superior to T5 for high_0p480 due to arch_fix recovery capability and lower emergency band usage.

---

### Option 2: DESIGN_T6H_SOFT_BLEND_NEXT

**Description**: Implement only T6H_soft_blend_arch_fix candidate (soft modulation of pitch/damping by 50%, not 100%).

**Rationale for this option**:
- Avoids T6F_sign_corrected's fatal flaw (preserves 50% stabilization)
- Simpler implementation than T6I
- Explicit safety overrides (pitch > 10°, wheel_vel > 7.0)
- May reduce fighting terms during arch_fix

**Rationale against this option**:
- Blend factors still modify stabilization terms (some risk remains)
- May not improve much over T6F if fighting terms are not the issue
- Single candidate approach → no fallback if T6H fails

**Verdict**: ⚠️ ACCEPTABLE — Safer than T6F_sign_corrected, but single-candidate approach has higher risk.

---

### Option 3: DESIGN_T6I_PHASE_AWARE_NEXT

**Description**: Implement only T6I_phase_aware_release candidate (detect convergence, gradually decay cap).

**Rationale for this option**:
- Preserves full pitch and damping authority (safest approach)
- Directly addresses potential overshoot mechanism
- Smooth cap transitions avoid discontinuities
- Higher estimated success probability (65-75% vs T6H's 60-70%)

**Rationale against this option**:
- More complex implementation (convergence detection, state tracking)
- Convergence detector may trigger prematurely
- Single candidate approach → no fallback if T6I fails

**Verdict**: ⚠️ ACCEPTABLE — Safest design, but single-candidate approach has higher risk.

---

### Option 4: DESIGN_T6H_AND_T6I_NEXT ✅

**Description**: Implement both T6H_soft_blend_arch_fix and T6I_phase_aware_release candidates, evaluate comparatively via 500-step diagnostic.

**Rationale for this option**:
- **Explores two distinct hypotheses**:
  - T6H: Fighting terms cause overshoot → soft blend reduces fighting
  - T6I: Prolonged high authority causes overshoot → phase-aware release reduces overshoot
- **Provides fallback**: If one candidate fails, the other may succeed
- **Comparative evaluation**: 500-step diagnostic reveals which approach is more promising
- **Risk mitigation**: Both candidates preserve stabilization authority (learned from T6F_sign_corrected)
- **Efficient resource use**: Implementation cost is moderate; 500-step diagnostic is fast

**Rationale against this option**:
- Higher upfront implementation effort (two candidates vs one)
- Both candidates may fail → wasted effort

**Risk analysis**:
- Probability both candidates fail AND provide no insight: LOW (~15-20%)
- Probability at least one candidate improves over T6F: MODERATE-HIGH (~55-65%)
- Probability comparative evaluation provides valuable insight even if both fail: HIGH (~80%)

**Verdict**: ✅ **RECOMMENDED** — Best balance of exploration, risk mitigation, and learning.

---

### Option 5: STOP_HIGH_0P480_TUNING_REVISE_TARGET

**Description**: Abandon high_0p480 tuning entirely and revise task target (e.g., relax drift tolerance or switch to different height).

**Rationale for this option**:
- T6F already achieves acceptable performance (0.203m max drift, no falls)
- Further tuning may have diminishing returns
- Could redirect effort to other heights or tasks

**Rationale against this option**:
- T6F is stable and provides good baseline for incremental improvement
- Two promising candidates (T6H, T6I) are already designed and low-risk
- high_0p480 is critical height for push recovery and step E validation
- Target (max drift < 0.15m) is achievable with better convergence

**Verdict**: ❌ REJECT — Premature to abandon tuning when safer candidates exist and T6F provides stable baseline.

---

## Final Decision: DESIGN_T6H_AND_T6I_NEXT

### Rationale

1. **Two distinct hypotheses**: T6H explores soft blending, T6I explores phase-aware release — complementary approaches

2. **Risk mitigation**: If one fails, the other may succeed; if both fail, comparative analysis provides insight for next iteration

3. **Learned from T6F_sign_corrected**: Both candidates preserve stabilization authority, avoiding fatal flaw

4. **Low cost, high value**: 
   - Implementation: ~1-2 days per candidate
   - 500-step diagnostic: ~1 hour total
   - Insight gained: Validates or invalidates two distinct hypotheses

5. **Safety-first**: Both candidates have mandatory 500-step gates; neither proceeds to long evaluation without passing

6. **Fallback available**: If both candidates fail, T6F baseline is validated and ready for 2000-step screening

### Implementation Order

**Sequential approach** (recommended if resource-constrained):
1. Implement T6H_soft_blend_arch_fix first (simpler)
2. Run T6H 500-step diagnostic
3. If T6H PASS → proceed to T6I implementation
4. If T6H FAIL → evaluate whether T6I is worth pursuing or return to T6F

**Parallel approach** (recommended if resources available):
1. Implement T6H and T6I in parallel
2. Run comparative 500-step diagnostic (T5, T6F, T6H, T6I)
3. Analyze all four profiles together
4. Select best candidate(s) for 1200-step evaluation

**Recommended**: Parallel approach for faster exploration and comparative insight.

---

## Implementation Timeline (Estimated)

### Phase 1: T6H Implementation (1-2 days)
- [ ] Add soft blend logic to sagittal controller
- [ ] Add T6H telemetry fields
- [ ] Add T6H profile to authority schedule
- [ ] Unit tests (blend factors, safety overrides)
- [ ] 100-step smoke test
- [ ] Code review and verification

### Phase 2: T6I Implementation (2-3 days)
- [ ] Add convergence detection logic
- [ ] Add cap decay and rate limiting
- [ ] Add T6I telemetry fields
- [ ] Add T6I profile to authority schedule
- [ ] Unit tests (convergence, cap decay, rate limit)
- [ ] 100-step smoke test
- [ ] Code review and verification

### Phase 3: 500-Step Diagnostic (0.5 days)
- [ ] Run T5, T6F, T6H, T6I at high_0p480 (500 steps each)
- [ ] Analyze telemetry and classify results
- [ ] Generate comparative report

### Phase 4: Decision Point
- [ ] If T6H PASS and T6I PASS → select best for 1200-step
- [ ] If T6H PASS and T6I FAIL → proceed with T6H
- [ ] If T6H FAIL and T6I PASS → proceed with T6I
- [ ] If both FAIL → return to T6F baseline for long evaluation

**Total estimated time**: 4-6 days from approval to 500-step results.

---

## Success Criteria (Review)

### T6H 500-Step PASS Criteria
- Max abs error ≤ 0.21m (equal or better than T6F 0.203m)
- Final error < 0.15m
- Max pitch < 11°
- Transition steps = 0
- Recovery steps = 0
- Terminated = False

### T6I 500-Step PASS Criteria
- Max abs error ≤ 0.21m
- Final error < 0.15m
- Max pitch < 11°
- Transition steps = 0
- Recovery steps = 0
- Terminated = False
- No premature release causing secondary divergence

### Comparative Metrics
When comparing T6H vs T6I (if both pass):
- Which achieves lower max drift?
- Which has smoother error trajectory?
- Which uses less torque/energy?
- Which has faster settling time?
- Which has simpler implementation and fewer edge cases?

**Sign correctness is NOT a gate**: Both candidates can proceed with 40-50% sign correctness if primary metrics pass.

---

## Risk Assessment

### T6H Risks
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Blend factors still degrade stability | LOW-MODERATE | HIGH | 500-step gate, safety overrides |
| No improvement over T6F | MODERATE | LOW | T6I provides fallback |
| Fighting terms not the issue | MODERATE | LOW | Learning value even if fails |

### T6I Risks
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Convergence detector triggers prematurely | MODERATE | HIGH | Convergence threshold tuning, 500-step gate |
| Secondary divergence after cap release | LOW-MODERATE | HIGH | Rate limit on cap decay |
| Complexity introduces bugs | LOW | MODERATE | Unit tests, code review |

### Overall Risk
- **Both candidates fail**: LOW (~20%) — Both preserve stabilization, avoid T6F_sign_corrected flaw
- **At least one candidate improves over T6F**: MODERATE-HIGH (~55-65%)
- **Implementation introduces regression**: LOW — T6F baseline unchanged, new profiles opt-in only

**Risk mitigation**: 500-step diagnostic is mandatory gate; no long evaluation without passing.

---

## Contingency Plan

### If Both T6H and T6I FAIL 500-Step

1. **Analyze failure modes**:
   - Document root cause for each candidate
   - Compare failure mechanisms between T6H and T6I
   - Identify common failure pattern if exists

2. **Extract learning**:
   - Did blend factors cause instability (T6H)?
   - Did convergence detection fail (T6I)?
   - Is overshoot mechanism different than hypothesized?

3. **Return to T6F baseline**:
   - T6F (T6F_budget_cap_raise) is validated and stable
   - Proceed with T6F 2000-step screening for Step E
   - Document T6F as current best high_0p480 profile

4. **Long-term options**:
   - Investigate root cause of T6F sign incorrectness (gain tuning, IK geometry)
   - Explore alternative heights (high_0p450, high_0p420) if 0p480 is inherently difficult
   - Accept T6F performance (0.203m max drift) as sufficient for current research phase

**Do NOT**: Force through unstable candidates or continue tuning indefinitely without progress.

---

## Constraints and Restrictions (Reminder)

### DO NOT (until explicit approval):
- ❌ Implement T6H or T6I (design approval required first)
- ❌ Run 1200-step evaluation for any profile
- ❌ Run 2000-step evaluation for any profile
- ❌ Run 5000-step evaluation for any profile
- ❌ Proceed to Step C validation
- ❌ Proceed to Step D validation
- ❌ Commit T6F_sign_corrected
- ❌ Make T6F_sign_corrected default
- ❌ Modify T5, T6F, or other baselines

### DO (documentation phase complete):
- ✅ Archive T6F_sign_corrected as failed design (complete)
- ✅ Document root cause reframing (complete)
- ✅ Design safer alternatives (complete)
- ✅ Present recommendation (complete)

**Awaiting user approval to proceed with T6H/T6I implementation.**

---

## Communication to Stakeholders

### Key Messages

1. **T6F_sign_corrected investigation was successful**:
   - All implementation bugs identified and fixed
   - Design hypothesis tested rigorously and invalidated
   - Systematic debugging methodology worked as intended

2. **Learning achieved**:
   - Sign incorrectness is symptom, not cause
   - Stabilization authority must be preserved
   - Component-level optimization can degrade system-level behavior

3. **Path forward is clear**:
   - Two safer candidates designed (T6H, T6I)
   - Both preserve stabilization authority
   - 500-step diagnostic will validate or reject each candidate
   - T6F baseline remains available as fallback

4. **No long evaluation time wasted**:
   - 500-step diagnostic caught design failure early
   - Prevented wasting time on 1200/2000/5000-step evaluation
   - Rapid iteration on new designs now possible

### Timeline

- Design phase: Complete (2026-06-12)
- Implementation phase: 4-6 days (pending approval)
- 500-step diagnostic: 0.5 days
- Decision point: ~1 week from approval

### Success Metrics

- At least one candidate improves max drift to < 0.20m (vs T6F 0.203m)
- Both candidates maintain zero mode transitions
- Sign correctness is diagnostic only, not a gate

---

## Conclusion

**Decision**: DESIGN_T6H_AND_T6I_NEXT ✅

**Rationale**:
- Two complementary hypotheses (soft blend vs phase-aware release)
- Both preserve stabilization authority (learned from T6F_sign_corrected failure)
- Moderate implementation cost, high learning value
- 500-step diagnostic provides safety gate
- T6F baseline available as fallback

**Recommended implementation order**: Parallel (T6H and T6I simultaneously) for comparative evaluation.

**Contingency**: If both candidates fail 500-step, return to T6F baseline for 2000-step screening.

**Next action**: Await user approval to proceed with T6H/T6I implementation.

---

## Appendix: Decision Matrix

| Option | Risk | Cost | Learning Value | Success Prob | Recommendation |
|--------|------|------|----------------|--------------|----------------|
| RETURN_TO_T5 | LOW | ZERO | LOW | N/A | ❌ REJECT |
| T6H_ONLY | MODERATE | LOW | MODERATE | 60-70% | ⚠️ ACCEPTABLE |
| T6I_ONLY | MODERATE | MODERATE | MODERATE | 65-75% | ⚠️ ACCEPTABLE |
| **T6H_AND_T6I** | **LOW-MODERATE** | **MODERATE** | **HIGH** | **55-65% either** | ✅ **RECOMMENDED** |
| STOP_TUNING | LOW | ZERO | LOW | N/A | ❌ REJECT |

**Selected**: DESIGN_T6H_AND_T6I_NEXT

---

**End of Recommendation**
