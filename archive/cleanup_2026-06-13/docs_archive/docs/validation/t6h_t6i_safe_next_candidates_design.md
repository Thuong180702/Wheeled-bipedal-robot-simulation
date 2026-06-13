# T6H/T6I Safe Next Candidates Design

**Date**: 2026-06-12  
**Context**: Post-T6F_sign_corrected design invalidation  
**Status**: DESIGN ONLY — DO NOT IMPLEMENT YET

---

## Executive Summary

Two safer candidate profiles are proposed to improve high_0p480 drift behavior while avoiding the T6F_sign_corrected failure modes:

1. **T6H_soft_blend_arch_fix**: Soft modulation approach (reduce fighting terms by 50%, not 100%)
2. **T6I_phase_aware_release**: Phase-aware release approach (detect convergence, decay cap gradually)

**Both candidates preserve continuous pitch control and velocity damping**, learning from T6F_sign_corrected's fundamental flaw.

**Implementation timeline**: Design approval required before implementation. 500-step diagnostic mandatory before any long evaluation.

---

## Design Principles (Post-T6F_sign_corrected)

### Learned Constraints

Based on T6F_sign_corrected empirical failure:

1. ✅ **MUST preserve pitch stabilization**: Never zero pitch torque
2. ✅ **MUST preserve velocity damping**: Never completely disable damping
3. ✅ **MUST use soft modulation**: Blend factors ∈ [0.5, 1.0], not {0, 1}
4. ✅ **MUST use gradual transitions**: Exponential fade, not step discontinuities
5. ✅ **MUST add safety overrides**: Restore full control if pitch/velocity exceeds safe thresholds
6. ✅ **MUST optimize primary metrics**: Drift and pitch, not sign correctness

### Architecture Inheritance

Both candidates inherit from T6F_budget_cap_raise:

- ✅ Keep arch_fix cap raise mechanism (4.0 → 8.0 Nm during emergency)
- ✅ Keep band state logic (normal/soft/desired/hard/emergency)
- ✅ Keep position authority budget and sagittal schedule
- ✅ Keep all safety gates (WBC, hidden joint, ownership, Step E)

**Only add**: Soft modulation or phase-aware release on top of working T6F architecture.

---

## Candidate A: T6H_soft_blend_arch_fix

### Design Intent

**Hypothesis**: T6F_sign_corrected failed because it removed stabilization authority completely (pitch suppression = 0.0, damping disabled). Reducing fighting terms by 50% instead of 100% may improve convergence while preserving partial stabilization.

**Goal**: Reduce overshoot and improve drift convergence without causing instability.

### Architecture

**Profile name**: `T6H_soft_blend_arch_fix`

**Base**: T6F_budget_cap_raise (inherits cap raise, band state, safety gates)

**New features**:

1. **Soft pitch blending during arch_fix**
2. **Soft damping blending during arch_fix**
3. **Pitch excursion safety override**
4. **Wheel velocity safety override**

### Feature 1: Soft Pitch Blending

**Activation condition**:
```python
soft_pitch_blend_active = (
    arch_fix_active and
    abs(sagittal_error) > 0.10  # same threshold as T6F_sign_corrected
)
```

**Blending logic**:
```python
if soft_pitch_blend_active:
    pitch_blend_factor = 0.50  # reduce by 50%, not 100%
    tau_pitch *= pitch_blend_factor
else:
    pitch_blend_factor = 1.0  # full pitch control
```

**Rationale**: 
- T6F_sign_corrected used `pitch_blend_factor = 0.0` → pitch grew to 19.7°
- T6H uses `pitch_blend_factor = 0.5` → preserve 50% pitch stabilization
- If error > 0.10m, reduce pitch authority but don't remove it

### Feature 2: Soft Damping Blending

**Activation condition**:
```python
soft_damping_blend_active = (
    arch_fix_active and
    wheel_velocity_opposes_error_correction
)
```

**Blending logic**:
```python
if soft_damping_blend_active:
    damping_blend_factor = 0.50  # reduce by 50%, not 100%
    velocity_damping_gain *= damping_blend_factor
else:
    damping_blend_factor = 1.0  # full damping
```

**Rationale**:
- T6F_sign_corrected disabled damping completely → drift grew to 0.383m
- T6H reduces damping by 50% → preserve partial energy dissipation
- If wheel velocity transiently opposes correction, reduce damping but don't remove it

### Feature 3: Pitch Excursion Safety Override

**Safety condition**:
```python
pitch_safety_active = (abs(pitch) > 10.0 * DEG_TO_RAD)
```

**Override logic**:
```python
if pitch_safety_active:
    pitch_blend_factor = 1.0  # restore full pitch control
    # log warning: pitch excursion safety triggered
```

**Rationale**: If pitch exceeds 10°, immediately restore full pitch control regardless of other conditions to prevent runaway growth.

### Feature 4: Wheel Velocity Safety Override

**Safety condition**:
```python
wheel_velocity_safety_active = (abs(mean_wheel_velocity) > 7.0)  # rad/s
```

**Override logic**:
```python
if wheel_velocity_safety_active:
    damping_blend_factor = 1.0  # restore full damping
    # log warning: wheel velocity safety triggered
```

**Rationale**: If wheel velocity exceeds 7.0 rad/s, immediately restore full damping to dissipate energy.

### Telemetry Fields (Required)

Add to telemetry CSV:

```python
"t6h_soft_pitch_blend_active": bool
"t6h_pitch_blend_factor": float  # 0.5 or 1.0
"t6h_soft_damping_blend_active": bool
"t6h_damping_blend_factor": float  # 0.5 or 1.0
"t6h_pitch_safety_active": bool
"t6h_wheel_velocity_safety_active": bool
```

### Expected Behavior

**Best case**:
- Reduces overshoot by 20-30% (blend factors reduce fighting terms)
- Max drift: 0.15-0.18m (improvement over T6F's 0.203m)
- Pitch excursion: 6-8° (comparable to or better than T6F's 8.4°)
- No mode transitions (blend factors preserve stabilization)
- Sign correctness: 50-55% (not a target, but may improve slightly)

**Acceptable case**:
- Max drift: 0.18-0.21m (comparable to T6F's 0.203m)
- Pitch excursion: 8-10° (slightly worse than T6F but within safety limit)
- No mode transitions

**Failure case** (reject if observed):
- Max drift > 0.25m (worse than T6F)
- Pitch excursion > 12° (safety boundary)
- Transition/recovery steps > 0 (mode instability)

### Tests Required Before 500-Step Diagnostic

1. **Blend factor bounds test**:
   - Verify `pitch_blend_factor ∈ {0.5, 1.0}`
   - Verify `damping_blend_factor ∈ {0.5, 1.0}`
   - Never 0.0

2. **Safety override test**:
   - If pitch = 12°, verify `pitch_blend_factor = 1.0`
   - If wheel_vel = 8.0 rad/s, verify `damping_blend_factor = 1.0`

3. **Telemetry presence test**:
   - Verify all 6 T6H telemetry fields present in CSV

4. **Integration test**:
   - 100-step smoke test at high_0p480
   - Verify no NaN, no falls, telemetry written

### 500-Step Diagnostic Gates

**PASS criteria** (proceed to 1200-step if ALL met):
- Max abs error < 0.21m (equal or better than T6F 0.203m)
- Final error < 0.15m
- Max pitch < 11°
- Transition steps = 0
- Recovery steps = 0
- Terminated = False

**REJECT criteria** (abandon T6H if ANY met):
- Max abs error > 0.25m
- Max pitch > 12°
- Transition steps > 0
- Terminated = True

**INCONCLUSIVE** (if between PASS and REJECT):
- Run 3 more 500-step seeds
- Average results must meet PASS criteria

---

## Candidate B: T6I_phase_aware_release

### Design Intent

**Hypothesis**: T6F holds high authority (8.0 Nm cap) during entire hard/emergency band. This may cause overshoot because the controller continues pushing at high authority even when error is converging. Detecting convergence and gradually releasing authority may reduce overshoot.

**Goal**: Reduce overshoot by exiting arch_fix smoothly when error trajectory improves.

### Architecture

**Profile name**: `T6I_phase_aware_release`

**Base**: T6F_budget_cap_raise (inherits cap raise, band state, safety gates)

**New features**:

1. **Error convergence detection**
2. **Gradual cap decay during convergence**
3. **Rate-limited cap transitions**

**DO NOT add**: Pitch suppression or damping override (preserve full stabilization)

### Feature 1: Error Convergence Detection

**State tracking**:
```python
# State variables (persistent across steps)
error_trajectory_history = deque(maxlen=10)  # last 10 steps
```

**Convergence detection**:
```python
# Update history
error_trajectory_history.append(sagittal_error)

# Compute trajectory
if len(error_trajectory_history) >= 5:
    recent_errors = list(error_trajectory_history)[-5:]
    error_trend = recent_errors[-1] - recent_errors[0]  # negative if converging toward zero
    
    # Convergence condition
    converging = (
        abs(sagittal_error) < 0.12 and  # error not too large
        sign(sagittal_error) == sign(error_trend) and  # error and trend same sign
        abs(error_trend) < 0.03  # error decreasing (converging)
    )
else:
    converging = False
```

**Rationale**: If error magnitude is decreasing over 5 steps and error is below 0.12m, the controller is successfully recovering. Begin releasing authority.

### Feature 2: Gradual Cap Decay During Convergence

**Cap logic**:
```python
# Current T6F logic
if band_state in [HARD, EMERGENCY]:
    target_cap = 8.0  # high authority
else:
    target_cap = 4.0  # normal authority

# T6I modification
if converging:
    # Decay target cap from 8.0 toward 4.0
    decay_rate = 0.10  # Nm per step (takes 40 steps to decay from 8.0 to 4.0)
    target_cap = max(4.0, current_cap - decay_rate)
else:
    # Normal T6F logic
    if band_state in [HARD, EMERGENCY]:
        target_cap = 8.0
    else:
        target_cap = 4.0
```

**Rationale**: Instead of holding cap at 8.0 until error < threshold (T6F behavior), gradually reduce cap when convergence is detected. This reduces overshoot by lowering authority as error approaches zero.

### Feature 3: Rate-Limited Cap Transitions

**Smooth cap transitions**:
```python
# Limit cap change per step
MAX_CAP_DELTA_PER_STEP = 0.30  # Nm

cap_delta = target_cap - current_cap
if abs(cap_delta) > MAX_CAP_DELTA_PER_STEP:
    cap_delta = sign(cap_delta) * MAX_CAP_DELTA_PER_STEP

new_cap = current_cap + cap_delta
```

**Rationale**: Avoid step discontinuities in authority. Smooth transitions reduce control discontinuities that can trigger instability.

### Telemetry Fields (Required)

Add to telemetry CSV:

```python
"t6i_error_converging": bool
"t6i_error_trend": float  # change in error over last 5 steps
"t6i_target_cap": float  # Nm
"t6i_current_cap": float  # Nm
"t6i_cap_delta_this_step": float  # Nm
"t6i_cap_change_rate_limited": bool  # True if delta was clipped
```

### Expected Behavior

**Best case**:
- Reduces overshoot by detecting convergence and releasing authority smoothly
- Max drift: 0.15-0.18m (improvement over T6F's 0.203m)
- Smoother error trajectory (less oscillation after arch_fix exit)
- Faster settling time (cap decays as error converges)
- No mode transitions

**Acceptable case**:
- Max drift: 0.18-0.21m (comparable to T6F's 0.203m)
- Slightly faster settling time
- No mode transitions

**Failure case** (reject if observed):
- Max drift > 0.25m (worse than T6F)
- Premature authority release causes secondary divergence
- Transition/recovery steps > 0 (mode instability)

### Tests Required Before 500-Step Diagnostic

1. **Convergence detection test**:
   - Construct synthetic error trajectory: [0.15, 0.14, 0.13, 0.12, 0.11]
   - Verify `converging = True`
   - Construct diverging trajectory: [0.11, 0.12, 0.13, 0.14, 0.15]
   - Verify `converging = False`

2. **Cap decay test**:
   - If `converging = True` and `current_cap = 8.0`, verify `target_cap = 7.9` after 1 step
   - After 40 steps, verify `target_cap = 4.0` (decay complete)

3. **Rate limit test**:
   - If `current_cap = 4.0` and `target_cap = 8.0`, verify `cap_delta = 0.30` (not 4.0)
   - After 1 step, verify `new_cap = 4.30`

4. **Telemetry presence test**:
   - Verify all 6 T6I telemetry fields present in CSV

5. **Integration test**:
   - 100-step smoke test at high_0p480
   - Verify no NaN, no falls, telemetry written

### 500-Step Diagnostic Gates

**PASS criteria** (proceed to 1200-step if ALL met):
- Max abs error < 0.21m (equal or better than T6F 0.203m)
- Final error < 0.15m
- Max pitch < 11°
- Transition steps = 0
- Recovery steps = 0
- Terminated = False
- No premature release causing secondary divergence

**REJECT criteria** (abandon T6I if ANY met):
- Max abs error > 0.25m
- Max pitch > 12°
- Transition steps > 0
- Terminated = True
- Convergence detector triggers prematurely (error diverges after release)

**INCONCLUSIVE** (if between PASS and REJECT):
- Run 3 more 500-step seeds
- Average results must meet PASS criteria

---

## Comparison: T6F vs T6H vs T6I

| Feature | T6F Baseline | T6H Soft Blend | T6I Phase Aware |
|---------|--------------|----------------|-----------------|
| Cap raise during emergency | ✅ 4.0 → 8.0 Nm | ✅ 4.0 → 8.0 Nm | ✅ 4.0 → 8.0 Nm |
| Pitch control | ✅ Always full | ✅ 50-100% blend | ✅ Always full |
| Velocity damping | ✅ Always full | ✅ 50-100% blend | ✅ Always full |
| Cap hold strategy | Hold at 8.0 until error < threshold | Hold at 8.0, soft blend pitch/damping | **Gradual decay when converging** |
| Transition smoothness | Step discontinuity | Soft blend factors | **Rate-limited cap** |
| Safety overrides | None (implicit) | **Pitch/wheel safety** | Convergence validation |
| Max drift (expected) | 0.203m | 0.15-0.21m | 0.15-0.21m |

### Design Trade-offs

**T6H advantages**:
- May reduce fighting terms during arch_fix
- Explicit safety overrides (pitch > 10°, wheel_vel > 7.0)
- Simpler logic (just blend factors)

**T6H risks**:
- Blend factors may still degrade stability (though 50% safer than T6F_sign_corrected's 0%)
- May not improve much over T6F if fighting terms are not the issue

**T6I advantages**:
- Preserves full pitch and damping authority (avoids T6F_sign_corrected failure mode)
- Addresses potential overshoot mechanism directly
- Smooth cap transitions (no discontinuities)

**T6I risks**:
- Convergence detector may trigger prematurely
- May release authority too early, causing secondary divergence
- More complex logic (state tracking, trajectory estimation)

---

## Implementation Phases (When Approved)

### Phase 1: T6H Implementation

1. Add T6H soft blend logic to `sagittal_velocity_damped_balance_controller.py`
2. Add T6H telemetry fields
3. Add `T6H_soft_blend_arch_fix` profile to authority schedule registry
4. Run unit tests (blend factor bounds, safety overrides, telemetry presence)
5. Run 100-step smoke test at high_0p480
6. Run 500-step diagnostic (T5, T6F, T6H)
7. Analyze results and classify (PASS/REJECT/INCONCLUSIVE)

### Phase 2: T6I Implementation (If T6H Approved or In Parallel)

1. Add T6I convergence detection logic to `sagittal_velocity_damped_balance_controller.py`
2. Add T6I cap decay and rate limiting logic
3. Add T6I telemetry fields
4. Add `T6I_phase_aware_release` profile to authority schedule registry
5. Run unit tests (convergence detection, cap decay, rate limit, telemetry presence)
6. Run 100-step smoke test at high_0p480
7. Run 500-step diagnostic (T5, T6F, T6I)
8. Analyze results and classify (PASS/REJECT/INCONCLUSIVE)

### Phase 3: Comparative Evaluation (If Both Candidates Pass 500-Step)

1. Run 1200-step diagnostic for T6F, T6H, T6I
2. Compare drift, pitch, torque, energy metrics
3. Select best candidate for 2000-step screening
4. Document winner and rationale

---

## Rejection Protocol

If a candidate FAILS 500-step diagnostic:

1. **Classify failure mode**:
   - Drift degradation (max error > 0.25m)
   - Pitch instability (max pitch > 12°)
   - Mode instability (transition/recovery steps > 0)
   - Fall (terminated = True)

2. **Document root cause** (e.g., `docs/validation/t6h_500_fail_report.md`)

3. **DO NOT proceed to 1200-step**

4. **DO NOT attempt parameter tuning** unless failure mode is clearly a threshold issue (e.g., blend factor 0.5 too aggressive, try 0.7)

5. **Consider alternative approach** or return to T6F baseline

**Avoid sunk-cost fallacy**: If both T6H and T6I fail, return to T6F for long evaluation. Do not force through candidates that degrade stability.

---

## Success Criteria for Candidate Approval

A candidate is approved for long evaluation (1200/2000-step) if:

1. ✅ **500-step diagnostic PASS** (all primary metrics equal or better than T6F)
2. ✅ **No mode instability** (transition/recovery steps = 0)
3. ✅ **No safety violations** (pitch < 12°, drift < 0.25m)
4. ✅ **Comparable or better efficiency** (torque RMS, energy proxy)
5. ✅ **Implementation verified** (all tests pass, telemetry present)

**Sign correctness is NOT a gate**: A candidate with 45% sign correctness can proceed if primary metrics pass.

---

## Design Confidence Assessment

### T6H Confidence: MODERATE

**Pros**:
- Avoids T6F_sign_corrected's fatal flaw (preserves 50% stabilization)
- Simple implementation
- Explicit safety overrides

**Cons**:
- Blend factors still modify stabilization terms (some risk)
- May not improve much over T6F if fighting terms are not the issue

**Estimated success probability**: 60-70%

### T6I Confidence: MODERATE-HIGH

**Pros**:
- Preserves full pitch and damping authority (safest)
- Directly addresses potential overshoot mechanism
- Smooth cap transitions avoid discontinuities

**Cons**:
- Convergence detection adds complexity
- May release authority prematurely
- Requires careful threshold tuning

**Estimated success probability**: 65-75%

### Recommendation

**Implement both candidates in sequence** (T6H first, then T6I) or **in parallel** if resources allow. Comparative 500-step diagnostic will reveal which approach is more promising.

If both fail 500-step diagnostic, **return to T6F baseline** for long evaluation. T6F is validated, stable, and ready for 2000-step screening.

---

## Conclusion

**T6H_soft_blend_arch_fix** and **T6I_phase_aware_release** are safer alternatives to T6F_sign_corrected, learning from its failure:

1. ✅ Preserve continuous pitch control
2. ✅ Preserve continuous velocity damping
3. ✅ Use soft modulation or phase-aware logic, not hard suppression
4. ✅ Add safety overrides and rate limits
5. ✅ Optimize primary metrics (drift, pitch), not sign correctness

**Implementation requires approval**. 500-step diagnostic is mandatory before any long evaluation.

**If both candidates fail**: Return to T6F baseline. Do not force through unstable designs.

---

**End of Safe Next Candidates Design**
