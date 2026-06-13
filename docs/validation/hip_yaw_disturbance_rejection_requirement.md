# Hip-Yaw Disturbance Rejection Requirement

**Date:** 2026-06-04  
**Status:** ACTIVE INVESTIGATION  
**Task:** Hip-yaw disturbance rejection under support drift

---

## Requirement Clarification

### Previous Misinterpretation

The hip-yaw boundary audit (2026-06-04) correctly identified that:
- Support position drift precedes hip-yaw drift by 329 steps (3.29 seconds)
- Hip-yaw controller references and signs are correct
- Hip-yaw torque is not saturated, not rate-limited, not overwritten
- Hip-yaw drift appears to be a symptom of upstream support drift

**However, this does NOT mean hip-yaw drift is acceptable.**

### Correct Interpretation

Hip-yaw posture control must **reject disturbances** regardless of what triggers them.

Even if support position drifts, the hip-yaw controller must maintain:
- `hip_yaw_abs_max <= 0.07 rad`
- `percent hip_yaw_abs_max > 0.10 rad = 0`
- Left/right hip-yaw divergence bounded
- No leg inward/outward collapse

This is a **disturbance rejection requirement**, not a root-cause elimination requirement.

---

## Current Failure Mode

### Baseline Behavior (low_0p300)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| hip_yaw_abs_max | 0.2137 rad | ≤ 0.07 rad | **FAIL (3.05×)** |
| support_position_error_max | 0.243 m | ≤ 0.15 m | **FAIL (1.62×)** |
| pitch_x_max | 0.095 rad | ≤ 0.10 rad | PASS |
| roll_y_max | 0.015 rad | ≤ 0.05 rad | PASS |

### Visual Failure

- Legs visibly rotate inward/outward
- Posture validity fails
- Robot appearance: "collapsed inward" or "splayed outward"

### Temporal Pattern

1. Support position begins drifting forward (step 89)
2. Hip-yaw remains stable initially
3. Hip-yaw begins drifting (step 418, delay=329 steps)
4. Hip-yaw drift continues despite correct controller torque (3.3 Nm applied)

---

## Why "Symptom" Does Not Mean "Acceptable"

### Control Systems Perspective

In a hierarchical control system:
- **Root cause:** Insufficient sagittal position authority
- **Disturbance:** Support drift creates coupling force on hip-yaw
- **Controller responsibility:** Hip-yaw posture controller must reject this disturbance

**Analogy:** If wind pushes an aircraft off course:
- Root cause: wind exists
- Disturbance: lateral force
- Controller responsibility: aileron control must reject lateral disturbance

The aircraft does NOT get to fail lateral stability just because "wind is the root cause."

### Step D/E Passing Criteria

Step E explicitly requires:
- Valid posture throughout episode
- All joint errors within threshold
- No visual collapse

**Hip-yaw drift > 0.07 rad constitutes posture failure**, regardless of what triggered it.

---

## Investigation Scope

This investigation focuses on:

1. **Why does hip-yaw keep drifting even though torque is correct?**
   - Damping insufficient?
   - Disturbance feedforward needed?
   - Control authority insufficient at extreme flexion?
   - Phase lag between error and torque response?

2. **Can hip-yaw disturbance rejection be improved without fixing support drift?**
   - Height-dependent damping schedule?
   - Support-drift-aware compensation?
   - Body-yaw coupling correction?

3. **What is the coupling mechanism between support drift and hip-yaw?**
   - Contact force asymmetry?
   - Kinematic coupling at extreme flexion?
   - Body pitch compensation creating yaw torque?

---

## Success Criteria

### Hip-Yaw Gate (Required)

- `hip_yaw_abs_max <= 0.07 rad`
- `percent hip_yaw_abs_max > 0.10 rad = 0`
- Left/right divergence bounded

### Support Position (Must Not Worsen)

- Support drift must not worsen by more than 10%
- If hip-yaw fix degrades support: **REJECT FIX**

### Other Constraints

- Pitch, roll, height, contact remain valid
- WBC remains disabled
- Ownership violations = 0
- Hidden torque = 0

---

## Relationship to Sagittal Fix

### Two Parallel Problems

1. **Sagittal position authority insufficient** (continuous k_position fix failed)
2. **Hip-yaw disturbance rejection insufficient** (this investigation)

### Possible Outcomes

**Outcome A:** Hip-yaw fix successful without support fix
- Hip-yaw rejects support-drift disturbance effectively
- Support drift persists but hip-yaw stays within threshold
- **Decision:** Accept hip-yaw fix, continue sagittal investigation separately

**Outcome B:** Hip-yaw requires support fix first
- No hip-yaw gain/damping/compensation profile can reject disturbance
- Coupling mechanism too strong
- **Decision:** Return to sagittal fix, implement joint solution

**Outcome C:** Hip-yaw and support must be fixed jointly
- Hip-yaw fix improves hip-yaw but worsens support
- Support fix improves support but worsens hip-yaw
- **Decision:** Design coupled sagittal-yaw controller

**Outcome D:** Hip-yaw disturbance rejection achieved, support improved as side effect
- Hip-yaw compensation indirectly reduces support drift
- Both problems solved
- **Decision:** Accept hip-yaw fix as primary solution

---

## Investigation Protocol

### Phase 2: Mandatory Isolation Experiments

Run controlled experiments to determine:
- Can hip-yaw reject disturbance with higher damping?
- Can hip-yaw reject disturbance with support-error feedforward?
- Does preventing hip-yaw drift improve or worsen support drift?
- What is the disturbance magnitude vs controller authority?

### Phase 3: Mechanism Classification

Analyze telemetry to classify:
- `hip_yaw_damping_insufficient_under_support_disturbance`
- `hip_yaw_kp_insufficient_under_support_disturbance`
- `hip_yaw_support_drift_feedforward_needed`
- `hip_yaw_not_fixable_without_support_fix`
- `hip_yaw_control_conflicts_with_support_position`

### Phase 4: Targeted Fix Implementation

Implement only justified fixes:
- Continuous hip-yaw damping schedule (HY-D)
- Continuous hip-yaw kp+kd schedule (HY-PD)
- Support-drift-aware hip-yaw compensation (HY-FF)
- Hybrid approach (HY-COMBO)

### Phase 5: Evaluation

Test candidates at low_0p300, high_0p480, nominal.

Accept only if:
- Hip-yaw gate passed
- Support not worsened >10%
- No regression on other metrics

---

## Restrictions

Do NOT:
- Add WBC
- Globally increase hip-yaw gains
- Use discontinuous schedules
- Use variant-name-only patches
- Relax thresholds
- Proceed to Step D until resolved

DO:
- Run isolation experiments
- Use continuous height-based schedules if justified
- Implement support-aware compensation if justified
- Add diagnostic telemetry
- Update tests
- Document mechanism

---

## Expected Timeline

- Phase 2 (isolation experiments): 1-2 runs
- Phase 3 (mechanism classification): analysis
- Phase 4 (fix implementation): 1 implementation cycle
- Phase 5 (evaluation): multi-height validation
- Phase 6 (tests): test updates
- Final report: synthesis

---

## Conclusion

Hip-yaw disturbance rejection is a **hard requirement** for Step E passing.

The fact that support drift triggers hip-yaw drift does NOT excuse the controller from rejecting this disturbance.

This investigation will determine whether hip-yaw can be fixed independently, whether it requires the sagittal fix first, or whether a coupled solution is needed.

**The goal is not to eliminate the disturbance source—it is to reject the disturbance effectively.**
