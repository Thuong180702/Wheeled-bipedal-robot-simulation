# High_0p480 Recenter Root Cause Reframing

**Date**: 2026-06-12  
**Context**: Post-T6F_sign_corrected design invalidation  
**Purpose**: Update interpretation of sign incorrectness and drift behavior

---

## Executive Summary

The T6F_sign_corrected experiment revealed that **sign incorrectness is likely a symptom of coupled dynamics, not a root cause of drift instability**.

**Old hypothesis** (invalidated):
> Final torque sign incorrectness causes drift. Correcting signs will improve stability.

**New interpretation** (evidence-based):
> Sign incorrectness is a symptom of coupled pitch-wheel-phase dynamics. Pitch torque and velocity damping are stabilizing terms that may appear "wrong" by component-level inspection but are phase-appropriate for global stability. Removing these terms to "fix" signs degrades stability.

---

## What We Learned from T6F_sign_corrected Failure

### Empirical Evidence

| Intervention | Sign Correctness | Max Drift | Pitch Excursion | Mode Transitions |
|--------------|------------------|-----------|-----------------|------------------|
| T6F baseline | 48.9% | 0.203m | 8.4° | 0 |
| T6F_sign_corrected | 43.5% (-5.4pp) | 0.383m (+88%) | 19.7° (+135%) | 152 |

**Key finding**: "Fixing" sign incorrectness made both sign correctness AND stability worse.

### Mechanistic Interpretation

#### 1. Pitch Suppression Removed Stabilization, Not "Bad Control"

**Implementation**:
```python
if arch_fix_active and abs(error) > 0.10:
    tau_pitch = 0.0  # "correct" the sign by zeroing
```

**Result**:
- Pitch excursion: -12.5° to +19.7° (vs T6F -0.5° to +8.4°)
- Large pitch changes required correction direction
- New sign conflicts emerged downstream

**Interpretation**: The original pitch torque was **phase-appropriate stabilization**, not a sign error. Zeroing it allowed pitch to grow unchecked.

#### 2. Damping Override Removed Energy Dissipation, Not "Opposition"

**Implementation**:
```python
if wheel_velocity_opposes_error_correction:
    disable_velocity_damping()  # "correct" by removing damping
```

**Result**:
- Max drift: 0.383m (vs T6F 0.203m, +88%)
- System became underdamped
- Recovery oscillations amplified

**Interpretation**: The original damping was **energy dissipation**, not fighting. The wheel velocity direction may be **transiently opposite but dynamically stabilizing**.

#### 3. Sign Correctness Paradox

**Expectation**: Correcting signs → sign correctness improves

**Reality**: Sign correctness degraded from 48.9% to 43.5%

**Mechanism**: Removing stabilization terms caused:
- Larger pitch excursions → changed required correction direction → new sign conflicts
- Larger drift → new error regions with different sign conventions
- Controller mode transitions → different control laws with different sign semantics

**Conclusion**: Sign incorrectness is a **symptom of the underlying coupled dynamics**, not an independent cause.

---

## Updated Root Cause Model

### Wheeled Biped Sagittal Balance as Coupled System

High_0p480 balance is governed by coupled pitch-wheel-phase dynamics:

```
pitch_accel = f(pitch, pitch_vel, wheel_pos, wheel_vel, tau_pitch, tau_wheel)
wheel_accel = g(pitch, pitch_vel, wheel_pos, wheel_vel, tau_pitch, tau_wheel)
```

**Key coupling mechanisms**:

1. **Pitch-wheel geometric coupling**: CoM height 0.480m creates moment arm; wheel acceleration affects pitch and vice versa

2. **Phase-dependent sign convention**: 
   - During forward drift: wheel accelerates backward (negative tau)
   - But pitch may tilt forward during transient → tau_pitch appears "wrong sign"
   - This is **phase lag**, not error

3. **Damping as energy dissipation**: Velocity damping opposes wheel velocity, which may transiently oppose error correction, but dissipates energy and prevents overshoot

4. **Support phase transitions**: During arch_fix, robot may transition from "recovering" to "overshooting" → sign convention flips mid-recovery

### Why Component-Level Sign Inspection Fails

**Component-level view** (incorrect):
> If error > 0 (forward drift), then tau_wheel should be < 0 (backward correction). If tau_wheel > 0, it's wrong.

**System-level view** (correct):
> If error > 0 and pitch is tilting forward during transient, tau_pitch may be positive to arrest pitch growth. Wheel torque may be positive if overshooting backward and needs forward correction. Both are phase-appropriate despite appearing "wrong" instantaneously.

**Implication**: **Do not optimize component-level sign correctness at the expense of global stability metrics.**

---

## Correct Metrics Hierarchy

### Primary Metrics (Stability)

These metrics directly measure task success and safety:

1. **Drift bounds**:
   - `outside_0p08_m_count` / `outside_0p08_m_pct`
   - `outside_0p10_m_count` / `outside_0p10_m_pct`
   - `outside_0p15_m_count` / `outside_0p15_m_pct`
   - `max_abs_error_m`
   - `final_error_m`
   - `peak_to_peak_drift_m`

2. **Pitch/roll stability**:
   - `max_pitch_deg`, `rms_pitch_deg`
   - `max_roll_deg`, `rms_roll_deg`
   - `pitch_excursion_range_deg`

3. **Controller mode stability**:
   - `transition_steps` = 0 (no mode transitions)
   - `recovery_steps` = 0 (no recovery mode)
   - `upright_pct` = 100% (always in upright mode)

4. **Contact stability**:
   - `both_wheels_contact_pct` > 95%
   - `unilateral_contact_pct` < 5%
   - `no_contact_pct` = 0%

5. **Termination**:
   - `terminated` = False (no falls)
   - `survival_time_s` = full episode

**Target for high_0p480**:
- `max_abs_error` < 0.15m (stretch: < 0.10m)
- `outside_0p10_m_pct` < 50%
- `max_pitch` < 10°
- `transition_steps` = 0
- `terminated` = False

### Secondary Metrics (Efficiency)

These metrics measure control effort and energy, important for optimization but not task-critical:

1. **Torque effort**:
   - `mean_abs_tau`, `rms_tau`, `max_tau`
   - `energy_proxy` = Σ|tau · qdot|

2. **Wheel velocity**:
   - `mean_abs_wheel_vel`, `rms_wheel_vel`, `max_wheel_vel`

3. **Action smoothness**:
   - `action_rate_rms`
   - `action_discontinuity_count`

### Tertiary Metrics (Diagnostic Only)

These metrics are useful for understanding controller behavior but should NOT be optimization targets:

1. **Sign correctness**:
   - `final_torque_sign_correctness_pct`
   - `sign_correctness_during_arch_fix_pct`
   - `sign_correctness_high_torque_pct`

2. **Arch fix activation**:
   - `arch_fix_active_pct`
   - `arch_fix_activation_rate_during_emergency`

3. **Band state distribution**:
   - `normal_pct`, `soft_pct`, `desired_pct`, `hard_pct`, `emergency_pct`

**Why tertiary**: These metrics describe internal controller state, not task success. A controller can have 40% sign correctness and still achieve stable balance with low drift.

---

## Decision Framework for Future Candidates

### Acceptance Criteria (500-Step Diagnostic)

A new profile must meet ALL primary criteria to proceed to 1200-step:

| Criterion | Target | Measured Against | Status Required |
|-----------|--------|------------------|-----------------|
| Sign correctness | Diagnostic only | N/A | (not a gate) |
| Max abs error | < 0.15m | T6F: 0.203m | IMPROVE or MATCH |
| Final error | < 0.10m | T6F: 0.141m | IMPROVE or MATCH |
| Peak-to-peak | < 0.25m | T6F: 0.219m | IMPROVE or MATCH |
| Outside ±0.10m | < 50% | T6F: 39.1% | IMPROVE or MATCH |
| Max pitch | < 12° | T6F: 8.4° | IMPROVE or MATCH |
| Transition steps | 0 | T6F: 0 | MATCH |
| Recovery steps | 0 | T6F: 0 | MATCH |
| Terminated | False | T6F: False | MATCH |

**Rejection criteria**:
- ANY primary metric significantly worse than T6F baseline → REJECT
- Transition/recovery steps > 0 → REJECT (indicates mode instability)
- Max drift > 0.25m → REJECT (safety boundary)

### Diagnostic Use of Sign Correctness

Sign correctness remains useful for:

1. **Understanding mechanism**: Does the controller rely on high-torque corrections with correct signs, or low-torque modulation with mixed signs?

2. **Debugging**: If drift is high AND sign correctness is low, investigate whether gains/IK are fundamentally mistuned

3. **Comparative analysis**: Does candidate A achieve same drift as candidate B with higher or lower sign correctness? (Efficiency insight)

**But**: Never reject a candidate solely because sign correctness < 80% if primary metrics pass.

---

## Implications for High_0p480 Tuning Strategy

### What NOT to Do

❌ **Do not optimize sign correctness directly**
- T6F_sign_corrected showed this degrades stability

❌ **Do not suppress pitch/damping to "fix" signs**
- Removes necessary stabilization authority

❌ **Do not use component-level sign inspection as design driver**
- Ignores phase coupling

❌ **Do not treat sign incorrectness as root cause**
- It's a symptom of coupled dynamics

### What to Consider

✅ **Phase-aware modulation**: Instead of hard on/off, modulate pitch/damping continuously based on error trajectory, not instantaneous sign

✅ **Energy-based gates**: Allow high authority when error kinetic energy is high; decay when converging

✅ **Soft blending**: Reduce fighting terms by 50%, not 100%, to preserve partial stabilization

✅ **Gradient-based cap transitions**: Avoid step discontinuities in torque caps

✅ **Recovery exit detection**: Detect when error is converging vs diverging; exit arch_fix early if converging

✅ **Gain tuning**: Adjust position/velocity gains rather than sign overrides

✅ **IK refinement**: Ensure nominal posture is geometrically sound for 0.480m height

---

## Why T6F Baseline Works Despite "Wrong Signs"

### T6F Characteristics

- Sign correctness: 48.9% (far below 80% target)
- Max drift: 0.203m (acceptable)
- Pitch excursion: 8.4° (safe)
- Mode transitions: 0 (stable)
- Arch fix activation: 100% during hard/emergency (working as designed)

### Interpretation

T6F achieves stability through:

1. **Arch fix cap raise**: Provides high authority (up to 8 Nm) during emergency without removing stabilization terms

2. **Continuous pitch control**: Pitch torque remains active, preventing runaway pitch growth

3. **Continuous velocity damping**: Damping dissipates energy, preventing overshoot oscillations

4. **Coupled stabilization**: Pitch and wheel torques work together as a coupled system, not as independent sign-corrected components

**The "wrong signs" are phase-appropriate transient behaviors, not errors.**

### Why Sign Correctness is Low

Possible reasons T6F has 48.9% sign correctness:

1. **Phase lag**: During transients, torque may lead or lag error by phase offset
2. **Overshoot prevention**: During backward recovery from forward drift, controller may apply forward torque to prevent overshoot
3. **Pitch-wheel coupling**: Torques optimized for coupled system, not independent error correction
4. **IK geometry**: Nominal posture at 0.480m may have inherent geometric bias
5. **Transient dynamics**: High-frequency components during arch_fix create sign flips

**None of these indicate a problem if primary metrics pass.**

---

## Candidate Design Principles (Updated)

### Avoid These Patterns (Learned from T6F_sign_corrected)

1. ❌ **Hard suppression**: `tau = 0.0` removes authority
2. ❌ **Binary on/off**: Step discontinuities trigger instability
3. ❌ **Component-level optimization**: Ignores coupling
4. ❌ **Symptom-focused fixes**: Treats sign incorrectness as cause

### Prefer These Patterns

1. ✅ **Soft modulation**: `tau *= blend_factor` where `blend_factor ∈ [0.5, 1.0]`
2. ✅ **Gradual transitions**: Exponential fade over 5-10 steps
3. ✅ **System-level objectives**: Optimize drift and pitch, not component signs
4. ✅ **Root cause investigation**: Understand why signs appear wrong before "fixing"

### Safety Boundaries

**Pitch excursion safety**:
- If `|pitch| > 10°`, restore full pitch control regardless of other conditions
- Rationale: Prevent runaway growth

**Energy dissipation safety**:
- If `wheel_vel > 7.0 rad/s`, restore full damping regardless of other conditions
- Rationale: Prevent wheel velocity runaway

**Cap gradient limit**:
- Limit cap change rate to 0.2-0.3 Nm/step
- Rationale: Avoid discontinuous authority transitions

**Mode stability**:
- If transition/recovery steps > 10, flag as unstable
- Rationale: Mode transitions indicate controller instability

---

## Next Steps

### DO NOT Repeat T6F_sign_corrected Mistake

- Do not design candidates that remove pitch control
- Do not design candidates that remove velocity damping
- Do not optimize sign correctness as primary objective

### DO Explore Safer Alternatives

Two candidate directions:

1. **Soft blend approach**: Reduce fighting terms by 50% instead of 100%, preserving partial stabilization

2. **Phase-aware release approach**: Detect error convergence and gradually release high authority instead of holding it until error < threshold

Both preserve continuous pitch control and velocity damping while potentially reducing overshoot and improving drift convergence.

---

## Conclusion

**Sign incorrectness at high_0p480 is a symptom of coupled pitch-wheel-phase dynamics, not a root cause of drift.**

**Correct optimization target**: Minimize drift and pitch excursion (primary metrics) using coupled system-level control.

**Incorrect optimization target**: Maximize sign correctness by component-level sign correction (removes stabilization).

**T6F baseline works because**: It provides high authority via cap raise while preserving coupled pitch-wheel stabilization, despite "wrong signs."

**T6F_sign_corrected failed because**: It removed pitch and damping authority to "fix" signs, breaking coupled stabilization and amplifying instability.

**Future candidates must**: Preserve stabilization authority while exploring soft modulation, phase-aware transitions, or root cause investigation (gains/IK).

---

**End of Root Cause Reframing**
