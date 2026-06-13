# E0c Reference Shaping Position Containment - Failure Analysis

**Date:** 2026-05-29  
**Status:** FAILED - Catastrophic drift increase (worse than E0b)  
**Approach:** Capture-point reference shaping based on position error

## Executive Summary

E0c reference shaping position containment **FAILED catastrophically**, making drift 25.5x worse than baseline and 4x worse than E0b:
- **Baseline (no containment):** 2.50 m max drift
- **E0b (direct torque):** 15.98 m max drift (6.4x worse)
- **E0c (reference shaping):** 63.72 m max drift (25.5x worse)

E0c was designed to avoid the direct torque-to-position coupling that caused E0b to fail, but the reference shaping approach proved even less effective.

## Approach Description

E0c attempted to contain position drift by biasing the capture point error based on position drift, letting the existing balance controller handle wheel torque naturally:

```
position_error_y_m = current_position_y_m - position_reference_y_m

# Apply deadband
if |position_error| <= deadband (0.10 m):
    desired_velocity_y = 0
else:
    # Proportional control: move back toward reference
    desired_velocity_y = -k_position_to_velocity * position_error

# Clip desired velocity
desired_velocity_y = clip(desired_velocity_y, -max_desired_velocity, +max_desired_velocity)

# Compute velocity error
velocity_error_y = current_velocity_y - desired_velocity_y

# Compute CP bias from velocity error
cp_bias_from_position = -k_velocity_to_cp_bias * velocity_error_y

# Apply balance priority gating
balance_priority_gate = exp(-((pitch/threshold)^2 + (roll/threshold)^2))
cp_bias_gated = cp_bias_from_position * balance_priority_gate

# Clip CP bias
cp_bias_final = clip(cp_bias_gated, -max_cp_bias, +max_cp_bias)

# Bias the capture point error
cp_error_y_m = (cp_y - com_y) + cp_bias_final
```

### Parameters Used
- `k_position_to_velocity`: 0.10 m/s per m of position error
- `max_desired_velocity_m_s`: 0.10 m/s
- `k_velocity_to_cp_bias`: 0.50 m of CP bias per m/s of velocity error
- `max_cp_bias_m`: 0.05 m
- `position_deadband_m`: 0.10 m
- `pitch_gate_threshold_rad`: 0.15 rad (8.6°)
- `roll_gate_threshold_rad`: 0.15 rad (8.6°)

## Validation Results

### 5000-Step Nominal Validation (FAILED)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Max drift | 63.72 m | ≤0.50 m | **FAIL** |
| Final drift | 63.72 m | ≤0.20 m | **FAIL** |
| Time in deadband | 2.0% | >80% | **FAIL** |
| Max position error | 63.71 m | <0.50 m | **FAIL** |
| Max CP bias | 0.050 m | N/A | Saturated |
| Mean CP bias | 0.050 m | N/A | Saturated |
| Balance gate active | 79.7% | <20% | **FAIL** |
| Pitch range | -1.21° to 2.07° | <5° | PASS |
| Roll range | 0.00° to 0.25° | <5° | PASS |
| Height stability | Maintained | Stable | PASS |

### Telemetry Analysis

**CP bias saturation:**
- Max CP bias: 0.050 m (saturated at limit)
- Mean CP bias: 0.050 m (saturated throughout run)
- CP bias was clipped 98% of the time

**Balance priority gating:**
- Mean balance priority gate: 0.203 (79.7% suppression)
- Gate active (pitch/roll approaching threshold): 79.7% of time
- Gate suppressed CP bias even more aggressively than E0b

**Drift progression:**
- Initial position: -0.014 m
- Position at step 1000: 12.7 m
- Position at step 2500: 31.8 m
- Position at step 5000: 63.7 m
- Drift rate: ~12.7 mm/step (1.27 m/s at 100 Hz)

**Desired velocity:**
- Max desired velocity: 0.10 m/s (saturated at limit)
- Mean desired velocity: 0.10 m/s (saturated throughout run)
- Actual velocity: 1.27 m/s (12.7x larger than desired)

## Root Cause Analysis

### Primary Failure Mode: Insufficient Authority

E0c failed because the **CP bias was orders of magnitude too weak** to counteract multi-meter drift:

1. **Position error grows** → desired velocity = 0.10 m/s backward
2. **Actual velocity is 1.27 m/s forward** → velocity error = 1.37 m/s
3. **CP bias = 0.05 m** (clipped at max) → negligible effect on balance controller
4. **Balance controller ignores tiny CP bias** → drift continues unchecked
5. **Cycle repeats** → runaway drift at 4x the rate of E0b

### Why Reference Shaping Failed

**1. CP Bias Too Weak**
- Max CP bias: 0.05 m
- Position error: 63.7 m
- CP bias is 0.08% of position error
- Balance controller cannot detect such a small bias

**2. Velocity Error Dominates**
- Desired velocity: 0.10 m/s
- Actual velocity: 1.27 m/s
- Velocity error: 1.37 m/s (13.7x larger than desired)
- Position control has no authority over velocity

**3. Balance Priority Gate Suppression**
- Gate suppressed CP bias 79.7% of the time
- Even more aggressive than E0b (65.6%)
- When correction was needed most, gate was most active

**4. Cascade of Saturations**
- Position error → desired velocity saturated
- Velocity error → CP bias saturated
- CP bias → balance gate suppressed
- Final CP bias → negligible effect

### Comparison to E0b

E0c was designed to avoid E0b's direct torque-to-position coupling, but it failed even worse:

| Aspect | E0b Direct Torque | E0c Reference Shaping |
|--------|-------------------|----------------------|
| Max drift | 15.98 m | 63.72 m |
| Drift rate | 3.2 mm/step | 12.7 mm/step |
| Correction strength | 3-15 Nm | 0.05 m CP bias |
| Balance gate suppression | 65.6% | 79.7% |
| Time in deadband | 2.2% | 2.0% |
| Result | 6.4x worse | 25.5x worse |

E0c's reference shaping was **4x less effective** than E0b's direct torque because:
- CP bias of 0.05 m has negligible effect on balance controller
- Direct torque of 3-15 Nm at least had measurable (though insufficient) effect
- Reference shaping added an extra layer of indirection that further weakened correction

### Architectural Limitation

Both E0b and E0c fail for the same fundamental reason: **position containment is added as a secondary correction on top of a balance controller with no inherent position awareness**.

```
Balance Controller (primary):
  - Maintains pitch/roll/height
  - No position feedback
  - Generates large torques (50+ Nm)
  - Tracks capture point for balance, not position

Position Containment (secondary):
  E0b: Add weak torque bias (3-15 Nm)
  E0c: Add tiny CP bias (0.05 m)
  
  Both: Suppressed by balance priority gate
  Both: Insufficient authority to counteract drift
  Both: Cannot overcome primary controller's lack of position awareness
```

The balance controller successfully maintains pitch/roll/height but has no reason to prevent position drift. Neither direct torque (E0b) nor reference shaping (E0c) can overcome this fundamental limitation.

## Comparison to Baseline

| Metric | Baseline | E0b | E0c | E0c vs Baseline |
|--------|----------|-----|-----|-----------------|
| Max drift | 2.50 m | 15.98 m | 63.72 m | +2449% |
| Drift rate | 0.50 mm/step | 3.20 mm/step | 12.74 mm/step | +2448% |
| Pitch stability | Stable | Stable | Stable | No change |
| Roll stability | Stable | Stable | Stable | No change |
| Height stability | Stable | Stable | Stable | No change |

E0c made drift **25.5x worse** than baseline and **4x worse than E0b**, confirming that reference shaping is even less effective than direct torque for position containment.

## Lessons Learned

1. **Reference shaping is weaker than direct torque**
   - CP bias of 0.05 m has negligible effect on balance controller
   - Direct torque of 3-15 Nm at least had measurable effect
   - Adding indirection further weakens correction authority

2. **Balance priority gating is even more aggressive with reference shaping**
   - Suppressed correction 79.7% of time (vs 65.6% for E0b)
   - Gate treats CP bias as less critical than direct torque
   - Creates stronger positive feedback loop

3. **Cascade of saturations prevents effective correction**
   - Position error → desired velocity saturated
   - Velocity error → CP bias saturated
   - CP bias → balance gate suppressed
   - Each layer reduces correction authority

4. **Conservative parameters made failure worse**
   - Larger deadband (0.10 m vs 0.08 m) delayed correction
   - Lower velocity limit (0.10 m/s) prevented aggressive return
   - Lower CP bias limit (0.05 m) ensured negligible effect
   - "Conservative" parameters guaranteed failure

## Why E0c Failed Worse Than E0b

E0c was designed to avoid E0b's direct torque-to-position coupling, but the reference shaping approach introduced additional weaknesses:

1. **Extra layer of indirection**
   - E0b: position error → torque (one step)
   - E0c: position error → desired velocity → velocity error → CP bias → torque (four steps)
   - Each step adds saturation and suppression

2. **Weaker correction authority**
   - E0b: 3-15 Nm direct torque
   - E0c: 0.05 m CP bias → ~1-2 Nm equivalent torque
   - Reference shaping is 5-10x weaker

3. **More aggressive gating**
   - E0b: 65.6% suppression
   - E0c: 79.7% suppression
   - Balance controller treats CP bias as less critical

4. **Velocity control has no authority**
   - Desired velocity: 0.10 m/s
   - Actual velocity: 1.27 m/s
   - Position control cannot command velocity

## Recommendations

### Do NOT Pursue
- ❌ Increasing CP bias limit (already saturated, would need 100x increase)
- ❌ Increasing velocity limits (position control has no velocity authority)
- ❌ Adjusting balance priority gate (creates positive feedback)
- ❌ Adding integral term (will wind up and destabilize)
- ❌ Any variant of reference shaping (fundamentally too weak)

### Fundamental Redesign Required
Position containment cannot be achieved with the current controller architecture. Both direct torque (E0b) and reference shaping (E0c) have failed catastrophically.

Any future attempt must:

1. **Integrate position awareness into the balance controller core**
   - Balance controller must track position as a primary objective
   - Cannot be added as secondary correction
   - Requires controller redesign, not parameter tuning

2. **Use model-based position control**
   - Predict future position based on current velocity
   - Command wheel torque to achieve desired position trajectory
   - Integrate with balance objectives, not fight them

3. **Consider multi-rate control architecture**
   - Slow outer loop: position controller commands desired velocity
   - Fast inner loop: balance controller tracks velocity reference
   - Requires fundamental architecture change

4. **Accept position drift as inherent limitation**
   - Current controller successfully maintains pitch/roll/height
   - Position drift may be acceptable for standing balance
   - Focus on recovery behaviors instead of containment

## Conclusion

E0c reference shaping position containment **FAILED catastrophically**, making drift 25.5x worse than baseline and 4x worse than E0b. The reference shaping approach is fundamentally weaker than direct torque and should not be pursued further.

Both E0b and E0c fail for the same reason: **position containment cannot be added as a secondary correction on top of a balance controller with no inherent position awareness**.

Position containment requires either:
1. Fundamental controller redesign to integrate position awareness
2. Acceptance that position drift is an inherent limitation of the current architecture

**Status:** E0c is DISABLED and should remain disabled.

## Comparison Summary

| Approach | Max Drift | vs Baseline | Status |
|----------|-----------|-------------|--------|
| Baseline (no containment) | 2.50 m | 1.0x | Reference |
| E0b (direct torque) | 15.98 m | 6.4x worse | FAILED |
| E0c (reference shaping) | 63.72 m | 25.5x worse | FAILED |

Both approaches made drift significantly worse. Position containment is not achievable with the current controller architecture.
