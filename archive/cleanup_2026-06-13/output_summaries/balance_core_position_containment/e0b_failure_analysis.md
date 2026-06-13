# E0b Direct Torque Position Containment - Failure Analysis

**Date:** 2026-05-29  
**Status:** FAILED - Catastrophic drift increase  
**Approach:** Multi-zone direct wheel torque bias based on position error

## Executive Summary

E0b direct torque position containment **FAILED catastrophically**, making drift 6.4x worse than baseline:
- **Baseline (no containment):** 2.50 m max drift
- **E0b (direct torque):** 15.98 m max drift
- **Result:** 540% increase in drift

The robot spent 94.5% of the 5000-step run beyond the 0.45 m hard containment limit, demonstrating complete failure of the position containment mechanism.

## Approach Description

E0b attempted to contain position drift by adding a direct wheel torque bias based on position error:

```
position_error_y_m = current_position_y_m - position_reference_y_m

# Multi-zone structure
if |position_error| <= deadband (0.08 m):
    position_correction = 0
elif |position_error| <= soft_limit (0.25 m):
    position_correction = -kp_position * 0.5 * (position_error - sign(error)*deadband)
elif |position_error| <= hard_limit (0.45 m):
    position_correction = -kp_position * 1.0 * (position_error - sign(error)*deadband)
else:
    containment_violation = True

# Add velocity damping
position_correction += -kd_position_velocity * com_vy_m_s

# Balance priority gating
balance_priority_gate = exp(-((pitch/threshold)^2 + (roll/threshold)^2))
position_bias = clip(position_correction * balance_priority_gate, -max_bias, +max_bias)

# Add to balance torque
balance_torque = term_pitch + term_pitch_rate + term_cp + term_com_vy + position_bias
```

### Parameters Used
- `kp_position`: 8.0 Nm/m (increased from 2.0)
- `kd_position_velocity`: 3.0 Nm/(m/s)
- `position_deadband_m`: 0.08 m
- `position_soft_limit_m`: 0.25 m
- `position_hard_limit_m`: 0.45 m
- `max_position_bias`: 15.0 Nm
- `pitch_gate_threshold_rad`: 0.15 rad (8.6°)
- `roll_gate_threshold_rad`: 0.15 rad (8.6°)

## Validation Results

### 5000-Step Nominal Validation (FAILED)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Max drift | 15.98 m | ≤0.50 m | **FAIL** |
| Final drift | 15.98 m | ≤0.20 m | **FAIL** |
| Containment violations | 4723/5000 steps (94.5%) | <5% | **FAIL** |
| Time in deadband | 2.2% | >80% | **FAIL** |
| Time in soft zone | 1.4% | <15% | **FAIL** |
| Time in hard zone | 1.9% | <5% | **FAIL** |
| Pitch range | 0-3.61° | <5° | PASS |
| Roll range | 0-0.17° | <5° | PASS |
| Height stability | Maintained | Stable | PASS |
| Contact state | Valid throughout | Valid | PASS |

### Telemetry Analysis

**Position correction strength:**
- Mean position correction: -3.0 Nm
- Max position correction: -8.5 Nm
- Position correction clipped to max_bias 15.0 Nm: Never reached

**Balance priority gating:**
- Mean balance priority gate: 0.344 (65.6% suppression)
- Gate active (pitch/roll approaching threshold): 65.6% of time
- Mean pitch during drift: 3.7°

**Drift progression:**
- Initial position: 0.00 m
- Position at step 1000: 3.2 m
- Position at step 2500: 7.9 m
- Position at step 5000: 15.98 m
- Drift rate: ~3.2 mm/step (0.32 m/s at 100 Hz)

## Root Cause Analysis

### Primary Failure Mode: Position-Balance Conflict

The direct torque position containment created a **fundamental conflict** between position control and balance control:

1. **Robot drifts forward** → position error increases
2. **Position correction adds backward torque** → robot pitches forward to maintain balance
3. **Forward pitch triggers balance priority gate** → position correction suppressed
4. **Drift continues unchecked** → position error grows
5. **Cycle repeats** → runaway drift

### Why Direct Torque Failed

**1. Insufficient Authority**
- Position correction torque (3-8 Nm) was too weak compared to balance torque (50+ Nm)
- Even at max authority (15 Nm), position correction was only ~20% of balance torque
- Balance controller easily overpowered position correction

**2. Balance Priority Gate Suppression**
- Gate suppressed position correction 65.6% of the time
- When robot needed position correction most (large pitch), gate was most active
- Created positive feedback loop: drift → pitch → gate → more drift

**3. Velocity Damping Insufficient**
- Velocity damping (3.0 Nm/(m/s)) could not prevent drift accumulation
- Forward velocity reached 0.32 m/s sustained
- Damping only slowed drift rate, did not reverse it

**4. Multi-Zone Logic Ineffective**
- Robot spent 94.5% of time beyond hard limit
- Soft/hard zone transitions had no practical effect
- Deadband (0.08 m) was reached only 2.2% of time

### Architectural Limitation

The fundamental issue is that **position containment was added as a secondary correction on top of a balance controller with no inherent position awareness**:

```
Balance Controller (primary):
  - Maintains pitch/roll/height
  - No position feedback
  - Generates large torques (50+ Nm)

Position Containment (secondary):
  - Tries to correct position drift
  - Weak authority (3-15 Nm)
  - Suppressed when balance is active
```

The balance controller successfully maintains pitch/roll/height but has no reason to prevent position drift. Adding position correction as a secondary bias cannot overcome the primary controller's lack of position awareness.

## Comparison to Baseline

| Metric | Baseline | E0b | Change |
|--------|----------|-----|--------|
| Max drift | 2.50 m | 15.98 m | +540% |
| Drift rate | 0.50 mm/step | 3.20 mm/step | +540% |
| Pitch stability | Stable | Stable | No change |
| Roll stability | Stable | Stable | No change |
| Height stability | Stable | Stable | No change |

E0b made drift **6.4x worse** while maintaining pitch/roll/height stability, confirming that the position containment mechanism actively destabilized position control without improving balance.

## Lessons Learned

1. **Direct torque position containment is incompatible with the current balance controller architecture**
   - Balance controller has no position awareness
   - Adding position correction as secondary bias creates conflict
   - Balance controller overpowers position correction

2. **Balance priority gating creates positive feedback loop**
   - Suppresses correction when it's needed most
   - Drift → pitch → gate → more drift
   - Cannot break the cycle

3. **Multi-zone structure provides no benefit**
   - Robot immediately exits deadband and never returns
   - Soft/hard zones have no practical effect
   - Containment violation flag is always true

4. **Velocity damping alone cannot prevent drift**
   - Can slow drift rate but not reverse it
   - Insufficient authority to counteract balance controller
   - Requires position feedback to be effective

## Recommendations

### Do NOT Pursue
- ❌ Increasing position correction gains (already tried, made it worse)
- ❌ Adjusting multi-zone thresholds (zones are ineffective)
- ❌ Tuning balance priority gate (creates positive feedback)
- ❌ Adding integral term (will wind up and destabilize)

### Fundamental Redesign Required
Position containment cannot be achieved with the current controller architecture. Any future attempt must:

1. **Integrate position awareness into the balance controller core**
   - Balance controller must track position as a primary objective
   - Cannot be added as secondary correction
   - Requires controller redesign, not parameter tuning

2. **Use reference shaping instead of direct torque**
   - Bias capture point reference based on position error
   - Let balance controller handle wheel torque naturally
   - Avoid direct position-to-torque coupling
   - **Note:** E0c attempted this and also failed (see E0c failure analysis)

3. **Consider higher-level position control**
   - Outer-loop position controller that commands desired velocity
   - Inner-loop balance controller tracks velocity reference
   - Requires multi-rate control architecture

4. **Accept position drift as inherent limitation**
   - Current controller successfully maintains pitch/roll/height
   - Position drift may be acceptable for standing balance
   - Focus on recovery behaviors instead of containment

## Conclusion

E0b direct torque position containment **FAILED catastrophically**, making drift 6.4x worse than baseline. The approach is fundamentally incompatible with the current balance controller architecture and should not be pursued further.

Position containment requires either:
1. Fundamental controller redesign to integrate position awareness
2. Acceptance that position drift is an inherent limitation of the current architecture

**Status:** E0b is DISABLED by default and should remain disabled.
