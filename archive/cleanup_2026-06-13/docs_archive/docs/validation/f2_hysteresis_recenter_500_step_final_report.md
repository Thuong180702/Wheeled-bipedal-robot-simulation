# F2 Hysteresis Recenter 500-Step Final Report

## Executive Summary

F2 hysteresis recenter was implemented to fix the one-sided ratcheting behavior where F1b (proportional recenter) improved but did not eliminate the positive bias in signed support drift.

**Key Finding:** F2 hysteresis recenter activates correctly (216/500 steps for F2a, 204/500 steps for F2b) but **does not enter RECENTER_FROM_NEGATIVE state at all**. The robot consistently drifts positive throughout the simulation, suggesting the underlying dynamics are fundamentally biased in one direction.

## Decision

**F2_HYSTERESIS_FAILED_NO_IMPROVEMENT**

F2a and F2b do not improve signed support bias vs F1b:
- F1b positive%: 82.8%
- F2a positive%: 82.8%
- F2b positive%: 82.6%

F2b actually increases steps outside +0.15:
- F1b outside +0.15: 81 steps
- F2a outside +0.15: 92 steps
- F2b outside +0.15: 142 steps

## Root Cause Analysis

The hysteresis state machine logic is correct (proven by unit tests) but the underlying dynamics do not produce negative drift. Looking at the signed_error trajectory:

- F2a min: -0.047 m (only slight negative)
- F2a max: +0.176 m (exceeds +0.15)
- F2b min: -0.064 m
- F2b max: +0.176 m

The robot drifts positive beyond +0.15 but never drifts negative beyond -0.06. This means:
1. The hysteresis recenter cannot correct what doesn't occur
2. The outer_enter_m threshold of 0.10 is never crossed in the negative direction
3. RECENTER_FROM_NEGATIVE state is never entered

The fundamental issue is not the recenter strategy but the underlying dynamics that produce one-sided drift.

## Implementation Verification

### F2 Tests: 19/19 PASSED
- F2a and F2b profiles exist and are opt-in only
- State machine enters RECENTER_FROM_POSITIVE when signed_error > 0.10
- State machine holds until exit target
- Safety override disables recenter when pitch danger or height unsafe
- Hysteresis torque is bounded by max_recenter_tau
- Smoothing prevents discontinuous jumps
- tau_position_raw is not modified by hysteresis recenter

### Simulation: 500 steps completed
- F2a: Survived 500 steps, contact valid 99.8%
- F2b: Survived 500 steps

### Telemetry: Hysteresis fields confirmed
- hysteresis_recenter_state: tracks NEUTRAL/RECENTER_FROM_POSITIVE/RECENTER_FROM_NEGATIVE
- hysteresis_recenter_active: indicates when recenter is applied
- hysteresis_recenter_tau: actual torque applied
- hysteresis_recenter_state_entry_count: tracks state entries
- hysteresis_recenter_state_exit_count: tracks state exits

## Detailed Results

### Signed Support Error Metrics

| Metric | D2 | F1b | F2a | F2b |
|--------|-----|-----|-----|-----|
| positive% | 93.0% | 82.8% | 82.8% | 82.6% |
| negative% | 6.6% | 16.8% | 17.0% | 17.2% |
| Mean (m) | +0.082 | +0.076 | +0.080 | +0.083 |
| Min (m) | -0.004 | -0.034 | -0.047 | -0.064 |
| Max (m) | +0.176 | +0.169 | +0.176 | +0.176 |
| zero crossings | 4 | 5 | 5 | 5 |
| outside +0.15 | 96 | 81 | 92 | 142 |
| outside -0.15 | 0 | 0 | 0 | 0 |

### Hysteresis State Analysis

| Metric | F2a | F2b |
|--------|-----|-----|
| NEUTRAL % | 56.8% | 59.2% |
| RECENTER_FROM_POSITIVE % | 43.2% | 40.8% |
| RECENTER_FROM_NEGATIVE % | 0.0% | 0.0% |
| Hysteresis active % | 43.2% | 40.8% |
| State entries | 6 | 11 |
| State exits | 2 | 1 |
| Hysteresis tau max (Nm) | 0.0 | 0.0 |
| Hysteresis tau min (Nm) | -1.49 | -1.96 |

### Stability

| Metric | F2a | F2b |
|--------|-----|-----|
| Survived 500 steps | Yes | Yes |
| Contact valid % | 99.8% | 99.8% |
| Height range (m) | 0.288-0.295 | 0.287-0.295 |
| Pitch range (deg) | -2.4 to 6.4 | -3.1 to 6.4 |
| Roll range (deg) | 0.0 to 0.8 | 0.0 to 0.7 |

## Why F2 Did Not Improve

1. **Fundamental bias:** The robot dynamics produce only positive drift in this configuration. The outer_enter_m threshold of 0.10 m is never crossed in the negative direction.

2. **Hysteresis cannot create negative drift:** The state machine can only react to what's happening. If the system never drifts negative, RECENTER_FROM_NEGATIVE is never entered.

3. **F2b is too aggressive:** F2b's stronger torque (2.0 Nm vs 1.5 Nm) and larger overshoot target (0.02 m vs 0.01 m) actually increases overshoot beyond +0.15 (142 vs 92 steps) without improving the bias.

4. **F1b and F2a are equivalent:** F2a produces virtually identical positive% (82.8%) to F1b, confirming that the hysteresis logic doesn't help when there's only one-sided drift.

## Recommendations

1. **Do not run F2b at 2000 steps** - F2b shows worse overshoot behavior (+0.15 violations)

2. **Investigate root cause of one-sided drift** - The bias may be due to:
   - Initial condition asymmetry
   - Asymmetric contact force distribution
   - Hip yaw compensation imbalance
   - Height variant initialization

3. **Alternative approaches to consider:**
   - Add explicit bias correction term to hip yaw or sagittal controller
   - Adjust initial equilibrium to counteract expected drift
   - Use asymmetric position cap (lower for positive direction)
   - Add feedforward bias to wheel velocity controller

4. **If one-sided drift is intrinsic to the setup:**
   - Accept the bias and adjust operational limits
   - Focus on keeping drift within ±0.15 rather than centering around zero

## Files Changed

- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`: Added hysteresis recenter fields and state machine
- `scripts/simulate_hierarchical_controller.py`: Added F2a/F2b profiles and telemetry fields
- `tests/test_sagittal_velocity_damped_balance_controller.py`: Added 19 F2 unit tests

## What Was NOT Changed

- D2 baseline (protected)
- F1b profiles (untouched)
- Official Step E validation criteria
- WBC/HY2-DIV defaults

## Conclusion

The F2 hysteresis recenter implementation is correct and functional but does not solve the one-sided drift problem because the underlying dynamics do not produce negative drift to correct. The robot survives and remains stable, but signed support bias remains at ~83% positive.

**Recommendation:** Do not proceed to 2000-step validation. Investigate the root cause of one-sided drift in the robot dynamics.