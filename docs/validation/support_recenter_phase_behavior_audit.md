# Support Recenter Phase Behavior Audit

**Date**: 2026-06-08
**Status**: COMPLETE

## Phase Reversal Detection

### Pitch Reversal Rate

| Variant | Total Steps | Pitch Reversals | Reversals with Large Support Error |
|---------|-------------|-----------------|-----------------------------------|
| D2 | 5000 | 1 | 0 |
| E2 | 500 | 1 | 0 |
| E2b | 500 | 1 | 0 |

**Finding**: The robot falls forward **almost continuously** without stabilization phases. Only 1 pitch reversal detected in 5000 steps for D2.

## Wheel Velocity Behavior

| Variant | Mean (rad/s) | Std (rad/s) | Min | Max |
|---------|-------------|------------|-----|-----|
| D2 | -0.028 | 1.74 | -4.39 | 2.77 |
| E2 | N/A | N/A | N/A | N/A |

- Mean wheel velocity is nearly zero (-0.028 rad/s)
- High std (1.74 rad/s) shows wheels oscillate actively
- Equal positive/negative time suggests balance correction

## tau_position Analysis

| Variant | Mean (Nm) | Min (Nm) | At Cap % | Cap |
|---------|----------|----------|----------|-----|
| D2 | -2.23 | -4.00 | 37.8% | 3.0 Nm |
| E2 | -2.29 | -5.00 | 42.2% | 5.0 Nm |
| E2b | -2.29 | -5.00 | 42.2% | 5.0 Nm |

**Observations**:
- tau_position is **always negative** (correcting forward fall)
- D2 hits cap at 3 Nm 37.8% of time
- E2 increased cap to 5 Nm but hits cap 42.2% (more saturation)
- Increased cap did NOT reduce crossings

## Phase Behavior Classification

**RECENTERING_WORKS**: Pitch reverses and support error reduces
**RECENTERING_TOO_WEAK**: Pitch reverses but support error does NOT reduce
**RECENTERING_PREMATURELY_REVERSED**: Wheel reverses too aggressively

**Finding**: Pitch reversal rate is too low to analyze recentering behavior statistically. The robot falls forward continuously without stabilization phases.

## Conclusion

**PHASE_BEHAVIOR_INCONCLUSIVE**: The pitch reversal rate is too low (1 in 5000 steps for D2) to analyze recentering behavior. The robot falls forward continuously without stabilization phases. A phase-aware recentering strategy would need to work during active fall recovery, not just after pitch reversal.