# APCR Recovery Gate Redesign

## Classification: APCR_RECOVERY_GATE_REDESIGN_COMPLETE

## Executive Summary

The old APC safety gate logic blocked activation when pitch was moderately large. The new APCR recovery gate logic allows activation during moderate pitch error, when the robot needs active recovery.

## Phase 3: Redesign APC Gates into APCR Recovery Gates

### Problem with Old APC Gate

Old gate logic:
```python
apc_pitch_safe = apc_pitch_abs < apc_pitch_safe_threshold_rad  # 0.05 rad = 2.86°
apc_pitch_danger = apc_pitch_abs > apc_pitch_danger_threshold_rad  # 0.10 rad = 5.73°
apc_gate_safe = apc_contact_safe and apc_height_safe and apc_roll_safe and apc_pitch_safe and not apc_pitch_danger
```

**Issue:** This blocks APCR when pitch is moderately large (> 5.73°), but APCR is designed to activate DURING moderate pitch error.

### New APCR Recovery Gate Design

Split into two separate gate types:

#### A. Hard Safety Gate (blocks only if truly unsafe)

Blocks APCR only if robot is in a truly unsafe state:
- `contact_invalid` - loss of wheel contact
- `height_unsafe` - height outside safe operating range
- `roll_unsafe` - roll exceeds safe threshold
- `pitch_hard_emergency` - pitch exceeds hard emergency threshold

**Hard thresholds:**
- `pitch_hard_stop_rad = 0.30 rad` (17.2°) - absolute emergency stop
- `roll_hard_stop_rad = 0.15 rad` (8.6°) - lateral stability
- `min_com_z_m = 0.27 m` - minimum safe height
- `max_com_z_m = 0.50 m` - maximum operating height

#### B. Recovery Activation Gate (allows activation when needed)

Allows APCR to activate when pitch and support drift are in the same direction:

**Positive recovery entry:**
- `signed_error > outer_enter_m` AND
- `pitch_x > pitch_enter_rad` OR `tau_pitch persistently positive` AND
- hard safety gate is clear

**Negative recovery entry:**
- `signed_error < -outer_enter_m` AND
- `pitch_x < -pitch_enter_rad` OR `tau_pitch persistently negative` AND
- hard safety gate is clear

**Key insight:** APCR is supposed to activate during moderate pitch error, NOT only when pitch is already safe.

### APCR Recovery Gate Thresholds

| Parameter | APCR1 Value | Rationale |
|-----------|-------------|-----------|
| `outer_enter_m` | 0.10 m | Same as old APC - signed error must exceed this |
| `inner_exit_m` | 0.05 m | Exit when signed error enters this inner band |
| `opposite_overshoot_m` | 0.01 m | Allow slight overshoot to opposite side |
| `pitch_enter_rad` | 0.03 rad (1.7°) | Pitch must exceed this to enter recovery |
| `pitch_hard_stop_rad` | 0.30 rad (17.2°) | Hard emergency stop - blocks APCR |
| `roll_hard_stop_rad` | 0.15 rad (8.6°) | Lateral stability threshold |
| `min_com_z_m` | 0.27 m | Minimum safe height |
| `max_com_z_m` | 0.50 m | Maximum operating height |
| `max_cross_tau` | 1.0 Nm | Maximum APCR torque (APCR1) |
| `smooth_alpha` | 0.10 | Smoothing factor for torque transitions |
| `max_rate_per_step` | 0.4 Nm/step | Rate limiting for torque changes |

### Key Differences from Old APC Gate

| Aspect | Old APC Gate | New APCR Gate |
|--------|--------------|---------------|
| Block condition | pitch > 5.73° | pitch > 17.2° |
| Entry condition | pitch < 2.86° OR recovering | pitch > 1.7° AND drift |
| Philosophy | Safety-first | Recovery-first |
| Hard stop | 10° | 17.2° |
| Safe entry | 2.86° | 1.7° |

### State Machine (from Phase 5 spec)

States:
- `NEUTRAL` - APCR not active
- `CROSS_FROM_POSITIVE` - Recovery from positive pitch + drift
- `CROSS_FROM_NEGATIVE` - Recovery from negative pitch + drift
- `HOLD_RECENTER_TO_ZERO` - Smoothly decay APCR torque
- `SAFETY_DECAY` - Decay APCR to zero on safety trigger

### Exit Conditions

**CROSS_FROM_POSITIVE:**
- Exit when: `signed_error <= inner_exit_m` OR `signed_error < 0` (crossed slightly)

**CROSS_FROM_NEGATIVE:**
- Exit when: `signed_error >= -inner_exit_m` OR `signed_error > 0` (crossed slightly)

**SAFETY_DECAY:**
- Exit when: `apc_tau <= 0` or hard safety clears

## Implementation Notes

1. Add `apcr_` prefix to all new parameters to distinguish from old `apc_` parameters
2. Keep old `apc_` parameters for backward compatibility
3. Add `active_pitch_crossing_recovery_gate_mode` flag to switch between old and new gate logic
4. Add `apc_pitch_hard_stop_rad` with default 0.30 rad

## Files Generated

- `docs/validation/apcr_recovery_gate_redesign.md` - This file
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr_recovery_gate_redesign.json` - Design data
