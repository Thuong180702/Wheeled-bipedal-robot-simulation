# F2 Hysteresis Recenter Design

## Problem Statement

F1b (proportional recenter) improves signed support drift but does not eliminate the one-sided ratcheting behavior:

- **D2 positive%**: 93.0%
- **F1b positive%**: 82.8%
- **F1b recenter active**: 65.8%
- **F1b still remains strongly positive-biased**

F1b uses:
```
recenter_tau = -k_recenter * signed_error
```
This is proportional and weakens near zero. It has no memory.

## Root Cause

The F1b proportional recenter reacts instantaneously to changes in signed_error. When the robot is drifting positive:
1. F1b applies negative recenter torque
2. As balance terms change, signed_error may decrease slightly
3. F1b weakens proportionally, possibly reversing before error crosses zero
4. This causes "flutter" near the threshold instead of sustained recentering

The result is that F1b reduces bias but does not make support drift oscillate around zero.

## F2 Design: Stateful Hysteresis Recenter

F2 adds a state machine with hysteresis to hold the recenter direction until the error returns to an inner band near zero or crosses zero slightly.

### State Machine

```
States:
- NEUTRAL: No recenter applied, waiting for outer threshold
- RECENTER_FROM_POSITIVE: Robot drifted positive, keep pushing negative until error returns to target
- RECENTER_FROM_NEGATIVE: Robot drifted negative, keep pushing positive until error returns to target
```

### State Transitions

```
Entry:
- NEUTRAL + signed_error > outer_enter_m  → RECENTER_FROM_POSITIVE
- NEUTRAL + signed_error < -outer_enter_m → RECENTER_FROM_NEGATIVE

Exit (hold until target):
- RECENTER_FROM_POSITIVE + signed_error <= exit_target_m → NEUTRAL
- RECENTER_FROM_NEGATIVE + signed_error >= exit_target_m → NEUTRAL

Safety override (any state):
- contact invalid → NEUTRAL (recenter decays to 0)
- pitch_danger → NEUTRAL (recenter decays to 0)
- height_unsafe → NEUTRAL (recenter decays to 0)
```

### Key Parameters

| Parameter | F2a Moderate | F2b Strong | Description |
|-----------|-------------|-----------|-------------|
| outer_enter_m | 0.10 | 0.10 | Outer threshold to enter recenter state |
| exit_target_m | 0.00 | 0.00 | Exit target - 0.00 means cross through zero |
| opposite_overshoot_m | 0.01 | 0.02 | Slight overshoot into opposite direction |
| k_recenter | 10.0 | 12.0 | Nm/m - gain for recenter term |
| max_recenter_tau | 1.5 Nm | 2.0 Nm | Max recenter torque |
| smooth_alpha | 0.10 | 0.10 | Smoothing factor |
| max_rate_per_step | 0.5 Nm/step | 0.5 Nm/step | Rate limit |
| deadband_m | 0.01 | 0.01 | Ignore small signed errors in NEUTRAL |

### Exit Target Logic

For RECENTER_FROM_POSITIVE:
- Exit when signed_error <= exit_target_m
- exit_target_m = 0.00 or slight opposite overshoot = -0.01 to -0.02
- This ensures the error crosses or nearly crosses zero before reversing

For RECENTER_FROM_NEGATIVE:
- Exit when signed_error >= exit_target_m
- exit_target_m = 0.00 or slight opposite overshoot = +0.01 to +0.02

### Safety Overrides

The state machine must exit to NEUTRAL (not hard switch) when:
- contact_invalid: contact is lost
- pitch_danger: pitch exceeds danger threshold
- height_unsafe: com_z outside safe range

The recenter torque must decay smoothly, not jump to zero.

### Implementation Notes

1. **F2 does NOT modify tau_position_raw** - Recentering is decoupled from position torque
2. **F2 adds a separate recenter term** - Like F1b, recenter is added to tau_common
3. **State is persisted** - The state machine maintains memory across timesteps
4. **Telemetry is comprehensive** - State transitions, torque values, gate reasons are all logged

## Difference from F1b

| Aspect | F1b | F2 |
|--------|-----|-----|
| Memory | None | State machine |
| Direction changes | Instantaneous | Held until exit target |
| Exit condition | Deadband only | Hysteresis band |
| Near-zero behavior | Weakens proportionally | Holds until crossing |

## F2a vs F2b

- **F2a**: Moderate recenter (1.5 Nm max), exit at zero
- **F2b**: Stronger recenter (2.0 Nm max), exit at zero or slight overshoot

Run F2a first. If it improves but not enough, run F2b.

## Telemetry Fields

```
hysteresis_recenter_enabled
hysteresis_recenter_state (NEUTRAL/RECENTER_FROM_POSITIVE/RECENTER_FROM_NEGATIVE)
hysteresis_recenter_state_id (0/1/2)
hysteresis_recenter_outer_enter_m
hysteresis_recenter_exit_target_m
hysteresis_recenter_signed_error_m
hysteresis_recenter_target_error_m
hysteresis_recenter_raw_tau
hysteresis_recenter_tau
hysteresis_recenter_tau_clipped
hysteresis_recenter_active
hysteresis_recenter_state_entry_count
hysteresis_recenter_state_exit_count
hysteresis_recenter_safety_override
hysteresis_recenter_gate_reason
```

## Expected Behavior

With F2, signed support should oscillate around zero:
- Positive excursions should be followed by negative excursions
- Zero crossings should increase vs F1b
- Longest same-sign interval should decrease vs F1b
- Support should remain inside ±0.15 m more consistently
