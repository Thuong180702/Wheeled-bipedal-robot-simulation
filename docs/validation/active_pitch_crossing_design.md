# Active Pitch Crossing Controller Design

## Overview

The Active Pitch Crossing (APC) controller is a stateful controller that actively drives the wheels to create controlled pitch-rate reversal when the robot has positive pitch AND positive signed support drift. Unlike F1/F2/G1 which are reactive or estimate bias, APC explicitly commands wheel torque in the direction needed to reverse pitch_rate.

## Design Principles

1. **Not a proportional recenter**: APC doesn't just add small correction - it actively drives pitch_rate sign reversal
2. **Not a bias estimator**: APC doesn't estimate persistent bias - it responds to current state
3. **Stateful**: APC holds its direction until support enters inner band, not until pitch changes slightly
4. **Safety-first**: Contact, height, roll, and pitch gates prevent destabilizing actions

## States

| State | Description |
|-------|-------------|
| NEUTRAL | No active crossing, normal balance control |
| CROSS_FROM_POSITIVE | Actively driving to reverse from positive pitch + positive drift |
| CROSS_FROM_NEGATIVE | Actively driving to reverse from negative pitch + negative drift |
| HOLD_RECENTER_TO_ZERO | Transitioning back to neutral after crossing |
| SAFETY_DECAY | Emergency decay of crossing torque |

## State Machine Diagram

```
                    signed_error > outer_enter_m
                    pitch_x > pitch_enter_rad
                    gates safe
                         │
                         ▼
    ┌──────────────────────────────────────┐
    │            NEUTRAL                    │
    └──────────────────────────────────────┘
                         │
                         │ YES: signed_error > outer AND pitch_x > threshold
                         │     AND safety gates OK
                         ▼
    ┌──────────────────────────────────────┐
    │     CROSS_FROM_POSITIVE              │
    │  Apply negative wheel torque         │
    │  (to create negative pitch_rate)     │
    └──────────────────────────────────────┘
                         │
                         │ signed_error <= inner_exit_m
                         │ OR signed_error < 0 (crossed slightly)
                         │ OR safety override
                         ▼
    ┌──────────────────────────────────────┐
    │     HOLD_RECENTER_TO_ZERO            │
    │  Reduce crossing torque smoothly    │
    │  Allow normal balance to continue     │
    └──────────────────────────────────────┘
                         │
                         │ tau_crossing ~= 0
                         ▼
                    NEUTRAL
```

## Entry Conditions

### CROSS_FROM_POSITIVE
- signed_error > outer_enter_m (default: 0.10 m)
- pitch_x > pitch_enter_rad (default: 0.03 rad)
- OR tau_pitch persistently positive (threshold: 0.5 Nm for 5+ steps)
- Contact valid
- Height safe (0.28m <= com_z <= 0.50m)
- Roll safe (|roll_y| < 0.15 rad)
- Pitch not in danger zone (|pitch_x| < 0.10 rad)

### CROSS_FROM_NEGATIVE
- signed_error < -outer_enter_m (default: -0.10 m)
- pitch_x < -pitch_enter_rad (default: -0.03 rad)
- OR tau_pitch persistently negative
- Same safety gates

## Exit Conditions

### CROSS_FROM_POSITIVE
- signed_error <= inner_exit_m (default: 0.05 m)
- OR signed_error < 0 (crossed slightly into negative)
- OR safety override triggered

### CROSS_FROM_NEGATIVE
- signed_error >= -inner_exit_m (default: -0.05 m)
- OR signed_error > 0 (crossed slightly into positive)
- OR safety override triggered

## Torque Computation

### CROSS_FROM_POSITIVE
```python
# Apply negative torque to make pitch_rate negative
# This creates a controlled forward lean that pushes support back
apc_tau = -max_cross_tau * pitch_aware_scale
```

### CROSS_FROM_NEGATIVE
```python
# Apply positive torque to make pitch_rate positive
# This creates a controlled backward lean that pushes support back
apc_tau = +max_cross_tau * pitch_aware_scale
```

### pitch_aware_scale
```python
# If pitch is already large, reduce crossing torque to avoid overcorrection
if abs(pitch_x) > pitch_safe_limit:
    scale = pitch_safe_limit / abs(pitch_x)  # Reduce scale
else:
    scale = 1.0  # Full torque
```

## Safety Gates

| Gate | Condition | Action |
|------|-----------|--------|
| Contact | left_contact_active AND right_contact_active | Block if invalid |
| Height | 0.28m <= com_z <= 0.50m | Block if unsafe |
| Roll | |roll_y| < 0.15 rad | Block if unsafe |
| Pitch Danger | |pitch_x| < 0.10 rad | Block if in danger |
| Pitch Rate | |pitch_rate_x| < 0.10 rad/s | Allow (pitch recovering) |

## Parameters

### APC1 (Moderate)
- outer_enter_m: 0.10 m
- inner_exit_m: 0.05 m
- opposite_overshoot_m: 0.01 m
- pitch_enter_rad: 0.03 rad
- pitch_safe_limit_rad: 0.08 rad
- max_cross_tau: 1.5 Nm
- smooth_alpha: 0.10
- max_rate_per_step: 0.5 Nm/step

### APC2 (Stronger, only if APC1 improves but not enough)
- Same as APC1 but:
- max_cross_tau: 2.0 Nm
- pitch_safe_limit_rad: 0.10 rad

## Integration with Existing Controller

APC adds to tau_common_unclipped AFTER other terms:
```python
tau_common_unclipped = (
    tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
    tau_support_velocity + tau_position + tau_cp + tau_com_vy +
    recenter_tau_clipped +   # F1
    hyst_tau_clipped +       # F2
    bias_tau_clipped +       # G1
    apc_tau_clipped          # NEW: Active Pitch Crossing
)
```

## Differences from F1/F2/G1

| Feature | F1 | F2 | G1 | APC |
|---------|----|----|----|-----|
| Proportional to error | Yes | No | Yes | No |
| Stateful (holds direction) | No | Yes | No | Yes |
| Explicit pitch_rate target | No | No | No | YES |
| Actively drives pitch reversal | No | No | No | YES |
| Based on bias estimation | No | No | Yes | No |
| Responds to current state | Yes | Yes | Partial | YES |

## Expected Behavior

1. **When robot leans forward AND drifts forward**:
   - APC enters CROSS_FROM_POSITIVE
   - Applies negative wheel torque
   - pitch_rate becomes negative (body tilts back)
   - Support moves back toward 0
   - Exits when signed_error reaches inner band

2. **When robot leans back AND drifts backward**:
   - APC enters CROSS_FROM_NEGATIVE
   - Applies positive wheel torque
   - pitch_rate becomes positive (body tilts forward)
   - Support moves back toward 0
   - Exits when signed_error reaches inner band

3. **Does NOT reverse when**:
   - pitch_x merely decreases (could be natural oscillation)
   - pitch_x increases but support hasn't drifted far
   - Safety gates fail

## Metrics to Track

- APC active percent
- State occupancy (time in each state)
- Crossings from positive/negative
- Time from entry to pitch_rate reversal
- Time from entry to support inner band
- Torque direction matches target
- Premature reversal count
- Safety override count
