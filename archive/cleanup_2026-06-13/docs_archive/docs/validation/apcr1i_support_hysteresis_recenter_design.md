# APCR1i Support Hysteresis Recenter Design

## Date
2026-06-10

## Based On
- APCR1h (after wiring fix) - correct torque sign, drift priority override

## Design Principle

The user's desired behavior:
> If support drift moves far away from zero:
> - wheels may move faster and reverse direction if needed
> - support drift recovery must be prioritized even if pitch is near balanced
> - the controller must keep driving support back toward zero
> - once the drift direction reverses, do not immediately switch back
> - hold the recenter phase until the support error reaches near zero or crosses slightly to the opposite side
> - then switch/release according to symmetric hysteresis

## Key Differences from APCR1h

| Aspect | APCR1h | APCR1i |
|--------|--------|--------|
| State machine | Proportional soft band | Symmetric hysteresis |
| Exit condition | Error moving toward zero | Error inside inner band OR crosses to opposite |
| Phase brake | Disabled during drift priority | Disabled while outside inner band |
| Hysteresis | None | Full symmetric hysteresis |
| Pitch gating | Pitch must be safe | Drift prioritized over pitch |

## APCR1i State Machine

```
States:
- NEUTRAL: No recenter active, error near zero
- RECENTER_FROM_POSITIVE: Positive drift, driving backward
- RECENTER_FROM_NEGATIVE: Negative drift, driving forward
- HOLD_THROUGH_ZERO: Error crossing zero, holding direction
- EMERGENCY: Error > 0.12m, maximum authority

Entries:
- NEUTRAL → RECENTER_FROM_POSITIVE: e > +outer_enter_m (0.08m)
- NEUTRAL → RECENTER_FROM_NEGATIVE: e < -outer_enter_m (-0.08m)
- RECENTER_FROM_POSITIVE → HOLD_THROUGH_ZERO: e crosses below inner_exit_m (0.03m)
- RECENTER_FROM_NEGATIVE → HOLD_THROUGH_ZERO: e crosses above -inner_exit_m (-0.03m)
- Any → NEUTRAL: e reaches near_zero_m AND e_dot < 0 (moving toward zero)

Exits:
- RECENTER_FROM_POSITIVE → NEUTRAL or RECENTER_FROM_NEGATIVE: 
  - if e <= +inner_exit_m (0.03m) AND e_dot < 0, OR
  - if e < -opposite_release_m (-0.03m)
- RECENTER_FROM_NEGATIVE → NEUTRAL or RECENTER_FROM_POSITIVE:
  - if e >= -inner_exit_m (-0.03m) AND e_dot < 0, OR
  - if e > +opposite_release_m (+0.03m)
- HOLD_THROUGH_ZERO → NEUTRAL:
  - if |e| < inner_exit_m AND |e_dot| < velocity_threshold
  - if e_dot changes sign (error starts moving away)
```

## APCR1i Parameters

### Entry/Exit Thresholds

| Parameter | Value | Description |
|-----------|-------|-------------|
| `outer_enter_m` | 0.08 | Enter recenter when |e| > this |
| `inner_exit_m` | 0.03 | Exit recenter when |e| <= this |
| `opposite_release_m` | 0.03 | Allow small overshoot into opposite direction |
| `near_zero_m` | 0.01 | Error considered near zero |
| `emergency_enter_m` | 0.12 | Emergency clamp activates |
| `hard_enter_m` | 0.15 | Hard safety activates |

### Authority Levels

| Level | tau_max (Nm) | Rate (Nm/step) | Description |
|-------|--------------|----------------|-------------|
| Base | 0.45 | 0.30 | Starting torque |
| Recenter | 1.75 | 0.90 | During recenter state |
| Emergency | 2.00 | 1.00 | When |e| > 0.12m |
| Hold | 1.50 | 0.70 | Hold through zero |
| Decay | - | 0.50 | Torque decay when returning |

### Phase Brake Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `phase_brake_threshold_m` | 0.05 | Enable phase brake when |e| < this |
| `phase_brake_disable_in_recenter` | True | Disable phase brake in recenter state |

### Safety Gates

| Gate | Threshold | Behavior |
|------|-----------|----------|
| Contact | Both wheels active | Block if invalid |
| Height | 0.27 < h < 0.50 m | Block if unsafe |
| Roll | |e_roll| < 0.15 rad | Block if unsafe |
| Pitch hard | |pitch| < 0.30 rad | Block if beyond |

## Telemetry Fields

Required telemetry for APCR1i:

| Field | Type | Description |
|-------|------|-------------|
| `active_pitch_crossing_hysteresis_enabled` | bool | Profile enables hysteresis |
| `active_pitch_crossing_hysteresis_state` | string | Current state name |
| `active_pitch_crossing_hysteresis_state_id` | int | Current state ID |
| `active_pitch_crossing_recenter_direction` | string | positive/negative/none |
| `active_pitch_crossing_recenter_hold_active` | bool | Hold-through-zero active |
| `active_pitch_crossing_recenter_entry_e` | float | Error at state entry |
| `active_pitch_crossing_recenter_exit_e` | float | Error at state exit |
| `active_pitch_crossing_inner_release_m` | float | Inner release threshold |
| `active_pitch_crossing_opposite_release_m` | float | Opposite release threshold |
| `active_pitch_crossing_emergency_clamp_active` | bool | Emergency clamp active |
| `active_pitch_crossing_selected_tau_limit` | float | Current tau limit |
| `active_pitch_crossing_selected_rate_limit` | float | Current rate limit |
| `active_pitch_crossing_phase_brake_disabled_reason` | string | Why phase brake disabled |
| `active_pitch_crossing_support_priority_over_pitch` | bool | Support prioritized over pitch |
| `active_pitch_crossing_physical_drift_column_used` | string | Column used for drift |

## Expected Behavior

### Scenario 1: Positive drift exceeding outer threshold

```
Time  | e (m) | State                  | tau direction | Phase brake
------|-------|------------------------|---------------|------------
t0    | 0.05  | NEUTRAL                | none          | enabled
t1    | 0.08  | RECENTER_FROM_POSITIVE | negative      | DISABLED
t2    | 0.10  | RECENTER_FROM_POSITIVE | negative      | DISABLED
t3    | 0.12  | RECENTER_FROM_POSITIVE | negative      | DISABLED (emergency)
t4    | 0.10  | RECENTER_FROM_POSITIVE | negative      | DISABLED
t5    | 0.05  | RECENTER_FROM_POSITIVE | negative      | DISABLED
t6    | 0.03  | HOLD_THROUGH_ZERO      | negative      | DISABLED
t7    | 0.00  | HOLD_THROUGH_ZERO      | negative      | DISABLED
t8    | -0.01  | HOLD_THROUGH_ZERO      | negative      | DISABLED
t9    | 0.01  | NEUTRAL                | none          | enabled
```

**Key behavior**: 
- Does NOT exit at t4-t5 when e starts decreasing (APCR1h would exit)
- Holds direction through zero crossing (t6-t8)
- Only exits when e_dot reverses AND e is near zero

### Scenario 2: Error oscillating

```
Time  | e (m) | State                  | tau direction
------|-------|------------------------|---------------
t0    | 0.08  | RECENTER_FROM_POSITIVE | negative
t1    | 0.10  | RECENTER_FROM_POSITIVE | negative
t2    | 0.09  | RECENTER_FROM_POSITIVE | negative (e_dot < 0!)
t3    | 0.08  | RECENTER_FROM_POSITIVE | negative
t4    | 0.07  | RECENTER_FROM_POSITIVE | negative
...
t10   | 0.05  | RECENTER_FROM_POSITIVE | negative
```

**Key behavior**:
- Does NOT switch when e_dot reverses while |e| > inner_exit_m
- Only switches when |e| <= inner_exit_m OR |e| > opposite_release_m

## Implementation Notes

1. **State machine**: Use `_apc_hysteresis_state` string state
2. **Entry tracking**: Track `_apc_hysteresis_entry_e` at state entry
3. **Direction**: `tau_direction = -sign(e)` during recenter
4. **Torque**: Use full recenter max when in recenter state
5. **Phase brake**: Disabled when `_apc_hysteresis_state != NEUTRAL`
6. **Safety**: Still check contact/height/roll gates

## Profile Definition

```python
APCR1I_SUPPORT_HYSTERESIS_RECENTER = SagittalAuthoritySchedule(
    profile_name="APCR1i_support_hysteresis_recenter",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    # Hysteresis recenter parameters
    apc_hysteresis_enabled=True,
    apc_hysteresis_outer_enter_m=0.08,
    apc_hysteresis_inner_exit_m=0.03,
    apc_hysteresis_opposite_release_m=0.03,
    apc_hysteresis_near_zero_m=0.01,
    apc_hysteresis_emergency_m=0.12,
    apc_hysteresis_hard_m=0.15,
    apc_hysteresis_base_tau=0.45,
    apc_hysteresis_recenter_max_tau=1.75,
    apc_hysteresis_emergency_max_tau=2.00,
    apc_hysteresis_hold_max_tau=1.50,
    apc_hysteresis_normal_rate=0.30,
    apc_hysteresis_recenter_rate=0.90,
    apc_hysteresis_emergency_rate=1.00,
    apc_hysteresis_decay_rate=0.50,
    apc_hysteresis_phase_brake_threshold_m=0.05,
    apc_hysteresis_phase_brake_disable_in_recenter=True,
    # Safety gates
    apc_contact_gate=True,
    apc_height_gate=True,
    apc_roll_gate=True,
    apc_pitch_gate=False,  # Drift prioritized over pitch
    apc_min_com_z_m=0.27,
    apc_max_com_z_m=0.50,
    apc_pitch_danger_threshold_rad=0.30,
    apc_roll_threshold_rad=0.15,
)
```
