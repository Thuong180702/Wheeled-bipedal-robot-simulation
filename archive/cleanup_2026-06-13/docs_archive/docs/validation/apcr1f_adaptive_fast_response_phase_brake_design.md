# APCR1f Adaptive Fast Response with Phase Brake Design

## Classification: `APCR1F_DESIGN_CANDIDATE`

## Purpose

APCR1f is designed to reduce max positive drift by:
1. **Earlier intervention**: React when error exceeds 0.035 m (softer threshold than APCR1e's 0.05 m)
2. **Faster response**: Higher torque rate limit (0.55 Nm/step vs 0.35 Nm/step)
3. **Higher authority ceiling**: max_tau = 1.40 Nm (vs APCR1e's 1.20 Nm)
4. **Phase-aware braking**: Decay torque faster when error is already returning toward zero

## Core Design Principle

> React earlier and stronger when error is GROWING away from zero.  
> Brake harder when error is ALREADY RETURNING toward zero.

This prevents the APCR1e failure mode where stronger torque increases the oscillation envelope.

## Parameter Comparison

| Parameter | APCR1e | APCR1f | Rationale |
|-----------|--------|--------|-----------|
| inner_deadband_m | 0.05 | 0.015 | Earlier deadband exit |
| soft_enter_m | 0.05 | 0.035 | Earlier soft response entry |
| desired_band_m | 0.05 | 0.08 | Wider comfortable band |
| full_torque_error_m | 0.12 | 0.10 | Full torque at smaller error |
| emergency_error_m | N/A | 0.12 | Emergency mode trigger |
| base_tau | 0.55 | 0.45 | Slightly lower base |
| max_tau | 1.20 | 1.40 | Higher ceiling |
| boost_tau_max | 0.65 | 0.95 | Larger boost capability |
| startup_boost_max_tau | 1.00 | 1.20 | Higher startup authority |
| max_rate_per_step | 0.35 | 0.55 | Faster response |
| boost_rate_per_step | N/A | 0.25 | Rate for adaptive boost |
| decay_rate_per_step | N/A | 0.45 | Faster decay when returning |
| smooth_alpha | 0.10 | 0.18 | More responsive smoothing |
| no_improvement_window | 8 | 5 | Faster boost activation |
| increasing_error_threshold | N/A | 3 | Boost when error grows 3 steps |
| phase_brake_threshold | N/A | 0.08 | Apply brake below this |
| phase_brake_damping | N/A | 0.6 | Additional damping factor |

## Control Regions

### Region 1: Deadband
- **Condition**: abs_error <= 0.015 m
- **Action**: tau -> 0
- **Rationale**: Ignore small oscillations within tolerance

### Region 2: Early Soft Response
- **Condition**: 0.015 < abs_error <= 0.035 m
- **Action**: Light proportional torque, scale = (error - deadband) / (soft_enter - deadband)
- **Rationale**: Gentle correction before error grows

### Region 3: Active Response
- **Condition**: 0.035 < abs_error <= 0.08 m
- **Action**: Proportional torque with smoothstep interpolation
- **Rationale**: Stronger correction as error grows

### Region 4: Strong Response
- **Condition**: 0.08 < abs_error <= 0.10 m
- **Action**: Base torque + adaptive boost
- **Rationale**: Error exceeds comfortable band

### Region 5: Emergency
- **Condition**: abs_error > 0.10 m OR error growing 3+ consecutive steps
- **Action**: Maximum adaptive authority
- **Rationale**: Error is critical, apply all available torque

## Phase Logic

### Moving Away From Zero
```
if signed_error * error_rate > 0:  # Error magnitude increasing
    disable velocity decay
    increase boost faster (boost_rate_per_step)
    allow adaptive_max_tau = 1.40 Nm
```

### Moving Toward Zero (Phase Brake)
```
if signed_error * error_rate < 0:  # Error magnitude decreasing
    if abs_error > 0.10:
        # Keep enough authority, don't decay too early
        maintain base tau level
    elif abs_error > 0.08:
        # Apply phase brake
        proportional_scale *= 0.6
    else:
        # Error near zero, decay quickly
        apply decay_rate_per_step
```

## Startup Behavior

### First 50 Steps
- Disable velocity decay
- Allow startup_boost up to 1.20 Nm
- If pitch > 0.02 rad or error > 0.04 m, engage boost

### After Startup
- Normal adaptive authority behavior
- No-improvement boost after 5 steps (vs 8 in APCR1e)
- Error-growing boost after 3 steps

## Safety Gates

### Hard Safety (block APCR entirely)
- contact_force_valid = False
- com_z < 0.27 m or com_z > 0.50 m
- roll_x > 0.15 rad
- pitch_x > 0.30 rad
- pitch_x < -0.30 rad
- ownership violation

### Soft Safety (reduce torque)
- pitch_x > 0.10 rad: scale torque by (0.10 / pitch_abs)
- pitch_x < -0.10 rad: scale torque by (0.10 / pitch_abs)

## Telemetry Fields

Required new fields:
- `active_pitch_crossing_fast_response_enabled`
- `active_pitch_crossing_phase_brake_enabled`
- `active_pitch_crossing_boost_rate`
- `active_pitch_crossing_decay_rate`
- `active_pitch_crossing_phase_brake_active`
- `active_pitch_crossing_phase_brake_tau`
- `active_pitch_crossing_increasing_error_count`
- `active_pitch_crossing_tau_before_rate_limit`
- `active_pitch_crossing_tau_after_rate_limit`
- `active_pitch_crossing_physical_drift_column_used`

## Target Metrics

| Metric | D2 | APCR1e | APCR1f Target |
|--------|-----|--------|---------------|
| max positive drift | +0.16 m | +0.17 m | < +0.14 m |
| min negative drift | -0.03 m | -0.06 m | >= -0.08 m |
| P2P | 0.162 m | 0.235 m | < 0.18 m |
| outside ±0.15 | 19.2% | 10.1% | < 8% |

## Symmetric Logic Invariant

The same formula applies for positive and negative signed_error:
```
direction = -sign(signed_error)  # Always push toward zero
```

## Files to Modify

1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
   - Add APCR1f parameters to SagittalAuthoritySchedule dataclass
   - Add APCR1f profile definition (APCR1F_FAST_RESPONSE_PHASE_BRAKE)
   - Add to profile registry
   - Add state variables for phase brake tracking
   - Implement phase brake logic
   - Implement faster rate limiting
   - Add new telemetry fields

2. `tests/test_sagittal_velocity_damped_balance_controller.py`
   - Add tests for APCR1f profile existence
   - Add tests for phase brake behavior
   - Add tests for symmetric torque
   - Add tests for increasing error boost