# APCR1g Predictive Fast Response Phase Brake - Design

## Profile Name
`APCR1g_predictive_fast_response_phase_brake`

## Version
1.0

## Created
2026-06-09

## Parent
`APCR1f_adaptive_fast_response_phase_brake`

## Design Principle
**Predict near-future drift using error rate and intervene earlier when error is moving away from zero.**

APCR1f reacts to current error magnitude. APCR1g predicts where the error will be in `lead_time_s` seconds and activates sooner if the predicted error exceeds thresholds.

## Problem Statement

APCR1f reaches +0.157 m max positive drift despite:
- Earlier intervention (soft_enter = 0.035 m vs 0.05 m in APCR1e)
- Phase brake when error returns toward zero
- Boost when error growing 3+ consecutive steps
- Higher max_tau (1.40 Nm vs 1.20 Nm)
- Faster rate limit (0.55 Nm/step vs 0.35 Nm/step)

**Root cause:** APCR1f only responds when current error exceeds thresholds. It does not predict future error based on drift rate.

## Core Innovation: Predictive Error

```
e = signed physical drift
e_dot = signed drift rate (sagittal_velocity_m_s)
e_pred = e + lead_time_s * e_dot  # predicted error in lead_time seconds

abs_e = abs(e)
abs_pred = abs(e_pred)
moving_away = e * e_dot > 0
moving_toward_zero = e * e_dot < 0
direction = -sign(e_pred if abs_pred > abs_e else e)
```

Key insight: If error is growing away from zero at 0.05 m/s, in 0.10 seconds the error will be 0.005 m larger. APCR1g activates when predicted error exceeds thresholds, not just current error.

## Symmetric Logic

APCR1g uses the same logic for positive and negative drift:
- `abs(e)`, `abs(e_dot)`, `abs(e_pred)` for magnitude
- `moving_away` / `moving_toward_zero` for direction
- `direction = -sign(...)` for torque sign

## Parameters

### Prediction
| Parameter | Value | Description |
|-----------|-------|-------------|
| predictive_enabled | True | Enable predictive error logic |
| lead_time_s | 0.10 | Seconds to predict ahead |
| predicted_enter_m | 0.07 | Activate when abs_pred > this AND moving_away |
| predicted_full_response_m | 0.10 | Boost authority when abs_pred > this |
| predicted_emergency_m | 0.12 | Emergency mode when abs_pred > this |

### Current-Error Thresholds
| Parameter | Value | Description |
|-----------|-------|-------------|
| inner_deadband_m | 0.012 | Error below this: tau decays to zero |
| soft_enter_m | 0.030 | Earlier entry than APCR1f's 0.035 |
| desired_band_m | 0.075 | Wider comfortable band |
| full_torque_error_m | 0.095 | Full torque at this error |
| emergency_error_m | 0.115 | Emergency mode trigger |

### Authority
| Parameter | Value | Description |
|-----------|-------|-------------|
| base_tau | 0.45 Nm | Base starting torque |
| max_tau | 1.55 Nm | Higher ceiling than APCR1f's 1.40 |
| boost_tau_max | 1.10 Nm | Larger boost capability than APCR1f's 0.95 |
| startup_boost_max_tau | 1.25 Nm | Higher startup authority than APCR1f's 1.20 |
| startup_boost_steps | 50 | Startup phase duration |

### Response Speed
| Parameter | Value | Description |
|-----------|-------|-------------|
| max_rate_per_step | 0.70 Nm/step | Faster than APCR1f's 0.55 |
| boost_rate_per_step | 0.35 Nm/step | Rate for adaptive boost |
| decay_rate_per_step | 0.55 Nm/step | Faster decay when returning |
| smooth_alpha | 0.22 | More responsive smoothing |

### Phase-Aware Braking
| Parameter | Value | Description |
|-----------|-------|-------------|
| phase_brake_enabled | True | Enable phase-aware braking |
| phase_brake_threshold_m | 0.075 | Apply brake below this |
| phase_brake_strong_threshold_m | 0.050 | Strong brake closer to zero |
| phase_brake_factor | 0.55 | Reduce scale by this when braking |
| phase_brake_strong_factor | 0.35 | Stronger reduction near zero |

### Adaptive Response
| Parameter | Value | Description |
|-----------|-------|-------------|
| no_improvement_window | 4 | Boost after 4 steps without improvement (vs 5 in APCR1f) |
| increasing_error_threshold_steps | 2 | Boost when error grows 2+ steps (vs 3 in APCR1f) |
| increasing_error_boost_factor | 0.35 | Boost factor for growing error |

## Control Rules

### Rule 1: Predictive Activation
```
IF abs_pred > predicted_enter_m AND moving_away:
    activate earlier even if abs_e is still below soft_enter
```

### Rule 2: Predictive Boost
```
IF abs_pred > predicted_full_response_m:
    boost authority before actual error reaches full_torque_error_m
```

### Rule 3: Predictive Emergency
```
IF abs_pred > predicted_emergency_m:
    emergency mode triggered early
```

### Rule 4: Strong Response When Moving Away
```
IF abs_e > desired_band_m AND moving_away:
    increase boost faster
```

### Rule 5: Disable Velocity Decay at High Error
```
IF abs_e > 0.10:
    disable velocity decay, use strong authority
```

### Rule 6: Phase Brake When Returning
```
IF moving_toward_zero AND abs_e <= 0.10:
    apply phase brake
```

### Rule 7: Strong Phase Brake Near Zero
```
IF moving_toward_zero AND abs_e <= 0.075:
    apply stronger phase brake and faster decay
```

### Rule 8: Decay Toward Zero
```
IF abs_e <= inner_deadband_m:
    tau decays toward zero
```

### Rule 9: No-Improvement Boost
```
IF error has not improved for 4 consecutive steps:
    increase boost
```

### Rule 10: Error-Increasing Boost
```
IF error is increasing for 2 consecutive steps:
    increase boost faster
```

### Rule 11: Max Torque Cap
```
Do not let torque exceed max_tau
```

## Hard Safety Gates
- contact invalid
- height unsafe
- roll unsafe
- pitch beyond hard stop
- non-wheel contact
- hidden torque
- ownership violation

## Telemetry Fields

### Predictive Logic
- `active_pitch_crossing_predictive_enabled`
- `active_pitch_crossing_lead_time_s`
- `active_pitch_crossing_predicted_error_m`
- `active_pitch_crossing_abs_predicted_error_m`
- `active_pitch_crossing_predicted_enter_m`
- `active_pitch_crossing_predicted_full_response_m`
- `active_pitch_crossing_predicted_emergency_m`
- `active_pitch_crossing_predictive_trigger_active`
- `active_pitch_crossing_predictive_boost_active`
- `active_pitch_crossing_direction_source`

### Phase Brake
- `active_pitch_crossing_phase_brake_strong_active`
- `active_pitch_crossing_phase_brake_factor_current`

### Adaptive
- `active_pitch_crossing_adaptive_tau_limit`
- `active_pitch_crossing_tau_before_rate_limit`
- `active_pitch_crossing_tau_after_rate_limit`
- `active_pitch_crossing_tau_final`

### Metric Discipline
- `active_pitch_crossing_physical_drift_column_used`

## Target Metrics

| Metric | Target | APCR1f | APCR1g Goal |
|--------|--------|--------|------------|
| max positive drift | < +0.14 m | +0.157 m | < +0.14 m |
| min negative drift | >= -0.08 m | -0.049 m | >= -0.08 m |
| P2P | < 0.18 m | 0.206 m | < 0.18 m |
| outside ±0.15 | <= APCR1f | 2.2% | < 2.2% |
| outside ±0.10 | lower than APCR1f | 32.6% | < 30% |

## Applies To Variants
- `low_0p300`
- `low_0p330`
- `low_0p360`
- `extreme_height`

## Key Differences from APCR1f

| Parameter | APCR1f | APCR1g | Rationale |
|-----------|--------|--------|-----------|
| prediction | No | Yes | Intervene earlier |
| lead_time_s | N/A | 0.10 | Predict 100ms ahead |
| predicted_enter_m | N/A | 0.07 | Earlier activation |
| max_tau | 1.40 Nm | 1.55 Nm | More authority |
| max_rate_per_step | 0.55 | 0.70 | Faster response |
| soft_enter_m | 0.035 | 0.030 | Earlier entry |
| desired_band_m | 0.08 | 0.075 | Tighter band |
| phase_brake_threshold_m | 0.08 | 0.075 | Earlier brake |
| phase_brake_strong_threshold_m | N/A | 0.050 | New strong brake |
| phase_brake_factor | 0.6 | 0.55 | Stronger damping |
| phase_brake_strong_factor | N/A | 0.35 | Strong damping |
| no_improvement_window | 5 | 4 | Faster boost |
| increasing_error_threshold_steps | 3 | 2 | Faster boost |
| smooth_alpha | 0.18 | 0.22 | More responsive |

## Expected Behavior

1. **Earlier intervention:** APCR1g activates when predicted error exceeds 0.07 m, even if current error is below 0.035 m threshold.

2. **Predictive boost:** When predicted error exceeds 0.10 m, APCR1g boosts authority before actual error reaches full torque threshold.

3. **Reduced positive peak:** By intervening earlier, APCR1g should prevent error from reaching +0.157 m.

4. **Symmetric response:** Same logic for positive and negative drift, using abs() for magnitude.

5. **Phase brake near zero:** Stronger damping when error is close to zero and returning.

6. **Startup stability:** Startup boost remains to handle initial transient.
