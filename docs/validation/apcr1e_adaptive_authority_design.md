# APCR1e Adaptive Authority Design

## Profile Name

`APCR1e_adaptive_symmetric_soft_band`

## Core Concept

APCR1d is symmetric and soft (max_tau=0.75), but the user requires automatic authority increase when correction force is insufficient. APCR1e should:
1. Start with moderate base torque
2. Automatically increase torque when error keeps growing or when startup needs stronger correction
3. Reduce torque near zero to avoid overshoot
4. Remain symmetric for positive and negative signed support error

## Design Parameters

### Base Torque Settings
- `base_tau = 0.55 Nm` - moderate starting torque
- `max_tau = 1.20 Nm` - maximum adaptive torque
- `boost_tau_max = 0.65 Nm` - maximum boost above base

### Error Thresholds
- `inner_deadband_m = 0.02 m` - below this, torque decays toward zero
- `soft_enter_m = 0.045 m` - start applying soft proportional torque
- `desired_band_m = 0.08 m` - target operating band
- `full_torque_error_m = 0.10 m` - error magnitude for full proportional torque

### Adaptive Boost Triggers
Authority increases if any of these conditions are true:
1. `abs_e > desired_band_m` (0.08 m) - error exceeds target band
2. `moving_away_from_zero` - `e * e_dot > 0` and `abs_e > boost_start_error_m`
3. `no_improvement_count >= no_improvement_window_steps` - error not reducing
4. `pitch_x and signed_error have same sign and pitch_rate worsening`
5. `startup_phase and high pitch risk` - first 50 steps

### Startup Settings
- `startup_boost_steps = 50` - startup phase duration
- `startup_boost_max_tau = 1.0 Nm` - max torque during startup
- `disable_velocity_decay_during_startup = True`

### Velocity Decay
- `velocity_decay_factor = 0.5` - multiply torque by this when moving toward zero
- `disable_velocity_decay_when_abs_e_gt = 0.10 m`
- `disable_velocity_decay_during_startup = True`

### Rate Limiting
- `max_rate_per_step = 0.35 Nm/step` - maximum torque change per step

### Smoothing
- `smooth_alpha = 0.10` - exponential smoothing factor

## Torque Shaping Law

```
e = signed_error
abs_e = abs(e)
e_dot = signed_error_rate
direction = -sign(e)

# Moving direction detection
moving_away = e * e_dot > 0
moving_toward_zero = e * e_dot < 0

# Error magnitude zones
in_deadband = abs_e <= inner_deadband_m
in_soft_zone = inner_deadband_m < abs_e < soft_enter_m
in_proportional_zone = soft_enter_m <= abs_e < full_torque_error_m
in_saturation_zone = abs_e >= full_torque_error_m
beyond_desired_band = abs_e > desired_band_m

# Proportional scale
if in_proportional_zone:
    scale_error = smoothstep((abs_e - inner_deadband_m) / (full_torque_error_m - inner_deadband_m))
else if in_saturation_zone:
    scale_error = 1.0
else:
    scale_error = 0.0

# Adaptive max tau
boost_tau = 0.0
boost_reason = "none"

# Condition 1: Beyond desired band
if beyond_desired_band:
    boost_tau = min(boost_tau_max, boost_tau + boost_tau_max * 0.5)
    boost_reason = "beyond_band"

# Condition 2: Moving away from zero
if moving_away and abs_e >= boost_start_error_m:
    boost_tau = min(boost_tau_max, boost_tau + boost_tau_max * 0.4)
    boost_reason = "moving_away"

# Condition 3: No improvement for N steps
if no_improvement_count >= no_improvement_window_steps:
    boost_tau = min(boost_tau_max, boost_tau + boost_tau_max * 0.3)
    boost_reason = "no_improvement"

# Condition 4: Startup boost
if step < startup_boost_steps and (abs_e > 0.04 or pitch_x > 0.02):
    boost_tau = min(boost_tau_max, boost_tau + 0.3)
    boost_reason = "startup"
    startup_boost_active = True

# Final adaptive max tau
adaptive_max_tau = base_tau + boost_tau

# Base torque calculation
tau_base = direction * proportional_scale * adaptive_max_tau

# Velocity decay
velocity_decay_applied = False
if moving_toward_zero and abs_e < disable_velocity_decay_when_abs_e_gt and not in_startup:
    tau_base *= velocity_decay_factor
    velocity_decay_applied = True

# Rate limiting
tau_clamped = clamp_rate(tau_base, tau_prev, max_rate_per_step)

# Smoothing
tau_smooth = smooth_alpha * tau_clamped + (1 - smooth_alpha) * tau_prev

# Final torque
tau_final = direction * min(abs(tau_smooth), adaptive_max_tau)
```

## Telemetry Fields

| Field | Description |
|-------|-------------|
| `adaptive_enabled` | Boolean, always True for APCR1e |
| `base_tau` | Base torque setting (0.55 Nm) |
| `adaptive_max_tau` | Current adaptive max torque |
| `boost_tau` | Current boost amount |
| `boost_reason` | Reason for boost: "none", "beyond_band", "moving_away", "no_improvement", "startup" |
| `moving_away_from_zero` | Boolean |
| `moving_toward_zero` | Boolean |
| `no_improvement_count` | Steps since error decreased |
| `startup_boost_active` | Boolean |
| `velocity_decay_applied` | Boolean |
| `velocity_decay_disabled_reason` | "none", "startup", "high_error" |
| `abs_error_m` | Current absolute signed_error |
| `error_rate_mps` | Current signed_error_rate |
| `proportional_scale` | Current proportional scale (0-1) |
| `tau_before_rate_limit` | Torque before rate limiting |
| `tau_after_rate_limit` | Torque after rate limiting |
| `tau_final` | Final output torque |

## Safety Gates

Same as APCR1d:
- contact invalid → disable
- height unsafe → disable
- roll unsafe → disable
- pitch beyond hard stop → disable
- non-wheel contact → disable
- hidden torque or ownership violation → disable

## Symmetry

Positive and negative signed_error of the same magnitude produce equal-magnitude opposite-sign torque, except for:
- Measured dynamics feedback through e_dot
- Error direction and magnitude differences
- Safety gate overrides

## Implementation Notes

1. Add `APCR1e_adaptive_symmetric_soft_band` to profile choices
2. Add `_AdaptiveAPCR1eProfile` class with parameters above
3. Add `no_improvement_count` tracking in controller state
4. Add adaptive torque calculation in `_compute_active_pitch_crossing_torque`
5. Add telemetry fields for adaptive behavior
6. Maintain backward compatibility with APCR1d structure

## Test Cases

1. Zero error → zero torque
2. Small error in deadband → torque decays toward zero
3. Error in proportional zone → proportional torque
4. Error beyond desired band → increased authority
5. Error moving away → increased authority
6. No improvement → increased authority
7. Startup phase with error → startup boost
8. Moving toward zero → velocity decay
9. High error → velocity decay disabled
10. Rate limit → torque changes smoothly
11. Symmetric positive/negative error → symmetric torque
