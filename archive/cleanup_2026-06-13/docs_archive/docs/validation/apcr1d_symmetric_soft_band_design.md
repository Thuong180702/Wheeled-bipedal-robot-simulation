# APCR1d Symmetric Soft Band Controller Design

## Context

APCR1c improved centering and reduced positive bias compared to D2, but it worsened oscillation amplitude:

| Metric | D2 | APCR1c |
|--------|-----|--------|
| max drift | +0.1757 m | +0.1682 m |
| min drift | +0.0142 m | -0.0716 m |
| peak-to-peak | 0.1615 m | 0.2398 m |

**Problem**: APCR1c's constant-torque bang-bang control improves centering but creates a wider oscillation envelope. The robot crosses zero but then swings too far negative.

## Design Principle

APCR1d uses **symmetric proportional torque shaping** instead of bang-bang control:

```
e = signed_support_error
abs_e = abs(e)
direction = -sign(e)  # Push toward zero

if abs_e <= inner_deadband_m:
    scale = 0
elif abs_e >= full_torque_error_m:
    scale = 1
else:
    scale = smoothstep((abs_e - inner_deadband_m) / (full_torque_error_m - inner_deadband_m))

raw_tau = direction * max_cross_tau * scale
```

Additionally, **velocity-aware decay** reduces torque when the error is already moving toward zero:

```
if e * e_dot < 0:  # Moving toward zero
    scale = scale * velocity_decay_factor  # Reduce to prevent overshoot
```

## Key Differences from APCR1c

| Aspect | APCR1c | APCR1d |
|--------|--------|--------|
| Torque shape | Constant (bang-bang) | Proportional (soft) |
| Entry threshold | 0.08 m | 0.05 m (softer, earlier) |
| Exit deadband | 0.07 m | 0.02 m (narrower) |
| Full torque at | 0.10 m | 0.08 m |
| Max torque | 1.0 Nm | 0.75 Nm |
| Velocity decay | None | 0.5 factor when moving toward zero |
| Symmetry | State-machine based | abs(error)-based, inherently symmetric |

## Parameters

### APCR1d Core Parameters

```yaml
# Entry/exit thresholds
apc_soft_enter_m: 0.05        # Enter soft recenter when |error| > 0.05
apc_inner_deadband_m: 0.02   # Exit when |error| <= 0.02
apc_full_torque_error_m: 0.08 # Full torque when |error| >= 0.08
apc_desired_band_m: 0.08      # Target band width

# Torque shaping
apc_max_cross_tau: 0.75       # Nm - max torque (lower than APCR1c's 1.0)
apc_smooth_alpha: 0.10        # Smoothing factor
apc_max_rate_per_step: 0.30   # Nm/step - rate limit

# Velocity decay
apc_velocity_decay_enabled: True
apc_velocity_decay_factor: 0.5  # Reduce torque by 50% when moving toward zero
```

### Safety Thresholds

```yaml
apc_pitch_enter_rad: 0.03     # Pitch threshold to enter
apcr_pitch_hard_stop_rad: 0.30 # Hard emergency stop
apcr_roll_hard_stop_rad: 0.15  # Lateral stability
apcr_min_com_z_m: 0.27        # Min safe height
apcr_max_com_z_m: 0.50        # Max operating height
```

### Gates

```yaml
apc_contact_gate: True
apc_height_gate: True
apc_roll_gate: True
```

## State Machine

APCR1d uses three states instead of APCR1c's four:

| State | Entry Condition | Exit Condition | Torque |
|-------|-----------------|----------------|--------|
| NEUTRAL | Default | | 0 |
| SOFT_RECENTER | \|error\| > soft_enter AND pitch compatible | \|error\| <= inner_deadband | Proportional to \|error\| |
| SAFETY_DECAY | Gate failure OR pitch danger | Gate restored | Decay to 0 |

## Torque Computation

```python
def compute_apcr1d_torque(e, e_dot, pitch_x_rad, pitch_rate_x_rad_s, gates):
    """APCR1d symmetric soft band torque computation."""
    
    abs_e = abs(e)
    direction = -sign(e) if e != 0 else 0
    
    # Gate checks
    if not gates.all_pass():
        return decay_tau(prev_tau, rate=0.3)
    
    # Pitch safety
    pitch_safe = abs(pitch_x_rad) < apc_pitch_enter_rad or \
                 (pitch_x_rad * pitch_rate_x_rad_s < 0)  # recovering
    pitch_danger = abs(pitch_x_rad) > apcr_pitch_hard_stop_rad
    
    if pitch_danger or not pitch_safe:
        return decay_tau(prev_tau, rate=0.3)
    
    # Entry: enter when |error| > soft_enter
    if abs_e <= apc_soft_enter_m:
        return decay_tau(prev_tau, rate=0.3)
    
    # Proportional shaping
    if abs_e <= apc_inner_deadband_m:
        scale = 0.0
    elif abs_e >= apc_full_torque_error_m:
        scale = 1.0
    else:
        # Smooth interpolation
        u = (abs_e - apc_inner_deadband_m) / (apc_full_torque_error_m - apc_inner_deadband_m)
        scale = smoothstep(u)
    
    # Velocity decay: reduce torque if moving toward zero
    velocity_decay = 1.0
    if apc_velocity_decay_enabled and e * e_dot < 0:
        velocity_decay = apc_velocity_decay_factor
    
    scale = scale * velocity_decay
    
    # Raw torque
    raw_tau = direction * apc_max_cross_tau * scale
    
    # Smooth and rate limit
    tau = smoothstep(raw_tau, prev_tau, alpha=apc_smooth_alpha)
    tau = rate_limit(tau, prev_tau, max_rate=apc_max_rate_per_step)
    
    return tau
```

## Expected Behavior

### Desired Drift Envelope

```
Target: signed_error oscillates between -0.08 m and +0.08 m

APCR1d should:
1. Start acting earlier than APCR1c (0.05 m vs 0.08 m entry)
2. Use softer proportional torque instead of constant torque
3. Reduce positive max drift from APCR1c's +0.1682 m
4. Keep negative drift bounded (target: >= -0.08 m)
5. Reduce peak-to-peak amplitude below APCR1c's 0.2398 m
```

### Comparison Table

| Metric | D2 | APCR1c | APCR1d Target |
|--------|-----|--------|--------------|
| max drift | +0.176 | +0.168 | < +0.15 |
| min drift | +0.014 | -0.072 | > -0.08 |
| peak-to-peak | 0.162 | 0.240 | < 0.20 |
| outside ±0.15 | ~10% | ~15% | < 5% |
| centering | positive bias | crosses zero | balanced |

## Telemetry Fields

### APCR1d Specific

- `active_pitch_crossing_torque_mode`: "proportional_soft_band"
- `active_pitch_crossing_soft_enter_m`: 0.05
- `active_pitch_crossing_inner_deadband_m`: 0.02
- `active_pitch_crossing_full_torque_error_m`: 0.08
- `active_pitch_crossing_desired_band_m`: 0.08
- `active_pitch_crossing_abs_error_m`: |signed_error|
- `active_pitch_crossing_error_rate_mps`: e_dot
- `active_pitch_crossing_error_moving_toward_zero`: bool
- `active_pitch_crossing_proportional_scale`: current scale [0, 1]
- `active_pitch_crossing_velocity_decay_factor`: 0.5
- `active_pitch_crossing_velocity_decay_active`: bool

### State Machine

- `active_pitch_crossing_state`: NEUTRAL | SOFT_RECENTER | SAFETY_DECAY
- `active_pitch_crossing_state_id`: 0 | 1 | 2
- `active_pitch_crossing_active`: bool

## Implementation Notes

1. **No WBC**: APCR1d does not enable WBC (whole-body controller) - that's a separate concern.

2. **No HY2-DIV**: APCR1d does not modify HY2-DIV behavior.

3. **Opt-in only**: APCR1d is NOT the default. It must be explicitly selected via `--vd-sagittal-authority-profile APCR1d_symmetric_soft_band_control`.

4. **D2 unchanged**: The D2 baseline profile remains unchanged.

5. **APCR1/APCR1b/APCR1c unchanged**: Previous profiles remain available.

## Success Criteria

APCR1d 500-step validation passes if:

- Survives 500 steps without falling
- peak-to-peak amplitude < APCR1c (0.240 m)
- max positive drift < APCR1c (+0.168 m)
- min negative drift >= -0.08 m
- outside ±0.15 m violations < APCR1c
- No instability introduced