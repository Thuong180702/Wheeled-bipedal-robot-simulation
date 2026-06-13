# APCR1m Conditional Pitch Blend Design

## Profile Name

`APCR1m_conditional_pitch_blend_recenter`

## Design Rationale

APCR1l uses **hard suppression** (`tau_pitch = 0`) during RECENTER state. This approach is binary and may be too aggressive for certain conditions. APCR1m uses **conditional pitch blending** that:

1. Preserves tau_pitch during startup (startup guard)
2. Applies tau_pitch scaling based on error magnitude (not binary suppression)
3. Only blends when safety conditions are met (contact, height, roll, pitch)
4. Never blends when not in RECENTER state

## Key Differences from APCR1l

| Aspect | APCR1l | APCR1m |
|--------|--------|---------|
| Suppression method | Binary (0 or 1) | Scaling (0.0 to 1.0) |
| Startup protection | None | Startup guard (100-150 steps) |
| Safety gates | None | Contact, height, roll, pitch |
| Error-dependent | No | Yes (error magnitude → blend scale) |
| NEUTRAL behavior | tau_pitch unchanged | tau_pitch unchanged |

## Configuration

### Base (from APCR1k)

```python
apc_hysteresis_outer_enter_m: 0.05
apc_hysteresis_inner_exit_m: 0.03
apc_hysteresis_opposite_release_m: 0.03
apc_max_cross_tau: 2.0
apc_hysteresis_recenter_max_tau: 2.0
apc_hysteresis_emergency_max_tau: 2.2
apc_contact_gate: True
apc_height_gate: True
apc_roll_gate: True
apc_min_com_z_m: 0.27
apc_max_com_z_m: 0.50
apc_roll_threshold_rad: 0.15
```

### New APCR1m Parameters

```python
# Pitch blend configuration
apc_pitch_blend_enabled: True

# Startup guard
apc_pitch_blend_startup_guard_steps: 100  # No pitch blending for first 100 steps

# Safety thresholds
apc_pitch_blend_safe_pitch_rad: 0.15  # ~8.6 degrees
apc_pitch_blend_safe_pitch_rate_rad_s: 0.5
apc_pitch_blend_min_com_z: 0.27
apc_pitch_blend_max_roll_rad: 0.15

# Error-dependent scaling thresholds
apc_pitch_blend_deep_error_m: 0.12  # If |e| > 0.12, blend deep
apc_pitch_blend_mid_error_m: 0.08   # If 0.08 < |e| <= 0.12, blend mid
apc_pitch_blend_soft_error_m: 0.05   # If 0.05 < |e| <= 0.08, blend soft

# Blend scales for each error band
apc_pitch_blend_scale_deep: 0.0    # tau_pitch * 0.0 (effectively off)
apc_pitch_blend_scale_mid: 0.25    # tau_pitch * 0.25
apc_pitch_blend_scale_soft: 0.5     # tau_pitch * 0.5
apc_pitch_blend_scale_near: 1.0    # tau_pitch * 1.0 (no blend when |e| <= 0.05)
```

## Blend Logic

### Startup Guard

```python
if step < startup_guard_steps:
    tau_pitch_scale = 1.0  # No blending during startup
    startup_guard_active = True
else:
    startup_guard_active = False
```

### Safety Gates

```python
pitch_safe = abs(pitch_x) < safe_pitch_threshold
pitch_rate_safe = abs(pitch_rate_x) < safe_pitch_rate
height_safe = com_z > min_com_z
roll_safe = abs(roll_y) < max_roll
contact_valid = contact_state == valid

all_safe = pitch_safe and pitch_rate_safe and height_safe and roll_safe and contact_valid
```

### RECENTER State Check

```python
recenter_active = hysteresis_state in ("RECENTER_FROM_POSITIVE", "RECENTER_FROM_NEGATIVE")
```

### Blend Scale Computation

```python
def compute_pitch_blend_scale(
    error_m: float,
    recenter_active: bool,
    startup_guard_active: bool,
    all_safe: bool,
    deep_threshold: float,
    mid_threshold: float,
    soft_threshold: float,
) -> float:
    """
    Compute tau_pitch blend scale based on conditions.
    
    Returns: tau_pitch_scale (0.0 to 1.0)
    """
    # Never blend outside RECENTER
    if not recenter_active:
        return 1.0
    
    # Never blend during startup guard
    if startup_guard_active:
        return 1.0
    
    # Only blend when all safety conditions met
    if not all_safe:
        return 1.0
    
    abs_error = abs(error_m)
    
    if abs_error > deep_threshold:
        return 0.0  # Deep blend - effectively off
    elif abs_error > mid_threshold:
        return 0.25  # Mid blend
    elif abs_error > soft_threshold:
        return 0.5  # Soft blend
    else:
        return 1.0  # Near zero error - no blend
```

### Final tau_pitch Computation

```python
tau_pitch_before = kp_pitch * pitch_x  # Original computation

# Apply conditional blend
tau_pitch_scale = compute_pitch_blend_scale(...)
tau_pitch = tau_pitch_before * tau_pitch_scale
```

## Telemetry Fields

| Field | Type | Description |
|-------|------|-------------|
| `apcr1m_pitch_blend_active` | bool | Blend logic activated |
| `apcr1m_pitch_blend_scale` | float | tau_pitch scaling factor (0.0 to 1.0) |
| `apcr1m_pitch_blend_block_reason` | str | Why blend was blocked ("startup", "not_recenter", "safety", "none") |
| `apcr1m_tau_pitch_before_blend` | float | tau_pitch before scaling |
| `apcr1m_tau_pitch_after_blend` | float | tau_pitch after scaling |
| `apcr1m_startup_guard_active` | bool | Startup guard is blocking blend |
| `apcr1m_recenter_active` | bool | In RECENTER state |
| `apcr1m_pitch_safe` | bool | Pitch within safe threshold |
| `apcr1m_height_safe` | bool | Height above minimum |
| `apcr1m_contact_safe` | bool | Contact state valid |

## Block Reasons

- `"none"`: All conditions met, blend applied
- `"startup"`: Startup guard blocking blend
- `"not_recenter"`: Not in RECENTER state
- `"pitch_unsafe"`: Pitch or pitch rate unsafe
- `"height_unsafe"`: Height below minimum
- `"roll_unsafe"`: Roll exceeds threshold
- `"contact_invalid"`: Contact state invalid

## Expected Behavior

### Startup Phase (steps 0-100)
- `startup_guard_active = True`
- `pitch_blend_active = False`
- `pitch_blend_scale = 1.0`
- tau_pitch unchanged (full authority for stabilization)

### RECENTER with Small Error (|e| <= 0.05)
- `recenter_active = True`
- `pitch_blend_scale = 1.0`
- tau_pitch unchanged (small error, allow normal stabilization)

### RECENTER with Medium Error (0.05 < |e| <= 0.08)
- `recenter_active = True`
- `pitch_blend_scale = 0.5`
- tau_pitch reduced by 50%

### RECENTER with Larger Error (0.08 < |e| <= 0.12)
- `recenter_active = True`
- `pitch_blend_scale = 0.25`
- tau_pitch reduced by 75%

### RECENTER with Deep Error (|e| > 0.12)
- `recenter_active = True`
- `pitch_blend_scale = 0.0`
- tau_pitch effectively suppressed

### Safety Failure at Any Point
- `pitch_blend_scale = 1.0`
- tau_pitch unchanged (safety override)

## Implementation Notes

1. Compute tau_pitch normally first (don't change the computation order)
2. Compute pitch blend scale based on conditions
3. Apply `tau_pitch = tau_pitch * scale`
4. Log all telemetry fields
5. Do NOT modify tau_position or APCR sign
6. Do NOT patch signs in multiple places