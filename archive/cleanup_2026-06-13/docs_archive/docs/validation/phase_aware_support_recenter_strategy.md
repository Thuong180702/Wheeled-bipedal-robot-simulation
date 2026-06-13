# Phase-Aware Support Recenter Strategy

**Date**: 2026-06-08
**Candidate**: F1_phase_aware_recenter_velocity_shaping
**Status**: DESIGN COMPLETE

## Design Principle

**When the robot is in dangerous fall recovery**: Balance wins.
**When the robot is recovering or pitch is safe**: Allow/support recentering.

## Core Idea

Instead of increasing position correction cap (which causes hip yaw), add a phase-aware recenter term that:
1. Detects when pitch is recovering (not falling dangerously)
2. In those phases, applies a gentle recenter force proportional to signed support error
3. Does NOT compete with balance when pitch is unsafe

## Signal Requirements

| Signal | Source | Purpose |
|--------|--------|---------|
| `pitch_x` | telemetry | Detect fall direction |
| `pitch_rate_x` | telemetry | Detect recovery vs fall |
| `hip_yaw_comp_support_error_m` | telemetry | Signed support error |
| `hip_yaw_abs_max` | telemetry | Bounds check |
| `tau_position` | telemetry | Current position torque |
| `contact_valid` | telemetry | Safety check |

## Phase Detection Logic

```
SAFE_TO_RECENTER:
  - abs(pitch_x) < pitch_safe_threshold (e.g., 0.05 rad ≈ 3°)
  - OR pitch_rate_x indicates recovery (pitch_rate opposite to pitch)
  - AND hip_yaw_abs_max < hip_yaw_threshold (e.g., 0.10 rad)
  - AND contact_valid

DANGEROUS_FALL:
  - abs(pitch_x) > pitch_danger_threshold (e.g., 0.10 rad ≈ 6°)
  - OR hip_yaw_abs_max > hip_yaw_threshold
  - OR pitch_rate indicates accelerating fall
```

## Recenter Term Design

```python
def compute_recenter_term(signed_error, pitch_x, pitch_rate, hip_yaw, contact_valid):
    """
    Compute phase-aware recenter torque.
    
    Args:
        signed_error: hip_yaw_comp_support_error_m (signed, in meters)
        pitch_x: current pitch angle (rad)
        pitch_rate: current pitch rate (rad/s)
        hip_yaw: hip_yaw_abs_max (rad)
        contact_valid: boolean
    
    Returns:
        recenter_tau: recenter torque (Nm)
    """
    # Phase detection
    pitch_safe = abs(pitch_x) < 0.05 or (pitch_x * pitch_rate < 0)
    hip_yaw_safe = hip_yaw < 0.10
    safe = pitch_safe and hip_yaw_safe and contact_valid

    if not safe:
        return 0.0  # Let balance command dominate

    # Bounded recenter correction
    # tau_recenter = -k_recenter * signed_error
    k_recenter = 10.0  # Tunable gain
    max_recenter_tau = 1.0  # Nm - much smaller than balance authority

    recenter_tau = -k_recenter * signed_error
    recenter_tau = clip(recenter_tau, -max_recenter_tau, max_recenter_tau)

    return recenter_tau
```

## Integration with Existing Controller

The recenter term should be added as a **separate term** that:
1. Does NOT affect tau_position (which drives hip yaw)
2. Affects wheel velocity command directly
3. Only activates in safe phases

```
tau_final = tau_balance + tau_position + tau_recenter
                ↓
        wheel command
```

## Anti-Windup / Anti-Chatter

- Use hysteresis on phase detection to avoid rapid switching
- Apply smooth ramp-up/ramp-down of recenter term
- Limit recenter rate of change

```python
def compute_recenter_term_smooth(..., prev_recenter_tau, alpha=0.1):
    raw_recenter = compute_recenter_term(...)
    
    # Smooth transition
    smoothed_recenter = alpha * raw_recenter + (1 - alpha) * prev_recenter_tau
    
    # Rate limiting
    max_rate = 0.5  # Nm per step
    recenter_tau = clip(
        smoothed_recenter,
        prev_recenter_tau - max_rate,
        prev_recenter_tau + max_rate
    )
    
    return recenter_tau
```

## Telemetry Fields to Add

| Field | Type | Description |
|-------|------|-------------|
| `recenter_phase_safe` | bool | Whether recentering is safe |
| `recenter_tau` | float | Recenter torque command |
| `recenter_tau_smooth` | float | Smoothed recenter torque |
| `recenter_active` | bool | Whether recenter term is active |

## Expected Behavior

1. **During dangerous fall**: recenter_tau = 0, balance dominates
2. **During safe recovery**: recenter_tau = -k * signed_error, gently recenters
3. **During transition**: recenter_tau smooth ramp-up/down

## Pass Criteria for 500-Step Evaluation

- support_position_error_m crossings >0.15 reduced vs D2 baseline
- hip_yaw_abs_max does not exceed D2 or official gate (0.10 rad)
- wheel velocity does not worsen
- contact/height/roll remain valid
- WBC gate pass
- hidden_torque_norm = 0
- ownership_violation_count = 0

## Alternative Approaches

### Option A: Modify tau_position gain scheduling
- Reduce position gain when hip_yaw is high
- Increase position gain when hip_yaw is low
- Risk: May reduce position correction when needed most

### Option B: Hip-yaw-aware position correction
- Add hip_yaw feedforward to position correction
- Decouples position correction from yaw
- Risk: More complex, requires tuning

### Option C: Direct wheel velocity shaping
- Apply recenter force directly to wheel velocity command
- Bypasses tau_position entirely
- Risk: May conflict with balance command

## Recommendation

**F1_phase_aware_recenter_velocity_shaping** (Option C variant) is recommended because:
1. It decouples recentering from tau_position
2. It does NOT affect hip yaw coupling
3. It only activates in safe phases
4. It is bounded and smooth

This breaks the feedback loop that E2/E2b could not solve.