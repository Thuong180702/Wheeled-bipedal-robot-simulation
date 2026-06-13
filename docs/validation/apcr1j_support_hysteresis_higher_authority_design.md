# APCR1j_support_hysteresis_higher_authority Design

## Overview

APCR1j is a new opt-in sagittal authority profile based on APCR1i_support_hysteresis_recenter, with explicit higher torque authority to overcome the 1.5 Nm universal cap that limits APCR1i.

## Root Cause Analysis

APCR1i correctly implements the support hysteresis recenter principle, but its configured recenter authority is not reaching final APCR torque.

**APCR1i diagnostic result:**
- survived 1000 steps
- max_e = +0.2550 m
- min_e = -0.0873 m
- P2P = 0.3424 m
- outside ±0.15 = 29.6%
- target max_e < +0.15 m
- observed APCR tau max = 1.5000 Nm
- configured recenter_max_tau = 1.75 Nm
- configured emergency_max_tau = 2.00 Nm

**Root cause:** Downstream `apc_max_cross_tau = 1.5` universal clip overrides APCR1i hysteresis authority.

Location in sagittal_velocity_damped_balance_controller.py around line 2262-2266:
```python
apc_raw_tau = float(jnp.clip(
    apc_raw_tau,
    -self.authority_schedule.apc_max_cross_tau,
    self.authority_schedule.apc_max_cross_tau
))
```

This uses `apc_max_cross_tau`, which remains 1.5 Nm for APCR1i because it doesn't explicitly override it.

## APCR1j Solution

APCR1j explicitly sets `apc_max_cross_tau = 2.0` so the hysteresis recenter can reach 2.0 Nm without being capped at 1.5 Nm.

## Configuration Differences from APCR1i

| Parameter | APCR1i | APCR1j | Change |
|-----------|--------|--------|--------|
| apc_max_cross_tau | 1.5 (default) | 2.0 | **+0.5 Nm** - Critical fix |
| apc_hysteresis_recenter_max_tau | 1.75 | 2.0 | +0.25 Nm |
| apc_hysteresis_emergency_max_tau | 2.00 | 2.2 | +0.2 Nm |
| apc_hysteresis_hold_max_tau | 1.50 | 1.75 | +0.25 Nm |
| apc_hysteresis_normal_rate | 0.30 | 0.40 | +0.1 Nm/step |
| apc_hysteresis_recenter_rate | 0.90 | 1.1 | +0.2 Nm/step |
| apc_hysteresis_emergency_rate | 1.00 | 1.3 | +0.3 Nm/step |

## Hysteresis Thresholds (unchanged from APCR1i)

| Parameter | Value |
|-----------|-------|
| apc_hysteresis_outer_enter_m | 0.08 m |
| apc_hysteresis_inner_exit_m | 0.03 m |
| apc_hysteresis_opposite_release_m | 0.03 m |
| apc_hysteresis_near_zero_m | 0.01 m |
| apc_hysteresis_emergency_m | 0.12 m |
| apc_hysteresis_hard_m | 0.15 m |

## Safety Behavior (unchanged from APCR1i)

- Support recenter should not exit because pitch is balanced
- Support recenter should not exit because pitch sign changes
- Support recenter exits only at inner band, opposite release, or hard safety
- Pitch danger still blocks APCR activation
- Contact, height, roll gates still enforced

## State Machine (unchanged from APCR1i)

APCR1j uses the same 4-state hysteresis state machine as APCR1i:
1. NEUTRAL - no recenter active
2. RECENTER_FROM_POSITIVE - applying negative torque to reduce positive drift
3. RECENTER_FROM_NEGATIVE - applying positive torque to reduce negative drift
4. HOLD_THROUGH_ZERO - holding through zero crossing until inside inner band

## Telemetry Fields

All APCR1i telemetry fields are preserved:
- active_pitch_crossing_hysteresis_state
- active_pitch_crossing_hysteresis_state_id
- active_pitch_crossing_hysteresis_entry_e
- active_pitch_crossing_hysteresis_exit_e
- active_pitch_crossing_hysteresis_entry_count
- active_pitch_crossing_hysteresis_exit_count
- active_pitch_crossing_hysteresis_inner_exit_m
- active_pitch_crossing_hysteresis_opposite_release_m
- active_pitch_crossing_hysteresis_emergency_active
- active_pitch_crossing_max_tau (now shows 2.0 for APCR1j)

## Expected Behavior

1. APCR1j can produce APCR tau magnitude up to 2.0 Nm (vs 1.5 Nm for APCR1i)
2. APCR1j can reach selected_tau_limit = 2.0 or 2.2 Nm
3. APCR1j should reduce drift compared to APCR1i
4. APCR1j should have fewer torque clipping events than APCR1i
5. APCR1j should have faster e_dot reversal than APCR1i

## Target Metrics

- survive 1000 steps
- max_e < APCR1i (0.2550 m) and preferably < APCR1h (0.1572 m)
- target max_e < 0.15 m
- P2P < APCR1i (0.3424 m) and preferably <= APCR1h
- outside ±0.15 < APCR1i (29.6%) and preferably <= APCR1h (2.6%)
- outside ±0.10 lower than APCR1i
- contact/height/roll stable
- pitch/hip-yaw/wheel velocity monitored but not primary blockers unless unstable

## Restrictions

- Do NOT modify APCR1i
- Do NOT modify APCR1h/APCR1f
- Do NOT run 2000-step until APCR1j improves at 1000-step
- Do NOT run 5000-step
- Do NOT run Step C
- Do NOT run Step D
- Do NOT commit
