# APCR Torque Composition Code Audit Report

## Executive Summary

**Classification: `TORQUE_COMPOSITION_APCR_CANCELLED_BY_PITCH_TORQUE`**

The root cause is now identified: during RECENTER state, the baseline `tau_pitch` term (designed for PITCH stabilization) produces torque in the OPPOSITE direction of what drift correction requires. The APCR applies correct torque, but `tau_pitch` dominates and overwhelms it.

## Torque Composition Formula

From `sagittal_velocity_damped_balance_controller.py`:

```python
tau_common_unclipped = (
    tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
    tau_support_velocity + tau_position + tau_cp + tau_com_vy
)
tau_common_unclipped = tau_common_unclipped + recenter_tau_clipped
tau_common_unclipped = tau_common_unclipped + hyst_tau_clipped
tau_common_unclipped = tau_common_unclipped + bias_tau_clipped
tau_common_unclipped = tau_common_unclipped + apc_tau_clipped

tau_common = self.wheel_torque_sign * tau_common_unclipped
tau_left = tau_common + tau_wheel_vel_left
tau_right = tau_common + tau_wheel_vel_right
```

Telemetry fields:
- `final_wheel_tau_with_apc`: `tau_common_unclipped + wheel_vel_avg` (includes APCR)
- `final_wheel_tau_without_apc`: `tau_common_unclipped - apc_tau_clipped + wheel_vel_avg` (without APCR)

## Component Analysis for Positive Drift (e > 0)

| Component | Mean Torque (Nm) | Sign | Effect |
|-----------|------------------|------|--------|
| tau_pitch | +4.62 | POSITIVE | WRONG - accelerates drift |
| tau_pitch_rate | +0.21 | POSITIVE | Minor |
| tau_position | -2.87 | NEGATIVE | CORRECT - opposes drift |
| tau_sagittal_velocity | -0.003 | ~0 | Neutral |
| tau_wheel_velocity | +2.30 | POSITIVE | Compensates for wheel motion |
| APCR | -1.89 | NEGATIVE | CORRECT - opposes drift |
| **Baseline Sum** | **+1.95** | POSITIVE | WRONG |

## The Pitch Torque Problem

### Physics of tau_pitch

```
tau_pitch = kp_pitch * pitch_x_rad
```

With `kp_pitch = 50.0` and `wheel_torque_sign = 1.0`:

- **Positive pitch** (nose up, leaning backward) → `tau_pitch = POSITIVE`
- **Positive tau_pitch** → wheels spin forward → robot accelerates forward

### During Drift Correction

When drift is positive (CoM past support):
1. Robot INTENTIONALLY leans back (positive pitch) to correct
2. `tau_pitch` computes POSITIVE torque
3. This ACCELERATES forward motion, fighting the correction

### tau_position vs tau_pitch Conflict

```
tau_position = -k_position * sagittal_position_error_m
```

- For positive drift: `tau_position = NEGATIVE` (correct)
- For positive pitch: `tau_pitch = POSITIVE` (wrong)

These two terms fight each other. During APCR:
- `tau_position` = -2.87 Nm
- `tau_pitch` = +4.62 Nm
- Net baseline = +1.95 Nm (wrong direction)

## Telemetry Verification

### Step 50 Example

```
pitch_x = +0.047 rad (leaning back)
position_error = +0.062 m (positive drift)
wheel_vel = -4.60 rad/s (braking)

tau_pitch = +2.35 Nm  ← WRONG direction
tau_position = -2.48 Nm  ← CORRECT direction
tau_wheel_vel = +2.30 Nm (braking compensation)
APCR = -0.76 Nm (correct)

Baseline (without APCR) = +2.35 - 2.48 + 2.30 + ... = +0.51 Nm  ← WRONG
Final (with APCR) = +0.51 - 0.76 = -0.25 Nm  ← STILL mostly wrong
```

## Why This Wasn't Caught Earlier

1. **tau_pitch is correct for PITCH stabilization**: When robot is falling forward (negative pitch), tau_pitch produces negative torque to catch it. This is correct behavior.

2. **tau_pitch is WRONG for DRIFT correction**: When robot is leaning back intentionally (positive pitch) to correct drift, tau_pitch produces positive torque that fights the correction.

3. **APCR was designed to CORRECT this**: APCR correctly applies opposing torque, but:
   - APCR max torque = 2.0 Nm
   - tau_pitch during drift = 4.62 Nm (mean)
   - APCR cannot overcome tau_pitch

## Root Cause

The `tau_pitch` term assumes that ANY pitch deviation requires wheel torque to bring pitch back to zero. But during RECENTER state, intentional pitch deviation (leaning to correct drift) should NOT produce pitch-correcting wheel torque.

The controller does NOT suppress tau_pitch during RECENTER state.

## Classification

```
TORQUE_COMPOSITION_APCR_CANCELLED_BY_PITCH_TORQUE
```

Not exactly "baseline overpowers APCR" - it's more specifically "tau_pitch overpowers the combination of APCR + tau_position".

## Fix Options

### Option 1: Suppress tau_pitch During RECENTER (Recommended)

During RECENTER state, set `tau_pitch = 0` to let APCR and tau_position handle drift correction without interference.

```python
if apc_hysteresis_active:
    tau_pitch = 0.0
```

### Option 2: Reduce tau_pitch Gain During RECENTER

Scale down tau_pitch contribution during RECENTER state.

### Option 3: Make tau_pitch Aware of RECENTER State

Add a flag that tells tau_pitch whether the pitch is intentional (correction) or unintentional (falling).

## Recommended Approach

**Option 1 with telemetry**: Suppress tau_pitch during RECENTER state, with clear telemetry showing the suppression and its effect on final torque direction.
