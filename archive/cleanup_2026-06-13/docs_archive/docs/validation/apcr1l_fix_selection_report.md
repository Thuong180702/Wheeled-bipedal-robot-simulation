# APCR1l Fix Selection Report

## Phase 1 Summary: Torque Sign Convention Audit

**Classification: `APCR_SIGN_CORRECT_BUT_PITCH_TORQUE_FIGHTS_CORRECTION`**

| Finding | Value |
|---------|-------|
| APCR contribution sign matches command | 715/715 (100%) |
| Final torque opposes drift | 58/715 (8.1%) |
| Final torque accelerates drift | 657/715 (91.9%) |

APCR is NOT sign-inverted. APCR contribution is mathematically correct.

## Phase 2 Summary: Torque Composition Code Audit

**Classification: `TORQUE_COMPOSITION_APCR_CANCELLED_BY_PITCH_TORQUE`**

| Component | Mean Torque (Nm) | Sign | Effect During Positive Drift |
|-----------|------------------|------|------------------------------|
| tau_pitch | +4.62 | POSITIVE | **WRONG** - accelerates drift |
| tau_position | -2.87 | NEGATIVE | CORRECT - opposes drift |
| APCR | -1.89 | NEGATIVE | CORRECT - opposes drift |
| **Net Baseline** | **+1.95** | POSITIVE | **WRONG** |

### Root Cause

During RECENTER state:
1. Robot intentionally leans back (positive pitch) to correct positive drift
2. `tau_pitch = +4.62 Nm` (intended for pitch stabilization, but fights correction)
3. `tau_position = -2.87 Nm` (correctly opposes drift)
4. `APCR = -1.89 Nm` (correctly opposes drift)
5. Net = +1.95 Nm (tau_pitch dominates)

The `tau_pitch` term assumes ANY pitch deviation requires correction torque. But during RECENTER, intentional pitch deviation (correction lean) should NOT produce pitch-correcting wheel torque.

## Fix Selection

### Chosen Fix: **Fix A' - Suppress tau_pitch During RECENTER**

This is a variant of Fix A (sign inversion fix), but instead of flipping APCR sign (wrong), we suppress tau_pitch during RECENTER state.

**Rationale:**
- tau_pitch sign IS correct for pitch stabilization (falling recovery)
- tau_pitch sign IS WRONG for drift correction (fights RECENTER)
- APCR sign IS correct
- Solution: Suppress tau_pitch during RECENTER to let APCR + tau_position correct drift

**Profile name:** `APCR1l_pitch_suppress_recenter`

### Implementation

During `apc_hysteresis_active == True`:
```python
tau_pitch_suppressed = 0.0  # or scaled down
```

Keep:
- APCR torque unchanged (correct sign)
- tau_position unchanged (correct direction)
- tau_pitch_rate, tau_sagittal_velocity, tau_wheel_velocity unchanged

### Why NOT Other Fixes

**Fix A (flip APCR sign):** INCORRECT. APCR sign is correct. Flipping would make it worse.

**Fix B (suppress baseline torque):** PARTIAL. The issue is specifically tau_pitch, not all baseline torque. tau_position is correct.

**Fix C (actuator sign):** INCORRECT. Actuator sign is correct.

**Fix D (e_dot not responding):** N/A. Torque direction is the issue, not dynamics.

## Decision

```
APCR_SIGN_CORRECT_BUT_PITCH_TORQUE_FIGHTS_CORRECTION
TORQUE_COMPOSITION_APCR_CANCELLED_BY_PITCH_TORQUE
Fix: Suppress tau_pitch during RECENTER state
```
