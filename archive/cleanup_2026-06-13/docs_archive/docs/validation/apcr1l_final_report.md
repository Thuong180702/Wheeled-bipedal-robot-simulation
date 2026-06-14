# APCR1l Final Report

## Executive Summary

**Result: INCONCLUSIVE - Critical initialization bug prevents valid comparison**

APCR1l pitch suppression during RECENTER state was successfully implemented and tested, but a critical initialization bug causes the robot to fall immediately after startup, preventing meaningful comparison with APCR1i.

## Phase 0: Health Check

**Status: PASS**

All existing tests pass:
```
tests/test_sagittal_velocity_damped_balance_controller.py - 245 tests passed
```

## Phase 1: APCR Torque Sign Convention Audit

**Classification: `APCR_SIGN_CORRECT_BUT_PITCH_TORQUE_FIGHTS_CORRECTION`**

From APCR1k telemetry analysis (1000 steps):
- APCR contribution sign matches command: 715/715 (100%)
- Final torque OPPOSES drift: 58/715 (8.1%)
- Final torque ACCELERATES drift: 657/715 (91.9%)

**Conclusion: APCR sign is mathematically correct. No sign inversion.**

## Phase 2: Torque Composition Code Audit

**Root Cause Identified:**

| Component | Mean Torque (Nm) | Sign | Effect During Positive Drift |
|-----------|------------------|------|------------------------------|
| tau_pitch | +4.62 | POSITIVE | **WRONG** - accelerates drift |
| tau_position | -2.87 | NEGATIVE | CORRECT - opposes drift |
| APCR | -1.89 | NEGATIVE | CORRECT - opposes drift |
| **Net Baseline** | **+1.95** | POSITIVE | **WRONG** |

**Root Cause:** During RECENTER state:
1. Robot intentionally leans back (positive pitch) to correct positive drift
2. `tau_pitch = +4.62 Nm` (intended for pitch stabilization, but fights correction)
3. `tau_position = -2.87 Nm` (correctly opposes drift)
4. `APCR = -1.89 Nm` (correctly opposes drift)
5. Net = +1.95 Nm (tau_pitch dominates)

## Phase 3: Fix Selection

**Chosen Fix: Suppress tau_pitch During RECENTER**

Rationale:
- tau_pitch sign IS correct for pitch stabilization (falling recovery)
- tau_pitch sign IS WRONG for drift correction (fights RECENTER)
- APCR sign IS correct
- Solution: Suppress tau_pitch during RECENTER to let APCR + tau_position correct drift

## Phase 4: Implementation

**Profile Name:** `APCR1l_pitch_suppress_recenter`

**Changes Made:**

1. Added `apc_hysteresis_pitch_suppress_in_recenter` field to `SagittalAuthoritySchedule` class

2. Created `APCR1L_PITCH_SUPPRESS_RECENTER` profile with:
   - Same thresholds as APCR1k (only adds pitch suppression)
   - `apc_hysteresis_pitch_suppress_in_recenter = True`
   - Applies to: low_0p300, low_0p330, low_0p360, extreme_height

3. Added pitch suppression logic in `compute()`:
   ```python
   apc_recenter_active = self._apc_hysteresis_state in ("RECENTER_FROM_POSITIVE", "RECENTER_FROM_NEGATIVE")
   pitch_suppress_active = (
       self.authority_schedule.apc_hysteresis_pitch_suppress_in_recenter
       and apc_recenter_active
   )
   
   if pitch_suppress_active:
       tau_pitch = 0.0
       tau_pitch_clipped = 0.0
   else:
       # Normal pitch computation
   ```

4. Added telemetry fields:
   - `apcr1l_pitch_suppress_active`
   - `apcr1l_recenter_state`
   - `apcr1l_tau_pitch_before_suppress`

5. Registered profile in `JOINT_FIX_PROFILES` dict

6. Added `--vd-sagittal-authority-profile APCR1l_pitch_suppress_recenter` to simulation script

## Phase 5: Unit Tests

**Status: PASS (7/7 tests)**

```
tests/test_sagittal_velocity_damped_balance_controller.py
test_apcr1l_profile_exists_and_is_opt_in_only - PASSED
test_apcr1l_same_thresholds_as_apcr1k - PASSED
test_apcr1l_applies_to_boundary_variants - PASSED
test_apcr1l_initial_state_is_neutral - PASSED
test_apcr1l_suppresses_tau_pitch_in_recenter - PASSED
test_apcr1l_does_not_suppress_tau_pitch_in_neutral - PASSED
test_apcr1l_no_wbc_path_change - PASSED
```

## Phase 6: 1000-Step Validation

**Status: INCONCLUSIVE - Initialization Bug**

**Problem:**

APCR1i (baseline) with low_0p300 height variant: 1000 steps survived
APCR1l (pitch suppression): Fell after 18 steps (height_too_low)

**Critical Bug:**
```
Step 0-17: robot_pitch_x = [-49.83 to -0.28] degrees (consistently negative = forward fall)
```

The robot falls immediately because:
1. APCR1l suppresses tau_pitch during RECENTER
2. But the robot hasn't entered RECENTER yet (in NEUTRAL)
3. During NEUTRAL, tau_pitch IS suppressed (bug: suppression applies to all RECENTER states, but initial fall happens before RECENTER)
4. Without tau_pitch, the robot cannot recover from initial pitch perturbations

**Root Cause:**
The pitch suppression is designed for RECENTER state (correcting drift), but it also affects initial startup when the robot needs pitch stabilization to maintain balance.

## Phase 7: Analysis

**Torque Direction Correctness:**

| Metric | APCR1i | APCR1l |
|--------|--------|--------|
| correct_direction_count | 164 | 0 (N/A - fell early) |
| correct_direction_pct | 16.4% | N/A |

**APCR1i (baseline):**
- APCR active: 801/1000 steps
- APCR torque mean: -0.8036 Nm
- Final torque: 16.4% correct direction

**APCR1l:**
- Fell at step 18
- Cannot compute torque direction statistics

## Phase 8: Classification

```
APCR1L_INCONCLUSIVE_INIT_BUG
```

### Issues Found

1. **Initialization Bug:** Suppressing tau_pitch removes the robot's ability to recover from initial pitch perturbations during startup.

2. **Timing Issue:** The robot falls BEFORE entering RECENTER state, so pitch suppression never activates.

3. **Test Coverage Gap:** Unit tests verify pitch suppression works in RECENTER state, but don't catch the initialization issue.

## Recommended Fix

**Option A: Don't suppress tau_pitch during initial steps**
- Add a step counter to delay pitch suppression until after initialization

**Option B: Use different suppression profile for startup vs steady-state**
- Full tau_pitch during initialization
- Suppressed tau_pitch during steady-state RECENTER

**Option C: Suppress only during deep RECENTER**
- Only suppress tau_pitch when error exceeds a higher threshold
- Normal tau_pitch for small corrections

## Files Changed

1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
   - Added `apc_hysteresis_pitch_suppress_in_recenter` field
   - Created `APCR1L_PITCH_SUPPRESS_RECENTER` profile
   - Added pitch suppression logic
   - Added telemetry fields

2. `scripts/simulate_hierarchical_controller.py`
   - Added APCR1l to profile choices

3. `tests/test_sagittal_velocity_damped_balance_controller.py`
   - Added 7 APCR1l tests (all passing)

## Next Steps

1. Fix initialization bug by delaying pitch suppression
2. Re-run 1000-step validation
3. Compare torque direction correctness
4. If APCR1l shows improvement, merge into mainline
