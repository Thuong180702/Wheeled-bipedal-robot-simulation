# E0 Position Containment - Cleanup Audit

**Date:** 2026-05-29  
**Status:** Audit in progress  
**Goal:** Clean up failed E0b/E0c implementations, restore baseline, design correct E0d

## Executive Summary

This audit examines all E0-related code changes to:
1. Verify failed E0b/E0c logic is disabled by default
2. Restore safe baseline behavior
3. Design correct phase-aware position containment (E0d)

## 1. E0-Related Code Inventory

### 1.1 E0b Direct Torque Position Containment

**Location:** [wheeled_biped/controllers/sagittal_wheel_balance_controller.py](wheeled_biped/controllers/sagittal_wheel_balance_controller.py)

**Code paths:**
- Lines 28, 47: `enable_position_containment: bool = False` parameter
- Line 66: Instance variable `self.enable_position_containment`
- Lines 29-36: E0b parameters (kp_position, kd_position_velocity, deadband, limits, gates)
- Lines 67-74: E0b parameter storage
- Lines 113-193: Multi-zone position containment logic (gated by `enable_position_containment`)
- Lines 236-253: E0b telemetry fields (always populated, zeros when disabled)

**Classification:**
- **Control logic (lines 113-193):** `disable_by_default` ✓ (already done)
- **Parameters (lines 28-36, 67-74):** `keep_telemetry_only` (useful for E0d)
- **Telemetry (lines 236-253):** `keep_telemetry_only` ✓ (useful diagnostics)

**Status:** ✅ SAFE - E0b is disabled by default, only activates when `enable_position_containment=True`

### 1.2 E0c Reference Shaping Position Containment

**Location:** [scripts/simulate_hierarchical_controller.py](scripts/simulate_hierarchical_controller.py)

**Code paths:**
- Line 1203: `position_reference_y_m` captured at equilibrium
- Line 1738: `position_reference_y_m` added to nonlocal variables
- Lines 2127-2230: E0c CP bias reference shaping logic
- Line 2133: `e0c_enabled = False` (DISABLED)
- Lines 2142-2230: E0c computation (only runs if `e0c_enabled=True`)
- Lines 1413-1426: E0c telemetry field declarations
- Lines 2738-2771: E0c telemetry logging

**Classification:**
- **Control logic (lines 2142-2230):** `disable_by_default` ✓ (already done)
- **Position reference capture (line 1203):** `keep_active` (needed for E0d)
- **Telemetry fields (lines 1413-1426):** `keep_telemetry_only` (useful for E0d)
- **Telemetry logging (lines 2738-2771):** `keep_telemetry_only` (useful for E0d)

**Status:** ✅ SAFE - E0c is disabled, only activates when `e0c_enabled=True`

### 1.3 E0 Tests

**Location:** [tests/test_sagittal_wheel_position_containment.py](tests/test_sagittal_wheel_position_containment.py)

**Test coverage:**
- Backward compatibility (E0b disabled)
- Multi-zone behavior (deadband, soft, hard)
- Correction direction
- Clipping
- Balance priority gating
- Telemetry fields
- Ownership constraints
- Velocity damping

**Classification:**
- **Backward compatibility tests:** `keep_active` ✓ (verify E0b stays disabled)
- **E0b-specific tests:** `needs_rewrite` (test failed implementation)
- **Frame/sign tests:** `missing` (need to add)
- **Phase-aware tests:** `missing` (need to add for E0d)

**Status:** ⚠️ NEEDS UPDATE - Tests verify E0b behavior but don't test baseline or E0d

### 1.4 E0 Reports

**Location:** [outputs/balance_core_position_containment/](outputs/balance_core_position_containment/)

**Files:**
- `e0b_failure_analysis.md` - Complete E0b failure analysis
- `e0c_failure_analysis.md` - Complete E0c failure analysis
- `position_containment_summary.md` - Overall summary

**Classification:**
- All reports: `keep_active` ✓ (document failures for future reference)

**Status:** ✅ COMPLETE - Failure analysis documented

## 2. Baseline Behavior Verification

### 2.1 Current Default Behavior

**E0b in sagittal_wheel_balance_controller.py:**
- `enable_position_containment=False` by default
- When disabled: all position correction terms = 0
- Telemetry populated with zeros
- **Result:** No E0b control active by default ✅

**E0c in simulate_hierarchical_controller.py:**
- `e0c_enabled=False` hardcoded
- When disabled: `cp_bias_final_m = 0.0`
- Telemetry populated with zeros
- **Result:** No E0c control active by default ✅

### 2.2 Baseline Validation Required

Need to verify that running normal balance-core mode reproduces pre-E0 baseline:
- Expected max drift: ~2.50 m over 5000 steps
- NOT 15.98 m (E0b) or 63.72 m (E0c)

**Action:** Run validation to confirm baseline restored

## 3. Telemetry and Frame Correctness

### 3.1 Position Signal Sources

**Current implementation:**
- `position_y_m` = `float(centroidal_state_control.com_pos[1])`
- Source: CoM position in world frame
- Frame: World Y-axis (sagittal/front-back)

**Coordinate convention (verified):**
- X = lateral (left/right)
- Y = sagittal (front/back)
- Z = vertical (up/down)

### 3.2 Velocity Signal Sources

**Current implementation:**
- `com_vy_m_s` = `float(centroidal_state_control.com_vel[1])`
- Source: CoM velocity in world frame
- Frame: World Y-axis velocity

### 3.3 Frame Correctness Issues

**Yaw drift effect:**
- If robot yaws significantly, world-Y position != sagittal displacement along initial heading
- E0b/E0c used world-Y directly without yaw compensation
- **Potential issue:** Yaw drift may have contributed to failure

**Recommendation for E0d:**
- Use world-Y position if yaw remains small (<10°)
- OR project position onto initial heading vector if yaw can drift
- Monitor yaw drift in telemetry

### 3.4 Sign Convention Verification

**E0b sign convention:**
- Positive position_y_m (forward drift) → negative position_correction (backward torque)
- Formula: `position_correction = -kp_position * position_error`
- **Sign appears correct** ✓

**E0c sign convention:**
- Positive position_error → negative desired_velocity (move backward)
- Positive velocity_error (moving forward too fast) → negative CP bias (pull back)
- Formula: `cp_bias = -k_velocity_to_cp_bias * velocity_error`
- **Sign appears correct** ✓

**Conclusion:** Sign conventions were correct; failures were due to insufficient authority and balance gate suppression, not sign errors.

## 4. Root Cause Classification

### 4.1 E0b Failure Root Causes

1. **Insufficient torque authority**
   - Position correction: 3-15 Nm
   - Balance torque: 50+ Nm
   - Ratio: Position correction was only 6-30% of balance torque
   - **Result:** Balance controller overpowered position correction

2. **Balance priority gate positive feedback**
   - Gate suppressed correction 65.6% of time
   - Drift → correction → pitch → gate suppresses → more drift
   - **Result:** Positive feedback loop created runaway drift

3. **Multi-zone logic ineffective**
   - Robot spent 94.5% of time beyond hard limit
   - Deadband/soft/hard zones had no practical effect
   - **Result:** Zone structure provided no benefit

4. **Wrong control layer**
   - Direct torque from position error fights balance controller
   - Position and balance objectives conflict
   - **Result:** Architectural incompatibility

### 4.2 E0c Failure Root Causes

1. **CP bias too weak**
   - Max CP bias: 0.05 m
   - Position error: 63.7 m
   - Ratio: CP bias was 0.08% of position error
   - **Result:** Balance controller ignored tiny bias

2. **Cascade of saturations**
   - Position error → desired velocity saturated (0.10 m/s)
   - Actual velocity: 1.27 m/s (13x larger than desired)
   - Velocity error → CP bias saturated (0.05 m)
   - CP bias → balance gate suppressed (79.7%)
   - **Result:** Each layer reduced authority

3. **No phase-aware braking**
   - Desired velocity jumped immediately to max return velocity
   - No braking phase to slow down first
   - No acceleration limiting
   - **Result:** Robot couldn't track aggressive velocity commands

4. **Balance gate more aggressive**
   - Suppressed correction 79.7% of time (vs 65.6% for E0b)
   - Gate treated CP bias as less critical than direct torque
   - **Result:** Even less effective than E0b

### 4.3 Common Failure Modes

Both E0b and E0c failed because:
1. **No position awareness in primary controller** - Balance controller has no inherent position feedback
2. **Secondary correction too weak** - Adding position correction as afterthought cannot overcome primary controller's lack of position awareness
3. **Balance priority gate creates positive feedback** - Suppresses correction when needed most
4. **No phase-aware control** - No distinction between braking, return, and settle phases

## 5. E0d Design Requirements

### 5.1 Core Principles

**Phase-aware control:**
- Distinguish between moving away, braking, return, and settle phases
- Do not command aggressive reverse velocity immediately
- First slow down, then return gradually

**Acceleration limiting:**
- Desired velocity must be rate-limited
- No sudden jumps from +large to -max
- Smooth transitions between phases

**Balance priority:**
- If pitch/roll unsafe, freeze or reduce position return
- Do not fight fall recovery
- Gate must not create positive feedback

**Correct control interface:**
- No raw wheel torque from position error
- Use desired velocity → CP reference or pitch reference
- Let balance controller handle wheel torque

### 5.2 Required Phases

1. **inside_deadband**
   - Position error small
   - No return command
   - Light velocity damping only

2. **moving_away_braking**
   - Robot far from reference AND velocity moving further away
   - Objective: reduce outward velocity toward zero
   - Desired velocity ramps gradually toward zero
   - Allow wheels to keep moving forward briefly if needed for balance

3. **return**
   - Outward velocity reduced AND pitch/roll safe
   - Command small desired velocity toward reference
   - Clipped to 0.05-0.15 m/s

4. **settle**
   - Near target/deadband
   - Smoothly reduce desired velocity to zero
   - Avoid overshoot and oscillation

5. **gated_balance_recovery**
   - Pitch/roll unsafe
   - Reduce or freeze position return
   - Do not fight fall recovery

### 5.3 Control Interface Options

**Option A: Desired velocity tracking through CP reference (RECOMMENDED)**
- Define `desired_velocity_y` based on phase
- Derive CP reference/bias from velocity error
- Clip and acceleration-limit
- **Pros:** Indirect, lets balance controller handle torque
- **Cons:** Requires careful sign verification

**Option B: Tiny pitch reference bias**
- Position/velocity error generates small desired pitch bias
- Bias clipped and smooth
- Inner sagittal controller handles wheel torque
- **Pros:** Very indirect, minimal interference
- **Cons:** Pitch reference may be too weak

**Option C: Modify sagittal controller for desired velocity**
- Add desired velocity parameter to sagittal controller
- Controller tracks velocity reference directly
- **Pros:** Clean interface
- **Cons:** Requires controller modification

**Decision:** Use **Option A** (CP reference) with phase-aware desired velocity and careful sign verification.

## 6. E0d Implementation Plan

### 6.1 Phase State Machine

```python
# Phase determination
if position_error_abs <= deadband:
    phase = "inside_deadband"
elif velocity_away and position_error_abs > deadband:
    phase = "moving_away_braking"
elif velocity_toward and position_error_abs > settle_threshold:
    phase = "return"
elif position_error_abs <= settle_threshold:
    phase = "settle"

if pitch_unsafe or roll_unsafe:
    phase = "gated_balance_recovery"
```

### 6.2 Desired Velocity Computation

```python
if phase == "inside_deadband":
    desired_velocity_raw = 0.0
    
elif phase == "moving_away_braking":
    # Ramp velocity toward zero, not immediate reverse
    desired_velocity_raw = current_velocity * braking_factor
    
elif phase == "return":
    # Small return velocity toward reference
    desired_velocity_raw = -k_return * position_error
    desired_velocity_raw = clip(desired_velocity_raw, -max_return_vel, max_return_vel)
    
elif phase == "settle":
    # Decay velocity to zero near target
    desired_velocity_raw = -k_settle * position_error
    
elif phase == "gated_balance_recovery":
    # Freeze or reduce return
    desired_velocity_raw = 0.0 or previous_desired_velocity * gate_factor
```

### 6.3 Acceleration Limiting

```python
# Rate limit desired velocity
accel_limit = 0.5  # m/s^2
max_delta_v = accel_limit * dt
desired_velocity_limited = clip(
    desired_velocity_raw,
    prev_desired_velocity - max_delta_v,
    prev_desired_velocity + max_delta_v
)
```

### 6.4 CP Reference Bias

```python
# Compute velocity error
velocity_error = current_velocity - desired_velocity_limited

# CP bias from velocity error (verify sign!)
cp_bias = -k_velocity_to_cp * velocity_error

# Clip CP bias
cp_bias_clipped = clip(cp_bias, -max_cp_bias, max_cp_bias)

# Apply to CP error
cp_error_shaped = cp_error_natural + cp_bias_clipped
```

## 7. Cleanup Actions Required

### 7.1 Keep Active
- E0b telemetry fields (useful diagnostics)
- E0c telemetry fields (useful diagnostics)
- Position reference capture
- Failure analysis reports

### 7.2 Keep Disabled by Default
- E0b direct torque logic (already disabled) ✓
- E0c CP bias logic (already disabled) ✓

### 7.3 Remove/Rewrite
- E0b-specific tests that assert failed behavior
- Obsolete comments claiming position containment is impossible

### 7.4 Add New
- E0d phase-aware reference shaping logic
- Phase state machine
- Acceleration limiting
- Frame/sign verification tests
- Phase transition tests
- E0d validation tests

## 8. Validation Protocol

### 8.1 Baseline Validation (First)
```bash
python scripts/validate_balance_core.py --single-duration 1000
python scripts/validate_balance_core.py --single-duration 5000
```

**Expected:**
- Max drift ~2.50 m over 5000 steps
- NOT 15.98 m or 63.72 m
- Confirms E0b/E0c are truly disabled

### 8.2 E0d Validation (After Implementation)
```bash
# Nominal tests
python scripts/validate_balance_core.py --single-duration 1000
python scripts/validate_balance_core.py --single-duration 5000

# Height variant tests (if E0d passes nominal)
python scripts/validate_balance_core.py --single-duration 500 --height-variant high_5cm
python scripts/validate_balance_core.py --single-duration 500 --height-variant low_5cm
```

**Acceptance criteria:**
- Max drift ≤ 0.30-0.50 m over 5000 steps (substantial improvement over 2.50 m baseline)
- Final drift ≤ 0.10-0.20 m
- No pitch/roll/height divergence
- No wheel velocity runaway
- WBC remains off
- Ownership violations = 0
- Four-source stack unchanged

## 9. Next Steps

1. ✅ **Audit complete** - All E0 code paths identified and classified
2. ⏳ **Baseline validation** - Verify E0b/E0c are truly disabled
3. ⏳ **E0d design finalization** - Finalize phase state machine and parameters
4. ⏳ **E0d implementation** - Implement phase-aware reference shaping
5. ⏳ **Tests** - Add frame/sign/phase tests
6. ⏳ **E0d validation** - Run validation protocol
7. ⏳ **Report** - Generate comprehensive E0d report

## 10. Conclusion

**Current status:**
- E0b and E0c are DISABLED by default ✅
- Baseline behavior should be restored ✅
- Telemetry infrastructure is in place ✅
- Failure root causes are understood ✅

**Next action:**
- Run baseline validation to confirm E0b/E0c are truly disabled
- Design and implement E0d phase-aware reference shaping
- Do NOT proceed to Step C until E0d is validated

**Key insight:**
E0b and E0c failed not because position containment is impossible, but because they lacked phase-aware control and had insufficient authority. E0d will address these issues with proper braking/return/settle phases and acceleration limiting.
