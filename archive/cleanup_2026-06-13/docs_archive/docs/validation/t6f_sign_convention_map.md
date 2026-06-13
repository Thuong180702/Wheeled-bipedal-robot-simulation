# T6F Torque Sign Convention Map

**Date:** 2026-06-12  
**Task:** Phase 1 of torque sign bug fix investigation  
**Purpose:** Document complete sign convention for sagittal support drift controller

---

## Executive Summary

**Physical Drift Sign Convention:**
- `sagittal_position_error_m > 0` → Robot support is **forward** of reference
- `sagittal_position_error_m < 0` → Robot support is **backward** of reference

**Expected Corrective Torque Sign:**
- To correct **positive** drift (+forward), wheels must apply **negative** torque (backward acceleration)
- To correct **negative** drift (-backward), wheels must apply **positive** torque (forward acceleration)

**Sign Rule:** `sign(tau_correction) * sign(drift_error)` should be **NEGATIVE** (opposite signs)

---

## Signal Trace: Physical Drift Error

### 1. Primary Drift Signal

**Column:** `active_pitch_crossing_signed_error_m`

**Computation Path:**
```
support_position_error_m (input to controller)
→ sagittal_position_error_m (same value, different name in different contexts)
→ active_pitch_crossing_signed_error_m (telemetry column)
```

**Sign Convention:**
- Positive (+) = Robot support forward of target
- Negative (-) = Robot support backward of target
- Zero (0) = At target position

**Source:** Computed in sim/environment from support center position vs reference

**Usage:** Input to both:
1. `tau_position` computation
2. Active Pitch Crossing Recovery (APCR) logic

---

## Signal Trace: Position Torque

### 2. tau_position Computation

**Line 1964:**
```python
tau_position_p = -effective_k_position * sagittal_position_error_m
```

**Sign Convention Check:**
- If `sagittal_position_error_m = +0.10` (forward drift):
  - `tau_position_p = -effective_k_position * (+0.10) = NEGATIVE`
  - Expected: NEGATIVE torque to push backward ✅ CORRECT
  
- If `sagittal_position_error_m = -0.10` (backward drift):
  - `tau_position_p = -effective_k_position * (-0.10) = POSITIVE`
  - Expected: POSITIVE torque to push forward ✅ CORRECT

**Verdict:** `tau_position_p` sign convention is **CORRECT** ✅

---

### 3. tau_position After Clipping

**Lines 2173:**
```python
tau_position = float(jnp.clip(tau_position_before_clip, -effective_max_position_tau, effective_max_position_tau))
```

**T6F Architecture Fix Path (Lines 2105-2171):**
```python
if self.authority_schedule.arch_fix_enabled:
    # Gate checks...
    if all_gates_pass:
        if in_emergency_band:
            arch_fix_requested_cap = self.authority_schedule.arch_fix_emergency_max_position_tau  # 7.0
        elif in_hard_band:
            arch_fix_requested_cap = self.authority_schedule.arch_fix_hard_max_position_tau  # 6.5
        
        effective_max_position_tau = max(
            float(effective_max_position_tau),
            float(arch_fix_requested_cap)
        )
```

**Sign Convention Check:**
- `tau_position` inherits sign from `tau_position_before_clip`
- Clipping preserves sign, only limits magnitude
- Architecture fix raises cap magnitude but **does NOT flip sign**

**Verdict:** `tau_position` after clip preserves sign convention ✅

---

## Signal Trace: Wheel Velocity Damping

### 4. tau_wheel_velocity

**Lines 1847-1848:**
```python
tau_wheel_vel_left = -effective_k_wheel_velocity * wheel_vel_left_rad_s
tau_wheel_vel_right = -effective_k_wheel_velocity * wheel_vel_right_rad_s
```

**Sign Convention Check:**
- If `wheel_vel_left_rad_s = +5.0` (wheel spinning forward):
  - `tau_wheel_vel_left = -effective_k_wheel_velocity * (+5.0) = NEGATIVE`
  - Expected: NEGATIVE torque to slow down forward spin ✅ CORRECT

- If `wheel_vel_left_rad_s = -5.0` (wheel spinning backward):
  - `tau_wheel_vel_left = -effective_k_wheel_velocity * (-5.0) = POSITIVE`
  - Expected: POSITIVE torque to slow down backward spin ✅ CORRECT

**Verdict:** Wheel velocity damping sign convention is **CORRECT** ✅

**Interaction with Drift Correction:**
- When correcting forward drift (+), `tau_position` is NEGATIVE
- If wheels already spinning forward (+), damping is also NEGATIVE
- Both terms push in same direction (backward) → **HELP** ✅
- If wheels spinning backward (-), damping is POSITIVE
- Damping opposes `tau_position` → **FIGHT** ❌
  - This is handled by APCR1n wheel damping override logic

---

## Signal Trace: APCR1n Wheel Damping Override

### 5. APCR1n Damping Logic

**Purpose:** Scale down wheel damping when it fights drift correction

**Lines 2363-2440 (not shown but documented):**

APCR1n checks:
```python
if recenter_priority_active and vd_wheel_damping_recenter_override_enabled:
    # Check if damping opposes position torque
    sign_position = sign(tau_position)
    sign_damping_mean = sign((tau_wheel_vel_left + tau_wheel_vel_right) / 2.0)
    
    if sign_position * sign_damping_mean > 0:
        # Same sign → damping helps drift correction
        preserve_damping = True
    else:
        # Opposite sign → damping fights drift correction
        scale_damping_to_30_percent = True
```

**Sign Convention Impact:**
- This logic depends on `tau_position` having correct sign
- If `tau_position` sign is wrong, the "helps vs fights" detection **INVERTS**

**Verdict:** APCR1n damping override is **DEPENDENT** on upstream sign correctness ⚠️

---

## Signal Trace: Final Wheel Torque Composition

### 6. tau_common_unclipped

**Lines 3661-3672:**
```python
tau_common_unclipped = (
    tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
    tau_support_velocity + tau_position + tau_cp + tau_com_vy
)
tau_common_unclipped = tau_common_unclipped + recenter_tau_clipped
tau_common_unclipped = tau_common_unclipped + hyst_tau_clipped
tau_common_unclipped = tau_common_unclipped + bias_tau_clipped
tau_common_unclipped = tau_common_unclipped + apc_tau_clipped
```

**Sign Convention Check:**
- All terms are additive
- No sign flips in composition
- `tau_position` maintains its sign through composition

**Verdict:** Composition preserves sign convention ✅

---

### 7. tau_common with wheel_torque_sign

**Line 3674:**
```python
tau_common = self.wheel_torque_sign * tau_common_unclipped
```

**Critical Sign Multiplier:**
- `wheel_torque_sign` defaults to `+1.0`
- Can be set to `-1.0` for sign correction
- This is the **ONLY** global sign flip in the entire path

**Current Configuration:**
- T5: `wheel_torque_sign = +1.0` (default)
- T6F: `wheel_torque_sign = +1.0` (default, same as T5)

**Sign Convention Check:**
- If drift is forward (+0.10):
  - `tau_position` is NEGATIVE
  - `tau_common_unclipped` includes NEGATIVE tau_position
  - `tau_common = (+1.0) * (NEGATIVE) = NEGATIVE` ✅
  - Expected: NEGATIVE to correct forward drift ✅

- If this multiplier were wrong (e.g., accidentally flipped in high-torque regime):
  - `tau_common = (-1.0) * (NEGATIVE) = POSITIVE` ❌
  - Result: POSITIVE torque on forward drift → makes drift worse ❌

**Verdict:** Sign multiplier is **CRITICAL** and currently set correctly for T5/T6F baseline ✅

---

### 8. Final Per-Wheel Torque

**Lines 3677-3678:**
```python
tau_left = tau_common + tau_wheel_vel_left
tau_right = tau_common + tau_wheel_vel_right
```

**Sign Convention Check:**
- Both wheels receive same `tau_common` (symmetric sagittal control)
- Individual wheel damping is added separately
- No additional sign flips

**Verdict:** Final wheel torque preserves sign convention ✅

---

## Signal Trace: Active Pitch Crossing (APC) Torque

### 9. APC Torque Sign Convention

**APC State Machine (not shown in excerpts but documented):**

APCR uses hysteresis state machine with states:
- `NEUTRAL`
- `RECENTER_FROM_POSITIVE` (when error > +threshold)
- `RECENTER_FROM_NEGATIVE` (when error < -threshold)

**Expected APC Sign Rule:**
- In `RECENTER_FROM_POSITIVE` → apply NEGATIVE torque to reduce forward drift
- In `RECENTER_FROM_NEGATIVE` → apply POSITIVE torque to reduce backward drift

**APC Composition:**
```python
tau_common_unclipped = tau_common_unclipped + apc_tau_clipped
```

**Sign Convention Check:**
- `apc_tau_clipped` must oppose drift error
- If drift is positive (+), `apc_tau_clipped` should be NEGATIVE
- If drift is negative (-), `apc_tau_clipped` should be POSITIVE

**Verdict:** APC torque sign must oppose drift sign ⚠️ (needs separate audit)

---

## Signal Trace: Hip Yaw Compensation

### 10. Hip Yaw Support Feedforward (HY-FF)

**Lines 301-302 in shape_posture_controller.py:**
```python
tau_comp_left_raw = self.support_comp_sign * self.k_support_hip_yaw * support_position_error * height_gate
tau_comp_right_raw = -self.support_comp_sign * self.k_support_hip_yaw * support_position_error * height_gate
```

**Purpose:** Hip yaw compensation to counter yaw divergence caused by sagittal drift

**Sign Convention:**
- `support_position_error` = same as `sagittal_position_error_m`
- `support_comp_sign` = configuration parameter (default +1.0 or -1.0)
- Left and right hips receive **opposite** signs (antisymmetric)

**Impact on Wheel Torque:**
- Hip yaw torque does **NOT** directly affect wheel torque
- Hip yaw joints are [1, 6], wheel joints are [4, 9]
- These are separate control paths

**Verdict:** Hip yaw compensation does **NOT** directly affect wheel torque sign ✅

---

## Sign Convention Summary Table

| Signal | Expected Sign Rule | Current Implementation | Status |
|--------|-------------------|----------------------|--------|
| `sagittal_position_error_m` | + forward, - backward | ✅ Correct | ✅ |
| `tau_position_p` | Oppose error sign | `= -k * error` | ✅ |
| `tau_position` after clip | Preserve tau_position_p sign | Clip magnitude only | ✅ |
| `tau_wheel_vel_*` | Oppose wheel velocity | `= -k * wheel_vel` | ✅ |
| `tau_common_unclipped` | Sum all terms | Additive composition | ✅ |
| `wheel_torque_sign` | Global multiplier | Currently +1.0 | ✅ |
| `tau_common` | `wheel_torque_sign * tau_common_unclipped` | Correctly applied | ✅ |
| `tau_left/right` | `tau_common + damping` | Additive | ✅ |
| APCR `apc_tau_clipped` | Oppose drift error | **NEEDS AUDIT** | ⚠️ |
| Hip yaw compensation | Antisymmetric L/R | Does not affect wheels | ✅ |

---

## Potential Sign Bug Locations

### Location 1: APCR Torque Sign (HIGH PRIORITY)

**Hypothesis:** APCR torque may have wrong sign in certain conditions

**Evidence from root-cause report:**
- T6F torque opposes drift only 47.5% of time
- T6F APCR activates at high authority (7.0 Nm)
- Sign bug becomes visible at high magnitude

**Audit Required:**
- Check APCR state machine torque sign logic
- Verify `apc_tau_clipped` computation in all states
- Check if sign flips based on band state or error magnitude

**File:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
**Lines:** APCR state machine logic (not shown in excerpts, ~2500-3500)

---

### Location 2: Architecture Fix Cap Composition (MEDIUM PRIORITY)

**Hypothesis:** Raised cap might interact with existing logic incorrectly

**Evidence:**
- Sign bug present in both normal (4.0 Nm) and raised (7.0 Nm) regimes
- But degradation only severe at 7.0 Nm

**Audit Required:**
- Verify `effective_max_position_tau` does not flip sign
- Check if any conditional logic uses cap value as sign indicator
- Verify APCR1nD band state does not invert sign convention

**File:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
**Lines:** 2105-2171 (architecture fix), 2190-2344 (APCR1nD)

---

### Location 3: APCR1n Wheel Damping Override Sign Logic (LOW PRIORITY)

**Hypothesis:** Damping override may invert "helps vs fights" detection if upstream sign is wrong

**Evidence:**
- APCR1n damping override depends on `tau_position` sign
- If `tau_position` sign is wrong, override logic inverts
- Would amplify wrong-direction torque instead of scaling it down

**Audit Required:**
- Verify APCR1n correctly detects when damping opposes position torque
- Check if damping scale factor is applied with correct sign

**File:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
**Lines:** APCR1n logic (~2346-2600, not shown)

---

### Location 4: wheel_torque_sign Configuration (LOW PRIORITY)

**Hypothesis:** `wheel_torque_sign` might be inadvertently flipped for T6F

**Evidence:**
- Global sign multiplier at final composition
- If accidentally set to -1.0 for T6F, would flip all torque

**Audit Required:**
- Check T6F profile configuration
- Verify `wheel_torque_sign` parameter is not overridden

**File:** `scripts/simulate_hierarchical_controller.py` or profile config

---

## Physical Wheel Sign Convention

### Wheel Torque → Motion Direction

**Question:** Does positive wheel torque move support forward or backward?

**Answer from code:**
- Positive wheel velocity (`wheel_vel > 0`) means wheel spinning in direction that moves support **FORWARD**
- Wheel velocity damping applies **negative** torque to **oppose** positive velocity
- Therefore: **POSITIVE wheel torque** → **FORWARD motion**
- And: **NEGATIVE wheel torque** → **BACKWARD motion**

**Verification:**
- To correct forward drift (+error), need backward motion → NEGATIVE torque ✅
- To correct backward drift (-error), need forward motion → POSITIVE torque ✅

This matches the sign convention throughout the controller.

---

## Classification

**T6F_SIGN_CONVENTION_CLEAR** ⚠️ with caveats:

- Base `tau_position` sign convention is correct ✅
- Wheel velocity damping sign convention is correct ✅
- Composition path preserves signs correctly ✅
- `wheel_torque_sign` multiplier is correctly set to +1.0 ✅

**BUT:**
- **APCR torque sign is NOT verified** ⚠️
- APCR activates at high drift and high authority
- T6F evidence shows sign correctness degrades to 47.5% (random)
- High-authority APCR torque is the most likely culprit

**Next Step:** Audit APCR torque computation in all states and bands

---

## Recommended Action

**Priority 1:** Audit APCR torque sign logic in state machine

Focus on:
1. `RECENTER_FROM_POSITIVE` state → should produce NEGATIVE torque
2. `RECENTER_FROM_NEGATIVE` state → should produce POSITIVE torque
3. Emergency/hard band escalation → verify sign preserved
4. Architecture fix raised cap → verify no sign interaction

**If APCR sign is wrong:**
- Fix at APCR torque computation (earliest point in pipeline)
- Do NOT patch at `wheel_torque_sign` (would break T5)
- Do NOT patch at final composition (would affect all controllers)

---

**Status:** Sign convention map COMPLETE  
**Classification:** T6F_SIGN_CONVENTION_CLEAR with APCR audit required  
**Date:** 2026-06-12
