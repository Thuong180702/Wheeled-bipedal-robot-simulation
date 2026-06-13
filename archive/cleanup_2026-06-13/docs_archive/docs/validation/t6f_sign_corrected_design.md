# T6F_sign_corrected Design Document

**Date:** 2026-06-12  
**Phase:** 4 of 8 (Design sign fix candidate)  
**Status:** Phase 4 COMPLETE  
**Classification:** T6F_SIGN_CORRECTED_DESIGN_READY

---

## Executive Summary

**T6F_sign_corrected is an opt-in profile that preserves the correct T6F architecture fix mechanism but conditionally disables fighting damping and pitch terms during high-authority emergency recenter.**

**Design Philosophy:**
- Keep what works: position torque path, architecture fix gates, raised cap mechanism
- Fix what fights: damping when it opposes position, pitch during large-error recenter
- Preserve safety: all gates, contact validation, height/roll/pitch limits unchanged
- Opt-in only: T5, T6F baseline, APCR1nD remain unchanged

**Expected Outcome:**
- Final torque sign correctness: >80% (vs T6F's 47.5%)
- Net correction authority: ~6.0-7.0 Nm (vs T6F's 2.5 Nm effective)
- Drift control: improve outside ±0.15m metric from 30.1% to <5%

---

## Base Profile

**Profile:** T6F_budget_cap_raise

**Keep Unchanged:**
- ✅ Architecture fix enabled
- ✅ Architecture fix height threshold: 0.45 m
- ✅ Architecture fix hard cap: 6.5 Nm
- ✅ Architecture fix emergency cap: 7.0 Nm
- ✅ APCR1nD band structure (0.10 hard, 0.12 emergency)
- ✅ All four gates (height, band, safety, recenter)
- ✅ Contact validation
- ✅ Height limits (0.35-0.50 m)
- ✅ Roll limit (0.35 rad)
- ✅ Pitch limit (0.52 rad)
- ✅ Recenter priority direct enabled
- ✅ APCR1n wheel damping override infrastructure
- ✅ APCR1m pitch blend infrastructure
- ✅ Final motor torque cap (unchanged)
- ❌ WBC disabled
- ❌ HY2-DIV disabled

---

## New Features

### Feature 1: Enhanced APCR1n Wheel Damping Override

**Current APCR1n Behavior:**
- Detects when damping fights position torque
- Scales damping to 30% when fighting
- Problem: 30% still allows ~1.0-1.5 Nm cancellation at high authority

**Enhanced Behavior:**
- **Disable damping completely** (0.0) when it fights position torque during arch_fix
- **Preserve damping** when it helps position torque
- **Preserve damping** when arch_fix is inactive (normal operation)

**Activation Condition:**
```
IF arch_fix_active == True
AND vd_wheel_damping_recenter_override_enabled == True
AND sign(tau_position) * sign(tau_damping_mean) < 0
THEN
    tau_wheel_vel_left = 0.0
    tau_wheel_vel_right = 0.0
ELSE
    tau_wheel_vel_left = (unchanged)
    tau_wheel_vel_right = (unchanged)
```

**Rationale:**
- Damping opposes wheel velocity, not drift error
- When wheel velocity direction conflicts with correction direction (overshoot/undershoot), damping fights position torque
- At 7.0 Nm authority, fighting damping can cancel 3.5-7.5 Nm of position torque
- Disabling fighting damping preserves full position correction authority

**Safety:**
- Only active during arch_fix (high-authority emergency recenter)
- Only disables when fighting (preserves when helping)
- Does not change T5 or T6F baseline behavior
- Does not affect normal-authority operation

**Expected Impact:**
```
Before: Net = -7.20 (position) + 7.50 (damping fights) = -0.30 Nm or worse
After:  Net = -7.20 (position) + 0.00 (disabled) = -7.20 Nm full authority
```

**Implementation Location:**
- File: `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- Section: After APCR1n damping override logic (lines ~2363-2440)
- Insert enhanced logic before final wheel torque composition

---

### Feature 2: Enhanced APCR1m Pitch Suppression

**Current APCR1m Behavior:**
- Blends pitch torque based on drift magnitude
- Scales pitch from 0.0 to 1.0 depending on error
- Problem: Still allows pitch torque during large-error recenter when it conflicts

**Enhanced Behavior:**
- **Suppress pitch completely** (0.0) when arch_fix active AND `|error| > 0.10 m`
- **Preserve pitch** when arch_fix inactive (normal operation)
- **Preserve pitch** when error small (< 0.10 m)

**Activation Condition:**
```
IF arch_fix_active == True
AND abs(sagittal_position_error_m) > 0.10
THEN
    tau_pitch = 0.0
    tau_pitch_rate = 0.0  # optional, TBD
ELSE
    tau_pitch = (unchanged)
    tau_pitch_rate = (unchanged)
```

**Rationale:**
- Pitch torque stabilizes pitch angle, not drift
- During emergency recenter (|e| > 0.10 m), robot intentionally leans to correct drift
- Pitch stabilization opposes this intentional lean, fighting drift correction
- Pitch torque consistently has wrong sign (4.8% correct in Phase 2)
- Suppressing pitch during large-error recenter lets position torque dominate

**Safety:**
- Only active during arch_fix AND large error (> 0.10 m)
- Preserves pitch control during normal operation
- Preserves pitch control during small errors (< 0.10 m)
- Does not affect T5 or T6F baseline

**Expected Impact:**
```
Before: Net = -7.20 (position) + 1.50 (pitch fights) = -5.70 Nm
After:  Net = -7.20 (position) + 0.00 (suppressed) = -7.20 Nm
```

**Implementation Location:**
- File: `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- Section: After pitch computation, before final composition
- Insert enhanced suppression logic after APCR1m pitch blend (if enabled)

---

## Configuration Parameters

### Schedule Configuration

**Profile Name:** `T6F_sign_corrected`

**New Fields:**
```yaml
# Sign fix enabled (master switch)
sign_fix_enabled: true

# Enhanced damping override
sign_fix_disable_fighting_damping_during_arch_fix: true

# Enhanced pitch suppression
sign_fix_suppress_pitch_during_arch_fix: true
sign_fix_pitch_error_threshold_m: 0.10  # Suppress when |e| > this

# Optional: suppress pitch rate too
sign_fix_suppress_pitch_rate: false  # Start with false, may enable if needed
```

**Inherited from T6F_budget_cap_raise:**
```yaml
# Architecture fix
arch_fix_enabled: true
arch_fix_height_threshold_m: 0.45
arch_fix_hard_max_position_tau: 6.5
arch_fix_emergency_max_position_tau: 7.0

# APCR1nD bands
apcr1nd_tuned_enabled: true
apcr1nd_tuned_variant_name: "T6F"
apcr1nd_hard_band_m: 0.10
apcr1nd_emergency_band_m: 0.12

# Recenter priority
recenter_priority_direct_enabled: true
recenter_priority_safe_min_com_z: 0.35
recenter_priority_safe_roll_rad: 0.35
recenter_priority_safe_pitch_rad: 0.52

# Damping override infrastructure
vd_wheel_damping_recenter_override_enabled: true

# Pitch blend infrastructure (may be used if sign fix disabled)
apc_pitch_blend_enabled: false  # Not needed with sign fix
```

---

## Telemetry Fields

### Required New Fields

**Master status:**
- `sign_fix_enabled` (bool) — Profile has sign fix enabled
- `sign_fix_active` (bool) — Sign fix logic is actively running this step

**Damping override:**
- `sign_fix_damping_disabled` (bool) — Damping was disabled this step
- `sign_fix_damping_helped` (bool) — Damping had correct sign (preserved)
- `sign_fix_damping_fought` (bool) — Damping had wrong sign (would have fought)
- `sign_fix_damping_original_nm` (float) — Original damping torque before override
- `sign_fix_damping_after_nm` (float) — Damping torque after override

**Pitch suppression:**
- `sign_fix_pitch_suppressed` (bool) — Pitch was suppressed this step
- `sign_fix_pitch_original_nm` (float) — Original pitch torque before suppression
- `sign_fix_pitch_after_nm` (float) — Pitch torque after suppression

**Sign correctness:**
- `sign_fix_tau_position_sign` (int) — Sign of position torque (-1, 0, +1)
- `sign_fix_damping_sign` (int) — Sign of damping torque before override
- `sign_fix_pitch_sign` (int) — Sign of pitch torque before suppression
- `sign_fix_final_tau_sign` (int) — Sign of final wheel torque
- `sign_fix_drift_sign` (int) — Sign of drift error
- `sign_fix_final_sign_correct` (bool) — Final torque opposes drift

**Diagnostic:**
- `sign_fix_reason` (str) — Reason for sign fix action: "damping_fought", "pitch_suppressed", "both", "neither", "arch_fix_inactive"

**Existing fields to verify:**
- `arch_fix_active` (bool) — Architecture fix is active
- `apcr1nd_tuned_band_state` (int) — Band state (0=normal, 1=soft, 2=desired, 3=hard, 4=emergency)
- `sagittal_position_error_m` (float) — Drift error
- `tau_position` (float) — Position torque
- `tau_wheel_velocity_left` (float) — Left wheel damping torque
- `tau_wheel_velocity_right` (float) — Right wheel damping torque
- `tau_pitch` (float) — Pitch torque
- `final_wheel_tau_with_apc` (float) — Final wheel torque

---

## Implementation Pseudocode

### Enhanced Damping Override

```python
# After computing tau_wheel_vel_left and tau_wheel_vel_right
# Location: ~line 2440 in sagittal_velocity_damped_balance_controller.py

# Initialize sign fix telemetry
sign_fix_active = False
sign_fix_damping_disabled = False
sign_fix_damping_helped = False
sign_fix_damping_fought = False
sign_fix_damping_original_nm = 0.0
sign_fix_damping_after_nm = 0.0

# Enhanced APCR1n damping override (sign fix)
if self.authority_schedule.sign_fix_enabled and self.authority_schedule.sign_fix_disable_fighting_damping_during_arch_fix:
    if arch_fix_active and self.authority_schedule.vd_wheel_damping_recenter_override_enabled:
        sign_fix_active = True
        
        # Compute mean damping torque
        tau_damping_mean = (tau_wheel_vel_left + tau_wheel_vel_right) / 2.0
        sign_fix_damping_original_nm = float(tau_damping_mean)
        
        # Check if damping opposes position
        sign_position = jnp.sign(tau_position)
        sign_damping = jnp.sign(tau_damping_mean)
        
        # If signs opposite, damping fights position → disable
        if sign_position * sign_damping < 0:
            # Damping fights correction
            sign_fix_damping_fought = True
            sign_fix_damping_disabled = True
            tau_wheel_vel_left = 0.0
            tau_wheel_vel_right = 0.0
            sign_fix_damping_after_nm = 0.0
        else:
            # Damping helps correction → preserve
            sign_fix_damping_helped = True
            sign_fix_damping_after_nm = sign_fix_damping_original_nm
```

---

### Enhanced Pitch Suppression

```python
# After computing tau_pitch
# Location: ~line 1870 in sagittal_velocity_damped_balance_controller.py

# Initialize pitch suppression telemetry
sign_fix_pitch_suppressed = False
sign_fix_pitch_original_nm = float(tau_pitch)
sign_fix_pitch_after_nm = float(tau_pitch)

# Enhanced APCR1m pitch suppression (sign fix)
if self.authority_schedule.sign_fix_enabled and self.authority_schedule.sign_fix_suppress_pitch_during_arch_fix:
    if arch_fix_active and abs(float(sagittal_position_error_m)) > self.authority_schedule.sign_fix_pitch_error_threshold_m:
        sign_fix_active = True
        sign_fix_pitch_suppressed = True
        tau_pitch = 0.0
        sign_fix_pitch_after_nm = 0.0
        
        # Optional: also suppress pitch rate
        if self.authority_schedule.sign_fix_suppress_pitch_rate:
            tau_pitch_rate = 0.0
```

---

### Sign Correctness Computation

```python
# After final wheel torque composition
# Location: ~line 3680 in sagittal_velocity_damped_balance_controller.py

# Compute sign correctness for telemetry
sign_fix_tau_position_sign = int(jnp.sign(tau_position))
sign_fix_damping_sign = int(jnp.sign((tau_wheel_vel_left + tau_wheel_vel_right) / 2.0))
sign_fix_pitch_sign = int(jnp.sign(tau_pitch))
sign_fix_final_tau_sign = int(jnp.sign(final_wheel_tau_with_apc))
sign_fix_drift_sign = int(jnp.sign(sagittal_position_error_m))

# Final sign correctness: torque should oppose drift
# Correct if sign(torque) * sign(drift) < 0 (opposite signs)
if sign_fix_drift_sign != 0:
    sign_fix_final_sign_correct = (sign_fix_final_tau_sign * sign_fix_drift_sign < 0)
else:
    sign_fix_final_sign_correct = True  # Zero drift, any torque acceptable

# Reason string
if sign_fix_damping_disabled and sign_fix_pitch_suppressed:
    sign_fix_reason = "both"
elif sign_fix_damping_disabled:
    sign_fix_reason = "damping_fought"
elif sign_fix_pitch_suppressed:
    sign_fix_reason = "pitch_suppressed"
elif sign_fix_active:
    sign_fix_reason = "arch_fix_active_no_fighting"
elif arch_fix_active:
    sign_fix_reason = "arch_fix_active_signfix_disabled"
else:
    sign_fix_reason = "arch_fix_inactive"
```

---

## Integration Tests Required

### Unit Tests (tests/test_t6f_torque_sign_convention.py)

Already created and passing ✅:
- 3 tests: Position torque sign correctness
- 4 tests: Damping sign detection and override logic
- 3 tests: Pitch sign conflict detection
- 1 test: Architecture fix sign preservation
- 1 test: Safety gate blocking
- 3 tests: Sign fix conditions
- 1 test: Comprehensive summary

### Integration Tests (tests/test_t6_high_height_variants.py)

Add tests for T6F_sign_corrected:
```python
def test_t6f_sign_corrected_profile_exists():
    """Verify T6F_sign_corrected profile exists and is opt-in."""
    
def test_t6f_sign_corrected_preserves_t5_baseline():
    """Verify T5 unchanged when T6F_sign_corrected loaded."""
    
def test_t6f_sign_corrected_preserves_t6f_baseline():
    """Verify T6F unchanged when T6F_sign_corrected loaded."""
    
def test_t6f_sign_corrected_disables_damping_when_fighting():
    """Verify damping disabled when it fights position during arch_fix."""
    
def test_t6f_sign_corrected_preserves_damping_when_helping():
    """Verify damping preserved when it helps position."""
    
def test_t6f_sign_corrected_suppresses_pitch_large_error():
    """Verify pitch suppressed when arch_fix active and |e| > 0.10."""
    
def test_t6f_sign_corrected_preserves_pitch_small_error():
    """Verify pitch preserved when |e| <= 0.10."""
    
def test_t6f_sign_corrected_arch_fix_still_transmits():
    """Verify architecture fix still transmits >4.0 Nm authority."""
    
def test_t6f_sign_corrected_final_motor_cap_respected():
    """Verify final motor torque cap still enforced."""
    
def test_t6f_sign_corrected_no_wbc_path():
    """Verify WBC and hidden paths remain disabled."""
```

### Controller Tests (tests/test_sagittal_velocity_damped_balance_controller.py)

Add sign fix logic tests:
```python
def test_sign_fix_damping_override_logic():
    """Test enhanced damping override activates correctly."""
    
def test_sign_fix_pitch_suppression_logic():
    """Test enhanced pitch suppression activates correctly."""
    
def test_sign_fix_telemetry_fields_exist():
    """Verify all sign fix telemetry fields are logged."""
```

### Telemetry Tests (tests/test_simulation_telemetry_csv_writer.py)

Add CSV writer tests:
```python
def test_sign_fix_telemetry_fields_in_csv():
    """Verify sign fix telemetry fields are written to CSV."""
```

---

## Risk Analysis

### Low Risk ✅

**Position Torque Path:**
- ✅ Unchanged
- ✅ Sign convention correct (100%)
- ✅ Architecture fix mechanism correct

**Architecture Fix:**
- ✅ Unchanged
- ✅ Gates unchanged
- ✅ Raised cap mechanism unchanged
- ✅ Sign preservation correct

**Safety Gates:**
- ✅ All unchanged
- ✅ Height gate (≥0.45 m)
- ✅ Band gate (hard/emergency)
- ✅ Safety gate (contact/height/roll/pitch)
- ✅ Recenter gate

**Baseline Preservation:**
- ✅ T5 unchanged (opt-in profile)
- ✅ T6F unchanged (opt-in profile)
- ✅ APCR1nD unchanged (opt-in profile)

**Implementation:**
- ✅ Only disables fighting terms, never flips signs
- ✅ Only active during arch_fix (high-authority emergency)
- ✅ Preserves helping damping
- ✅ Preserves normal-operation pitch

---

### Medium Risk ⚠️

**Wheel Velocity Oscillation:**
- **Risk:** Disabling damping may increase wheel velocity oscillation during recenter
- **Mitigation:** Only disable when fighting; preserve when helping
- **Mitigation:** Only active during arch_fix (emergency), not normal operation
- **Monitoring:** Track `wheel_velocity_max` and `wheel_velocity_RMS` in diagnostics

**Pitch Excursion:**
- **Risk:** Suppressing pitch may allow larger pitch excursions during recenter
- **Mitigation:** Only suppress when |e| > 0.10 m (emergency recenter)
- **Mitigation:** Pitch hard-stop gate (0.52 rad) still enforced
- **Monitoring:** Track `pitch_max_rad` and `pitch_RMS_rad` in diagnostics

---

### No Risk ❌

**No Global Sign Flips:**
- ✅ No changes to `wheel_torque_sign`
- ✅ No changes to tau_position sign convention
- ✅ No changes to final composition logic

**No Structural Changes:**
- ✅ No changes to controller architecture
- ✅ No changes to WBC/hidden/ownership paths
- ✅ No changes to joint group assignment

**No Default Changes:**
- ✅ No changes to default profile
- ✅ No changes to existing profile registry
- ✅ Opt-in only

---

## Expected Performance

### Sign Correctness

**Target Metrics:**
- Final torque sign correctness: **>80%** (vs T6F's 47.5%)
- Improvement over T6F: **>25 percentage points**

**Baseline for Comparison:**
- T5 (4.0 Nm cap): 47.3% correct
- T6F (7.0 Nm cap, no fix): 47.5% correct
- T6F_sign_corrected (7.0 Nm cap, with fix): **>80%** target

---

### Drift Control

**Target Metrics:**
- Steps outside ±0.15 m: **<5%** (vs T6F's 30.1%)
- Max drift: **<0.20 m** (vs T6F's 0.234 m)
- Mean |drift|: **<0.06 m** (vs T6F's ~0.08 m)

**Baseline for Comparison:**
- T5: Outside ±0.15m = 4.4% (89 steps), max = 0.171 m
- T6F: Outside ±0.15m = 30.1% (601 steps), max = 0.234 m
- T6F_sign_corrected target: Outside ±0.15m **<5%**, max **<0.20 m**

---

### Authority Transmission

**Target Metrics:**
- Architecture fix transmits **>4.0 Nm** when active
- Peak position torque: **6.0-7.0 Nm** during emergency
- Net correction authority: **6.0-7.0 Nm** (vs T6F's ~2.5 Nm effective)

**Sign Fix Impact:**
```
Without sign fix:
Net = 7.0 Nm (position) - 3.5 Nm (damping fights) - 1.0 Nm (pitch fights) = 2.5 Nm

With sign fix:
Net = 7.0 Nm (position) - 0.0 Nm (damping disabled) - 0.0 Nm (pitch suppressed) = 7.0 Nm
```

**Improvement:** From 2.5 Nm to 7.0 Nm = **2.8× increase** in effective authority.

---

### Stability

**Target Metrics:**
- No falls in 2000 steps
- Contact valid >99.5%
- Height stable (0.45-0.52 m range)
- Roll RMS <0.10 rad
- Pitch RMS <0.15 rad

---

## Validation Plan

### Phase 5: 500-Step Diagnostic

**Run Profiles:**
- T5 (reference)
- T6F (baseline)
- T6F_sign_corrected (sign fix)

**Metrics:**
- Sign correctness by component
- Final torque sign correctness
- Sign fix activation rate
- Damping disabled rate
- Pitch suppressed rate
- Drift metrics (max, mean, outside ±0.10/±0.15)
- Stability metrics (contact, height, roll, pitch)
- Authority transmission (>4.0 Nm)

**Pass Criteria:**
- Sign correctness >80%
- Improvement >25 pp vs T6F
- No falls
- Architecture fix still transmits >4.0 Nm

---

### Phase 6: 2000-Step Screening

**Only if Phase 5 passes.**

**Run:**
- T6F_sign_corrected high_0p480 2000 steps

**Compare:**
- T5 first 2000 steps
- T6F rejected 2000 steps
- T6F_sign_corrected new 2000 steps

**Metrics:**
- All Phase 5 metrics
- Extended drift metrics over 2000 steps
- Torque stability over time
- Sign correctness consistency

**Pass Criteria:**
- Sign correctness >80%
- Outside ±0.15m <5%
- Max drift <0.20 m
- Stable over 2000 steps

---

## Files to Modify

### Controller Implementation

**File:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`

**Sections:**
1. Add sign fix fields to `SagittalAuthoritySchedule` (lines ~100-440)
2. Add enhanced damping override logic (after line ~2440)
3. Add enhanced pitch suppression logic (after line ~1870)
4. Add sign correctness telemetry computation (after line ~3680)

**Estimated lines:** ~150 new lines

---

### Profile Configuration

**File:** `wheeled_biped/configs/authority_schedules.py` (or equivalent)

**Add:**
```python
T6F_SIGN_CORRECTED = SagittalAuthoritySchedule(
    profile_name="T6F_sign_corrected",
    # Inherit from T6F
    arch_fix_enabled=True,
    arch_fix_height_threshold_m=0.45,
    arch_fix_hard_max_position_tau=6.5,
    arch_fix_emergency_max_position_tau=7.0,
    apcr1nd_tuned_enabled=True,
    apcr1nd_tuned_variant_name="T6F",
    apcr1nd_hard_band_m=0.10,
    apcr1nd_emergency_band_m=0.12,
    recenter_priority_direct_enabled=True,
    vd_wheel_damping_recenter_override_enabled=True,
    # New sign fix fields
    sign_fix_enabled=True,
    sign_fix_disable_fighting_damping_during_arch_fix=True,
    sign_fix_suppress_pitch_during_arch_fix=True,
    sign_fix_pitch_error_threshold_m=0.10,
    sign_fix_suppress_pitch_rate=False,
)
```

---

### Telemetry Writer

**File:** `wheeled_biped/sim/simulation_telemetry_csv_writer.py`

**Add columns:**
- `sign_fix_enabled`
- `sign_fix_active`
- `sign_fix_damping_disabled`
- `sign_fix_damping_helped`
- `sign_fix_damping_fought`
- `sign_fix_damping_original_nm`
- `sign_fix_damping_after_nm`
- `sign_fix_pitch_suppressed`
- `sign_fix_pitch_original_nm`
- `sign_fix_pitch_after_nm`
- `sign_fix_tau_position_sign`
- `sign_fix_damping_sign`
- `sign_fix_pitch_sign`
- `sign_fix_final_tau_sign`
- `sign_fix_drift_sign`
- `sign_fix_final_sign_correct`
- `sign_fix_reason`

**Estimated lines:** ~30 new lines

---

### Tests

**Files:**
- `tests/test_t6f_torque_sign_convention.py` (already created, 16 tests passing ✅)
- `tests/test_t6_high_height_variants.py` (add 10 integration tests)
- `tests/test_sagittal_velocity_damped_balance_controller.py` (add 3 logic tests)
- `tests/test_simulation_telemetry_csv_writer.py` (add 1 CSV test)

**Estimated lines:** ~500 new test lines

---

## Implementation Checklist

### Phase 4: Design ✅

- [x] Create design document
- [x] Define activation conditions
- [x] Define telemetry fields
- [x] Specify implementation location
- [x] Create pseudocode
- [x] Define integration tests
- [x] Analyze risks
- [x] Define validation plan

### Phase 5: Implementation (Next)

- [ ] Add sign fix fields to `SagittalAuthoritySchedule`
- [ ] Implement enhanced damping override logic
- [ ] Implement enhanced pitch suppression logic
- [ ] Add sign correctness telemetry computation
- [ ] Create T6F_sign_corrected profile configuration
- [ ] Update telemetry CSV writer
- [ ] Add integration tests
- [ ] Run unit tests (expect all pass)
- [ ] Run integration tests
- [ ] Run 500-step diagnostic
- [ ] Create Phase 5 report

---

## What NOT to Do

❌ **Do NOT modify position torque sign convention**  
❌ **Do NOT flip wheel_torque_sign**  
❌ **Do NOT patch at final composition**  
❌ **Do NOT modify T5 baseline**  
❌ **Do NOT modify T6F baseline**  
❌ **Do NOT modify APCR1nD baseline**  
❌ **Do NOT make T6F_sign_corrected default**  
❌ **Do NOT enable WBC**  
❌ **Do NOT enable HY2-DIV**  
❌ **Do NOT relax Step E gates**  
❌ **Do NOT run 5000-step yet**  
❌ **Do NOT commit yet**

✅ **DO create opt-in profile**  
✅ **DO preserve all safety gates**  
✅ **DO preserve architecture fix mechanism**  
✅ **DO disable only fighting terms**  
✅ **DO preserve helping terms**  
✅ **DO add comprehensive telemetry**  
✅ **DO run 500-step diagnostic first**

---

## Summary

**T6F_sign_corrected design is complete and ready for implementation.**

**Key Design Principles:**
1. Keep what works: position torque, architecture fix, gates
2. Fix what fights: damping when opposing, pitch during large error
3. Preserve safety: all gates, limits, validation unchanged
4. Opt-in only: no impact on existing profiles

**Expected Outcome:**
- Sign correctness: 47.5% → >80% (+33 pp improvement)
- Effective authority: 2.5 Nm → 7.0 Nm (2.8× improvement)
- Drift control: 30.1% outside ±0.15m → <5% (6× improvement)

**Ready to proceed to Phase 5 implementation.**

---

**Status:** Phase 4 Design COMPLETE  
**Next Phase:** Phase 5 Implementation  
**Classification:** T6F_SIGN_CORRECTED_DESIGN_READY  
**Date:** 2026-06-12
