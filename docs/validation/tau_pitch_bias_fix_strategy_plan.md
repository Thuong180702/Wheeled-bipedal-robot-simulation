# Tau_pitch Bias Fix Strategy Plan

**Date:** 2026-06-08
**Task:** Design fix strategy for persistent positive tau_pitch bias at low_0p300
**Status:** NO IMPLEMENTATION - Design only

---

## 1. Root Cause Summary

**Primary Root Cause:** Initial hip_pitch error of 0.45 rad creates steady forward pitch moment

**Secondary Root Cause:** Position authority cap (4.0 Nm) insufficient to cancel tau_pitch (up to 5.5 Nm)

**NOT the root cause:**
- tau_pitch computation (correct)
- tau_pitch sign (correct)
- Pitch reference (correct = 0.0)

---

## 2. Design Principles

Per task instructions:
- D2 remains protected
- New fix opt-in only
- No WBC
- No HY2-DIV
- Validate 500-step first
- Primary metrics: signed_support positive%, time outside ±0.15, tau_pitch mean/positive%

---

## 3. Candidate Fixes

### Fix A: Increase Position Authority at Low Height

**Description:** Continuous height-scheduled position cap that increases smoothly below 0.40 m.

**Implementation:**
```python
# In SagittalVelocityDampedBalanceController
if self.authority_schedule.continuous_max_position_tau:
    # Already exists: scheduled_k_position(
    #     z_ref=schedule_height_ref,
    #     k_nominal=self.authority_schedule.max_position_tau_nominal,
    #     k_low_max=self.authority_schedule.max_position_tau_low_max,
    #     ...
    # )
    pass

# Need: Increase max_position_tau_low_max from 6.0 to 8.0 Nm for low_0p300
```

**Proposed profile H1 (height-optimized position cap):**
```python
H1_LOW_HEIGHT_POSITION_CAP = SagittalAuthoritySchedule(
    profile_name="H1_low_height_position_cap",
    applies_to_variants=("low_0p300", "low_5cm"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=3.0,
    max_position_tau_low_max=8.0,  # Increased from 6.0
    k_position_z_low=0.300,
    k_position_z_high=0.393,
)
```

**Rationale:**
- At 0.30 m: cap = 8.0 Nm (vs current 4.0 Nm at nominal)
- Can cancel tau_pitch up to 5.5 Nm peak
- Smoothly transitions between heights

**Risk:** Medium
- Must validate at nominal heights to ensure no regression

**Validation plan:**
1. Run H1 at low_0p300 for 500 steps
2. Compare tau_pitch positive%, position saturation%, outside ±0.15 vs D2
3. Run H1 at nominal for 500 steps
4. Confirm no regression in nominal stability

---

### Fix B: Reduce kp_pitch at Low Height

**Description:** Reduce pitch proportional gain at low height to reduce tau_pitch magnitude.

**Implementation:**
```python
# New continuous kp_pitch scheduling
if self.authority_schedule.continuous_kp_pitch:
    effective_kp_pitch = scheduled_k_position(
        z_ref=schedule_height_ref,
        k_nominal=self.authority_schedule.kp_pitch_nominal,
        k_low_max=self.authority_schedule.kp_pitch_low_max,
        ...
    )
```

**Proposed profile H2 (low-height pitch gain reduction):**
```python
H2_LOW_HEIGHT_PITCH_REDUCTION = SagittalAuthoritySchedule(
    profile_name="H2_low_height_pitch_reduction",
    applies_to_variants=("low_0p300", "low_5cm"),
    continuous_kp_pitch=True,
    kp_pitch_nominal=50.0,
    kp_pitch_low_max=35.0,  # 30% reduction at lowest height
    kp_pitch_z_low=0.300,
    kp_pitch_z_high=0.393,
)
```

**Rationale:**
- kp_pitch = 35 at 0.30 m
- tau_pitch peak = 35 * 0.11 = 3.85 Nm (vs 5.5 Nm)
- Within 4.0 Nm cap
- Reduces tau_pitch magnitude without changing sign

**Risk:** Medium-High
- Lower pitch gain may reduce balance responsiveness
- Must validate carefully

**Validation plan:**
1. Run H2 at low_0p300 for 500 steps
2. Compare pitch RMS, pitch max, fall rate vs D2
3. Must not increase fall rate

---

### Fix C: Add Continuous Pitch Rate Damping at Low Height

**Description:** Increase kd_pitch (pitch rate damping) at low height to prevent pitch buildup.

**Implementation:**
```python
# New continuous kd_pitch scheduling
if self.authority_schedule.continuous_kd_pitch:
    effective_kd_pitch = scheduled_k_position(
        z_ref=schedule_height_ref,
        k_nominal=self.authority_schedule.kd_pitch_nominal,
        k_low_max=self.authority_schedule.kd_pitch_low_max,
        ...
    )
```

**Proposed profile H3 (increased damping at low height):**
```python
H3_LOW_HEIGHT_DAMPING = SagittalAuthoritySchedule(
    profile_name="H3_low_height_damping",
    applies_to_variants=("low_0p300", "low_5cm"),
    continuous_kd_pitch=True,
    kd_pitch_nominal=10.0,
    kd_pitch_low_max=20.0,  # 2x damping at lowest height
    kd_pitch_z_low=0.300,
    kd_pitch_z_high=0.393,
)
```

**Rationale:**
- At 0.30 m: kd_pitch = 20.0
- tau_pitch_rate = 20 * 0.21 = 4.2 Nm peak (vs 2.1 Nm)
- Better damping prevents pitch buildup
- Does not change equilibrium behavior

**Risk:** Medium
- May cause jerky motion if too high
- Must validate at nominal heights

**Validation plan:**
1. Run H3 at low_0p300 for 500 steps
2. Compare pitch_rate RMS, pitch overshoot vs D2
3. Check for oscillatory behavior

---

### Fix D: Height-Dependent Pitch Reference Offset

**Description:** Add a small forward pitch reference for low heights to match natural posture.

**Implementation:**
```python
# In simulate_hierarchical_controller.py
# Modify pitch reference based on height
if commanded_height_ref_m < 0.35:
    # Low height needs slight forward pitch for equilibrium
    pitch_x_ref_offset = -0.03  # rad (nose down = negative pitch)
else:
    pitch_x_ref_offset = 0.0
    
pitch_x_ref = float(pitch_x_eq) + pitch_x_ref_offset
pitch_x_error = float(centroidal_state_control.body_pitch_x) - pitch_x_ref
```

**Rationale:**
- At extreme squat (hip=78.85°, knee=134.56°), robot may naturally need forward pitch
- Adding small offset reduces pitch error from initial condition bias
- tau_pitch mean reduces proportionally

**Risk:** High
- Requires equilibrium re-search with pitch constraint
- May break other heights
- Most invasive change

**Validation plan:**
1. Re-run equilibrium search for low_0p300 with pitch_x constraint
2. Validate new equilibrium at low_0p300
3. Validate nominal heights unaffected

---

### Fix E: Fix Initial Hip_pitch Error at Simulation Start

**Description:** Ensure robot starts AT equilibrium, not with 0.45 rad error.

**Implementation:**
- Check initialization in simulate_hierarchical_controller.py
- Ensure joint_pos matches equilibrium_joint_pos at step 0
- hip_pitch_error should be < 0.05 rad at start

**Rationale:**
- Initial condition bias is the root cause
- Fixing it removes the steady forward moment
- tau_pitch positive% should drop significantly

**Risk:** Low
- Fixes root cause directly
- Does not change controller behavior

**Validation plan:**
1. Check initialization code
2. Confirm hip_pitch_error < 0.05 rad at step 0
3. Run for 500 steps and compare vs D2

---

## 4. Recommended Fix Strategy

**Primary recommendation: Fix E (initial condition) + Fix A (position cap)**

**Rationale:**
- Fix E addresses the root cause (initial hip_pitch error)
- Fix A addresses the symptom (position authority insufficient)
- Both are low-to-medium risk
- Can be validated independently

**Do NOT implement:**
- Fix D (height-dependent pitch reference) without further investigation
- G1c or stronger bias cancellation (proven to worsen outside-band)

---

## 5. Next Executable Experiment Plan

### Experiment H1: Position Cap Increase

**Command:**
```bash
python scripts/simulate_hierarchical_controller.py \
    --controller balance-core \
    --height-variant low_0p300 \
    --sagittal-controller velocity-damped \
    --sagittal-authority-profile H1_low_height_position_cap \
    --num-steps 500 \
    --seed 42 \
    --output-dir outputs/step_e_extreme_support_fix_eval/h1_low_height_position_cap_500
```

**Success criteria:**
- tau_pitch positive% < 85% (vs D2 89.2%)
- tau_position saturation% < 20% (vs D2 35.4%)
- time outside ±0.15 < 15% (vs D2 19.2%)

---

## 6. Summary

| Fix | Risk | Addresses | Recommended |
|-----|------|-----------|-------------|
| A: Position cap increase | Medium | Position saturation | ✅ Yes |
| B: kp_pitch reduction | Medium-High | tau_pitch magnitude | ⚠️ Maybe |
| C: kd_pitch increase | Medium | Pitch damping | ⚠️ Maybe |
| D: Pitch reference offset | High | Root cause | ❌ No (yet) |
| E: Initial condition fix | Low | Root cause | ✅ Yes |

**Do NOT:**
- Continue increasing G1 bias cancel gains
- Implement HY2-DIV
- Add WBC
- Enable legacy WBC
- Relax Step E gates