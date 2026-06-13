# T6F Sign Fix Phase 6 Failure Root Cause Report

**Date**: 2026-06-12  
**Author**: Systematic debugging investigation  
**Task**: Phase 6 500-step diagnostic failure analysis

---

## Executive Summary

The T6F_sign_corrected profile **FAILED** the 500-step diagnostic with sign correctness of 49.3% (target: >80%, improvement vs T6F: -1.4pp vs target +25pp). 

Root cause investigation identified **TWO CRITICAL IMPLEMENTATION BUGS**:

1. **Pitch Suppression Placement Bug**: Pitch suppression code reads `arch_fix_active` variable **before it is set**, causing 0% activation despite conditions being met 166 times (33.3% of steps).

2. **Band State Logic Bug**: APCR1nD band state remains at 0 (normal) for 100% of steps despite error reaching 0.19m (above emergency threshold 0.12m), preventing arch_fix from raising position cap above 4.0 Nm.

These bugs prevent the sign fix from functioning as designed. **No further 2000-step evaluation should proceed** until these bugs are fixed.

---

## Phase 6 Results Summary

### Sign Correctness Performance

| Profile | Final Torque Sign Correct | vs T6F | Target Met? |
|---------|---------------------------|--------|-------------|
| T5 | 37.3% | -13.4pp | ❌ |
| T6F | 50.7% | baseline | ❌ |
| **T6F_sign_corrected** | **49.3%** | **-1.4pp** | **❌ FAILED** |
| **Target** | **>80%** | **+25pp** | - |

### Activation Summary

| Metric | Count | % |
|--------|-------|---|
| sign_fix_active | 156 | 31.3% |
| damping_disabled | 73 | 14.6% |
| damping_helped | 83 | 16.6% |
| **pitch_suppressed** | **0** | **0.0%** ← BUG |
| arch_fix_active | 169 | 33.9% |
| high_authority (>4.0 Nm) | 8 | 1.6% |

### Drift Performance

| Profile | Outside ±0.15m | Max Abs Error |
|---------|----------------|---------------|
| T5 | 17.8% | 0.180m |
| T6F | 24.2% | 0.189m |
| **T6F_sign_corrected** | **25.5%** | **0.192m** |

T6F_sign_corrected drift is **worse** than T6F, not better.

---

## Root Cause Analysis

### Phase A: Telemetry Integrity

**Status**: ✅ VALID (with minor issue)

**Findings**:
- All critical telemetry fields present
- sign_fix_enabled: True ✓
- arch_fix_enabled: True ✓
- Row count: 499 (expected 500, acceptable)
- Missing field: `vd_sagittal_authority_profile` (non-critical string identifier)

**Conclusion**: Telemetry is valid for analysis.

---

### Phase B: Pitch Suppression Activation Audit

**Status**: ❌ **CRITICAL BUG IDENTIFIED**

**Design Intent**:
Pitch suppression should activate when:
- `arch_fix_active == True` AND
- `abs(sagittal_position_error_m) > 0.10m`

**Expected Activation**: 166 steps (33.3%) met both conditions  
**Actual Activation**: 0 steps (0.0%)

**Evidence**:

```
Condition Analysis:
  Steps where arch_fix_active == True: 169 (33.9%)
  Steps where abs(error) > 0.10m: 194 (38.9%)
  Steps where BOTH conditions true: 166 (33.3%)
  Steps where sign_fix_pitch_suppressed == True: 0 (0.0%)
```

**Error Distribution During arch_fix**:
- min: 0.0976m (just below threshold)
- max: 0.1916m (well above threshold)
- mean: 0.1552m
- **98.3% of arch_fix steps had error > 0.10m**

**Sample Discrepancy**:
```
Step 87:
  arch_fix_active: True
  abs(error): 0.1014 m (above 0.10m threshold)
  sign_fix_pitch_suppressed: False  ← BUG
  tau_pitch: 3.507 Nm (should be suppressed to 0.0)
  sign_fix_reason: sign_fix_disabled
```

**Root Cause**:

The pitch suppression code at **line 2027-2046** reads `arch_fix_active` **before** it is set to True.

```python
# Line 2027: Pitch suppression checks arch_fix_active
if (self.authority_schedule.sign_fix_enabled and
    self.authority_schedule.sign_fix_suppress_pitch_during_arch_fix and
    arch_fix_active):  # ← Still False here!
    
    if abs_sagittal_error > pitch_error_threshold:
        tau_pitch = 0.0
        sign_fix_pitch_suppressed = True

# Line 2253: arch_fix_active is set to True 226 lines later
arch_fix_active = True
```

**Code Order**:
1. Line 1810: `arch_fix_active = False` (initialization)
2. Line 2027: Pitch suppression reads `arch_fix_active` (still False)
3. Line 2253: `arch_fix_active = True` (set in arch fix logic)
4. Line 2589: Damping override reads `arch_fix_active` (now True, works correctly)

**Why Damping Override Works but Pitch Suppression Doesn't**:
- Damping override (line 2589) is placed **after** arch_fix logic, so it sees the correct value
- Pitch suppression (line 2027) is placed **before** arch fix logic, so it always sees False

**Classification**: `PITCH_SUPPRESSION_BUG_CONDITION_TRUE_BUT_NOT_ACTIVE`

---

### Phase C: High Authority Activation Audit

**Status**: ❌ **CRITICAL BUG IDENTIFIED**

**Design Intent**:
T6F_sign_corrected should raise position cap to 6.5-7.0 Nm during hard/emergency bands, allowing high torque transmission.

**Expected Behavior**:
- APCR1nD bands activate when error crosses thresholds:
  - hard_band: 0.10m → cap 6.5 Nm
  - emergency_band: 0.12m → cap 7.0 Nm
- Arch fix raises upstream cap when band is hard/emergency AND height > 0.45m

**Actual Behavior**:

```
Band State Distribution:
  Band state 0 (normal): 499 (100.0%)
  Band state 1 (soft): 0 (0.0%)
  Band state 2 (hard): 0 (0.0%)
  Band state 3 (emergency): 0 (0.0%)
```

**ALL steps remained in "normal" band** despite:
- Error reaching 0.1916m (above emergency threshold 0.12m)
- 169 steps with arch_fix_active
- Position torque demand reaching 7.00 Nm

**Evidence**:

The 8 high-authority steps show the bug clearly:

```
Step 305 (highest torque):
  final_tau_mean: 6.12 Nm (transmitted)
  tau_position: -4.00 Nm (clipped at default cap!)
  arch_fix_active: False  ← Should be True
  band_state: 0 (normal) ← Should be 3 (emergency)
  abs(error): 0.1867 m (well above 0.12m emergency threshold)
```

**Position Torque Stats**:
- tau_position max: 7.00 Nm (demand exists)
- tau_position_after_clip: -4.00 Nm (clipped at default cap)
- Steps with |tau_position| > 4.0 Nm: 169 (33.9%)
- arch_fix_requested_cap when active: 6.50-7.00 Nm
- But position torque was clipped at 4.0 Nm anyway

**Root Cause**:

The APCR1nD band logic is **not transitioning out of normal state** even when error exceeds all thresholds. This suggests:

1. Band transition logic has a bug (entry conditions not met), OR
2. Band state is being reset/overwritten after computation, OR
3. A gate is blocking band transition that shouldn't be active

Because arch_fix shows as active 169 times but band_state is always 0, and the high-authority steps show arch_fix_active=False when it should be True, there's likely a **gate logic bug** preventing proper band/arch_fix activation.

**Classification**: `HIGH_AUTHORITY_RARE_BECAUSE_ARCH_FIX_GATED_OFF`

---

## Classification

**Final Classification**: `T6F_SIGN_FIX_FAILURE_MIXED_CAUSES`

**Primary Bugs**:
1. **Pitch suppression placement bug** (Phase B) - reads arch_fix_active before it's set
2. **Band state / arch fix gate bug** (Phase C) - bands never transition, arch_fix doesn't raise cap during high-authority steps

**Contributing Factors**:
- 500-step window may be too short to fully characterize behavior
- Sign fix design assumes pitch and damping are primary causes, but bugs prevent testing that hypothesis

---

## Recommendation

**`FIX_PITCH_SUPPRESSION_IMPLEMENTATION`** + **`FIX_ARCH_FIX_GATE_LOGIC`**

### Required Fixes

#### Fix 1: Move Pitch Suppression After arch_fix Logic

**Current (WRONG)**:
```python
# Line 2027: Too early - arch_fix_active still False
if (sign_fix_enabled and sign_fix_suppress_pitch and arch_fix_active):
    if abs_sagittal_error > pitch_error_threshold:
        tau_pitch = 0.0
        sign_fix_pitch_suppressed = True

# Line 2253: arch_fix_active set here
arch_fix_active = True
```

**Fixed (CORRECT)**:
```python
# Line 2253: arch_fix_active set here
arch_fix_active = True

# Move pitch suppression to AFTER arch_fix logic (after line 2253)
if (sign_fix_enabled and sign_fix_suppress_pitch and arch_fix_active):
    if abs_sagittal_error > pitch_error_threshold:
        tau_pitch = 0.0
        sign_fix_pitch_suppressed = True
```

#### Fix 2: Debug Band State / Arch Fix Gate Logic

**Investigation needed**:
1. Add telemetry for ALL arch_fix gates:
   - `arch_fix_height_gate_pass`
   - `arch_fix_band_gate_pass` (why is this False when band should be hard/emergency?)
   - `arch_fix_safety_gate_pass`
   - `arch_fix_recenter_gate_pass`

2. Add telemetry for APCR1nD band transition logic:
   - Band entry conditions
   - Band hold conditions
   - Converging release logic
   - Which gate/condition is preventing band 2/3 activation?

3. Check if band_state telemetry is showing the RIGHT band_state:
   - Is it captured before or after band logic runs?
   - Is there a separate "requested" vs "actual" band state?

**Hypothesis**: The arch_fix_band_gate_pass is checking a stale or wrong band_state variable, or the band state logic has a separate bug preventing transition to hard/emergency.

### Verification Plan

After fixes:

1. **Rerun 500-step diagnostic** with enhanced telemetry
2. **Verify pitch suppression activates** when arch_fix active AND error > 0.10m
3. **Verify band state transitions** to hard (2) when error > 0.10m, emergency (3) when error > 0.12m
4. **Verify arch_fix raises cap** to 6.5/7.0 Nm when band is hard/emergency
5. **Verify high-authority transmission** >4.0 Nm during arch_fix
6. **Measure sign correctness improvement** - should exceed 80% if design is correct

Only proceed to **1200-step diagnostic** after 500-step shows:
- Pitch suppression >0%
- Band state transitions working
- High-authority >4.0 Nm transmission working
- Sign correctness improving

### Do NOT Proceed With

- ❌ 2000-step evaluation
- ❌ 5000-step evaluation  
- ❌ Step C validation
- ❌ Step D validation
- ❌ Commit T6F_sign_corrected
- ❌ Paper claims about sign fix

---

## Phase 6 Classification

**T6F_SIGN_FIX_500_FAIL_SIGN_STILL_WRONG**

**Explanation**: Sign correctness 49.3% vs target >80%, improvement -1.4pp vs target +25pp. The implementation has critical bugs preventing evaluation of the design hypothesis.

---

## Lessons Learned

1. **Variable ordering matters**: Reading a variable before it's set is a classic bug that passed code review because the variable exists (initialized to False).

2. **Multi-component systems need component-level telemetry**: Band state being always 0 was only discoverable because we logged it. Without band_state telemetry, we'd see "arch_fix not working" without knowing why.

3. **500-step diagnostics caught the bugs early**: If we'd run 2000-step first, we'd have wasted more compute on a broken implementation.

4. **Systematic debugging worked**: Following the 6-phase audit protocol (telemetry → pitch → authority → sign metric → damping → report) identified both bugs within hours.

---

## Next Steps

1. **Fix pitch suppression placement** (move after arch_fix logic)
2. **Debug band state bug** (add gate telemetry, investigate transition logic)
3. **Rerun 500-step diagnostic** with fixes
4. **Verify activation telemetry** shows expected behavior
5. **Only then consider 1200-step** evaluation

**Status**: BLOCKED on implementation fixes. Do not proceed to Phase 7.
