# T6F Sign Fix Pitch Suppression Placement Fix

**Date**: 2026-06-12  
**Task**: Phase 2 - Fix pitch suppression placement bug

---

## Problem Statement

**Bug ID**: Bug 1 from Phase 6 root cause investigation

**Location**: `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py:2027`

**Description**: Pitch suppression code checked `arch_fix_active` at line 2027, **before** `arch_fix_active` was set to `True` at line 2253 (226 lines later).

**Evidence**:
- Condition was TRUE 166 times (33.3% of steps): `arch_fix_active == True AND abs(error) > 0.10m`
- Pitch suppression activated: **0 times (0.0%)**
- **98.3%** of arch_fix steps had error > 0.10m threshold
- Max error during arch_fix: 0.1916m (well above 0.10m threshold)

**Impact**: Pitch suppression never activated, preventing evaluation of whether pitch is a primary cause of sign incorrectness.

---

## Root Cause

Variable ordering bug:

```python
# Line 1810: arch_fix_active initialized to False
arch_fix_active = False

# Line 2027: Pitch suppression checks arch_fix_active (STILL FALSE!)
if (sign_fix_enabled and sign_fix_suppress_pitch and arch_fix_active):
    if abs_sagittal_error > pitch_error_threshold:
        tau_pitch = 0.0

# Line 2253: arch_fix_active set to True (226 lines AFTER check!)
arch_fix_active = True
```

The pitch suppression code executed before `arch_fix_active` was computed, so it always saw `False`.

---

## Solution

**Moved pitch suppression code from line 2027 to immediately after line 2253** where `arch_fix_active = True` is set.

### Before (WRONG):

```python
# Line 2027 - TOO EARLY
if (sign_fix_enabled and sign_fix_suppress_pitch and arch_fix_active):
    if abs_sagittal_error > pitch_error_threshold:
        tau_pitch = 0.0
        sign_fix_pitch_suppressed = True

# ... 226 lines later ...

# Line 2253
arch_fix_active = True
```

### After (CORRECT):

```python
# Line 2253
arch_fix_active = True

# IMMEDIATELY AFTER - pitch suppression now sees correct arch_fix_active value
if (sign_fix_enabled and sign_fix_suppress_pitch):
    if abs_sagittal_error > pitch_error_threshold:
        tau_pitch = 0.0
        sign_fix_pitch_suppressed = True
```

**Note**: Removed `arch_fix_active` from the condition check since we're now inside the `if all_gates_pass:` block where `arch_fix_active = True` was just set.

---

## Changes Made

### Modified Files

**wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py**:
1. **Removed** pitch suppression code from line 2027 (before arch_fix_active set)
2. **Added** pitch suppression code after line 2253 (immediately after arch_fix_active = True)
3. **Simplified** condition: removed redundant `arch_fix_active` check since we're inside the arch_fix activation block

---

## Verification

### Compilation

```bash
python -m py_compile wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py
```

**Result**: ✅ PASSED

### Unit Tests

```bash
pytest tests/test_t6f_torque_sign_convention.py -v
```

**Result**: ✅ 16/16 PASSED

Key test: `test_pitch_suppress_condition` - Verifies pitch suppression activates when conditions are met

---

## Expected Behavior After Fix

After this fix, pitch suppression **should activate** when:
1. ✅ `sign_fix_enabled == True`
2. ✅ `sign_fix_suppress_pitch_during_arch_fix == True`
3. ✅ **arch_fix is actually active** (all gates passed)
4. ✅ `abs(sagittal_position_error_m) > 0.10m`

In the 500-step diagnostic, this means:
- **Expected activation**: ~166 steps (33.3%) where error > 0.10m during arch_fix
- **Previous activation**: 0 steps (bug)
- **After fix**: Should activate on eligible steps

---

## Verification in Next 500-Step Run

Phase 5 will verify:
- `sign_fix_pitch_suppressed > 0%` (should be ~33%)
- `tau_pitch == 0.0` when suppression active
- Sign correctness improves toward target >80%

---

## Classification

**PITCH_SUPPRESSION_PLACEMENT_FIXED**

---

## Related Bugs

This fix only addresses **Bug 1**. **Bug 2** (band state logic) remains and will be fixed in Phase 3.

---

## Files Modified

- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` - Moved pitch suppression ~25 lines

## Files Created

- `docs/validation/t6f_sign_fix_pitch_suppression_placement_fix.md` (this file)
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_sign_fix_pitch_suppression_placement_fix.json` (pending)
