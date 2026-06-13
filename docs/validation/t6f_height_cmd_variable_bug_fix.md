# T6F height_cmd Variable Bug Fix

**Date:** 2026-06-12  
**Phase:** 7C (during paired diagnostics)  
**Status:** Fixed

---

## Bug Description

During Phase 7C (paired 1200-step diagnostics), T6F simulation failed with:

```
NameError: name 'height_cmd' is not defined. Did you mean: 'height_safe'?
```

**Location:** Two instances in `sagittal_velocity_damped_balance_controller.py`:
- Line 1768 (early initialization section)
- Line 2107 (architecture fix application section)

---

## Root Cause

The T6F architecture fix code referenced `height_cmd`, but the correct variable name in the controller is `schedule_height_ref`.

The `schedule_height_ref` variable is computed at line 1707-1714:

```python
if commanded_height_ref_m is not None:
    schedule_height_ref = commanded_height_ref_m
    schedule_height_source = "target_reference"
else:
    alpha_filter = 0.9
    self._filtered_com_z = alpha_filter * self._filtered_com_z + (1.0 - alpha_filter) * float(com_z_m)
    schedule_height_ref = self._filtered_com_z
    schedule_height_source = "filtered_current_fallback"
```

---

## Fix Applied

**Changed:**
```python
arch_fix_height_gate_pass = height_cmd >= self.authority_schedule.arch_fix_height_threshold_m
```

**To:**
```python
arch_fix_height_gate_pass = schedule_height_ref >= self.authority_schedule.arch_fix_height_threshold_m
```

**Files modified:**
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` (2 instances)

---

## Verification

After fix:
- `python -m py_compile wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` → success
- `grep -n "height_cmd" wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` → no matches
- T6F simulation started successfully

---

## Why This Was Missed

The bug was introduced during Phase 6 implementation but only discovered during Phase 7C runtime because:

1. Phase 6 implementation was pure code addition (no runtime test)
2. Existing unit tests don't exercise T6F architecture fix code paths
3. The variable name was assumed from design documentation but not verified against actual code

---

## Prevention

Future architecture fixes should:
1. Run short smoke simulation immediately after implementation
2. Add unit test for new telemetry fields
3. Verify variable names against existing code before writing

---

**Status:** Bug fixed, T6F simulation restarted successfully
