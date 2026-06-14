# T5 Tuned Telemetry CSV Fix Report

**Date:** 2026-06-12  
**Issue:** Tuned telemetry fields not logged in CSV  
**Classification:** T5_TUNED_TELEMETRY_CSV_FIXED

---

## Problem

Low_0p300 T5 5000-step validation report indicated that tuned telemetry fields were not present in the CSV output:

**Expected fields (19 total):**
- tuned_variant_name
- tuned_recenter_active
- tuned_band_state
- tuned_band_state_id
- tuned_abs_error
- tuned_error_rate
- tuned_moving_away
- tuned_converging
- tuned_release_allowed
- tuned_active_reason
- tuned_block_reason
- tuned_position_cap_current
- tuned_wheel_damping_scale
- tuned_wheel_damping_override_active
- tuned_outside_band_active
- tuned_outside_band_inactive
- tuned_recenter_held
- tuned_release_counter
- tuned_final_torque_direction_correct

**Available in CSV:** 0/19

---

## Root Cause

The sagittal velocity-damped balance controller correctly included tuned telemetry fields in its diagnostics dict (lines 3690-3709 in `sagittal_velocity_damped_balance_controller.py`), but the simulation script did not append ALL diagnostics fields to the telemetry CSV.

The simulation script only appended explicitly handled fields, and tuned telemetry fields were added after the low_0p300 5000-step run.

---

## Fix Applied

**File:** `scripts/simulate_hierarchical_controller.py`

**Change 1: Dynamic telemetry field creation (line 2009-2015)**

Before:
```python
for name, value in result.telemetry.items():
    if isinstance(value, tuple):
        telemetry[name].append(",".join(str(v) for v in value))
    else:
        telemetry[name].append(value)
```

After:
```python
for name, value in result.telemetry.items():
    if isinstance(value, tuple):
        telemetry.setdefault(name, []).append(",".join(str(v) for v in value))
    else:
        telemetry.setdefault(name, []).append(value)
```

**Rationale:** Use `setdefault` to dynamically create telemetry columns for diagnostics fields that weren't pre-initialized.

**Change 2: Generic sagittal diagnostics append loop (after line 5414)**

Added:
```python
# Append all remaining sagittal diagnostics fields (including tuned telemetry)
# Use setdefault to dynamically create columns for new fields
if is_balance_core_mode(args):
    for key, value in sagittal_diag.items():
        # Skip fields already explicitly handled above
        if key not in telemetry or len(telemetry[key]) < step:
            if isinstance(value, (int, float, bool, str)):
                telemetry.setdefault(key, []).append(value)
            else:
                # Convert other types to string for CSV compatibility
                telemetry.setdefault(key, []).append(str(value))
```

**Rationale:** Append ALL sagittal diagnostics fields (including tuned telemetry) that aren't already explicitly handled, ensuring future diagnostics fields are automatically logged without manual CSV plumbing.

---

## Verification

**Test:** 50-step T5 simulation at low_0p300

**Command:**
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile APCR1nD_T5_band_limited_balanced \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 50 \
  --telemetry-decimation 1 \
  --failure-window-steps 50
```

**Result:**
- ✅ Simulation completed successfully (50/50 steps)
- ✅ All 19 tuned telemetry fields present in CSV
- ✅ Field values correct:
  - `tuned_variant_name` = "T5"
  - `tuned_band_state` = "normal" (expected for minimal drift)
  - `tuned_band_state_id` = 0 (normal band)
  - `tuned_recenter_active` = False (no large drift in 50 steps)
  - `tuned_abs_error` mean = 0.0 m (minimal drift)

---

## Impact

✅ **Fixed:** Tuned telemetry fields now logged to CSV  
✅ **Future-proof:** Generic loop captures future diagnostics fields automatically  
✅ **No breaking changes:** Existing telemetry fields unchanged  
✅ **Ready for high_0p480 validation:** Can now verify tuned feature activation

---

## Classification

**T5_TUNED_TELEMETRY_CSV_FIXED**

All 19 expected tuned telemetry fields are now present in CSV output and contain valid values.

---

**Date:** 2026-06-12  
**Phase:** 1 (Fix Tuned Telemetry CSV Logging) COMPLETE
