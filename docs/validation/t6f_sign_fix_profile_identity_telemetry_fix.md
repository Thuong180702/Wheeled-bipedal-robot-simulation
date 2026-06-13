# T6F Sign Fix Profile Identity Telemetry Fix

**Date**: 2026-06-12  
**Task**: Phase 1 - Fix telemetry identity fields for profile validation

---

## Problem Statement

During Phase 6 500-step diagnostic investigation, telemetry files lacked profile identity fields. This made it impossible to verify which profile produced which telemetry file when analyzing multiple variants.

**Missing fields**:
- `vd_sagittal_authority_profile` - Which sagittal authority schedule was used
- `controller_mode` - Legacy vs balance-core mode
- `sagittal_controller` - Baseline vs velocity-damped
- `height_variant_setup_name` - Which height setup was loaded

**Impact**: Cannot distinguish T5 / T6F / T6F_sign_corrected telemetry without checking filenames or directory structure.

---

## Solution

Added four profile identity fields to telemetry CSV:

```python
# In simulate_hierarchical_controller.py after balance-core columns init:
telemetry.setdefault("controller_mode", [])
telemetry.setdefault("sagittal_controller", [])
telemetry.setdefault("vd_sagittal_authority_profile", [])
telemetry.setdefault("height_variant_setup_name", [])
```

And populate them each step:

```python
if is_balance_core_mode(args):
    telemetry["controller_mode"].append("balance-core")
    telemetry["sagittal_controller"].append(args.sagittal_controller)
    telemetry["vd_sagittal_authority_profile"].append(args.vd_sagittal_authority_profile)
    telemetry["height_variant_setup_name"].append(setup_name)
else:
    telemetry["controller_mode"].append("legacy")
    # ... empty strings for balance-core-only fields
```

---

## Changes Made

### Modified Files

**scripts/simulate_hierarchical_controller.py**:
1. Added profile identity field initialization after balance-core columns (line ~3785)
2. Added profile identity population in main loop (line ~5095)

**tests/test_simulation_telemetry_csv_writer.py**:
1. Added `test_profile_identity_telemetry_fields_exist` test

---

## Verification

### Test Results

```bash
pytest tests/test_simulation_telemetry_csv_writer.py::TestTelemetryCSVWriting::test_profile_identity_telemetry_fields_exist -v
```

**Result**: ✅ PASSED

### Manual Verification

After fix, all new telemetry CSVs will contain:

| controller_mode | sagittal_controller | vd_sagittal_authority_profile | height_variant_setup_name |
|-----------------|---------------------|-------------------------------|---------------------------|
| balance-core    | velocity-damped     | T6F_sign_corrected            | high_0p480_setup          |

---

## Classification

**PROFILE_IDENTITY_TELEMETRY_FIXED**

---

## Next Steps

Proceed to Phase 2: Fix pitch suppression placement bug.

---

## Files Modified

- `scripts/simulate_hierarchical_controller.py` - Added 4 telemetry fields
- `tests/test_simulation_telemetry_csv_writer.py` - Added identity fields test

## Files Created

- `docs/validation/t6f_sign_fix_profile_identity_telemetry_fix.md` (this file)
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_sign_fix_profile_identity_telemetry_fix.json` (pending)
