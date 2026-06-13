# Phase 4 Complete: APCR1nD Tuned Variants Testing

**Date:** 2026-06-12  
**Status:** ✅ COMPLETE

---

## Summary

Phase 4 testing is complete with all 31 tuned variant tests passing. Implementation has been verified to match design specification. The tuned variants are ready for Phase 5 (2000-step simulation runs).

---

## Accomplishments

### 1. Added Missing Telemetry Field

- **Issue:** `tuned_band_state_id` was missing from telemetry
- **Fix:** Modified `_compute_tuned_band_state()` to return tuple `(band_name, band_id)`
- **Result:** All 19 tuned telemetry fields now present

### 2. Created Comprehensive Test Suite

- **File:** `tests/test_apcr1nd_tuned_variants.py`
- **Tests:** 31 comprehensive tests
- **Coverage:**
  - Profile existence and opt-in behavior
  - Baseline unchanged verification
  - T1-T5 parameter verification
  - Safety gate preservation
  - Telemetry field existence
  - Band state computation
  - WBC path preservation

### 3. Updated CLI for Phase 5

- **File:** `scripts/simulate_hierarchical_controller.py`
- **Changes:**
  - Added 5 tuned variant profiles to `--vd-sagittal-authority-profile` choices
  - Added imports for T1-T5 profile constants
  - Added profiles to `SAGITTAL_AUTHORITY_PROFILES` dictionary

---

## Test Results

**Total:** 31 tests  
**Passed:** 31  
**Failed:** 0  
**Duration:** 1.71s

**Existing tests:** 285/285 still passing (4.28s)

**Classification:** APCR1ND_TUNED_TESTS_PASS

---

## Files Modified (Phase 4)

1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
   - Fixed `_compute_tuned_band_state()` to return band ID
   - Added `tuned_band_state_id` to telemetry (line 3690)

2. `tests/test_apcr1nd_tuned_variants.py` (NEW)
   - Created 31 comprehensive tests
   - Verified all five tuned variants
   - Verified baseline unchanged
   - Verified telemetry fields

3. `scripts/simulate_hierarchical_controller.py`
   - Added T1-T5 imports (lines 67-82)
   - Added T1-T5 to argparser choices (lines 2537-2591)
   - Added T1-T5 to SAGITTAL_AUTHORITY_PROFILES dict (lines 1303-1308)

4. `docs/validation/apcr1nd_tuned_variant_tests_report.md` (NEW)
   - Comprehensive test report

5. `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1nd_tuned_variant_tests_summary.json` (NEW)
   - JSON test summary

---

## Ready for Phase 5

✅ **Implementation complete**  
✅ **All tests pass**  
✅ **CLI updated for simulation runs**  
✅ **Telemetry complete (19 fields)**  
✅ **Baseline unchanged**  
✅ **Backward compatible**

---

## Phase 5 Commands

Run each tuned variant at low_0p300 for 2000 steps:

```bash
# T5 Band-Limited Balanced (recommended)
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile APCR1nD_T5_band_limited_balanced \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 2000 \
  --telemetry-decimation 1 \
  --failure-window-steps 2000

# T1 Early Entry
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile APCR1nD_T1_early_entry \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 2000 \
  --telemetry-decimation 1 \
  --failure-window-steps 2000

# T2 Hold Outside Band
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile APCR1nD_T2_hold_outside_band \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 2000 \
  --telemetry-decimation 1 \
  --failure-window-steps 2000

# T3 Early Entry Plus Hold
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile APCR1nD_T3_early_entry_plus_hold \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 2000 \
  --telemetry-decimation 1 \
  --failure-window-steps 2000

# T4 Stronger Authority
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile APCR1nD_T4_stronger_authority \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 2000 \
  --telemetry-decimation 1 \
  --failure-window-steps 2000
```

After each run, move telemetry to variant-specific directory:

```powershell
Move-Item outputs/hierarchical_controller_sim/telemetry_*.csv `
  outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/tuned_2000_APCR1nD_T5_band_limited_balanced/
```

---

## Implementation Summary

**Phase 3:** Implementation ✅  
**Phase 4:** Testing ✅  
**Phase 5:** 2000-step simulations (READY)  
**Phase 6:** Analysis (pending Phase 5 data)  
**Phase 7:** Best variant selection (pending Phase 5 data)  
**Phase 8:** Final report (pending Phases 5-7)

---

**Next:** Run Phase 5 simulations to generate telemetry data for comparative analysis.
