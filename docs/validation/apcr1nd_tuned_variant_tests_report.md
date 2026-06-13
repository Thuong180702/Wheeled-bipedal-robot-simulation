# APCR1nD Tuned Variant Tests Report

**Date:** 2026-06-12  
**Phase:** 4 (Testing)  
**Status:** ✅ PASS

---

## Overview

Added 31 comprehensive tests for APCR1nD tuned variants (T1-T5). All tests pass, confirming implementation matches design specification.

---

## Test Summary

**Total tests:** 31  
**Passed:** 31  
**Failed:** 0  
**Duration:** 1.71s

---

## Test Coverage

### 1. Profile Existence and Registry (3 tests)

✅ `test_all_five_tuned_profiles_exist` - All five tuned profiles exist in registry  
✅ `test_all_tuned_profiles_are_opt_in` - All tuned variants have `apcr1nd_tuned_enabled=True`  
✅ `test_apcr1nd_baseline_not_tuned` - APCR1nD baseline has `apcr1nd_tuned_enabled=False`

### 2. APCR1nD Baseline Unchanged (1 test)

✅ `test_apcr1nd_baseline_unchanged` - Verifies APCR1nD baseline parameters remain unchanged:
- `recenter_priority_direct_enter_m = 0.08`
- `recenter_priority_direct_exit_m = 0.02`
- `position_cap_normal_nm = 4.0`
- `position_cap_recenter_nm = 5.0`
- `position_cap_emergency_nm = 6.0`

### 3. T1 Early Entry (3 tests)

✅ `test_t1_early_entry_thresholds` - T1 enters at 0.06 (earlier than baseline 0.08)  
✅ `test_t1_release_logic` - T1 releases at inner band 0.02  
✅ `test_t1_does_not_hold_outside_band` - T1 does not use hold logic

### 4. T2 Hold Outside Band (4 tests)

✅ `test_t2_hold_outside_band_enabled` - T2 has `hold_outside_band=True`  
✅ `test_t2_desired_band_threshold` - T2 desired band is 0.08  
✅ `test_t2_release_inner_band` - T2 releases at 0.05  
✅ `test_t2_entry_threshold` - T2 enters at 0.08 (same as baseline)

### 5. T3 Early Entry + Hold (3 tests)

✅ `test_t3_early_entry_plus_hold` - T3 combines early entry (0.06) and hold logic  
✅ `test_t3_strict_release` - T3 uses strict release threshold (0.03)  
✅ `test_t3_converging_release_steps` - T3 uses 20-step converging counter

### 6. T4 Stronger Authority (3 tests)

✅ `test_t4_stronger_position_caps` - T4 uses stronger caps (4.0, 6.0, 7.0 Nm)  
✅ `test_t4_aggressive_damping` - T4 uses aggressive damping (desired=0.20, hard=0.10)  
✅ `test_t4_early_entry` - T4 enters at 0.06

### 7. T5 Band-Limited Balanced (5 tests)

✅ `test_t5_graduated_position_caps` - T5 graduated caps: 4.0 → 4.5 → 5.5 → 6.5 → 7.0 Nm  
✅ `test_t5_graduated_damping_scales` - T5 graduated damping: 1.0 → 0.50 → 0.30 → 0.15 → 0.10  
✅ `test_t5_band_thresholds` - T5 defines all five bands (0.05, 0.06, 0.08, 0.10, 0.12)  
✅ `test_t5_preserves_damping_if_helps` - T5 has `preserve_damping_if_helps=True`  
✅ `test_t5_hold_and_strict_release` - T5 uses hold logic and 0.03 release threshold

### 8. Hold Outside Band Behavior (1 test)

✅ `test_hold_outside_band_profiles` - T2, T3, T5 use hold logic; T1, T4 do not

### 9. Safety Gates Preserved (3 tests)

✅ `test_tuned_variants_preserve_startup_guard` - All variants use 100-step startup guard  
✅ `test_tuned_variants_preserve_safety_thresholds` - All variants preserve CoM/roll/pitch thresholds  
✅ `test_tuned_variants_enable_recenter_priority` - All variants enable recenter priority features

### 10. Telemetry Fields (3 tests)

✅ `test_tuned_telemetry_fields_exist` - All 19 tuned telemetry fields exist in diagnostics  
✅ `test_tuned_telemetry_variant_name` - Each variant reports correct name (T1-T5)  
✅ `test_band_state_computation` - Band state computed correctly for all five levels

**Tuned telemetry fields verified:**
1. `tuned_variant_name`
2. `tuned_recenter_active`
3. `tuned_band_state`
4. `tuned_band_state_id`
5. `tuned_abs_error`
6. `tuned_error_rate`
7. `tuned_moving_away`
8. `tuned_converging`
9. `tuned_release_allowed`
10. `tuned_active_reason`
11. `tuned_block_reason`
12. `tuned_position_cap_current`
13. `tuned_wheel_damping_scale`
14. `tuned_wheel_damping_override_active`
15. `tuned_outside_band_active`
16. `tuned_outside_band_inactive`
17. `tuned_recenter_held`
18. `tuned_release_counter`
19. `tuned_final_torque_direction_correct`

### 11. No WBC/HY2-DIV Changes (2 tests)

✅ `test_tuned_variants_no_wbc_path_change` - All variants preserve WBC gates  
✅ `test_tuned_variants_produce_wheel_output` - All variants produce correct wheel torque

---

## Implementation Verification

### ✅ Code Quality Checks

- **Clean implementation:** Reuses existing APCR1nD infrastructure
- **No code duplication:** All variants share common logic with config-driven branching
- **Type safe:** All config fields properly typed in dataclass
- **Backward compatible:** APCR1nD baseline unchanged, all old tests pass (285/285)

### ✅ Design Match Verification

1. **T1 release logic:** Confirmed 0.02 matches APCR1nD baseline - design match ✓
2. **Telemetry field count:** Added missing `tuned_band_state_id` - now complete (19 fields) ✓
3. **T4 authority:** Confirmed caps (4.0, 6.0, 7.0) and damping (0.20, 0.10) - design match ✓
4. **T5 authority:** Confirmed all graduated caps and damping scales - design match ✓
5. **APCR1nD baseline:** Confirmed unchanged - design match ✓

### ✅ Existing Tests Still Pass

**Main controller tests:** 285/285 passed in 4.28s

All existing APCR1nD tests continue to pass, confirming backward compatibility.

---

## Classification

**APCR1ND_TUNED_TESTS_PASS**

All 31 tuned variant tests pass. Implementation matches design specification. Ready for Phase 5 (2000-step simulation runs).

---

## Next Steps

**Phase 5:** Run 2000-step tuned variant study
- T1 Early Entry
- T2 Hold Outside Band
- T3 Early Entry + Hold
- T4 Stronger Authority
- T5 Band-Limited Balanced

Each variant will be evaluated at low_0p300 for 2000 steps with full telemetry to measure band control performance.

---

**Test suite location:** `tests/test_apcr1nd_tuned_variants.py`  
**Test count:** 31  
**Status:** ✅ All tests pass  
**Ready for Phase 5:** Yes
