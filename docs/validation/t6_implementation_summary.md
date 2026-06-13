# T6 High-Height Transient Suppression Implementation Summary

**Date:** 2026-06-12  
**Status:** Implementation complete, ready for 2000-step screening

---

## Implementation Complete

**Phase 3 deliverables:**
- ✅ T6A_high_early_hard_band implemented
- ✅ T6B_high_stronger_emergency implemented
- ✅ T6C_high_early_plus_stronger implemented
- ✅ T6D_high_transient_boost (aliased to T6C for now)
- ✅ T6E_high_pitch_aware_boost (aliased to T6C for now)
- ✅ All variants added to JOINT_FIX_PROFILES registry
- ✅ T5 unchanged (verified by diff and tests)
- ✅ 36 tests passed

---

## T6 Variant Configurations

### T6A: High_Early_Hard_Band
**Strategy:** Earlier entry into hard/emergency bands  
**Target root cause:** EMERGENCY_TOO_LATE

**Changes from T5:**
```
desired_band_m: 0.07 (was 0.08)
hard_band_m: 0.085 (was 0.10)
emergency_band_m: 0.105 (was 0.12)
```

**Caps:** Same as T5 (4.0, 4.5, 5.5, 6.5, 7.0 Nm)  
**Damping:** Same as T5 (1.0, 0.50, 0.30, 0.15, 0.10)

---

### T6B: High_Stronger_Emergency
**Strategy:** Stronger authority in high bands  
**Target root cause:** AUTHORITY_TOO_WEAK, DAMPING_TOO_STRONG

**Changes from T5:**
```
Caps:
  desired_cap: 5.8 Nm (was 5.5)
  hard_cap: 7.0 Nm (was 6.5)
  emergency_cap: 8.0 Nm (was 7.0)

Damping:
  hard_scale: 0.10 (was 0.15)
  emergency_scale: 0.05 (was 0.10)
```

**Thresholds:** Same as T5 (0.08, 0.10, 0.12 m)

---

### T6C: High_Early_Plus_Stronger
**Strategy:** Combined T6A + T6B (maximum transient suppression)  
**Target root cause:** All three (EMERGENCY_TOO_LATE + AUTHORITY_TOO_WEAK + DAMPING_TOO_STRONG)

**Changes from T5:**
```
Thresholds (T6A):
  desired_band_m: 0.07
  hard_band_m: 0.085
  emergency_band_m: 0.105

Caps (T6B):
  desired_cap: 5.8 Nm
  hard_cap: 7.0 Nm
  emergency_cap: 8.0 Nm

Damping (T6B + slightly more aggressive desired):
  desired_scale: 0.25 (was 0.30)
  hard_scale: 0.10 (was 0.15)
  emergency_scale: 0.05 (was 0.10)
```

---

### T6D and T6E (Future Work)
**Status:** Currently aliased to T6C for screening

T6D and T6E require additional conditional logic (transient-only and pitch-aware boost) that needs state tracking beyond simple parameter changes. For initial screening, they are aliased to T6C to test the upper bound of authority.

If T6C succeeds, T6D/T6E can be implemented properly with:
- T6D: Transient boost active only during steps 500-3500 at high height
- T6E: Pitch-aware boost when abs(pitch) > 4 deg or abs(pitch_rate) > 0.10 rad/s

---

## Test Results

**All 36 tests passed:**
- ✅ All 5 T6 profiles exist and are opt-in
- ✅ T5 unchanged (thresholds, caps, damping)
- ✅ T6A has tighter thresholds, same caps/damping
- ✅ T6B has stronger caps/damping, same thresholds
- ✅ T6C combines T6A thresholds + T6B caps/damping
- ✅ All T6 variants preserve startup guard (100 steps)
- ✅ All T6 variants preserve safety thresholds
- ✅ All T6 caps bounded (emergency ≤ 8.0 Nm)
- ✅ All T6 damping scales bounded (emergency ≥ 0.05)
- ✅ All T6 variants have tuned telemetry enabled
- ✅ Variant names correct (T6A, T6B, T6C)
- ✅ No WBC path changes

---

## Code Changes

**File:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`

**Added:**
- T6A_HIGH_EARLY_HARD_BAND schedule (after line 1241)
- T6B_HIGH_STRONGER_EMERGENCY schedule
- T6C_HIGH_EARLY_PLUS_STRONGER schedule
- T6D/T6E aliases to T6C
- Registry entries in JOINT_FIX_PROFILES

**Unchanged:**
- T5 (APCR1ND_T5_BAND_LIMITED_BALANCED)
- All APCR1nD baselines
- D2 baseline
- WBC paths
- HY2-DIV defaults

---

## CLI Usage

All T6 variants are opt-in via `--vd-sagittal-authority-profile` flag:

```bash
# T6A: Earlier entry
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile T6A_high_early_hard_band \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 2000 \
  --telemetry-decimation 1 \
  --failure-window-steps 2000 \
  --write-run-summary-sidecar

# T6B: Stronger authority
...--vd-sagittal-authority-profile T6B_high_stronger_emergency...

# T6C: Combined (most aggressive)
...--vd-sagittal-authority-profile T6C_high_early_plus_stronger...

# T6D: Transient boost (currently = T6C)
...--vd-sagittal-authority-profile T6D_high_transient_boost...

# T6E: Pitch-aware (currently = T6C)
...--vd-sagittal-authority-profile T6E_high_pitch_aware_boost...
```

---

## Next Phase: 2000-Step Screening

**Phase 5 plan:**
1. Run all 5 T6 variants at high_0p480 for 2000 steps
2. Compare against T5 baseline (first 2000 steps from 5000-step run)
3. Rank by "outside ±0.08 m %" (primary metric)
4. Select best candidate for 5000-step validation

**Target improvement:**
- T5 high_0p480: 68.5% outside ±0.08 m during steps 500-3500
- T6 goal: ≤30% outside ±0.08 m

**Expected ranking (hypothesis):**
1. T6C (most aggressive, addresses all root causes)
2. T6B (stronger authority, addresses 2/3 root causes)
3. T6A (earlier entry, addresses 1/3 root causes)
4. T5 (baseline reference)

---

**Status:** Phase 3 (Implementation) COMPLETE  
**Next:** Phase 5 (2000-step screening)  
**Date:** 2026-06-12
