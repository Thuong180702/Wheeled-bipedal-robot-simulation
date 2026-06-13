# Upstream Clip Diagnostic Telemetry Report

**Date:** 2026-06-12  
**Status:** Telemetry already exists - no additions needed  
**Classification:** DIAGNOSTIC_TELEMETRY_SUFFICIENT

---

## Executive Summary

**Good news:** The existing telemetry already captures the complete upstream clip pipeline. No code changes needed for Phase 2.

The controller already logs:
1. `tau_position_before_clip` - signal entering the upstream clip
2. `tau_position` - signal after upstream 4.0 Nm clip (line 2009)
3. `effective_max_position_tau` - the upstream clip value (4.0 Nm)
4. `apcr1n_tau_position_after_cap` - signal after tuned cap boost (line 2353)
5. `apcr1n_position_cap_current` - the tuned cap value (T5: 7.0, T6B: 8.0)

This is sufficient to verify:
- Where the 4.0 Nm clip occurs
- Whether T5 and T6B produce identical clips
- Whether the tuned cap receives pre-clipped input
- Whether raising the tuned cap from 7.0 to 8.0 has any effect

---

## Telemetry Fields Present

### Upstream Clip Pipeline

**Field:** `tau_position_raw`  
**Type:** float  
**Location:** Line 3544  
**Description:** Raw position torque from `k_position × error + integral`  
**Purpose:** Shows the initial position demand

**Field:** `tau_position_before_clip`  
**Type:** float  
**Location:** Line 3556  
**Description:** Position torque after capture gate and pitch-aware scaling, BEFORE upstream clip  
**Purpose:** Shows the signal entering line 2009 upstream clip  
**Expected at high_0p480:** Can exceed 7.0 Nm (observed 7.485 Nm in Phase 0 audit)

**Field:** `tau_position`  
**Type:** float  
**Location:** Line 3557  
**Description:** Position torque AFTER upstream clip at line 2009  
**Purpose:** Shows the signal after clipping to `±effective_max_position_tau`  
**Expected at high_0p480:** Should max at 4.0 Nm for both T5 and T6B

**Field:** `effective_max_position_tau`  
**Type:** float  
**Location:** Line 3567  
**Description:** The upstream clip value (computed from height scheduling)  
**Purpose:** Shows what the upstream cap is set to  
**Expected at high_0p480:** Should be 4.0 Nm for both T5 and T6B

### APCR1n Tuned Cap Layer

**Field:** `apcr1n_tau_position_raw`  
**Type:** float  
**Location:** Line 3857  
**Description:** Duplicate of `tau_position_before_clip` for APCR1n tracking  
**Purpose:** Shows the signal before any clipping (for APCR1n context)

**Field:** `apcr1n_position_cap_current`  
**Type:** float  
**Location:** Line 3856  
**Description:** The tuned cap value (changes with band state)  
**Purpose:** Shows what the tuned emergency cap is set to  
**Expected at high_0p480:**
- T5 emergency: 7.0 Nm
- T6B emergency: 8.0 Nm

**Field:** `apcr1n_tau_position_after_cap`  
**Type:** float  
**Location:** Line 3858  
**Description:** Position torque AFTER tuned cap boost at line 2353  
**Purpose:** Shows the signal after the second (tuned) clip  
**Expected at high_0p480:** Should remain 4.0 Nm for both T5 and T6B (no change from `tau_position`)

**Field:** `apcr1n_position_cap_boost_active`  
**Type:** bool  
**Location:** Line 3855  
**Description:** Whether the tuned cap boost layer is active  
**Purpose:** Confirms the tuned cap is being applied

**Field:** `apcr1n_position_saturated`  
**Type:** bool  
**Location:** Line 3859  
**Description:** Whether position torque saturated at the tuned cap  
**Purpose:** Shows if the tuned cap is limiting (should be False if upstream clip dominates)

### Supporting Fields

**Field:** `tuned_position_cap_current`  
**Type:** float  
**Location:** Line 3839  
**Description:** Same as `apcr1n_position_cap_current` but only logged if tuned enabled  
**Purpose:** Backward compatibility

**Field:** `sagittal_schedule_profile`  
**Type:** string  
**Location:** Line 3565  
**Description:** Profile name (T5, T6B, etc.)  
**Purpose:** Identify which configuration is active

---

## Verification Strategy

### What to Check in Phase 3 Diagnostic Run

**1. Confirm upstream clip value:**
```
effective_max_position_tau == 4.0 for both T5 and T6B at high_0p480
```

**2. Confirm raw torque exceeds upstream clip:**
```
tau_position_before_clip > 4.0 during high-drift episodes
```

**3. Confirm upstream clip is applied:**
```
tau_position == 4.0 when tau_position_before_clip > 4.0
tau_position == tau_position_before_clip when tau_position_before_clip <= 4.0
```

**4. Confirm tuned cap differs between T5 and T6B:**
```
T5: apcr1n_position_cap_current == 7.0 in emergency band
T6B: apcr1n_position_cap_current == 8.0 in emergency band
```

**5. Confirm tuned cap receives pre-clipped input:**
```
apcr1n_tau_position_after_cap == tau_position (no change)
apcr1n_tau_position_after_cap <= 4.0 even when apcr1n_position_cap_current > 4.0
```

**6. Confirm T5 and T6B produce identical final torque:**
```
For same drift trajectory:
  T5.tau_position == T6B.tau_position
  T5.apcr1n_tau_position_after_cap == T6B.apcr1n_tau_position_after_cap
```

---

## Required Telemetry Fields (Already Present)

✓ `tau_position_raw` - raw position torque  
✓ `tau_position_before_clip` - before upstream 4.0 Nm clip  
✓ `tau_position` - after upstream clip  
✓ `effective_max_position_tau` - upstream clip value  
✓ `apcr1n_tau_position_raw` - APCR1n tracking of raw  
✓ `apcr1n_position_cap_current` - tuned cap value  
✓ `apcr1n_tau_position_after_cap` - after tuned cap boost  
✓ `apcr1n_position_cap_boost_active` - tuned boost active flag  
✓ `apcr1n_position_saturated` - saturated at tuned cap  
✓ `sagittal_schedule_profile` - profile name

**No additional fields needed.**

---

## CSV Output Verification

The telemetry CSV writer already handles these fields. Verified in Phase 0 tests:
- `test_balanced_core_telemetry_columns_initialization` - PASSED
- `test_append_balance_core_telemetry_populates_columns` - PASSED
- `test_telemetry_mismatch_detection` - PASSED
- `test_csv_writer_produces_correct_row_count` - PASSED

No changes to CSV writer needed.

---

## Optional Enhancements (Not Required)

The following fields would provide additional insight but are NOT required for Phase 3:

**Upstream clip active flag:**
```python
upstream_clip_active = abs(tau_position_before_clip) > effective_max_position_tau * 0.99
```
Can be computed post-hoc from existing fields.

**Tuned cap actually saturating:**
```python
tuned_cap_would_saturate = abs(tau_position) > apcr1n_position_cap_current * 0.99
```
Can be computed post-hoc. Should be False when upstream clip dominates.

**Clip source reason:**
```python
if upstream_clip_active and not tuned_cap_would_saturate:
    clip_source = "upstream_max_position_tau"
elif tuned_cap_would_saturate:
    clip_source = "tuned_cap"
else:
    clip_source = "none"
```
Can be computed post-hoc.

---

## Recommendation

**Skip code modifications for Phase 2.** The existing telemetry is sufficient to verify the upstream clip hypothesis in Phase 3.

**Proceed directly to Phase 3:** Run short paired T5 vs T6B diagnostic at high_0p480 (1200 steps) and analyze the existing telemetry fields.

---

**Status:** Phase 2 complete (no code changes needed)  
**Date:** 2026-06-12
