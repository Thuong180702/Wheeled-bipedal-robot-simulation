# Hip-Yaw HY-FF Integration Bug Fix Summary

**Date:** 2026-06-04  
**Status:** BUG FIXED - Ready for Phase 5 Re-evaluation

---

## Bug Description

**Original Issue:** HY-FF (Hip-Yaw Support-Error Feedforward) compensation never activated during Phase 5 evaluation, causing all candidates to produce identical results to baseline.

**Symptoms:**
- `hip_yaw_comp_height_gate = 0.000` (should be ~1.0 at low_0p300)
- `hip_yaw_comp_support_error_m = 0.000` (should be ~0.24)
- `hip_yaw_comp_tau_left/right = 0.000` (consequence of above)
- All HY-FF candidates (k=2.0, 4.0, 6.0, 8.0) produced byte-for-byte identical results

---

## Root Cause Analysis

### Timing Issue

**Controller execution order:**
1. Line 3061: `shape_posture.compute()` called
2. Line 3245: `sagittal_wheel_balance.compute()` called  
   → `sagittal_diag["support_position_error_m"]` populated here

**Problem:** Shape controller runs BEFORE sagittal controller computes support error.

### Height Source Issue

**Original code at line 3068:**
```python
target_com_height=float(height_variant_setup.get("target_com_z_m", height_cmd))
```

This correctly passed `target_com_z_m = 0.300` from setup JSON.

**Height gate function receives 0.300m correctly**, so height source was NOT the bug.

### Support Error Source Issue  

**Original code at line 3067:**
```python
support_position_error=sagittal_diag.get("support_position_error_m", 0.0)
```

**Problem:** `sagittal_diag` is initialized as `{}` at line 3027, then `shape_posture.compute()` runs at line 3061 BEFORE `sagittal_wheel_balance.compute()` populates it at line 3245.

Result: `.get("support_position_error_m", 0.0)` returns default `0.0` every step.

---

## Fix Implementation

### Solution: Previous-Step Support Error

Since shape controller must run before sagittal controller (controller execution order is architecturally fixed), use **previous-step support error** for HY-FF.

This introduces a **1-step delay** (5ms at 200Hz control rate) which is acceptable for this feedforward compensation.

### Code Changes

**File:** `scripts/simulate_hierarchical_controller.py`

**1. Initialize previous-step tracking (line 2377):**
```python
prev_support_error = 0.0  # Previous-step support position error for HY-FF (m)
```

**2. Add to nonlocal variables (line 2667):**
```python
nonlocal ... prev_support_error
```

**3. Pass previous-step support error to shape controller (line 3067):**
```python
support_position_error=prev_support_error,  # Use previous-step (sagittal computes after shape)
```

**4. Update previous-step support error after sagittal completes (line 3308):**
```python
# Update previous-step support error for next iteration's HY-FF
prev_support_error = sagittal_diag.get("support_position_error_m", 0.0)
```

**5. Add debug telemetry (8 new columns):**
- `hy_ff_height_passed_to_shape`
- `hy_ff_support_error_passed_to_shape`
- `hy_ff_support_error_from_sagittal`
- `hy_ff_prev_support_error`
- `hy_ff_setup_target_com_z_m`
- `hy_ff_setup_achieved_com_z_m`
- `hy_ff_root_z_m`
- `hy_ff_current_com_z_m`

---

## Smoke Test Verification

**Test configuration:**
- Variant: low_0p300
- Steps: 200
- HY-FF: enabled, k=2.0, tau_max=1.0, sign=+1.0

**Results:**

| Metric | Before Fix | After Fix | Status |
|--------|------------|-----------|--------|
| `hip_yaw_comp_height_gate` | 0.000 | 1.000 | ✓ FIXED |
| `hip_yaw_comp_support_error_m` max | 0.000 | 0.2372 | ✓ FIXED |
| `hip_yaw_comp_tau_left` max | 0.000 | 0.4745 | ✓ FIXED |
| `hip_yaw_comp_tau_right` min | 0.000 | -0.4745 | ✓ FIXED |

**Debug telemetry confirms:**
- Height passed to shape: 0.300 m (correct from setup)
- Support error passed to shape: up to 0.2372 m (correct)
- Support error from sagittal: up to 0.2375 m (matches)
- Setup target CoM z: 0.300 m (correct)
- Root z: 0.394-0.397 m (NOT used for gate, correct)

**Verdict:** ✓✓✓ INTEGRATION BUG FIXED ✓✓✓

---

## Impact Analysis

### What Changed

- HY-FF now uses previous-step support error (5ms delay)
- Height gate activation logic unchanged
- Compensation computation unchanged
- Controller execution order unchanged

### What Did NOT Change

- No WBC added
- No hip-roll modification
- No global gain changes
- No variant-name patches
- No discontinuous schedules
- No threshold relaxation

### One-Step Delay Acceptability

**Delay magnitude:** 5ms (1 control step at 200Hz)

**Acceptable because:**
1. Support error changes slowly (~0.24m developed over 200 steps = 1 second)
2. HY-FF is feedforward compensation, not feedback control
3. Shape controller PD gains provide immediate feedback
4. 5ms delay << support error development timescale (1000ms)
5. Industry practice: feedforward often uses filtered/delayed signals

**Alternative rejected:** Reordering controllers would require extensive validation and risk breaking existing authority budget and composer logic.

---

## Next Steps

1. ✓ **Smoke test passed** - Compensation activates
2. **Phase 5 re-evaluation in progress** - Testing all candidates
3. **Acceptance criteria validation** - Check if any candidate passes:
   - hip_yaw_abs_max <= 0.07 rad
   - percent(hip_yaw > 0.10) = 0%
   - support_position_error does not worsen by >10%
   - pitch/roll/height/contact gates pass
   - WBC applied = false
   - ownership violations = 0

---

## Files Modified

1. `scripts/simulate_hierarchical_controller.py`
   - Added `prev_support_error` tracking
   - Updated `shape_posture.compute()` call
   - Added 8 debug telemetry columns
   - Added telemetry logging

2. `scripts/analyze_hy_ff_smoke_test.py` (created)
   - Smoke test verification script

---

## Restrictions Compliance

✓ All restrictions satisfied:
- No WBC added
- No legacy WBC enabled
- No hip-roll modified
- No global hip-yaw gain changes
- No variant-name patches
- No discontinuous schedules
- No threshold relaxation
- No Step D progression
- No BOUNDARY_RANGE_PASS claimed

---

**Bug Fix Status:** COMPLETE  
**Integration Status:** VERIFIED  
**Ready for:** Phase 5 Re-evaluation
