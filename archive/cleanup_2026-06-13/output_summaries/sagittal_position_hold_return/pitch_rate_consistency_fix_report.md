# Pitch Rate Consistency Fix Report

**Date:** 2026-05-31  
**Status:** FIX IMPLEMENTED BUT INEFFECTIVE — ROOT CAUSE AUDIT WAS INCORRECT  
**Recommendation:** REVERT FIX, ACCEPT FALLBACK GATE FROM STEP E CALIBRATION

---

## Executive Summary

The pitch rate consistency estimator was implemented correctly and works as designed. It successfully detects sign mismatches between measured `qvel` pitch rate and finite-difference derivative, and substitutes the FD rate when inconsistent.

**However, the transient peak is unchanged** (0.5951 m vs 0.595 m before), and the wheel acceleration spike still occurs. The root cause audit conclusion was incorrect: the pitch rate artifact at step 1236 was NOT the cause of the Step E transient.

Additionally, the low-pass filter introduces lag that causes height variant regressions — both high_5cm and low_5cm fell, whereas they survived in the previous Step E calibration.

**Recommendation:** Revert the pitch rate fix and accept the fallback final-drift gate (≤0.10 m) from the previous Step E gain calibration (k_position=20.0, max_position_tau=3.0, no pitch rate correction).

---

## What Was Implemented

### 1. PitchRateConsistencyEstimator

**File:** `wheeled_biped/controllers/pitch_rate_consistency_estimator.py`

**Design:**
- Stateful estimator maintains previous pitch angle for finite-difference computation
- Computes FD rate: `pitch_rate_fd = (pitch_x[t] - pitch_x[t-1]) / dt`
- Detects sign mismatch when both rates exceed threshold and have opposite signs
- Substitutes FD rate when mismatch detected
- Applies low-pass filter to corrected rate: `corrected[t] = alpha * corrected[t-1] + (1-alpha) * selected[t]`

**Parameters:**
- `dt`: 0.01 s (100 Hz control)
- `min_rate_for_sign_check`: 0.01 rad/s (threshold for sign check)
- `filter_alpha`: 0.3 (low-pass filter coefficient)

### 2. Integration into Simulation Script

**File:** `scripts/simulate_hierarchical_controller.py`

**Changes:**
- Added CLI args: `--vd-pitch-rate-filter-alpha`, `--vd-pitch-rate-min-sign-check`
- Instantiated estimator after `control_dt` is defined (line ~1640)
- Wired estimator into velocity-damped controller call site (line ~2260)
- Added telemetry fields: `pitch_rate_measured_x_rad_s`, `pitch_rate_fd_x_rad_s`, `pitch_rate_corrected_x_rad_s`, `pitch_rate_consistency_error_rad_s`, `pitch_rate_sign_mismatch`, `pitch_rate_source_used`

### 3. Tests

**File:** `tests/test_pitch_rate_consistency_estimator.py`

**Coverage:**
- 12 tests, all passing
- Sign mismatch detection
- FD rate substitution
- Low-pass filtering
- Near-zero rate handling
- Step E transient scenario reproduction

---

## Results — 5000-Step Nominal Run

| Metric | Before (no fix) | After (with fix) | Change |
|--------|----------------|------------------|--------|
| Max support position error | 0.595 m | 0.5951 m | +0.0001 m |
| Final support position error | 0.053 m | 0.0527 m | -0.0003 m |
| Steady-state (last 1000 steps) | 0.0527 m | 0.0527 m | 0.0 m |
| Transient peak step | 1360 | 1360 | Same |
| Pitch rate artifact step | 1236 | 1235 | -1 step |
| Sign mismatches detected | 1 | 2 | +1 |
| Wheel acc spike at step 1237 | -106.14 rad/s² | -109.56 rad/s² | -3.42 rad/s² |
| Completed 5000 steps | Yes | Yes | Same |

**Conclusion:** The transient peak is essentially unchanged. The pitch rate fix did not eliminate or reduce the transient.

---

## Pitch Rate Estimator Behavior

### Sign Mismatch Detection

**Step 1235:**
- `pitch_x`: 5.578 deg (increasing from 5.545 deg)
- `pitch_rate_measured`: -0.0164 rad/s (negative, artifact)
- `pitch_rate_fd`: +0.0575 rad/s (positive, correct)
- `sign_mismatch`: **True**
- `source_used`: **finite_difference**
- `pitch_rate_corrected`: +0.0571 rad/s (FD rate used)

**Step 1236:**
- `pitch_x`: 5.545 deg (decreasing)
- `pitch_rate_measured`: -0.1056 rad/s (negative, correct)
- `pitch_rate_fd`: -0.0574 rad/s (negative, correct)
- `sign_mismatch`: False
- `source_used`: measured
- `pitch_rate_corrected`: -0.0568 rad/s

**Step 1237:**
- `wheel_acc_left`: -109.56 rad/s² (spike still occurs)

The estimator correctly detected the artifact at step 1235 and used the FD rate. However, the wheel acceleration spike still occurred at step 1237.

---

## Why the Fix Was Ineffective

### 1. Transient Builds Gradually, Not Suddenly

The support position error increases gradually from step 1000 to step 1360:

| Steps | Max error | Final error |
|-------|-----------|-------------|
| 0-1000 | 0.133 m | 0.128 m |
| 1000-2000 | **0.595 m** | 0.039 m |
| 2000-3000 | 0.053 m | 0.053 m |
| 3000-4000 | 0.053 m | 0.053 m |
| 4000-5000 | 0.053 m | 0.053 m |

The transient peak occurs at step 1360, **125 steps after the pitch rate artifact**. The position error builds continuously during the entire pitch excursion (steps 1000-1360), not suddenly at step 1236.

### 2. Wheel Acceleration Spike Occurs After Correction

The wheel acceleration spike at step 1237 occurs **after** the pitch rate correction at step 1235. The correction did not prevent the spike.

### 3. Fundamental TWIP Limitation

The actual root cause is the fundamental TWIP limitation: during large pitch excursions, wheels must move forward to balance the robot. The position-return term fights this motion. Term-level clipping prevents wheel torque saturation but cannot prevent large position errors during pitch disturbances.

The pitch rate artifact at step 1236 was a **symptom**, not the cause.

---

## Height Variant Regression

### Before (Step E calibration, no pitch rate fix)

| Variant | Steps | Max error | Final error | Gate ±0.15m | Survived |
|---------|-------|-----------|-------------|-------------|---------|
| high_5cm | 500 | 0.224 m | 0.220 m | FAIL | **Yes** |
| low_5cm | 500 | 0.083 m | 0.041 m | PASS | **Yes** |

### After (with pitch rate fix, filter_alpha=0.3)

| Variant | Steps | Termination | Survived |
|---------|-------|-------------|---------|
| high_5cm | 500 | orientation_fail_roll_y_-0.80 | **No** |
| low_5cm | 500 | orientation_fail_pitch_x_-0.81 | **No** |

**Regression:** Both height variants fell with the pitch rate fix enabled.

**Root cause:** The low-pass filter (alpha=0.3) is always active, not just during sign mismatches. This introduces lag in the pitch rate signal even during normal operation, degrading damping performance during height perturbations.

---

## Root Cause Audit Was Incorrect

The Step E transient root cause audit (file: `outputs/sagittal_position_hold_return/step_e_transient_root_cause_report.md`) concluded:

> **Primary Classification: B — Pitch Damping Insufficient**  
> **Specific sub-cause:** Pitch rate measurement artifact causing damping sign flip

This conclusion was **incorrect**. The evidence shows:

1. The transient builds gradually over 360 steps, not suddenly at the artifact step
2. The transient peak occurs 125 steps after the artifact
3. Correcting the artifact does not reduce the transient peak
4. The wheel acceleration spike occurs after the correction

The actual root cause is the fundamental TWIP limitation during large pitch excursions, as documented in the Step E gain calibration report.

---

## Verification

| Check | Status |
|-------|--------|
| Estimator detects sign mismatches | ✓ CONFIRMED |
| Estimator substitutes FD rate when inconsistent | ✓ CONFIRMED |
| Estimator applies low-pass filter | ✓ CONFIRMED |
| Transient peak reduced | ✗ FAIL (0.5951 m vs 0.595 m) |
| Wheel acceleration spike prevented | ✗ FAIL (-109.56 rad/s² vs -106.14 rad/s²) |
| Height variants stable | ✗ FAIL (both fell) |
| No WBC changes | ✓ CONFIRMED |
| No E0b/E0c/E0d reintroduced | ✓ CONFIRMED |
| Torque ownership unchanged | ✓ CONFIRMED |
| Ownership violation count = 0 | ✓ CONFIRMED |
| Hidden torque norm = 0.0 | ✓ CONFIRMED |

---

## Recommendation

**Do NOT proceed with the pitch rate consistency fix.**

**Recommended action:**
1. Revert the pitch rate fix (remove estimator, restore direct use of measured pitch_rate)
2. Accept the fallback final-drift gate (≤0.10 m) from the previous Step E gain calibration
3. Document the transient peak (0.595 m) as a known limitation of the position-return approach
4. Proceed to Step C with the configuration from Step E gain calibration:
   - `k_position=20.0`
   - `k_velocity=15.0`
   - `max_position_tau=3.0`
   - No pitch rate correction

**Rationale:**
- The pitch rate fix does not reduce the transient
- The filter introduces lag that causes height variant regressions
- The previous Step E calibration achieved the same steady-state performance (0.053 m) without the regressions
- The transient peak is caused by the fundamental TWIP limitation, not the pitch rate artifact

---

## Files Modified

- `wheeled_biped/controllers/pitch_rate_consistency_estimator.py` (new)
- `scripts/simulate_hierarchical_controller.py` (modified)
- `tests/test_pitch_rate_consistency_estimator.py` (new)

---

## Step C Recommendation

**DO NOT PROCEED** to Step C with the pitch rate fix enabled.

**Proceed to Step C** with the configuration from Step E gain calibration (no pitch rate correction).
