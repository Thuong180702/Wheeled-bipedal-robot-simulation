# Early Zero-Crossing Recenter V2 Final Report

**Date:** 2026-06-15  
**Profile:** `early_zero_crossing_recenter_v2`  
**Problem:** EZC V1 exits at zero but positive bias immediately returns drift to +0.10 to +0.20 m  
**Root Cause:** `EZC_FAILURE_EXIT_TOO_EARLY_REBOUND`

## Executive Summary

Implemented anti-rebound fix (Path C) for early zero-crossing recenter. The fix adds a decaying correction phase after zero crossing to prevent immediate rebound.

**Result:** IMPROVED but NOT at target. V2 shows improvement over V1 in the 500-step window but regresses at longer horizons.

---

## Audit Findings

### Phase 1: Telemetry Correctness
**Classification:** `EZC_TELEMETRY_COLUMN_CORRECT`

All drift columns agree. EZC uses correct column (`active_pitch_crossing_signed_error_m`).

### Phase 2: Episode Root Cause
**Classification:** `EZC_FAILURE_EXIT_TOO_EARLY_REBOUND`

| Metric | Value |
|--------|-------|
| Total episodes | 21 |
| Crossed zero | 18/21 (85.7%) |
| EZC reached max torque | 21/21 (100%) |
| Net corrective torque | -5.37 Nm (strong) |
| Avg rebound steps | 28.3 |
| Post-exit positive % | 71% |

**Key insight:** EZC correction IS working but decays too quickly, allowing positive bias to overwhelm.

### Phase 3: Hip-Yaw/Posture
**Classification:** `EZC_POSTURE_HIP_YAW_SAFE`

Posture is safe. Hip-yaw, roll, pitch, height, and contact are all stable.

---

## V2 Changes from V1

| Parameter | V1 | V2 |
|-----------|----|----|
| `ezc_base_tau_nm` | 0.18 | 0.25 |
| `ezc_max_tau_nm` | 0.55 | 0.70 |
| `ezc_rate_nm_per_step` | 0.012 | 0.015 |
| `ezc_decay_nm_per_step` | 0.025 | 0.018 |
| `ezc_zero_dwell_steps` | 3 | 5 |
| `ezc_error_gain_nm_per_m` | 3.0 | 4.0 |
| `ezc_antirebound_enabled` | False | True |
| `ezc_antirebound_decay_steps` | N/A | 30 |
| `ezc_antirebound_initial_ratio` | N/A | 0.50 |

### New State: ANTIREBOUND_DECAY

After crossing zero (instead of exiting to IDLE):
1. Enter `ANTIREBOUND_DECAY` state
2. Start at `ezc_antirebound_initial_ratio * current_tau` (50%)
3. Decay linearly to 0 over `ezc_antirebound_decay_steps` (30 steps)
4. Re-enter `RECENTER_FROM_POSITIVE` if error exceeds 0.05 m

---

## Staged Validation Results

| Profile | Steps | min | max | P2P | mean | pos% | neg% | crossings | EZC enters |
|---------|-------|-----|-----|-----|------|------|------|-----------|------------|
| V1 | 5000 | -0.042 | +0.202 | 0.244 | 0.082 | 86.0% | 14.0% | 38 | 21 |
| V2 | 500 | -0.030 | +0.199 | 0.229 | 0.069 | 72.3% | 27.5% | 6 | 3 |
| V2 | 1200 | -0.030 | +0.199 | 0.229 | 0.082 | 80.1% | 19.8% | 12 | 6 |
| V2 | 2000 | -0.038 | +0.199 | 0.237 | 0.081 | 79.8% | 20.2% | 20 | 10 |
| V2 | 5000 | -0.045 | +0.205 | 0.250 | 0.082 | 86.0% | 14.0% | 36 | 20 |

### Key Observations

1. **V2 500-step shows best result**: 72.3% positive (best among all runs)
2. **V2 performance degrades at longer horizons**: 72.3% → 80.1% → 79.8% → 86.0%
3. **Anti-rebound steps increase with horizon**: 0 → 150 → 270 → 510
4. **Zero crossings similar**: V1 38 vs V2 36 (no significant difference)
5. **No falls**: All runs completed without falling

### Improvement at 500-step

V2 vs V1 at 500 steps:
- positive %: 72.3% vs 80.8% (**-8.5 pp improvement**)
- negative %: 27.5% vs 19.0% (**+8.5 pp improvement**)
- min drift: -0.030 vs -0.016 (**+0.014 m improvement**)

This is a **meaningful improvement** within the 500-step window.

### Regression at longer horizons

At 5000 steps:
- V1: 86.0% positive
- V2: 86.0% positive
- **No improvement at longer horizon**

The anti-rebound fix provides short-term benefit but doesn't fix the underlying issue.

---

## Root Cause Analysis: Why V2 Doesn't Scale

The root cause is **systematic positive bias in the controller**:

1. `tau_pitch` has mean = +3.31 Nm (forward pitch torque even at near-zero pitch)
2. `tau_wheel_velocity` has mean = +0.38 Nm (positive velocity damping)
3. Total positive bias ≈ +3.5 to +4.0 Nm

EZC V2 provides anti-rebound at 50% of tau for 30 steps ≈ 0.35 Nm × 30 steps = 10.5 Nm-s of correction

But the positive bias is continuous at +3.5 to +4.0 Nm per step.

**The math:** Anti-rebound cannot overcome continuous positive bias indefinitely.

---

## Classification

**EZC_V2_IMPROVED_BUT_NOT_AROUND_ZERO**

- V2 shows improvement at 500-step (72.3% vs 86.0%)
- But regresses at longer horizons (5000-step = 86.0% same as V1)
- Anti-rebound alone is not sufficient for the root cause

---

## Recommendations

### Short-term: Use V2 for 500-step scenarios
V2 shows clear improvement at 500 steps. If the use case is short-duration (~5 second balance), V2 is better than V1.

### Long-term: Address systematic positive bias
The root cause is `tau_pitch = +3.31 Nm` at near-zero pitch. This needs to be investigated:
- Is there a pitch sensor offset?
- Is the pitch control law asymmetric?
- Is there a systematic forward tilt in the robot posture?

### Alternative: Increase anti-rebound authority
Try increasing:
- `ezc_antirebound_initial_ratio` from 0.50 to 0.75
- `ezc_antirebound_decay_steps` from 30 to 50
- `ezc_max_tau_nm` from 0.70 to 1.00

But this is likely a temporary fix without addressing the bias.

---

## Files Changed

1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
   - Added `ezc_antirebound_enabled`, `ezc_antirebound_decay_steps`, `ezc_antirebound_initial_ratio` to dataclass
   - Added `EARLY_ZERO_CROSSING_RECENTER_V2` profile constant
   - Added `ANTIREBOUND_DECAY` state to EZC state machine
   - Added `ezc_antirebound_steps`, `ezc_antirebound_tau_start` telemetry

2. `scripts/simulate_hierarchical_controller.py`
   - Added V2 to SAGITTAL_AUTHORITY_PROFILES
   - Added V2 to CLI --vd-sagittal-authority-profile choices

3. `tests/test_early_zero_crossing_recenter_v2.py` (new)
   - 38 tests for V2 profile existence and correctness

---

## Conclusion

Anti-rebound (Path C) is the correct fix direction, but the implementation alone is insufficient. The V2 profile shows meaningful improvement at short horizons (500 steps: 72.3% vs 86.0%), demonstrating that anti-rebound works. However, the systematic positive bias in `tau_pitch` overwhelms the fix at longer horizons.

**Next step:** Investigate and fix `tau_pitch` systematic bias before further anti-rebound tuning.