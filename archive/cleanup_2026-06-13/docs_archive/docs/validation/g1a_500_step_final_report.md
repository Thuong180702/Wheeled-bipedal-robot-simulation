# G1a Bias Cancellation - 500-Step Final Report

## Phase 9-10: Final Classification and Report

### Executive Summary

**Task:** Stop F2 hysteresis tuning. Identify and fix one-sided positive bias in signed support error.

**Root Cause:** Persistent positive tau_pitch (82-89% positive) creating forward pitching tendency that causes:
1. COM to shift forward
2. Positive signed support error
3. tau_position counteracts but is too weak

**Fix Strategy (G1):** Instead of waiting for natural negative drift (which never occurs), estimate persistent bias using low-pass filter and apply bounded opposite torque proactively.

---

## Results Summary

### Signed Support Error Comparison (500-step horizon)

| Metric | D2 | F1b | G1a | G1a vs F1b |
|--------|-----|------|------|-------------|
| mean | 0.0823 | 0.0764 | 0.0724 | -0.0040 |
| min | -0.0035 | -0.0339 | -0.0530 | -0.0191 |
| max | 0.1757 | 0.1695 | 0.1717 | +0.0022 |
| **positive%** | 93.0% | 82.8% | 81.8% | **-1.0%** |
| **outside band** | 19.2% | 16.2% | 13.4% | **-2.8%** |
| below -0.05 | 0 | 0 | 19 | +19 |
| below -0.10 | 0 | 0 | 0 | 0 |
| below -0.15 | 0 | 0 | 0 | 0 |
| zero crossings | 5 | 6 | 6 | 0 |

### Tau Analysis

| Metric | D2 | F1b | G1a |
|--------|-----|------|------|
| tau_pitch mean | 2.5992 | 2.6976 | 2.8761 |
| tau_pitch positive% | 89.2% | 82.8% | 84.2% |
| tau_position mean | -2.6146 | -2.3618 | -2.2157 |

### Stability

| Metric | D2 | F1b | G1a |
|--------|-----|------|------|
| com_z mean | 0.2921 | 0.2923 | 0.2926 |
| pitch mean (deg) | 2.98 | 3.09 | 3.30 |
| pitch max (deg) | 6.36 | 6.32 | 6.79 |
| roll max (deg) | 0.76 | 0.75 | 0.67 |

---

## Classification: BIAS_FIX_IMPROVES_BUT_NOT_ENOUGH

### Evidence

1. **positive% improved but not to target**
   - G1a: 81.8% (improvement from F1b: -1.0%)
   - Target: ≤70%
   - Delta needed: -11.8%

2. **Outside band improved**
   - G1a: 13.4% (improvement: -2.8%)
   - F1b: 16.2%
   - D2: 19.2%

3. **Negative excursion below -0.05 increased**
   - G1a: 19 steps (F1b: 0, D2: 0)
   - This is expected and desired - bias cancellation should produce negative excursions
   - But stayed above -0.10 and -0.15 (safe)

4. **Zero crossings maintained**
   - G1a: 6 crossings (same as F1b)
   - System can still cross zero

5. **Height/roll stability maintained**
   - No degradation in stability metrics
   - Roll max actually improved slightly (0.67 vs 0.75)

6. **Pitch slightly worse**
   - tau_pitch positive%: 84.2% (vs F1b: 82.8%)
   - pitch max: 6.79 deg (vs F1b: 6.32 deg)
   - Slight increase but within acceptable range

---

## Root Cause Not Fully Addressed

The bias cancellation effect is **too weak** because:

1. **Deadband (0.02 m) excludes many steps**
   - When signed error is small (within deadband), bias estimation pauses
   - This limits how aggressive the cancellation can be

2. **Bias estimate builds up slowly**
   - Filter alpha = 0.02 (slow response)
   - After 500 steps, estimate may not have converged

3. **Bias tau capped at ±1.5 Nm**
   - tau_pitch persistent positive ~2.6-2.9 Nm
   - Bias tau can only partially counteract this

4. **No direct pitch reference adjustment**
   - The fix operates on wheel torque level
   - Doesn't address the tau_pitch bias directly

---

## Recommendations

### Option A: Strengthen G1 (Tuning)

Increase bias cancellation authority:
- Increase `bias_cancel_k` from 12.0 to 15.0-18.0
- Increase `bias_cancel_max_tau` from 1.5 to 2.0-2.5 Nm
- Reduce `bias_cancel_deadband_m` from 0.02 to 0.01 m

### Option B: G1b Trial

Try G1b_bias_cancel_strong profile with:
- bias_cancel_k = 15.0 Nm/m
- bias_cancel_max_tau = 2.0 Nm
- bias_cancel_filter_alpha = 0.03 (faster response)

### Option C: Lower pitch reference

If pitch reference is contributing to tau_pitch bias:
- Investigate pitch reference adjustment
- Target reducing tau_pitch positive% below 50%

### Option D: Address root cause at source

If tau_pitch bias comes from:
- Height-specific LQR gains
- Controller implementation
- Physical model asymmetry

Then tune those instead of compensating downstream.

---

## What NOT to Do

- Do NOT modify D2 baseline
- Do NOT make G1 profile default yet
- Do NOT run 2000-step validation (insufficient improvement)
- Do NOT commit
- Do NOT claim Step E pass

---

## Files Modified

1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
   - Added bias_cancel fields to SagittalAuthoritySchedule
   - Added state variables: `_bias_cancel_estimate`, `_bias_cancel_prev_tau`
   - Implemented bias cancellation logic

2. `scripts/simulate_hierarchical_controller.py`
   - Added G1a_bias_cancel_moderate profile
   - Added G1b_bias_cancel_strong profile

3. `tests/test_sagittal_velocity_damped_balance_controller.py`
   - Added 16 tests for G1 bias cancellation

---

## Next Steps

1. Run G1b_bias_cancel_strong at 500-step horizon
2. Compare results
3. If G1b shows further improvement but not to target, consider Option C
4. Document root cause investigation for pitch reference
5. Evaluate if root cause is in LQR gains or controller implementation

