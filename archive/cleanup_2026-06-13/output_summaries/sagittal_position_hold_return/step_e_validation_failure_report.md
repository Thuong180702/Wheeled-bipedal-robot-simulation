# Step E Validation Failure Report

**Date:** 2026-05-31  
**Validation Sequence:** V1 (500 steps), V2 (1000 steps), V3 (2000 steps), V4 (5000 steps)  
**Result:** ALL GATES FAILED  
**Status:** Step E REJECTED - Cannot proceed to Step C

---

## Executive Summary

The support velocity damping fix (k_support_velocity=10.0) successfully reduced velocity oscillations but **failed to prevent position drift and transient excursions**. All validation runs (V2, V3, V4) failed all three acceptance gates due to a **position authority saturation transient** occurring around steps 1000-1500.

**Critical failure:** Max support position error = **0.493m** at step 1411 (14.1 seconds), exceeding the hard minimum gate (0.30m) by **64%**.

---

## Validation Results Summary

### V1: 500 steps (MISLEADING - missed transient)
- **Max error:** 0.085m at step 499
- **Final error:** 0.085m
- **Gate result:** Preferred PASS (max ≤0.10m, final ≤0.05m: FAIL on final only)
- **Issue:** Run too short to capture the transient at steps 1000-1500

### V2: 1000 steps (caught early drift)
- **Max error:** 0.122m at step 999
- **Final error:** 0.122m
- **Gate result:** ALL FAIL
- **Pattern:** Monotonic forward drift at ~6-7 mm/s

### V3: 2000 steps (caught full transient)
- **Max error:** 0.493m at step 1411
- **Final error:** 0.068m
- **Gate result:** ALL FAIL
- **Pattern:** Drift → saturation → explosion → recovery

### V4: 5000 steps (confirmed pattern + steady-state)
- **Max error:** 0.493m at step 1411
- **Final error:** 0.053m
- **Gate result:** ALL FAIL
- **Pattern:** Same transient, then stabilizes at 0.053m after step 2500

---

## Error Evolution Timeline

| Step | V1 (500) | V2 (1000) | V3 (2000) | V4 (5000) | Phase |
|------|----------|-----------|-----------|-----------|-------|
| 100  | 0.060m   | 0.060m    | 0.060m    | 0.060m    | Initialization |
| 500  | 0.085m   | 0.085m    | 0.085m    | 0.085m    | Early drift |
| 1000 | -        | 0.122m    | 0.122m    | 0.122m    | Drift acceleration |
| 1411 | -        | -         | **0.493m** | **0.493m** | **PEAK (saturation)** |
| 2000 | -        | -         | 0.068m    | 0.068m    | Recovery |
| 2500 | -        | -         | -         | 0.047m    | Stabilization |
| 5000 | -        | -         | -         | 0.053m    | Steady-state |

---

## Gate Compliance Results

### Preferred Gate (max ≤0.10m, final ≤0.05m)
- **V1:** FAIL (final 0.085m > 0.05m)
- **V2:** FAIL (max 0.122m > 0.10m, final 0.122m > 0.05m)
- **V3:** FAIL (max 0.493m > 0.10m, final 0.068m > 0.05m)
- **V4:** FAIL (max 0.493m > 0.10m, final 0.053m > 0.05m)

### Fallback Gate (max ≤0.15m, final ≤0.10m)
- **V1:** FAIL (final 0.085m, but max would exceed on longer run)
- **V2:** FAIL (max 0.122m, final 0.122m)
- **V3:** FAIL (max 0.493m)
- **V4:** FAIL (max 0.493m)

### Hard Minimum Gate (max ≤0.30m, final ≤0.10m)
- **V1:** Would FAIL on longer run
- **V2:** FAIL (max 0.122m, final 0.122m)
- **V3:** FAIL (max 0.493m >> 0.30m)
- **V4:** FAIL (max 0.493m >> 0.30m)

**Conclusion:** Even the hard minimum gate failed by 64% margin.

---

## Root Cause Analysis

### Primary Failure Mechanism: Position Authority Saturation

**Transient window (steps 1300-1500):**
- Position authority **saturated for 200+ consecutive steps**
- tau_position_raw commanded: -9.0 to -9.9 Nm
- tau_position_clipped applied: -3.0 Nm (max_position_tau limit)
- **Authority deficit:** ~6-7 Nm unavailable
- Result: Controller could not arrest forward drift

**Saturation analysis (steps 1361-1461, ±50 around peak):**
```
Support velocity range: [-0.052, 0.166] m/s
tau_position_raw range: [-9.866, -9.014] Nm  ← commanded
tau_position_clipped: -3.0 Nm (saturated)     ← applied
tau_support_velocity range: [-1.660, 0.521] Nm
Pitch range: [3.03, 7.82] deg
Position saturation: 100/100 steps (100%)
```

### Failure Sequence

1. **Phase 1 (steps 0-1000): Slow drift**
   - Position error accumulates from 0 → 0.122m
   - Drift rate: ~6-7 mm/s
   - Position control authority: -1.6 Nm mean (not saturated)
   - Velocity damping active but insufficient

2. **Phase 2 (steps 1000-1300): Drift acceleration**
   - Error grows from 0.122m → 0.337m
   - Position controller demands increasing torque
   - Approaching saturation threshold

3. **Phase 3 (steps 1300-1500): Saturation explosion**
   - Position authority saturates at -3.0 Nm
   - Controller demands -9 to -10 Nm but only gets -3.0 Nm
   - Error explodes from 0.337m → 0.493m
   - **Authority deficit prevents correction**

4. **Phase 4 (steps 1500-2500): Recovery**
   - Error large enough that even saturated authority starts working
   - Gradual recovery: 0.493m → 0.047m
   - Saturation ends as error reduces

5. **Phase 5 (steps 2500-5000): Steady-state**
   - Error stabilizes at ~0.053m
   - No further drift or oscillation
   - Position control maintains equilibrium

### Contributing Factors

1. **Insufficient position gain (k_position=20.0)**
   - Too weak to prevent initial drift
   - Requires saturation-level torque to correct moderate errors

2. **Insufficient authority limit (max_position_tau=3.0 Nm)**
   - Saturates when error exceeds ~0.15m
   - Cannot provide the -9 Nm needed during transient

3. **Velocity damping consumes authority**
   - k_support_velocity=10.0 produces up to 1.7 Nm
   - Reduces available authority for position correction
   - During transient: 1.7 Nm velocity + 3.0 Nm position = 4.7 Nm total
   - Still insufficient vs. required ~10 Nm

4. **No integral action**
   - Cannot eliminate steady-state bias
   - Final error settles at 0.053m instead of zero

---

## Failure Classification

**Primary:** `torque_saturation`  
**Secondary:** `position_velocity_conflict_with_pitch`  
**Tertiary:** `insufficient_position_gain`

**NOT:**
- ❌ `support_velocity_gain_too_high` - velocity damping worked as intended
- ❌ `support_velocity_sign_error` - signs correct
- ❌ `oscillatory_position_hunting` - no oscillation observed
- ❌ `wheel_velocity_runaway` - wheels stable
- ❌ `height_regression` - CoM height stable (4.6cm range)
- ❌ `contact_invalid` - contact maintained throughout
- ❌ `metric_reporting_error` - telemetry consistent across runs

---

## Controller Integrity Verification

### WBC Disabled (Correct)
- `tau_wbc_scaled_per_joint`: "0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000" at all steps
- `ownership_violation_count`: 0
- `hidden_torque_norm`: 0.0
- Balance-core mode active as intended

### E0 Logic Disabled (Correct)
- `kp_cp`: 0.0 (capture-point position control disabled)
- Pitch reference active
- Pitch-rate correction filter disabled

### Sagittal Controller Active (Correct)
- `active_sagittal_controller`: velocity-damped
- Baseline SagittalWheelBalanceController inactive
- SagittalVelocityDampedBalanceController active

---

## Posture and Stance Metrics

### Posture Stability (ACCEPTABLE)
- **Pitch range:** 8.36° (acceptable for transient)
- **Roll range:** 2.67° (good)
- **CoM height range:** 4.6 cm (acceptable)
- **No leg collapse or splay**

### Contact State (VALID)
- Both feet maintained ground contact throughout
- No flight phase or tip-over

### Torque Distribution
- **Hip roll max:** 20.79 Nm (during transient)
- **Wheels max:** 1.31 Nm (stable)
- **Legs max:** 0.00 Nm (position control, not direct torque)

---

## Support Velocity Damping Effectiveness

### Velocity Metrics (V2, 1000 steps)
- **Support velocity RMS:** 28.5 mm/s (reduced from previous)
- **Support velocity max:** 118.5 mm/s
- **tau_support_velocity RMS:** 0.285 Nm

### Velocity Metrics (V4, 5000 steps)
- **Support velocity RMS:** Similar to V2
- **tau_support_velocity range:** [-1.660, 0.645] Nm

**Conclusion:** Velocity damping (k=10.0) successfully reduced velocity oscillations but could not prevent position drift due to insufficient position authority.

---

## Comparison to Previous Best

### Previous Run (WBC active, no velocity damping)
- **Max error:** 0.595m at step 1360
- **Final error:** ~0.053m
- **Pattern:** Similar transient, slightly worse peak

### Current Run (WBC disabled, velocity damping active)
- **Max error:** 0.493m at step 1411
- **Final error:** 0.053m
- **Improvement:** 17% reduction in peak error
- **Issue:** Still fails all gates

**Conclusion:** Support velocity damping provides modest improvement but does not solve the fundamental position control problem.

---

## Recommended Fixes (DO NOT IMPLEMENT YET)

### Option 1: Increase Position Authority (Preferred)
```yaml
vd_max_position_tau: 6.0  # was 3.0
```
- Allows controller to apply sufficient torque during transients
- Risk: May increase aggressiveness

### Option 2: Increase Position Gain
```yaml
vd_k_position: 40.0  # was 20.0
```
- Stronger position correction before saturation
- Risk: May cause oscillation or overshoot

### Option 3: Reduce Velocity Damping
```yaml
vd_k_support_velocity: 5.0  # was 10.0
```
- Frees up authority for position control
- Risk: Velocity oscillations may return

### Option 4: Combined Approach
```yaml
vd_k_position: 30.0          # was 20.0
vd_max_position_tau: 5.0     # was 3.0
vd_k_support_velocity: 8.0   # was 10.0
```
- Balanced increase in position authority and gain
- Slight reduction in velocity damping

### Option 5: Add Integral Action
- Implement position error integrator
- Eliminate steady-state bias (0.053m → 0.0m)
- Risk: Integral windup during saturation

---

## Tests Run

### V1: 500 steps nominal
```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 500 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 20.0 \
  --vd-k-velocity 15.0 \
  --vd-k-support-velocity 10.0 \
  --vd-max-position-tau 3.0 \
  --vd-enable-capture-gate \
  --vd-capture-gate-use-cp
```
**Result:** MISLEADING PASS (too short)

### V2: 1000 steps nominal
```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 1000 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 20.0 \
  --vd-k-velocity 15.0 \
  --vd-k-support-velocity 10.0 \
  --vd-max-position-tau 3.0 \
  --vd-enable-capture-gate \
  --vd-capture-gate-use-cp
```
**Result:** ALL GATES FAIL (max 0.122m)  
**Telemetry:** `outputs/hierarchical_controller_sim/telemetry_1780210585.csv`

### V3: 2000 steps nominal
```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 2000 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 20.0 \
  --vd-k-velocity 15.0 \
  --vd-k-support-velocity 10.0 \
  --vd-max-position-tau 3.0 \
  --vd-enable-capture-gate \
  --vd-capture-gate-use-cp
```
**Result:** ALL GATES FAIL (max 0.493m)  
**Telemetry:** `outputs/hierarchical_controller_sim/telemetry_1780211033.csv`

### V4: 5000 steps nominal
```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 5000 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 20.0 \
  --vd-k-velocity 15.0 \
  --vd-k-support-velocity 10.0 \
  --vd-max-position-tau 3.0 \
  --vd-enable-capture-gate \
  --vd-capture-gate-use-cp
```
**Result:** ALL GATES FAIL (max 0.493m)  
**Telemetry:** `outputs/hierarchical_controller_sim/telemetry_1780211559.csv`

---

## Conclusion

**Step E validation FAILED.** The support velocity damping fix (k_support_velocity=10.0) successfully reduced velocity oscillations but **failed to prevent position drift and transient excursions** due to **position authority saturation**.

### Key Findings

1. ✅ **Support velocity damping works** - reduces velocity oscillations
2. ✅ **WBC correctly disabled** - balance-core mode active
3. ✅ **Posture stable** - no collapse or splay
4. ❌ **Position control fails** - max error 0.493m >> 0.30m gate
5. ❌ **Authority saturation** - controller demands 10 Nm, gets 3 Nm
6. ❌ **All gates fail** - preferred, fallback, and hard minimum

### Cannot Proceed to Step C

Step E must pass at least the **hard minimum gate** (max ≤0.30m, final ≤0.10m) before Step C can begin. Current max error (0.493m) exceeds this by **64%**.

### Next Steps

1. **User review required** - do not proceed without explicit direction
2. **Tune controller gains** - increase position authority and/or gain
3. **Re-run V2/V3/V4 sequence** - validate fix addresses saturation
4. **Only after hard minimum gate passes** - proceed to Step C

---

## Appendix: Detailed Metrics

### V2 (1000 steps)
```
Duration: 10.0 s
Support position error: min=-0.0083m, max=0.1215m, final=0.1215m
Support velocity: min=-0.0645m/s, max=0.1185m/s, RMS=0.0285m/s
Pitch range: 3.23°
Roll range: 0.50°
CoM height range: 0.5 cm
tau_position_raw: [-2.430, 0.166] Nm
tau_support_velocity: [-1.185, 0.645] Nm, RMS=0.285 Nm
Position saturation: 0/1000 steps (0.0%)
```

### V3 (2000 steps)
```
Duration: 20.0 s
Support position error: min=-0.0083m, max=0.4933m, final=0.0679m
Peak at step 1411 (14.1 seconds)
Pitch range: 8.36°
Roll range: 2.67°
CoM height range: 4.6 cm
Position saturation near peak: 100/100 steps (100%)
tau_position_raw near peak: [-9.866, -9.014] Nm (saturated at -3.0 Nm)
```

### V4 (5000 steps)
```
Duration: 50.0 s
Support position error: min=-0.0083m, max=0.4933m, final=0.0527m
Peak at step 1411 (same as V3)
Steady-state (steps 2500-5000): ~0.053m
Pitch range: 8.36°
Roll range: 2.67°
CoM height range: 4.6 cm
```

---

**Report generated:** 2026-05-31  
**Status:** Step E REJECTED - awaiting user direction
