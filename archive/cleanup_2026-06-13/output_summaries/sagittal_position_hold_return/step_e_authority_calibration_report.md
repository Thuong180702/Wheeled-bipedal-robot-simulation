# Step E Authority Calibration Report

**Date:** 2026-05-31  
**Objective:** Find controller gains that pass at least hard minimum gate (max ≤0.30m, final ≤0.10m)  
**Result:** **FAILED** - All candidates failed hard minimum gate

---

## Executive Summary

Tested four gain combinations across 1000-step and 2000-step validations. **All candidates failed** the hard minimum gate due to position authority saturation during the transient at steps 1400-1500.

**Key Finding:** Increasing position gain from 20.0 to 25.0 made the problem WORSE by increasing demanded torque from 10 Nm to 11 Nm, while max_position_tau=4.0 remained insufficient.

**Root Cause:** Position controller demands 10-11 Nm during transient but authority limit clips it to 3-4 Nm, causing error explosion to 0.43-0.49m.

---

## Torque Budget Analysis

Analyzed baseline V4 telemetry during saturation window (steps 1300-1500):

```
Position Authority:
  tau_position_raw range: [-9.866, -6.731] Nm
  tau_position_clipped: -3.000 Nm (100% saturated)
  Authority deficit: 6.87 Nm

Support Velocity Damping:
  tau_support_velocity RMS: 1.193 Nm

Total Wheel Torque:
  max_tau_wheel: 0.20 Nm << 10.0 Nm limit

DIAGNOSIS: Final wheel torque NOT saturated
→ Position term saturation is the TRUE LIMITER
→ Safe to increase max_position_tau to 6-7 Nm
```

**Conclusion:** Plenty of torque budget available. Position authority is the bottleneck.

---

## Candidate Results

### 1000-Step Validation

| Candidate | k_position | k_support_velocity | max_position_tau | Max Error | Final Error | Pos Sat % | Gate |
|-----------|------------|-------------------|------------------|-----------|-------------|-----------|------|
| A         | 20.0       | 10.0              | 4.0              | 0.1215m   | 0.1215m     | 0.0%      | FAIL |
| B         | 20.0       | 10.0              | 5.0              | 0.1215m   | 0.1215m     | 0.0%      | FAIL |
| C         | 25.0       | 10.0              | 4.0              | 0.1006m   | 0.1006m     | 0.0%      | FAIL |
| D         | 25.0       | 8.0               | 4.0              | 0.1015m   | 0.1015m     | 0.0%      | FAIL |

**Observations:**
- All show monotonic drift, no transient captured (too short)
- Increasing max_position_tau alone (A→B) had NO effect
- Increasing k_position (C, D) reduced drift slightly but insufficient
- 1000 steps too short to evaluate - transient occurs at 1400-1500

### 2000-Step Validation

| Candidate | k_position | k_support_velocity | max_position_tau | Max Error | Max Step | Final Error | Pos Sat % | Gate |
|-----------|------------|-------------------|------------------|-----------|----------|-------------|-----------|------|
| C         | 25.0       | 10.0              | 4.0              | 0.4314m   | 1451     | 0.0379m     | 25.8%     | FAIL |
| D         | 25.0       | 8.0               | 4.0              | 0.4421m   | 1443     | 0.0361m     | 24.9%     | FAIL |

**Critical Finding:**
- Transient still occurs around step 1400-1500
- Max error 0.43-0.44m >> 0.30m hard minimum gate
- Position authority STILL saturates (~25% of steps)
- Controller demands **11.0 Nm** but only gets 4.0 Nm
- **Authority deficit: 7.0 Nm**

**Counterproductive Effect:**
Increasing k_position from 20.0 to 25.0 made the controller MORE aggressive, demanding MORE torque (11 Nm vs baseline 10 Nm), which worsened saturation.

---

## Comparison to Baseline

| Metric | Baseline (k=20, tau=3.0) | Best Candidate C (k=25, tau=4.0) | Change |
|--------|--------------------------|----------------------------------|--------|
| Max error | 0.493m at step 1411 | 0.431m at step 1451 | -12.6% |
| Final error | 0.053m | 0.038m | -28.3% |
| Position saturation | 100% (steps 1300-1500) | 25.8% (steps 1400-1500) | -74% |
| Demanded torque | 9.9 Nm | 11.1 Nm | +12% ⚠️ |

**Conclusion:** Modest improvement but still fails all gates. Increasing position gain was counterproductive.

---

## Failure Analysis

### Why Candidates Failed

1. **Insufficient Authority:** max_position_tau=4.0 Nm << demanded 11 Nm
2. **Aggressive Gain:** k_position=25.0 increased demand instead of reducing error
3. **No Integral Action:** Cannot eliminate steady-state bias
4. **Velocity Damping Conflict:** k_support_velocity=10.0 consumes ~1.2 Nm, reducing available authority

### Why Increasing Gain Made It Worse

Higher k_position → larger correction torque → hits saturation sooner → larger error during saturation → even larger correction demand → vicious cycle

---

## Recommended Next Steps

### Option 1: Substantial Authority Increase (Preferred)

```yaml
vd_k_position: 20.0          # Keep moderate gain
vd_k_velocity: 15.0
vd_k_support_velocity: 10.0
vd_max_position_tau: 7.0     # Increase to 7.0 Nm (covers 11 Nm demand with margin)
```

**Rationale:**
- Torque budget analysis confirms 7.0 Nm is safe (final wheel torque only 0.2 Nm)
- Allows controller to apply demanded torque without saturation
- Avoids aggressive gain that increases demand

**Risk:** May increase aggressiveness, but torque budget has headroom

### Option 2: Add Integral Action

Implement position error integrator to eliminate steady-state bias:

```python
position_error_integral += position_error * dt
tau_integral = k_integral * position_error_integral
tau_position = k_position * position_error + tau_integral
```

**Rationale:**
- Eliminates 0.05m steady-state error
- Reduces need for high proportional gain
- Standard control technique

**Risk:** Integral windup during saturation - needs anti-windup

### Option 3: Combined Moderate Approach

```yaml
vd_k_position: 22.0          # Slight increase
vd_k_velocity: 15.0
vd_k_support_velocity: 8.0   # Reduce to free authority
vd_max_position_tau: 6.0     # Moderate increase
```

**Rationale:**
- Balanced increase in gain and authority
- Reduced velocity damping frees ~0.5 Nm
- Less aggressive than Option 1

**Risk:** May still saturate if demand exceeds 6.0 Nm

---

## Tests Run

### Candidate A: 1000 steps
```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 1000 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 20.0 \
  --vd-k-velocity 15.0 \
  --vd-k-support-velocity 10.0 \
  --vd-max-position-tau 4.0 \
  --vd-enable-capture-gate \
  --vd-capture-gate-use-cp
```
**Result:** Max error 0.1215m, FAIL (monotonic drift, no transient)

### Candidate B: 1000 steps
```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 1000 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 20.0 \
  --vd-k-velocity 15.0 \
  --vd-k-support-velocity 10.0 \
  --vd-max-position-tau 5.0 \
  --vd-enable-capture-gate \
  --vd-capture-gate-use-cp
```
**Result:** Max error 0.1215m, FAIL (identical to A - authority increase had no effect)

### Candidate C: 1000 steps + 2000 steps
```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 2000 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 25.0 \
  --vd-k-velocity 15.0 \
  --vd-k-support-velocity 10.0 \
  --vd-max-position-tau 4.0 \
  --vd-enable-capture-gate \
  --vd-capture-gate-use-cp
```
**Result:** Max error 0.4314m at step 1451, FAIL (transient captured, still saturates)

### Candidate D: 1000 steps + 2000 steps
```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 2000 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 25.0 \
  --vd-k-velocity 15.0 \
  --vd-k-support-velocity 8.0 \
  --vd-max-position-tau 4.0 \
  --vd-enable-capture-gate \
  --vd-capture-gate-use-cp
```
**Result:** Max error 0.4421m at step 1443, FAIL (similar to C)

---

## Controller Integrity

✓ WBC disabled (balance-core mode active)  
✓ E0 logic disabled (kp_cp=0.0)  
✓ Baseline sagittal controller inactive  
✓ Velocity-damped controller active  
✓ No ownership violations  
✓ No hidden torque

---

## Conclusion

**Step E authority calibration FAILED.** All tested candidates failed the hard minimum gate (max ≤0.30m).

**Root Cause:** Position authority saturation. Controller demands 10-11 Nm during transient but max_position_tau=4.0 Nm is insufficient.

**Key Insight:** Increasing position gain (k_position=20→25) was counterproductive - it increased demanded torque from 10 Nm to 11 Nm, worsening saturation.

**Recommendation:** Increase max_position_tau to 7.0 Nm with moderate k_position=20.0, OR implement integral action to reduce proportional gain requirement.

**Cannot proceed to Step C** until at least hard minimum gate passes.

---

**Report generated:** 2026-05-31  
**Status:** Step E FAILED - awaiting user direction on next tuning iteration
