# APCR1g Early Transient Root-Cause Audit

## Date
2026-06-09

## Objective
Understand why APCR1g drifts worse than APCR1f despite having better pitch stability.

## Summary

**Classification: `APCR1G_DRIFT_WORSE_FROM_WRONG_TORQUE_SIGN`**

APCR1g applies **WRONG SIGN TORQUE** when drift exceeds the entry threshold. This is the root cause of the drift regression.

---

## Key Finding

### Critical Torque Sign Analysis

| Condition | APCR1f tau | APCR1g tau | Expected | APCR1f Correct | APCR1g Correct |
|-----------|------------|------------|----------|-----------------|-----------------|
| drift > +0.10 | **-1.00 Nm** | **+1.33 Nm** | negative | 100% | **0%** |
| drift < -0.10 | **+0.40 Nm** | **-0.40 Nm** | positive | ~100% | ~100% |

**When drift is positive (> +0.10 m):**
- **APCR1f applies NEGATIVE torque** → correctly opposes positive drift
- **APCR1g applies POSITIVE torque** → incorrectly accelerates positive drift

This single issue explains the entire drift regression in APCR1g.

---

## Evidence

### Windowed Drift Comparison (0-500 steps)

| Metric | APCR1f | APCR1g | Delta |
|--------|--------|--------|-------|
| max_e (m) | 0.157 | 0.369 | **+0.212** |
| outside ±0.15 (%) | 7.2 | 82.0 | **+74.8** |
| pitch_rms (deg) | 3.81 | 3.70 | -0.11 |
| apc_active (%) | 57.6 | 92.8 | +35.2 |
| wheel_vel_max (rad/s) | 4.69 | 4.20 | -0.49 |
| moving_away (%) | 52.0 | 99.2 | +47.2 |

### Torque Sign When Drift > +0.10

| Metric | APCR1f | APCR1g |
|--------|--------|--------|
| n steps | 652 | 1019 |
| tau mean | **-1.00 Nm** | **+1.33 Nm** |
| tau < 0 (correct) | **652 (100%)** | **0 (0%)** |
| tau > 0 (WRONG) | 0 (0%) | **1019 (100%)** |

### When Drift > 0.15 m

| Metric | APCR1f | APCR1g |
|--------|--------|--------|
| n steps | 45 | 694 |
| mean e_dot | +0.000022 m/step | +0.000002 m/step |
| steps with e_dot < 0 | 22 (48.9%) | 230 (33.1%) |
| mean apc_tau | **-0.91 Nm** | **+1.35 Nm** |

---

## Root Cause Mechanism

### APCR1f (Correct Behavior)
1. Drift exceeds threshold → enters active pitch crossing
2. Positive drift → applies **negative** torque
3. Negative torque → decelerates forward wheel motion
4. Support position moves backward toward zero
5. Drift recovers toward zero

### APCR1g (Wrong Behavior)
1. Drift exceeds threshold → enters active pitch crossing
2. Positive drift → applies **positive** torque
3. Positive torque → accelerates forward wheel motion
4. Support position moves further forward away from zero
5. **Drift INCREASES** rather than decreasing

The APCR1g predictive logic likely reverses the torque sign when it predicts future drift based on pitch dynamics, but this causes the wheel to accelerate in the wrong direction for support recovery.

---

## Why APCR1g Has Better Pitch Stability

Despite wrong support recovery torque, APCR1g has slightly better pitch stability because:
- Higher position torque authority (14.9 vs 6.3 Nm max)
- Lower sagittal pitch torque usage (4.7 vs 6.2 Nm max)
- More active APC engagement (93% vs 62%)

But pitch stability does not compensate for support drift.

---

## Conclusion

APCR1g's drift regression is caused by **applying positive torque when drift is positive**, which accelerates drift rather than opposing it. APCR1f correctly applies negative torque when drift is positive.

The fix for APCR1h must ensure torque sign is correct for support recovery, not just pitch stabilization.

---

## Files Generated

- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1g_early_transient_root_cause_audit.json`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1g_early_transient_events.csv`
