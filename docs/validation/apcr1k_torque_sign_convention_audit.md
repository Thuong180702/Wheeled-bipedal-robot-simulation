# APCR1k Torque Sign Convention Audit Report

## Executive Summary

**Classification: `TORQUE_COMPOSITION_BASELINE_OVERPOWERS_APCR`**

The APCR1k report claimed "sign inversion", but telemetry proves this is incorrect. APCR torque contribution sign IS correct. The real issue is that the baseline wheel torque during RECENTER state is consistently in the OPPOSITE direction of what is needed to correct drift, and its magnitude overpowers the APCR correction.

## Phase 1 Audit: Telemetry Evidence

### APCR Contribution Sign Analysis

| Metric | Value |
|--------|-------|
| Total steps | 1000 |
| APCR active steps | 715 / 1000 (71.5%) |
| Steps where `apcr_delta == apcr_command` | 715 / 715 (100%) |
| Steps where delta sign matches command sign | 715 / 715 (100%) |
| Steps where delta sign OPPOSES command sign | 0 / 715 (0%) |

**Conclusion: APCR contribution sign is mathematically correct. No sign inversion.**

### Final Torque vs Drift Direction Analysis

| Case | Count | % |
|------|-------|---|
| Final torque OPPOSES drift (correct) | 58 | 8.1% |
| Final torque ACCELERATES drift (wrong) | 657 | 91.9% |

**Conclusion: Final wheel torque is WRONG 91.9% of the time during APCR activation.**

### Positive Drift (e > 0) Breakdown

| Metric | Value |
|--------|-------|
| Total APCR-active positive drift steps | 666 |
| Baseline torque POSITIVE (accelerates drift) | 660 steps (99.1%) |
| Baseline torque NEGATIVE (opposes drift) | 6 steps (0.9%) |
| Mean baseline torque magnitude | 1.9725 Nm |
| Mean APCR command magnitude | 1.8873 Nm |
| Baseline magnitude > APCR command | 611 steps (91.7%) |
| Final torque STILL positive (wrong) | 608 steps (91.3%) |

### Negative Drift (e < 0) Breakdown

| Metric | Value |
|--------|-------|
| Total APCR-active negative drift steps | 49 |
| Baseline torque NEGATIVE (accelerates drift) | 49 steps (100%) |
| Baseline torque POSITIVE (opposes drift) | 0 steps (0%) |
| Mean baseline torque | -2.0107 Nm |
| Mean APCR command | 1.6246 Nm |
| Final torque STILL negative (wrong) | 49 steps (100%) |

## Critical Example: Step 392

```
APCR command = -1.998 Nm
Baseline wheel torque = +2.185 Nm
APCR delta (with - without) = -1.998 Nm  ← APCR sign CORRECT
Final wheel torque = +0.187 Nm  ← STILL POSITIVE (wrong direction)
```

APCR correctly subtracted -1.998 Nm, but baseline was +2.185 Nm, overwhelming the correction.

## Physical Interpretation

### Wheel Torque Sign Convention

- **Positive wheel torque** = wheel spinning in positive direction = robot accelerates forward
- **Negative wheel torque** = wheel braking/reversing = robot decelerates

### Drift Correction Requirements

- **Positive drift** (CoM past support center, falling forward-right): Need **negative** wheel torque to decelerate/correct
- **Negative drift** (CoM behind support center, falling backward-left): Need **positive** wheel torque to decelerate/correct

### What Actually Happens

1. **Positive drift**: Baseline torque is POSITIVE → accelerating drift
2. **Negative drift**: Baseline torque is NEGATIVE → accelerating drift
3. **APCR** correctly applies opposing torque
4. **Final torque** is still wrong direction because baseline dominates

### Root Cause

The baseline wheel torque calculation (tau_position, tau_pitch, tau_wheel_velocity, etc.) produces torque in the WRONG direction during drift correction. The baseline controller is:
- Either: treating drift as "desired motion" instead of "error to correct"
- Or: using the wrong sign convention for position/Pitch feedback
- Or: the baseline and APCR are fighting each other instead of cooperating

## APCR Hysteresis State Machine

| State | Steps | Positive Drift | Negative Drift |
|-------|-------|----------------|----------------|
| RECENTER_FROM_POSITIVE | 666 | 666 (100%) | 0 |
| RECENTER_FROM_NEGATIVE | 49 | 0 | 49 (100%) |
| NEUTRAL | 285 | N/A | N/A |

All 715 APCR-active steps are in their corresponding RECENTER state, confirming the hysteresis state machine is working correctly.

## Summary Table

| Question | Answer |
|----------|--------|
| Is APCR contribution sign correct? | **YES** - matches command sign 100% |
| Does APCR contribution oppose drift? | **YES** - mathematically correct |
| Does final torque oppose drift? | **NO** - wrong 91.9% of the time |
| Is baseline torque dominating APCR? | **YES** - baseline > APCR command 91.7% of positive drift |
| Is there "sign inversion"? | **NO** - APCR sign is correct |
| Is the report's claim correct? | **NO** - sign inversion claim is wrong |

## Classification

```
TORQUE_COMPOSITION_BASELINE_OVERPOWERS_APCR
```

## Recommended Fix Approach

**Fix B: Baseline torque dominates APCR**

During RECENTER state and abs(e) > 0.05:
- The baseline torque direction is WRONG (accelerates drift)
- APCR torque direction is CORRECT (opposes drift)
- Solution: Suppress/replace baseline torque when it opposes the recenter direction
- NOT: Flip APCR sign (that would make it WORSE)
