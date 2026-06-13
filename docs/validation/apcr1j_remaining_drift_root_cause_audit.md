# APCR1j Remaining Drift Root Cause Audit

## Summary

**Classification: APCR1J_REMAINING_DRIFT_FROM_TORQUE_TRANSMISSION_LOSS**

APCR1j produces 2.0 Nm torque command, but final wheel torque is limited to ~1.64 Nm (82% transmission ratio). Additionally, entry at 0.08 m allows drift momentum to accumulate before recentering begins.

## Key Findings

### 1. Entry Timing (LATE ENTRY)

| Threshold | First Step | Entry e (m) |
|-----------|------------|-------------|
| 0.03 | 36 | 0.0301 |
| 0.05 | 46 | 0.0521 |
| **0.08** | **58** | **0.0817** |
| 0.10 | 66 | 0.1026 |
| 0.12 | 73 | 0.1205 |
| 0.15 | 87 | 0.1506 |

**Problem:** RECENTER starts at step 58 when e = 0.0817 m. By then, drift has already accumulated significant momentum.

### 2. Torque Transmission Loss (PRIMARY CAUSE)

| Metric | APCR Command | Final Wheel |
|--------|-------------|-------------|
| Max torque | 2.0000 Nm | 1.6386 Nm |
| Reaches 2.0 Nm | 411 steps | 0 steps |
| Transmission ratio | - | **0.8193** |

**Problem:** APCR reaches 2.0 Nm at 411 steps, but final wheel torque is capped at 1.64 Nm (~18% loss).

### 3. Gate Interference (NOT A CAUSE)

- Large error steps (|e| > 0.08): 531
- APCR inactive during large error: **0**
- Pitch blocked: 0
- Contact blocked: 0
- Height blocked: 0

**Finding:** Safety gates are NOT blocking APCR activation. This is NOT a gate interference problem.

### 4. Hysteresis Episodes

| Episode | Entry Step | Entry e | Max e | Duration | Max APCR tau | Max Final tau |
|---------|-----------|---------|-------|----------|--------------|---------------|
| 1 | 58 | 0.0817 | 0.1826 | 177 | 2.0000 | 0.2238 |
| 2 | 354 | 0.0813 | 0.1802 | 163 | 2.0000 | 0.2129 |
| 3 | 621 | 0.0826 | 0.1679 | 162 | 2.0000 | 0.2559 |
| 4 | 891 | 0.0812 | 0.1654 | 109 | 2.0000 | 0.2335 |

**Observation:** Even with 2.0 Nm APCR command, final wheel torque peaks at only 0.22-0.26 Nm during recentering episodes.

## Root Cause Analysis

### Primary Cause: Torque Transmission Loss

The APCR torque is being reduced by ~18% in the downstream path. This could be due to:

1. **Torque blending**: APCR tau is added to other torque terms and may be partially cancelled
2. **Rate limiting**: Wheel torque rate limiter reduces peak torque
3. **Final clipping**: Other constraints clip the final wheel torque
4. **Torque budget**: Position authority budget limits available torque

### Secondary Cause: Late Entry at 0.08 m

By the time RECENTER starts (step 58, e = 0.0817 m), drift has accumulated momentum. Earlier entry at 0.05 m would catch drift before it reaches 0.08 m.

## Comparison: APCR1j vs APCR1h

APCR1h achieved max_e = 0.1572 m vs APCR1j's 0.1826 m. The difference may be:
- Different transmission path
- Different torque blending behavior
- Different wheel dynamics

## APCR1k Solution

Lowering entry threshold from 0.08 m to 0.05 m will:
1. Start RECENTER 12 steps earlier (step 46 vs step 58)
2. Catch drift before it accumulates to 0.08 m
3. Reduce maximum drift by preventing momentum buildup

This does NOT fix the torque transmission loss, but it addresses the late entry problem.

## Recommendations

1. **Implement APCR1k**: Lower outer_enter_m from 0.08 to 0.05
2. **Investigate torque transmission loss**: Understand why final tau is only 82% of APCR command
3. **Do NOT increase torque cap further**: APCR1j already reaches 2.0 Nm; the issue is transmission, not command

## Classification

**APCR1J_REMAINING_DRIFT_FROM_TORQUE_TRANSMISSION_LOSS**

This classification is assigned because:
1. APCR reaches 2.0 Nm correctly
2. Final wheel torque is only 1.64 Nm (18% loss)
3. Gate interference is 0 (not a factor)
4. Late entry is a contributing factor but not the primary cause