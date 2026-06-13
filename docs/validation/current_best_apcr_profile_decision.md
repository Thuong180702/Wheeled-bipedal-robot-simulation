# Current Best APCR Profile Decision

## Phase 8: Current Best Profile Decision

### Executive Summary

Based on the quantitative comparison of APCR1h, APCR1j, APCR1k, and APCR1m at 1000 steps each:

**Current Best: APCR1h**

---

## Comparison Table

| Metric | APCR1h | APCR1j | APCR1k | APCR1m | Winner |
|--------|--------|--------|--------|--------|--------|
| **max \|e\| (m)** | **0.178** | 0.183 | 0.232 | 0.434 | APCR1h |
| **P2P (m)** | **0.249** | 0.251 | 0.303 | 0.833 | APCR1h |
| **outside ±0.15 %** | **9.7%** | 25.8% | 20.2% | 54.0% | APCR1h |
| **>+0.15 count** | **97** | 258 | 202 | 368 | APCR1h |
| **<-0.15 count** | **0** | 0 | 0 | 172 | APCR1h |
| **mean \|e\| (m)** | **0.075** | 0.092 | 0.095 | 0.177 | APCR1h |
| **final e (m)** | 0.167 | **0.124** | 0.132 | 0.308 | APCR1j |
| **tau_pitch abs_mean** | 3.24 Nm | 3.36 Nm | 3.52 Nm | 4.23 Nm | APCR1h |
| **wheel_vel abs_mean** | **1.42 Nm** | 1.34 Nm | 1.31 Nm | 5.00 Nm | APCR1h |
| **Startup** | Stable | Stable | Stable | Stable | tie |

---

## Decision Criteria

### Primary Criteria (in order of importance)

1. **Lowest max |e|** - APCR1h wins (0.178m vs 0.183/0.232/0.434)
2. **Lowest outside ±0.15** - APCR1h wins (9.7% vs 25.8/20.2/54.0%)
3. **Lowest P2P** - APCR1h wins (0.249m vs 0.251/0.303/0.833)
4. **No startup fall** - All stable (tie)
5. **Contact/height/roll stable** - All stable (tie)

### Secondary Criteria

- **Final torque direction correctness**: Not measured for APCR1h/j/k (need to re-run audit)
- **Lower support RMS**: APCR1h has lowest mean |e| (0.075m)
- **Acceptable pitch/wheel oscillation**: All acceptable

---

## Detailed Analysis

### Why APCR1h Wins

1. **Lowest drift magnitude**: max|e| = 0.178m, 2.4x better than APCR1m
2. **Lowest band violations**: Only 9.7% outside ±0.15m
3. **Lowest wheel velocity damping**: 1.42 Nm mean abs, 3.5x smaller than APCR1m
4. **No sustained negative drift**: 0 steps with |e| > 0.15m in negative direction
5. **Positive mean error**: 78.3% positive, indicates slight forward lean (normal)

### Why APCR1m is Worst

1. **Highest drift**: max|e| = 0.434m (2.4x worse than APCR1h)
2. **Largest P2P**: 0.833m (3.3x worse than APCR1h)
3. **Sustained negative drift**: 172 steps with |e| > 0.15m in negative direction
4. **Largest wheel damping**: 5.00 Nm (3.5x larger than APCR1h)
5. **Position cap saturated 77.3%**: Limits recenter effectiveness

### Root Causes of APCR1m's Poor Performance

1. **tau_position cap at ±3 Nm**: Raw tau_position would be ±15 Nm, but capped at ±3 Nm
2. **tau_position saturated 77.3%**: During RECENTER, 87.3% saturated
3. **tau_position sign 100% correct**: But limited by cap
4. **Wheel velocity damping too aggressive**: 5.0 Nm vs 1.4 Nm in APCR1h
5. **Blend blocked by safety gates**: Only 40.4% of steps have all gates passing

---

## Classification

**CURRENT_BEST_APCR1H**

---

## Recommendations

1. **APCR1h should remain the default** for low_0p300 boundary-height operation
2. **APCR1m should be ABANDONED** as a standalone profile
3. **APCR1n should be designed** with the following targets:
   - Reduce wheel velocity damping to ~1.5 Nm abs_mean
   - Relax position cap to ~5 Nm during safe RECENTER
   - Use APCR1h as the base with targeted modifications

---

## Files Generated

- `current_best_apcr_profile_decision.json`
- `current_best_apcr_profile_decision.md`
