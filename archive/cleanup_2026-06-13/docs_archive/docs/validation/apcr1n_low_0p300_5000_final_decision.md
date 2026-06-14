# APCR1n Low_0p300 5000-Step Final Decision

**Date:** 2026-06-11  
**Profile:** APCR1n_recenter_priority_torque_boost  
**Setup:** low_0p300  
**Steps:** 5000

## Classification: APCR1N_LOW_0P300_5000_PASS_WITH_MONITORING

## Summary

APCR1n survived 5000 steps at low_0p300 with bounded drift. Key metrics:
- **max |e|:** 0.1714 m (same as 2000-step, no degradation)
- **P2P:** 0.2099 m (acceptable for 5000 steps)
- **outside ±0.15:** 1.1% (53 steps total, all in first 500 steps)
- **final e:** -0.0129 m (near zero, acceptable)
- **drift accumulation:** bounded (ratio 1.10, first 1000 vs last 1000)

## Drift Accumulation Analysis

| Period | mean |e| (m) | max |e| (m) | outside ±0.15 |
|--------|--------------|-----------|---------------|
| First 500 | 0.0702 | 0.1714 | 53 |
| 500-1000 | 0.0518 | 0.1090 | 0 |
| 1000-1500 | 0.0654 | 0.1186 | 0 |
| 1500-2000 | 0.0559 | 0.1188 | 0 |
| 2000-2500 | 0.0567 | 0.1184 | 0 |
| 2500-3000 | 0.0647 | 0.1196 | 0 |
| 3000-3500 | 0.0552 | 0.1220 | 0 |
| 3500-4000 | 0.0537 | 0.1258 | 0 |
| 4000-4500 | 0.0680 | 0.1348 | 0 |
| 4500-5000 | 0.0661 | 0.1431 | 0 |

**Key observations:**
1. The worst drift occurred in the first 500 steps (max 0.1714 m, 53 steps outside ±0.15)
2. After step 500, max |e| stayed below 0.15 m throughout
3. The final 1000 steps (4500-5000) show max |e| = 0.1431 m, well within bounds
4. No drift accumulation trend - drift is bounded and self-correcting

## Stability

| Metric | Value | Status |
|--------|-------|--------|
| survived | 5000/5000 | ✓ |
| contact | 100% double contact | ✓ |
| CoM Z range | 0.271-0.295 m | ✓ |
| pitch range | -2.1 to 7.8 deg | ✓ |
| roll range | 0.0-0.8 deg | ✓ |
| wheel vel max | 4.77 rad/s | ✓ (< 5.0 threshold) |
| wheel vel RMS | 2.41 rad/s | ✓ |

## APCR1n Feature Activation

| Feature | Count | Total | Percentage |
|---------|-------|-------|------------|
| recenter_priority_active | 0 | 5000 | 0.0% |
| position_cap_boost_active | 0 | 5000 | 0.0% |

**Interpretation:** APCR1n-specific features did not activate because drift stayed bounded. The base APCR1h parameters (soft-band mode, drift priority, fast response) are sufficient for this benign scenario.

## Decision

**APCR1N_LOW_0P300_5000_PASS_WITH_MONITORING**

APCR1n passes the 5000-step low_0p300 validation with the following observations:
1. ✓ Survived 5000 steps
2. ✓ No drift accumulation (ratio 1.10 < 1.5 threshold)
3. ✓ max |e| stays bounded after first 500 steps
4. ✓ Contact/height/roll stable
5. ✓ Wheel velocity acceptable
6. ✓ No WBC/hidden/ownership violation

**Classification: PASS_WITH_MONITORING** (not PASS_READY_FOR_HIGH_0P480) because:
- The first 500 steps showed the worst drift behavior (53 steps outside ±0.15)
- There is a slight upward trend in max |e| in the final windows (0.1348, 0.1431)
- While bounded, this warrants monitoring before proceeding to high_0p480

## Recommendation

1. APCR1n is the current best profile for low_0p300
2. Proceed to prepare for high_0p480 evaluation when appropriate
3. Do NOT claim official Step E pass yet
4. Do NOT commit changes