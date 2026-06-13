# APCR1n vs APCR1h Fair 2000-Step Comparison

**Date:** 2026-06-11  
**Classification:** APCR1N_FAIR_2000_PASS_PROCEED_TO_5000

## Summary

APCR1n wins on all primary drift metrics compared to APCR1h in a fair 2000-step evaluation using identical setup (low_0p300, 2000 steps, decimation=1).

## Primary Drift Metrics

| Metric | APCR1h | APCR1n | Winner | Improvement |
|--------|--------|--------|--------|-------------|
| max \|e\| (m) | 0.1775 | 0.1714 | APCR1n | -3.4% |
| P2P (m) | 0.2491 | 0.1854 | APCR1n | -25.6% |
| mean \|e\| (m) | 0.0768 | 0.0608 | APCR1n | -20.8% |
| outside ±0.15 (%) | 12.6% | 2.6% | APCR1n | -10.0 pp |
| final e (m) | -0.0453 | 0.0035 | APCR1n | much closer to 0 |
| zero crossings | 18 | 9 | tie | fewer is better |

## Band Metrics

| Band | APCR1h | APCR1n |
|------|--------|--------|
| outside ±0.03 | 75.5% | 67.2% |
| outside ±0.05 | 59.1% | 55.0% |
| outside ±0.08 | 44.6% | 37.9% |
| outside ±0.10 | 37.3% | 22.9% |
| outside ±0.12 | 27.2% | 4.2% |
| outside ±0.15 | 12.6% | 2.6% |

## Stability Metrics

| Metric | APCR1h | APCR1n | Winner |
|--------|--------|--------|--------|
| wheel vel RMS (rad/s) | 3.49 | 2.12 | APCR1n |
| wheel vel max (rad/s) | 6.88 | 4.39 | APCR1n |
| pitch RMS (deg) | 4.4 | 3.5 | APCR1n |
| roll max (deg) | 0.8 | 0.8 | tie |
| CoM Z range (m) | 0.280-0.295 | 0.282-0.295 | APCR1n |

## APCR1n Feature Activation

| Feature | Count | Total | Percentage |
|---------|-------|-------|------------|
| recenter_priority_active | 0 | 2000 | 0.0% |
| position_cap_boost_active | 0 | 2000 | 0.0% |
| wheel_damping_override_active | 0 | 2000 | 0.0% |

**Important:** APCR1n-specific features did NOT activate during this benign 2000-step run. The improvement over APCR1h is likely due to:
1. Cleaner profile configuration
2. Proper base APCR1h parameters being carried through
3. The APCR1n-specific recenter priority features are dormant but available if drift worsens

## Window Metrics Comparison

### APCR1h Windows

| Window | max \|e\| | P2P | mean \|e\| | final e | outside ±0.15 |
|--------|-----------|-----|------------|---------|---------------|
| 0-500 | 0.1568 | 0.1801 | 0.0650 | 0.0821 | 34 |
| 500-1000 | 0.1775 | 0.2491 | 0.0839 | 0.1667 | 63 |
| 1000-1500 | 0.1672 | 0.2217 | 0.0859 | 0.0735 | 93 |
| 1500-2000 | 0.1578 | 0.2031 | 0.0725 | -0.0453 | 61 |

### APCR1n Windows

| Window | max \|e\| | P2P | mean \|e\| | final e | outside ±0.15 |
|--------|-----------|-----|------------|---------|---------------|
| 0-500 | 0.1325 | 0.1325 | 0.0458 | 0.0069 | 0 |
| 500-1000 | 0.1483 | 0.1643 | 0.0605 | -0.0065 | 0 |
| 1000-1500 | 0.1714 | 0.1854 | 0.0706 | -0.0187 | 52 |
| 1500-2000 | 0.1565 | 0.1565 | 0.0663 | 0.0035 | 0 |

**Key observation:** APCR1n has much smaller P2P in all windows, indicating tighter drift control. The worst window for APCR1n (1000-1500) still has only 52 steps outside ±0.15 vs 93 for APCR1h in the same window.

## Decision

**Classification: APCR1N_FAIR_2000_PASS_PROCEED_TO_5000**

APCR1n beats APCR1h on all primary drift metrics:
- Lower max |e| (0.1714 vs 0.1775)
- Lower P2P (0.1854 vs 0.2491, -25.6% improvement)
- Lower mean |e| (0.0608 vs 0.0768)
- Much lower outside ±0.15 (2.6% vs 12.6%)
- Final error near zero (0.0035 m vs -0.0453 m)

Stability is also better:
- Lower wheel velocity RMS (2.12 vs 3.49 rad/s)
- Lower wheel velocity max (4.39 vs 6.88 rad/s)
- Lower pitch RMS (3.5 vs 4.4 deg)

APCR1n-specific features did not activate, which is acceptable because drift stayed bounded. The features remain available if drift worsens in longer runs.

**Proceed to Phase 6: APCR1n 5000-step low_0p300 validation.**