# APCR1nD 2000-step Drift Comparison

## Summary

This report compares drift metrics across four profiles:
- **D2** (baseline)
- **APCR1h** (support drift priority fast recenter)
- **APCR1n** (recenter priority torque boost)
- **APCR1nD** (direct support recenter features)

## Drift Metrics Comparison

| Metric | D2 | APCR1h | APCR1n | APCR1nD | Winner |
|--------|-----|--------|--------|---------|--------|
| Max \|e\| | 0.2463 | 0.1775 | 0.1714 | **0.1691** | APCR1nD ✅ |
| P2P | 0.2733 | 0.2491 | 0.1854 | **0.1795** | APCR1nD ✅ |
| Mean \|e\| | 0.0923 | 0.0768 | 0.0608 | **0.0607** | APCR1nD ✅ |
| Final e | +0.0720 | -0.0453 | +0.0035 | +0.0038 | APCR1n ✅ |
| Outside ±0.10 | 565 (28.2%) | 746 (37.3%) | 459 (22.9%) | **446 (22.3%)** | APCR1nD ✅ |
| Outside ±0.15 | 357 (17.9%) | 251 (12.6%) | **53 (2.6%)** | 55 (2.8%) | APCR1n ✅ |

## Key Findings

### 1. APCR1nD is the Clear Winner on Primary Metrics

APCR1nD achieves:
- **LOWEST max |e|**: 0.1691 m (vs APCR1n 0.1714, APCR1h 0.1775, D2 0.2463)
- **LOWEST P2P**: 0.1795 m (vs APCR1n 0.1854, APCR1h 0.2491, D2 0.2733)
- **LOWEST Mean |e|**: 0.0607 m (vs APCR1n 0.0608, APCR1h 0.0768, D2 0.0923)
- **LOWEST Outside ±0.10**: 446 steps (22.3%) (vs APCR1n 459, APCR1h 746, D2 565)

### 2. APCR1n has Slight Edge on Extreme Violations

APCR1n has slightly fewer violations beyond ±0.15 (53 vs 55), but this is a marginal difference of 2 steps.

### 3. Window Analysis

#### Window 0-500 (Startup)
| Metric | D2 | APCR1h | APCR1n | APCR1nD |
|--------|-----|--------|--------|---------|
| Max \|e\| | 0.2463 | 0.1568 | 0.1714 | 0.1691 |
| Mean \|e\| | 0.1794 | 0.0650 | 0.0702 | 0.0693 |
| Outside ±0.15 | 357 | 34 | 53 | 55 |

D2 has significantly more startup drift (357 vs 34-55).

#### Windows 500-2000 (Steady State)
- APCR1nD consistently outperforms APCR1h on all metrics
- APCR1nD and APCR1n are comparable in steady state
- APCR1h shows recurring drift spikes in windows 500-1000 and 1000-1500

### 4. Final State Comparison

| Profile | Final e | Interpretation |
|--------|---------|----------------|
| D2 | +0.0720 | Persistent positive drift |
| APCR1h | -0.0453 | Negative drift |
| APCR1n | +0.0035 | Near zero - best |
| APCR1nD | +0.0038 | Near zero - near best |

## Conclusion

**APCR1nD is the best performing profile** on most drift metrics:
1. Lowest max |e| ✅
2. Lowest P2P ✅
3. Lowest mean |e| ✅
4. Fewest ±0.10 violations ✅
5. Near-zero final drift ✅

APCR1n is a close second with slightly fewer extreme ±0.15 violations.
