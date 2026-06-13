# APCR1h Baseline Audit for APCR1n Development

## Classification

**APCR1H_BASELINE_READY_FOR_APCR1N**

## Source

- Profile: `APCR1h_support_drift_priority_fast_recenter`
- Run source: `comparison_1000_apcr1h`
- Steps: 1000
- Survived: YES

## Drift Metrics

| Metric | Value |
|--------|-------|
| min e | -0.0716 m |
| max e | +0.1775 m |
| max \|e\| | 0.1775 m |
| P2P | 0.2491 m |
| mean e | +0.0594 m |
| mean \|e\| | 0.0745 m |
| final e | +0.1668 m |
| positive % | 78.3% |
| negative % | 21.6% |
| zero crossings | 8 |

## Band Metrics

| Band | Outside % |
|------|-----------|
| ±0.03 m | 73.2% |
| ±0.05 m | 61.6% |
| ±0.08 m | 43.6% |
| ±0.10 m | 35.6% |
| ±0.12 m | 23.2% |
| **±0.15 m** | **9.7%** |
| ±0.18 m | 0.0% |

## Window Metrics

| Window | max \|e\| | P2P | mean \|e\| |
|--------|-----------|-----|------------|
| 0-250 | 0.1568 | 0.1801 | 0.0765 |
| 250-500 | 0.1188 | 0.1391 | 0.0535 |
| 500-750 | 0.1512 | 0.2126 | 0.0749 |
| 750-1000 | 0.1775 | 0.2491 | 0.0930 |

## Stability Metrics

| Metric | Value |
|--------|-------|
| left_contact_mean | 1.000 |
| right_contact_mean | 1.000 |
| n_contacts_mean | 2.037 |
| n_contacts_min | 2.0 |
| CoM Z min | 0.2838 m |
| CoM Z mean | 0.2913 m |
| CoM Z max | 0.2954 m |
| pitch RMS | 0.0084 rad |
| pitch max | 0.0136 rad |
| roll RMS | 0.0774 rad |
| roll max | 0.1366 rad |
| wheel tau mean abs | 0.1913 Nm |
| wheel tau max | 1.6566 Nm |

## Key Observations

1. **Survived full 1000 steps** - no termination
2. **Positive drift bias** - 78.3% of error values are positive
3. **Final drift accumulation** - ends at +0.167 m
4. **P2P = 0.249 m** - indicates oscillation around positive drift
5. **9.7% outside ±0.15 m** - primary target for APCR1n
6. **Wheel damping abs_mean = 1.42 Nm** (from prior audit)
7. **Contact stable** - always 2 contacts throughout
8. **Height stable** - CoM Z stays in [0.284, 0.295] m range
9. **Roll max = 0.137 rad** - acceptable but notable
10. **Pitch very stable** - max 0.014 rad only

## APCR1n Targets

Based on APCR1h baseline:

| Target Metric | APCR1h | APCR1n Target |
|---------------|--------|---------------|
| max \|e\| | 0.1775 m | < 0.1775 m |
| outside ±0.15 | 9.7% | < 9.7% |
| P2P | 0.2491 m | ≤ 0.2491 m |
| wheel damping | 1.42 Nm | 1.5-2.0 Nm during RECENTER |

## Conclusion

APCR1h baseline is valid and ready for APCR1n development.

APCR1h shows:
- Good startup stability (0-250 window healthy)
- Positive drift accumulation trend
- 9.7% outside ±0.15 m band
- Acceptable stability metrics

APCR1n should target reducing max |e| and outside ±0.15 % while preserving stability.
