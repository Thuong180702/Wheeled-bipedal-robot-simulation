# APCR1e Adaptive Behavior Telemetry Audit

## Purpose

Separate validation of APCR1e adaptive authority behavior, independent of the physical drift metrics issue.

## Adaptive Authority Metrics

| Metric | 500-step | 2000-step |
|--------|----------|-----------|
| APCR tau max (Nm) | 1.16 | 1.16 |
| APCR tau mean (Nm) | 0.50 | 0.53 |
| APCR tau RMS (Nm) | 0.66 | 0.67 |
| APCR active % | 55.4% | 60.6% |

### Comparison with Previous Profiles

| Profile | tau_max | tau_mean | Notes |
|---------|---------|----------|-------|
| D2 | 0.75 | ~0.50 | Fixed cap |
| APCR1c | 0.75 | ~0.50 | Fixed cap |
| APCR1d | 0.75 | N/A | Failed at step 18 |
| **APCR1e** | **1.16** | **0.53** | **Adaptive cap** |

## Interpretation

1. **APCR tau max = 1.16 Nm**: The adaptive max was reached, demonstrating that APCR1e's adaptive authority increase mechanism is functioning.

2. **APCR tau mean = 0.53 Nm**: Slightly higher than D2/APCR1c (~0.50 Nm), consistent with adaptive boosting.

3. **APCR active % = 60.6%**: Majority of steps have APCR correction active, which is expected at low height (0.30 m).

## Conclusion

The adaptive authority behavior is **REAL and VALID**. The APCR1e adaptive mechanism successfully increased torque beyond APCR1d's fixed 0.75 Nm cap.

This does not change the overall classification because physical drift (P2P = 0.235 m) is still worse than D2 (0.162 m).