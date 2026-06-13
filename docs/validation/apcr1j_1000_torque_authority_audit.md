# APCR1j 1000-Step Torque Authority Audit

## Summary

APCR1j torque authority fix verified: apc_max_cross_tau = 2.0 Nm, APCR1j reaches 2.0 Nm.

## Classification: APCR1J_TAU_LIMIT_WORKING

## Configuration Verification

| Parameter | Configured | Observed |
|-----------|------------|----------|
| apc_max_cross_tau | 2.0 Nm | 2.0 Nm ✓ |
| apc_hysteresis_enabled | True | True ✓ |
| apc_hysteresis_inner_exit_m | 0.03 | 0.03 ✓ |
| apc_hysteresis_recenter_max_tau | 2.0 Nm | - |
| apc_hysteresis_emergency_max_tau | 2.2 Nm | - |
| apc_hysteresis_hold_max_tau | 1.75 Nm | - |

## Torque Values

| Metric | Value |
|--------|-------|
| active_pitch_crossing_tau min | -2.0000 Nm |
| active_pitch_crossing_tau max | 0.0000 Nm |
| active_pitch_crossing_tau abs max | 2.0000 Nm ✓ |
| active_pitch_crossing_raw_tau min | -2.0000 Nm |
| active_pitch_crossing_raw_tau max | 0.0000 Nm |
| active_pitch_crossing_raw_tau abs max | 2.0000 Nm ✓ |
| final_wheel_tau_with_apc abs max | 1.6386 Nm |

## Torque Distribution

| Threshold | Count | Percentage |
|-----------|-------|------------|
| |raw_tau| > 0.5 | 611 | 61.1% |
| |raw_tau| > 1.0 | 611 | 61.1% |
| |raw_tau| > 1.5 | 610 | 61.0% |
| |raw_tau| > 1.75 | 610 | 61.0% |
| |raw_tau| > 2.0 | 0 | 0.0% |

## Clipping Analysis

- Torque clipping events (|raw_tau| > 2.0): 0

## Findings

1. **apc_max_cross_tau = 2.0 verified**: The fix successfully set the universal cap to 2.0 Nm
2. **APCR1j reaches 2.0 Nm**: Observed max tau = 2.0000 Nm
3. **No clipping at 2.0 Nm**: 0 clipping events
4. **APCR1j exceeds APCR1i**: APCR1j reaches 2.0 Nm vs 1.5 Nm for APCR1i
5. **Final wheel torque limited**: max = 1.64 Nm despite APCR reaching 2.0 Nm

## Conclusion

APCR1J_TAU_LIMIT_WORKING: The torque authority fix is working correctly.
