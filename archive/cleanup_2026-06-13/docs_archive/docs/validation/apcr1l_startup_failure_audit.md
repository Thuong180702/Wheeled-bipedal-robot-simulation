# APCR1l Startup Failure Audit

## Classification

```
APCR1L_TELEMETRY_MISMATCH
```

The reported APCR1l "fall at step 18 with height_too_low" is **NOT reproducible** with the current telemetry. The current telemetry shows 1000 steps survived with pitch in normal range (-5.5 to +8.6 deg).

## Evidence

### Current Telemetry Analysis

From `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1l_1000_validation/apcr1l_1000_telemetry.csv`:

| Metric | Value |
|--------|-------|
| Total rows | 1000 (steps 0-999) |
| Pitch range | -5.49 to +8.59 degrees |
| CoM Z range | 0.350 to 0.350 m |
| NEUTRAL steps | 260 (26%) |
| RECENTER_FROM_POSITIVE | 404 (40.4%) |
| RECENTER_FROM_NEGATIVE | 336 (33.6%) |
| Pitch suppression active | 739/1000 (73.9%) |

### Steps 0-25 Pitch Analysis

Steps 0-25 show healthy pitch values:

| Step | Pitch (deg) | CoM Z | State | tau_pitch |
|------|-------------|-------|-------|-----------|
| 0 | +0.14 | 0.350 | NEUTRAL | 0.124 |
| 1 | +0.59 | 0.350 | NEUTRAL | 0.519 |
| 2 | +0.66 | 0.350 | NEUTRAL | 0.579 |
| ... | ... | ... | ... | ... |
| 22 | -0.21 | 0.350 | NEUTRAL | -0.180 |
| 26 | +0.71 | 0.350 | NEUTRAL | 0.618 |
| 27 | +1.04 | 0.350 | NEUTRAL | 0.907 |

- All pitches are in normal range
- tau_pitch is NOT suppressed during NEUTRAL (correct behavior)
- No height issues detected

### Pitch Suppression Behavior

Pitch suppression activates correctly in RECENTER states:
- Steps 0-259: NEUTRAL, suppress=False (correct)
- Steps 260+: RECENTER states with suppress=True (correct)

## Key Finding

The APCR1l code:
1. Does NOT suppress tau_pitch during NEUTRAL (correct)
2. DOES suppress tau_pitch during RECENTER states (correct per design)
3. Survives 1000 steps in current telemetry

## Discrepancy

The `apcr1l_final_report.md` claims:
- "Fell after 18 steps (height_too_low)"
- "robot_pitch_x = [-49.83 to -0.28] degrees"

But the telemetry shows:
- 1000 steps survived
- pitch_x = [-5.49 to +8.59] degrees

This discrepancy indicates either:
1. The telemetry was captured from a different run than the one reported in the final report
2. There was a code change between the failed run and the telemetry run
3. The "failed" summary data was fabricated or from a different validation

## Answers to Phase 1 Questions

1. **Is APCR1l pitch suppression active during NEUTRAL?** NO - Correctly inactive
2. **Is APCR1l entering RECENTER at startup before pitch is safe?** NO - NEUTRAL for first 260 steps
3. **Is tau_pitch actually zeroed before step 18?** NO - tau_pitch has normal values (0.12 to 0.9 Nm)
4. **Is the startup fall caused by pitch suppression or by initialization/state bug?** NEITHER - No startup fall in telemetry
5. **Does APCR1l have the same initial qpos/target_joint_pos as APCR1i?** Cannot verify - no failure in telemetry
6. **Is low_0p300 equilibrium applied correctly?** CoM Z = 0.350m consistent
7. **Does pitch_x start extremely wrong only for APCR1l?** NO - pitch starts normally in telemetry
8. **Does suppress_active telemetry match the actual tau_pitch after suppression?** YES - In NEUTRAL suppress=False, tau_pitch has normal values

## Conclusion

APCR1l pitch suppression works correctly:
- Does NOT suppress during NEUTRAL (startup safe)
- DOES suppress during RECENTER (correct per design)
- Survives 1000 steps

The "fall at step 18" claim in the final report is inconsistent with the available telemetry data.
