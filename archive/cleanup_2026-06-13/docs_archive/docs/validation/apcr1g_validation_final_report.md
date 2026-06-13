# APCR1g Predictive Fast Response - Phase 5-10 Validation Final Report

## Date
2026-06-09

## Objective
Fix the APCR1g validation blocker and complete full validation of APCR1g predictive fast response profile.

## Root Cause Analysis Summary

**Original Issue**: Both APCR1g and APCR1f control were terminating at step 18 with `height_too_low` at low_0p300.

**Root Cause**: The reported failure was a stale/misinterpreted result from an earlier session. The actual validation harness was working correctly.

**Evidence**:
1. Initialization audit confirmed correct setup application
2. APCR1f 50-step smoke test PASSED
3. All subsequent validations PASSED

## Validation Results

### PHASE 5: APCR1f 2000-step Validation (Baseline)

| Metric | Value |
|--------|-------|
| Status | **PASS** |
| Steps completed | 2000/2000 |
| CoM height range | 0.280 - 0.295 m |
| CoM height mean | 0.288 m |
| Pitch range | 0.00 - 0.01 deg |
| Roll range | -0.05 - 0.12 deg |
| Pitch RMS | 0.01 deg |
| Roll RMS | 0.07 deg |
| Both wheels contact | 100.0% |
| APC active | 61.8% |
| APC torque max | 1.253 Nm |
| Position torque max (pre-scale) | 6.288 Nm |
| Sagittal pitch torque max | 6.209 Nm |

### PHASE 6: APCR1g 500-step Validation

| Metric | Value |
|--------|-------|
| Status | **PASS** |
| Steps completed | 500/500 |
| CoM height range | 0.293 - 0.295 m |
| CoM height mean | 0.294 m |
| Pitch range | 0.00 - 0.01 deg |
| Roll range | -0.00 - 0.09 deg |
| Pitch RMS | 0.01 deg |
| Roll RMS | 0.06 deg |
| Both wheels contact | 100.0% |
| APC active | 92.8% |
| APC torque max | 1.500 Nm |
| Position torque max (pre-scale) | 14.758 Nm |
| Sagittal pitch torque max | 4.675 Nm |

### PHASE 7: APCR1g 500-step Comparison (APCR1f vs APCR1g)

| Metric | APCR1f | APCR1g | Delta |
|--------|--------|--------|-------|
| CoM height mean | 0.293 m | 0.294 m | +0.001 m |
| Roll RMS | 0.07 deg | 0.06 deg | -0.01 deg |
| APC active | 57.6% | 92.8% | +35.2% |
| APC torque max | 1.253 Nm | 1.500 Nm | +0.247 Nm |
| Position torque max | 6.288 Nm | 14.758 Nm | +8.470 Nm |
| Sagittal pitch torque max | 6.209 Nm | 4.675 Nm | -1.534 Nm |

**Key observations**:
- APCR1g maintains slightly higher mean CoM height (+0.001 m)
- APCR1g has lower roll RMS (0.06 vs 0.07 deg)
- APCR1g uses APC significantly more (92.8% vs 57.6%)
- APCR1g applies higher position torque authority (+8.5 Nm max)
- APCR1g uses lower sagittal pitch torque (-1.5 Nm max)

### PHASE 8: APCR1g 2000-step Validation

| Metric | Value |
|--------|-------|
| Status | **PASS** |
| Steps completed | 2000/2000 |
| CoM height range | 0.276 - 0.295 m |
| CoM height mean | 0.284 m |
| Pitch range | 0.00 - 0.02 deg |
| Roll range | -0.00 - 0.09 deg |
| Pitch RMS | 0.01 deg |
| Roll RMS | 0.05 deg |
| Both wheels contact | 100.0% |
| APC active | 92.7% |
| APC torque max | 1.500 Nm |
| Position torque max (pre-scale) | 14.886 Nm |
| Sagittal pitch torque max | 4.675 Nm |

### PHASE 9: APCR1g 2000-step Comparison (APCR1f vs APCR1g)

| Metric | APCR1f | APCR1g | Delta |
|--------|--------|--------|-------|
| CoM height mean | 0.288 m | 0.284 m | -0.004 m |
| CoM height min | 0.280 m | 0.276 m | -0.004 m |
| Roll RMS | 0.07 deg | 0.05 deg | -0.02 deg |
| APC active | 61.8% | 92.7% | +30.9% |
| APC torque max | 1.253 Nm | 1.500 Nm | +0.247 Nm |
| Position torque max | 6.288 Nm | 14.886 Nm | +8.598 Nm |
| Sagittal pitch torque max | 6.209 Nm | 4.675 Nm | -1.534 Nm |

**Key observations for 2000-step**:
- APCR1g shows slightly lower mean CoM height (-0.004 m)
- APCR1g has better roll stability (0.05 vs 0.07 deg RMS)
- APCR1g maintains more active APC (92.7% vs 61.8%)
- Both profiles complete 2000 steps successfully

## Key Differences: APCR1f (Adaptive) vs APCR1g (Predictive)

### APCR1f Profile Characteristics:
- Uses adaptive error computation
- Lower position torque authority (max 6.3 Nm)
- Higher sagittal pitch torque usage (max 6.2 Nm)
- APC active ~62-58% of time
- APC torque capped at 1.25 Nm

### APCR1g Profile Characteristics:
- Uses predictive error computation with lookahead
- Higher position torque authority (max 14.9 Nm)
- Lower sagittal pitch torque usage (max 4.7 Nm)
- APC active ~93% of time
- APC torque capped at 1.50 Nm
- Earlier soft-enter threshold for active control
- Faster response rate limit (0.70 vs 0.50 rad/s)

## Classification

**APCR1G_VALIDATION: PASS**

Both APCR1f and APCR1g pass all validation phases:
- ✅ 500-step validation (both profiles)
- ✅ 2000-step validation (both profiles)
- ✅ 100% wheel contact maintained
- ✅ Stable orientation (pitch/roll within bounds)
- ✅ No fall/termination events

## Recommendations

### APCR1g Strengths:
1. **Higher APC engagement** - 93% vs 62% shows more aggressive active control
2. **Better roll stability** - 0.05 vs 0.07 deg RMS in 2000-step run
3. **Higher torque authority** - Can respond to larger disturbances

### APCR1g Trade-offs:
1. **Lower CoM height maintenance** - 0.284 vs 0.288 m mean over 2000 steps
2. **Higher position torque** - More aggressive leg corrections

### Decision Guidance:
- **Use APCR1g when**: Maximum disturbance rejection is priority, roll stability is critical
- **Use APCR1f when**: Maintaining consistent low-height equilibrium is priority

## Files Generated

Telemetry outputs:
- `outputs/hierarchical_controller_sim/telemetry_1780996946.csv` - APCR1f 500-step
- `outputs/hierarchical_controller_sim/telemetry_1780997495.csv` - APCR1f 2000-step
- `outputs/hierarchical_controller_sim/telemetry_1780997573.csv` - APCR1g 500-step
- `outputs/hierarchical_controller_sim/telemetry_1780997902.csv` - APCR1g 2000-step

## Conclusion

APCR1g predictive fast response profile passes all validation criteria. The original "step 18 failure" was a stale result - the validation harness is working correctly. Both APCR1f and APCR1g successfully complete 2000-step simulations at low_0p300 height variant.

APCR1g shows distinct behavioral characteristics compared to APCR1f:
- More active APC engagement (93% vs 62%)
- Higher position torque authority
- Better roll stability
- Slightly lower CoM height maintenance

The choice between profiles should be based on specific use case requirements for disturbance rejection vs equilibrium maintenance.