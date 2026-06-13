# APCR1 Active Pitch Crossing Recovery 500-Step Report

## Classification: APCR1_PASS_PROCEED_TO_2000

## Executive Summary

APCR1 successfully improved signed support drift metrics compared to D2 baseline while maintaining 500-step survival:

- **Signed error mean**: 18.2% improvement (0.0824 → 0.0674)
- **Position error outside ±0.15**: 28.1% reduction (19.2% → 13.8%)
- **Positive bias reduction**: 14.8% (93.2% → 79.4%)
- **Pitch RMS**: Slightly worse (3.60 → 4.00 deg), expected due to APCR torque during moderate pitch

## Phase 8: APCR1 500-Step Evaluation

### Run Configuration

- **Profile**: APCR1_active_pitch_crossing_recovery_moderate
- **Height setup**: low_0p300
- **Steps**: 500
- **Telemetry**: telemetry_1780926507.csv (523 columns, 500 rows)
- **Result**: Survived 500 steps ✓

### Key Configuration Differences from Old APC1

| Parameter | Old APC1 | New APCR1 |
|----------|----------|-----------|
| `enable_active_pitch_crossing` | True | True |
| `active_pitch_crossing_recovery_gate_mode` | False | **True** |
| `apc_pitch_danger_threshold_rad` | 0.10 (5.73°) | 0.10 (same) |
| `apcr_pitch_hard_stop_rad` | N/A | **0.30 (17.2°)** |
| `apc_max_cross_tau` | 1.5 Nm | **1.0 Nm** |
| `apc_max_rate_per_step` | 0.5 Nm/step | **0.4 Nm/step** |

The key change is the **recovery gate mode** which allows APCR to activate during moderate pitch error instead of blocking until pitch is safe.

## Phase 9: Comparison

### D2 vs APC1 vs APCR1

| Metric | D2 | APC1 | APCR1 | APCR1 vs D2 |
|--------|-----|------|-------|--------------|
| Survived | 500 | 500 | 500 | = |
| Pitch X min (deg) | -0.48 | -15.90 | -3.34 | Worse |
| Pitch X max (deg) | 6.36 | 14.96 | 6.88 | Similar |
| Pitch X RMS (deg) | 3.60 | 7.50 | 4.00 | +11% worse |
| Signed error mean | 0.0824 | 1.6 | **0.0674** | **-18.2%** |
| Signed error max | 0.1757 | 1.6 | 0.1714 | -2.4% |
| Outside ±0.15 (%) | 19.2% | 100% | **13.8%** | **-28.1%** |
| Positive % | 93.2% | 100% | **79.4%** | **-14.8%** |
| Height mean (m) | 0.2921 | 0.291 | 0.2922 | Similar |

### Analysis

**APCR1 improvements:**
1. **Signed error mean reduced by 18.2%** (0.0824 → 0.0674) - the most important metric
2. **Position error band violations reduced by 28.1%** (19.2% → 13.8%)
3. **Positive drift bias reduced by 14.8%** (93.2% → 79.4%) - robot more balanced

**Expected tradeoffs:**
1. Pitch RMS slightly worse (+11%) - due to APCR applying torque during moderate pitch
2. Pitch min more negative (-0.48 → -3.34 deg) - APCR creates controlled lean-back
3. Pitch max similar (+6.36 → +6.88 deg) - no significant change

### APC1 vs APCR1 Comparison

APC1 never activated (stayed in NEUTRAL throughout) because:
- Old pitch safety gate blocked activation when pitch > 5.73°
- Pitch oscillated between -15.9° and +15.0°, frequently exceeding threshold

APCR1 activates because:
- Recovery gate mode separates hard safety (pitch > 17.2°) from recovery entry
- APCR can activate during moderate pitch (1.7° to 17.2°) when drift is present
- Hard safety only blocks at true emergency threshold

## Why APCR1 Works

1. **Recovery gate mode allows activation when needed**: Old APC blocked when pitch > 5.73°, but APCR allows activation when 1.7° < pitch < 17.2° AND signed error > 0.10

2. **Moderate torque avoids overcorrection**: APCR1 uses 1.0 Nm (vs 1.5 Nm for APC2), which is enough to influence drift without destabilizing

3. **Exit conditions prevent early reversal**: APCR holds recovery direction until signed error enters inner band (≤0.05 m) or crosses slightly negative

4. **Rate limiting prevents discontinuities**: 0.4 Nm/step max rate ensures smooth torque transitions

## Phase 10: Classification

**APCR1_PASS_PROCEED_TO_2000**

Criteria met:
- ✓ APCR activates when expected
- ✓ Signed error mean decreases (18.2% improvement)
- ✓ Outside ±0.15 band decreases (28.1% reduction)
- ✓ Positive drift bias decreases (14.8% reduction)
- ✓ No overcorrection below -0.15 (APCR1 min = -0.0721)
- ✓ Contact/height/roll remain valid
- ✓ Survived 500 steps

## Recommendations

1. **Proceed to 2000-step validation** with APCR1
2. **Monitor pitch RMS** - slightly worse but acceptable given drift improvements
3. **Consider APCR2** (1.5 Nm) if 2000-step shows insufficient improvement
4. **Do NOT use old APC1** - it cannot activate for low_0p300 due to pitch oscillation

## Files Generated

- `outputs/hierarchical_controller_sim/telemetry_1780926507.csv` - APCR1 500-step telemetry
- `docs/validation/apcr1_active_pitch_crossing_recovery_500_step_report.md` - This report
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1_500_comparison.json` - Comparison data
