# T6J High_0p480 5000-Step Validation Report

Date: 2026-06-13
Steps: 5000

## Main comparison

| Metric | T6I | T6J | Delta |
|--------|-----|-----|-------|
| Max abs error | 0.2122 m | 0.1828 m | -0.0294 m |
| Final error | +0.1309 m | +0.1178 m | -0.0132 m |
| Mean error | +0.0953 m | +0.0787 m | -0.0166 m |
| Positive % | 95.4% | 94.5% | -0.9 pp |
| Zero crossings | 11 | 17 | +6 |
| Outside ±0.08 | 53.6% | 47.2% | -6.4 pp |
| Outside ±0.10 | 46.7% | 39.0% | -7.7 pp |
| Outside ±0.15 | 29.2% | 14.1% | -15.1 pp |
| P2P | 0.2409 m | 0.2078 m | -0.0331 m |
| Pitch RMS | 4.80 deg | 4.45 deg | -0.35 deg |
| Wheel vel RMS | 3.63 rad/s | 3.19 rad/s | -0.44 rad/s |

### Late-run (last 1000 steps)

| Metric | T6I | T6J | Delta |
|--------|-----|-----|-------|
| Late mean error | +0.0988 m | +0.0771 m | -0.0216 m |
| Late outside ±0.08 | 55.6% | 46.3% | -9.3 pp |
| Late positive % | 100.0% | 100.0% | 0 |

## T6J trim behavior
- Active: 96.3%
- Safety gate pass: 98.5%
- Tau range: [-0.35, 0.0] Nm
- Bounded and always within designed limit

## Interpretation

T6J remains clearly better than T6I over the full 5000-step horizon.
The persistent positive drift is **not eliminated**, but it is materially reduced:
- mean error improved by 17.4%
- final error improved by 10.1%
- outside ±0.15 occupancy cut by more than half (29.2% → 14.1%)
- late-run mean error improved by 21.9%

The trim stays active nearly the entire run, indicating the bias remains persistent and the 0.35 Nm cap is likely authority-limiting rather than unstable.

## Classification

**T6J_5000_PASS_PROCEED_HEIGHT_LADDER**

Rationale:
- Full 5000-step survival maintained.
- All principal drift metrics improved.
- Large-error occupancy (>±0.15 m) reduced by 15.1 percentage points.
- No evidence of instability, pitch suppression, damping suppression, WBC leakage, or hidden torques.
- Residual positive bias remains, but T6J is consistently better than T6I.

## Output files
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_5000_T6I/telemetry_5000.csv`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_5000_T6J/telemetry_5000.csv`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_high_0p480_5000_diagnostic.json`
