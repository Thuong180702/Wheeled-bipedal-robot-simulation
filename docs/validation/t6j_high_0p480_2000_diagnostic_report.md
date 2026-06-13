# T6J High_0p480 2000-Step Diagnostic Report

Date: 2026-06-13
Steps: 2000

## Drift metrics comparison

| Metric | T6I | T6J | Delta |
|--------|-----|-----|-------|
| Max abs error | 0.2122 m | 0.1828 m | -0.0294 m |
| Final error | +0.0294 m | +0.0483 m | +0.0189 m |
| Mean error | +0.0921 m | +0.0757 m | -0.0164 m |
| Positive % | 91.3% | 87.9% | -3.4 pp |
| Zero crossings | 7 | 13 | +6 |
| Outside ±0.08 | 51.9% | 45.5% | -6.3 pp |
| Outside ±0.10 | 45.7% | 38.9% | -6.8 pp |
| Outside ±0.15 | 30.1% | 20.5% | -9.6 pp |
| P2P | 0.2409 m | 0.2078 m | -0.0331 m |

### Late-run (last 500 steps)

| Metric | T6I | T6J | Delta |
|--------|-----|-----|-------|
| Late mean error | +0.0952 m | +0.0625 m | -0.0327 m |
| Late outside ±0.08 | 54.8% | 38.2% | -16.6 pp |

## T6J trim behavior
- Active: 91.5%
- Safety gate pass: 96.9%
- Direction correct: 100%
- Tau range: [-0.35, 0.0] Nm

## Classification

**T6J_2000_PASS_PROCEED_5000**

Late-run improvement is substantial: mean error reduced by 34% and outside ±0.08 reduced by 16.6 pp.
Final error slightly worse for T6J (+0.0189 m) but late-run behavior clearly better.

## Output files
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_2000_T6I/telemetry_2000.csv`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_2000_T6J/telemetry_2000.csv`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_high_0p480_2000_diagnostic.json`
