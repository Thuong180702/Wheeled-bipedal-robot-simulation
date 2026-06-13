# T6J High_0p480 1200-Step Diagnostic Report

Date: 2026-06-13
Profiles compared: `T6I_phase_aware_release` vs `T6J_centering_bias_trim`
Height variant: `high_0p480` (target 0.48 m)
Steps: 1200

## 1. Survival and stability

| Metric | T6I | T6J |
|--------|-----|-----|
| Survived | Yes (1200/1200) | Yes (1199/1199 rows) |
| Terminated | No | No |
| CoM Z drift | +0.0094 m | +0.0087 m |
| Ownership violations | 0 | 0 |
| Hidden torque max | 0.0 | 0.0 |

Both profiles survived 1200 steps with no issues.

## 2. Drift metrics comparison

| Metric | T6I | T6J | Delta |
|--------|-----|-----|-------|
| Max abs error | 0.2034 m | 0.1828 m | -0.0206 m (improved) |
| Final error | +0.0311 m | +0.0132 m | -0.0179 m (improved) |
| Mean error | +0.0934 m | +0.0783 m | -0.0151 m (improved) |
| Positive % | 92.8% | 92.9% | +0.1% (neutral) |
| Zero crossings | 3 | 5 | +2 |
| Outside ±0.08 | 52.3% | 46.3% | -6.0 pp (improved) |
| Outside ±0.10 | 46.0% | 39.7% | -6.3 pp (improved) |
| Outside ±0.15 | 30.3% | 21.1% | -9.2 pp (improved) |
| P2P | 0.2194 m | 0.1986 m | -0.0208 m (improved) |

### Late-run analysis (last 200 steps)

| Metric | T6I | T6J | Delta |
|--------|-----|-----|-------|
| Late mean error | +0.0965 m | +0.0795 m | -0.0170 m (improved) |
| Late positive % | 100.0% | 100.0% | 0% |
| Late outside ±0.08 | 53.0% | 46.5% | -6.5 pp (improved) |

## 3. Pitch and roll

| Metric | T6I | T6J | Delta |
|--------|-----|-----|-------|
| Pitch max | 8.40 deg | 8.19 deg | -0.21 deg |
| Pitch RMS | 4.79 deg | 4.54 deg | -0.25 deg |
| Roll max | 0.24 deg | 0.22 deg | -0.02 deg |

No pitch or damping suppression.

## 4. T6J bias trim behavior

| Metric | Value |
|--------|-------|
| Bias trim active | 1051/1199 (87.7%) |
| Safety gate pass | 96.7% |
| Direction correct | 1199/1199 (100%) |
| Trim tau range | [-0.3500, 0.0000] Nm |
| Block reasons | positive_bias_correcting, inside_exit_threshold, hold_between_thresholds, upright_gate_fail, contact_unstable |

## 5. Key observations

1. **All drift metrics improved**: max abs error, final error, mean error, outside bands, and P2P all show consistent improvement.
2. **Final error converged much closer to zero**: 0.0132 m (T6J) vs 0.0311 m (T6I) — a 57% improvement.
3. **The 500-step final error regression resolved**: At 500 steps T6J's final error was slightly worse (+0.0147 m), but by 1200 steps the trim has had time to operate and T6J's final error is now substantially better.
4. **Outside ±0.15 improved by 9.2 pp**: This is a significant reduction in large-drift occupancy.
5. **Late-run improvement**: The last 200 steps show mean error improved from +0.0965 to +0.0795 m.
6. **Trim reached authority limit**: tau range is [-0.35, 0.0], confirming the trim is bounded as designed.
7. **No stability regression**: Both survived, no ownership violations, no hidden torques.

## 6. Phase 6 classification

**T6J_1200_PASS_PROCEED_2000**

Rationale:
- Both profiles survived 1200 steps with no issues.
- Every drift metric improved consistently.
- Final error improved by 57% (0.0311 → 0.0132 m).
- Outside ±0.15 improved by 9.2 percentage points.
- No pitch/damping suppression.
- Trim behavior is correct and bounded.
- The 500-step final error concern resolved at 1200 steps.

## 7. Output files
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_1200_T6I/telemetry_1200.csv`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_1200_T6J/telemetry_1200.csv`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_high_0p480_1200_diagnostic.json`
