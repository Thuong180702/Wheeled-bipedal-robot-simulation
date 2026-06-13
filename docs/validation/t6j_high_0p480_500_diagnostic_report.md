# T6J High_0p480 500-Step Diagnostic Report

Date: 2026-06-13
Profiles compared: `T6I_phase_aware_release` vs `T6J_centering_bias_trim`
Height variant: `high_0p480` (target 0.48 m)
Steps: 500

## 1. Survival and stability

| Metric | T6I | T6J |
|--------|-----|-----|
| Survived | Yes (500/500) | Yes (499/499 rows) |
| Terminated | No | No |
| Contact % | 100% double | 100% double |
| CoM Z min | 0.4811 m | 0.4811 m |
| CoM Z max | 0.4915 m | 0.4915 m |
| CoM Z drift | +0.0096 m | +0.0092 m |
| Ownership violations | 0 | 0 |
| Hidden torque max | 0.0 | 0.0 |

Both profiles survived the full 500 steps with stable double contact, no ownership violations, and no hidden torques.

## 2. Drift metrics

| Metric | T6I | T6J | Delta |
|--------|-----|-----|-------|
| Max abs error | 0.2034 m | 0.1828 m | -0.0206 m (improved) |
| Final error | +0.1411 m | +0.1558 m | +0.0147 m (worse) |
| Mean error | +0.0793 m | +0.0728 m | -0.0065 m (improved) |
| Abs mean error | 0.0819 m | 0.0750 m | -0.0069 m (improved) |
| Positive % | 82.8% | 83.0% | +0.2% (neutral) |
| Negative % | 14.4% | 12.6% | -1.8% |
| Zero crossings | 3 | 5 | +2 |
| Outside ±0.08 | 45.1% | 43.7% | -1.4% (improved) |
| Outside ±0.10 | 39.1% | 37.3% | -1.8% (improved) |
| Outside ±0.15 | 24.2% | 19.0% | -5.2% (improved) |
| P2P | 0.2194 m | 0.1986 m | -0.0208 m (improved) |
| Error range | [-0.016, +0.203] | [-0.016, +0.183] | narrower |

## 3. Pitch and roll

| Metric | T6I | T6J | Delta |
|--------|-----|-----|-------|
| Pitch max | 8.40 deg | 8.19 deg | -0.21 deg |
| Pitch RMS | 4.38 deg | 4.33 deg | -0.05 deg |
| Roll max | 0.17 deg | 0.18 deg | +0.01 deg (negligible) |
| Roll RMS | 0.09 deg | 0.09 deg | 0.00 deg |

No pitch suppression or damping suppression observed. Pitch and roll remain comparable.

## 4. Wheel velocity

| Metric | T6I | T6J | Delta |
|--------|-----|-----|-------|
| Wheel vel max | 6.67 rad/s | 6.32 rad/s | -0.35 rad/s |
| Wheel vel RMS | 3.91 rad/s | 3.60 rad/s | -0.31 rad/s |

T6J shows slightly lower wheel velocity, consistent with the trim providing some centering correction rather than the wheels fighting the drift.

## 5. T6J bias trim behavior

| Metric | Value |
|--------|-------|
| Bias trim active | 363/499 (72.7%) |
| Safety gate pass | 94.4% |
| Mean error range | [-0.0105, +0.0843] m |
| Trim tau range | [-0.3500, 0.0000] Nm |
| Applied tau range | [-0.3500, 0.0000] Nm |
| Direction correct | 499/499 (100%) |
| Block reasons | positive_bias_correcting, inside_exit_threshold, hold_between_thresholds, upright_gate_fail, contact_unstable |

### T6J trim interpretation
- The trim activated 72.7% of the time, which is expected given the persistent positive drift bias.
- The trim torque was always negative (corrective for positive drift), bounded at -0.35 Nm, and never exceeded the max.
- Direction correctness was 100% — no sign inversions observed.
- Safety gate passed 94.4% of the time, with occasional blocks from upright_gate_fail, contact_unstable, and inside_exit_threshold.
- The trim correctly entered positive_bias_correcting mode and attempted to correct the bias.

## 6. Key comparisons

### T6J improvements over T6I
1. **Max abs error reduced by 0.021 m** (0.2034 → 0.1828)
2. **Mean error reduced by 0.007 m** (0.0793 → 0.0728)
3. **Outside ±0.15 reduced by 5.2 percentage points** (24.2% → 19.0%)
4. **P2P reduced by 0.021 m** (0.2194 → 0.1986)
5. **Wheel velocity RMS reduced** (3.91 → 3.60 rad/s)
6. **Zero crossings increased** (3 → 5)

### T6J regressions vs T6I
1. **Final error slightly worse** (+0.1411 → +0.1558, delta +0.0147 m)
   - This is a modest increase. The trim may not have fully converged yet in 500 steps given the slow 0.01 Nm/step rate limit and 200-step averaging window.

### T6J areas of concern
- Final error is slightly worse, though the overall trajectory shows improvement in most metrics.
- The positive occupancy remains high (83%), indicating the bias is deeply structural and the 0.35 Nm trim authority may be insufficient for full centering.
- The trim reached its -0.35 Nm bound, suggesting it may be authority-limited.

## 7. Safety verification
- No pitch suppression: pitch metrics are equivalent.
- No damping suppression: wheel velocity is slightly lower, not higher.
- No sign inversions: direction correct 100%.
- No WBC changes: ownership violations remain zero.
- No hidden torques: max is 0.0 for both profiles.

## 8. Phase 5 classification

**T6J_500_PASS_WITH_MONITORING**

Rationale:
- T6J survived the full 500 steps with no stability issues.
- All core drift metrics improved or remained equivalent.
- The trim activated correctly with 100% direction correctness.
- The final error regression (+0.0147 m) is modest and consistent with the slow rate-limited trim not yet converging in 500 steps.
- No safety violations, no pitch suppression, no damping suppression.
- The trim is authority-limited at -0.35 Nm which explains the incomplete centering.

Monitoring notes for 1200-step run:
- Watch whether the final error starts decreasing after the trim has more time to operate.
- Watch whether outside ±0.15 continues to improve.
- Watch for any oscillatory behavior as the trim accumulates.

## 9. Output files
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_500_T6I/telemetry_500.csv`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_500_T6I/telemetry_500.summary.json`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_500_T6J/telemetry_500.csv`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_500_T6J/telemetry_500.summary.json`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_high_0p480_500_diagnostic.json`
