# Phase 7: Long-Run Strict Verification

**Date:** 2026-06-29
**Source data:** `outputs/k2_jax_dedicated_promotion_validation/all_metrics_comparison.json`
**Individual summaries:** `outputs/k2_jax_dedicated_promotion_validation/long_run/*/summary.json`

---

## All 5 Long-Run Equilibrium Cases (6000 steps each)

### Results Table

| case | fell (cand) | fell (orig) | pitch_rms cand (deg) | pitch_rms orig (deg) | pitch_rms delta (deg) | pitch_final orig (deg) | hip_yaw_max cand (rad) | hip_yaw_max orig (rad) | hip_yaw_max delta (rad) | class |
|---|---|---|---|---|---|---|---|---|---|---|
| low_0p330 | no | no | 5.07 | 3.97 | +1.10 | 4.34 | 0.1887 | 0.2048 | -0.0161 | SAFE_BUT_WORSE |
| mid_0p400 | no | no | 1.75 | 1.84 | -0.09 | 2.51 | 0.1917 | 0.1071 | +0.0846 | SAFE_BUT_WORSE |
| high_0p430 | no | no | 3.77 | 5.60 | -1.83 | 5.69 | 0.2158 | 0.0496 | +0.1662 | SAFE_BUT_WORSE |
| high_0p450 | no | no | 4.55 | 3.45 | +1.10 | 3.72 | 0.2213 | 0.0882 | +0.1331 | SAFE_BUT_WORSE |
| high_0p480 | no | no | 4.69 | 5.15 | -0.46 | 5.69 | 0.1962 | 0.0574 | +0.1388 | SAFE_BUT_WORSE |

Notes:
- `pitch_final` not available in candidate summary files; original `pitch_final_deg` provided for reference.
- `pitch_rms delta` = candidate - original (negative = candidate better).
- `hip_yaw_max delta` = candidate - original (negative = candidate better).
- Values rounded to 2 decimal places (pitch_rms) and 4 decimal places (hip_yaw_max).
- All source values are measured; no estimates or interpolations used.

---

### Key Findings

**Survival**
- All 5 cases completed 6000 steps without falls in both candidate (JAX) and original (K2) backends.
- 0 falls across all runs.

**Pitch RMS**
- **mid_0p400** has the best candidate pitch RMS at **1.75 deg** (marginally better than original 1.84 deg, delta -0.09).
- **low_0p330** has the worst candidate pitch RMS at **5.07 deg** (worse than original 3.97 deg, delta +1.10).
- Candidate beats original on pitch at 3 of 5 heights (mid_0p400: -0.09, high_0p430: -1.83, high_0p480: -0.46).
- Candidate loses on pitch at 2 of 5 heights (low_0p330: +1.10, high_0p450: +1.10).

**Hip Yaw Divergence**
- All 5 candidate hip_yaw_max values exceed their original counterparts except low_0p330.
- low_0p330: candidate 0.1887 rad is marginally better than original 0.2048 rad (delta -0.0161).
- Worst hip_yaw degradation: high_0p430 at +0.1662 rad and high_0p480 at +0.1388 rad.
- Original baseline reference values:
  - low_0p330 hy = 0.2048
  - mid_0p400 hy = 0.1071
  - high_0p430 hy = 0.0496
  - high_0p450 hy = 0.0882
  - high_0p480 hy = 0.0574

**Classification**
- Each case classified as SAFE_BUT_WORSE because:
  - All 5 survived (no SAFETY_FAIL).
  - But hip_yaw_max is degraded in 4 of 5 cases, and pitch_rms is degraded in 2 of 5 cases, so the candidate is not a strict improvement.

---

### Summary

| Metric | Value |
|---|---|
| Cases completed | 5/5 |
| Falls | 0 |
| SAFETY_FAIL cases | 0 |
| SAFE_BUT_WORSE cases | 5 |
| STRICT_IMPROVEMENT cases | 0 |
| **Overall classification** | **SAFE_BUT_WORSE** |

**Verdict:** The JAX dedicated runner completes all 5 long-run equilibrium cases without falls, meeting the safety requirement. However, hip_yaw divergence is systematically worse than the original K2 baseline across 4 of 5 heights, and pitch RMS is worse at 2 heights. The runner passes the promotion safety gate (no falls) but does not achieve parity on hip_yaw divergence or pitch stability. Further tuning of the mode_div mechanism and/or dynamic_qref interpolation at higher heights may reduce the hip_yaw gap.

---

### Raw Data Sources

- `outputs/k2_jax_dedicated_promotion_validation/long_run/low_0p330/summary.json`
- `outputs/k2_jax_dedicated_promotion_validation/long_run/mid_0p400/summary.json`
- `outputs/k2_jax_dedicated_promotion_validation/long_run/high_0p430/summary.json`
- `outputs/k2_jax_dedicated_promotion_validation/long_run/high_0p450/summary.json`
- `outputs/k2_jax_dedicated_promotion_validation/long_run/high_0p480/summary.json`
- `outputs/k2_jax_dedicated_promotion_validation/all_metrics_comparison.json`

All values above are directly transcribed from these files. No estimates or derived values are used.
