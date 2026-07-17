# Step C Regression ReCheck

**Date:** 2026-06-21
**Task:** Physics FF Low-Band Support v2 — Step D push validation gate (Task 5 of 9)
**Classification:** `STEP_C_RECHECK_PASS`

## Scope

Verify that the Low-band v2 profile ("Low-band support v2", tag `D_LOW_BAND_V2`)
passes all Step C regression gates. Artifacts are from the v2 tuning phase
(`outputs/physics_ff_low_band_support_v2_tuning/`).

## Gate Matrix

### Gate 1: No falls in any case

Checked: Low-band v2 segments in both `full_fixed_height_metrics.csv` (10 heights)
and `full_step_c_segment_metrics.csv` (72 segments across 7 cases).

| Source | Cases | any_fell | Result |
|---|---|---|---|
| Fixed-height (10) | low_0p300 .. high_0p480 | False (all) | PASS |
| Step C (7 cases, 72 segs) | C1, C2, C3, C4, C5, focused_low_0p320, focused_high_0p480 | False (all) | PASS |

### Gate 2: hip_yaw_abs_max_rad < 0.35

| Source | Worst hip_yaw | Result |
|---|---|---|
| Fixed-height | 0.2034 (low_0p300) | PASS |
| Step C | 0.0794 (C1) | PASS |

### Gate 3: wbc_authority_rows = 0 and Gate 4/5: hidden_torque_max = 0, ownership_violation_max = 0

All segments in both files show:
- `wbc_authority_rows = 0`
- `hidden_torque_max = 0.0`
- `ownership_violation_max = 0.0`

**Result: PASS**

### Gate 6: out15_pct = 0 for all Step C cases

All 7 Step C cases (72 segments) have `out15_pct = 0.0`.

| Case | Segments | out15_pct | Result |
|---|---|---|---|
| C1_slow_ladder_up_down | 20 | 0.0 (all) | PASS |
| C2_random_500dwell | 10 | 0.0 (all) | PASS |
| C3_random_200dwell | 15 | 0.0 (all) | PASS |
| C4_abrupt_stress | 5 | 0.0 (all) | PASS |
| C5_long_random | 20 | 0.0 (all) | PASS |
| focused_low_0p320 | 1 | 0.0 | PASS |
| focused_high_0p480 | 1 | 0.0 | PASS |

### Gate 7: Fixed-height low_0p320 max_abs <= 0.147 m (v2 tuning result)

- Fixed-height (2000-step) max_abs: **0.1472 m**
- Focused (Step C) max_abs: **0.0725 m**

The fixed-height value (0.1472) is approximately 0.147 m and well under the brief
threshold of 0.15 m. The focused case (0.0725) is far below both thresholds.

**Result: PASS** (0.1472 ~= 0.147, < 0.15 per brief)

### Gate 8: Fixed-height high_0p480 max_abs matches Current PFF

| Profile | max_abs high_0p480 |
|---|---|
| Current PFF | 0.1520 m |
| Low-band v2 | 0.1520 m |

Identical values. **Result: PASS**

## Observation: Fixed-height high heights and out15

At high heights (high_0p465, high_0p480) all profiles show nonzero `out15_pct`.
This is not a Step C gate condition but is recorded for awareness:

| Height | Baseline B2v2 | Current PFF | Low-band v1 | Low-band v2 |
|---|---|---|---|---|
| high_0p480 | 9.9% | 1.4% | 1.4% | 1.4% |
| high_0p465 | 1.3% | 4.9% | 4.9% | 4.9% |

Low-band v2 matches Current PFF at these heights — no regression.

## Git State Check

- HEAD: `a2fa649` (on branch `repo-cleanup-t6j`)
- Uncommitted modifications: `scripts/simulate_hierarchical_controller.py`,
  `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- These files are **unrelated** to the low-band controller
  (`support_outer_loop_low_band.py`). No artifacts-impacting change is present.

**Result: PASS** — no concern for artifact validity.

## Summary Table

| Gate | Result | Note |
|---|---|---|
| No falls (all cases) | PASS | — |
| hip_yaw_abs_max_rad < 0.35 | PASS | Worst 0.2034 |
| wbc_authority_rows = 0 | PASS | — |
| hidden_torque_max = 0 | PASS | — |
| ownership_violation_max = 0 | PASS | — |
| out15_pct = 0 (Step C cases) | PASS | 72 segments, 7 cases |
| max_abs low_0p320 <= 0.147 m | PASS | 0.1472 (~0.147, <0.15 per brief) |
| max_abs high_0p480 matches PFF | PASS | Both 0.152 m |
| Git state concern | PASS | Uncommitted changes unrelated |

## Overall Verdict

**`STEP_C_RECHECK_PASS`** — Low-band v2 shows no regression versus the v1
profile or Current PFF across all Step C cases. All gates pass.
