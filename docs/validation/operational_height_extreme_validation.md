# Operational Height Extreme Validation

**Date:** 2026-06-02  
**Status:** **PASS** for conservative controller-ready extrema

## Scope

This follow-up extends the existing Step E and Step C validation beyond the original five Step B variants.

The goal was **not** to infer absolute mechanical limits from joint ranges. The goal was to:

1. define operational standing height using physical static validity,
2. search for broader low/high standing poses using true posture changes and root-z calibration,
3. reject root-z-only candidates,
4. run fresh Step E and Step C telemetry at the selected extremes,
5. report only **conservative validated extrema** that pass dynamic criteria with the current controller.

## Operational height definition

Operational height was defined in [extreme_height_operational_definition.md](extreme_height_operational_definition.md).

A candidate height is valid only if it satisfies all of the following:

- both wheels remain in valid floor contact,
- no non-wheel body part touches the floor,
- CoM projection remains close to the support center,
- pitch/roll/yaw remain near equilibrium,
- hip-yaw stays near reference,
- symmetric hip-pitch/knee posture is used,
- root_z is calibrated from wheel-floor contact,
- the pose is **not** root-z-only,
- equilibrium references can be captured from the pose,
- WBC remains off,
- hidden torque remains zero,
- ownership violations remain zero.

## Search method

The envelope search reused the Step B posture-search pattern:

- symmetric hip-pitch/knee search,
- calibrated root_z from wheel-floor contact,
- MuJoCo forward validation,
- CoM/support centering checks,
- posture checks,
- joint-limit margin checks,
- root-z-only rejection.

Artifacts:

- `outputs/operational_height_envelope_search/operational_height_search_grid.csv`
- `outputs/operational_height_envelope_search/operational_height_valid_candidates.json`
- `outputs/operational_height_envelope_search/operational_height_envelope_summary.json`
- `outputs/operational_height_envelope_search/static_extreme_validation.json`

## Important finding: static-valid is broader than dynamic-valid

Static search produced controller-ready static extrema broader than the final validated dynamic envelope:

- static-ready min candidate: **0.3800659587 m** CoM
- static-ready max candidate: **0.4267224587 m** CoM

Fresh Step E telemetry showed both of those still fail dynamic acceptance:

### Static-ready min candidate dynamic failure

- support max abs: `0.1387403542 m`
- hip-yaw max abs: `0.2481258363 rad`
- pitch max abs: `0.1000168431 rad`
- wheel velocity max abs: `5.5191009045 rad/s`
- final height drift vs achieved initial: `-0.0151934435 m`

### Static-ready max candidate dynamic failure

- support max abs: `0.1790720323 m`
- hip-yaw max abs: `0.1329053938 rad`
- pitch max abs: `0.1132340090 rad`
- wheel velocity max abs: `7.9205155373 rad/s`
- final height drift vs achieved initial: `-0.0016027308 m`

So the final reported extrema must be based on **fresh dynamic validation**, not static pose validity alone.

## Boundary probing and conservative dynamic envelope

Fresh outward boundary probes were run using the selected D2 profile:

### Low side

- `search_low_0p389m` **FAIL**
  - support max abs: `0.1459076470 m`
  - hip-yaw max abs: `0.2985576689 rad`
  - wheel velocity max abs: `5.9064631462 rad/s`
  - final height error: `-0.0211026414 m`

- `search_low_0p394m` **PASS**
  - support max abs: `0.1061633846 m`
  - hip-yaw max abs: `0.0584058389 rad`
  - pitch max abs: `0.0714031231 rad`
  - roll max abs: `0.0142093679 rad`
  - wheel velocity max abs: `3.8380130529 rad/s`
  - final height error: `0.0033017450 m`

### High side

- `search_high_0p414m` **FAIL**
  - support max abs: `0.1630999934 m`
  - hip-yaw max abs: `0.1468264908 rad`
  - wheel velocity max abs: `6.6353118420 rad/s`
  - final height error: `-0.0042460117 m`

- existing `high_small` **PASS**
  - support max abs: `0.1354918957 m`
  - hip-yaw max abs: `0.0296155605 rad`
  - pitch max abs: `0.0960037677 rad`
  - roll max abs: `0.0091194582 rad`
  - wheel velocity max abs: `4.7700369358 rad/s`
  - final height error: `0.0047191070 m`

## Final conservative validated extrema

The final controller-ready extreme pair is:

- **Minimum operational height:** `0.3932865805 m` CoM
- **Maximum operational height:** `0.4128130092 m` CoM

These are **conservative validated extrema**, not absolute mechanical extrema.

Selection basis:

- min = lowest dynamically passing candidate from the low-side probe set,
- max = highest dynamically passing candidate from the high-side probe set.

Controller-ready extrema artifacts:

- `outputs/operational_height_envelope_search/controller_ready_extrema/min_operational_height_setup.json`
- `outputs/operational_height_envelope_search/controller_ready_extrema/max_operational_height_setup.json`
- `outputs/operational_height_envelope_search/controller_ready_extrema/controller_ready_extrema_summary.json`

## Static validation result

Static validation status: **PASS**

For both final controller-ready extrema:

- left/right wheel contact: `true`
- non-wheel floor contact count: `0`
- root-z-only: `false`
- pitch/roll/yaw: near equilibrium
- equilibrium capture: available
- WBC applied: `false`
- hidden torque: `0.0`
- ownership violations: `0`

### Final static metrics

| Case | Achieved CoM (m) | Support error norm (m) | Joint limit margin (rad) | Wheel floor force (N) |
|---|---:|---:|---:|---:|
| min_operational_height | 0.3932865805 | 0.0000560746 | 0.8873502857 | 113.5161 |
| max_operational_height | 0.4128130092 | 0.0003707630 | 0.8042680000 | 121.2859 |

## Step E extreme-height result

Step E controller-ready extreme validation status: **PASS**

Artifacts:

- `outputs/step_e_extreme_height_position_hold/step_e_extreme_case_matrix.json`
- `outputs/step_e_extreme_height_position_hold/step_e_extreme_metrics.json`
- `outputs/step_e_extreme_height_position_hold/step_e_extreme_position_hold_summary.json`
- `outputs/step_e_extreme_height_position_hold/step_e_extreme_position_hold_report.md`

### Step E metrics

| Case | Verdict | Support max (m) | Support final (m) | HipYaw max (rad) | Pitch max (rad) | Roll max (rad) | Wheel max (rad/s) | Final height error (m) | Contact valid (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| min_operational_height | PASS | 0.1328167633 | 0.1154914446 | 0.0562800691 | 0.0955508836 | 0.0141554463 | 4.7212777138 | 0.0035492533 | 100.0 adjusted / 99.98 raw |
| max_operational_height | PASS | 0.1354918957 | 0.1202197922 | 0.0296155605 | 0.0960037677 | 0.0091194582 | 4.7700369358 | 0.0047191070 | 100.0 adjusted / 99.98 raw |

Structural invariants preserved for both cases:

- WBC applied: `false`
- hidden torque norm max: `0.0`
- ownership violation count max: `0`
- controller behavior changed: `false`
- profile used: `candidate_D2_wheel_velocity_damping_light`

## Step C extreme-height result

Step C controller-ready extreme validation status: **PASS**

Artifacts:

- `outputs/step_c_extreme_height_recovery/step_c_extreme_case_matrix.json`
- `outputs/step_c_extreme_height_recovery/step_c_extreme_metrics.json`
- `outputs/step_c_extreme_height_recovery/step_c_extreme_pass_fail_summary.json`
- `outputs/step_c_extreme_height_recovery/step_c_extreme_height_recovery_report.md`

### Step C metrics

| Case | Verdict | Recovery time (s) | Support max (m) | Support final (m) | HipYaw max (rad) | Pitch max (rad) | Roll max (rad) | Wheel max (rad/s) | Final height error (m) | Contact valid (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| min_operational_height | PASS | 0.0 | 0.1328167633 | 0.1154914446 | 0.0562800691 | 0.0955508836 | 0.0141554463 | 4.7212777138 | 0.0035492533 | 100.0 adjusted / 99.98 raw |
| max_operational_height | PASS | 0.0 | 0.1354918957 | 0.1202197922 | 0.0296155605 | 0.0960037677 | 0.0091194582 | 4.7700369358 | 0.0047191070 | 100.0 adjusted / 99.98 raw |

Notes:

- Both cases started already inside the accepted height band, so `recovery_time_s = 0.0` is valid because the hold window remained satisfied.
- Local transition recovery around the extrema was **NOT RUN** in this pass.

Structural invariants preserved for both cases:

- WBC applied: `false`
- hidden torque norm max: `0.0`
- ownership violation count max: `0`
- Step E invariants preserved: `true`
- controller behavior changed: `false`

## Final decision

**EXTREME_HEIGHT_VALIDATION_PASS**

This pass applies to the **controller-ready conservative extrema**:

- `min_operational_height = 0.3932865805 m`
- `max_operational_height = 0.4128130092 m`

It does **not** claim:

- absolute mechanical min/max height,
- success for the broader static-only envelope,
- success for local transition recovery beyond the hold/sanity checks performed here.

## Limitations

- The broader static search envelope contains poses that are physically valid but not dynamically valid under the current controller.
- Local transition recovery near the extrema was not exercised in this run and remains future validation work.
- These results are specific to the current simulation/controller stack and selected D2 profile.
