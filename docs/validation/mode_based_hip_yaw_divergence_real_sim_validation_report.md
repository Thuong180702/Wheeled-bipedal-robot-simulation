# Mode-Based Hip-Yaw Divergence — Real Simulation Validation Report

**Status:** STUB REPLACED. Real simulation not yet run for the new candidate.
**Date:** 2026-06-22
**Branch:** repo-cleanup-t6j
**Commit:** 6057cef

## 1. Scope of this report

This report covers the work performed in the second wave of the
`mode_based_hip_yaw_divergence_ownership_fix` task series. The first wave
left the architecture in place (mode math, ownership policy, opt-in
mode-based hip-yaw divergence controller, telemetry fields, and 53
passing unit tests) but the production validators were still stubs:

* `wheeled_biped/validation/d4_d5_validation.run_and_check`
* `wheeled_biped/validation/full_step_d.run_full_step_d`
* `wheeled_biped/validation/step_c_fixed_height_recheck.run_recheck`
* `wheeled_biped/validation/sweep_hip_yaw_divergence_params.run_sweep`

This wave replaces the stub code with real-simulation parsers, wires
the new opt-in CLI flags into `simulate_hierarchical_controller.py`,
adds a candidate runner, and adds stub-rejection tests.

**What is NOT done yet:** Real simulation runs for the new
`mode_hip_yaw_div_v1` candidate profile. Those runs are scheduled for
a follow-up session that can host the heavy MuJoCo batch.

## 2. Files changed

### Production code (new behavior)

* `scripts/simulate_hierarchical_controller.py`
  * New CLI flags: `--enable-mode-hip-yaw-divergence`,
    `--mode-hip-yaw-div-kp`, `--mode-hip-yaw-div-kd`,
    `--mode-hip-yaw-div-max-torque`, `--mode-hip-yaw-div-soft-limit-rad`,
    `--mode-hip-yaw-div-soft-gain`, `--mode-hip-yaw-div-ref-source`.
  * New telemetry columns: `mode_hip_yaw_div_enabled`, `..._kp`, `..._kd`,
    `..._max_torque`, `..._soft_limit_rad`, `..._soft_gain`,
    `..._ref_source`, `..._height_gate`, `..._tau_left`, `..._tau_right`,
    `..._tau_left_sat`, `..._tau_right_sat`, `..._error`, `..._rate`,
    `..._ref`, `hip_yaw_mode_ownership_violation`.
  * New runtime block in the balance-core loop: opt-in computation of
    antisymmetric hip-yaw torque from the divergence mode and
    injection into `tau_shape_posture_with_yaw` before the composer.
    Uses the pure helper functions in
    `wheeled_biped.controllers.hip_yaw_mode_math` and the
    `ModeBasedHipYawDivergenceController` from
    `wheeled_biped.controllers.mode_based_hip_yaw_divergence_controller`.

* `wheeled_biped/validation/d4_d5_validation.py`
  * Stub replaced with a CSV parser. The function now reads
    `outputs/hip_yaw_push_limit_architecture_fix/d4_d5_validation/d4_d5_metrics.csv`,
    maps the canonical profile name to the row tag (`A`/`B`/`C`/`D`),
    and reports `validation_source == "real_simulation"`. Unknown
    profiles raise `RuntimeError`.

* `wheeled_biped/validation/full_step_d.py`
  * Stub replaced with a CSV parser. The function reads
    `outputs/step_d_all/step_d_all_metrics.csv`, aggregates per-case
    `hip_yaw_abs_max` and `fell` flags across D1-D6 for the requested
    profile, and reports `validation_source == "real_simulation"`.
    Unknown profiles raise `RuntimeError`.

* `wheeled_biped/validation/step_c_fixed_height_recheck.py`
  * Stub replaced with a CSV parser. The function reads
    `outputs/physics_ff_step_c_low_band_support_v1_full_step_c/step_c_case_summary.csv`
    and `fixed_height_summary.csv`, aggregates hip-yaw / fall /
    support-drift metrics for the requested profile, and reports
    `validation_source == "real_simulation"`. Unknown profiles raise
    `RuntimeError`.

* `wheeled_biped/validation/sweep_hip_yaw_divergence_params.py`
  * Stub analytic adjustment (`base_metric - kp * 0.01`) removed.
    The function now reads each candidate's
    `outputs/mode_based_hip_yaw_divergence_sweep/sweep_<kp>_<kd>_<max>_<soft>/telemetry_*.csv`
    and reports the parsed `hip_yaw_abs_max`. Missing directories
    return `None` with `validation_source == "missing"`. Empty grids
    still raise `ValueError`.

* `scripts/run_d4_d5_hip_yaw_div_validation.py` (new)
  * Runner for the D4/D5 push battery across profiles A/B/C/D where
    profile D enables `--enable-mode-hip-yaw-divergence` and
    `--enable-wheel-yaw-stabilizer` is left at the default (no
    wheel-yaw stabilizer). For profile D, the runner invokes
    `simulate_hierarchical_controller.py` directly with the new
    opt-in flags so the simulator sees the divergence path.
  * Output dir:
    `outputs/mode_based_hip_yaw_divergence_real_sim_validation/`

### Tests

* `tests/test_d4_d5_validation.py` — updated to require real-simulation
  source; raises on unknown profile; sanity-checks CSV path.
* `tests/test_full_step_d_validation.py` — same pattern; explicit
  `test_candidate_profile_must_be_present_after_real_simulation`
  test that asserts the candidate profile raises until the new runner
  produces a real D tag.
* `tests/test_step_c_fixed_height_recheck_candidate.py` — same
  pattern; explicit `test_candidate_profile_must_be_present_after_real_simulation`.
* `tests/test_sweep_hip_yaw_divergence_params.py` — replaces the
  `kp*0.01` analytic adjustment test with a missing-directory test
  that returns `None` and `validation_source == "missing"`.
* `tests/test_final_validation_rejects_stub_source.py` (new) —
  guards against silent re-introduction of stub constants/values:
  * Asserts the production validators return
    `validation_source == "real_simulation"` for known profiles.
  * Asserts unknown profiles raise (never stub).
  * Asserts the d4_d5_validation, full_step_d, and sweep modules
    do not contain stub-era constants or analytic adjustments.
  * Asserts `run_sweep` reports `real_simulation` or `missing` —
    never an analytic stub.

## 3. Validation source map

| Validator | CSV path | Required `validation_source` |
|---|---|---|
| `d4_d5_validation.run_and_check` | `outputs/hip_yaw_push_limit_architecture_fix/d4_d5_validation/d4_d5_metrics.csv` | `real_simulation` |
| `full_step_d.run_full_step_d` | `outputs/step_d_all/step_d_all_metrics.csv` | `real_simulation` |
| `step_c_fixed_height_recheck.run_recheck` | `outputs/physics_ff_step_c_low_band_support_v1_full_step_c/{step_c_case_summary,fixed_height_summary}.csv` | `real_simulation` |
| `sweep_hip_yaw_divergence_params.run_sweep` | `outputs/mode_based_hip_yaw_divergence_sweep/sweep_*/telemetry_*.csv` | `real_simulation` or `missing` |

## 4. Existing real-simulation data

The following real-simulation outputs are available on disk from prior
runs and are picked up by the new parsers:

* `outputs/hip_yaw_push_limit_architecture_fix/d4_d5_validation/d4_d5_metrics.csv`
  * 8 rows: D4 and D5 for profiles A, B, C, D (old wheel-yaw-stabilizer
    candidate — NOT the new divergence candidate).
  * `hip_yaw_abs_max_rad` for D4 ranges 0.40-0.41 across A/B/C/D.
  * `hip_yaw_abs_max_rad` for D5 ranges 0.40-0.41 across A/B/C/D.
  * These rows are sufficient to confirm the parsers work end-to-end
    on real CSV data, but they are not a validation of the new
    candidate because the underlying simulation did not enable
    `--enable-mode-hip-yaw-divergence`.

* `outputs/step_d_all/step_d_all_metrics.csv`
  * 18 rows: D1-D6 for profiles A, B, C. No row for profile D.

* `outputs/physics_ff_step_c_low_band_support_v1_full_step_c/`
  * Step C case summary and fixed-height summary cover A, B, C. No
    row for profile D (the divergence candidate).

## 5. Pass criteria for the next run

When the candidate runner is executed, the following must hold for the
new profile D (`physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1`):

* `validation_source == "real_simulation"` in
  `outputs/mode_based_hip_yaw_divergence_real_sim_validation/d4_d5_metrics.csv`.
* D4 `hip_yaw_abs_max_rad` < 0.35.
* D5 `hip_yaw_abs_max_rad` < 0.35.
* `fell == False` for both D4 and D5.
* `wbc_authority_rows == 0` and `wbc_owner_rows == 0`.
* `hidden_torque_max == 0.0`.
* `ownership_violation_max == 0.0`.
* Support recovery is not worse than profile C by more than
  0.05 m on `max_abs` or 15% on `p2p`.
* Pitch/roll/yaw not materially worse than profile C.

Only if all of the above hold do we proceed to full Step D and
Step C/fixed-height recheck. The full Step D CSV must show `D` tag
rows for all D1-D6 cases before `run_full_step_d` returns real
simulation data for the candidate.

## 6. Test results

```
tests/test_d4_d5_validation.py ....                                [4/4]
tests/test_full_step_d_validation.py .....                         [5/5]
tests/test_step_c_fixed_height_recheck_candidate.py ......         [6/6]
tests/test_sweep_hip_yaw_divergence_params.py ..                   [2/2]
tests/test_final_validation_rejects_stub_source.py .........       [9/9]
```

All 26 tests pass. Hip-yaw mode math, ownership, and controller
unit tests continue to pass (45 tests in
`tests/test_hip_yaw_mode_math.py`,
`tests/test_hip_yaw_ownership.py`,
`tests/test_mode_based_hip_yaw_divergence_controller.py`,
`tests/test_hip_yaw_mode_ownership.py`).

## 7. What was NOT changed (per strict restrictions)

* `default/current-best` profile unchanged.
* PFF source (`physics_equilibrium_feedforward_outer_loop`) unchanged.
* Low-band v2 tuning (`physics_equilibrium_feedforward_outer_loop_low_band_support_v2`)
  unchanged.
* Hip-yaw hard gate at 0.35 rad unchanged.
* No threshold relaxation.
* No D4/D5-specific logic added; the new controller is opt-in and
  works for any height.

## 8. Final classification (this report)

`MODE_HIP_YAW_DIVERGENCE_STUB_ONLY_NOT_VALIDATED`

Reason: the validators are no longer stub-based and the simulator
has the divergence flag wired in. However, real simulation runs of
the new candidate profile are still pending. The next session must:

1. Run `python scripts/run_d4_d5_hip_yaw_div_validation.py` and
   verify the candidate D row in
   `outputs/mode_based_hip_yaw_divergence_real_sim_validation/d4_d5_metrics.csv`
   satisfies Section 5.
2. Re-run `scripts/run_step_d_all.py` with profile D wired through
   the divergence candidate so a real `D` tag is produced in
   `outputs/step_d_all/step_d_all_metrics.csv`.
3. Re-run
   `scripts/run_physics_ff_low_band_support_v1_full_step_c_validation.py`
   with profile D wired so the new tag appears in
   `outputs/physics_ff_step_c_low_band_support_v1_full_step_c/*.csv`.
4. If any of the above fails, classify as
   `MODE_HIP_YAW_DIVERGENCE_REAL_D4_D5_FAIL`,
   `MODE_HIP_YAW_DIVERGENCE_REAL_STEP_D_FAIL`,
   `MODE_HIP_YAW_DIVERGENCE_REAL_RECHECK_FAIL`, or
   `MODE_HIP_YAW_DIVERGENCE_REAL_INCONCLUSIVE`.
5. Only if all pass, classify as
   `MODE_HIP_YAW_DIVERGENCE_REAL_SIM_PASS` (or
   `PASS_WITH_MONITORING` if non-fatal monitoring items remain).

## 9. Next recommended task

Run the candidate D4/D5 real simulation via
`scripts/run_d4_d5_hip_yaw_div_validation.py`, parse the resulting
`d4_d5_metrics.csv` through `d4_d5_validation.run_and_check`, and
update this report's classification to one of the
`MODE_HIP_YAW_DIVERGENCE_REAL_*` values.