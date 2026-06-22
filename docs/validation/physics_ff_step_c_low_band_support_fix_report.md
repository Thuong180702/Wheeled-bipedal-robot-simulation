# Physics FF Step C Low-Band Support Fix Report

Task: `low_band_support_outer_loop_pff_step_c_fix`

Date: 2026-06-21

Classification: `PHYSICS_FF_STEP_C_LOW_BAND_FIX_PASS_WITH_MONITORING`

## Scope

This pass addressed the local PFF focused `low_0p320` support-position regression without changing:

- PFF source calibration table, PCHIP interpolation, bounds, or version.
- Existing `physics_equilibrium_feedforward_outer_loop` behavior/defaults.
- B2v2 baseline profile behavior.
- WBC, HY2, hip-yaw gates, ownership gates, or Step D.

The change is an opt-in candidate profile:

- `physics_equilibrium_feedforward_outer_loop_low_band_support_v1`

## Files Read

Primary local files inspected:

- `wheeled_biped/controllers/physics_equilibrium_feedforward.py`
- `wheeled_biped/controllers/calibrated_outer_loop_functions_v2.py`
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- `wheeled_biped/controllers/support_outer_loop_low_band.py`
- `scripts/simulate_hierarchical_controller.py`
- `scripts/run_outer_loop_step_c_random_height.py`
- `scripts/run_physics_equilibrium_feedforward_fixed_height_validation.py`
- `tests/test_physics_equilibrium_feedforward_outer_loop.py`
- `tests/test_calibrated_support_position_outer_loop_pitch_ref_v2.py`
- `tests/test_sagittal_velocity_damped_balance_controller.py`
- `tests/test_simulation_telemetry_csv_writer.py`

Validation/output files inspected:

- `outputs/physics_ff_step_c/step_c_random_height_metrics.csv`
- `outputs/physics_ff_step_c/step_c_focused_low_0p320_A/seg000_low_0p320_300/telemetry_300.csv`
- `outputs/physics_ff_step_c/step_c_focused_low_0p320_B/seg000_low_0p320_300/telemetry_300.csv`
- `outputs/physics_ff_step_c/step_c_focused_high_0p480_B/seg000_high_0p480_300/telemetry_300.csv`
- `docs/validation/calibrated_outer_loop_fixed_height_v2_report.md`

Requested but missing locally:

- `docs/validation/physics_equilibrium_feedforward_step_c_report.md`
- `docs/validation/physics_ff_height_continuity_and_source_audit.md`
- `docs/validation/hip_yaw_telemetry_naming_and_gate_policy_fix.md`
- `docs/validation/integrated_yaw_hip_yaw_posture_architecture_audit.md`
- `docs/validation/physics_equilibrium_feedforward_fixed_height_audit.md`
- `tests/test_height_dependent_continuity.py`
- `tests/test_hip_yaw_gate_policy_and_metrics.py`

## Root Cause

Local focused `low_0p320` PFF telemetry did not contain a `0.145 m` max-abs support spike. The current-code focused PFF rerun produced:

- support max-abs: `0.1052808031 m`
- support peak-to-peak: `0.1643790036 m`

The regression source is still clear:

- PFF replaces the empirical pitch-ref height schedule with `physics_ff_pitch_eq_no_off_deg`.
- At `0.320 m`, PFF uses `-3.026 deg`.
- B2v2 uses about `-2.0 deg` plus a small dynamic support outer-loop term.
- Existing PFF keeps `outer_loop_height_schedule_required=True` while `pitch_ref_height_schedule_enabled=False`, so its dynamic support pitch-ref correction is inactive.

## Change

Added `wheeled_biped/controllers/support_outer_loop_low_band.py` with a smooth Gaussian height scale centered at `0.320 m`, sigma `0.006 m`.

Added opt-in profile:

- `physics_equilibrium_feedforward_outer_loop_low_band_support_v1`

At `0.320 m`, this profile applies:

- low-band pitch-ref trim: `+1.0 deg`
- effective support outer-loop Kp: `1.5 deg/m`
- Kd unchanged

Away from the low band, the trim and effective Kp fade to numerical zero. At `high_0p480`, the rerun candidate matched current PFF metrics exactly.

## Focused Metrics

Focused `low_0p320`, 300 steps, current-code PFF rerun:

| Profile | support max-abs m | support p2p m | out15 rows | pitch max-abs rad | hip-yaw max rad |
|---|---:|---:|---:|---:|---:|
| B2v2 baseline | 0.0715256847 | 0.1409512889 | 0 | 0.0960094013 | 0.0602129027 |
| Current PFF rerun | 0.1052808031 | 0.1643790036 | 0 | 0.1038886884 | 0.0569710284 |
| Low-band candidate | 0.0798309233 | 0.1458657813 | 0 | 0.1051791022 | 0.0617863126 |

Candidate deltas:

- vs current PFF: max-abs `-24.17%`, p2p `-11.26%`
- vs B2v2: max-abs `+11.61%`, p2p `+3.49%`

Focused artifacts:

- `outputs/physics_ff_step_c_low_band_support_fix/focused_low_0p320_summary.csv`
- `outputs/physics_ff_step_c_low_band_support_fix/focused_low_0p320_root_cause.json`
- `outputs/physics_ff_step_c_low_band_support_fix/focused_low_0p320_comparison_with_candidate.csv`
- `outputs/physics_ff_step_c_low_band_support_fix/focused_low_0p320_candidate_telemetry_300.csv`
- `outputs/physics_ff_step_c_low_band_support_fix/focused_low_0p320_current_pff_telemetry_300.csv`

## High-Band Check

Focused `high_0p480`, 300 steps, current-code rerun:

| Profile | support max-abs m | support p2p m | low-band scale | effective Kp | pitch trim deg |
|---|---:|---:|---:|---:|---:|
| Current PFF rerun | 0.0275865155 | 0.0394664469 | 0.0 | 1.0 telemetry default | 0.0 |
| Low-band candidate | 0.0275865155 | 0.0394664469 | 3.8387e-155 | 5.7581e-155 | 3.8387e-155 |

The scoped low-band profile did not change high-band behavior in the current-code rerun.

## Fixed-Height Sanity

Fresh 10-height, 300-step candidate sanity sweep:

- Output: `outputs/physics_ff_step_c_low_band_support_fix/fixed_height_sanity_candidate_scoped_summary.csv`
- JSON: `outputs/physics_ff_step_c_low_band_support_fix/fixed_height_sanity_candidate_scoped_summary.json`
- all safe: `true`
- max support max-abs: `0.1253498540 m`
- max support p2p: `0.1703097069 m`
- total rows above `0.15 m`: `0`
- no falls
- no WBC ownership rows
- no hidden torque or ownership violations
- hip-yaw remained below gate

Worst per-height candidate rows:

| Metric | Height | Value |
|---|---|---:|
| max support max-abs | `low_0p340` | 0.1253498540 m |
| max support p2p | `low_0p330` | 0.1703097069 m |
| focused low support max-abs | `low_0p320` | 0.0798309233 m |
| protected high support max-abs | `high_0p480` | 0.0275865155 m |

## Tests

Compile check passed:

- `scripts/simulate_hierarchical_controller.py`
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- `wheeled_biped/controllers/support_outer_loop_low_band.py`
- `tests/test_support_outer_loop_low_band_pff.py`

Pytest passed:

- `tests/test_support_outer_loop_low_band_pff.py`
- `tests/test_physics_equilibrium_feedforward_outer_loop.py`
- `tests/test_calibrated_support_position_outer_loop_pitch_ref_v2.py`
- `tests/test_sagittal_velocity_damped_balance_controller.py`
- `tests/test_simulation_telemetry_csv_writer.py`

Result: `557 passed in 5.33s`

Missing tests not run:

- `tests/test_height_dependent_continuity.py`
- `tests/test_hip_yaw_gate_policy_and_metrics.py`

## Decision

`PHYSICS_FF_STEP_C_LOW_BAND_FIX_PASS_WITH_MONITORING`

Reason:

- The focused `low_0p320` support-position regression is reduced.
- The candidate is within B2v2 tolerance on focused p2p.
- The 10-height fixed sanity sweep is safe.
- High-band current-code behavior is preserved when rerun.
- The full random/changing-height Step C candidate suite has not been rerun, so this is not a promotion result.

Step C status:

- Focused low-band fix: pass.
- Fixed-height sanity: pass.
- Full random/changing-height Step C suite: not claimed, still required before promotion.

Step D status:

- Blocked.
- Do not proceed to Step D.
- Do not promote PFF from this report alone.

Recommended next task:

- Run the full random/changing-height Step C suite for `physics_equilibrium_feedforward_outer_loop_low_band_support_v1`, compare against B2v2 and current PFF, and only then decide whether the opt-in candidate can be promoted or carried forward.
