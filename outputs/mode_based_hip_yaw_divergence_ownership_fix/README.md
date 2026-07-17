# Mode-Based Hip-Yaw Divergence Ownership Fix - Outputs

This directory is the canonical output drop for the
`2026-06-22-mode_based_hip_yaw_divergence_ownership_fix` series.

## Status

The unit-test, ownership, and stub-validation phases of the fix are complete
(see `docs/validation/mode_based_hip_yaw_divergence_ownership_fix_report.md`).
This directory is intentionally a **placeholder** until the heavy simulation
runs (D4/D5, full Step D, Step C fixed-height recheck, parameter sweep) are
executed against the new candidate profile.

## What will go here

| Subdir / file | Source stub | Description |
| --- | --- | --- |
| `d4_d5/` | `wheeled_biped.validation.d4_d5_validation.run_and_check` | Per-scenario D4/D5 push recovery CSVs and aggregated `hip_yaw_abs_max` |
| `full_step_d/` | `wheeled_biped.validation.full_step_d.run_full_step_d` | Aggregated Step D scenario battery hip-yaw metrics |
| `step_c_recheck/` | `wheeled_biped.validation.step_c_fixed_height_recheck.run_recheck` | Step C fixed-height recheck output (h = 0.70 to 0.40 m) |
| `sweep/` | `wheeled_biped.validation.sweep_hip_yaw_divergence_params.run_sweep` | Parameter sweep CSV (`kp`, `kd`, `max_torque` grid) |
| `reconstruct/` | `wheeled_biped.validation.reconstruct_hip_yaw_divergence.reconstruct` | Reconstructed hip-yaw common / divergence metrics per (profile, case) |

## Candidate profile to run

`physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1`

The opt-in `mode_hip_yaw_divergence` block in
`configs/training/balance_residual.yaml` must be set to `enabled: true` for
this profile.

## Related files

- Validation report: `docs/validation/mode_based_hip_yaw_divergence_ownership_fix_report.md`
- Controller: `wheeled_biped/controllers/mode_based_hip_yaw_divergence_controller.py`
- Ownership: `wheeled_biped/controllers/hip_yaw_ownership.py`
- Mode math: `wheeled_biped/controllers/hip_yaw_mode_math.py`
- HY2-DIV integration: `wheeled_biped/controllers/shape_posture_controller.py`

## Outstanding gates

- D4/D5 push recovery: real simulation pending.
- Full Step D: real simulation pending.
- Step C fixed-height recheck: real simulation pending.
- Parameter sweep: real simulation pending.

Until these are run, the validation report classifies the fix as
`MODE_HIP_YAW_DIVERGENCE_FIX_PASS_WITH_MONITORING`.
