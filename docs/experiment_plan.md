# T6J Centering Bias Trim Implementation Plan

## Goal
Implement a new opt-in sagittal authority profile, `T6J_centering_bias_trim`, based on `T6I_phase_aware_release`, to reduce persistent one-sided positive support drift without changing T6I/T6F/T5 behavior, without pitch suppression or damping suppression, and without altering default controller paths.

## What I inspected
- Profile registry and CLI wiring in [scripts/simulate_hierarchical_controller.py](scripts/simulate_hierarchical_controller.py)
- Controller schedule, internal state, telemetry, and T6I logic in [wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py](wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py)
- Existing controller tests in [tests/test_sagittal_velocity_damped_balance_controller.py](tests/test_sagittal_velocity_damped_balance_controller.py)
- Existing T6I 5000-step summary artifacts in:
  - [outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6i_high_0p480_5000_validation.json](outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6i_high_0p480_5000_validation.json)
  - [outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6i_high_0p480_5000_window_metrics.csv](outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6i_high_0p480_5000_window_metrics.csv)

## Findings that drive the design
1. T6I already has a stateful release mechanism (`_t6i_error_history`, `_t6i_current_cap`, `_t6i_converging`) and preserves pitch/damping; this must remain untouched.
2. The controller already supports stateful add-on mechanisms via schedule flags plus private controller state, so T6J can follow that pattern rather than add a separate subsystem.
3. Existing artifacts suggest T6I’s positive drift is persistent, not just an early transient:
   - 0-crossing windows include multiple 500-step segments with zero crossings = 0.
   - Late windows still show high positive occupancy and outside-band percentages.
   - T6I convergence/release activity is present but low, suggesting cap decay is not a centering mechanism.
4. There is already a `bias_cancel` concept in the controller, but the user explicitly requested a new `T6J` profile on top of T6I with different gating, telemetry, staged validation, and no direct reuse claim. I will implement the requested T6J mechanism explicitly rather than silently repurpose another path.

## Root-cause audit approach (Phase 1)
Before implementing T6J, complete the required T6I audit using the existing 5000-step telemetry file for the high_0p480 run.

### Data source resolution
Use the correct drift column priority exactly as requested:
1. `active_pitch_crossing_signed_error_m`
2. `sagittal_position_error_m`
3. `support_position_error_m`
4. `hip_yaw_comp_support_error_m`

If the existing T6I telemetry CSV is not already copied into the Step E active-pitch-crossing directory, locate the matching run sidecar/CSV by timestamp or profile name first, then analyze that exact file.

### Audit outputs
Produce:
- [docs/validation/t6i_positive_bias_root_cause_audit.md](docs/validation/t6i_positive_bias_root_cause_audit.md)
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6i_positive_bias_root_cause_audit.json`

### Audit contents
Compute and report:
- min/max/mean/median/final error
- positive %, negative %, zero crossings
- outside ±0.03/0.05/0.08/0.10/0.15
- 500-step window means, positive %, zero crossings
- correlation with T6I cap decay / convergence flags
- correlation with `arch_fix_active`
- correlation with pitch, wheel velocity, final wheel torque
- explicit answers to the 8 required diagnostic questions
- one required classification label

## T6J design approach (Phase 2)
Create a new explicit add-on mechanism in the controller schedule and state machine:

### New schedule fields in `SagittalAuthoritySchedule`
Add T6J-only fields, all default-off / zeroed so no existing behavior changes:
- `t6j_bias_trim_enabled`
- `t6j_bias_trim_window_steps`
- `t6j_bias_trim_enter_threshold_m`
- `t6j_bias_trim_exit_threshold_m`
- `t6j_bias_trim_max_tau_nm`
- `t6j_bias_trim_rate_nm_per_step`
- `t6j_bias_trim_decay_rate_nm_per_step`
- `t6j_bias_trim_only_when_upright`
- `t6j_bias_trim_only_when_contact_stable`
- `t6j_bias_trim_disable_if_pitch_gt_deg`
- `t6j_bias_trim_disable_if_roll_gt_deg`
- `t6j_bias_trim_disable_if_wheel_vel_gt_rad_s`
- `t6j_bias_trim_disable_if_abs_error_gt_m`

### New profile definition
Define `T6J_CENTERING_BIAS_TRIM` as a copy-pattern from T6I with all T6I values preserved and T6J fields enabled using the user-provided initial settings.

### Mechanism placement
Apply the trim on the same support recenter / position torque path after T6I cap selection but before final torque composition.

Reasoning:
- This satisfies the user’s requested placement.
- It preserves T6I’s phase-aware cap decay semantics.
- It avoids modifying pitch and damping terms.
- It keeps the correction bounded and additive rather than globally changing authority.

### Sign rule
Use the established `tau_position` sign convention:
- positive mean error → negative trim torque
- negative mean error → positive trim torque

Do not derive sign from final wheel torque.

### Safety / gating logic
T6J trim integrates only when all requested safety conditions pass; otherwise it decays toward zero. If `abs(error)` is too large, T6J will stop integrating and yield to the existing T6I/T6F emergency logic.

### T6J telemetry
Add the requested fields exactly, emitted every step with safe defaults when disabled.

### Design outputs
Produce:
- [docs/validation/t6j_centering_bias_trim_design.md](docs/validation/t6j_centering_bias_trim_design.md)
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_centering_bias_trim_design.json`

## Implementation plan (Phase 3)
### Files to modify
1. [wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py](wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py)
   - add schedule fields
   - add `T6J_CENTERING_BIAS_TRIM`
   - add registry entry
   - add controller private state for history / trim / durations
   - compute mean-error window, safety gates, trim target, rate limiting, decay, and applied trim
   - inject trim after T6I cap selection and before final torque composition
   - emit telemetry
2. [scripts/simulate_hierarchical_controller.py](scripts/simulate_hierarchical_controller.py)
   - register T6J in `SAGITTAL_AUTHORITY_PROFILES`
   - ensure CLI choice exposure if needed
   - ensure run summary/telemetry output includes new fields
3. [tests/test_t6j_centering_bias_trim.py](tests/test_t6j_centering_bias_trim.py)
   - add the requested dedicated tests
4. [tests/test_simulation_telemetry_csv_writer.py](tests/test_simulation_telemetry_csv_writer.py)
   - extend CSV field expectations for T6J telemetry if the dedicated test does not already cover writer behavior sufficiently
5. Optionally extend [tests/test_sagittal_velocity_damped_balance_controller.py](tests/test_sagittal_velocity_damped_balance_controller.py) only if shared helpers or regression assertions belong there.

## Test strategy (Phase 4)
Add a focused dedicated test file for T6J and then rerun the required regression suite.

### New tests
Cover all requested invariants:
- profile existence and opt-in status
- T6J inherits T6I values
- T6I/T6F/T5 unchanged
- positive and negative mean error activation
- sign correctness of trim direction
- threshold gating / exit decay / bound / rate limit
- safety gate disable cases
- no pitch suppression / no damping suppression
- T6I cap decay still works
- final motor cap still respected
- telemetry presence
- CSV writer presence
- no WBC path change / no HY2-DIV default change

### Regression suite
Rerun exactly the required Phase 4 tests listed by the user.

### Test artifacts
Produce:
- [docs/validation/t6j_centering_bias_trim_implementation_tests_report.md](docs/validation/t6j_centering_bias_trim_implementation_tests_report.md)
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_centering_bias_trim_implementation_tests_summary.json`

## Validation workflow (Phases 5-9)
Run staged validation exactly in order, stopping on failure gates:
1. 500-step T6I vs T6J diagnostic at high_0p480
2. 1200-step T6J high_0p480 only if Phase 5 passes
3. 2000-step T6J high_0p480 only if Phase 6 passes
4. 5000-step T6J high_0p480 only if Phase 7 passes
5. 2000-step height ladder sanity only if Phase 8 passes/pass-with-monitoring

For each stage, create the required report markdown and JSON summary in the exact user-specified locations.

### Validation analysis approach
For each run:
- prefer sidecar + telemetry CSV copied into dedicated output folders
- compute the requested drift metrics using the same drift-column priority as Phase 1
- include safety/ownership/WBC checks from telemetry
- include T6J trim activation %, mean bias, tau range, and block reasons for T6J runs
- compare against T6I reference where requested

## Final report (Phase 10)
Produce:
- [docs/validation/t6j_centering_bias_trim_final_report.md](docs/validation/t6j_centering_bias_trim_final_report.md)
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_centering_bias_trim_summary.json`

The final report will answer the 10 required questions and end with exactly one required decision label.

## Risks / checks
1. **Wrong injection point** could accidentally change emergency recovery instead of slow centering. I will inject only after T6I cap selection and keep trim bounded/rate-limited.
2. **Silent pitch/damping interaction** could violate restrictions. I will keep pitch and damping terms untouched and test for this explicitly.
3. **Telemetry drift source mismatch** could invalidate conclusions. I will use the requested priority order consistently across audit and validation.
4. **State carryover bugs** are possible because this controller already holds many internal states. I will keep T6J state names isolated and default-off.
5. **High_0p480 improvement but low-height regression** is a real risk, hence the ladder sanity stage remains mandatory before any “better than T6I” conclusion.

## Minimal-diff principle
This plan intentionally avoids:
- editing T6I/T6F/T5 behavior directly
- changing default profiles
- enabling WBC/HY2-DIV/legacy WBC
- pitch suppression, damping suppression, sign flips, or broad authority increases
- Step C / Step D runs
- commits

## Requested approval boundary
After approval, I will execute the plan in this order:
1. Phase 1 audit artifacts
2. Phase 2 design artifacts
3. T6J implementation
4. tests and implementation report
5. staged simulations and validation reports
6. final report and final decision
