# Balance-Core Extended Nominal Longevity Validation Specification

Date: 2026-05-28
Status: specification only
Scope: Step A only — extended nominal longevity validation for the current balance-core controller

---

## Section 1: Goal, Scope, and Non-Goals

**Goal:**

Validate how long the current nominal balance-core controller can survive beyond the already-confirmed 10000-step nominal standing runs, without changing controller behavior.

This specification covers only:
- long-duration nominal validation orchestration
- validation-only telemetry decimation and failure-window preservation
- per-duration summary reporting
- study summary generation under a dedicated output directory
- tests required for any validation infrastructure changes

**Target validation durations:**
- 10000 steps (baseline rerun)
- 20000 steps
- 50000 steps
- 100000 steps
- optional 200000 steps only if 100000 steps passes and runtime/file size remain reasonable

**Strict scope boundary:**

This work must stop after Step A is complete and reported.

**Non-goals:**
- no controller behavior changes
- no gain tuning
- no SagittalWheelBalanceController changes
- no ShapePostureController changes
- no SupportFeedforwardController changes
- no LateralRollBalanceController changes
- no joint ownership changes
- no four-source torque stack changes
- no new controller stages
- no legacy torque reintroduction
- no fake contact force
- no WBC reintroduction
- no true standing-height variants
- no height recovery
- no dynamic height transitions
- no position-hold / drift correction controller
- no random push robustness testing
- no roadmap steps B, C, D, E, or F

Only validation harness, CLI, telemetry decimation, summary reporting, and tests are in scope.

---

## Section 2: Existing System and Reuse Requirements

The implementation must reuse the current validation pipeline rather than introducing a duplicate long-run workflow unless the current validator proves structurally unable to support the required behavior.

**Existing components to reuse first:**
- [scripts/validate_balance_core.py](scripts/validate_balance_core.py)
- [scripts/simulate_hierarchical_controller.py](scripts/simulate_hierarchical_controller.py)
- [wheeled_biped/validation/balance_core_validator.py](wheeled_biped/validation/balance_core_validator.py)
- [wheeled_biped/validation/study_aggregator.py](wheeled_biped/validation/study_aggregator.py)
- [wheeled_biped/validation/failure_classifier.py](wheeled_biped/validation/failure_classifier.py)
- [wheeled_biped/validation/classification_report.py](wheeled_biped/validation/classification_report.py)
- [wheeled_biped/validation/telemetry_schema_checker.py](wheeled_biped/validation/telemetry_schema_checker.py)
- [wheeled_biped/validation/structural_invariant_checker.py](wheeled_biped/validation/structural_invariant_checker.py)

**Design rule:**
- extend the existing validator/classifier/reporting path
- do not create a second, parallel long-run validator unless reuse becomes clearly impossible
- if a limitation forces a small helper or extension, it must plug into the existing pipeline rather than duplicate it

---

## Section 3: Validation Workflow

The extended nominal longevity workflow must support arbitrary duration lists, stop at first failure by default, and optionally continue through all durations if explicitly requested.

**Preferred command shape:**

```bash
python scripts/validate_balance_core.py --durations 10000,20000,50000,100000
```

**Required behavior:**
1. Validate durations in requested order
2. Run nominal balance-core mode only
3. Reuse the existing simulation + validation + classification pipeline
4. Stop at first failing duration by default
5. Support `--continue-all` to run remaining durations after a failure
6. Produce per-duration artifacts and an overall study summary in a dedicated Step A output directory

**Required default output directory for Step A:**

```text
outputs/balance_core_extended_longevity/
```

**Required artifacts:**
- `extended_longevity_summary.json`
- `extended_longevity_summary.md`
- per-duration reports
- telemetry file paths or decimated telemetry file paths
- failure report if any duration fails

---

## Section 4: Telemetry Decimation Requirements

Long runs must avoid unnecessarily large telemetry files, but decimation must be strictly logging-only.

### 4.1 Logging-only rule

Telemetry decimation must not change simulation or controller behavior.

That means:
- no change to control timestep
- no change to physics timestep
- no change to controller update frequency
- no change to controller inputs
- no change to termination logic
- no change to torque composition
- no change to balance-core ownership or source stack

Only the telemetry writing/reporting path may change.

### 4.2 Default behavior must remain unchanged

The existing telemetry behavior must remain unchanged unless explicit long-run logging flags are used.

If the user runs the current nominal commands without long-run logging flags, output behavior must remain compatible with current usage.

### 4.3 Decimation trigger and mode

For Step A, the preferred design is explicit write-time telemetry decimation controlled by validation-only CLI flags.

Recommended behavior:
- default: full-rate telemetry, same as current behavior
- long-run mode: save every Nth row to the main CSV
- preserve a full-rate rolling window around failure if failure occurs
- keep continuous summary metric accumulation over all steps when required metrics would otherwise be distorted by decimation

### 4.4 Decimation must not hide failures

Decimated telemetry must not hide threshold crossings, architecture regressions, or failure timing.

Therefore, at least one of the following must be true:
1. required summary/failure metrics are accumulated online over all steps during simulation, not inferred only from decimated rows, or
2. any metric that is only computed from decimated rows must be clearly marked as approximate/limited in the report

For Step A, the design should prefer online accumulation of required whole-run summary metrics so the final report remains representative of the full run.

### 4.5 Failure-window preservation

If a run fails, the output must preserve a full-rate rolling telemetry window around the failure.

Required behavior:
- maintain a rolling in-memory buffer of recent full-rate rows during long-run logging mode
- on termination/failure, write a full-rate failure-window artifact that includes the pre-failure window and the final failure rows
- this failure window must be sufficient for the existing classifier/reporting diagnostics or any post-run inspection

The main decimated CSV may remain decimated, but the failure-window artifact must capture the local dynamics at full rate.

### 4.6 Schema compatibility

Telemetry decimation must not break schema validation or structural invariant checking for the produced artifacts that the validator consumes.

If validation continues to run on the main telemetry CSV, that CSV must preserve the required column schema.

If full-run summary statistics are accumulated online, the summary/report generator may combine:
- decimated CSV data for row-based artifact persistence
- online summary accumulators for whole-run maxima/minima/counts/percentages
- failure-window full-rate data for local failure diagnosis

---

## Section 5: Metrics and Reporting Requirements

For every tested duration, the final Step A reporting must include the following fields.

### 5.1 Required per-duration fields

- `requested_steps`
- `survived_steps`
- `pass_fail`
- `terminated`
- `termination_reason`
- `final_sim_time_s`
- `primary_failure_mode` if any
- `secondary_failure_modes` if any
- `structural_invariants_passed`
- `ownership_violation_count_max`
- `hidden_torque_norm_max`
- `tau_wbc_norm_max`
- `pitch_x_rad_min`
- `pitch_x_rad_max`
- `pitch_x_rad_rms`
- `roll_y_rad_min`
- `roll_y_rad_max`
- `roll_y_rad_rms`
- `com_z_m_min`
- `com_z_m_max`
- `com_z_m_drift`
- `wheel_vel_mean_rad_s_min`
- `wheel_vel_mean_rad_s_max`
- `wheel_vel_mean_rad_s_rms`
- `wheel_velocity_trend`
- `contact_state_summary`
- `left_wheel_contact_validity`
- `right_wheel_contact_validity`
- `torque_saturation_percentage_per_joint`
- `torque_rate_saturation_percentage_per_joint`
- `telemetry_csv_path`
- `validation_report_path`

### 5.2 Whole-run metric integrity requirement

The following metrics must be computed over all simulation steps, not only decimated rows, unless an explicit limitation is documented in the report:
- max pitch
- max roll
- min com_z
- wheel velocity max
- saturation percentage
- contact validity summary
- ownership violation maxima/counts
- hidden torque maxima
- WBC norm maxima

Preferred design:
- compute these online over all steps during simulation or validation
- store them in summary metadata or a summary sidecar
- surface them in per-duration and overall study reports

If any metric cannot be computed over all steps without deeper refactoring, the limitation must be made explicit in both JSON and markdown outputs rather than silently implying full-run exactness.

### 5.3 Failure reporting

If a duration fails:
- stop at that duration unless `--continue-all` is explicitly set
- classify the failure using the existing temporal root-cause classifier
- preserve the normal failure report path
- report the next allowed fix scope only
- do not implement any controller fix inside this Step A work

### 5.4 Success reporting

If all required durations pass through 100000 steps, the final summary must state:

```text
long_duration_survival_passed_up_to_100000_steps
```

Do not claim infinite stability.
Do not claim anything beyond the maximum tested duration.

---

## Section 6: CLI and Output Design

The implementation should extend existing CLIs rather than replacing them.

### 6.1 Validator CLI expectations

[scripts/validate_balance_core.py](scripts/validate_balance_core.py) should remain the main entry point for Step A.

It should support:
- custom duration lists
- stop-at-first-failure by default
- `--continue-all`
- output directory selection
- long-run logging controls that are explicitly opt-in

Potential long-run logging controls may include:
- telemetry decimation interval
- failure-window size
- summary-only or summary-sidecar toggles if needed

The exact flag names should remain minimal and validation-oriented.

### 6.2 Simulator CLI expectations

[scripts/simulate_hierarchical_controller.py](scripts/simulate_hierarchical_controller.py) may be extended with validation-only logging flags, but these flags must:
- affect logging only
- default to current behavior when omitted
- remain compatible with balance-core mode

### 6.3 Output structure

Preferred structure:

```text
outputs/balance_core_extended_longevity/
  extended_longevity_summary.json
  extended_longevity_summary.md
  longevity/
    longevity_10000/
      telemetry_10000.csv
      validation_report.json or md
      failure_report_10000.md   # only if failed
      failure_window_10000.csv  # only if failed in decimated mode
    longevity_20000/
    longevity_50000/
    longevity_100000/
```

Exact filenames may align with the current validator/study aggregator conventions, but the dedicated Step A directory and top-level summary filenames are required.

---

## Section 7: Testing Requirements

If infrastructure changes are made, tests must be added or updated.

Minimum required coverage:
- arbitrary duration list parsing
- stop-at-first-failure behavior
- `--continue-all` behavior
- long-run summary generation
- telemetry decimation preserves required schema
- telemetry decimation does not break structural invariant checks
- failure-window preservation behavior
- no controller behavior change is required for long-run validation mode

Tests should focus on validation infrastructure and artifact behavior, not controller retuning.

---

## Section 8: Constraints on Interpretation and Claims

The final Step A outputs and report must explicitly preserve the following truths:
- no controller behavior was modified
- no gains were tuned
- WBC remained off
- no legacy torque source was activated
- torque ownership remained unchanged
- the four-source balance-core stack remained unchanged
- only validation/logging/reporting infrastructure was extended

The final report must also include:
- files changed
- tests added or updated
- tests run and results
- exact commands run
- maximum confirmed survival steps
- whether failure occurred before 100000
- first failing duration, if any
- primary failure mode, if any
- output directory

---

## Section 9: Recommended Implementation Shape

The preferred implementation shape is a minimal extension of the current system:

1. Extend [scripts/validate_balance_core.py](scripts/validate_balance_core.py) to drive Step A durations and write the required top-level summary files under `outputs/balance_core_extended_longevity/`
2. Extend [wheeled_biped/validation/balance_core_validator.py](wheeled_biped/validation/balance_core_validator.py) to pass through long-run logging options, preserve stop-on-first-failure semantics, and surface richer per-duration results
3. Extend [scripts/simulate_hierarchical_controller.py](scripts/simulate_hierarchical_controller.py) with opt-in logging-only decimation and failure-window preservation
4. Extend [wheeled_biped/validation/study_aggregator.py](wheeled_biped/validation/study_aggregator.py) to emit the required Step A summary fields and whole-run metrics
5. Reuse the existing failure classifier and report generator without changing controller behavior

This approach stays within Step A, avoids duplicate wrappers, and keeps the work limited to validation harness, CLI, telemetry management, and reporting.

---

## Section 10: Approval Gate

After this specification is approved, the next step is to create an implementation plan for the minimal validator extension.

Do not begin implementation before the spec is reviewed and approved.
