# Step C Height Recovery Specification

Status: **DRAFT — awaiting approval**  
Date: 2026-06-01  
Roadmap step: **Step C — Height recovery to target height**

## 1. Scope

Step C validates whether the existing WBC-off balance-core controller can recover body height from kinematically consistent small standing-height variants while preserving the completed Step E standing-position behavior.

Step C covers:

- target height definition from official Step E telemetry and equilibrium metadata
- staged low/high kinematically consistent standing-height variants
- height recovery metrics and pass/fail thresholds
- position-hold, posture, balance, contact, and structural invariant preservation
- telemetry requirements for diagnosing failure before changing controller behavior
- official output artifacts required before Step C can be marked done

Initial Step C work is diagnostic-only. Controller behavior must remain unchanged until baseline diagnostics prove a specific height-recovery failure mode and motivate a minimal targeted fix.

## 2. Non-goals

Step C is not:

- dynamic height transition while running
- push robustness
- WBC reintroduction
- blind gain tuning
- a full-body QP/WBC controller
- aggressive height trajectory tracking
- stand-up recovery
- Step D or Step F validation

Do not claim Step C complete until official Step C validation artifacts pass.

## 3. Starting controller and architecture

Step C starts from the official Step E production controller:

- simulation entry point: [simulate_hierarchical_controller.py](../../scripts/simulate_hierarchical_controller.py)
- controller mode: `balance-core`
- sagittal controller: `velocity-damped`
- four-source torque stack:
  1. `ShapePostureController`
  2. `SupportFeedforwardController`
  3. `SagittalVelocityDampedBalanceController`
  4. `LateralRollBalanceController`
- WBC raw computation may remain diagnostic-only, but WBC must not contribute to applied torque

The existing simulation supports `--initial-root-z-perturbation`, but the Step C initialization audit found that this path changes only `qpos[2]` after equilibrium capture and does not update hip/knee posture or capture a height-specific equilibrium. Root-z-only perturbation is therefore not the official Step C height-change method unless a later static validation proves it physically valid.

## 4. Target height definition

The controlled height signal for Step C is **CoM height**, using `com_z_m` when present and falling back to `com_z` only for legacy telemetry compatibility.

Rationale:

- official Step E validation reports balance height through `com_z`
- balance-core telemetry exposes explicit `com_z_m`
- `root_z` is an initialization coordinate and may not equal body/CoM standing height
- torso height is not currently the official Step E validation metric

The Step C target height is the validated Step E standing CoM height:

- primary target: median `com_z_m` over the final 500 rows of official Step E telemetry
- official Step E reference telemetry: `outputs/hierarchical_controller_sim/telemetry_1780289121.csv`
- computed preliminary value from official telemetry: `target_com_z_m = 0.40774276852607727`
- supporting official Step E values:
  - first CoM height: `0.4041118323802948` m
  - final CoM height: `0.408227801322937` m
  - min CoM height: `0.4038352966308594` m
  - max CoM height: `0.40854722261428833` m
  - all-run median CoM height: `0.40779018402099609` m

The official Step C reference extractor must write the exact target and source statistics to `outputs/step_c_height_recovery/step_c_height_reference.json`.

## 5. Initial disturbances and test cases

Step C official cases must use kinematically consistent standing-height variant initialization, not root-z-only displacement. Each height case must start from a physically valid pose that preserves wheel-floor contact, symmetric left/right leg posture, support-center alignment, and near-equilibrium pitch/roll/yaw before the dynamic rollout begins.

Root-z-only offsets are diagnostic-only after the initialization audit in `outputs/step_c_initialization_method_audit/`. The audit confirmed that `--initial-root-z-perturbation` changes only `qpos[2]` after equilibrium capture, which can create physically inconsistent startup contact transients and wheel velocity spikes.

Preferred initialization sources:

1. Reuse existing Step B true standing-height variant setup JSONs when the target height matches an available variant.
2. If a required height is missing, add a diagnostic-only symmetric hip-pitch/knee pose search or IK generator that calibrates root height for wheel-floor contact before simulation.
3. Capture per-variant equilibrium references from the validated initial pose before running the dynamic Step C case.

Required official case matrix:

| Case | Target standing-height variant | Purpose |
|---|---:|---|
| nominal | validated Step E/Step B nominal pose | sanity check and Step E parity |
| low_1cm | nominal CoM height `-0.01 m` valid pose | first low-height recovery gate |
| high_1cm | nominal CoM height `+0.01 m` valid pose | first high-height recovery gate |
| low_2cm | nominal CoM height `-0.02 m` valid pose if feasible | medium low-height recovery gate |
| high_2cm | nominal CoM height `+0.02 m` valid pose if feasible | medium high-height recovery gate |
| low_3cm | nominal CoM height `-0.03 m` valid pose if feasible | final low-height diagnostic gate |
| high_3cm | nominal CoM height `+0.03 m` valid pose if feasible | final high-height diagnostic gate |

Static initialization validation is mandatory before each dynamic case:

- both wheel contacts valid after reset
- left/right wheel floor contacts true
- no non-wheel ground penetration/contact
- contact force positive and physically reasonable
- support center near body/CoM projection
- CoM height close to requested target
- pitch/roll/yaw near equilibrium
- hip/knee within joint limits
- left/right leg symmetry preserved unless intentionally testing asymmetry
- Step E position reference preserved or explicitly re-captured for the height variant

Stop-gated progression:

1. run nominal first
2. run `±1 cm` only if nominal passes structural and safety gates
3. run `±2 cm` only if both `±1 cm` cases pass
4. run `±3 cm` only if both `±2 cm` cases pass

If a case fails, stop escalation and classify the failure before proposing any controller change.

## 6. Height recovery success criteria

A Step C case passes only if all criteria pass.

### 6.1 Survival

- final official validation: must survive `5000` control steps
- short diagnostic gates may use fewer steps only during development and must be labeled non-official
- no early termination from height or orientation checks

### 6.2 Height return

Height error is:

```text
height_error_m = com_z_m - target_com_z_m
height_error_abs_m = abs(height_error_m)
```

Thresholds:

- minimum final absolute height error: `<= 0.02 m`
- preferred final absolute height error: `<= 0.01 m`
- minimum steady-state absolute height error: `<= 0.02 m`
- preferred steady-state absolute height error: `<= 0.01 m`
- preferred recovery time: `<= 2.0 s`
- minimum recovery time: `<= 5.0 s`

Recovery time is the first time after perturbation when `height_error_abs_m <= 0.02 m` and remains within that band for a continuous hold window. The hold window must be defined by the validator; initial proposal: `0.5 s` / `50` control steps.

### 6.3 Height safety

- CoM height must never fall below the Step E safety floor derived from official Step E data minus a small diagnostic margin
- initial specification floor: `com_z_m >= 0.38 m`
- the validator must also report margin to official Step E minimum `0.4038352966308594 m`

### 6.4 Overshoot and oscillation

The validator must flag:

- `height_overshoot` when height crosses beyond the target by more than `0.02 m` after recovery
- `height_oscillation` when height repeatedly exits and re-enters the recovery band after first recovery, or when final-window peak-to-peak height exceeds `0.02 m`

These flags make the case fail unless explicitly classified as benign in a later approved spec revision.

## 7. Position hold preservation criteria

Step C must preserve Step E position hold.

Metric:

- primary: `support_position_error_m`
- fallback only if missing: reconstruct from support center telemetry if available; otherwise classify as `unclear_requires_more_telemetry`

Thresholds:

- max absolute support-position error: `<= 0.15 m`
- preferred max absolute support-position error: `<= 0.12 m`
- final absolute support-position error: `<= 0.15 m`
- preferred final absolute support-position error: `<= 0.10 m`

Official Step E reference:

- max absolute: `0.1044567514034454 m`
- final: `0.09135177303814725 m`
- RMS: `0.05704655732559888 m`

A height-recovery fix must not improve height by sacrificing Step E position hold.

## 8. Posture validity preservation criteria

Step C must preserve valid standing posture, not merely avoid falling.

Required thresholds:

- hip-yaw max absolute error: `<= 0.07 rad`
- percent absolute hip-yaw error `> 0.10 rad`: `0.0%`
- hip-yaw RMS must be reported
- hip pitch and knee errors must be reported when available
- joint-position error norm must be reported

Official Step E reference:

- hip-yaw max absolute: `0.0567 rad`
- hip-yaw RMS: `0.02281944898545975 rad`
- percent absolute hip-yaw error `> 0.10 rad`: `0.0%`

Failure to reconstruct posture validity is `unclear_requires_more_telemetry`, not pass.

## 9. Balance and contact safety criteria

Required thresholds:

- pitch_x max absolute: `<= 0.10 rad`
- roll_y max absolute: `<= 0.05 rad`
- contact valid percent: `>= 99.9%`
- wheel mean velocity max absolute: `<= 5.0 rad/s` preferred
- no termination from `height_too_low` or orientation failure
- no non-wheel floor contact unless explicitly classified and justified

Official Step E reference:

- pitch_x max absolute: `0.07077135067308149 rad`
- roll_y max absolute: `0.012998944689273586 rad`
- contact valid percent: `100.0%`
- wheel mean velocity max absolute: `3.8395681381225586 rad/s`

## 10. Structural invariants

Step C must preserve Step E structural invariants:

- applied WBC contribution norm max: `0.0`
- WBC applied: `false`
- WBC computed only as diagnostic: `true`
- WBC contributed to `tau_total_raw_per_joint`: `false`
- `active_torque_owner_per_joint` must not include WBC
- `ownership_violation_count` max: `0`
- hidden torque norm max: `0.0`
- legacy torque paths remain off
- balance-core four-source ownership remains clean

Important interpretation rule:

- nonzero raw `tau_wbc_norm` does not count as WBC application if `applied_wbc_contribution_norm == 0`, WBC is absent from active owners, and WBC does not contribute to `tau_total_raw_per_joint`

## 11. Telemetry requirements

Step C telemetry must include or allow exact reconstruction of:

### Time and case metadata

- `source_step_index`
- `time`
- case name
- requested `initial_root_z_perturbation_m`
- `nominal_equilibrium_com_z_m`
- `initial_com_z_m_after_perturbation`
- `perturbation_applied_after_equilibrium_capture`

### Height metrics

- `target_height_m`
- `com_z_m`
- `height_error_m`
- `height_error_abs_m`
- `height_recovered`
- `height_recovery_time_s`
- `root_z` if available

### Position and posture metrics

- `support_position_error_m`
- hip-yaw errors or reconstructable hip-yaw joint/reference values
- hip pitch and knee errors if available
- joint position error norm
- `pitch_x_rad`
- `roll_y_rad`
- `yaw_z_rad`

### Contact and wheel metrics

- left/right wheel contact state
- `contact_force_valid`
- contact valid percent
- non-wheel floor contact count
- `wheel_vel_mean_rad_s`
- wheel torque saturation flags
- wheel torque-rate saturation flags when available

### Torque and ownership metrics

- four-source torque vectors:
  - `tau_shape_posture_per_joint`
  - `tau_support_feedforward_per_joint`
  - `tau_sagittal_wheel_balance_per_joint`
  - `tau_lateral_roll_balance_per_joint`
- `tau_total_raw_per_joint`
- `tau_total_clipped_per_joint`
- `tau_final_per_joint`
- `active_torque_owner_per_joint`
- `ownership_violation_count`
- `applied_wbc_contribution_norm`
- `hidden_torque_norm`
- torque saturation
- torque-rate saturation when available

If any required field is absent and cannot be reconstructed safely, the case is inconclusive and must be classified as `unclear_requires_more_telemetry`.

## 12. Validation gates

### Gate C0 — Spec approval

This specification must be reviewed and approved before implementation planning.

### Gate C1 — Height reference extraction

- compute target height from official Step E telemetry
- write `step_c_height_reference.json`
- no controller changes

### Gate C1b — Kinematically consistent height initialization

- root-z-only perturbation must not be used as the official Step C height-change method unless static validation proves it physically valid
- select or generate a valid standing-height pose for each requested target height
- validate wheel contacts, no ground penetration, support-center alignment, CoM target accuracy, posture symmetry, joint limits, and near-equilibrium orientation before dynamic rollout
- capture equilibrium references from the validated pose
- no controller torque/gain/ownership changes

### Gate C2 — Diagnostic runner readiness

- define case matrix
- run cases through existing Step E production path with validated height-variant initialization
- no controller changes

### Gate C3 — Validator readiness

- compute metrics, pass/fail, and failure classifications from telemetry
- validate structural invariants
- no controller changes

### Gate C4 — Baseline diagnostic sweep

- run stop-gated nominal and small perturbation cases
- decide whether current balance-core already recovers height
- no controller changes unless C4 fails and root cause is classified

### Gate C5 — Decision gate

Outcomes:

- `STEP_C_BASELINE_PASS`: all official cases pass without controller change
- `STEP_C_FIX_REQUIRED`: one or more cases fail with a clear root cause
- `STEP_C_INCONCLUSIVE`: telemetry is insufficient or results are contradictory

### Gate C6 — Optional minimal fix

Only allowed if C5 returns `STEP_C_FIX_REQUIRED`. The fix must be targeted to the classified failure and must preserve Step E invariants.

### Gate C7 — Official production validation

- run final 5000-step validation cases
- write official artifacts to `outputs/step_c_height_recovery/`
- Step C may be marked done only if official pass/fail summary passes

## 13. Failure classifications

Validators must classify failures using these labels:

- `height_not_recovered`: final or steady-state height error exceeds minimum band
- `height_recovery_too_slow`: recovery occurs after minimum recovery-time threshold
- `height_overshoot`: height overshoots beyond allowed band after recovery
- `height_oscillation`: repeated band exits or excessive final-window oscillation
- `position_regression`: support-position error exceeds Step E threshold
- `posture_regression`: hip-yaw or support-shape posture exceeds threshold
- `pitch_regression`: pitch exceeds safe Step C threshold
- `roll_regression`: roll exceeds safe Step C threshold
- `yaw_regression`: yaw/hip-yaw validity exceeds threshold
- `contact_invalid`: contact validity falls below threshold
- `wheel_velocity_runaway`: wheel mean velocity exceeds threshold or trends unstably
- `hidden_torque_nonzero`: hidden torque invariant fails
- `wbc_applied`: applied WBC contribution is nonzero or WBC owns any applied joint
- `ownership_violation`: ownership violation count exceeds zero
- `legacy_torque_path_enabled`: legacy torque path contributes in balance-core mode
- `torque_saturation_persistent`: torque saturation persists in a way that plausibly prevents recovery
- `torque_rate_saturation_persistent`: torque-rate saturation persists in a way that plausibly prevents recovery
- `root_z_only_initialization_artifact`: startup failure caused by displacing root height without updating leg posture or revalidating contacts/equilibrium
- `physically_inconsistent_height_perturbation`: height variant starts from invalid or unvalidated kinematics, contact, support-center, or posture state
- `unclear_requires_more_telemetry`: required telemetry is missing or contradictory

Each failed case must include primary and secondary classifications.

## 14. Allowed and forbidden implementation changes

### Allowed before baseline diagnostics

- add metric extraction utilities
- add validator/reporting scripts
- add tests for metrics and classification
- add non-invasive telemetry fields needed for Step C validation
- use existing `--initial-root-z-perturbation` only for diagnostic audits, not official Step C pass/fail cases unless static validation proves it physically valid
- write artifacts under `outputs/step_c_height_recovery/`

### Forbidden before baseline diagnostics prove root cause

- adding WBC to applied torque
- enabling legacy WBC torque paths
- flipping sagittal axis
- modifying hip-roll logic
- modifying sagittal position gains
- modifying position integral or support-position gains
- changing controller ownership rules
- weakening Step E validation thresholds
- claiming Step C done

### Allowed only after failed diagnostics and approved plan revision

- a minimal height-recovery modification targeted to a classified root cause
- added telemetry needed to prove or disprove the root cause
- focused tests that lock the root-cause behavior and Step E invariants

## 15. Required output artifacts

Official Step C output directory:

```text
outputs/step_c_height_recovery/
```

Required artifacts:

- `step_c_height_reference.json`
- `step_c_height_case_matrix.json`
- per-case telemetry CSVs
- `step_c_height_recovery_metrics.json`
- `step_c_failure_classification.json`
- `step_c_height_recovery_report.md`
- `step_c_pass_fail_summary.json`

The pass/fail summary must report:

- overall Step C verdict: `PASS`, `FAIL`, or `INCONCLUSIVE`
- final decision: `STEP_C_DONE`, `STEP_C_FIX_REQUIRED`, or `STEP_C_INCONCLUSIVE`
- whether controller behavior changed
- whether WBC was applied
- whether Step E invariants remained preserved
- exact commands used for official validation
- artifact paths

## 16. Approval requirement

No implementation plan or code changes may begin until this Step C specification is approved.

After approval, the next required artifact is:

```text
docs/validation/step_c_height_recovery_plan.md
```

The plan must be task-based, gated, and diagnostic-first.
