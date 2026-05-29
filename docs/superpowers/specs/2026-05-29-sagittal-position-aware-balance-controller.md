# Sagittal Position-Aware Balance Controller

## Summary

This specification defines the architecture for replacing failed add-on position-containment experiments with a proper sagittal balance controller that regulates sagittal position as part of the balance state itself. The design preserves the validated balance-core stack structure, keeps WBC off, preserves torque ownership rules, and explicitly avoids direct torque bias patches, capture-point bias patches, and phase-aware add-on shaping.

## Problem Statement

The current clean balance-core stack is:

- `tau_shape_posture`
- `tau_support_feedforward`
- `tau_sagittal_wheel_balance`
- `tau_lateral_roll_balance`

This stack can keep the robot upright for long durations, but it does not keep the robot near its standing location. The robot balances by rolling away along the sagittal axis.

### Root Cause

The current sagittal controller stabilizes pitch and sagittal balance indicators well enough to prevent falling, but it does not include sagittal position regulation in the state it is controlling. As a result, wheel motion that preserves upright balance is allowed to accumulate into large displacement.

The failed E0 approaches all added position regulation outside the core sagittal balance problem:

- **E0b direct torque containment** added wheel torque from position error.
- **E0c capture-point/reference shaping** added CP bias from position/velocity mismatch.
- **E0d phase-aware reference shaping** added staged braking/return logic around the same external correction idea.

All three failed because they treated position return as a secondary correction layered on top of an inner balance loop that was not position-aware. The inner loop interpreted return-to-position behavior as a disturbance, or the outer loop saturated before meaningful containment occurred.

## Design Goal

Design a sagittal balance controller that regulates uprightness, sagittal velocity, wheel velocity, and return-to-reference behavior as one coupled balance problem.

The target outcome is not hard position locking. The target outcome is smooth bounded drift with balance priority preserved.

## Non-Goals

This specification explicitly excludes:

- WBC reintroduction
- fake contact forces
- hidden torque paths
- push robustness expansion
- dynamic height transitions
- height recovery
- locomotion
- terrain or stair handling
- RL methods
- real robot deployment

## Coordinate and Frame Contract

### Axis Convention

The controller shall use the project’s established coordinate convention:

- `X`: lateral
- `Y`: sagittal
- `Z`: vertical

### Position Reference Frame

Sagittal displacement shall be measured in an **initial-heading frame**, not raw world-frame Y alone.

Rationale:

- Pure world-frame Y is acceptable only if yaw remains negligible.
- Over long standing runs, yaw can drift enough that world-frame Y no longer represents the robot’s perceived forward/backward displacement.
- The controller should regulate motion along the robot’s initial sagittal heading, not a globally fixed axis that may become misaligned.

### Frame Definition

At equilibrium capture, record:

- initial CoM position in world frame
- initial yaw heading
- initial sagittal unit vector projected onto the world XY plane

Sagittal displacement shall be computed as:

- current planar displacement relative to equilibrium origin
- projected onto the initial sagittal unit vector

Sagittal velocity shall be computed as either:

1. CoM planar velocity projected onto the same initial sagittal unit vector, or
2. finite-difference sagittal displacement over control time

Preferred runtime source:

- Use projected CoM velocity from the stable estimator when available.
- Cross-check with finite-difference displacement during validation.

## Controlled State

The minimum regulated state is:

```text
x = [
    sagittal_position_error,
    sagittal_velocity,
    pitch_x,
    pitch_rate_x,
    wheel_velocity_mean,
]
```

Where:

- `sagittal_position_error`: signed displacement from equilibrium along initial-heading sagittal axis
- `sagittal_velocity`: signed CoM velocity along the same axis
- `pitch_x`: robot-frame sagittal tilt
- `pitch_rate_x`: sagittal angular velocity
- `wheel_velocity_mean`: mean of left/right wheel velocities

### Optional State Extensions

These may be evaluated during system identification, but are optional and must be justified by measurable improvement:

- `capture_point_y`
- `com_y`
- `com_vy`
- `wheel_position_mean`
- explicit initial-heading-frame sagittal displacement state if separate from projected CoM state representation

The first production design should stay with the minimum 5-state controller unless identification quality clearly requires augmentation.

## Control Output Contract

The controller output is:

- a sagittal wheel torque command applied only to the two wheel joints

The controller must never:

- command leg torques directly
- inject hidden torque through WBC
- create new non-wheel ownership paths

The output remains owned by the sagittal wheel balance slot in the balance-core architecture, or by its direct replacement if the component is renamed.

## Candidate Controller Architectures

### Option A — Discrete-Time State-Feedback Regulator from Closed-Loop Identified Dynamics

Model a local linear discrete-time sagittal system around stable standing behavior using closed-loop data, then design a full-state feedback controller over the identified dynamics.

Advantages:

- directly uses measured robot behavior under the validated stack
- naturally includes coupled position/velocity/pitch/wheel effects
- fits the actual simulator/controller stack rather than an idealized pencil model
- easiest to extend across height variants with repeated identification

Risks:

- depends on identification quality
- may produce poor gains if data are insufficiently exciting or badly conditioned

### Option B — LQR-Style Regulator on the Identified Discrete Model

Use the same identified state-space model as Option A, but compute gains with explicit quadratic cost design.

Advantages:

- gives a principled tradeoff between balance recovery, drift suppression, and wheel-speed damping
- easy to tune conceptually through `Q` and `R`
- naturally supports gain scheduling by standing height later if needed

Risks:

- still depends on model quality
- poorly chosen weights can imitate the same external-bias failure mode if they over-prioritize position too aggressively

### Option C — Linearized Wheeled Inverted Pendulum Model with Hand-Aligned Gains

Construct a simplified analytical sagittal model and derive a controller from that model.

Advantages:

- more interpretable physically
- less dependent on identification tooling

Risks:

- likely too idealized for the existing robot and balance-core stack
- mismatch between simplified model and actual closed-loop behavior could waste time

### Recommendation

**Recommended approach: Option B — LQR-style regulator built on a closed-loop identified discrete-time sagittal model.**

Why:

- It integrates position regulation into the same state as pitch and wheel dynamics.
- It avoids the failed outer-loop patch pattern.
- It is the most compatible with the project’s current validation-first workflow.
- It supports later extension to multiple standing heights without changing the control concept.

## Model Identification Strategy

### Core Rule

Do not identify the model from naive open-loop perturbations that make the robot fall.

### Data Source

Use **closed-loop trajectories from the validated balance-core nominal controller** and true standing-height variants that are already known to be stable.

### Identification Dataset Contents

Record time-aligned sequences of:

- sagittal position error in initial-heading frame
- sagittal velocity
- pitch_x
- pitch_rate_x
- wheel_velocity_mean
- wheel torque command from the sagittal controller
- optional left/right wheel velocities separately for diagnostics
- CoM height label / standing-height variant label

### Excitation Strategy

Data should come from safe closed-loop runs with bounded variation, for example:

- nominal standing drift episodes
- small injected reference offsets, if they do not destabilize the stack
- small bounded wheel-command excitation envelopes around stable runs
- height variants already known to be feasible (nominal and ±5 cm)

Do not run open-loop falling experiments for identification.

### Model Form

Preferred model form:

```text
x[k+1] = A x[k] + B u[k]
```

Where:

- `x` is the 5-state sagittal balance vector
- `u` is the scalar sagittal wheel torque command, or an equivalent mean wheel torque command if represented symmetrically

### Model Acceptance

Before controller design, report:

- one-step prediction quality
- multi-step rollout quality over short horizons
- residual structure
- stability/usefulness assessment across nominal and ±5 cm height cases

If the identified model is unusable, stop and revise identification before designing the controller.

## Controller Architecture

### Replacement Strategy

The new controller must **replace or subsume** the current sagittal wheel balance behavior. It must not be added as another external correction path that fights the existing sagittal controller.

### Allowed Integration Shapes

#### Preferred Production End State

Keep:

- `ShapePostureController`
- `SupportFeedforwardController`
- `LateralRollBalanceController`

Replace:

- `SagittalWheelBalanceController`

With:

- `SagittalPositionAwareBalanceController`

#### Safe Validation Path

Before production replacement, introduce:

- `SagittalPositionAwareBalanceController`

as an experimental alternative mode, disabled by default, while the existing sagittal controller remains the default validated baseline.

### Naming Rules

Use professional names only, such as:

- `SagittalPositionAwareBalanceController`
- `SagittalBalanceState`
- `SagittalBalanceReference`
- `SagittalPositionRegulationConfig`
- `PositionAwareBalanceValidator`

Forbidden names include temporary or experiment labels such as:

- `E0Controller`
- `Stage2E`
- `temp_position_fix`
- `position_patch`
- `quick_fix`

## Safety and Constraints

The new design must satisfy all of the following:

- WBC remains off in balance-core mode
- no fake contact force
- no hidden torque path
- ownership violation count remains zero
- only wheel joints receive sagittal controller torque
- wheel torque limits remain enforced
- wheel torque rate limits remain enforced
- balance recovery has priority over position return
- no hard position locking behavior
- no discontinuous phase-switch torque hacks
- position return must be smooth, bounded, and stable

## Validation Protocol

Validation shall proceed progressively.

### Required Sequence

1. nominal 1000 steps
2. nominal 5000 steps
3. nominal 10000 steps if 5000 passes
4. `high_5cm` 500 steps
5. `low_5cm` 500 steps

### Baseline Comparison

The current clean baseline without position containment drifted approximately:

- `35.22 m` over `5000` steps

The new controller must be compared directly against this baseline.

### Drift Objective

Primary objective:

- reduce drift substantially versus 35.22 m baseline

Ideal target:

- max drift `<= 0.30–0.50 m` over 5000 steps

If the ideal target is not achieved, the result must still report the best stable tradeoff reached between:

- reduced drift
- upright stability
- wheel-speed containment

## Required Metrics

Each validation run must report:

- max sagittal drift
- final sagittal drift
- max planar drift
- final planar drift
- pitch range
- roll range
- yaw drift
- CoM Z range
- wheel velocity range
- wheel torque range
- saturation rate
- contact state summary
- ownership violations
- WBC norm
- hidden torque norm

Recommended additions:

- wheel torque RMS
- sagittal position RMS
- sagittal velocity RMS
- percentage of time within target drift band

## Acceptance Criteria

### Minimum Acceptance

The redesign is acceptable only if it:

- substantially outperforms the `35.22 m / 5000 step` baseline
- avoids catastrophic multi-meter runaway worse than baseline
- avoids pitch divergence
- avoids roll divergence
- avoids height collapse
- avoids wheel velocity runaway
- preserves valid contact behavior
- keeps ownership violation count at `0`
- keeps WBC off
- clearly documents whether the old sagittal controller was replaced or run as an experimental alternative

### Preferred Acceptance

Preferred outcome:

- max drift `<= 0.50 m` over 5000 steps
- final drift `<= 0.20 m`
- stable on nominal and both ±5 cm height variants

## Telemetry and Reporting Requirements

The redesign must add telemetry specifically for the integrated sagittal state, not outer-loop patches. At minimum, record:

- sagittal position error
- sagittal velocity
- pitch_x
- pitch_rate_x
- wheel_velocity_mean
- commanded wheel torque
- controller saturation flags
- initial-heading frame reference data

Outputs shall be written under a dedicated future directory:

- `outputs/sagittal_position_aware_balance/`

Expected report artifacts:

- model identification report
- validation summary JSON
- validation summary markdown
- comparison against baseline drift
- comparison against failed E0 approaches as historical context only

## Rollback and Isolation Strategy

Until the new controller is validated, the current clean balance-core stack remains the default baseline. Any experimental position-aware controller must be explicitly selected and disabled by default.

Rollback means:

- switch the sagittal controller selection back to the current validated sagittal wheel controller
- preserve the rest of the four-source stack unchanged
- preserve the existing validation workflow and reports

## Self-Review

This specification has been reviewed against the requested constraints:

- no temporary E0/stage naming in the production design
- no direct torque bias recommendation
- no CP bias patch recommendation
- no WBC reintroduction
- no torque ownership change beyond replacing the sagittal wheel controller role itself
- no implementation instructions beyond architecture/spec level
- no placeholders or TBD sections
