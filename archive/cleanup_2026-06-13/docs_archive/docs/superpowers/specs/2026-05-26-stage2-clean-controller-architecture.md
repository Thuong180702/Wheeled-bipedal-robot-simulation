# Balance-Core Controller Architecture Specification

Date: 2026-05-26
Status: specification only
Scope: controller architecture restructuring and consolidation for the wheeled-biped standing-balance stack

This document defines a maintainable, physics-consistent controller architecture for the wheeled-biped robot project. It is not an implementation plan, gain-tuning recipe, or short-term patch proposal.

The architecture is named **balance-core**. It may also be described as the **standing-balance core** when discussing the control objective. It must use functional, production-style controller names rather than debug or experiment-stage names.

## 1. Problem statement

The current controller stack has accumulated experimental torque paths, wrappers, ablation flags, and overlapping controllers. This makes the robot hard to reason about because multiple modules can command the same joints for different physical objectives in the same control tick.

The desired architecture is a clean standing-balance controller that follows principles used in real humanoid and biped robots, not short-term simulation patches. It must:

- separate posture/shape support from balance regulation;
- give balance higher priority than posture;
- hold posture and height softly instead of rigidly locking the robot;
- use wheels as the primary sagittal forward/backward balance actuator;
- allow forward/backward wheel motion and body motion during recovery;
- use hip-roll/lateral actuation as the primary lateral balance actuator;
- keep hip-pitch and knee torques focused on compliant shape, height, and support;
- keep the center of mass and capture point in a recoverable region without preventing temporary motion needed for balance;
- return slowly toward the initial position after balance is recovered through an outer-loop position regulation concept;
- supervise contact state and avoid fake contact forces;
- support future whole-body control, contact recovery, residual RL, locomotion, robustness testing, and real-robot deployment;
- prevent hidden torque ownership conflicts.

The architecture must prioritize physical correctness, maintainability, and extensibility over passing a short simulation rollout.

## 2. Current architecture problems

The repository currently contains several valid controller ideas, but they are composed through a script-level stack with many flags and implicit interactions.

Observed controller families include:

- `IntegratedWBC`: centroidal wrench computation, force distribution, and contact-Jacobian torque mapping.
- `CentroidalWrenchComputer`: desired force/moment generation from centroidal state.
- `SimpleForceDistributor`: wrench-to-contact-force and hip-roll mapping, including contact-aware behavior.
- `ContactJacobian`: MuJoCo Jacobian mapping from wheel contact forces to joint torques.
- `StaticPostureHoldingController`: current file name for the concept that should become `ShapePostureController`.
- `StaticFeedforwardController`: current file name for the concept that should become `SupportFeedforwardController`.
- `Stage2BRollDirectController`: current temporary experimental file name for the concept that should become `LateralRollBalanceController`.
- `Stage2BSagittalWheelController`: current temporary experimental file name for the concept that should become `SagittalWheelBalanceController`.
- `Stage2CSagittalStateFeedbackController`: current temporary experimental file name for `ExperimentalSagittalStateFeedbackController`.
- `Stage2DSagittalLQRController`: current temporary experimental file name for `ExperimentalSagittalLQRController`.
- `MomentumCoordinator`, `PostureRegularizer`, `LegPositionController`: earlier hierarchical support layers.
- legacy hip-roll centering and secondary wheel balance helpers in `simulate_hierarchical_controller.py`.
- `StaticBalanceController` wrapper for correction-only WBC bias cancellation.

The main problems are:

1. **Joint ownership is not globally enforced.** Several controllers can produce nonzero torque for hip-roll, wheels, hip-pitch, and knees in the same control tick.
2. **Debug and stage-based names leak into architecture.** Names tied to experiment chronology obscure functional responsibility.
3. **Script flags define architecture.** The effective controller is assembled in `scripts/simulate_hierarchical_controller.py`, not through a single explicit functional controller mode.
4. **WBC and direct controllers can overlap.** WBC can command hip-roll and wheel joints while direct roll or sagittal wheel controllers also command the same groups unless masking is carefully maintained.
5. **Posture controllers can be too rigid.** Strong joint-position controllers can fight the balance controller by suppressing the body and wheel motion needed for recovery.
6. **Legacy paths remain active candidates.** Hip-roll centering, secondary wheel balance, posture regularization, leg position control, and ablation modes can re-enter experiments without a clear ownership contract.
7. **Contact semantics are mixed.** Some code correctly avoids fake force on non-contact wheels, while other recovery concepts still risk implying forces through unavailable contacts.
8. **Telemetry names mix frames and semantics.** Generic Euler names and robot-frame control variables can be confused, especially for `pitch_x` and `roll_y`.
9. **Torque-source accounting is scattered.** There is no single authoritative torque-stack report that proves each joint was commanded by exactly one owner group.

## 3. Physical principles and design constraints

Balance-core must preserve these principles.

### 3.1 Task hierarchy

1. **Balance has higher priority than posture.** The controller must stabilize the floating base and contact state before enforcing nominal joint shape.
2. **Posture is compliant, not rigid.** Posture control should restore a useful internal configuration without preventing the body, wheels, and joints from moving during recovery.
3. **Support joints preserve shape and height.** Hip-pitch and knee control should support height and leg geometry, but must allow small body motion and posture deviation when balance requires it.
4. **Posture must not fight wheel balance.** If the robot is falling forward or backward, wheel torque and body motion are allowed to recover balance; posture control must not rigidly oppose that recovery.

### 3.2 Floating-base dynamics

The robot is an underactuated floating-base wheeled biped. Balance-core must consider:

- base inertia and angular rates;
- joint and wheel inertia;
- wheel velocity and wheel acceleration;
- actuator torque limits;
- actuator torque-rate limits;
- CoM and capture-point motion;
- contact transitions and contact-force validity;
- the fact that not every desired base wrench is feasible under the current contact state.

A controller that only locks joint angles is not a valid standing-balance architecture for this system.

### 3.3 Sagittal balance

1. **Wheels are the primary sagittal actuator.** Wheel torques regulate forward/backward balance, `pitch_x`, `pitch_rate_x`, sagittal capture-point error, CoM velocity, and wheel velocity damping.
2. **Recovery may require motion.** The robot may move forward or backward to regain balance. Zero position error is not more important than avoiding a fall.
3. **Outer position regulation is slow.** After balance is recovered, a slow outer-loop position controller may bias the sagittal wheel balance reference to return toward the initial position. This outer loop must not overpower the inner pitch/wheel balance loop.

### 3.4 Lateral balance

1. **Hip-roll/lateral actuation controls `roll_y`.** Hip-roll torques regulate lateral tilt and lateral recovery.
2. **Roll control is separate from sagittal wheel balance.** Lateral roll control must not be hidden inside wheel balance, and sagittal wheel balance must not secretly command hip-roll.
3. **Lateral balance owns hip-roll.** Only the lateral-roll balance owner may command dynamic hip-roll balance torque in balance-core.

### 3.5 Posture compliance

1. **The robot holds posture and height softly.** It should not rigidly lock hip-pitch, knee, or yaw targets.
2. **Falling states may require posture deviation.** If the robot is falling forward, it may lean or adjust posture slightly while wheel torque recovers balance.
3. **Posture support is subordinate to balance.** Hip-pitch and knee should preserve shape, height, and support, not act as competing sagittal balance controllers.

### 3.6 Contact and recovery

1. **No fake contact forces.** A non-contact wheel cannot receive a ground reaction force.
2. **Contact recovery is state-machine behavior.** Re-contact recovery must estimate state and command feasible joint/wheel motion; it must not inject force through a missing contact.
3. **Contact supervision is mandatory from the beginning.** Full recovery behavior may be deferred, but balance-core must expose a contact supervisor interface from the start.

### 3.7 Naming and ownership

1. **Orientation naming is robot-frame explicit.** `pitch_x` means sagittal forward/backward tilt. `roll_y` means lateral tilt. World Euler labels must not be used as controller variables unless clearly marked diagnostic-only.
2. **Every torque source has one owner.** The architecture must make it impossible for multiple active controllers to secretly command the same joint group.
3. **Functional names replace experiment chronology.** Controller names should describe responsibility, not the historical experiment that produced them.

## 4. Joint ownership table

Balance-core uses fixed ownership by joint group. Each active torque source must declare its owned joints and must produce zero torque outside those joints.

| Index | Joint | Balance-core owner | Allowed control purpose | Disallowed in balance-core |
|---:|---|---|---|---|
| 0 | `l_hip_roll` | `tau_lateral_roll_balance` | lateral/roll balance; limited posture trim only if explicitly merged into the lateral owner | legacy hip-roll centering; WBC hip-roll torque unless an explicit WBC mode is selected |
| 1 | `l_hip_yaw` | `tau_shape_posture` | yaw/posture hold, symmetry, weak damping | balance torque, WBC correction by default |
| 2 | `l_hip_pitch` | `tau_shape_posture` + `tau_support_feedforward` | leg shape, height support, compliant posture | sagittal balance torque that competes with wheels |
| 3 | `l_knee` | `tau_shape_posture` + `tau_support_feedforward` | leg shape, height support, support feedforward | balance torque that locks pitch or CoM |
| 4 | `l_wheel` | `tau_sagittal_wheel_balance` | sagittal balance, wheel damping, slow outer position recovery | posture hold, WBC wheel torque by default, secondary wheel balance |
| 5 | `r_hip_roll` | `tau_lateral_roll_balance` | lateral/roll balance; limited posture trim only if explicitly merged into the lateral owner | legacy hip-roll centering; WBC hip-roll torque unless an explicit WBC mode is selected |
| 6 | `r_hip_yaw` | `tau_shape_posture` | yaw/posture hold, symmetry, weak damping | balance torque, WBC correction by default |
| 7 | `r_hip_pitch` | `tau_shape_posture` + `tau_support_feedforward` | leg shape, height support, compliant posture | sagittal balance torque that competes with wheels |
| 8 | `r_knee` | `tau_shape_posture` + `tau_support_feedforward` | leg shape, height support, support feedforward | balance torque that locks pitch or CoM |
| 9 | `r_wheel` | `tau_sagittal_wheel_balance` | sagittal balance, wheel damping, slow outer position recovery | posture hold, WBC wheel torque by default, secondary wheel balance |

Joint groups:

```text
HIP_ROLL = [0, 5]
HIP_YAW = [1, 6]
HIP_PITCH = [2, 7]
KNEE = [3, 8]
SUPPORT_SHAPE = [1, 2, 3, 6, 7, 8]
SUPPORT_FEEDFORWARD = [2, 3, 7, 8]
WHEELS = [4, 9]
```

## 5. Proposed balance-core controller architecture

### Architecture alternatives considered

**Option A — WBC-first cleanup.** Make `IntegratedWBC` the primary controller and remove most direct controllers. This is attractive long-term, but it is not the right default for balance-core because the current WBC path still mixes contact-force reasoning, correction-only behavior, force distribution, and joint-torque mapping. It is too large to use as the baseline while the project is trying to untangle ownership.

**Option B — Functional direct torque stack. Recommended.** Use a small number of physically assigned torque sources with explicit joint ownership:

```text
tau_total_raw =
    tau_shape_posture
  + tau_support_feedforward
  + tau_sagittal_wheel_balance
  + tau_lateral_roll_balance
```

This preserves the strongest validated ideas while making controller conflicts impossible by construction. It also provides a stable interface for later WBC, contact recovery, RL, locomotion, and real-robot work.

**Option C — Keep current flag-driven stack and add stronger masks.** This is the smallest code movement, but it preserves the underlying architectural problem: the script remains the controller definition, and new flags can reintroduce hidden conflicts.

Balance-core should use Option B.

### Balance-core block diagram

```text
Centroidal/contact state estimator
    ├─ robot-frame orientation: pitch_x, roll_y, yaw_z
    ├─ robot-frame rates: pitch_rate_x, roll_rate_y, yaw_rate_z
    ├─ CoM/capture point: com_y, com_vy, cp_y, cp_error_y
    ├─ wheel state: wheel_vel_left, wheel_vel_right, wheel_vel_mean
    └─ contact state: left_contact, right_contact, normal forces, force validity

Contact supervisor
    ├─ contact mode: double_contact, left_only, right_only, no_contact
    ├─ transition timing and force validity
    └─ future re-contact recovery state hook

Shape/posture reference provider
    └─ desired soft leg/yaw posture and height-compatible support geometry

Controllers
    ├─ ShapePostureController
    │     owns [1,2,3,6,7,8]
    │     outputs tau_shape_posture
    ├─ SupportFeedforwardController
    │     owns [2,3,7,8]
    │     outputs tau_support_feedforward
    ├─ SagittalWheelBalanceController
    │     owns [4,9]
    │     outputs tau_sagittal_wheel_balance
    └─ LateralRollBalanceController
          owns [0,5]
          outputs tau_lateral_roll_balance

Torque ownership validator
    ├─ rejects nonzero torque outside declared ownership
    ├─ rejects two active dynamic balance owners for the same joint group
    └─ logs per-joint owner/source accounting

Safety layer
    ├─ actuator torque clamp
    ├─ torque-rate clamp
    ├─ wheel acceleration/velocity sanity telemetry
    └─ contact-state gating

Actuator command
    └─ tau_final
```

## 6. Role of each existing controller

| Existing controller/module | Balance-core role | Rationale |
|---|---|---|
| `CentroidalStateEstimator` | Core state provider | Provides CoM, capture point, contact state, and robot-frame orientation fields needed by balance-core. |
| `orientation_utils` | Core naming/transform utility | Must remain the canonical source for robot-frame `pitch_x`, `roll_y`, `yaw_z` semantics. |
| `StaticPostureHoldingController` | Temporary existing implementation of the `ShapePostureController` concept | Useful as a starting point for `tau_shape_posture`, but balance-core must narrow and soften the concept. It should own yaw, hip-pitch, and knee support; hip-roll posture trim must be off or explicitly merged into lateral balance. |
| `StaticFeedforwardController` | Temporary existing implementation of the `SupportFeedforwardController` concept | Useful for `tau_support_feedforward`, limited to hip-pitch/knee support joints. It must not become a hidden balance controller. |
| `Stage2BRollDirectController` | Temporary experimental module for the `LateralRollBalanceController` concept | Its direct hip-roll ownership is physically clear. It should be renamed or merged into a functional lateral-roll balance component before becoming core architecture. |
| `Stage2BSagittalWheelController` | Temporary experimental module for the `SagittalWheelBalanceController` concept | Simple wheel balance is useful for sign tests and comparisons, but balance-core should use one selected functional sagittal wheel balance owner, not multiple wheel controllers. |
| `Stage2CSagittalStateFeedbackController` | Temporary experimental module; future name `ExperimentalSagittalStateFeedbackController` | It includes wheel velocity damping and may inform the production sagittal wheel balance controller, but the current stage-based file should remain experimental until renamed or merged. |
| `Stage2DSagittalLQRController` | Temporary experimental module; future name `ExperimentalSagittalLQRController` | LQR is conceptually appropriate for diagnostics or comparison, but it depends on identified dynamics and offline model assumptions. It should not be silently active as another layer. |
| `IntegratedWBC` | Optional advanced layer, off by default in balance-core | Must not be deleted. It may be reintroduced later through diagnostic-only, wheel-only, roll-only, and then full correction modes. When reintroduced, it must respect torque ownership. |
| `CentroidalWrenchComputer` | WBC infrastructure and diagnostics | Useful for future WBC, not part of the default balance-core torque stack. |
| `SimpleForceDistributor` | WBC infrastructure and diagnostics | Useful contact-aware logic, especially no fake non-contact force. Not part of the default balance-core torque stack except as future WBC infrastructure. |
| `ContactJacobian` | WBC infrastructure and diagnostics | Correct tool for force-to-torque mapping, but balance-core default is direct torque ownership by joint group. |
| `StaticBalanceController` | Diagnostic WBC wrapper only | Useful for correction-only WBC experiments, not balance-core default. |
| `MomentumCoordinator` | Deprecated from balance-core | It overlaps with balance ownership and can become another hidden torque source. Its ideas may inform future WBC/RL but should not command torque in balance-core. |
| `PostureRegularizer` | Deprecated from balance-core | It overlaps with shape posture and can fight balance if independently active. |
| `LegPositionController` | Deprecated from balance-core | Strong leg position control can mask dynamic balance behavior. Balance-core posture support must be compliant and explicitly owned. |
| legacy hip-roll centering | Deprecated from balance-core | It duplicates lateral roll ownership. |
| legacy secondary wheel balance | Deprecated from balance-core | It duplicates sagittal wheel ownership. |

## 7. What to keep, deprecate, merge, or defer

### Keep as core

- Centroidal/contact state estimation.
- Robot-frame orientation utilities.
- Contact supervisor interface.
- `ShapePostureController` concept owning `[1,2,3,6,7,8]`.
- `SupportFeedforwardController` concept owning `[2,3,7,8]`.
- `SagittalWheelBalanceController` concept owning `[4,9]`.
- `LateralRollBalanceController` concept owning `[0,5]`.
- A single torque ownership validator and torque-stack telemetry contract.
- Torque clipping and torque-rate limiting.

### Rename or merge concepts into functional components

- `StaticPostureHoldingController` concept should become `ShapePostureController`, not a rigid full-leg lock.
- `StaticFeedforwardController` concept should become `SupportFeedforwardController`, constrained to support joints.
- `Stage2BRollDirectController` concept should become `LateralRollBalanceController`.
- `Stage2BSagittalWheelController` concept should become `SagittalWheelBalanceController` if selected as the production sagittal owner.
- `Stage2CSagittalStateFeedbackController` should become `ExperimentalSagittalStateFeedbackController` or be merged into `SagittalWheelBalanceController` after validation.
- `Stage2DSagittalLQRController` should become `ExperimentalSagittalLQRController` or remain diagnostic until model assumptions are validated.

### Keep as diagnostics or explicit experiments

- `IntegratedWBC`, `CentroidalWrenchComputer`, `SimpleForceDistributor`, `ContactJacobian`.
- Current stage-named sagittal/roll files only as temporary experimental modules until renamed or merged.
- `StaticBalanceController` for correction-only WBC experiments.
- WBC force-distribution and contact-Jacobian diagnostics.

### Deprecate from balance-core

- legacy hip-roll centering;
- legacy secondary wheel balance;
- `PostureRegularizer` as an independent torque source;
- `LegPositionController` as a strong posture-locking source;
- `MomentumCoordinator` as an active torque source;
- experiment-stage ablation stacks as the definition of the clean controller;
- simultaneous experimental sagittal wheel torque sources.

### Defer

- WBC as primary control;
- full re-contact recovery behavior;
- disturbance/push robustness tuning;
- residual RL integration;
- locomotion and terrain behaviors;
- real-robot deployment;
- aggressive gain tuning;
- adding another controller stage.

## 8. Balance-core controller mode requirements

A single explicit mode should define the clean architecture:

```text
--controller-mode balance-core
```

The alias `--controller-mode standing-balance` may be supported only if it maps to the same functional architecture. The canonical name is `balance-core`.

This mode must be the only way to activate the clean torque stack. It must not be assembled indirectly through a collection of legacy flags.

### balance-core includes

- centroidal/contact state estimation;
- contact supervisor interface;
- robot-frame orientation naming using `pitch_x`, `roll_y`, `yaw_z`;
- compliant shape posture on `[1,2,3,6,7,8]`;
- bounded support feedforward on `[2,3,7,8]`;
- exactly one sagittal wheel balance source on `[4,9]`;
- exactly one lateral roll balance source on `[0,5]`;
- slow outer-loop sagittal position regulation as a bias into the wheel balance reference, not as a second wheel torque source;
- torque ownership validation;
- torque limits and torque-rate limits;
- standardized telemetry.

### balance-core excludes

- WBC torque by default;
- `MomentumCoordinator` torque;
- `PostureRegularizer` torque;
- `LegPositionController` torque;
- legacy hip-roll centering;
- legacy secondary wheel balance;
- experiment-stage ablation mode composition;
- simultaneous experimental sagittal wheel torque sources;
- any torque source that writes nonzero torque outside its declared joint ownership;
- force injection on non-contact wheels;
- patches whose only purpose is to pass a short simulation rollout.

### Explicit compatibility rule

Existing flags may remain for legacy experiments, but `--controller-mode balance-core` must either ignore or reject incompatible flags. Incompatible flags should fail fast with a clear error instead of silently changing the torque stack.

## 9. Torque stack requirements

The clean torque stack is:

```text
tau_total_raw =
    tau_shape_posture
  + tau_support_feedforward
  + tau_sagittal_wheel_balance
  + tau_lateral_roll_balance
```

Then:

```text
tau_total_clipped = actuator_limit_clip(tau_total_raw)
tau_final = torque_rate_limit(tau_total_clipped, tau_prev)
```

### Source requirements

#### `tau_shape_posture`

- Produced by the `ShapePostureController` concept.
- Owns `[1,2,3,6,7,8]`.
- Holds yaw, hip-pitch, and knee shape softly.
- Does not command wheels.
- Does not command hip-roll unless the architecture explicitly assigns low-authority hip-roll posture trim to the lateral-roll owner.
- Uses compliant gains and damping so body/wheel balance can move.
- Must not rigidly lock the robot when `pitch_x`, `roll_y`, or contact state indicates recovery motion is needed.
- Supports height-compatible posture references later, but this spec does not tune those references.

#### `tau_support_feedforward`

- Produced by the `SupportFeedforwardController` concept.
- Owns `[2,3,7,8]`.
- Provides bounded baseline support against gravity/static load.
- Must be explicit feedforward, not hidden inside WBC or posture logic.
- Must be logged separately from posture feedback.
- Must not command wheels or hip-roll.

#### `tau_sagittal_wheel_balance`

- Produced by the `SagittalWheelBalanceController` concept.
- Owns `[4,9]` only.
- Regulates `pitch_x`, `pitch_rate_x`, capture-point/sagittal CoM state, and wheel velocity damping.
- Allows the robot to move forward/backward to recover balance.
- May include a slow outer-loop position term, but only as a reference bias or low-bandwidth term inside the same sagittal owner.
- Must not directly command hip-pitch or knee to correct pitch.
- Must support both wheels receiving the same sagittal torque for common-mode balance unless differential wheel yaw control is explicitly added later.
- Must expose saturation, wheel damping, wheel velocity, and wheel acceleration telemetry.

#### `tau_lateral_roll_balance`

- Produced by the `LateralRollBalanceController` concept.
- Owns `[0,5]` only.
- Regulates `roll_y` and `roll_rate_y`.
- May use lateral CoM/capture-point state if available, but must remain the sole hip-roll dynamic balance owner.
- Must not command wheels, hip-pitch, or knees.
- Must expose moment command and left/right hip-roll torque telemetry.

### Ownership validator

Before summing, each torque source must pass an ownership check:

```text
source.output[joint not in source.owned_joints] == 0 within tolerance
```

The controller composition must also assert:

```text
no two active sources claim the same exclusive dynamic balance ownership
```

Support feedforward and shape posture may both own hip-pitch/knee because they are explicitly paired as support terms, but they must be logged as separate support components and may not contain balance terms.

## 10. Telemetry and naming requirements

Telemetry must make the clean architecture auditable from one rollout log.

### Required control-state names

- `pitch_x_rad`: robot-frame sagittal forward/backward tilt.
- `roll_y_rad`: robot-frame lateral tilt.
- `yaw_z_rad`: robot-frame yaw.
- `pitch_rate_x_rad_s`: robot-frame sagittal angular rate.
- `roll_rate_y_rad_s`: robot-frame lateral angular rate.
- `yaw_rate_z_rad_s`: robot-frame yaw rate.
- `com_x_m`, `com_y_m`, `com_z_m`.
- `com_vx_m_s`, `com_vy_m_s`, `com_vz_m_s`.
- `cp_x_m`, `cp_y_m`.
- `cp_error_y_m` for sagittal capture-point error.
- `wheel_vel_left_rad_s`, `wheel_vel_right_rad_s`, `wheel_vel_mean_rad_s`.
- `wheel_acc_left_rad_s2`, `wheel_acc_right_rad_s2`, `wheel_acc_mean_rad_s2`.
- `left_wheel_contact`, `right_wheel_contact`.
- `left_normal_force_n`, `right_normal_force_n` when valid.
- `contact_force_valid`.
- `contact_supervisor_state`.

World-frame Euler terms may be logged only with explicit names such as `world_euler_roll_rad`; they must not be used interchangeably with controller-frame variables.

### Required torque-source telemetry

- `tau_shape_posture_per_joint`.
- `tau_support_feedforward_per_joint`.
- `tau_sagittal_wheel_balance_per_joint`.
- `tau_lateral_roll_balance_per_joint`.
- `tau_total_raw_per_joint`.
- `tau_total_clipped_per_joint`.
- `tau_final_per_joint`.
- `tau_rate_limited_per_joint` or equivalent rate-limit delta.
- `torque_saturation_mask_per_joint`.
- `torque_rate_saturation_mask_per_joint`.
- `active_torque_owner_per_joint`.
- `ownership_violation_count`.

### Required controller diagnostics

- sagittal terms: pitch, pitch-rate, capture-point, CoM velocity, wheel damping, wheel acceleration, outer-position contribution;
- lateral terms: roll, roll-rate, lateral capture/CoM contribution if used;
- posture terms: joint position error and joint velocity contribution by support group;
- feedforward terms: selected joint group, applied scale/source, per-joint value;
- contact supervisor state and contact transition events.

## 11. Contact supervisor and re-contact recovery requirements

Balance-core must include a contact supervisor interface even if full re-contact recovery is deferred.

### Contact supervisor responsibilities

- Determine contact state: `double_contact`, `left_only`, `right_only`, `flight_or_no_contact`.
- Report contact force validity separately from contact geometry presence.
- Gate any force-based controller path so non-contact wheels receive no fake ground reaction force.
- Inform sagittal and lateral controllers when contact is degraded.
- Log contact transitions and duration in each contact state.
- Provide a future hook for re-contact recovery state transitions.
- Provide enough state for future recovery to use IMU, joint positions, wheel/foot kinematics, and contact state.

### balance-core behavior by contact state

- **Double contact:** normal clean torque stack active.
- **Single contact:** no fake force on the non-contact wheel; direct joint torque components may still exist only if physically meaningful for airborne motion, but force-distribution paths must be disabled or contact-aware.
- **No contact:** balance torque must not pretend to create ground reaction. The controller may command bounded posture preparation and wheel spin damping, but re-contact recovery is deferred.

### Future re-contact recovery requirements

Re-contact recovery must be a state machine, not force injection. Future recovery must use IMU state, joint positions, wheel/foot kinematics, and contact state.

If one wheel or foot loses contact, the controller must estimate where the lost-contact limb/wheel is relative to the body and ground, then command feasible posture and wheel motion to regain contact. It must not assign a ground reaction force to the missing contact.

Future states should include at least:

```text
normal_balance
contact_degraded
flight_or_lost_contact
recontact_prepare
recontact_impact_absorb
return_to_balance
```

Each state must define allowed torque owners, torque limits, sensor inputs, state estimates, and transition conditions.

## 12. WBC role and reintroduction path

`IntegratedWBC` must not be deleted. WBC is an advanced optional layer, off by default in balance-core.

WBC may be reintroduced later in explicit modes only, in this order:

1. **Diagnostic-only mode:** compute desired wrench, contact forces, and mapped torques for logging only; do not apply WBC torque.
2. **Wheel-only correction mode:** allow WBC-derived corrections only on wheel joints if they respect sagittal ownership and do not conflict with `SagittalWheelBalanceController`.
3. **Roll-only correction mode:** allow WBC-derived corrections only on hip-roll joints if they respect lateral ownership and do not conflict with `LateralRollBalanceController`.
4. **Full correction mode:** allow broader WBC correction only after ownership, contact feasibility, support semantics, and safety behavior are validated.

Any WBC reintroduction must:

- declare torque ownership;
- produce zero torque outside allowed joints for the selected WBC mode;
- obey contact supervisor constraints;
- avoid fake non-contact forces;
- log WBC diagnostic torque separately from applied torque;
- fail ownership validation if it conflicts with an active balance-core torque source.

## 13. Safety, limits, and dynamic consistency requirements

Balance-core must include these safety and dynamic consistency constraints:

1. **Actuator torque limits.** Final torque must respect MuJoCo actuator control ranges or a stricter configured limit.
2. **Torque-rate limits.** The final applied torque must pass through a rate limiter unless explicitly disabled for diagnostics.
3. **Per-joint saturation telemetry.** Saturation must be logged by joint and source.
4. **Wheel velocity and acceleration telemetry.** The sagittal controller must report wheel velocity damping and excessive wheel-speed conditions.
5. **Floating-base dynamics awareness.** Controller terms must account for base angular rates, CoM/capture-point motion, wheel velocity, contact transitions, and actuator limits.
6. **No hidden inverse dynamics term.** If inverse dynamics is used, it must be a named torque source with ownership and telemetry; it is not included in the default balance-core stack.
7. **No blind body-weight `J^T f` baseline.** Support load should be handled through explicit support feedforward or future validated WBC, not an unverified static contact-force mapping.
8. **Contact validity gating.** Contact force measurements at invalid times, startup transients, or no-contact states must be marked invalid and not treated as real support.
9. **Inner-loop priority.** Outer position regulation must be low bandwidth and bounded so it cannot overpower pitch stabilization.
10. **No silent mode mixing.** Incompatible flags and torque sources must fail fast in balance-core.
11. **Deterministic composition.** Given the same state and controller references, the torque stack must produce the same source decomposition and final torque.

## 14. Required tests

The following tests are required to prevent future controller conflicts. They are requirements, not an execution plan.

### Ownership and composition tests

- Each clean torque source outputs shape `(10,)`.
- `tau_shape_posture` is zero outside `[1,2,3,6,7,8]`.
- `tau_support_feedforward` is zero outside `[2,3,7,8]`.
- `tau_sagittal_wheel_balance` is zero outside `[4,9]`.
- `tau_lateral_roll_balance` is zero outside `[0,5]`.
- The clean torque stack equals the sum of the four named sources before clipping.
- Ownership validator rejects a source that commands an unowned joint.
- Ownership validator rejects simultaneous active sagittal wheel owners.
- Ownership validator rejects simultaneous active lateral hip-roll owners.
- Support feedforward and shape posture may coexist on support joints only because they are declared compatible support terms.

### Mode isolation tests

- `--controller-mode balance-core` disables WBC torque by default.
- `--controller-mode balance-core` disables `MomentumCoordinator` torque.
- `--controller-mode balance-core` disables `PostureRegularizer` torque.
- `--controller-mode balance-core` disables `LegPositionController` torque.
- `--controller-mode balance-core` disables legacy hip-roll centering.
- `--controller-mode balance-core` disables legacy secondary wheel balance.
- Incompatible legacy flags fail fast or are explicitly ignored with a logged warning.

### Physical sign and semantics tests

- Positive `pitch_x` produces wheel torque in the documented corrective direction.
- Positive wheel velocity damping produces opposing wheel torque.
- Positive `roll_y` produces the documented restoring hip-roll torque pair.
- Hip-pitch/knee support terms do not change when only sagittal pitch error changes, except through explicitly defined posture reference updates.
- Wheels are never posture-controlled.
- Hip-pitch/knee are never the primary sagittal balance actuators.
- Posture compliance tests verify posture torque remains subordinate to balance demands.

### Contact tests

- Double-contact state permits normal stack operation.
- Single-contact state never assigns fake contact force to the non-contact wheel.
- No-contact state disables force-based ground-reaction assumptions.
- Contact transitions are logged.
- Contact force validity is false during invalid startup/measurement windows.
- Re-contact recovery hooks expose IMU, joint position, wheel/foot kinematics, and contact-state inputs for future state-machine recovery.

### Telemetry tests

- Required `pitch_x`/`roll_y`/`yaw_z` fields are logged with robot-frame names.
- Required torque-source vectors are logged every control tick.
- `active_torque_owner_per_joint` is present and matches ownership declarations.
- Saturation masks are present for torque and torque-rate limiting.
- Contact supervisor state is logged.
- Wheel velocity and acceleration are logged.
- No generic Euler field is used as a clean controller input without robot-frame naming.

### Safety tests

- Final torque respects actuator limits.
- Torque-rate limiter bounds per-step torque changes.
- Saturation telemetry is correct when raw torque exceeds limits.
- The clean stack produces finite torques for nominal standing states.
- The clean stack produces finite, bounded torques for degraded contact states.
- Wheel velocity remains bounded under nominal balance-core operation.
- Knee torques and knee states remain bounded under nominal balance-core operation.

## 15. Acceptance criteria

Balance-core is complete when all of the following are true:

1. A single explicit `--controller-mode balance-core` mode exists and defines the full clean torque stack.
2. The active clean torque stack is exactly:

   ```text
   tau_shape_posture
 + tau_support_feedforward
 + tau_sagittal_wheel_balance
 + tau_lateral_roll_balance
   ```

3. No hidden legacy torque source contributes to `tau_final`.
4. WBC is off by default in balance-core.
5. Posture control is compliant, not rigid.
6. Wheel torque controls sagittal balance.
7. Hip-roll/lateral actuation controls `roll_y`.
8. `pitch_x` and `roll_y` remain bounded in nominal standing-balance scenarios.
9. Knees remain stable and do not collapse or rigidly lock the floating base.
10. Wheel velocity remains bounded.
11. Legacy hip-roll centering and secondary wheel balance are off in balance-core.
12. `MomentumCoordinator`, `PostureRegularizer`, and `LegPositionController` do not command torque in balance-core.
13. Exactly one sagittal wheel balance owner commands `[4,9]`.
14. Exactly one lateral roll balance owner commands `[0,5]`.
15. Shape posture and support feedforward are the only clean sources commanding `[1,2,3,6,7,8]`.
16. All clean torque sources declare ownership and pass ownership validation.
17. Incompatible flags cannot silently modify the clean torque stack.
18. Telemetry logs robot-frame orientation names and per-source torque vectors.
19. Contact state is supervised and logged.
20. Non-contact wheels never receive fake ground reaction force.
21. Torque limits and torque-rate limits are applied and logged.
22. Required ownership, mode isolation, sign, contact, telemetry, and safety tests pass.
23. The architecture remains extensible for future WBC, contact recovery, residual RL, locomotion, real-robot deployment, and robustness testing.
24. The implementation does not add another quick-patch controller layer.
25. The implementation does not tune gains as a substitute for architectural cleanup.
26. The implementation does not claim full WBC, contact recovery, disturbance robustness, RL readiness, locomotion readiness, or real-robot readiness unless those are separately implemented and evaluated.

## 16. Out-of-scope items for this spec

This specification intentionally excludes:

- implementation steps or task-by-task execution planning;
- gain tuning;
- adding another controller stage;
- reintroducing full WBC as the default controller;
- making short-term patches solely to pass a rollout;
- full contact recovery behavior;
- push-disturbance robustness validation;
- terrain, stair, locomotion, or stand-up recovery control;
- residual RL integration;
- real-robot deployment claims;
- paper-result claims;
- deleting WBC infrastructure.

WBC remains important future infrastructure. Its correct role is optional and advanced in balance-core, and it should be reintroduced only after direct torque ownership, telemetry, contact semantics, and safety behavior are unambiguous.
