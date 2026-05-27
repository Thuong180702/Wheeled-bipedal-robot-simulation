# Balance-Core Performance Validation and Stabilization Specification

Date: 2026-05-26
Status: specification only
Scope: performance validation and stabilization workflow for the balance-core controller architecture

---

## Section 1: Overview and Validation Workflow

**Goal:** Validate and stabilize the existing balance-core architecture until the robot passes 500-step and 1000-step nominal standing-balance runs, using diagnostic-first analysis rather than blind tuning.

**Current Status:**
- Balance-core architecture is structurally complete
- Four torque sources are active
- Ownership validation passes
- Contact supervision and telemetry are available
- WBC is preserved but off by default
- 50-step architecture validation passes

**Validation Workflow:**

The workflow combines iterative diagnostic cycles with progressive duration scaling.

**Duration ladder:** 100 → 200 → 500 → 1000 steps

**At each duration:**
1. Run validation
2. Capture telemetry
3. Check structural invariants first (Priority 0)
4. If the run fails, stop advancing
5. Classify the failure mode from telemetry
6. Map the failure to the responsible balance-core component
7. Apply one targeted fix inside that component only
8. Re-run the same duration
9. Advance only after the current duration passes

**Constraints:**
- No blind gain tuning
- No new controller stages
- No full WBC reintroduction
- No fake contact force
- No legacy torque source
- No multi-component fix unless telemetry proves coupled failure
- No architecture expansion; only targeted changes inside existing balance-core components are allowed

**Cycle output:**

Each diagnostic cycle must produce a short report including:
- Command used
- Telemetry file
- Survival steps
- Termination reason
- Failure classification
- Responsible component
- Proposed fix
- Before/after result

---

## Section 2: Validation Commands and Telemetry Requirements

**Validation commands:**

100-step:
```bash
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --steps 100
```

200-step:
```bash
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --steps 200
```

500-step:
```bash
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --steps 500
```

1000-step:
```bash
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --steps 1000
```

**Each run must produce:**
- Telemetry CSV
- Validation summary JSON or markdown
- Command used
- Survival steps
- Termination reason
- Pass/fail status

**Required telemetry groups:**

All required telemetry groups from the balance-core architecture must be present:
- Metadata fields (controller_mode, step, time, etc.)
- State fields (pitch_x_rad, roll_y_rad, yaw_z_rad, rates, CoM, capture point, wheel state)
- Posture fields (joint positions, joint velocities, joint errors)
- Contact fields (contact_supervisor_state, contact forces, contact validity)
- Torque fields (tau_shape_posture_per_joint, tau_support_feedforward_per_joint, tau_sagittal_wheel_balance_per_joint, tau_lateral_roll_balance_per_joint, tau_total_raw_per_joint, tau_total_clipped_per_joint, tau_final_per_joint, active_torque_owner_per_joint, ownership_violation_count)
- Actuator fields (actuator_ctrl_per_joint)
- Safety fields (torque_saturation_mask_per_joint, torque_rate_saturation_mask_per_joint)
- Hidden/legacy torque validation fields (tau_wbc_norm, hidden_torque_norm, legacy source norms)

---


## Section 3: Structural Invariant Checks

Before classifying any performance failure, every validation run must first pass structural invariant checks. If any structural invariant fails, stop immediately. This is an architecture regression, not a performance issue.

**Invariant 1: Correct controller mode**
```python
assert (df["controller_mode"] == "balance-core").all()
```

**Invariant 2: Required telemetry fields exist**

Required groups:
- Metadata fields
- State fields
- Posture fields
- Contact fields
- Torque fields
- Hidden/legacy torque fields

If any required field is missing, stop.

**Invariant 3: Zero ownership violations**
```python
assert df["ownership_violation_count"].sum() == 0
```

**Invariant 4: Valid torque owners**

Parse `active_torque_owner_per_joint` and verify every joint owner matches the balance-core ownership table.

No WBC, legacy wheel balance, hip-roll centering, posture regularizer, leg position, inverse dynamics, or experimental controller may appear as an active owner in balance-core.

**Invariant 5: WBC and hidden legacy torque remain zero**

If present:
- `tau_wbc_norm < tolerance`
- `tau_legacy_wheel_balance_norm < tolerance`
- `tau_legacy_hip_roll_centering_norm < tolerance`
- `tau_posture_regularizer_norm < tolerance`
- `tau_leg_position_norm < tolerance`
- `tau_inverse_dynamics_norm < tolerance`

Required:
- `hidden_torque_norm < tolerance`

Recommended tolerance: 1e-6 for exact disabled sources, or 1e-5 if floating-point aggregation noise exists.

**Invariant 6: All torque values are finite**

For scalar torque fields: values must be finite

For per-joint vector torque fields:
- Parse vector
- Every element must be finite

Fields include:
- `tau_shape_posture_per_joint`
- `tau_support_feedforward_per_joint`
- `tau_sagittal_wheel_balance_per_joint`
- `tau_lateral_roll_balance_per_joint`
- `tau_total_raw_per_joint`
- `tau_total_clipped_per_joint`
- `tau_final_per_joint`
- `actuator_ctrl_per_joint`

**Invariant 7: Safety masks are valid**

Parse:
- `torque_saturation_mask_per_joint`
- `torque_rate_saturation_mask_per_joint`

Each must have length 10 and contain only boolean or 0/1 values.

**Invariant 8: Contact supervisor state is valid**

`contact_supervisor_state` must be one of the defined states:
- DOUBLE_CONTACT
- SINGLE_LEFT
- SINGLE_RIGHT
- NO_CONTACT
- UNKNOWN or INIT only during initialization if explicitly allowed

`contact_duration_s` must be non-negative.

**Invariant 9: No fake contact force**

In balance-core, controller-side logic must not assign fake ground reaction force to a non-contact wheel.

If contact force command telemetry exists:
- Left non-contact → left assigned contact force = 0
- Right non-contact → right assigned contact force = 0

**Invariant 10: No non-wheel floor contact in nominal validation**

If `non_wheel_floor_contact_count` exists:
```python
assert df["non_wheel_floor_contact_count"].max() == 0
```

**If any invariant fails:**
- Classify as architecture regression
- Do not proceed to performance failure classification
- Fix the structural issue first

**If all invariants pass:**
- Proceed to performance failure mode classification

---


## Section 4: Failure Mode Classification

After all structural invariants pass, classify the failure mode using temporal analysis. The classifier must identify the root cause, not just the final termination state.

**Classification principle:**

The primary failure mode is the earliest meaningful threshold crossing that explains subsequent failures. Later events such as height collapse or contact loss may be secondary consequences.

**Classification process:**

1. Parse telemetry time series
2. Identify threshold crossings with step index and timestamp
3. Determine temporal order of violations
4. Identify the earliest root-cause crossing
5. Record secondary effects that followed
6. Map the root cause to the responsible balance-core component
7. Produce evidence fields for the classification

**Failure priority order:**

- **Priority 0:** Architecture Regression (not performance failures)
- **Priority 1:** Support and Contact Failures
- **Priority 2:** Primary Balance Axis Failures
- **Priority 3:** Dynamic Quality Failures

---



## Section 5: Failure Mode Definitions and Component Mapping

**Classification Principle:** The primary failure mode is the earliest meaningful threshold crossing that explains subsequent failures. Later events (height collapse, contact loss, wheel liftoff) are often secondary consequences, not root causes.

### Priority 0: Architecture Regression

These are not balance performance failures. If any Priority 0 failure occurs, stop immediately.

**F0.1-F0.8:** Hidden legacy torque, ownership violation, non-finite torque, WBC active, fake contact force, invalid torque owner, architectural torque saturation, architectural torque-rate saturation.

For each: Detection rule, temporal rule, responsible component, evidence fields, action required.

### Priority 1: Support and Contact Failures

Temporal rule: Only primary if before pitch_x/roll_y divergence.

**F1.1:** Knee/Support Collapse - ShapePostureController or SupportFeedforwardController
**F1.2:** Height Collapse - ShapePostureController or SupportFeedforwardController  
**F1.3:** Contact Loss - ContactSupervisor if primary; otherwise earlier failure

### Priority 2: Primary Balance Axis Failures

Temporal rule: Primary if earliest crossing before height/contact collapse.

**F2.1:** Pitch Divergence - SagittalWheelBalanceController
**F2.2:** Roll Divergence - LateralRollBalanceController

### Priority 3: Dynamic Quality Failures

Temporal rule: Only primary if no earlier support/contact/pitch/roll divergence.

**F3.1:** Wheel Velocity Runaway - SagittalWheelBalanceController
**F3.2:** Excessive Wheel Acceleration - SagittalWheelBalanceController or SafetyLimiter
**F3.3:** Oscillation - Controller for oscillating axis
**F3.4:** Position Drift - Future outer-loop controller (defer, do not fix inner balance)

### Initial Thresholds

| Threshold | Value |
|-----------|-------|
| pitch_x_max | 0.30 rad |
| roll_y_max | 0.20 rad |
| com_z_drop_max | 0.05 m |
| knee_error_max | 0.15 rad |
| wheel_vel_max | 50 rad/s |
| wheel_acc_max | 100 rad/s² |
| position_drift_max | 0.5 m |

---

## Section 6: Allowed Fixes and Fix Scope

**General Rules:**
- Fix one primary component per cycle
- No blind parameter adjustment
- No new controller stages
- No WBC reintroduction
- No legacy torque sources
- No fake contact force
- Evidence-bounded changes only

**Failure-to-Component Mapping:**

**Priority 0:** Fix architecture regression (ownership, WBC disable, torque limits, rate-limit init)

**Priority 1:**
- F1.1/F1.2: ShapePostureController or SupportFeedforwardController only
- F1.3: ContactSupervisor if primary; otherwise fix earlier failure

**Priority 2:**
- F2.1: SagittalWheelBalanceController only (diagnostic order: verify inputs, sign, saturation, then adjust)
- F2.2: LateralRollBalanceController only (diagnostic order: verify inputs, sign, saturation, then adjust)

**Priority 3:**
- F3.1/F3.2: SagittalWheelBalanceController
- F3.3: Controller for oscillating axis
- F3.4: Defer to future outer-loop (do not modify inner balance)

**Stop Conditions:**
- Multi-component fix required
- WBC required (request separate spec)
- New controller stage required
- Contact recovery required
- Joint ownership change required
- Architecture principle conflict
- Repeated failure after 3 cycles
- Cascading new failures

---

## Section 7: Acceptance Criteria

### 7.1 Structural Acceptance

- controller_mode == "balance-core"
- ownership_violation_count == 0
- Valid active_torque_owner_per_joint
- WBC torque zero
- Hidden legacy torque zero
- All torques finite
- Safety masks valid
- Contact supervisor state valid
- No fake contact force
- All required telemetry groups present

### 7.2 Performance Acceptance by Duration

For 100, 200, 500, 1000 steps:
- Simulation completes duration
- pitch_x bounded
- roll_y bounded
- com_z maintained
- Knee/support errors bounded
- Wheel velocity bounded
- Contact valid
- No Priority 0 failures

### 7.3 Duration Progression

- 100 → 200 → 500 → 1000
- Failure triggers diagnostic cycle at same duration
- Do not skip durations

### 7.4 Diagnostic-Cycle Acceptance

- Primary failure classified
- Responsible component identified
- Fix within allowed scope
- Structural invariants pass after fix
- Same failure resolved
- Fix cycle report complete

### 7.5 Completion Criteria

Balance-core performance stabilization complete when:
- Structural acceptance passes
- 500-step validation passes (milestone)
- **1000-step validation passes (completion)**
- No open Priority 0 failures
- No open Priority 1 failures
- No open Priority 2 failures
- No unresolved wheel velocity runaway, acceleration, or oscillation
- Only stable position drift may be deferred

### 7.6 Partial Milestone

500-step pass is a milestone, not final completion.

If 1000-step fails:
- Priority 0: Must fix
- Priority 1: Must fix or request architecture review
- Priority 2 pitch/roll: Must continue diagnostic cycles or request review
- Priority 3 wheel/acceleration/oscillation: Must fix unless explicitly reviewed
- Position drift only (with bounded pitch/roll/height/contact/wheel): Defer to future outer-loop

---

## Section 8: Out of Scope

**Not included:**
- Gain tuning recipes
- WBC reintroduction (separate spec)
- New controller stages
- Full re-contact recovery
- Outer-loop position controller
- Push-disturbance robustness
- Terrain/locomotion control
- Residual RL integration
- Real-robot deployment

**Explicitly forbidden:**
- Blind tuning
- New controller layers
- WBC as performance fix
- Legacy torque sources
- Fake contact force
- Changing joint ownership
- Multi-component fixes without evidence
- Treating position drift as inner-balance failure

**May be deferred:**
- Outer-loop position return
- Re-contact recovery state machine
- WBC reintroduction
- Architecture review

---

## Section 9: Summary and Next Steps

**Goal:** Validate and stabilize balance-core until 500-step and 1000-step nominal standing passes.

**Approach:** Diagnostic-first iterative cycles with progressive duration scaling (100→200→500→1000).

**Key Constraints:**
- No blind tuning
- No new stages
- No WBC reintroduction
- No architecture expansion
- Fixes only within existing components

**Workflow:**
1. Run validation at current duration
2. Check structural invariants
3. If pass: advance
4. If fail: classify → map to component → fix → re-validate

**Completion:**
- 500-step pass = milestone
- 1000-step pass = performance completion
- All structural invariants maintained
- No open Priority 0/1/2 failures
- Priority 3 wheel velocity runaway, excessive acceleration, and growing oscillation must be resolved or escalated for review; only stable position drift may be deferred to future outer-loop position control

**Next Step:** Write implementation plan using superpowers:writing-plans skill.

---
