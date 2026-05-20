# Full WBC C1 Correctness Rebuild Design

**Goal:** Rebuild the hierarchical whole-body controller path so sensing, orientation convention, contact forces, wrench computation, Jacobian torque mapping, torque allocation, and telemetry are physically truthful before gain tuning.

**Selected approach:** C1 — staged full-WBC rebuild with hard invariants.

**Canonical convention:**

```text
World/body axes:
X = lateral / roll axis
Y = sagittal / pitch axis
Z = vertical / yaw axis

roll  = rotation about X
pitch = rotation about Y
yaw   = rotation about Z

roll error  -> Mx
pitch error -> My
yaw error   -> Mz
```

---

## Background and current root causes

The current hierarchical controller simulation fails quickly: the telemetry file `outputs/hierarchical_controller_sim/telemetry_1779152450.csv` shows termination after roughly 0.52 s with `height_too_low`, CoM height dropping from about 0.413 m to 0.349 m, and pitch diverging to about -27.5 deg.

The observed weak reaction force is not caused by MuJoCo failing to compute kinematics. The main issues are controller wiring and truthfulness of state/contact interpretation:

1. `tau_wbc` is computed but is not the primary torque applied to `mj_data.ctrl` in `scripts/simulate_hierarchical_controller.py`.
2. Wheel contact geom IDs are hardcoded incorrectly in `wheeled_biped/controllers/centroidal_state_estimator.py`; the model's wheel collision geom IDs are resolved by name as `l_wheel_collision` and `r_wheel_collision`.
3. Measured contact force must use `mujoco.mj_contactForce()` and be transformed to world frame; direct `efc_force[i]` usage is not sufficient as a world vertical contact-force measurement.
4. Existing orientation helper logic must be corrected to the selected convention: roll about X, pitch about Y.
5. WBC angular-rate inputs must come from base angular velocity, not ambiguous observation indices that currently overlap joint positions.
6. The desired wrench must use actual model mass, not a hardcoded diagnostic target such as 147.4 N.
7. Contact-force distribution must not command support through wheels that are not actually in contact.
8. Torque allocation currently mixes independent wheel balancing, leg position control, and hip-roll control in a way that can fight or bypass the WBC path.

---

## Architecture boundary

The corrected pipeline should be layered as:

```text
MuJoCo state
  -> State/Contact Sensing
  -> Orientation Convention Adapter
  -> Desired Wrench Computer
  -> Contact-Constrained Force Distributor
  -> Contact Jacobian Mapper
  -> Torque Allocator
  -> MuJoCo ctrl
  -> Telemetry Verification
```

The script `scripts/simulate_hierarchical_controller.py` should become an experiment runner that wires components and logs telemetry. Controller math should live in controller modules with focused tests.

Initial files expected to change:

- `wheeled_biped/controllers/orientation_utils.py` — canonical roll/pitch/yaw extraction.
- `wheeled_biped/controllers/centroidal_state_estimator.py` — contact geom lookup by name, CoM state, measured contact forces.
- `wheeled_biped/controllers/centroidal_wrench_computer.py` — desired wrench with roll=Mrx/Mx and pitch=My.
- `wheeled_biped/controllers/contact_jacobian.py` — force-to-torque sign and contact-point correctness.
- `wheeled_biped/controllers/simple_force_distributor.py` or a focused replacement — contact-aware feasible force distribution.
- `wheeled_biped/controllers/integrated_wbc.py` — corrected WBC orchestration.
- `scripts/simulate_hierarchical_controller.py` — applies WBC torque as primary torque and logs verification telemetry.
- `tests/` — add invariant tests for each layer.

---

## Non-negotiable invariants

### 1. Coordinate convention

All controller code must obey:

```text
roll angle  -> roll moment Mx
pitch angle -> pitch moment My
yaw angle   -> yaw moment Mz
```

`orientation_utils.py` is the single source of truth for orientation extraction.

### 2. State mapping

The WBC state must explicitly carry:

```text
base_quat
base_rotmat
base_ang_vel
roll, pitch, yaw
roll_rate, pitch_rate, yaw_rate
joint_pos[10]
joint_vel[10]
com_pos
com_vel
left/right wheel contact state
left/right measured contact force
```

The desired wrench computer should not read roll rate or pitch rate from ambiguous `obs` indices.

### 3. Contact truth

Wheel contact must be detected by geom names:

```text
l_wheel_collision
r_wheel_collision
```

Contact force measurement must distinguish:

```text
contact active
contact normal force
world-frame contact force vector
world vertical contact force
```

No-support state must be represented explicitly.

### 4. Desired wrench truth

Gravity compensation must use:

```text
Fz_gravity = sum(mj_model.body_mass) * gravity
```

Desired wrench layout is:

```text
[Fx, Fy, Fz, Mx, My, Mz]
```

Where:

```text
Mx = roll stabilization
My = pitch stabilization
Mz = yaw stabilization, initially zero unless intentionally controlled
```

### 5. Force distribution truth

The force distributor must not invent ground support:

```text
if both wheels contact:
    split/support Fz through both wheels
elif only left wheel contacts:
    command left force only
elif only right wheel contacts:
    command right force only
else:
    command zero contact force and report infeasible/no-support
```

The distributor must return diagnostics indicating feasibility and support state.

### 6. Torque mapping truth

Jacobian sign convention must be locked by MuJoCo-based tests or diagnostics. A commanded upward support force at reset should not reduce measured support force or immediately remove contact.

### 7. Torque allocation truth

Final torque applied to `mj_data.ctrl[:]` must be driven primarily by corrected WBC torque. Secondary terms are allowed only if they are explicit and logged:

```text
tau_wbc
tau_posture
tau_wheel
tau_total
tau_total_clipped
```

No controller should overwrite WBC-controlled joints silently.

---

## Implementation phases

### Phase 1 — Convention and state truth

Fix orientation utilities and state mapping so WBC receives actual base orientation and angular velocity.

Verification:

```text
- identity quaternion gives roll=0, pitch=0, yaw=0
- pure X rotation gives roll only
- pure Y rotation gives pitch only
- gravity-vector orientation agrees with quaternion orientation for small roll/pitch
- WBC receives roll_rate/pitch_rate from base angular velocity, not joint qpos
```

### Phase 2 — Contact sensing and measured force truth

Resolve wheel geom IDs by name and measure contact forces using MuJoCo's contact-force API.

Verification:

```text
- reset keyframe detects wheel contacts
- reset measured Fz is positive
- no-contact state reports zero support
- total reset contact force is plausibly near model weight
```

### Phase 3 — Desired wrench correctness

Compute desired wrench from explicit state and actual model mass.

Verification:

```text
- upright static desired Fz approximately equals mass * gravity
- positive height error increases Fz
- negative height error decreases Fz
- positive roll generates corrective Mx
- positive pitch generates corrective My
- zero roll/pitch/rates gives zero moment
```

### Phase 4 — Contact-aware force distribution

Upgrade or replace the simple force distributor to consume contact state.

Verification:

```text
- both contacts split Fz
- left-only contact sends right force to zero
- right-only contact sends left force to zero
- no contact outputs zero forces and infeasible=True
- achieved wrench diagnostic reflects lost support
```

### Phase 5 — Jacobian and torque sign validation

Validate that mapped torques produce the intended physical effect in MuJoCo.

Verification:

```text
- upward force command does not reduce support force at reset
- symmetric vertical force command produces symmetric hip/knee torque pattern
- lateral/pitch commands generate expected moment direction
```

### Phase 6 — Integrated WBC torque path

Make `IntegratedWBC` orchestrate the corrected sensing, wrench, distribution, mapping, clipping, and diagnostics path. Make the simulation script apply WBC torque as primary control.

Verification:

```text
- WBC torque is nonzero and appears in mj_data.ctrl
- no diagnostic force target is hardcoded
- actual contact force is logged every step
- contact loss is visible in telemetry
- no independent controller silently overrides WBC torque
```

### Phase 7 — Static balance validation

Run short static standing validation before any gain tuning.

Verification target:

```text
- survival time improves beyond the current 0.52 s failure
- contact force telemetry remains nonzero for most of the rollout
- no false desired-force success when contacts are gone
- failure reason is physically interpretable from telemetry
```

### Phase 8 — Disturbance-ready WBC

Add small lean/push checks only after static validation passes.

Verification:

```text
- small perturbation produces corrective torque in expected direction
- controller recovers or logs a physically meaningful infeasibility
```

---

## Diagnostics and failure classification

The corrected telemetry must distinguish:

```text
desired wrench exists but contact is inactive -> no physical support possible
contact is active but measured force is low -> torque mapping / actuator / contact model issue
measured vertical force is enough but pitch/roll diverges -> moment distribution / sign / gain issue
torques saturate immediately -> infeasible command or bad allocation
robot falls while telemetry claims success -> diagnostic bug
```

Required telemetry fields:

```text
mass_kg
weight_N
roll_rad
pitch_rad
yaw_rad
roll_rate_rad_s
pitch_rate_rad_s
yaw_rate_rad_s
left_contact_active
right_contact_active
n_contacts
left_contact_force_world_x
left_contact_force_world_y
left_contact_force_world_z
right_contact_force_world_x
right_contact_force_world_y
right_contact_force_world_z
total_contact_force_z
desired_wrench_Fx
desired_wrench_Fy
desired_wrench_Fz
desired_wrench_Mx
desired_wrench_My
desired_wrench_Mz
distributed_left_fx
distributed_left_fy
distributed_left_fz
distributed_right_fx
distributed_right_fy
distributed_right_fz
achieved_virtual_wrench_error_norm
force_distribution_feasible
tau_wbc_max
tau_posture_max
tau_wheel_max
tau_total_max
tau_saturation_rate
contact_loss_reason
termination_reason
```

Rollout failure classification should use one primary reason:

```text
no_support_contact_lost
height_too_low
pitch_limit
roll_limit
torque_saturation
force_distribution_infeasible
nan_or_invalid_state
completed
```

---

## Success gates before tuning

No gain tuning should happen until these pass:

```text
- orientation convention tests
- contact detection tests
- measured-force tests
- desired-wrench sign tests
- torque-application smoke test
```

The first C1 pass is considered correct when:

```text
- both wheel contacts are correctly detected at reset
- measured total vertical contact force is near robot weight at static reset
- desired Fz equals model mass * gravity plus height correction
- WBC torque is actually applied to mj_data.ctrl
- telemetry shows actual contact force, not only desired force
- robot survives longer than the current immediate failure or reports a physically meaningful failure reason
```

---

## Scope exclusions

This design does not attempt to solve long-horizon PPO residual training, sim-to-real validation, stand-up recovery, locomotion, stair climbing, or rough terrain. It only rebuilds and verifies the full classical WBC path needed before those layers can be trusted.
