# Balance-Core Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the current wheeled-biped controller stack into the approved `balance-core` architecture with explicit mode isolation, torque ownership, contact supervision, standardized telemetry, and a four-source clean torque stack.

**Architecture:** Add functional balance-core components beside the existing experimental controllers, then route `scripts/simulate_hierarchical_controller.py --controller-mode balance-core` through one explicit `BalanceCoreTorqueComposer`. Keep legacy and experimental modules available but isolated. WBC remains in the repository and is off by default in balance-core.

**Tech Stack:** Python, NumPy, JAX NumPy, MuJoCo, pytest, existing `wheeled_biped.controllers` package, existing telemetry CSV flow in `scripts/simulate_hierarchical_controller.py`.

---

## Non-negotiable constraints

- Do not tune gains while implementing this plan.
- Do not add another experimental controller stage.
- Do not create new production files, classes, or functions with stage-based names.
- Do not make `IntegratedWBC` the default balance-core controller.
- Do not delete `IntegratedWBC`, `ContactJacobian`, `SimpleForceDistributor`, or WBC infrastructure.
- Do not write short-term rollout patches.
- Do not let legacy torque sources contribute to `tau_final` in `--controller-mode balance-core`.
- Do not assign fake ground reaction force to non-contact wheels.
- Do not commit unless the user explicitly approves committing.

## File structure map

### New production files

- `wheeled_biped/controllers/balance_core_types.py`
  - Joint index constants, torque source names, telemetry dataclasses, and balance-core result containers.
- `wheeled_biped/controllers/torque_ownership_validator.py`
  - Validates per-source joint ownership and returns owner labels/violation counts.
- `wheeled_biped/controllers/contact_supervisor.py`
  - Converts left/right contact flags and contact force validity into a contact supervisor state.
- `wheeled_biped/controllers/shape_posture_controller.py`
  - Functional replacement concept for posture support; outputs only `tau_shape_posture` on `[1,2,3,6,7,8]`.
- `wheeled_biped/controllers/support_feedforward_controller.py`
  - Functional replacement concept for support feedforward; outputs only `tau_support_feedforward` on `[2,3,7,8]`.
- `wheeled_biped/controllers/sagittal_wheel_balance_controller.py`
  - Functional sagittal balance owner; outputs only `tau_sagittal_wheel_balance` on `[4,9]`.
- `wheeled_biped/controllers/lateral_roll_balance_controller.py`
  - Functional lateral balance owner; outputs only `tau_lateral_roll_balance` on `[0,5]`.
- `wheeled_biped/controllers/balance_core_torque_composer.py`
  - Composes exactly the four approved torque sources, applies clipping/rate limiting, validates ownership, and emits telemetry fields.

### Existing files to modify

- `wheeled_biped/controllers/__init__.py`
  - Export balance-core production components.
- `scripts/simulate_hierarchical_controller.py`
  - Add `--controller-mode balance-core`.
  - Add incompatible-flag validation.
  - Instantiate balance-core components.
  - Route balance-core through `BalanceCoreTorqueComposer`.
  - Preserve legacy behavior outside balance-core.
  - Add required telemetry fields.
- Existing experimental files remain for legacy experiments:
  - `wheeled_biped/controllers/stage2b_roll_direct_controller.py`
  - `wheeled_biped/controllers/stage2b_sagittal_wheel_controller.py`
  - `wheeled_biped/controllers/stage2c_sagittal_state_feedback_controller.py`
  - `wheeled_biped/controllers/stage2d_sagittal_lqr_controller.py`

### New tests

- `tests/test_torque_ownership_validator.py`
- `tests/test_contact_supervisor.py`
- `tests/test_balance_core_components.py`
- `tests/test_balance_core_torque_composer.py`
- `tests/test_balance_core_mode_isolation.py`
- `tests/test_balance_core_telemetry_schema.py`

---

## Task 1: Add balance-core type definitions and constants

**Objective:** Define canonical joint groups, source names, contact states, telemetry containers, and composition result types. This prevents every controller from duplicating joint ownership definitions.

**Files:**
- Create: `wheeled_biped/controllers/balance_core_types.py`
- Test: `tests/test_torque_ownership_validator.py`

**Required behavior:**
- Action/torque dimension is exactly 10.
- Joint groups match the approved ownership table.
- Source names are functional and production-style.
- No new stage-named production symbol is introduced.

**Dependencies:** None.

**Rollback/safety notes:** This task only adds isolated types and constants. It should not change runtime behavior.

- [ ] **Step 1: Write failing constants test**

Add this initial content to `tests/test_torque_ownership_validator.py`:

```python
import jax.numpy as jnp

from wheeled_biped.controllers.balance_core_types import (
    ACTION_DIM,
    HIP_ROLL_INDICES,
    HIP_YAW_INDICES,
    HIP_PITCH_INDICES,
    KNEE_INDICES,
    SUPPORT_SHAPE_INDICES,
    SUPPORT_FEEDFORWARD_INDICES,
    WHEEL_INDICES,
    TorqueSourceName,
)


def test_balance_core_joint_groups_match_approved_ownership_table():
    assert ACTION_DIM == 10
    assert HIP_ROLL_INDICES == (0, 5)
    assert HIP_YAW_INDICES == (1, 6)
    assert HIP_PITCH_INDICES == (2, 7)
    assert KNEE_INDICES == (3, 8)
    assert SUPPORT_SHAPE_INDICES == (1, 2, 3, 6, 7, 8)
    assert SUPPORT_FEEDFORWARD_INDICES == (2, 3, 7, 8)
    assert WHEEL_INDICES == (4, 9)


def test_balance_core_torque_source_names_are_functional():
    assert TorqueSourceName.SHAPE_POSTURE.value == "tau_shape_posture"
    assert TorqueSourceName.SUPPORT_FEEDFORWARD.value == "tau_support_feedforward"
    assert TorqueSourceName.SAGITTAL_WHEEL_BALANCE.value == "tau_sagittal_wheel_balance"
    assert TorqueSourceName.LATERAL_ROLL_BALANCE.value == "tau_lateral_roll_balance"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_torque_ownership_validator.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wheeled_biped.controllers.balance_core_types'`.

- [ ] **Step 3: Add `balance_core_types.py`**

Create `wheeled_biped/controllers/balance_core_types.py`:

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import jax.numpy as jnp
from jax import Array

ACTION_DIM = 10

HIP_ROLL_INDICES = (0, 5)
HIP_YAW_INDICES = (1, 6)
HIP_PITCH_INDICES = (2, 7)
KNEE_INDICES = (3, 8)
SUPPORT_SHAPE_INDICES = (1, 2, 3, 6, 7, 8)
SUPPORT_FEEDFORWARD_INDICES = (2, 3, 7, 8)
WHEEL_INDICES = (4, 9)


class TorqueSourceName(str, Enum):
    SHAPE_POSTURE = "tau_shape_posture"
    SUPPORT_FEEDFORWARD = "tau_support_feedforward"
    SAGITTAL_WHEEL_BALANCE = "tau_sagittal_wheel_balance"
    LATERAL_ROLL_BALANCE = "tau_lateral_roll_balance"


class ContactSupervisorState(str, Enum):
    DOUBLE_CONTACT = "double_contact"
    LEFT_ONLY = "left_only"
    RIGHT_ONLY = "right_only"
    FLIGHT_OR_NO_CONTACT = "flight_or_no_contact"


@dataclass(frozen=True)
class TorqueSource:
    name: TorqueSourceName
    tau: Array
    owned_indices: tuple[int, ...]
    compatible_shared_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class OwnershipValidationResult:
    active_torque_owner_per_joint: tuple[str, ...]
    ownership_violation_count: int
    violations: tuple[str, ...]


@dataclass(frozen=True)
class ContactSupervisorOutput:
    state: ContactSupervisorState
    previous_state: ContactSupervisorState | None
    left_wheel_contact: bool
    right_wheel_contact: bool
    contact_force_valid: bool
    left_normal_force_n: float
    right_normal_force_n: float
    contact_duration_s: float
    transition_event: str
    recovery_hook_fields: Mapping[str, object]


@dataclass(frozen=True)
class BalanceCoreState:
    pitch_x_rad: float
    roll_y_rad: float
    yaw_z_rad: float
    pitch_rate_x_rad_s: float
    roll_rate_y_rad_s: float
    yaw_rate_z_rad_s: float
    com_pos_m: Array
    com_vel_m_s: Array
    cp_xy_m: Array
    cp_error_y_m: float
    wheel_vel_left_rad_s: float
    wheel_vel_right_rad_s: float
    wheel_acc_left_rad_s2: float
    wheel_acc_right_rad_s2: float
    contact: ContactSupervisorOutput


@dataclass(frozen=True)
class BalanceCoreTorqueResult:
    tau_shape_posture: Array
    tau_support_feedforward: Array
    tau_sagittal_wheel_balance: Array
    tau_lateral_roll_balance: Array
    tau_total_raw: Array
    tau_total_clipped: Array
    tau_final: Array
    torque_saturation_mask: Array
    torque_rate_saturation_mask: Array
    active_torque_owner_per_joint: tuple[str, ...]
    ownership_violation_count: int
    telemetry: Mapping[str, object] = field(default_factory=dict)


def zeros_action() -> Array:
    return jnp.zeros(ACTION_DIM)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_torque_ownership_validator.py -v
```

Expected: PASS for the two tests added in this task.

**Acceptance criteria:**
- Joint group constants exactly match the approved ownership table.
- Source names match the required telemetry names.
- No stage-based production names are added.

---

## Task 2: Add TorqueOwnershipValidator

**Objective:** Enforce joint ownership for every torque source and detect hidden torque conflicts before torques are summed.

**Files:**
- Create: `wheeled_biped/controllers/torque_ownership_validator.py`
- Modify: `tests/test_torque_ownership_validator.py`

**Required behavior:**
- A source with nonzero torque outside its owned indices is rejected as an unowned-joint violation.
- Duplicate `TorqueSourceName` values are invalid unless explicitly allowed; balance-core allows no duplicate source names.
- `tau_shape_posture` and `tau_support_feedforward` may share only support joints `[2,3,7,8]`.
- Any other multi-source command on the same joint is rejected as an exclusive-owner conflict.
- Validator returns `active_torque_owner_per_joint` and `ownership_violation_count`.

**Dependencies:** Task 1.

**Rollback/safety notes:** Validator should be pure and independent from MuJoCo. It can be tested without simulation.

- [ ] **Step 1: Add failing validator tests**

Append to `tests/test_torque_ownership_validator.py`:

```python
import pytest

from wheeled_biped.controllers.balance_core_types import TorqueSource
from wheeled_biped.controllers.torque_ownership_validator import TorqueOwnershipValidator


def test_validator_accepts_approved_balance_core_sources():
    validator = TorqueOwnershipValidator()
    sources = [
        TorqueSource(TorqueSourceName.SHAPE_POSTURE, jnp.array([0, 1, 2, 3, 0, 0, 6, 7, 8, 0], dtype=float), SUPPORT_SHAPE_INDICES, SUPPORT_FEEDFORWARD_INDICES),
        TorqueSource(TorqueSourceName.SUPPORT_FEEDFORWARD, jnp.array([0, 0, 2, 3, 0, 0, 0, 7, 8, 0], dtype=float), SUPPORT_FEEDFORWARD_INDICES, SUPPORT_FEEDFORWARD_INDICES),
        TorqueSource(TorqueSourceName.SAGITTAL_WHEEL_BALANCE, jnp.array([0, 0, 0, 0, 4, 0, 0, 0, 0, 9], dtype=float), WHEEL_INDICES),
        TorqueSource(TorqueSourceName.LATERAL_ROLL_BALANCE, jnp.array([1, 0, 0, 0, 0, 5, 0, 0, 0, 0], dtype=float), HIP_ROLL_INDICES),
    ]

    result = validator.validate(sources)

    assert result.ownership_violation_count == 0
    assert result.active_torque_owner_per_joint == (
        "tau_lateral_roll_balance",
        "tau_shape_posture",
        "tau_shape_posture+tau_support_feedforward",
        "tau_shape_posture+tau_support_feedforward",
        "tau_sagittal_wheel_balance",
        "tau_lateral_roll_balance",
        "tau_shape_posture",
        "tau_shape_posture+tau_support_feedforward",
        "tau_shape_posture+tau_support_feedforward",
        "tau_sagittal_wheel_balance",
    )


def test_validator_rejects_torque_outside_owned_joint_group():
    validator = TorqueOwnershipValidator()
    bad_source = TorqueSource(
        TorqueSourceName.SAGITTAL_WHEEL_BALANCE,
        jnp.array([0, 0, 3, 0, 4, 0, 0, 0, 0, 9], dtype=float),
        WHEEL_INDICES,
    )

    with pytest.raises(ValueError, match="tau_sagittal_wheel_balance commands unowned joint 2"):
        validator.validate([bad_source])


def test_validator_rejects_duplicate_source_name_even_on_disjoint_joints():
    validator = TorqueOwnershipValidator()
    sources = [
        TorqueSource(TorqueSourceName.SAGITTAL_WHEEL_BALANCE, jnp.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float), WHEEL_INDICES),
        TorqueSource(TorqueSourceName.SAGITTAL_WHEEL_BALANCE, jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float), WHEEL_INDICES),
    ]

    with pytest.raises(ValueError, match="duplicate torque source name: tau_sagittal_wheel_balance"):
        validator.validate(sources)


def test_validator_rejects_exclusive_owner_conflict():
    validator = TorqueOwnershipValidator()
    source_a = TorqueSource(TorqueSourceName.SHAPE_POSTURE, jnp.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float), SUPPORT_SHAPE_INDICES)
    source_b = TorqueSource(TorqueSourceName.LATERAL_ROLL_BALANCE, jnp.array([0, 0, 2, 0, 0, 0, 0, 0, 0, 0], dtype=float), (2,))

    with pytest.raises(ValueError, match="joint 2 has conflicting exclusive owners"):
        validator.validate([source_a, source_b])


def test_validator_allows_only_shape_and_support_feedforward_sharing():
    validator = TorqueOwnershipValidator()
    sources = [
        TorqueSource(TorqueSourceName.SHAPE_POSTURE, jnp.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float), SUPPORT_SHAPE_INDICES, SUPPORT_FEEDFORWARD_INDICES),
        TorqueSource(TorqueSourceName.SUPPORT_FEEDFORWARD, jnp.array([0, 0, 2, 0, 0, 0, 0, 0, 0, 0], dtype=float), SUPPORT_FEEDFORWARD_INDICES, SUPPORT_FEEDFORWARD_INDICES),
    ]

    result = validator.validate(sources)

    assert result.active_torque_owner_per_joint[2] == "tau_shape_posture+tau_support_feedforward"
    assert result.ownership_violation_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_torque_ownership_validator.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `torque_ownership_validator`.

- [ ] **Step 3: Add validator implementation**

Create `wheeled_biped/controllers/torque_ownership_validator.py`:

```python
from collections import defaultdict

import numpy as np

from wheeled_biped.controllers.balance_core_types import (
    ACTION_DIM,
    OwnershipValidationResult,
    TorqueSource,
)


class TorqueOwnershipValidator:
    def __init__(self, tolerance: float = 1e-9, allow_duplicate_source_names: bool = False):
        self.tolerance = tolerance
        self.allow_duplicate_source_names = allow_duplicate_source_names

    def validate(self, sources: list[TorqueSource]) -> OwnershipValidationResult:
        owners_by_joint: dict[int, set[str]] = defaultdict(set)
        violations: list[str] = []
        seen_source_names: set[str] = set()

        for source in sources:
            source_name = source.name.value
            if not self.allow_duplicate_source_names and source_name in seen_source_names:
                message = f"duplicate torque source name: {source_name}"
                violations.append(message)
                raise ValueError(message)
            seen_source_names.add(source_name)

            tau = np.asarray(source.tau, dtype=float)
            if tau.shape != (ACTION_DIM,):
                raise ValueError(f"{source_name} torque must have shape (10,), got {tau.shape}")

            owned = set(source.owned_indices)
            for idx, value in enumerate(tau):
                if abs(float(value)) <= self.tolerance:
                    continue
                if idx not in owned:
                    message = f"{source_name} commands unowned joint {idx} with torque {float(value):.6g}"
                    violations.append(message)
                    raise ValueError(message)
                owners_by_joint[idx].add(source_name)

        for idx, owners in owners_by_joint.items():
            unique_owners = sorted(owners)
            if len(unique_owners) <= 1:
                continue
            allowed_shared = {
                "tau_shape_posture",
                "tau_support_feedforward",
            }
            if set(unique_owners) == allowed_shared:
                continue
            message = f"joint {idx} has conflicting exclusive owners: {unique_owners}"
            violations.append(message)
            raise ValueError(message)

        owner_labels = []
        for idx in range(ACTION_DIM):
            unique_owners = sorted(owners_by_joint.get(idx, set()))
            owner_labels.append("+".join(unique_owners) if unique_owners else "none")

        return OwnershipValidationResult(
            active_torque_owner_per_joint=tuple(owner_labels),
            ownership_violation_count=len(violations),
            violations=tuple(violations),
        )
```

- [ ] **Step 4: Run ownership validator tests**

Run:

```bash
pytest tests/test_torque_ownership_validator.py -v
```

Expected: PASS.

**Acceptance criteria:**
- Unowned-joint violations raise clear `ValueError` messages.
- Duplicate source names are rejected in balance-core.
- Exclusive owner conflicts are rejected.
- Compatible sharing is accepted only for `tau_shape_posture` and `tau_support_feedforward`.
- Active owners are returned per joint without duplicate owner labels.

---

## Task 3: Add ContactSupervisor

**Objective:** Add a contact supervisor interface that classifies contact state and exposes contact force validity without implementing full re-contact recovery.

**Files:**
- Create: `wheeled_biped/controllers/contact_supervisor.py`
- Test: `tests/test_contact_supervisor.py`

**Required behavior:**
- Contact states: `double_contact`, `left_only`, `right_only`, `flight_or_no_contact`.
- Force validity is separate from contact geometry.
- Output contains normal forces and booleans used by telemetry.
- No fake force is created for non-contact wheels.
- Supervisor tracks previous contact state, current contact duration, and transition event.
- Output includes future recovery hook fields without implementing full re-contact recovery.

**Dependencies:** Task 1.

**Rollback/safety notes:** This is a read-only classifier. It must not command torque or force.

- [ ] **Step 1: Write failing contact supervisor tests**

Create `tests/test_contact_supervisor.py`:

```python
from wheeled_biped.controllers.balance_core_types import ContactSupervisorState
from wheeled_biped.controllers.contact_supervisor import ContactSupervisor


def test_contact_supervisor_reports_double_contact():
    supervisor = ContactSupervisor()
    output = supervisor.update(
        left_wheel_contact=True,
        right_wheel_contact=True,
        contact_force_valid=True,
        left_normal_force_n=40.0,
        right_normal_force_n=41.0,
    )

    assert output.state == ContactSupervisorState.DOUBLE_CONTACT
    assert output.left_wheel_contact is True
    assert output.right_wheel_contact is True
    assert output.contact_force_valid is True
    assert output.left_normal_force_n == 40.0
    assert output.right_normal_force_n == 41.0


def test_contact_supervisor_reports_left_only_without_fake_right_force():
    supervisor = ContactSupervisor()
    output = supervisor.update(
        left_wheel_contact=True,
        right_wheel_contact=False,
        contact_force_valid=True,
        left_normal_force_n=55.0,
        right_normal_force_n=999.0,
    )

    assert output.state == ContactSupervisorState.LEFT_ONLY
    assert output.left_normal_force_n == 55.0
    assert output.right_normal_force_n == 0.0


def test_contact_supervisor_reports_no_contact_and_zero_forces():
    supervisor = ContactSupervisor(control_dt=0.02)
    output = supervisor.update(
        left_wheel_contact=False,
        right_wheel_contact=False,
        contact_force_valid=False,
        left_normal_force_n=10.0,
        right_normal_force_n=20.0,
    )

    assert output.state == ContactSupervisorState.FLIGHT_OR_NO_CONTACT
    assert output.contact_force_valid is False
    assert output.left_normal_force_n == 0.0
    assert output.right_normal_force_n == 0.0


def test_contact_supervisor_exposes_future_recovery_hook_fields():
    supervisor = ContactSupervisor(control_dt=0.02)
    first = supervisor.update(True, True, True, 40.0, 41.0)
    second = supervisor.update(True, False, True, 40.0, 41.0)

    assert first.previous_state is None
    assert first.transition_event == "initial_double_contact"
    assert second.previous_state == ContactSupervisorState.DOUBLE_CONTACT
    assert second.transition_event == "double_contact_to_left_only"
    assert second.contact_duration_s == 0.02
    assert second.recovery_hook_fields == {
        "entered_single_contact": True,
        "entered_no_contact": False,
        "force_valid_for_recovery": True,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_contact_supervisor.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `contact_supervisor`.

- [ ] **Step 3: Add contact supervisor implementation**

Create `wheeled_biped/controllers/contact_supervisor.py`:

```python
from wheeled_biped.controllers.balance_core_types import (
    ContactSupervisorOutput,
    ContactSupervisorState,
)


class ContactSupervisor:
    def __init__(self, control_dt: float = 0.02):
        self.control_dt = control_dt
        self.previous_state = None
        self.contact_duration_s = 0.0

    def update(
        self,
        left_wheel_contact: bool,
        right_wheel_contact: bool,
        contact_force_valid: bool,
        left_normal_force_n: float,
        right_normal_force_n: float,
    ) -> ContactSupervisorOutput:
        if left_wheel_contact and right_wheel_contact:
            state = ContactSupervisorState.DOUBLE_CONTACT
        elif left_wheel_contact:
            state = ContactSupervisorState.LEFT_ONLY
        elif right_wheel_contact:
            state = ContactSupervisorState.RIGHT_ONLY
        else:
            state = ContactSupervisorState.FLIGHT_OR_NO_CONTACT

        previous_state = self.previous_state
        if previous_state == state:
            self.contact_duration_s += self.control_dt
            transition_event = "none"
        else:
            self.contact_duration_s = 0.0 if previous_state is None else self.control_dt
            transition_event = f"initial_{state.value}" if previous_state is None else f"{previous_state.value}_to_{state.value}"
        self.previous_state = state

        left_force = float(left_normal_force_n) if left_wheel_contact and contact_force_valid else 0.0
        right_force = float(right_normal_force_n) if right_wheel_contact and contact_force_valid else 0.0
        recovery_hook_fields = {
            "entered_single_contact": state in {ContactSupervisorState.LEFT_ONLY, ContactSupervisorState.RIGHT_ONLY} and previous_state != state,
            "entered_no_contact": state == ContactSupervisorState.FLIGHT_OR_NO_CONTACT and previous_state != state,
            "force_valid_for_recovery": bool(contact_force_valid),
        }

        return ContactSupervisorOutput(
            state=state,
            previous_state=previous_state,
            left_wheel_contact=bool(left_wheel_contact),
            right_wheel_contact=bool(right_wheel_contact),
            contact_force_valid=bool(contact_force_valid),
            left_normal_force_n=left_force,
            right_normal_force_n=right_force,
            contact_duration_s=float(self.contact_duration_s),
            transition_event=transition_event,
            recovery_hook_fields=recovery_hook_fields,
        )
```

- [ ] **Step 4: Run contact supervisor tests**

Run:

```bash
pytest tests/test_contact_supervisor.py -v
```

Expected: PASS.

**Acceptance criteria:**
- Single/no-contact states zero non-contact normal forces.
- Supervisor exposes state, previous state, contact duration, transition event, force validity, and future recovery hook fields for telemetry and future re-contact recovery.

---

## Task 4: Add ShapePostureController

**Objective:** Add compliant shape posture support that outputs only on hip-yaw, hip-pitch, and knee joints. It must not command wheels or hip-roll.

**Files:**
- Create: `wheeled_biped/controllers/shape_posture_controller.py`
- Test: `tests/test_balance_core_components.py`

**Required behavior:**
- Output shape is `(10,)`.
- Nonzero output allowed only on `[1,2,3,6,7,8]`.
- Hip-roll and wheels remain zero.
- Controller exposes diagnostics for posture error and torque norm.
- Posture is compliant, not a rigid pose lock: provide `posture_weight` and `contact_degraded_scale` inputs so balance/contact logic can soften posture authority without changing gains.
- No gain tuning is performed; initial values mirror existing posture concepts only as defaults.

**Dependencies:** Task 1.

**Rollback/safety notes:** This file is isolated. Existing `StaticPostureHoldingController` remains untouched for legacy experiments.

- [ ] **Step 1: Write failing shape posture tests**

Create `tests/test_balance_core_components.py`:

```python
import jax.numpy as jnp

from wheeled_biped.controllers.balance_core_types import SUPPORT_SHAPE_INDICES
from wheeled_biped.controllers.shape_posture_controller import ShapePostureController


def test_shape_posture_outputs_only_on_support_shape_joints():
    controller = ShapePostureController()
    q_ref = jnp.zeros(10)
    q = jnp.array([1.0, 0.2, -0.1, 0.3, 9.0, -1.0, -0.2, 0.1, -0.3, -9.0])
    qd = jnp.ones(10) * 0.5

    tau, diagnostics = controller.compute(q_ref, q, qd)

    assert tau.shape == (10,)
    for idx in range(10):
        if idx in SUPPORT_SHAPE_INDICES:
            continue
        assert tau[idx] == 0.0
    assert diagnostics["tau_shape_posture_norm"] >= 0.0
    assert diagnostics["shape_posture_error_norm"] >= 0.0


def test_shape_posture_zero_error_returns_zero_torque():
    controller = ShapePostureController()
    q_ref = jnp.zeros(10)
    q = jnp.zeros(10)
    qd = jnp.zeros(10)

    tau, diagnostics = controller.compute(q_ref, q, qd)

    assert jnp.allclose(tau, jnp.zeros(10))
    assert diagnostics["shape_posture_error_norm"] == 0.0


def test_shape_posture_can_be_softened_for_balance_priority():
    controller = ShapePostureController()
    q_ref = jnp.zeros(10)
    q = jnp.ones(10) * 0.1
    qd = jnp.zeros(10)

    tau_nominal, _ = controller.compute(q_ref, q, qd, posture_weight=1.0, contact_degraded_scale=1.0)
    tau_soft, diagnostics = controller.compute(q_ref, q, qd, posture_weight=0.25, contact_degraded_scale=0.5)

    assert jnp.allclose(tau_soft, tau_nominal * 0.125)
    assert diagnostics["posture_weight"] == 0.25
    assert diagnostics["contact_degraded_scale"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_balance_core_components.py::test_shape_posture_outputs_only_on_support_shape_joints -v
```

Expected: FAIL with `ModuleNotFoundError` for `shape_posture_controller`.

- [ ] **Step 3: Add controller implementation**

Create `wheeled_biped/controllers/shape_posture_controller.py`:

```python
import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.balance_core_types import SUPPORT_SHAPE_INDICES, zeros_action


class ShapePostureController:
    def __init__(
        self,
        kp_hip_yaw: float = 5.0,
        kd_hip_yaw: float = 1.0,
        kp_hip_pitch: float = 30.0,
        kd_hip_pitch: float = 4.0,
        kp_knee: float = 40.0,
        kd_knee: float = 5.0,
        max_torque_hip_yaw: float = 15.0,
        max_torque_hip_pitch: float = 30.0,
        max_torque_knee: float = 30.0,
    ):
        self.kp_hip_yaw = kp_hip_yaw
        self.kd_hip_yaw = kd_hip_yaw
        self.kp_hip_pitch = kp_hip_pitch
        self.kd_hip_pitch = kd_hip_pitch
        self.kp_knee = kp_knee
        self.kd_knee = kd_knee
        self.max_torque_hip_yaw = max_torque_hip_yaw
        self.max_torque_hip_pitch = max_torque_hip_pitch
        self.max_torque_knee = max_torque_knee

    def compute(
        self,
        q_ref: Array,
        joint_pos: Array,
        joint_vel: Array,
        posture_weight: float = 1.0,
        contact_degraded_scale: float = 1.0,
    ) -> tuple[Array, dict]:
        pos_error = q_ref - joint_pos
        authority_scale = posture_weight * contact_degraded_scale
        tau = zeros_action()

        for idx in (1, 6):
            raw = self.kp_hip_yaw * pos_error[idx] - self.kd_hip_yaw * joint_vel[idx]
            tau = tau.at[idx].set(jnp.clip(raw, -self.max_torque_hip_yaw, self.max_torque_hip_yaw))

        for idx in (2, 7):
            raw = self.kp_hip_pitch * pos_error[idx] - self.kd_hip_pitch * joint_vel[idx]
            tau = tau.at[idx].set(jnp.clip(raw, -self.max_torque_hip_pitch, self.max_torque_hip_pitch))

        for idx in (3, 8):
            raw = self.kp_knee * pos_error[idx] - self.kd_knee * joint_vel[idx]
            tau = tau.at[idx].set(jnp.clip(raw, -self.max_torque_knee, self.max_torque_knee))

        tau = tau * authority_scale
        support_error = pos_error[jnp.array(SUPPORT_SHAPE_INDICES)]
        diagnostics = {
            "shape_posture_error_norm": float(jnp.linalg.norm(support_error)),
            "tau_shape_posture_norm": float(jnp.linalg.norm(tau)),
            "posture_weight": float(posture_weight),
            "contact_degraded_scale": float(contact_degraded_scale),
        }
        return tau, diagnostics
```

- [ ] **Step 4: Run component tests**

Run:

```bash
pytest tests/test_balance_core_components.py -v
```

Expected: PASS for shape posture tests.

**Acceptance criteria:**
- `ShapePostureController` never commands hip-roll or wheels.
- `posture_weight` and `contact_degraded_scale` can soften posture torque without changing controller gains.
- Existing stage/static controllers remain available but are not used by balance-core production path.

---

## Task 5: Add SupportFeedforwardController

**Objective:** Add support feedforward that outputs only on hip-pitch and knee joints with functional naming.

**Files:**
- Create: `wheeled_biped/controllers/support_feedforward_controller.py`
- Modify: `tests/test_balance_core_components.py`

**Required behavior:**
- Output shape is `(10,)`.
- Nonzero output allowed only on `[2,3,7,8]`.
- Controller accepts a 10-element empirical support vector and applies selected support group only.
- No hidden balance behavior.

**Dependencies:** Task 1.

**Rollback/safety notes:** Keep `StaticFeedforwardController` for legacy experiments.

- [ ] **Step 1: Add failing support feedforward tests**

Append to `tests/test_balance_core_components.py`:

```python
from wheeled_biped.controllers.balance_core_types import SUPPORT_FEEDFORWARD_INDICES
from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController


def test_support_feedforward_outputs_only_on_support_feedforward_joints():
    empirical = jnp.arange(10, dtype=float)
    controller = SupportFeedforwardController(empirical_support=empirical, scale=0.5, joint_group="hip_pitch_knee")

    tau, diagnostics = controller.compute()

    assert tau.shape == (10,)
    for idx in range(10):
        if idx in SUPPORT_FEEDFORWARD_INDICES:
            assert tau[idx] == empirical[idx] * 0.5
        else:
            assert tau[idx] == 0.0
    assert diagnostics["support_feedforward_joint_group"] == "hip_pitch_knee"
    assert diagnostics["tau_support_feedforward_norm"] >= 0.0


def test_support_feedforward_knee_group_does_not_command_hip_pitch():
    empirical = jnp.arange(10, dtype=float)
    controller = SupportFeedforwardController(empirical_support=empirical, scale=1.0, joint_group="knee")

    tau, diagnostics = controller.compute()

    assert tau[2] == 0.0
    assert tau[7] == 0.0
    assert tau[3] == empirical[3]
    assert tau[8] == empirical[8]
    assert diagnostics["support_feedforward_joint_group"] == "knee"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_balance_core_components.py::test_support_feedforward_outputs_only_on_support_feedforward_joints -v
```

Expected: FAIL with `ModuleNotFoundError` for `support_feedforward_controller`.

- [ ] **Step 3: Add support feedforward implementation**

Create `wheeled_biped/controllers/support_feedforward_controller.py`:

```python
import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.balance_core_types import zeros_action

_SUPPORT_GROUPS = {
    "knee": (3, 8),
    "hip_pitch": (2, 7),
    "hip_pitch_knee": (2, 3, 7, 8),
}


class SupportFeedforwardController:
    def __init__(self, empirical_support: Array, scale: float = 0.5, joint_group: str = "knee"):
        empirical_support = jnp.asarray(empirical_support)
        if empirical_support.shape != (10,):
            raise ValueError(f"empirical_support must have shape (10,), got {empirical_support.shape}")
        if joint_group not in _SUPPORT_GROUPS:
            raise ValueError(f"joint_group must be one of {sorted(_SUPPORT_GROUPS)}, got {joint_group}")
        self.empirical_support = empirical_support
        self.scale = scale
        self.joint_group = joint_group
        self.joint_indices = _SUPPORT_GROUPS[joint_group]

    def compute(self) -> tuple[Array, dict]:
        tau = zeros_action()
        for idx in self.joint_indices:
            tau = tau.at[idx].set(self.scale * self.empirical_support[idx])
        diagnostics = {
            "support_feedforward_joint_group": self.joint_group,
            "support_feedforward_scale": self.scale,
            "tau_support_feedforward_norm": float(jnp.linalg.norm(tau)),
        }
        return tau, diagnostics
```

- [ ] **Step 4: Run component tests**

Run:

```bash
pytest tests/test_balance_core_components.py -v
```

Expected: PASS for shape posture and support feedforward tests.

**Acceptance criteria:**
- Support feedforward never commands wheels or hip-roll.
- Functional telemetry name is `tau_support_feedforward`.

---

## Task 6: Add SagittalWheelBalanceController

**Objective:** Add one production sagittal wheel balance owner that commands only wheel joints and includes wheel velocity damping and slow outer position bias input without tuning gains.

**Files:**
- Create: `wheeled_biped/controllers/sagittal_wheel_balance_controller.py`
- Modify: `tests/test_balance_core_components.py`

**Required behavior:**
- Output shape is `(10,)`.
- Nonzero output allowed only on `[4,9]`.
- Sign convention is explicit through `wheel_torque_sign`, where `+1.0` means positive `pitch_x` produces positive wheel torque and `-1.0` means positive `pitch_x` produces negative wheel torque.
- Positive wheel velocity damping opposes wheel motion.
- The controller may include an `outer_position_bias` input, but it remains inside the single sagittal owner.
- No hip-pitch/knee sagittal balance torque.

**Dependencies:** Task 1.

**Rollback/safety notes:** Existing experimental sagittal controllers remain untouched and off by default in balance-core.

- [ ] **Step 1: Add failing sagittal wheel tests**

Append to `tests/test_balance_core_components.py`:

```python
from wheeled_biped.controllers.balance_core_types import WHEEL_INDICES
from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController


def test_sagittal_wheel_balance_outputs_only_on_wheels():
    controller = SagittalWheelBalanceController()
    tau, diagnostics = controller.compute(
        pitch_x_rad=0.1,
        pitch_rate_x_rad_s=0.2,
        cp_error_y_m=0.03,
        com_vy_m_s=0.04,
        wheel_vel_left_rad_s=1.0,
        wheel_vel_right_rad_s=1.5,
        outer_position_bias=0.0,
    )

    assert tau.shape == (10,)
    for idx in range(10):
        if idx in WHEEL_INDICES:
            continue
        assert tau[idx] == 0.0
    assert tau[4] == tau[9]
    assert diagnostics["wheel_vel_mean_rad_s"] == 1.25
    assert "term_wheel_velocity_damping" in diagnostics


def test_positive_pitch_x_creates_restoring_wheel_torque_with_verified_sign():
    controller = SagittalWheelBalanceController(
        k_pitch=10.0,
        k_pitch_rate=0.0,
        k_cp_y=0.0,
        k_com_vy=0.0,
        k_wheel_vel=0.0,
        max_tau_wheel=10.0,
        wheel_torque_sign=1.0,
    )
    tau, diagnostics = controller.compute(
        pitch_x_rad=0.1,
        pitch_rate_x_rad_s=0.0,
        cp_error_y_m=0.0,
        com_vy_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        outer_position_bias=0.0,
    )

    assert tau[4] == 1.0
    assert tau[9] == 1.0
    assert diagnostics["wheel_torque_sign"] == 1.0
    assert diagnostics["sign_convention"] == "positive_pitch_x_to_positive_wheel_torque"


def test_sagittal_wheel_velocity_damping_opposes_positive_velocity():
    controller = SagittalWheelBalanceController(
        k_pitch=0.0,
        k_pitch_rate=0.0,
        k_cp_y=0.0,
        k_com_vy=0.0,
        k_wheel_vel=0.5,
        max_tau_wheel=10.0,
        wheel_torque_sign=1.0,
    )
    tau, diagnostics = controller.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        cp_error_y_m=0.0,
        com_vy_m_s=0.0,
        wheel_vel_left_rad_s=2.0,
        wheel_vel_right_rad_s=2.0,
        outer_position_bias=0.0,
    )

    assert tau[4] == -1.0
    assert tau[9] == -1.0
    assert diagnostics["term_wheel_velocity_damping"] == -1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_balance_core_components.py::test_sagittal_wheel_balance_outputs_only_on_wheels -v
```

Expected: FAIL with `ModuleNotFoundError` for `sagittal_wheel_balance_controller`.

- [ ] **Step 3: Add sagittal controller implementation**

Create `wheeled_biped/controllers/sagittal_wheel_balance_controller.py`:

```python
import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.balance_core_types import zeros_action


class SagittalWheelBalanceController:
    def __init__(
        self,
        k_pitch: float = 20.0,
        k_pitch_rate: float = 6.0,
        k_cp_y: float = 8.0,
        k_com_vy: float = 0.0,
        k_wheel_vel: float = 0.3,
        max_tau_wheel: float = 8.0,
        wheel_torque_sign: float = 1.0,
    ):
        if wheel_torque_sign not in (-1.0, 1.0):
            raise ValueError("wheel_torque_sign must be -1.0 or 1.0")
        self.k_pitch = k_pitch
        self.k_pitch_rate = k_pitch_rate
        self.k_cp_y = k_cp_y
        self.k_com_vy = k_com_vy
        self.k_wheel_vel = k_wheel_vel
        self.max_tau_wheel = max_tau_wheel
        self.wheel_torque_sign = wheel_torque_sign

    def compute(
        self,
        pitch_x_rad: float,
        pitch_rate_x_rad_s: float,
        cp_error_y_m: float,
        com_vy_m_s: float,
        wheel_vel_left_rad_s: float,
        wheel_vel_right_rad_s: float,
        outer_position_bias: float = 0.0,
    ) -> tuple[Array, dict]:
        wheel_vel_mean = 0.5 * (wheel_vel_left_rad_s + wheel_vel_right_rad_s)
        term_pitch = self.wheel_torque_sign * self.k_pitch * pitch_x_rad
        term_pitch_rate = self.wheel_torque_sign * self.k_pitch_rate * pitch_rate_x_rad_s
        term_cp_y = self.wheel_torque_sign * self.k_cp_y * cp_error_y_m
        term_com_vy = self.wheel_torque_sign * self.k_com_vy * com_vy_m_s
        term_wheel_velocity_damping = -self.k_wheel_vel * wheel_vel_mean
        tau_raw = term_pitch + term_pitch_rate + term_cp_y + term_com_vy + term_wheel_velocity_damping + outer_position_bias
        tau_clipped = jnp.clip(tau_raw, -self.max_tau_wheel, self.max_tau_wheel)

        tau = zeros_action()
        tau = tau.at[4].set(tau_clipped)
        tau = tau.at[9].set(tau_clipped)

        diagnostics = {
            "term_pitch": float(term_pitch),
            "term_pitch_rate": float(term_pitch_rate),
            "term_cp_y": float(term_cp_y),
            "term_com_vy": float(term_com_vy),
            "term_wheel_velocity_damping": float(term_wheel_velocity_damping),
            "outer_position_bias": float(outer_position_bias),
            "wheel_vel_mean_rad_s": float(wheel_vel_mean),
            "wheel_torque_sign": float(self.wheel_torque_sign),
            "sign_convention": "positive_pitch_x_to_positive_wheel_torque" if self.wheel_torque_sign > 0 else "positive_pitch_x_to_negative_wheel_torque",
            "tau_sagittal_wheel_raw": float(tau_raw),
            "tau_sagittal_wheel_clipped": float(tau_clipped),
            "sagittal_wheel_saturated": bool(jnp.abs(tau_raw) > self.max_tau_wheel),
        }
        return tau, diagnostics
```

- [ ] **Step 4: Run component tests**

Run:

```bash
pytest tests/test_balance_core_components.py -v
```

Expected: PASS for shape, feedforward, and sagittal tests.

**Acceptance criteria:**
- Sagittal wheel balance owns only `[4,9]`.
- Positive `pitch_x` produces restoring wheel torque according to the documented `wheel_torque_sign` convention.
- Wheel velocity damping opposes positive wheel velocity.
- Outer position return is represented only as an input inside the sagittal owner.

---

## Task 7: Add LateralRollBalanceController

**Objective:** Add one production lateral roll balance owner that commands only hip-roll joints and keeps roll control separate from sagittal wheel balance.

**Files:**
- Create: `wheeled_biped/controllers/lateral_roll_balance_controller.py`
- Modify: `tests/test_balance_core_components.py`

**Required behavior:**
- Output shape is `(10,)`.
- Nonzero output allowed only on `[0,5]`.
- Sign convention is explicit through `hip_roll_torque_sign`, where the verified default maps positive `roll_y` to the restoring hip-roll pair `tau_left > 0`, `tau_right < 0`.
- Positive `roll_y` produces the documented restoring hip-roll pair.
- No wheel, hip-pitch, knee, or yaw torque.

**Dependencies:** Task 1.

**Rollback/safety notes:** Existing experimental direct roll controller remains untouched and off by default in balance-core.

- [ ] **Step 1: Add failing lateral roll tests**

Append to `tests/test_balance_core_components.py`:

```python
from wheeled_biped.controllers.balance_core_types import HIP_ROLL_INDICES
from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController


def test_lateral_roll_balance_outputs_only_on_hip_roll():
    controller = LateralRollBalanceController()
    tau, diagnostics = controller.compute(roll_y_rad=0.1, roll_rate_y_rad_s=0.2)

    assert tau.shape == (10,)
    for idx in range(10):
        if idx in HIP_ROLL_INDICES:
            continue
        assert tau[idx] == 0.0
    assert diagnostics["m_roll_clipped"] != 0.0


def test_positive_roll_y_produces_restoring_hip_roll_pair_with_verified_mapping():
    controller = LateralRollBalanceController(
        k_roll=100.0,
        k_roll_rate=0.0,
        tau_hip_roll_max=15.0,
        hip_roll_torque_sign=1.0,
    )
    tau, diagnostics = controller.compute(roll_y_rad=0.1, roll_rate_y_rad_s=0.0)

    assert diagnostics["m_roll_cmd"] < 0.0
    assert tau[0] > 0.0
    assert tau[5] < 0.0
    assert diagnostics["hip_roll_torque_sign"] == 1.0
    assert diagnostics["sign_convention"] == "positive_roll_y_to_left_positive_right_negative"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_balance_core_components.py::test_lateral_roll_balance_outputs_only_on_hip_roll -v
```

Expected: FAIL with `ModuleNotFoundError` for `lateral_roll_balance_controller`.

- [ ] **Step 3: Add lateral controller implementation**

Create `wheeled_biped/controllers/lateral_roll_balance_controller.py`:

```python
import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.balance_core_types import zeros_action


class LateralRollBalanceController:
    def __init__(
        self,
        k_roll: float = 100.0,
        k_roll_rate: float = 20.0,
        tau_hip_roll_max: float = 15.0,
        max_roll_moment: float | None = None,
        hip_roll_torque_sign: float = 1.0,
    ):
        if hip_roll_torque_sign not in (-1.0, 1.0):
            raise ValueError("hip_roll_torque_sign must be -1.0 or 1.0")
        self.k_roll = k_roll
        self.k_roll_rate = k_roll_rate
        self.tau_hip_roll_max = tau_hip_roll_max
        self.max_roll_moment = max_roll_moment if max_roll_moment is not None else 2.0 * tau_hip_roll_max
        self.hip_roll_torque_sign = hip_roll_torque_sign

    def compute(self, roll_y_rad: float, roll_rate_y_rad_s: float) -> tuple[Array, dict]:
        m_roll_cmd = -self.k_roll * roll_y_rad - self.k_roll_rate * roll_rate_y_rad_s
        m_roll_clipped = jnp.clip(m_roll_cmd, -self.max_roll_moment, self.max_roll_moment)
        tau_left = jnp.clip(-self.hip_roll_torque_sign * m_roll_clipped / 2.0, -self.tau_hip_roll_max, self.tau_hip_roll_max)
        tau_right = jnp.clip(self.hip_roll_torque_sign * m_roll_clipped / 2.0, -self.tau_hip_roll_max, self.tau_hip_roll_max)

        tau = zeros_action()
        tau = tau.at[0].set(tau_left)
        tau = tau.at[5].set(tau_right)

        diagnostics = {
            "m_roll_cmd": float(m_roll_cmd),
            "m_roll_clipped": float(m_roll_clipped),
            "tau_hip_roll_left": float(tau_left),
            "tau_hip_roll_right": float(tau_right),
            "hip_roll_torque_sign": float(self.hip_roll_torque_sign),
            "sign_convention": "positive_roll_y_to_left_positive_right_negative" if self.hip_roll_torque_sign > 0 else "positive_roll_y_to_left_negative_right_positive",
            "lateral_roll_saturated": bool(jnp.abs(m_roll_cmd) > self.max_roll_moment),
        }
        return tau, diagnostics
```

- [ ] **Step 4: Run component tests**

Run:

```bash
pytest tests/test_balance_core_components.py -v
```

Expected: PASS for all balance-core component tests.

**Acceptance criteria:**
- Lateral roll balance owns only `[0,5]`.
- Positive `roll_y` produces restoring hip-roll torque according to the documented `hip_roll_torque_sign` convention.
- Roll control remains separate from sagittal wheel control.

---

## Task 8: Add BalanceCoreTorqueComposer

**Objective:** Compose exactly the four approved torque sources, apply actuator clipping and torque-rate limiting, validate ownership, and return standardized telemetry.

**Files:**
- Create: `wheeled_biped/controllers/balance_core_torque_composer.py`
- Test: `tests/test_balance_core_torque_composer.py`

**Required behavior:**
- `tau_total_raw = tau_shape_posture + tau_support_feedforward + tau_sagittal_wheel_balance + tau_lateral_roll_balance`.
- `tau_total_clipped = clip(tau_total_raw, -torque_limit, torque_limit)`.
- `tau_final = tau_prev + clip((tau_total_clipped - tau_prev) / control_dt, -max_torque_rate, max_torque_rate) * control_dt`.
- WBC, momentum, posture regularizer, leg position, legacy centering, secondary wheel balance, wrapper, inverse dynamics are not inputs.
- Telemetry includes all required torque fields.

**Dependencies:** Tasks 1 and 2.

**Rollback/safety notes:** Composer is pure and independently testable.

- [ ] **Step 1: Write failing composer tests**

Create `tests/test_balance_core_torque_composer.py`:

```python
import jax.numpy as jnp

from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer


def test_composer_sums_exactly_four_approved_sources_before_clipping():
    composer = BalanceCoreTorqueComposer(control_dt=0.02, max_torque_rate=1000.0)
    result = composer.compose(
        tau_shape_posture=jnp.array([0, 1, 2, 3, 0, 0, 6, 7, 8, 0], dtype=float),
        tau_support_feedforward=jnp.array([0, 0, 20, 30, 0, 0, 0, 70, 80, 0], dtype=float),
        tau_sagittal_wheel_balance=jnp.array([0, 0, 0, 0, 4, 0, 0, 0, 0, 9], dtype=float),
        tau_lateral_roll_balance=jnp.array([10, 0, 0, 0, 0, 50, 0, 0, 0, 0], dtype=float),
        tau_prev=jnp.zeros(10),
        torque_limit=jnp.ones(10) * 100.0,
    )

    expected_raw = jnp.array([10, 1, 22, 33, 4, 50, 6, 77, 88, 9], dtype=float)
    assert jnp.allclose(result.tau_total_raw, expected_raw)
    assert jnp.allclose(result.tau_total_clipped, expected_raw)
    assert jnp.allclose(result.tau_final, expected_raw)
    assert result.ownership_violation_count == 0
    assert result.telemetry["tau_shape_posture_per_joint"] == tuple([0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 6.0, 7.0, 8.0, 0.0])


def test_composer_applies_torque_limit_and_rate_limit():
    composer = BalanceCoreTorqueComposer(control_dt=0.02, max_torque_rate=100.0)
    result = composer.compose(
        tau_shape_posture=jnp.array([0, 0, 50, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        tau_support_feedforward=jnp.zeros(10),
        tau_sagittal_wheel_balance=jnp.zeros(10),
        tau_lateral_roll_balance=jnp.zeros(10),
        tau_prev=jnp.zeros(10),
        torque_limit=jnp.ones(10) * 10.0,
    )

    assert result.tau_total_raw[2] == 50.0
    assert result.tau_total_clipped[2] == 10.0
    assert result.tau_final[2] == 2.0
    assert result.torque_saturation_mask[2] == True
    assert result.torque_rate_saturation_mask[2] == True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_balance_core_torque_composer.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `balance_core_torque_composer`.

- [ ] **Step 3: Add composer implementation**

Create `wheeled_biped/controllers/balance_core_torque_composer.py`:

```python
import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.balance_core_types import (
    HIP_ROLL_INDICES,
    SUPPORT_FEEDFORWARD_INDICES,
    SUPPORT_SHAPE_INDICES,
    WHEEL_INDICES,
    BalanceCoreTorqueResult,
    TorqueSource,
    TorqueSourceName,
)
from wheeled_biped.controllers.torque_ownership_validator import TorqueOwnershipValidator


def _as_tuple(x: Array) -> tuple[float, ...]:
    return tuple(float(v) for v in jnp.asarray(x))


class BalanceCoreTorqueComposer:
    def __init__(self, control_dt: float, max_torque_rate: float = 400.0):
        self.control_dt = control_dt
        self.max_torque_rate = max_torque_rate
        self.validator = TorqueOwnershipValidator()

    def compose(
        self,
        tau_shape_posture: Array,
        tau_support_feedforward: Array,
        tau_sagittal_wheel_balance: Array,
        tau_lateral_roll_balance: Array,
        tau_prev: Array,
        torque_limit: Array,
    ) -> BalanceCoreTorqueResult:
        sources = [
            TorqueSource(TorqueSourceName.SHAPE_POSTURE, tau_shape_posture, SUPPORT_SHAPE_INDICES, SUPPORT_FEEDFORWARD_INDICES),
            TorqueSource(TorqueSourceName.SUPPORT_FEEDFORWARD, tau_support_feedforward, SUPPORT_FEEDFORWARD_INDICES, SUPPORT_FEEDFORWARD_INDICES),
            TorqueSource(TorqueSourceName.SAGITTAL_WHEEL_BALANCE, tau_sagittal_wheel_balance, WHEEL_INDICES),
            TorqueSource(TorqueSourceName.LATERAL_ROLL_BALANCE, tau_lateral_roll_balance, HIP_ROLL_INDICES),
        ]
        ownership = self.validator.validate(sources)

        tau_total_raw = tau_shape_posture + tau_support_feedforward + tau_sagittal_wheel_balance + tau_lateral_roll_balance
        tau_total_clipped = jnp.clip(tau_total_raw, -torque_limit, torque_limit)
        tau_rate_vec = (tau_total_clipped - tau_prev) / self.control_dt
        tau_rate_vec_clipped = jnp.clip(tau_rate_vec, -self.max_torque_rate, self.max_torque_rate)
        tau_final = tau_prev + tau_rate_vec_clipped * self.control_dt

        torque_saturation_mask = jnp.abs(tau_total_raw) > torque_limit
        torque_rate_saturation_mask = jnp.abs(tau_rate_vec) > self.max_torque_rate

        telemetry = {
            "tau_shape_posture_per_joint": _as_tuple(tau_shape_posture),
            "tau_support_feedforward_per_joint": _as_tuple(tau_support_feedforward),
            "tau_sagittal_wheel_balance_per_joint": _as_tuple(tau_sagittal_wheel_balance),
            "tau_lateral_roll_balance_per_joint": _as_tuple(tau_lateral_roll_balance),
            "tau_total_raw_per_joint": _as_tuple(tau_total_raw),
            "tau_total_clipped_per_joint": _as_tuple(tau_total_clipped),
            "tau_final_per_joint": _as_tuple(tau_final),
            "active_torque_owner_per_joint": ownership.active_torque_owner_per_joint,
            "ownership_violation_count": ownership.ownership_violation_count,
            "torque_saturation_mask_per_joint": tuple(bool(v) for v in torque_saturation_mask),
            "torque_rate_saturation_mask_per_joint": tuple(bool(v) for v in torque_rate_saturation_mask),
        }

        return BalanceCoreTorqueResult(
            tau_shape_posture=tau_shape_posture,
            tau_support_feedforward=tau_support_feedforward,
            tau_sagittal_wheel_balance=tau_sagittal_wheel_balance,
            tau_lateral_roll_balance=tau_lateral_roll_balance,
            tau_total_raw=tau_total_raw,
            tau_total_clipped=tau_total_clipped,
            tau_final=tau_final,
            torque_saturation_mask=torque_saturation_mask,
            torque_rate_saturation_mask=torque_rate_saturation_mask,
            active_torque_owner_per_joint=ownership.active_torque_owner_per_joint,
            ownership_violation_count=ownership.ownership_violation_count,
            telemetry=telemetry,
        )
```

- [ ] **Step 4: Run composer tests**

Run:

```bash
pytest tests/test_balance_core_torque_composer.py -v
```

Expected: PASS.

**Acceptance criteria:**
- Composer accepts exactly the four approved source arrays.
- Composer has no WBC, legacy, or inverse-dynamics torque inputs.
- Telemetry names match the spec.

---

## Task 9: Export balance-core components

**Objective:** Make balance-core components importable from the controller package without exposing new stage-based production names.

**Files:**
- Modify: `wheeled_biped/controllers/__init__.py`
- Test: `tests/test_balance_core_components.py`

**Required behavior:**
- Existing exports remain available.
- Functional balance-core classes are exported.

**Dependencies:** Tasks 1-8.

**Rollback/safety notes:** Import-only change. No runtime behavior change.

- [ ] **Step 1: Add import test**

Append to `tests/test_balance_core_components.py`:

```python

def test_balance_core_components_export_from_controllers_package():
    from wheeled_biped.controllers import (  # noqa: F401
        BalanceCoreTorqueComposer,
        ContactSupervisor,
        LateralRollBalanceController,
        SagittalWheelBalanceController,
        ShapePostureController,
        SupportFeedforwardController,
        TorqueOwnershipValidator,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_balance_core_components.py::test_balance_core_components_export_from_controllers_package -v
```

Expected: FAIL with `ImportError` for the new package exports.

- [ ] **Step 3: Inspect and append to `__init__.py` without overwriting existing exports**

First inspect the current contents of `wheeled_biped/controllers/__init__.py`. Preserve every existing import and every existing `__all__` entry. Append only the new balance-core imports and extend `__all__` if it already exists.

Add these imports after the existing imports:

```python
from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer
from wheeled_biped.controllers.contact_supervisor import ContactSupervisor
from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController
from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController
from wheeled_biped.controllers.shape_posture_controller import ShapePostureController
from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController
from wheeled_biped.controllers.torque_ownership_validator import TorqueOwnershipValidator
```

If `__all__` already exists, append only these names to the existing list:

```python
"BalanceCoreTorqueComposer",
"ContactSupervisor",
"LateralRollBalanceController",
"SagittalWheelBalanceController",
"ShapePostureController",
"SupportFeedforwardController",
"TorqueOwnershipValidator",
```

If `__all__` does not exist, do not invent a package-wide replacement list for existing exports. Either leave `__all__` absent or create it only if the current package style already uses one elsewhere and include all existing public exports plus the new names.

- [ ] **Step 4: Run package export test**

Run:

```bash
pytest tests/test_balance_core_components.py::test_balance_core_components_export_from_controllers_package -v
```

Expected: PASS.

**Acceptance criteria:**
- Functional class imports work.
- Existing `LQRBalanceController` export remains available.

---

## Task 10: Add balance-core mode validation helpers in simulation script

**Objective:** Add `--controller-mode balance-core` and reject incompatible legacy flags before the simulation loop.

**Files:**
- Modify: `scripts/simulate_hierarchical_controller.py`
- Test: `tests/test_balance_core_mode_isolation.py`

**Required behavior:**
- `--controller-mode balance-core` exists.
- In balance-core, incompatible legacy flags fail fast.
- Balance-core rejects static wrapper, WBC correction flags, legacy wheel balance, legacy hip-roll centering controls, and simultaneous experimental sagittal controllers.
- Legacy mode remains available for existing experiments.

**Dependencies:** None for helper tests; later tasks use this helper in runtime.

**Rollback/safety notes:** Keep default `controller-mode` as `legacy` until user explicitly switches to balance-core to preserve existing behavior.

- [ ] **Step 1: Write failing mode isolation tests**

Create `tests/test_balance_core_mode_isolation.py`:

```python
from argparse import Namespace

import pytest

from scripts.simulate_hierarchical_controller import validate_balance_core_mode_args


def _args(**overrides):
    defaults = dict(
        controller_mode="balance-core",
        enable_static_dynamics_wrapper=False,
        enable_secondary_wheel_balance=False,
        enable_stage2_static_posture_hold=False,
        enable_stage2b_gravity_feedforward=False,
        enable_stage2b_roll_direct=False,
        enable_stage2b_sagittal_wheel=False,
        enable_stage2c_sagittal_state_feedback=False,
        enable_stage2d_sagittal_lqr=False,
        disable_wbc_correction=False,
        disable_hip_roll_centering=False,
        disable_wheel_balance=False,
        initialize_tau_prev_from_wbc=False,
        use_per_actuator_wbc_authority=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_balance_core_mode_accepts_clean_flags():
    validate_balance_core_mode_args(_args())


@pytest.mark.parametrize(
    "flag_name",
    [
        "enable_static_dynamics_wrapper",
        "enable_secondary_wheel_balance",
        "enable_stage2_static_posture_hold",
        "enable_stage2b_gravity_feedforward",
        "enable_stage2b_roll_direct",
        "enable_stage2b_sagittal_wheel",
        "enable_stage2c_sagittal_state_feedback",
        "enable_stage2d_sagittal_lqr",
        "initialize_tau_prev_from_wbc",
        "use_per_actuator_wbc_authority",
    ],
)
def test_balance_core_rejects_incompatible_legacy_flags(flag_name):
    with pytest.raises(ValueError, match="Use --controller-mode balance-core alone") as exc_info:
        validate_balance_core_mode_args(_args(**{flag_name: True}))
    assert flag_name in str(exc_info.value)


def test_legacy_mode_does_not_reject_legacy_flags():
    validate_balance_core_mode_args(_args(controller_mode="legacy", enable_stage2b_sagittal_wheel=True))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_balance_core_mode_isolation.py -v
```

Expected: FAIL with `ImportError` for `validate_balance_core_mode_args`.

- [ ] **Step 3: Add parser argument and validation function**

Modify `scripts/simulate_hierarchical_controller.py` near parser setup:

```python
parser.add_argument(
    "--controller-mode",
    type=str,
    default="legacy",
    choices=["legacy", "balance-core", "standing-balance"],
    help="Controller architecture mode. Use balance-core for the clean four-source standing-balance stack.",
)
```

Add this top-level helper near other helper functions:

```python
def validate_balance_core_mode_args(args):
    if args.controller_mode not in {"balance-core", "standing-balance"}:
        return

    incompatible_true_flags = [
        "enable_static_dynamics_wrapper",
        "enable_secondary_wheel_balance",
        "enable_stage2_static_posture_hold",
        "enable_stage2b_gravity_feedforward",
        "enable_stage2b_roll_direct",
        "enable_stage2b_sagittal_wheel",
        "enable_stage2c_sagittal_state_feedback",
        "enable_stage2d_sagittal_lqr",
        "initialize_tau_prev_from_wbc",
        "use_per_actuator_wbc_authority",
    ]
    enabled = [name for name in incompatible_true_flags if getattr(args, name, False)]
    if enabled:
        raise ValueError(
            "Use --controller-mode balance-core alone. Do not combine it with stage/legacy controller flags. "
            "Incompatible flags: " + ", ".join(enabled)
        )
```

Call it immediately after `args = parser.parse_args()`:

```python
args = parser.parse_args()
validate_balance_core_mode_args(args)
```

- [ ] **Step 4: Run mode isolation tests**

Run:

```bash
pytest tests/test_balance_core_mode_isolation.py -v
```

Expected: PASS.

**Acceptance criteria:**
- `balance-core` mode is explicit.
- Legacy flags fail fast only in balance-core.
- Existing legacy experiments are not blocked when `controller_mode="legacy"`.

---

## Task 11: Add telemetry schema helper for balance-core

**Objective:** Centralize required telemetry field names and formatting so simulation logging cannot silently omit balance-core fields.

**Files:**
- Create or modify: `wheeled_biped/controllers/balance_core_types.py`
- Test: `tests/test_balance_core_telemetry_schema.py`

**Required behavior:**
- Required state and torque telemetry names are listed once.
- Telemetry helper can initialize list-valued CSV columns.
- Names match the approved spec exactly.

**Dependencies:** Task 1.

**Rollback/safety notes:** Type/schema-only change.

- [ ] **Step 1: Write failing telemetry schema tests**

Create `tests/test_balance_core_telemetry_schema.py`:

```python
from wheeled_biped.controllers.balance_core_types import (
    BALANCE_CORE_REQUIRED_STATE_TELEMETRY,
    BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY,
    make_balance_core_telemetry_columns,
)


def test_balance_core_required_state_telemetry_names_match_spec():
    expected = {
        "pitch_x_rad",
        "roll_y_rad",
        "yaw_z_rad",
        "pitch_rate_x_rad_s",
        "roll_rate_y_rad_s",
        "yaw_rate_z_rad_s",
        "com_x_m",
        "com_y_m",
        "com_z_m",
        "com_vx_m_s",
        "com_vy_m_s",
        "com_vz_m_s",
        "cp_x_m",
        "cp_y_m",
        "cp_error_y_m",
        "wheel_vel_left_rad_s",
        "wheel_vel_right_rad_s",
        "wheel_vel_mean_rad_s",
        "wheel_acc_left_rad_s2",
        "wheel_acc_right_rad_s2",
        "wheel_acc_mean_rad_s2",
        "left_wheel_contact",
        "right_wheel_contact",
        "contact_supervisor_state",
        "contact_previous_state",
        "contact_duration_s",
        "contact_transition_event",
        "contact_force_valid",
        "contact_recovery_hook_fields",
    }
    assert set(BALANCE_CORE_REQUIRED_STATE_TELEMETRY) == expected


def test_balance_core_required_torque_telemetry_names_match_spec():
    expected = {
        "tau_shape_posture_per_joint",
        "tau_support_feedforward_per_joint",
        "tau_sagittal_wheel_balance_per_joint",
        "tau_lateral_roll_balance_per_joint",
        "tau_total_raw_per_joint",
        "tau_total_clipped_per_joint",
        "tau_final_per_joint",
        "active_torque_owner_per_joint",
        "ownership_violation_count",
        "torque_saturation_mask_per_joint",
        "torque_rate_saturation_mask_per_joint",
    }
    assert set(BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY) == expected


def test_make_balance_core_telemetry_columns_initializes_all_required_lists():
    columns = make_balance_core_telemetry_columns()
    for name in BALANCE_CORE_REQUIRED_STATE_TELEMETRY + BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY:
        assert name in columns
        assert columns[name] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_balance_core_telemetry_schema.py -v
```

Expected: FAIL with import errors for missing telemetry constants.

- [ ] **Step 3: Add telemetry schema constants**

Append to `wheeled_biped/controllers/balance_core_types.py`:

```python
BALANCE_CORE_REQUIRED_STATE_TELEMETRY = (
    "pitch_x_rad",
    "roll_y_rad",
    "yaw_z_rad",
    "pitch_rate_x_rad_s",
    "roll_rate_y_rad_s",
    "yaw_rate_z_rad_s",
    "com_x_m",
    "com_y_m",
    "com_z_m",
    "com_vx_m_s",
    "com_vy_m_s",
    "com_vz_m_s",
    "cp_x_m",
    "cp_y_m",
    "cp_error_y_m",
    "wheel_vel_left_rad_s",
    "wheel_vel_right_rad_s",
    "wheel_vel_mean_rad_s",
    "wheel_acc_left_rad_s2",
    "wheel_acc_right_rad_s2",
    "wheel_acc_mean_rad_s2",
    "left_wheel_contact",
    "right_wheel_contact",
    "contact_supervisor_state",
    "contact_previous_state",
    "contact_duration_s",
    "contact_transition_event",
    "contact_force_valid",
    "contact_recovery_hook_fields",
)

BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY = (
    "tau_shape_posture_per_joint",
    "tau_support_feedforward_per_joint",
    "tau_sagittal_wheel_balance_per_joint",
    "tau_lateral_roll_balance_per_joint",
    "tau_total_raw_per_joint",
    "tau_total_clipped_per_joint",
    "tau_final_per_joint",
    "active_torque_owner_per_joint",
    "ownership_violation_count",
    "torque_saturation_mask_per_joint",
    "torque_rate_saturation_mask_per_joint",
)


def make_balance_core_telemetry_columns() -> dict[str, list]:
    return {
        name: []
        for name in BALANCE_CORE_REQUIRED_STATE_TELEMETRY + BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY
    }
```

- [ ] **Step 4: Run telemetry schema tests**

Run:

```bash
pytest tests/test_balance_core_telemetry_schema.py -v
```

Expected: PASS.

**Acceptance criteria:**
- Required telemetry names are centralized.
- Names match the spec exactly.

---

## Task 12: Integrate balance-core controller construction in simulation script

**Objective:** Instantiate balance-core components in `scripts/simulate_hierarchical_controller.py` without changing legacy behavior.

**Files:**
- Modify: `scripts/simulate_hierarchical_controller.py`
- Test: `tests/test_balance_core_mode_isolation.py`

**Required behavior:**
- In `balance-core`, instantiate `ContactSupervisor`, `ShapePostureController`, `SupportFeedforwardController`, `SagittalWheelBalanceController`, `LateralRollBalanceController`, and `BalanceCoreTorqueComposer`.
- Balance-core construction resolves support feedforward through a functional helper such as `resolve_support_feedforward_vector()`; it must not directly call Stage2B/Stage2C/Stage2D-named helpers.
- In legacy mode, existing setup remains available.
- WBC may still be constructed for legacy diagnostics only if existing script structure requires it, but balance-core must not use `tau_wbc` in `tau_final`.

**Dependencies:** Tasks 3-8 and 10.

**Rollback/safety notes:** Keep the runtime branch small: add a balance-core path rather than rewriting the full script.

- [ ] **Step 1: Add import block**

Modify imports in `scripts/simulate_hierarchical_controller.py`:

```python
from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer
from wheeled_biped.controllers.balance_core_types import make_balance_core_telemetry_columns
from wheeled_biped.controllers.contact_supervisor import ContactSupervisor
from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController
from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController
from wheeled_biped.controllers.shape_posture_controller import ShapePostureController
from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController
```

- [ ] **Step 2: Add functional support-vector resolver and balance-core construction helper**

Add top-level helper to `scripts/simulate_hierarchical_controller.py`. If the implementation reuses a legacy empirical vector internally, keep that reference inside `resolve_support_feedforward_vector()` only; `build_balance_core_controllers()` and the balance-core runtime path must call the functional helper, not any stage-named helper.

```python
def resolve_support_feedforward_vector() -> np.ndarray:
    support = np.zeros(10, dtype=float)
    support[3] = 8.0
    support[8] = 8.0
    return support


def build_balance_core_controllers(control_dt: float, support_feedforward_vector: np.ndarray):
    return {
        "contact_supervisor": ContactSupervisor(control_dt=control_dt),
        "shape_posture": ShapePostureController(),
        "support_feedforward": SupportFeedforwardController(
            empirical_support=jnp.array(support_feedforward_vector),
            scale=0.5,
            joint_group="knee",
        ),
        "sagittal_wheel_balance": SagittalWheelBalanceController(wheel_torque_sign=1.0),
        "lateral_roll_balance": LateralRollBalanceController(hip_roll_torque_sign=1.0),
        "composer": BalanceCoreTorqueComposer(control_dt=control_dt, max_torque_rate=400.0),
    }
```

This is not gain tuning: it preserves existing defaults until separate tuning is approved. The helper name is functional so balance-core production code does not directly depend on Stage2B/Stage2C/Stage2D naming.

- [ ] **Step 3: Add construction smoke test**

Append to `tests/test_balance_core_mode_isolation.py`:

```python
import numpy as np

from scripts.simulate_hierarchical_controller import build_balance_core_controllers, resolve_support_feedforward_vector


def test_resolve_support_feedforward_vector_uses_functional_name():
    support = resolve_support_feedforward_vector()

    assert support.shape == (10,)


def test_build_balance_core_controllers_returns_functional_components():
    controllers = build_balance_core_controllers(control_dt=0.02, support_feedforward_vector=np.zeros(10))

    assert sorted(controllers) == [
        "composer",
        "contact_supervisor",
        "lateral_roll_balance",
        "sagittal_wheel_balance",
        "shape_posture",
        "support_feedforward",
    ]
```

- [ ] **Step 4: Run mode isolation tests**

Run:

```bash
pytest tests/test_balance_core_mode_isolation.py -v
```

Expected: PASS.

**Acceptance criteria:**
- Balance-core construction exists and returns only functional components.
- No stage-named production component is used in the balance-core construction helper.

---

## Task 13: Route balance-core torque path through composer

**Objective:** Add the actual runtime balance-core branch in the simulation loop so `tau_final` is produced only from the four approved torque sources.

**Files:**
- Modify: `scripts/simulate_hierarchical_controller.py`
- Test: `tests/test_balance_core_mode_isolation.py`
- Test: `tests/test_balance_core_torque_composer.py`

**Required behavior:**
- For `--controller-mode balance-core`, the torque stack is exactly:

```text
tau_shape_posture + tau_support_feedforward + tau_sagittal_wheel_balance + tau_lateral_roll_balance
```

- The balance-core branch sets these legacy torques to zero for telemetry clarity:
  - `tau_wbc_correction`
  - `tau_posture`
  - `tau_leg_position`
  - `tau_hip_roll_centering`
  - `tau_wheel_balance`
  - `tau_inverse_dynamics`
- `tau_final` in balance-core uses composer result.

**Dependencies:** Tasks 8, 10, 12.

**Rollback/safety notes:** Keep the legacy branch unchanged. The balance-core branch should be selected only by `args.controller_mode in {"balance-core", "standing-balance"}`.

- [ ] **Step 1: Add helper for mode check**

Add near validation helper:

```python
def is_balance_core_mode(args) -> bool:
    return args.controller_mode in {"balance-core", "standing-balance"}
```

- [ ] **Step 2: Instantiate balance-core controllers after control timestep is known**

In `main()`, after `control_dt` and empirical feedforward helper are available, add:

```python
balance_core_controllers = None
if is_balance_core_mode(args):
    support_feedforward_vector = resolve_support_feedforward_vector()
    balance_core_controllers = build_balance_core_controllers(
        control_dt=control_dt,
        support_feedforward_vector=support_feedforward_vector,
    )
    print("[BALANCE-CORE] Functional four-source controller stack enabled")
```

- [ ] **Step 3: Add balance-core branch inside the control loop before legacy torque composition**

At the point where the legacy code currently computes `tau_total_raw`, branch as follows:

```python
if is_balance_core_mode(args):
    contact_output = balance_core_controllers["contact_supervisor"].update(
        left_wheel_contact=bool(centroidal_state_control.left_wheel_contact),
        right_wheel_contact=bool(centroidal_state_control.right_wheel_contact),
        contact_force_valid=bool(centroidal_state_control.contact_force_valid),
        left_normal_force_n=float(centroidal_state_control.left_wheel_force),
        right_normal_force_n=float(centroidal_state_control.right_wheel_force),
    )

    tau_shape_posture, shape_diag = balance_core_controllers["shape_posture"].compute(
        q_ref=equilibrium_joint_pos,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        posture_weight=1.0,
        contact_degraded_scale=1.0,
    )
    tau_support_feedforward, support_diag = balance_core_controllers["support_feedforward"].compute()

    wheel_vel_left = float(joint_vel[4])
    wheel_vel_right = float(joint_vel[9])
    wheel_acc_left = (wheel_vel_left - prev_wheel_vel_left) / control_dt
    wheel_acc_right = (wheel_vel_right - prev_wheel_vel_right) / control_dt
    prev_wheel_vel_left = wheel_vel_left
    prev_wheel_vel_right = wheel_vel_right

    cp_error_y_m = float(centroidal_state_control.capture_point[1] - centroidal_state_control.com_pos[1])
    tau_sagittal_wheel_balance, sagittal_diag = balance_core_controllers["sagittal_wheel_balance"].compute(
        pitch_x_rad=float(centroidal_state_control.body_pitch_x),
        pitch_rate_x_rad_s=float(centroidal_state_control.body_pitch_rate_x),
        cp_error_y_m=cp_error_y_m,
        com_vy_m_s=float(centroidal_state_control.com_vel[1]),
        wheel_vel_left_rad_s=wheel_vel_left,
        wheel_vel_right_rad_s=wheel_vel_right,
        outer_position_bias=0.0,
    )
    tau_lateral_roll_balance, lateral_diag = balance_core_controllers["lateral_roll_balance"].compute(
        roll_y_rad=float(centroidal_state_control.body_roll_y),
        roll_rate_y_rad_s=float(centroidal_state_control.body_roll_rate_y),
    )

    torque_limit = jnp.array(mj_model.actuator_ctrlrange[:, 1])
    balance_core_result = balance_core_controllers["composer"].compose(
        tau_shape_posture=tau_shape_posture,
        tau_support_feedforward=tau_support_feedforward,
        tau_sagittal_wheel_balance=tau_sagittal_wheel_balance,
        tau_lateral_roll_balance=tau_lateral_roll_balance,
        tau_prev=tau_prev,
        torque_limit=torque_limit,
    )

    tau_total_raw = balance_core_result.tau_total_raw
    tau_total_clipped = balance_core_result.tau_total_clipped
    tau_smooth = balance_core_result.tau_final
    tau_prev = tau_smooth

    tau_wbc_correction = jnp.zeros(10)
    tau_wbc_scaled = jnp.zeros(10)
    tau_posture = jnp.zeros(10)
    tau_static_posture = tau_shape_posture
    tau_static_feedforward = tau_support_feedforward
    tau_leg_position = jnp.zeros(10)
    tau_hip_roll_centering = jnp.zeros(10)
    tau_wheel_balance = jnp.zeros(10)
    tau_inverse_dynamics = jnp.zeros(10)
else:
    # Existing legacy torque composition remains here.
```

Before the loop, initialize wheel velocity memory:

```python
prev_wheel_vel_left = 0.0
prev_wheel_vel_right = 0.0
```

- [ ] **Step 4: Add no hidden legacy torque unit test by extracting a helper if necessary**

If the branch is too hard to unit test directly, add this small top-level helper:

```python
def zero_legacy_torque_sources_for_balance_core():
    return {
        "tau_wbc_correction": jnp.zeros(10),
        "tau_wbc_scaled": jnp.zeros(10),
        "tau_posture": jnp.zeros(10),
        "tau_leg_position": jnp.zeros(10),
        "tau_hip_roll_centering": jnp.zeros(10),
        "tau_wheel_balance": jnp.zeros(10),
        "tau_inverse_dynamics": jnp.zeros(10),
    }
```

Append this test to `tests/test_balance_core_mode_isolation.py`:

```python
from scripts.simulate_hierarchical_controller import zero_legacy_torque_sources_for_balance_core


def test_balance_core_legacy_torque_sources_are_zeroed():
    sources = zero_legacy_torque_sources_for_balance_core()

    assert sorted(sources) == [
        "tau_hip_roll_centering",
        "tau_inverse_dynamics",
        "tau_leg_position",
        "tau_posture",
        "tau_wbc_correction",
        "tau_wbc_scaled",
        "tau_wheel_balance",
    ]
    for value in sources.values():
        assert jnp.allclose(value, jnp.zeros(10))
```

Use this helper in the runtime branch instead of manually constructing the same dictionary.

- [ ] **Step 5: Run mode isolation and composer tests**

Run:

```bash
pytest tests/test_balance_core_mode_isolation.py tests/test_balance_core_torque_composer.py -v
```

Expected: PASS.

**Acceptance criteria:**
- Balance-core `tau_final` comes from composer only.
- Legacy torques are zero in balance-core.
- Existing legacy branch remains available outside balance-core.

---

## Task 14: Add balance-core telemetry logging to simulation

**Objective:** Ensure CSV telemetry includes all required balance-core state and torque fields.

**Files:**
- Modify: `scripts/simulate_hierarchical_controller.py`
- Test: `tests/test_balance_core_telemetry_schema.py`

**Required behavior:**
- Required state fields are appended each control tick in balance-core.
- Required torque fields are appended from `BalanceCoreTorqueResult.telemetry`.
- Robot-frame names use `pitch_x` and `roll_y`, not ambiguous world Euler names.

**Dependencies:** Tasks 8, 11, 13.

**Rollback/safety notes:** Telemetry additions should not affect torque computation.

- [ ] **Step 1: Add helper to append telemetry**

Add to `scripts/simulate_hierarchical_controller.py`:

```python
def append_balance_core_telemetry(
    telemetry: dict,
    result,
    centroidal_state,
    contact_output,
    cp_error_y_m: float,
    wheel_vel_left_rad_s: float,
    wheel_vel_right_rad_s: float,
    wheel_acc_left_rad_s2: float,
    wheel_acc_right_rad_s2: float,
):
    wheel_vel_mean = 0.5 * (wheel_vel_left_rad_s + wheel_vel_right_rad_s)
    wheel_acc_mean = 0.5 * (wheel_acc_left_rad_s2 + wheel_acc_right_rad_s2)
    state_values = {
        "pitch_x_rad": float(centroidal_state.body_pitch_x),
        "roll_y_rad": float(centroidal_state.body_roll_y),
        "yaw_z_rad": float(centroidal_state.body_yaw_z),
        "pitch_rate_x_rad_s": float(centroidal_state.body_pitch_rate_x),
        "roll_rate_y_rad_s": float(centroidal_state.body_roll_rate_y),
        "yaw_rate_z_rad_s": float(centroidal_state.body_yaw_rate_z),
        "com_x_m": float(centroidal_state.com_pos[0]),
        "com_y_m": float(centroidal_state.com_pos[1]),
        "com_z_m": float(centroidal_state.com_pos[2]),
        "com_vx_m_s": float(centroidal_state.com_vel[0]),
        "com_vy_m_s": float(centroidal_state.com_vel[1]),
        "com_vz_m_s": float(centroidal_state.com_vel[2]),
        "cp_x_m": float(centroidal_state.capture_point[0]),
        "cp_y_m": float(centroidal_state.capture_point[1]),
        "cp_error_y_m": float(cp_error_y_m),
        "wheel_vel_left_rad_s": float(wheel_vel_left_rad_s),
        "wheel_vel_right_rad_s": float(wheel_vel_right_rad_s),
        "wheel_vel_mean_rad_s": float(wheel_vel_mean),
        "wheel_acc_left_rad_s2": float(wheel_acc_left_rad_s2),
        "wheel_acc_right_rad_s2": float(wheel_acc_right_rad_s2),
        "wheel_acc_mean_rad_s2": float(wheel_acc_mean),
        "left_wheel_contact": bool(contact_output.left_wheel_contact),
        "right_wheel_contact": bool(contact_output.right_wheel_contact),
        "contact_supervisor_state": contact_output.state.value,
        "contact_previous_state": contact_output.previous_state.value if contact_output.previous_state is not None else "none",
        "contact_duration_s": float(contact_output.contact_duration_s),
        "contact_transition_event": contact_output.transition_event,
        "contact_force_valid": bool(contact_output.contact_force_valid),
        "contact_recovery_hook_fields": str(contact_output.recovery_hook_fields),
    }
    for name, value in state_values.items():
        telemetry[name].append(value)

    for name, value in result.telemetry.items():
        if isinstance(value, tuple):
            telemetry[name].append(",".join(str(v) for v in value))
        else:
            telemetry[name].append(value)
```

- [ ] **Step 2: Add test for telemetry append helper**

Append to `tests/test_balance_core_telemetry_schema.py`:

```python
import jax.numpy as jnp
from types import SimpleNamespace

from wheeled_biped.controllers.balance_core_types import (
    ContactSupervisorOutput,
    ContactSupervisorState,
    make_balance_core_telemetry_columns,
)
from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer
from scripts.simulate_hierarchical_controller import append_balance_core_telemetry


def test_append_balance_core_telemetry_populates_required_fields():
    telemetry = make_balance_core_telemetry_columns()
    result = BalanceCoreTorqueComposer(control_dt=0.02).compose(
        tau_shape_posture=jnp.zeros(10),
        tau_support_feedforward=jnp.zeros(10),
        tau_sagittal_wheel_balance=jnp.zeros(10),
        tau_lateral_roll_balance=jnp.zeros(10),
        tau_prev=jnp.zeros(10),
        torque_limit=jnp.ones(10) * 10.0,
    )
    centroidal_state = SimpleNamespace(
        body_pitch_x=0.1,
        body_roll_y=0.2,
        body_yaw_z=0.3,
        body_pitch_rate_x=0.4,
        body_roll_rate_y=0.5,
        body_yaw_rate_z=0.6,
        com_pos=jnp.array([1.0, 2.0, 3.0]),
        com_vel=jnp.array([4.0, 5.0, 6.0]),
        capture_point=jnp.array([7.0, 8.0]),
    )
    contact = ContactSupervisorOutput(
        state=ContactSupervisorState.DOUBLE_CONTACT,
        previous_state=None,
        left_wheel_contact=True,
        right_wheel_contact=True,
        contact_force_valid=True,
        left_normal_force_n=40.0,
        right_normal_force_n=41.0,
        contact_duration_s=0.0,
        transition_event="initial_double_contact",
        recovery_hook_fields={},
    )

    append_balance_core_telemetry(
        telemetry,
        result,
        centroidal_state,
        contact,
        cp_error_y_m=0.7,
        wheel_vel_left_rad_s=1.1,
        wheel_vel_right_rad_s=1.3,
        wheel_acc_left_rad_s2=2.1,
        wheel_acc_right_rad_s2=2.3,
    )

    assert telemetry["pitch_x_rad"] == [0.1]
    assert telemetry["roll_y_rad"] == [0.2]
    assert telemetry["contact_supervisor_state"] == ["double_contact"]
    assert telemetry["contact_previous_state"] == ["none"]
    assert telemetry["contact_transition_event"] == ["initial_double_contact"]
    assert telemetry["wheel_vel_mean_rad_s"] == [1.2]
    assert telemetry["wheel_acc_mean_rad_s2"] == [2.2]
    assert len(telemetry["tau_final_per_joint"]) == 1
```

- [ ] **Step 3: Initialize balance-core telemetry columns in main telemetry dict**

Where telemetry is initialized, add:

```python
if is_balance_core_mode(args):
    for key, values in make_balance_core_telemetry_columns().items():
        telemetry.setdefault(key, values)
```

- [ ] **Step 4: Append balance-core telemetry inside balance-core branch**

After `balance_core_result` is computed in the balance-core branch:

```python
append_balance_core_telemetry(
    telemetry,
    balance_core_result,
    centroidal_state_control,
    contact_output,
    cp_error_y_m=cp_error_y_m,
    wheel_vel_left_rad_s=wheel_vel_left,
    wheel_vel_right_rad_s=wheel_vel_right,
    wheel_acc_left_rad_s2=wheel_acc_left,
    wheel_acc_right_rad_s2=wheel_acc_right,
)
```

- [ ] **Step 5: Run telemetry tests**

Run:

```bash
pytest tests/test_balance_core_telemetry_schema.py -v
```

Expected: PASS.

**Acceptance criteria:**
- Required telemetry fields are present and appended.
- State naming is robot-frame explicit.

---

## Task 15: Prevent duplicate telemetry appends in balance-core branch

**Objective:** Ensure legacy telemetry appends do not overwrite or double-append balance-core telemetry fields.

**Files:**
- Modify: `scripts/simulate_hierarchical_controller.py`
- Test: `tests/test_balance_core_telemetry_schema.py`

**Required behavior:**
- Balance-core required fields are appended once per control tick.
- Legacy fields may still be appended for backwards-compatible CSVs, but they must reflect zeroed legacy torques in balance-core.
- Balance-core required torque fields must come from `BalanceCoreTorqueResult.telemetry`.

**Dependencies:** Task 14.

**Rollback/safety notes:** Avoid broad telemetry refactor. Use a small branch around balance-core-specific fields.

- [ ] **Step 1: Add helper test for required field lengths**

Append to `tests/test_balance_core_telemetry_schema.py`:

```python
from wheeled_biped.controllers.balance_core_types import (
    BALANCE_CORE_REQUIRED_STATE_TELEMETRY,
    BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY,
)


def test_balance_core_required_telemetry_fields_have_equal_lengths_after_append():
    telemetry = make_balance_core_telemetry_columns()
    result = BalanceCoreTorqueComposer(control_dt=0.02).compose(
        tau_shape_posture=jnp.zeros(10),
        tau_support_feedforward=jnp.zeros(10),
        tau_sagittal_wheel_balance=jnp.zeros(10),
        tau_lateral_roll_balance=jnp.zeros(10),
        tau_prev=jnp.zeros(10),
        torque_limit=jnp.ones(10) * 10.0,
    )
    centroidal_state = SimpleNamespace(
        body_pitch_x=0.0,
        body_roll_y=0.0,
        body_yaw_z=0.0,
        body_pitch_rate_x=0.0,
        body_roll_rate_y=0.0,
        body_yaw_rate_z=0.0,
        com_pos=jnp.zeros(3),
        com_vel=jnp.zeros(3),
        capture_point=jnp.zeros(2),
    )
    contact = ContactSupervisorOutput(
        state=ContactSupervisorState.FLIGHT_OR_NO_CONTACT,
        previous_state=None,
        left_wheel_contact=False,
        right_wheel_contact=False,
        contact_force_valid=False,
        left_normal_force_n=0.0,
        right_normal_force_n=0.0,
        contact_duration_s=0.0,
        transition_event="initial_flight_or_no_contact",
        recovery_hook_fields={},
    )

    append_balance_core_telemetry(telemetry, result, centroidal_state, contact, 0.0, 0.0, 0.0, 0.0, 0.0)

    for name in BALANCE_CORE_REQUIRED_STATE_TELEMETRY + BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY:
        assert len(telemetry[name]) == 1
```

- [ ] **Step 2: Run telemetry tests**

Run:

```bash
pytest tests/test_balance_core_telemetry_schema.py -v
```

Expected: PASS.

- [ ] **Step 3: Guard legacy-specific stage telemetry names**

In the telemetry append section, keep existing legacy telemetry but ensure balance-core names are not overwritten by legacy aliases. For balance-core, append functional source names from `append_balance_core_telemetry`; for legacy, keep existing stage/static fields.

Use this pattern:

```python
if is_balance_core_mode(args):
    telemetry["tau_wbc_correction"].append(",".join(f"{x:.4f}" for x in np.zeros(10)))
    telemetry["tau_static_feedforward"].append(",".join(f"{x:.4f}" for x in np.array(tau_support_feedforward)))
    telemetry["tau_static_posture"].append(",".join(f"{x:.4f}" for x in np.array(tau_shape_posture)))
else:
    telemetry["tau_wbc_correction"].append(",".join(f"{x:.4f}" for x in np.array(tau_wbc_correction)))
    telemetry["tau_static_feedforward"].append(",".join(f"{x:.4f}" for x in np.array(tau_static_feedforward)))
    telemetry["tau_static_posture"].append(",".join(f"{x:.4f}" for x in np.array(tau_static_posture)))
```

Do not rename existing legacy telemetry in this task; only ensure balance-core required telemetry is present and correct.

- [ ] **Step 4: Run telemetry and mode tests**

Run:

```bash
pytest tests/test_balance_core_telemetry_schema.py tests/test_balance_core_mode_isolation.py -v
```

Expected: PASS.

**Acceptance criteria:**
- Required balance-core fields are append-safe.
- Legacy compatibility fields do not hide balance-core telemetry.

---

## Task 16: Add WBC-off-by-default tests

**Objective:** Prove `IntegratedWBC` torque is not part of balance-core `tau_final`.

**Files:**
- Modify: `tests/test_balance_core_mode_isolation.py`
- Modify: `scripts/simulate_hierarchical_controller.py` if helper extraction is needed

**Required behavior:**
- Balance-core composer has no WBC input.
- Runtime balance-core branch zeroes WBC-applied torque telemetry.
- WBC remains importable and unmodified.

**Dependencies:** Tasks 8 and 13.

**Rollback/safety notes:** Do not delete WBC files or alter WBC behavior.

- [ ] **Step 1: Add WBC preservation and off-default test**

Append to `tests/test_balance_core_mode_isolation.py`:

```python
import inspect

from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer


def test_integrated_wbc_remains_available_but_composer_has_no_wbc_input():
    assert IntegratedWBC is not None
    signature = inspect.signature(BalanceCoreTorqueComposer.compose)
    assert "tau_wbc" not in signature.parameters
    assert "tau_wbc_correction" not in signature.parameters
    assert "tau_inverse_dynamics" not in signature.parameters
```

- [ ] **Step 2: Run WBC isolation test**

Run:

```bash
pytest tests/test_balance_core_mode_isolation.py::test_integrated_wbc_remains_available_but_composer_has_no_wbc_input -v
```

Expected: PASS.

**Acceptance criteria:**
- WBC is preserved.
- Balance-core torque composition has no WBC input.

---

## Task 17: Add incompatible experimental sagittal controller guard

**Objective:** Ensure balance-core cannot silently run simultaneous experimental sagittal controllers or any experimental sagittal controller by default.

**Files:**
- Modify: `tests/test_balance_core_mode_isolation.py`
- Modify: `scripts/simulate_hierarchical_controller.py`

**Required behavior:**
- `--controller-mode balance-core` rejects all current experimental sagittal flags.
- Legacy mode still enforces existing mutual exclusion for experimental sagittal controllers.

**Dependencies:** Task 10.

**Rollback/safety notes:** No changes to experimental controllers themselves.

- [ ] **Step 1: Add explicit simultaneous sagittal rejection test**

Append to `tests/test_balance_core_mode_isolation.py`:

```python

def test_balance_core_rejects_simultaneous_experimental_sagittal_controllers():
    args = _args(
        enable_stage2b_sagittal_wheel=True,
        enable_stage2c_sagittal_state_feedback=True,
        enable_stage2d_sagittal_lqr=True,
    )
    with pytest.raises(ValueError, match="enable_stage2b_sagittal_wheel"):
        validate_balance_core_mode_args(args)
```

- [ ] **Step 2: Run mode isolation tests**

Run:

```bash
pytest tests/test_balance_core_mode_isolation.py -v
```

Expected: PASS if Task 10 validation already rejects all three flags.

**Acceptance criteria:**
- Balance-core cannot enable any current experimental sagittal controller.
- Functional `SagittalWheelBalanceController` is the only balance-core wheel owner.

---

## Task 18: Add safety and finite-output tests

**Objective:** Verify safety limits, rate limits, and finite torques under nominal and degraded-contact inputs.

**Files:**
- Modify: `tests/test_balance_core_torque_composer.py`
- Modify: `tests/test_balance_core_components.py`

**Required behavior:**
- Final torque respects actuator limits and rate limits.
- Component outputs are finite.
- Contact-degraded state does not create non-finite torque.

**Dependencies:** Tasks 3-8.

**Rollback/safety notes:** Unit tests only. Do not change gains to pass these tests.

- [ ] **Step 1: Add finite output tests**

Append to `tests/test_balance_core_components.py`:

```python

def test_balance_core_component_outputs_are_finite_for_nominal_inputs():
    shape = ShapePostureController()
    support = SupportFeedforwardController(jnp.zeros(10))
    sagittal = SagittalWheelBalanceController()
    lateral = LateralRollBalanceController()

    tau_shape, _ = shape.compute(jnp.zeros(10), jnp.ones(10) * 0.01, jnp.zeros(10))
    tau_support, _ = support.compute()
    tau_sagittal, _ = sagittal.compute(0.01, 0.02, 0.01, 0.02, 0.1, 0.1)
    tau_lateral, _ = lateral.compute(0.01, 0.02)

    for tau in [tau_shape, tau_support, tau_sagittal, tau_lateral]:
        assert jnp.all(jnp.isfinite(tau))
```

Append to `tests/test_balance_core_torque_composer.py`:

```python

def test_composer_final_torque_respects_rate_limited_step():
    composer = BalanceCoreTorqueComposer(control_dt=0.02, max_torque_rate=50.0)
    result = composer.compose(
        tau_shape_posture=jnp.array([0, 0, 100, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        tau_support_feedforward=jnp.zeros(10),
        tau_sagittal_wheel_balance=jnp.zeros(10),
        tau_lateral_roll_balance=jnp.zeros(10),
        tau_prev=jnp.zeros(10),
        torque_limit=jnp.ones(10) * 200.0,
    )

    assert result.tau_final[2] == 1.0
    assert result.torque_rate_saturation_mask[2] == True
```

- [ ] **Step 2: Run safety tests**

Run:

```bash
pytest tests/test_balance_core_components.py tests/test_balance_core_torque_composer.py -v
```

Expected: PASS.

**Acceptance criteria:**
- Safety behavior is covered at unit-test level.
- No tuning is introduced to satisfy the tests.

---

## Task 19: Add short balance-core smoke simulation command support

**Objective:** Verify the script can run `--controller-mode balance-core` for a short headless simulation and produce telemetry with required fields.

**Files:**
- Modify: `tests/test_balance_core_mode_isolation.py` only if subprocess test is affordable
- No test file change if this remains a manual validation command due runtime cost

**Required behavior:**
- Command runs without incompatible flag errors.
- Telemetry CSV contains required balance-core fields.
- Short smoke is not treated as proof of balance performance.

**Dependencies:** Tasks 10-15.

**Rollback/safety notes:** Keep smoke short. Do not tune gains based on this task.

- [ ] **Step 1: Run short smoke simulation manually**

Run:

```bash
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --steps 50
```

Expected: process exits 0, prints `[BALANCE-CORE] Functional four-source controller stack enabled`, and writes telemetry under `outputs/hierarchical_controller_sim/` using the existing script behavior.

- [ ] **Step 2: Check telemetry columns manually with Python**

Run:

```bash
python - <<'PY'
import csv
from pathlib import Path

paths = sorted(Path('outputs/hierarchical_controller_sim').glob('*.csv'), key=lambda p: p.stat().st_mtime)
assert paths, 'no telemetry csv found'
path = paths[-1]
with path.open(newline='') as f:
    reader = csv.DictReader(f)
    fields = set(reader.fieldnames or [])
required = {
    'pitch_x_rad', 'roll_y_rad', 'yaw_z_rad',
    'tau_shape_posture_per_joint', 'tau_support_feedforward_per_joint',
    'tau_sagittal_wheel_balance_per_joint', 'tau_lateral_roll_balance_per_joint',
    'tau_final_per_joint', 'active_torque_owner_per_joint',
    'ownership_violation_count', 'contact_supervisor_state', 'contact_force_valid',
}
missing = sorted(required - fields)
assert not missing, f'missing fields: {missing}'
print(f'checked {path}')
PY
```

Expected: prints `checked <path>` and no assertion failure.

**Acceptance criteria:**
- Short smoke run starts and exits cleanly.
- Telemetry includes required balance-core fields.

---

## Task 20: Add separated architecture and performance validation commands

**Objective:** Validate balance-core structure independently from balance performance so architecture consolidation is not failed only because the untuned controller terminates early under physics.

**Files:**
- No code changes expected unless validation reveals an architectural defect.

**Required behavior:**
- Architecture validation checks that the script runs without crash, required telemetry exists, ownership violations are zero, WBC is off, hidden legacy torque is zero, and torques are finite.
- Architecture validation does not require 500 rows or bounded pitch/roll.
- Performance validation is separate and checks 500-step survival plus bounded pitch, roll, wheel velocity, and knees.
- Balance-core architecture implementation can be considered structurally complete after architecture validation passes.
- Balance-core controller performance is complete only after performance validation passes.

**Dependencies:** Task 19.

**Rollback/safety notes:** If architecture validation fails, fix architecture bugs: hidden torque, missing telemetry, NaNs, ownership leaks, WBC leakage, or script errors. If performance validation fails due physics, do not tune gains under this plan; record it as controller-performance work outside architecture consolidation.

- [ ] **Step 1: Run architecture validation smoke**

Run:

```bash
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --steps 50
```

Expected: process exits 0 and writes telemetry CSV. Early physical instability after producing fewer than 50 telemetry rows is acceptable only if the script exits cleanly and the telemetry checker below can inspect at least one balance-core row.

- [ ] **Step 2: Run architecture telemetry checker**

Run:

```bash
python - <<'PY'
import csv
import math
from pathlib import Path

paths = sorted(Path('outputs/hierarchical_controller_sim').glob('*.csv'), key=lambda p: p.stat().st_mtime)
assert paths, 'no telemetry csv found'
path = paths[-1]
rows = list(csv.DictReader(path.open(newline='')))
assert rows, 'no telemetry rows found'

required = {
    'pitch_x_rad', 'roll_y_rad', 'tau_shape_posture_per_joint',
    'tau_support_feedforward_per_joint', 'tau_sagittal_wheel_balance_per_joint',
    'tau_lateral_roll_balance_per_joint', 'tau_final_per_joint',
    'active_torque_owner_per_joint', 'ownership_violation_count',
    'contact_supervisor_state', 'contact_force_valid',
}
missing = sorted(required - set(rows[0]))
assert not missing, f'missing fields: {missing}'

violations = [int(float(r['ownership_violation_count'])) for r in rows]
assert max(violations) == 0, f'ownership violations present: {max(violations)}'

for legacy_field in ['tau_wbc_correction', 'tau_inverse_dynamics', 'tau_wheel_balance']:
    if legacy_field in rows[0]:
        for row in rows:
            values = [float(v) for v in row[legacy_field].split(',') if v]
            assert all(abs(v) < 1e-9 for v in values), f'{legacy_field} leaked nonzero torque'

for field in ['tau_shape_posture_per_joint', 'tau_support_feedforward_per_joint', 'tau_sagittal_wheel_balance_per_joint', 'tau_lateral_roll_balance_per_joint', 'tau_final_per_joint']:
    for row in rows:
        values = [float(v) for v in row[field].split(',') if v]
        assert values, f'{field} had no values'
        assert all(math.isfinite(v) for v in values), f'{field} contains non-finite values'

print(f'architecture checked {path} rows={len(rows)}')
PY
```

Expected: prints `architecture checked <path> rows=<n>` and no assertion failure.

- [ ] **Step 3: Run separate performance validation only after architecture validation passes**

Run:

```bash
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --steps 500
```

Expected for performance completion: process exits 0 and writes at least 500 telemetry rows.

- [ ] **Step 4: Run performance telemetry checker**

Run:

```bash
python - <<'PY'
import csv
import math
from pathlib import Path

paths = sorted(Path('outputs/hierarchical_controller_sim').glob('*.csv'), key=lambda p: p.stat().st_mtime)
assert paths, 'no telemetry csv found'
path = paths[-1]
rows = list(csv.DictReader(path.open(newline='')))
assert len(rows) >= 500, f'expected at least 500 rows, got {len(rows)}'

for field in ['pitch_x_rad', 'roll_y_rad', 'wheel_vel_mean_rad_s', 'ownership_violation_count']:
    assert field in rows[0], f'missing {field}'

pitch = [abs(float(r['pitch_x_rad'])) for r in rows]
roll = [abs(float(r['roll_y_rad'])) for r in rows]
wheel = [abs(float(r['wheel_vel_mean_rad_s'])) for r in rows]
violations = [int(float(r['ownership_violation_count'])) for r in rows]
assert all(math.isfinite(v) for v in pitch + roll + wheel), 'non-finite balance telemetry'
assert max(violations) == 0, f'ownership violations present: {max(violations)}'
assert max(pitch) < 1.5, f'pitch_x exceeded sanity bound: {max(pitch)}'
assert max(roll) < 1.5, f'roll_y exceeded sanity bound: {max(roll)}'
assert max(wheel) < 200.0, f'wheel velocity exceeded sanity bound: {max(wheel)}'
print(f'performance checked {path}')
PY
```

Expected for performance completion: prints `performance checked <path>` and no assertion failure.

**Acceptance criteria:**
- Structural architecture acceptance requires only the architecture validation smoke and architecture telemetry checker.
- Performance acceptance additionally requires the 500-step run and performance telemetry checker.

---

## Task 21: Run focused and full verification suite

**Objective:** Verify balance-core work without relying only on smoke simulations.

**Files:**
- No code changes expected.

**Required behavior:**
- Unit tests pass.
- Mode isolation tests pass.
- Ownership tests pass.
- Telemetry tests pass.
- Architecture validation passes for structural completion.
- Performance validation is reported separately and is not required to claim architecture consolidation is structurally complete.

**Dependencies:** Tasks 1-20.

**Rollback/safety notes:** If a broad existing test fails outside balance-core, inspect whether this plan caused it before changing unrelated behavior.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
pytest \
  tests/test_torque_ownership_validator.py \
  tests/test_contact_supervisor.py \
  tests/test_balance_core_components.py \
  tests/test_balance_core_torque_composer.py \
  tests/test_balance_core_mode_isolation.py \
  tests/test_balance_core_telemetry_schema.py \
  -v
```

Expected: all selected tests PASS.

- [ ] **Step 2: Run existing non-slow controller checks**

Run:

```bash
pytest tests/ --ignore=tests/test_env.py -m "not slow" -v
```

Expected: PASS or identify failures caused by balance-core integration. Do not mark complete with unresolved failures.

- [ ] **Step 3: Run short smoke simulation**

Run:

```bash
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --steps 50
```

Expected: exits 0.

- [ ] **Step 4: Run architecture validation checker**

Run the architecture telemetry checker from Task 20.

Expected: exits 0, confirms required telemetry, confirms `ownership_violation_count = 0`, confirms WBC/legacy torque leakage is zero, and confirms balance-core torques are finite.

- [ ] **Step 5: Run separate performance validation only for controller-performance acceptance**

Run:

```bash
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --steps 500
```

Expected for performance acceptance: exits 0 and the performance telemetry checker from Task 20 passes. If this fails due physical instability while architecture validation passes, structural architecture consolidation can still be accepted; balance performance remains pending.

**Acceptance criteria:**
- All focused tests pass.
- Full non-slow test suite either passes or any unrelated pre-existing failures are documented with evidence.
- Architecture validation passes without tuning gains.
- Performance validation is reported separately from architecture completion.

---

## Balance-core implementation acceptance criteria

The structural architecture implementation is complete only when all criteria below are verified with fresh commands:

- `--controller-mode balance-core` exists.
- Clean torque stack contains exactly:

```text
tau_shape_posture
+ tau_support_feedforward
+ tau_sagittal_wheel_balance
+ tau_lateral_roll_balance
```

- No hidden legacy torque contributes to `tau_final` in balance-core.
- Balance-core production code does not directly call Stage2B/Stage2C/Stage2D-named helpers.
- WBC is off by default in balance-core and remains preserved in the repository.
- Posture is compliant, not rigidly assigned to hip-roll or wheels, and exposes softening inputs.
- Wheel torque controls sagittal balance through `SagittalWheelBalanceController` with explicit `wheel_torque_sign`.
- Hip-roll controls `roll_y` through `LateralRollBalanceController` with explicit `hip_roll_torque_sign`.
- Contact state is supervised and logged, including previous state, duration, transition event, force validity, and future recovery hook fields.
- Non-contact wheels never receive fake force from `ContactSupervisor`.
- Ownership tests pass, including unowned joint rejection, duplicate source name rejection, exclusive owner conflict rejection, and allowed shape/support sharing.
- Mode isolation tests pass.
- Telemetry tests pass.
- Safety and torque-rate tests pass.
- Architecture validation passes: script runs without crash, telemetry fields exist, `ownership_violation_count = 0`, WBC/legacy torque leakage is zero, and torques are finite.
- The implementation does not introduce new stage-named production components.
- No gains are tuned as part of architecture consolidation.

Balance-core controller performance is complete only when the separate performance validation passes:

- 500-step nominal standing-balance run produces at least 500 telemetry rows.
- `pitch_x` and `roll_y` remain bounded.
- Wheel velocity remains bounded.
- Knees remain stable in telemetry sanity checks.
- Ownership violations remain zero during the performance run.

## Required validation commands summary

Focused structural tests:

```bash
pytest tests/test_torque_ownership_validator.py -v
pytest tests/test_contact_supervisor.py -v
pytest tests/test_balance_core_components.py -v
pytest tests/test_balance_core_torque_composer.py -v
pytest tests/test_balance_core_mode_isolation.py -v
pytest tests/test_balance_core_telemetry_schema.py -v
pytest tests/test_torque_ownership_validator.py tests/test_contact_supervisor.py tests/test_balance_core_components.py tests/test_balance_core_torque_composer.py tests/test_balance_core_mode_isolation.py tests/test_balance_core_telemetry_schema.py -v
pytest tests/ --ignore=tests/test_env.py -m "not slow" -v
```

Architecture validation for structural completion:

```bash
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --steps 50
# then run the architecture telemetry checker from Task 20
```

Separate performance validation, not required for structural architecture completion:

```bash
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --steps 500
# then run the performance telemetry checker from Task 20
```

## Plan self-review checklist

- Spec coverage: tasks cover mode isolation, four-source torque stack, component refactors, contact supervisor, ownership validation, telemetry, WBC preservation, tests, validation commands, and acceptance criteria.
- Naming check: new production names are functional: `balance-core`, `ShapePostureController`, `SupportFeedforwardController`, `SagittalWheelBalanceController`, `LateralRollBalanceController`, `ContactSupervisor`, `TorqueOwnershipValidator`, and `BalanceCoreTorqueComposer`.
- Stage-name containment: existing stage-named files and flags are referenced only as legacy or temporary experimental modules/flags. No new production file/class/function uses a stage-based name.
- Scope check: plan does not tune gains, delete WBC, make WBC default, implement full contact recovery, implement RL, or add another controller stage.
- Verification check: plan includes focused unit tests, mode isolation tests, telemetry tests, architecture validation for structural completion, and separate 500-step performance validation.
