# Full WBC C1 Correctness Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the hierarchical whole-body controller path so orientation, state mapping, contact sensing, desired wrench, force distribution, Jacobian torque mapping, torque allocation, and telemetry are physically truthful before gain tuning.

**Architecture:** Keep the existing controller modules, but make each layer explicit and tested: MuJoCo state/contact sensing -> roll/pitch convention -> desired wrench -> contact-aware force distribution -> Jacobian torque mapping -> integrated WBC torque application -> telemetry verification. The simulation script becomes an experiment runner; WBC torque becomes the primary control path.

**Tech Stack:** Python, MuJoCo Python API, JAX/JAX NumPy, pytest, existing wheeled_biped controller modules.

---

## File structure

- Modify `wheeled_biped/controllers/orientation_utils.py`: canonical `roll=X`, `pitch=Y`, `yaw=Z` extraction from quaternion, rotation matrix, and gravity vector.
- Modify `wheeled_biped/controllers/centroidal_state_estimator.py`: resolve contact geom IDs by name, expose measured wheel contact forces from `mj_contactForce()`, and include base angular velocity/orientation fields.
- Modify `wheeled_biped/controllers/centroidal_wrench_computer.py`: consume explicit state fields or state-like data, use actual roll/pitch convention, compute gravity compensation from configured/model mass.
- Modify `wheeled_biped/controllers/simple_force_distributor.py`: make distribution contact-aware and return diagnostics.
- Modify `wheeled_biped/controllers/contact_jacobian.py`: keep MuJoCo Jacobian path, expose sign-controlled mapping, and support diagnostics.
- Modify `wheeled_biped/controllers/integrated_wbc.py`: orchestrate corrected wrench -> distribution -> torque mapping -> clipping with diagnostics.
- Modify `scripts/simulate_hierarchical_controller.py`: apply `tau_wbc` as primary torque, compose secondary torque explicitly, and log actual contact-force telemetry.
- Add/modify tests:
  - `tests/test_orientation_utils.py`
  - `tests/test_centroidal_state_estimator.py`
  - `tests/test_centroidal_wrench_computer.py`
  - `tests/test_simple_force_distributor.py`
  - `tests/test_contact_jacobian.py`
  - `tests/test_integrated_wbc.py`

---

### Task 1: Canonical orientation convention

**Files:**
- Modify: `wheeled_biped/controllers/orientation_utils.py`
- Test: `tests/test_orientation_utils.py`

- [ ] **Step 1: Write failing orientation tests**

Create or replace focused tests in `tests/test_orientation_utils.py`:

```python
import math

import jax.numpy as jnp
import numpy as np

from wheeled_biped.controllers.orientation_utils import (
    compute_orientation_from_gravity,
    compute_orientation_from_quaternion,
)


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    return np.array([
        math.cos(half),
        axis[0] * math.sin(half),
        axis[1] * math.sin(half),
        axis[2] * math.sin(half),
    ])


def gravity_body_from_quat(quat):
    w, x, y, z = quat
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return rot.T @ np.array([0.0, 0.0, -9.81])


def test_identity_quaternion_is_level():
    roll, pitch, yaw = compute_orientation_from_quaternion(np.array([1.0, 0.0, 0.0, 0.0]))
    assert abs(roll) < 1e-9
    assert abs(pitch) < 1e-9
    assert abs(yaw) < 1e-9


def test_x_axis_rotation_is_roll_only():
    quat = quat_from_axis_angle([1.0, 0.0, 0.0], 0.2)
    roll, pitch, yaw = compute_orientation_from_quaternion(quat)
    assert abs(roll - 0.2) < 1e-6
    assert abs(pitch) < 1e-6
    assert abs(yaw) < 1e-6


def test_y_axis_rotation_is_pitch_only():
    quat = quat_from_axis_angle([0.0, 1.0, 0.0], -0.15)
    roll, pitch, yaw = compute_orientation_from_quaternion(quat)
    assert abs(roll) < 1e-6
    assert abs(pitch + 0.15) < 1e-6
    assert abs(yaw) < 1e-6


def test_gravity_and_quaternion_paths_agree_for_small_angles():
    quat = quat_from_axis_angle([1.0, 0.0, 0.0], 0.08)
    gravity_body = gravity_body_from_quat(quat)
    roll_q, pitch_q, _ = compute_orientation_from_quaternion(quat)
    roll_g, pitch_g = compute_orientation_from_gravity(jnp.array(gravity_body))
    assert abs(float(roll_g) - roll_q) < 1e-3
    assert abs(float(pitch_g) - pitch_q) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails before implementation**

Run:

```bash
pytest tests/test_orientation_utils.py -v
```

Expected before implementation: at least `test_x_axis_rotation_is_roll_only`, `test_y_axis_rotation_is_pitch_only`, or gravity/quaternion agreement fails under the old convention.

- [ ] **Step 3: Implement canonical orientation helpers**

Update `wheeled_biped/controllers/orientation_utils.py` so the public API returns roll first for quaternion and gravity helpers:

```python
def compute_orientation_from_gravity(gravity_body: Array) -> tuple[float, float]:
    """Compute roll and pitch from gravity vector in body frame.

    Convention: roll is rotation about X, pitch is rotation about Y.
    Gravity in body frame is [0, 0, -g] when upright.
    """
    gx, gy, gz = gravity_body[0], gravity_body[1], gravity_body[2]
    roll = jnp.arctan2(-gy, -gz)
    pitch = jnp.arctan2(gx, -gz)
    return roll, pitch


def compute_orientation_from_quaternion(quat: np.ndarray) -> tuple[float, float, float]:
    """Compute roll, pitch, yaw from quaternion [w, x, y, z]."""
    w, x, y, z = quat

    roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    return roll, pitch, yaw
```

- [ ] **Step 4: Update call sites expecting old tuple order**

Search:

```bash
python - <<'PY'
from pathlib import Path
for path in Path('.').rglob('*.py'):
    text = path.read_text(encoding='utf-8', errors='ignore')
    if 'compute_orientation_from_gravity' in text or 'compute_orientation_from_quaternion' in text:
        print(path)
PY
```

For each call site, ensure unpacking matches:

```python
roll, pitch = compute_orientation_from_gravity(gravity_body)
roll, pitch, yaw = compute_orientation_from_quaternion(quat)
```

- [ ] **Step 5: Run orientation tests**

Run:

```bash
pytest tests/test_orientation_utils.py -v
```

Expected: all tests pass.

---

### Task 2: Contact and state sensing truth

**Files:**
- Modify: `wheeled_biped/controllers/centroidal_state_estimator.py`
- Test: `tests/test_centroidal_state_estimator.py`

- [ ] **Step 1: Write failing contact-state tests**

Add tests:

```python
import jax.numpy as jnp
import mujoco

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)

MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


def make_model_data():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def test_wheel_geom_ids_are_resolved_by_name():
    model, data = make_model_data()
    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(robot_mass=8.1, torso_inertia=jnp.array([0.1, 0.1, 0.05])),
        mj_model=model,
    )
    assert estimator.left_wheel_geom_id == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    assert estimator.right_wheel_geom_id == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")


def test_reset_keyframe_detects_wheel_contact_and_force():
    model, data = make_model_data()
    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(robot_mass=sum(model.body_mass), torso_inertia=jnp.array([0.1, 0.1, 0.05])),
        mj_model=model,
    )
    state, _ = estimator.estimate(jnp.zeros(42), data, None)
    assert state.left_wheel_contact
    assert state.right_wheel_contact
    assert state.left_wheel_force > 0.0
    assert state.right_wheel_force > 0.0
    assert state.total_contact_force_z > 0.5 * sum(model.body_mass) * abs(model.opt.gravity[2])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_centroidal_state_estimator.py -v
```

Expected before implementation: constructor signature or contact ID assertions fail because IDs are hardcoded.

- [ ] **Step 3: Extend `CentroidalState` fields**

Add fields to the dataclass:

```python
base_quat: Array
base_ang_vel: Array
roll: float
pitch: float
yaw: float
roll_rate: float
pitch_rate: float
yaw_rate: float
left_contact_force_world: Array
right_contact_force_world: Array
total_contact_force_z: float
```

- [ ] **Step 4: Resolve geom IDs by name when MuJoCo model is provided**

Update constructor:

```python
def __init__(self, config: CentroidalStateEstimatorConfig, mj_model=None):
    self.config = config
    self.dt = 0.02
    self.mj_model = mj_model

    if mj_model is not None:
        self.left_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
        self.right_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
        if self.left_wheel_geom_id == -1 or self.right_wheel_geom_id == -1:
            raise ValueError("Wheel collision geoms not found")
    else:
        self.left_wheel_geom_id = 15
        self.right_wheel_geom_id = 28
```

- [ ] **Step 5: Measure contact forces with `mj_contactForce()`**

Inside `estimate()`, replace direct `data.contact.force` logic with:

```python
left_force_world = jnp.zeros(3)
right_force_world = jnp.zeros(3)

if self.mj_model is not None:
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(self.mj_model, data, i, force_contact)
        frame = np.array(contact.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]

        if geom1 == self.left_wheel_geom_id or geom2 == self.left_wheel_geom_id:
            left_wheel_contact = True
            left_force_world = left_force_world + jnp.array(force_world)
            left_wheel_force = float(left_force_world[2])

        if geom1 == self.right_wheel_geom_id or geom2 == self.right_wheel_geom_id:
            right_wheel_contact = True
            right_force_world = right_force_world + jnp.array(force_world)
            right_wheel_force = float(right_force_world[2])
```

Also compute:

```python
total_contact_force_z = float(left_force_world[2] + right_force_world[2])
```

- [ ] **Step 6: Extract base orientation and angular velocity**

Inside `estimate()`:

```python
base_quat = jnp.array(data.qpos[3:7])
base_ang_vel = jnp.array(data.qvel[3:6])
roll, pitch, yaw = compute_orientation_from_quaternion(np.array(data.qpos[3:7]))
roll_rate = float(base_ang_vel[0])
pitch_rate = float(base_ang_vel[1])
yaw_rate = float(base_ang_vel[2])
```

- [ ] **Step 7: Run contact tests**

Run:

```bash
pytest tests/test_centroidal_state_estimator.py -v
```

Expected: all tests pass.

---

### Task 3: Desired wrench correctness

**Files:**
- Modify: `wheeled_biped/controllers/centroidal_wrench_computer.py`
- Test: `tests/test_centroidal_wrench_computer.py`

- [ ] **Step 1: Write failing desired-wrench tests**

Add tests:

```python
import jax.numpy as jnp

from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState
from wheeled_biped.controllers.centroidal_wrench_computer import CentroidalWrenchComputer


def make_state(roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, com_z=0.42):
    return CentroidalState(
        com_pos=jnp.array([0.0, 0.0, com_z]),
        com_vel=jnp.zeros(3),
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=40.0,
        right_wheel_force=40.0,
        base_quat=jnp.array([1.0, 0.0, 0.0, 0.0]),
        base_ang_vel=jnp.array([roll_rate, pitch_rate, 0.0]),
        roll=roll,
        pitch=pitch,
        yaw=0.0,
        roll_rate=roll_rate,
        pitch_rate=pitch_rate,
        yaw_rate=0.0,
        left_contact_force_world=jnp.array([0.0, 0.0, 40.0]),
        right_contact_force_world=jnp.array([0.0, 0.0, 40.0]),
        total_contact_force_z=80.0,
    )


def test_static_fz_equals_weight_at_target_height():
    computer = CentroidalWrenchComputer(robot_mass=8.1, gravity=9.81, k_height=50.0)
    force, moment = computer.compute_desired_wrench_from_state(make_state(com_z=0.42), height_cmd=0.42)
    assert abs(float(force[2]) - 8.1 * 9.81) < 1e-5
    assert abs(float(moment[0])) < 1e-8
    assert abs(float(moment[1])) < 1e-8


def test_positive_roll_generates_corrective_mx():
    computer = CentroidalWrenchComputer(k_roll=10.0, k_roll_rate=0.0, robot_mass=8.1)
    _, moment = computer.compute_desired_wrench_from_state(make_state(roll=0.2), height_cmd=0.42)
    assert float(moment[0]) < 0.0
    assert abs(float(moment[1])) < 1e-8


def test_positive_pitch_generates_corrective_my():
    computer = CentroidalWrenchComputer(k_pitch=10.0, k_pitch_rate=0.0, robot_mass=8.1)
    _, moment = computer.compute_desired_wrench_from_state(make_state(pitch=0.2), height_cmd=0.42)
    assert abs(float(moment[0])) < 1e-8
    assert float(moment[1]) < 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_centroidal_wrench_computer.py -v
```

Expected before implementation: `compute_desired_wrench_from_state` missing.

- [ ] **Step 3: Add state-based wrench method**

In `CentroidalWrenchComputer`, add:

```python
def compute_desired_wrench_from_state(self, state: CentroidalState, height_cmd: float) -> tuple[Array, Array]:
    f_gravity = jnp.array([0.0, 0.0, self.robot_mass * self.gravity])
    height_error = height_cmd - state.com_pos[2]
    f_height = jnp.array([0.0, 0.0, self.k_height * height_error])

    f_com_lateral = jnp.array([
        0.0,
        -self.k_com_lateral * state.com_pos[1] - self.k_com_lateral_damping * state.com_vel[1],
        0.0,
    ])
    f_com_sagittal = jnp.array([
        -self.k_com_sagittal * state.com_pos[0] - self.k_com_sagittal_damping * state.com_vel[0],
        0.0,
        0.0,
    ])
    f_cp = jnp.array([
        -self.k_cp_sagittal * state.capture_point[0],
        -self.k_cp_lateral * state.capture_point[1],
        0.0,
    ])

    desired_force = f_gravity + f_height + f_com_lateral + f_com_sagittal + f_cp

    m_roll = -self.k_roll * state.roll - self.k_roll_rate * state.roll_rate
    m_pitch = -self.k_pitch * state.pitch - self.k_pitch_rate * state.pitch_rate
    desired_moment = jnp.array([m_roll, m_pitch, 0.0])

    return desired_force, desired_moment
```

Then update `compute_desired_wrench_vector()` to call this method when used by integrated WBC, while keeping existing API available for legacy callers.

- [ ] **Step 4: Run desired-wrench tests**

Run:

```bash
pytest tests/test_centroidal_wrench_computer.py -v
```

Expected: all tests pass.

---

### Task 4: Contact-aware force distribution

**Files:**
- Modify: `wheeled_biped/controllers/simple_force_distributor.py`
- Test: `tests/test_simple_force_distributor.py`

- [ ] **Step 1: Write failing force-distribution tests**

Add tests:

```python
import jax.numpy as jnp

from wheeled_biped.controllers.simple_force_distributor import SimpleForceDistributor


def test_both_contacts_split_vertical_force():
    distributor = SimpleForceDistributor()
    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        jnp.array([0.0, 0.0, 80.0, 0.0, 0.0, 0.0]),
        left_contact=True,
        right_contact=True,
    )
    assert float(f_left[2]) == 40.0
    assert float(f_right[2]) == 40.0
    assert diagnostics["feasible"]


def test_left_only_contact_sends_right_force_to_zero():
    distributor = SimpleForceDistributor()
    f_left, f_right, _, diagnostics = distributor.distribute_wrench_contact_aware(
        jnp.array([0.0, 0.0, 80.0, 0.0, 0.0, 0.0]),
        left_contact=True,
        right_contact=False,
    )
    assert float(f_left[2]) == 80.0
    assert float(f_right[2]) == 0.0
    assert diagnostics["feasible"]


def test_no_contact_outputs_zero_and_infeasible():
    distributor = SimpleForceDistributor()
    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        jnp.array([1.0, 2.0, 80.0, 3.0, 4.0, 5.0]),
        left_contact=False,
        right_contact=False,
    )
    assert float(jnp.linalg.norm(f_left)) == 0.0
    assert float(jnp.linalg.norm(f_right)) == 0.0
    assert float(jnp.linalg.norm(tau_hip_roll)) == 0.0
    assert not diagnostics["feasible"]
    assert diagnostics["reason"] == "no_support_contact_lost"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_simple_force_distributor.py -v
```

Expected before implementation: `distribute_wrench_contact_aware` missing.

- [ ] **Step 3: Implement contact-aware distribution**

Add method:

```python
def distribute_wrench_contact_aware(
    self,
    desired_wrench: Array,
    left_contact: bool,
    right_contact: bool,
) -> tuple[Array, Array, Array, dict]:
    Fx, Fy, Fz, Mx, My, Mz = desired_wrench

    if not left_contact and not right_contact:
        return (
            jnp.zeros(3),
            jnp.zeros(3),
            jnp.zeros(2),
            {"feasible": False, "reason": "no_support_contact_lost"},
        )

    active_count = int(left_contact) + int(right_contact)
    f_left = jnp.zeros(3)
    f_right = jnp.zeros(3)

    if left_contact:
        f_left = jnp.array([Fx / active_count, Fy / active_count, Fz / active_count])
    if right_contact:
        f_right = jnp.array([Fx / active_count, Fy / active_count, Fz / active_count])

    tau_hip_roll = jnp.array([Mx / active_count if left_contact else 0.0, Mx / active_count if right_contact else 0.0])
    diagnostics = {"feasible": True, "reason": "ok"}
    return f_left, f_right, tau_hip_roll, diagnostics
```

Keep the existing `distribute_wrench()` for legacy callers, but route new WBC through this method.

- [ ] **Step 4: Run force-distribution tests**

Run:

```bash
pytest tests/test_simple_force_distributor.py -v
```

Expected: all tests pass.

---

### Task 5: Contact Jacobian mapping diagnostics

**Files:**
- Modify: `wheeled_biped/controllers/contact_jacobian.py`
- Test: `tests/test_contact_jacobian.py`

- [ ] **Step 1: Write Jacobian shape and sign tests**

Add tests:

```python
import jax.numpy as jnp
import mujoco
import numpy as np

from wheeled_biped.controllers.contact_jacobian import ContactJacobian

MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


def make_model_data():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def test_wheel_jacobians_have_expected_shape():
    model, data = make_model_data()
    contact_jacobian = ContactJacobian(model)
    j_left, j_right = contact_jacobian.compute_wheel_jacobians(data)
    assert j_left.shape == (3, 10)
    assert j_right.shape == (3, 10)


def test_symmetric_upward_force_maps_to_nonzero_leg_torque():
    model, data = make_model_data()
    contact_jacobian = ContactJacobian(model)
    tau = contact_jacobian.map_contact_forces_to_torques(
        data,
        jnp.array([0.0, 0.0, 40.0]),
        jnp.array([0.0, 0.0, 40.0]),
        jnp.array([0.0, 0.0]),
    )
    assert tau.shape == (10,)
    assert abs(float(tau[3])) > 1.0
    assert abs(float(tau[8])) > 1.0
    assert np.isfinite(np.array(tau)).all()
```

- [ ] **Step 2: Run Jacobian tests**

Run:

```bash
pytest tests/test_contact_jacobian.py -v
```

Expected: shape test should pass; sign/torque test locks current nonzero behavior.

- [ ] **Step 3: Add diagnostics without changing sign yet**

Add a helper to `ContactJacobian`:

```python
def compute_force_mapping_diagnostics(self, mj_data: mujoco.MjData, f_left: Array, f_right: Array) -> dict:
    J_left, J_right = self.compute_wheel_jacobians(mj_data)
    tau_left = J_left.T @ (-f_left)
    tau_right = J_right.T @ (-f_right)
    return {
        "left_jacobian_z_row": J_left[2],
        "right_jacobian_z_row": J_right[2],
        "tau_left_from_force": tau_left,
        "tau_right_from_force": tau_right,
        "tau_total_from_force": tau_left + tau_right,
    }
```

- [ ] **Step 4: Run focused Jacobian tests again**

Run:

```bash
pytest tests/test_contact_jacobian.py -v
```

Expected: all tests pass.

---

### Task 6: Integrated WBC uses corrected state, contact-aware distribution, and rich diagnostics

**Files:**
- Modify: `wheeled_biped/controllers/integrated_wbc.py`
- Test: `tests/test_integrated_wbc.py`

- [ ] **Step 1: Write integrated WBC diagnostics test**

Add test:

```python
import jax.numpy as jnp
import mujoco

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import CapturePointEstimator, CapturePointEstimatorConfig
from wheeled_biped.controllers.integrated_wbc import IntegratedWBC

MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


def test_integrated_wbc_reports_contact_aware_diagnostics():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(robot_mass=sum(model.body_mass), torso_inertia=jnp.array([0.1, 0.1, 0.05])),
        mj_model=model,
    )
    state, _ = estimator.estimate(jnp.zeros(42), data, None)
    state = CapturePointEstimator(CapturePointEstimatorConfig()).update(state)

    controller = IntegratedWBC(model, robot_mass=sum(model.body_mass), gravity=abs(model.opt.gravity[2]))
    tau, diagnostics = controller.compute_wbc_torque(data, jnp.zeros(42), state, float(state.com_pos[2]))

    assert tau.shape == (10,)
    assert diagnostics["left_contact_active"]
    assert diagnostics["right_contact_active"]
    assert diagnostics["force_distribution_feasible"]
    assert diagnostics["desired_wrench_Fz"] > 0.0
    assert diagnostics["total_contact_force_z"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_integrated_wbc.py -v
```

Expected before implementation: missing diagnostics keys or constructor mismatch.

- [ ] **Step 3: Update `IntegratedWBC.compute_wbc_torque()` orchestration**

Replace desired wrench and distribution block with:

```python
desired_force, desired_moment = self.wrench_computer.compute_desired_wrench_from_state(state, height_cmd)
desired_wrench = jnp.concatenate([desired_force, desired_moment])

solve_start = time.perf_counter()
f_left, f_right, tau_hip_roll, distribution_diagnostics = self.force_distributor.distribute_wrench_contact_aware(
    desired_wrench,
    left_contact=bool(state.left_wheel_contact),
    right_contact=bool(state.right_wheel_contact),
)
solve_time_ms = (time.perf_counter() - solve_start) * 1000.0

tau_wbc = self.contact_jacobian.map_contact_forces_to_torques(mj_data, f_left, f_right, tau_hip_roll)
tau_wbc = self.clip_to_authority_budget(tau_wbc)
```

- [ ] **Step 4: Replace actual contact force measurement**

Do not use `_measure_total_vertical_contact_force()` based on `efc_force`. Use state fields:

```python
actual_fz_total = float(state.total_contact_force_z)
desired_fz_total = float(f_left[2] + f_right[2])
```

Remove or stop calling the old `efc_force` measurement path.

- [ ] **Step 5: Add required diagnostics keys**

Ensure diagnostics includes:

```python
"left_contact_active": bool(state.left_wheel_contact),
"right_contact_active": bool(state.right_wheel_contact),
"left_contact_force_world": state.left_contact_force_world,
"right_contact_force_world": state.right_contact_force_world,
"total_contact_force_z": actual_fz_total,
"force_distribution_feasible": bool(distribution_diagnostics["feasible"]),
"force_distribution_reason": distribution_diagnostics["reason"],
"distributed_left_fx": float(f_left[0]),
"distributed_left_fy": float(f_left[1]),
"distributed_left_fz": float(f_left[2]),
"distributed_right_fx": float(f_right[0]),
"distributed_right_fy": float(f_right[1]),
"distributed_right_fz": float(f_right[2]),
```

- [ ] **Step 6: Run integrated WBC tests**

Run:

```bash
pytest tests/test_integrated_wbc.py -v
```

Expected: all tests pass.

---

### Task 7: Simulation applies WBC torque and logs actual contact telemetry

**Files:**
- Modify: `scripts/simulate_hierarchical_controller.py`
- Test: no new unit test required; verified by smoke command.

- [ ] **Step 1: Update estimator construction**

Change:

```python
centroidal_estimator = CentroidalStateEstimator(
    CentroidalStateEstimatorConfig(robot_mass=8.1, torso_inertia=jnp.array([0.1, 0.1, 0.05]))
)
```

to:

```python
robot_mass = float(np.sum(mj_model.body_mass))
gravity = float(abs(mj_model.opt.gravity[2]))
centroidal_estimator = CentroidalStateEstimator(
    CentroidalStateEstimatorConfig(
        robot_mass=robot_mass,
        torso_inertia=jnp.array([0.1, 0.1, 0.05]),
    ),
    mj_model=mj_model,
)
```

- [ ] **Step 2: Update WBC construction mass/gravity**

Use:

```python
wbc_controller = IntegratedWBC(
    mj_model,
    k_roll=15.0,
    k_roll_rate=3.0,
    k_pitch=25.0,
    k_pitch_rate=5.0,
    k_com_lateral=15.0,
    k_com_lateral_damping=3.0,
    k_com_sagittal=10.0,
    k_com_sagittal_damping=2.0,
    k_cp_lateral=25.0,
    k_cp_sagittal=20.0,
    k_height=50.0,
    robot_mass=robot_mass,
    gravity=gravity,
    wbc_authority_budget=0.70,
    max_actuator_torque=60.0,
    force_feedback_gain=0.0,
)
```

Set force feedback to `0.0` for C1 correctness until actual measured contact feedback is designed and tested.

- [ ] **Step 3: Fix orientation unpacking in script**

Use:

```python
roll, pitch = compute_orientation_from_gravity(gravity_body)
roll, pitch, yaw = compute_orientation_from_quaternion(quat)
```

Everywhere in the script.

- [ ] **Step 4: Apply WBC torque as primary torque path**

Replace the final torque assembly with explicit composition:

```python
tau_posture = leg_position_controller.compute_leg_torques(joint_pos, joint_vel)
tau_wheel_secondary = jnp.zeros(10)
tau_total = tau_wbc + 0.15 * tau_posture + tau_wheel_secondary

torque_limit = jnp.array(mj_model.actuator_ctrlrange[:, 1])
tau_total = jnp.clip(tau_total, -torque_limit, torque_limit)
mj_data.ctrl[:] = np.array(tau_total)
```

Do not overwrite hip roll, hip pitch, knee, or wheel torques independently after this point.

- [ ] **Step 5: Add telemetry fields**

Add fields to telemetry initialization:

```python
"mass_kg": [],
"weight_N": [],
"roll_rate_rad_s": [],
"pitch_rate_rad_s": [],
"yaw_rate_rad_s": [],
"left_contact_active": [],
"right_contact_active": [],
"n_contacts": [],
"left_contact_force_world_x": [],
"left_contact_force_world_y": [],
"left_contact_force_world_z": [],
"right_contact_force_world_x": [],
"right_contact_force_world_y": [],
"right_contact_force_world_z": [],
"total_contact_force_z": [],
"force_distribution_feasible": [],
"force_distribution_reason": [],
"distributed_left_fx": [],
"distributed_left_fy": [],
"distributed_left_fz": [],
"distributed_right_fx": [],
"distributed_right_fy": [],
"distributed_right_fz": [],
"tau_saturation_rate": [],
```

Append values from `centroidal_state`, `qp_diagnostics`, and `tau_total` every step.

- [ ] **Step 6: Remove hardcoded 147.4 N diagnostic**

Replace diagnostic print with:

```python
weight_n = robot_mass * gravity
print(f"  Total contact force z: {qp_diagnostics['total_contact_force_z']:.2f} N (weight: {weight_n:.2f} N)")
print(f"  Desired Fz: {qp_diagnostics['desired_wrench_Fz']:.2f} N")
```

- [ ] **Step 7: Run simulation smoke**

Run:

```bash
python scripts/simulate_hierarchical_controller.py
```

Expected:

```text
- script completes or terminates with a physical failure reason
- telemetry CSV is written
- first diagnostic prints model weight around 79.46 N for 8.1 kg robot
- actual total contact force z is printed/logged
- WBC torque contributes directly to mj_data.ctrl
```

---

### Task 8: Verification gate

**Files:**
- No new code unless tests reveal implementation bugs.

- [ ] **Step 1: Run focused C1 tests**

Run:

```bash
pytest tests/test_orientation_utils.py tests/test_centroidal_state_estimator.py tests/test_centroidal_wrench_computer.py tests/test_simple_force_distributor.py tests/test_contact_jacobian.py tests/test_integrated_wbc.py -v
```

Expected: all pass.

- [ ] **Step 2: Run fast existing tests that avoid slow env test**

Run:

```bash
pytest tests/ --ignore=tests/test_env.py -m "not slow" -v
```

Expected: all pass, or failures are unrelated and documented with exact failing test names.

- [ ] **Step 3: Run simulation and inspect telemetry summary**

Run:

```bash
python scripts/simulate_hierarchical_controller.py
```

Expected: telemetry exists under `outputs/hierarchical_controller_sim/telemetry_*.csv` and includes actual contact-force fields.

- [ ] **Step 4: Analyze latest telemetry**

Run:

```bash
python - <<'PY'
import csv
from pathlib import Path
path = max(Path('outputs/hierarchical_controller_sim').glob('telemetry_*.csv'), key=lambda p: p.stat().st_mtime)
with path.open(newline='') as f:
    rows = list(csv.DictReader(f))
print(path)
print('rows', len(rows))
for key in ['total_contact_force_z', 'desired_wrench_Fz', 'tau_wbc_max', 'tau_total_max', 'pitch', 'roll', 'termination_reason']:
    values = [r.get(key, '') for r in rows]
    print(key, values[0], values[-1])
PY
```

Expected: `total_contact_force_z` is present and non-empty; failure, if any, is interpretable from contact force, torque saturation, pitch/roll, and termination reason.

---

## Stop conditions

Stop and report before continuing if any of these happens:

```text
- orientation convention tests cannot be made to agree between quaternion and gravity paths
- reset keyframe does not produce wheel contacts after geom IDs are fixed
- measured reset contact force is near zero while visual contacts exist
- WBC torque is nonzero but cannot be applied because actuator/control semantics conflict
- changing torque sign makes contact support worse in measured MuJoCo behavior
- more than three fix attempts fail in the same layer
```
