# Phase 8 Upright Standing Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the standalone hierarchical torque controller stand more upright by separating leg posture tracking, bounded roll recovery, and active wheel-based pitch/CoM recovery.

**Architecture:** Keep the existing WBC pipeline, but stop treating WBC as the only owner of leg shape. Use the height-dependent posture target as the single target source, add diagnostics first, then progressively add leg PD tracking, WBC leg authority scaling, hip-roll soft centering, wheel sagittal recovery, and mode arbitration.

**Tech Stack:** Python, MuJoCo, JAX/JAX NumPy, pytest, CSV telemetry from `scripts/simulate_hierarchical_controller.py`.

---

## Problem Summary

Current Phase 7 telemetry shows the target posture remains symmetric, but the robot does not track it:

```text
step 157:
l_hip_roll = -0.7066, target = 0.0
r_hip_roll = +0.7076, target = 0.0
l_hip_yaw  = -0.4019, target = -0.0007
r_hip_yaw  = +0.4024, target = +0.0009
l_knee     = 1.6740, target = 1.7484
r_knee     = 1.5269, target = 1.7484
joint_error_norm = 2.4981
```

Root causes:

1. `LegPositionController` exists but is not active in `tau_total_raw`.
2. Its fixed constructor targets (`0.674267`, `1.668071`) do not match the active `height_cmd=0.40` target (`0.926052`, `1.748364`).
3. WBC currently commands all leg joints via `tau_wbc_masked = tau_wbc_raw`, so it can override posture.
4. Hip-roll is intentionally free, so WBC uses opposite-sign hip-roll and spreads the legs up to about `±0.70 rad`.
5. Pitch recovery has `m_pitch = 0.0` and only weak wheel torque, so the robot does not actively move the support point under the CoM.

---

## File Map

- `wheeled_biped/controllers/leg_position_controller.py`
  - Refactor from stale fixed hip/knee targets to per-step full-target input.
  - Keep output torque semantics unchanged: direct torque command array, wheels zero.

- `wheeled_biped/controllers/posture_regularizer.py`
  - Remains the single posture target source through `compute_target_posture_from_height(height_cmd)`.
  - Later tasks may add hip-roll soft centering.

- `wheeled_biped/controllers/integrated_wbc.py`
  - Later tasks scale WBC authority by joint group.
  - Step 1 must not alter WBC torque output.

- `scripts/simulate_hierarchical_controller.py`
  - Step 1 removes stale target arguments from `LegPositionController` construction.
  - Step 1 adds telemetry for per-joint torque components and per-group errors.
  - Later tasks add leg PD torque, wheel torque assist, and mode arbitration.

- `tests/test_leg_position_controller.py`
  - Add tests proving `LegPositionController` consumes a full per-step target.
  - Add tests proving wheels and hip-roll are not position controlled by leg PD.

---

## Task 1 — Save plan and add target-source tests

**Files:**
- Modify: `tests/test_leg_position_controller.py`
- Read-only reference: `wheeled_biped/controllers/leg_position_controller.py`

- [ ] **Step 1: Add a failing test for full-target leg PD input**

Add this test to `tests/test_leg_position_controller.py`:

```python
def test_leg_position_controller_uses_per_step_full_target():
    controller = LegPositionController(
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=20.0,
        kd_hip_pitch=2.0,
        kp_knee=30.0,
        kd_knee=3.0,
        max_torque=40.0,
    )
    joint_pos = jnp.array([0.2, -0.2, 0.8, 1.6, 1.0, -0.2, 0.2, 1.0, 1.9, -1.0])
    joint_vel = jnp.zeros(10)
    target_joint_pos = jnp.array([0.0, -0.1, 1.0, 1.7, 0.0, 0.0, 0.1, 0.9, 1.8, 0.0])

    tau = controller.compute_leg_torques(joint_pos, joint_vel, target_joint_pos)

    assert jnp.isclose(tau[0], 0.0)
    assert jnp.isclose(tau[1], 0.5)
    assert jnp.isclose(tau[2], 4.0)
    assert jnp.isclose(tau[3], 3.0)
    assert jnp.isclose(tau[4], 0.0)
    assert jnp.isclose(tau[5], 0.0)
    assert jnp.isclose(tau[6], -0.5)
    assert jnp.isclose(tau[7], -2.0)
    assert jnp.isclose(tau[8], -3.0)
    assert jnp.isclose(tau[9], 0.0)
```

- [ ] **Step 2: Add a failing test for velocity damping and torque clipping**

Add this test to `tests/test_leg_position_controller.py`:

```python
def test_leg_position_controller_damps_velocity_and_clips():
    controller = LegPositionController(
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=20.0,
        kd_hip_pitch=2.0,
        kp_knee=30.0,
        kd_knee=3.0,
        max_torque=5.0,
    )
    joint_pos = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    joint_vel = jnp.array([0.0, 2.0, -2.0, -2.0, 0.0, 0.0, -2.0, 2.0, 2.0, 0.0])
    target_joint_pos = jnp.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0])

    tau = controller.compute_leg_torques(joint_pos, joint_vel, target_joint_pos)

    assert jnp.isclose(tau[1], -2.0)
    assert jnp.isclose(tau[2], 5.0)
    assert jnp.isclose(tau[3], 5.0)
    assert jnp.isclose(tau[6], 2.0)
    assert jnp.isclose(tau[7], -5.0)
    assert jnp.isclose(tau[8], -5.0)
```

- [ ] **Step 3: Run the tests to verify they fail before implementation**

Run:

```bash
pytest tests/test_leg_position_controller.py -q
```

Expected before implementation:

```text
FAILED ... TypeError: LegPositionController.__init__() got an unexpected keyword argument 'kp_hip_yaw'
```

or:

```text
FAILED ... TypeError: compute_leg_torques() takes 3 positional arguments but 4 were given
```

---

## Task 2 — Refactor `LegPositionController` target input without using it in dynamics yet

**Files:**
- Modify: `wheeled_biped/controllers/leg_position_controller.py`
- Test: `tests/test_leg_position_controller.py`

- [ ] **Step 1: Replace fixed target constructor fields with hip-yaw gains**

Change `LegPositionController.__init__` to this signature:

```python
def __init__(
    self,
    kp_hip_yaw: float = 5.0,
    kd_hip_yaw: float = 1.0,
    kp_hip_pitch: float = 20.0,
    kd_hip_pitch: float = 2.0,
    kp_knee: float = 30.0,
    kd_knee: float = 3.0,
    max_torque: float = 30.0,
):
```

Set these fields:

```python
self.kp_hip_yaw = kp_hip_yaw
self.kd_hip_yaw = kd_hip_yaw
self.kp_hip_pitch = kp_hip_pitch
self.kd_hip_pitch = kd_hip_pitch
self.kp_knee = kp_knee
self.kd_knee = kd_knee
self.max_torque = max_torque
self.LEG_POSTURE_INDICES = [1, 2, 3, 6, 7, 8]
```

- [ ] **Step 2: Replace `compute_leg_torques` with a full-target implementation**

Use this body:

```python
def compute_leg_torques(self, joint_pos: Array, joint_vel: Array, target_joint_pos: Array) -> Array:
    tau = jnp.zeros(10)

    joint_gains = {
        1: (self.kp_hip_yaw, self.kd_hip_yaw),
        2: (self.kp_hip_pitch, self.kd_hip_pitch),
        3: (self.kp_knee, self.kd_knee),
        6: (self.kp_hip_yaw, self.kd_hip_yaw),
        7: (self.kp_hip_pitch, self.kd_hip_pitch),
        8: (self.kp_knee, self.kd_knee),
    }

    for joint_idx, (kp, kd) in joint_gains.items():
        pos_error = target_joint_pos[joint_idx] - joint_pos[joint_idx]
        vel_error = -joint_vel[joint_idx]
        tau_raw = kp * pos_error + kd * vel_error
        tau = tau.at[joint_idx].set(jnp.clip(tau_raw, -self.max_torque, self.max_torque))

    return tau
```

This function intentionally leaves hip-roll and wheels at zero.

- [ ] **Step 3: Run targeted controller tests**

Run:

```bash
pytest tests/test_leg_position_controller.py -q
```

Expected:

```text
passed
```

---

## Task 3 — Update simulation construction to remove stale target values

**Files:**
- Modify: `scripts/simulate_hierarchical_controller.py`

- [ ] **Step 1: Replace stale `LegPositionController` construction**

Replace the construction with:

```python
leg_position_controller = LegPositionController(
    kp_hip_yaw=5.0,
    kd_hip_yaw=1.0,
    kp_hip_pitch=20.0,
    kd_hip_pitch=3.0,
    kp_knee=35.0,
    kd_knee=4.0,
    max_torque=25.0,
)
```

Do not add `tau_leg_position` to `tau_total_raw` in this task. Step 1 must not change dynamics.

- [ ] **Step 2: Compile simulation script**

Run:

```bash
python -m py_compile scripts/simulate_hierarchical_controller.py wheeled_biped/controllers/leg_position_controller.py
```

Expected:

```text
no output, exit code 0
```

---

## Task 4 — Add Step 1 telemetry without changing torque dynamics

**Files:**
- Modify: `scripts/simulate_hierarchical_controller.py`

- [ ] **Step 1: Add telemetry fields**

Add these keys to the `telemetry` dictionary:

```python
"tau_wbc_per_joint": [],
"tau_posture_per_joint": [],
"tau_leg_position_per_joint": [],
"tau_wheel_balance_per_joint": [],
"tau_total_per_joint": [],
"hip_roll_abs_max": [],
"hip_yaw_abs_max": [],
"hip_pitch_error_max": [],
"knee_error_max": [],
"wheel_balance_torque": [],
"control_mode": [],
```

- [ ] **Step 2: Compute diagnostic-only zero torque components**

Immediately after `target_joint_pos` and `joint_pos_error` are computed, add:

```python
tau_leg_position = jnp.zeros(10)
tau_wheel_balance = jnp.zeros(10)
control_mode = "upright"
hip_roll_indices = jnp.array([0, 5])
hip_yaw_indices = jnp.array([1, 6])
hip_pitch_indices = jnp.array([2, 7])
knee_indices = jnp.array([3, 8])
hip_roll_abs_max = float(jnp.max(jnp.abs(joint_pos[hip_roll_indices])))
hip_yaw_abs_max = float(jnp.max(jnp.abs(joint_pos[hip_yaw_indices])))
hip_pitch_error_max = float(jnp.max(jnp.abs(joint_pos_error[hip_pitch_indices])))
knee_error_max = float(jnp.max(jnp.abs(joint_pos_error[knee_indices])))
wheel_balance_torque = 0.0
```

Do not add `tau_leg_position` or `tau_wheel_balance` to `tau_total_raw` yet.

- [ ] **Step 3: Append telemetry rows**

After existing motor tracking telemetry appends, add:

```python
telemetry["tau_wbc_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_wbc)))
telemetry["tau_posture_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_posture)))
telemetry["tau_leg_position_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_leg_position)))
telemetry["tau_wheel_balance_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_wheel_balance)))
telemetry["tau_total_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_total)))
telemetry["hip_roll_abs_max"].append(hip_roll_abs_max)
telemetry["hip_yaw_abs_max"].append(hip_yaw_abs_max)
telemetry["hip_pitch_error_max"].append(hip_pitch_error_max)
telemetry["knee_error_max"].append(knee_error_max)
telemetry["wheel_balance_torque"].append(wheel_balance_torque)
telemetry["control_mode"].append(control_mode)
```

- [ ] **Step 4: Run short simulation smoke check**

Run:

```bash
python scripts/simulate_hierarchical_controller.py --steps 5
```

Expected:

```text
Total steps: 5
Telemetry saved to: outputs\hierarchical_controller_sim\telemetry_<timestamp>.csv
```

- [ ] **Step 5: Verify telemetry columns exist**

Run:

```bash
python - <<'PY'
import csv
from pathlib import Path
p = max(Path('outputs/hierarchical_controller_sim').glob('telemetry_*.csv'), key=lambda x: x.stat().st_mtime)
with p.open(newline='') as f:
    header = next(csv.reader(f))
required = [
    'tau_wbc_per_joint',
    'tau_posture_per_joint',
    'tau_leg_position_per_joint',
    'tau_wheel_balance_per_joint',
    'tau_total_per_joint',
    'hip_roll_abs_max',
    'hip_yaw_abs_max',
    'hip_pitch_error_max',
    'knee_error_max',
    'wheel_balance_torque',
    'control_mode',
]
missing = [name for name in required if name not in header]
print('latest_csv=', p)
print('missing=', missing)
assert not missing
PY
```

Expected:

```text
missing= []
```

---

## Task 5 — Confirm Step 1 changed diagnostics only

**Files:**
- Read/verify: latest telemetry CSV

- [ ] **Step 1: Run 100-step comparison smoke**

Run:

```bash
python scripts/simulate_hierarchical_controller.py --steps 100
```

Expected:

```text
Total steps: 100
Terminated: False
```

- [ ] **Step 2: Inspect diagnostic-only torque fields**

Run:

```bash
python - <<'PY'
import csv
from pathlib import Path
p = max(Path('outputs/hierarchical_controller_sim').glob('telemetry_*.csv'), key=lambda x: x.stat().st_mtime)
with p.open(newline='') as f:
    rows = list(csv.DictReader(f))
def arr(s):
    return [float(x) for x in s.split(',')]
leg_max = max(max(abs(x) for x in arr(r['tau_leg_position_per_joint'])) for r in rows)
wheel_balance_max = max(max(abs(x) for x in arr(r['tau_wheel_balance_per_joint'])) for r in rows)
print('latest_csv=', p)
print('rows=', len(rows))
print('tau_leg_position_max=', leg_max)
print('tau_wheel_balance_max=', wheel_balance_max)
assert leg_max == 0.0
assert wheel_balance_max == 0.0
PY
```

Expected:

```text
tau_leg_position_max= 0.0
tau_wheel_balance_max= 0.0
```

This confirms Step 1 does not yet alter dynamics.

---

## Future Task 6 — Enable real leg posture PD torque

Do not execute this task in Step 1.

Later torque path:

```python
tau_leg_position = leg_position_controller.compute_leg_torques(
    joint_pos,
    joint_vel,
    target_joint_pos,
)

tau_total_raw = (
    tau_wbc
    + tau_leg_position
    + tau_posture
    + tau_wheel_secondary
    + tau_inverse_dynamics
)
```

Success target:

```text
knee error < 0.08–0.12 rad most of the time
hip_pitch error < 0.08–0.12 rad most of the time
joint_error_norm lower than Phase 7 baseline
```

---

## Future Task 7 — Scale WBC authority on leg joints

Do not execute this task in Step 1.

Later `IntegratedWBC` direction:

```python
wbc_joint_scale = jnp.array([1.0, 0.3, 0.25, 0.25, 1.0, 1.0, 0.3, 0.25, 0.25, 1.0])
tau_wbc_masked = tau_wbc_raw * wbc_joint_scale
```

Test variants:

```text
leg_wbc_scale = 1.00
leg_wbc_scale = 0.50
leg_wbc_scale = 0.25
```

---

## Future Task 8 — Add soft hip-roll centering

Do not execute this task in Step 1.

Target behavior:

```text
hip_roll remains free inside ±0.15 rad
soft correction starts beyond ±0.25 rad
strong correction prevents sustained ±0.70 rad spread
```

---

## Future Task 9 — Add wheel-based pitch/CoM recovery

Do not execute this task in Step 1.

Target behavior:

```text
wheel torque increases from <1 Nm max to bounded 2–5 Nm recovery authority
positive/negative wheel torque sign verified before enabling in normal rollout
```

---

## Future Task 10 — Add upright/recovery mode arbitration

Do not execute this task in Step 1.

Mode sketch:

```python
upright_mode = abs(roll) < 0.20 and abs(pitch) < 0.15
recovery_mode = abs(roll) > 0.30 or abs(pitch) > 0.25
```

---

## Self-Review

- Spec coverage: Step 1 covers target-source cleanup and diagnostic telemetry without changing dynamics. Future tasks cover leg PD, WBC scaling, hip-roll centering, wheel pitch recovery, and mode arbitration.
- Placeholder scan: No implementation step contains TBD/TODO/fill-in placeholders.
- Type consistency: New `LegPositionController.compute_leg_torques(joint_pos, joint_vel, target_joint_pos)` signature is used consistently in tests and future torque-path notes.
