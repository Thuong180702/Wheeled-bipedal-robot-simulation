# Hierarchical Controller Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add targeted diagnostics that identify whether the 1-second balance failure is caused by WBC torque sign/mapping, hip-roll convention, weak posture tracking, missing inverse dynamics, torque-rate jerk, or contact-force artifacts.

**Architecture:** Add standalone scripts under `scripts/` that reset the existing MuJoCo model to keyframe 0, apply one isolated control mechanism at a time, and print numeric evidence. Do not change controller behavior while collecting evidence. Keep diagnostics small, deterministic, and runnable from the repo root.

**Tech Stack:** Python, MuJoCo, NumPy/JAX where existing controller code requires JAX.

---

## File Structure

- Create: `scripts/check_wbc_torque_sign.py` — compares one-step response for `tau = J^T f` versus `tau = -J^T f` using the existing contact Jacobian.
- Create: `scripts/check_hip_roll_sign.py` — applies same-sign and opposite-sign hip-roll torque patterns and logs roll/contact response.
- Create: `scripts/check_controller_isolation_modes.py` — runs short WBC-only, posture-only, and combined-controller modes to separate tracking failure from WBC instability.
- Do not modify production controller files during diagnostics unless the evidence later justifies a fix.

---

### Task 1: Save the diagnostic plan

**Files:**
- Create: `docs/superpowers/plans/2026-05-20-hierarchical-controller-diagnostics.md`

- [ ] **Step 1: Write this plan file**

Create the plan at the exact path above.

- [ ] **Step 2: Continue with read-only inspection**

Inspect:
- `wheeled_biped/controllers/contact_jacobian.py`
- `wheeled_biped/controllers/integrated_wbc.py`
- `wheeled_biped/controllers/posture_regularizer.py`
- `wheeled_biped/controllers/centroidal_state_estimator.py`
- `assets/robot/wheeled_biped_real.xml`

Expected: identify actuator ordering, joint indices, keyframe state, and existing helper APIs.

---

### Task 2: WBC torque-sign diagnostic

**Files:**
- Create: `scripts/check_wbc_torque_sign.py`

- [ ] **Step 1: Implement reset and measurement helpers**

Use MuJoCo to load `assets/robot/wheeled_biped_real.xml`, reset keyframe 0, zero velocities/accelerations, run `mj_forward`, and measure:
- roll/pitch from base quaternion
- actuated joint positions/velocities
- total vertical contact force using `mj_contactForce`

- [ ] **Step 2: Implement two torque-sign cases**

Compute wheel contact Jacobians with `ContactJacobian.map_contact_forces_to_torques(...)` using symmetric vertical force near half robot weight per wheel. Run:
- `positive_jtf`: apply `tau = mapped_tau`
- `negative_jtf`: apply `tau = -mapped_tau`

Use a small number of steps, reset between cases, and print deltas from baseline.

- [ ] **Step 3: Verify script runs**

Run:

```bash
python scripts/check_wbc_torque_sign.py
```

Expected: script exits 0 and prints both cases with contact-force, roll/pitch, and joint deltas.

---

### Task 3: Hip-roll sign diagnostic

**Files:**
- Create: `scripts/check_hip_roll_sign.py`

- [ ] **Step 1: Implement four hip-roll torque patterns**

Run from the same keyframe reset:
- `same_positive`: left `+tau`, right `+tau`
- `same_negative`: left `-tau`, right `-tau`
- `opposite_left_positive`: left `+tau`, right `-tau`
- `opposite_right_positive`: left `-tau`, right `+tau`

Use hip-roll indices 0 and 5 in the 10-actuator control vector.

- [ ] **Step 2: Measure response**

For each case, log:
- final roll delta
- roll-rate estimate
- left/right vertical contact-force deltas
- hip-roll joint deltas

- [ ] **Step 3: Verify script runs**

Run:

```bash
python scripts/check_hip_roll_sign.py
```

Expected: script exits 0 and prints numeric response for all four cases.

---

### Task 4: Controller isolation diagnostic

**Files:**
- Create: `scripts/check_controller_isolation_modes.py`

- [ ] **Step 1: Reuse existing controller setup**

Instantiate the same components as `scripts/simulate_hierarchical_controller.py` where needed:
- `CentroidalStateEstimator`
- `CapturePointEstimator`
- `IntegratedWBC`
- `PostureRegularizer`

Keep the run short, deterministic, and headless.

- [ ] **Step 2: Implement modes**

Run reset between modes:
- `posture_only`: `tau_total = tau_posture`
- `wbc_only`: `tau_total = tau_wbc`
- `combined`: `tau_total = tau_wbc + tau_posture`
- `combined_inverse_dynamics`: `tau_total = tau_wbc + tau_posture + qfrc_inverse[6:16]`

Apply the same torque clipping and optional torque-rate limiting in every mode.

- [ ] **Step 3: Print comparable metrics**

For each mode, print:
- steps completed
- pitch/roll min/max
- joint position error norm min/max/final
- max torque norm
- contact loss count
- termination reason if any

- [ ] **Step 4: Verify script runs**

Run:

```bash
python scripts/check_controller_isolation_modes.py
```

Expected: script exits 0 and prints one summary per mode.

---

### Task 5: Summarize evidence and stop before fixing

**Files:**
- No controller modifications in this task.

- [ ] **Step 1: Run all diagnostics**

Run:

```bash
python scripts/check_wbc_torque_sign.py
python scripts/check_hip_roll_sign.py
python scripts/check_controller_isolation_modes.py
```

- [ ] **Step 2: Report root-cause evidence**

Summarize which hypothesis is supported:
- WBC global sign issue
- hip-roll sign convention issue
- posture target/gain issue
- missing inverse dynamics issue
- torque-rate jerk issue
- contact-force artifact issue

- [ ] **Step 3: Do not fix without explicit root-cause decision**

Stop after reporting evidence unless the user explicitly asks to apply the fix.
