# Hierarchical Controller Seven-Error Fix Plan

> **For agentic workers:** Continue using systematic debugging discipline. Fix one root-cause group at a time, verify after each phase, and do not apply bundled tuning changes that hide which change mattered.

**Goal:** Fix the seven identified controller errors causing the wheeled-biped hierarchical controller to survive only briefly with large roll/pitch oscillation, poor posture tracking, hip-roll saturation, contact artifacts, and unstable inverse-dynamics behavior.

**Current baseline evidence:**

```text
combined roll_range ≈ [-24.5, +14.6] deg
combined pitch_range ≈ [-21.9, +1.7] deg
combined joint_error_final ≈ 1.31
wbc_only fails around roll ≈ -46.6 deg
posture_only collapses around 16 steps
hip-roll max torque ≈ 57 Nm
```

**Architecture:** Keep production controller changes minimal and evidence-driven. Use the existing standalone diagnostics under `scripts/` to compare behavior after each phase.

---

## Step A — Lock baseline diagnostics

**Purpose:** Save current behavior before any fix so every later change can be compared against a known baseline.

**Commands:**

```bash
python scripts/check_hip_roll_sign.py
python scripts/check_wbc_torque_sign.py
python scripts/check_motor_joint_coupling.py
python scripts/check_controller_isolation_modes.py
python scripts/check_wbc_variants.py
python scripts/simulate_hierarchical_controller.py
```

**Record:**

- roll/pitch ranges
- termination reasons
- joint error final
- max torque norm/rate
- hip-roll torque saturation
- contact-force artifacts
- selected WBC variant behavior

**Pass condition for this step:** All diagnostics run and current baseline values are recorded.

---

## Phase 1 — Fix hip-roll sign convention

**Error fixed:** `SimpleForceDistributor` assumes same-sign hip-roll torques create roll moment, but diagnostics show opposite-sign hip-roll produces the measurable roll response for the current MJCF.

**Target files:**

- `wheeled_biped/controllers/simple_force_distributor.py`
- `scripts/check_hip_roll_sign.py`
- optional tests under `tests/`

**Evidence:**

```text
same_positive roll_delta=-0.0011 deg
opposite_left_positive roll_delta=-0.0104 deg
opposite_right_positive roll_delta=+0.0097 deg
```

**Change:**

Add one explicit helper for the measured model convention and use it in all force-distribution branches:

```python
def _roll_moment_to_hip_roll_torque(self, mx: Array) -> Array:
    ...
```

Use it in:

- both-wheel contact branch
- single-contact recovery branch
- no-contact anticipatory branch
- legacy `distribute_wrench`

**Verification:**

```bash
python scripts/check_hip_roll_sign.py
python scripts/check_wbc_variants.py
python scripts/check_controller_isolation_modes.py
```

**Pass criteria:**

```text
WBC variant with corrected hip-roll reduces early roll drift.
wbc_only no longer rapidly drives roll monotonically toward -40 deg.
combined roll range improves from about [-24.5, +14.6] deg.
```

---

## Phase 2 — Fix WBC force-to-torque sign consistency

**Error fixed:** `contact_jacobian.py` documents `tau = J.T @ f`, but `IntegratedWBC` globally negates the mapped torque. Diagnostics show a blind global sign flip collapses height, so contact-force torque and direct hip-roll torque must be separated.

**Target files:**

- `wheeled_biped/controllers/integrated_wbc.py`
- `wheeled_biped/controllers/contact_jacobian.py`
- `scripts/check_wbc_torque_sign.py`
- `scripts/check_wbc_variants.py`

**Change:**

Replace global negation of mixed contact + hip torque with explicit assembly:

```python
tau_contact = self.contact_jacobian.map_contact_forces_to_torques(
    mj_data, f_left, f_right, tau_hip_roll=None
)
tau_hip = self._build_direct_hip_roll_torque(tau_hip_roll)
tau_wbc_raw = contact_sign * tau_contact + hip_sign * tau_hip
```

Add helper:

```python
def _build_direct_hip_roll_torque(self, tau_hip_roll: Array) -> Array:
    tau = jnp.zeros(10)
    tau = tau.at[0].set(tau_hip_roll[0])
    tau = tau.at[5].set(tau_hip_roll[1])
    return tau
```

**Verification:**

```bash
python scripts/check_wbc_torque_sign.py
python scripts/check_wbc_variants.py
python scripts/simulate_hierarchical_controller.py
```

**Pass criteria:**

```text
No immediate height collapse.
Roll does not monotonically drift to one side in first 30 steps.
100-step rollout has smaller roll range than current [-24.5, +14.6] deg.
Wrench error does not explode.
```

---

## Phase 3 — Reduce hip-roll saturation and overcorrection

**Error fixed:** WBC generates huge hip-roll torques and over-corrects roll.

**Target files:**

- `wheeled_biped/controllers/centroidal_wrench_computer.py`
- `wheeled_biped/controllers/integrated_wbc.py`
- `wheeled_biped/controllers/simple_force_distributor.py`
- `scripts/simulate_hierarchical_controller.py`

**Change:**

Add explicit roll-moment limiting and reduce roll gains only after sign consistency is fixed.

Initial conservative targets:

```text
max_roll_moment: 20–30 Nm
tau_hip_roll_max: 10–20 Nm
k_roll: reduce from 200 to 40–80
k_roll_rate: reduce from 40 to 8–20
k_roll_integral: disable initially, 0.0
```

Add telemetry:

- desired `Mx`
- clipped `Mx`
- hip-roll saturation flag
- hip-roll torque left/right
- roll error
- roll rate

**Verification:**

```bash
python scripts/check_wbc_variants.py
python scripts/simulate_hierarchical_controller.py
```

**Pass criteria:**

```text
hip-roll max torque < previous 57 Nm
roll does not cross from -24 deg to +14 deg in one second
roll range shrinks materially
torque-rate norm decreases
```

---

## Phase 4 — Strengthen posture/height support safely

**Error fixed:** `posture_only` collapses; posture support is too weak to hold the nominal leg configuration.

**Target files:**

- `wheeled_biped/controllers/posture_regularizer.py`
- `scripts/simulate_hierarchical_controller.py`
- `scripts/check_controller_isolation_modes.py`

**Change:**

Increase posture authority for hip pitch/knee first. Keep hip roll weak so it does not fight WBC roll control, and keep wheel posture torque zero.

Initial per-joint gain intent:

```text
hip_roll: 2–4
hip_yaw: 1–2
hip_pitch: 20–40
knee: 20–40
wheel: 0
```

**Verification:**

```bash
python scripts/check_controller_isolation_modes.py
```

**Pass criteria:**

```text
posture_only survives longer than 16 steps
combined joint_error_final materially lower than 1.31
combined roll instability does not increase
```

---

## Phase 5 — Keep inverse dynamics disabled until filtered/projected

**Error fixed:** Raw inverse dynamics destabilizes the controller when added directly.

**Target files:**

- `scripts/simulate_hierarchical_controller.py`
- `scripts/check_controller_isolation_modes.py`

**Change:**

Keep production behavior disabled:

```python
USE_INVERSE_DYNAMICS = False
```

Rename raw diagnostic mode to make it explicit:

```text
combined_raw_inverse_dynamics_diagnostic
```

Future filtered mode may only apply support torques to hip pitch/knee after Phases 1–4 are stable.

**Verification:**

```bash
python scripts/check_controller_isolation_modes.py
```

**Pass criteria:**

```text
combined remains better than raw inverse-dynamics diagnostic
no production path depends on raw qfrc_inverse
```

---

## Phase 6 — Fix torque-rate jerk and initial step discontinuity

**Error fixed:** The first control step bypasses torque-rate limiting, and max torque rate remains high.

**Target files:**

- `scripts/simulate_hierarchical_controller.py`
- `scripts/check_controller_isolation_modes.py`

**Change:**

Initialize previous torque from current control and apply rate limiting from step 0:

```python
tau_prev = np.array(data.ctrl)
tau_rate_vec = np.clip(
    (tau_total - tau_prev) / CONTROL_DT,
    -MAX_TORQUE_RATE,
    MAX_TORQUE_RATE,
)
tau_smooth = tau_prev + tau_rate_vec * CONTROL_DT
```

Only lower `MAX_TORQUE_RATE` after WBC sign fixes are stable.

**Verification:**

```bash
python scripts/check_controller_isolation_modes.py
python scripts/simulate_hierarchical_controller.py
```

**Pass criteria:**

```text
max_tau_rate decreases substantially
first-step torque no longer jumps directly to large WBC torque
height does not collapse due to overly slow torque ramp
```

---

## Phase 7 — Fix contact-force artifact and feedback timing

**Error fixed:** `mj_forward()` contact force can report large transient artifacts, e.g. 846 N versus desired 79 N.

**Target files:**

- `wheeled_biped/controllers/centroidal_state_estimator.py`
- `wheeled_biped/controllers/integrated_wbc.py`
- `scripts/simulate_hierarchical_controller.py`
- diagnostic scripts under `scripts/`

**Change:**

Force feedback should only use post-step, warmup-valid contact forces.

Rules:

```text
force_feedback_warmup_steps >= 5
contact_force_valid only after at least one mj_step
never use mj_forward contact force as diagnostic baseline
```

**Verification:**

```bash
python scripts/check_wbc_torque_sign.py
python scripts/check_hip_roll_sign.py
python scripts/simulate_hierarchical_controller.py
```

**Pass criteria:**

```text
no controller reaction to 846 N reset artifact
force feedback activates only after warmup
actual Fz after mj_step remains near physical weight unless real impact occurs
```

---

## Final completion criteria

The seven-error fix set is complete only when:

```text
1. check_motor_joint_coupling.py still shows correct actuator/joint mapping.
2. check_hip_roll_sign.py convention matches SimpleForceDistributor convention.
3. check_wbc_torque_sign.py no longer contradicts IntegratedWBC comments/implementation.
4. check_controller_isolation_modes.py improves posture_only, wbc_only, and combined behavior.
5. check_wbc_variants.py shows the selected WBC sign/hip convention is best or near-best.
6. simulate_hierarchical_controller.py completes 100 steps with materially smaller roll/pitch oscillation.
7. no fix relies on contact-force artifact from mj_forward().
```
