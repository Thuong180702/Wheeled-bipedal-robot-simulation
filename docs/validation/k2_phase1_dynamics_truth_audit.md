# K2 Phase 1 — Dynamics Truth Layer Audit Report

**Generated:** 2026-07-01T23:42:38.461963+00:00
**Model:** `F:\ROBOTCUATAO\Wheeled-bipedal-robot-simulation\assets\robot\wheeled_biped_real.xml`
**Total robot mass:** 8.100 kg

## 1. Executive Summary

**Phase 2 Readiness Verdict: `PARTIAL_READY`**

- Model loaded with **11 joints**, **12 bodies**, **10 actuators**
- State snapshot: qpos/qvel finite → **True/True**
- Mass matrix available via CPU MuJoCo → **True**
- COM position plausible → **True**
- Contacts after passive settle: **2** active contacts
- Jacobian FD validation: **5 pass**, 0 warn, 0 fail
- Torque sign probes: **6 MEASURED**, 4 AMBIGUOUS, 0 MISSING

## 2. Controller Non-Modification Statement

> **Controller code and `K2_JAX_DEDICATED_DEFAULT_V3` parameters were not modified.**
>
> This audit is purely diagnostic. No controller profiles were loaded or executed.
> No training, promotion, or regression evaluation was run.

**Changed files:**

- `wheeled_biped/dynamics/__init__.py` (new)
- `wheeled_biped/dynamics/model_inspector.py` (new)
- `wheeled_biped/dynamics/jacobian_checks.py` (new)
- `wheeled_biped/dynamics/contact_inspector.py` (new)
- `wheeled_biped/dynamics/torque_sign_checks.py` (new)
- `scripts/phase1_dynamics_truth_audit.py` (new)
- `tests/test_phase1_dynamics_truth_layer.py` (new)
- `docs/validation/k2_phase1_dynamics_truth_audit.md` (new — this report)

**Files NOT touched:**

- `wheeled_biped/controllers/k2_jax_controller.py`
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- All controller profile definitions and promotion scripts
- All config YAML files

## 3. Model Dimensions

| Property | Value |
|----------|-------|
| `nq` (generalized positions) | 17 |
| `nv` (generalized velocities) | 16 |
| `nu` (actuators/controls) | 10 |
| `nbody` (bodies) | 12 |
| `njnt` (joints) | 11 |
| `ngeom` (geoms) | 29 |
| `nsite` (sites) | 8 |
| `nkey` (keyframes) | 1 |
| `nsensor` (sensors) | 25 |

## 4. Joint / Actuator Mapping

| Index | Joint | Actuator | Ctrl Range (Nm) | Force Range (Nm) |
|-------|-------|----------|-----------------|------------------|
| 0 | `l_hip_roll` | `l_hip_roll_motor` | [-30.0, 30.0] | [-60.0, 60.0] |
| 1 | `l_hip_yaw` | `l_hip_yaw_motor` | [-30.0, 30.0] | [-60.0, 60.0] |
| 2 | `l_hip_pitch` | `l_hip_pitch_motor` | [-150.0, 150.0] | [-150.0, 150.0] |
| 3 | `l_knee` | `l_knee_motor` | [-150.0, 150.0] | [-150.0, 150.0] |
| 4 | `l_wheel` | `l_wheel_motor` | [-30.0, 30.0] | [-60.0, 60.0] |
| 5 | `r_hip_roll` | `r_hip_roll_motor` | [-30.0, 30.0] | [-60.0, 60.0] |
| 6 | `r_hip_yaw` | `r_hip_yaw_motor` | [-30.0, 30.0] | [-60.0, 60.0] |
| 7 | `r_hip_pitch` | `r_hip_pitch_motor` | [-150.0, 150.0] | [-150.0, 150.0] |
| 8 | `r_knee` | `r_knee_motor` | [-150.0, 150.0] | [-150.0, 150.0] |
| 9 | `r_wheel` | `r_wheel_motor` | [-30.0, 30.0] | [-60.0, 60.0] |

**Actuator count: 10** — MATCH (expected 10)

**Joint name verification:**
- ✅ `l_hip_roll` — id=1, type=hinge
- ✅ `l_hip_yaw` — id=2, type=hinge
- ✅ `l_hip_pitch` — id=3, type=hinge
- ✅ `l_knee` — id=4, type=hinge
- ✅ `l_wheel` — id=5, type=hinge
- ✅ `r_hip_roll` — id=6, type=hinge
- ✅ `r_hip_yaw` — id=7, type=hinge
- ✅ `r_hip_pitch` — id=8, type=hinge
- ✅ `r_knee` — id=9, type=hinge
- ✅ `r_wheel` — id=10, type=hinge

## 5. Body Mapping

| Body Name | ID | Parent | Mass (kg) |
|-----------|-----|--------|-----------|
| `torso` | 1 | `world` | 2.500 |
| `l_thigh` | 4 | `l_hip_yaw_link` | 0.800 |
| `r_thigh` | 9 | `r_hip_yaw_link` | 0.800 |
| `l_knee_link` | 5 | `l_thigh` | 0.600 |
| `r_knee_link` | 10 | `r_thigh` | 0.600 |
| `l_hip_roll_link` | 2 | `torso` | 0.500 |
| `r_hip_roll_link` | 7 | `torso` | 0.500 |
| `l_hip_yaw_link` | 3 | `l_hip_roll_link` | 0.800 |
| `r_hip_yaw_link` | 8 | `r_hip_roll_link` | 0.800 |
| `l_wheel_link` | 6 | `l_knee_link` | 0.100 |
| `r_wheel_link` | 11 | `r_knee_link` | 0.100 |

✅ All mandatory body names found.

## 6. Actuator Limits

| Actuator | Ctrl Min | Ctrl Max | Force Min | Force Max | Issues |
|----------|----------|----------|-----------|-----------|--------|
| `l_hip_roll_motor` | -30.0 | 30.0 | -60.0 | 60.0 | zero_not_in_range |
| `l_hip_yaw_motor` | -30.0 | 30.0 | -60.0 | 60.0 | zero_not_in_range |
| `l_hip_pitch_motor` | -150.0 | 150.0 | -150.0 | 150.0 | zero_not_in_range |
| `l_knee_motor` | -150.0 | 150.0 | -150.0 | 150.0 | zero_not_in_range |
| `l_wheel_motor` | -30.0 | 30.0 | -60.0 | 60.0 | zero_not_in_range |
| `r_hip_roll_motor` | -30.0 | 30.0 | -60.0 | 60.0 | zero_not_in_range |
| `r_hip_yaw_motor` | -30.0 | 30.0 | -60.0 | 60.0 | zero_not_in_range |
| `r_hip_pitch_motor` | -150.0 | 150.0 | -150.0 | 150.0 | zero_not_in_range |
| `r_knee_motor` | -150.0 | 150.0 | -150.0 | 150.0 | zero_not_in_range |
| `r_wheel_motor` | -30.0 | 30.0 | -60.0 | 60.0 | zero_not_in_range |

- Symmetric limits: ✅ all symmetric
- Any zero-range: ✅ none

## 7. State Snapshot Summary

- **Base position:** [0.0000, 0.0000, 0.5319] (world)
- **Base quaternion:** [1.0000, 0.0000, 0.0000, 0.0000]
- **Joint positions:** ['0.0000', '0.0000', '0.9261', '1.7484', '0.0000', '0.0000', '0.0000', '0.9261', '1.7484', '0.0000']
- **qpos finite:** True
- **qvel finite:** True
- **COM position:** [-7.408496604254533e-08, -0.013535414193508717, 0.40020012440172836]
- **COM velocity:** [0.0, 0.0, 0.0]

## 8. COM Check

- **COM position:** [-7.408496604254533e-08, -0.013535414193508717, 0.40020012440172836]
- **Base Z:** 0.531943
- **COM relative to base Z:** -0.1317428755982717
- **Plausible:** True

## 9. Contact Inspection Summary

- **Settle steps:** 50 (0.100 s)
- **Active contacts:** 2
- **Left wheel in contact:** True
- **Right wheel in contact:** True
- **Total contact force (world):** ['-0.004', '-2.625', '50.169']

| # | Geom 1 | Geom 2 | Body 1 | Body 2 | Force World (N) | Dist |
|---|--------|--------|--------|--------|-----------------|------|
| 0 | `floor` | `l_wheel_collision` | `world` | `l_wheel_link` | [-2.355, -1.312, 25.084] | -0.000027 |
| 1 | `floor` | `r_wheel_collision` | `world` | `r_wheel_link` | [2.351, -1.312, 25.085] | -0.000027 |

## 10. Jacobian Validation

### 10.1 Analytic Jacobians

| Target | Type | ID | JacP Shape | JacP Rank | Finite |
|--------|------|-----|------------|-----------|--------|
| `torso` | body | 1 | [3, 16] | 3 | True |
| `l_wheel_link` | body | 6 | [3, 16] | 3 | True |
| `r_wheel_link` | body | 11 | [3, 16] | 3 | True |
| `l_knee_link` | body | 5 | [3, 16] | 3 | True |
| `r_knee_link` | body | 10 | [3, 16] | 3 | True |

### 10.2 Finite-Difference Validation

Free-joint columns (v[0:6]) are skipped — not FD-validated for position Jacobians.
Only actuated joint columns (v[6:16]) are checked.

**Thresholds:** PASS < 0.001, WARN < 0.01, FAIL ≥ 0.01

| Target | Max Abs Error | Max Rel Error | Verdict |
|--------|--------------|---------------|---------|
| `torso` | 0.000000e+00 | 0.000000e+00 | **PASS** |
| `l_wheel_link` | 6.277328e-10 | 1.665642e-09 | **PASS** |
| `r_wheel_link` | 6.280457e-10 | 1.667278e-09 | **PASS** |
| `l_knee_link` | 3.642454e-10 | 1.666764e-09 | **PASS** |
| `r_knee_link` | 3.643741e-10 | 1.667096e-09 | **PASS** |

### 10.3 Per-Joint FD Detail (torso)

| Joint | Abs Error | Rel Error | Verdict |
|-------|-----------|-----------|---------|
| `l_hip_roll` | 0.000000e+00 | 0.000000e+00 | PASS |
| `l_hip_yaw` | 0.000000e+00 | 0.000000e+00 | PASS |
| `l_hip_pitch` | 0.000000e+00 | 0.000000e+00 | PASS |
| `l_knee` | 0.000000e+00 | 0.000000e+00 | PASS |
| `l_wheel` | 0.000000e+00 | 0.000000e+00 | PASS |
| `r_hip_roll` | 0.000000e+00 | 0.000000e+00 | PASS |
| `r_hip_yaw` | 0.000000e+00 | 0.000000e+00 | PASS |
| `r_hip_pitch` | 0.000000e+00 | 0.000000e+00 | PASS |
| `r_knee` | 0.000000e+00 | 0.000000e+00 | PASS |
| `r_wheel` | 0.000000e+00 | 0.000000e+00 | PASS |

## 11. Torque Sign Validation

**Probe torque:** ±10.0 Nm, 1 simulation step

| Joint | Actuator | qacc(+) | qacc(-) | Consistent | Convention | Outcome |
|-------|----------|---------|---------|------------|------------|---------|
| `l_hip_roll` | `l_hip_roll_motor` | 84.083623 | -403.570644 | ✅ | positive_ctrl_→_positive_qacc | **MEASURED** |
| `l_hip_yaw` | `l_hip_yaw_motor` | 71.693837 | -98.432167 | ✅ | positive_ctrl_→_positive_qacc | **MEASURED** |
| `l_hip_pitch` | `l_hip_pitch_motor` | 479.983299 | 123.223905 | ❌ | positive_ctrl_→_positive_qacc | **AMBIGUOUS** |
| `l_knee` | `l_knee_motor` | 1614.021364 | 1505.716748 | ❌ | positive_ctrl_→_positive_qacc | **AMBIGUOUS** |
| `l_wheel` | `l_wheel_motor` | 316.713043 | -1712.230903 | ✅ | positive_ctrl_→_positive_qacc | **MEASURED** |
| `r_hip_roll` | `r_hip_roll_motor` | 403.157234 | -84.497356 | ✅ | positive_ctrl_→_positive_qacc | **MEASURED** |
| `r_hip_yaw` | `r_hip_yaw_motor` | 100.861437 | -69.268925 | ✅ | positive_ctrl_→_positive_qacc | **MEASURED** |
| `r_hip_pitch` | `r_hip_pitch_motor` | 471.794157 | 115.050487 | ❌ | positive_ctrl_→_positive_qacc | **AMBIGUOUS** |
| `r_knee` | `r_knee_motor` | 1596.187770 | 1487.882447 | ❌ | positive_ctrl_→_positive_qacc | **AMBIGUOUS** |
| `r_wheel` | `r_wheel_motor` | 324.909321 | -1703.785024 | ✅ | positive_ctrl_→_positive_qacc | **MEASURED** |

> **Note:** Left/right mirrored joints may have differing sign conventions due to
> physical mirroring of the kinematic tree. This is expected and NOT a controller bug.
> All outcomes are labeled MEASURED/CONSISTENT/AMBIGUOUS, not pass/fail.

## 12. Mass Matrix (CPU MuJoCo)

- **Available via CPU:** True
- **Shape:** [16, 16]
- **Finite:** True
- **Symmetric:** True
- **Diagonal positive:** True
- **Condition number:** 1010.7165073632176

- **Available via MJX:** False
- **MJX note:** mj_fullM / mjData.qM are CPU-only. MJX does not expose the mass matrix directly. For future real-time/JAX WBC, the mass matrix must be computed via a separate JAX port or accessed through the MJX C++ internals.

## 13. Limitations

### What MuJoCo CPU exposes (used in this audit):

- ✅ `mj_jac` — task-space Jacobians for bodies and sites
- ✅ `mj_fullM` / `data.qM` — mass matrix
- ✅ `mj_contactForce` — per-contact force vectors
- ✅ `data.contact` — contact geometry pairs, positions, normals, distances
- ✅ `data.qpos`, `data.qvel`, `data.qacc` — full state
- ✅ `data.xpos`, `data.xmat`, `data.subtree_com` — body poses and COM
- ✅ `data.qfrc_bias`, `data.qfrc_passive`, `data.qfrc_actuator` — force components

### What MJX exposes (for future real-time/JAX integration):

- ✅ `qpos`, `qvel`, `qacc` — full state vectors
- ✅ `xpos`, `xmat` — body poses
- ✅ `ctrl`, `act` — control/activation
- ⚠️ `contact` — contact array (limited fields vs CPU; `dist` and `pos` available, `frame` partial)
- ❌ `mj_jac` — **not available** in MJX. Jacobians must be hand-computed or ported.
- ❌ `mj_fullM` — **not available** in MJX. Mass matrix must be computed via CRBA port.
- ❌ `mj_contactForce` — **not available** in MJX. Contact forces must be computed from constraint solver outputs.
- ❌ `subtree_com` — **not available** in MJX. COM must be computed from body masses and poses.

### Impact on QP-WBC development:

- **Jacobians:** Must be hand-computed from kinematics or a separate JAX CRBA/kinematics port.
- **Mass matrix:** Must be computed via JAX Composite Rigid Body Algorithm (CRBA).
- **Contact forces:** MJX constraint solver outputs (`efc_force`, `efc_J`) are internal; contact force extraction requires understanding the constraint model.
- **Bias forces:** `qfrc_bias` is available in MJX as `data.qfrc_bias` — this is a significant positive for QP-WBC.

These limitations mean that a pure-MJX/JIT WBC requires a dedicated dynamics/kinematics port, not just wrapping existing MuJoCo API calls. This audit provides the ground-truth reference for validating such a port.

## 14. Phase 2 Readiness Verdict

**Verdict: `PARTIAL_READY`**

All structural checks pass, but key physics quantities (mass matrix, Jacobians,
contact forces) are only available through CPU MuJoCo and must be ported to
JAX/MJX before they can be used in a real-time QP-WBC pipeline. The ground
truth is established and can be used to validate any future JAX port.

## 15. Required Fixes Before Phase 2

- ⚠️ Torque sign AMBIGUOUS for `l_hip_pitch`
- ⚠️ Torque sign AMBIGUOUS for `l_knee`
- ⚠️ Torque sign AMBIGUOUS for `r_hip_pitch`
- ⚠️ Torque sign AMBIGUOUS for `r_knee`

---

*Report generated by `scripts/phase1_dynamics_truth_audit.py`*
