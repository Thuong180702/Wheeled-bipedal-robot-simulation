# K2 Phase 1.5 — Dynamics Truth Layer Refinement Report

**Generated:** 2026-08-01T03:56:17.689563+00:00
**Model:** `/Users/admin/Wheeled-bipedal-robot-simulation/assets/robot/wheeled_biped_real.xml`
**Total robot mass:** 8.100 kg

## 1. Executive Summary

**Phase 2A Readiness Verdict: `PARTIAL_READY`**

- Model loaded with **11 joints**, **12 bodies**, **10 actuators**
- State snapshot: qpos/qvel finite → **True/True**
- Mass matrix available via CPU MuJoCo → **True**
- COM position plausible → **True**
- Contacts after passive settle: **2** active contacts
- Jacobian FD validation: **5 pass**, 0 warn, 0 fail
- Torque sign probes: **10 MEASURED**, 0 AMBIGUOUS, 0 MISSING

## 2. Controller Non-Modification Statement

> **Controller code and `K2_JAX_DEDICATED_DEFAULT_V3` parameters were not modified.**
>
> This audit is purely diagnostic. No controller profiles were loaded or executed.
> No training, promotion, or regression evaluation was run.

**Changed files (Phase 1.5):**

- `wheeled_biped/dynamics/torque_sign_checks.py` (modified — bias-subtracted probe)
- `scripts/phase1_dynamics_truth_audit.py` (modified — Phase 1.5 paths, actuator limit fix, delta-based torque sign table)
- `tests/test_phase1_5_dynamics_truth_refinement.py` (new)
- `docs/validation/k2_phase1_5_dynamics_truth_refinement.md` (new — this report)
- `docs/validation/k2_phase1_5_dynamics_truth_refinement.json` (new)

**Files NOT touched:**

- `wheeled_biped/controllers/k2_jax_controller.py`
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- All controller profile definitions and promotion scripts
- All config YAML files
- `K2_JAX_DEDICATED_DEFAULT_V3` (no profile changes)

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
| `l_hip_roll_motor` | -30.0 | 30.0 | -60.0 | 60.0 | none |
| `l_hip_yaw_motor` | -30.0 | 30.0 | -60.0 | 60.0 | none |
| `l_hip_pitch_motor` | -150.0 | 150.0 | -150.0 | 150.0 | none |
| `l_knee_motor` | -150.0 | 150.0 | -150.0 | 150.0 | none |
| `l_wheel_motor` | -30.0 | 30.0 | -60.0 | 60.0 | none |
| `r_hip_roll_motor` | -30.0 | 30.0 | -60.0 | 60.0 | none |
| `r_hip_yaw_motor` | -30.0 | 30.0 | -60.0 | 60.0 | none |
| `r_hip_pitch_motor` | -150.0 | 150.0 | -150.0 | 150.0 | none |
| `r_knee_motor` | -150.0 | 150.0 | -150.0 | 150.0 | none |
| `r_wheel_motor` | -30.0 | 30.0 | -60.0 | 60.0 | none |

- Symmetric limits: ✅ all symmetric
- Any zero-range: ✅ none

## 7. State Snapshot Summary

- **Base position:** [0.0000, 0.0000, 0.5319] (world)
- **Base quaternion:** [1.0000, 0.0000, 0.0000, 0.0000]
- **Joint positions:** ['0.0000', '0.0000', '0.9261', '1.7484', '0.0000', '0.0000', '0.0000', '0.9261', '1.7484', '0.0000']
- **qpos finite:** True
- **qvel finite:** True
- **COM position:** [-7.408496605625179e-08, -0.013535414193508712, 0.40020012440172825]
- **COM velocity:** [0.0, 0.0, 0.0]

## 8. COM Check

- **COM position:** [-7.408496605625179e-08, -0.013535414193508712, 0.40020012440172825]
- **Base Z:** 0.531943
- **COM relative to base Z:** -0.1317428755982718
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
| `l_wheel_link` | 6.280104e-10 | 1.666443e-09 | **PASS** |
| `r_wheel_link` | 6.279071e-10 | 1.667279e-09 | **PASS** |
| `l_knee_link` | 3.640372e-10 | 1.666392e-09 | **PASS** |
| `r_knee_link` | 3.643743e-10 | 1.668122e-09 | **PASS** |

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

**Probe method:** Bias-subtracted (zero/+probe/−probe) with escalation up to 25% of actuator ctrl range limit.

| Joint | Actuator | qacc(0) | qacc(+) | qacc(−) | Δ+ | Δ− | Δ± | Probe (Nm) | Δ-Consistent | Convention | Outcome |
|-------|----------|---------|---------|---------|----|----|----|-----------|-------------|------------|---------|
| `l_hip_roll` | `l_hip_roll_motor` | -159.4082 | 23.7136 | -342.5300 | 183.1218 | -183.1218 | 366.2436 | 7.5 | ✅ | positive_ctrl_increases_joint_acceleration | **MEASURED** |
| `l_hip_yaw` | `l_hip_yaw_motor` | -13.3692 | 50.4281 | -77.1664 | 63.7973 | -63.7973 | 127.5945 | 7.5 | ✅ | positive_ctrl_increases_joint_acceleration | **MEASURED** |
| `l_hip_pitch` | `l_hip_pitch_motor` | 301.6036 | 479.9833 | 123.2239 | 178.3797 | -178.3797 | 356.7594 | 10.0 | ✅ | positive_ctrl_increases_joint_acceleration | **MEASURED** |
| `l_knee` | `l_knee_motor` | 1559.8691 | 1614.0214 | 1505.7167 | 54.1523 | -54.1523 | 108.3046 | 10.0 | ✅ | positive_ctrl_increases_joint_acceleration | **MEASURED** |
| `l_wheel` | `l_wheel_motor` | -697.7589 | 63.0950 | -1458.6129 | 760.8540 | -760.8540 | 1521.7080 | 7.5 | ✅ | positive_ctrl_increases_joint_acceleration | **MEASURED** |
| `r_hip_roll` | `r_hip_roll_motor` | 158.9964 | 342.1170 | -24.1243 | 183.1207 | -183.1207 | 366.2413 | 7.5 | ✅ | positive_ctrl_increases_joint_acceleration | **MEASURED** |
| `r_hip_yaw` | `r_hip_yaw_motor` | 15.7963 | 79.5951 | -48.0026 | 63.7989 | -63.7989 | 127.5978 | 7.5 | ✅ | positive_ctrl_increases_joint_acceleration | **MEASURED** |
| `r_hip_pitch` | `r_hip_pitch_motor` | 293.4223 | 471.7942 | 115.0505 | 178.3718 | -178.3718 | 356.7437 | 10.0 | ✅ | positive_ctrl_increases_joint_acceleration | **MEASURED** |
| `r_knee` | `r_knee_motor` | 1542.0351 | 1596.1878 | 1487.8824 | 54.1527 | -54.1527 | 108.3053 | 10.0 | ✅ | positive_ctrl_increases_joint_acceleration | **MEASURED** |
| `r_wheel` | `r_wheel_motor` | -689.4379 | 71.3225 | -1450.1982 | 760.7604 | -760.7604 | 1521.5208 | 7.5 | ✅ | positive_ctrl_increases_joint_acceleration | **MEASURED** |

**Phase 1.5 torque sign summary:** 10 MEASURED, 0 AMBIGUOUS

> **Note:** Left/right mirrored joints may have differing sign conventions due to
> physical mirroring of the kinematic tree. This is expected and NOT a controller bug.
> All outcomes are labeled MEASURED/AMBIGUOUS/MISSING, not pass/fail.

## 12. Phase 1 Comparison

| Metric | Phase 1 | Phase 1.5 | Change |
|--------|---------|-----------|--------|
| Torque signs MEASURED | 6 | 10 | +4 |
| Torque signs AMBIGUOUS | 4 | 0 | -4 |
| Actuator `zero_not_in_range` false labels | Yes (bug) | No (fixed) | ✅ fixed |
| Probe method | Absolute qacc signs | Bias-subtracted deltas | ✅ improved |
| Probe escalation | None | Up to 25% actuator limit | ✅ added |

**All 10 torque signs now MEASURED.** ✅

The bias-subtracted delta probe resolved the 4 previously ambiguous
joints (l_hip_pitch, l_knee, r_hip_pitch, r_knee) that were gravity-dominated
with the Phase 1 absolute-sign-only measurement.

## 13. Mass Matrix (CPU MuJoCo)

- **Available via CPU:** True
- **Shape:** N/A
- **Finite:** False
- **Symmetric:** N/A
- **Diagonal positive:** N/A
- **Condition number:** N/A
- **Error:** mj_fullM(): incompatible function arguments. The following argument types are supported:
    1. (m: mujoco._structs.MjModel, d: mujoco._structs.MjData, dst: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.c_contiguous"]) -> None

Invoked with: <mujoco._structs.MjModel object at 0x10f3f4db0>, array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([ 8.10000000e+00,  8.10000000e+00,  0.00000000e+00,  8.10000000e+00,
        0.00000000e+00,  0.00000000e+00,  4.00904968e-01, -1.09636855e-01,
        1.06711729e+00,  0.00000000e+00,  4.34027382e-01, -7.09689089e-07,
        6.00088225e-07,  0.00000000e+00, -1.06711729e+00,  1.12640604e-01,
       -5.79790507e-02,  8.78369396e-06,  0.00000000e+00, -6.00088225e-07,
        1.09636855e-01,  4.20094085e-02,  1.92504032e-02, -2.89918678e-02,
        1.17975735e-03, -2.19266091e-07, -2.39912850e-02,  5.48165708e-02,
        1.05412464e-01, -2.25454686e-02, -2.25457238e-02,  1.14188793e-01,
        2.22733541e-03,  2.39941350e-02, -2.14168882e-06, -3.16941746e-01,
        9.73630029e-02, -8.87307957e-04, -4.13635309e-03, -3.29917528e-02,
       -1.29710873e-02, -1.10999477e-01,  1.05067865e-01, -2.50920123e-01,
        3.90859327e-06,  5.08783200e-02, -2.44673575e-02, -2.06440945e-03,
        1.91736301e-03,  1.28425600e-02, -1.38271697e-02,  3.56744561e-02,
        1.02286936e-01,  9.50028560e-02, -1.92797877e-07,  8.18832000e-03,
        1.88320000e-04, -1.88320000e-04, -3.09617516e-11, -8.71879362e-11,
       -8.40487051e-10,  1.28706089e-09,  1.88320000e-04,  6.85275742e-14,
       -1.66220341e-13,  1.44185902e-18,  4.20099720e-02,  1.92505892e-02,
       -2.89871829e-02, -1.17816074e-03, -2.19281328e-07,  2.39946984e-02,
        5.48202841e-02,  1.05383483e-01, -2.25402110e-02, -2.25402131e-02,
        1.14152638e-01, -2.22833334e-03, -2.39925420e-02,  9.59750458e-08,
       -3.16866466e-01,  9.73621294e-02,  8.86787305e-04,  4.13624843e-03,
        3.29925996e-02,  1.29689801e-02, -1.10987270e-01,  1.05062798e-01,
       -2.50921190e-01,  4.20252032e-07,  5.08783200e-02, -2.44669207e-02,
        2.06431826e-03, -1.91719229e-03, -1.28423604e-02,  1.38276006e-02,
        3.56684594e-02,  1.02289293e-01,  9.50003184e-02,  2.28000371e-07,
        8.18832000e-03,  1.88320000e-04, -1.88320000e-04, -8.31476194e-10,
        1.10573630e-09,  3.52446962e-10, -8.31499237e-10,  1.88320000e-04,
       -1.61916866e-13,  8.11970238e-14,  6.61550758e-19])

- **Available via MJX:** False
- **MJX note:** mj_fullM / mjData.qM are CPU-only. MJX does not expose the mass matrix directly. For future real-time/JAX WBC, the mass matrix must be computed via a separate JAX port or accessed through the MJX C++ internals.

## 14. Limitations

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

## 15. Phase 2A Readiness Verdict

**Verdict: `PARTIAL_READY`**

All structural checks pass, but one or more torque signs remain ambiguous
or key physics quantities (mass matrix, Jacobians, contact forces) are only
available through CPU MuJoCo and must be ported to JAX/MJX before they can
be used in a real-time QP-WBC pipeline.

## 16. Remaining Items Before Phase 2A

- ⚠️ Mass matrix contains non-finite values
- ⚠️ Mass matrix error: mj_fullM(): incompatible function arguments. The following argument types are supported:
    1. (m: mujoco._structs.MjModel, d: mujoco._structs.MjData, dst: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.c_contiguous"]) -> None

Invoked with: <mujoco._structs.MjModel object at 0x10f3f4db0>, array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([ 8.10000000e+00,  8.10000000e+00,  0.00000000e+00,  8.10000000e+00,
        0.00000000e+00,  0.00000000e+00,  4.00904968e-01, -1.09636855e-01,
        1.06711729e+00,  0.00000000e+00,  4.34027382e-01, -7.09689089e-07,
        6.00088225e-07,  0.00000000e+00, -1.06711729e+00,  1.12640604e-01,
       -5.79790507e-02,  8.78369396e-06,  0.00000000e+00, -6.00088225e-07,
        1.09636855e-01,  4.20094085e-02,  1.92504032e-02, -2.89918678e-02,
        1.17975735e-03, -2.19266091e-07, -2.39912850e-02,  5.48165708e-02,
        1.05412464e-01, -2.25454686e-02, -2.25457238e-02,  1.14188793e-01,
        2.22733541e-03,  2.39941350e-02, -2.14168882e-06, -3.16941746e-01,
        9.73630029e-02, -8.87307957e-04, -4.13635309e-03, -3.29917528e-02,
       -1.29710873e-02, -1.10999477e-01,  1.05067865e-01, -2.50920123e-01,
        3.90859327e-06,  5.08783200e-02, -2.44673575e-02, -2.06440945e-03,
        1.91736301e-03,  1.28425600e-02, -1.38271697e-02,  3.56744561e-02,
        1.02286936e-01,  9.50028560e-02, -1.92797877e-07,  8.18832000e-03,
        1.88320000e-04, -1.88320000e-04, -3.09617516e-11, -8.71879362e-11,
       -8.40487051e-10,  1.28706089e-09,  1.88320000e-04,  6.85275742e-14,
       -1.66220341e-13,  1.44185902e-18,  4.20099720e-02,  1.92505892e-02,
       -2.89871829e-02, -1.17816074e-03, -2.19281328e-07,  2.39946984e-02,
        5.48202841e-02,  1.05383483e-01, -2.25402110e-02, -2.25402131e-02,
        1.14152638e-01, -2.22833334e-03, -2.39925420e-02,  9.59750458e-08,
       -3.16866466e-01,  9.73621294e-02,  8.86787305e-04,  4.13624843e-03,
        3.29925996e-02,  1.29689801e-02, -1.10987270e-01,  1.05062798e-01,
       -2.50921190e-01,  4.20252032e-07,  5.08783200e-02, -2.44669207e-02,
        2.06431826e-03, -1.91719229e-03, -1.28423604e-02,  1.38276006e-02,
        3.56684594e-02,  1.02289293e-01,  9.50003184e-02,  2.28000371e-07,
        8.18832000e-03,  1.88320000e-04, -1.88320000e-04, -8.31476194e-10,
        1.10573630e-09,  3.52446962e-10, -8.31499237e-10,  1.88320000e-04,
       -1.61916866e-13,  8.11970238e-14,  6.61550758e-19])

---

*Report generated by `scripts/phase1_dynamics_truth_audit.py`*
