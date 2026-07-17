# Phase 2A — JAX Kinematics / COM / Jacobian Audit Report

**Timestamp:** 2026-07-02T01:22:16.110538+00:00
**Model:** `F:\ROBOTCUATAO\Wheeled-bipedal-robot-simulation\assets\robot\wheeled_biped_real.xml`

## 1. Executive Summary

Phase 2A ports the K2 robot's forward kinematics, COM computation, and translational Jacobians to pure JAX, validated against CPU MuJoCo ground truth from Phase 1.5.

**Verdict: `READY_FOR_PHASE_2B_MASS_MATRIX_CRBA_PORT`**

- FK: 11 PASS / 0 WARN / 0 FAIL (max pos err: 6.87e-08 m)
- COM: PASS (err: 2.56e-08 m)
- Jacobians: 7 PASS / 0 WARN / 0 FAIL (max abs err: 2.98e-08)
- JIT: FK ✓, COM ✓, Jacobian ✓

## 2. Controller Integrity

Controller code and `K2_JAX_DEDICATED_DEFAULT_V3` parameters were **not** modified.

## 3. Changed Files

| File | Status |
|------|--------|
| `wheeled_biped/dynamics/jax_kinematics.py` | **new** — JAX FK |
| `wheeled_biped/dynamics/jax_com.py` | **new** — JAX COM |
| `wheeled_biped/dynamics/jax_jacobians.py` | **new** — JAX Jacobians |
| `wheeled_biped/dynamics/__init__.py` | modified — added exports |
| `scripts/phase2a_jax_kinematics_audit.py` | **new** — this script |
| `tests/test_phase2a_jax_kinematics.py` | **new** — tests |
| `docs/validation/k2_phase2a_jax_kinematics_audit.md` | **new** — this report |
| `docs/validation/k2_phase2a_jax_kinematics_audit.json` | **new** — JSON summary |

## 4. Phase 1.5 Reference Summary

- Verdict: `READY_FOR_PHASE_2A_JAX_KINEMATICS_PORT`
- 10/10 torque signs MEASURED, 0 AMBIGUOUS
- Jacobian FD: 5/5 PASS
- Actuator limits clean
- nbody=12, njnt=11, nq=17, nv=16, nu=10

## 5. Kinematic Constants Summary

- nbody: 12
- njnt: 11
- nq: 17
- nv: 16

### Joint Order

| Index | Joint | Type | qpos_adr | dof_adr |
|-------|-------|------|----------|---------|
| 0 | root | free | 0 | 0 |
| 1 | l_hip_roll | hinge | 7 | 6 |
| 2 | l_hip_yaw | hinge | 8 | 7 |
| 3 | l_hip_pitch | hinge | 9 | 8 |
| 4 | l_knee | hinge | 10 | 9 |
| 5 | l_wheel | hinge | 11 | 10 |
| 6 | r_hip_roll | hinge | 12 | 11 |
| 7 | r_hip_yaw | hinge | 13 | 12 |
| 8 | r_hip_pitch | hinge | 14 | 13 |
| 9 | r_knee | hinge | 15 | 14 |
| 10 | r_wheel | hinge | 16 | 15 |

### Target Body IDs

| Body | ID |
|------|----|
| torso | 1 |
| l_hip_roll_link | 2 |
| l_hip_yaw_link | 3 |
| l_thigh | 4 |
| l_knee_link | 5 |
| l_wheel_link | 6 |
| r_hip_roll_link | 7 |
| r_hip_yaw_link | 8 |
| r_thigh | 9 |
| r_knee_link | 10 |
| r_wheel_link | 11 |

## 6. FK Position + Orientation Validation

Thresholds: PASS < 1e-4, WARN < 1e-3, FAIL ≥ 1e-3

| Body | Pos Error (m) | Ori Error (rad equiv) | Verdict |
|------|---------------|-----------------------|---------|
| torso | 2.32e-08 | 0.00e+00 | PASS |
| l_wheel_link | 5.21e-08 | 2.60e-08 | PASS |
| r_wheel_link | 4.65e-08 | 1.01e-07 | PASS |
| l_knee_link | 4.28e-08 | 1.46e-08 | PASS |
| r_knee_link | 6.87e-08 | 9.30e-08 | PASS |
| l_thigh | 1.75e-08 | 1.57e-08 | PASS |
| r_thigh | 3.48e-08 | 8.60e-08 | PASS |
| l_hip_roll_link | 1.56e-08 | 2.98e-08 | PASS |
| r_hip_roll_link | 1.56e-08 | 2.98e-08 | PASS |
| l_hip_yaw_link | 2.27e-08 | 3.78e-08 | PASS |
| r_hip_yaw_link | 2.27e-08 | 8.29e-08 | PASS |

**Max FK position error:** 6.87e-08 m
**Max FK orientation error:** 1.01e-07 (rotation matrix element)

## 7. COM Validation

- JAX COM: computed from body positions + inertial offsets, weighted by mass
- CPU COM: `data.subtree_com[1]` (torso subtree)
- Error: 2.56e-08 m
- Verdict: **PASS**

## 8. Translational Jacobian Validation

Actuated columns (qvel[6:16]) validated against CPU `jacp[:, 6:16]`.
Free-joint columns (v[0:6]) skipped — require quaternion-to-angular-velocity conversion.

Thresholds: PASS < 1e-3, WARN < 1e-2, FAIL ≥ 1e-2

| Body | Max Abs Error | Max Rel Error | Free-Joint Status | Verdict |
|------|---------------|---------------|-------------------|---------|
| torso | 0.00e+00 | 0.00e+00 | skipped — not validated in Phase 2A | PASS |
| l_wheel_link | 2.98e-08 | 7.90e-08 | skipped — not validated in Phase 2A | PASS |
| r_wheel_link | 2.98e-08 | 7.90e-08 | skipped — not validated in Phase 2A | PASS |
| l_knee_link | 1.66e-08 | 6.39e-08 | skipped — not validated in Phase 2A | PASS |
| r_knee_link | 1.77e-08 | 6.81e-08 | skipped — not validated in Phase 2A | PASS |
| l_thigh | 1.86e-09 | 6.20e-08 | skipped — not validated in Phase 2A | PASS |
| r_thigh | 2.00e-09 | 6.68e-08 | skipped — not validated in Phase 2A | PASS |

**Max Jacobian actuated-column abs error:** 2.98e-08

## 9. JIT Compatibility

| Function | JIT Status | Notes |
|----------|------------|-------|
| FK | ✓ | max pos err vs CPU: 6.867956653389129e-08 |
| COM | ✓ | err vs CPU: 2.5648752899343208e-08 |
| Jacobian | ✓ | finite: True |

## 10. Limitations

1. Free-joint Jacobian columns (v[0:6]) not validated — requires quaternion-to-angular-velocity conversion
2. Rotational Jacobians not implemented — only translational Jacobians ported
3. Mass matrix / CRBA not implemented (targeted for Phase 2B)
4. Contact force port not implemented
5. vmap / batch Jacobian not tested

## 11. Phase 2B Readiness Verdict

```text
READY_FOR_PHASE_2B_MASS_MATRIX_CRBA_PORT
```

