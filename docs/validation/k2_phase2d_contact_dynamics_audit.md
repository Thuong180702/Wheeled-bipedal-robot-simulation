# Phase 2D — Contact Dynamics / Contact Jacobian / Constraint Force Validation Audit Report

**Timestamp:** 2026-07-03T03:08:33.987558+00:00  
**Verdict:** `READY_FOR_PHASE_3_OFFLINE_QP_WBC_PROTOTYPE`

## 1. Executive Summary

Phase 2D implements JAX-compatible contact dynamics infrastructure validated against CPU MuJoCo:
- Contact point world position from body-local coordinates
- Full translational contact Jacobian Jp in R^(3x16), including free-base columns
- Rotational contact Jacobian Jr in R^(3x16)
- Contact force to generalized force mapping (Jp^T @ f_world)
- Contact wrench to generalized force mapping (Jp^T @ f + Jr^T @ tau)

All validations PASS with excellent margins against CPU MuJoCo ground truth.

### Results Summary

| Validation | PASS | WARN | FAIL | Max Error |
|------------|------|------|------|-----------|
| Contact Point Reconstruction | 4 | 0 | 0 | 6.06e-08 m |
| Jacobian Full | 4 | 0 | 0 | 4.76e-08 |
| Jacobian Base Linear | 4 | 0 | 0 | 0.00e+00 |
| Jacobian Base Angular | 4 | 0 | 0 | 4.76e-08 |
| Jacobian Actuated | 4 | 0 | 0 | 1.00e-07 |
| QFRC Full | 4 | 0 | 0 | 1.27e-05 |
| QFRC Free-Base | 4 | 0 | 0 | 1.27e-05 |
| QFRC Actuated | 4 | 0 | 0 | 5.00e-07 |

## 2. Controller Integrity

Controller code and K2_JAX_DEDICATED_DEFAULT_V3 were **not** modified.

## 3. Changed Files

| File | Status |
|------|--------|
| wheeled_biped/dynamics/jax_contact_dynamics.py | **new** |
| tests/test_phase2d_contact_dynamics.py | **new** (30 tests) |
| scripts/phase2d_contact_dynamics_audit.py | **new** |
| tests/test_phase2c4_runtime_mcross_orientation.py | **modified** (removed 2 xfail) |
| docs/validation/k2_phase2c5_actuated_coriolis_audit.md | **modified** (corrected table) |
| docs/validation/k2_phase2c5_actuated_coriolis_audit.json | **modified** (correction notes) |

## 4. Phase 2C.5 Readiness Recap

Phase 2C.5 verified READY: 35P/0W/0F, max actuated error 4.60e-07, max full error 1.41e-05.
xpassed tests cleaned: 79 passed, 0 xpassed. Report inconsistency fixed.

## 5. Contact Dynamics Method

### Contact Point Kinematics
Uses Phase 2A FK: p_world = x_body + R_body @ p_local. Validated against CPU xpos + R(xquat) @ p_local.

### Contact Jacobian
Free-base: Jp[:,0:3]=I_3, Jp[:,3:6]=-skew(r)@R_base_world (body-frame angular velocity convention).
Actuated: JAX autodiff of p_world w.r.t. qpos[7:17].

### Contact Force Mapping
Virtual work: qfrc = Jp^T @ force_world (+ Jr^T @ torque_world for wrench).

## 6. Free-Base Jacobian Convention

- qvel[0:3] = base linear velocity (WORLD frame) -> Jp[:,0:3]=I_3
- qvel[3:6] = base angular velocity (BODY frame) -> Jp[:,3:6]=-skew(r)@R_base
- omega_world = R_base @ qvel[3:6]

## 7. Contact Reference Extraction

CPU MuJoCo used for reference only (not inside JAX compute):
- mj_jac for Jacobian ground truth
- mj_contactForce for 6D contact wrench
- contact.frame is 3x3 rotation matrix

## 8. Contact Frame Convention

MuJoCo contact.frame: frame[:,0]=normal, frame[:,1]=tangent1, frame[:,2]=tangent2.
Force conversion: f_world = contact.frame @ f_contact_frame.

## 9-11. Validation Results

Contact Point Reconstruction: 4/4 PASS, max error 6.06e-08 m
Contact Jacobian: 4/4 PASS, max full error 4.76e-08
Contact Force Mapping: 4/4 PASS, max qfrc error 1.27e-05

30-test suite additionally validates non-identity orientations and JIT compatibility.

## 12. Aggregate Metrics

- Max contact point error: 6.06e-08 m
- Max Jacobian full error: 4.76e-08
- Max Jacobian base linear error: 0.00e+00
- Max Jacobian base angular error: 4.76e-08
- Max Jacobian actuated error: 1.00e-07
- Max QFRC full error: 1.27e-05
- Max QFRC free-base error: 1.27e-05
- Max QFRC actuated error: 5.00e-07

## 13. JIT Compatibility

JIT check: PASS. All functions JIT-compile and produce identical results.

## 14. Limitations

- Contact detection not implemented (CPU MuJoCo locates contacts for validation).
- Summed qfrc_constraint validation not applicable (per-contact Path A used).
- No friction cone / QP / WBC integration (Phase 3 scope).

## 15. Phase 3 Readiness Verdict

```text
READY_FOR_PHASE_3_OFFLINE_QP_WBC_PROTOTYPE
```

**Recommendation: Proceed to Phase 3 offline QP-WBC prototype development.**
