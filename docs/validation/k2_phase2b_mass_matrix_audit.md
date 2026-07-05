# Phase 2B — JAX Mass Matrix / CRBA Port Audit Report

**Timestamp:** 2026-07-02T02:40:35.765344+00:00
**Model:** `F:\ROBOTCUATAO\Wheeled-bipedal-robot-simulation\assets\robot\wheeled_biped_real.xml`

## 1. Executive Summary

Phase 2B implements a JAX-compatible full generalized mass matrix :math:`M(q) \in \mathbb{R}^{16 \times 16}` for the K2 wheeled-biped robot, validated against CPU MuJoCo `mj_fullM` ground truth.

**Verdict: `READY_FOR_PHASE_2C_BIAS_FORCES_PORT`**

- Full M: 9/9 PASS, max abs error 3.81e-07
- Actuated block: 9/9 PASS, max abs error 9.39e-08
- Symmetry: 9/9 PASS, max asymmetry 0.00e+00
- JIT compatible: ✓

## 2. Controller Integrity

Controller code and `K2_JAX_DEDICATED_DEFAULT_V3` parameters were **not** modified.

## 3. Changed Files

| File | Status |
|------|--------|
| `wheeled_biped/dynamics/jax_mass_matrix.py` | **new** — JAX mass matrix |
| `wheeled_biped/dynamics/__init__.py` | modified — added exports |
| `scripts/phase2b_mass_matrix_audit.py` | **new** — this script |
| `tests/test_phase2b_mass_matrix.py` | **new** — tests |
| `docs/validation/k2_phase2b_mass_matrix_audit.md` | **new** — this report |
| `docs/validation/k2_phase2b_mass_matrix_audit.json` | **new** — JSON summary |

## 4. Phase 2A Reference Summary

- Verdict: `READY_FOR_PHASE_2B_MASS_MATRIX_CRBA_PORT`
- FK: 11/11 PASS (max pos error 6.87e-08 m)
- COM: PASS (error 2.56e-08 m)
- Jacobians: 7/7 PASS (max abs error 2.98e-08)
- JIT: FK ✓, COM ✓, Jacobian ✓

## 5. Mass Matrix Method

**Kinetic Energy Hessian** (mathematically equivalent to CRBA):

1. Compute body spatial velocities recursively through kinematic tree
2. Compute kinetic energy: T = Σ 0.5*m*||v_COM||² + 0.5*ω^T*I_COM*ω
3. M = ∇²_{q̇} T(q, q̇) |_{q̇=0} via `jax.hessian`

**Free-base handling:** Full 16×16 M(q) including 6 free-base DOFs + 10 actuated DOFs.

**Inertial frame convention:** MuJoCo `body_inertia` (COM inertia, diagonal in inertial frame) rotated to world frame via `body_quat * body_iquat`.

**Armature:** DOF armature (reflected rotor inertias from `model.dof_armature`) added to diagonal to match MuJoCo `mj_fullM` convention.

**Symmetrization:** M_sym = 0.5 * (M + M^T) applied to correct floating-point autodiff asymmetries (~1e-15).

## 6. Constants Summary

- nbody: 12
- nq: 17
- nv: 16
- Total body mass: 8.1000 kg
- DOF armature (free base): [0. 0. 0. 0. 0. 0.]
- DOF armature (actuated):  [0.02  0.02  0.02  0.02  0.008 0.02  0.02  0.02  0.02  0.008]

## 7. Validation Poses

| # | Pose | Description |
|---|------|-------------|
| 1 | keyframe | Keyframe/nominal |
| 2 | low_height | Height-like |
| 3 | mid_height | Height-like |
| 4 | high_height | Height-like |
| 5 | random_1 | Random perturbation |
| 6 | random_2 | Random perturbation |
| 7 | random_3 | Random perturbation |
| 8 | random_4 | Random perturbation |
| 9 | random_5 | Random perturbation |

## 8. Full Mass Matrix Validation

Thresholds: PASS < 0.001, WARN < 0.01, FAIL ≥ 0.01

| Pose | CPU shape | JAX shape | Max Abs Err | Max Rel Err | Symmetry Err | Cond | Verdict |
|------|-----------|-----------|-------------|-------------|--------------|------|---------|
| keyframe | (16,16) | (16,16) | 3.81e-07 | 4.71e-08 | 0.00e+00 | 1010.7 | PASS |
| low_height | (16,16) | (16,16) | 3.81e-07 | 4.71e-08 | 0.00e+00 | 997.9 | PASS |
| mid_height | (16,16) | (16,16) | 3.81e-07 | 4.71e-08 | 0.00e+00 | 1003.9 | PASS |
| high_height | (16,16) | (16,16) | 3.81e-07 | 4.71e-08 | 0.00e+00 | 1014.0 | PASS |
| random_1 | (16,16) | (16,16) | 3.81e-07 | 4.71e-08 | 0.00e+00 | 1010.5 | PASS |
| random_2 | (16,16) | (16,16) | 3.81e-07 | 4.71e-08 | 0.00e+00 | 1010.1 | PASS |
| random_3 | (16,16) | (16,16) | 3.81e-07 | 4.71e-08 | 0.00e+00 | 1010.6 | PASS |
| random_4 | (16,16) | (16,16) | 3.81e-07 | 4.71e-08 | 0.00e+00 | 1011.0 | PASS |
| random_5 | (16,16) | (16,16) | 3.81e-07 | 4.71e-08 | 0.00e+00 | 1010.5 | PASS |

## 9. Actuated Block Validation

Thresholds: PASS < 0.001, WARN < 0.01, FAIL ≥ 0.01

| Pose | Max Abs Err | Max Rel Err | Verdict |
|------|-------------|-------------|---------|
| keyframe | 2.26e-08 | 2.15e-07 | PASS |
| low_height | 1.13e-08 | 2.14e-07 | PASS |
| mid_height | 1.85e-08 | 2.63e-07 | PASS |
| high_height | 2.27e-08 | 1.79e-07 | PASS |
| random_1 | 6.09e-08 | 5.66e-07 | PASS |
| random_2 | 4.44e-08 | 4.29e-07 | PASS |
| random_3 | 4.94e-08 | 4.35e-07 | PASS |
| random_4 | 9.39e-08 | 8.60e-07 | PASS |
| random_5 | 2.39e-08 | 2.25e-07 | PASS |

## 10. Diagonal Validation

| Pose | Min Diag | Max Diag | All Positive |
|------|----------|----------|-------------|
| keyframe | 8.1883e-03 | 8.1000e+00 | True |
| low_height | 8.1883e-03 | 8.1000e+00 | True |
| mid_height | 8.1883e-03 | 8.1000e+00 | True |
| high_height | 8.1883e-03 | 8.1000e+00 | True |
| random_1 | 8.1883e-03 | 8.1000e+00 | True |
| random_2 | 8.1883e-03 | 8.1000e+00 | True |
| random_3 | 8.1883e-03 | 8.1000e+00 | True |
| random_4 | 8.1883e-03 | 8.1000e+00 | True |
| random_5 | 8.1883e-03 | 8.1000e+00 | True |

## 11. JIT Compatibility

JIT mass matrix: ✓ PASS

## 12. Limitations

1. Mass matrix includes dof_armature (reflected rotor inertias) to match MuJoCo convention
2. Kinetic energy Hessian method used (not CRBA) — mathematically equivalent
3. Free-joint Jacobian columns validated indirectly through mass matrix matching
4. Rotational Jacobians validated indirectly through mass matrix matching
5. No contact-consistent dynamics port (targeted for Phase 2C)

## 13. Phase 2C Readiness Verdict

```text
READY_FOR_PHASE_2C_BIAS_FORCES_PORT
```

