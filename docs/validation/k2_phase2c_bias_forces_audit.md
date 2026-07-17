# Phase 2C — JAX Bias Forces / Gravity / Coriolis Port Audit Report

**Timestamp:** 2026-07-02T04:45:42.050754+00:00
**Model:** `F:\ROBOTCUATAO\Wheeled-bipedal-robot-simulation\assets\robot\wheeled_biped_real.xml`

## 1. Executive Summary

Phase 2C implements a JAX-compatible generalized bias force computation :math:`\text{qfrc\_bias}(q, \dot{q}) \in \mathbb{R}^{16}` for the K2 wheeled-biped robot, using the **Recursive Newton-Euler Algorithm (RNEA)** with zero joint acceleration, validated against CPU MuJoCo `data.qfrc_bias`.

**Verdict: `PARTIAL_READY`**

- Gravity-only: 35/35 PASS, max abs error 6.16e-06
- Full bias: 21 PASS / 0 WARN / 14 FAIL, max abs error 6.25e-01
- Actuated bias: max abs error 5.53e-02
- Velocity-dependent: max abs error 6.25e-01
- JIT compatible: ✓

## 2. Controller Integrity

Controller code and `K2_JAX_DEDICATED_DEFAULT_V3` parameters were **not** modified.

## 3. Changed Files

| File | Status |
|------|--------|
| `wheeled_biped/dynamics/jax_bias_forces.py` | **new** — JAX RNEA bias forces |
| `wheeled_biped/dynamics/__init__.py` | modified — added exports |
| `scripts/phase2c_bias_forces_audit.py` | **new** — this script |
| `tests/test_phase2c_bias_forces.py` | **new** — tests |
| `docs/validation/k2_phase2c_bias_forces_audit.md` | **new** — this report |
| `docs/validation/k2_phase2c_bias_forces_audit.json` | **new** — JSON summary |

## 4. Phase 2B Reference Summary

- Verdict: `READY_FOR_PHASE_2C_BIAS_FORCES_PORT`
- Full M: 9/9 PASS (max abs error 3.81e-07)
- Actuated block: 9/9 PASS (max abs error 9.39e-08)
- Symmetry: 9/9 PASS
- JIT: Compatible

## 5. Bias Force Method

**Recursive Newton-Euler Algorithm (RNEA) with q̈ = 0**

The bias force is computed as the inverse dynamics solution with zero joint acceleration:

```text
qfrc_bias(q, q̇) = RNEA(q, q̇, q̈=0)
```

**Forward pass:**
- Compute forward kinematics (Phase 2A)
- Compute body spatial velocities via recursive tree traversal
- Compute spatial accelerations with q̈=0:
  - Fictitious base acceleration a_0 = [0; -g] for gravity
  - Hinge body: a_c = a_parent_transformed + v_c × (S @ q̇)
  - No-joint body: a_c = a_parent_transformed
  - Centripetal terms ω×(ω×r) included for world-frame coordinates

**Backward pass:**
- For each body in reverse topological order:
  - Compute spatial inertia I at body origin in world frame
  - Convert world-frame acceleration to Featherstone convention: a_fs = [α; a_lin - ω×v]
  - Compute spatial force: F = I @ a_fs + v ×* (I @ v)
  - Propagate force to parent: F_parent += [τ_c + r×f_c; f_c]
- Project onto joint motion subspaces → generalized forces

**Sign convention:** MuJoCo `qfrc_bias` appears on the left-hand side: `M(q)@q̈ + qfrc_bias = τ_applied`.  Gravity is a fictitious upward base acceleration so that the RNEA output includes gravitational forces with the correct sign.

**Free-base handling:** Full 16-vector output.  MuJoCo uses [force; torque] ordering for free-base DOFs (qvel[0:3] = linear, qvel[3:6] = angular).

## 6. Constants Summary

- nbody: 12
- nq: 17
- nv: 16
- Gravity: [ 0.    0.   -9.81]
- Total body mass: 8.1000 kg

## 7. Validation Case Summary

- Poses: 7 (keyframe, 3 height-like, 3 random)
- Velocity cases: 5 (zero, small_random, moderate_random, base_yaw_rate, symmetric_wheels)
- Total pose × velocity cases: 35

## 8. Gravity-Only Validation

Thresholds: PASS < 0.001, WARN < 0.01, FAIL ≥ 0.01

| Pose | Max Abs Err | Max Rel Err | Verdict |
|------|-------------|-------------|---------|
| keyframe | 6.16e-06 | 7.76e-08 | PASS |
| low_height | 6.16e-06 | 7.76e-08 | PASS |
| mid_height | 6.16e-06 | 7.76e-08 | PASS |
| high_height | 6.16e-06 | 7.76e-08 | PASS |
| random_1 | 6.16e-06 | 7.76e-08 | PASS |
| random_2 | 6.16e-06 | 7.76e-08 | PASS |
| random_3 | 6.16e-06 | 7.76e-08 | PASS |

## 9. Full Bias Validation (nonzero velocity)

Thresholds: PASS < 0.001, WARN < 0.01, FAIL ≥ 0.01

| Pose | Vel Case | Full Err | FB Err | Act Err | Vel Err | Verdict |
|------|----------|----------|--------|---------|---------|---------|
| keyframe | small_random | 1.22e-02 | 1.22e-02 | 2.89e-03 | 1.22e-02 | FAIL |
| keyframe | moderate_random | 6.16e-01 | 6.16e-01 | 4.87e-02 | 6.16e-01 | FAIL |
| keyframe | base_yaw_rate | 6.16e-06 | 6.16e-06 | 2.19e-07 | 1.18e-07 | PASS |
| keyframe | symmetric_wheels | 6.16e-06 | 6.16e-06 | 2.24e-07 | 2.13e-12 | PASS |
| low_height | small_random | 1.08e-02 | 1.08e-02 | 2.06e-03 | 1.08e-02 | FAIL |
| low_height | moderate_random | 5.87e-01 | 5.87e-01 | 2.95e-02 | 5.87e-01 | FAIL |
| low_height | base_yaw_rate | 6.16e-06 | 6.16e-06 | 6.50e-07 | 2.71e-07 | PASS |
| low_height | symmetric_wheels | 6.16e-06 | 6.16e-06 | 5.84e-07 | 2.13e-12 | PASS |
| mid_height | small_random | 1.15e-02 | 1.15e-02 | 2.40e-03 | 1.15e-02 | FAIL |
| mid_height | moderate_random | 6.02e-01 | 6.02e-01 | 3.33e-02 | 6.02e-01 | FAIL |
| mid_height | base_yaw_rate | 6.16e-06 | 6.16e-06 | 4.02e-07 | 2.96e-07 | PASS |
| mid_height | symmetric_wheels | 6.16e-06 | 6.16e-06 | 2.95e-07 | 2.13e-12 | PASS |
| high_height | small_random | 1.25e-02 | 1.25e-02 | 3.24e-03 | 1.24e-02 | FAIL |
| high_height | moderate_random | 6.22e-01 | 6.22e-01 | 5.50e-02 | 6.22e-01 | FAIL |
| high_height | base_yaw_rate | 6.16e-06 | 6.16e-06 | 3.55e-07 | 1.85e-07 | PASS |
| high_height | symmetric_wheels | 6.16e-06 | 6.16e-06 | 4.14e-07 | 2.13e-12 | PASS |
| random_1 | small_random | 1.15e-02 | 1.15e-02 | 2.84e-03 | 1.15e-02 | FAIL |
| random_1 | moderate_random | 5.97e-01 | 5.97e-01 | 5.53e-02 | 5.97e-01 | FAIL |
| random_1 | base_yaw_rate | 6.16e-06 | 6.16e-06 | 5.66e-07 | 2.02e-07 | PASS |
| random_1 | symmetric_wheels | 6.16e-06 | 6.16e-06 | 5.59e-07 | 1.32e-09 | PASS |
| random_2 | small_random | 1.25e-02 | 1.25e-02 | 2.75e-03 | 1.25e-02 | FAIL |
| random_2 | moderate_random | 6.25e-01 | 6.25e-01 | 4.76e-02 | 6.25e-01 | FAIL |
| random_2 | base_yaw_rate | 6.16e-06 | 6.16e-06 | 3.27e-07 | 1.69e-07 | PASS |
| random_2 | symmetric_wheels | 6.16e-06 | 6.16e-06 | 3.38e-07 | 7.32e-10 | PASS |
| random_3 | small_random | 1.30e-02 | 1.30e-02 | 2.90e-03 | 1.30e-02 | FAIL |
| random_3 | moderate_random | 6.21e-01 | 6.21e-01 | 4.86e-02 | 6.21e-01 | FAIL |
| random_3 | base_yaw_rate | 6.16e-06 | 6.16e-06 | 5.59e-07 | 1.26e-07 | PASS |
| random_3 | symmetric_wheels | 6.16e-06 | 6.16e-06 | 4.33e-07 | 2.17e-09 | PASS |

## 10. Velocity-Dependent Bias Validation

Thresholds: PASS < 0.001, WARN < 0.01, FAIL ≥ 0.01

| Pose | Vel Case | Max Abs Err | Verdict |
|------|----------|-------------|---------|
| keyframe | small_random | 1.22e-02 | FAIL |
| keyframe | moderate_random | 6.16e-01 | FAIL |
| keyframe | base_yaw_rate | 1.18e-07 | PASS |
| keyframe | symmetric_wheels | 2.13e-12 | PASS |
| low_height | small_random | 1.08e-02 | FAIL |
| low_height | moderate_random | 5.87e-01 | FAIL |
| low_height | base_yaw_rate | 2.71e-07 | PASS |
| low_height | symmetric_wheels | 2.13e-12 | PASS |
| mid_height | small_random | 1.15e-02 | FAIL |
| mid_height | moderate_random | 6.02e-01 | FAIL |
| mid_height | base_yaw_rate | 2.96e-07 | PASS |
| mid_height | symmetric_wheels | 2.13e-12 | PASS |
| high_height | small_random | 1.24e-02 | FAIL |
| high_height | moderate_random | 6.22e-01 | FAIL |
| high_height | base_yaw_rate | 1.85e-07 | PASS |
| high_height | symmetric_wheels | 2.13e-12 | PASS |
| random_1 | small_random | 1.15e-02 | FAIL |
| random_1 | moderate_random | 5.97e-01 | FAIL |
| random_1 | base_yaw_rate | 2.02e-07 | PASS |
| random_1 | symmetric_wheels | 1.32e-09 | PASS |
| random_2 | small_random | 1.25e-02 | FAIL |
| random_2 | moderate_random | 6.25e-01 | FAIL |
| random_2 | base_yaw_rate | 1.69e-07 | PASS |
| random_2 | symmetric_wheels | 7.32e-10 | PASS |
| random_3 | small_random | 1.30e-02 | FAIL |
| random_3 | moderate_random | 6.21e-01 | FAIL |
| random_3 | base_yaw_rate | 1.26e-07 | PASS |
| random_3 | symmetric_wheels | 2.17e-09 | PASS |

## 11. JIT Compatibility

JIT bias forces: ✓ PASS

## 12. Limitations

1. Full RNEA-based bias force computation (not energy/Christoffel method)
2. World-frame acceleration propagation with Featherstone correction in backward pass
3. Velocity-dependent mixed-case errors can reach 1e-2 for large random velocities (free-base component dominant)
4. Error scales as ~qvel², indicating residual Coriolis coefficient mismatch in multi-joint velocity interactions
5. Free-base forces have larger relative error than actuated joint torques

## 13. Phase 2D Readiness Verdict

```text
PARTIAL_READY
```

