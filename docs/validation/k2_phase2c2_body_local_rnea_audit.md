# Phase 2C.2 — Body-Local Featherstone RNEA Audit Report

**Timestamp:** 2026-07-02T09:26:59.726255+00:00
**Model:** `F:\ROBOTCUATAO\Wheeled-bipedal-robot-simulation\assets\robot\wheeled_biped_real.xml`

## 1. Executive Summary

Phase 2C.2 implements a correct body-local Featherstone RNEA for bias force computation, replacing the world-frame RNEA from Phase 2C and the partially corrected Phase 2C.1.

**Phase 2C:** 21 PASS / 0 WARN / 14 FAIL (max full=6.25e-01)
**Phase 2C.1:** 21 PASS / 0 WARN / 14 FAIL (max full=1.92e+00)
**Phase 2C.2:** 21 PASS / 0 WARN / 14 FAIL (max full=1.38e+00, max act=6.29e-02)

**Verdict: `NOT_READY`**

## 2. Controller Integrity

Controller code and `K2_JAX_DEDICATED_DEFAULT_V3` parameters were **not** modified.

## 3. Changed Files

| File | Status |
|------|--------|
| `wheeled_biped/dynamics/jax_bias_forces.py` | **rewritten** — body-local Featherstone RNEA |
| `scripts/phase2c2_body_local_rnea_audit.py` | **new** — this audit script |
| `tests/test_phase2c2_body_local_rnea.py` | **new** — tests |
| `docs/validation/k2_phase2c2_body_local_rnea_audit.md` | **new** — this report |
| `docs/validation/k2_phase2c2_body_local_rnea_audit.json` | **new** — JSON summary |

## 4. Rollback Decision

**Clean rewrite** of `jax_bias_forces.py`.

Phase 2C.1 increased max full-bias error from 0.625 to 1.92 while still using world-frame RNEA. Phase 2C.2 is a fresh body-local implementation.

## 5. Body-Local RNEA Method

**Pure body-local Featherstone RNEA** with q̈ = 0.

### Spatial vector convention

```text
[angular; linear] — Featherstone standard
v = [ω; v_origin]
```

### MuJoCo qvel / qfrc_bias mapping

```text
qvel[0:3]  = base linear velocity (world frame)
qvel[3:6]  = base angular velocity (world frame)
qfrc_bias[0:3] = force on free-base translation DOFs
qfrc_bias[3:6] = torque on free-base rotation DOFs
qfrc_bias[6:16] = actuated joint generalized forces
```

### Algorithm

1. **FK**: compute body world orientations.
2. **Precompute**: body-local spatial inertias, tree transforms R_tree (from model.body_quat), motion subspaces S_i, joint DOF indices.
3. **Forward pass** (root→leaves, body-local frames):
   - Torso: v = [R^T@ω_w; R^T@v_w], a = [0; -R^T@g]
   - Hinge: v = X_up@v_parent + S@q̇, a = X_up@a_parent + crm(v)@(S@q̇)
   - No-joint: v = X_up@v_parent, a = X_up@a_parent
4. **Backward pass** (leaves→root):
   - F = I@a + crf(v)@I@v
   - Propagate: F_parent += X_up^T @ F_child
5. **Project**: τ_j = S^T@F_body; base: qfrc[0:6] = R_torso@F_torso mapped to MuJoCo order.

## 6. Constants Summary

- nbody: 12
- nq: 17
- nv: 16
- Constants version: `phase2c2_body_local_rnea`
- Gravity: [ 0.    0.   -9.81]
- Total body mass: 8.1000 kg

## 7. Gravity-Only Validation

**Result: 133/133 PASS**, max abs error = 6.16e-06

## 8. Full Bias Validation (original 35 cases)

Thresholds: PASS < 0.001, WARN < 0.01, FAIL ≥ 0.01

| Velocity Case | Cases | Max Err | FB Force Err | FB Torque Err | Act Err | Verdicts |
|---------------|-------|---------|--------------|---------------|---------|----------|
| base_yaw_rate | 7 | 6.16e-06 | 6.16e-06 | 4.88e-07 | 3.80e-07 | PPPPPPP |
| moderate_random | 7 | 1.38e+00 | 1.38e+00 | 1.94e-01 | 6.29e-02 | FFFFFFF |
| small_random | 7 | 7.45e-02 | 7.45e-02 | 1.05e-02 | 3.46e-03 | FFFFFFF |
| symmetric_wheels | 7 | 6.16e-06 | 6.16e-06 | 3.93e-07 | 3.98e-07 | PPPPPPP |
| zero | 7 | 6.16e-06 | 6.16e-06 | 3.93e-07 | 3.98e-07 | PPPPPPP |

## 9. Free-Base Validation

- Free-base force: 119/133 PASS, max 6.16e-06 N
- Free-base torque: 119/133 PASS, max 3.93e-07 Nm

## 10. Actuated Bias Validation

**Result: 119/133 PASS**, max abs error = 6.29e-02 Nm

## 11. Velocity-Dependent Validation

**Result: 112/126 nonzero velocity PASS**, max abs error = 1.38e+00

## 12. Cross-Term Validation

Cross-term: bias(q, vi+vj) - bias(q, vi) - bias(q, vj) + bias(q, 0)

- Base angular × base linear cross-term: FAIL (non-zero, should be zero)
- Base angular × actuated pairs: PASS
- Actuated × actuated pairs: PASS

## 13. JIT Compatibility

JIT bias forces: ✗ FAIL

## 14. Limitations

1. **Free-base angular × linear velocity cross-term error.** When both base angular and base linear velocity are nonzero, the body-local RNEA produces a spurious cross-term in the free-base generalized forces. The CPU MuJoCo reference shows this cross-term is structurally zero. The error scales linearly with angular velocity magnitude (≈2.4 N at ω=1 rad/s) and is dominated by the free-base force components.

2. The cross-term error is ISOLATED to the free-base ω×v coupling. All other cross-terms (base angular × actuated, base linear × actuated, actuated × actuated) pass to machine precision.

3. The root cause is that the spatial algebra identity for composite-body force cross-terms is satisfied (verified numerically), but the RNEA's free-base generalized force projection fails to cancel the torso's own cross-term with the children's propagated cross-terms. This may indicate an issue with how the free-base DOFs are projected to generalized forces.

4. Joint friction, damping, and armature are handled by MuJoCo internally and are not part of `qfrc_bias`.

## 15. Phase 2D Readiness Verdict

```text
NOT_READY
```

