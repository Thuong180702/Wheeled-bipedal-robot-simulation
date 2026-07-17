# K2 Phase 3 — Offline QP-WBC Prototype Audit

**Verdict:** `READY_FOR_PHASE_3B_OFFLINE_TASK_STACK_EXPANSION`
**Timestamp:** 2026-07-03T11:22:05.256736+00:00

## 1. Executive Summary

- **Scenarios requested:** 12
- **Scenarios solved:** 12
- **Scenarios failed:** 0
- **Total contacts:** 44
- **Solver:** SLSQP (fallback: True)
- **Dynamics residual max:** 2.817e-14
- **Contact accel residual max:** 5.329e-15
- **Friction violation max:** 7.739e-12
- **Torque violation max:** 0.000e+00

## 2. Controller Integrity

- **Controller modified:** False
- **QP torque injected:** False
- **K2_JAX_DEDICATED_DEFAULT_V3 unchanged:** True

## 3. Changed Files

- `wheeled_biped/wbc/__init__.py` (new)
- `wheeled_biped/wbc/offline_qp_wbc.py` (new)
- `scripts/phase3_offline_qp_wbc_audit.py` (new)
- `tests/test_phase3_offline_qp_wbc.py` (new)
- `docs/validation/k2_phase3_offline_qp_wbc_audit.md` (new)
- `docs/validation/k2_phase3_offline_qp_wbc_audit.json` (new)

## 4. Phase 2 Readiness Recap

Phase 2C.5, 2D, and 2D.1 dynamics stack validated. All tests pass.
Controller unchanged throughout Phase 2 audit series.

## 5. QP Formulation

### Variables

```text
z = [qdd (16), tau (10), lambda (3m), slack (k)]
```

### Cost

```text
minimize:
  w_qdd      * ||qdd||^2
+ w_tau      * ||tau||^2
+ w_lambda   * ||lambda||^2
+ w_slack    * ||slack||^2
```

### Dynamics Equality

```text
M @ qdd + h = S @ tau + JcT @ lambda
-> [M, -S, -JcT] @ [qdd; tau; lambda] = -h
```

## 6. Contact Acceleration Constraints

- **Jdot qdot implemented:** True
- **Jdot qdot validated:** True
- **Method:** Central finite difference, eps=1e-5
- **qpos integration validated:** PASS (max err: 0.000e+00)

## 7. Friction Cone

- **Model:** Linearized pyramid, μ = 0.8
- **Inequalities:** 5 per contact (fn>=0, ±ft1≤μfn, ±ft2≤μfn)

## 8. Torque Limits

- **Source:** `actuator_forcerange` from MuJoCo model
- **Bounds:** hip_roll/hip_yaw/wheel ±60 Nm, hip_pitch/knee ±150 Nm

## 9. Solver Backend

- **Name:** SLSQP
- **Available:** True
- **Fallback used:** True
- **OSQP available:** False
- **Settings:** {'method': 'SLSQP', 'maxiter': 500, 'ftol': 1e-08}

## 10. Scenario Results

| # | Scenario | Contacts | Solved | Dyn Res | Contact Accel | Friction | Torque |
|---|----------|----------|--------|---------|---------------|----------|--------|
| keyframe_static | 4 | OK | 2.1e-15 | 7.5e-16 | 0.0e+00 | 0.0e+00 |
| passive_settle_keyframe | 4 | OK | 2.1e-15 | 7.5e-16 | 0.0e+00 | 0.0e+00 |
| low_height_settle | 4 | OK | 2.6e-14 | 3.6e-15 | 3.0e-12 | 0.0e+00 |
| mid_height_settle | 4 | OK | 2.1e-15 | 7.5e-16 | 0.0e+00 | 0.0e+00 |
| high_height_settle | 4 | OK | 4.3e-15 | 3.6e-15 | 0.0e+00 | 0.0e+00 |
| small_forward_velocity | 4 | OK | 2.1e-15 | 7.5e-16 | 0.0e+00 | 0.0e+00 |
| small_lateral_velocity | 4 | OK | 4.0e-15 | 1.8e-15 | 0.0e+00 | 0.0e+00 |
| small_yaw_rate | 4 | OK | 4.8e-15 | 1.8e-15 | 7.2e-12 | 0.0e+00 |
| small_roll_tilt | 4 | OK | 2.8e-14 | 9.4e-16 | 7.7e-12 | 0.0e+00 |
| small_pitch_tilt | 2 | OK | 2.1e-15 | 1.8e-15 | 0.0e+00 | 0.0e+00 |
| random_pose_small_perturbation_1 | 4 | OK | 1.5e-14 | 5.3e-15 | 4.7e-12 | 0.0e+00 |
| random_pose_small_perturbation_2 | 2 | OK | 2.7e-15 | 1.7e-15 | 0.0e+00 | 0.0e+00 |

## 11. Dynamics Residual Validation

- **Max full residual:** 2.817e-14
- **Threshold PASS:** 1e-5
- **Threshold WARN:** 1e-4

## 12. Contact Normal Acceleration Validation

- **Max residual:** 5.329e-15
- **Threshold PASS:** 1e-4

## 13. Friction Validation

- **Max violation:** 7.739e-12
- **Min normal force:** -7.739e-12

## 14. Torque Limit Validation

- **Max violation:** 0.000e+00

## 15. Solution Magnitude Sanity

- **Max |qdd|:** 57.247
- **Max |tau|:** 2.337
- **Max |lambda|:** 1.005

## 16. JIT Compatibility

- **Dynamics calls use JAX operations:** True
- **JIT-compatible:** True
- **Scipy solver outside JIT:** True

## 17. Limitations

- SLSQP fallback used (OSQP not available)
- Jdot qdot uses finite difference (not analytical)
- No tangential rolling constraint (wheel rolling unmodeled)
- Offline only — no realtime integration

## 18. Phase 3B Readiness

**Verdict:** `READY_FOR_PHASE_3B_OFFLINE_TASK_STACK_EXPANSION`

Proceed to Phase 3B — Offline Task Stack Expansion.
