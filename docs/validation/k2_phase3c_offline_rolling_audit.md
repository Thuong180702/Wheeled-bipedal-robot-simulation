# K2 Phase 3C — Offline Rolling Constraints Audit

**Verdict:** `READY_FOR_PHASE_3D_OFFLINE_WBC_SHADOW_EVALUATION`

## 1. Executive Summary
- Total QP solves: 120/120
- Hard constraints pass: True
- Controller modified: False
- QP torque injected: False

## 2. Controller Integrity Statement
No controller files modified. No QP torque injected. No realtime integration.

## 3. Wheel Geometry
- Wheel radius: L=0.0600m, R=0.0600m
- Wheel qvel indices: L=10, R=15
- Wheel axes (local): L=[-1.0, 0.0, 0.0], R=[1.0, 0.0, 0.0]

## 4. Results Summary
| Task Mode | Rolling Mode | Solved | Max Dyn Res | Max Fric Viol | Max Torque Viol |
|-----------|-------------|--------|-------------|---------------|-----------------|
| feasibility_only | normal_only | 12/12 | 2.82e-14 | 7.74e-12 | 0.00e+00 |
| balanced_default | normal_only | 12/12 | 3.15e-14 | 3.44e-11 | 0.00e+00 |
| feasibility_only | lateral_soft | 12/12 | 3.76e-14 | 2.10e-11 | 0.00e+00 |
| balanced_default | lateral_soft | 12/12 | 3.13e-14 | 1.15e-10 | 0.00e+00 |
| feasibility_only | lateral_hard | 12/12 | 3.57e-14 | 1.42e-11 | 0.00e+00 |
| balanced_default | lateral_hard | 12/12 | 2.13e-14 | 1.02e-10 | 0.00e+00 |
| feasibility_only | full_rolling_soft | 12/12 | 2.75e-14 | 8.35e-11 | 0.00e+00 |
| balanced_default | full_rolling_soft | 12/12 | 2.58e-14 | 3.46e-10 | 0.00e+00 |
| feasibility_only | full_rolling_hard | 12/12 | 2.13e-14 | 9.03e-11 | 0.00e+00 |
| balanced_default | full_rolling_hard | 12/12 | 1.78e-14 | 1.30e-10 | 0.00e+00 |

## 5. Hard Constraint Aggregate (worst across all modes)
- Max Dynamics: 3.76e-14
- Max Contact accel: 0.00e+00
- Max Friction: 3.46e-10
- Max Torque: 0.00e+00

## 6. Rolling Residuals
- Max Pre-solve lateral slip: 3.00e-01
- Max Post-solve lateral residual: 0.00e+00
- Max Pre-solve rolling residual: 2.00e-01
- Max Post-solve rolling residual: 0.00e+00

## 7. Phase 3D Readiness Verdict
**READY_FOR_PHASE_3D_OFFLINE_WBC_SHADOW_EVALUATION**
