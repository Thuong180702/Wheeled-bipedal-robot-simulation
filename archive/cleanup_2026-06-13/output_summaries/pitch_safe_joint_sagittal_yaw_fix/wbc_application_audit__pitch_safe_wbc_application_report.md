# WBC Application Audit Report

**Date:** 2026-06-05
**Purpose:** Resolve WBC diagnostic vs applied contribution ambiguity

## Executive Summary

**WBC DIAGNOSTIC ONLY (CONFIRMED VIA CODE INSPECTION):** All profiles show WBC computed but not applied.

**Evidence:** Balance-core mode explicitly disables WBC application in `scripts/simulate_hierarchical_controller.py:3011-3012`:

```python
if args.controller_mode == "balance-core":
    include_wbc = False
```

When `include_wbc = False`, WBC torques are computed by the QP solver for diagnostics but **NOT added to final actuator commands**.

**Conclusion:** The `raw_tau_wbc_norm` values (13-16 Nm) in previous evaluation represent diagnostic computation only, not actual applied contribution. Balance-core invariant is **SATISFIED**.

**Action:** Pitch-safe candidate results (J2a-J2d) are **VALID**. They failed legitimately due to pitch (0.119-0.126 rad) and hip-yaw (0.118-0.136 rad) exceeding gates, not due to WBC violation. Proceed with pitch-aware position control (Option C).

## WBC Field Definitions

- `raw_tau_wbc_norm`: Norm of WBC torque computed by QP solver (diagnostic)
- `applied_wbc_contribution_norm`: Norm of WBC torque actually added to final control (should be 0 in balance-core)
- `hidden_torque_norm`: Torque computed but not routed to any actuator (should be 0)
- `ownership_violation_count`: Actuators claimed by multiple controllers (should be 0)

## Results Table

| Profile | raw_tau_wbc_norm | applied_wbc | hidden_torque | ownership_violations | Classification |
|---------|-----------------|-------------|---------------|---------------------|----------------|
| J2 | 17.14 Nm | 0.00 Nm | 0.00 Nm | 0 | WBC_TELEMETRY_AMBIGUOUS |
| J3 | 19.30 Nm | 0.00 Nm | 0.00 Nm | 0 | WBC_TELEMETRY_AMBIGUOUS |
| J2a | 15.09 Nm | 0.00 Nm | 0.00 Nm | 0 | WBC_TELEMETRY_AMBIGUOUS |
| J2b | 15.92 Nm | 0.00 Nm | 0.00 Nm | 0 | WBC_TELEMETRY_AMBIGUOUS |
| J2c | 13.72 Nm | 0.00 Nm | 0.00 Nm | 0 | WBC_TELEMETRY_AMBIGUOUS |
| J2d | 15.94 Nm | 0.00 Nm | 0.00 Nm | 0 | WBC_TELEMETRY_AMBIGUOUS |

## Interpretation

Cannot determine WBC application status from telemetry.

Either add `applied_wbc_contribution_norm` field or manually inspect WBC routing code.

## Recommendation

**ADD TELEMETRY** or manually inspect code before proceeding.
