# Phase 1: WBC / Torque Ownership Audit for low_0p300

## Executive Summary

**Conclusion:** WBC_COMPUTED_HIGH
**WBC Status:** WBC torques computed (max=13.71 Nm). Need to verify if applied or diagnostic only.

## Simulation Details

- Telemetry: `outputs\low_0p300_initialization_contact_audit\low_0p300_first_30_steps_telemetry.csv`
- Total steps: 30
- Terminated: False
- Termination reason: nan

## WBC Fields Present

- [OK] `tau_wbc_norm`
- [OK] `tau_wbc_max`
- [OK] `tau_posture_max`
- [OK] `tau_total_max`
- [OK] `qp_converged`
- [OK] `wrench_error_norm`

## WBC Analysis

- `tau_wbc_norm_max`: 13.7086
- `tau_wbc_norm_mean`: 12.9459
- `tau_wbc_max_max`: 9.6680
- `tau_wbc_max_mean`: 9.0513
- `tau_posture_max_max`: 0.0000
- `tau_posture_max_mean`: 0.0000
- `tau_total_max_max`: 8.8827
- `tau_total_max_mean`: 8.3708
- `qp_converged_max`: 1.0000
- `qp_converged_mean`: 1.0000
- `wrench_error_norm_max`: 3.2652
- `wrench_error_norm_mean`: 1.9731

## Ownership Analysis

- `ownership_violation_count_max`: 0.0
- `hidden_torque_norm_max`: 0.0

## Torque Analysis

- `wbc_norm_max`: 13.708630561828612
- `wbc_nonzero_steps`: 30
- `wbc_nonzero_percent`: 100.0
- `tau_total_max`: 8.882697105407715

## Contact Analysis

- `active_wheels_initial`: 2
- `active_wheels_final`: 2
- `contact_lost_step`: None
- `contact_valid_initial`: False
- `contact_valid_percent`: 96.66666666666667

## Next Steps

If WBC torques are being applied:
- **STOP IMMEDIATELY**
- Fix WBC routing/invariant before any dynamics conclusions
- Balance-core mode should have WBC OFF

If WBC is diagnostic only:
- Proceed to Phase 2: Static setup validation
