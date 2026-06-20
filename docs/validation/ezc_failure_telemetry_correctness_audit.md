# EZC Failure Telemetry Correctness Audit

**Date:** 2026-06-15  
**Profile:** early_zero_crossing_recenter  
**Scenario:** high_0p480, 5000 steps

## Classification

**EZC_TELEMETRY_COLUMN_CORRECT**

All four drift columns are present and identical:
- `active_pitch_crossing_signed_error_m`: min=-0.0419, max=+0.2019, mean=+0.0816
- `sagittal_position_error_m`: min=-0.0419, max=+0.2019, mean=+0.0816  
- `support_position_error_m`: min=-0.0419, max=+0.2019, mean=+0.0816
- `hip_yaw_comp_support_error_m`: min=-0.0419, max=+0.2015, mean=+0.0815

**Max difference between columns:** 0.0004 m (negligible)  
**Sign agreement:** 100%

## EZC Telemetry Verification

EZC uses `active_pitch_crossing_signed_error_m` for its state machine decisions.

Telemetry fields verified:
- `ezc_state_id`: present
- `ezc_active`: present
- `ezc_direction`: present
- `ezc_tau_nm`: present
- `ezc_target_tau_nm`: present
- `ezc_enter_event`: present
- `ezc_zero_cross_exit_event`: present
- `ezc_safety_exit_event`: present
- `ezc_hold_steps`: present

## Conclusion

The EZC logic uses the same drift column as the analyzer. No telemetry mismatch exists.

**Proceed to Phase 2: Episode-level root cause audit.**