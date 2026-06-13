# Balance-Core True Height Variant Dynamic Validation Report (B5-B10)

## Validation Method

Progressive passive simulation with support feedforward only (no active control).

## Support Feedforward Configuration (B5)

- **Vector**: [0.0, 0.0, 4.1, -15.5, 0.0, 0.0, 0.0, 3.2, -15.8, 0.0]
- **Scale**: 0.5
- **Joint group**: hip_pitch_knee
- **Indices**: [2, 3, 7, 8]

## Summary

- **Variants tested**: 5
- **Total validation runs**: 5
- **Support feedforward consistency**: 0/5 (0.0%)

## Maximum Confirmed Steps Per Variant

- **high_small**: 363 steps
- **high_tiny**: 462 steps
- **low_small**: 522 steps
- **low_tiny**: 367 steps
- **nominal**: 403 steps

## Failures

### nominal (failed at step 403)

- **Primary root cause**: pitch_divergence
- **Secondary causes**: support_feedforward_mismatch, yaw_drift_issue, height_drift
- **Responsible component**: SagittalWheelBalanceController or SupportFeedforwardController
- **Recommended fix scope**: sagittal_balance_or_support_feedforward

### high_tiny (failed at step 462)

- **Primary root cause**: height_collapse
- **Secondary causes**: support_feedforward_mismatch, height_drift
- **Responsible component**: ShapePostureController or SupportFeedforwardController
- **Recommended fix scope**: posture_reference_or_support_feedforward

### high_small (failed at step 363)

- **Primary root cause**: pitch_divergence
- **Secondary causes**: support_feedforward_mismatch, yaw_drift_issue, height_drift
- **Responsible component**: SagittalWheelBalanceController or SupportFeedforwardController
- **Recommended fix scope**: sagittal_balance_or_support_feedforward

### low_tiny (failed at step 367)

- **Primary root cause**: pitch_divergence
- **Secondary causes**: support_feedforward_mismatch, yaw_drift_issue, height_drift
- **Responsible component**: SagittalWheelBalanceController or SupportFeedforwardController
- **Recommended fix scope**: sagittal_balance_or_support_feedforward

### low_small (failed at step 522)

- **Primary root cause**: height_collapse
- **Secondary causes**: support_feedforward_mismatch
- **Responsible component**: ShapePostureController or SupportFeedforwardController
- **Recommended fix scope**: posture_reference_or_support_feedforward


## Controller Status

- **WBC**: off
- **Ownership violations**: 0
- **Four-source stack**: unchanged
