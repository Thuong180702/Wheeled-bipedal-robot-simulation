# Balance-Core Failure Classification Report

**Primary Failure Mode:** F2.1
**First Threshold Crossing:** Step 165 (t=1.650s)
**Responsible Component:** SagittalWheelBalanceController
**Fix Allowed in Balance-Core:** Yes

## Recommended Fix Scope

SagittalWheelBalanceController: verify inputs, sign, saturation, then adjust gains

## Secondary Threshold Crossings

- **F1.2** at step 185 (t=1.850s): value=0.3534, threshold=0.3541

## Evidence Fields

- **primary_failure_value:** 0.3006
- **primary_failure_threshold:** 0.3000
- **pitch_max_rad:** 0.4889
- **pitch_rate_max_rad_s:** 0.0216
