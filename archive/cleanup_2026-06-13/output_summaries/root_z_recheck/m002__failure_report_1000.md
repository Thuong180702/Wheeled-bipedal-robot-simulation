# Balance-Core Failure Classification Report

**Primary Failure Mode:** F2.1
**First Threshold Crossing:** Step 12 (t=0.120s)
**Responsible Component:** SagittalWheelBalanceController
**Fix Allowed in Balance-Core:** Yes

## Recommended Fix Scope

SagittalWheelBalanceController: verify inputs, sign, saturation, then adjust gains

## Evidence Fields

- **primary_failure_value:** -0.3077
- **primary_failure_threshold:** 0.3000
- **pitch_max_rad:** 0.5324
- **pitch_rate_max_rad_s:** 0.0363
