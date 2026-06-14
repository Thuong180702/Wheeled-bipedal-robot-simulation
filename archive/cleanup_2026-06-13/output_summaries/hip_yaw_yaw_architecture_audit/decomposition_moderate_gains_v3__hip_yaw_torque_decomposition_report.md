# Hip-Yaw Torque Decomposition Audit Report

**Telemetry:** `outputs/hierarchical_controller_sim/telemetry_1780655374.csv`
**Total steps:** 212

## Mode Decomposition Summary

### Common Mode (Body Yaw Rotation)

- Error RMS: 0.3399 rad
- Error max: 0.4455 rad
- Error final: 0.4020 rad
- Torque RMS: 6.0900 Nm
- Torque max: 23.2787 Nm
- Control correctness: CORRECT: 100.0% of steps have correct sign opposition

### Divergence Mode (Leg Geometry / Twist)

- Error RMS: 0.0470 rad
- Error max: 0.1187 rad
- Error final: -0.0014 rad
- Torque RMS: 1.5412 Nm
- Torque max: 8.8507 Nm
- Control correctness: CORRECT: 78.7% of steps have correct sign opposition

## Failure Classification

- `common_torque_dominates_divergence`

## Diagnosis

## Mode Control Analysis

**Common mode (body yaw):**
  - Opposes error: True
  - Sign correctness: 100.0%
  - Correlation: -0.536
  - RMS torque: 6.090 Nm

**Divergence mode (leg geometry):**
  - Opposes error: True
  - Sign correctness: 78.7%
  - Correlation: -0.436
  - RMS torque: 1.541 Nm

## Failure Mechanisms Detected

- common_torque_dominates_divergence

## Interpretation

**WARNING:** Common-mode torque RMS exceeds divergence-mode torque RMS.
This is unusual - typically symmetric posture control should dominate divergence.
May indicate mode mixing or incorrect decomposition.

## Key Observations


## Recommendations
