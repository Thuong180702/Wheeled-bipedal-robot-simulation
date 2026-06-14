# Hip-Yaw Kinematic Coupling Analysis

**Date:** 2026-06-05
**Status:** PHASE 4 - Coupling analysis from baseline telemetry

## Objective

Measure kinematic coupling between hip-yaw torques and system response:
1. Can hip-yaw common-mode torque control body yaw?
2. Can hip-yaw divergence-mode torque control leg geometry?
3. Where does authority lie?

## Method

Computed transfer functions using cross-correlation between:
- Input: Hip-yaw torque modes (common and divergence)
- Output: Body yaw angle and hip-yaw position modes

Analyzed 212 steps of baseline telemetry.

## Results

### Common-Mode Torque → Body Yaw

- **Correlation:** -0.122
- **Coupling strength:** very_weak
- **Gain:** 0.096745 rad/Nm
- **Interpretation:** Hip-yaw torque does NOT control body yaw

### Common-Mode Torque → Hip-Yaw Common Position

- **Correlation:** 0.536
- **Coupling strength:** moderate
- **Gain:** 0.028997 rad/Nm
- **Interpretation:** Hip-yaw control is weak

### Divergence-Mode Torque → Hip-Yaw Divergence Position

- **Correlation:** 0.436
- **Coupling strength:** moderate
- **Gain:** 0.018515 rad/Nm
- **Interpretation:** Divergence mode control is weak

## Key Findings

- `hip_yaw_common_torque_weakly_coupled_to_body_yaw`
- `CRITICAL_hip_yaw_kinematically_decoupled_from_body_yaw`

## Conclusion

Hip-yaw common-mode torque controls hip-yaw joint angles (r=0.536) but NOT body yaw rotation (r=-0.122). Kinematic decoupling confirmed.

## Classification

### Body Yaw Authority
**Classification:** `body_yaw_requires_differential_wheel_control`

Hip-yaw common-mode torque cannot control body yaw rotation (r=-0.122).
Body yaw must be controlled through differential wheel velocity.

### Hip-Yaw Divergence Authority
**Classification:** `divergence_mode_controllable_by_hip_yaw`

Hip-yaw divergence-mode torque controls leg geometry (r=0.436).
Divergence posture control should remain on hip-yaw joints.

## Next Steps

1. Design differential wheel velocity controller for body yaw stabilization
2. Design mode-based hip-yaw divergence controller for leg geometry
3. Ensure both controllers have clear ownership and don't conflict
4. Implement and validate
