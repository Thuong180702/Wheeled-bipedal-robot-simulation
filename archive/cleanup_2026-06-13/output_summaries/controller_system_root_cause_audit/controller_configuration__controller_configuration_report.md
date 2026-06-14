# Controller Configuration Audit Report

**Date:** 2026-06-05
**Phase:** Phase 2

## Summary

This audit identifies the active controller configuration used in the Step E 5000 evaluations.

## Controller Mode

**Active Mode:** `balance-core`

| Property | Value |
|----------|-------|
| WBC Status | diagnostic_only |
| Torque Sources | shape_posture, support_feedforward, sagittal_wheel, lateral_roll |
| Torque Composer | BalanceCoreTorqueComposer |

## Sagittal Controller

**Active Controller:** `velocity-damped`

| Property | Value |
|----------|-------|
| Controller Class | SagittalVelocityDampedBalanceController |
| Selected Profile | J3 |
| Profile Type | JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING |

### Profile Parameters

| Parameter | Value |
|-----------|-------|
| k_position | 80.0 |
| max_position_tau | 6.0 Nm |
| k_velocity | 30.0 |
| schedule_type | continuous_smoothstep |
| z_low | 0.300 m |
| z_high | 0.393 m |

## Controller Enabled States by Variant

| Variant | hip_yaw_div | hip_yaw_comp | yaw_aware | boundary_sagittal |
|---------|-------------|--------------|-----------|-------------------|
| low_0p300 | False | False | True | True |
| nominal | False | False | True | False |
| high_0p480 | False | False | True | True |

## Key Observations

1. **WBC is diagnostic-only:** The balance-core mode correctly specifies WBC as diagnostic-only, meaning it should not contribute to final torque.

2. **Yaw-aware compensation is active:** All variants have `yaw_aware_position_compensation_active=True`, meaning the sagittal controller accounts for yaw-induced position error.

3. **Hip-yaw divergence damping is disabled:** All variants show `hip_yaw_div_active=False`, meaning the HY2-DIV divergence damping is not applied.

4. **Hip-yaw support feedforward is disabled:** All variants show `hip_yaw_comp_active=False`, meaning the HY-FF support feedforward is not applied.

5. **Boundary sagittal authority is enabled:** Only boundary variants (low_0p300, high_0p480) have boundary_sagittal_authority=True.

## Controller Flags Summary

| Flag | low_0p300 | nominal | high_0p480 |
|------|-----------|---------|------------|
| balance-core mode | ✓ | ✓ | ✓ |
| velocity-damped sagittal | ✓ | ✓ | ✓ |
| J3 profile | ✓ | ✓ | ✓ |
| hip_yaw_div | ✗ | ✗ | ✗ |
| hip_yaw_comp | ✗ | ✗ | ✗ |
| yaw_aware | ✓ | ✓ | ✓ |
| hip_yaw_integral | ✗ | ✗ | ✗ |
| boundary_sagittal | ✓ | ✗ | ✓ |

## Active Controller Paths

```
shape_posture_controller.py
  └── ShapePostureController.compute()
      ├── Hip-yaw PD control (indices 1, 6)
      ├── Hip-pitch PD control (indices 2, 7)
      └── Knee PD control (indices 3, 8)

sagittal_velocity_damped_balance_controller.py
  └── SagittalVelocityDampedBalanceController.compute()
      ├── Position control (k_position=80.0)
      ├── Velocity damping (k_velocity=30.0)
      ├── Pitch control
      └── Wheel velocity damping

balance_core_torque_composer.py
  └── BalanceCoreTorqueComposer.compose()
      ├── tau_shape_posture
      ├── tau_support_feedforward
      ├── tau_sagittal_wheel_balance
      ├── tau_lateral_roll_balance
      └── Final clipping and rate limiting
```

## Files

- Configuration summary: `outputs/controller_system_root_cause_audit/controller_configuration/controller_configuration_summary.json`