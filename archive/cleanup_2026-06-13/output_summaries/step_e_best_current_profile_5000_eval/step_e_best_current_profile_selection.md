# Step E Best Current Profile Selection

**Date:** 2026-06-05  
**Selection Time:** 14:14:14 CST

## Selected Profile: J3

**Velocity-damped sagittal authority profile with strong damping**

### Profile Parameters
- `k_position`: 80.0 (scheduled, nominal: 40.0)
- `max_position_tau`: 6.0 Nm (scheduled, nominal: 3.0 Nm)
- `k_velocity`: 30.0 (scheduled, nominal: 15.0)
- Schedule type: continuous smoothstep
- Schedule range: z_low=0.300m, z_high=0.393m

## Selection Criteria

Best current profile selected using normalized max violation:

```
normalized_violation = max(
    support_error_max_abs / 0.15,
    hip_yaw_abs_max / 0.07,
    pitch_x_max_abs / 0.10
)
```

Select profile with lowest normalized_violation.

## Candidate Comparison

Based on recent 1000-step evaluations at low_0p300:

| Profile | Support Max | Hip-Yaw Max | Pitch Max | Normalized Violation | Decision |
|---------|-------------|-------------|-----------|----------------------|----------|
| **J2** | 0.1142 m (PASS) | 0.1370 rad (FAIL) | 0.1571 rad (FAIL) | **1.957** | - |
| **J3** | 0.1252 m (PASS) | 0.0884 rad (FAIL) | 0.1513 rad (FAIL) | **1.513** | ✓ SELECTED |

### Why J3?

1. **Lowest normalized violation:** 1.513 vs J2's 1.957 (23% better)
2. **Better hip-yaw control:** 0.0884 rad vs J2's 0.1370 rad (35% improvement)
3. **Comparable pitch:** 0.1513 rad vs J2's 0.1571 rad (4% better)
4. **Support within tolerance:** 0.1252 m still passes 0.15 m threshold

### Trade-offs

- J3 has slightly higher support error (0.1252 vs 0.1142), but both pass the 0.15m gate
- J3's stronger velocity damping (k_velocity=30 vs 25) provides better hip-yaw rejection
- Both profiles currently fail strict pitch gate (0.10 rad), but this is a known boundary limitation at z=0.300m

## Evaluation Plan

Running three 5000-step Step E position-hold evaluations:

1. **low_0p300:** Using `outputs/physical_target_height_setups/low_0p300_setup.json`
2. **nominal:** Using standard nominal height setup
3. **high_0p480:** Using `outputs/physical_target_height_setups/high_0p480_setup.json`

All runs use:
- `--controller-mode balance-core`
- `--sagittal-controller velocity-damped`
- `--vd-sagittal-authority-profile J3`
- 5000 steps (50 seconds simulation time)

## References

- Source data: `outputs/visual_inspection_low_0p300/J2_telemetry.csv`, `J3_telemetry.csv`
- Profile design: `docs/validation/joint_low_height_sagittal_yaw_fix_design.md`
- Pitch-safe candidates (failed): `docs/validation/pitch_safe_joint_sagittal_yaw_fix_report.md`
