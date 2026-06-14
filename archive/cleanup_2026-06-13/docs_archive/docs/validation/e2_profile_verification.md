# E2 Profile Verification Report

## Profile Name
`E2_support_integral_higher_cap`

## Verification Status: PASSED

## E2 Profile Definition

| Field | Value |
|-------|-------|
| profile_name | E2_support_integral_higher_cap |
| applies_to_variants | BOUNDARY_HEIGHT_VARIANTS (low_0p300, high_0p480) |
| opt_in_only | Yes (via --vd-sagittal-authority-profile) |
| D2_baseline_unchanged | Yes |

### Position Integral Settings
| Field | Value |
|-------|-------|
| enable_position_integral | True |
| ki_position_integral | 2.0 |
| integral_max_abs | 1.0 Nm |
| integral_pitch_error_threshold_rad | 0.03 |
| integral_support_velocity_threshold_m_s | 0.03 |
| integral_wheel_velocity_threshold_rad_s | 1.0 |
| integral_min_com_z_m | 0.28 |
| integral_max_com_z_m | 0.50 |

### Position Cap (Key Difference from E1)
| Field | E1 | E2 | Difference |
|-------|-----|-----|------------|
| max_position_tau_nominal | 4.0 Nm | 4.0 Nm | Same |
| max_position_tau_low_max | 4.0 Nm | **5.0 Nm** | +25% increase |

### Other Settings
| Field | Value |
|-------|-------|
| velocity_damping_scale | 1.10 |
| continuous_max_position_tau | True |

## Comparison: E1 vs E2

E2 is the same as E1 except:
1. `max_position_tau_low_max` increased from 4.0 Nm to 5.0 Nm (25% increase)
2. `integral_pitch_error_threshold_rad` is 0.03 (E1 after fix was 0.12)

The pitch threshold difference is notable: E1 was fixed to 0.12, E2 uses 0.03.

## Telemetry Fields (Verified Present)
- `sagittal_schedule_profile`
- `effective_max_position_tau`
- `tau_position_raw`
- `tau_position_final`
- `tau_position_saturated`
- `tau_position_integral`
- `integral_active`
- `support_position_error_m`

## Conclusion
E2 profile is properly defined and wired. The key difference from E1 is the 25% higher position torque cap (5.0 Nm vs 4.0 Nm).