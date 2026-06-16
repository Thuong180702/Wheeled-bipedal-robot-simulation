# Phase 7: Leg-Yaw / Hip-Yaw Stability Audit

**Profile:** `height_scheduled_pitch_equilibrium_trim` (sched) vs `adaptive_support_centering_trim` (offset-0 baseline)

**Classification: `LEG_YAW_HIP_YAW_STABLE`**

The height-scheduled pitch_ref offset is a sagittal coordination change. This audit confirms it does not couple into hip-yaw instability: hip-yaw angle, yaw drift growth, and left/right asymmetry stay bounded and are not materially worse than the accepted adaptive baseline at any height.

## Per-height hip-yaw metrics (sched profile)

| height | hy_abs_max (rad) | baseline hy_abs_max | yaw_drift_max | yaw_drift_growth | lr_asym_rms | hy-drift corr | verdict |
|---|---|---|---|---|---|---|---|
| low_0p300 | 0.2052 | 0.2035 | 0.0110 | 0.0113 | 0.0058 | -0.30 | STABLE |
| low_0p320 | 0.1699 | 0.1873 | 0.0239 | 0.0145 | 0.0088 | -0.40 | STABLE |
| low_0p330 | 0.2034 | 0.2139 | 0.0474 | 0.0432 | 0.0113 | -0.50 | STABLE |
| low_0p340 | 0.1709 | 0.1709 | 0.0205 | 0.0297 | 0.0094 | -0.40 | STABLE |
| low_0p360 | 0.1813 | 0.1992 | 0.0604 | 0.0693 | 0.0164 | -0.53 | STABLE |
| low_0p380 | 0.0488 | 0.2712 | 0.0949 | 0.0891 | 0.0100 | -0.61 | STABLE |
| high_0p430 | 0.0378 | 0.1177 | 0.0481 | 0.0567 | 0.0077 | -0.29 | STABLE |
| high_0p450 | 0.0691 | 0.0941 | 0.0479 | 0.0542 | 0.0163 | -0.19 | STABLE |
| high_0p465 | 0.0487 | 0.0337 | 0.0956 | 0.0997 | 0.0123 | -0.06 | STABLE |
| high_0p480 | 0.0412 | 0.0667 | 0.0640 | 0.0768 | 0.0074 | 0.09 | STABLE |

## Verdict criteria

- **UNSAFE** if hy_abs_max > 0.35 rad, yaw_drift_growth > 0.15 rad, or lr_asym_rms > 0.10 rad.
- **MONITORING** if hy_abs_max or yaw_drift_max is > 0.05 rad worse than the adaptive baseline at that height, or a large-amplitude high-frequency oscillation is present.
- **STABLE** otherwise.
