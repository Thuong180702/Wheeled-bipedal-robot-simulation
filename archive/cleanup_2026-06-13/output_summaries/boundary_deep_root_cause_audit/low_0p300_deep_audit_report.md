# Deep Root-Cause Audit: low_0p300

**Setup:** `outputs/physical_target_height_setups/low_0p300_setup.json`

## A. Reference Consistency

- Target CoM Z: 0.3000 m
- Achieved CoM Z: 0.2955 m
- Root Z: 0.3971 m
- Reference consistent: **True**
- Max joint error: 0.000000e+00 rad

## B. Static Inverse Dynamics / Holding Torque

| Joint | Required (Nm) | PD @ Zero Error | PD @ Threshold | PD @ Observed | Deficit @ Zero |
|-------|---------------|-----------------|----------------|---------------|----------------|
| l_hip_roll   |   -0.00 |   +0.00 |   +0.00 |   +0.00 |   -0.00 |
| l_hip_yaw    |   +0.00 |   +0.00 |   +1.05 |   +2.25 |   +0.00 |
| l_hip_pitch  |   +0.00 |   +0.00 |   +0.00 |   +0.00 |   +0.00 |
| l_knee       |   -0.00 |   +0.00 |   +0.00 |   +0.00 |   -0.00 |
| l_wheel      |   -0.00 |   +0.00 |   +0.00 |   +0.00 |   -0.00 |
| r_hip_roll   |   +0.00 |   +0.00 |   +0.00 |   +0.00 |   +0.00 |
| r_hip_yaw    |   -0.00 |   +0.00 |   +1.05 |   +2.25 |   -0.00 |
| r_hip_pitch  |   +0.00 |   +0.00 |   +0.00 |   +0.00 |   +0.00 |
| r_knee       |   -0.00 |   +0.00 |   +0.00 |   +0.00 |   -0.00 |
| r_wheel      |   -0.00 |   +0.00 |   +0.00 |   +0.00 |   -0.00 |

## Root Cause Indicators

- **Hip yaw has nonzero holding torque:** False
- **Max hip yaw deficit at zero error:** 0.00 Nm
- **Observed yaw error (from previous failures):** 0.150 rad

### ⚠️  Hip yaw holding torque is near-zero

Static inverse dynamics shows negligible hip yaw holding torque requirement.
Root cause may lie elsewhere (support reference, coupling, etc.).
