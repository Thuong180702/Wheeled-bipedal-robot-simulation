# Low 0.300m Harness Debug Report
Date: 2026-06-05

## Phase A: Schedule Telemetry - COMPLETE

Added missing telemetry fields to controller and simulation script:
- `low_height_sagittal_schedule_active`
- `effective_k_position`
- `effective_k_velocity`
- `sagittal_schedule_height_reference_m`
- `sagittal_schedule_height_source`
- `sagittal_schedule_u`
- `sagittal_schedule_smoothstep`

Files changed:
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` (added effective_k_velocity)
- `scripts/simulate_hierarchical_controller.py` (added 7 schedule telemetry fields)

## Phase B: Baseline Behavior - CRITICAL FINDING

**FINDING**: low_0p300 baseline genuinely fails at step 17 with pitch -0.83 rad for ALL profiles (J0, J1, J2, J3).

This is NOT an evaluation harness bug - it's a real simulation failure.

### Simulation Behavior

Command run:
```bash
python scripts/simulate_hierarchical_controller.py \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 1000 \
  --vd-sagittal-authority-profile baseline
```

Result: **Terminated at step 17** with `orientation_fail_pitch_x_-0.83_roll_y_-0.03`

### Timeline of Failure

| Step | CoM Z (m) | Active Wheels | Pitch (deg) | Status |
|------|-----------|---------------|-------------|--------|
| 0    | 0.295     | 2             | -0.0        | OK     |
| 10   | 0.311     | 1             | -16.1       | Degrading |
| 12   | 0.314     | 0             | Contact lost | Critical |
| 17   | 0.306     | 0             | -42.4       | TERMINATED |

### Root Cause Analysis

The robot configuration at z=0.300m is **physically unstable** despite passing static feasibility checks.

Evidence:
1. Wheel contact lost by step 12 (`active_wheels=0`, `actual_fz=0.0N`)
2. Large WBC torques applied (57 Nm) but ineffective without contact
3. Robot tips forward as center of pressure moves behind support polygon

The setup JSON claims:
```json
{
  "achieved_com_z_m": 0.2954845595126816,
  "static_feasible": true,
  "setup_valid": true
}
```

But simulation shows this configuration is **dynamically unstable** - small perturbations from mj_forward cause immediate tip-over.

### Contradiction with User's Claim

User stated: "Previously, low_0p300 baseline ran for 1000 steps with support_error ≈0.243m, hip_yaw ≈0.214 rad, pitch ≈0.095 rad"

**This does not match current behavior** - all profiles terminate at step 17 with identical pitch failure.

Possible explanations:
1. Recent commit 64367b0 (June 2) broke low_0p300 initialization
2. User's memory refers to a different height variant (low_0p330 or low_0p360)
3. Previous runs used different initialization parameters

## Phase C: Next Steps

**BLOCKER**: Cannot proceed with Phase 6 evaluation until low_0p300 initialization is fixed.

Required actions:
1. Investigate commit 64367b0 changes to height variant initialization
2. Check if qvel zeroing or mj_forward timing changed
3. Test if low_0p330 or low_0p360 are stable (sanity check)
4. Determine if low_0p300 is fundamentally too low for this robot morphology

## Status: FIX_HARNESS_STILL_REQUIRED

The harness CLI parameters are correct, but low_0p300 itself is broken.
