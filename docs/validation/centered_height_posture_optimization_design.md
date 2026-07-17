# Centered Height Posture Optimization Design

**Date:** 2026-06-19  
**Status:** READY  

## Context

The Phase 1 posture geometry audit revealed:

1. **Static sagittal CoM-x is already centered** (abs(com_support_error_x) <= 5e-6 m at all 10 heights). The previous pitch_ref_offset was compensating dynamic control coupling, not a static geometry error.

2. **Lateral CoM-y is biased at 6/10 heights** (abs(com_support_error_y) > 0.005 m), with values up to ±0.017 m.

3. **The hip_pitch_ref/knee_ref height schedule is non-monotonic** at two transitions (0.330→0.340 and 0.360→0.380), caused by the 0.075-rad coarse grid step in the original search.

## Optimizer Design

### Decision Variables

| Variable | Range | Grid Step | Purpose |
|----------|-------|-----------|---------|
| `hip_pitch_ref` | [0.3, 2.5] rad | 0.015 rad (5× finer than original) | Smooth monotonic height schedule |
| `knee_ref` | [1.0, 2.8] rad | 0.020 rad (3.75× finer) | Smooth height schedule |
| `hip_roll_left` | [-0.03, +0.03] rad | 0.005 rad | Lateral CoM centering |
| `hip_roll_right` | [-0.03, +0.03] rad | 0.005 rad | Lateral CoM centering |

Hip_yaw is fixed at zero.

### Multi-Phase Search

**Phase A** — 2D grid over (hip_pitch, knee), symmetric, no hip_roll. Each candidate is root-z calibrated, forward-kinematics evaluated, and scored. Keep top-K.

**Phase B** — For each Phase-A top-K candidate, 2D grid over (hip_roll_left, hip_roll_right). This decouples the lateral fix from the sagittal posture search.

**Phase C** — Fit smooth monotone PCHIP through the Phase-B optimized values. Re-evaluate at smoothed values to produce the final centered posture set.

### Objective Function

```
J = w_height * |com_z - target|^2
  + w_com_xy * com_support_error_norm
  + w_pitch * pitch_x^2
  + w_roll * roll_y^2
  + w_yaw * yaw_z^2
  + w_hip_roll * (|l_hip_roll| + |r_hip_roll|)/2
  + joint_limit_penalty (soft below 0.05 rad margin)
  + contact_penalty (missing wheel contact or extra contact)
  + smoothness_penalty (deviation from neighbor-smoothed prior)
```

### Hard Constraints

| Constraint | Limit | Rationale |
|------------|-------|-----------|
| sagittal CoM error | ≤ 0.005 m | Preserve existing centering |
| lateral CoM error | ≤ 0.005 m | New target for lateral bias fix |
| height error | ≤ 0.005 m (preferred) ≤ 0.010 m (max) | Height tracking |
| wheel contacts | Both valid | Must stand on wheels |
| non-wheel contacts | 0 | Clean contact state |
| joint limit margin | ≥ 0.05 rad | Safety |
| hip_roll magnitude | ≤ 0.05 rad per side | Small corrections only |
| hip_yaw | 0 | Avoid yaw coupling |

## Output Artifacts

- `outputs/physical_target_height_setups_centered/*_setup.json` — 10 centered setup JSONs
- `centered_posture_summary.csv` — per-height metrics table
- `centered_posture_summary.json` — structured summary
- `centered_posture_height_functions.json` — calibration artifact for continuous height functions

## Classification

`CENTERED_POSTURE_OPTIMIZATION_DESIGN_READY`
