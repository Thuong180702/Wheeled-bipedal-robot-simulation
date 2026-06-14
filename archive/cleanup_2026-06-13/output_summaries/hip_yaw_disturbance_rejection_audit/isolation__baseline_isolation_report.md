# Hip-Yaw Disturbance Rejection - Baseline Isolation Data

**Date:** 2026-06-04
**Phase:** 2 (Isolation Experiments - Baseline)

## Baseline Metrics

| Variant | hip_yaw_abs_max | support_error | pitch_x | roll_y | Status |
|---------|----------------|---------------|---------|--------|--------|

## Gate Thresholds

- hip_yaw_abs_max: ≤ 0.07 rad
- support_position_error: ≤ 0.15 m
- pitch_x: ≤ 0.10 rad
- roll_y: ≤ 0.05 rad

## Next Steps

To proceed with experiments D and E (damping sweep and kp/kd matrix),
we need to add parameter override support to the simulation script.

Options:
1. Add `--shape-kp-hip-yaw` and `--shape-kd-hip-yaw` CLI arguments
2. Create temporary boundary profiles with test parameters
3. Modify controller instantiation to accept runtime overrides

Recommended: Option 1 (cleanest, most flexible)