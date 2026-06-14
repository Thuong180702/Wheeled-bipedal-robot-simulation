# HY2-DIV Implementation Verification

**Date:** 2026-06-05
**Status:** PASS

## Verification Checklist

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | HY2-DIV disabled by default | PASS | `enable_hip_yaw_divergence_damping=False` in `ShapePostureController.__init__` |
| 2 | Requires explicit flag | PASS | `--enable-hip-yaw-divergence-damping` required in simulate_hierarchical_controller.py |
| 3 | Uses target_com_height for gate | PASS | `target_com_height` parameter, line 227 |
| 4 | Height gate is continuous | PASS | smoothstep: `3.0 * u**2 - 2.0 * u**3`, line 228 |
| 5 | Torque clamp applied | PASS | `jnp.clip()` on lines 243-244 (bug fixed) |
| 6 | Torque opposes divergence | PASS | Lines 239-240: antisymmetric torque |
| 7 | Only affects hip-yaw | PASS | Lines 254-259: only indices [1, 6] |
| 8 | No hip-roll changes | PASS | Hip-roll not in hip-yaw PD loop |
| 9 | WBC remains diagnostic-only | PASS | `is_balance_core_mode(args)` check |
| 10 | Ownership clean | PASS | HY2-DIV torques added to shape_posture, not wbc |

## Implementation Details

### Height Gate Formula
```python
u = clip((z_high - z_ref) / (z_high - z_low), 0.0, 1.0)
div_gate = 3.0 * u**2 - 2.0 * u**3  # smoothstep
```

Default range: z_low=0.300m, z_high=0.393m

### Divergence Damping Formula
```python
divergence = l_error - r_error  # l_hip_yaw - r_hip_yaw
divergence_rate = l_vel - r_vel  # l_hip_yaw_vel - r_hip_yaw_vel

tau_div_L = -(k_div * divergence + kd_div * divergence_rate) * gate
tau_div_R = +(k_div * divergence + kd_div * divergence_rate) * gate
```

### Telemetry Fields
- `hip_yaw_div_active` - boolean, HY2-DIV enabled
- `hip_yaw_div_height_gate` - float [0,1], gate value
- `hip_yaw_div_left` / `hip_yaw_div_right` - raw torques
- `hip_yaw_div_left_clipped` / `hip_yaw_div_right_clipped` - clipping flags
- `hip_yaw_div_k_divergence` / `hip_yaw_div_k_divergence_rate` / `hip_yaw_div_tau_max` - gains

## Conclusion

Implementation is correct. No bugs found.

## Next Step

Proceed to Phase 2: Candidate Design.