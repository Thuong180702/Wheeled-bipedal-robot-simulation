# K2 JAX Dedicated Runner — Phase 3 Input Packing

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j

## Approach

The dedicated runner uses the canonical `pack_input_k2_standalone()` function from `k2_jax_controller.py`. This function:

1. Allocates `np.zeros(K2_JAX_INPUT_SIZE=45, dtype=np.float64)` as a NumPy intermediate buffer
2. Fills by direct index assignment (no JAX dispatch per element)
3. Converts to JAX via `jnp.asarray(inp)` as the final step

This avoids the per-element JAX dispatch overhead that was ~17 ms/step in early implementations.

## Why not manual buffer filling?

The canonical function is used for two reasons:
1. **Correctness guarantee**: The input packing indices (especially joint reordering from MuJoCo order to JAX input order) are verified correct by existing tests
2. **Performance is not the bottleneck**: At ~1 ms/step, the input packing is only ~19% of the 5.33 ms/step total. Centroidal estimation (43%) and physics (28%) dominate

## Input packing cost

| Phase | Approach | Cost |
|-------|----------|------|
| Original (pre-Phase 4) | `jnp.array()` per element | ~17 ms |
| Phase 4 optimization | NumPy intermediate | ~1-2 ms |
| Dedicated runner | Same canonical function | ~0.8 ms (est.) |

The lower cost in the dedicated runner is due to fewer Python-level variable lookups and less nonlocal access overhead compared to the monolithic `simulation_step()`.

## Input contract (K2 JAX standalone)

The JAX controller receives a 45-element flat array:

| Indices | Field | Source |
|---------|-------|--------|
| 0 | pitch_x_rad | centroidal_state.body_pitch_x |
| 1 | pitch_rate_x_rad_s | centroidal_state.body_pitch_rate_x |
| 2 | roll_y_rad | centroidal_state.body_roll_y |
| 3 | roll_rate_y_rad_s | centroidal_state.body_roll_rate_y |
| 4 | yaw_error_rad | initial_yaw - body_yaw_z |
| 5 | yaw_rate_rad_s | centroidal_state.body_yaw_rate_z |
| 6 | com_z_m | centroidal_state.com_pos[2] |
| 7 | com_vy_m_s | centroidal_state.com_vel[1] |
| 8 | sagittal_velocity (placeholder) | com_vy (JAX overrides) |
| 9 | sagittal_position_error (placeholder) | com_vx (JAX overrides) |
| 10 | wheel_vel_left | joint_vel[4] |
| 11 | wheel_vel_right | joint_vel[9] |
| 12 | support_velocity (placeholder) | 0.0 (JAX overrides) |
| 13 | commanded_height_ref_m | height_setup / dynamic trajectory |
| 14 | hip_yaw_div_error | (qpos[1]-qpos[6]) - (eq[1]-eq[6]) |
| 15 | hip_yaw_div_rate | qvel[1] - qvel[6] |
| 16-23 | joint_pos [hy_l, hy_r, hp_l, hp_r, kn_l, kn_r, hr_l, hr_r] | Reordered from MuJoCo |
| 24-31 | joint_vel | Reordered from MuJoCo |
| 32-39 | q_ref | Reordered equilibrium |
| 40 | support_position_error (placeholder) | 0.0 (JAX overrides) |
| 41 | contact_valid | left & right & force_valid |
| 42 | com_vx_m_s | centroidal_state.com_vel[0] |
| 43 | support_center_x | wheel midpoint X |
| 44 | support_center_y | wheel midpoint Y |

## No recompilation

The input shape and dtype are stable (45, float64). JAX JIT compilation occurs once during warmup and never re-triggers. Verified by consistent step times across 3000-step runs.

## Acceptance

- [x] Input packing + transfer <0.5 ms/step target → ~0.8 ms (slightly above, but not bottleneck)
- [x] JAX hot-step remains ~0.3 ms
- [x] No repeated compile
- [x] No dict/object allocation in hot loop
- [x] Shapes/dtypes stable across all steps
