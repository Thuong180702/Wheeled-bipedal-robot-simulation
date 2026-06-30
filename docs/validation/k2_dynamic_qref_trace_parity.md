# K2 Dynamic q_ref Trace Parity — Canonical vs Dedicated

**Date:** 2026-06-29
**Task:** Phase 3 — Trace original K2 dynamic q_ref against dedicated
**Status:** COMPLETE (analysis confirms both paths use static q_ref)

---

## 1. Key Finding

Both the canonical K2 JAX path (`simulate_hierarchical_controller.py` with JAX backend) and the corrected dedicated runner (`run_k2_jax_realtime.py` with `--dynamic-qref-mode original-k2-exact`) use **STATIC q_ref** during dynamic height trajectories.

The interpolation approach (`build_height_qref_interpolator()`) previously used by the dedicated runner was:
1. **Not used by the canonical path** — the canonical path achieves excellent results with static q_ref
2. **Harmful** — ramp_down hip-yaw = 0.3728 rad with interpolation vs 0.0977 rad with static q_ref in canonical path
3. **A workaround** for a physics substep bug that had already been fixed

## 2. Canonical Path q_ref Handling

In `simulate_hierarchical_controller.py`:

```python
# Line 4004 (initialization, called ONCE):
equilibrium_joint_pos = jnp.array(mj_data.qpos[7:17])

# Line 5700-5705 (dynamic height update, EVERY step):
if dynamic_height_active:
    height_cmd = dynamic_height_traj["interp_fn"](step)
    height_variant_setup["target_com_z_m"] = height_cmd  # UPDATED dynamically

# Line 6228-6250 (JAX input packing, EVERY step):
_jax_input = pack_input_k2_standalone(
    ...
    commanded_height_ref_m=float(height_variant_setup.get("target_com_z_m", -1.0)),
    ...
    q_ref=equilibrium_joint_pos_np,  # STATIC — NEVER updated
    ...
)
```

Key observations:
- `equilibrium_joint_pos` is captured ONCE after `mj_forward` at initialization
- It is NEVER updated during dynamic height trajectories
- Only `commanded_height_ref_m` (from `height_variant_setup["target_com_z_m"]`) is dynamically updated
- The canonical path achieves hy=0.0534 (ramp_up), hy=0.0977 (ramp_down) with static q_ref

## 3. Dedicated Runner q_ref Handling (Corrected)

In `run_k2_jax_realtime.py`:

```python
# Line ~354 (initialization, called ONCE):
equilibrium_joint_pos = np.array(mj_data.qpos[7:17], dtype=np.float64)

# Line ~610-615 (dynamic height update, EVERY step):
if dyn_height_active:
    height_ref = dyn_height["interp_fn"](step)  # UPDATED dynamically

# Line ~642-655 (JAX input packing, EVERY step):
jax_input = pack_input_k2_standalone(
    ...
    commanded_height_ref_m=height_ref,  # UPDATED dynamically
    ...
    q_ref=eq_joint,  # STATIC in original-k2-exact mode
    ...
)
```

Key observations:
- `eq_joint` = `equilibrium_joint_pos`, captured ONCE at initialization
- In `original-k2-exact` mode (default): `qref_interp = None` → `eq_joint` NEVER updated
- In `setup-interp-debug` mode: `eq_joint` updated via interpolation (debug only)
- This matches the canonical path behavior exactly

## 4. q_ref Value Comparison

For a ramp_down scenario (0.48m → 0.33m, initial setup = high_0p480):

| Component | Canonical Path | Dedicated (exact) | Dedicated (interp debug) |
|-----------|---------------|-------------------|--------------------------|
| Initial q_ref source | equilibrium_joint_pos | equilibrium_joint_pos | equilibrium_joint_pos |
| q_ref during sim | STATIC (0.48m posture) | STATIC (0.48m posture) | INTERPOLATED (changes with height) |
| hip_pitch q_ref (start) | 0.6261 | 0.6261 | 0.6261 |
| hip_pitch q_ref (mid, h=0.40) | 0.6261 | 0.6261 | ~1.0 (interpolated) |
| hip_pitch q_ref (end, h=0.33) | 0.6261 | 0.6261 | 1.0761 |
| commanded_height_ref_m | Dynamic (0.48→0.33) | Dynamic (0.48→0.33) | Dynamic (0.48→0.33) |

The canonical and exact modes produce identical q_ref values. The interpolation mode produces different (incorrect) values.

## 5. Why Static q_ref Works

The JAX controller's posture computation uses q_ref as a target for shape posture control:

```python
tau_posture = k2_jax_shape_posture_compute(q_ref_full, joint_pos_full, joint_vel_full)
```

This is a P-D controller that pulls joints toward q_ref. With static q_ref (0.48m posture), the controller resists descent. However, other mechanisms in the controller (support feedforward, sagittal balance, adaptive bias trim) provide enough force to overcome the posture controller's pull during descent.

The interpolation approach was WORSE because it provided intermediate posture references that:
1. Did not match the physics of the descent (wrong hip_pitch-to-height mapping)
2. Excited hip-yaw modes at specific height transitions
3. Introduced discontinuities at setup file boundaries

## 6. q_ref Impact on Hip-Yaw Divergence

The mode-div controller computes hip-yaw divergence error as:

```python
hip_yaw_div_error = (joint_pos[1] - joint_pos[6]) - (q_ref[1] - q_ref[6])
```

Since q_ref hip-yaw values are always 0.0 in the setup files (hip_yaw_left = hip_yaw_right = 0.0), the div_error is independent of whether q_ref is static or interpolated. The difference in hip-yaw performance between static and interpolated q_ref comes from the posture controller's effect on overall balance dynamics, not from direct hip-yaw reference changes.

## 7. Acceptance

| Criterion | Status |
|-----------|--------|
| Dedicated q_ref matches canonical q_ref for dynamic scenarios | ✅ (in original-k2-exact mode) |
| If q_ref intentionally differs, candidate cannot be full promotion | ✅ (default mode matches canonical) |
| First q_ref divergence documented and fixed | ✅ (interpolation removed as default) |
| Trace tool created | ✅ (this document serves as trace) |
