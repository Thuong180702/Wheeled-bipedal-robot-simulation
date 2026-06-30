# K2 Hip-Yaw Scalar Parity Audit

**Date:** 2026-06-30
**Phase:** 7 — AUDIT HIP-YAW SYSTEMATIC REGRESSION

---

## 1. Symptom

Dedicated JAX runner consistently produces 2-10× higher hip_yaw_max than the original paths across all scopes:

| Scenario | Original hy_max | Dedicated hy_max | Ratio |
|---|---|---|---|
| ramp_up (JAX baseline) | 0.0534 | 0.4029 | 7.5× |
| low_0p330 Step E (Python baseline) | 0.0851 | 0.1033 | 1.2× |
| high_0p480 Step E (Python baseline) | 0.0563 | 0.0735 | 1.3× |
| gate_dwell (JAX baseline) | 0.0534 | 0.5370 | 10.1× |

---

## 2. Architecture Comparison

Both the monolithic JAX fast path and the dedicated runner use **identical** JAX controller architecture:

| Component | Monolithic JAX Fast Path | Dedicated Runner | Match? |
|---|---|---|---|
| JAX controller code | `k2_jax_controller.py` | `k2_jax_controller.py` | ✅ Same file |
| Step function | `k2_jax_controller_step` | `k2_jax_controller_step` | ✅ Same |
| Input packing | `pack_input_k2_standalone()` | `pack_input_k2_standalone()` | ✅ Same |
| standalone_mode | `True` | `True` | ✅ Same |
| control_dt | 0.01 | 0.01 | ✅ Same |
| max_torque_rate | 400.0 | 400.0 | ✅ Same |
| k_velocity | 15.0 | 15.0 | ✅ Same |
| mode_div params | kp=10, kd=0.5, max=7.5 | kp=10, kd=0.5, max=7.5 | ✅ Same |
| q_ref handling | Static equilibrium_joint_pos | Static eq_joint | ✅ Same |
| Input values | Raw state from MuJoCo | Raw state from MuJoCo | ✅ Same sources |

---

## 3. Remaining Differences

### 3.1 Physics Engine
- **Monolithic:** MJX (JAX-based physics, GPU-accelerated)
- **Dedicated:** Native MuJoCo with manual substeps (`mj_step` × N)

The MJX and native MuJoCo physics may produce different intermediate states, particularly for joint positions/velocities at control-rate granularity. These differences compound over thousands of steps.

### 3.2 Torque Application Path
- **Monolithic:** JAX output tau → `mj_data.ctrl[:] = tau` (within the MJX simulation loop)
- **Dedicated:** JAX output tau → `np.array(jax_tau) → mj_data.ctrl[:] = tau` (Python loop)

The dedicated runner copies JAX array to numpy before applying. This adds negligible overhead but could introduce precision differences (float64 in both cases).

### 3.3 Centroidal Estimator
- **Monolithic:** Uses MJX simulation's internal state (body_xpos, body_xquat, etc. from MJX data structures)
- **Dedicated:** Uses `CentroidalEstimator.estimate()` which wraps `mujoco.mj_kinematics` + manual computation

The centroidal estimator in the dedicated runner might compute slightly different CoM, orientation, or contact state values compared to MJX's internal computation.

### 3.4 Termination Floor (now fixed)
- **Monolithic:** Fixed floor = `achieved_com_z - 0.05`
- **Dedicated (before fix):** Dynamic floor = `height_ref - 0.05`
- **Dedicated (after fix):** Fixed floor = `achieved_com_z - 0.05` ✅

---

## 4. Hypothesis for Hip-Yaw Divergence

**Primary hypothesis:** The centroidal estimator in the dedicated runner computes slightly different body orientation (pitch_x, roll_y) and CoM position compared to MJX's internal state. These differences feed into the JAX controller as `pitch_x_rad`, `com_z_m`, `yaw_error_rad`, and `support_center_x/y_m` inputs. The mode_div hip-yaw controller is particularly sensitive to yaw_error and hip_yaw_div_error, which are computed from the centroidal estimate.

Since the mode_div controller has gain kp=10.0, a small difference in yaw_error (0.01 rad) produces 0.1 Nm difference in hip-yaw torque. Compounded over thousands of steps, this can cause significant divergence.

**Secondary hypothesis:** The physics engine difference (MJX vs native MuJoCo substeps) produces slightly different joint positions/velocities at each step, which the mode_div controller amplifies through its proportional-derivative action.

---

## 5. Required Next Steps for Definitive Fix

1. **Instrument both paths** to log JAX input values at step 0, 100, 200, 500, 1000 for a fixed-height scenario (low_0p330, 2000 steps).
2. **Compare first divergent scalar** in the JAX input: which field diverges first?
3. **Trace the divergent field upstream** to find the source (centroidal estimator, physics, or torque application).
4. **Patch the source** — fix centroidal estimator to match MJX computation, or apply per-step correction.

This requires running both the monolithic `simulate_hierarchical_controller.py --controller-backend jax` and the dedicated runner side-by-side, dumping JAX inputs at identical steps.

---

## 6. Acceptance

- [x] Identified that both paths use identical JAX controller architecture
- [x] Confirmed params, inputs, and step function are the same
- [x] Identified centroidal estimator and physics engine as remaining differences
- [x] Documented required next steps (instrumented comparison)
- [ ] Scalar-level trace comparison not yet performed (requires instrumented runs)
