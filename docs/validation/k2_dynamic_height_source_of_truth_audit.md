# K2 Dynamic Height Source of Truth Audit

**Date:** 2026-06-29
**Phase:** 5 — AUDIT DYNAMIC HEIGHT SOURCE OF TRUTH

---

## 1. Contradiction Resolution

### Contradiction
Reports conflicted on whether original K2 dynamic height used static or dynamic q_ref, and whether `original-k2-exact` mode is truly "exact."

### Resolution
The original K2 dynamic height validation was run with `--controller-backend jax` on `simulate_hierarchical_controller.py`. This is the canonical monolithic JAX path. The source_file paths in `k2_original_metrics.json` confirm this (e.g., `ramp_up_0p330_to_0p480_K2_JAX` — the `_JAX` suffix is from `f"K2_{args.controller_backend.upper()}"`).

---

## 2. Original K2 Dynamic Height Behavior (Monolithic JAX Path)

### q_ref handling
**STATIC.** `equilibrium_joint_pos` is captured once at initialization from the initial height setup's joint positions (line 4004 of simulate_hierarchical_controller.py). It is NEVER updated during dynamic height.

### LQR gain handling
**DYNAMIC.** The height target is updated per-step from the trajectory:
```python
dynamic_height_target_m = dynamic_height_traj["interp_fn"](step)
height_cmd = dynamic_height_target_m
height_variant_setup["target_com_z_m"] = dynamic_height_target_m
```
The JAX controller receives `height_cmd` via its input, computes `schedule_h = height_ref`, and recomputes all height-dependent gains (LQR K, calibrated outer loop, notch gate, physics FF) using the updated `schedule_h`.

### Termination floor
**FIXED.** Set once before the loop:
```python
termination_height_floor_m = achieved_com_z - 0.05
```
This is NEVER updated during dynamic height. For ramp_up from 0.33m setup: floor = 0.285m. For ramp_down from 0.48m setup: floor = 0.431m.

### Height tracking quality
**POOR BUT SAFE.** With static q_ref + dynamic LQR gains, the CoM stays near the initial posture height. Height RMSE is 0.10-0.11m. The robot survives because the fixed floor is based on the initial achieved CoM, not the target.

---

## 3. Dedicated Runner Dynamic Height Behavior

### q_ref handling
**STATIC** (in `original-k2-exact` mode). Matches monolithic path.

### LQR gain handling
**DYNAMIC.** Matches monolithic path — `commanded_height_ref_m` is updated per-step from trajectory and passed to JAX controller.

### Termination floor
**DYNAMIC (BUG).** Updated every step:
```python
height_floor = height_ref - 0.05
```
This causes `height_floor` to TRACK the target height. During ramp_up from 0.33→0.48m:
- Step 0: floor = 0.33 - 0.05 = 0.28m
- Step 1509: height_ref ≈ 0.38m, floor ≈ 0.33m
- CoM stuck at ~0.33m → falls below floor → termination!

This is the PRIMARY ROOT CAUSE of dynamic height falls.

---

## 4. First Divergent Control-Affecting Field

| Field | Monolithic | Dedicated | Match? |
|---|---|---|---|
| q_ref | Static (equilibrium_joint_pos) | Static (eq_joint) | ✅ |
| height_ref (commanded) | Dynamic from trajectory | Dynamic from trajectory | ✅ |
| LQR gain scheduling | Dynamic via schedule_h | Dynamic via schedule_h | ✅ |
| Termination floor | FIXED (achieved_com_z - 0.05) | DYNAMIC (height_ref - 0.05) | ❌ |
| Height floor update | Never | Every step | ❌ |

**First divergent field: termination height floor.**

---

## 5. What Original K2 Actually Used

| Question | Answer |
|---|---|
| Static q_ref? | YES |
| Dynamic q_ref recomputation? | NO |
| Precomputed trajectory q_ref? | NO |
| Different initial height setup? | NO — same setup files |
| Different termination floor? | YES — FIXED, not dynamic |
| Different actuator/torque composition? | NO |

---

## 6. Fix Strategy

1. **Primary fix:** Use FIXED `height_floor = achieved_com_z - 0.05` during dynamic height (match monolithic path).
2. **Secondary:** `gate_dwell` hip_yaw > 0.35 requires separate Phase 7 investigation — likely a porting gap in mode_div or ABS/APCR1ND trim.

---

## 7. Acceptance

- [x] Determined original K2 dynamic height uses static q_ref + dynamic LQR gains
- [x] Identified termination floor as first divergent field
- [x] Root cause: dedicated runner dynamically updates height_floor to track target
- [x] Fix identified: use fixed floor matching monolithic path
