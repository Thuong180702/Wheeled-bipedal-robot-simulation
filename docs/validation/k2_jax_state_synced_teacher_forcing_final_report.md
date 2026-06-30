# K2 JAX State-Synced Teacher-Forcing — Final Report

**Date:** 2026-06-27
**Classification:** `K2_JAX_STATE_SYNCED_PARITY_IMPROVED`

---

## 1. Summary

State-synced teacher-forcing rerun after three targeted parity fixes:
1. Notch filter state capture (Phase 1-2)
2. Velocity damping scale (Phase 3-4)
3. Support velocity input (previous session)

### Before Fixes
```
Step 4: tau_pitch_rate diff ≈ 0.207 Nm (~6%)
        tau_sagittal_velocity diff ≈ 0.032 Nm (~10%)
        max_abs_diff ≈ 0.21 Nm
```

### After Fixes
```
Step 4: tau_pitch_rate diff = 0.0
        tau_sagittal_velocity diff = 0.0
        max_abs_diff ≈ 0.015 Nm (at hip-yaw indices only)
```

## 2. Per-Source Torque Comparison (Step 4, high_0p480)

| Source | Python | JAX | DIFF | Status |
|--------|--------|-----|------|--------|
| tau_pitch | -2.731020 | -2.731020 | 0.0 | ✓ MATCH |
| tau_pitch_rate | 3.281906 | 3.281906 | 0.0 | ✓ MATCH |
| tau_sagittal_velocity | -0.349612 | -0.349612 | 0.0 | ✓ MATCH |
| tau_support_velocity | 0.0 | 0.0 | 0.0 | ✓ MATCH |
| tau_cp | 0.0 | 0.0 | 0.0 | ✓ MATCH |
| tau_com_vy | -0.105943 | -0.105943 | 0.0 | ✓ MATCH |
| tau_position | -0.172628 | -0.172628 | 0.0 | ✓ MATCH |
| tau_wheel_vel_left | 0.586930 | 0.586930 | 0.0 | ✓ MATCH |
| tau_wheel_vel_right | 0.615046 | 0.615046 | 0.0 | ✓ MATCH |
| Composer | — | — | ~0.015 | ✗ (hip-yaw cascade) |

**All sagittal terms now match exactly.** The remaining ~0.015 Nm difference
is at hip-yaw indices [1,6] only, from a pre-existing posture-path divergence
that predates the fixes in this task.

## 3. Per-Step Max Abs Diff Summary

| Step | Max Abs Diff | Divergent Index | Notes |
|------|-------------|-----------------|-------|
| 0 | 4.77e-08 | 2 (composer float) | Near-perfect (always was) |
| 1 | 1.57e-03 | 6 (hip_yaw_r) | Hip-yaw posture path |
| 2 | 5.17e-03 | 1 (hip_yaw_l) | Hip-yaw posture path |
| 3 | 9.64e-03 | 1 (hip_yaw_l) | Hip-yaw posture path |
| 4 | 1.46e-02 | 6 (hip_yaw_r) | Hip-yaw posture path |

The hip-yaw divergence grows from ~0 to ~0.015 Nm over 4 steps.
It does NOT involve any sagittal terms (all verified matching).

## 4. Remaining Issue

### Root Cause of Hip-Yaw Divergence

The hip-yaw [1,6] torque difference is anti-symmetric (equal magnitude, opposite sign),
consistent with a yaw/mode-div/posture path mismatch. All inputs to the posture path
(q_ref, joint_pos, joint_vel, yaw_err, yaw_rate, div_err, div_rate) match identically.
The gains also match (kp_hip_yaw=15.0, kd_hip_yaw=3.0, kp_yaw=8.0, kd_yaw=2.0).

The remaining hypothesis is a subtle floating-point difference in the mode-div
computation (`k2_jax_mode_div_compute()`) or the yaw computation
(`k2_jax_yaw_compute()`), or a difference in how Python's shape_posture controller
handles the hip-yaw PD computation vs JAX's equivalent.

**Status:** Documented, not blocking functional validation. Requires dedicated
investigation of the posture path (NOT the notch or sagittal paths).

## 5. Classification

**`K2_JAX_STATE_SYNCED_PARITY_IMPROVED`**

- Step 0: Near-perfect (4.77e-08) ✓
- tau_pitch_rate: PERFECT (0.0 diff) ✓ — WAS ~6% mismatch, now FIXED
- tau_sagittal_velocity: PERFECT (0.0 diff) ✓ — WAS ~10% mismatch, now FIXED
- tau_pitch, tau_position, tau_wheel_vel: PERFECT ✓
- Full 10-dim tau: NOT <1e-5 (hip-yaw ~1.5e-02) ✗

**NOT `K2_JAX_STATE_SYNCED_PARITY_PASS`** — hip-yaw divergence still exceeds 1e-5.
**NOT `K2_JAX_STATE_SYNCED_PARITY_FAIL_WITH_ROOT_CAUSE`** — two major fixes applied successfully.
