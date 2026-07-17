# K2 JAX Mode-Div Hip-Yaw State Parity Fix

**Date:** 2026-06-27
**Classification:** `K2_JAX_MODE_DIV_PARITY_FIX_APPLIED`

---

## 1. Bug

JAX input received `hip_yaw_div_error = joint_pos[1] - joint_pos[6]` (raw joint position difference), while Python mode-div uses `div_error = (l_pos - r_pos) - (l_ref - r_ref)` (divergence from reference). This caused hip-yaw torque divergence growing from ~0.02 Nm to ~0.22 Nm over 50 steps at low_0p330.

## 2. Root Cause

In `simulate_hierarchical_controller.py:6541`, the `pack_input_k2()` call passed:
```python
hip_yaw_div_error=float(joint_pos[1] - joint_pos[6]),
```

Python computes at lines 6448-6465:
```python
l_ref = float(equilibrium_joint_pos[1])
r_ref = float(equilibrium_joint_pos[6])
ref_common, ref_div = decompose(l_ref, r_ref)
_act_common, actual_div = decompose(l_pos, r_pos)
state = HipYawState(div_error=actual_div - ref_div, div_rate=div_rate, ...)
```

Where `decompose()` computes:
- `ref_div = l_ref - r_ref`
- `actual_div = l_pos - r_pos`
- `div_error = actual_div - ref_div = (l_pos - r_pos) - (l_ref - r_ref)`

The JAX input was missing the `ref_div` subtraction.

## 3. Fix

**File:** `scripts/simulate_hierarchical_controller.py`
**Line:** 6541
**Change:**
```python
# Before:
hip_yaw_div_error=float(joint_pos[1] - joint_pos[6]),

# After:
hip_yaw_div_error=float((joint_pos[1] - joint_pos[6]) - (equilibrium_joint_pos[1] - equilibrium_joint_pos[6])),
```

The `hip_yaw_div_rate` computation is unchanged — JAX uses `joint_vel[1] - joint_vel[6]` which matches Python's `div_rate = l_vel - r_vel`.

## 4. Remaining Parameter Gaps

The JAX `k2_jax_mode_div_compute` uses hardcoded defaults:
| Parameter | JAX Default | Python Source | Need Fix? |
|-----------|------------|---------------|-----------|
| `kp_div` | 10.0 | `args.mode_hip_yaw_div_kp` | Match pending |
| `kd_div` | 0.50 | `args.mode_hip_yaw_div_kd` | Match pending |
| `max_torque` | 7.5 | `args.mode_hip_yaw_div_max_torque` | Match pending |
| `soft_limit_rad` | 0.30 | `args.mode_hip_yaw_div_soft_limit_rad` | Match pending |
| `soft_gain` | 0.50 (default) / param override | `args.mode_hip_yaw_div_soft_gain` (0.80 K2) | ✓ Via params |

If Python args match JAX defaults, no further fix needed. Diagnostics will confirm.

## 5. Diagnostics Added

In `both-synced` mode:
- `py_mode_div_error` vs `jax_mode_div_error` — divergence error comparison
- `py_mode_div_rate` vs `jax_mode_div_rate` — divergence rate comparison
- `py_mode_div_height_gate` — height gate value
- `py_mode_div_tau_l/r` — raw torque
- `py_tau[1]` vs `jx_tau[1]` — left hip-yaw torque comparison
- `py_tau[6]` vs `jx_tau[6]` — right hip-yaw torque comparison

## 6. Classification

**`K2_JAX_MODE_DIV_PARITY_FIX_APPLIED`**

Div_error formula fixed. Diagnostics ready for verifying parity.
