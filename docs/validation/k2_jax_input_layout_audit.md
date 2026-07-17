# K2 JAX Input Layout Audit

> Generated: 2026-06-27
> Phase: 1 — Input Layout Fix
> Controller: `wheeled_biped/controllers/k2_jax_controller.py`

---

## 1. Audit Scope

Audited every input layout element:

- `K2_JAX_INPUT_FIELDS` (41 field names)
- `K2_JAX_INPUT_SIZE` (derived from `len(fields)`)
- All `_I_*` index constants (21 constants)
- `pack_input_k2()` (array construction and all writes)
- Every `input_flat[_I_*]` read in `k2_jax_step()`
- All callers and tests referencing `K2_JAX_INPUT_SIZE`

---

## 2. Input Field Inventory (41 fields)

| Index | Field Name | Symbol |
|-------|-----------|--------|
| 0 | `pitch_x_rad` | `_I_PITCH_X` |
| 1 | `pitch_rate_x_rad_s` | `_I_PITCH_RATE` |
| 2 | `roll_y_rad` | `_I_ROLL_Y` |
| 3 | `roll_rate_y_rad_s` | `_I_ROLL_RATE` |
| 4 | `yaw_error_rad` | `_I_YAW_ERR` |
| 5 | `yaw_rate_rad_s` | `_I_YAW_RATE` |
| 6 | `com_z_m` | `_I_COM_Z` |
| 7 | `com_vy_m_s` | `_I_COM_VY` |
| 8 | `sagittal_velocity_m_s` | `_I_SAG_VEL` |
| 9 | `sagittal_position_error_m` | `_I_SAG_POS_ERR` |
| 10 | `wheel_vel_left_rad_s` | `_I_WHEEL_VEL_L` |
| 11 | `wheel_vel_right_rad_s` | `_I_WHEEL_VEL_R` |
| 12 | `support_velocity_m_s` | `_I_SUPPORT_VEL` |
| 13 | `commanded_height_ref_m` | `_I_HEIGHT_REF` |
| 14 | `hip_yaw_div_error` | `_I_HY_DIV_ERR` |
| 15 | `hip_yaw_div_rate` | `_I_HY_DIV_RATE` |
| 16–23 | `q_*` (8 joint positions) | `_I_Q_START` |
| 24–31 | `qd_*` (8 joint velocities) | `_I_QD_START` |
| 32–39 | `q_ref_*` (8 reference positions) | `_I_QREF_START` |
| 40 | `support_position_error_m` | `_I_SUPPORT_POS_ERR` |

**Count: 41 fields → `K2_JAX_INPUT_SIZE = 41`**

---

## 3. Bug Found: `_I_TARGET_COM_HEIGHT = 41` (OOB Index)

### Location

`k2_jax_controller.py:931` (before fix)

```python
_I_TARGET_COM_HEIGHT = 41  # alias for height ref used by support FF
```

### Nature of Bug

- `K2_JAX_INPUT_SIZE = 41` → valid indices are `[0, 40]`
- `_I_TARGET_COM_HEIGHT = 41` → **out of bounds**
- Written at `pack_input_k2` line 975: `inp = inp.at[_I_TARGET_COM_HEIGHT].set(commanded_height_ref_m)`
- The value `commanded_height_ref_m` is already correctly written to `_I_HEIGHT_REF = 13` (valid)
- `_I_TARGET_COM_HEIGHT` is **never read** anywhere — write-only dead code

### JAX OOB Behavior (verified empirically)

Tested with both `x64` enabled and disabled:

```python
jnp.zeros(41).at[41].set(value)  # silently drops the OOB write
```

- **Shape unchanged**: stays `(41,)`
- **No extend**: index 41 is NOT created
- **No clamp**: index 40 is NOT overwritten
- **No error**: JAX silently discards OOB scatter indices

**Impact classification: HARMLESS BUT STILL A BUG**

The dead OOB write was silently dropped by JAX. It never corrupted `_I_SUPPORT_POS_ERR = 40` or any other field. This was NOT the root cause of the teacher-forcing hip-yaw divergence.

### Fix Applied

1. Removed `_I_TARGET_COM_HEIGHT = 41  # alias for height ref used by support FF`
2. Removed `inp = inp.at[_I_TARGET_COM_HEIGHT].set(commanded_height_ref_m)` from `pack_input_k2`
3. Fixed misleading comment: `# 42` → `# 41` on `K2_JAX_INPUT_SIZE`

**Outcome**: Input layout is now clean — `size == len(fields)`, all indices in-bounds, no dead writes.

---

## 4. Index Constant Audit (Post-Fix)

All index constants verified in-bounds `[0, 40]`:

| Constant | Value | Range | In-Bounds |
|----------|-------|-------|-----------|
| `_I_PITCH_X` | 0 | single | ✅ |
| `_I_PITCH_RATE` | 1 | single | ✅ |
| `_I_ROLL_Y` | 2 | single | ✅ |
| `_I_ROLL_RATE` | 3 | single | ✅ |
| `_I_YAW_ERR` | 4 | single | ✅ |
| `_I_YAW_RATE` | 5 | single | ✅ |
| `_I_COM_Z` | 6 | single | ✅ |
| `_I_COM_VY` | 7 | single | ✅ |
| `_I_SAG_VEL` | 8 | single | ✅ |
| `_I_SAG_POS_ERR` | 9 | single | ✅ |
| `_I_WHEEL_VEL_L` | 10 | single | ✅ |
| `_I_WHEEL_VEL_R` | 11 | single | ✅ |
| `_I_SUPPORT_VEL` | 12 | single | ✅ |
| `_I_HEIGHT_REF` | 13 | single | ✅ |
| `_I_HY_DIV_ERR` | 14 | single | ✅ |
| `_I_HY_DIV_RATE` | 15 | single | ✅ |
| `_I_Q_START` | 16 | [16, 23] | ✅ |
| `_I_QD_START` | 24 | [24, 31] | ✅ |
| `_I_QREF_START` | 32 | [32, 39] | ✅ |
| `_I_SUPPORT_POS_ERR` | 40 | single | ✅ |

---

## 5. Joint Position Ordering in `pack_input_k2`

```python
inp[_I_Q_START + 0] = joint_pos[1]  # l_hip_yaw
inp[_I_Q_START + 1] = joint_pos[6]  # r_hip_yaw
inp[_I_Q_START + 2] = joint_pos[2]  # l_hip_pitch
inp[_I_Q_START + 3] = joint_pos[7]  # r_hip_pitch
inp[_I_Q_START + 4] = joint_pos[3]  # l_knee
inp[_I_Q_START + 5] = joint_pos[8]  # r_knee
inp[_I_Q_START + 6] = joint_pos[0]  # l_hip_roll
inp[_I_Q_START + 7] = joint_pos[5]  # r_hip_roll
```

Same ordering applied to `qd` (velocities) and `q_ref` (references).

**Verified**: All 8 joint positions, velocities, and references are packed and unpacked with the same ordering.

---

## 6. Caller Audit

### `simulate_hierarchical_controller.py:6494`

```python
_jax_input = pack_input_k2(
    pitch_x_rad, pitch_rate_x_rad_s, roll_y_rad, roll_rate_y_rad_s,
    yaw_error_rad, yaw_rate_rad_s, com_z_m, com_vy_m_s,
    sagittal_velocity_m_s, sagittal_position_error_m,
    wheel_vel_left_rad_s, wheel_vel_right_rad_s,
    support_velocity_m_s, commanded_height_ref_m,
    hip_yaw_div_error, hip_yaw_div_rate,
    joint_pos=actuator_positions,
    joint_vel=actuator_velocities,
    q_ref=q_ref_current,
    support_position_error_m=support_position_error_m,
)
```

**17 positional arguments + 3 keyword arguments = matches `pack_input_k2` signature exactly.**

### Tests (`test_k2_jax_step_parity.py`)

Uses `K2_JAX_INPUT_SIZE` for array allocation — shape consistent with post-fix value of 41.

---

## 7. Resolution

**Classification**: `K2_JAX_INPUT_LAYOUT_BUG_CONFIRMED_AND_FIXED`

**Fix**: Removed dead `_I_TARGET_COM_HEIGHT = 41` constant and its OOB write. No other input layout issues found.

**Verdict**: The `_I_TARGET_COM_HEIGHT = 41` bug was **harmless** (JAX silently drops OOB `.at[]` writes) and was **NOT the root cause** of the teacher-forcing hip-yaw divergence. The bug was correctly classified as "obsolete index constant" — the field was removed from `K2_JAX_INPUT_FIELDS` but the index constant and write were not cleaned up.

**Unit tests**: 131/131 PASS (no regressions).

**Next step**: Phase 2 — investigate the actual root cause of step-1 hip-yaw teacher-forcing divergence on `[1,6]`.
