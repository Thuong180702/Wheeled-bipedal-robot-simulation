# Low 0p300 Initialization Path Audit

## Classification

**INIT_SETUP_REFERENCE_SPLIT_BUG**

## Root Cause Summary

The `low_0p300` height-variant setup is correctly applied to MuJoCo `qpos` at simulation initialization, but the **equilibrium joint position reference** used by the controller is captured AFTER setup application from `mj_data.qpos[7:17]`, which correctly shows `hip_pitch = 1.3761 rad`. However, the **target joint position** used for `joint_pos_error` computation comes from `posture_regularizer.compute_target_posture_from_height(height_cmd)`, which interpolates from a hardcoded `height_targets` table with `hip_pitch = 0.9261 rad` at `h=0.40`.

The error: `joint_pos_error = target_joint_pos - joint_pos = 0.9261 - 1.3761 = -0.45 rad`

## Evidence

### 1. Setup is correctly applied to qpos

From `simulate_hierarchical_controller.py` lines 1864-1879:
```python
if height_variant_setup:
    print("[HEIGHT VARIANT] Applying variant posture...")
    # Apply hip_pitch and knee references (symmetric left/right)
    mj_data.qpos[9] = height_variant_setup["hip_pitch_ref"]   # l_hip_pitch
    mj_data.qpos[10] = height_variant_setup["knee_ref"]        # l_knee
    mj_data.qpos[14] = height_variant_setup["hip_pitch_ref"]  # r_hip_pitch
    mj_data.qpos[15] = height_variant_setup["knee_ref"]        # r_knee
```

After applying setup, verification shows:
- `qpos[9]` (l_hip_pitch) = 1.3761 rad ✓
- `qpos[14]` (r_hip_pitch) = 1.3761 rad ✓
- Matches `low_0p300_setup.json` equilibrium_joint_pos ✓

### 2. Target joint position comes from wrong source

From `simulate_hierarchical_controller.py` line 3340:
```python
target_joint_pos = posture_regularizer.compute_target_posture_from_height(height_cmd)
```

This uses `posture_regularizer.height_targets` which has hardcoded values:
```python
self.height_targets = jnp.array([
    [0.40, 0.926052, 1.748364, 0.926052, 1.748364],  # hip_pitch = 0.9261 rad
    ...
])
```

At `height_cmd = 0.40` (the low_0p300 target), this produces:
- `l_hip_pitch = 0.9261 rad`
- `r_hip_pitch = 0.9261 rad`

### 3. Computed joint position error

From `simulate_hierarchical_controller.py` line 3341:
```python
joint_pos_error = target_joint_pos - joint_pos
```

Result:
- `joint_pos_error[2]` (l_hip_pitch) = 0.9261 - 1.3761 = -0.4500 rad
- `joint_pos_error[7]` (r_hip_pitch) = 0.9261 - 1.3761 = -0.4500 rad

### 4. Telemetry confirmation

From old D2 run CSV (step 0):
```
joint_pos: "0.0000,0.0000,1.3761,2.3484,0.0000,0.0000,0.0000,1.3761,2.3484,0.0000"
joint_pos_error: "0.0000,-0.0007,-0.4500,-0.6000,0.0000,0.0000,0.0009,-0.4500,-0.6000,0.0000"
hip_pitch_error_left_rad: -0.45
hip_pitch_error_right_rad: -0.45
hip_pitch_error_max: 0.4500000476837158
```

The actual joint positions are correct (1.3761 rad), but the error is computed against the wrong reference.

## Why hip_pitch_error_max = 0.45 rad in telemetry

The `hip_pitch_error_max` telemetry field (from `compute_step1_joint_diagnostics`) uses `joint_pos_error` which is computed as:
```python
joint_pos_error = target_joint_pos - joint_pos
```

Where `target_joint_pos` comes from `posture_regularizer.compute_target_posture_from_height(height_cmd)`, NOT from the height-variant setup.

## Joint Index Verification

The qpos indices used in setup application (lines 1867-1870) are correct:
- `qpos[9]` = l_hip_pitch (verified by MuJoCo XML)
- `qpos[14]` = r_hip_pitch (verified by MuJoCo XML)
- Actual qpos at step 0 = 1.3761 rad = setup.hip_pitch_ref ✓

## Impact on tau_pitch

From the tau_pitch bias audit, `tau_pitch` computation is correct:
- `tau_pitch = kp_pitch * pitch_x_error`
- `pitch_x_error = pitch_x_ref - pitch_x`

But the **large initial `joint_pos_error`** triggers the **position feedback term** (`tau_position`) which:
1. Sends large tau to hip_pitch/knee to "correct" the error
2. Causes sagittal instability and forward lean
3. The forward lean then triggers real `pitch_x_error` → `tau_pitch`

## Fix Strategy

The fix should make `target_joint_pos` use the height-variant setup equilibrium when a setup is provided, instead of interpolating from `posture_regularizer.height_targets`.

Options:
1. **Option A (Minimal)**: When `height_variant_setup` is provided and `height_cmd` matches setup's target height, use setup's `equilibrium_joint_pos` directly.
2. **Option B (Comprehensive)**: Replace `posture_regularizer.compute_target_posture_from_height` with height-variant setup values when available.

## Files Involved

- `scripts/simulate_hierarchical_controller.py`:
  - Line 3340: `target_joint_pos = posture_regularizer.compute_target_posture_from_height(height_cmd)`
  - This should use setup equilibrium when height_variant_setup is provided.

## Verification Plan

After fix:
1. Run `scripts/audit_low_0p300_step0_state.py`
2. Verify `hip_pitch_error_max < 0.05 rad` at step 0
3. Run D2 500-step validation
4. Compare with old D2 to verify tau_pitch bias reduction