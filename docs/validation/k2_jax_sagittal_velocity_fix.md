# K2 JAX Sagittal Velocity Parity Fix

**Date:** 2026-06-27
**Classification:** `K2_JAX_SAGITTAL_VELOCITY_DAMPING_FIXED`

---

## 1. Root Cause

### Mismatch: `velocity_damping_scale` hardcoded to 1.0 in JAX

JAX's sagittal torque assembly used hardcoded values:
```python
effective_k_velocity=15.0, effective_velocity_damping_scale=1.0
```

Python's K2 controller profiles inherit `velocity_damping_scale=1.10` from
`ADAPTIVE_SUPPORT_CENTERING_TRIM` (line 2266 of `sagittal_velocity_damped_balance_controller.py`).

The profile chain:
```
ADAPTIVE_SUPPORT_CENTERING_TRIM (velocity_damping_scale=1.10, applies_to_variants includes high_0p480)
  → PITCH_EQUILIBRIUM_TRIM
  → HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM
  → SUPPORT_POSITION_OUTER_LOOP_PITCH_REF
  → CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF_V2
  → PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP
  → PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2
  → K1_PITCH_RATE_NOTCH
  → K2_NOTCH_LOW_Q_V1
```

K2 inherits `velocity_damping_scale=1.10` through the entire chain. Since
`applies_to_variants` includes `high_0p480`, `schedule_active=True`, and
Python's effective velocity damping scale = 1.10 (not 1.0).

Python effective coefficient = 15.0 × 1.10 = 16.5
JAX effective coefficient = 15.0 × 1.0 = 15.0
Difference = 10%

### Evidence (step 1, fixed_high_0p480, before fix):
```
tau_sagittal_velocity: py=-1.069206e-01  jx=-9.720053e-02
Ratio = 1.09997  (exactly 1.10 within floating point)
```

## 2. Fix

### Change 1: Add velocity damping params to JAX params layout

Added `k_velocity` and `velocity_damping_scale` to `K2_JAX_PARAMS_FIELDS_STAGE2`
and corresponding index constants. Params size increased from 31 to 33.

### Change 2: Update `pack_params_stage2()` and `unpack_params_stage2()`

Added `k_velocity` and `velocity_damping_scale` parameters with defaults (15.0, 1.0).
Values are stored at indices 31 and 32 in the params array.

### Change 3: Update `k2_jax_controller_step()` to read and use params

Read `_k_velocity` and `_velocity_damping_scale` from params_flat and pass them
to `k2_jax_sagittal_torque_assembly()` instead of hardcoded values.

### Change 4: Pass profile values at JAX init

In `simulate_hierarchical_controller.py`, read actual profile values:
```python
k_velocity=float(balance_core_controllers["sagittal_wheel_balance"].k_velocity),
velocity_damping_scale=float(balance_core_controllers["sagittal_wheel_balance"].authority_schedule.velocity_damping_scale),
```

## 3. Verification

### Before fix:
```
tau_sagittal_velocity: py=-3.496125e-01  jx=-3.178295e-01  DIFF=3.18e-02
```

### After fix:
```
tau_sagittal_velocity: py=-1.069206e-01  jx=-1.069206e-01  DIFF=0.0
tau_sagittal_velocity: py=-3.874994e-01  jx=-3.874994e-01  DIFF=0.0
tau_sagittal_velocity: py=-3.763333e-01  jx=-3.763333e-01  DIFF=0.0
```
All sagittal velocity terms now match.

## 4. Files Changed

| File | Change |
|------|--------|
| `wheeled_biped/controllers/k2_jax_controller.py` | Params layout: +2 fields (k_velocity, velocity_damping_scale), 31→33 |
| `wheeled_biped/controllers/k2_jax_controller.py` | `pack_params_stage2()`: accept new params |
| `wheeled_biped/controllers/k2_jax_controller.py` | `unpack_params_stage2()`: include new fields |
| `wheeled_biped/controllers/k2_jax_controller.py` | `k2_jax_controller_step()`: read and pass new params |
| `scripts/simulate_hierarchical_controller.py` | JAX init: pass profile values from authority schedule |

## 5. Classification

**`K2_JAX_SAGITTAL_VELOCITY_DAMPING_FIXED`** — Root cause identified and fixed.
tau_sagittal_velocity now matches Python exactly.
