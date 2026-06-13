# Tau_pitch Computation Audit

**Date:** 2026-06-08
**Profile:** D2/F1b/G1a/G1b at low_0p300
**Objective:** Find exact tau_pitch computation, pitch reference, sign convention, gain values

---

## 1. Tau_pitch Formula

### Formula (from `sagittal_velocity_damped_balance_controller.py` line 599)

```python
tau_pitch_raw = self.kp_pitch * pitch_x_rad
```

Where:
- `kp_pitch = 50.0` (default, line 364)
- `pitch_x_rad` is the **pitch error** (not raw pitch angle)

### Pitch Error Computation (from `simulate_hierarchical_controller.py` line 3616-3617)

```python
pitch_x_ref = float(pitch_x_eq)
pitch_x_error = float(centroidal_state_control.body_pitch_x) - pitch_x_ref
```

Then passed to controller as `pitch_x_rad=pitch_x_error` (line 3708).

---

## 2. Pitch Reference Source

### At equilibrium initialization (lines 2216-2220)

```python
R_eq = np.array(mj_data.xmat[base_body_id]).reshape(3, 3)
gravity_world = np.array([0.0, 0.0, -gravity])
gravity_body_eq = R_eq.T @ gravity_world
pitch_x_eq, roll_y_eq = compute_orientation_from_gravity(jnp.array(gravity_body_eq))
```

### Orientation computation (from `orientation_utils.py`)

```python
def compute_robot_frame_orientation_from_gravity(gravity_body: Array) -> tuple[float, float]:
    gx, gy, gz = gravity_body[0], gravity_body[1], gravity_body[2]
    body_pitch_x = jnp.arctan2(-gy, -gz)
    body_roll_y = jnp.arctan2(gx, -gz)
    return body_pitch_x, body_roll_y
```

---

## 3. low_0p300 Setup

From `outputs/physical_target_height_setups/low_0p300_setup.json`:

```json
{
  "pitch_x_rad": 0.0,
  "roll_y_rad": 0.0,
  "yaw_z_rad": 0.0,
  "equilibrium_pitch_x": 0.0,
  "equilibrium_roll_y": 0.0,
  "equilibrium_yaw_z": 0.0
}
```

**The setup file explicitly stores `equilibrium_pitch_x = 0.0`.**

---

## 4. Sign Convention (from controller docstring)

```
Control law:
    tau = k_pitch * pitch_x + k_pitch_rate * pitch_rate_x + ...

Signs verified by unit tests:
    - positive pitch → restoring torque (opposes tilt)
    - positive pitch_rate → damping torque (opposes angular velocity)
```

### Physical interpretation:
- `body_pitch_x > 0` means robot nose UP (falling forward)
- `tau_pitch > 0` produces **forward** wheel torque (pushes robot back)

---

## 5. D2 Telemetry Evidence

### pitch_x_ref_rad (from telemetry)
```
mean=0.000000, min=-0.000000, max=-0.000000
```
**Confirmed: pitch reference is exactly 0.0 throughout the run.**

### pitch_x (raw body pitch)
```
mean=+0.0520 rad (+2.98 deg)
min=-0.0083 rad (-0.48 deg)
max=+0.1111 rad (+6.36 deg)
positive%=89.0%
```

### pitch_x_error_rad (error = pitch - ref)
```
mean=+0.0520 rad (+2.98 deg)
positive%=89.2%
```
**pitch_x ≈ pitch_x_error since ref=0**

### tau_pitch computation
```
tau_pitch = 50.0 * pitch_x_error
tau_pitch_mean = +2.5992 Nm
tau_pitch_positive% = 89.2%
```

### tau_pitch_rate (damping term)
```
tau_pitch_rate = kd_pitch * pitch_rate_x
kd_pitch = 10.0
tau_pitch_rate_mean = +0.0946 Nm
tau_pitch_rate_positive% = 42.0%
```

---

## 6. Correlation Analysis

| Correlation | Value | Interpretation |
|-------------|-------|-----------------|
| corr(pitch_x, tau_pitch) | 1.0000 | tau_pitch directly proportional to pitch |
| corr(pitch_x, sagittal_position_error_m) | 0.9353 | pitch and position error highly correlated |
| corr(tau_pitch, sagittal_position_error_m) | 0.9345 | same |
| corr(pitch_rate_x, tau_pitch) | -0.0882 | weak negative (damping effect) |

---

## 7. Key Observations

### Observation 1: Pitch reference is correctly set to 0
- `pitch_x_ref_rad` telemetry confirms ref=0 throughout
- Setup file confirms `equilibrium_pitch_x = 0.0`
- **TAU_PITCH_REFERENCE_SUSPECT: NO** - reference is not wrong

### Observation 2: tau_pitch_raw = tau_pitch = tau_pitch_scheduled
- No clipping or scheduling applied to tau_pitch in D2
- `effective_pitch_scale = 1.0`
- `effective_pitch_tau_cap = none`
- **TAU_PITCH_SATURATION_SUSPECT: NO** - not saturated/clipped

### Observation 3: pitch_x is consistently positive
- 89% of steps have pitch_x > 0
- Mean pitch_x = +2.98 deg
- Max pitch_x = +6.36 deg
- **This is a systematic positive bias, not noise**

### Observation 4: pitch_x and sagittal_position_error are correlated
- corr = 0.9353
- When pitch is positive (forward lean), position error is also positive (forward drift)
- **Both are symptoms of forward instability tendency**

---

## 8. Classification

| Check | Result | Evidence |
|-------|--------|----------|
| TAU_PITCH_FORMULA_OK | ✅ PASS | `tau_pitch = kp_pitch * pitch_x_error` is correct |
| TAU_PITCH_SIGN_CORRECT | ✅ PASS | Sign convention matches physics (positive pitch → forward torque) |
| TAU_PITCH_REFERENCE_SUSPECT | ❌ NO | pitch_x_ref = 0.0 is correct per setup file |
| TAU_PITCH_RATE_TERM_SUSPECT | ❌ NO | tau_pitch_rate mean ≈ 0, evenly distributed |
| TAU_PITCH_SATURATION_SUSPECT | ❌ NO | tau_pitch_raw = tau_pitch, no clipping |
| TAU_PITCH_COMPUTATION_INCONCLUSIVE | ❌ NO | Computation is straightforward and correct |

---

## 9. Conclusion: Computation is Correct

**tau_pitch computation is not the source of the problem.**

The persistent positive tau_pitch is a **correct response** to:
1. Robot body persistently leaning forward (pitch_x > 0)
2. Support center persistently drifting forward (sagittal_position_error > 0)

**The real question is: why is the robot leaning forward in the first place?**

This is a physics/control problem, not a computation bug.

---

## 10. Next: Investigate Why pitch_x is Consistently Positive

Possible causes to investigate:
1. **Geometry mismatch**: low_0p300 requires different hip_pitch for equilibrium
2. **Initial condition bias**: robot starts with forward lean
3. **Control coupling**: other controllers (WBC, hip yaw) create forward moment
4. **Unmodeled dynamics**: wheel friction, contact model at low height