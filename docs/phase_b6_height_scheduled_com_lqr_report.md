# Phase B.6: Height-Scheduled CoM Feedback LQR Report

**Date:** 2026-05-06  
**Status:** Implementation Complete, Tests Passing  
**Variant:** `height_scheduled_com_lqr_ik`

---

## Executive Summary

Phase B.6 implements a literature-informed stronger classical baseline by adding **Center-of-Mass (CoM) feedback** to the height-scheduled LQR/IK prior. This variant extends the 4D state vector (pitch, pitch_rate, wheel_pos, wheel_vel) to a **6D state vector** that includes CoM error terms:

```
x = [pitch, pitch_rate, com_y_error, com_y_error_rate, wheel_pos, wheel_vel]
```

where `com_y_error = whole_body_CoM_y - wheel_contact_y`.

The controller uses **height-scheduled gains K(h)** optimized for heights [0.70, 0.65, 0.60, 0.55, 0.50] m, with linear interpolation between grid points. This addresses the geometric CoM misalignment that increases at lower heights, providing stronger corrective feedback where it's most needed.

**Key limitation:** This variant uses **simulator CoM** (not hardware-ready). Hardware deployment requires a CoM estimator such as a momentum observer.

---

## Motivation

The geometric LQR/IK variant (Phase B.5) uses height-dependent inverse kinematics but does not explicitly account for CoM position errors. At lower heights, the robot's whole-body CoM shifts forward relative to the wheel contact point, creating a larger moment arm that destabilizes pitch control.

**CoM error at different heights** (approximate):
- h = 0.70m: ~4.6 cm forward
- h = 0.65m: ~8 cm forward
- h = 0.60m: ~12 cm forward
- h = 0.55m: ~18 cm forward
- h = 0.50m: ~25 cm forward

Adding CoM feedback allows the controller to:
1. Directly compensate for CoM misalignment
2. Improve balance stability at low heights
3. Provide a stronger classical baseline for residual RL comparison

---

## State Vector and Dynamics

### 6D State Vector

```
x = [pitch, pitch_rate, com_y_error, com_y_error_rate, wheel_pos, wheel_vel]
```

**State definitions:**
- `pitch`: Torso pitch angle (rad)
- `pitch_rate`: Torso pitch angular velocity (rad/s)
- `com_y_error`: Lateral CoM error = whole_body_CoM_y - wheel_contact_y (m)
- `com_y_error_rate`: Time derivative of CoM error (m/s)
- `wheel_pos`: Average wheel position (rad)
- `wheel_vel`: Average wheel velocity (rad/s)

### TWIP Model with CoM Augmentation

The dynamics are based on a Two-Wheeled Inverted Pendulum (TWIP) model augmented with CoM error states:

```
A = [
    [0,   1,   0,   0,   0,   0],  # pitch_dot = pitch_rate
    [g/h, 0,   0,   0,   0,   0],  # pitch_rate_dot ≈ (g/h) * pitch
    [0,   0,   0,   1,   0,   0],  # com_y_error_dot = com_y_error_rate
    [0,   0.5, 0,   0,   0,   0],  # com_y_error_rate_dot ≈ 0.5 * pitch_rate (kinematic coupling)
    [0,   0,   0,   0,   0,   1],  # wheel_pos_dot = wheel_vel
    [0,   0,   0,   0,   0,   0],  # wheel_vel_dot = u / r
]

B = [0, -1/h, 0, 0, 0, 1/r]^T
```

where:
- `g = 9.81 m/s²` (gravity)
- `h`: height-dependent CoM height above wheel axis
- `r = 0.0762 m` (wheel radius)

---

## Height-Scheduled Gains

### Gain Grid

Gains are defined at 5 height points:

```yaml
height_grid: [0.70, 0.65, 0.60, 0.55, 0.50]
```

### Gain Structure

For each height, the LQR gain vector is:

```
K(h) = [k_pitch, k_pitch_rate, k_com, k_com_rate, k_wheel_pos, k_wheel_vel]
```

### Initial Gain Values

| Height | k_pitch | k_pitch_rate | k_com | k_com_rate | k_wheel_pos | k_wheel_vel |
|--------|---------|--------------|-------|------------|-------------|-------------|
| 0.70m  | 180.0   | 25.0         | 120.0 | 15.0       | -8.0        | -2.5        |
| 0.65m  | 200.0   | 28.0         | 150.0 | 18.0       | -10.0       | -3.0        |
| 0.60m  | 220.0   | 32.0         | 180.0 | 22.0       | -12.0       | -3.5        |
| 0.55m  | 250.0   | 36.0         | 220.0 | 26.0       | -15.0       | -4.0        |
| 0.50m  | 280.0   | 40.0         | 260.0 | 30.0       | -18.0       | -4.5        |

**Gain trends:**
- `k_com` increases significantly at lower heights (120 → 260) to compensate for larger CoM misalignment
- `k_pitch` and `k_pitch_rate` increase moderately for stronger pitch stabilization
- `k_wheel_pos` and `k_wheel_vel` increase in magnitude for more aggressive wheel control

### Gain Interpolation

Between grid points, gains are **linearly interpolated**:

```python
K_interpolated(h) = np.interp(h, height_grid, K_values)
```

Outside the grid range [0.50, 0.70], gains are **clamped** to the nearest boundary value.

---

## Implementation Details

### Config File

**Location:** `configs/controllers/height_scheduled_com_lqr.yaml`

Key sections:
- `variant`: "height_scheduled_com_lqr_ik"
- `com_feedback`: enabled, uses simulator CoM
- `height_grid`: [0.70, 0.65, 0.60, 0.55, 0.50]
- `lqr_gains`: per-height gain dictionaries
- `interpolation`: linear with clamping
- `height_ik`: same as geometric variant

### Controller Extension

**File:** `wheeled_biped/controllers/lqr_ik_prior.py`

**Changes:**
1. Added `height_scheduled_gains_enabled` and `height_scheduled_gains` fields to `LQRIKConfig`
2. Extended `from_yaml()` to auto-detect height-scheduled config format
3. Added `_build_gain_interpolators()` method to construct interpolation functions
4. Modified `compute_action()` to branch between 4D (geometric) and 6D (height-scheduled) LQR control

**6D LQR control flow:**
```python
# Compute CoM error
com_y = self._compute_com_y(qpos)
wheel_contact_y = self._compute_wheel_contact_y(qpos)
com_error_y = com_y - wheel_contact_y

# Construct 6D state
x_lqr = [pitch_error, pitch_rate, com_error_y, com_vel_y, wheel_pos, wheel_vel]

# Interpolate gains at current height
K = [k_pitch(h), k_pitch_rate(h), k_com(h), k_com_rate(h), k_wheel_pos(h), k_wheel_vel(h)]

# Compute wheel velocity command
wheel_vel_cmd = -(K @ x_lqr)
```

### Tuning Script

**File:** `scripts/tune_height_scheduled_com_lqr.py`

Optimizes LQR gains via:
1. TWIP dynamics with CoM augmentation at each height
2. Continuous-time Algebraic Riccati Equation (CARE) solver
3. Configurable Q and R matrices

**Usage:**
```bash
python scripts/tune_height_scheduled_com_lqr.py \
    --heights 0.70 0.65 0.60 0.55 0.50 \
    --q-diag 100.0 10.0 50.0 5.0 1.0 1.0 \
    --r-val 1.0 \
    --output configs/controllers/height_scheduled_com_lqr_tuned.yaml
```

### Evaluation Script

**File:** `scripts/eval_classical_plus.py`

Compares classical prior variants:
- `geometric_lqr_ik` (Phase B.5)
- `height_scheduled_com_lqr_ik` (Phase B.6)

**Scenarios:**
- Nominal standing
- Fixed-height sweep (0.70 → 0.40 m)
- Push recovery

**Usage:**
```bash
python scripts/eval_classical_plus.py \
    --variants geometric_lqr_ik height_scheduled_com_lqr_ik \
    --scenarios fixed_height_sweep \
    --episodes 20 \
    --output-dir outputs/classical_plus_eval
```

---

## Test Suite

**File:** `tests/test_height_scheduled_com_lqr.py`

**Test coverage:**
1. ✅ Config loading and validation
2. ✅ Gain interpolator construction (6 interpolators)
3. ✅ Interpolation accuracy at grid points
4. ✅ Interpolation between grid points
5. ✅ Clamping outside grid range
6. ✅ `compute_action()` output shape and bounds
7. ✅ Height-scheduled gains usage verification
8. ✅ CoM feedback enabled checks
9. ✅ Comparison with geometric variant
10. ✅ NaN/inf safety
11. ✅ Height IK consistency

**Test results:** All 11 tests passing.

---

## Comparison with Geometric Variant

| Feature | Geometric LQR/IK | Height-Scheduled CoM LQR |
|---------|------------------|--------------------------|
| State dimension | 4D | 6D |
| CoM feedback | No | Yes (simulator only) |
| Gain scheduling | No | Yes (5-point grid) |
| Height IK | Yes | Yes (same) |
| Hardware-ready | Yes | No (requires CoM estimator) |
| Expected performance | Good at high heights | Better at low heights |

---

## Limitations and Future Work

### Limitations

1. **Simulator CoM only:** Uses `mj_data.subtree_com` which is not available on hardware
2. **No hardware CoM estimator:** Requires momentum observer or similar for deployment
3. **Initial gains not optimized:** Current gains are hand-tuned, not LQR-optimal
4. **No evaluation results yet:** Performance comparison pending

### Future Work

1. **Gain tuning:** Run `tune_height_scheduled_com_lqr.py` with systematic Q/R sweep
2. **Evaluation:** Compare with geometric variant on fixed-height sweep and push recovery
3. **Hardware CoM estimator:** Implement momentum-based CoM observer for hardware deployment
4. **Residual RL integration:** Use as nominal prior for Phase C residual PPO training
5. **Ablation study:** Quantify CoM feedback contribution vs. gain scheduling alone

---

## Files Created/Modified

### Created
- `configs/controllers/height_scheduled_com_lqr.yaml` (242 lines)
- `scripts/tune_height_scheduled_com_lqr.py` (246 lines)
- `scripts/eval_classical_plus.py` (274 lines)
- `tests/test_height_scheduled_com_lqr.py` (184 lines)
- `docs/phase_b6_height_scheduled_com_lqr_report.md` (this file)

### Modified
- `wheeled_biped/controllers/lqr_ik_prior.py`:
  - Added `height_scheduled_gains_enabled` and `height_scheduled_gains` config fields
  - Extended `from_yaml()` for backward-compatible config loading
  - Added `_build_gain_interpolators()` method
  - Modified `compute_action()` to support 6D LQR control

---

## Conclusion

Phase B.6 successfully implements a height-scheduled CoM feedback LQR variant as a stronger classical baseline. The implementation:
- ✅ Extends state vector to include CoM error terms
- ✅ Uses height-scheduled gains with linear interpolation
- ✅ Maintains backward compatibility with geometric variant
- ✅ Passes comprehensive test suite (11/11 tests)
- ✅ Provides tuning and evaluation infrastructure

**Next steps:**
1. Optional: Run gain tuning to optimize Q/R matrices
2. Optional: Run evaluation to compare with geometric variant
3. Proceed to Phase C: Implement `ResidualBalanceEnv` for bounded residual PPO training

**Hardware deployment note:** This variant is **not hardware-ready** due to simulator CoM dependency. Hardware deployment requires implementing a CoM estimator (e.g., momentum observer) before use.
