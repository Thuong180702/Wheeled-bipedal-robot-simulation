# K2 JAX Dedicated Default V1 — Drift Controller Design Spec

**Date**: 2026-06-30
**Phase**: `K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_CONTROL`
**Base**: `K2_JAX_DEDICATED_DEFAULT_V1`
**Candidate**: `K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED_CANDIDATE`

## Problem Statement

The K2 JAX Dedicated Default V1 controller exhibits persistent drift:
- Fixed height: robot drifts too fast, drift does not decay, heading not preserved
- Dynamic height: same issues, worse (~4 m travel in ~60 s)
- Push recovery: drift becomes faster, takes too long to decay

## Design Goals

1. Robot stays near initial world position
2. Small bounded oscillatory drift acceptable; persistent velocity drift not acceptable
3. Heading/yaw returns toward initial heading
4. Push recovery reduces drift velocity quickly after disturbance
5. Fixed height, dynamic height, and push cases must all improve
6. No degradation to posture stability, hip-yaw safety, Step D, dynamic height safety
7. Performance >= 50 Hz maintained

## Hard Rules

- No scenario-specific hacks
- No discrete if-height buckets
- No hardcoded "if drift > X then gain = Y" patches
- All gains/weights continuous functions of physical state
- Use smoothstep/sigmoid schedules where needed
- Controllers share state and are aware of each other
- No isolated blind drift controller that fights sagittal balance, yaw, or posture
- Preserve posture stability, hip-yaw safety, Step D, dynamic height safety, realtime >=50 Hz

---

## Architecture

### Integration Point

Insert drift controller after the pitch damping boost (step 13) and before the torque composer (step 14) in `k2_jax_controller_step()`. Pattern follows the existing pitch damping boost at [k2_jax_controller.py:2254](wheeled_biped/controllers/k2_jax_controller.py#L2254).

### Data Flow

```
MuJoCo sim → centroidal estimator → world_x, world_y, yaw, vx, vy, yaw_rate
    ↓
realtime runner packs into estimator input fields (est_* prefix)
    ↓
k2_jax_controller_step():
    step 0: latch initial pose → drift_ref_world_x/y/yaw
    steps 1-13: existing balance pipeline
    step 13.5: DRIFT CONTROLLER
        - compute world/body drift from estimator input
        - compute continuous authority gates
        - compute drift correction torques (smooth tanh bounded)
    step 14: torque composer sums drift torques
```

### Estimator Interface

All world-pose inputs use `est_` prefix — hardware-compatible API. In simulation,
the runner provides these from MuJoCo. On hardware, the same fields come from
IMU + wheel odometry + body kinematics/contact estimator.

### Torque Application

Drift correction applied through wheel torques [4, 9] only — not through leg joints.
This avoids disturbing the FF-PD co-contraction that proved important for stability.

---

## Schema Changes

### State Fields (+4, from 836→840)

```python
"drift_ref_world_x",     # initial world x latched at step 0
"drift_ref_world_y",     # initial world y latched at step 0
"drift_ref_yaw",         # initial yaw latched at step 0
"drift_ref_latched",     # 0.0 → 1.0 after first latch
```

### Input Fields (+6, from 45→51 standalone format)

```python
"est_world_x_m",         # estimated world x position
"est_world_y_m",         # estimated world y position
"est_yaw_rad",           # estimated world yaw
"est_world_vx_m_s",      # estimated world x velocity
"est_world_vy_m_s",      # estimated world y velocity
"est_yaw_rate_rad_s",    # estimated world yaw rate
```

### Params Fields (+7, from 54→61)

```python
"drift_k_vel",           # 8.0 Nm/(m/s) — velocity damping gain
"drift_k_pos",           # 2.0 Nm/m — position return gain (intentionally weak)
"drift_k_heading",       # 4.0 Nm/rad — heading hold proportional gain
"drift_k_heading_rate",  # 1.0 Nm/(rad/s) — heading rate damping
"drift_push_damp_mult",  # 2.0 — max additional velocity damping during push-like states
"drift_max_tau",         # 6.0 Nm — per-wheel max drift torque (smooth tanh bound)
"drift_enabled",         # 1.0 = enabled, 0.0 = disabled (ablation flag)
```

### Diag Fields (+14, from 106→120)

```python
# Drift state
"drift_world_x_m",       "drift_world_y_m",
"drift_body_x_m",        "drift_body_y_m",
"drift_distance_m",      "drift_velocity_m_s",
"yaw_error_drift_rad",

# Authority gates
"drift_stability_gate",  "drift_heading_gate",
"drift_position_gate",

# Torque contributions (raw + bounded)
"tau_drift_raw_l_nm",    "tau_drift_raw_r_nm",
"tau_drift_bounded_l_nm","tau_drift_bounded_r_nm",
```

---

## Controller Structure

### Component Priority (highest to lowest)

```
balance/existing pipeline (untouchable)
    ↓
1. Sagittal drift velocity damping  ← always on, yields only to large pitch
    ↓
2. Push recovery drift damping       ← inferred from state continuously
    ↓
3. Heading hold (yaw correction)     ← gated by stability + hip-yaw safety
    ↓
4. Position return (weak)            ← only when stable and already drifting
    ↓
torque composer (clipping, rate limiting, final safety)
```

### Continuous Authority Gates

All gates use smoothstep — no hard thresholds. Parameter names encode physical scale.

```python
def _jax_smoothstep01(x):
    """Smoothstep from 0→1 on [0,1], clamped outside."""
    xc = jnp.clip(x, 0.0, 1.0)
    return xc * xc * (3.0 - 2.0 * xc)

# Core stability gate: robot must be upright and in contact
stability_gate = (
    _jax_smoothstep01((0.21 - pitch_abs) / (0.21 - 0.035))          # pitch 2→12 deg
    * _jax_smoothstep01((0.262 - pitch_rate_abs) / (0.262 - 0.035)) # pitch_rate 2→15 deg/s
    * _jax_smoothstep01((0.087 - roll_abs) / (0.087 - 0.017))       # roll 1→5 deg
    * contact_quality                                                 # already 0→1
)

# Height transition gate: reduce during fast CoM changes
height_gate = 1.0 - _jax_smoothstep01(
    (com_z_vel_abs - 0.005) / (0.03 - 0.005)
)

# Component 1: Velocity damping gate
vel_gate = stability_gate * height_gate

# Component 2: Push inference (continuous, no scenario flag)
push_inference = (
    _jax_smoothstep01((drift_vel_mag - 0.05) / (0.30 - 0.05))
    * _jax_smoothstep01((pitch_rate_abs - 0.087) / (0.35 - 0.087))
)
push_damping_mult = 1.0 + drift_push_damp_mult * push_inference  # 1.0→(1.0+push_damp_mult)

# Component 3: Heading gate — reduce if hip-yaw diverging
heading_gate = (
    stability_gate
    * height_gate
    * _jax_smoothstep01((yaw_error_abs - 0.03) / (0.15 - 0.03))
    * (1.0 - _jax_smoothstep01((hip_yaw_div - 0.05) / (0.15 - 0.05)))
)

# Component 4: Position gate — weak, heavily gated
position_gate = (
    stability_gate
    * height_gate
    * _jax_smoothstep01((drift_distance - 0.02) / (0.20 - 0.02))
)
position_gate *= (1.0 - 0.5 * _jax_smoothstep01(
    (drift_vel_mag - 0.02) / (0.15 - 0.02)
))
```

### Body-Frame Drift Decomposition

```python
# World-frame drift
world_drift_x = est_world_x - drift_ref_world_x
world_drift_y = est_world_y - drift_ref_world_y
yaw_error = est_yaw - drift_ref_yaw

# Rotate into body frame
cos_yaw = jnp.cos(est_yaw)
sin_yaw = jnp.sin(est_yaw)
body_drift_x =  cos_yaw * world_drift_x + sin_yaw * world_drift_y  # +forward
body_drift_y = -sin_yaw * world_drift_x + cos_yaw * world_drift_y  # +left
body_drift_vx = cos_yaw * est_world_vx + sin_yaw * est_world_vy    # sagittal velocity

drift_distance = jnp.sqrt(body_drift_x**2 + body_drift_y**2)
drift_vel_mag = jnp.sqrt(body_drift_vx**2 + body_drift_vy**2)
```

### Torque Computation

```python
# Component 1: Velocity damping (symmetric — both wheels same sign)
# Negative body_drift_vx = drifting backward → positive torque = accelerate forward
tau_drift_vel = -drift_k_vel * body_drift_vx * vel_gate * push_damping_mult

# Component 3: Heading hold (antisymmetric — opposite signs)
# Positive yaw_error = turned CCW → negative diff torque = turn CW back toward reference
heading_torque = (
    -drift_k_heading * yaw_error
    - drift_k_heading_rate * est_yaw_rate
) * heading_gate

# Component 4: Position return (symmetric, very weak)
tau_drift_pos = -drift_k_pos * body_drift_x * position_gate

# Assemble wheel torques
tau_wheel_symmetric = tau_drift_vel + tau_drift_pos
tau_wheel_antisymmetric = heading_torque

tau_drift_raw_l = tau_wheel_symmetric + tau_wheel_antisymmetric   # index 4
tau_drift_raw_r = tau_wheel_symmetric - tau_wheel_antisymmetric   # index 9

# Smooth tanh bound (NOT hard clip — final safety clip belongs to composer)
tau_drift_bounded_l = drift_max_tau * jnp.tanh(tau_drift_raw_l / drift_max_tau)
tau_drift_bounded_r = drift_max_tau * jnp.tanh(tau_drift_raw_r / drift_max_tau)
```

### Ablation Flag

```python
# When drift_enabled != 1.0, all drift torques are zero.
# This allows single-code-path comparison vs DEFAULT_V1.
do_drift = (drift_enabled > 0.5)
tau_drift_bounded_l *= jnp.where(do_drift, 1.0, 0.0)
tau_drift_bounded_r *= jnp.where(do_drift, 1.0, 0.0)
```

### Coordination with Existing Controllers

| Existing Controller | Joints Used | Drift Overlap | Coordination |
|---|---|---|---|
| Sagittal balance | wheels [4,9] | YES | Drift is downstream; vel_gate yields to pitch |
| Yaw controller | hip_yaw [1,6] | NO | heading_gate reduced by hip_yaw_div |
| Mode-div controller | hip_yaw [1,6] | NO | heading_gate reduced by hip_yaw_div |
| Posture PD | legs [0-3,5-8] | NO | No joint conflict |
| Support FF | hip_yaw, hip_pitch | NO | No joint conflict |
| Pitch damping boost | wheels [4,9] | YES | Both contribute; composer handles sum |

### Lateral Drift Policy

Lateral drift (`body_drift_y`) is diagnostic only. Since the robot is wheeled,
there is no direct lateral authority. Lateral drift is reduced indirectly through
heading correction (realigning sagittal axis) and sagittal position return.
No fake lateral wheel controller. No body_drift_y torque term.

---

## Sign Conventions

These must be verified in Phase C visual sign checks:

1. **Forward body-frame velocity** (body_drift_vx > 0) → braking wheel torque (negative)
2. **Backward body-frame velocity** (body_drift_vx < 0) → forward recovery torque (positive)
3. **Positive yaw error** (turned CCW from reference) → differential torque turning CW back
4. **Bounded torque** never exceeds `drift_max_tau` before final composer

---

## Implementation Order

1. **Schema changes** — extend state (+4), input (+6), params (+7), diag (+14) field tuples
2. **Index constants** — add fast-access index constants for all new fields
3. **Pack/unpack functions** — update `pack_state`, `unpack_state`, `pack_params_stage2`, `unpack_params_stage2`, `pack_input_k2_standalone`
4. **Drift reference latch** — latch initial pose at step 0 in `k2_jax_controller_step()`
5. **Drift torque function** — pure JAX function: `k2_jax_drift_controller_step(state, input, params) → (tau_drift_l, tau_drift_r, diag)`
6. **Pipeline integration** — insert drift step into `k2_jax_controller_step()` after pitch damping boost, before composer
7. **Runner: estimator input packing** — compute `est_*` fields from MuJoCo `mj_data` in `run_k2_jax_realtime.py`
8. **Runner: initial pose capture** — capture world pose at step 0 for reference latch
9. **Profile variant** — add `K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED` in `sagittal_velocity_damped_balance_controller.py`
10. **Profile param passthrough** — ensure drift params flow from profile → `pack_params_stage2`

---

## Validation Plan

### Phase C.1: Visual Sign Checks

Single runs with `--visual` to confirm torque sign conventions.

### Phase C.2: Targeted Visual Runs

```bash
# Fixed height
python scripts/run_k2_jax_realtime.py --height low_0p320 --visual
python scripts/run_k2_jax_realtime.py --height low_0p380 --visual
python scripts/run_k2_jax_realtime.py --height mid_0p400 --visual
python scripts/run_k2_jax_realtime.py --height high_0p450 --visual

# Dynamic height
python scripts/run_k2_jax_realtime.py --scenario dynamic_height --case ramp_up_0p330_to_0p480 --visual
python scripts/run_k2_jax_realtime.py --scenario dynamic_height --case ramp_down_0p480_to_0p330 --visual
python scripts/run_k2_jax_realtime.py --scenario dynamic_height --case up_down_cycle_0p330_0p480_0p330 --visual

# Push recovery
python scripts/run_k2_jax_realtime.py --scenario step_d --height low_0p330 --push sagittal_forward_60N --visual
python scripts/run_k2_jax_realtime.py --scenario step_d --height mid_0p400 --push sagittal_backward_60N --visual
python scripts/run_k2_jax_realtime.py --scenario step_d --height high_0p480 --push sagittal_forward_90N --visual
```

### Phase C.3: Full 39-Scenario Validation

```bash
python scripts/validate_k2_jax_dedicated_promotion.py --scope all --output-dir outputs/k2_default_v1_drift_candidate
```

### Phase C.4: Quality Analysis + Before/After Comparison

```bash
python scripts/analyze_k2_behavior_quality.py --input-dir outputs/k2_default_v1_drift_candidate --output docs/validation/k2_default_v1_drift_candidate_quality.md
python scripts/evaluate_k2_stability_improvement.py --baseline docs/validation/k2_default_v1_quality.json --candidate docs/validation/k2_default_v1_drift_candidate_quality.json --output docs/validation/k2_default_v1_drift_candidate_eval.md
```

---

## Acceptance Criteria

| Metric | Direction | Threshold |
|---|---|---|
| falls | = | 0 |
| SAFETY_FAIL | = | 0 |
| performance | ≥ | 50 Hz |
| final_displacement_m | ↓ | decrease |
| max_displacement_m | ↓ | decrease |
| drift_velocity_rms | ↓ | decrease |
| yaw_drift_deg | ↓ | decrease |
| wheel_travel_asymmetry | ≤ | no worse |
| hip_yaw_max_rad | ≤ | no worse |
| hip_yaw_rms_rad | ≤ | no worse |
| pitch_rms_deg | ≤ | no material worsening (+0.5 deg tolerance) |
| dynamic height safety | = | preserved |
| push recovery drift decay | ↓ | faster (post_push support_rms decreases) |

---

## Conservative First Pass Gains

Start conservative, sweep upward if stable:

```python
drift_k_vel = 6.0          # Nm/(m/s) — velocity damping
drift_k_pos = 1.5          # Nm/m — position return (very weak)
drift_k_heading = 3.0      # Nm/rad — heading proportional
drift_k_heading_rate = 0.8 # Nm/(rad/s) — heading rate
drift_push_damp_mult = 1.5 # additional multiplier during push-like states
drift_max_tau = 5.0        # Nm — per-wheel max drift contribution
drift_enabled = 1.0
```

Balance must remain dominant. If pitch RMS increases or hip-yaw grows, reduce `drift_k_vel` first.

---

## Profile Definition

```python
K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED = replace(
    K2_JAX_DEDICATED_DEFAULT_V1,
    profile_name="k2_jax_dedicated_default_v1_drift_fixed",
    # Drift controller: coordinated wheel-torque drift correction
    # with continuous stability/height/contact/hip-yaw gating.
    enable_drift_controller=True,
    drift_k_vel=6.0,
    drift_k_pos=1.5,
    drift_k_heading=3.0,
    drift_k_heading_rate=0.8,
    drift_push_damp_mult=1.5,
    drift_max_tau=5.0,
)
```
