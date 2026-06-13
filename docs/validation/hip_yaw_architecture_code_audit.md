# Hip-Yaw Architecture Code Audit

**Date:** 2026-06-05

**Status:** ARCHITECTURE AUDIT - Phase 3

## Executive Summary

Code audit of hip-yaw torque composition architecture to understand why antisymmetric yaw control (6.09 Nm RMS, 100% sign correctness) fails to stabilize body yaw despite correct execution.

**Key finding from Phase 2:** Hip-yaw common-mode has weak correlation with body yaw (r=0.304) and common torque barely affects yaw rate (r=0.307), suggesting kinematic decoupling.

## Audit Questions

1. Where is hip-yaw shape posture torque computed?
2. Is it per-joint PD only?
3. Does it understand common/divergence modes?
4. Where is yaw controller torque added?
5. Is yaw torque added before or after shape posture final clipping?
6. Is yaw torque clipped separately or together?
7. Does the torque composer sum torques or choose ownership?
8. Can a symmetric posture term erase or dominate an antisymmetric yaw term?
9. Are left/right hip-yaw axes mirrored or same-signed?
10. Is body yaw being controlled through the correct mode?

## Code Analysis

### 1. Shape Posture Controller Hip-Yaw Torque

**File:** [wheeled_biped/controllers/shape_posture_controller.py:246-254](wheeled_biped/controllers/shape_posture_controller.py#L246-L254)

```python
# Hip-yaw: PD + optional HY-FF compensation + optional HY2-DIV divergence damping
# SIGN FIX: Hip-yaw joint axes are inverted in MJCF model
# Negate entire PD output to account for inverted axis convention
for idx in [1, 6]:
    tau_pd = -(self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx])
    tau_comp = tau_comp_left_final if idx == 1 else tau_comp_right_final
    tau_div = tau_div_left_raw if idx == 1 else tau_div_right_raw
    tau_total = authority_scale * tau_pd + tau_comp + tau_div
    tau = tau.at[idx].set(tau_total)
```

**Analysis:**
- **Per-joint PD:** Yes, each joint has independent PD control
- **Mode awareness:** NO - does not decompose into common/divergence modes
- **Sign fix:** Applied globally (entire PD output negated)
- **Authority scale:** Multiplied by `authority_scale * contact_degraded_scale`

**Problem:** Per-joint PD naturally generates both symmetric (divergence-mode) and antisymmetric (common-mode) components depending on left/right error asymmetry. This is uncontrolled mode mixing.

### 2. Yaw Controller Torque

**File:** [wheeled_biped/controllers/yaw_controller.py:49-59](wheeled_biped/controllers/yaw_controller.py#L49-L59)

```python
# Antisymmetric PD control (damping term has negative sign)
tau_antisym_raw = self.kp_yaw * yaw_error - self.kd_yaw * yaw_rate

# Clip to actuator limits
tau_antisym = jnp.clip(tau_antisym_raw, -self.max_yaw_torque, self.max_yaw_torque)

# Apply antisymmetrically to hip-yaw joints
# Positive yaw moment: left negative, right positive
tau = zeros_action()
tau = tau.at[1].set(-tau_antisym)  # left hip-yaw
tau = tau.at[6].set(tau_antisym)   # right hip-yaw
```

**Analysis:**
- **Pure antisymmetric:** Yes, generates only common-mode torque
- **Sign convention:** Left negative, right positive for positive yaw error
- **Clipping:** Separate clipping before composition

### 3. Torque Composition

**File:** [scripts/simulate_hierarchical_controller.py:3407-3418](scripts/simulate_hierarchical_controller.py#L3407-L3418)

```python
# Compute yaw stabilization torque (antisymmetric hip-yaw)
# Compute yaw directly from quaternion since centroidal_state yaw is NaN during control phase
quat = np.array(mj_data.qpos[3:7])
_, _, current_yaw = compute_orientation_from_quaternion(quat)
yaw_rate = float(mj_data.qvel[5])  # Body-frame yaw rate (z-axis angular velocity)
yaw_error = 0.0 - current_yaw  # Reference yaw is zero
tau_yaw, yaw_diag = balance_core_controllers["yaw_controller"].compute(
    yaw_error=yaw_error,
    yaw_rate=yaw_rate,
)

# Compose yaw torque with shape posture at hip-yaw joints [1, 6]
# Shape posture provides symmetric PD control, yaw provides antisymmetric stabilization
tau_shape_posture_with_yaw = tau_shape_posture.at[1].add(tau_yaw[1])
tau_shape_posture_with_yaw = tau_shape_posture_with_yaw.at[6].add(tau_yaw[6])
```

**Analysis:**
- **Composition method:** Simple addition (`tau_shape + tau_yaw`)
- **Order:** Yaw torque added AFTER shape posture computation (including sign fix and clipping)
- **No mode decomposition:** Direct per-joint addition
- **Clipping stage:** Yaw controller clips before addition, composer clips again later

**Problem:** Additive composition assumes orthogonality between shape posture and yaw control, but shape posture generates both modes.

### 4. Balance-Core Composer

**File:** [wheeled_biped/controllers/balance_core_torque_composer.py](wheeled_biped/controllers/balance_core_torque_composer.py)

The composer receives `tau_shape_posture_with_yaw` as a single input, treating it as one torque source. It does not distinguish between shape posture and yaw components.

```python
def compose(
    self,
    tau_shape_posture: Array,  # Actually contains shape + yaw
    tau_support_feedforward: Array,
    tau_sagittal_wheel_balance: Array,
    tau_lateral_roll_balance: Array,
    tau_prev: Array,
) -> BalanceCoreTorqueResult:
```

**Analysis:**
- **Torque sources:** 4 inputs (shape posture, support feedforward, sagittal, lateral)
- **Composition:** Summation with rate limiting and final clipping
- **Mode awareness:** NO - treats all inputs as joint-space torques
- **Ownership:** No ownership model - all sources contribute additively

### 5. Joint Axis Convention

**MJCF Model:** Hip-yaw joint axes are inverted in the robot model.

**Sign convention:**
- Positive torque → joint position decreases
- Negative torque → joint position increases

**Shape posture handling:** Entire PD output negated to account for inversion
**Yaw controller handling:** No explicit axis inversion (assumes standard convention)

**Potential issue:** If yaw controller assumes standard axis convention but applies to inverted axes, sign may be wrong. However, telemetry shows 100% sign correctness for common-mode control, so this is NOT the issue.

### 6. Left/Right Hip-Yaw Axes

**MJCF joint definitions:** Both hip-yaw axes point in same direction (both inverted).

**Implications:**
- Antisymmetric torque `(tau_L = -tau_R)` generates yaw moment correctly
- Symmetric torque `(tau_L = tau_R)` generates divergence/twist correctly
- No mirroring required

## Architecture Issues Identified

### Issue 1: Per-Joint PD Generates Uncontrolled Mode Mixing

Shape posture applies independent PD to left and right hip-yaw:
- `tau_L = -(kp * error_L + kd * vel_L)`
- `tau_R = -(kp * error_R + kd * vel_R)`

This naturally generates:
- Common component: `0.5 * (tau_L + tau_R) = -0.5 * kp * (error_L + error_R) - 0.5 * kd * (vel_L + vel_R)`
- Divergence component: `0.5 * (tau_L - tau_R) = -0.5 * kp * (error_L - error_R) - 0.5 * kd * (vel_L - vel_R)`

Both modes are active simultaneously based on error asymmetry, not control objectives.

### Issue 2: Additive Composition Assumes Orthogonality

Current composition:
```python
tau_final[1] = tau_shape[1] + tau_yaw[1]
tau_final[6] = tau_shape[6] + tau_yaw[6]
```

Assumes shape and yaw torques are independent, but:
- Shape generates both common and divergence modes
- Yaw generates only common mode
- Common-mode components add/subtract unpredictably based on shape posture errors

### Issue 3: No Kinematic Model of Hip-Yaw to Body-Yaw Coupling

**Observation from Phase 2:** Hip-yaw common-mode correlates weakly with body yaw (r=0.304).

**Implication:** Hip-yaw joint angles may not directly control body yaw rotation on a wheeled biped. Body yaw may be driven primarily by:
- Wheel-ground contact forces (friction, slip)
- Support polygon geometry
- Centrifugal forces during motion
- Hip-yaw indirectly through leg geometry affecting support polygon

**Missing:** Kinematic analysis of how hip-yaw torque couples to body yaw through contact forces.

### Issue 4: Double Clipping May Destroy Mode Structure

1. Yaw controller clips antisymmetric torque: `tau_antisym = clip(tau_antisym_raw, -max, max)`
2. Shape posture clips per-joint: `tau_shape[i] = clip(...)`
3. Composer clips final sum: `tau_final = clip(tau_shape + tau_yaw, -limit, limit)`

After clipping, antisymmetric structure may be destroyed:
- If `tau_shape[1] + tau_yaw[1]` saturates but `tau_shape[6] + tau_yaw[6]` doesn't
- Result is no longer purely antisymmetric
- Common-mode signal corrupted

## Classification

Based on code audit, the failure mechanism is:

**Primary:** `hip_yaw_kinematically_decoupled_from_body_yaw`
- Hip-yaw common-mode (joint angles) has weak correlation with body yaw (r=0.304)
- Common torque barely affects yaw rate (r=0.307)
- Hip-yaw control is ineffective for body yaw stabilization on wheeled biped

**Secondary:** `per_joint_pd_generates_uncontrolled_mode_mixing`
- Shape posture PD generates both common and divergence modes simultaneously
- No explicit mode decomposition or control objectives
- Additive composition assumes orthogonality that doesn't exist

**Tertiary:** `double_clipping_destroys_mode_structure`
- Multiple clipping stages can corrupt antisymmetric torque structure
- Final torque may not maintain desired mode characteristics

## Recommendations for Phase 4 (Isolation Experiments)

### Critical Experiment: Kinematic Decoupling Test

**Objective:** Verify that hip-yaw torque is kinematically decoupled from body yaw.

**Method:**
1. Disable all other controllers (sagittal, lateral, support feedforward)
2. Apply pure antisymmetric hip-yaw torque pulse: `tau_L = -5 Nm, tau_R = +5 Nm` for 20 steps
3. Measure body yaw response

**Expected if decoupled:** Body yaw changes <5° despite large hip-yaw torque

### Secondary Experiment: Wheel Yaw Control Test

**Objective:** Test if differential wheel velocity can control body yaw.

**Method:**
1. Apply differential wheel velocity command: `v_wheel_L = -1 rad/s, v_wheel_R = +1 rad/s`
2. Measure body yaw response

**Expected if coupled:** Body yaw rotates proportionally to wheel velocity difference

### Mode Decomposition Experiment

**Objective:** Verify mode decomposition theory.

**Method:**
1. Implement mode-based hip-yaw control:
   - Compute common/divergence errors
   - Apply separate PD controllers
   - Recompose to joint torques
2. Compare to per-joint PD baseline

**Expected:** Mode-based control should show cleaner mode separation in telemetry

## Next Steps

1. Run kinematic decoupling test (Phase 4)
2. If hip-yaw is decoupled from body yaw: design wheel-based yaw control
3. If hip-yaw is coupled: design mode-based hip-yaw architecture
4. Implement and validate minimal candidate

## Related Files

- Shape posture controller: [wheeled_biped/controllers/shape_posture_controller.py](wheeled_biped/controllers/shape_posture_controller.py)
- Yaw controller: [wheeled_biped/controllers/yaw_controller.py](wheeled_biped/controllers/yaw_controller.py)
- Balance-core composer: [wheeled_biped/controllers/balance_core_torque_composer.py](wheeled_biped/controllers/balance_core_torque_composer.py)
- Integration point: [scripts/simulate_hierarchical_controller.py:3407-3418](scripts/simulate_hierarchical_controller.py#L3407-L3418)
- Phase 2 audit: [outputs/hip_yaw_yaw_architecture_audit/decomposition_moderate_gains_v3/](outputs/hip_yaw_yaw_architecture_audit/decomposition_moderate_gains_v3/)
