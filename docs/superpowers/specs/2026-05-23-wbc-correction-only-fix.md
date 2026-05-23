# WBC Correction-Only Fix Specification

**Date:** 2026-05-23  
**Status:** Approved with revisions  
**Replaces:** StaticBalanceController wrapper approach (failed validation)

## Problem Statement

The wheeled biped robot falls after 14-15 steps due to a vertical contact force deficit. Root cause analysis revealed:

1. **Current WBC behavior:** Maps entire baseline body weight through joint-only contact Jacobian (J^T f), producing large support-joint torques that fight against contact constraints
2. **Physics reality:** MuJoCo contact constraints already provide baseline body-weight support through normal forces at wheel-floor contacts
3. **Result:** WBC-commanded torques and contact-provided support interfere, causing force deficit and eventual collapse

**Failed approach:** StaticBalanceController wrapper attempted to cancel WBC static bias using inverse dynamics reference torques. Validation revealed fundamental flaw: `mj_inverse` computes torques WITHOUT accounting for contact forces, producing large negative torques (-242 Nm, -204 Nm) that made performance 14× worse.

**Correct approach:** Separate baseline support (handled by contact constraints) from correction wrench (mapped through WBC). At calibrated equilibrium with zero errors, WBC should produce near-zero torque. With perturbations, WBC produces stabilizing corrections only.

## Design Principle

**Core invariant:** Baseline body weight is supported by contact constraints in static equilibrium. WBC maps only deviation-driven correction wrench to joint torques.

```
Baseline support:     mg → contact constraints → normal forces → zero joint torque
Correction support:   error → WBC → correction wrench → J^T f_correction → joint torques
```

At calibrated equilibrium:
- pitch_x ≈ 0, roll_y ≈ 0, height_error ≈ 0, com_error ≈ 0, cp_error ≈ 0
- correction_wrench ≈ [0, 0, 0, 0, 0, 0]
- tau_wbc ≈ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- Contact forces provide full body-weight support

With perturbations:
- pitch_x = 0.05 rad → correction_Fy produces stabilizing sagittal force
- height_error = -0.02 m → correction_Fz > 0 (upward correction)
- roll_y = 0.03 rad → correction_My produces stabilizing roll moment

## Height and Mass Conventions

**Robot mass:**
```python
# Compute from MuJoCo model (same as simulate_hierarchical_controller.py)
robot_mass = float(np.sum(mj_model.body_mass))
gravity = float(abs(mj_model.opt.gravity[2]))
model_weight = robot_mass * gravity  # Baseline body weight in N
```

**Height definitions:**
- `root_z`: MuJoCo root body z-position (qpos[2])
- `com_z`: Center of mass z-position (data.subtree_com[1, 2])
- `torso_height`: Torso body z-position
- `contact_height`: Wheel contact point z-position

**Height command convention:**
- `height_cmd` refers to desired CoM z-position (com_z)
- `CentroidalWrenchComputer` uses `height_error = height_cmd - com_z`
- Calibrated root_z is NOT the same as height_cmd
- Tests must use consistent height definition

## Architecture

### Component Changes

**1. CentroidalWrenchComputer**

Add method to compute separate baseline and correction wrenches:

```python
def compute_baseline_and_correction_wrench(
    self,
    obs: Array,
    state: CentroidalState,
    height_cmd: float,
    roll_integral: float = 0.0,
) -> tuple[Array, Array]:
    """Compute baseline wrench (diagnostic) and correction wrench (control).
    
    Args:
        obs: Observation array
        state: CentroidalState with CoM, velocities, orientation
        height_cmd: Desired CoM z-position (NOT root_z)
        roll_integral: Accumulated roll error for integral control
    
    Returns:
        Tuple of (baseline_wrench, correction_wrench) where:
            - baseline_wrench: (6,) [0, 0, mg, 0, 0, 0] - diagnostic only
            - correction_wrench: (6,) [Fx, Fy, Fz, Mx, My, Mz] - control output
    """
```

**Baseline wrench (diagnostic only):**
```python
baseline_wrench = jnp.array([0.0, 0.0, self.robot_mass * self.gravity, 0.0, 0.0, 0.0])
```

**Correction wrench (control output):**
```python
# Extract state
com_pos = state.com_pos  # (3,) [x, y, z] in world frame
com_vel = state.com_vel  # (3,) [vx, vy, vz]
pitch_x = state.pitch_x  # Rotation about X-axis (sagittal plane)
roll_y = state.roll_y    # Rotation about Y-axis (frontal plane)
pitch_rate_x = state.angular_vel[0]
roll_rate_y = state.angular_vel[1]
cp = state.capture_point  # (2,) [x, y]

# Height tracking: proportional + damping (NO baseline mg)
height_error = height_cmd - com_pos[2]
correction_Fz = self.k_height * height_error - self.k_height_damping * com_vel[2]

# CoM lateral regulation
correction_Fx = -self.k_com_lateral * com_pos[0] - self.k_com_lateral_damping * com_vel[0]

# CoM sagittal regulation
correction_Fy_com = -self.k_com_sagittal * com_pos[1] - self.k_com_sagittal_damping * com_vel[1]

# Capture point corrections
correction_Fx += -self.k_cp_lateral * cp[0]
correction_Fy_com += -self.k_cp_sagittal * cp[1]

# Pitch stabilization (inverted pendulum control)
correction_Fy_pitch = -self.k_pitch * pitch_x - self.k_pitch_rate * pitch_rate_x

# Total sagittal force
correction_Fy = correction_Fy_com + correction_Fy_pitch

# Roll stabilization (PID control)
correction_My = -self.k_roll * roll_y - self.k_roll_rate * roll_rate_y - self.k_roll_integral * roll_integral
correction_My = self._limit_roll_moment(correction_My)

correction_wrench = jnp.array([
    correction_Fx,
    correction_Fy,
    correction_Fz,
    0.0,  # Mx
    correction_My,
    0.0,  # Mz
])
```

**Correction limits (configurable):**
```python
# Add to CentroidalWrenchComputer.__init__
self.max_correction_fz_fraction = 0.35  # 35% of model weight
self.max_correction_fxy_fraction = 0.20  # 20% of model weight

# In compute_baseline_and_correction_wrench
max_correction_fz = self.max_correction_fz_fraction * (self.robot_mass * self.gravity)
max_correction_fxy = self.max_correction_fxy_fraction * (self.robot_mass * self.gravity)

correction_Fz = jnp.clip(correction_Fz, -max_correction_fz, max_correction_fz)
correction_Fx = jnp.clip(correction_Fx, -max_correction_fxy, max_correction_fxy)
correction_Fy = jnp.clip(correction_Fy, -max_correction_fxy, max_correction_fxy)
```

**2. IntegratedWBC**

Modify `compute_wbc_torque_with_diagnostics` to use correction-only wrench:

```python
def compute_wbc_torque_with_diagnostics(
    self,
    mj_data: mujoco.MjData,
    obs: Array,
    state: CentroidalState,
    height_cmd: float,
    hip_roll_authority_scale: float = 1.0,
) -> tuple[Array, dict]:
    # Update roll integral (existing logic)
    # ...
    
    # Compute baseline (diagnostic) and correction (control) wrenches
    baseline_wrench, correction_wrench = self.wrench_computer.compute_baseline_and_correction_wrench(
        obs, state, height_cmd, self.roll_integral
    )
    
    # CRITICAL: Only pass correction_wrench to force distributor
    # Baseline mg is handled by contact constraints, NOT mapped through J^T f
    wheel_pos_left, wheel_pos_right = self._compute_wheel_positions_relative_to_com(
        mj_data, state.com_pos
    )
    
    f_left, f_right, tau_hip_roll, distribution_diagnostics = (
        self.force_distributor.distribute_wrench_contact_aware(
            correction_wrench,  # NOT baseline_wrench, NOT baseline + correction
            left_contact=bool(state.left_wheel_contact),
            right_contact=bool(state.right_wheel_contact),
            wheel_pos_left=wheel_pos_left,
            wheel_pos_right=wheel_pos_right,
            hip_roll_authority_scale=hip_roll_authority_scale,
        )
    )
    
    # Map correction forces to joint torques
    tau_contact = self.contact_jacobian.map_contact_forces_to_torques(
        mj_data, f_left, f_right, tau_hip_roll=None
    )
    tau_hip = self._build_direct_hip_roll_torque(tau_hip_roll)
    tau_wbc_correction = tau_contact + tau_hip
    
    # CRITICAL: Disable or adapt force feedback for correction-only mode
    # Force feedback was designed to scale torques based on (actual - desired) force error
    # In correction-only mode, desired force is correction_Fz (small), not baseline + correction
    # Scaling correction torques to compensate for baseline body weight is incorrect
    #
    # Option 1: Disable force feedback entirely (safest for initial validation)
    force_scale = 1.0
    
    # Option 2: Adapt force feedback to use correction-only reference (future work)
    # desired_fz_correction = f_left[2] + f_right[2]
    # actual_fz_correction = actual_fz_total - baseline_wrench[2]
    # force_error_ratio = (actual_fz_correction - desired_fz_correction) / desired_fz_correction
    # force_scale = 1.0 - self.force_feedback_gain * force_error_ratio
    
    # Apply authority budget
    tau_wbc_correction_scaled = tau_wbc_correction * force_scale
    tau_wbc = self.clip_to_authority_budget(tau_wbc_correction_scaled)
    
    # Diagnostics
    actual_fz_total = float(state.total_contact_force_z)
    baseline_fz = float(baseline_wrench[2])
    correction_fz = float(correction_wrench[2])
    distributor_fz_sum = float(f_left[2] + f_right[2])
    
    diagnostics = {
        # Baseline wrench (diagnostic only)
        "baseline_wrench_Fx": float(baseline_wrench[0]),
        "baseline_wrench_Fy": float(baseline_wrench[1]),
        "baseline_wrench_Fz": baseline_fz,  # Should equal model_weight
        "baseline_wrench_Mx": float(baseline_wrench[3]),
        "baseline_wrench_My": float(baseline_wrench[4]),
        "baseline_wrench_Mz": float(baseline_wrench[5]),
        
        # Correction wrench (control output)
        "correction_wrench_Fx": float(correction_wrench[0]),
        "correction_wrench_Fy": float(correction_wrench[1]),
        "correction_wrench_Fz": correction_fz,
        "correction_wrench_Mx": float(correction_wrench[3]),
        "correction_wrench_My": float(correction_wrench[4]),
        "correction_wrench_Mz": float(correction_wrench[5]),
        
        # Force breakdown
        "baseline_fz": baseline_fz,
        "correction_fz": correction_fz,
        "distributor_fz_sum": distributor_fz_sum,  # Should match correction_fz
        "actual_contact_fz": actual_fz_total,
        "force_error": actual_fz_total - baseline_fz,  # Error relative to model_weight
        
        # Torque breakdown
        "tau_wbc_correction": tau_wbc_correction,
        "tau_wbc_final": tau_wbc,
        "tau_wbc_support_joints_rms": float(jnp.sqrt(jnp.mean(tau_wbc[[2,3,7,8]]**2))),
        
        # Existing diagnostics...
    }
    
    return tau_wbc, diagnostics
```

**3. SimpleForceDistributor Audit**

**Issue:** `min_wheel_force` parameter may reintroduce baseline Fz even when correction_wrench ≈ 0.

**Audit requirement:**
```python
# In distribute_wrench_contact_aware, when correction_wrench_Fz ≈ 0:
# - f_left[2] + f_right[2] should be ≈ 0, NOT ≈ mg
# - min_wheel_force should only affect force asymmetry limits, not total Fz
# - Verify that min_wheel_force does not add a force floor to f_left[2] or f_right[2]
```

**Test case:**
```python
# At equilibrium with correction_wrench = [0, 0, 0, 0, 0, 0]
f_left, f_right, tau_hip_roll, diag = distributor.distribute_wrench_contact_aware(
    jnp.zeros(6), left_contact=True, right_contact=True, ...
)
assert abs(f_left[2] + f_right[2]) < 1.0, "Distributor must not add force floor"
```

**If audit fails:** Modify `SimpleForceDistributor` to ensure zero correction wrench produces zero distributed force.

**4. ContactJacobian**

No changes required. Already maps contact forces to joint torques via J^T f.

## Validation Tests

### Test 1: Equilibrium Correction Wrench Near Zero

**Setup:** 
- Calibrated initialization (root_z adjusted for -5e-4 contact penetration)
- Zero velocities (qvel = 0, qacc = 0)
- height_cmd = calibrated CoM z-position

**Expected:**
```python
model_weight = robot_mass * gravity
correction_wrench_norm = jnp.linalg.norm(correction_wrench)
assert correction_wrench_norm < 0.10 * model_weight, "Correction wrench should be < 10% of model weight"
assert abs(correction_wrench[2]) < 0.05 * model_weight, "Correction Fz should be < 5% of model weight"
```

### Test 2: Equilibrium WBC Torque Near Zero

**Setup:** Same as Test 1

**Expected:**
```python
SUPPORT_JOINTS = [2, 3, 7, 8]  # l_hip_pitch, l_knee, r_hip_pitch, r_knee
tau_wbc_support_max = jnp.max(jnp.abs(tau_wbc[SUPPORT_JOINTS]))
assert tau_wbc_support_max < 1.0, "WBC torque on support joints should be < 1.0 Nm at equilibrium"
```

### Test 3: Height Drop Produces Positive Correction

**Setup:** 
- Start from equilibrium
- Drop CoM by 0.02 m: `com_pos = com_pos.at[2].add(-0.02)`
- Set downward velocity: `com_vel = com_vel.at[2].set(-0.1)`

**Expected:**
```python
assert correction_wrench[2] > 0.05 * model_weight, "Height drop should produce upward correction > 5% of model weight"
```

**Physical validation:**
- Upward correction force should reduce height error over time
- If applied for one timestep, should produce upward acceleration

### Test 4: Pitch Perturbation Produces Stabilizing Correction

**Setup:**
- Start from equilibrium
- Apply forward pitch: `pitch_x = 0.05` rad (forward tilt)
- Zero pitch rate: `pitch_rate_x = 0.0`

**Expected:**
```python
# Pitch correction should produce sagittal force that opposes tilt
# For forward pitch (positive pitch_x), expect backward force (negative Fy)
# to create restoring moment through wheel contact
assert correction_wrench[1] < -0.03 * model_weight, "Forward pitch should produce backward correction force"
```

**Physical validation:**
- Apply correction for one timestep
- Verify pitch error decreases or pitch_rate becomes negative (restoring)

### Test 5: Roll Perturbation Produces Stabilizing Correction

**Setup:**
- Start from equilibrium
- Apply right roll: `roll_y = 0.03` rad (right tilt)
- Zero roll rate: `roll_rate_y = 0.0`

**Expected:**
```python
# Roll correction should produce moment that opposes tilt
# For right roll (positive roll_y), expect left roll moment (negative My)
assert correction_wrench[4] < -0.02 * model_weight * 0.2, "Right roll should produce left correction moment"
```

**Physical validation:**
- Apply correction for one timestep
- Verify roll error decreases or roll_rate becomes negative (restoring)

### Test 6: Force Audit - Baseline mg Not Mapped Through J^T f

**Setup:** Calibrated equilibrium, instrument force distribution pipeline

**Audit steps:**
1. Capture `correction_wrench` passed to `SimpleForceDistributor`
2. Verify `correction_wrench[2]` < 0.10 * model_weight (not baseline mg)
3. Capture `f_left`, `f_right` from force distributor
4. Verify `distributor_fz_sum = f_left[2] + f_right[2]` ≈ `correction_wrench[2]` (not mg)
5. Capture `tau_wbc` from contact Jacobian
6. Verify `tau_wbc[SUPPORT_JOINTS]` near zero (< 1.0 Nm)

**Assertions:**
```python
model_weight = robot_mass * gravity
assert correction_wrench_to_distributor[2] < 0.10 * model_weight, "Baseline mg must not be passed to distributor"
assert abs(distributor_fz_sum - correction_wrench_to_distributor[2]) < 0.05 * model_weight, "Distributed forces should match correction wrench"
assert jnp.max(jnp.abs(tau_wbc[SUPPORT_JOINTS])) < 1.0, "Support joint torques should be near zero"
```

### Test 7: Static Support Parity Comparison

**Cases:**
- **Case A:** Old WBC (baseline + correction mapped through J^T f)
- **Case B:** Correction-only WBC (only correction mapped through J^T f)
- **Case C:** Zero control (tau = 0, contact constraints only)

**Metrics:**
- Survival time (steps before termination)
- Contact force error: `|actual_fz - model_weight|`
- Support joint torque RMS
- Pitch/roll RMS

**Expected:**
- Case B survival time ≥ Case A survival time
- Case B contact force error ≤ Case A contact force error
- Case B support joint torque RMS < Case A support joint torque RMS

### Test 8: 100-Step Static Standing

**Setup:** 
- Calibrated initialization
- height_cmd = calibrated CoM z-position
- Full controller pipeline (WBC + posture + leg PD)

**Success criteria:**
- Survive 100 steps without termination
- Contact force remains within 15% of model_weight
- Pitch/roll remain < 0.1 rad (< 5.7 degrees)
- CoM height remains within ±0.05 m of height_cmd

**Failure analysis (if < 100 steps):**
- Termination < 20 steps: likely static equilibrium issue (WBC still injecting bias)
- Termination 20-50 steps: likely secondary controller interference (posture/leg PD)
- Termination 50-100 steps: likely contact solver or actuator limits
- Telemetry must identify next blocking layer

## Implementation Scope

**In scope:**
1. Add `CentroidalWrenchComputer.compute_baseline_and_correction_wrench()`
2. Add configurable correction limits (max_correction_fz_fraction, max_correction_fxy_fraction)
3. Modify `IntegratedWBC.compute_wbc_torque_with_diagnostics()` to use correction-only wrench
4. Disable force feedback in correction-only mode (or add TODO for future adaptation)
5. Audit `SimpleForceDistributor` to ensure zero correction wrench produces zero distributed force
6. Add telemetry for baseline/correction wrench breakdown
7. Implement 8 validation tests
8. Run static support parity comparison
9. Run 100-step static standing test

**Out of scope (future work):**
- QP-based force distribution
- Contact recovery logic
- Trajectory planning
- Full inverse dynamics WBC
- Stand-up recovery
- Locomotion
- Adaptive force feedback for correction-only mode

## Success Criteria

**Primary goal:** Achieve ≥ 100 step static standing with correction-only WBC

**Secondary goal (if primary fails):** Produce telemetry that clearly identifies next blocking layer:
- Posture/leg PD interference (secondary controllers reintroduce bias > 5 Nm)
- Contact solver issues (contact forces unstable)
- Actuator limits (clipping/rate limiting)
- Missing contact recovery (single-wheel contact handling)

**Validation checklist:**
- [ ] Test 1: Equilibrium correction wrench < 10% model weight
- [ ] Test 2: Equilibrium WBC torque < 1 Nm on support joints
- [ ] Test 3: Height drop produces positive correction Fz
- [ ] Test 4: Pitch perturbation produces stabilizing correction (verified by error reduction)
- [ ] Test 5: Roll perturbation produces stabilizing correction (verified by error reduction)
- [ ] Test 6: Force audit confirms mg not mapped through J^T f
- [ ] Test 7: Correction-only WBC outperforms old WBC in parity test
- [ ] Test 8: 100-step static standing OR clear failure telemetry

## Risk Mitigation

**Risk 1: Correction-only WBC still fails to achieve 100 steps**

Mitigation: Comprehensive telemetry identifies next blocker. If secondary controllers (posture/leg PD) reintroduce bias > 5 Nm on support joints, flag for follow-up fix. If contact forces remain unstable, investigate contact solver parameters.

**Risk 2: Removing baseline mg causes robot to collapse immediately**

Mitigation: Contact constraints should provide baseline support. If robot collapses at t=0, verify:
1. Calibration produces proper wheel-floor contact (penetration -5e-4 m)
2. Contact solver is enabled and configured correctly
3. Floor/wheel collision geoms are active
4. Contact stiffness/damping parameters are reasonable

**Risk 3: Correction limits too restrictive, prevent recovery**

Mitigation: Start with generous limits (35% model weight for Fz, 20% for Fxy). If recovery fails, increase limits incrementally and re-test. Limits should be configurable parameters, not hard-coded constants.

**Risk 4: SimpleForceDistributor reintroduces baseline Fz via min_wheel_force**

Mitigation: Audit distributor with zero correction wrench test case. If audit fails, modify distributor to ensure min_wheel_force only affects asymmetry limits, not total Fz.

## Telemetry Requirements

**Per-step logging:**
```python
{
    # Model parameters
    "robot_mass": 15.0,  # kg
    "gravity": 9.81,     # m/s²
    "model_weight": 147.15,  # N
    
    # Baseline wrench (diagnostic)
    "baseline_fz": 147.15,  # Should equal model_weight
    
    # Correction wrench (control)
    "correction_fx": 0.5,
    "correction_fy": -2.3,
    "correction_fz": 3.1,
    "correction_mx": 0.0,
    "correction_my": -1.2,
    "correction_mz": 0.0,
    "correction_wrench_norm": 4.2,
    
    # Force breakdown
    "distributor_fz_sum": 3.1,  # Should match correction_fz
    "actual_contact_fz": 148.7,
    "force_error": 1.55,  # actual - model_weight
    
    # Torque breakdown
    "tau_wbc_correction": [0.1, 0.0, 0.3, 0.2, 0.0, 0.1, 0.0, 0.3, 0.2, 0.0],
    "tau_wbc_final": [0.1, 0.0, 0.3, 0.2, 0.0, 0.1, 0.0, 0.3, 0.2, 0.0],
    "tau_wbc_support_joints_rms": 0.25,
    
    # State
    "pitch_x": 0.002,
    "roll_y": -0.001,
    "height_error": 0.005,
    "com_pos_x": 0.001,
    "com_pos_y": -0.002,
    "com_pos_z": 0.534,
}
```

## References

- **Failed approach:** [StaticBalanceController wrapper](../plans/2026-05-23-static-dynamics-consistency-fix-plan.md)
- **Root cause analysis:** Phases 0-3 diagnostics (scripts/debug_*.py)
- **Validation failure:** debug_static_support_parity_v2.py output showing 14× worse performance
- **Physics principle:** Contact constraints provide baseline support in static equilibrium
- **Mass convention:** simulate_hierarchical_controller.py line 377: `robot_mass = float(np.sum(mj_model.body_mass))`
