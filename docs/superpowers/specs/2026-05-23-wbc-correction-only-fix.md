# WBC Correction-Only Fix Specification

**Date:** 2026-05-23  
**Status:** Approved (Revised)  
**Replaces:** StaticBalanceController wrapper approach (failed validation)

## Problem Statement

The wheeled biped robot falls after 14-15 steps due to a vertical contact force deficit. Root cause analysis revealed:

1. **Current WBC behavior:** Maps entire baseline body weight (mg) through joint-only contact Jacobian (J^T f), producing large support-joint torques that fight against contact constraints
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
- pitch_x = 0.05 rad → correction_Fy < 0 (stabilizing sagittal force)
- height_error = -0.02 m → correction_Fz > 0 (upward correction)
- roll_y = 0.03 rad → correction_My < 0 (stabilizing roll moment)

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
    
    Returns:
        Tuple of (baseline_wrench, correction_wrench) where:
            - baseline_wrench: (6,) [0, 0, mg, 0, 0, 0] - diagnostic only
            - correction_wrench: (6,) [Fx, Fy, Fz, Mx, My, Mz] - control output
    """
```

**Baseline wrench (diagnostic only):**
```python
# Baseline vertical force = total model weight
# robot_mass should be computed as: float(np.sum(mj_model.body_mass))
# gravity should be: abs(float(mj_model.opt.gravity[2]))
baseline_fz = self.robot_mass * self.gravity
baseline_wrench = jnp.array([0.0, 0.0, baseline_fz, 0.0, 0.0, 0.0])
```

**Correction wrench (control output):**
```python
# Height tracking: proportional + damping (NO baseline mg)
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
# Clamp correction_Fz to prevent reintroducing baseline mg
# Express limits as fraction of model weight for generality
max_correction_fz_fraction = 0.35  # 35% of body weight (configurable parameter)
max_correction_fxy_fraction = 0.20  # 20% of body weight (configurable parameter)

model_weight = self.robot_mass * self.gravity
MAX_CORRECTION_FZ = max_correction_fz_fraction * model_weight
MAX_CORRECTION_FXY = max_correction_fxy_fraction * model_weight

correction_Fz = jnp.clip(correction_Fz, -MAX_CORRECTION_FZ, MAX_CORRECTION_FZ)
correction_Fx = jnp.clip(correction_Fx, -MAX_CORRECTION_FXY, MAX_CORRECTION_FXY)
correction_Fy = jnp.clip(correction_Fy, -MAX_CORRECTION_FXY, MAX_CORRECTION_FXY)
```

**2. IntegratedWBC**

Modify `compute_wbc_torque_with_diagnostics` to use correction-only wrench.

**CRITICAL: Force feedback adaptation**

The existing force feedback mechanism scales WBC torque based on `(actual_fz - desired_fz) / desired_fz`. In the old WBC, `desired_fz` included baseline mg, so the feedback compensated for contact force errors relative to total body weight. In correction-only mode, `desired_fz` from the force distributor will be near zero at equilibrium, making the ratio undefined or unstable.

**Solution:** Redefine force feedback to use error relative to baseline model_weight:

```python
def compute_wbc_torque_with_diagnostics(
    self,
    mj_data: mujoco.MjData,
    obs: Array,
    state: CentroidalState,
    height_cmd: float,
    hip_roll_authority_scale: float = 1.0,
) -> tuple[Array, dict]:
    # Compute baseline (diagnostic) and correction (control) wrenches
    baseline_wrench, correction_wrench = self.wrench_computer.compute_baseline_and_correction_wrench(
        obs, state, height_cmd, self.roll_integral
    )
    
    # CRITICAL: Only pass correction_wrench to force distributor
    # Baseline mg is handled by contact constraints, NOT mapped through J^T f
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
    
    # Force feedback adaptation for correction-only mode
    # OLD: force_scale = 1.0 - gain * (actual - desired) / desired
    # NEW: force_scale based on error relative to baseline model_weight
    model_weight = self.wrench_computer.robot_mass * self.wrench_computer.gravity
    if contact_force_valid and model_weight > 1e-3:
        force_error = actual_fz_total - model_weight
        force_error_fraction = force_error / model_weight
        # Reduce torque if actual > model_weight (too much support)
        # Increase torque if actual < model_weight (too little support)
        force_scale = 1.0 - self.force_feedback_gain * force_error_fraction
        force_scale = float(jnp.clip(force_scale, 0.1, 2.0))
    else:
        force_scale = 1.0
    
    # Apply authority budget and force feedback
    tau_wbc = self.clip_to_authority_budget(tau_wbc_correction * force_scale)
    
    # Diagnostics
    diagnostics = {
        # Baseline wrench (diagnostic only)
        "baseline_wrench_Fx": float(baseline_wrench[0]),
        "baseline_wrench_Fy": float(baseline_wrench[1]),
        "baseline_wrench_Fz": float(baseline_wrench[2]),  # Should equal model_weight
        "baseline_wrench_Mx": float(baseline_wrench[3]),
        "baseline_wrench_My": float(baseline_wrench[4]),
        "baseline_wrench_Mz": float(baseline_wrench[5]),
        
        # Correction wrench (control output)
        "correction_wrench_Fx": float(correction_wrench[0]),
        "correction_wrench_Fy": float(correction_wrench[1]),
        "correction_wrench_Fz": float(correction_wrench[2]),
        "correction_wrench_Mx": float(correction_wrench[3]),
        "correction_wrench_My": float(correction_wrench[4]),
        "correction_wrench_Mz": float(correction_wrench[5]),
        
        # Force breakdown
        "baseline_fz": float(baseline_wrench[2]),  # Should equal model_weight
        "correction_fz": float(correction_wrench[2]),
        "distributor_fz_sum": float(f_left[2] + f_right[2]),  # Should match correction_fz
        "total_expected_support_fz": float(baseline_wrench[2] + correction_wrench[2]),
        "actual_contact_fz": actual_fz_total,
        "force_error": actual_fz_total - float(baseline_wrench[2]),  # Error relative to model_weight
        
        # Torque breakdown
        "tau_wbc_correction": tau_wbc_correction,
        "tau_wbc_final": tau_wbc,
        
        # Existing diagnostics...
    }
    
    return tau_wbc, diagnostics
```

**3. SimpleForceDistributor**

**CRITICAL AUDIT REQUIRED:** The force distributor must be verified to NOT reintroduce baseline body weight when correction_wrench ≈ 0.

Current implementation has `min_wheel_force = 10.0 N` safety floor. At equilibrium with correction_wrench_Fz ≈ 0:
- If distributor enforces `f_left[2] >= 10.0` and `f_right[2] >= 10.0`
- Then `f_left[2] + f_right[2] >= 20.0 N` even when correction_wrench_Fz = 0
- This reintroduces 20N of baseline force through J^T f

**Required behavior:**
- At equilibrium: `f_left[2] + f_right[2]` should match `correction_wrench_Fz` (near zero)
- `min_wheel_force` should only apply during single-wheel contact recovery, NOT at two-wheel equilibrium
- Force floor logic must not silently add baseline support

**Validation:**
```python
# Test: correction_wrench = [0, 0, 0, 0, 0, 0], both wheels in contact
f_left, f_right, tau_hip_roll, _ = distributor.distribute_wrench_contact_aware(
    correction_wrench, left_contact=True, right_contact=True, ...
)
assert abs(f_left[2] + f_right[2]) < 1.0, "Distributor must not add baseline force at equilibrium"
```

**4. ContactJacobian**

No changes required. Already maps contact forces to joint torques via J^T f.

## Validation Tests

### Test 1: Equilibrium Correction Wrench Near Zero

**Setup:**
- Calibrated initialization: `mj_resetDataKeyframe` → `mj_forward` → `calibrate_root_z_for_wheel_floor_contact(target_dist=-5e-4)` → zero qvel/qacc → `mj_forward`
- Compute model_weight: `model_weight = float(np.sum(mj_model.body_mass)) * abs(float(mj_model.opt.gravity[2]))`
- Build observation with height_cmd = current CoM z (from `data.subtree_com[1, 2]`)

**Height definition clarity:**
- `root_z` = `data.qpos[2]` (floating base z position)
- `com_z` = `data.subtree_com[1, 2]` (center of mass z, body index 1 = torso)
- `height_cmd` = desired CoM z (NOT root_z)
- At equilibrium: `height_cmd` should equal current `com_z` to produce zero height error

**Expected:**
- `correction_wrench_Fz` < 0.05 * model_weight (5% of body weight)
- `correction_wrench_Fx` < 0.02 * model_weight (2% of body weight)
- `correction_wrench_Fy` < 0.02 * model_weight (2% of body weight)
- `correction_wrench_My` < 5.0 Nm (small roll correction)
- `baseline_wrench_Fz` = model_weight

**Assertion:**
```python
model_weight = float(np.sum(mj_model.body_mass)) * abs(float(mj_model.opt.gravity[2]))
assert abs(correction_wrench[2]) < 0.05 * model_weight, "Correction Fz should be < 5% of model weight"
assert jnp.linalg.norm(correction_wrench) < 0.10 * model_weight, "Total correction wrench should be small"
assert abs(baseline_wrench[2] - model_weight) < 0.01, "Baseline Fz should equal model weight"
```

### Test 2: Equilibrium WBC Torque Near Zero

**Setup:** Same as Test 1

**Expected:**
- `tau_wbc[SUPPORT_JOINTS]` < 1.0 Nm per joint (support joints: [2, 3, 7, 8])
- `jnp.linalg.norm(tau_wbc)` < 5.0 Nm (total WBC torque magnitude)

**Assertion:**
```python
SUPPORT_JOINTS = [2, 3, 7, 8]
assert jnp.max(jnp.abs(tau_wbc[SUPPORT_JOINTS])) < 1.0, "WBC torque on support joints should be near zero"
```

### Test 3: Height Drop Produces Positive Correction

**Setup:** Drop CoM by 0.02 m, set com_vel[2] = -0.1 m/s

**Expected:**
- `correction_wrench_Fz` > 10.0 N (upward correction)
- `tau_wbc[SUPPORT_JOINTS]` produces net upward force through contact Jacobian

**Assertion:**
```python
assert correction_wrench[2] > 10.0, "Height drop should produce positive correction Fz"
```

### Test 4: Pitch Perturbation Produces Stabilizing Correction

**Setup:** Perturb pitch_x = 0.05 rad (forward tilt), pitch_rate_x = 0.0

**Physical stabilization criterion:**
- Forward pitch (pitch_x > 0) means CoM is ahead of support base
- Stabilizing correction requires backward force (Fy < 0) to decelerate forward motion
- Through contact Jacobian, backward Fy produces negative wheel torques (backward wheel acceleration)

**Expected:**
- `correction_wrench_Fy` < 0 (backward force to counteract forward tilt)
- Sign backed by inverted pendulum dynamics: F = -k * theta for stabilization

**Assertion:**
```python
# Pitch correction force should oppose pitch angle (negative gain)
pitch_correction_fy = -k_pitch * pitch_x - k_pitch_rate * pitch_rate_x
assert pitch_correction_fy < 0, "Forward pitch should produce backward correction Fy (inverted pendulum)"
assert correction_wrench[1] < 0, "Total correction Fy should be negative for forward pitch"
```

### Test 5: Roll Perturbation Produces Stabilizing Correction

**Setup:** Perturb roll_y = 0.03 rad (right tilt), roll_rate_y = 0.0

**Physical stabilization criterion:**
- Right roll (roll_y > 0) means robot tilting to the right
- Stabilizing correction requires left roll moment (My < 0) to counteract right tilt
- Roll moment convention: My < 0 produces left roll correction

**Expected:**
- `correction_wrench_My` < 0 (left roll moment to counteract right tilt)
- Sign backed by PID control: M = -k * roll for stabilization

**Assertion:**
```python
# Roll correction moment should oppose roll angle (negative gain)
roll_correction_my = -k_roll * roll_y - k_roll_rate * roll_rate_y
assert roll_correction_my < 0, "Right roll should produce left correction My (negative gain)"
assert correction_wrench[4] < 0, "Total correction My should be negative for right roll"
```

### Test 6: Force Audit - Baseline mg Not Mapped Through J^T f

**Setup:** Calibrated equilibrium, instrument force distribution pipeline

**Audit:**
1. Compute `model_weight = float(np.sum(mj_model.body_mass)) * abs(float(mj_model.opt.gravity[2]))`
2. Capture `correction_wrench` passed to `SimpleForceDistributor`
3. Verify `correction_wrench[2]` does NOT contain baseline mg (should be < 0.10 * model_weight)
4. Capture `f_left`, `f_right` from force distributor
5. Verify `f_left[2] + f_right[2]` ≈ `correction_wrench[2]` (NOT model_weight)
6. Verify distributor does not add force floor at two-wheel equilibrium
7. Capture `tau_wbc` from contact Jacobian
8. Verify `tau_wbc[SUPPORT_JOINTS]` near zero (< 1.0 Nm per joint)

**Assertion:**
```python
model_weight = float(np.sum(mj_model.body_mass)) * abs(float(mj_model.opt.gravity[2]))
assert correction_wrench_to_distributor[2] < 0.10 * model_weight, "Baseline mg must not be passed to distributor"
distributor_fz_sum = f_left[2] + f_right[2]
assert abs(distributor_fz_sum - correction_wrench_to_distributor[2]) < 0.02 * model_weight, \
    "Distributed forces should match correction wrench, not add baseline"
assert abs(distributor_fz_sum) < 0.15 * model_weight, \
    "Distributor must not reintroduce baseline force via min_wheel_force at equilibrium"
```

### Test 7: Static Support Parity Comparison

**Cases:**
- **Case A:** Old WBC (baseline + correction mapped through J^T f)
- **Case B:** Correction-only WBC (only correction mapped through J^T f)
- **Case C:** Zero control (tau = 0, contact constraints only)
- **Case D:** Inverse dynamics baseline (for reference, known to be flawed)

**Metrics:**
- Survival time (steps before termination)
- Contact force error: `|actual_fz - model_weight|` where `model_weight = float(np.sum(mj_model.body_mass)) * abs(float(mj_model.opt.gravity[2]))`
- Support joint torque RMS (joints [2, 3, 7, 8])
- Pitch/roll RMS

**Expected:**
- Case B (correction-only) should survive ≥ 100 steps or show clear improvement over Case A
- Case B contact force error < Case A contact force error
- Case B support joint torque RMS < Case A support joint torque RMS
- Case B should show near-zero WBC torque at equilibrium (first 5 steps)

### Test 8: 100-Step Static Standing

**Setup:**
- Calibrated initialization
- Compute `model_weight = float(np.sum(mj_model.body_mass)) * abs(float(mj_model.opt.gravity[2]))`
- Set `height_cmd = data.subtree_com[1, 2]` (current CoM z at equilibrium)
- Full controller pipeline (WBC + posture + leg PD)

**Success criteria:**
- Survive 100 steps without termination
- Contact force remains within 15% of model_weight: `0.85 * model_weight < actual_fz < 1.15 * model_weight`
- Pitch/roll remain < 0.1 rad (< 5.7 degrees)
- CoM height remains within ±0.05 m of height_cmd

**Failure analysis (if < 100 steps):**
- If termination < 20 steps: likely static equilibrium issue (WBC still injecting bias)
- If termination 20-50 steps: likely secondary controller interference (posture/leg PD)
- If termination 50-100 steps: likely contact solver or actuator limits
- Telemetry must identify next blocking layer

## Implementation Scope

**In scope:**
1. Modify `CentroidalWrenchComputer.compute_baseline_and_correction_wrench()`
2. Add configurable correction limit parameters (max_correction_fz_fraction, max_correction_fxy_fraction)
3. Modify `IntegratedWBC.compute_wbc_torque_with_diagnostics()` to use correction-only wrench
4. Adapt force feedback to use error relative to model_weight
5. Audit `SimpleForceDistributor` to ensure no baseline force reintroduction at equilibrium
6. Add telemetry for baseline/correction wrench breakdown with distributor_fz_sum
7. Implement 8 validation tests with model_weight computation
8. Run static support parity comparison
9. Run 100-step static standing test

**Out of scope (future work):**
- QP-based force distribution
- Contact recovery logic
- Trajectory planning
- Full inverse dynamics WBC
- Stand-up recovery
- Locomotion

## Success Criteria

**Primary goal:** Achieve ≥ 100 step static standing with correction-only WBC

**Secondary goal (if primary fails):** Produce telemetry that clearly identifies next blocking layer:
- Posture/leg PD interference (secondary controllers reintroduce bias)
- Contact solver issues (contact forces unstable)
- Actuator limits (clipping/rate limiting)
- Missing contact recovery (single-wheel contact handling)

**Validation checklist:**
- [ ] Test 1: Equilibrium correction wrench < 10% of model_weight norm
- [ ] Test 2: Equilibrium WBC torque < 1 Nm on support joints
- [ ] Test 3: Height drop produces positive correction Fz
- [ ] Test 4: Pitch perturbation produces stabilizing correction Fy (with physical criterion)
- [ ] Test 5: Roll perturbation produces stabilizing correction My (with physical criterion)
- [ ] Test 6: Force audit confirms mg not mapped through J^T f, distributor doesn't add baseline
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

**Risk 3: Correction limits too restrictive, prevent recovery**

Mitigation: Start with generous limits (max_correction_fz_fraction = 0.35, max_correction_fxy_fraction = 0.20). If recovery fails, increase fractions incrementally and re-test.

**Risk 4: SimpleForceDistributor reintroduces baseline via min_wheel_force**

Mitigation: Audit distributor behavior at equilibrium. If force floor applies at two-wheel contact, modify logic to only enforce minimum during single-wheel recovery, not at equilibrium.

## Telemetry Requirements

**Per-step logging:**
```python
{
    # Model parameters (computed once at initialization)
    "model_weight": 147.15,  # float(np.sum(mj_model.body_mass)) * abs(float(mj_model.opt.gravity[2]))
    
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
    "distributor_fz_sum": 3.0,  # f_left[2] + f_right[2], should match correction_fz
    "total_expected_support_fz": 150.25,  # baseline + correction
    "actual_contact_fz": 148.7,
    "force_error": 1.55,  # actual_contact_fz - model_weight (NOT - total_expected)
    
    # Torque breakdown
    "tau_wbc_correction": [0.1, 0.0, 0.3, 0.2, 0.0, 0.1, 0.0, 0.3, 0.2, 0.0],
    "tau_wbc_final": [0.1, 0.0, 0.3, 0.2, 0.0, 0.1, 0.0, 0.3, 0.2, 0.0],
    "tau_wbc_support_joints_rms": 0.25,
    
    # State
    "pitch_x": 0.002,
    "roll_y": -0.001,
    "height_error": 0.005,  # height_cmd - com_z
    "com_z": 0.534,  # data.subtree_com[1, 2]
    "root_z": 0.595,  # data.qpos[2]
    "com_pos_x": 0.001,
    "com_pos_y": -0.002,
}
```

## References

- **Failed approach:** [StaticBalanceController wrapper](../plans/2026-05-23-static-dynamics-consistency-fix-plan.md)
- **Root cause analysis:** Phases 0-3 diagnostics (scripts/debug_*.py)
- **Validation failure:** debug_static_support_parity_v2.py output showing 14× worse performance
- **Physics principle:** Contact constraints provide baseline support in static equilibrium
- **Mass computation convention:** `float(np.sum(mj_model.body_mass))` from simulate_hierarchical_controller.py
