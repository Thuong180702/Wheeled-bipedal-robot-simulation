# WBC Correction-Only Fix Specification

**Date:** 2026-05-23  
**Status:** Approved  
**Replaces:** StaticBalanceController wrapper approach (failed validation)

## Problem Statement

The wheeled biped robot falls after 14-15 steps due to a 15-20N vertical contact force deficit (desired ~79N vs actual 60-67N, 19-25% shortfall). Root cause analysis revealed:

1. **Current WBC behavior:** Maps entire baseline body weight (mg = 147N) through joint-only contact Jacobian (J^T f), producing large support-joint torques that fight against contact constraints
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
baseline_wrench = jnp.array([0.0, 0.0, self.robot_mass * self.gravity, 0.0, 0.0, 0.0])
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

**Correction limits:**
```python
# Clamp correction_Fz to prevent reintroducing baseline mg
MAX_CORRECTION_FZ = 50.0  # N (reasonable correction range)
correction_Fz = jnp.clip(correction_Fz, -MAX_CORRECTION_FZ, MAX_CORRECTION_FZ)

# Clamp horizontal forces
MAX_CORRECTION_FXY = 30.0  # N
correction_Fx = jnp.clip(correction_Fx, -MAX_CORRECTION_FXY, MAX_CORRECTION_FXY)
correction_Fy = jnp.clip(correction_Fy, -MAX_CORRECTION_FXY, MAX_CORRECTION_FXY)
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
    
    # Apply authority budget and force feedback
    tau_wbc = self.clip_to_authority_budget(tau_wbc_correction * force_scale)
    
    # Diagnostics
    diagnostics = {
        # Baseline wrench (diagnostic only)
        "baseline_wrench_Fx": float(baseline_wrench[0]),
        "baseline_wrench_Fy": float(baseline_wrench[1]),
        "baseline_wrench_Fz": float(baseline_wrench[2]),  # Should be mg
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
        "baseline_fz": float(baseline_wrench[2]),
        "correction_fz": float(correction_wrench[2]),
        "total_expected_support_fz": float(baseline_wrench[2] + correction_wrench[2]),
        "actual_contact_fz": actual_fz_total,
        
        # Torque breakdown
        "tau_wbc_correction": tau_wbc_correction,
        "tau_wbc_final": tau_wbc,
        
        # Existing diagnostics...
    }
    
    return tau_wbc, diagnostics
```

**3. SimpleForceDistributor**

No changes required. Already accepts 6D wrench and distributes to wheel forces + hip roll torques.

**4. ContactJacobian**

No changes required. Already maps contact forces to joint torques via J^T f.

## Validation Tests

### Test 1: Equilibrium Correction Wrench Near Zero

**Setup:** Calibrated initialization (root_z adjusted for -5e-4 contact penetration), zero velocities

**Expected:**
- `correction_wrench_Fz` < 5.0 N (small height tracking correction)
- `correction_wrench_Fx` < 2.0 N (small lateral correction)
- `correction_wrench_Fy` < 2.0 N (small sagittal correction)
- `correction_wrench_My` < 5.0 Nm (small roll correction)
- `baseline_wrench_Fz` = 147.15 N (15 kg × 9.81 m/s²)

**Assertion:**
```python
assert abs(correction_wrench[2]) < 5.0, "Correction Fz should be near zero at equilibrium"
assert jnp.linalg.norm(correction_wrench) < 10.0, "Total correction wrench should be small"
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

**Setup:** Set pitch_x = 0.05 rad (forward tilt), pitch_rate_x = 0.0

**Expected:**
- `correction_wrench_Fy` < 0 (backward force to counteract forward tilt)
- Wheel torques produce backward acceleration

**Assertion:**
```python
assert correction_wrench[1] < -5.0, "Forward pitch should produce backward correction Fy"
```

### Test 5: Roll Perturbation Produces Stabilizing Correction

**Setup:** Set roll_y = 0.03 rad (right tilt), roll_rate_y = 0.0

**Expected:**
- `correction_wrench_My` < 0 (left roll moment to counteract right tilt)
- Hip roll torques produce left roll correction

**Assertion:**
```python
assert correction_wrench[4] < -2.0, "Right roll should produce left correction My"
```

### Test 6: Force Audit - Baseline mg Not Mapped Through J^T f

**Setup:** Calibrated equilibrium, instrument force distribution pipeline

**Audit:**
1. Capture `correction_wrench` passed to `SimpleForceDistributor`
2. Verify `correction_wrench[2]` does NOT contain baseline mg (should be < 10 N)
3. Capture `f_left`, `f_right` from force distributor
4. Verify `f_left[2] + f_right[2]` ≈ `correction_wrench[2]` (not mg)
5. Capture `tau_wbc` from contact Jacobian
6. Verify `tau_wbc[SUPPORT_JOINTS]` near zero (< 1.0 Nm)

**Assertion:**
```python
assert correction_wrench_to_distributor[2] < 10.0, "Baseline mg must not be passed to distributor"
assert abs((f_left[2] + f_right[2]) - correction_wrench_to_distributor[2]) < 1.0, "Distributed forces should match correction wrench"
```

### Test 7: Static Support Parity Comparison

**Cases:**
- **Case A:** Old WBC (baseline + correction mapped through J^T f)
- **Case B:** Correction-only WBC (only correction mapped through J^T f)
- **Case C:** Zero control (tau = 0, contact constraints only)
- **Case D:** Inverse dynamics baseline (for reference, known to be flawed)

**Metrics:**
- Survival time (steps before termination)
- Contact force error: `|actual_fz - 79.5 N|`
- Support joint torque RMS
- Pitch/roll RMS

**Expected:**
- Case B (correction-only) should survive ≥ 100 steps or show clear improvement over Case A
- Case B contact force error < Case A contact force error
- Case B support joint torque RMS < Case A support joint torque RMS

### Test 8: 100-Step Static Standing

**Setup:** Calibrated initialization, height_cmd = 0.534 m, full controller pipeline (WBC + posture + leg PD)

**Success criteria:**
- Survive 100 steps without termination
- Contact force remains 70-90 N (within 15% of 79.5 N)
- Pitch/roll remain < 0.1 rad (< 5.7 degrees)
- CoM height remains 0.50-0.57 m

**Failure analysis (if < 100 steps):**
- If termination < 20 steps: likely static equilibrium issue (WBC still injecting bias)
- If termination 20-50 steps: likely secondary controller interference (posture/leg PD)
- If termination 50-100 steps: likely contact solver or actuator limits
- Telemetry must identify next blocking layer

## Implementation Scope

**In scope:**
1. Modify `CentroidalWrenchComputer.compute_baseline_and_correction_wrench()`
2. Modify `IntegratedWBC.compute_wbc_torque_with_diagnostics()` to use correction-only wrench
3. Add correction limits (MAX_CORRECTION_FZ, MAX_CORRECTION_FXY)
4. Add telemetry for baseline/correction wrench breakdown
5. Implement 8 validation tests
6. Run static support parity comparison
7. Run 100-step static standing test

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
- [ ] Test 1: Equilibrium correction wrench < 10 N norm
- [ ] Test 2: Equilibrium WBC torque < 1 Nm on support joints
- [ ] Test 3: Height drop produces positive correction Fz
- [ ] Test 4: Pitch perturbation produces stabilizing correction Fy
- [ ] Test 5: Roll perturbation produces stabilizing correction My
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

**Risk 3: Correction limits too restrictive, prevent recovery**

Mitigation: Start with generous limits (MAX_CORRECTION_FZ = 50 N, MAX_CORRECTION_FXY = 30 N). If recovery fails, increase limits incrementally and re-test.

## Telemetry Requirements

**Per-step logging:**
```python
{
    # Baseline wrench (diagnostic)
    "baseline_fz": 147.15,  # Should be constant mg
    
    # Correction wrench (control)
    "correction_fx": 0.5,
    "correction_fy": -2.3,
    "correction_fz": 3.1,
    "correction_mx": 0.0,
    "correction_my": -1.2,
    "correction_mz": 0.0,
    "correction_wrench_norm": 4.2,
    
    # Force breakdown
    "total_expected_support_fz": 150.25,  # baseline + correction
    "actual_contact_fz": 148.7,
    "force_error": -1.55,
    
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
}
```

## References

- **Failed approach:** [StaticBalanceController wrapper](../plans/2026-05-23-static-dynamics-consistency-fix-plan.md)
- **Root cause analysis:** Phases 0-3 diagnostics (scripts/debug_*.py)
- **Validation failure:** debug_static_support_parity_v2.py output showing 14× worse performance
- **Physics principle:** Contact constraints provide baseline support in static equilibrium
