# Static Dynamics Consistency Fix

**Date**: 2026-05-23  
**Status**: Draft  
**Scope**: Minimal fix for static standing balance - wrapper approach

## Problem Statement

Phases 0-3 diagnostics identified the root cause of the 15-20N force gap:

**Current WBC behavior**: Maps the entire baseline vertical support force (Fz = mg ≈ 79N) through a joint-only contact Jacobian (J^T f), generating ~15 Nm knee torques at the calibrated standing keyframe.

**Physics reality**: Inverse dynamics shows that support joints [2,3,7,8] require ≈ 0 Nm at the calibrated equilibrium because contact constraints support body weight, not the actuators.

**Consequence**: WBC injects unnecessary support torques, creating a static dynamics inconsistency that prevents stable standing.

## Core Principle

**Separate baseline static equilibrium from active balance corrections:**

- **Baseline support**: Contact constraints support body weight in static equilibrium
- **Actuator role**: Provide minimal joint torques as shown by inverse dynamics (≈ 0 Nm at calibrated keyframe)
- **WBC role**: Generate correction torques only for deviations from equilibrium (pitch/roll/height errors)

**Key insight**: Don't map baseline mg through joint-only J^T f. Only map correction forces/moments through the Jacobian.

## Solution Architecture

### Approach: Controller Wrapper (Minimal Invasive)

Create `StaticBalanceController` wrapper that:
- Wraps existing WBC pipeline without modifying it
- Computes static reference torques once at initialization
- Cancels WBC equilibrium bias at runtime
- Outputs only correction torques for deviations from equilibrium

**Why this approach:**
- Minimal changes to existing tested code
- Easy to validate and toggle on/off for A/B comparison
- Proves the static-bias hypothesis before deeper refactoring
- Follows "minimal fix" principle

**Architecture:**
```
tau_wbc_current = existing WBC output
tau_wbc_correction = tau_wbc_current - tau_wbc_equilibrium
tau_wbc_wrapped = tau_static_ref + tau_wbc_correction

Then simulation continues:
tau_total_raw = tau_wbc_wrapped + tau_posture + tau_leg_position + ...
tau_final = apply_clipping_and_smoothing(tau_total_raw)
```

## Initialization and Reference Computation

### Calibrated Initialization Sequence

Same as diagnostic scripts to ensure consistency:

1. Reset to keyframe pose
2. Call `mj_forward(model, data)`
3. Calibrate `root_z` to achieve wheel-floor contact distance = -5e-4 m (0.5mm penetration)
4. Zero `qvel` and `qacc`
5. Call `mj_forward(model, data)` again

**Important**: Use a **copied MuJoCo data object** for reference computation, not the live simulation data, to avoid mutating active sim state.

### Reference State Capture

After calibration, capture equilibrium state:

```python
equilibrium_state = {
    'com_z': measured CoM height at calibrated equilibrium,
    'pitch_x': measured pitch at equilibrium (should be ≈ 0),
    'roll_y': measured roll at equilibrium (should be ≈ 0),
    'joint_pos': qpos[7:17] at equilibrium,
    'contact_points': wheel contact positions at equilibrium,
}
```

### Reference Torque Computation

**1. tau_static_ref** (inverse dynamics at equilibrium):
```python
# At calibrated equilibrium with qvel=0, qacc=0
mj_inverse(model, data_copy)
tau_static_ref = qfrc_inverse[6:16]

# Also store for diagnostics:
qfrc_inverse_ref = qfrc_inverse[6:16]
qfrc_bias_ref = qfrc_bias[6:16]
qfrc_constraint_ref = qfrc_constraint[6:16]  # if available
```

Expected: `tau_static_ref[2,3,7,8]` ≈ 0 Nm (per diagnostics)

**2. tau_wbc_equilibrium** (WBC output at equilibrium):
```python
# Build observation with zero errors:
# - qvel = 0, qacc = 0
# - com_vel = 0
# - pitch_rate_x = 0, roll_rate_y = 0
# - height_cmd = equilibrium_com_z (zero height error)
# - capture point velocity contribution = 0

obs_equilibrium = build_zero_error_observation(equilibrium_state)
tau_wbc_equilibrium = wbc_pipeline.compute(obs_equilibrium)
```

This captures the "static bias" that needs to be cancelled.

**3. Diagnostic logging at initialization**:
```
[STATIC BALANCE CONTROLLER INITIALIZATION]

Equilibrium State:
  com_z: 0.400 m
  pitch_x: 0.001 rad
  roll_y: -0.002 rad
  
Static Reference Torques (from inverse dynamics):
  tau_static_ref[2,3,7,8] = [0.2, -0.1, 0.2, -0.1] Nm
  
WBC Equilibrium Bias:
  tau_wbc_equilibrium[2,3,7,8] = [8.5, 15.2, 8.5, 15.2] Nm
  
Support Bias Removed:
  support_bias[2,3,7,8] = [8.3, 15.3, 8.3, 15.3] Nm
```

## Runtime Behavior

### Each Control Step

**1. Compute current WBC torque**:
```python
tau_wbc_current = wbc_pipeline.compute(obs_current)
```

**2. Compute correction torque** (remove equilibrium bias):
```python
tau_wbc_correction = tau_wbc_current - tau_wbc_equilibrium
```

This isolates only the correction component for deviations from equilibrium.

**3. Compute wrapped WBC torque**:
```python
tau_wbc_wrapped = tau_static_ref + tau_wbc_correction
```

**4. Compute equilibrium error metrics** (for validity tracking):
```python
# Separate errors, not one mixed-unit norm
posture_error_norm = ||joint_pos_current - joint_pos_equilibrium||
com_height_error = com_z_current - com_z_equilibrium
pitch_x_error = pitch_x_current - pitch_x_equilibrium
roll_y_error = roll_y_current - roll_y_equilibrium
com_velocity_norm = ||com_vel_current||
angular_velocity_norm = ||angular_vel_current||
```

**5. Safety diagnostic** (not a hard control switch):
```python
# Log warning if far from equilibrium
if posture_error_norm > 0.1 or abs(com_height_error) > 0.05:
    log_warning("Fixed static reference may no longer be physically exact")
```

### Expected Behavior

**At calibrated equilibrium**:
- `tau_wbc_correction[2,3,7,8]` ≈ 0 Nm (< 0.5 Nm)
- `tau_wbc_wrapped[2,3,7,8]` ≈ `tau_static_ref[2,3,7,8]`
- Old WBC bias (~15 Nm knee torques) is cancelled

**With perturbations**:
- Pitch_x error → `tau_wbc_correction` nonzero, opposes pitch error
- Roll_y error → `tau_wbc_correction` nonzero, opposes roll error
- Height error → `tau_wbc_correction` nonzero, increases upward force
- Corrections are stabilizing (reduce error or acceleration)

### Telemetry

Log each step (especially first 20 steps when wrapper enabled):

**Torque components** (support joints [2,3,7,8]):
- `tau_static_ref`
- `tau_wbc_equilibrium`
- `tau_wbc_current`
- `tau_wbc_correction`
- `tau_wbc_wrapped`
- `support_joint_bias_removed` (= tau_wbc_equilibrium)

**Equilibrium error metrics**:
- `posture_error_norm`
- `com_height_error`
- `pitch_x_error`
- `roll_y_error`
- `com_velocity_norm`
- `angular_velocity_norm`

**Pipeline audit**:
- `tau_total_raw` (after adding posture/leg PD/etc)
- `tau_final` (after clipping/smoothing)

## Integration with Simulation Pipeline

### Wrapper Scope

`StaticBalanceController` wraps **only the WBC component**, not the entire torque pipeline.

### Integration Point

```python
# In simulate_hierarchical_controller.py or equivalent

# 1. Compute WBC torque (existing)
tau_wbc_raw = wbc_pipeline.compute(obs)

# 2. Apply static dynamics wrapper (new)
if enable_static_dynamics_wrapper:
    tau_wbc_wrapped, telemetry = static_balance_controller.wrap(
        tau_wbc_raw, 
        current_state
    )
    # Log telemetry
    log_wrapper_telemetry(telemetry)
else:
    tau_wbc_wrapped = tau_wbc_raw

# 3. Add other torque sources (existing)
tau_total_raw = (
    tau_wbc_wrapped
    + tau_posture
    + tau_leg_position
    + tau_hip_roll_centering
    + tau_wheel_balance
)

# 4. Apply clipping, smoothing, etc. (existing)
tau_final = apply_limits_and_smoothing(tau_total_raw)
```

### Command-line Flag

`--enable-static-dynamics-wrapper` (default: False for initial validation)

Allows A/B comparison:
- Old WBC: wrapper disabled
- Fixed WBC: wrapper enabled

### Wrapper Interface

```python
class StaticBalanceController:
    """Wrapper that cancels WBC static equilibrium bias.
    
    Computes static reference torques once at initialization,
    then removes WBC equilibrium bias at runtime to output
    only correction torques for deviations from equilibrium.
    """
    
    def __init__(
        self,
        mj_model,
        mj_data,
        wbc_pipeline,
        calibration_config,
    ):
        """Initialize with calibrated equilibrium references.
        
        Args:
            mj_model: MuJoCo model
            mj_data: MuJoCo data (will be copied, not mutated)
            wbc_pipeline: Existing WBC pipeline to wrap
            calibration_config: Config for calibrated initialization
        """
        # Compute references using copied data
        pass
    
    def wrap(
        self,
        tau_wbc_current: ndarray,
        current_state: dict,
    ) -> tuple[ndarray, dict]:
        """Wrap WBC torque to remove equilibrium bias.
        
        Args:
            tau_wbc_current: Current WBC output (10,)
            current_state: Current robot state for error metrics
            
        Returns:
            tau_wbc_wrapped: Bias-corrected WBC torque (10,)
            telemetry: Dict with all diagnostic values
        """
        pass
```

## Testing and Acceptance Criteria

### Unit Tests (`tests/test_static_balance_controller.py`)

**Test 1: Equilibrium Reference Computation**
```python
def test_equilibrium_reference_computation():
    """Verify references computed correctly at calibrated keyframe."""
    controller = StaticBalanceController(model, data, wbc, config)
    
    # tau_static_ref should match inverse dynamics
    assert np.allclose(
        controller.tau_static_ref[SUPPORT_JOINTS],
        expected_inverse_dynamics[SUPPORT_JOINTS],
        atol=0.5  # Nm
    )
    
    # tau_wbc_equilibrium should capture old WBC bias
    assert np.any(np.abs(controller.tau_wbc_equilibrium[SUPPORT_JOINTS]) > 5.0)
    
    # Equilibrium state stored correctly
    assert controller.equilibrium_state['com_z'] > 0.35
    assert abs(controller.equilibrium_state['pitch_x']) < 0.01
```

**Test 2: Bias Cancellation at Equilibrium**
```python
def test_bias_cancellation_at_equilibrium():
    """At calibrated equilibrium, correction should be near zero."""
    controller = StaticBalanceController(model, data, wbc, config)
    
    # Build zero-error observation
    obs_eq = build_zero_error_observation(controller.equilibrium_state)
    tau_wbc_current = wbc.compute(obs_eq)
    
    tau_wbc_wrapped, telemetry = controller.wrap(tau_wbc_current, equilibrium_state)
    
    # Correction should be near zero
    assert np.allclose(
        telemetry['tau_wbc_correction'][SUPPORT_JOINTS],
        0.0,
        atol=0.5  # Nm
    )
    
    # Wrapped output should match static reference
    assert np.allclose(
        tau_wbc_wrapped[SUPPORT_JOINTS],
        controller.tau_static_ref[SUPPORT_JOINTS],
        atol=1.0  # Nm
    )
    
    # Support bias removed should match diagnostic findings
    support_bias = telemetry['support_joint_bias_removed'][SUPPORT_JOINTS]
    assert np.any(np.abs(support_bias) > 5.0)  # Significant bias
```

**Test 3: Correction Response to Perturbations**
```python
def test_correction_response_to_perturbations():
    """Perturbations should produce stabilizing corrections."""
    controller = StaticBalanceController(model, data, wbc, config)
    
    # Test pitch perturbation
    state_pitch = perturb_pitch(equilibrium_state, +0.05)  # rad
    obs_pitch = build_observation(state_pitch)
    tau_wbc_current = wbc.compute(obs_pitch)
    tau_wbc_wrapped, telemetry = controller.wrap(tau_wbc_current, state_pitch)
    
    # Correction should be nonzero
    assert np.any(np.abs(telemetry['tau_wbc_correction'][SUPPORT_JOINTS]) > 1.0)
    
    # Correction should oppose pitch error (stabilizing)
    # Physical check: does correction reduce pitch acceleration?
    assert verify_stabilizing_correction(tau_wbc_wrapped, state_pitch, 'pitch')
    
    # Test roll perturbation
    state_roll = perturb_roll(equilibrium_state, +0.03)  # rad
    # ... similar checks
    
    # Test height perturbation
    state_height = perturb_height(equilibrium_state, -0.02)  # m
    # ... correction should increase upward force
```

**Test 4: Telemetry Completeness**
```python
def test_telemetry_completeness():
    """Verify all telemetry fields populated correctly."""
    controller = StaticBalanceController(model, data, wbc, config)
    
    tau_wbc_current = wbc.compute(obs)
    tau_wbc_wrapped, telemetry = controller.wrap(tau_wbc_current, current_state)
    
    # Check all required fields present
    required_fields = [
        'tau_static_ref', 'tau_wbc_equilibrium', 'tau_wbc_current',
        'tau_wbc_correction', 'tau_wbc_wrapped', 'support_joint_bias_removed',
        'posture_error_norm', 'com_height_error', 'pitch_x_error',
        'roll_y_error', 'com_velocity_norm', 'angular_velocity_norm',
    ]
    for field in required_fields:
        assert field in telemetry
        assert not np.any(np.isnan(telemetry[field]))
```

**Test 5: Reference Computation Does Not Mutate Live Data**
```python
def test_reference_computation_no_mutation():
    """Verify reference computation doesn't mutate live mj_data."""
    data_copy = copy.deepcopy(mj_data)
    
    controller = StaticBalanceController(model, mj_data, wbc, config)
    
    # Live data should be unchanged
    assert np.allclose(mj_data.qpos, data_copy.qpos)
    assert np.allclose(mj_data.qvel, data_copy.qvel)
    assert np.allclose(mj_data.qacc, data_copy.qacc)
```

### Integration Tests (`tests/test_static_balance_simulation.py`)

**Test 6: 100-Step Survival with Wrapper** (smoke test)
```python
def test_100_step_survival_with_wrapper():
    """Simulation with wrapper should survive ≥100 steps."""
    sim = setup_simulation(enable_static_dynamics_wrapper=True)
    
    for step in range(100):
        sim.step()
        
        # Should not terminate early
        if sim.terminated:
            pytest.fail(f"Terminated at step {step}")
    
    # Contact force should remain near weight
    # Use stable window (steps 20-100) to avoid transients
    contact_fz_mean = np.mean(sim.contact_fz_history[20:100])
    assert 75.0 < contact_fz_mean < 83.0  # 79N ± 5%
    
    # CoM height should remain stable (no continuous drift)
    com_z_std = np.std(sim.com_z_history[20:100])
    assert com_z_std < 0.01  # m
```

**Test 7: A/B Comparison (Old WBC vs Wrapped)**
```python
def test_ab_comparison_old_vs_wrapped():
    """Wrapped version should outperform old WBC."""
    # Run with wrapper disabled (old WBC)
    sim_old = setup_simulation(enable_static_dynamics_wrapper=False)
    survival_old = run_until_termination(sim_old, max_steps=100)
    
    # Run with wrapper enabled
    sim_wrapped = setup_simulation(enable_static_dynamics_wrapper=True)
    survival_wrapped = run_until_termination(sim_wrapped, max_steps=100)
    
    # Wrapped should survive longer
    assert survival_wrapped > survival_old
    
    # Wrapped should have better contact force
    contact_fz_old = np.mean(sim_old.contact_fz_history[10:survival_old])
    contact_fz_wrapped = np.mean(sim_wrapped.contact_fz_history[20:100])
    assert contact_fz_wrapped > contact_fz_old
    
    # Wrapped should have better CoM stability
    com_z_drift_old = np.max(sim_old.com_z_history) - np.min(sim_old.com_z_history)
    com_z_drift_wrapped = np.max(sim_wrapped.com_z_history[20:100]) - np.min(sim_wrapped.com_z_history[20:100])
    assert com_z_drift_wrapped < com_z_drift_old
```

**Test 8: Secondary Controller Audit**
```python
def test_secondary_controller_audit():
    """Check if posture/leg PD reintroduce static bias after WBC fix."""
    sim = setup_simulation(enable_static_dynamics_wrapper=True)
    
    # Run at equilibrium for 10 steps
    for step in range(10):
        sim.step()
        
        # Log torque components
        tau_wbc_wrapped = sim.telemetry['tau_wbc_wrapped'][SUPPORT_JOINTS]
        tau_posture = sim.telemetry['tau_posture'][SUPPORT_JOINTS]
        tau_leg_position = sim.telemetry['tau_leg_position'][SUPPORT_JOINTS]
        tau_total_raw = sim.telemetry['tau_total_raw'][SUPPORT_JOINTS]
        
        # Check if secondary controllers reintroduce bias
        secondary_bias = tau_total_raw - tau_wbc_wrapped
        
        # Flag if secondary bias is significant
        if np.any(np.abs(secondary_bias) > 5.0):
            print(f"WARNING: Secondary controllers reintroduce {secondary_bias} Nm bias")
            # This is diagnostic, not a failure - may need follow-up fix
```

### Regression Tests (`scripts/debug_static_support_parity_v2.py`)

**Test 9: Static Support Parity with Wrapper**
```python
def test_static_support_parity_with_wrapper():
    """Wrapped WBC should behave closer to inverse dynamics than old WBC."""
    
    # Case B: Current pipeline with wrapper disabled
    tau_old_wbc, contact_fz_old = run_case_b(enable_wrapper=False)
    
    # Case B': Current pipeline with wrapper enabled
    tau_wrapped, contact_fz_wrapped = run_case_b(enable_wrapper=True)
    
    # Case D: Inverse dynamics baseline
    tau_inverse_dynamics, contact_fz_id = run_case_d()
    
    # Wrapped should be closer to inverse dynamics than old WBC
    error_old = np.linalg.norm(tau_old_wbc[SUPPORT_JOINTS] - tau_inverse_dynamics[SUPPORT_JOINTS])
    error_wrapped = np.linalg.norm(tau_wrapped[SUPPORT_JOINTS] - tau_inverse_dynamics[SUPPORT_JOINTS])
    
    assert error_wrapped < error_old
    
    # Contact force should also be closer
    fz_error_old = abs(contact_fz_old - 79.5)
    fz_error_wrapped = abs(contact_fz_wrapped - 79.5)
    
    assert fz_error_wrapped < fz_error_old
```

## Acceptance Criteria

### ✅ Initialization Success

- References computed without errors using copied MuJoCo data
- `tau_static_ref[2,3,7,8]` matches inverse dynamics output (≈ 0 Nm per diagnostics)
- `tau_wbc_equilibrium[2,3,7,8]` captures measured old WBC static bias (significant, > 5 Nm)
- Equilibrium state stored correctly
- Initialization logging shows support bias removed

### ✅ Runtime Correctness

**At calibrated equilibrium**:
- `tau_wbc_correction[2,3,7,8]` < 0.5 Nm (near zero)
- `tau_wbc_wrapped[2,3,7,8]` ≈ `tau_static_ref[2,3,7,8]` (within 1.0 Nm)
- Old WBC bias is cancelled by subtraction

**With perturbations**:
- `tau_wbc_correction` ≠ 0 and physically stabilizing:
  - Pitch_x perturbation → correction opposes pitch error or reduces pitch acceleration
  - Roll_y perturbation → correction opposes roll error or reduces roll acceleration
  - Height drop → correction increases upward force or reduces downward acceleration

### ✅ Simulation Performance

- Survives ≥ 100 steps (vs ~14 steps with old WBC)
- Mean total contact force (steps 20-100) ≈ 79N ± 5% (vs 60-67N with old WBC)
- CoM height stable with no continuous drift (std < 0.01 m over steps 20-100)
- A/B comparison shows clear improvement: survival time, contact force, CoM stability

### ✅ Diagnostic Validation

- Static support parity: wrapped behavior closer to inverse dynamics than old WBC
  - Torque RMSE on support joints reduced
  - Contact force error reduced
- Telemetry shows bias cancellation working correctly
- Equilibrium error metrics logged and valid
- First 20 steps of wrapper telemetry logged when enabled

### ✅ Secondary Controller Audit

- `tau_wbc_wrapped` vs `tau_total_raw` comparison shows whether posture/leg PD reintroduce bias
- If secondary bias detected (> 5 Nm), flagged for follow-up fix (not blocking for this phase)

## Failure Decision Rules

If acceptance criteria not met, use these rules to diagnose:

**1. tau_wbc_wrapped correct but tau_total_raw wrong**
- **Diagnosis**: Secondary controller interference (posture/leg PD reintroduce bias)
- **Action**: Extend wrapper to pipeline-level or fix offending secondary controller

**2. tau_total_raw correct but tau_final wrong**
- **Diagnosis**: Clipping or rate limiting too aggressive
- **Action**: Adjust actuator limits or rate limiting parameters

**3. tau_final correct but contact Fz still low**
- **Diagnosis**: Contact solver, contact model, or wheel slip issue
- **Action**: Investigate contact parameters, friction model, or Jacobian mapping

**4. All torques correct but robot still falls**
- **Diagnosis**: Fixed static reference invalid far from equilibrium, or missing contact recovery
- **Action**: Add online inverse dynamics or implement contact recovery logic (out of scope for minimal fix)

**5. tau_wbc_correction not near zero at equilibrium**
- **Diagnosis**: Equilibrium reference computation incorrect or observation not truly zero-error
- **Action**: Debug reference computation, verify qvel=0, qacc=0, com_vel=0, capture point=0

**6. Perturbation corrections not stabilizing**
- **Diagnosis**: WBC gains wrong or correction sign error
- **Action**: Debug WBC pipeline, verify Jacobian signs, check gain values

## Out of Scope

The following are explicitly **not** part of this minimal static fix:

- **Online inverse dynamics computation**: Fixed reference only, computed once at initialization
- **Full wrench-level baseline/correction split**: Wrapper approach, not refactoring WBC internals
- **QP force distribution**: Keep existing simple force distributor
- **Contact recovery logic**: Assumes robot stays near equilibrium
- **Trajectory planning**: Static standing only, no dynamic transitions
- **Gain tuning**: Use existing gains, only fix static bias
- **Equilibrium tracking**: Fixed reference, no adaptation to configuration changes

These will be addressed in follow-on phases once static standing is proven stable.

## Success Metrics

**Primary**: Robot survives ≥ 100 steps with contact force ≈ 79N ± 5%

**Secondary**:
- Torque RMSE on support joints reduced by ≥ 50% vs old WBC
- Contact force error reduced by ≥ 50% vs old WBC (60-67N → 75-83N)
- CoM height drift reduced by ≥ 50% vs old WBC

**Diagnostic**:
- Static support parity shows wrapped behavior matches inverse dynamics within 20%
- Telemetry confirms bias cancellation working (tau_wbc_correction ≈ 0 at equilibrium)
- A/B comparison shows clear improvement in all metrics

## Next Steps After This Fix

Once this minimal wrapper proves the static-bias hypothesis:

1. **If successful**: Refactor into proper wrench-level baseline/correction split (cleaner architecture)
2. **If secondary bias detected**: Extend wrapper to pipeline-level or fix posture/leg PD
3. **If contact issues remain**: Investigate contact solver, friction model, or Jacobian mapping
4. **If robot falls far from equilibrium**: Add online inverse dynamics or contact recovery

The wrapper is a proof-of-concept layer. The final architecture should implement the same principle (separate baseline from corrections) at the wrench computation level, not as a post-hoc wrapper.
