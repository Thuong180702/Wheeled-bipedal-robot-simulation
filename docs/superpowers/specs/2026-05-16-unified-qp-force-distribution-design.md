# Unified QP Force Distribution for Wheeled Biped

**Date:** 2026-05-16  
**Status:** Approved  
**Control Frequency:** 100Hz  
**Implementation Time:** 4 days

## Problem Statement

The current hierarchical WBC force distributor assumes humanoid morphology with laterally-separated feet. The wheeled biped has wheels with **zero lateral separation** (0.27m front-to-back, 0.0m side-to-side), causing division by near-zero and producing massive forces (6,283N on right wheel), leading to immediate robot failure.

The robot's morphology means:
- **Wheels cannot generate roll moments** (no lateral separation)
- **Wheels cannot generate lateral forces** (wheels roll forward/backward only)
- **Hip roll joints must handle roll stabilization**

## Architecture Overview

### High-Level Flow

```
Centroidal objectives (roll, CoM, CP, height)
    ↓
Centroidal wrench computer (existing)
    ↓
Desired wrench: [Fx, Fy, Fz, Mx, My, Mz]
    ↓
NEW: Unified QP Force Distributor
    ↓
Decision variables: [f_left (3D), f_right (3D), tau_hip_roll_L, tau_hip_roll_R]
    ↓
Contact Jacobian maps to joint torques (10D)
```

### Key Components

1. **ContactJacobianComputer** (refine existing)
   - Computes J_left, J_right mapping wheel contact forces → joint torques
   - Computes J_hip_roll mapping hip roll torques → joint torques

2. **UnifiedForceDistributor** (new)
   - Formulates QP: minimize cost subject to wrench equality + contact inequality constraints
   - Uses OSQP-JAX for real-time solving at 100Hz
   - Returns: wheel forces + hip roll torques

3. **Integration layer** (modify existing)
   - Replace current force distributor with unified version
   - Keep centroidal wrench computer unchanged

## QP Formulation

### Decision Variables (8D)

```
x = [f_left_x, f_left_y, f_left_z,     # Left wheel contact force (3D)
     f_right_x, f_right_y, f_right_z,   # Right wheel contact force (3D)
     tau_hip_roll_L,                     # Left hip roll torque
     tau_hip_roll_R]                     # Right hip roll torque
```

### Equality Constraint (Wrench Matching)

```
A_wrench @ x = b_desired_wrench

where:
- A_wrench (6×8): maps decision variables to centroidal wrench [Fx, Fy, Fz, Mx, My, Mz]
- b_desired_wrench (6×1): desired wrench from centroidal controller
```

### Inequality Constraints

```
1. Contact forces compressive: f_left_z ≥ 0, f_right_z ≥ 0
2. Torque limits: -tau_max ≤ tau_hip_roll_L/R ≤ tau_max
3. (Optional) Friction cone: sqrt(fx² + fy²) ≤ μ * fz for each wheel
```

### Cost Function (Weighted Quadratic)

```
minimize: x^T Q x + c^T x

where Q is diagonal with weights:
- w_force: penalize large contact forces (effort minimization)
- w_torque: penalize large hip roll torques (effort minimization)
- w_smoothness: penalize deviation from previous solution (temporal smoothness)
```

### Solver

OSQP-JAX with warm-starting (use previous solution as initial guess)

## Contact Jacobian Computation

### Building A_wrench Matrix (6×8)

The matrix maps decision variables to centroidal wrench:

```
[Fx]     [J_left_x^T    J_right_x^T    0  0]   [f_left]
[Fy]     [J_left_y^T    J_right_y^T    0  0]   [f_right]
[Fz]  =  [J_left_z^T    J_right_z^T    0  0] @ [tau_hip_roll_L]
[Mx]     [J_roll_L      J_roll_R       1  1]   [tau_hip_roll_R]
[My]     [J_pitch_L     J_pitch_R      0  0]
[Mz]     [J_yaw_L       J_yaw_R        0  0]
```

### Key Components

**1. Wheel Force Jacobians (J_left, J_right):**
- Computed via `mujoco.mj_jacBody()` for each wheel body
- Extract rows for [x, y, z] translation
- Extract columns for joint DOFs (skip floating base)
- Result: (3×10) matrix per wheel

**2. Hip Roll Moment Arm:**
- Hip roll torques directly contribute to roll moment (Mx)
- Coefficient = 1.0 (direct mapping)
- No contribution to other wrench components

**3. Moment Computation from Forces:**
- Mx (roll): `r_y * Fz - r_z * Fy` for each wheel
- My (pitch): `r_z * Fx - r_x * Fz` for each wheel
- Mz (yaw): `r_x * Fy - r_y * Fx` for each wheel
- Where r = wheel position relative to CoM

### Implementation Approach

- Pre-compute wheel positions relative to CoM each timestep
- Build A_wrench matrix dynamically (depends on current configuration)
- Cache Jacobian computation results within a timestep

## OSQP-JAX Integration (100Hz)

### Timing Constraint

- 100Hz control → 10ms per solve (half the budget of 50Hz)
- Requires tighter solver settings and aggressive warm-starting

### Solver Setup

```python
import osqp_jax

# Problem dimensions
n_vars = 8  # [f_left (3), f_right (3), tau_hip_roll (2)]
n_eq = 6    # wrench matching constraints
n_ineq = 4  # contact compressive + torque limits

# Initialize solver (done once at controller init)
solver = osqp_jax.OSQP()
```

### Per-Timestep Solve

```python
# Build QP matrices (configuration-dependent)
P = build_cost_matrix(weights)  # (8×8) quadratic cost
q = build_linear_cost(prev_solution, weights)  # (8,) for smoothness
A = build_constraint_matrix(J_left, J_right, wheel_positions)  # (10×8) [eq + ineq]
l = build_lower_bounds(desired_wrench, torque_limits)  # (10,)
u = build_upper_bounds(desired_wrench, torque_limits)  # (10,)

# Solve with warm start
solution = solver.solve(P, q, A, l, u, x_init=prev_solution)

# Extract results
f_left = solution.x[0:3]
f_right = solution.x[3:6]
tau_hip_roll = solution.x[6:8]
```

### Warm-Starting Strategy

- Use previous timestep's solution as initial guess
- At 100Hz, state changes less between timesteps → better warm start
- Expected iterations: 2-4 (vs 2-5 at 50Hz)

### Solver Settings (100Hz Optimized)

```python
max_iter: 10       # Reduced from 20 (need faster convergence)
eps_abs: 1e-3      # Relaxed from 1e-4 (trade accuracy for speed)
eps_rel: 1e-3      # Relaxed from 1e-4
polish: False      # Disable refinement step (saves ~2-3ms)
```

**Tradeoff:** Slightly relaxed accuracy (1e-3 vs 1e-4 tolerance) for 2x faster control rate.

## Cost Function Weights

### Quadratic Cost Matrix Q (8×8 Diagonal)

```python
Q = diag([
    w_force, w_force, w_force,  # Left wheel forces [x, y, z]
    w_force, w_force, w_force,  # Right wheel forces [x, y, z]
    w_torque, w_torque          # Hip roll torques [L, R]
])
```

### Linear Cost for Smoothness

```python
q = -2 * w_smoothness * Q @ x_prev
# Penalizes deviation from previous solution
```

### Recommended Initial Weights

- `w_force = 0.01` - Small penalty on contact forces (effort minimization)
- `w_torque = 0.1` - Larger penalty on hip roll torques (prefer wheels for balance)
- `w_smoothness = 0.5` - Moderate smoothness penalty (avoid chattering)

### Tuning Strategy

- Start with these defaults
- If robot is too aggressive: increase w_force/w_torque
- If robot is too sluggish: decrease w_smoothness
- If hip rolls saturate: decrease w_torque (allow more hip roll usage)
- If wheels slip: increase w_force (reduce commanded forces)

## Integration with Existing Controller

### Current Architecture

```
CentroidalBalanceController → compute_centroidal_wbc_torque() → returns tau (10D)
MomentumCoordinator → compute_momentum_coordinator_torque() → returns tau (10D)
PostureRegularizer → compute_posture_regularizer_torque() → returns tau (10D)
```

### New Architecture

```
CentroidalBalanceController → compute_desired_wrench() → returns wrench (6D)
                                      ↓
                          UnifiedForceDistributor (NEW)
                                      ↓
                    [f_left, f_right, tau_hip_roll] (8D)
                                      ↓
                          ContactJacobian.map_to_torques()
                                      ↓
                                  tau_wbc (10D)

MomentumCoordinator → (unchanged) → tau_momentum (10D)
PostureRegularizer → (unchanged) → tau_posture (10D)

Final: tau_total = tau_wbc + tau_momentum + tau_posture
```

### Key Changes

1. **CentroidalBalanceController** refactored to output wrench instead of torques
2. **UnifiedForceDistributor** added as new component
3. **ContactJacobian** extended to handle hip roll torques
4. **Momentum/Posture layers** unchanged (still output joint torques directly)

### Migration Strategy

- Keep old `compute_centroidal_wbc_torque()` temporarily for comparison
- Add new `compute_desired_wrench()` method
- Switch simulation script to use new path
- Remove old method after validation

## Error Handling and Fallback Strategy

### Solver Failure Modes

1. **No convergence within max_iter** - solver doesn't reach tolerance in 10 iterations
2. **Infeasible problem** - constraints are contradictory (e.g., desired wrench physically impossible)
3. **Numerical issues** - ill-conditioned matrices, NaN/Inf values

### Fallback Strategy

```python
# Primary: Use OSQP solution if converged
if solution.status == "solved":
    return solution.x

# Fallback 1: Use previous solution (graceful degradation)
elif solution.status == "max_iter_reached":
    log_warning("QP max iterations, using previous solution")
    return prev_solution

# Fallback 2: Emergency zero-wrench (robot freefalls but doesn't explode)
else:
    log_error(f"QP failed: {solution.status}")
    return zero_solution  # [0, 0, mg/2, 0, 0, mg/2, 0, 0]
```

### Telemetry for Debugging

- Log solve time, iteration count, status every timestep
- Track convergence rate over time
- Alert if fallback is used more than 5% of timesteps
- Record QP matrices when solver fails (for offline debugging)

### Validation Checks

- Verify wrench error: `||A @ x - b|| < threshold`
- Check contact forces are compressive: `fz >= -epsilon`
- Verify torques within limits: `|tau| < tau_max`

## Testing and Validation Strategy

### Unit Tests

```python
# test_unified_force_distributor.py
- test_wrench_matching_accuracy() - verify A @ x ≈ b within tolerance
- test_contact_forces_compressive() - verify fz >= 0 for both wheels
- test_torque_limits_respected() - verify |tau| <= tau_max
- test_cost_function_minimization() - verify solution minimizes effort
- test_warm_start_convergence() - verify faster convergence with warm start
- test_infeasible_wrench_fallback() - verify graceful degradation
```

### Integration Tests

```python
# test_hierarchical_controller_integration.py
- test_100hz_timing() - verify solve completes within 10ms budget
- test_static_balance() - robot stands still without falling
- test_roll_disturbance_rejection() - apply roll moment, verify recovery
- test_pitch_disturbance_rejection() - apply pitch moment, verify recovery
- test_smooth_transitions() - verify no torque discontinuities
```

### Validation Metrics

- **Solve time:** mean, p95, p99 (target: <10ms)
- **Convergence rate:** % of timesteps that converge within max_iter
- **Wrench error:** ||A @ x - b|| (target: <0.01 N or Nm)
- **Contact force violations:** % of timesteps with fz < 0
- **Fallback rate:** % of timesteps using fallback solution

### Comparison Baseline

- Run old controller (direct torque output) vs new controller (QP-based)
- Compare: survival time, roll/pitch RMS, energy consumption
- Target: new controller >= old controller performance

## Implementation Phases

### Phase 1: Contact Jacobian Foundation (Day 1)

**Tasks:**
- Extend `ContactJacobian` to compute wheel Jacobians + hip roll mapping
- Implement `build_wrench_matrix()` that constructs A_wrench from Jacobians
- Unit tests: verify Jacobian dimensions, wrench mapping accuracy

**Deliverable:** Working Jacobian computation, tested in isolation

### Phase 2: OSQP-JAX Integration (Day 2)

**Tasks:**
- Install osqp-jax dependency
- Implement `UnifiedForceDistributor` class with QP formulation
- Build cost matrices (P, q) and constraint matrices (A, l, u)
- Unit tests: verify QP solves simple cases, respects constraints

**Deliverable:** Working QP solver, tested with synthetic inputs

### Phase 3: Centroidal Controller Refactor (Day 3)

**Tasks:**
- Refactor `CentroidalBalanceController` to output wrench instead of torques
- Add `compute_desired_wrench()` method
- Keep old `compute_centroidal_wbc_torque()` for comparison
- Unit tests: verify wrench computation matches old torque output

**Deliverable:** Dual-mode controller (old path + new path)

### Phase 4: Integration and Validation (Day 4)

**Tasks:**
- Wire new force distributor into simulation script
- Add telemetry logging (solve time, iterations, convergence)
- Run side-by-side comparison: old vs new controller
- Tune QP weights based on observed behavior

**Deliverable:** Working 100Hz controller, performance validated

## Success Criteria

1. **Functional:** Robot balances for >5 seconds without falling
2. **Performance:** QP solves in <10ms at p95
3. **Correctness:** Wrench error <0.01 N/Nm, contact forces compressive
4. **Robustness:** Fallback rate <5%, graceful degradation when solver fails
5. **Comparison:** New controller >= old controller survival time

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| OSQP doesn't converge in 10ms | Relax tolerances, reduce max_iter, use previous solution fallback |
| Infeasible wrench requests | Add soft constraints, validate wrench before QP, emergency fallback |
| Jacobian computation too slow | Cache results within timestep, pre-compute static components |
| QP weights hard to tune | Start with conservative defaults, add auto-tuning based on telemetry |
| Integration breaks existing controller | Keep dual-mode during migration, extensive A/B testing |

## Future Extensions

- Add friction cone constraints (pyramid approximation)
- Implement ZMP constraints for dynamic walking
- Extend to 4-contact (hands + wheels) for manipulation tasks
- Add torque rate limits for smoother control
- Implement predictive QP (MPC-style) for anticipatory control
