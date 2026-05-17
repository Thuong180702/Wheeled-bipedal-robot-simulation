# Upstream Controller Debug Plan

**Date**: 2026-05-17  
**Context**: Unified QP force distribution is working (produces non-zero torques, all tests pass), but robot fails at 0.32s with rapid divergence (pitch -36°, roll +45°). QP wrench tracking error is 35.46 N/Nm (3546x over target).

---

## Problem Statement

The QP solver is working correctly but cannot track the desired wrench:
- **QP performance**: Produces valid compressive forces, no fallback to zeros
- **Wrench tracking**: 35.46 N/Nm error (target: <0.01 N/Nm)
- **Robot behavior**: Rapid divergence despite control effort
- **Survival time**: 0.32s (worse than 0.52s with hard constraints producing 0.0 Nm)

**Hypothesis**: The desired wrench from upstream controllers is either:
1. Infeasible given contact geometry constraints
2. Destabilizing due to incorrect state estimation
3. Conflicting with momentum/posture controllers in hierarchical fusion

---

## Debug Tasks

### Task 1: Analyze Contact Geometry

**Goal**: Verify if the wheeled biped morphology makes the desired wrench feasible.

**Key facts**:
- Zero lateral wheel separation (0.0m side-to-side)
- 0.27m front-to-back separation
- Wheels are the only contact points
- Hip roll torques provide additional roll moment authority

**Checks**:
1. Compute wheel positions relative to CoM from telemetry
2. Build wrench matrix A for first timestep
3. Check matrix conditioning (condition number, rank, null space)
4. Identify which wrench components are controllable
5. Compare desired wrench to feasible wrench subspace

**Expected findings**:
- Zero lateral separation may limit lateral force/moment authority
- Roll moment may be primarily controlled by hip roll torques
- Some wrench components may be weakly controllable

---

### Task 2: Validate Centroidal Wrench Computation

**Goal**: Check if desired wrenches are physically reasonable for standing balance.

**Checks**:
1. Extract desired wrench components from telemetry
2. Verify vertical force Fz ≈ 147N (gravity compensation for 15kg robot)
3. Check horizontal forces Fx, Fy are reasonable (<30N for standing)
4. Check roll moment Mx is reasonable (<20Nm)
5. Analyze wrench evolution over time (should be smooth, not oscillating)

**Expected findings**:
- Desired wrench may be too aggressive (large horizontal forces/moments)
- Wrench may be oscillating due to poor gain tuning
- Height tracking may be producing large vertical force errors

---

### Task 3: Validate State Estimation

**Goal**: Check if CoM, capture point, and contact state are accurate.

**Checks**:
1. Compare estimated CoM position to MuJoCo ground truth
2. Check capture point computation (should be CoM + vel/omega_0)
3. Verify contact state (both wheels should be in contact)
4. Check for NaNs or discontinuities in state estimates

**Expected findings**:
- State estimation may be using simulator-clean signals (not hardware-ready)
- CoM velocity may be noisy or incorrect
- Capture point may be diverging due to incorrect omega_0

---

### Task 4: Check Hierarchical Fusion

**Goal**: Ensure WBC + momentum + posture controllers don't conflict.

**Checks**:
1. Extract individual controller torques from telemetry
2. Check if WBC + momentum + posture sum exceeds actuator limits
3. Verify authority budgets are respected (WBC: 18Nm, momentum: 6Nm, posture: 6Nm)
4. Check for sign conflicts (controllers fighting each other)

**Expected findings**:
- WBC may be saturating its authority budget
- Momentum controller may be saturated (6.0/6.0 Nm in telemetry)
- Controllers may be producing conflicting commands

---

### Task 5: Tune Controller Gains

**Goal**: Adjust centroidal controller gains for stability.

**Current gains** (from `CentroidalBalanceConfig`):
```python
k_roll = 20.0
k_roll_rate = 4.0
k_com_lateral = 15.0
k_com_lateral_damping = 3.0
k_com_sagittal = 10.0
k_com_sagittal_damping = 2.0
k_cp_lateral = 25.0
k_cp_sagittal = 20.0
k_height = 5.0
```

**Tuning strategy**:
1. Start with reduced gains (50% of current)
2. Increase damping gains first (stabilize before tracking)
3. Tune roll gains separately from CoM/CP gains
4. Use telemetry to verify wrench error decreases

---

## Implementation Order

1. **Task 1** (Contact Geometry) - Identifies fundamental feasibility limits
2. **Task 2** (Wrench Validation) - Checks if desired commands are reasonable
3. **Task 3** (State Estimation) - Validates input signals to controllers
4. **Task 4** (Hierarchical Fusion) - Checks for controller conflicts
5. **Task 5** (Gain Tuning) - Adjusts controller behavior

---

## Success Criteria

- Wrench tracking error <1.0 N/Nm (100x improvement)
- Robot survives >2.0s (6x improvement)
- Pitch/roll divergence rate <50°/s (vs current 478°/s pitch, 230°/s roll)
- QP solve time <100ms (vs current 1136ms)

---

## Files to Modify

- `wheeled_biped/controllers/centroidal_wrench_computer.py` - Wrench computation
- `wheeled_biped/controllers/centroidal_state_estimator.py` - State estimation
- `wheeled_biped/controllers/centroidal_balance_controller.py` - Controller gains
- `scripts/simulate_hierarchical_controller.py` - Add diagnostic logging
- `scripts/analyze_contact_geometry.py` - New analysis script

---

## Next Steps

Start with Task 1: Analyze contact geometry to understand fundamental feasibility limits.
