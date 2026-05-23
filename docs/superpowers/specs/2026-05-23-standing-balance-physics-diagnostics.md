# Standing Balance Physics Diagnostics Spec

**Date**: 2026-05-23  
**Status**: Draft  
**Scope**: Diagnostic phases 0-3 to identify root cause of force gap

## Problem Statement

The wheeled biped robot is falling after 14-15 steps with termination reason `height_too_low`. The dominant failure mode is insufficient vertical contact force:

- **Desired vertical force**: ~79 N (robot weight)
- **Actual contact force**: 60-67 N
- **Missing support**: 15-20 N (19-25% deficit)
- **Consequence**: CoM velocity remains negative and grows in magnitude, leading to collapse

**Goal**: Identify exactly where and why the desired 79N vertical force becomes 60-67N actual contact force through systematic diagnostics.

## Physical Invariants and Conventions

### Coordinate System (MuJoCo XML)
- **X-axis**: Lateral (left/right), left is +X
- **Y-axis**: Sagittal (front/back), front is -Y
- **Z-axis**: Vertical (up/down), up is +Z

### Orientation Convention
- **pitch_x**: Rotation about X-axis (forward/backward tilt)
- **roll_y**: Rotation about Y-axis (left/right tilt)
- **yaw_z**: Rotation about Z-axis (heading)

### Force and Torque Sign Conventions
- **Contact forces**: Ground reaction forces on robot, Fz > 0 means upward support
- **Jacobian mapping**: `tau = J^T f` where f is ground reaction force
- **Actuator torques**: Positive torque direction per joint axis definition in XML
- **Wrench convention**: `[Fx, Fy, Fz, Mx, My, Mz]` in world frame about CoM

### Control Semantics
- **Actuator control**: `mj_data.ctrl` is treated as torque-level command in `simulate_hierarchical_controller.py`
- **No low-level PID assumption**: Diagnostics operate at torque level unless explicitly auditing a separate legacy path
- **WBC output**: Joint torques applied directly to `mj_data.ctrl`
- **Support-critical joints**: `[2, 3, 7, 8]` = `[l_hip_pitch, l_knee, r_hip_pitch, r_knee]`

### Static Equilibrium Requirements
For standing balance at calibrated height (root_z ≈ 0.536 m, CoM ≈ 0.40 m):
- Total vertical contact force ≈ robot weight (79.5 N)
- Both wheels in contact with floor (contact.dist ≈ -0.5 mm)
- CoM velocity ≈ 0
- Pitch_x ≈ 0, Roll_y ≈ 0
- Joint torques sufficient to counteract gravity on leg segments

## Diagnostic Phases

### Phase 0: Instrumentation & Truth Diagnostics

**Purpose**: Make the force pipeline transparent by adding comprehensive telemetry at every stage.

**Telemetry to Add**:

1. **Force tracking at each stage**:
   - `desired_fz_total`: WBC wrench computer output
   - `distributed_fz_left`, `distributed_fz_right`: Force distributor output
   - `f_left_z_actual`, `f_right_z_actual`: Actual contact forces from MuJoCo
   - `fz_error = desired_fz_total - (f_left_z_actual + f_right_z_actual)`

2. **Torque tracking per joint**:
   - `tau_wbc[j]`: WBC output torque for joint j
   - `tau_wbc_scaled[j]`: After joint-specific scaling
   - `tau_total_raw[j]`: Sum of all torque sources before clipping
   - `tau_total_clipped[j]`: After actuator limit clipping
   - `tau_smooth[j]`: After rate limiting
   - `tau_applied[j]`: Final control signal to MuJoCo

3. **Support joint diagnostics** (indices [2,3,7,8]):
   - `tau_ideal[j]`: Torque from `J^T f` with f = weight/2 per wheel
   - `support_ratio[j] = tau_applied[j] / tau_ideal[j]`
   - `support_deficit[j] = tau_ideal[j] - tau_applied[j]`

3b. **Cancellation diagnostics for support joints [2,3,7,8]**:
   - `tau_wbc[j]`: WBC contribution
   - `tau_leg_position[j]`: Leg position controller contribution
   - `tau_posture[j]`: Posture regularizer contribution
   - `tau_hip_roll_centering[j]`: Hip roll centering contribution
   - `tau_wheel_balance[j]`: Wheel balance contribution
   - `tau_total_raw[j]`: Sum before clipping
   - Classification: For each secondary term, determine if it assists (same sign as tau_wbc) or opposes (opposite sign)

4. **Actuator saturation tracking**:
   - `saturation_flags[j]`: Boolean per joint, true if |tau| >= limit
   - `rate_limit_flags[j]`: Boolean per joint, true if rate limited
   - `saturation_rate`: Fraction of joints saturated

5. **Contact state**:
   - `contact_count`: Number of active contacts
   - `contact_distances`: Per-contact penetration depth
   - `wheel_slip_velocity`: Tangential velocity at contact point
   - `contact_normal_alignment`: Angle between contact normal and Z-axis

6. **Acceleration diagnostics**:
   - `qacc`: Joint accelerations from MuJoCo (if available)
   - `com_acc_z`: Vertical CoM acceleration
   - `expected_acc_z = (fz_actual - weight) / mass`

**Deliverable**: `scripts/debug_force_gap.py`

**Script behavior**:
- Load robot at calibrated standing keyframe
- Run one control cycle (compute WBC torque, apply to robot, step physics once)
- Print force audit trail showing values at each stage:
  ```
  [FORCE AUDIT TRAIL - Step 0]
  1. WBC Wrench Computer:
     desired_fz_total = 79.5 N
  
  2. Force Distributor:
     distributed_fz_left = 39.75 N
     distributed_fz_right = 39.75 N
     distributed_fz_total = 79.5 N
  
  3. Contact Jacobian Mapping:
     tau_from_jacobian[2,3,7,8] = [X, X, X, X] Nm
  
  4. Torque Pipeline:
     tau_wbc[2,3,7,8] = [X, X, X, X] Nm
     tau_wbc_scaled[2,3,7,8] = [X, X, X, X] Nm
     tau_total_raw[2,3,7,8] = [X, X, X, X] Nm
     tau_total_clipped[2,3,7,8] = [X, X, X, X] Nm
     tau_smooth[2,3,7,8] = [X, X, X, X] Nm
  
  5. MuJoCo Contact Forces (after mj_step):
     f_left_z_actual = 60.2 N
     f_right_z_actual = 60.5 N
     f_total_z_actual = 60.7 N
  
  6. Force Gap Analysis:
     fz_error = 79.5 - 60.7 = 18.8 N (23.6% deficit)
     Stage with largest loss: [IDENTIFIED STAGE]
  ```

**Acceptance Criteria**:
- Script runs without errors
- All telemetry values are populated (no NaN/None)
- Force gap is reproduced (actual < desired by 15-20N)
- Audit trail clearly shows which stage loses the most force

---

### Phase 1: Static Support Parity Test

**Purpose**: Test whether the controller can hold the robot at the calibrated standing keyframe under different torque sources.

**Test Cases**:

**Case A: Zero Control (Gravity Only)**
- Set `ctrl[:] = 0`
- Step physics 1, 5, 10, 20 times
- Measure: contact Fz, com_z, com_vz, joint qacc
- Expected: Robot collapses, contact force drops to near zero
- Purpose: Baseline showing gravity alone causes collapse

**Case B: WBC Desired Torque (Current Pipeline)**
- Compute WBC torque via current pipeline
- Apply final `tau_smooth` to robot
- Step physics 1, 5, 10, 20 times
- Measure: contact Fz, com_z, com_vz, joint qacc
- Expected: Current behavior (60-67N contact force, slow collapse)
- Purpose: Reproduce current failure mode

**Case C: Ideal J^T f (Theoretical Perfect Support)**
- Compute `f_left = f_right = [0, 0, weight/2]`
- Compute `tau_ideal = J_left^T @ f_left + J_right^T @ f_right`
- Apply `tau_ideal` directly (no clipping, no rate limiting)
- Step physics 1, 5, 10, 20 times
- Measure: contact Fz, com_z, com_vz, joint qacc
- Expected: Contact force should improve vs Case A, but may not hold posture perfectly (no joint-space holding torques)
- Purpose: Verify Jacobian mapping produces vertical support; compare against Case D (inverse dynamics) for full static equilibrium

**Case D: Inverse Dynamics (MuJoCo's Answer)**
- Set `qvel[:] = 0` and `qacc[:] = 0` before calling `mj_inverse`
- Call `mj_inverse(model, data)` to compute required holding torques
- Extract and report:
  - `tau_id = qfrc_inverse[6:16]` (joint torques)
  - `qfrc_bias[6:16]` (Coriolis + gravity terms)
  - `qfrc_constraint[6:16]` (constraint forces, if available)
- Apply `tau_id` directly
- Step physics 1, 5, 10, 20 times
- Measure: contact Fz, com_z, com_vz, joint qacc
- Expected: Contact force ≈ weight if contact-constrained static support is feasible
- Purpose: Ground truth for required torques; treat as diagnostic unless verified

**Case E: Final Pipeline Torque (With All Modifications)**
- Compute WBC torque
- Apply all scaling, clipping, rate limiting
- Apply final `tau_smooth`
- Step physics 1, 5, 10, 20 times
- Measure: contact Fz, com_z, com_vz, joint qacc
- Expected: Same as Case B (current behavior)
- Purpose: Isolate effect of pipeline modifications

**Deliverable**: `scripts/debug_static_support_parity.py`

**Script behavior**:
- Load robot at calibrated standing keyframe
- Run all 5 test cases
- For each case, print table:
  ```
  Case A: Zero Control
  Steps | contact_fz | com_z  | com_vz  | max_qacc
  ------|------------|--------|---------|----------
  1     | 0.0 N      | 0.400m | -0.01   | -2.5
  5     | 0.0 N      | 0.395m | -0.05   | -3.0
  10    | 0.0 N      | 0.385m | -0.10   | -3.5
  20    | 0.0 N      | 0.360m | -0.20   | -4.0
  ```
- Compare Cases C and D (should both work) vs Case B (current failure)
- Identify which pipeline stage causes Case B to differ from Cases C/D

**Acceptance Criteria**:
- Case D (inverse dynamics) produces contact force ≈ 79N and stable standing (if feasible)
- Case C (ideal J^T f) shows improved vertical support vs Case A, compared against Case D
- Case B reproduces current failure (60-67N, slow collapse)
- Comparison clearly identifies which stage loses force (see Decision Rules section below)

---

### Phase 2: Actuator Sign & Authority Validation

**Purpose**: Verify every actuator produces force in the expected direction with expected magnitude.

**Tests**:

**Test 2.1: Individual Actuator Sign Test**
- For each joint j in [0..9]:
  - Load calibrated standing keyframe
  - Apply `ctrl[j] = +1.0 Nm`, all others = 0
  - Step physics once
  - Measure `qacc[j]` and sign
  - Apply `ctrl[j] = -1.0 Nm`, all others = 0
  - Step physics once
  - Measure `qacc[j]` and sign
  - Verify: `sign(qacc[+1.0]) = -sign(qacc[-1.0])`

**Test 2.2: Support Joint Authority Test**
- For each support joint j in [2, 3, 7, 8]:
  - Load calibrated standing keyframe
  - Apply `ctrl[j] = +10.0 Nm`, all others = 0
  - Step physics once
  - Measure: contact Fz change, joint qacc[j]
  - Apply `ctrl[j] = -10.0 Nm`, all others = 0
  - Step physics once
  - Measure: contact Fz change, joint qacc[j]
  - Record which sign (+τ or -τ) increases Fz or reduces downward qacc
  - Do NOT assume positive torque increases support - test both directions

**Test 2.3: Left/Right Symmetry Test**
- For each joint pair (left, right):
  - Load calibrated standing keyframe
  - Apply `ctrl[left] = +5.0 Nm`, all others = 0
  - Measure `qacc[left]`
  - Apply `ctrl[right] = +5.0 Nm`, all others = 0
  - Measure `qacc[right]`
  - Verify: `|qacc[left]| ≈ |qacc[right]|` (within 10%)

**Test 2.4: Posture Controller Interference Test**
- Load calibrated standing keyframe
- Compute WBC torque (should be near zero at equilibrium)
- Compute posture controller torque
- Compute leg position controller torque
- Check for support joints [2,3,7,8]:
  - If `sign(tau_posture[j]) ≠ sign(tau_wbc[j])` → CONFLICT
  - If `|tau_posture[j]| > 0.5 * |tau_wbc[j]|` → SIGNIFICANT INTERFERENCE

**Deliverable**: `tests/test_actuator_signs.py`

**Test structure**:
```python
def test_actuator_sign_consistency():
    """Verify each actuator produces expected acceleration direction."""
    # Test 2.1 implementation
    
def test_support_joint_authority():
    """Verify support joints increase contact force when commanded."""
    # Test 2.2 implementation
    
def test_left_right_symmetry():
    """Verify left/right joint pairs have symmetric response."""
    # Test 2.3 implementation
    
def test_posture_controller_interference():
    """Verify posture controller doesn't fight WBC on support joints."""
    # Test 2.4 implementation
```

**Acceptance Criteria**:
- All actuator signs are consistent (no sign errors)
- Support joints produce expected contact force changes
- Left/right symmetry is within 10%
- Posture controller does not significantly oppose WBC on support joints
- Any bugs found have regression tests added

---

### Phase 3: Inverse Dynamics Baseline

**Purpose**: Establish ground truth for what torques are physically required to hold the standing posture.

**Analysis**:

**Step 3.1: Compute Required Holding Torques**
- Load calibrated standing keyframe
- Call `mj_inverse(model, data)` to compute inverse dynamics
- Extract `tau_required = qfrc_inverse[6:16]`
- Extract `qfrc_bias` (Coriolis + gravity terms)
- Extract `qfrc_gravcomp` if available

**Step 3.2: Compare Against Controller Torques**
- Compute WBC torque: `tau_wbc`
- Compute posture torque: `tau_posture`
- Compute leg position torque: `tau_leg_position`
- Compute hip roll centering: `tau_hip_roll_centering`
- Compute wheel balance: `tau_wheel_balance`
- Compute total: `tau_total = tau_wbc + tau_posture + tau_leg_position + tau_hip_roll_centering + tau_wheel_balance`
- Compute final: `tau_final` (after clipping, rate limiting)
- For each support joint, classify whether each secondary term assists or opposes WBC

**Step 3.3: Torque Budget Analysis**
For each support joint j in [2,3,7,8]:
- `required[j]`: From inverse dynamics
- `wbc[j]`: From WBC
- `posture[j]`: From posture controller
- `leg_position[j]`: From leg position controller
- `total[j]`: Sum of all sources
- `final[j]`: After clipping/rate limiting
- `deficit[j] = required[j] - final[j]`

**Deliverable**: `scripts/debug_static_inverse_dynamics.py`

**Script output**:
```
[INVERSE DYNAMICS BASELINE - Standing Keyframe]

Required Holding Torques (from mj_inverse):
  l_hip_pitch [2]:  12.5 Nm
  l_knee [3]:       -8.3 Nm
  r_hip_pitch [7]:  12.5 Nm
  r_knee [8]:       -8.3 Nm

Controller Torque Budget:
Joint | Required | WBC   | Posture | Leg_Pos | Total | Final | Deficit
------|----------|-------|---------|---------|-------|-------|--------
[2]   | 12.5     | 8.2   | 2.1     | 1.5     | 11.8  | 11.8  | +0.7
[3]   | -8.3     | -5.1  | -1.2    | -0.8    | -7.1  | -7.1  | -1.2
[7]   | 12.5     | 8.2   | 2.1     | 1.5     | 11.8  | 11.8  | +0.7
[8]   | -8.3     | -5.1  | -1.2    | -0.8    | -7.1  | -7.1  | -1.2

Analysis:
- Knee joints [3,8] have 1.2 Nm deficit (14% under-torqued)
- Hip pitch joints [2,7] have 0.7 Nm surplus
- Total vertical force deficit: ~15N (matches observed gap)
- Root cause: Knee torques insufficient for static support
```

**Acceptance Criteria**:
- Inverse dynamics computation succeeds
- Required holding torques are physically reasonable (within actuator limits)
- Torque budget clearly shows which joints are under-torqued
- Deficit magnitude correlates with observed force gap (15-20N)

---

## Decision Rules

Based on the diagnostic results, use these rules to identify the root cause:

**Rule 1: Contact solver / geometry issue**
- Condition: `tau_applied ≈ tau_ideal` (within 20%) BUT `actual_fz << desired_fz` (>15% deficit)
- Interpretation: Torques are correct, but contact forces don't match
- Likely causes: Contact point geometry wrong, Jacobian using wrong contact point, wheel slip, contact solver parameters, friction model

**Rule 2: Torque pipeline issue**
- Condition: `tau_applied << tau_ideal` (>30% deficit)
- Interpretation: Applied torques are insufficient
- Likely causes: Clipping too aggressive, rate limiting too restrictive, authority budget too low, cancellation between torque sources

**Rule 3: Jacobian mapping issue**
- Condition: Case C (ideal J^T f) fails to improve support vs Case A (zero control) OR Case D (inverse dynamics) works but Case C fails
- Interpretation: Jacobian mapping is incorrect
- Likely causes: Contact point mismatch between Jacobian and wrench matrix, sign error in J^T f, wrong contact frame

**Rule 4: Force distribution issue**
- Condition: `desired_fz_total ≈ weight` BUT `distributed_fz_total << weight` (>15% deficit)
- Interpretation: Force distributor is losing force
- Likely causes: Heuristic split logic, missing constraints, force asymmetry limits too tight

**Rule 5: Cancellation issue**
- Condition: For support joints [2,3,7,8], secondary torque sources oppose WBC with magnitude >30% of WBC
- Interpretation: Multiple controllers fighting each other
- Likely causes: Posture controller, leg position controller, or hip roll centering opposing WBC support torques

**Rule 6: Static equilibrium infeasible**
- Condition: Case D (inverse dynamics) also fails to hold standing
- Interpretation: The calibrated keyframe may not be a true static equilibrium
- Likely causes: Keyframe configuration, actuator limits insufficient, contact model doesn't support static standing

## Critical Questions to Answer

By the end of Phase 3, we must be able to answer:

1. **Where does the force disappear?**
   - In the force distributor? (desired 79N → distributed 60N)
   - In the Jacobian mapping? (distributed 79N → tau produces 60N contact)
   - In torque clipping? (tau_raw 79N-capable → tau_clipped 60N-capable)
   - In rate limiting? (tau_clipped 79N-capable → tau_smooth 60N-capable)
   - In the contact solver? (tau_smooth 79N-capable → actual contact 60N)

2. **Is it a sign error, magnitude error, or cancellation error?**
   - Sign error: Torque applied in wrong direction
   - Magnitude error: Torque too small by constant factor
   - Cancellation error: Multiple torque sources fighting each other

3. **Which joints are the bottleneck?**
   - Hip pitch? Knee? Hip roll? Wheels?
   - Are all support joints under-torqued equally, or is one joint the limiting factor?

4. **Is the Jacobian mapping correct?**
   - Does `J^T f` with f=weight produce contact force ≈ weight?
   - Is the contact point geometry correct?
   - Is the wrench matrix using the same contact point as the Jacobian?

5. **Are the actuators capable of producing required torques?**
   - Do actuator limits allow sufficient torque?
   - Is the WBC authority budget too restrictive?
   - Is rate limiting preventing rapid torque changes?

6. **Do secondary controllers interfere with WBC?**
   - Does posture controller reduce support torques?
   - Does leg position controller fight WBC?
   - Do multiple torque sources cancel each other?

---

## Success Criteria

**Phase 0 Success**:
- Force audit trail script runs and prints clear breakdown
- Force gap is reproduced (15-20N deficit)
- All telemetry values are valid (no NaN/None)

**Phase 1 Success**:
- Cases C (ideal J^T f) and D (inverse dynamics) produce stable standing
- Case B (current pipeline) reproduces failure
- Comparison identifies which pipeline stage causes divergence

**Phase 2 Success**:
- All actuator signs verified correct
- Support joints produce expected force changes
- No significant posture controller interference detected
- Any bugs found have regression tests

**Phase 3 Success**:
- Inverse dynamics baseline established
- Torque budget shows which joints are under-torqued
- Deficit magnitude (Nm) correlates with force gap (N)
- Root cause hypothesis is testable

**Overall Success**:
- We have a minimal reproduction case (single step, known state, clear discrepancy)
- We know exactly which stage loses the 15-20N
- We know if it's sign, magnitude, or cancellation
- We have regression tests to prevent re-introducing bugs
- We have a clear hypothesis for the fix (to be designed in follow-on phases)

---

## Out of Scope

The following are explicitly **not** part of this diagnostic spec:

- Implementing fixes (that comes after diagnosis)
- QP force distribution improvements
- Vertical impedance controller
- Contact recovery controller
- Equilibrium target unification
- Trajectory planning or rate limiting improvements
- Gain tuning (diagnostics must work with current gains)

These will be addressed in follow-on phases once the root cause is identified.

---

## Testing Strategy

**Unit Tests** (`tests/test_actuator_signs.py`):
- Actuator sign consistency
- Support joint authority
- Left/right symmetry
- Posture controller interference

**Diagnostic Scripts** (run manually, print human-readable output):
- `scripts/debug_force_gap.py`: Force audit trail
- `scripts/debug_static_support_parity.py`: Multi-case comparison
- `scripts/debug_static_inverse_dynamics.py`: Torque budget analysis

**Acceptance Test**:
- Run all diagnostic scripts
- All scripts complete without errors
- Output clearly identifies root cause
- Hypothesis is testable with minimal code change

---

## Next Steps After Diagnosis

Once phases 0-3 are complete and root cause is identified, we will design targeted fix phases:

- If **force distributor** is the bottleneck → Phase 4: QP-based force distribution
- If **Jacobian mapping** is wrong → Phase 4: Contact Jacobian correction
- If **torque clipping** is too aggressive → Phase 4: Authority budget adjustment
- If **posture controller** interferes → Phase 4: Torque coordination
- If **contact solver** loses force → Phase 4: Contact parameters tuning

The fix design will be informed by the diagnostic results, not prescribed in advance.
