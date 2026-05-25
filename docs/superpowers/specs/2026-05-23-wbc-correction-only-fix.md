# WBC Correction-Only Fix Specification

**Date:** 2026-05-23  
**Status:** Two-stage implementation required after diagnostic failures  
**Replaces:** StaticBalanceController wrapper approach (failed validation)

## Problem Statement

The wheeled biped robot falls after 14-15 steps due to a vertical contact force deficit. Root cause analysis revealed:

1. **Current WBC behavior:** Maps entire baseline body weight through joint-only contact Jacobian (J^T f), producing large support-joint torques that fight against contact constraints
2. **Physics reality:** MuJoCo contact constraints already provide baseline body-weight support through normal forces at wheel-floor contacts
3. **Result:** WBC-commanded torques and contact-provided support interfere, causing force deficit and eventual collapse

**Failed approach:** StaticBalanceController wrapper attempted to cancel WBC static bias using inverse dynamics reference torques. Validation revealed fundamental flaw: `mj_inverse` computes torques WITHOUT accounting for contact forces, producing large negative torques (-242 Nm, -204 Nm) that made performance 14× worse.

## Pre-Implementation Diagnostic Results

Three diagnostics were run to validate physics assumptions before implementation:

### Diagnostic A: Zero Correction Equilibrium Check - FAILED
- **Expected:** correction_wrench_norm < 10% model_weight (~7.9 N)
- **Actual:** correction_wrench_norm = 944.839 N
- **Root cause:** Current WBC computes corrections relative to **absolute zero**, not calibrated equilibrium
  - CoM at y = -0.013535 m (13.5mm backward) → 944.508 N sagittal correction force
  - Current code uses: `correction_Fy = -k_com_sagittal * com_pos[1]` (absolute)
  - Should use: `correction_Fy = -k_com_sagittal * (com_pos[1] - equilibrium_com_pos[1])` (relative)

### Diagnostic B: Distributor Zero-Input Check - FAILED
- **Expected:** Zero correction wrench → zero distributed force
- **Actual:** Non-contact wheel receives 50 N force in single-contact case
- **Root cause:** `min_recovery_force=50` injects fake force on non-contact wheels
- **Impact:** Violates correction-only semantics (zero correction should produce zero force)

### Diagnostic C: Passive Contact Feasibility - FAILED
- **Expected:** Robot stable with tau=0 for 20 steps
- **Actual:** Robot falls, contact forces drop from 106 N → 30-40 N (should be 79.46 N)
- **Interpretation:** Contact constraints alone do NOT provide stable baseline support
- **Implication:** Correction-only WBC must be paired with a separate posture/static holding controller

## Two-Stage Implementation Plan

Based on diagnostic failures, implementation is split into two stages:

---

## Stage 1: Equilibrium-Reference and Distributor-Semantics Fix

**Goal:** Fix equilibrium reference computation and distributor semantics so that Diagnostics A and B pass.

**Scope:** Do NOT implement full correction-only WBC integration yet. Only fix the reference frame and distributor behavior.

### Stage 1 Changes

#### 1. Capture Calibrated Equilibrium State

Add equilibrium state capture at initialization:

```python
class CentroidalWrenchComputer:
    def __init__(self, ...):
        # Existing parameters
        self.robot_mass = robot_mass
        self.gravity = gravity
        # ... existing gains ...
        
        # Equilibrium reference (set via set_equilibrium_reference)
        self.equilibrium_com_pos = None
        self.equilibrium_com_z = None
        self.equilibrium_pitch_x = None
        self.equilibrium_roll_y = None
        self.equilibrium_capture_point = None
        self.equilibrium_joint_pos = None
    
    def set_equilibrium_reference(
        self,
        com_pos: Array,
        com_z: float,
        pitch_x: float,
        roll_y: float,
        capture_point: Array,
        joint_pos: Array,
    ):
        """Set equilibrium reference for computing relative corrections.
        
        Must be called after calibrated initialization before computing corrections.
        
        Args:
            com_pos: Equilibrium CoM position (3,) [x, y, z]
            com_z: Equilibrium CoM z-position
            pitch_x: Equilibrium pitch angle (rad)
            roll_y: Equilibrium roll angle (rad)
            capture_point: Equilibrium capture point (2,) [x, y]
            joint_pos: Equilibrium joint positions (10,)
        """
        self.equilibrium_com_pos = com_pos
        self.equilibrium_com_z = com_z
        self.equilibrium_pitch_x = pitch_x
        self.equilibrium_roll_y = roll_y
        self.equilibrium_capture_point = capture_point
        self.equilibrium_joint_pos = joint_pos
```

#### 2. Compute Equilibrium-Relative Corrections

Modify `compute_desired_wrench` to use equilibrium-relative errors:

```python
def compute_desired_wrench(
    self,
    obs: Array,
    state: CentroidalState,
    height_cmd: float,
    roll_integral: float = 0.0,
) -> tuple[Array, Array]:
    """Compute desired 6D wrench on CoM from control objectives.
    
    All correction terms are computed relative to calibrated equilibrium.
    """
    # Verify equilibrium reference is set
    if self.equilibrium_com_pos is None:
        raise RuntimeError(
            "Equilibrium reference not set. Call set_equilibrium_reference() "
            "after calibrated initialization."
        )
    
    # Extract state
    com_pos = state.com_pos
    com_vel = state.com_vel
    pitch_x = state.pitch_x
    roll_y = state.roll_y
    pitch_rate_x = state.pitch_rate_x
    roll_rate_y = state.roll_rate_y
    cp = state.capture_point
    
    # CRITICAL: Compute equilibrium-relative errors
    com_error = com_pos - self.equilibrium_com_pos
    cp_error = cp - self.equilibrium_capture_point
    pitch_error = pitch_x - self.equilibrium_pitch_x
    roll_error = roll_y - self.equilibrium_roll_y
    height_error = self.equilibrium_com_z - com_pos[2]
    
    # === Force objectives (equilibrium-relative) ===
    
    # Gravity compensation: baseline vertical force
    f_gravity = jnp.array([0.0, 0.0, self.robot_mass * self.gravity])
    
    # Height tracking: proportional + damping
    f_height = jnp.array([
        0.0,
        0.0,
        self.k_height * height_error - self.k_height_damping * com_vel[2],
    ])
    
    # CoM lateral regulation (equilibrium-relative)
    f_com_lateral = jnp.array([
        -self.k_com_lateral * com_error[0] - self.k_com_lateral_damping * com_vel[0],
        0.0,
        0.0
    ])
    
    # CoM sagittal regulation (equilibrium-relative)
    f_com_sagittal = jnp.array([
        0.0,
        -self.k_com_sagittal * com_error[1] - self.k_com_sagittal_damping * com_vel[1],
        0.0
    ])
    
    # Capture point corrections (equilibrium-relative)
    f_cp = jnp.array([
        -self.k_cp_lateral * cp_error[0],
        -self.k_cp_sagittal * cp_error[1],
        0.0
    ])
    
    # Total desired force
    desired_force = f_gravity + f_height + f_com_lateral + f_com_sagittal + f_cp
    
    # === Moment objectives (equilibrium-relative) ===
    
    # Roll stabilization: PID control (equilibrium-relative)
    m_roll_y = -self.k_roll * roll_error - self.k_roll_rate * roll_rate_y - self.k_roll_integral * roll_integral
    m_roll_y = self._limit_roll_moment(m_roll_y)
    
    # Pitch stabilization: inverted pendulum control (equilibrium-relative)
    pitch_correction_force = -self.k_pitch * pitch_error - self.k_pitch_rate * pitch_rate_x
    desired_force = desired_force.at[1].add(pitch_correction_force)
    
    desired_moment = jnp.array([0.0, m_roll_y, 0.0])
    
    return desired_force, desired_moment
```

#### 3. Add Correction Breakdown Telemetry

Add detailed correction breakdown to diagnostics:

```python
# In IntegratedWBC.compute_wbc_torque_with_diagnostics
diagnostics = {
    # Equilibrium reference
    "equilibrium_com_x": float(self.wrench_computer.equilibrium_com_pos[0]),
    "equilibrium_com_y": float(self.wrench_computer.equilibrium_com_pos[1]),
    "equilibrium_com_z": float(self.wrench_computer.equilibrium_com_z),
    "equilibrium_pitch_x": float(self.wrench_computer.equilibrium_pitch_x),
    "equilibrium_roll_y": float(self.wrench_computer.equilibrium_roll_y),
    
    # Equilibrium-relative errors
    "com_error_x": float(state.com_pos[0] - self.wrench_computer.equilibrium_com_pos[0]),
    "com_error_y": float(state.com_pos[1] - self.wrench_computer.equilibrium_com_pos[1]),
    "pitch_error": float(state.pitch_x - self.wrench_computer.equilibrium_pitch_x),
    "roll_error": float(state.roll_y - self.wrench_computer.equilibrium_roll_y),
    "height_error": float(self.wrench_computer.equilibrium_com_z - state.com_pos[2]),
    
    # Correction force breakdown (compute these separately for telemetry)
    "correction_Fx_com": ...,
    "correction_Fx_cp": ...,
    "correction_Fy_com": ...,
    "correction_Fy_cp": ...,
    "correction_Fy_pitch": ...,
    "correction_Fz_height": ...,
    "correction_My_roll": ...,
    
    # Total correction wrench
    "correction_wrench_Fx": ...,
    "correction_wrench_Fy": ...,
    "correction_wrench_Fz": ...,
    "correction_wrench_My": ...,
    "correction_wrench_norm": ...,
    
    # Existing diagnostics...
}
```

#### 4. Fix SimpleForceDistributor Correction-Only Behavior

Modify `SimpleForceDistributor` to ensure zero correction → zero force:

```python
class SimpleForceDistributor:
    def distribute_wrench_contact_aware(
        self,
        desired_wrench: Array,
        left_contact: bool,
        right_contact: bool,
        wheel_pos_left: Array,
        wheel_pos_right: Array,
        hip_roll_authority_scale: float = 1.0,
        recovery_mode: bool = False,  # NEW: explicit recovery mode flag
    ) -> tuple[Array, Array, Array, dict]:
        """Distribute wrench to contact forces.
        
        Args:
            desired_wrench: (6,) [Fx, Fy, Fz, Mx, My, Mz]
            left_contact: Left wheel in contact
            right_contact: Right wheel in contact
            wheel_pos_left: Left wheel position relative to CoM
            wheel_pos_right: Right wheel position relative to CoM
            hip_roll_authority_scale: Hip roll authority scaling
            recovery_mode: If True, apply min_recovery_force behavior.
                          If False (default), zero wrench produces zero force.
        
        Returns:
            Tuple of (f_left, f_right, tau_hip_roll, diagnostics)
        """
        # CRITICAL: In normal mode (recovery_mode=False), zero wrench must produce zero force
        # Only apply min_recovery_force when explicitly in recovery mode
        
        if not left_contact and not right_contact:
            # No contact: zero force
            return (
                jnp.zeros(3),
                jnp.zeros(3),
                jnp.zeros(2),
                {"mode": "no_contact"},
            )
        
        # Extract wrench components
        desired_fz = desired_wrench[2]
        
        # Check if wrench is near zero (correction-only mode at equilibrium)
        wrench_norm = jnp.linalg.norm(desired_wrench)
        is_near_zero = wrench_norm < 1.0  # 1 N threshold
        
        if is_near_zero and not recovery_mode:
            # Zero correction wrench in normal mode → zero force
            return (
                jnp.zeros(3),
                jnp.zeros(3),
                jnp.zeros(2),
                {"mode": "zero_correction"},
            )
        
        # Single contact case
        if left_contact and not right_contact:
            # Left wheel only: solve for wrench, right wheel gets ZERO force
            f_left = self._solve_single_contact(desired_wrench, wheel_pos_left)
            f_right = jnp.zeros(3)  # CRITICAL: No fake force on non-contact wheel
            tau_hip_roll = jnp.zeros(2)
            return (f_left, f_right, tau_hip_roll, {"mode": "single_left"})
        
        if right_contact and not left_contact:
            # Right wheel only: solve for wrench, left wheel gets ZERO force
            f_right = self._solve_single_contact(desired_wrench, wheel_pos_right)
            f_left = jnp.zeros(3)  # CRITICAL: No fake force on non-contact wheel
            tau_hip_roll = jnp.zeros(2)
            return (f_left, f_right, tau_hip_roll, {"mode": "single_right"})
        
        # Double contact case: distribute wrench between both wheels
        # Apply min_recovery_force ONLY if recovery_mode=True
        if recovery_mode:
            min_fz_per_wheel = 50.0  # Recovery mode: maintain minimum force
        else:
            min_fz_per_wheel = 0.0  # Normal mode: allow zero force
        
        # ... existing double-contact distribution logic ...
```

### Stage 1 Acceptance Criteria

**Must pass before proceeding to Stage 2:**

1. **Equilibrium reference captured:**
   - `equilibrium_com_pos`, `equilibrium_com_z`, `equilibrium_pitch_x`, `equilibrium_roll_y`, `equilibrium_capture_point`, `equilibrium_joint_pos` stored
   - Set via `set_equilibrium_reference()` after calibrated initialization

2. **Corrections computed relative to equilibrium:**
   - `com_error = state.com_pos - equilibrium_com_pos`
   - `cp_error = state.capture_point - equilibrium_capture_point`
   - `pitch_error = state.pitch_x - equilibrium_pitch_x`
   - `roll_error = state.roll_y - equilibrium_roll_y`
   - `height_error = equilibrium_com_z - state.com_pos[2]`

3. **Correction breakdown telemetry added:**
   - Log individual correction components: `correction_Fx_com`, `correction_Fx_cp`, `correction_Fy_com`, `correction_Fy_cp`, `correction_Fy_pitch`, `correction_Fz_height`, `correction_My_roll`
   - Log total correction wrench and norm

4. **At calibrated equilibrium (height_cmd = equilibrium_com_z):**
   - `correction_wrench_norm < 10% model_weight` (~7.9 N)
   - `correction_Fz < 5% model_weight` (~4.0 N)
   - `correction_Fy` should NOT be hundreds of Newtons (was 944 N, should be < 10 N)
   - If only correction WBC is active (no posture/leg PD), `tau_wbc_support_joints` should be near zero

5. **SimpleForceDistributor correction-only behavior fixed:**
   - Zero correction wrench → zero distributed force (both wheels)
   - Non-contact wheel receives zero force (no fake 50 N injection)
   - `min_recovery_force=50` removed from normal distribution or gated behind `recovery_mode=True`
   - Recovery mode is out of scope for Stage 1

6. **Diagnostic A passes:**
   - Rerun `scripts/debug_wbc_correction_only_diagnostics.py`
   - Diagnostic A: correction_wrench_norm < 10% model_weight
   - Diagnostic A: correction_Fz < 5% model_weight

7. **Diagnostic B passes:**
   - Rerun `scripts/debug_wbc_correction_only_diagnostics.py`
   - Diagnostic B: Double contact with zero correction → total Fz < 1.0 N
   - Diagnostic B: Single contact with zero correction → non-contact wheel Fz < 0.1 N
   - Diagnostic B: No contact with zero correction → total Fz < 0.1 N

**Do NOT proceed to Stage 2 until Diagnostics A and B pass.**

### Stage 1 Out of Scope

- Do NOT integrate correction-only WBC into `IntegratedWBC.compute_wbc_torque_with_diagnostics` yet
- Do NOT remove baseline mg from wrench computation yet (keep `f_gravity` in `desired_force`)
- Do NOT add static posture holding controller yet
- Do NOT tune gains
- Do NOT claim 100-step standing
- Do NOT implement recovery mode logic

---

## Stage 2: Static Posture Holding + Correction-Only WBC Integration

**Goal:** Integrate correction-only WBC with a separate static posture holding controller to achieve stable standing.

**Prerequisite:** Stage 1 complete and Diagnostics A and B passing.

### Stage 2 Rationale

Diagnostic C revealed that contact constraints alone do NOT provide stable baseline support:
- Robot falls with tau=0
- Contact forces drop from 106 N → 30-40 N (should be 79.46 N)
- CoM downward velocity reaches -0.278 m/s

**Interpretation:** Actuator torques ARE required to maintain internal joint posture against gravity. Correction-only WBC handles perturbations, but a separate posture controller is needed for baseline joint holding.

### Stage 2 Architecture

```
tau_total = tau_static_posture_hold + tau_wbc_correction
```

Where:
- `tau_static_posture_hold`: Maintains internal joint posture at equilibrium (computed via IK or static torque reference)
- `tau_wbc_correction`: Handles perturbations and stabilization (correction-only WBC)

### Stage 2 Changes

#### 1. Add Static Posture Holding Controller

Create a separate controller that maintains equilibrium joint posture:

```python
class StaticPostureHoldingController:
    """Maintains internal joint posture at calibrated equilibrium.
    
    Provides baseline joint torques to counteract gravity and hold posture.
    Does NOT map baseline body weight through contact Jacobian.
    """
    
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        equilibrium_joint_pos: NDArray,
        kp_posture: float = 20.0,
        kd_posture: float = 2.0,
    ):
        self.equilibrium_joint_pos = equilibrium_joint_pos
        self.kp_posture = kp_posture
        self.kd_posture = kd_posture
    
    def compute_posture_holding_torque(
        self,
        joint_pos: NDArray,
        joint_vel: NDArray,
    ) -> NDArray:
        """Compute torque to maintain equilibrium posture.
        
        Args:
            joint_pos: Current joint positions (10,)
            joint_vel: Current joint velocities (10,)
        
        Returns:
            Posture holding torque (10,)
        """
        pos_error = self.equilibrium_joint_pos - joint_pos
        tau_posture = self.kp_posture * pos_error - self.kd_posture * joint_vel
        return tau_posture
```

#### 2. Modify IntegratedWBC to Use Correction-Only Wrench

```python
def compute_wbc_torque_with_diagnostics(
    self,
    mj_data: mujoco.MjData,
    obs: Array,
    state: CentroidalState,
    height_cmd: float,
    hip_roll_authority_scale: float = 1.0,
) -> tuple[Array, dict]:
    # Update roll integral
    # ...
    
    # Compute desired wrench (still includes baseline mg for now)
    desired_force, desired_moment = self.wrench_computer.compute_desired_wrench(
        obs, state, height_cmd, self.roll_integral
    )
    
    # CRITICAL: Separate baseline and correction
    baseline_wrench = jnp.array([0.0, 0.0, self.robot_mass * self.gravity, 0.0, 0.0, 0.0])
    total_wrench = jnp.concatenate([desired_force, desired_moment])
    correction_wrench = total_wrench - baseline_wrench
    
    # CRITICAL: Only pass correction_wrench to force distributor
    # Baseline mg is NOT mapped through J^T f
    wheel_pos_left, wheel_pos_right = self._compute_wheel_positions_relative_to_com(
        mj_data, state.com_pos
    )
    
    f_left, f_right, tau_hip_roll, distribution_diagnostics = (
        self.force_distributor.distribute_wrench_contact_aware(
            correction_wrench,  # NOT total_wrench, NOT baseline_wrench
            left_contact=bool(state.left_wheel_contact),
            right_contact=bool(state.right_wheel_contact),
            wheel_pos_left=wheel_pos_left,
            wheel_pos_right=wheel_pos_right,
            hip_roll_authority_scale=hip_roll_authority_scale,
            recovery_mode=False,  # Normal correction-only mode
        )
    )
    
    # Map correction forces to joint torques
    tau_contact = self.contact_jacobian.map_contact_forces_to_torques(
        mj_data, f_left, f_right, tau_hip_roll=None
    )
    tau_hip = self._build_direct_hip_roll_torque(tau_hip_roll)
    tau_wbc_correction = tau_contact + tau_hip
    
    # Disable force feedback in correction-only mode
    force_scale = 1.0
    
    # Apply authority budget
    tau_wbc_correction_scaled = tau_wbc_correction * force_scale
    tau_wbc = self.clip_to_authority_budget(tau_wbc_correction_scaled)
    
    # Diagnostics
    diagnostics = {
        "baseline_fz": float(baseline_wrench[2]),
        "correction_wrench_Fx": float(correction_wrench[0]),
        "correction_wrench_Fy": float(correction_wrench[1]),
        "correction_wrench_Fz": float(correction_wrench[2]),
        "correction_wrench_My": float(correction_wrench[4]),
        "correction_wrench_norm": float(jnp.linalg.norm(correction_wrench)),
        "tau_wbc_correction": tau_wbc_correction,
        "force_feedback_disabled": True,
        # ... existing diagnostics ...
    }
    
    return tau_wbc, diagnostics
```

#### 3. Integrate Posture Holding + Correction WBC

In the main control loop:

```python
# Compute correction-only WBC torque
tau_wbc_correction, wbc_diagnostics = wbc_controller.compute_wbc_torque_with_diagnostics(
    mj_data, obs, centroidal_state, height_cmd
)

# Compute static posture holding torque
tau_posture_hold = posture_controller.compute_posture_holding_torque(
    joint_pos, joint_vel
)

# Combine: total = posture holding + correction
tau_total = tau_posture_hold + tau_wbc_correction

# Apply to robot (with existing smoothing/rate limiting/PID)
mj_data.ctrl[:] = tau_total
```

### Stage 2 Acceptance Criteria

1. **Static posture holding controller implemented:**
   - Maintains equilibrium joint posture via PD control
   - Does NOT map baseline mg through contact Jacobian

2. **IntegratedWBC uses correction-only wrench:**
   - Baseline mg NOT passed to force distributor
   - Only correction wrench mapped through J^T f

3. **Torque composition:**
   - `tau_total = tau_posture_hold + tau_wbc_correction`
   - Telemetry logs both components separately

4. **100-step static standing:**
   - Survive 100 steps without termination
   - Contact force within 15% of model_weight
   - Pitch/roll < 0.1 rad (< 5.7 degrees)
   - CoM height within ±0.05 m of height_cmd

5. **Diagnostic C interpretation:**
   - Diagnostic C failure does NOT invalidate correction-only WBC
   - It confirms that posture holding IS needed alongside correction WBC
   - Document this requirement clearly

### Stage 2 Out of Scope

- QP-based force distribution
- Contact recovery logic
- Trajectory planning
- Full inverse dynamics WBC
- Stand-up recovery
- Locomotion
- Adaptive force feedback for correction-only mode

---

## Important Constraints

**Do NOT:**
- Tune gains (use existing gain values)
- Map baseline mg through J^T f (only correction wrench)
- Add fake contact force to non-contact wheels
- Claim 100-step standing until Stage 2 complete
- Implement recovery mode logic in Stage 1
- Proceed to Stage 2 until Diagnostics A and B pass

**Do:**
- Fix equilibrium reference computation first (Stage 1)
- Fix distributor semantics first (Stage 1)
- Rerun Diagnostics A and B after Stage 1
- Only proceed to Stage 2 after A and B pass
- Document that Diagnostic C failure requires posture controller

---

## Mass and Height Conventions

**Robot mass:**
```python
robot_mass = float(np.sum(mj_model.body_mass))  # ~8.1 kg
gravity = float(abs(mj_model.opt.gravity[2]))   # 9.81 m/s²
model_weight = robot_mass * gravity              # ~79.46 N
```

**Height definitions:**
- `root_z`: MuJoCo root body z-position (qpos[2])
- `com_z`: Center of mass z-position (data.subtree_com[1, 2])
- `height_cmd`: Desired CoM z-position (NOT root_z)

**Equilibrium reference:**
- Captured at calibrated initialization (root_z adjusted for -5e-4 contact penetration)
- All corrections computed relative to equilibrium, not absolute zero

---

## References

- **Failed approach:** [StaticBalanceController wrapper](../plans/2026-05-23-static-dynamics-consistency-fix-plan.md)
- **Root cause analysis:** Phases 0-3 diagnostics (scripts/debug_*.py)
- **Validation failure:** debug_static_support_parity_v2.py output showing 14× worse performance
- **Physics principle:** Contact constraints provide baseline support in static equilibrium
- **Mass convention:** simulate_hierarchical_controller.py line 377: `robot_mass = float(np.sum(mj_model.body_mass))`
- **Mass consistency fix:** All controllers now derive mass from MuJoCo model (~8.1 kg), not hardcoded 15 kg default
- **Pre-implementation diagnostics:** scripts/debug_wbc_correction_only_diagnostics.py
  - Diagnostic A: FAILED (correction_wrench_norm = 944.839 N, should be < 7.9 N)
  - Diagnostic B: FAILED (non-contact wheel receives 50 N fake force)
  - Diagnostic C: FAILED (robot falls with tau=0, posture controller needed)
