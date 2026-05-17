# Centroidal/Capture-Point Dynamic Balance Controller Design

**Date:** 2026-05-14  
**Phase:** B.9 Step 5.26  
**Target:** Transform Step 5.25 posture stabilizer into true dynamic balance controller  
**Success Criteria:** >20 seconds survival time

## Executive Summary

This design transforms the Step 5.25 hierarchical torque fusion controller (0.87s survival, 83.8% WBC authority) into a true dynamic balance system through integrated centroidal dynamics control. The approach replaces separate WBC + stabilization layers with a unified Centroidal WBC that directly incorporates CoM regulation, capture point tracking, and momentum coordination.

**Key Innovation:** Integrated centroidal-aware WBC with height-dependent Linear Inverted Pendulum (LIP) capture point estimation, enabling dynamic balance behavior rather than rigid posture stabilization.

**Target Performance:** >20 seconds mean survival time (23× improvement over Step 5.25 baseline)

## 1. Overall Architecture

### Three-Level Hierarchy

```
Level 1: Centroidal WBC (60% authority)
├─ Roll stabilization (existing k_roll gains)
├─ CoM regulation (NEW - deadband control)
├─ Capture point tracking (NEW - height-dependent LIP)
└─ Height tracking (existing height IK)

Level 2: Momentum Coordinator (20% authority)
├─ Lateral momentum damping
├─ Angular momentum damping
├─ Feedforward compensation (height transitions)
└─ Contact-aware recovery redistribution

Level 3: Posture Regularization (20% authority)
└─ Weak posture restoration (gated by WBC error)
```

### Key Architectural Changes from Step 5.25

1. **WBC becomes centroidal-aware** - CoM and capture point objectives computed inside WBC torque generation
2. **Authority reallocation** - WBC 60% (down from 83.8%), momentum 20% (new), posture 20% (up from ~4%)
3. **Contact stabilization absorbed** - Contact-aware recovery moves into momentum coordinator
4. **Oscillation damping absorbed** - Momentum damping replaces separate oscillation-triggered damping

### Control Flow

```
Observation → Centroidal State Estimator
                ↓
            [CoM position, velocity]
            [Capture point estimate]
            [Momentum (linear, angular)]
            [Contact forces]
                ↓
            Centroidal WBC (60%)
                ↓
            Momentum Coordinator (20%)
                ↓
            Posture Regularization (20%)
                ↓
            τ_total = τ_wbc + τ_momentum + τ_posture
                ↓
            Clip to actuator limits
                ↓
            Robot
```

## 2. Centroidal State Estimator

### Required Centroidal States

```python
@dataclass
class CentroidalState:
    # CoM kinematics
    com_pos: np.ndarray  # [x, y, z] in world frame
    com_vel: np.ndarray  # [vx, vy, vz] in world frame
    
    # Capture point (height-dependent LIP)
    capture_point: np.ndarray  # [x_cp, y_cp] in world frame
    divergence: np.ndarray     # [div_x, div_y] divergent component
    
    # Momentum
    linear_momentum: np.ndarray   # [px, py, pz] total system
    angular_momentum: np.ndarray  # [Lx, Ly, Lz] about CoM
    
    # Contact state
    left_wheel_contact: bool
    right_wheel_contact: bool
    left_wheel_force: float   # normal force [N]
    right_wheel_force: float  # normal force [N]
```

### Extraction from MJX Data

- **CoM position/velocity**: `data.subtree_com[1]` (torso subtree) - already available in Step 5.25
- **Linear momentum**: `total_mass * com_vel` (simplified)
- **Angular momentum**: `body_angular_velocity * body_inertia` (approximate, torso only)
- **Contact forces**: Extract from `data.contact.force` for wheel geoms (fixes Step 5.25's 0.0% contact activation)

### Height-Dependent Capture Point

Using Linear Inverted Pendulum with height-varying natural frequency:

```
ω(h) = √(g / h_com)
x_cp = x_com + vx_com / ω(h)
y_cp = y_com + vy_com / ω(h)

divergence = [x_com - x_support, y_com - y_support] + [vx_com, vy_com] / ω(h)
```

Where:
- `h_com` = current CoM height above ground (varies with commanded height)
- `x_support, y_support` = center of support polygon (midpoint between wheels)
- Divergence > 0 means CoM moving away from support (unstable)

### Soft Deadband Control Parameters

```yaml
centroidal_control:
  # CoM position deadband
  com_deadband_lateral: 0.02   # ±2cm lateral drift allowed
  com_deadband_sagittal: 0.03  # ±3cm fore-aft drift allowed
  
  # Capture point deadband
  cp_deadband: 0.05            # ±5cm capture point error allowed
  
  # Momentum deadband
  momentum_deadband_linear: 2.0   # kg⋅m/s
  momentum_deadband_angular: 0.5  # kg⋅m²/s
```

Control only activates when state exceeds deadband, allowing natural sway within bounds.

## 3. Centroidal WBC Implementation

### WBC Torque Computation

The Centroidal WBC computes torques to achieve four simultaneous objectives: roll stabilization, CoM regulation, capture point tracking, and height tracking.

```python
def compute_centroidal_wbc_torque(
    state: CentroidalState,
    obs: np.ndarray,
    config: CentroidalWBCConfig
) -> np.ndarray:
    """
    Compute WBC torque with integrated centroidal objectives.
    Returns τ_wbc with 60% authority budget.
    """
    
    # 1. Roll stabilization (existing from Step 5.25)
    roll_error = obs[roll_idx]
    roll_rate = obs[roll_rate_idx]
    τ_roll = -config.k_roll * roll_error - config.k_roll_rate * roll_rate
    
    # 2. CoM regulation (NEW - deadband control)
    com_error_lateral = state.com_pos[1] - 0.0  # y-axis
    com_error_sagittal = state.com_pos[0] - 0.0  # x-axis
    
    # Apply deadband
    if abs(com_error_lateral) < config.com_deadband_lateral:
        com_error_lateral = 0.0
    if abs(com_error_sagittal) < config.com_deadband_sagittal:
        com_error_sagittal = 0.0
    
    τ_com = compute_com_correction_torque(
        com_error_lateral, com_error_sagittal,
        state.com_vel, config
    )
    
    # 3. Capture point tracking (NEW - height-dependent LIP)
    cp_error = state.capture_point - support_center
    
    # Apply deadband
    if np.linalg.norm(cp_error) < config.cp_deadband:
        cp_error = np.zeros(2)
    
    τ_cp = compute_capture_point_torque(
        cp_error, state.divergence, config
    )
    
    # 4. Height tracking (existing height IK)
    τ_height = compute_height_tracking_torque(obs, config)
    
    # Combine objectives with task priorities
    τ_wbc_desired = (
        config.w_roll * τ_roll +
        config.w_com * τ_com +
        config.w_cp * τ_cp +
        config.w_height * τ_height
    )
    
    # Clip to 60% authority budget
    τ_wbc = clip_to_budget(τ_wbc_desired, budget=0.6)
    
    return τ_wbc
```

### CoM Correction Torque Mapping

CoM errors map to joint torques through virtual model control:

- **Lateral CoM error** → hip roll torques (both legs, symmetric)
- **Sagittal CoM error** → wheel torques (differential for pitch correction)
- **CoM velocity** → damping term (opposes drift)

### Capture Point Torque Mapping

Capture point errors indicate divergent instability and require aggressive correction:

- **Lateral divergence** → hip roll (asymmetric) + wheel differential
- **Sagittal divergence** → wheel common mode

### Proposed Gains

```yaml
centroidal_wbc:
  # Roll stabilization (from Step 5.25)
  k_roll: 20.0
  k_roll_rate: 4.0
  
  # CoM regulation (NEW)
  k_com_lateral: 15.0
  k_com_lateral_damping: 3.0
  k_com_sagittal: 10.0
  k_com_sagittal_damping: 2.0
  
  # Capture point tracking (NEW)
  k_cp_lateral: 25.0
  k_cp_sagittal: 20.0
  k_cp_wheel_diff: 8.0
  
  # Height tracking (existing)
  k_height: 5.0
  
  # Task weights
  w_roll: 1.0
  w_com: 0.8
  w_cp: 1.2      # Highest priority - divergence is critical
  w_height: 0.6
  
  # Authority budget
  wbc_authority_budget: 0.6  # 60% of actuator range
```

## 4. Momentum Coordinator

The Momentum Coordinator provides Level 2 stabilization with 20% authority budget, focusing on momentum regulation and contact-aware recovery.

### Three Components

1. **Momentum damping** - Reactive stabilization opposing unwanted momentum
2. **Feedforward compensation** - Proactive compensation during commanded height transitions
3. **Contact-aware recovery** - Asymmetric support redistribution based on wheel loading

### Momentum Damping

```python
def compute_momentum_damping_torque(
    state: CentroidalState,
    config: MomentumCoordinatorConfig
) -> np.ndarray:
    """
    Damp centroidal momentum to prevent oscillation buildup.
    Only activates outside deadband.
    """
    
    τ = np.zeros(10)
    
    # Linear momentum damping (lateral and sagittal)
    linear_momentum_mag = np.linalg.norm(state.linear_momentum[:2])
    
    if linear_momentum_mag > config.momentum_deadband_linear:
        # Lateral momentum → hip roll damping
        lateral_momentum = state.linear_momentum[1]
        τ[L_HIP_ROLL] = -config.k_momentum_lateral * lateral_momentum
        τ[R_HIP_ROLL] = -config.k_momentum_lateral * lateral_momentum
        
        # Sagittal momentum → wheel damping
        sagittal_momentum = state.linear_momentum[0]
        τ[L_WHEEL] = -config.k_momentum_sagittal * sagittal_momentum
        τ[R_WHEEL] = -config.k_momentum_sagittal * sagittal_momentum
    
    # Angular momentum damping (roll axis most critical)
    angular_momentum_mag = abs(state.angular_momentum[0])  # roll axis
    
    if angular_momentum_mag > config.momentum_deadband_angular:
        # Roll momentum → differential hip roll
        roll_momentum = state.angular_momentum[0]
        τ[L_HIP_ROLL] += -config.k_angular_roll * roll_momentum
        τ[R_HIP_ROLL] += config.k_angular_roll * roll_momentum  # opposite
    
    return τ
```

### Feedforward Compensation

During commanded height transitions, predict momentum changes and preemptively compensate:

- Squatting (height decreasing) → anticipate forward pitch
- Standing (height increasing) → anticipate backward pitch
- Preemptive wheel torque + hip pitch adjustment

### Contact-Aware Recovery

Use wheel contact forces to detect unloading and redistribute support:

- Compute force imbalance ratio
- Detect unloading (one wheel losing contact)
- Shift support toward loaded wheel via hip roll + wheel differential

### Proposed Gains

```yaml
momentum_coordinator:
  # Momentum damping
  k_momentum_lateral: 0.8
  k_momentum_sagittal: 1.2
  k_angular_roll: 1.5
  
  # Feedforward compensation
  k_feedforward: 5.0
  k_feedforward_hip: 2.0
  height_transition_threshold: 0.05  # m/s
  
  # Contact-aware recovery
  k_contact_recovery: 10.0
  k_contact_wheel_diff: 4.0
  unloading_threshold: 0.3  # 30% force asymmetry
  
  # Authority budget
  momentum_authority_budget: 0.2  # 20% of actuator range
```

## 5. Posture Regularization

The Posture Regularization layer provides Level 3 stabilization with 20% authority budget, offering weak posture restoration when dynamic balance objectives are satisfied.

### Design Philosophy

Posture correction should be **permissive** - it guides the robot toward good posture but doesn't fight against dynamic balance requirements.

### Two-Level Gating

1. **WBC error gate**: If Centroidal WBC error is large (>30% of capacity), posture regularization is completely disabled
2. **Momentum coordinator gate**: If momentum coordinator is actively damping or recovering, posture authority is reduced by 50%

### Per-Joint Deadbands

```yaml
posture_regularization:
  # Proportional gain
  k_posture: 2.0  # Weak compared to WBC gains
  
  # Per-joint deadbands (radians)
  posture_deadband:
    hip_roll: 0.05    # ±2.9° - allow lateral sway
    hip_yaw: 0.03     # ±1.7° - tighter, yaw drift is bad
    hip_pitch: 0.08   # ±4.6° - allow squat variation
    knee: 0.10        # ±5.7° - allow knee bend variation
    wheel: 0.0        # wheels don't have posture target
  
  # Gating thresholds
  wbc_error_threshold: 0.3        # 30% of WBC capacity
  momentum_active_scale: 0.5      # 50% authority when momentum active
  
  # Authority budget
  posture_authority_budget: 0.2   # 20% of actuator range
```

## 6. Ablation Study Design

Sequential additive approach, starting from Step 5.25 baseline:

```
Baseline: Step 5.25 (hierarchical torque fusion) - 0.87s
  ↓
Candidate 1: + CoM stabilization only
  ↓
Candidate 2: + CoM + Capture point tracking
  ↓
Candidate 3: + CoM + Capture point + Momentum damping
  ↓
Candidate 4: + CoM + Capture point + Momentum damping + Feedforward
  ↓
Candidate 5: + CoM + Capture point + Momentum damping + Feedforward + Contact recovery
  ↓
Final: Full dynamic balance stack (Centroidal WBC)
```

### Evaluation Protocol

- 50 episodes per candidate at h=0.60m
- Compare against Step 5.25 baseline (0.87s survival)
- Analyze incremental contribution of each component

### Expected Outcomes

- **CoM regulation**: 5-10% improvement
- **Capture point**: 5-10% improvement
- **Momentum damping**: 3-5% improvement
- **Feedforward**: 2-5% improvement
- **Contact recovery**: 5-10% improvement

**Target**: Final candidate >20s survival to justify architectural complexity

## 7. Evaluation Metrics and Validation

### Core Performance Metrics

```yaml
# Primary success criteria
survival_time_mean: float      # Must exceed 20.0s (target)
survival_time_std: float
fall_rate: float

# Dynamic balance indicators (NEW)
com_drift_lateral_rms: float
com_drift_sagittal_rms: float
com_drift_max: float
capture_point_error_rms: float
divergence_episodes: int

# Momentum regulation (NEW)
linear_momentum_rms: float
angular_momentum_rms: float
momentum_oscillation_freq: float

# Contact stability (NEW)
contact_asymmetry_mean: float
contact_loss_events: int
recovery_success_rate: float

# Existing metrics
roll_rms_deg: float
pitch_rms_deg: float
torque_rms_nm: float
energy_sum_abs_tau_qdot: float

# Authority allocation
wbc_authority_mean: float      # Must remain >60%
momentum_authority_mean: float # Target ~20%
posture_authority_mean: float  # Target ~20%
saturation_rate: float         # Must remain <5%
```

### Dynamic Balance Behavior Indicators

```python
# Distinguish dynamic balance from posture stabilization
sway_amplitude_lateral: float
sway_amplitude_sagittal: float
sway_frequency: float
torque_intermittency: float     # % of time torque < 10% max
torque_peak_to_rms_ratio: float
recovery_time_mean: float
recovery_smoothness: float
hip_wheel_correlation: float
left_right_symmetry: float
```

### Success Criteria

The final candidate must meet ALL of these:

1. **Survival improvement**: >20.0s mean survival
2. **WBC authority preserved**: >60% mean WBC authority
3. **Low saturation**: <5% saturation rate
4. **Dynamic balance emergence**:
   - CoM drift RMS < 0.05m
   - Capture point error RMS < 0.08m
   - Torque intermittency >30%
5. **Smooth recovery**: Recovery time <0.5s after disturbances

### Failure Criteria

Any of these indicate design failure:

- Survival < 0.87s (regression from Step 5.25)
- WBC authority < 60% (authority suppression)
- Saturation > 10% (actuator limits hit)
- CoM drift RMS > 0.10m (uncontrolled sway)
- Divergence in >20% of episodes (unstable)

## 8. Implementation Structure

### File Structure

```
wheeled_biped/controllers/
├── centroidal_balance_controller.py  (NEW - main controller)
├── centroidal_state_estimator.py     (NEW - state extraction)
├── capture_point_estimator.py        (NEW - LIP-based CP)
├── momentum_coordinator.py           (NEW - Level 2 coordinator)
└── dual_rate_balance_controller.py   (EXISTING - Step 5.25 baseline)

configs/controllers/
├── step5_26_baseline.yaml            (Step 5.25 config)
├── step5_26_candidate_1_com.yaml
├── step5_26_candidate_2_com_cp.yaml
├── step5_26_candidate_3_momentum.yaml
├── step5_26_candidate_4_feedforward.yaml
├── step5_26_candidate_5_contact.yaml
└── step5_26_final_full_stack.yaml

scripts/
└── phase_b9_step5_26_centroidal_balance_evaluation.py

outputs/phase_b9_step5_26_dynamic_balance_controller/
├── centroidal_balance_design.md
├── ablation_results.csv
├── candidate_*.json
├── runtime_signal_trace.csv
├── full_validation.csv
└── step5_26_summary.json
```

### Controller Class Structure

```python
class CentroidalBalanceController:
    """
    Integrated centroidal/capture-point dynamic balance controller.
    """
    
    def __init__(self, config: CentroidalBalanceConfig):
        self.config = config
        self.state_estimator = CentroidalStateEstimator(config)
        self.cp_estimator = CapturePointEstimator(config)
        self.momentum_coordinator = MomentumCoordinator(config)
        
    def compute_action(self, obs: np.ndarray, data: mjx.Data) -> np.ndarray:
        """Main control loop."""
        
        # 1. Extract centroidal state
        centroidal_state = self.state_estimator.estimate(obs, data)
        
        # 2. Estimate capture point
        centroidal_state = self.cp_estimator.update(centroidal_state)
        
        # 3. Level 1: Centroidal WBC (60% authority)
        τ_wbc = self.compute_centroidal_wbc(centroidal_state, obs)
        
        # 4. Level 2: Momentum coordinator (20% authority)
        τ_momentum = self.momentum_coordinator.compute(...)
        
        # 5. Level 3: Posture regularization (20% authority)
        τ_posture = self.compute_posture_regularization(...)
        
        # 6. Hierarchical fusion
        τ_total = τ_wbc + τ_momentum + τ_posture
        
        # 7. Convert torques to normalized actions
        action = self.torque_to_action(τ_total, obs, data)
        
        return action
```

## 9. Error Handling and Safety

### Critical Safety Checks

```python
class SafetyMonitor:
    """Monitor centroidal state for dangerous conditions."""
    
    def check_safety(self, state: CentroidalState, obs: np.ndarray) -> SafetyStatus:
        # 1. Contact loss detection
        # 2. Extreme pitch/roll (>20 degrees)
        # 3. CoM outside support polygon (<1cm margin)
        # 4. Divergence runaway (>15cm)
        # 5. NaN detection
```

### Fallback Behaviors

```yaml
safety:
  emergency_stop: freeze_current_posture
  emergency_recovery: max_wbc_authority, disable_momentum
  aggressive_recovery: boost_capture_point_gains
  capture_point_emergency: wheel_only_stabilization
  state_estimation_failure: fallback_to_step5_25
```

### Graceful Degradation

If centroidal state estimation fails, fall back to Step 5.25 baseline controller.

## 10. Testing and Validation Approach

### Unit Tests

```python
# tests/test_centroidal_state_estimator.py
test_com_extraction_from_mjx()
test_capture_point_computation()
test_capture_point_height_dependency()

# tests/test_momentum_coordinator.py
test_momentum_damping_deadband()
test_feedforward_height_transition()
test_contact_recovery_asymmetry()

# tests/test_centroidal_wbc.py
test_wbc_authority_budget()
test_com_deadband_control()
test_hierarchical_fusion()
```

### Integration Tests

```python
# tests/test_centroidal_controller_integration.py
test_no_nan_rollout()
test_action_bounds()
test_safety_fallback()
test_step5_25_parity()
```

### Full Evaluation Protocol

```yaml
evaluation:
  episodes_per_candidate: 50
  height: 0.60  # meters
  max_episode_length: 1000  # 20 seconds at 50Hz
  
  baseline_config: step5_26_baseline.yaml  # Step 5.25
  
  candidates:
    - step5_26_baseline.yaml
    - step5_26_candidate_1_com.yaml
    - step5_26_candidate_2_com_cp.yaml
    - step5_26_candidate_3_momentum.yaml
    - step5_26_candidate_4_feedforward.yaml
    - step5_26_candidate_5_contact.yaml
    - step5_26_final_full_stack.yaml
  
  target_survival: 20.0  # seconds
```

## 11. Implementation Phases

### Phase 1: Infrastructure (Week 1)

**Tasks:**
- Create CentroidalStateEstimator class
- Implement height-dependent LIP capture point computation
- Fix MuJoCo contact force extraction (Step 5.25's 0.0% contact issue)
- Add unit tests for state estimation and capture point
- Verify no NaN rollouts

**Validation:**
- Unit tests pass
- Contact forces extracted correctly (non-zero values)
- Capture point computation matches analytical LIP
- 100-step rollout produces no NaN

**Deliverables:**
- centroidal_state_estimator.py
- capture_point_estimator.py
- tests/test_centroidal_state.py

### Phase 2: Centroidal WBC Core (Week 2)

**Tasks:**
- Implement CentroidalBalanceController skeleton
- Add CoM regulation with deadband control
- Add capture point tracking
- Integrate with existing height IK and roll stabilization
- Implement 60% authority budget clipping

**Validation:**
- Candidate 1 (CoM only): survival > 0.87s
- Candidate 2 (CoM + CP): survival > Candidate 1
- WBC authority remains >60%
- Saturation <5%

**Deliverables:**
- centroidal_balance_controller.py
- step5_26_candidate_1_com.yaml
- step5_26_candidate_2_com_cp.yaml

### Phase 3: Momentum Coordinator (Week 3)

**Tasks:**
- Implement MomentumCoordinator class
- Add momentum damping with deadband
- Add feedforward compensation for height transitions
- Add contact-aware recovery
- Implement 20% authority budget

**Validation:**
- Candidate 3 (+ momentum): survival > Candidate 2
- Candidate 4 (+ feedforward): survival > Candidate 3
- Candidate 5 (+ contact): survival > Candidate 4
- Momentum authority ~20%

**Deliverables:**
- momentum_coordinator.py
- step5_26_candidate_3_momentum.yaml
- step5_26_candidate_4_feedforward.yaml
- step5_26_candidate_5_contact.yaml

### Phase 4: Full Integration (Week 4)

**Tasks:**
- Integrate posture regularization (20% authority)
- Implement hierarchical fusion (60/20/20 split)
- Add safety monitoring and fallback
- Full ablation study (7 candidates × 50 episodes)
- Generate all required artifacts

**Validation:**
- Final candidate: survival >20s (target)
- WBC authority >60%
- Saturation <5%
- All dynamic balance metrics meet criteria
- Ablation shows clear incremental value

**Deliverables:**
- step5_26_final_full_stack.yaml
- phase_b9_step5_26_centroidal_balance_evaluation.py
- All outputs/ artifacts
- step5_26_summary.json

## 12. Summary and Key Decisions

### Architectural Decisions

1. **Approach 2 (Integrated Centroidal WBC)** - Unified centroidal-aware WBC
2. **Simplified CoM-based dynamics** - CoM + body angular momentum only
3. **Height-dependent LIP** - Capture point with ω(h) = √(g/h)
4. **Hybrid damping + feedforward** - Reactive + proactive stabilization
5. **MuJoCo contact forces** - Direct extraction from MJX data
6. **Soft deadband control** - Allow natural sway (±2-3cm CoM, ±5cm CP)
7. **60/20/20 authority allocation** - WBC/momentum/posture
8. **Sequential additive ablation** - Build from Step 5.25 baseline

### Critical Success Criteria

1. **Survival**: >20 seconds mean survival time
2. **WBC authority**: >60% maintained
3. **Low saturation**: <5% saturation rate
4. **Controlled sway**: CoM drift RMS <0.05m, CP error <0.08m
5. **Dynamic behavior**: Torque intermittency >30%
6. **Smooth recovery**: Recovery time <0.5s

### Risk Mitigation

- Phased implementation with validation gates
- Safety fallback to Step 5.25 baseline
- Ablation study to identify component contributions
- Comprehensive unit and integration tests

### Expected Timeline

4 weeks (1 week per phase)

---

**Design Status:** Complete and approved  
**Next Step:** Create implementation plan via writing-plans skill
