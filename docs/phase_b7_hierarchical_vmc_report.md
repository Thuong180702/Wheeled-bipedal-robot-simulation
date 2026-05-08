# Phase B.7: Hierarchical VMC+LQR Controller Evaluation Report

**Date:** 2026-05-07  
**Status:** EVALUATION COMPLETE - REJECTED  
**Baseline:** Height-scheduled dynamic LQR/IK (Phase B.6)  
**Candidate:** Hierarchical VMC+LQR (Phase B.7)  
**Reference:** Geometric LQR (Phase B.5)

---

## Executive Summary

**Objective:** Evaluate whether hierarchical VMC+LQR controller provides sufficient improvement over the Phase B.6 baseline to warrant adoption as the new classical prior for residual PPO training.

**Decision Rule:** Adopt if candidate achieves ANY of:
- ≥ +20% survival time improvement
- ≥ +10 percentage points fall rate reduction
- ≥ +20% pitch RMS reduction

**Result:** Hierarchical VMC+LQR controller **FAILED** all adoption criteria:
- Survival time: **-81.7%** (0.31s vs 1.71s baseline) — criterion NOT met
- Fall rate: **+10.0 pp** (100% vs 90% baseline) — criterion NOT met (worse, not better)
- Pitch RMS: **+14.2%** (26.47° vs 23.17° baseline) — criterion NOT met (worse, not better)

**Decision:** **REJECT** — Hierarchical VMC+LQR controller performs significantly worse than the Phase B.6 baseline across all metrics. The Phase B.6 height-scheduled dynamic LQR/IK prior remains the current best classical controller for residual PPO training.

---

## 1. Motivation

### 1.1 Phase B.6 Baseline Limitations

The height-scheduled dynamic LQR/IK prior (Phase B.6) achieved +121% survival time improvement over the geometric baseline but exhibited several limitations:

1. **Simulator-dependent CoM computation** - Uses MuJoCo `subtree_com`, not hardware-ready
2. **No wheel saturation compensation** - Cannot detect or respond to actuator limits
3. **Rigid height tracking** - IK provides fixed posture without CoM awareness
4. **Stateless LQR** - No integral action or disturbance estimation
5. **Limited roll/yaw authority** - Minimal lateral and heading stabilization
6. **No disturbance awareness** - Purely reactive, no feedforward compensation

### 1.2 Hierarchical VMC+LQR Architecture

The Phase B.7 candidate addresses these limitations through a 4-layer hierarchical control architecture:

**Layer 1: Posture/Height IK**
- Geometric inverse kinematics for height tracking
- Lookup table built via grid search over joint space
- Provides base leg configuration (hip_pitch, knee)

**Layer 2: CoM/Posture VMC**
- Virtual Model Control with spring-damper force model
- Corrects CoM error relative to wheel contact point
- Maps virtual force to leg joint adjustments
- Parameters: k_com=150 N/m, k_com_dot=30 N·s/m, max_force=50 N

**Layer 3: Wheel Balance LQR**
- 6D state: [pitch, pitch_rate, fwd_vel, fwd_pos, com_error, com_rate]
- Height-scheduled gains across 7 heights (0.40-0.70m)
- Wheel command filtering (alpha=0.7, max_delta=2.0)

**Layer 4: Roll/Yaw Stabilization**
- Roll correction via hip_roll (kp=2.0, kd=0.4)
- Yaw correction via differential wheel velocity (kp=3.0, kd=0.3)

### 1.3 Theoretical Advantages

**CoM feedback integration:**
- VMC layer explicitly corrects CoM error
- Provides anticipatory posture adjustments
- Reduces reliance on reactive wheel corrections

**Hierarchical decomposition:**
- Separates height tracking, CoM correction, balance, and stabilization
- Each layer operates at appropriate timescale
- Reduces coupling between control objectives

**Enhanced disturbance rejection:**
- CoM velocity damping in VMC layer
- Wheel command filtering reduces high-frequency oscillations
- Roll/yaw stabilization improves lateral robustness

---

## 2. Implementation

### 2.1 Controller Components

**Files created:**
- `wheeled_biped/controllers/hierarchical_vmc_lqr.py` - Main controller implementation
- `configs/controllers/hierarchical_vmc_lqr.yaml` - Configuration file
- `wheeled_biped/controllers/qp_allocator.py` - Optional QP-based action allocator
- `scripts/tune_hierarchical_vmc.py` - Automatic hyperparameter tuning

**Tests created:**
- `tests/test_hierarchical_vmc_lqr.py` - Controller unit tests
- `tests/test_qp_allocator.py` - QP allocator tests
- `tests/test_telemetry.py` - Telemetry system tests

### 2.2 Telemetry System

Enhanced evaluation infrastructure for Phase B.7:

**TelemetrySnapshot** - Per-timestep data:
- Pitch, roll, CoM error/velocity
- Wheel velocity command vs actual
- Wheel saturation rate
- LQR state component contributions
- Height IK error
- Joint commands

**EpisodeTelemetry** - Per-episode aggregation:
- Survival time, fall status
- Pitch/roll/CoM RMS metrics
- Wheel saturation duration
- Failure mode classification

**Failure mode classification:**
- `pitch_oscillation` - High-frequency pitch instability
- `com_drift` - Unbounded CoM error growth
- `wheel_saturation` - Prolonged actuator saturation
- `leg_config` - Joint limit violations or IK failure
- `unknown` - Insufficient data or unclear failure

### 2.3 Evaluation Protocol

**Script:** `scripts/eval_phase_b7_comprehensive.py`

**Controllers compared:**
1. Baseline: `height_scheduled_dynamic_lqr` (Phase B.6)
2. Candidate: `hierarchical_vmc_lqr` (Phase B.7)
3. Reference: `gain_scheduled_lqr` (Phase B.5 geometric baseline)

**Test heights:** 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40 m

**Episodes per height:** 20 (configurable)

**Metrics computed:**
- Survival time (mean ± std)
- Fall rate
- Pitch RMS (degrees)
- Roll RMS (degrees)
- CoM error RMS (meters)
- Wheel saturation duration (seconds)

---

## 3. Results

### 3.1 Per-Height Performance

| Height (m) | Baseline Survival (s) | Candidate Survival (s) | Reference Survival (s) | Baseline Pitch RMS (°) | Candidate Pitch RMS (°) |
|------------|----------------------|------------------------|------------------------|------------------------|-------------------------|
| 0.70       | 1.91 ± 3.27          | 0.32 ± 0.29            | 1.16 ± 2.07            | 23.20                  | 26.29                   |
| 0.65       | 1.48 ± 2.85          | 0.30 ± 0.26            | 2.01 ± 3.37            | 22.91                  | 26.43                   |
| 0.60       | 1.65 ± 2.89          | 0.29 ± 0.23            | 1.56 ± 2.83            | 23.25                  | 26.42                   |
| 0.55       | 1.75 ± 2.93          | 0.28 ± 0.20            | 1.11 ± 2.06            | 23.08                  | 26.49                   |
| 0.50       | 1.70 ± 2.89          | 0.32 ± 0.28            | 1.58 ± 2.82            | 23.15                  | 26.17                   |
| 0.45       | 1.89 ± 2.97          | 0.34 ± 0.21            | 1.50 ± 2.84            | 23.69                  | 26.51                   |
| 0.40       | 1.56 ± 2.83          | 0.33 ± 0.18            | 1.92 ± 3.25            | 22.92                  | 26.95                   |

**Key observations:**
- Candidate controller fails catastrophically at all heights (0.28-0.34s survival vs 1.48-1.91s baseline)
- Candidate fall rate is 100% across all heights (baseline: 90%)
- Candidate pitch RMS is consistently worse (26.17-26.95° vs 22.91-23.69° baseline)
- No height shows any improvement for the candidate controller

### 3.2 Aggregate Comparison

| Metric | Baseline | Candidate | Improvement | Criterion | Met? |
|--------|----------|-----------|-------------|-----------|------|
| Survival Time (s) | 1.71 | 0.31 | **-81.7%** | >= +20% | **NO** |
| Pitch RMS (°) | 23.17 | 26.47 | **+14.2%** | >= +20% reduction | **NO** |
| Fall Rate | 90.0% | 100.0% | **+10.0 pp** | >= +10 pp reduction | **NO** |

**Verdict:** All three adoption criteria FAILED. The candidate controller performs catastrophically worse than the baseline across all metrics.

### 3.3 Failure Mode Analysis

**Baseline failure modes (Phase B.6):**
- All episodes fell within ~2 seconds
- 90% fall rate across all heights
- Failures primarily due to pitch instability and limited disturbance rejection

**Candidate failure modes (Phase B.7):**
- **100% fall rate** - every episode failed
- **Catastrophic early failure** - mean survival 0.31s (vs 1.71s baseline)
- **Consistent across heights** - no height showed improvement
- **High wheel saturation** - 0.08-0.10s saturation duration (vs 0.0-0.0005s baseline)
- **Worse pitch control** - 26.47° RMS vs 23.17° baseline

**Key observation:** The hierarchical controller does not improve upon the baseline in any scenario. Instead, it introduces systematic failures that cause immediate instability across all test conditions.

---

## 4. Analysis

### 4.1 Performance Degradation

The hierarchical VMC+LQR controller shows catastrophic performance degradation compared to the Phase B.6 baseline:

**Survival time:** The candidate achieves only 0.31s mean survival time compared to 1.71s for the baseline, representing an **81.7% reduction** in performance. This is far below the +20% improvement criterion required for adoption.

**Fall rate:** The candidate has a 100% fall rate (every episode fails) compared to 90% for the baseline. This represents a **10 percentage point increase** in fall rate, moving in the wrong direction from the -10pp reduction criterion.

**Pitch stability:** The candidate shows worse pitch control with 26.47° RMS compared to 23.17° baseline, representing a **14.2% increase** in pitch error rather than the required 20% reduction.

**Wheel saturation:** The candidate exhibits high wheel saturation (0.08-0.10s duration) compared to near-zero saturation in the baseline (0.0-0.0005s), suggesting the controller is commanding infeasible wheel velocities.

### 4.2 Root Cause Hypothesis

The catastrophic failure suggests fundamental issues in the controller implementation rather than simple parameter tuning problems:

1. **Sign errors:** Pitch correction, CoM correction, or wheel command signs may be inverted, causing the controller to amplify disturbances rather than reject them.

2. **Unit mismatches:** Height error, CoM error, or force calculations may have unit conversion errors (m vs cm, rad vs deg, N vs Nm).

3. **VMC layer instability:** The Virtual Model Control layer may be generating excessive or oscillatory leg corrections that destabilize the system rather than improving CoM tracking.

4. **Action composition conflicts:** The hierarchical layers (IK + VMC + LQR + roll/yaw) may be fighting each other, with corrections from one layer being counteracted by another.

5. **Wheel command saturation:** The LQR layer may be commanding wheel velocities that immediately saturate, causing the controller to lose balance authority.

### 4.3 CoM Feedback Ineffectiveness

The original hypothesis was that explicit CoM feedback through the VMC layer would improve balance performance. The evaluation results strongly reject this hypothesis:

- The baseline controller (without explicit CoM feedback) achieves 5.5× longer survival time
- The candidate controller (with VMC CoM correction) fails immediately across all heights
- High wheel saturation in the candidate suggests the CoM correction is generating excessive wheel commands

This suggests that either:
1. The CoM feedback implementation has critical bugs
2. The VMC layer parameters are severely mistuned
3. The hierarchical decomposition introduces harmful coupling between layers
4. Explicit CoM feedback is not beneficial for this system (contradicting the original hypothesis)

### 4.4 Computational Cost

Despite the performance failure, the computational cost analysis remains valid:

**Controller complexity:**
- Baseline: 6D LQR + height IK + wheel filtering (~50 floating-point ops)
- Candidate: 4-layer hierarchy (IK + VMC + LQR + roll/yaw) (~150 floating-point ops, 3× increase)

**Real-time feasibility:**
- Control frequency: 50 Hz (20ms period)
- Estimated compute time: <1ms (well within budget)
- Hardware deployment: computationally feasible, but performance makes this irrelevant

The 3× computational cost increase would be acceptable if the controller provided performance benefits, but given the catastrophic failure, the added complexity is unjustified.

---

## 5. Limitations and Caveats

### 5.1 Simulator-Only Validation

**CoM computation:**
- Uses MuJoCo `subtree_com` (simulator-only)
- Hardware deployment requires state estimation
- Options: IMU integration, kinematic estimation, or vision-based

**Wheel contact assumption:**
- Assumes continuous ground contact
- No slip detection or compensation
- May fail on low-friction or uneven terrain

### 5.2 Tuning and Generalization

**Hyperparameter sensitivity:**
- VMC gains (k_com, k_com_dot) hand-tuned
- LQR gains inherited from Phase B.6
- Roll/yaw gains set heuristically
- Automatic tuning infrastructure available but not yet applied

**Height range:**
- Evaluated on 0.40-0.70m range
- Extrapolation beyond this range not validated
- IK lookup table may degrade outside training range

### 5.3 Disturbance Robustness

**Push recovery:**
- Not evaluated in Phase B.7 protocol
- VMC layer may improve push rejection via CoM damping
- Requires separate push-recovery evaluation

**Model uncertainty:**
- Friction, mass, damping variations not tested
- Robustness to parameter mismatch unknown
- Phase B.7 focuses on nominal performance only

---

## 6. Literature Context

### 6.1 Virtual Model Control

**Foundational work:**
- Pratt et al., "Virtual Model Control: An Intuitive Approach for Bipedal Locomotion", ICRA 2001
- Introduced virtual spring-damper models for legged robots
- Demonstrated on planar biped and quadruped platforms

**Key insight:**
- Map desired forces/torques to joint commands via Jacobian transpose
- Intuitive tuning through physical parameters (stiffness, damping)
- Widely adopted in humanoid and legged robot control

**Application to wheeled bipeds:**
- Novel application domain (most VMC work on legged locomotion)
- CoM correction via virtual force is natural extension
- Wheel actuation provides direct force application

### 6.2 Hierarchical Control for Wheeled Inverted Pendulums

**Relevant work:**
- Grasser et al., "JOE: A Mobile, Inverted Pendulum", IEEE TIE 2002
- Hierarchical control for two-wheeled inverted pendulum
- Separates balance, position, and heading control

**Architectural parallels:**
- Layer decomposition by timescale and objective
- Wheel velocity as primary balance actuator
- Posture control as secondary layer

**Differences:**
- JOE is single rigid body, wheeled biped has articulated legs
- Height variation adds complexity not present in JOE
- VMC layer is novel contribution for wheeled bipeds

### 6.3 Gain Scheduling and Adaptive Control

**Height-dependent dynamics:**
- Pendulum length varies with leg configuration
- Linearization point shifts with height
- Gain scheduling is standard approach (Rugh & Shamma, "Research on Gain Scheduling", Automatica 2000)

**Limitations of gain scheduling:**
- Assumes slow height variation (quasi-static)
- No stability guarantees during transitions
- VMC layer may provide smoother adaptation

---

## 7. Decision and Recommendations

### 7.1 Adoption Decision

**DECISION: REJECT**

The hierarchical VMC+LQR controller is **REJECTED** for adoption as the residual PPO prior. The evaluation results show catastrophic performance degradation across all metrics:

- **81.7% reduction** in survival time (0.31s vs 1.71s baseline)
- **100% fall rate** (vs 90% baseline)
- **14.2% increase** in pitch RMS (worse, not better)
- **High wheel saturation** indicating fundamental control issues

**Action items:**
- **KEEP** `height_scheduled_dynamic_lqr` as the current residual PPO prior
- **DO NOT** update `configs/training/balance_residual.yaml` to use `hierarchical_vmc_lqr`
- **DO NOT** proceed with residual PPO training using hierarchical controller
- **Document** Phase B.7 findings as a negative result for future reference
- **Investigate** root causes before attempting further hierarchical controller development

### 7.2 Future Work

**Before attempting further hierarchical controller development:**

1. **Root cause diagnosis (CRITICAL):**
   - Verify all sign conventions (pitch correction, CoM correction, wheel commands)
   - Check unit conversions (m vs cm, rad vs deg, N vs Nm)
   - Validate action composition (no double-addition of corrections)
   - Test each layer in isolation (IK-only, IK+VMC, IK+VMC+LQR)
   - Add telemetry logging to track layer contributions

2. **Ablation study to isolate failure:**
   - Test baseline (Phase B.6) alone
   - Test baseline + wheel filtering only
   - Test baseline + height PD only
   - Test baseline + VMC layer only
   - Test baseline + roll/yaw stabilization only
   - Identify which component(s) cause the catastrophic failure

3. **Sign and unit verification:**
   - `height_error = height_cmd - height_actual` (check sign)
   - `com_error_y` computation and sign
   - `pitch_ref_from_com` sign and magnitude
   - Hip pitch/knee correction signs from VMC
   - Wheel command sign from LQR
   - Roll correction sign
   - Yaw differential sign

4. **Alternative approaches if hierarchical control is abandoned:**
   - Automatic tuning of Phase B.6 controller (CMA-ES, Optuna)
   - Disturbance observer integration
   - Adaptive gain scheduling based on online system identification
   - Model predictive control (MPC) for height transitions
   - Direct residual PPO training without attempting to improve the prior

**Recommended immediate action:**
Proceed directly to Phase D (residual PPO training) using the Phase B.6 `height_scheduled_dynamic_lqr` prior. The Phase B.6 controller, while limited, provides a stable baseline for residual learning. Further classical controller development should be deferred until after residual PPO results are available, as the residual policy may compensate for prior limitations more effectively than manual controller tuning.

---

## 8. Reproducibility

### 8.1 Running the Evaluation

```bash
# Full evaluation (7 heights, 20 episodes each)
python scripts/eval_phase_b7_comprehensive.py \
    --baseline height_scheduled_dynamic_lqr \
    --candidate hierarchical_vmc_lqr \
    --reference gain_scheduled_lqr \
    --heights 0.70 0.65 0.60 0.55 0.50 0.45 0.40 \
    --episodes 20 \
    --output-dir outputs/phase_b7_eval

# Quick evaluation (3 heights, 5 episodes each)
python scripts/eval_phase_b7_comprehensive.py \
    --heights 0.55 0.60 0.65 \
    --episodes 5 \
    --output-dir outputs/phase_b7_eval_quick
```

### 8.2 Running Tests

```bash
# All Phase B.7 tests
pytest tests/test_hierarchical_vmc_lqr.py -v
pytest tests/test_qp_allocator.py -v
pytest tests/test_telemetry.py -v

# Skip tests requiring cvxpy (if not installed)
pytest tests/test_qp_allocator.py -v -m "not skipif"
```

### 8.3 Automatic Tuning (Optional)

```bash
# CMA-ES optimization
python scripts/tune_hierarchical_vmc.py \
    --optimizer cmaes \
    --heights 0.55 0.60 0.65 \
    --episodes 5 \
    --max-evals 100 \
    --output-dir outputs/phase_b7_tuning

# Optuna optimization
python scripts/tune_hierarchical_vmc.py \
    --optimizer optuna \
    --heights 0.55 0.60 0.65 \
    --episodes 5 \
    --n-trials 50 \
    --output-dir outputs/phase_b7_tuning
```

---

## 9. References

1. Pratt, J., Carff, J., Drakunov, S., & Goswami, A. (2001). "Virtual Model Control: An Intuitive Approach for Bipedal Locomotion". *IEEE International Conference on Robotics and Automation (ICRA)*.

2. Grasser, F., D'Arrigo, A., Colombi, S., & Rufer, A. C. (2002). "JOE: A Mobile, Inverted Pendulum". *IEEE Transactions on Industrial Electronics*, 49(1), 107-114.

3. Rugh, W. J., & Shamma, J. S. (2000). "Research on Gain Scheduling". *Automatica*, 36(10), 1401-1425.

4. Sentis, L., & Khatib, O. (2005). "Synthesis of Whole-Body Behaviors through Hierarchical Control of Behavioral Primitives". *International Journal of Humanoid Robotics*, 2(4), 505-518.

5. Kanoun, O., Lamiraux, F., & Wieber, P. B. (2011). "Kinematic Control of Redundant Manipulators: Generalizing the Task-Priority Framework to Inequality Task". *IEEE Transactions on Robotics*, 27(4), 785-792.

---

## Appendix A: Configuration Files

### A.1 Hierarchical VMC+LQR Config

See: `configs/controllers/hierarchical_vmc_lqr.yaml`

Key parameters:
- VMC: k_com=150.0, k_com_dot=30.0, max_force=50.0
- LQR: height-scheduled gains for 7 heights
- Wheel filtering: alpha=0.7, max_delta=2.0
- Roll: kp=2.0, kd=0.4
- Yaw: kp=3.0, kd=0.3

### A.2 Baseline Config

See: `configs/controllers/height_scheduled_dynamic_lqr.yaml`

Key parameters:
- LQR: 6D state with height-scheduled gains
- Wheel filtering: alpha=0.7, max_delta=2.0
- No VMC layer
- Minimal roll/yaw stabilization

---

## Appendix B: Telemetry Schema

### B.1 TelemetrySnapshot

```python
@dataclass
class TelemetrySnapshot:
    time: float
    pitch_deg: float
    pitch_rate_deg_s: float
    roll_deg: float
    com_error_y_m: float
    com_vel_y_m_s: float
    wheel_vel_cmd_rad_s: float
    wheel_vel_actual_rad_s: float
    wheel_saturation_rate: float
    lqr_pitch_contrib: float
    lqr_pitch_rate_contrib: float
    lqr_fwd_vel_contrib: float
    lqr_com_contrib: float
    lqr_com_rate_contrib: float
    height_cmd_m: float
    height_actual_m: float
    height_ik_error_m: float
    hip_pitch_cmd_rad: float
    knee_cmd_rad: float
```

### B.2 EpisodeTelemetry

```python
@dataclass
class EpisodeTelemetry:
    episode_id: int
    height_cmd_m: float
    survival_time_s: float
    fell: bool
    failure_mode: str
    failure_reason: str
    snapshots: list[TelemetrySnapshot]
    pitch_rms_deg: float
    roll_rms_deg: float
    com_error_rms_m: float
    wheel_saturation_duration_s: float
```

---

**Report Status:** EVALUATION COMPLETE - REJECTED  
**Decision:** Hierarchical VMC+LQR controller REJECTED due to catastrophic performance degradation (-81.7% survival time)  
**Next Action:** Proceed to Phase D residual PPO training using Phase B.6 `height_scheduled_dynamic_lqr` prior  
**Contact:** See CLAUDE.md for project guidelines and contribution instructions
