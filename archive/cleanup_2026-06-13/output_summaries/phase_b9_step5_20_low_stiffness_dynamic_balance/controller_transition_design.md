# Phase B.9 Step 5.20: Low-Stiffness Dynamic Balance Transition

## Design Document

### Motivation

**Hypothesis**: The current controller is over-stiff and fighting natural balancing dynamics.

**Evidence**:
1. Pure RL previously balanced successfully without persistent max torque saturation
2. Current PID stack saturates at ±30 Nm continuously (Step 5.18c)
3. PID saturation is dominant, WBC residuals are small (~1 Nm)
4. Stronger WBC gains improved survival from 0.52s → 0.86s (Step 5.18c)
5. Authority reallocation (Step 5.19) showed marginal improvement at best

**Interpretation**: The plant is stabilizable (RL proved this), but the current classical control structure may be inefficient and over-constraining.

### Core Design Shift

**Current behavior** (pose-first):
- Rigid posture tracking with high PID gains
- Balance treated as correction on top of posture maintenance
- PID dominates authority, saturates continuously
- Small deviations trigger aggressive corrections

**Target behavior** (balance-first):
- Soft posture compliance with reduced PID gains
- Balance survival prioritized over exact pose
- Allow natural torso lean, temporary asymmetry, CoM movement
- PID becomes soft tracking layer, balance controller becomes primary

### Implementation Strategy

#### 1. Soft Dynamic Balance Mode

Add opt-in controller mode in `DualRateBalanceController`:

```yaml
soft_dynamic_balance:
  enabled: false  # disabled by default
  posture_stiffness_reduction: 0.5  # multiply PID gains by this factor
  posture_deadband_deg: 2.0  # don't correct small deviations
  posture_restore_delay_s: 0.5  # delay aggressive recentering
  balance_authority_boost: 2.0  # increase WBC authority
  allow_torso_lean: true
  allow_temporary_asymmetry: true
  max_torso_lean_deg: 10.0
  max_wheel_offset_m: 0.1
```

#### 2. Low-Stiffness Posture Tracking

**Mechanism**: Systematically reduce posture rigidity

**Implementation**:
- Multiply PID Kp, Kd by `posture_stiffness_reduction` factor
- Add deadband: don't correct if error < threshold
- Add restore delay: wait before aggressive recentering
- Softer torso restoration: gradual return to nominal pose

**Purpose**: Allow dynamic balancing motion instead of forcing exact posture

#### 3. Balance-First Prioritization

**Mechanism**: Relax posture constraints when they conflict with balance

**Implementation**:
- Wider acceptable ranges for torso lean, asymmetry, wheel offset
- Reduce correction aggressiveness near zero-crossing
- Allow temporary deviations if they improve stability

**Purpose**: Survival > exact pose tracking

#### 4. Dynamic Compliance

**Mechanism**: Smooth, gradual corrections instead of aggressive snapping

**Implementation**:
- Posture restore delay: wait before recentering
- Gradual recentering: ramp corrections over time
- Soft wheel recentering: allow drift, recover slowly
- Reduced high-frequency fighting

**Purpose**: Avoid fighting natural dynamics

#### 5. Hybrid Authority Transition

**Mechanism**: Shift authority from PID to balance controller

**Current**: Hard PID (dominant) + weak WBC residual
**Target**: Soft PID (compliant) + strong WBC (primary)

**Implementation**:
- Reduce PID gains by `posture_stiffness_reduction`
- Increase WBC authority by `balance_authority_boost`
- Smooth blending, no abrupt switching

#### 6. Torque Efficiency Telemetry

**New metrics**:
- `actuator_rms_torque`: RMS torque across all actuators
- `mean_abs_torque`: Mean absolute torque
- `saturation_fraction`: Fraction of time saturated
- `posture_error_rms`: RMS deviation from nominal pose
- `balance_energy_proxy`: Sum of |torque × velocity|
- `correction_smoothness`: Rate of change of control output

**Purpose**: Quantify efficiency improvement vs stiff PID baseline

### Evaluation Plan

#### Phase 1: h=0.60 Quick Evaluation

Test candidates:
- Baseline (soft mode disabled)
- Soft mode with stiffness_reduction = 0.7
- Soft mode with stiffness_reduction = 0.5
- Soft mode with stiffness_reduction = 0.3

Metrics:
- Survival time
- Fall rate
- Saturation rate
- RMS torque
- Roll RMS
- Posture error RMS

**Success criteria**: 
- Survival > 0.86s (Step 5.18c best)
- Saturation rate < 0.90 (Step 5.18c was 0.9375)
- RMS torque significantly lower

#### Phase 2: Full Validation

Only for best candidate from Phase 1.

Heights: 0.65, 0.60, 0.55, 0.50, 0.45, 0.40
Episodes per height: 5

**Success criteria**:
- Overall survival > 3.8167s (reset-fixed baseline)
- Saturation rate < 0.80
- Torque efficiency improved

### Expected Outcomes

**Optimistic**: Soft mode dramatically improves survival and efficiency
- Lower saturation (e.g., 50-70% vs 93%)
- Longer survival (e.g., 2-3s at h=0.60 vs 0.86s)
- Lower RMS torque (e.g., 15-20 Nm vs 30 Nm)
- Slightly worse pose tracking but much better balance

**Neutral**: Soft mode trades pose accuracy for efficiency without survival gain
- Lower saturation
- Lower RMS torque
- Similar survival
- Worse pose tracking

**Pessimistic**: Soft mode destabilizes the robot
- Faster falls
- Loss of posture control
- No efficiency gain

### Decision Framework

**If soft mode improves survival + efficiency**:
- Adopt as new baseline
- Proceed to Step 6 with soft mode enabled

**If soft mode improves efficiency but not survival**:
- Document efficiency gain
- Consider hybrid approach
- May still indicate over-constraining problem

**If soft mode degrades performance**:
- Conclude that current stiffness is necessary
- Classical control architecture may be fundamentally limited
- Consider pure RL or different architecture

### Implementation Checklist

- [ ] Add `soft_dynamic_balance` config block to controller
- [ ] Implement posture stiffness reduction
- [ ] Implement deadband logic
- [ ] Implement restore delay
- [ ] Implement balance authority boost
- [ ] Add torque efficiency telemetry
- [ ] Create evaluation script
- [ ] Create test configs
- [ ] Run h=0.60 evaluation
- [ ] Run full validation if promising
- [ ] Add tests
- [ ] Update reports

### Critical Success Factors

1. **Systematic reduction**: Not random tuning, but principled stiffness reduction
2. **Clear telemetry**: Must be able to see efficiency gains
3. **Honest evaluation**: Accept if soft mode doesn't help
4. **No heuristic explosion**: Simplification, not more layers
5. **Evidence-based conclusions**: Data drives decisions

### Risks

1. **Soft mode may destabilize**: Lower stiffness could cause faster falls
2. **May not address root cause**: Problem may be deeper than stiffness
3. **May need different architecture**: Classical control may be fundamentally limited

### Next Steps After Step 5.20

**If successful**: Proceed to Step 6 (PPO training) with soft mode as baseline

**If unsuccessful**: Consider:
- Pure WBC control (no PID)
- Pure RL (no classical control)
- Different hybrid architecture
- Architectural redesign
