# Phase B.9 Step 5.25 — Hierarchical Task-Priority Torque Fusion Architecture

## Executive Summary

Step 5.24 failed because naive additive fusion (`tau_total = tau_wbc + tau_damping + tau_impedance`) allowed stabilization components to overwhelm WBC authority (11.3% vs >70% requirement).

This step implements **hierarchical task-priority fusion** where:
- WBC has guaranteed minimum authority
- Stabilization acts only in nullspace or with explicit budgets
- State-dependent activation prevents continuous interference
- Contact-aware logic adapts to robot state

---

## Architecture Overview

### Priority Hierarchy

```
Level 1: Balance-Critical WBC Torque (guaranteed authority)
    ↓
Level 2: Contact Stabilization (state-dependent, contact-aware)
    ↓
Level 3: Velocity Damping (oscillation-triggered, not continuous)
    ↓
Level 4: Posture Regularization (weak, nullspace-only)
```

### Control Flow

```
obs → WBC gains → tau_wbc_desired
                      ↓
                  authority_budget(tau_wbc_desired, ctrl_limits)
                      ↓
                  tau_wbc_allocated (guaranteed minimum authority %)
                      ↓
                  remaining_budget = ctrl_limits - tau_wbc_allocated
                      ↓
    ┌─────────────────┴─────────────────┐
    ↓                                   ↓
contact_stabilization              velocity_damping
(if contact_asymmetry > thresh)    (if oscillation_detected)
    ↓                                   ↓
tau_contact (clipped to budget)    tau_damping (clipped to budget)
    ↓                                   ↓
    └─────────────────┬─────────────────┘
                      ↓
              posture_regularization
              (weak, nullspace-only)
                      ↓
              tau_posture (clipped aggressively)
                      ↓
          tau_total = tau_wbc + tau_contact + tau_damping + tau_posture
                      ↓
                  clip(tau_total, ctrl_min, ctrl_max)
```

---

## Component Specifications

### 1. Explicit Authority Allocation

**WBC Authority Budget:**
```python
wbc_authority_min = 0.60  # guaranteed minimum 60%
wbc_budget = ctrl_limit * wbc_authority_min
tau_wbc_allocated = clip(tau_wbc_desired, -wbc_budget, wbc_budget)
remaining_budget = ctrl_limit - abs(tau_wbc_allocated)
```

**Stabilization Budget:**
```python
stabilization_budget = remaining_budget * 0.8  # use 80% of remaining
posture_budget = remaining_budget * 0.2        # use 20% of remaining
```

**Key Invariant:**
```
|tau_wbc| >= 60% * ctrl_limit  (always)
|tau_wbc + tau_stabilization| <= 100% * ctrl_limit
```

### 2. State-Dependent Stabilization

**Contact Stabilization (Level 2):**
```python
# Only activate when contact asymmetry detected
left_load = contact_force_left / (contact_force_left + contact_force_right)
right_load = 1.0 - left_load
contact_asymmetry = abs(left_load - 0.5)

if contact_asymmetry > 0.15:  # >15% asymmetry
    # Apply corrective hip roll torque
    tau_contact = contact_stabilization_gain * (left_load - right_load)
    tau_contact = clip(tau_contact, -stabilization_budget, stabilization_budget)
else:
    tau_contact = 0.0
```

**Velocity Damping (Level 3):**
```python
# Only activate during oscillation
joint_vel_rms = sqrt(mean(joint_vel**2))
oscillation_detected = joint_vel_rms > oscillation_threshold

if oscillation_detected:
    tau_damping = -damping_gain * joint_vel
    tau_damping = clip(tau_damping, -stabilization_budget, stabilization_budget)
else:
    tau_damping = 0.0
```

**Posture Regularization (Level 4):**
```python
# Only activate when WBC error is small
wbc_error_norm = sqrt(sum((tau_wbc_desired)**2))
wbc_error_normalized = wbc_error_norm / (num_joints * ctrl_limit)

if wbc_error_normalized < 0.3:  # WBC error < 30% of capacity
    pos_error = nominal_pose - current_pose
    tau_posture = impedance_kp * pos_error
    tau_posture = clip(tau_posture, -posture_budget, posture_budget)
else:
    tau_posture = 0.0  # disable during recovery
```

### 3. Contact-Aware Control

**Contact Force Estimation:**
```python
# Use vertical ground reaction forces from MuJoCo contact sensors
left_foot_contact = sum(contact_forces[left_foot_geoms])
right_foot_contact = sum(contact_forces[right_foot_geoms])

# Detect unloading (roll-induced)
left_unloaded = left_foot_contact < unload_threshold
right_unloaded = right_foot_contact < unload_threshold
```

**Corrective Redistribution:**
```python
if left_unloaded and not right_unloaded:
    # Left foot unloading → apply corrective left hip roll torque
    tau_contact[l_hip_roll] = +contact_correction_gain
elif right_unloaded and not left_unloaded:
    # Right foot unloading → apply corrective right hip roll torque
    tau_contact[r_hip_roll] = -contact_correction_gain
```

### 4. Dynamic Balance Design

**Desired Behavior:**
- Soft sway (not rigid locking)
- Intermittent corrections (not continuous high torque)
- Low average torque
- Recovery motion allowed
- Dynamic CoM stabilization

**Anti-Patterns to Avoid:**
- Frozen pose tracking
- Continuous high torque fighting
- Rigid impedance control
- Always-on damping

**Implementation:**
- Use state-dependent activation (stabilization only when needed)
- Use weak gains (allow natural dynamics)
- Use aggressive clipping (prevent saturation)
- Monitor torque efficiency metrics

---

## Candidate Configurations

### Ablation Study

1. **baseline_pure_wbc**
   - WBC only, no stabilization
   - Authority: 100% WBC

2. **wbc_authority_budget**
   - WBC with explicit 60% authority budget
   - Remaining 40% unused
   - Authority: 100% WBC (no stabilization active)

3. **wbc_contact_aware**
   - WBC + contact stabilization (state-dependent)
   - Authority: WBC 60-80%, contact 20-40%

4. **wbc_oscillation_damping**
   - WBC + velocity damping (oscillation-triggered)
   - Authority: WBC 60-80%, damping 20-40%

5. **wbc_contact_damping**
   - WBC + contact + damping (both state-dependent)
   - Authority: WBC 60-75%, contact 10-20%, damping 10-20%

6. **wbc_contact_damping_posture**
   - WBC + contact + damping + posture (all state-dependent)
   - Authority: WBC 60-70%, contact 10-15%, damping 10-15%, posture 5-10%

7. **hierarchical_full**
   - Full hierarchical stack with all components
   - Aggressive state-dependent activation
   - Authority: WBC 60-70%, stabilization 30-40%

8. **hierarchical_aggressive_wbc**
   - Hierarchical with 70% WBC authority minimum
   - Authority: WBC 70-80%, stabilization 20-30%

9. **hierarchical_dynamic_budget**
   - Dynamic authority reallocation based on WBC error magnitude
   - High WBC error → 80% WBC authority
   - Low WBC error → 60% WBC authority, more stabilization
   - Authority: WBC 60-80%, stabilization 20-40%

---

## Telemetry Requirements

### Authority Tracking
- `tau_wbc_rms`: RMS magnitude of WBC torque
- `tau_contact_rms`: RMS magnitude of contact stabilization
- `tau_damping_rms`: RMS magnitude of velocity damping
- `tau_posture_rms`: RMS magnitude of posture regularization
- `tau_total_rms`: RMS magnitude of total torque
- `wbc_authority_pct`: WBC torque / total torque * 100
- `contact_authority_pct`: Contact torque / total torque * 100
- `damping_authority_pct`: Damping torque / total torque * 100
- `posture_authority_pct`: Posture torque / total torque * 100

### Activation Tracking
- `contact_activation_rate`: Fraction of timesteps contact stabilization active
- `damping_activation_rate`: Fraction of timesteps damping active
- `posture_activation_rate`: Fraction of timesteps posture regularization active
- `contact_asymmetry_mean`: Mean contact asymmetry
- `oscillation_detection_rate`: Fraction of timesteps oscillation detected
- `wbc_error_mean`: Mean WBC error magnitude

### Clipping Tracking
- `wbc_clipping_rate`: Fraction of timesteps WBC torque clipped
- `contact_clipping_rate`: Fraction of timesteps contact torque clipped
- `damping_clipping_rate`: Fraction of timesteps damping torque clipped
- `posture_clipping_rate`: Fraction of timesteps posture torque clipped
- `total_saturation_rate`: Fraction of timesteps total torque saturated

### Efficiency Metrics
- `torque_efficiency`: Mean torque / max torque
- `energy_proxy`: Sum of |tau * qdot|
- `recovery_smoothness`: Std dev of torque rate
- `sway_amplitude`: Std dev of roll angle

---

## Success Criteria

### Primary Metrics
- **Survival time > 0.86s** (beat Step 5.18c)
- **WBC authority > 60%** (maintain dominance)
- **Saturation rate < 80%** (reduce from Step 5.24's 85.7%)
- **Dynamic balance behavior** (soft sway, intermittent corrections)

### Secondary Metrics
- Contact stabilization activates only during asymmetry
- Damping activates only during oscillation
- Posture regularization activates only when WBC error small
- Torque efficiency improved vs Step 5.24
- Energy usage reduced vs Step 5.24

### Failure Indicators
- WBC authority < 60%
- Continuous stabilization activation (>80% of timesteps)
- High saturation (>85%)
- Rigid pose locking behavior
- No improvement over Step 5.24

---

## Implementation Plan

1. **Enhance `torque_first_wbc_control()` in `low_level_control.py`**
   - Add explicit authority budgeting
   - Add state-dependent activation logic
   - Add contact-aware stabilization
   - Add telemetry tracking

2. **Update `balance_env.py`**
   - Add contact force extraction
   - Add oscillation detection
   - Add state-dependent parameters to info dict
   - Propagate telemetry

3. **Create evaluation script**
   - Test 9 ablation candidates
   - Track all telemetry metrics
   - Compare against Step 5.18c/5.22/5.24 baselines
   - Generate comprehensive analysis

4. **Generate artifacts**
   - `hierarchical_architecture.md` (this document)
   - `authority_budget_analysis.csv`
   - `suppression_detection.csv`
   - `state_dependent_activation.csv`
   - `contact_stabilization_analysis.csv`
   - `ablation_results.csv`
   - `step5_25_summary.json`
   - `step5_25_summary.md`

---

## Expected Outcomes

### Best Case
- Hierarchical fusion achieves >0.90s survival
- WBC authority maintained at 65-70%
- Stabilization activates intermittently (30-50% of timesteps)
- Dynamic balance behavior emerges
- Saturation reduced to <75%
- Torque-first architecture validated

### Realistic Case
- Hierarchical fusion achieves 0.80-0.85s survival
- WBC authority maintained at 60-65%
- Stabilization activates moderately (40-60% of timesteps)
- Some improvement in dynamic behavior
- Saturation reduced to 75-80%
- Torque-first architecture viable but not superior to position control

### Worst Case
- Hierarchical fusion achieves <0.80s survival
- WBC authority drops below 60%
- Stabilization activates continuously (>70% of timesteps)
- Rigid behavior persists
- Saturation remains high (>80%)
- Torque-first architecture still not viable

---

## Next Steps After Step 5.25

### If Successful (survival > 0.86s, authority > 60%)
- Accept hierarchical torque-first WBC as viable alternative to position control
- Compare position control vs torque-first WBC for Step 6 baseline
- Consider hybrid approach: position control for legs, torque control for wheels

### If Marginal (survival 0.80-0.86s, authority > 60%)
- Document hierarchical torque-first as viable but not superior
- Proceed with position control (Step 5.18c) as canonical baseline
- Archive torque-first work as alternative approach

### If Failed (survival < 0.80s or authority < 60%)
- Accept that torque-first WBC is not viable for this robot
- Proceed with position control (Step 5.18c) as canonical baseline
- Document lessons learned about authority allocation and control fusion
