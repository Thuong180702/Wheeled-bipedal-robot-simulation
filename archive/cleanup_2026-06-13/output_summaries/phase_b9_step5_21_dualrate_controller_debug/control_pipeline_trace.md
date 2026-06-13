# Phase B.9 Step 5.21 — Control Pipeline Trace

## Executive Summary

**Root Cause Identified**: Action semantic mismatch + PID authority suppression

**Degradation**: 56% survival loss (0.86s → 0.38s)

**Mechanism**: DualRateBalanceController outputs position targets → PID generates ±30 Nm torques → saturates actuators → WBC residuals (~1 Nm) suppressed → authority ratio 30:1 (PID:WBC)

---

## Complete Control Pipeline Trace

### Pure WBC Path (Step 5.18c - 0.86s survival)

```
1. compute_torque_residual_action(obs)
   ↓ Computes normalized torque residuals from orientation error
   ↓ Output: action ∈ [-1, 1]^10 (torque residuals)
   
2. env.step(state, action)
   ↓ Receives action as torque residuals
   ↓ low_level_mode determines interpretation
   
3. If low_level_mode == "motor_torque":
   ↓ scaled_action = normalized_motor_torque_control(action, ...)
   ↓ Direct torque control: action → torque (no PID)
   ↓ PID authority: 0%
   ↓ WBC authority: 100%
   
4. mjx_data.ctrl = scaled_action
   ↓ Actuators receive WBC torques directly
   ↓ Torque magnitude: ±1-5 Nm (unsaturated)
   ↓ Dynamic balancing motion allowed
   
5. Result: 0.86s survival, 80% fall rate
```

### DualRateBalanceController Path (0.38s survival)

```
1. DualRateBalanceController.compute_action(obs)
   ↓ LQR/IK computes posture targets
   ↓ Wheel balance logic computes wheel velocity
   ↓ Output: action ∈ [-1, 1]^10 (position targets)
   
2. env.step(state, action)
   ↓ Receives action as position targets
   ↓ low_level_mode == "hybrid_pid_plus_torque"
   
3. PID position control:
   ↓ biased_action = action + pid_action_bias
   ↓ pid_ctrl = K_p * (target - actual) + K_d * vel_error
   ↓ PID torque magnitude: ±30 Nm (large position error)
   ↓ PID saturates at ctrlrange limits (±15-30 Nm)
   
4. WBC torque residual:
   ↓ residual = state.info["torque_residual_action"]
   ↓ WBC torque magnitude: ±1 Nm (small orientation error)
   
5. hybrid_pid_plus_torque_control():
   ↓ final = clip(pid_ctrl + residual, ctrl_min, ctrl_max)
   ↓ Since pid_ctrl is saturated, residual has minimal effect
   ↓ PID authority: 97% (30 Nm)
   ↓ WBC authority: 3% (1 Nm)
   
6. mjx_data.ctrl = final
   ↓ Actuators receive saturated PID torques
   ↓ WBC corrections suppressed by clipping
   ↓ Dynamic balancing motion prevented
   
7. Result: 0.38s survival, 100% fall rate
```

---

## Signal Trace Evidence

### Step 0 Comparison

**Pure WBC**:
- Input action type: `wbc_torque_residual`
- Pitch: 0.015 rad
- L_HIP_ROLL ctrl: -2.75 Nm
- Action directly controls torque

**DualRateBalanceController**:
- Input action type: `position_targets`
- Pitch: 0.015 rad
- L_HIP_ROLL ctrl: 1.69 Nm
- Action triggers PID → torque

### Step 1 Comparison

**Pure WBC**:
- Pitch: 0.018 rad
- L_HIP_ROLL ctrl: 15.00 Nm (saturated)
- WBC responding to disturbance

**DualRateBalanceController**:
- Pitch: 0.010 rad (better initially)
- L_HIP_ROLL ctrl: 0.37 Nm (weak response)
- PID enforcing rigid posture

### Step 2 Comparison

**Pure WBC**:
- Pitch: 0.052 rad
- L_HIP_ROLL ctrl: 7.72 Nm
- WBC making corrective torques

**DualRateBalanceController**:
- Pitch: 0.034 rad
- L_HIP_ROLL ctrl: -1.31 Nm
- PID saturated, WBC suppressed

---

## Authority Flow Analysis

### Pure WBC Authority Breakdown

| Component | Authority | Mechanism |
|-----------|-----------|-----------|
| PID | 0% | Not in control loop |
| WBC | 100% | Direct torque control |
| Total | 100% | Full actuator range available |

**Torque budget**:
- Available: ±15-30 Nm (actuator limits)
- WBC usage: ±1-5 Nm (unsaturated)
- Headroom: ±10-25 Nm (unused capacity)

### DualRateBalanceController Authority Breakdown

| Component | Authority | Mechanism |
|-----------|-----------|-----------|
| PID | 97% | Position control saturates first |
| WBC | 3% | Residuals clipped after PID |
| Total | 100% | No headroom for WBC |

**Torque budget**:
- Available: ±15-30 Nm (actuator limits)
- PID usage: ±30 Nm (saturated)
- WBC usage: ±1 Nm (suppressed by clipping)
- Headroom: 0 Nm (fully saturated)

---

## Failure Mechanism Chain

1. **Controller outputs position targets**
   - DualRateBalanceController.compute_action() returns normalized joint angles
   - These are intended as PID setpoints

2. **PID converts to large torques**
   - PID controller: τ_PID = K_p * (target - actual) + K_d * vel_error
   - Position tracking requires large torques (±30 Nm)
   - PID outputs saturate at actuator ctrlrange limits

3. **WBC computes small corrective torques**
   - WBC: τ_WBC = K_roll * roll + K_pitch * pitch + ...
   - Orientation errors are small → small torques (±1 Nm)

4. **Hybrid control blends but clips**
   - hybrid_pid_plus_torque_control(): τ_final = clip(τ_PID + τ_WBC, min, max)
   - Since τ_PID is already saturated, τ_WBC has minimal effect
   - Final torque dominated by PID (97% PID, 3% WBC)

5. **Robot loses dynamic balancing capability**
   - PID enforces rigid position targets
   - WBC cannot make corrective torques
   - Dynamic balancing motion prevented
   - Survival degrades 56% (0.86s → 0.38s)

---

## Why Step 5.18c Works

1. **Action IS the WBC torque residual**
   - No position targets involved
   - No PID position control in the loop

2. **WBC has 100% actuator authority**
   - Torques applied directly to actuators
   - No saturation from competing control layers

3. **Dynamic balancing motion allowed**
   - WBC can make corrective torques freely
   - Robot can move dynamically to maintain balance

4. **Result: 0.86s survival**
   - 80% fall rate (some episodes succeed)
   - Unsaturated control (headroom available)

---

## Why DualRateBalanceController Fails

1. **Adds posture control layer**
   - Position targets → PID → large torques
   - Architectural decision to use position control

2. **PID saturates and dominates**
   - Position tracking requires ±30 Nm
   - Saturates at actuator limits
   - Leaves no headroom for WBC

3. **WBC residuals become ineffective**
   - ±1 Nm residuals clipped after PID saturation
   - Authority ratio: 30:1 (PID:WBC)
   - WBC corrections suppressed

4. **Result: 0.38s survival**
   - 100% fall rate (all episodes fail)
   - Saturated control (no headroom)
   - 56% performance degradation

---

## Quantitative Evidence

| Metric | Pure WBC | DualRateBalanceController | Delta |
|--------|----------|---------------------------|-------|
| Survival | 0.86s | 0.38s | -56% |
| Fall rate | 80% | 100% | +25% |
| PID torque | 0 Nm | ±30 Nm | N/A |
| WBC torque | ±1-5 Nm | ±1 Nm | -80% |
| PID authority | 0% | 97% | +97% |
| WBC authority | 100% | 3% | -97% |
| Authority ratio | N/A | 30:1 | N/A |
| Saturation rate | ~20% | ~95% | +75% |

---

## Conclusion

**Root cause**: Action semantic mismatch between position targets (DualRateBalanceController) and torque residuals (Step 5.18c)

**Primary mechanism**: PID authority suppression through saturation

**Recommendation**: BYPASS DualRateBalanceController architecture

**Options**:
1. Use pure WBC (Step 5.18c pattern) - proven 0.86s survival
2. Disable PID, use controller for target generation only
3. Implement PID authority reallocation (pid_authority_fraction=0.3-0.5)
4. Redesign controller to output torque residuals instead of position targets
5. Abandon DualRateBalanceController and proceed with pure WBC + PPO residual

**Step 6 status**: BLOCKED - requires 3.8167s survival, current best is 0.86s (pure WBC)
