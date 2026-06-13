# Phase B.9 Step 5.23 — Full Configuration Equivalence Audit

## Executive Summary

**ROOT CAUSE IDENTIFIED**: Step 5.18c and Step 5.22 use fundamentally different control architectures, making the performance comparison invalid.

- **Step 5.18c (0.86s survival)**: WBC gains → position targets → PID position control → torques
- **Step 5.22 (0.68s survival)**: WBC gains → direct torque commands → torques

The 21% performance gap is NOT due to configuration mismatch, but due to **control semantics mismatch**.

---

## Control Architecture Comparison

### Step 5.18c Actual Control Path

**Script**: `scripts/phase_b9_step5_18c_torque_gain_saturation_calibration.py`

**Config Loading** (line 133):
```python
config = yaml.safe_load(f)  # Loads balance.yaml as-is
```

**balance.yaml Low-Level Mode**:
```yaml
# NOT SPECIFIED - defaults to "pid_position_velocity"
```

**BalanceEnv Default** (balance_env.py:138):
```python
self._low_level_mode = str(ll_cfg.get("mode", "pid_position_velocity"))
```

**Action Processing** (step5_18c script line 265):
```python
residual_action = compute_torque_residual_action(obs_np, candidate)
# WBC gains produce normalized values in [-1, 1]
```

**Env Step Call** (step5_18c script line 270):
```python
state = env.step(state, action)
# Passes action directly, does NOT set state.info["torque_residual_action"]
```

**Actual Control Path** (balance_env.py:441-458):
```python
if self._pid_enabled:
    if self._pid_bias_disabled:
        biased_action = control_action
    else:
        biased_action = jnp.clip(control_action + self._pid_action_bias, -1.0, 1.0)
    raw_pid_ctrl, pid_integral = self._pid_low_level_ctrl(
        state.mjx_data,
        biased_action,
        state.info["pid_integral"],
    )
```

**Control Semantics**:
- Action interpreted as **POSITION TARGET** (normalized joint angles)
- PID controller converts position error to torque
- Implicit damping from PID derivative term
- Integral action eliminates steady-state error
- Position feedback provides restoring force

---

### Step 5.22 Actual Control Path

**Script**: `scripts/phase_b9_step5_22_torque_first_wbc_evaluation.py`

**Config Modification** (lines 93-99):
```python
config["low_level_control"] = {
    "mode": "torque_first_wbc",
    "torque_control": {
        "enabled": True,
        "max_ctrl_fraction": 1.0,
    },
}
```

**Action Processing** (step5_22 script line 153):
```python
wbc_action = controller.compute_torque(obs_np)
# WBC gains produce normalized TORQUE commands
```

**State Info Setting** (step5_22 script lines 156-158):
```python
state = state._replace(
    info={**state.info, "torque_residual_action": jnp.array(wbc_action)}
)
```

**Actual Control Path** (balance_env.py:420-439):
```python
elif self._low_level_mode == "torque_first_wbc" and self._torque_control_enabled:
    from wheeled_biped.sim.low_level_control import torque_first_wbc_control
    
    scaled_action = torque_first_wbc_control(
        state.mjx_data,
        state.info["torque_residual_action"],
        self._ctrl_min,
        self._ctrl_max,
        wbc_authority_fraction=self._torque_max_ctrl_fraction,
        damping_gain=0.0,  # No damping by default
        smoothing_alpha=0.0,  # No smoothing by default
        prev_ctrl=None,
    )
```

**Control Semantics**:
- Action interpreted as **TORQUE COMMAND** (normalized motor torques)
- Direct torque application to actuators
- No position feedback
- No implicit damping (damping_gain=0.0)
- No integral action
- No restoring force from position error

---

## Parameter-by-Parameter Comparison

| Parameter | Step 5.18c | Step 5.22 | Match? |
|-----------|------------|-----------|--------|
| **Control Architecture** |
| low_level_mode | `pid_position_velocity` (default) | `torque_first_wbc` (explicit) | ❌ **MISMATCH** |
| Control semantics | Position targets | Torque commands | ❌ **FUNDAMENTAL** |
| PID enabled | `true` | `false` (bypassed) | ❌ |
| **Environment Config** |
| height | 0.60 m (fixed) | 0.60 m (fixed) | ✅ |
| num_episodes | 5 | 5 | ✅ |
| max_steps | 60 | 60 | ✅ |
| random_seeds | 42+ep | 42+ep | ✅ |
| **Action Processing** |
| action_smoothing_alpha | 0.5 (from balance.yaml) | 0.0 (torque_first_wbc default) | ❌ |
| action_delay_steps | 0 | 0 | ✅ |
| **WBC Gains** |
| k_roll | 20.0 | 20.0 | ✅ |
| k_roll_rate | 2.0 | 2.0 | ✅ |
| k_pitch | 5.0 | 5.0 | ✅ |
| k_pitch_rate | 0.5 | 0.5 | ✅ |
| **Torque Scaling** |
| max_ctrl_fraction | 0.5 (strong_k20) | 1.0 (moderate_k15) | ❌ |
| **Wheel Torque** |
| allow_wheel_torque | false | false (best candidate) | ✅ |

---

## Why Step 5.18c Performed Better

**Step 5.18c benefits from PID position control**:

1. **Implicit Damping**: PID derivative term (kd) provides velocity damping
   - Hip roll kd = 3.0
   - Hip pitch kd = 4.0
   - Knee kd = 4.0
   - Damping stabilizes oscillations

2. **Integral Action**: PID integral term (ki) eliminates steady-state error
   - Hip roll ki = 0.8
   - Hip pitch ki = 1.0
   - Knee ki = 1.0
   - Prevents drift accumulation

3. **Position Feedback**: Position error creates restoring force
   - Large position errors → large corrective torques
   - Automatic stabilization around target pose

4. **Proportional Gain Amplification**: PID proportional term (kp) amplifies WBC commands
   - Hip roll kp = 55.0
   - Hip pitch kp = 70.0
   - Knee kp = 70.0
   - WBC position targets get multiplied by high kp gains

5. **Action Smoothing**: alpha=0.5 filters high-frequency noise
   - Reduces control jitter
   - Improves stability

---

## Why Step 5.22 Underperformed

**Step 5.22 lacks stabilizing mechanisms**:

1. **No Position Feedback**: Pure torque control has no restoring force
   - Robot can drift away from nominal pose
   - No automatic correction for accumulated errors

2. **No Implicit Damping**: damping_gain=0.0
   - Oscillations not suppressed
   - Energy not dissipated

3. **No Integral Action**: No mechanism to eliminate steady-state error
   - Small biases accumulate over time
   - Robot drifts from target orientation

4. **WBC Gains Alone Insufficient**: Proportional gains (k_roll=20, k_pitch=5) too weak
   - For 2° roll error: torque = 20 * 0.035 rad = 0.7 (normalized)
   - Physical torque = 0.7 * 15 Nm = 10.5 Nm
   - Insufficient for rapid stabilization

5. **No Action Smoothing**: smoothing_alpha=0.0
   - High-frequency noise not filtered
   - Control jitter destabilizes robot

---

## Comparison Validity

**VERDICT**: **INVALID COMPARISON**

Step 5.18c and Step 5.22 are testing fundamentally different control architectures:
- **Position control** (Step 5.18c) vs **Torque control** (Step 5.22)

This is equivalent to comparing:
- A car with power steering vs a car without power steering
- A thermostat-controlled heater vs a manual heater
- A cruise-control car vs a manual-throttle car

The performance difference (0.86s vs 0.68s) reflects the architectural difference, not a configuration bug.

---

## What Step 5.18c Actually Demonstrated

**Claim**: "Pure WBC with 100% authority"

**Reality**: "WBC-generated position targets with PID position control"

**Authority Breakdown**:
- WBC: Generates position targets based on orientation error
- PID: Converts position targets to torques with kp/ki/kd gains
- **Effective authority**: WBC sets targets, PID provides stabilization

**This is NOT pure WBC torque control** - it's a hybrid architecture where:
1. WBC computes desired joint angles
2. PID tracks those angles with position feedback
3. PID gains (kp=55-70, ki=0.8-1.0, kd=3-4) provide the actual control authority

---

## Correct Comparisons Needed

To validate the torque-first architecture, we need apples-to-apples comparisons:

### Option A: Both Use Position Control
```python
# Step 5.18c (baseline)
low_level_mode = "pid_position_velocity"
action = wbc_position_targets

# Step 5.22 (modified)
low_level_mode = "pid_position_velocity"
action = wbc_position_targets
```

### Option B: Both Use Torque Control
```python
# Step 5.18c (modified)
low_level_mode = "motor_torque"
action = wbc_torque_commands

# Step 5.22 (baseline)
low_level_mode = "torque_first_wbc"
action = wbc_torque_commands
```

### Option C: Add Stabilization to Torque Control
```python
# Step 5.22 (enhanced)
low_level_mode = "torque_first_wbc"
damping_gain = 0.5  # Add velocity damping
smoothing_alpha = 0.5  # Add action smoothing
# Increase WBC gains to compensate for lack of PID amplification
k_roll = 40.0  # 2x stronger
k_pitch = 10.0  # 2x stronger
```

---

## Recommended Path Forward

**Immediate Action**: Clarify the canonical architecture

**Question**: What is the TRUE target architecture for Phase B.9?

### If Target = Position Control (WBC generates position targets)
- Step 5.18c is already correct
- Step 5.22 should be abandoned or converted to position control
- Continue with Step 5.18c as baseline for Step 6

### If Target = Torque Control (WBC generates torque commands)
- Step 5.18c needs to be re-run with `motor_torque` mode
- Step 5.22 needs stabilization enhancements (damping, smoothing, stronger gains)
- Expect lower performance than position control (torque control is harder)

### If Target = Hybrid (Position + Torque Residuals)
- This is the `hybrid_pid_plus_torque` mode
- Already analyzed in Step 5.21 (PID suppresses WBC)
- Not recommended due to authority conflict

---

## Step 6 Implications

**Current Status**: BLOCKED

**Gate Requirement**: 3.8167s survival (reset-fixed baseline)

**Current Best**:
- Step 5.18c: 0.86s (position control)
- Step 5.22: 0.68s (torque control)

**Gap**: 2.96s improvement needed (78% improvement required)

**Outlook**:
- Position control (Step 5.18c) is closer to gate (2.96s gap)
- Torque control (Step 5.22) is further from gate (3.14s gap)
- **Neither architecture is close to Step 6 gate**
- PPO residual learning will be required regardless of choice

---

## Conclusion

The 21% performance degradation between Step 5.18c (0.86s) and Step 5.22 (0.68s) is caused by **fundamental control architecture mismatch**, not configuration details.

**Step 5.18c** uses position control with PID stabilization (implicit damping, integral action, position feedback).

**Step 5.22** uses pure torque control without stabilization (no damping, no integral action, no position feedback).

**The comparison is invalid** because they test different control paradigms.

**Recommended next action**: Decide on canonical architecture (position vs torque control) and run valid apples-to-apples comparison before proceeding to Step 6.
