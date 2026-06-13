# Phase B.9 Step 5.22 — Performance Degradation Analysis

## Executive Summary

**Unexpected Finding**: Torque-first WBC architecture achieved 0.68s survival (21% worse than Step 5.18c's 0.86s baseline), despite successfully eliminating PID authority suppression.

**Authority Distribution**:
- Step 5.18c: 100% WBC (0.86s survival)
- Step 5.22: 100% WBC (0.68s survival)
- Hybrid mode: 97% PID, 3% WBC (0.38s survival)

**Key Question**: Why did torque-first WBC underperform when both architectures have 100% WBC authority?

---

## Results Comparison

### Step 5.18c Baseline (Pure WBC)

| Metric | Value |
|--------|-------|
| Survival | 0.86s |
| Fall rate | 80% |
| Saturation | 93.75% |
| Architecture | motor_torque mode |
| WBC authority | 100% |

### Step 5.22 Torque-First WBC

| Candidate | Survival | Fall Rate | Saturation | Torque RMS |
|-----------|----------|-----------|------------|------------|
| strong_k20 | 0.64s | 100% | 80.15% | N/A |
| strong_k20_wheels | 0.45s | 100% | 84.35% | N/A |
| moderate_k15 | 0.68s | 100% | 62.67% | N/A |

**Best**: moderate_k15 with 0.68s survival (21% degradation vs baseline)

---

## Hypotheses for Performance Degradation

### Hypothesis 1: Different Low-Level Control Modes

**Step 5.18c** likely uses:
```python
low_level_mode = "motor_torque"
# Direct path: action -> normalized_motor_torque_control() -> actuators
```

**Step 5.22** uses:
```python
low_level_mode = "torque_first_wbc"
# Path: state.info["torque_residual_action"] -> torque_first_wbc_control() -> actuators
```

**Analysis**: Both should be equivalent if implemented correctly. The scaling logic is identical:
```python
# normalized_motor_torque_control()
ctrl_limit = jnp.minimum(jnp.abs(ctrl_min), jnp.abs(ctrl_max)) * fraction
ctrl = normalized_torque * ctrl_limit

# torque_first_wbc_control()
ctrl_limit = jnp.minimum(jnp.abs(ctrl_min), jnp.abs(ctrl_max)) * wbc_fraction
wbc_ctrl = normalized_wbc_torque * ctrl_limit
```

**Verdict**: Unlikely to be the cause.

### Hypothesis 2: Integration Bug

**Potential issue**: The action passed to `env.step()` might be used differently.

In Step 5.18c:
```python
action = compute_torque_residual_action(obs, candidate)
state = env.step(state, action)
# Action is directly used by motor_torque mode
```

In Step 5.22:
```python
wbc_action = controller.compute_torque(obs)
state = state._replace(info={**state.info, "torque_residual_action": jnp.array(wbc_action)})
state = env.step(state, jnp.array(wbc_action))
# Action passed to env.step() is IGNORED, only state.info["torque_residual_action"] is used
```

**Analysis**: The torque_first_wbc mode reads from `state.info["torque_residual_action"]`, not from the action parameter. This is correct for the architecture, but creates a dependency on state.info being set correctly.

**Verdict**: Integration appears correct, but worth verifying.

### Hypothesis 3: Saturation Still High

**Observation**: Saturation rates remain high:
- Step 5.18c: 93.75%
- Step 5.22 strong_k20: 80.15%
- Step 5.22 moderate_k15: 62.67%

**Analysis**: Lower saturation in Step 5.22 suggests WBC is not saturating as much, which should be GOOD. But survival is worse. This is counterintuitive.

**Possible explanation**: Lower saturation might mean WBC is not generating enough corrective torque to maintain balance.

**Verdict**: Requires deeper investigation of torque magnitudes and control effectiveness.

### Hypothesis 4: Step 5.18c Baseline Needs Re-Verification

**Question**: Does Step 5.18c actually achieve 0.86s with the same WBC controller?

**Evidence needed**:
1. Verify Step 5.18c uses motor_torque mode
2. Verify Step 5.18c uses same WBC gains (k_roll=20, k_pitch=5)
3. Verify Step 5.18c evaluation methodology

**Verdict**: Critical to verify baseline is apples-to-apples comparison.

### Hypothesis 5: Missing Smoothing or Damping

**Observation**: Step 5.22 uses:
```python
damping_gain=0.0,  # No damping by default
smoothing_alpha=0.0,  # No smoothing by default
```

**Analysis**: Step 5.18c might have implicit smoothing from the environment's action smoothing or PID integral term (even if PID is disabled).

Looking at BalanceEnv:
```python
if self._pid_enabled and self._pid_smoothing_alpha > 0.0:
    smooth_action = (
        self._pid_smoothing_alpha * state.prev_action
        + (1.0 - self._pid_smoothing_alpha) * action
    )
```

If Step 5.18c has `pid_enabled=True` with smoothing, but Step 5.22 has `pid_enabled=False`, this could explain the difference.

**Verdict**: Likely contributor - need to verify smoothing configuration.

### Hypothesis 6: Action Delay Buffer

**Observation**: BalanceEnv has action delay buffer:
```python
if self._action_delay_steps > 0:
    delayed_action = state.info["action_delay_buffer"][0]
```

**Analysis**: If Step 5.18c and Step 5.22 have different delay configurations, this could affect performance.

**Verdict**: Need to verify delay configuration is identical.

### Hypothesis 7: Wheel Torque Hurts Performance

**Observation**: strong_k20_with_wheels (0.45s) performed WORSE than strong_k20 (0.64s).

**Analysis**: Adding wheel torque for roll stabilization degraded performance by 30%. This suggests wheel torque might be destabilizing rather than helpful.

**Verdict**: Wheel torque should remain disabled for now.

---

## Critical Questions to Answer

1. **What exact low_level_mode does Step 5.18c use?**
   - Need to inspect Step 5.18c evaluation script
   - Verify motor_torque vs hybrid_pid_plus_torque

2. **What smoothing/delay configuration does Step 5.18c use?**
   - Check pid_smoothing_alpha
   - Check action_delay_steps
   - Check if PID is enabled (even with zero gains)

3. **Are WBC gains identical?**
   - Step 5.18c: k_roll=20, k_pitch=5
   - Step 5.22: k_roll=20, k_pitch=5 (strong_k20)
   - Verify these are truly identical

4. **Is the evaluation methodology identical?**
   - Same height (h=0.60)
   - Same number of episodes (5)
   - Same max steps (60)
   - Same random seeds

5. **What are the actual torque magnitudes?**
   - Step 5.18c: ±1-5 Nm (from Step 5.21 analysis)
   - Step 5.22: Need to measure actual torque RMS

---

## Recommended Next Steps

### Step 1: Verify Step 5.18c Configuration

Read Step 5.18c evaluation script to determine:
- Exact low_level_mode used
- Smoothing configuration
- Delay configuration
- WBC gains

### Step 2: Run Apples-to-Apples Comparison

Create a controlled comparison:
- Use identical environment configuration
- Use identical WBC gains
- Use identical evaluation methodology
- Measure torque magnitudes and patterns

### Step 3: Test Smoothing Hypothesis

Run Step 5.22 with smoothing enabled:
```python
damping_gain=0.0,
smoothing_alpha=0.1,  # Add light smoothing
```

### Step 4: Investigate Torque Effectiveness

Analyze time-series data:
- Torque magnitude over time
- Pitch/roll error over time
- Control effectiveness (torque → state change)

### Step 5: Consider Hybrid Approach

If pure torque-first underperforms, consider:
- Light damping (damping_gain=0.1-0.5)
- Temporal smoothing (smoothing_alpha=0.1-0.3)
- Authority reallocation (wbc_authority_fraction=0.7-0.9)

---

## Preliminary Conclusions

1. **Architecture is correct**: Eliminating PID suppression was the right move (100% WBC authority achieved)

2. **Performance gap is real**: 0.68s vs 0.86s (21% degradation) requires explanation

3. **Saturation decreased**: 62-80% vs 93.75% suggests WBC is less saturated, but this didn't improve survival

4. **Wheel torque is harmful**: Adding wheel torque degraded performance by 30%

5. **Configuration mismatch likely**: Step 5.18c and Step 5.22 probably have different smoothing/delay configurations

6. **Baseline verification needed**: Must verify Step 5.18c actually achieves 0.86s with pure WBC

---

## Impact on Step 6

**Status**: BLOCKED

**Gate requirement**: 3.8167s survival (reset-fixed baseline)

**Current best**: 0.68s (Step 5.22 torque-first WBC)

**Gap**: 2.49s improvement needed (78% improvement required)

**Outlook**: Even if we match Step 5.18c's 0.86s, we're still 2.96s short of Step 6 gate. The controller architecture alone is insufficient - PPO residual learning will be required.

---

## Conclusion

The torque-first WBC architecture successfully eliminated PID authority suppression (100% WBC authority vs 3% in hybrid mode), but unexpectedly underperformed the Step 5.18c baseline by 21%.

The most likely explanation is a configuration mismatch (smoothing, delay, or other environment settings) rather than a fundamental architectural flaw.

**Immediate action**: Verify Step 5.18c configuration and run controlled apples-to-apples comparison before drawing conclusions about the torque-first architecture.
