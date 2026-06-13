# Phase B.9 Step 5.20: Evaluation Analysis

## Executive Summary

**Status**: Evaluation INVALID - setup error detected

**Finding**: All candidates failed catastrophically, including the baseline (0.24s vs expected 0.86s). This indicates a fundamental evaluation setup error, not a valid test of the stiffness reduction hypothesis.

---

## Evaluation Results (INVALID)

| Candidate | Stiffness | Deadband | Survival | Fall Rate | Roll RMS | RMS Torque |
|-----------|-----------|----------|----------|-----------|----------|------------|
| baseline | 1.0 | 0.0° | 0.24s | 1.00 | 21.83° | 7.16 Nm |
| conservative | 0.7 | 1.0° | 0.24s | 1.00 | 21.90° | 6.85 Nm |
| moderate | 0.5 | 2.0° | 0.25s | 1.00 | 21.95° | 6.56 Nm |
| aggressive | 0.3 | 3.0° | 0.28s | 1.00 | 23.10° | 6.29 Nm |

**Expected baseline**: 0.86s survival (Step 5.18c)
**Actual baseline**: 0.24s survival (72% degradation)

---

## Root Cause: Evaluation Setup Error

### The Problem

The evaluation script creates `DualRateBalanceController` and passes its output directly to `env.step()`:

```python
# Create controller
controller = DualRateBalanceController(config, env.mj_model)

# Use controller directly
action = controller.compute_action(obs)
state = env.step(state, jnp.array(action))
```

### Why This Fails

1. **Controller output format mismatch**: `DualRateBalanceController` outputs actions in a specific format/range designed for a particular control pipeline
2. **Missing control integration**: The environment expects normalized actions that go through its low-level control (PID, smoothing, delay)
3. **Bypassed control path**: The evaluation bypasses the `hybrid_pid_plus_torque_control` path that Step 5.18c uses

### Analogy

This is like testing a car engine by connecting it directly to the wheels without the transmission. The engine might produce power, but the car won't move correctly because the power delivery system is missing.

---

## Evidence of Setup Error

### 1. Baseline Degradation

The baseline (stiffness=1.0, no reduction) should match Step 5.18c performance but doesn't:
- Expected: 0.86s survival
- Actual: 0.24s survival
- Degradation: 72%

**This proves the evaluation setup is wrong**, not that the controller is wrong.

### 2. Misleading Torque Efficiency

All candidates show "improved" torque efficiency:
- Step 5.18c: ~30 Nm RMS
- Evaluation: 6-7 Nm RMS

But this is because **the robot is falling**, not because it's balancing more efficiently. Lower torque during collapse is not an improvement.

### 3. Uniform Failure

All candidates fail uniformly (0.24-0.28s), suggesting the failure mode is independent of stiffness reduction. This indicates a common setup error, not a stiffness-dependent behavior.

---

## What Went Wrong

### Missing Integration

The `DualRateBalanceController` was designed to work within a specific control architecture:

```
obs → DualRateBalanceController → normalized_action
    → env.low_level_control (PID + WBC) → motor_commands
    → robot
```

The evaluation bypassed this:

```
obs → DualRateBalanceController → action → env.step() → ???
```

### Action Semantics Mismatch

The controller outputs actions in a format that assumes:
1. Specific normalization/scaling
2. Integration with PID control
3. Blending with WBC torque residuals
4. Smoothing and delay

The environment's `step()` method expects actions in a different format.

---

## Correct Evaluation Approach

### Option 1: Use Existing Evaluation Infrastructure

Use `scripts/eval_classical_prior_with_telemetry.py` or similar scripts that properly integrate the controller with the environment's control pipeline.

### Option 2: Fix the Evaluation Script

Modify `phase_b9_step5_20_evaluation.py` to:
1. Configure the environment to use the controller properly
2. Ensure action semantics match
3. Validate baseline performance before testing variants

### Option 3: Use Training Environment

Create a minimal training loop that uses the controller as the policy, ensuring proper integration with the environment's control pipeline.

---

## Answers to Required Questions

### 1. Did lower stiffness improve survival?

**Cannot determine** - evaluation setup is invalid. The baseline itself failed catastrophically, so comparisons are meaningless.

### 2. Did saturation decrease?

**Cannot determine** - the evaluation doesn't measure actuator saturation. The lower torque values are due to immediate collapse, not reduced saturation during stable balancing.

### 3. Did torque efficiency improve?

**No** - the lower torque is because the robot is falling, not because it's balancing more efficiently. This is a misleading metric when the robot fails immediately.

### 4. Is the controller still over-constrained?

**Cannot determine** - the evaluation setup error prevents testing this hypothesis. The controller may or may not be over-constrained, but this evaluation doesn't provide valid evidence either way.

### 5. Is behavior now closer to the previous successful pure RL behavior?

**No** - the behavior is immediate collapse, which is nothing like the previous successful pure RL behavior. This is due to the evaluation setup error, not the controller design.

---

## Conclusion

**The Step 5.20 evaluation is INVALID due to a fundamental setup error.**

The evaluation script bypasses the environment's control pipeline, causing all candidates (including the baseline) to fail catastrophically. This prevents any valid conclusions about the stiffness reduction hypothesis.

**Recommendation**: Fix the evaluation setup before attempting to test the hypothesis. The controller implementation itself may be correct, but the evaluation infrastructure is broken.

---

## Step 6 Gate Status

**BLOCKED** - remains blocked

The invalid evaluation provides no evidence for or against the stiffness reduction hypothesis. Step 6 gate requirement (beat reset-fixed baseline of 3.8167s) is not met.

---

## Next Steps

1. **Fix evaluation setup**: Ensure proper integration with environment control pipeline
2. **Validate baseline**: Confirm baseline matches Step 5.18c performance (0.86s)
3. **Re-run evaluation**: Test stiffness reduction hypothesis with valid setup
4. **If hypothesis still fails**: Consider alternative architectural approaches

---

## Technical Debt

This evaluation error highlights a broader issue: **lack of standardized controller evaluation infrastructure**. Different evaluation scripts use different integration approaches, leading to inconsistent results and setup errors.

**Recommendation**: Create a unified controller evaluation framework that ensures consistent integration with the environment's control pipeline across all evaluation scripts.
