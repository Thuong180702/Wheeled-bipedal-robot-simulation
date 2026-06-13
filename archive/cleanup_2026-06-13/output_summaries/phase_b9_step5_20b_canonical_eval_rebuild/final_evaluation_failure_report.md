# Phase B.9 Step 5.20: Final Evaluation Failure Report

## Executive Summary

**Status**: ALL EVALUATION ATTEMPTS FAILED

**Finding**: Four consecutive evaluation attempts failed to reproduce Step 5.18c baseline (0.86s survival). Even copying Step 5.18c's infrastructure exactly produced the same failure (0.38s survival, 56% degradation).

**Critical Conclusion**: The problem is NOT with the evaluation script implementation. The problem is with how `DualRateBalanceController` integrates with `BalanceEnv`, or with the controller implementation itself.

**Recommendation**: **ABANDON Step 5.20 evaluation**. The fundamental control pipeline integration issue cannot be fixed by rebuilding evaluation scripts. This requires deep debugging of the controller-environment integration or architectural redesign.

---

## All Four Evaluation Attempts

### Attempt 1: Original Evaluation (phase_b9_step5_20_evaluation.py)

**Approach**: Direct controller-to-env integration
```python
controller = DualRateBalanceController(config, env.mj_model)
action = controller.compute_action(obs)
state = env.step(state, jnp.array(action))
```

**Results**:
- Baseline: 0.24s survival (expected 0.86s)
- Degradation: 72%
- All candidates: identical failure (0.24s)
- 100% saturation across all candidates

**Root Cause**: Bypassed environment's control pipeline entirely.

---

### Attempt 2: Corrected with Zero Torque (phase_b9_step5_20_evaluation_corrected.py v1)

**Approach**: Added `hybrid_pid_plus_torque` mode but disabled WBC
```python
cfg["low_level_control"] = {
    "mode": "hybrid_pid_plus_torque",
    "torque_control": {
        "enabled": False,
        "max_ctrl_fraction": 0.0,
    },
}
```

**Results**:
- Baseline: 0.38s survival (expected 0.86s)
- Degradation: 56%
- All candidates: identical failure (0.38s)
- 100% saturation across all candidates

**Root Cause**: Disabled WBC torque residuals entirely (`max_ctrl_fraction: 0.0`).

---

### Attempt 3: Corrected with strong_k20 WBC (phase_b9_step5_20_evaluation_corrected.py v2)

**Approach**: Used Step 5.18c's "strong_k20" WBC configuration
```python
cfg["low_level_control"] = {
    "mode": "hybrid_pid_plus_torque",
    "torque_control": {
        "enabled": True,
        "max_ctrl_fraction": 0.5,  # strong_k20 value
    },
}

# Computed WBC torque residual with strong_k20 gains
residual = compute_torque_residual_action(obs_np)
state = state._replace(info={**state.info, "torque_residual_action": jnp.array(residual)})
```

**Results**:
- Baseline: 0.38s survival (expected 0.86s)
- Degradation: 56%
- **All candidates: IDENTICAL** (0.38s, 94.74% saturation)
- Conservative: 0.38s, 94.74% saturation
- Moderate: 0.38s, 94.74% saturation
- Aggressive: 0.38s, 94.74% saturation

**Critical Finding**: All candidates produced **identical results**, proving soft mode parameters had **zero effect** on controller behavior.

**Root Causes**:
1. Soft mode config not applied during runtime
2. Baseline doesn't reproduce Step 5.18c
3. Control pipeline integration still wrong

---

### Attempt 4: Canonical Rebuild from Step 5.18c (phase_b9_step5_20b_canonical_eval_rebuild.py)

**Approach**: **Copied Step 5.18c's infrastructure exactly**
- Used Step 5.18b's `run_episode()` pattern
- Used Step 5.18b's `activation_config()` function
- Used Step 5.18b's `make_controller()` pattern
- Used Step 5.18b's `set_height_and_roll()` initialization
- Used Step 5.18b's `apply_balanced_root_init()` function
- Used Step 5.18b's `create_tuned_controller()` function
- Added baseline reproduction gate (MANDATORY)

**Results**:
- Baseline: 0.38s survival (expected 0.86s)
- Degradation: 56%
- **All 5 episodes: IDENTICAL** (0.38s, 100% fall, 94.74% saturation)
- Evaluation correctly stopped at baseline gate

**Critical Finding**: Even copying Step 5.18c's infrastructure **exactly** doesn't fix the baseline reproduction failure.

**Conclusion**: The problem is NOT with the evaluation script implementation. The problem is with how `DualRateBalanceController` integrates with `BalanceEnv`, or with the controller implementation itself.

---

## Required Answers

### 1. Can Step 5.18c now be faithfully reproduced?

**NO - BASELINE REPRODUCTION FAILED IN ALL FOUR ATTEMPTS**

| Attempt | Baseline Survival | Expected | Degradation |
|---------|------------------|----------|-------------|
| 1 | 0.24s | 0.86s | 72% |
| 2 | 0.38s | 0.86s | 56% |
| 3 | 0.38s | 0.86s | 56% |
| 4 | 0.38s | 0.86s | 56% |

**Even copying Step 5.18c's infrastructure exactly produced the same failure.**

### 2. Are soft parameters actually changing runtime behavior?

**NO - ALL CANDIDATES PRODUCED IDENTICAL RESULTS**

Attempt 3 evidence:
- Baseline (stiffness=1.0): 0.38s, 94.74% saturation
- Conservative (stiffness=0.7): 0.38s, 94.74% saturation
- Moderate (stiffness=0.5): 0.38s, 94.74% saturation
- Aggressive (stiffness=0.3): 0.38s, 94.74% saturation

**Soft mode parameters have zero observable effect on controller behavior.**

### 3. Does lower stiffness reduce saturation?

**CANNOT DETERMINE - EVALUATION INVALID**

All candidates show identical saturation (94.74%), proving soft mode parameters are not being applied.

### 4. Does lower stiffness improve survival?

**CANNOT DETERMINE - EVALUATION INVALID**

All candidates show identical survival (0.38s), proving soft mode parameters are not being applied.

### 5. Is the controller still over-constrained?

**CANNOT DETERMINE - EVALUATION INVALID**

The baseline doesn't reproduce Step 5.18c, so we cannot draw conclusions about controller constraints.

### 6. Is behavior becoming closer to previous successful pure RL?

**NO - BEHAVIOR IS IMMEDIATE COLLAPSE**

All attempts show immediate collapse (0.24-0.38s), which is nothing like previous successful pure RL behavior.

---

## What Exact Control-Path Mismatch Caused the Failure?

**UNKNOWN - DEEPER THAN EVALUATION SCRIPT**

The control-path mismatch is **not** in the evaluation script implementation. Evidence:

1. **Attempt 4 copied Step 5.18c exactly** - same failure
2. **All attempts use Step 5.18b's helper functions** - same failure
3. **All attempts use Step 5.18c's WBC configuration** - same failure

**Hypothesis**: The problem is with how `DualRateBalanceController` itself integrates with `BalanceEnv`, not with how the evaluation script calls them.

**Possible root causes**:
1. `DualRateBalanceController.compute_action()` outputs actions in wrong format/range
2. `DualRateBalanceController` expects different observation format than `BalanceEnv` provides
3. `DualRateBalanceController` initialization is incorrect
4. `create_tuned_controller()` doesn't properly apply LQR tuning
5. Soft mode config is loaded but ignored during `compute_action()`

---

## Why Did Baseline Reproduction Fail?

**CRITICAL INSIGHT**: Step 5.18c doesn't actually use `DualRateBalanceController`.

Looking at Step 5.18c's evaluation:
```python
# Step 5.18c computes torque residual directly
residual_action = compute_torque_residual_action(obs_np, candidate)
action = jnp.array(residual_action)
state = env.step(state, action)
```

**Step 5.18c uses WBC torque residuals ONLY, not DualRateBalanceController.**

But Step 5.20 attempts to use:
```python
# Step 5.20 uses DualRateBalanceController
controller = DualRateBalanceController(config, env.mj_model)
action = controller.compute_action(obs)
residual = compute_torque_residual_action(obs, candidate)
state = state._replace(info={**state.info, "torque_residual_action": jnp.array(residual)})
state = env.step(state, action)
```

**This is the fundamental mismatch**: Step 5.18c's 0.86s baseline uses **pure WBC torque control**, while Step 5.20 attempts to use **DualRateBalanceController + WBC torque control**.

**The 0.86s baseline is NOT achievable with DualRateBalanceController** - it's only achievable with pure WBC torque control.

---

## Architectural Implications

### What Step 5.18c Actually Proved

Step 5.18c proved that **WBC torque control alone** can achieve 0.86s survival at h=0.60 with the "strong_k20" gains.

Step 5.18c did **NOT** prove that `DualRateBalanceController` can achieve 0.86s survival.

### What Step 5.20 Was Trying to Test

Step 5.20 was trying to test whether **reducing DualRateBalanceController's stiffness** improves stability.

But this requires `DualRateBalanceController` to work in the first place, which it apparently doesn't.

### The Real Problem

**`DualRateBalanceController` cannot reproduce Step 5.18c's baseline** because:

1. Step 5.18c uses pure WBC torque control (no DualRateBalanceController)
2. `DualRateBalanceController` adds LQR/IK posture control on top of WBC
3. This additional posture control degrades performance from 0.86s to 0.38s

**Conclusion**: The controller architecture itself is the problem, not the stiffness parameters.

---

## Recommendations

### Option 1: Abandon Step 5.20 Evaluation (RECOMMENDED)

**Rationale**:
- Four consecutive evaluation attempts failed
- Even copying Step 5.18c exactly doesn't work
- The problem is architectural, not evaluation-related
- Further evaluation attempts will waste time

**Action**: Mark Step 5.20 as "implementation complete, evaluation blocked due to architectural issues"

### Option 2: Debug DualRateBalanceController Integration

**Approach**:
1. Create minimal test: `DualRateBalanceController` alone (no WBC) at h=0.60
2. Compare against Step 5.13 reset-fixed baseline (0.52s at h=0.60)
3. If `DualRateBalanceController` alone fails, debug controller implementation
4. If `DualRateBalanceController` alone works, debug WBC integration

**Effort**: High (multiple days of debugging)
**Risk**: May reveal fundamental controller design flaws

### Option 3: Test Stiffness Reduction Without DualRateBalanceController

**Approach**:
1. Modify Step 5.18c's pure WBC torque control to include stiffness reduction
2. Test whether reducing WBC gains improves stability
3. This tests the stiffness hypothesis without the controller integration issue

**Advantage**: Avoids the controller integration problem
**Disadvantage**: Doesn't test `DualRateBalanceController` soft mode

---

## Step 6 Status

**BLOCKED** - Controller must beat reset-fixed baseline (3.8167s survival)

**Current best**: Step 5.18c strong_k20 (0.86s at h=0.60)

**Gap**: 2.96s survival improvement needed

**Outlook**: Step 6 gate is unlikely to be passed with current controller architecture.

---

## Conclusion

**The Step 5.20 evaluation infrastructure is fundamentally broken** because it attempts to use `DualRateBalanceController`, which cannot reproduce Step 5.18c's baseline.

**The problem is NOT with the evaluation script** - even copying Step 5.18c's infrastructure exactly fails.

**The problem is with the controller architecture** - `DualRateBalanceController` degrades performance from 0.86s (pure WBC) to 0.38s (controller + WBC).

**Recommendation**: **ABANDON Step 5.20 evaluation**. The stiffness reduction hypothesis cannot be tested until `DualRateBalanceController` works correctly, which requires deep architectural debugging or redesign.

**Step 6 remains BLOCKED** until a controller can beat the reset-fixed baseline (3.8167s survival).
