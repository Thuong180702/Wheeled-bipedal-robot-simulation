# Phase B.9 Step 5.20: Evaluation Pipeline Audit

## Executive Summary

**Status**: EVALUATION INFRASTRUCTURE INVALID

**Finding**: The Step 5.20 evaluation infrastructure is fundamentally broken and cannot produce valid results. Multiple attempts to fix it have failed because the approach of adapting Step 5.18c's evaluation pattern is insufficient.

**Evidence**: All 4 candidates (baseline, conservative, moderate, aggressive) produce **identical results** (0.38s survival, 94.74% saturation), proving the stiffness reduction parameters have **zero effect**.

**Root Cause**: The evaluation script attempts to integrate `DualRateBalanceController` with `BalanceEnv` in a way that bypasses or incorrectly implements the control pipeline that Step 5.18c uses.

**Recommendation**: **STOP incremental fixes**. Rebuild evaluation from scratch by copying Step 5.18c's exact evaluation pattern, not adapting it.

---

## Evaluation Attempts and Failures

### Attempt 1: Original Evaluation (INVALID)

**Script**: `phase_b9_step5_20_evaluation.py`

**Approach**:
```python
controller = DualRateBalanceController(config, env.mj_model)
action = controller.compute_action(obs)
state = env.step(state, jnp.array(action))
```

**Results**:
- Baseline: 0.24s survival (expected 0.86s)
- All candidates: identical failure (0.24s)
- 100% saturation across all candidates

**Root Cause**: Bypassed environment's control pipeline entirely. Controller outputs were passed directly to `env.step()` without proper integration with `hybrid_pid_plus_torque_control`.

---

### Attempt 2: Corrected Evaluation with Zero Torque (INVALID)

**Script**: `phase_b9_step5_20_evaluation_corrected.py` (initial version)

**Approach**:
```python
# Added activation_config to enable hybrid_pid_plus_torque mode
cfg["low_level_control"] = {
    "mode": "hybrid_pid_plus_torque",
    "torque_control": {
        "enabled": False,
        "max_ctrl_fraction": 0.0,  # Zero torque
    },
}
```

**Results**:
- Baseline: 0.38s survival (expected 0.86s)
- All candidates: identical failure (0.38s)
- 100% saturation across all candidates

**Root Cause**: Set `max_ctrl_fraction: 0.0`, which disabled WBC torque residuals entirely. This is wrong - Step 5.18c's 0.86s baseline used `max_ctrl_fraction: 0.5` with specific WBC torque gains.

---

### Attempt 3: Corrected Evaluation with strong_k20 WBC (INVALID)

**Script**: `phase_b9_step5_20_evaluation_corrected.py` (final version)

**Approach**:
```python
# Used Step 5.18c's "strong_k20" WBC configuration
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
- All candidates: **identical** failure (0.38s, 94.74% saturation)
- Conservative: 0.38s, 94.74% saturation
- Moderate: 0.38s, 94.74% saturation
- Aggressive: 0.38s, 94.74% saturation

**Critical Finding**: All candidates produce **identical results**, proving the stiffness reduction parameters are having **zero effect**.

**Root Causes**:
1. **Soft mode config not applied**: Despite loading soft mode YAML files and merging them, the stiffness reduction parameters don't affect controller behavior
2. **Baseline doesn't reproduce Step 5.18c**: 0.38s vs expected 0.86s (56% degradation)
3. **Control pipeline integration still wrong**: Despite using strong_k20 WBC config, the integration pattern doesn't match Step 5.18c

---

## Evidence of Fundamental Failure

### 1. Identical Results Across All Candidates

If stiffness reduction was working, we would see **different behavior** between:
- Baseline (stiffness=1.0, deadband=0.0°)
- Conservative (stiffness=0.7, deadband=1.0°)
- Moderate (stiffness=0.5, deadband=2.0°)
- Aggressive (stiffness=0.3, deadband=3.0°)

**Actual results**: All produce exactly 0.38s survival and 94.74% saturation.

**Conclusion**: The stiffness reduction parameters are not being applied to the controller's behavior.

### 2. Baseline Doesn't Reproduce Step 5.18c

**Expected** (Step 5.18c strong_k20):
- Survival: 0.86s
- Fall rate: 0.80
- Roll RMS: 15.9°
- Saturation: 93.75%

**Actual** (Attempt 3 baseline):
- Survival: 0.38s (56% worse)
- Fall rate: 1.00 (25% worse)
- Roll RMS: 24.94° (57% worse)
- Saturation: 94.74% (1% worse)

**Conclusion**: The evaluation setup does not reproduce the Step 5.18c baseline, proving the control pipeline integration is wrong.

### 3. Soft Mode Config Loading Verified But Not Applied

The evaluation script:
1. ✓ Loads soft mode YAML files correctly
2. ✓ Merges soft_dynamic_balance section into controller config
3. ✓ Creates temporary merged config file
4. ✓ Loads merged config via `DualRateConfig.from_yaml()`
5. ✗ **Stiffness reduction has zero effect on behavior**

**Conclusion**: The soft mode config is loaded but not applied during controller execution.

---

## Root Cause Analysis

### Why Soft Mode Config Has Zero Effect

**Hypothesis 1**: Controller reset clears soft mode config

The evaluation calls `controller.reset()` after initialization:
```python
controller = make_controller(model, candidate.config_path)
# ...
state = set_height_and_roll(state, env, height, init_table)
controller.reset()  # Does this clear soft mode config?
```

**Hypothesis 2**: Soft mode config not used in compute_action

The `DualRateBalanceController.compute_action()` method may not actually use the soft mode config fields during execution. The config may be loaded but ignored.

**Hypothesis 3**: Control pipeline integration bypasses controller

The way the evaluation integrates the controller with the environment may bypass the controller's internal logic, rendering the soft mode config irrelevant.

### Why Baseline Doesn't Reproduce Step 5.18c

**Hypothesis 1**: Missing initialization state

Step 5.18c uses `apply_balanced_root_init()` and `set_height_and_roll()` from `phase_b9_step5_lqr_gain_strengthening.py`. The corrected evaluation uses these functions, but may not use them correctly.

**Hypothesis 2**: Different episode length or termination conditions

Step 5.18c uses specific episode lengths and termination conditions that may differ from the corrected evaluation.

**Hypothesis 3**: Missing controller tuning

Step 5.18c uses `create_tuned_controller()` which applies LQR gain tuning from `best_lqr_config.yaml`. The corrected evaluation uses this function, but may not apply the tuning correctly.

---

## What Went Wrong: Incremental Fixing Approach

The fundamental mistake was trying to **adapt** Step 5.18c's evaluation pattern rather than **copying** it exactly.

### Adaptation Approach (Failed)

1. Read Step 5.18c evaluation script
2. Identify key patterns (activation_config, make_controller, etc.)
3. Reimplement these patterns in a new script
4. Hope the reimplementation matches the original behavior

**Why this failed**:
- Subtle differences in implementation details
- Missing helper functions or initialization steps
- Incorrect assumptions about control pipeline integration
- No validation that baseline reproduces Step 5.18c

### Correct Approach (Not Attempted)

1. **Copy** Step 5.18c evaluation script exactly
2. **Modify** only the candidate definitions to test stiffness reduction
3. **Verify** baseline reproduces Step 5.18c results FIRST
4. **Then** test stiffness reduction variants

---

## Path Forward

### Option 1: Rebuild Evaluation from Step 5.18c (Recommended)

**Approach**:
1. Copy `phase_b9_step5_18c_torque_gain_saturation_calibration.py` to `phase_b9_step5_20_evaluation_v2.py`
2. Keep all helper functions, initialization, and control pipeline integration **exactly as-is**
3. Replace `TorqueGainCandidate` with `SoftModeCandidate` that includes:
   - Same WBC torque gains as strong_k20 (k_roll=20.0, etc.)
   - Additional soft mode config parameters (stiffness_reduction, deadband_deg)
4. Modify `make_controller()` to merge soft mode config into controller config
5. **Verify baseline reproduces Step 5.18c** (0.86s survival) before testing variants
6. If baseline fails, STOP and debug until it matches Step 5.18c

**Advantages**:
- Minimal changes to proven evaluation infrastructure
- High confidence that baseline will reproduce Step 5.18c
- Clear validation gate (baseline must match before proceeding)

**Disadvantages**:
- Requires understanding Step 5.18c's evaluation pattern in detail
- May reveal that soft mode config integration is more complex than expected

### Option 2: Investigate Why Soft Mode Config Has Zero Effect

**Approach**:
1. Add debug logging to `DualRateBalanceController.compute_action()`
2. Verify soft mode config fields are actually used during execution
3. Check if `controller.reset()` clears soft mode config
4. Verify stiffness reduction is applied to LQR gains
5. Verify deadband is applied to pitch error

**Advantages**:
- May reveal simple bug in soft mode implementation
- Could fix evaluation without rebuilding from scratch

**Disadvantages**:
- Doesn't address baseline reproduction failure
- May reveal deeper architectural issues
- Time-consuming debugging with uncertain outcome

### Option 3: Abandon Step 5.20 Evaluation

**Approach**:
1. Document that Step 5.20 evaluation infrastructure is invalid
2. Mark Step 5.20 as "implementation complete, evaluation blocked"
3. Move forward with other Phase B.9 steps or Step 6

**Advantages**:
- Avoids sinking more time into broken evaluation infrastructure
- Focuses effort on more productive work

**Disadvantages**:
- Leaves Step 5.20 hypothesis untested
- Doesn't provide evidence for or against stiffness reduction approach
- Step 6 remains blocked (requires controller improvement)

---

## Recommendation

**Pursue Option 1: Rebuild Evaluation from Step 5.18c**

**Rationale**:
1. Step 5.18c's evaluation infrastructure is proven to work (0.86s survival)
2. Copying it exactly minimizes risk of introducing new bugs
3. Clear validation gate (baseline must match Step 5.18c) prevents wasted effort
4. If baseline still fails, we know the problem is deeper than evaluation setup

**Implementation Plan**:
1. Copy `phase_b9_step5_18c_torque_gain_saturation_calibration.py` → `phase_b9_step5_20_evaluation_v2.py`
2. Keep `run_h060_survival_evaluation()` function exactly as-is
3. Replace `TorqueGainCandidate` with `SoftModeCandidate`:
   ```python
   @dataclass
   class SoftModeCandidate:
       name: str
       # WBC torque gains (same as strong_k20)
       k_roll: float = 20.0
       k_roll_rate: float = 2.0
       k_pitch: float = 5.0
       k_pitch_rate: float = 0.5
       max_ctrl_fraction: float = 0.5
       allow_wheel_torque: bool = False
       wheel_roll_gain: float = 0.0
       # Soft mode parameters
       stiffness_reduction: float = 1.0
       deadband_deg: float = 0.0
   ```
4. Modify `make_controller()` to merge soft mode config
5. Run baseline candidate FIRST and verify 0.86s survival
6. If baseline fails, STOP and debug
7. If baseline succeeds, test stiffness reduction variants

**Success Criteria**:
- Baseline reproduces Step 5.18c: ~0.86s survival, ~94% saturation
- Stiffness reduction variants show **different behavior** from baseline
- Results are trustworthy and can inform architectural decisions

---

## Conclusion

The Step 5.20 evaluation infrastructure is fundamentally broken and cannot produce valid results. Three attempts to fix it have failed because the approach of adapting Step 5.18c's evaluation pattern is insufficient.

**The evaluation must be rebuilt from scratch by copying Step 5.18c's exact evaluation pattern, not adapting it.**

Until the evaluation infrastructure is fixed and baseline reproduction is verified, no conclusions can be drawn about the stiffness reduction hypothesis.

**Step 6 remains BLOCKED** - the controller must beat the reset-fixed baseline (3.8167s survival) before Step 6 can begin.
