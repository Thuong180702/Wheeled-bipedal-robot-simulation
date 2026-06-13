# Phase B.9 Step 5.20: Low-Stiffness Dynamic Balance Transition - Summary

## Executive Summary

**Status**: Implementation complete, tests passing (7/7), evaluation pending.

**Core Hypothesis**: The current controller is over-stiff and fighting natural balancing dynamics.

**Evidence**:
- Pure RL previously balanced successfully without persistent saturation
- Current PID saturates at ±30 Nm continuously (Step 5.18c)
- Plant is stabilizable, but classical control structure may be inefficient

**Implementation**: Soft dynamic balance mode with systematic stiffness reduction and deadband logic.

---

## What Was Implemented

### 1. Soft Dynamic Balance Mode

Added opt-in controller mode in [DualRateBalanceController](wheeled_biped/controllers/dual_rate_balance_controller.py):

**Config parameters**:
```yaml
soft_dynamic_balance:
  enabled: false  # disabled by default
  posture_stiffness_reduction: 1.0  # multiply LQR gains by this factor
  posture_deadband_deg: 0.0  # don't correct pitch errors < threshold
  posture_restore_delay_s: 0.0  # delay aggressive recentering
  balance_authority_boost: 1.0  # increase WBC authority
  allow_torso_lean: false
  allow_temporary_asymmetry: false
  max_torso_lean_deg: 5.0
  max_wheel_offset_m: 0.05
```

### 2. Stiffness Reduction Logic

**Implementation** in [compute_action:498-520](wheeled_biped/controllers/dual_rate_balance_controller.py#L498-L520):

```python
# Soft dynamic balance mode: reduce posture stiffness
if self.config.soft_dynamic_balance_enabled:
    stiffness_reduction = self.config.soft_posture_stiffness_reduction
    gains = {k: v * stiffness_reduction for k, v in gains.items()}

# Soft dynamic balance: apply deadband to pitch error
if self.config.soft_dynamic_balance_enabled:
    deadband_rad = np.deg2rad(self.config.soft_posture_deadband_deg)
    if abs(pitch_error) < deadband_rad:
        pitch_error = 0.0
```

**Purpose**: Systematically reduce LQR gains and add deadband to allow dynamic balancing motion instead of forcing exact posture.

### 3. Test Configs Created

Four test configs in `outputs/phase_b9_step5_20_low_stiffness_dynamic_balance/`:

| Config | Stiffness Reduction | Deadband | Description |
|--------|---------------------|----------|-------------|
| `soft_baseline.yaml` | 1.0 (100%) | 0.0° | Disabled (backward compatible) |
| `soft_conservative.yaml` | 0.7 (70%) | 1.0° | Conservative reduction |
| `soft_moderate.yaml` | 0.5 (50%) | 2.0° | Moderate reduction |
| `soft_aggressive.yaml` | 0.3 (30%) | 3.0° | Aggressive reduction |

### 4. Evaluation Script

Created [phase_b9_step5_20_evaluation.py](scripts/phase_b9_step5_20_evaluation.py):
- Tests all 4 candidates at h=0.60
- Measures survival, fall rate, pitch/roll RMS, torque efficiency
- Compares against Step 5.18c baseline (0.86s survival, 93.75% saturation)
- Provides decision framework based on results

### 5. Tests

All 7 tests passing in [test_phase_b9_step5_20_low_stiffness_dynamic_balance.py](tests/test_phase_b9_step5_20_low_stiffness_dynamic_balance.py):
- ✓ Soft mode disabled by default
- ✓ Config loading
- ✓ Stiffness reduction bounds
- ✓ Action dimension unchanged
- ✓ No protected file modification
- ✓ Backward compatibility
- ✓ Test configs valid

---

## Design Rationale

### Current Architecture (Pose-First)
- Rigid posture tracking with high PID gains
- Balance treated as correction on top of posture maintenance
- PID dominates authority, saturates continuously
- Small deviations trigger aggressive corrections

### Target Architecture (Balance-First)
- Soft posture compliance with reduced PID gains
- Balance survival prioritized over exact pose
- Allow natural torso lean, temporary asymmetry, CoM movement
- PID becomes soft tracking layer, balance controller becomes primary

### Key Design Shift

**From**: `final = hard_PID(dominant) + weak_WBC(residual)`

**To**: `final = soft_PID(compliant) + strong_WBC(primary)`

---

## Expected Outcomes

### Optimistic: Soft Mode Improves Stability
- Lower saturation (e.g., 50-70% vs 93%)
- Longer survival (e.g., 2-3s at h=0.60 vs 0.86s)
- Lower RMS torque (e.g., 15-20 Nm vs 30 Nm)
- Slightly worse pose tracking but much better balance

**Interpretation**: Controller was over-stiff, soft mode is more efficient.

**Action**: Adopt soft mode as new baseline, proceed to Step 6.

### Neutral: Efficiency Gain Without Survival Gain
- Lower saturation
- Lower RMS torque
- Similar survival
- Worse pose tracking

**Interpretation**: Stiffness reduction helps efficiency but not stability.

**Action**: Document efficiency gain, consider hybrid approach.

### Pessimistic: Soft Mode Degrades Performance
- Faster falls
- Loss of posture control
- No efficiency gain

**Interpretation**: Current stiffness is necessary, not excessive. Classical control architecture may be fundamentally limited.

**Action**: Consider alternative approaches (pure RL, different architecture).

---

## Evaluation Plan

### Phase 1: h=0.60 Quick Evaluation

**Command**:
```bash
python scripts/phase_b9_step5_20_evaluation.py
```

**Candidates**: baseline, conservative, moderate, aggressive

**Metrics**:
- Survival time
- Fall rate
- Pitch/roll RMS
- Mean torque
- RMS torque

**Baseline (Step 5.18c)**:
- Survival: 0.86s
- Fall rate: 0.80
- Roll RMS: 15.9 deg
- Saturation: 93.75%
- RMS torque: ~30 Nm

**Success criteria**:
- Survival > 0.86s
- Saturation < 90%
- RMS torque significantly lower

### Phase 2: Full Validation (If Promising)

Only for best candidate from Phase 1.

**Heights**: 0.65, 0.60, 0.55, 0.50, 0.45, 0.40
**Episodes per height**: 5

**Baseline (reset-fixed)**:
- Survival: 3.8167s
- Fall rate: 0.8333

**Success criteria**:
- Overall survival > 3.8167s (beats reset-fixed baseline)
- Step 6 gate passes

---

## Files Modified

1. [wheeled_biped/controllers/dual_rate_balance_controller.py](wheeled_biped/controllers/dual_rate_balance_controller.py)
   - Added soft dynamic balance config fields to `DualRateConfig`
   - Added config loading in `from_yaml`
   - Added stiffness reduction and deadband logic in `compute_action`

## Files Created

1. [outputs/phase_b9_step5_20_low_stiffness_dynamic_balance/controller_transition_design.md](outputs/phase_b9_step5_20_low_stiffness_dynamic_balance/controller_transition_design.md) - Design document
2. [outputs/phase_b9_step5_20_low_stiffness_dynamic_balance/soft_baseline.yaml](outputs/phase_b9_step5_20_low_stiffness_dynamic_balance/soft_baseline.yaml) - Baseline config
3. [outputs/phase_b9_step5_20_low_stiffness_dynamic_balance/soft_conservative.yaml](outputs/phase_b9_step5_20_low_stiffness_dynamic_balance/soft_conservative.yaml) - Conservative config
4. [outputs/phase_b9_step5_20_low_stiffness_dynamic_balance/soft_moderate.yaml](outputs/phase_b9_step5_20_low_stiffness_dynamic_balance/soft_moderate.yaml) - Moderate config
5. [outputs/phase_b9_step5_20_low_stiffness_dynamic_balance/soft_aggressive.yaml](outputs/phase_b9_step5_20_low_stiffness_dynamic_balance/soft_aggressive.yaml) - Aggressive config
6. [scripts/phase_b9_step5_20_evaluation.py](scripts/phase_b9_step5_20_evaluation.py) - Evaluation script
7. [tests/test_phase_b9_step5_20_low_stiffness_dynamic_balance.py](tests/test_phase_b9_step5_20_low_stiffness_dynamic_balance.py) - Tests (7/7 passing)

---

## Critical Constraints Satisfied

✓ Soft mode disabled by default (backward compatible)
✓ No modification to `configs/training/balance_residual.yaml`
✓ No PPO training
✓ Action dimension unchanged (10)
✓ Action ordering unchanged
✓ Deployable motor torque path preserved
✓ hybrid_pid_plus_torque path preserved
✓ All existing tests preserved
✓ All existing telemetry compatibility preserved

---

## Next Steps

1. **Run evaluation**:
   ```bash
   python scripts/phase_b9_step5_20_evaluation.py
   ```

2. **Analyze results**: Compare against expected outcomes

3. **Make decision**:
   - If soft mode improves stability → Adopt as baseline, proceed to Step 6
   - If marginal improvement → Consider hybrid approach
   - If degrades performance → Question classical control architecture

4. **Update reports**: Document findings in Phase B.9 reports

---

## Architectural Significance

This is NOT another heuristic layer. This is an **architectural investigation** to test whether:

1. The controller is over-constraining posture
2. Lower stiffness allows natural balancing dynamics
3. The plant is stabilizable with softer control

**If soft mode succeeds**: Validates that classical control was fighting natural dynamics. Soft mode becomes new baseline.

**If soft mode fails**: Suggests classical control architecture is fundamentally limited. May need pure RL or different architecture.

---

## Step 6 Gate Status

**Current**: Step 6 remains BLOCKED

**Gate requirement**: Controller must beat reset-fixed baseline (3.8167s survival across all heights)

**Current best**: Step 5.18c strong_k20 (0.86s at h=0.60)

**Soft mode potential**: If evaluation shows dramatic improvement, may pass gate. Otherwise, alternative approaches needed.

---

## Conclusion

Phase B.9 Step 5.20 implementation is **complete and tested**. The soft dynamic balance mode is ready for evaluation.

This step tests a fundamental hypothesis: that the controller is over-stiff and fighting natural balancing dynamics. The evaluation will determine whether reducing posture rigidity improves stability, or whether the current stiffness is necessary.

**Evaluation pending**: Run `python scripts/phase_b9_step5_20_evaluation.py` to test the hypothesis.
