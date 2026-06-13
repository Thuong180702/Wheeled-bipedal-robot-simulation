# Step E: Torque-Budget-Aware Position Authority Validation Report

**Date**: 2026-05-31  
**Objective**: Validate torque-budget-aware position authority allocation to reduce transient peak below 0.30m  
**Result**: **FAILED** - Torque-budget-aware approach provides no improvement over legacy fixed cap

---

## Executive Summary

The torque-budget-aware position authority allocation was implemented and validated across three test durations (1000, 2000, 5000 steps). **Critical finding**: The approach provides **zero improvement** over the legacy fixed cap of 3.0 Nm because the physical budget constraint dominates during the transient.

**Key Results**:
- V4 baseline (legacy fixed cap 3.0 Nm): max error 0.493m at step 1411
- V3 torque-budget (cap=7.0, reserve=2.0): max error 0.493m at step 1411 (identical)
- Physical budget constraint limits position authority to 3.0 Nm during transient
- Torque-budget-aware logic correctly protects pitch balance but doesn't increase position authority
- All gates FAILED: preferred, fallback, and hard minimum

**Conclusion**: The torque-budget-aware approach is **functionally equivalent** to the legacy fixed cap because `tau_position_budget_allowed = min(position_tau_budget_cap - pitch_reserve_tau, max_tau_wheel - abs(tau_balance)) = min(7.0 - 2.0, 10.0 - 3.2) = min(5.0, 6.8) = 5.0 Nm` in theory, but during the transient when `tau_balance ≈ 3.2 Nm`, the physical budget constraint caps position authority at `10.0 - 3.2 = 6.8 Nm`, which is then further limited by the fixed cap logic to 3.0 Nm.

**Root cause**: The implementation still uses a fixed cap of 3.0 Nm in the clipping logic, overriding the budget-aware calculation.

---

## 1. Test Configuration

### V4 Baseline (Legacy Fixed Cap)
```yaml
controller_mode: balance-core
sagittal_controller: velocity-damped
k_position: 20.0
k_velocity: 15.0
k_support_velocity: 10.0
max_position_tau: 3.0  # Fixed cap
enable_capture_gate: true
capture_gate_use_cp: true
enable_torque_budget_aware_position: false
```

### V1/V2/V3 Torque-Budget-Aware
```yaml
controller_mode: balance-core
sagittal_controller: velocity-damped
k_position: 20.0
k_velocity: 15.0
k_support_velocity: 10.0
enable_capture_gate: true
capture_gate_use_cp: true
enable_torque_budget_aware_position: true
position_tau_budget_cap: 7.0  # Soft cap
pitch_reserve_tau: 2.0  # Reserve for pitch balance
```

**Budget Allocation Logic**:
```python
tau_position_budget_available = max_tau_wheel - abs(tau_balance_before_position)
tau_position_budget_allowed = min(
    position_tau_budget_cap - pitch_reserve_tau,  # 7.0 - 2.0 = 5.0 Nm
    tau_position_budget_available  # 10.0 - abs(tau_balance)
)
tau_position_clipped = clip(tau_position_raw, -tau_position_budget_allowed, tau_position_budget_allowed)
```

---

## 2. Validation Results

### V1: 1000-Step Validation

**Duration**: 10.0 seconds  
**Outcome**: FAIL - Monotonic drift, transient not captured

| Metric | Value |
|--------|-------|
| Max support position error | 0.1215 m at step 999 |
| Final support position error | 0.1215 m |
| Pitch range | [-0.10, 3.13] deg |
| Roll range | [-0.33, 0.17] deg |
| CoM height range | [0.404, 0.409] m |
| tau_position_raw range | [-2.430, 0.166] Nm |
| tau_position_clipped range | [-2.430, 0.166] Nm |
| tau_budget_allowed range | [1.707, 3.000] Nm |
| Saturation (none) | 1000/1000 (100.0%) |
| Saturation (physical_budget) | 0/1000 (0.0%) |

**Analysis**: 1000 steps too short to capture transient at step 1411. Shows monotonic drift only.

---

### V2: 2000-Step Validation

**Duration**: 20.0 seconds  
**Outcome**: FAIL - Transient captured, identical to V4 baseline

| Metric | Value |
|--------|-------|
| Max support position error | **0.4933 m at step 1411** |
| Final support position error | 0.0679 m |
| Pitch range | [-0.10, 8.26] deg |
| Roll range | [-2.48, 0.17] deg |
| CoM height range | [0.363, 0.409] m |
| tau_position_raw range | [-9.866, 0.166] Nm |
| tau_position_clipped range | **[-3.000, 0.166] Nm** |
| tau_budget_allowed range | **[1.707, 3.000] Nm** |
| Saturation (none) | 1266/2000 (63.3%) |
| Saturation (physical_budget) | **734/2000 (36.7%)** |

**Analysis**: Transient occurs at step 1411 with max error 0.493m, identical to V4 baseline. Physical budget constraint limits position authority to 3.0 Nm during transient.

---

### V3: 5000-Step Validation

**Duration**: 50.0 seconds  
**Outcome**: FAIL - Identical to V4 baseline

| Metric | Value |
|--------|-------|
| Max support position error | **0.4933 m at step 1411** |
| Final support position error | 0.0527 m |
| Pitch range | [-0.10, 8.26] deg |
| Roll range | [-2.48, 0.17] deg |
| CoM height range | [0.363, 0.409] m |
| tau_position_raw range | [-9.866, 0.166] Nm |
| tau_position_clipped range | **[-3.000, 0.166] Nm** |
| tau_budget_allowed range | **[1.707, 3.000] Nm** |
| Saturation (none) | 4266/5000 (85.3%) |
| Saturation (physical_budget) | **734/5000 (14.7%)** |

**Analysis**: Identical transient behavior to V4 baseline. Physical budget constraint dominates during transient (steps 1300-1500).

---

## 3. Comparison: V4 Baseline vs V3 Torque-Budget-Aware

| Metric | V4 Baseline | V3 Torque-Budget | Difference |
|--------|-------------|------------------|------------|
| Max support position error | 0.4933 m | 0.4933 m | **0.000 m (0.0%)** |
| Max error step | 1411 | 1411 | 0 |
| Final support position error | 0.0527 m | 0.0527 m | 0.000 m |
| Pitch range | [-0.10, 8.26] deg | [-0.10, 8.26] deg | Identical |
| Roll range | [-2.48, 0.17] deg | [-2.48, 0.17] deg | Identical |
| CoM height range | [0.363, 0.409] m | [0.363, 0.409] m | Identical |
| tau_position_raw range | [-9.866, 0.166] Nm | [-9.866, 0.166] Nm | Identical |
| tau_position_clipped range | [-3.000, 0.166] Nm | [-3.000, 0.166] Nm | **Identical** |

**Conclusion**: Torque-budget-aware approach provides **zero improvement** over legacy fixed cap.

---

## 4. Root Cause Analysis

### Why Torque-Budget-Aware Doesn't Help

1. **Physical Budget Constraint Dominates**:
   - During transient (steps 1300-1500), `tau_balance ≈ 3.2 Nm`
   - Physical budget available: `10.0 - 3.2 = 6.8 Nm`
   - Budget-aware allowed: `min(7.0 - 2.0, 6.8) = min(5.0, 6.8) = 5.0 Nm`
   - **But actual clipped range is [-3.000, 0.166] Nm, not [-5.000, 0.166] Nm**

2. **Implementation Issue**:
   - The budget-aware logic calculates `tau_position_budget_allowed` correctly
   - But the clipping still uses a fixed cap of 3.0 Nm somewhere in the pipeline
   - This overrides the budget-aware calculation

3. **Saturation Reason Confirms**:
   - V2/V3 show 36.7%/14.7% saturation due to `physical_budget` constraint
   - This means the budget-aware logic is active and limiting position authority
   - But the limit is 3.0 Nm, not the expected 5.0 Nm

### Expected vs Actual Behavior

**Expected**:
```
tau_balance = 3.2 Nm
tau_budget_available = 10.0 - 3.2 = 6.8 Nm
tau_budget_allowed = min(5.0, 6.8) = 5.0 Nm
tau_position_clipped = clip(tau_position_raw, -5.0, 5.0)
```

**Actual**:
```
tau_balance = 3.2 Nm
tau_budget_available = 10.0 - 3.2 = 6.8 Nm
tau_budget_allowed = min(5.0, 6.8) = 5.0 Nm  # Calculated correctly
tau_position_clipped = clip(tau_position_raw, -3.0, 3.0)  # Fixed cap still applied!
```

---

## 5. Gate Compliance

### Preferred Gate (max ≤0.10m, final ≤0.05m)
- V1: **FAIL** (max 0.1215m > 0.10m)
- V2: **FAIL** (max 0.4933m > 0.10m)
- V3: **FAIL** (max 0.4933m > 0.10m)

### Fallback Gate (max ≤0.15m, final ≤0.10m)
- V1: **FAIL** (max 0.1215m > 0.15m, but final 0.1215m > 0.10m)
- V2: **FAIL** (max 0.4933m > 0.15m)
- V3: **FAIL** (max 0.4933m > 0.15m)

### Hard Minimum Gate (max ≤0.30m, final ≤0.10m)
- V1: **FAIL** (max 0.1215m < 0.30m, but final 0.1215m > 0.10m)
- V2: **FAIL** (max 0.4933m > 0.30m)
- V3: **FAIL** (max 0.4933m > 0.30m)

**All gates FAILED.**

---

## 6. Recommended Next Steps

### Option 1: Fix Implementation (Immediate)
**Diagnose why fixed cap of 3.0 Nm is still applied despite budget-aware calculation.**

Investigation needed:
1. Check if `max_position_tau` parameter is still being used in clipping
2. Verify that `tau_position_budget_allowed` is actually passed to clipping function
3. Add telemetry to confirm clipping limits at each step

Expected fix:
```python
# Current (suspected):
tau_position_clipped = jnp.clip(tau_position_raw, -max_position_tau, max_position_tau)

# Should be:
tau_position_clipped = jnp.clip(tau_position_raw, -tau_position_budget_allowed, tau_position_budget_allowed)
```

### Option 2: Reduce Pitch Reserve (If Implementation is Correct)
If the implementation is correct and budget-aware logic is truly limiting to 3.0 Nm due to high balance torque, then:

**Test pitch_reserve_tau = 1.0 Nm** (currently 2.0 Nm):
- This would allow `tau_budget_allowed = min(7.0 - 1.0, 6.8) = min(6.0, 6.8) = 6.0 Nm`
- Risk: Less pitch balance authority during transient
- Benefit: More position authority to suppress transient

### Option 3: Increase Position Budget Cap
**Test position_tau_budget_cap = 9.0 Nm** (currently 7.0 Nm):
- This would allow `tau_budget_allowed = min(9.0 - 2.0, 6.8) = min(7.0, 6.8) = 6.8 Nm`
- Uses all available physical budget after balance torque
- Risk: None if physical budget constraint is respected

### Option 4: Abandon Budget-Aware Approach
If budget-aware approach cannot provide more than 3.0 Nm position authority:
- Return to simple fixed cap approach
- Increase `max_position_tau` directly to 6.0-7.0 Nm
- Accept that position hold may steal pitch balance authority during transients
- Rely on pitch controller to recover

---

## 7. Conclusion

**The torque-budget-aware position authority allocation provides zero improvement over the legacy fixed cap of 3.0 Nm.** This is either due to:

1. **Implementation bug**: Fixed cap of 3.0 Nm still applied despite budget-aware calculation (most likely)
2. **Design limitation**: Physical budget constraint during transient limits position authority to 3.0 Nm

**Immediate action required**: Diagnose implementation to determine if budget-aware clipping limits are actually being used. If not, fix the implementation. If yes, then the approach is fundamentally limited by the high balance torque during transient and alternative strategies are needed.

**Step E status**: **FAILED** - Cannot proceed to Step C until at least hard minimum gate passes.

---

## Appendix A: Telemetry Files

- V4 baseline: `outputs/hierarchical_controller_sim/telemetry_1780211559.csv`
- V1 1000-step: `outputs/hierarchical_controller_sim/telemetry_1780215336.csv`
- V2 2000-step: `outputs/hierarchical_controller_sim/telemetry_1780215969.csv`
- V3 5000-step: `outputs/hierarchical_controller_sim/telemetry_1780216397.csv`

## Appendix B: Analysis Scripts

- `analyze_validation_run.py`: Single-run analysis
- `outputs/sagittal_position_hold_return/compare_torque_budget_variants.py`: Multi-run comparison
- `outputs/sagittal_position_hold_return/step_e_torque_budget_comparison.json`: Structured results

---

**Report generated**: 2026-05-31  
**Next action**: Diagnose implementation bug or pivot to alternative strategy
