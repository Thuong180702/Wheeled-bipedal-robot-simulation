# Step E: Root Cause Analysis and Fix Report

**Date**: 2026-05-31  
**Objective**: Diagnose why torque-budget-aware position authority failed to improve transient performance  
**Result**: **ROOT CAUSE IDENTIFIED** - pitch_reserve_tau too conservative  

---

## Executive Summary

The torque-budget-aware position authority implementation is **working correctly**, but the `pitch_reserve_tau=2.0 Nm` parameter is too conservative, limiting position authority to 3.0 Nm during transients—identical to the legacy fixed cap. Reducing `pitch_reserve_tau` from 2.0 to 1.0 Nm will increase position authority to 4.0 Nm (+33% improvement) while still protecting pitch balance.

**Key Findings**:
1. Budget-aware clipping logic is implemented correctly (no implementation bug)
2. `pitch_reserve_tau=2.0 Nm` is the limiting factor during transients
3. One-sided position error (+0.493 m peak) is caused by insufficient corrective torque
4. Steady-state offset (+0.053 m) is equilibrium, not a bug

**Recommended Fix**: Reduce `pitch_reserve_tau` from 2.0 to 1.0 Nm

---

## Phase 1: Torque Budget Implementation Audit

### Finding 1: Budget-Aware Clipping is Working Correctly

**Evidence from V3 telemetry (5000 steps, pitch_reserve_tau=2.0):**

| Step | tau_position_raw | tau_position_clipped | tau_budget_allowed | Expected (budget) | Expected (fixed 3.0) | Match |
|------|------------------|----------------------|--------------------|--------------------|----------------------|-------|
| 1300 | -6.731 Nm | -3.000 Nm | 3.000 Nm | -3.000 Nm | -3.000 Nm | ✓ Budget |
| 1360 | -8.981 Nm | -3.000 Nm | 3.000 Nm | -3.000 Nm | -3.000 Nm | ✓ Budget |
| 1411 | -9.866 Nm | -3.000 Nm | 3.000 Nm | -3.000 Nm | -3.000 Nm | ✓ Budget |
| 1450 | -9.605 Nm | -3.000 Nm | 3.000 Nm | -3.000 Nm | -3.000 Nm | ✓ Budget |
| 1500 | -9.057 Nm | -3.000 Nm | 3.000 Nm | -3.000 Nm | -3.000 Nm | ✓ Budget |

**Conclusion**: `tau_position_clipped` matches the budget-aware logic perfectly. This is **NOT an implementation bug**.

### Finding 2: pitch_reserve_tau is the Limiting Factor

**Budget calculation during transient (step 1411):**

```
tau_balance_before_position = +3.154 Nm (positive, forward-tilting)
tau_position_raw = -9.866 Nm (negative, wants to correct forward drift)

Directional budget logic (lines 189-194 in controller):
  Since tau_position_raw < 0 (negative direction):
    available_budget = max_tau_wheel - max(0, -tau_balance_before_position)
    available_budget = 5.0 - max(0, -3.154)
    available_budget = 5.0 - 0 = 5.0 Nm

  available_position_tau = max(0, available_budget - pitch_reserve_tau)
  available_position_tau = max(0, 5.0 - 2.0) = 3.0 Nm

  allowed_position_tau = min(available_position_tau, position_tau_budget_cap)
  allowed_position_tau = min(3.0, 7.0) = 3.0 Nm

Result: tau_position_clipped = clip(-9.866, -3.0, 3.0) = -3.0 Nm
```

**Root Cause**: `pitch_reserve_tau=2.0 Nm` limits position authority to 3.0 Nm, making the budget-aware approach functionally equivalent to the legacy fixed cap.

**Classification**: `pitch_reserve_tau_too_conservative`

---

## Phase 2: One-Sided Position Error Investigation

### Finding 3: One-Sided Error is Steady-State Equilibrium

**Steady-state analysis (steps 4000-5000):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| support_position_error_m mean | +0.0527 m | Forward of reference |
| support_position_velocity_m_s mean | ~0.000 m/s | No drift |
| pitch_x_rad mean | +1.207 deg | Slightly forward-tilted |
| tau_balance_before_position mean | +1.053 Nm | Forward torque |
| tau_position_raw mean | -1.053 Nm | Corrective torque |
| Net torque | ~0 Nm | **Equilibrium** |

**Conclusion**: The +0.053 m steady-state offset is **NOT a bug**. It is the equilibrium position required to balance the forward pitch bias. The robot cannot achieve pitch=0 deg at support_position_error=0 m simultaneously.

### Finding 4: Transient Peak Caused by Insufficient Position Authority

**Transient analysis (steps 1300-1500):**

- `tau_position_raw` reaches -9.9 Nm (controller wants strong correction)
- Clipped to -3.0 Nm due to `pitch_reserve_tau`
- Insufficient corrective torque allows error to grow to 0.493 m
- Controller eventually recovers but peak exceeds all gates

**Classification**: `insufficient_position_authority_during_transient`

---

## Recommended Fix

### Fix: Reduce pitch_reserve_tau from 2.0 to 1.0 Nm

**Rationale**:
- Current: `available_position_tau = 5.0 - 2.0 = 3.0 Nm`
- Proposed: `available_position_tau = 5.0 - 1.0 = 4.0 Nm`
- **+33% more position authority during transient**
- Still reserves 1.0 Nm for pitch balance protection
- Physical budget constraint still enforced (`max_tau_wheel = 5.0 Nm`)

**Expected Improvement**:
- Stronger corrective torque during transient: -4.0 Nm instead of -3.0 Nm
- Reduced peak position error: target < 0.30 m (hard minimum gate)
- Faster recovery to steady-state
- Estimated improvement: 20-30% reduction in peak error

**Risk Assessment**:
- **Risk**: Slightly less pitch balance authority during transient
- **Mitigation**: Pitch controller is already strong (`kp_pitch=50.0`, `kd_pitch=10.0`)
- **Acceptable**: 1.0 Nm reserve is sufficient for pitch protection

**Implementation**:
```python
# In scripts/simulate_hierarchical_controller.py, line 623:
vd_pitch_reserve_tau: float = 1.0,  # Changed from 2.0
```

---

## Alternative Fixes Considered and Rejected

### Alternative 1: Increase position_tau_budget_cap to 9.0 Nm
- Would allow `available_position_tau = min(9.0 - 2.0, 5.0) = 5.0 Nm`
- **Rejected**: Doesn't address the root cause (pitch_reserve_tau)
- Less intuitive parameter relationship

### Alternative 2: Abandon budget-aware approach, increase max_position_tau to 6.0 Nm
- Simpler but loses dynamic budget allocation
- **Rejected**: Budget-aware approach is correct, just needs tuning
- Would allow position to steal pitch authority during critical moments

### Alternative 3: Add integral action for steady-state bias
- **Rejected**: Steady-state offset is equilibrium, not drift
- Integral would fight the natural equilibrium
- Adds complexity and potential instability

---

## Validation Plan

### V5: 2000-step validation with pitch_reserve_tau=1.0

**Expected results**:
- Max position error: 0.25-0.35 m (down from 0.493 m)
- `tau_position_budget_allowed` during transient: ~4.0 Nm
- `tau_position_clipped` during transient: -4.0 to -3.5 Nm
- Hard minimum gate: **PASS** (max ≤ 0.30 m)

### V6: 5000-step validation with pitch_reserve_tau=1.0

**Required for Step C progression**:
- Confirm hard minimum gate passes over full duration
- Verify no pitch/posture regression
- Check steady-state error remains acceptable

### Acceptance Criteria

**Hard Minimum Gate (required for Step C)**:
- Max absolute position error ≤ 0.30 m
- Final absolute position error ≤ 0.10 m
- No termination
- Pitch range within [-10, +10] deg
- Roll range within [-5, +5] deg

**Preferred (stretch goal)**:
- Max absolute position error ≤ 0.10 m
- Final absolute position error ≤ 0.05 m

---

## Code Review: Controller Implementation

**File**: `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`

**Lines 178-203**: Torque-budget-aware position authority allocation

```python
if self.enable_torque_budget_aware_position:
    # Compute balance torque before position (all terms except position)
    tau_balance_before_position = (
        tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
        tau_support_velocity + tau_cp + tau_com_vy +
        0.5 * (tau_wheel_vel_left + tau_wheel_vel_right)
    )

    # Compute available budget in the direction tau_position_before_clip requests
    if tau_position_before_clip >= 0:
        # Positive direction: available = max_tau_wheel - max(0, tau_balance)
        available_budget = self.max_tau_wheel - max(0.0, float(tau_balance_before_position))
    else:
        # Negative direction: available = max_tau_wheel - max(0, -tau_balance)
        available_budget = self.max_tau_wheel - max(0.0, -float(tau_balance_before_position))

    # Apply pitch reserve to protect pitch balance authority
    available_position_tau = max(0.0, available_budget - self.pitch_reserve_tau)

    # Apply configurable upper cap
    allowed_position_tau = min(available_position_tau, self.position_tau_budget_cap)

    # Clip tau_position to allowed budget
    tau_position = float(jnp.clip(tau_position_before_clip, -allowed_position_tau, allowed_position_tau))
```

**Verification**: ✓ Implementation is correct. The directional budget logic properly accounts for the sign of `tau_balance_before_position` and `tau_position_before_clip`.

---

## Torque Budget Unit Convention

**Clarification**: 
- `max_tau_wheel = 5.0 Nm` is the **per-wheel** physical limit
- `tau_common` is applied to **both** wheel joints [4, 9]
- Budget calculation uses per-wheel convention correctly
- No unit conversion error

---

## Summary

| Issue | Status | Classification | Fix |
|-------|--------|----------------|-----|
| Implementation bug? | ✗ No | N/A | None needed |
| Budget-aware logic correct? | ✓ Yes | Verified | None needed |
| pitch_reserve_tau too conservative? | ✓ Yes | Root cause | Reduce to 1.0 Nm |
| One-sided error a bug? | ✗ No | Equilibrium | None needed |
| Insufficient position authority? | ✓ Yes | Consequence | Fixed by pitch_reserve_tau |

**Next Action**: Run V5 and V6 validations with `pitch_reserve_tau=1.0 Nm` to confirm hard minimum gate passes.

**Step E Status**: Root cause identified, fix ready for validation.

---

## Appendix A: Telemetry Files

- V3 5000-step (pitch_reserve_tau=2.0): `outputs/hierarchical_controller_sim/telemetry_1780216397.csv`
- V4 baseline (legacy fixed cap): `outputs/hierarchical_controller_sim/telemetry_1780211559.csv`

## Appendix B: Analysis Scripts

- `outputs/sagittal_position_hold_return/analyze_validation_run.py`
- `outputs/sagittal_position_hold_return/compare_torque_budget_variants.py`
- `outputs/sagittal_position_hold_return/analyze_pitch_reserve_fix.py`

---

**Report generated**: 2026-05-31  
**Author**: Diagnostic analysis of Step E torque-budget-aware validation  
**Status**: Root cause identified, awaiting validation of fix
