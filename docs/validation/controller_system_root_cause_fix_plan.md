# Controller System Root-Cause Fix Plan

**Date:** 2026-06-05
**Phase:** Phase 8

## Executive Summary

This fix plan is derived from the root-cause map (Phase 7). It identifies the fixes needed to address the confirmed issues that are blocking Step E acceptance.

## Critical Finding: WBC Misclassification Corrected

**Previous Report Error:** The previous Step E evaluation report incorrectly classified the controller as having WBC applied to joints, resulting in "STEP_E_5000_INVALID_DUE_TO_WBC".

**Actual State:** WBC is correctly configured as diagnostic-only in balance-core mode:
- `tau_wbc_scaled_per_joint` = all zeros
- `hidden_torque_norm` = 0.0 Nm
- Ownership violations = 0

**This means the controller is structurally sound. The remaining issues are functional, not structural.**

---

## Confirmed Issues Requiring Fix

### Issue 1: Hip-Yaw Torque Sign Convention Error

**Priority:** 1 (BLOCKING)

**Root Cause:**
Hip-yaw torque sign correctness is 0% across all heights. The PD control law in `shape_posture_controller.py` applies torque with incorrect sign.

**Evidence:**
- `hip_yaw_torque_sign_correct_left` = 0%
- `hip_yaw_torque_sign_correct_right` = 0%
- Affects: low_0p300, nominal, high_0p480

**Current Code (shape_posture_controller.py:246-254):**
```python
# SIGN FIX: Hip-yaw joint axes are inverted in MJCF model
# Negate entire PD output to account for inverted axis convention
for idx in [1, 6]:
    tau_pd = -(self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx])
```

**Problem:**
The sign negation comment says axes are inverted, but this may be incorrect. The sign_correct_left/right telemetry shows 0%, meaning torque does NOT oppose error.

**Fix Approach:**

**Step 1:** Audit the sign convention in the telemetry
1. Check if `hip_yaw_torque_sign_correct_left/right` computation matches the expected behavior
2. Verify that `posture_error = q_ref - joint_pos` is the correct error definition
3. Check if torque should be `kp * error + kd * error_dot` (standard) or negated

**Step 2:** Determine the correct sign
1. The comment says "axes are inverted in MJCF model"
2. But if axes are inverted, positive torque should make position decrease
3. If error = ref - pos, then positive error means pos < ref
4. To correct positive error (pos < ref), we need positive torque (to increase pos)
5. If axes are inverted, positive torque decreases position... which is WRONG

**Step 3:** Fix the sign if confirmed
- If the current negation is wrong: remove the negation
- If the error calculation is wrong: fix the error sign
- If the axes are not inverted as assumed: remove the negation

**Validation:**
- Run 100-step smoke test
- Expected: `hip_yaw_torque_sign_correct_left/right` > 95%
- Expected: `hip_yaw_abs_max` < 0.05 rad at nominal
- Expected: divergence_rms reduced at boundary heights

**Rollback:**
- Revert the sign change
- Re-evaluate with telemetry

---

### Issue 2: Hip-Yaw Divergence at Boundary Heights

**Priority:** 2 (BLOCKING)

**Root Cause:**
Hip-yaw divergence (left-right asymmetry) is 8x higher at boundary heights:
- low_0p300: 0.3575 rad RMS
- nominal: 0.0447 rad RMS
- high_0p480: 0.2825 rad RMS

**Hypothesis:**
This is likely a secondary effect of the torque sign convention error (Issue 1). When torque doesn't oppose error correctly, divergence accumulates over time.

**Evidence:**
- First event timing shows hip_yaw_0.10 at step 699 (low_0p300) and step 2258 (high_0p480)
- Divergence RMS is 8x higher at boundary heights vs nominal
- Sign correctness is 0% everywhere, causing uncontrolled divergence

**Fix Approach:**

**Step 1:** Fix Issue 1 (Hip-Yaw Sign Convention)
- After fixing the sign convention, re-evaluate divergence
- Expected: divergence_rms will decrease significantly

**Step 2:** Evaluate if additional divergence damping is needed
1. Run 5000-step evaluation at all three heights
2. Measure divergence_rms after sign fix
3. If divergence_rms > 0.1 rad at boundary heights, consider adding HY2-DIV divergence damping

**Step 3:** If HY2-DIV is needed (conditional fix)
- Enable `hip_yaw_divergence_damping` with:
  - k_divergence: 5.0 (baseline) or 10.0 (aggressive)
  - k_divergence_rate: 1.0 (baseline) or 2.0 (aggressive)
  - tau_max_divergence: 0.5 Nm (baseline) or 1.0 Nm (aggressive)
- Height gate: z_low=0.300, z_high=0.393

**Validation:**
- After sign fix: `hip_yaw_abs_max` < 0.07 rad at all heights
- After sign fix + HY2-DIV (if needed): `hip_yaw_abs_max` < 0.05 rad at nominal

**Rollback:**
- Disable HY2-DIV if not beneficial
- Revert sign change if divergence worsens

---

### Issue 3: Hip-Roll Saturation at Low Heights (Diagnostic)

**Priority:** 3 (NON-BLOCKING)

**Root Cause:**
Hip-roll abs_max is 3x higher at low_0p300 (0.2167 rad) vs high_0p480 (0.0773 rad).

**Hypothesis:**
Lateral roll balance controller may have insufficient authority or gain scheduling issue at low heights.

**Fix Approach:**
This is diagnostic only. Do NOT modify hip-roll or lateral controller in this fix plan.

**Validation:**
Monitor hip-roll saturation as a secondary metric after fixing Issues 1 and 2.

---

## Fix Implementation Sequence

### Sequence: Fix Hip-Yaw Sign Convention

```
1. AUDIT (Diagnostic - no code change)
   - Examine hip_yaw_torque_sign_correct computation
   - Verify posture_error definition
   - Check if sign negation is correct

2. FIX (Code change - single small diff)
   - Option A: Remove sign negation if confirmed wrong
   - Option B: Change error sign if error definition is wrong
   - Only ONE change per iteration

3. VALIDATE (No code change - run simulation)
   - Run 100-step smoke test
   - Check sign_correct_left/right > 95%
   - If not, REVERT and re-audit

4. EVALUATE (No code change - run simulation)
   - Run 5000-step evaluation at all three heights
   - Check hip_yaw_abs_max < 0.07 rad
   - Check divergence_rms reduced

5. ITERATE or COMMIT
   - If validation passes: proceed to HY2-DIV evaluation if needed
   - If validation fails: revert and re-audit
```

---

## What NOT to Fix Yet

The following are explicitly NOT part of this fix plan:

1. **Differential wheel yaw control** - Not in scope for Step E
2. **Mode-based hip-yaw control** - Not in scope for Step E
3. **Hip-roll tuning** - Do not modify hip-roll or lateral controller
4. **Gain tuning** - Do not tune gains without evidence
5. **Threshold relaxation** - Do not relax thresholds
6. **WBC changes** - WBC is correctly configured
7. **Step C/Step D** - Do not proceed until Step E passes

---

## Rollback Plan

If any fix does not improve the metrics:

1. **Revert code change** to the last known good state
2. **Re-run 5000-step evaluation**
3. **Verify metrics return to baseline**
4. **Do not proceed with subsequent fixes**

---

## Validation Gates

The following gates must pass for Step E acceptance:

| Gate | Metric | Threshold | Current | After Fix |
|------|--------|-----------|---------|-----------|
| 1 | hip_yaw_sign_correct | > 95% | 0% | TBD |
| 2 | hip_yaw_abs_max (nominal) | < 0.07 rad | 0.058 rad | TBD |
| 3 | hip_yaw_abs_max (low/high) | < 0.15 rad | 0.281/0.262 | TBD |
| 4 | divergence_rms (boundary) | < 0.10 rad | 0.358/0.283 | TBD |
| 5 | WBC applied | = 0 Nm | 0 Nm | 0 Nm |
| 6 | ownership_violations | = 0 | 0 | 0 |

---

## Files to Modify

1. `wheeled_biped/controllers/shape_posture_controller.py`
   - Hip-yaw PD control sign (if confirmed wrong)

2. `wheeled_biped/controllers/shape_posture_controller.py` (conditional)
   - HY2-DIV divergence damping (if needed after sign fix)

---

## Files NOT to Modify

- `scripts/simulate_hierarchical_controller.py` - Main simulation script
- `wheeled_biped/controllers/lateral_roll_balance_controller.py` - Lateral controller
- `wheeled_biped/controllers/balance_core_torque_composer.py` - Torque composer
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` - Sagittal controller
- Any WBC-related files

---

## Summary

| Item | Status |
|------|--------|
| Structural Invariants | ✓ CLEAN |
| Hip-Yaw Sign Convention | ✗ BLOCKING (0% correctness) |
| Hip-Yaw Divergence | ✗ BLOCKING (8x at boundary) |
| Body Yaw Stability | ✓ PASS |
| Pitch | ⚠ EXCLUDED (task-aware later) |
| WBC | ✓ CORRECT |
| Ownership | ✓ CLEAN |

**Decision:** ROOT_CAUSE_MAP_COMPLETE_READY_FOR_FIX_PLAN

**Next Step:** Implement fix for Issue 1 (Hip-Yaw Sign Convention) following the sequence above.