# Controller System Root-Cause Audit Report

**Date:** 2026-06-05
**Audit Type:** Diagnostic Root-Cause Analysis
**Scope:** Step E 5000-step controller evaluation at three heights

## Executive Summary

**Final Decision:** `ROOT_CAUSE_MAP_COMPLETE_READY_FOR_FIX_PLAN`

This audit has completed a systematic root-cause analysis of the controller system. Two confirmed issues were identified that are blocking Step E acceptance:

1. **Hip-Yaw Torque Sign Convention Error** (Priority 1) - Torque sign correctness is 0% across all heights
2. **Hip-Yaw Divergence at Boundary Heights** (Priority 2) - Divergence is 8x higher at boundary heights

**Critical Correction:** The previous Step E evaluation report incorrectly classified WBC as applied to joints. This audit confirms WBC is correctly configured as diagnostic-only in balance-core mode.

## Key Findings

| Finding | Severity | Confidence | Status |
|---------|----------|------------|--------|
| Hip-Yaw Sign Convention Error | HIGH | HIGH | CONFIRMED - Blocking |
| Hip-Yaw Divergence at Boundaries | HIGH | HIGH | CONFIRMED - Blocking |
| Hip-Roll Saturation | MEDIUM | MEDIUM | LIKELY - Non-blocking |
| WBC Applied | - | HIGH | RULED OUT |
| Body Yaw Instability | - | HIGH | RULED OUT |

## Telemetry Summary

| Height | Survived | Hip-Yaw Max | Hip-Yaw Sign Correct | Divergence RMS | Pitch Max | Roll Max |
|--------|----------|-------------|---------------------|----------------|-----------|----------|
| low_0p300 | ✓ 5000 | 0.281 rad | 0% | 0.358 rad | 0.157 rad | 0.014 rad |
| nominal | ✓ 5000 | 0.058 rad | 0% | 0.045 rad | 0.071 rad | 0.013 rad |
| high_0p480 | ✓ 5000 | 0.262 rad | 0% | 0.283 rad | 0.093 rad | 0.002 rad |

## Structural Invariant Status

| Invariant | Status | Evidence |
|-----------|--------|----------|
| WBC Diagnostic vs Applied | ✓ CLEAN | `tau_wbc_scaled_per_joint` = all zeros |
| Hidden Torque | ✓ CLEAN | `hidden_torque_norm` = 0.0 Nm |
| Ownership Violations | ✓ CLEAN | `ownership_violation_count` = 0 |
| Legacy Path | ✓ INACTIVE | No legacy torque detected |

## Controller Configuration

| Parameter | Value |
|-----------|-------|
| Controller Mode | balance-core |
| Sagittal Controller | velocity-damped |
| Sagittal Profile | J3 |
| WBC Status | diagnostic_only |
| Hip-Yaw Divergence Damping | DISABLED |
| Hip-Yaw Support Feedforward | DISABLED |
| Yaw-Aware Compensation | ENABLED |

## Event Order Analysis

| Height | Classification | Hip-Yaw 0.03 | Hip-Yaw 0.10 | Notes |
|--------|----------------|--------------|--------------|-------|
| low_0p300 | hip_yaw_first | step 273 | step 699 | Divergence dominates |
| nominal | stable | step 464 | - | Within limits |
| high_0p480 | hip_yaw_first | step 716 | step 2258 | Slower divergence |

## Subsystem Audit Summary

### A. Hip-Yaw Subsystem
- **Classification:** one_sided_drift (low/high), divergence_present (nominal)
- **Key Issue:** sign_correct = 0% everywhere
- **Root Cause:** Torque sign convention error in PD control

### B. Body Yaw Subsystem
- **Classification:** stable
- **Key Issue:** None
- **Status:** PASS

### C. Support/Sagittal Subsystem
- **Classification:** large_pitch (low), controlled (nominal/high)
- **Key Issue:** Pitch at low_0p300 (0.157 rad) - excluded from fix scope
- **Status:** Monitor only

### D. Roll/Lateral Subsystem
- **Classification:** hip_roll_saturation (low/nominal), stable (high)
- **Key Issue:** Hip-roll abs_max elevated at low heights
- **Status:** Monitor only

### E. Height/Contact Subsystem
- **Classification:** controlled
- **Key Issue:** None
- **Status:** PASS

### F. Torque Composer
- **Classification:** CLEAN
- **Key Issue:** None
- **Status:** PASS

## Fix Plan Summary

### Priority 1: Hip-Yaw Sign Convention Fix
- **Location:** `wheeled_biped/controllers/shape_posture_controller.py`
- **Action:** Audit and fix hip-yaw PD control sign
- **Expected Outcome:** sign_correct > 95%
- **Validation:** 100-step smoke test → 5000-step evaluation

### Priority 2: Divergence Evaluation
- **Action:** After sign fix, evaluate if HY2-DIV is needed
- **Expected Outcome:** divergence_rms < 0.10 rad at all heights
- **Validation:** 5000-step evaluation

### Priority 3: Monitor Hip-Roll
- **Action:** Monitor hip-roll saturation as secondary metric
- **Expected Outcome:** No degradation after sign fix
- **Validation:** Include in 5000-step evaluation

## What NOT to Fix

- WBC (correctly configured)
- Body yaw stability (stable)
- Pitch behavior (task-aware later)
- Differential wheel yaw (not in scope)
- Mode-based hip-yaw (not in scope)
- Hip-roll modification (do not modify)

## Artifacts Generated

### Audit Scripts
- `scripts/audit_controller_configuration_state.py`
- `scripts/audit_controller_structural_invariants.py`
- `scripts/audit_controller_event_order.py`
- `scripts/audit_controller_subsystems.py`

### Reports
- `outputs/controller_system_root_cause_audit/telemetry_freshness_report.md`
- `outputs/controller_system_root_cause_audit/controller_configuration/controller_configuration_report.md`
- `outputs/controller_system_root_cause_audit/structural_invariants/structural_invariant_report.md`
- `outputs/controller_system_root_cause_audit/event_order/event_order_report.md`
- `outputs/controller_system_root_cause_audit/subsystems/subsystem_audit_report.md`
- `outputs/controller_system_root_cause_audit/telemetry_completeness_report.md`
- `outputs/controller_system_root_cause_audit/root_cause_map.md`
- `docs/validation/controller_system_root_cause_fix_plan.md`

### Machine-Readable
- `outputs/controller_system_root_cause_audit/controller_system_root_cause_summary.json`

## Constraints Followed

✓ Did NOT add WBC
✓ Did NOT enable legacy WBC paths
✓ Did NOT proceed to Step C
✓ Did NOT proceed to Step D
✓ Did NOT implement differential wheel yaw control
✓ Did NOT implement mode-based hip-yaw control
✓ Did NOT revert hip-yaw sign without audit
✓ Did NOT modify hip-roll
✓ Did NOT tune gains
✓ Did NOT relax thresholds
✓ Did NOT commit

---

## Next Steps

1. **Audit the hip-yaw sign convention** in `shape_posture_controller.py`
2. **Fix the sign** if confirmed incorrect
3. **Validate with 100-step smoke test**
4. **Run 5000-step evaluation** at all three heights
5. **Evaluate if HY2-DIV is needed** based on divergence after sign fix
6. **Proceed only if validation gates pass**

## Validation Gates for Step E Acceptance

| Gate | Metric | Threshold |
|------|--------|-----------|
| 1 | hip_yaw_sign_correct | > 95% |
| 2 | hip_yaw_abs_max (nominal) | < 0.07 rad |
| 3 | hip_yaw_abs_max (low/high) | < 0.15 rad |
| 4 | divergence_rms (boundary) | < 0.10 rad |
| 5 | WBC applied | = 0 Nm |
| 6 | ownership_violations | = 0 |