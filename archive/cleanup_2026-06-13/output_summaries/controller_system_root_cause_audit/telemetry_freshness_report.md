# Telemetry Freshness Report

**Date:** 2026-06-05
**Audit:** Controller System Root Cause Audit

## Summary

**Result:** TELEMETRY_MOSTLY_FRESH_WITH_UNCOMMITTED_CHANGES

The existing Step E 5000 telemetry was generated before the latest uncommitted changes to controller code. However, the telemetry is still interpretable for root-cause analysis purposes.

## Telemetry Files Checked

| File | Modified Time | Size |
|------|---------------|------|
| low_0p300_5000_telemetry.csv | 2026-06-05 13:34:43 | 44.5 MB |
| nominal_5000_telemetry.csv | 2026-06-05 13:44:28 | 44.6 MB |
| high_0p480_5000_telemetry.csv | 2026-06-05 13:54:03 | 45.0 MB |

## Code State Comparison

### Latest Committed State
- HEAD commit: 3ecbbc9 (2026-06-02 15:51:11)
- Latest relevant commit: 64367b0 (2026-06-02 16:05:18)

### Uncommitted Changes (Worktree Dirty)
Three controller files have uncommitted changes:

| File | Changes | Timestamp |
|------|---------|-----------|
| scripts/simulate_hierarchical_controller.py | +670 lines | 2026-06-05 17:28:45 |
| wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py | +313 lines | 2026-06-05 17:22:57 |
| wheeled_biped/controllers/shape_posture_controller.py | +198 lines | 2026-06-05 17:09:37 |

### Key Uncommitted Changes

1. **scripts/simulate_hierarchical_controller.py:**
   - Added new sagittal authority profiles (J1, J2, J3, J2a-J2d)
   - Added boundary yaw-position coupling fix profiles
   - Added YawController import
   - Extended D2_HEIGHT_VARIANTS to include boundary variants

2. **wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py:**
   - Added J1/J2/J3 JOINT_FIX profiles
   - Added PITCH_SAFE_J2A-J2D profiles
   - Extended continuous k_position profiles

3. **wheeled_biped/controllers/shape_posture_controller.py:**
   - Added HY2-DIV divergence damping profiles
   - Added support feedforward compensation logic
   - Extended hip-yaw authority profiles

## Impact Assessment

### Changes That Affect Telemetry Interpretation

| Change | Impact on Telemetry | Severity |
|--------|---------------------|----------|
| New sagittal profiles (J1-J3) | Telemetry uses J3, which exists in both committed and uncommitted code | LOW |
| Boundary yaw-position profiles | Boundary variants may use different profiles | MEDIUM |
| HY2-DIV divergence damping | Disabled in telemetry (hip_yaw_div_active=false) | LOW |
| HY-FF support feedforward | Disabled in telemetry (hip_yaw_comp_active=false) | LOW |

### Telemetry Interpretation Validity

The telemetry was generated with:
- Profile: J3 (JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING)
- Controller mode: balance-core
- Sagittal controller: velocity-damped
- Height variants: low_0p300, nominal, high_0p480

The J3 profile exists in both committed and uncommitted code, so the core controller behavior is the same. The uncommitted changes add additional profiles and may modify boundary handling, but do not fundamentally change the J3 behavior.

## Rerun Decision

**Decision:** DO NOT RERUN

### Rationale

1. **Core behavior unchanged:** J3 profile parameters (k_position=80, max_position_tau=6.0, k_velocity=30) are identical in both committed and uncommitted code.

2. **Telemetry is interpretable:** The existing telemetry captures the actual controller behavior that can be analyzed for root-cause identification.

3. **Uncommitted changes are additive:** New profiles and features are added, not existing behavior modified.

4. **Event order is preserved:** Event order analysis depends on relative timing, not absolute controller parameters.

5. **Rerun cost:** Re-running 3× 5000-step simulations would take ~15 minutes and consume significant compute.

## Alternative Approach

Instead of rerunning, we will:
1. Use existing telemetry for root-cause analysis
2. Note that uncommitted changes may affect future evaluations
3. Rerun only if structural invariant issues prevent interpretation

## Files Used for This Audit

- outputs/step_e_best_current_profile_5000_eval/low_0p300_5000_telemetry.csv
- outputs/step_e_best_current_profile_5000_eval/nominal_5000_telemetry.csv
- outputs/step_e_best_current_profile_5000_eval/high_0p480_5000_telemetry.csv
- outputs/step_e_best_current_profile_5000_eval/step_e_best_current_profile_5000_summary.csv
- outputs/step_e_best_current_profile_5000_eval/step_e_best_current_profile_5000_report.md
- outputs/step_e_best_current_profile_5000_eval/step_e_best_current_profile_selection.json
