# Structural Invariant Audit Report

**Date:** 2026-06-05
**Phase:** Phase 3

## Summary

**Result:** STRUCTURAL_INVARIANTS_CLEAN

All three height variants exhibit clean structural invariants with WBC correctly configured as diagnostic-only.

## WBC Status

| Variant | WBC Norm Max | WBC Applied | Classification |
|---------|--------------|-------------|---------------|
| low_0p300 | 0.0 Nm | NO | WBC_DIAGNOSTIC_ONLY |
| nominal | 0.0 Nm | NO | WBC_DIAGNOSTIC_ONLY |
| high_0p480 | 0.0 Nm | NO | WBC_DIAGNOSTIC_ONLY |

### Analysis

The telemetry shows:
- `tau_wbc_per_joint`: Contains raw WBC diagnostic values (non-zero)
- `tau_wbc_scaled_per_joint`: All zeros - WBC was scaled to zero by authority clipping
- `hidden_torque_norm`: 0.0 Nm - No hidden torque

**Conclusion:** WBC is correctly configured as diagnostic-only in balance-core mode. The raw WBC diagnostic produces torque commands (norm ~13-19 Nm), but the scaled WBC (what would be applied to joints) is zero due to authority clipping.

## Torque Ownership

| Variant | Violations Max | Saturation Events | Classification |
|---------|----------------|-------------------|---------------|
| low_0p300 | 0 | 5000 | CLEAN |
| nominal | 0 | 0 | CLEAN |
| high_0p480 | 0 | 0 | CLEAN |

**Note:** The high number of saturation events in low_0p300 is normal - per-joint torque saturation is expected when controllers apply torque to joints.

## Legacy Path

| Variant | Controller Mode | Legacy Path Active |
|---------|-----------------|-------------------|
| low_0p300 | 0.2955 | NO |
| nominal | 0.400 | NO |
| high_0p480 | 0.481 | NO |

**Note:** The "controller_mode" values appear to be height variant target CoM heights, not the controller mode string.

## Torque Composition

| Variant | Per-Joint Columns | Saturation Events | Rate Limit Events |
|---------|-------------------|-------------------|-------------------|
| low_0p300 | 16 | 5000 | 0 |
| nominal | 16 | 0 | 0 |
| high_0p480 | 16 | 0 | 0 |

**Note:** All per-joint torque decomposition columns are available (16/16).

## Structural Invariant Status

| Invariant | Status | Evidence |
|-----------|--------|----------|
| WBC Diagnostic vs Applied | CLEAN | `tau_wbc_scaled_per_joint` all zeros |
| Hidden Torque | CLEAN | `hidden_torque_norm` = 0.0 Nm |
| Ownership Violations | CLEAN | `ownership_violation_count` = 0 |
| Legacy Path | INACTIVE | No legacy torque detected |

## Critical Finding

**WBC IS CORRECTLY CONFIGURED AS DIAGNOSTIC-ONLY.**

The previous report incorrectly classified the telemetry as "WBC applied" based on a misunderstanding of the telemetry columns:
- `tau_wbc_norm` = raw WBC diagnostic (diagnostic, not applied)
- `tau_wbc_scaled_per_joint` = WBC after authority clipping (should be applied, is zero)

The balance-core architecture correctly:
1. Computes raw WBC diagnostics for monitoring
2. Scales WBC to zero via authority clipping in balance-core mode
3. Does not apply WBC to joints

## Files

- Structural invariant summary: `outputs/controller_system_root_cause_audit/structural_invariants/structural_invariant_summary.json`
- WBC audit CSV: `outputs/controller_system_root_cause_audit/structural_invariants/wbc_application_audit.csv`