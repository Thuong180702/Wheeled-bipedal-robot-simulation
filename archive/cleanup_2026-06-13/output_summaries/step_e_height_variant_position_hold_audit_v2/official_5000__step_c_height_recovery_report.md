# Step C Height Recovery Report

## Summary

- Overall verdict: **FAIL**
- Final decision: **STEP_C_FIX_REQUIRED**
- Controller behavior changed: `False`
- WBC applied: `False`
- Step E invariants preserved: `False`

## Case results

| Case | Verdict | Primary failure |
|---|---|---|
| nominal | PASS |  |
| low_tiny | PASS |  |
| high_tiny | FAIL | position_regression |
| low_small | PASS |  |
| high_small | FAIL | position_regression |

## Artifacts

- height_reference: `outputs\step_e_height_variant_position_hold_audit_v2\official_5000\step_c_height_reference.json`
- case_matrix: `outputs\step_e_height_variant_position_hold_audit_v2\official_5000\step_c_height_case_matrix.json`
- metrics: `outputs\step_e_height_variant_position_hold_audit_v2\official_5000\step_c_height_recovery_metrics.json`
- failure_classification: `outputs\step_e_height_variant_position_hold_audit_v2\official_5000\step_c_failure_classification.json`
- summary: `outputs\step_e_height_variant_position_hold_audit_v2\official_5000\step_c_pass_fail_summary.json`
- report: `outputs\step_e_height_variant_position_hold_audit_v2\official_5000\step_c_height_recovery_report.md`
