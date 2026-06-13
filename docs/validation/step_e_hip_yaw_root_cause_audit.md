# Step E Hip-Yaw Root Cause Audit

**Date:** 2026-06-05

**Objective:** Diagnostic-first systematic analysis of hip-yaw posture failure

## Executive Summary

This audit investigates the root cause of hip-yaw posture failure across
three height variants (low_0p300, nominal, high_0p480) in the Step E controller.

### Priority Order

1. Robot must survive and keep contact/height
2. Robot must keep posture, especially hip-yaw (legs must not twist)
3. Support-position drift should be improved
4. Pitch handled later by task-aware pitch control (not blocking this audit)

## Telemetry Source

- **Directory:** `outputs\step_e_best_current_profile_5000_eval`
- **Simulations rerun:** False
- **Files used:**
  - low_0p300_5000_telemetry.csv
  - nominal_5000_telemetry.csv
  - high_0p480_5000_telemetry.csv

## Metrics Summary

| Case | Survived | Support Max | Hip-Yaw Max | Pitch Max | Roll Max | Height Error |
|------|----------|-------------|-------------|-----------|----------|---------------|
| low_0p300 | ✓ | 0.1142 m | 0.2807 rad | 0.1572 rad | 0.0140 rad | 0.0159 m |
| nominal | ✓ | 0.1045 m | 0.0576 rad | 0.0708 rad | 0.0130 rad | 0.0082 m |
| high_0p480 | ✓ | 0.2336 m | 0.2619 rad | 0.0926 rad | 0.0023 rad | 0.0151 m |

## Event Order Analysis

| Case | Classification | Hip-Yaw 0.03 | Support 0.05 | Hip-Yaw 0.10 | Support 0.15 |
|------|----------------|--------------|--------------|--------------|---------------|
| low_0p300 | support_first | 273 | 42 | 699 | -1 |
| nominal | support_first | 464 | 84 | -1 | -1 |
| high_0p480 | support_first | 716 | 71 | 2258 | 108 |

## WBC Structural Invariant Status

### low_0p300

- **Classification:** `WBC_DIAGNOSTIC_ONLY`
- **WBC norm max:** 19.2993 Nm
- **Hidden torque max:** 0.0000 Nm

### nominal

- **Classification:** `WBC_DIAGNOSTIC_ONLY`
- **WBC norm max:** 14.2073 Nm
- **Hidden torque max:** 0.0000 Nm

### high_0p480

- **Classification:** `WBC_DIAGNOSTIC_ONLY`
- **WBC norm max:** 20.3686 Nm
- **Hidden torque max:** 0.0000 Nm

## Hip-Yaw Failure Mechanism Classification

### low_0p300

**Mechanism:** `hip_yaw_torque_sign_error`

**Evidence:**

- divergence RMS (0.3575) > common mode RMS
- torque does not consistently oppose error
- event order: support_first

### nominal

**Mechanism:** `hip_yaw_torque_sign_error`

**Evidence:**

- divergence RMS (0.0447) > common mode RMS
- torque does not consistently oppose error
- event order: support_first

### high_0p480

**Mechanism:** `hip_yaw_torque_sign_error`

**Evidence:**

- divergence RMS (0.2825) > common mode RMS
- torque does not consistently oppose error
- event order: support_first

## Pitch Policy Statement

Pitch is tracked and reported in this audit but is **not the primary objective**.

Pitch will later be converted to task-aware pitch control:
- Static pitch reference for standing
- Dynamic pitch reference for future forward/backward motion
- Absolute safety bound

For this audit:
- Pitch metrics are recorded
- Pitch must not cause fall/contact loss/height failure
- Hip-yaw posture failure is prioritized

## Final Decision

**Decision:** `HIP_YAW_ROOT_CAUSE_IDENTIFIED`

## Fix Recommendations

### Priority 1: Hip-Yaw Torque Sign Correction (REQUIRED)

**Root Cause:** Shape posture controller applies hip-yaw torque with wrong sign (0.22-14.88% correct).

**Fix Options:**

**Option A: Sign Convention Audit (RECOMMENDED FIRST STEP)**
- Audit `wheeled_biped/controllers/shape_posture_controller.py` hip-yaw PD control law
- Check if error calculation has wrong sign: `error = ref - pos` vs `error = pos - ref`
- Check if torque application has wrong sign: `tau = kp * error + kd * error_dot` vs `tau = -(kp * error + kd * error_dot)`
- Check joint indexing: verify left/right hip-yaw joint indices are not swapped
- **Validation:** Rerun 100-step test, expect torque sign correctness > 95%

**Option B: Direct Sign Flip**
- If Option A audit reveals consistent sign error, apply negation to hip-yaw torque output
- Location: `shape_posture_controller.py`, hip-yaw torque computation
- **Risk:** Low if audit confirms consistent sign error across all scenarios
- **Validation:** 100-step test → 5000-step nominal → 5000-step low/high

**Option C: Reference Sign Flip**
- If hip-yaw reference sign convention is inverted in equilibrium capture
- Check `simulate_hierarchical_controller.py` equilibrium reference setup
- **Risk:** Medium - affects initialization logic
- **Validation:** Same as Option B

**Implementation Sequence:**
1. Run Option A audit (diagnostic script, no controller changes)
2. Based on audit findings, implement Option B or C
3. Validate with 100-step smoke test (expect sign correctness > 95%)
4. If successful, run full 5000-step Step E evaluation at all three heights
5. Verify hip-yaw abs max < 0.05 rad at nominal, < 0.15 rad at low/high

**Required Telemetry for Validation:**
- `hip_yaw_torque_sign_correct_left`
- `hip_yaw_torque_sign_correct_right`
- `l_hip_yaw_error`, `r_hip_yaw_error`
- `l_hip_yaw_tau_shape_final`, `r_hip_yaw_tau_shape_final`
- `hip_yaw_abs_max`

**Expected Outcome:**
- Hip-yaw torque sign correctness: > 95%
- Hip-yaw abs max at nominal: < 0.03 rad (currently 0.058 rad)
- Hip-yaw abs max at low_0p300: < 0.15 rad (currently 0.281 rad)
- Hip-yaw abs max at high_0p480: < 0.15 rad (currently 0.262 rad)
- Divergence RMS should approach 0 (antisymmetric instability eliminated)

### Priority 2: Support Drift (AFTER Priority 1)

Support drift is secondary to hip-yaw posture failure. Address after hip-yaw sign is corrected and validated.

### Pitch Policy Reminder

Pitch is tracked but **not the primary fix objective**. Pitch will be handled by task-aware pitch control in a separate phase.

## Restrictions Followed

- ✓ Did NOT add WBC
- ✓ Did NOT enable legacy WBC paths
- ✓ Did NOT modify hip-roll
- ✓ Did NOT proceed to Step C or Step D
- ✓ Did NOT commit

## Artifacts Generated

All artifacts saved to: `outputs/hip_yaw_root_cause_audit/`

- `hip_yaw_root_cause_summary.json`
- `hip_yaw_event_order_comparison.csv`
- `hip_yaw_reference_command_audit.csv`
- `hip_yaw_torque_authority_audit.csv`
- `hip_yaw_correlation_lag_audit.csv`
- `low_0p300_hip_yaw_peak_window.csv`
- `nominal_hip_yaw_peak_window.csv`
- `high_0p480_hip_yaw_peak_window.csv`

