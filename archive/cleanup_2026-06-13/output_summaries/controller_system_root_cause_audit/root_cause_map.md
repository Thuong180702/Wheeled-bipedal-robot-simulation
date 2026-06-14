# Root-Cause Map

**Date:** 2026-06-05
**Phase:** Phase 7

## Summary

| Category | Count |
|----------|-------|
| Confirmed Issues | 2 |
| Likely Issues | 1 |
| Ruled Out Issues | 2 |
| Ambiguous Issues | 1 |
| Non-Issues | 1 |

## Confirmed Issues

### CONFIRMED_1: Hip-Yaw Torque Sign Convention Error

**Severity:** HIGH | **Confidence:** HIGH | **Blocks Step E:** Yes

**Affected Heights:** low_0p300, nominal, high_0p480

**Evidence:**
- `hip_yaw_torque_sign_correct_left` = 0%
- `hip_yaw_torque_sign_correct_right` = 0%
- Evidence type: direct_telemetry

**First Event Timing:**
| Height | Step | Time |
|--------|------|------|
| low_0p300 | 273 | 2.73s |
| nominal | 464 | 4.64s |
| high_0p480 | 716 | 7.16s |

**Description:**
Hip-yaw torque sign correctness is 0% across all heights. The torque does not consistently oppose the error. This causes divergence accumulation at boundary heights and prevents the controller from correcting hip-yaw posture.

**Root Cause Hypothesis:**
The shape_posture_controller applies hip-yaw torque with incorrect sign. The PD control law may have inverted sign convention.

**Fix Validation Needed:**
Verify hip-yaw PD control law sign convention in `wheeled_biped/controllers/shape_posture_controller.py`.

---

### CONFIRMED_2: Hip-Yaw Divergence at Boundary Heights

**Severity:** HIGH | **Confidence:** HIGH | **Blocks Step E:** Yes

**Affected Heights:** low_0p300, high_0p480

**Evidence:**
| Height | Divergence RMS |
|--------|----------------|
| low_0p300 | 0.3575 rad |
| nominal | 0.0447 rad |
| high_0p480 | 0.2825 rad |

Evidence type: direct_telemetry

**First Event Timing:**
| Height | Step | Time |
|--------|------|------|
| low_0p300 | 699 | 6.99s |
| high_0p480 | 2258 | 22.58s |

**Description:**
Hip-yaw divergence (left-right asymmetry) is 8x higher at boundary heights (0.3575, 0.2825) compared to nominal (0.0447). Legs twist inward/outward at low and high heights.

**Root Cause Hypothesis:**
This is likely a secondary effect of the torque sign convention error (CONFIRMED_1). When torque doesn't oppose error correctly, divergence accumulates. May also need mode-based control (separate common/divergence modes).

**Fix Validation Needed:**
1. Fix hip-yaw sign first (CONFIRMED_1)
2. Evaluate if additional divergence damping is needed

---

## Likely Issues

### LIKELY_1: Hip-Roll Saturation at Low Heights

**Severity:** MEDIUM | **Confidence:** MEDIUM | **Blocks Step E:** No (Diagnostic only)

**Affected Heights:** low_0p300, nominal

**Evidence:**
| Height | Hip-Roll Abs Max |
|--------|------------------|
| low_0p300 | 0.2167 rad |
| nominal | 0.1749 rad |
| high_0p480 | 0.0773 rad |

Evidence type: correlated_telemetry

**Description:**
Hip-roll abs_max is 3x higher at low_0p300 (0.2167) vs high_0p480 (0.0773). May indicate lateral controller saturation or insufficient authority at low heights.

**Root Cause Hypothesis:**
Lateral roll balance controller may have insufficient authority or gain scheduling issue at low heights.

**Fix Validation Needed:**
Correlate with roll_y and lateral controller outputs to determine if this is a cause or effect.

---

## Ruled Out Issues

### RULED_OUT_1: WBC Applied to Joints

**Confidence:** HIGH

**Evidence:**
- `tau_wbc_scaled_per_joint` = all zeros
- `hidden_torque_norm` = 0.0 Nm

**Description:**
WBC is correctly configured as diagnostic-only in balance-core mode. The previous report incorrectly classified this.

---

### RULED_OUT_2: Body Yaw Instability

**Confidence:** HIGH

**Evidence:**
| Height | Yaw Drift Max |
|--------|---------------|
| low_0p300 | 0.0149 rad |
| nominal | 0.0946 rad |
| high_0p480 | 0.1036 rad |

**Description:**
Body yaw is stable across all heights. yaw_drift_max < 0.1 rad at all heights. Classification: stable.

---

## Ambiguous Issues

### AMBIGUOUS_1: Support Position Error Reporting

**Severity:** LOW | **Confidence:** LOW

**Affected Heights:** all

**Evidence:**
- `support_position_error` column shows 0.0 m
- But other metrics (com_x, com_y) show some drift

**Description:**
May be a column naming/calculation issue in telemetry. Requires investigation of how support_position_error is computed.

---

## Non-Issues

### NON_ISSUE_1: Pitch Behavior at Low Heights

**Description:**
Pitch is larger at low_0p300 (0.1571 rad) vs nominal (0.0708 rad) and high_0p480 (0.0926 rad). This is expected behavior and will be handled by task-aware pitch control in a future phase. Not a root cause to fix in this audit.

---

## Priority Order for Fix

| Priority | Issue ID | Name | Reason |
|----------|----------|------|--------|
| 1 | CONFIRMED_1 | Hip-Yaw Torque Sign Convention Error | 0% sign correctness means controller cannot correct hip-yaw error |
| 2 | CONFIRMED_2 | Hip-Yaw Divergence at Boundary Heights | Likely secondary effect of sign error; 8x higher at boundary heights |
| 3 | LIKELY_1 | Hip-Roll Saturation at Low Heights | May improve with lateral controller tuning; not blocking |

---

## Fix Plan Constraints

The following are NOT allowed:
- Add WBC
- Enable legacy WBC paths
- Proceed to Step C
- Proceed to Step D
- Implement differential wheel yaw control
- Implement mode-based hip-yaw control
- Revert hip-yaw sign (do not change sign without audit)
- Modify hip-roll
- Tune gains
- Relax thresholds
- Use root-z-only perturbation
- Use discontinuous schedules
- Use variant-name-only patches
- Commit