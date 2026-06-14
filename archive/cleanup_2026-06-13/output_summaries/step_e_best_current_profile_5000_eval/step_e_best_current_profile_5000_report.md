# Step E Best Current Profile 5000-Step Evaluation Report

**Date:** 2026-06-05  
**Evaluation Time:** 13:34-13:54 CST  
**Profile:** J3 (velocity-damped sagittal with strong damping)

---

## Executive Summary

**Evaluation Status:** STEP_E_5000_INVALID_DUE_TO_WBC

All three height evaluations (low_0p300, nominal, high_0p480) completed 5000 steps but **ALL FAILED** due to WBC invariant violation. WBC torque was applied despite `--controller-mode balance-core`, which requires WBC to remain diagnostic-only with zero contribution to final torque.

**Critical Blocker:** WBC applied at all three heights (19.3 Nm, 14.2 Nm, 20.4 Nm respectively). This invalidates the evaluation for Step E acceptance.

**Secondary Blockers (if WBC were fixed):**
- **low_0p300:** Hip yaw 0.281 rad (4.0× threshold), Pitch 0.157 rad (1.57× threshold)
- **high_0p480:** Support error 0.234 m (1.56× threshold), Hip yaw 0.262 rad (3.74× threshold)
- **nominal:** Would PASS all strict gates if WBC were fixed

---

## Selected Profile: J3

### Selection Criteria

Best current profile selected using normalized max violation metric from recent 1000-step evaluations:

```
normalized_violation = max(
    support_error_max_abs / 0.15,
    hip_yaw_abs_max / 0.07,
    pitch_x_max_abs / 0.10
)
```

### Candidate Comparison

| Profile | Support Max | Hip-Yaw Max | Pitch Max | Normalized Violation | Decision |
|---------|-------------|-------------|-----------|----------------------|----------|
| J2 | 0.114 m | 0.137 rad | 0.157 rad | **1.957** | - |
| **J3** | 0.125 m | 0.088 rad | 0.151 rad | **1.513** | ✓ SELECTED |

**Why J3:**
- Lowest normalized violation (23% better than J2)
- Superior hip-yaw control (0.088 vs 0.137 rad, 35% improvement)
- Comparable pitch performance
- Both profiles pass support gate (<0.15 m)

### Profile Parameters

```yaml
k_position: 80.0 (scheduled, nominal: 40.0)
max_position_tau: 6.0 Nm (scheduled, nominal: 3.0 Nm)  
k_velocity: 30.0 (scheduled, nominal: 15.0)
schedule_type: continuous_smoothstep
schedule_range:
  z_low: 0.300 m
  z_high: 0.393 m
```

---

## Evaluation Commands

### 1. low_0p300 (13:34)

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile J3 \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 5000
```

**Setup:** `outputs/physical_target_height_setups/low_0p300_setup.json`  
**Target CoM:** 0.295 m (physical boundary height)

### 2. nominal (13:44)

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile J3 \
  --steps 5000
```

**Setup:** Standard nominal height (no height-variant setup)  
**Target CoM:** 0.400 m (nominal operational height)

### 3. high_0p480 (13:54)

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile J3 \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 5000
```

**Setup:** `outputs/physical_target_height_setups/high_0p480_setup.json`  
**Target CoM:** 0.480 m (physical upper boundary)

---

## Summary Table

| Height | Survived | Support Max | Support RMS | Hip-Yaw Max | Hip-Yaw RMS | Pitch Max | Pitch RMS | Roll Max | Height Error | Contact Valid | Non-Wheel | WBC Max | Hidden τ | Ownership | Verdict |
|--------|----------|-------------|-------------|-------------|-------------|-----------|-----------|----------|--------------|---------------|-----------|---------|----------|-----------|---------|
| **low_0p300** | ✓ | 0.114 m | 0.062 m | **0.281 rad** | 0.181 rad | **0.157 rad** | 0.088 rad | 0.014 rad | 0.016 m | 99.98% | 0 | **19.3 Nm** | 0.0 | 0 | **FAIL** |
| **nominal** | ✓ | 0.104 m | 0.057 m | 0.058 rad | 0.029 rad | 0.071 rad | 0.041 rad | 0.013 rad | 0.008 m | 100.0% | 0 | **14.2 Nm** | 0.0 | 0 | **FAIL** |
| **high_0p480** | ✓ | **0.234 m** | 0.155 m | **0.262 rad** | 0.181 rad | 0.093 rad | 0.058 rad | 0.002 rad | 0.018 m | 99.96% | 0 | **20.4 Nm** | 0.0 | 0 | **FAIL** |

**Strict Step E Gates:**
- Support position error ≤ 0.15 m
- Hip yaw abs max ≤ 0.07 rad
- Pitch x max abs ≤ 0.10 rad
- Roll y max abs ≤ 0.05 rad
- Final height error ≤ 0.02 m
- Contact valid ≥ 99.9%
- Non-wheel contacts = 0
- **WBC applied = false** ← **PRIMARY BLOCKER**
- Hidden torque = 0
- Ownership violations = 0

---

## Detailed Metrics by Height

### low_0p300

**Run Identity:**
- Telemetry: `outputs/step_e_best_current_profile_5000_eval/low_0p300_5000_telemetry.csv`
- Rows: 5000
- Survived: true
- Setup: `outputs/physical_target_height_setups/low_0p300_setup.json`
- Target CoM z: 0.295 m
- Achieved CoM z: 0.280 m (final)
- Schedule active: true
- Effective k_position: 80.0
- Effective k_velocity: 30.0
- Effective max_position_tau: 6.0 Nm

**Support Position / Drift:**
- Max abs: 0.114 m ✓ PASS (<0.15)
- Final: 0.089 m
- RMS: 0.062 m
- Mean: 0.035 m
- Min/Max: -0.059 / 0.114 m
- Time above 0.15 m: 0.0%
- Time above 0.10 m: 12.0%

**Hip-Yaw Posture:**
- Max abs: **0.281 rad** ✗ FAIL (>0.07)
- Final: 0.230 rad
- RMS: 0.181 rad
- Time above 0.07 rad: 84.8%
- Time above 0.10 rad: 80.0%

**Pitch:**
- Max abs: **0.157 rad** ✗ FAIL (>0.10)
- Final: 0.129 rad
- RMS: 0.088 rad
- Mean: 0.046 rad
- Min/Max: -0.097 / 0.157 rad
- Time above 0.10 rad: 33.2%
- Time above 0.15 rad: 2.3%

**Roll:**
- Max abs: 0.014 rad ✓ PASS (<0.05)
- Final: 0.002 rad
- RMS: 0.005 rad
- Time above 0.05 rad: 0.0%

**Height / CoM:**
- CoM z final: 0.280 m
- CoM z min/max: 0.274 / 0.295 m
- Height error max abs: 0.021 m
- Height error final: 0.016 m ✓ PASS (<0.02)
- Height error RMS: 0.012 m
- Time above 0.02 m error: 2.5%

**Contact Validity:**
- Contact valid: 99.98% ✓ PASS (≥99.9%)
- Non-wheel floor contacts max: 0 ✓ PASS
- Non-wheel floor contacts any: 0
- Left/right wheel contact: present

**Wheel Velocity:**
- Mean max abs: 7.25 rad/s
- Mean final: -4.16 rad/s
- Mean RMS: 3.97 rad/s

**WBC / Ownership Invariants:**
- **WBC norm max: 19.3 Nm** ✗ **FAIL (>0)**
- WBC applied: **TRUE** ✗ **INVARIANT VIOLATION**
- Hidden torque norm max: 0.0 ✓ PASS
- Ownership violations max: 0 ✓ PASS
- Controller mode: balance-core (confirmed)

**Verdict:** FAIL (WBC applied + hip-yaw + pitch)

---

### nominal

**Run Identity:**
- Telemetry: `outputs/step_e_best_current_profile_5000_eval/nominal_5000_telemetry.csv`
- Rows: 5000
- Survived: true
- Setup: nominal (no height-variant setup)
- Target CoM z: 0.400 m
- Achieved CoM z: 0.408 m (final)
- Schedule active: false (above z_high threshold)
- Effective k_position: 40.0 (nominal)
- Effective k_velocity: 15.0 (nominal)
- Effective max_position_tau: 3.0 Nm (nominal)

**Support Position / Drift:**
- Max abs: 0.104 m ✓ PASS (<0.15)
- Final: 0.091 m
- RMS: 0.057 m
- Mean: 0.033 m
- Min/Max: -0.045 / 0.104 m
- Time above 0.15 m: 0.0%
- Time above 0.10 m: 2.0%

**Hip-Yaw Posture:**
- Max abs: 0.058 rad ✓ PASS (<0.07)
- Final: 0.009 rad
- RMS: 0.029 rad
- Time above 0.07 rad: 0.0%
- Time above 0.10 rad: 0.0%

**Pitch:**
- Max abs: 0.071 rad ✓ PASS (<0.10)
- Final: 0.066 rad
- RMS: 0.041 rad
- Mean: 0.023 rad
- Min/Max: -0.037 / 0.071 rad
- Time above 0.10 rad: 0.0%

**Roll:**
- Max abs: 0.013 rad ✓ PASS (<0.05)
- Final: 0.004 rad
- RMS: 0.007 rad
- Time above 0.05 rad: 0.0%

**Height / CoM:**
- CoM z final: 0.408 m
- CoM z min/max: 0.404 / 0.409 m
- Height error max abs: 0.009 m
- Height error final: 0.008 m ✓ PASS (<0.02)
- Height error RMS: 0.008 m
- Time above 0.02 m error: 0.0%

**Contact Validity:**
- Contact valid: 100.0% ✓ PASS (≥99.9%)
- Non-wheel floor contacts max: 0 ✓ PASS
- Non-wheel floor contacts any: 0
- Left/right wheel contact: present

**Wheel Velocity:**
- Mean max abs: 5.15 rad/s
- Mean final: -3.67 rad/s
- Mean RMS: 3.16 rad/s

**WBC / Ownership Invariants:**
- **WBC norm max: 14.2 Nm** ✗ **FAIL (>0)**
- WBC applied: **TRUE** ✗ **INVARIANT VIOLATION**
- Hidden torque norm max: 0.0 ✓ PASS
- Ownership violations max: 0 ✓ PASS
- Controller mode: balance-core (confirmed)

**Verdict:** FAIL (WBC applied only - all other gates PASS)

---

### high_0p480

**Run Identity:**
- Telemetry: `outputs/step_e_best_current_profile_5000_eval/high_0p480_5000_telemetry.csv`
- Rows: 5000
- Survived: true
- Setup: `outputs/physical_target_height_setups/high_0p480_setup.json`
- Target CoM z: 0.480 m
- Achieved CoM z: 0.470 m (final)
- Schedule active: false (above z_high threshold)
- Effective k_position: 40.0 (nominal)
- Effective k_velocity: 15.0 (nominal)
- Effective max_position_tau: 3.0 Nm (nominal)

**Support Position / Drift:**
- Max abs: **0.234 m** ✗ FAIL (>0.15)
- Final: 0.230 m
- RMS: 0.155 m
- Mean: 0.083 m
- Min/Max: -0.149 / 0.234 m
- Time above 0.15 m: 51.3%
- Time above 0.10 m: 90.6%

**Hip-Yaw Posture:**
- Max abs: **0.262 rad** ✗ FAIL (>0.07)
- Final: 0.241 rad
- RMS: 0.181 rad
- Time above 0.07 rad: 98.7%
- Time above 0.10 rad: 95.6%

**Pitch:**
- Max abs: 0.093 rad ✓ PASS (<0.10)
- Final: 0.072 rad
- RMS: 0.058 rad
- Mean: 0.029 rad
- Min/Max: -0.055 / 0.093 rad
- Time above 0.10 rad: 0.0%

**Roll:**
- Max abs: 0.002 rad ✓ PASS (<0.05)
- Final: 0.001 rad
- RMS: 0.001 rad
- Time above 0.05 rad: 0.0%

**Height / CoM:**
- CoM z final: 0.470 m
- CoM z min/max: 0.462 / 0.488 m
- Height error max abs: 0.018 m
- Height error final: 0.010 m ✓ PASS (<0.02)
- Height error RMS: 0.011 m
- Time above 0.02 m error: 0.0%

**Contact Validity:**
- Contact valid: 99.96% ✓ PASS (≥99.9%)
- Non-wheel floor contacts max: 0 ✓ PASS
- Non-wheel floor contacts any: 0
- Left/right wheel contact: present

**Wheel Velocity:**
- Mean max abs: 12.01 rad/s
- Mean final: -8.52 rad/s
- Mean RMS: 7.32 rad/s

**WBC / Ownership Invariants:**
- **WBC norm max: 20.4 Nm** ✗ **FAIL (>0)**
- WBC applied: **TRUE** ✗ **INVARIANT VIOLATION**
- Hidden torque norm max: 0.0 ✓ PASS
- Ownership violations max: 0 ✓ PASS
- Controller mode: balance-core (confirmed)

**Verdict:** FAIL (WBC applied + support error + hip-yaw)

---

## Final Decision

**Overall Verdict:** STEP_E_5000_INVALID_DUE_TO_WBC_OR_OWNERSHIP

All three heights failed due to **WBC invariant violation**. Despite `--controller-mode balance-core`, WBC contributed 14-20 Nm to final torque at all heights, invalidating the evaluation.

### Primary Blocker

**WBC Applied = True** at all heights:
- low_0p300: 19.3 Nm
- nominal: 14.2 Nm  
- high_0p480: 20.4 Nm

This violates the balance-core invariant requirement that WBC must remain diagnostic-only with zero contribution to applied torque. The simulator or controller integration has a bug that allows WBC to contribute despite balance-core mode.

### Secondary Blockers (if WBC were fixed)

**low_0p300:**
1. Hip yaw: 0.281 rad (401% of 0.07 threshold)
2. Pitch: 0.157 rad (157% of 0.10 threshold)

**nominal:**
- Would **PASS** all strict gates

**high_0p480:**
1. Support error: 0.234 m (156% of 0.15 threshold)
2. Hip yaw: 0.262 rad (374% of 0.07 threshold)

---

## Step E Reliability Assessment

**Current State:**

- **low_0p300 (z=0.300m):** NOT RELIABLE
  - Even if WBC fixed: hip-yaw and pitch exceed thresholds by 4× and 1.6× respectively
  - Fundamental position-pitch coupling at boundary height
  - J3 profile insufficient for strict acceptance

- **nominal (z=0.400m):** POTENTIALLY RELIABLE
  - If WBC fixed: all gates would PASS
  - Support: 0.104 m (✓), Hip-yaw: 0.058 rad (✓), Pitch: 0.071 rad (✓), Roll: 0.013 rad (✓)
  - Only blocker is WBC bug

- **high_0p480 (z=0.480m):** NOT RELIABLE  
  - Even if WBC fixed: support error and hip-yaw exceed thresholds
  - Nominal baseline gains insufficient at high height
  - J3 schedule only active below z=0.393m, does not help at 0.480m

---

## Top Blockers by Height

### low_0p300
1. **WBC applied** (19.3 Nm) - invalidates evaluation
2. **Hip-yaw** (0.281 rad, 4.0× threshold) - fundamental drift at boundary
3. **Pitch** (0.157 rad, 1.57× threshold) - position-pitch coupling

### nominal
1. **WBC applied** (14.2 Nm) - ONLY blocker, all other gates pass

### high_0p480
1. **WBC applied** (20.4 Nm) - invalidates evaluation
2. **Support error** (0.234 m, 1.56× threshold) - baseline gains insufficient
3. **Hip-yaw** (0.262 rad, 3.74× threshold) - high-height instability

---

## Artifacts Generated

- `step_e_best_current_profile_selection.md` - Profile selection rationale
- `step_e_best_current_profile_selection.json` - Selection metadata
- `low_0p300_5000_telemetry.csv` - Low height telemetry (5000 rows)
- `nominal_5000_telemetry.csv` - Nominal height telemetry (5000 rows)
- `high_0p480_5000_telemetry.csv` - High height telemetry (5000 rows)
- `step_e_best_current_profile_5000_metrics.json` - Complete metrics for all heights
- `step_e_best_current_profile_5000_pass_fail_summary.json` - Gate verdicts
- `step_e_best_current_profile_5000_report.md` - This report

---

## Recommendations

### Immediate Action Required

**Fix WBC invariant violation before any further Step E evaluation.**

The simulator or controller integration has a bug that allows WBC to contribute torque despite `--controller-mode balance-core`. This must be debugged and fixed first, as it invalidates all current Step E evaluations.

Verify:
1. `simulate_hierarchical_controller.py` balance-core mode initialization
2. Torque composition in balance-core mode
3. WBC authority scaling/gating in balance-core
4. Telemetry reporting of WBC norm vs applied contribution

### After WBC Fix

**nominal:** Re-evaluate immediately - likely to pass all gates

**low_0p300:** Consider:
- Relaxing pitch threshold to 0.15 rad for boundary heights
- OR accepting z=0.330m as operational lower bound
- OR implementing pitch-aware position control

**high_0p480:** Requires:
- High-height sagittal schedule (k_position, max_tau increases above z=0.393m)
- OR accepting z=0.450m as operational upper bound

---

## References

- Profile selection: [J3 vs J2 comparison](step_e_best_current_profile_selection.md)
- Profile design: [Joint Low-Height Sagittal-Yaw Fix Design](../../docs/validation/joint_low_height_sagittal_yaw_fix_design.md)
- Pitch-safe candidates (failed): [Pitch-Safe Joint Fix Report](../../docs/validation/pitch_safe_joint_sagittal_yaw_fix_report.md)
- Setup files: `outputs/physical_target_height_setups/`
