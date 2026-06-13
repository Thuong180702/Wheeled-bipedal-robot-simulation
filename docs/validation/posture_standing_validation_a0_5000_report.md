# Posture/Standing Validation A0 5000-Step Report

**Date:** 2026-06-06
**Phase:** POSTURE_STANDING_VALIDATION
**Profile:** HY2-DIV A0 (k=5.0, kd=1.0, tau_max=0.5, z_low=0.300, z_high=0.393)
**Steps:** 5000

## Executive Summary

A0 5000-step validation completed at all three heights. All runs **survived full steps** with valid contact and bounded roll. However, **hip-yaw divergence exceeded targets** at all heights, and **HY2-DIV A0 authority is insufficient** at low height (88.74% clipping).

**Final decision:** `POSTURE_REQUIRES_STRONGER_HY2_PROFILE`

This is NOT:
- Official Step E pass
- Ready for Step C or Step D

This IS:
- Evidence that A0 gate pass-through works correctly
- Evidence that A0 is safe (survived, no collapse, no WBC)
- Evidence that A0 torque authority (tau_max=0.5) is insufficient at low height

## Key Metrics Summary

| Height | Survived | Div RMS (rad) | Target (rad) | HY2 Gate | HY2 Clip% | Roll Max (rad) | Contact% |
|--------|----------|---------------|--------------|----------|-----------|----------------|----------|
| nominal | ✓ | 0.245 | < 0.10 | 0% | 0% | 0.014 | 99.98% |
| low_0p300 | ✓ | 0.493 | < 0.30 | 100% | **88.74%** | 0.012 | 99.98% |
| high_0p480 | ✓ | 0.340 | < 0.25 | 0% | 0% | 0.002 | 99.98% |

## Detailed Results

### 1. nominal (0.404m target)

**Survival/Contact/Height:**
- Survived: ✓ (5000/5000 steps)
- Contact valid: 99.98%
- Left/right wheel contact: 100%/100%
- Non-wheel contacts: 0
- Height error: max=0.017m, final=0.017m, RMS=0.007m
- Final COM z: 0.387m (target 0.404m)

**Hip-Yaw/Posture:**
- Divergence RMS: **0.245 rad** (target < 0.10 rad) ✗
- Divergence max: 0.507 rad
- Hip-yaw abs max: 0.254 rad (target < 0.30 rad) ✓
- Common mode: RMS=0.007 rad (bounded)

**HY2-DIV Telemetry:**
- Enabled: ✓
- Gate active: **0%** (com_z ~0.39 > z_high=0.393)
- Effective k/kd: 0.0 (not active)
- Torque: 0.0 (not active)
- Clipping: 0%

**Roll:** 0.014 rad max (bounded) ✓

**Pitch (DEFERRED):** max=0.089 rad, RMS=0.044 rad ✓

**Support Drift (DEFERRED):** max=0.159m, final=0.065m, RMS=0.066m

**Posture Result:** POSTURE_REQUIRES_STRONGER_HY2_PROFILE

### 2. low_0p300 (0.300m target)

**Survival/Contact/Height:**
- Survived: ✓ (5000/5000 steps)
- Contact valid: 99.98%
- Left/right wheel contact: 100%/100%
- Non-wheel contacts: 0
- Height error: max=0.033m, final=0.027m, RMS=0.019m
- Final COM z: 0.269m (target 0.300m)

**Hip-Yaw/Posture:**
- Divergence RMS: **0.493 rad** (target < 0.30 rad) ✗
- Divergence max: 0.785 rad
- Hip-yaw abs max: **0.393 rad** (target < 0.30 rad) ✗
- Common mode: RMS=0.003 rad (bounded)

**HY2-DIV Telemetry:**
- Enabled: ✓
- Gate active: **100%** (com_z ~0.27 < z_low=0.300)
- Effective k/kd: 5.0/1.0 (fully active)
- Torque max: 0.5 Nm (at tau_max limit)
- **Clipping: 88.74%** ← INSUFFICIENT AUTHORITY

**Roll:** 0.012 rad max (bounded) ✓

**Pitch (DEFERRED):** max=0.154 rad, RMS=0.087 rad ✓

**Support Drift (DEFERRED):** max=0.110m, final=0.103m, RMS=0.061m

**Posture Result:** POSTURE_REQUIRES_STRONGER_HY2_PROFILE

### 3. high_0p480 (0.480m target)

**Survival/Contact/Height:**
- Survived: ✓ (5000/5000 steps)
- Contact valid: 99.98%
- Left/right wheel contact: 100%/100%
- Non-wheel contacts: 0
- Height error: max=0.030m, final=0.030m, RMS=0.011m
- Final COM z: 0.451m (target 0.480m)

**Hip-Yaw/Posture:**
- Divergence RMS: **0.340 rad** (target < 0.25 rad) ✗
- Divergence max: 0.689 rad
- Hip-yaw abs max: **0.345 rad** (target < 0.30 rad) ✗
- Common mode: RMS=0.0005 rad (bounded)

**HY2-DIV Telemetry:**
- Enabled: ✓
- Gate active: **0%** (com_z ~0.45 > z_high=0.393)
- Effective k/kd: 0.0 (not active)
- Torque: 0.0 (not active)
- Clipping: 0%

**Roll:** 0.002 rad max (bounded) ✓

**Pitch (DEFERRED):** max=0.092 rad, RMS=0.058 rad ✓

**Support Drift (DEFERRED):** max=0.378m, final=0.039m, RMS=0.278m

**Posture Result:** POSTURE_REQUIRES_STRONGER_HY2_PROFILE

## Gate Results

### Priority 1: Survival/Contact/Height ✓ ALL PASS
- gate_survived_full_run: ✓ (all 3)
- gate_contact_valid: ✓ (all 3 at 98%)
- gate_no_nonwheel_contacts: ✓ (all 0)
- gate_height_error_acceptable: ✓ (all < 0.05m)
- gate_no_height_collapse: ✓ (all com_z > 0.2m)

### Priority 2: Posture ✗ ALL FAIL
- gate_hip_yaw_divergence_bounded: ✗ (all 3 exceed targets)
- gate_hip_yaw_abs_max_bounded: ✗ (2 of 3 exceed 0.30 rad)
- gate_roll_bounded: ✓ (all < 0.02 rad)
- gate_no_collapse: ✓ (all survived)

### Priority 3: Pitch (DEFERRED) ✓ ALL RECORDED
- All pitch values are within stable range

### Priority 4: Support Drift (DEFERRED) ✓ ALL RECORDED
- All support drift values recorded
- No contact/height/roll failure caused by drift

## Root Cause Analysis

### HY2-DIV Gate Behavior
The gate (z_low=0.300, z_high=0.393) means:
- **nominal (~0.39m):** com_z > z_high → gate = 0 → HY2-DIV **inactive**
- **low_0p300 (~0.27m):** com_z < z_low → gate = 1 → HY2-DIV **fully active**
- **high_0p480 (~0.45m):** com_z > z_high → gate = 0 → HY2-DIV **inactive**

This explains why divergence is high at nominal and high_0p480 (HY2-DIV not helping) and still high at low_0p300 despite HY2-DIV being active (insufficient torque).

### A0 Insufficient Authority at Low Height
At low_0p300:
- HY2-DIV was fully active (gate=100%)
- But **88.74% of torque commands were clipped**
- This means tau_max=0.5 Nm is insufficient to control divergence

### Divergence Drivers (Not HY2-DIV Only)
Even with HY2-DIV active at low height, divergence continued to grow. Possible causes:
1. Sagittal controller coupling with yaw
2. Support-position drift causing kinematic coupling
3. Insufficient hip-yaw PD gains in shape posture controller
4. Height-dependent dynamics not compensated

## Structural Invariants

All passed:
- WBC applied: false
- Hidden torque: 0.0
- Ownership violations: 0.0

## Files Changed

- `docs/validation/posture_standing_validation_gate_definition.md` - gate definition
- `scripts/evaluate_posture_standing_a0_5000.py` - evaluation script
- `outputs/posture_standing_validation_a0_5000/` - results

## Tests Run

Phase 0 health check:
- `pytest test_hip_yaw_divergence_control.py` → 35 passed
- `pytest test_shape_posture_hip_yaw_sign.py` → 9 passed
- `pytest test_step_e_hip_yaw_authority_fix.py` → 5 passed

## Next Recommended Actions

### Option 1: Stronger HY2-DIV Profile
Increase A0 tau_max to provide more authority at low height:
- A1: tau_max=1.0 Nm
- A2: tau_max=2.0 Nm
- A3: k=7.5, kd=1.5, tau_max=1.0 Nm

**Risk:** May worsen nominal/high divergence without solving low-height.

### Option 2: Extend HY2-DIV Gate
Extend z_high to cover nominal and high_0p480:
- B1: z_high=0.500 (partially covers nominal)
- B2: z_high=0.500 + tau_max=2.0

**Risk:** May cause nominal instability with HY2-DIV active.

### Option 3: Hybrid Approach
- Keep A0 for low-height authority
- Add separate controller for nominal/high divergence
- Consider boundary-aware hip-yaw position controller

### Option 4: Investigate Root Cause
Before adding more authority, investigate why divergence grows:
1. Check sagittal controller coupling with yaw
2. Check support-position drift effect on hip-yaw
3. Check if shape posture hip-yaw gains need adjustment

## Final Decision

```
POSTURE_REQUIRES_STRONGER_HY2_PROFILE
```

**What this means:**
- A0 is safe (survived, no collapse, no WBC)
- A0 gate pass-through is working correctly
- A0 torque authority (tau_max=0.5) is insufficient at low height
- Divergence is not controlled at any height

**What this does NOT mean:**
- Official Step E pass
- Ready for Step C or Step D
- HY2-DIV is the wrong approach

**Next phase:** Stronger HY2-DIV profile evaluation or root cause investigation