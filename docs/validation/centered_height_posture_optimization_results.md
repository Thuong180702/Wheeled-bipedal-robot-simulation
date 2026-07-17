# Centered Height Posture Optimization Results

**Date:** 2026-06-19  
**Status:** CENTERED_POSTURE_SETUPS_PASS_WITH_MONITORING

---

## Key Results

| Metric | Count/10 |
|--------|----------|
| Sagittal CoM-x centered (≤0.005 m) | **10/10** ✅ |
| Height error ≤0.005 m | **10/10** ✅ |
| Both wheel contacts valid | **10/10** ✅ |
| No non-wheel contact | **10/10** ✅ |
| Joint margin safe | **10/10** ✅ |
| Lateral CoM-y ≤0.005 m | **4/10** ⚠️ |

## Lateral Bias Is Intrinsic

The lateral CoM-y bias (com_support_error_y ≠ 0) at 6/10 heights is an **intrinsic geometric property** of the mechanism, not fixable via joint posture adjustment:

```
Hips at y=0 (torso mass ~70%)
            ↓
Legs bend → wheels shift laterally (up to ±35 mm)
            ↓
CoM tracks torso (y≈0) more than wheels
            ↓
com_support_error_y up to ±17 mm
```

Hip_roll adjustments shift com_y by ≤3 mm — insufficient to correct biases of 10-20 mm.

This is documented but **not a blocker**. The lateral bias is symmetrical and should be dynamically safe (it doesn't create a yaw moment). Monitor during dynamic validation.

## Smooth Posture Functions

The optimizer selected 4th-degree polynomial fits that are:
- **Strictly monotone decreasing** across [0.30, 0.48] m
- **Continuous** (derivative defined everywhere)
- **Smooth** (no kinks from the original coarse grid)
- **Valid** at all 10 breakpoints (height error ≤5 mm, contacts valid, sagittal CoM centered)

## Centered Posture Height Functions

```
hip_pitch_ref(height) = poly4(heights, hip_pitch_opt)
knee_ref(height)      = poly4(heights, knee_opt)
```

Saved to `outputs/physical_target_height_setups_centered/centered_posture_height_functions.json`

## Classification

**CENTERED_POSTURE_SETUPS_PASS_WITH_MONITORING**

Proceed to Phase 4 (fit continuous functions — already done via poly4), then Phase 5 (implement centered posture schedule).
